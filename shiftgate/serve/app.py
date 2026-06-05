"""
FastAPI application exposing shiftgate routing behind the OpenAI API.

Endpoints
---------
POST /v1/chat/completions
    OpenAI-compatible chat completions.  When ``model == "auto"`` the request
    is routed through shiftgate's semantic router and the chosen adapter's
    backend name is substituted before forwarding upstream.  Any other model
    id bypasses routing and is forwarded verbatim.

GET  /v1/models
    Lists ``"auto"`` plus every registered adapter as OpenAI model objects.

GET  /health
    Liveness/readiness probe.

The actual upstream HTTP is delegated to a *forwarder* object stored on
``app.state.forwarder`` so it can be swapped out in tests.
"""

from __future__ import annotations

import logging
import time
import uuid
from typing import Any, AsyncIterator, Optional

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from sse_starlette.sse import EventSourceResponse
from starlette.concurrency import run_in_threadpool

from shiftgate.runtime.backend import (
    BackendError,
    BackendRouter,
    BaseBackend,
    effective_backend_name,
)

logger = logging.getLogger(__name__)

_ROUTE_HEADER = "X-Shiftgate-Route"
_READ_TIMEOUT = 120.0
_RUNTIMES_TTL = 60.0  # seconds to cache the active backend's loaded-runtime list


# ---------------------------------------------------------------------------
# Upstream forwarder (swappable for tests)
# ---------------------------------------------------------------------------

class HttpxForwarder:
    """Default forwarder that proxies to an upstream OpenAI-compatible server."""

    async def complete(
        self, url: str, headers: dict[str, str], body: dict[str, Any]
    ) -> tuple[int, Any]:
        """Forward a non-streaming request and return ``(status_code, json)``."""
        async with httpx.AsyncClient(timeout=_READ_TIMEOUT) as client:
            r = await client.post(url, json=body, headers=headers)
            try:
                data = r.json()
            except Exception:
                data = {"error": "upstream returned non-JSON response", "body": r.text}
            return r.status_code, data

    async def stream(
        self, url: str, headers: dict[str, str], body: dict[str, Any]
    ) -> AsyncIterator[str]:
        """Forward a streaming request, yielding raw SSE lines from upstream."""
        async with httpx.AsyncClient(timeout=_READ_TIMEOUT) as client:
            async with client.stream("POST", url, json=body, headers=headers) as r:
                async for line in r.aiter_lines():
                    yield line


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _openai_envelope(text: str, model: str) -> dict[str, Any]:
    """Wrap a plain text completion in an OpenAI chat.completion response shape.

    Used to translate non-OpenAI backends (e.g. Cloudflare Workers AI) into a
    response any OpenAI client understands.
    """
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
            }
        ],
    }


def _last_user_message(messages: list[dict[str, Any]]) -> str | None:
    """Return the content of the last user message, or None."""
    for msg in reversed(messages):
        if msg.get("role") == "user":
            content = msg.get("content")
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                # OpenAI "parts" format — concatenate text parts.
                parts = [p.get("text", "") for p in content if isinstance(p, dict)]
                return " ".join(p for p in parts if p)
    return None


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

def create_app(
    *,
    backend: str = "auto",
    cerebras_api_key: Optional[str] = None,
    task_reg: Any | None = None,
    adapter_reg: Any | None = None,
    embedder: Any | None = None,
    backend_router: BackendRouter | None = None,
    forwarder: Any | None = None,
) -> FastAPI:
    """Build the shiftgate serve FastAPI application.

    All dependencies are injectable so the app can be exercised in tests
    without touching disk or the network.  When omitted, registries are loaded
    from ``~/.shiftgate`` and a fresh :class:`BackendRouter` is created.
    """
    app = FastAPI(title="shiftgate serve", version="0.1")

    if task_reg is None or adapter_reg is None:
        from shiftgate.registry.adapter_registry import AdapterRegistry
        from shiftgate.registry.task_registry import TaskRegistry

        task_reg = task_reg or TaskRegistry.load()
        adapter_reg = adapter_reg or AdapterRegistry.load()

    if backend_router is None:
        backend_router = BackendRouter(cerebras_api_key=cerebras_api_key)
    backend_router.select(backend)

    app.state.task_reg = task_reg
    app.state.adapter_reg = adapter_reg
    app.state.embedder = embedder
    app.state.backend_router = backend_router
    app.state.backend_choice = backend
    app.state.forwarder = forwarder or HttpxForwarder()
    # (timestamp, runtimes set) cache so we don't ping the backend on every request.
    app.state.runtimes_cache = None

    def _embedder():
        if app.state.embedder is None:
            from shiftgate.router.embedder import Embedder

            app.state.embedder = Embedder()
        return app.state.embedder

    def _active_backend() -> BaseBackend | None:
        router: BackendRouter = app.state.backend_router
        if router.active_backend is None:
            # A backend may have come up after startup — try once more.
            router.select(app.state.backend_choice)
        return router.active_backend

    def _available_runtimes() -> set[str] | None:
        """Loaded runtimes on the active backend, cached with a 60s TTL.

        Returns ``None`` when no backend is active (no filtering).
        """
        active = _active_backend()
        if active is None:
            return None
        cache = app.state.runtimes_cache
        now = time.monotonic()
        if cache is not None and (now - cache[0]) < _RUNTIMES_TTL:
            return cache[1]
        runtimes = set(active.list_loaded_adapters())

        # Cloudflare base models are always available without any finetune
        # upload, so every registered @cf/ adapter with no finetune runtime is
        # usable (base-model inference).
        from shiftgate.runtime.backend import CloudflareBackend

        if isinstance(active, CloudflareBackend) and app.state.adapter_reg is not None:
            for adapter in app.state.adapter_reg.list_adapters():
                is_cf_base = (adapter.base_model or "").startswith("@cf/")
                has_finetune = bool((adapter.runtime_name or "").strip())
                if is_cf_base and not has_finetune:
                    runtimes.add(adapter.effective_backend_name())

        app.state.runtimes_cache = (now, runtimes)
        return runtimes

    # -- health -------------------------------------------------------------
    @app.get("/health")
    def health() -> dict[str, Any]:
        router: BackendRouter = app.state.backend_router
        if router.active_backend is None:
            router.select(app.state.backend_choice)
        return {
            "status": "ok",
            "backend": router.active_backend_name,
            "adapters": len(app.state.adapter_reg),
        }

    # -- models -------------------------------------------------------------
    @app.get("/v1/models")
    def list_models() -> dict[str, Any]:
        data: list[dict[str, Any]] = [
            {"id": "auto", "object": "model", "owned_by": "shiftgate"}
        ]
        for adapter in app.state.adapter_reg.list_adapters():
            data.append(
                {"id": adapter.id, "object": "model", "owned_by": "shiftgate"}
            )
        return {"object": "list", "data": data}

    # -- chat completions ---------------------------------------------------
    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        body = await request.json()
        model = body.get("model")
        messages = body.get("messages") or []
        stream = bool(body.get("stream", False))

        if not model:
            return JSONResponse(
                status_code=400, content={"error": "missing 'model' field"}
            )

        route_header: str | None = None
        selected_adapter = None  # AdapterEntry for the non-OpenAI generate() path
        query = _last_user_message(messages)

        # --- model == "auto": run shiftgate's router ---
        if model == "auto":
            from shiftgate.router import router as routing

            if not query:
                return JSONResponse(
                    status_code=400,
                    content={"error": "no user message found to route"},
                )

            try:
                _trace, match = routing.route(
                    query,
                    app.state.task_reg,
                    app.state.adapter_reg,
                    _embedder(),
                    available_runtimes=_available_runtimes(),
                )
            except ValueError as exc:
                # Embeddings not initialised, etc.
                return JSONResponse(status_code=400, content={"error": str(exc)})

            if match.selected_adapter is None:
                return JSONResponse(
                    status_code=400,
                    content={
                        "error": "no adapter available for this query",
                        "matched_task": match.matched_task.id,
                    },
                )

            selected_adapter = match.selected_adapter
            body["model"] = effective_backend_name(selected_adapter)
            route_header = f"{selected_adapter.id} ({match.similarity_score:.2f})"
        else:
            # Bypass routing: look up the explicit adapter (needed by non-OpenAI
            # backends that build the request from the AdapterEntry).
            selected_adapter = app.state.adapter_reg.get_adapter(model)

        # --- resolve the active backend ---
        active = _active_backend()
        if active is None:
            return JSONResponse(
                status_code=503,
                content={"error": "no inference backend available"},
            )

        # --- OpenAI-compatible backends: forward raw (existing fast path) ---
        if active.is_openai_compatible:
            url = f"{active.openai_base_url()}/chat/completions"
            headers = {"Content-Type": "application/json", **active.auth_headers()}
            forwarder = app.state.forwarder

            if stream:
                async def event_gen() -> AsyncIterator[dict[str, str]]:
                    async for line in forwarder.stream(url, headers, body):
                        if not line:
                            continue
                        s = line.strip()
                        if not s:
                            continue
                        payload = s[len("data:"):].strip() if s.startswith("data:") else s
                        yield {"data": payload}

                sse_headers = {_ROUTE_HEADER: route_header} if route_header else None
                return EventSourceResponse(event_gen(), headers=sse_headers)

            status_code, data = await forwarder.complete(url, headers, body)
            resp_headers = {_ROUTE_HEADER: route_header} if route_header else None
            return JSONResponse(content=data, status_code=status_code, headers=resp_headers)

        # --- non-OpenAI backends (e.g. Cloudflare): translate via generate() ---
        backend_label = app.state.backend_router.active_backend_name or "backend"

        if stream:
            # TODO(v0.3): wrap generate() output in a single SSE event so
            # streaming clients work against non-OpenAI backends too.
            return JSONResponse(
                status_code=501,
                content={"error": f"streaming not yet supported for backend: {backend_label}"},
            )

        if selected_adapter is None:
            return JSONResponse(
                status_code=400,
                content={
                    "error": (
                        f"backend '{backend_label}' needs a registered adapter, "
                        f"but model '{model}' is not in the registry"
                    )
                },
            )
        if not query:
            return JSONResponse(
                status_code=400,
                content={"error": "no user message found to generate from"},
            )

        try:
            text = await run_in_threadpool(active.generate, query, selected_adapter)
        except BackendError as exc:
            return JSONResponse(status_code=502, content={"error": str(exc)})

        envelope = _openai_envelope(text, body.get("model", model))
        resp_headers = {_ROUTE_HEADER: route_header} if route_header else None
        return JSONResponse(content=envelope, headers=resp_headers)

    return app
