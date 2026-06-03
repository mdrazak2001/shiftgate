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
from typing import Any, AsyncIterator, Optional

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from sse_starlette.sse import EventSourceResponse

from shiftgate.runtime.backend import (
    BackendRouter,
    BaseBackend,
    effective_backend_name,
)

logger = logging.getLogger(__name__)

_ROUTE_HEADER = "X-Shiftgate-Route"
_READ_TIMEOUT = 120.0


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

        # --- model == "auto": run shiftgate's router ---
        if model == "auto":
            from shiftgate.router import router as routing

            query = _last_user_message(messages)
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

            body["model"] = effective_backend_name(match.selected_adapter)
            route_header = f"{match.selected_adapter.id} ({match.similarity_score:.2f})"

        # --- resolve the active backend ---
        active = _active_backend()
        if active is None:
            return JSONResponse(
                status_code=503,
                content={"error": "no inference backend available"},
            )

        url = f"{active.openai_base_url()}/chat/completions"
        headers = {"Content-Type": "application/json", **active.auth_headers()}
        forwarder = app.state.forwarder

        # --- streaming ---
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

        # --- non-streaming ---
        status_code, data = await forwarder.complete(url, headers, body)
        resp_headers = {_ROUTE_HEADER: route_header} if route_header else None
        return JSONResponse(content=data, status_code=status_code, headers=resp_headers)

    return app
