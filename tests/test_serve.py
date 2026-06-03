"""
Tests for `shiftgate serve` — the OpenAI-compatible proxy.

Uses fastapi.testclient.TestClient with a fully injected app: a synthetic task
registry, a deterministic fake embedder, a force-selected backend, and a fake
forwarder so no real network or model download happens.
"""

from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient

from shiftgate.registry.adapter_registry import AdapterRegistry
from shiftgate.registry.schemas import AdapterEntry, TaskCluster
from shiftgate.registry.task_registry import TaskRegistry
from shiftgate.runtime.backend import BackendRouter
from shiftgate.serve import create_app


# ---------------------------------------------------------------------------
# Fakes / fixtures
# ---------------------------------------------------------------------------

def _unit(v):
    arr = np.array(v, dtype=np.float32)
    return (arr / np.linalg.norm(arr)).tolist()


class FakeEmbedder:
    """Always returns a vector aligned with task_x → adapter-x."""

    def embed(self, text: str):
        return np.array([1.0, 0.0, 0.0], dtype=np.float32)


class FakeForwarder:
    """Captures the forwarded body and returns a canned response."""

    def __init__(self):
        self.last_body = None
        self.last_url = None

    async def complete(self, url, headers, body):
        self.last_url = url
        self.last_body = body
        return 200, {"id": "cmpl-1", "model": body.get("model"), "choices": []}

    async def stream(self, url, headers, body):  # pragma: no cover - not used here
        self.last_url = url
        self.last_body = body
        for chunk in ['data: {"x":1}', "data: [DONE]"]:
            yield chunk


def _task(task_id, centroid, adapters):
    return TaskCluster(
        id=task_id,
        name=task_id,
        description="t",
        validation_examples=["x"],
        embedding_centroid=_unit(centroid),
        preferred_adapters=adapters,
    )


def _adapter(adapter_id, runtime_name=None):
    return AdapterEntry(
        id=adapter_id,
        name=adapter_id,
        base_model="test/base",
        task_tags=[],
        runtime_name=runtime_name,
        status="linked",
    )


@pytest.fixture()
def forwarder():
    return FakeForwarder()


@pytest.fixture()
def client(tmp_path, forwarder):
    tasks = [
        _task("task_x", [1, 0, 0], ["adapter-x"]),
        _task("task_y", [0, 1, 0], ["adapter-y"]),
        _task("orphan", [0, 0, 1], ["missing-adapter"]),
    ]
    task_reg = TaskRegistry(tasks=tasks, source_path=tmp_path / "tasks.json")
    adapter_reg = AdapterRegistry(
        adapters=[_adapter("adapter-x", runtime_name="adapter-x-vllm"), _adapter("adapter-y")],
        source_path=tmp_path / "adapters.json",
    )

    router = BackendRouter()
    router.select("vllm")  # force-select; no availability ping

    app = create_app(
        backend="vllm",
        task_reg=task_reg,
        adapter_reg=adapter_reg,
        embedder=FakeEmbedder(),
        backend_router=router,
        forwarder=forwarder,
    )
    return TestClient(app)


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------

def test_health_ok(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["backend"] == "vllm"
    assert body["adapters"] == 2


# ---------------------------------------------------------------------------
# /v1/models
# ---------------------------------------------------------------------------

def test_models_lists_auto_first_and_adapters(client):
    r = client.get("/v1/models")
    assert r.status_code == 200
    data = r.json()
    assert data["object"] == "list"
    ids = [m["id"] for m in data["data"]]
    assert ids[0] == "auto"
    assert "adapter-x" in ids
    assert "adapter-y" in ids


# ---------------------------------------------------------------------------
# /v1/chat/completions
# ---------------------------------------------------------------------------

def test_auto_triggers_routing_and_substitutes_model(client, forwarder):
    r = client.post(
        "/v1/chat/completions",
        json={"model": "auto", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert r.status_code == 200
    # adapter-x has runtime_name "adapter-x-vllm" → that's the substituted model.
    assert forwarder.last_body["model"] == "adapter-x-vllm"


def test_auto_sets_route_header(client):
    r = client.post(
        "/v1/chat/completions",
        json={"model": "auto", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert r.status_code == 200
    header = r.headers.get("X-Shiftgate-Route")
    assert header is not None
    assert header.startswith("adapter-x (")


def test_specific_model_bypasses_routing(client, forwarder):
    r = client.post(
        "/v1/chat/completions",
        json={"model": "my-explicit-model", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert r.status_code == 200
    # Forwarded verbatim, no substitution.
    assert forwarder.last_body["model"] == "my-explicit-model"
    assert r.headers.get("X-Shiftgate-Route") is None


def test_no_adapter_returns_400(tmp_path, forwarder):
    # Query embeds to task_x, but its preferred adapter isn't registered.
    tasks = [_task("task_x", [1, 0, 0], ["missing-adapter"])]
    task_reg = TaskRegistry(tasks=tasks, source_path=tmp_path / "tasks.json")
    adapter_reg = AdapterRegistry(adapters=[], source_path=tmp_path / "adapters.json")
    router = BackendRouter()
    router.select("vllm")
    app = create_app(
        backend="vllm",
        task_reg=task_reg,
        adapter_reg=adapter_reg,
        embedder=FakeEmbedder(),
        backend_router=router,
        forwarder=forwarder,
    )
    client = TestClient(app)

    r = client.post(
        "/v1/chat/completions",
        json={"model": "auto", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert r.status_code == 400
    body = r.json()
    assert body["error"] == "no adapter available for this query"
    assert body["matched_task"] == "task_x"


def test_missing_model_field_returns_400(client):
    r = client.post("/v1/chat/completions", json={"messages": []})
    assert r.status_code == 400
