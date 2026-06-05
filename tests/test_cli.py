"""
Integration tests for CLI commands — specifically the backend-aware adapter
filtering wiring in `shiftgate run`.

Regression guard for the bug where `run` selected an adapter whose runtime was
NOT loaded on the active backend (e.g. picking `code-writer` on Cerebras, which
only serves `gpt-oss-120b`, producing a 404).
"""

from __future__ import annotations

import numpy as np
import pytest
from typer.testing import CliRunner

from shiftgate import cli
from shiftgate.registry.adapter_registry import AdapterRegistry
from shiftgate.registry.schemas import AdapterEntry, TaskCluster
from shiftgate.registry.task_registry import TaskRegistry

runner = CliRunner()


def _unit(v):
    arr = np.array(v, dtype=np.float32)
    return (arr / np.linalg.norm(arr)).tolist()


class _FixedEmbedder:
    """Always embeds to the code_python centroid."""

    def embed(self, text: str):
        return np.array([1.0, 0.0, 0.0], dtype=np.float32)


class _FakeActiveBackend:
    """Cerebras-like backend that only serves gpt-oss-120b."""

    def list_loaded_adapters(self):
        return ["gpt-oss-120b"]

    def list_loaded_adapters_cached(self):
        return self.list_loaded_adapters()


class _FakeBackendRouter:
    """Stand-in for BackendRouter with a fixed active backend + capture."""

    captured: dict = {}

    def __init__(self, *args, **kwargs):
        self._active = None

    def detect(self):
        self._active = _FakeActiveBackend()
        return "cerebras"

    @property
    def active_backend(self):
        return self._active

    @property
    def active_backend_name(self):
        return "cerebras" if self._active is not None else None

    def generate(self, prompt, adapter):
        # Record which adapter the run command chose to send to the backend.
        type(self).captured["adapter_id"] = adapter.id
        type(self).captured["effective"] = adapter.effective_backend_name()
        return "stubbed response"


@pytest.fixture()
def registries(tmp_path):
    task = TaskCluster(
        id="code_python",
        name="Python Code Generation",
        description="Writing code",
        validation_examples=["x"],
        embedding_centroid=_unit([1, 0, 0]),
        # code-writer is linked FIRST (would win without filtering).
        preferred_adapters=["code-writer", "cerebras-gptoss"],
    )
    task_reg = TaskRegistry(tasks=[task], source_path=tmp_path / "tasks.json")

    code_writer = AdapterEntry(
        id="code-writer",
        name="Code Writer",
        base_model="unknown",
        task_tags=["code"],
        runtime_name="code",
        status="linked",
    )
    cerebras_gptoss = AdapterEntry(
        id="cerebras-gptoss",
        name="Cerebras Gptoss",
        base_model="unknown",
        task_tags=["code"],
        runtime_name="gpt-oss-120b",
        status="linked",
    )
    adapter_reg = AdapterRegistry(
        adapters=[code_writer, cerebras_gptoss],
        source_path=tmp_path / "adapters.json",
    )
    return task_reg, adapter_reg


def test_run_selects_loaded_adapter_not_first_linked(monkeypatch, registries):
    """`run` must skip code-writer (runtime 'code' not loaded) and pick gpt-oss-120b."""
    task_reg, adapter_reg = registries
    _FakeBackendRouter.captured = {}

    monkeypatch.setattr(cli, "_load_registries", lambda: (task_reg, adapter_reg))
    monkeypatch.setattr(cli, "_get_embedder", lambda: _FixedEmbedder())
    monkeypatch.setattr("shiftgate.runtime.backend.BackendRouter", _FakeBackendRouter)
    monkeypatch.setattr("shiftgate.feedback.loop.record_trace", lambda trace: None)

    result = runner.invoke(cli.app, ["run", "write a python sorting function"])

    assert result.exit_code == 0, result.output
    # The adapter forwarded to the backend must be the loaded one.
    assert _FakeBackendRouter.captured.get("adapter_id") == "cerebras-gptoss"
    assert _FakeBackendRouter.captured.get("effective") == "gpt-oss-120b"


def test_run_routing_logic_filters_via_active_runtimes(registries):
    """Directly exercise the run command's routing logic (filtering path)."""
    from shiftgate.router import router as routing

    task_reg, adapter_reg = registries
    backend_router = _FakeBackendRouter()
    backend_router.detect()
    available_runtimes = cli._active_runtimes(backend_router)

    assert available_runtimes == {"gpt-oss-120b"}

    _trace, match = routing.route(
        "write a python sorting function",
        task_reg,
        adapter_reg,
        _FixedEmbedder(),
        available_runtimes=available_runtimes,
    )
    assert match.selected_adapter is not None
    assert match.selected_adapter.id == "cerebras-gptoss"
