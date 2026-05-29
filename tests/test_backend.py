"""
Tests for the inference backend layer.

Focus areas:
  - effective_backend_name() name resolution (runtime_name vs id)
  - BackendRouter.verify_adapter() against a stubbed loaded-adapter list
No real HTTP calls are made.
"""

from __future__ import annotations

import pytest

from shiftgate.registry.schemas import AdapterEntry
from shiftgate.runtime.backend import (
    BackendRouter,
    VLLMBackend,
    effective_backend_name,
)


def _adapter(**kwargs) -> AdapterEntry:
    base = dict(id="sql-lora", name="SQL", base_model="llama3")
    base.update(kwargs)
    return AdapterEntry(**base)


# ---------------------------------------------------------------------------
# effective_backend_name()
# ---------------------------------------------------------------------------

class TestEffectiveBackendName:
    def test_uses_runtime_name_when_set(self):
        adapter = _adapter(runtime_name="sql-lora-vllm")
        assert effective_backend_name(adapter) == "sql-lora-vllm"

    def test_falls_back_to_id_when_runtime_name_none(self):
        adapter = _adapter(runtime_name=None, hf_repo="org/sql-lora")
        assert effective_backend_name(adapter) == "sql-lora"

    def test_falls_back_to_id_when_runtime_name_blank(self):
        # An empty / whitespace runtime_name must not be used.
        adapter = _adapter(runtime_name="   ", hf_repo="org/sql-lora")
        assert effective_backend_name(adapter) == "sql-lora"


# ---------------------------------------------------------------------------
# generate() uses the effective name
# ---------------------------------------------------------------------------

class TestGenerateUsesEffectiveName:
    def test_vllm_generate_sends_runtime_name(self, monkeypatch):
        captured = {}

        class _Resp:
            def raise_for_status(self):
                pass

            def json(self):
                return {"choices": [{"message": {"content": "ok"}}]}

        def fake_post(url, json, timeout):
            captured["model"] = json["model"]
            return _Resp()

        monkeypatch.setattr("shiftgate.runtime.backend.httpx.post", fake_post)

        backend = VLLMBackend()
        adapter = _adapter(runtime_name="sql-lora-vllm")
        backend.generate("hello", adapter)
        assert captured["model"] == "sql-lora-vllm"

    def test_vllm_generate_explicit_override_wins(self, monkeypatch):
        captured = {}

        class _Resp:
            def raise_for_status(self):
                pass

            def json(self):
                return {"choices": [{"message": {"content": "ok"}}]}

        def fake_post(url, json, timeout):
            captured["model"] = json["model"]
            return _Resp()

        monkeypatch.setattr("shiftgate.runtime.backend.httpx.post", fake_post)

        backend = VLLMBackend()
        adapter = _adapter(runtime_name="sql-lora-vllm")
        # Explicit lora_name must win over runtime_name.
        backend.generate("hello", adapter, lora_name="override-name")
        assert captured["model"] == "override-name"


# ---------------------------------------------------------------------------
# BackendRouter.verify_adapter()
# ---------------------------------------------------------------------------

class TestVerifyAdapter:
    def test_no_backend_returns_false_none(self, monkeypatch):
        router = BackendRouter()
        monkeypatch.setattr(router, "detect", lambda: None)
        # Active stays None → verification skipped.
        is_loaded, backend_name = router.verify_adapter(_adapter(runtime_name="x"))
        assert is_loaded is False
        assert backend_name is None

    def test_loaded_returns_true(self, monkeypatch):
        router = BackendRouter()
        # Force vLLM active and stub its loaded list.
        router._active = router._vllm
        monkeypatch.setattr(router._vllm, "list_loaded_adapters", lambda: ["base", "sql-lora-vllm"])
        is_loaded, backend_name = router.verify_adapter(_adapter(runtime_name="sql-lora-vllm"))
        assert is_loaded is True
        assert backend_name == "vllm"

    def test_not_loaded_returns_false_with_backend(self, monkeypatch):
        router = BackendRouter()
        router._active = router._vllm
        monkeypatch.setattr(router._vllm, "list_loaded_adapters", lambda: ["base"])
        is_loaded, backend_name = router.verify_adapter(_adapter(runtime_name="sql-lora-vllm"))
        assert is_loaded is False
        assert backend_name == "vllm"


# ---------------------------------------------------------------------------
# list_loaded_adapters() degrades gracefully when offline
# ---------------------------------------------------------------------------

class TestListLoadedAdaptersOffline:
    def test_vllm_returns_empty_on_connection_error(self, monkeypatch):
        def boom(*a, **k):
            raise OSError("connection refused")

        monkeypatch.setattr("shiftgate.runtime.backend.httpx.get", boom)
        assert VLLMBackend().list_loaded_adapters() == []
