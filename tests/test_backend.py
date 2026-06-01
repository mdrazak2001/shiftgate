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
    CerebrasBackend,
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


# ---------------------------------------------------------------------------
# Cerebras backend
# ---------------------------------------------------------------------------

class TestCerebrasBackend:
    def test_effective_name_uses_runtime_name(self, monkeypatch):
        captured = {}

        class _Resp:
            def raise_for_status(self):
                pass

            def json(self):
                return {"choices": [{"message": {"content": "ok"}}]}

        def fake_post(url, json, headers, timeout):
            captured["model"] = json["model"]
            captured["headers"] = headers
            return _Resp()

        monkeypatch.setattr("shiftgate.runtime.backend.httpx.post", fake_post)

        backend = CerebrasBackend(api_key="csk-test")
        adapter = _adapter(runtime_name="llama3.1-8b")
        backend.generate("hi", adapter)
        assert captured["model"] == "llama3.1-8b"

    def test_effective_name_falls_back_to_id(self, monkeypatch):
        captured = {}

        class _Resp:
            def raise_for_status(self):
                pass

            def json(self):
                return {"choices": [{"message": {"content": "ok"}}]}

        def fake_post(url, json, headers, timeout):
            captured["model"] = json["model"]
            return _Resp()

        monkeypatch.setattr("shiftgate.runtime.backend.httpx.post", fake_post)

        backend = CerebrasBackend(api_key="csk-test")
        adapter = _adapter(runtime_name=None, hf_repo="org/sql-lora")
        backend.generate("hi", adapter)
        assert captured["model"] == "sql-lora"

    def test_is_available_false_without_key_makes_no_network_call(self, monkeypatch):
        # Ensure CEREBRAS_API_KEY is not picked up from the environment.
        monkeypatch.delenv("CEREBRAS_API_KEY", raising=False)

        def boom(*a, **k):
            raise AssertionError("network call must not happen without an API key")

        monkeypatch.setattr("shiftgate.runtime.backend.httpx.get", boom)
        assert CerebrasBackend(api_key=None).is_available() is False

    def test_generate_includes_bearer_header(self, monkeypatch):
        captured = {}

        class _Resp:
            def raise_for_status(self):
                pass

            def json(self):
                return {"choices": [{"message": {"content": "ok"}}]}

        def fake_post(url, json, headers, timeout):
            captured["headers"] = headers
            captured["url"] = url
            return _Resp()

        monkeypatch.setattr("shiftgate.runtime.backend.httpx.post", fake_post)

        backend = CerebrasBackend(api_key="csk-secret")
        backend.generate("hi", _adapter(runtime_name="m"))
        assert captured["headers"]["Authorization"] == "Bearer csk-secret"
        assert captured["url"].startswith("https://api.cerebras.ai/v1")

    def test_api_key_read_from_env(self, monkeypatch):
        monkeypatch.setenv("CEREBRAS_API_KEY", "csk-from-env")
        assert CerebrasBackend().api_key == "csk-from-env"

    def test_router_detects_cerebras_when_only_cloud_available(self, monkeypatch):
        router = BackendRouter(cerebras_api_key="csk-test")
        monkeypatch.setattr(router._ollama, "is_available", lambda: False)
        monkeypatch.setattr(router._vllm, "is_available", lambda: False)
        monkeypatch.setattr(router._cerebras, "is_available", lambda: True)
        assert router.detect() == "cerebras"
        assert router.active_backend_name == "cerebras"
        assert router.active_backend_url == "https://api.cerebras.ai/v1"
