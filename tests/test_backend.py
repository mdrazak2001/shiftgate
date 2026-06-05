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
    BackendError,
    BackendRouter,
    CerebrasBackend,
    CloudflareBackend,
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


# ---------------------------------------------------------------------------
# Cloudflare backend
# ---------------------------------------------------------------------------

def _cf_adapter(**kwargs) -> AdapterEntry:
    base = dict(
        id="sql-lora",
        name="SQL",
        base_model="@cf/mistral/mistral-7b-instruct-v0.2-lora",
    )
    base.update(kwargs)
    return AdapterEntry(**base)


class _CFResp:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


class TestCloudflareBackend:
    def test_is_not_openai_compatible(self):
        assert CloudflareBackend(account_id="a", api_token="t").is_openai_compatible is False

    def test_is_available_false_without_creds_no_network(self, monkeypatch):
        monkeypatch.delenv("CLOUDFLARE_ACCOUNT_ID", raising=False)
        monkeypatch.delenv("CLOUDFLARE_API_TOKEN", raising=False)

        def boom(*a, **k):
            raise AssertionError("no network call should happen without creds")

        monkeypatch.setattr("shiftgate.runtime.backend.httpx.get", boom)
        assert CloudflareBackend(account_id=None, api_token=None).is_available() is False

    def test_creds_read_from_env(self, monkeypatch):
        monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", "acc123")
        monkeypatch.setenv("CLOUDFLARE_API_TOKEN", "tok456")
        backend = CloudflareBackend()
        assert backend.account_id == "acc123"
        assert backend.api_token == "tok456"
        assert "acc123" in backend.base_url

    def test_generate_sends_lora_and_model_in_url(self, monkeypatch):
        captured = {}

        def fake_post(url, json, headers, timeout):
            captured["url"] = url
            captured["json"] = json
            captured["headers"] = headers
            return _CFResp({"success": True, "result": {"response": "hello world"}})

        monkeypatch.setattr("shiftgate.runtime.backend.httpx.post", fake_post)

        backend = CloudflareBackend(account_id="acc", api_token="tok")
        adapter = _cf_adapter(runtime_name="my-sql-lora")
        text = backend.generate("hi", adapter)

        # base model in the URL path
        assert captured["url"].endswith("/run/@cf/mistral/mistral-7b-instruct-v0.2-lora")
        # lora as a separate body field (= runtime_name), with raw=True
        assert captured["json"]["lora"] == "my-sql-lora"
        assert captured["json"]["raw"] is True
        # bearer auth
        assert captured["headers"]["Authorization"] == "Bearer tok"
        # response extracted from result.response
        assert text == "hello world"

    def test_generate_lora_name_override_wins(self, monkeypatch):
        captured = {}

        def fake_post(url, json, headers, timeout):
            captured["json"] = json
            return _CFResp({"success": True, "result": {"response": "ok"}})

        monkeypatch.setattr("shiftgate.runtime.backend.httpx.post", fake_post)
        backend = CloudflareBackend(account_id="acc", api_token="tok")
        backend.generate("hi", _cf_adapter(runtime_name="rt"), lora_name="override")
        assert captured["json"]["lora"] == "override"

    def test_generate_omits_lora_for_base_model_only(self, monkeypatch):
        """No finetune runtime_name => base-model inference, no `lora` field."""
        captured = {}

        def fake_post(url, json, headers, timeout):
            captured["json"] = json
            return _CFResp({"success": True, "result": {"response": "ok"}})

        monkeypatch.setattr("shiftgate.runtime.backend.httpx.post", fake_post)
        backend = CloudflareBackend(account_id="acc", api_token="tok")
        backend.generate("hi", _cf_adapter(runtime_name=None))
        assert "lora" not in captured["json"]
        assert captured["json"]["raw"] is True

    def test_verify_cf_base_model_adapter_always_available(self, monkeypatch):
        """A @cf/ adapter with no finetune is runnable without any upload."""
        router = BackendRouter(cloudflare_account_id="acc", cloudflare_api_token="tok")
        router._active = CloudflareBackend(account_id="acc", api_token="tok")
        # finetunes endpoint reports nothing
        monkeypatch.setattr(
            router._active, "list_loaded_adapters", lambda: []
        )
        is_loaded, name = router.verify_adapter(_cf_adapter(runtime_name=None))
        assert is_loaded is True
        assert name == "cloudflare"

    def test_verify_cf_finetune_still_requires_upload(self, monkeypatch):
        """A @cf/ adapter WITH a finetune runtime is only available if uploaded."""
        router = BackendRouter(cloudflare_account_id="acc", cloudflare_api_token="tok")
        router._active = CloudflareBackend(account_id="acc", api_token="tok")
        monkeypatch.setattr(router._active, "list_loaded_adapters", lambda: [])
        is_loaded, _ = router.verify_adapter(_cf_adapter(runtime_name="not-uploaded"))
        assert is_loaded is False

    def test_generate_raises_when_base_model_not_cloudflare(self):
        backend = CloudflareBackend(account_id="acc", api_token="tok")
        adapter = _cf_adapter(base_model="llama3")  # not @cf/
        with pytest.raises(BackendError, match="Cloudflare base model"):
            backend.generate("hi", adapter)

    def test_generate_raises_on_success_false(self, monkeypatch):
        def fake_post(url, json, headers, timeout):
            return _CFResp({"success": False, "errors": [{"message": "bad lora"}]})

        monkeypatch.setattr("shiftgate.runtime.backend.httpx.post", fake_post)
        backend = CloudflareBackend(account_id="acc", api_token="tok")
        with pytest.raises(BackendError, match="bad lora"):
            backend.generate("hi", _cf_adapter(runtime_name="rt"))

    def test_list_loaded_adapters_flat_list(self, monkeypatch):
        def fake_get(url, headers, timeout):
            if url.endswith("/finetunes/public"):
                return _CFResp({"result": []})
            return _CFResp({"result": [{"name": "a"}, {"name": "b"}]})

        monkeypatch.setattr("shiftgate.runtime.backend.httpx.get", fake_get)
        backend = CloudflareBackend(account_id="acc", api_token="tok")
        assert backend.list_loaded_adapters() == ["a", "b"]

    def test_list_loaded_adapters_list_of_lists(self, monkeypatch):
        def fake_get(url, headers, timeout):
            if url.endswith("/finetunes/public"):
                return _CFResp({"result": []})
            return _CFResp({"result": [[{"name": "a"}, {"name": "b"}]]})

        monkeypatch.setattr("shiftgate.runtime.backend.httpx.get", fake_get)
        backend = CloudflareBackend(account_id="acc", api_token="tok")
        assert backend.list_loaded_adapters() == ["a", "b"]

    def test_list_loaded_adapters_includes_public_finetunes(self, monkeypatch):
        def fake_get(url, headers, timeout):
            if url.endswith("/finetunes/public"):
                return _CFResp({"result": [{"name": "cf-public-magicoder"}]})
            return _CFResp({"result": [{"name": "my-lora"}]})

        monkeypatch.setattr("shiftgate.runtime.backend.httpx.get", fake_get)
        backend = CloudflareBackend(account_id="acc", api_token="tok")
        assert backend.list_loaded_adapters() == ["my-lora", "cf-public-magicoder"]

    def test_list_loaded_adapters_empty_without_creds(self, monkeypatch):
        monkeypatch.delenv("CLOUDFLARE_ACCOUNT_ID", raising=False)
        monkeypatch.delenv("CLOUDFLARE_API_TOKEN", raising=False)
        assert CloudflareBackend(account_id=None, api_token=None).list_loaded_adapters() == []

    def test_effective_backend_name_for_cf_adapter(self):
        assert effective_backend_name(_cf_adapter(runtime_name="rt")) == "rt"
        assert effective_backend_name(_cf_adapter(runtime_name=None)) == "sql-lora"

    def test_router_detects_cloudflare_last(self, monkeypatch):
        router = BackendRouter(
            cloudflare_account_id="acc", cloudflare_api_token="tok"
        )
        monkeypatch.setattr(router._ollama, "is_available", lambda: False)
        monkeypatch.setattr(router._vllm, "is_available", lambda: False)
        monkeypatch.setattr(router._cerebras, "is_available", lambda: False)
        monkeypatch.setattr(router._cloudflare, "is_available", lambda: True)
        assert router.detect() == "cloudflare"
        assert router.active_backend_name == "cloudflare"

    def test_router_prefers_local_over_cloudflare(self, monkeypatch):
        router = BackendRouter(
            cloudflare_account_id="acc", cloudflare_api_token="tok"
        )
        monkeypatch.setattr(router._ollama, "is_available", lambda: False)
        monkeypatch.setattr(router._vllm, "is_available", lambda: True)
        monkeypatch.setattr(router._cerebras, "is_available", lambda: False)
        monkeypatch.setattr(router._cloudflare, "is_available", lambda: True)
        assert router.detect() == "vllm"

    def test_detect_skips_cloud_http_without_credentials(self, monkeypatch):
        """No cloud env vars and no local backends → zero HTTP calls."""
        monkeypatch.delenv("CEREBRAS_API_KEY", raising=False)
        monkeypatch.delenv("CLOUDFLARE_ACCOUNT_ID", raising=False)
        monkeypatch.delenv("CLOUDFLARE_API_TOKEN", raising=False)

        def boom(*args, **kwargs):
            raise AssertionError("httpx must not be called when no backends are configured")

        monkeypatch.setattr("shiftgate.runtime.backend.httpx.get", boom)
        monkeypatch.setattr("shiftgate.runtime.backend.httpx.post", boom)

        router = BackendRouter(
            cerebras_api_key=None,
            cloudflare_account_id=None,
            cloudflare_api_token=None,
        )
        monkeypatch.setattr(router._ollama, "is_available", lambda: False)
        monkeypatch.setattr(router._vllm, "is_available", lambda: False)

        assert router.detect() is None
        assert router.active_backend_name is None

    def test_list_loaded_adapters_cached_fetches_once(self, monkeypatch):
        calls = {"n": 0}

        def fake_get(url, headers, timeout):
            calls["n"] += 1
            return _CFResp({"result": [{"name": "a"}]})

        monkeypatch.setattr("shiftgate.runtime.backend.httpx.get", fake_get)
        backend = CloudflareBackend(account_id="acc", api_token="tok")

        assert backend.list_loaded_adapters_cached() == ["a"]
        assert backend.list_loaded_adapters_cached() == ["a"]
        # /finetunes + /finetunes/public on first uncached call only.
        assert calls["n"] == 2
