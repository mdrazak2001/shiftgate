"""
Thin clients for local LLM inference backends.

Two backends are supported:

Ollama (http://localhost:11434)
    Open-source inference server that supports LoRA adapters via custom
    Modelfiles.  To use a LoRA with Ollama, create a Modelfile like:

        FROM llama3
        ADAPTER /path/to/adapter.safetensors

    and run ``ollama create my-model -f Modelfile``.  shiftgate sets
    ``model`` to the adapter's Ollama model name (derived from adapter.id).

vLLM  (http://localhost:8000)
    Provides an OpenAI-compatible ``/v1/chat/completions`` endpoint.
    LoRA adapters are loaded at server start-up with ``--lora-modules``
    and are addressed by name via the ``model`` field in the request.

Both backends are auto-detected by ``BackendRouter``, which pings each
health endpoint and delegates to whichever is available.
"""

from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod

import httpx

from shiftgate.registry.schemas import AdapterEntry

logger = logging.getLogger(__name__)

# Default timeouts (seconds).
_CONNECT_TIMEOUT = 3.0
_READ_TIMEOUT = 120.0


def effective_backend_name(adapter: AdapterEntry) -> str:
    """Return the name the inference backend knows this adapter by.

    When the adapter was registered with ``--runtime <name>`` the user has
    explicitly told us the backend loaded it under that name (e.g. a vLLM
    ``--lora-modules`` key or an Ollama Modelfile model name).  In that case we
    must send ``runtime_name`` — sending ``adapter.id`` would address a model
    the backend has never heard of.

    Priority: ``runtime_name`` (if set and non-empty) > ``id``.
    """
    runtime = (adapter.runtime_name or "").strip()
    return runtime if runtime else adapter.id


class BaseBackend(ABC):
    """Abstract base for inference backends."""

    # Whether this backend speaks the OpenAI ``/v1/chat/completions`` wire
    # format.  Defaults to True so existing custom backends keep working with
    # the serve proxy's raw-forwarding path.  Backends with a bespoke API
    # (e.g. Cloudflare Workers AI) override this to False.
    is_openai_compatible: bool = True

    @abstractmethod
    def is_available(self) -> bool:
        """Return True if the backend can be reached."""

    @abstractmethod
    def generate(self, prompt: str, adapter: AdapterEntry) -> str:
        """Send ``prompt`` to the backend and return the generated text."""

    @abstractmethod
    def list_loaded_adapters(self) -> list[str]:
        """Return the model/adapter names currently loaded in the backend.

        Must use a short timeout and silently return ``[]`` if the backend is
        unreachable — this method is only used for informational verification.
        """

    # -- OpenAI-compatible proxying (used by `shiftgate serve`) --------------

    def openai_base_url(self) -> str:
        """Return the base URL of the backend's OpenAI-compatible API.

        Ollama and vLLM expose it at ``<base_url>/v1``.  Backends whose
        ``base_url`` already ends in ``/v1`` (e.g. Cerebras) override this.
        """
        return f"{self.base_url}/v1"

    def auth_headers(self) -> dict[str, str]:
        """Return extra HTTP headers needed to authenticate (empty for local)."""
        return {}


# ---------------------------------------------------------------------------
# Ollama
# ---------------------------------------------------------------------------

class OllamaBackend(BaseBackend):
    """Thin httpx client for the Ollama inference server.

    Ollama API reference: https://github.com/ollama/ollama/blob/main/docs/api.md

    LoRA adapters in Ollama
    -----------------------
    Ollama does not have a first-class "load this adapter onto this base model"
    API at request time.  Instead you pre-register a composite model via a
    Modelfile::

        FROM llama3
        ADAPTER /path/to/my-lora.safetensors

        ollama create my-lora-model -f Modelfile

    shiftgate uses ``adapter.id`` as the Ollama model name by convention.
    Ensure your Ollama model names match shiftgate adapter IDs.
    """

    def __init__(self, base_url: str = "http://localhost:11434") -> None:
        self.base_url = base_url.rstrip("/")

    def is_available(self) -> bool:
        """Return True if the Ollama server responds to a health ping."""
        try:
            r = httpx.get(f"{self.base_url}/api/tags", timeout=_CONNECT_TIMEOUT)
            return r.status_code == 200
        except Exception:
            return False

    def generate(
        self,
        prompt: str,
        adapter: AdapterEntry,
        *,
        model_name: str | None = None,
        stream: bool = False,
    ) -> str:
        """Generate text via Ollama's ``/api/generate`` endpoint.

        Parameters
        ----------
        prompt:
            The full prompt string.
        adapter:
            The selected ``AdapterEntry``; ``adapter.id`` is used as the
            Ollama model name unless overridden by ``model_name``.
        model_name:
            Override the Ollama model name (useful when the Ollama model name
            differs from the shiftgate adapter ID).
        stream:
            If True, Ollama streams response tokens.  This client reads the
            full stream and returns the concatenated text.
        """
        # Explicit override wins; otherwise use the backend-effective name
        # (runtime_name when set, else adapter.id).
        model = model_name or effective_backend_name(adapter)
        payload = {"model": model, "prompt": prompt, "stream": stream}

        logger.debug("Ollama generate: model=%s", model)
        try:
            r = httpx.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=_READ_TIMEOUT,
            )
            r.raise_for_status()
        except httpx.HTTPError as exc:
            raise BackendError(f"Ollama request failed: {exc}") from exc

        data = r.json()
        return data.get("response", "")

    def list_loaded_adapters(self) -> list[str]:
        """Return the names of all models loaded in Ollama (``GET /api/tags``).

        Silently returns ``[]`` if Ollama is unreachable.
        """
        try:
            r = httpx.get(f"{self.base_url}/api/tags", timeout=_CONNECT_TIMEOUT)
            r.raise_for_status()
            models = r.json().get("models", [])
            return [m["name"] for m in models if "name" in m]
        except Exception as exc:
            logger.debug("Ollama list_loaded_adapters failed: %s", exc)
            return []


# ---------------------------------------------------------------------------
# vLLM
# ---------------------------------------------------------------------------

class VLLMBackend(BaseBackend):
    """Thin httpx client for the vLLM OpenAI-compatible inference server.

    vLLM API reference: https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html

    LoRA adapters in vLLM
    ---------------------
    vLLM loads LoRA adapters at startup via the ``--lora-modules`` flag::

        python -m vllm.entrypoints.openai.api_server \\
            --model meta-llama/Meta-Llama-3-8B \\
            --lora-modules my-lora=/path/to/adapter \\
            --enable-lora

    After that, passing ``"model": "my-lora"`` in a chat completion request
    automatically activates the adapter.
    """

    def __init__(self, base_url: str = "http://localhost:8000") -> None:
        self.base_url = base_url.rstrip("/")

    def is_available(self) -> bool:
        """Return True if the vLLM server responds on the health endpoint."""
        try:
            r = httpx.get(f"{self.base_url}/health", timeout=_CONNECT_TIMEOUT)
            return r.status_code == 200
        except Exception:
            return False

    def generate(
        self,
        prompt: str,
        adapter: AdapterEntry,
        *,
        lora_name: str | None = None,
        system_prompt: str = "You are a helpful assistant.",
    ) -> str:
        """Generate text via vLLM's ``/v1/chat/completions`` endpoint.

        Parameters
        ----------
        prompt:
            The user message content.
        adapter:
            The selected ``AdapterEntry``; ``adapter.id`` is used as the
            ``model`` field (which vLLM maps to the LoRA name) unless
            overridden by ``lora_name``.
        lora_name:
            Override the vLLM model/lora name.
        system_prompt:
            System message prepended before the user message.
        """
        # Explicit override wins; otherwise use the backend-effective name
        # (runtime_name when set, else adapter.id).
        model = lora_name or effective_backend_name(adapter)
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
        }

        logger.debug("vLLM generate: model=%s", model)
        try:
            r = httpx.post(
                f"{self.base_url}/v1/chat/completions",
                json=payload,
                timeout=_READ_TIMEOUT,
            )
            r.raise_for_status()
        except httpx.HTTPError as exc:
            raise BackendError(f"vLLM request failed: {exc}") from exc

        data = r.json()
        try:
            return data["choices"][0]["message"]["content"]
        except (KeyError, IndexError) as exc:
            raise BackendError(f"Unexpected vLLM response format: {data}") from exc

    def list_loaded_adapters(self) -> list[str]:
        """Return all model/LoRA ids served by vLLM (``GET /v1/models``).

        The ``data`` array lists the base model plus every ``--lora-modules``
        key.  Silently returns ``[]`` if vLLM is unreachable.
        """
        try:
            r = httpx.get(f"{self.base_url}/v1/models", timeout=_CONNECT_TIMEOUT)
            r.raise_for_status()
            data = r.json().get("data", [])
            return [m["id"] for m in data if "id" in m]
        except Exception as exc:
            logger.debug("vLLM list_loaded_adapters failed: %s", exc)
            return []


# ---------------------------------------------------------------------------
# Cerebras (cloud, OpenAI-compatible)
# ---------------------------------------------------------------------------

class CerebrasBackend(BaseBackend):
    """Thin httpx client for the Cerebras cloud inference API.

    Cerebras exposes an OpenAI-compatible API at ``https://api.cerebras.ai/v1``
    and authenticates with a bearer token read from the ``CEREBRAS_API_KEY``
    environment variable (or passed explicitly to the constructor).

    LoRA adapters on Cerebras
    -------------------------
    Today shiftgate routes to Cerebras' base-model inference: the ``model``
    field is set to ``effective_backend_name(adapter)``.  When Cerebras
    Multi-LoRA becomes public, register your adapter with
    ``--runtime <cerebras-lora-id>`` and routing works unchanged.
    """

    def __init__(self, api_key: str | None = None) -> None:
        self.base_url = "https://api.cerebras.ai/v1"
        self.api_key = api_key or os.getenv("CEREBRAS_API_KEY")

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}"}

    def openai_base_url(self) -> str:
        # Cerebras' base_url already includes the /v1 prefix.
        return self.base_url

    def auth_headers(self) -> dict[str, str]:
        return self._headers()

    def is_available(self) -> bool:
        """Return True only if an API key is set and ``/models`` returns 200.

        No network call is made when the API key is unset.
        """
        if not self.api_key:
            return False
        try:
            r = httpx.get(
                f"{self.base_url}/models",
                headers=self._headers(),
                timeout=_CONNECT_TIMEOUT,
            )
            return r.status_code == 200
        except Exception:
            return False

    def generate(
        self,
        prompt: str,
        adapter: AdapterEntry,
        *,
        lora_name: str | None = None,
        system_prompt: str = "You are a helpful assistant.",
    ) -> str:
        """Generate text via Cerebras' ``/chat/completions`` endpoint.

        Mirrors :meth:`VLLMBackend.generate`: an explicit ``lora_name`` wins,
        otherwise the adapter's effective backend name is used as ``model``.
        """
        model = lora_name or effective_backend_name(adapter)
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
        }

        logger.debug("Cerebras generate: model=%s", model)
        try:
            r = httpx.post(
                f"{self.base_url}/chat/completions",
                json=payload,
                headers=self._headers(),
                timeout=_READ_TIMEOUT,
            )
            r.raise_for_status()
        except httpx.HTTPError as exc:
            raise BackendError(f"Cerebras request failed: {exc}") from exc

        data = r.json()
        try:
            return data["choices"][0]["message"]["content"]
        except (KeyError, IndexError) as exc:
            raise BackendError(f"Unexpected Cerebras response format: {data}") from exc

    def list_loaded_adapters(self) -> list[str]:
        """Return all model ids served by Cerebras (``GET /models``).

        Silently returns ``[]`` if the API key is unset or Cerebras is
        unreachable.
        """
        if not self.api_key:
            return []
        try:
            r = httpx.get(
                f"{self.base_url}/models",
                headers=self._headers(),
                timeout=_CONNECT_TIMEOUT,
            )
            r.raise_for_status()
            data = r.json().get("data", [])
            return [m["id"] for m in data if "id" in m]
        except Exception as exc:
            logger.debug("Cerebras list_loaded_adapters failed: %s", exc)
            return []


# ---------------------------------------------------------------------------
# Cloudflare Workers AI (cloud, NOT OpenAI-compatible)
# ---------------------------------------------------------------------------

class CloudflareBackend(BaseBackend):
    """Thin httpx client for Cloudflare Workers AI inference with LoRA.

    Cloudflare's API is architecturally different from vLLM/Cerebras:

    * The **base model** lives in the URL path
      (``/ai/run/@cf/mistral/mistral-7b-instruct-v0.2-lora``).
    * The **LoRA name** is a separate ``lora`` field in the request body — not
      the ``model`` value.
    * The response is wrapped in ``{"result": {"response": ...}, "success": ...}``
      and is NOT OpenAI-compatible, so it needs translation.

    Auth uses ``Authorization: Bearer {CLOUDFLARE_API_TOKEN}`` and requires a
    ``CLOUDFLARE_ACCOUNT_ID``.  Both can be passed explicitly or read from env.

    Reference: https://developers.cloudflare.com/workers-ai/
    """

    # Cloudflare has a bespoke request/response shape — not OpenAI-compatible.
    is_openai_compatible: bool = False

    def __init__(
        self,
        account_id: str | None = None,
        api_token: str | None = None,
    ) -> None:
        self.account_id = account_id or os.getenv("CLOUDFLARE_ACCOUNT_ID")
        self.api_token = api_token or os.getenv("CLOUDFLARE_API_TOKEN")
        self.base_url = (
            f"https://api.cloudflare.com/client/v4/accounts/{self.account_id}/ai"
        )

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.api_token}"}

    def is_available(self) -> bool:
        """Return True only if both credentials are set and ``/finetunes`` 200s.

        No network call is made when either credential is missing.
        """
        if not (self.account_id and self.api_token):
            return False
        try:
            r = httpx.get(
                f"{self.base_url}/finetunes",
                headers=self._headers(),
                timeout=_CONNECT_TIMEOUT,
            )
            return r.status_code == 200
        except Exception:
            return False

    def list_loaded_adapters(self) -> list[str]:
        """Return finetune names available on the account (private + public).

        Merges ``/finetunes`` (user uploads) with ``/finetunes/public``
        (Cloudflare-hosted LoRAs like ``cf-public-magicoder``).  Cloudflare's
        ``result`` may be a flat list or a list-of-lists; both are handled.
        Silently returns ``[]`` if credentials are missing or the API is
        unreachable.
        """
        if not (self.account_id and self.api_token):
            return []
        names: list[str] = []
        for path in ("/finetunes", "/finetunes/public"):
            try:
                r = httpx.get(
                    f"{self.base_url}{path}",
                    headers=self._headers(),
                    timeout=_CONNECT_TIMEOUT,
                )
                r.raise_for_status()
                names.extend(self._extract_finetune_names(r.json().get("result", [])))
            except Exception as exc:
                logger.debug("Cloudflare list_loaded_adapters %s failed: %s", path, exc)
        # Preserve order while deduplicating (private upload wins over public).
        return list(dict.fromkeys(names))

    @staticmethod
    def _extract_finetune_names(result) -> list[str]:
        """Pull ``name`` fields from a flat list or a list-of-lists of finetunes."""
        if not isinstance(result, list):
            return []
        # Flatten one level if Cloudflare wrapped the list in result[0].
        items: list = []
        for entry in result:
            if isinstance(entry, list):
                items.extend(entry)
            else:
                items.append(entry)
        return [it["name"] for it in items if isinstance(it, dict) and "name" in it]

    def generate(
        self,
        prompt: str,
        adapter: AdapterEntry,
        *,
        lora_name: str | None = None,
        system_prompt: str = "You are a helpful assistant.",
    ) -> str:
        """Generate text via Cloudflare's ``/ai/run/{base_model}`` endpoint.

        The base model comes from ``adapter.base_model`` and MUST be a
        Cloudflare-prefixed model name (``@cf/...``).  The LoRA name is sent as
        the ``lora`` body field (an explicit ``lora_name`` override wins).
        """
        base_model = (adapter.base_model or "").strip()
        if not base_model.startswith("@cf/"):
            raise BackendError(
                "Cloudflare backend requires a Cloudflare base model name. "
                f"Adapter '{adapter.id}' has base_model='{adapter.base_model or ''}'. "
                "Re-register it with a Cloudflare model, e.g.:\n"
                f"  shiftgate adapter add {adapter.id} --runtime {effective_backend_name(adapter)} "
                "--base @cf/mistral/mistral-7b-instruct-v0.2-lora --tags <task>"
            )

        # `lora` is only sent when the adapter names an actual finetune
        # (explicit lora_name override, or a non-empty runtime_name).  When no
        # finetune is named the request runs the base model directly — the
        # same as calling Cloudflare with no `lora` field.
        lora = lora_name or ((adapter.runtime_name or "").strip() or None)
        payload: dict = {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            "raw": True,
        }
        if lora:
            payload["lora"] = lora

        logger.debug(
            "Cloudflare generate: base_model=%s lora=%s", base_model, lora or "(base model)"
        )
        try:
            r = httpx.post(
                f"{self.base_url}/run/{base_model}",
                json=payload,
                headers=self._headers(),
                timeout=_READ_TIMEOUT,
            )
            r.raise_for_status()
        except httpx.HTTPError as exc:
            raise BackendError(f"Cloudflare request failed: {exc}") from exc

        data = r.json()
        if not data.get("success", False):
            errors = data.get("errors", [])
            raise BackendError(f"Cloudflare inference failed: {errors}")
        try:
            return data["result"]["response"]
        except (KeyError, TypeError) as exc:
            raise BackendError(f"Unexpected Cloudflare response format: {data}") from exc


# ---------------------------------------------------------------------------
# BackendRouter — auto-detects which backend is live
# ---------------------------------------------------------------------------

class BackendRouter:
    """Detects and delegates to whichever backend is running.

    Priority: Ollama → vLLM → Cerebras → Cloudflare (local backends first,
    cloud as fallback).  If none is available, calls to ``generate`` raise
    ``NoBackendError`` with a helpful message.
    """

    def __init__(
        self,
        ollama_url: str = "http://localhost:11434",
        vllm_url: str = "http://localhost:8000",
        cerebras_api_key: str | None = None,
        cloudflare_account_id: str | None = None,
        cloudflare_api_token: str | None = None,
    ) -> None:
        self._ollama = OllamaBackend(ollama_url)
        self._vllm = VLLMBackend(vllm_url)
        self._cerebras = CerebrasBackend(cerebras_api_key)
        self._cloudflare = CloudflareBackend(
            cloudflare_account_id, cloudflare_api_token
        )
        self._active: BaseBackend | None = None

    def detect(self) -> str | None:
        """Probe backends and return the name of the first that responds.

        Returns ``"ollama"``, ``"vllm"``, ``"cerebras"``, ``"cloudflare"``, or
        ``None``.
        """
        if self._ollama.is_available():
            self._active = self._ollama
            return "ollama"
        if self._vllm.is_available():
            self._active = self._vllm
            return "vllm"
        if self._cerebras.is_available():
            self._active = self._cerebras
            return "cerebras"
        if self._cloudflare.is_available():
            self._active = self._cloudflare
            return "cloudflare"
        self._active = None
        return None

    def select(self, name: str | None) -> str | None:
        """Force-select a backend by name, or auto-detect when ``None``/``auto``.

        Unlike :meth:`detect`, an explicitly named backend is activated even if
        an availability ping would fail — the caller asked for it specifically.
        Returns the active backend name (or ``None`` if auto-detect found none).
        """
        if name in (None, "", "auto"):
            return self.detect()
        mapping = {
            "ollama": self._ollama,
            "vllm": self._vllm,
            "cerebras": self._cerebras,
            "cloudflare": self._cloudflare,
        }
        if name not in mapping:
            raise ValueError(
                f"Unknown backend '{name}'. "
                "Choose ollama, vllm, cerebras, cloudflare, or auto."
            )
        self._active = mapping[name]
        return name

    @property
    def active_backend(self) -> BaseBackend | None:
        """Return the currently active backend instance (or None)."""
        return self._active

    def generate(self, prompt: str, adapter: AdapterEntry) -> str:
        """Route the prompt to the active backend.

        If no backend was detected yet, ``detect()`` is called automatically.

        Raises
        ------
        NoBackendError
            If neither Ollama nor vLLM is reachable.
        """
        if self._active is None:
            self.detect()
        if self._active is None:
            raise NoBackendError(
                "No inference backend detected.\n"
                "  • Start Ollama  : ollama serve\n"
                "  • Start vLLM    : python -m vllm.entrypoints.openai.api_server "
                "--model <base_model> --enable-lora\n"
                "  • Use Cerebras  : export CEREBRAS_API_KEY=csk-...\n"
                "  • Use Cloudflare: export CLOUDFLARE_ACCOUNT_ID=... CLOUDFLARE_API_TOKEN=...\n\n"
                "shiftgate can route queries without a backend. "
                "Use `shiftgate route` to see routing decisions without inference."
            )
        return self._active.generate(prompt, adapter)

    @property
    def active_backend_name(self) -> str | None:
        """Return 'ollama', 'vllm', 'cerebras', 'cloudflare', or None."""
        if self._active is self._ollama:
            return "ollama"
        if self._active is self._vllm:
            return "vllm"
        if self._active is self._cerebras:
            return "cerebras"
        if self._active is self._cloudflare:
            return "cloudflare"
        return None

    @property
    def active_backend_url(self) -> str | None:
        """Return the base URL of the active backend, or None."""
        if self._active is not None:
            return self._active.base_url
        return None

    def verify_adapter(self, adapter: AdapterEntry) -> tuple[bool, str | None]:
        """Check whether an adapter is actually loaded in the active backend.

        Auto-detects a backend if one hasn't been probed yet.

        Returns
        -------
        ``(is_loaded, backend_name)``
            - ``(True, "<name>")``  — backend running and the adapter's
              effective name is present in its loaded model list.
            - ``(False, "<name>")`` — backend running but the name is absent.
            - ``(False, None)``     — no backend reachable (verification skipped).

        Never raises: HTTP failures degrade to ``(False, None)``.
        """
        if self._active is None:
            self.detect()
        if self._active is None:
            return (False, None)

        # Cloudflare base models are always available without any upload.  A
        # Cloudflare adapter with a @cf/ base model and no finetune runtime name
        # is therefore always runnable (base-model inference).
        if isinstance(self._active, CloudflareBackend):
            is_cf_base = (adapter.base_model or "").startswith("@cf/")
            has_finetune = bool((adapter.runtime_name or "").strip())
            if is_cf_base and not has_finetune:
                return (True, "cloudflare")

        target = effective_backend_name(adapter)
        loaded = self._active.list_loaded_adapters()
        return (target in loaded, self.active_backend_name)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class BackendError(RuntimeError):
    """Raised when an inference backend returns an error or unexpected response."""


class NoBackendError(RuntimeError):
    """Raised when no local inference backend is reachable."""
