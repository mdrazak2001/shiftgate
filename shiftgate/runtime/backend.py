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
# BackendRouter — auto-detects which backend is live
# ---------------------------------------------------------------------------

class BackendRouter:
    """Detects and delegates to whichever backend is running.

    Priority: Ollama → vLLM → Cerebras (local backends first, cloud as
    fallback).  If none is available, calls to ``generate`` raise
    ``NoBackendError`` with a helpful message.
    """

    def __init__(
        self,
        ollama_url: str = "http://localhost:11434",
        vllm_url: str = "http://localhost:8000",
        cerebras_api_key: str | None = None,
    ) -> None:
        self._ollama = OllamaBackend(ollama_url)
        self._vllm = VLLMBackend(vllm_url)
        self._cerebras = CerebrasBackend(cerebras_api_key)
        self._active: BaseBackend | None = None

    def detect(self) -> str | None:
        """Probe backends and return the name of the first that responds.

        Returns ``"ollama"``, ``"vllm"``, ``"cerebras"``, or ``None``.
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
        }
        if name not in mapping:
            raise ValueError(f"Unknown backend '{name}'. Choose ollama, vllm, cerebras, or auto.")
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
                "  • Use Cerebras  : export CEREBRAS_API_KEY=csk-...\n\n"
                "shiftgate can route queries without a backend. "
                "Use `shiftgate route` to see routing decisions without inference."
            )
        return self._active.generate(prompt, adapter)

    @property
    def active_backend_name(self) -> str | None:
        """Return 'ollama', 'vllm', or None depending on what was detected."""
        if self._active is self._ollama:
            return "ollama"
        if self._active is self._vllm:
            return "vllm"
        if self._active is self._cerebras:
            return "cerebras"
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
