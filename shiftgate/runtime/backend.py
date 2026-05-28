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
from abc import ABC, abstractmethod

import httpx

from shiftgate.registry.schemas import AdapterEntry

logger = logging.getLogger(__name__)

# Default timeouts (seconds).
_CONNECT_TIMEOUT = 3.0
_READ_TIMEOUT = 120.0


class BaseBackend(ABC):
    """Abstract base for inference backends."""

    @abstractmethod
    def is_available(self) -> bool:
        """Return True if the backend can be reached."""

    @abstractmethod
    def generate(self, prompt: str, adapter: AdapterEntry) -> str:
        """Send ``prompt`` to the backend and return the generated text."""


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
        model = model_name or adapter.id
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
        model = lora_name or adapter.id
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


# ---------------------------------------------------------------------------
# BackendRouter — auto-detects which backend is live
# ---------------------------------------------------------------------------

class BackendRouter:
    """Detects and delegates to whichever local backend is running.

    Priority: Ollama first, then vLLM.  If neither is available, calls to
    ``generate`` raise ``NoBackendError`` with a helpful message.
    """

    def __init__(
        self,
        ollama_url: str = "http://localhost:11434",
        vllm_url: str = "http://localhost:8000",
    ) -> None:
        self._ollama = OllamaBackend(ollama_url)
        self._vllm = VLLMBackend(vllm_url)
        self._active: BaseBackend | None = None

    def detect(self) -> str | None:
        """Probe both backends and return the name of the one that responds.

        Returns ``"ollama"``, ``"vllm"``, or ``None``.
        """
        if self._ollama.is_available():
            self._active = self._ollama
            return "ollama"
        if self._vllm.is_available():
            self._active = self._vllm
            return "vllm"
        self._active = None
        return None

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
                "  • Start Ollama : ollama serve\n"
                "  • Start vLLM  : python -m vllm.entrypoints.openai.api_server "
                "--model <base_model> --enable-lora\n\n"
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
        return None


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class BackendError(RuntimeError):
    """Raised when an inference backend returns an error or unexpected response."""


class NoBackendError(RuntimeError):
    """Raised when no local inference backend is reachable."""
