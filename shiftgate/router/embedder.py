"""
Text embedder backed by fastembed.

Uses ``BAAI/bge-small-en-v1.5`` — a compact (33 M param) model that runs
efficiently on CPU.  The model is downloaded once by fastembed and cached in
``~/.shiftgate/fastembed_cache`` (avoids Windows ``%TEMP%`` corruption issues).

A module-level singleton (``_MODEL``) is created lazily on first use so that
importing this module is cheap.  The model is NOT re-created between calls.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# Stable cache location — fastembed defaults to %TEMP% on Windows, which is
# prone to partial downloads and "file sizes do not match" corruption.
FASTEMBED_CACHE_DIR = Path.home() / ".shiftgate" / "fastembed_cache"

# -------------------------------------------------------------------------
# Default model — small, CPU-friendly, strong quality/speed trade-off.
# -------------------------------------------------------------------------
DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"

# HuggingFace downloads can flake; retry a few times before giving up.
_LOAD_RETRIES = 3
_LOAD_RETRY_DELAY_S = 2.0

# Module-level singleton; populated on first call to `_get_model()`.
_MODEL: Any | None = None


def _is_retryable_download_error(exc: BaseException) -> bool:
    """Return True for transient HuggingFace / network failures."""
    msg = str(exc).lower()
    needles = (
        "server disconnected",
        "connection reset",
        "connection aborted",
        "timed out",
        "timeout",
        "temporary failure",
        "503",
        "502",
        "429",
    )
    return any(n in msg for n in needles)


def _format_load_error(model_name: str, exc: BaseException) -> str:
    cache_hint = str(FASTEMBED_CACHE_DIR)
    if _is_retryable_download_error(exc):
        return (
            f"Failed to download embedding model '{model_name}' from HuggingFace: {exc}\n"
            "This is usually a transient network/rate-limit issue. Retry:\n"
            f"  uv run shiftgate init\n"
            "Optional: set HF_TOKEN for higher HuggingFace rate limits.\n"
            f"If downloads keep failing, delete '{cache_hint}' and retry."
        )
    return (
        f"Failed to load embedding model '{model_name}': {exc}\n"
        "If you see NO_SUCHFILE or 'file sizes do not match', delete the cache "
        "and retry:\n"
        f"  Remove-Item -Recurse -Force '{cache_hint}'\n"
        "Also clear any stale copy at $env:TEMP\\fastembed_cache, then run "
        "`shiftgate init` again."
    )


def _get_model(model_name: str = DEFAULT_MODEL) -> Any:
    """Return the fastembed TextEmbedding singleton, creating it if needed.

    The model is loaded once per process.  If you need a different model,
    call ``reset_model()`` first.
    """
    global _MODEL
    if _MODEL is None:
        try:
            from fastembed import TextEmbedding  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "fastembed is required for shiftgate routing. "
                "Install it with: pip install fastembed"
            ) from exc

        FASTEMBED_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        logger.info(
            "Loading embedding model '%s' (first use — one-time download may occur)…",
            model_name,
        )
        last_exc: BaseException | None = None
        for attempt in range(1, _LOAD_RETRIES + 1):
            try:
                _MODEL = TextEmbedding(
                    model_name=model_name,
                    cache_dir=str(FASTEMBED_CACHE_DIR),
                )
                break
            except Exception as exc:
                last_exc = exc
                if attempt < _LOAD_RETRIES and _is_retryable_download_error(exc):
                    logger.warning(
                        "Embedder load attempt %d/%d failed (%s); retrying…",
                        attempt,
                        _LOAD_RETRIES,
                        exc,
                    )
                    time.sleep(_LOAD_RETRY_DELAY_S * attempt)
                    continue
                raise RuntimeError(_format_load_error(model_name, exc)) from exc
        else:
            assert last_exc is not None
            raise RuntimeError(_format_load_error(model_name, last_exc)) from last_exc
        logger.info("Embedding model loaded.")
    return _MODEL


def warm_up(model_name: str = DEFAULT_MODEL) -> int:
    """Load the embedder and run a dummy embed. Returns embedding dimension."""
    vec = Embedder(model_name).embed("warmup")
    return int(vec.shape[0])


def reset_model() -> None:
    """Force the next embed call to recreate the model singleton.

    Useful in tests or when switching models at runtime.
    """
    global _MODEL
    _MODEL = None


class Embedder:
    """Thin wrapper around the fastembed TextEmbedding model.

    All embedding operations are synchronous and run on CPU.  The model
    is shared across all ``Embedder`` instances via the module-level singleton.
    """

    def __init__(self, model_name: str = DEFAULT_MODEL) -> None:
        self._model_name = model_name

    @property
    def _model(self) -> Any:
        return _get_model(self._model_name)

    def embed(self, text: str) -> np.ndarray:
        """Embed a single text string.

        Returns a 1-D float32 numpy array of shape ``(dim,)``.
        The vector is **not** L2-normalised here; normalisation is done
        where appropriate (e.g. when computing task centroids).
        """
        # fastembed returns a generator of numpy arrays.
        results = list(self._model.embed([text]))
        return np.array(results[0], dtype=np.float32)

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        """Embed a list of strings.

        Returns a 2-D float32 numpy array of shape ``(n, dim)`` where
        ``n = len(texts)``.
        """
        if not texts:
            raise ValueError("embed_batch received an empty list.")
        results = list(self._model.embed(texts))
        return np.array(results, dtype=np.float32)
