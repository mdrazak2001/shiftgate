"""
Text embedder backed by fastembed.

Uses ``BAAI/bge-small-en-v1.5`` — a compact (33 M param) model that runs
efficiently on CPU.  The model is downloaded once by fastembed and cached in
``~/.cache/fastembed``.

A module-level singleton (``_MODEL``) is created lazily on first use so that
importing this module is cheap.  The model is NOT re-created between calls.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# -------------------------------------------------------------------------
# Default model — small, CPU-friendly, strong quality/speed trade-off.
# -------------------------------------------------------------------------
DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"

# Module-level singleton; populated on first call to `_get_model()`.
_MODEL: Any | None = None


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

        logger.info("Loading embedding model '%s' (first use — one-time download may occur)…", model_name)
        _MODEL = TextEmbedding(model_name=model_name)
        logger.info("Embedding model loaded.")
    return _MODEL


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
