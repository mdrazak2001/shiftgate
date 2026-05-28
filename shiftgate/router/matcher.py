"""
Cosine-similarity matcher: maps query embeddings to task clusters and adapters.

This module is deliberately stateless — all context (registries, embeddings)
is passed explicitly so the functions are easy to test in isolation.
"""

from __future__ import annotations

import logging

import numpy as np

from shiftgate.registry.schemas import AdapterEntry, TaskCluster

logger = logging.getLogger(__name__)


def top_k_tasks(
    query_embedding: np.ndarray,
    task_clusters: list[TaskCluster],
    k: int = 3,
) -> list[tuple[TaskCluster, float]]:
    """Return the top-K task clusters by cosine similarity to the query.

    Parameters
    ----------
    query_embedding:
        1-D float32 array of shape ``(dim,)``.  Need not be L2-normalised;
        this function normalises internally.
    task_clusters:
        All clusters in the registry.  Clusters without a computed centroid
        are silently skipped.
    k:
        Number of top clusters to return.

    Returns
    -------
    list of ``(TaskCluster, score)`` pairs sorted by score descending.
    """
    eligible = [t for t in task_clusters if t.embedding_centroid is not None]
    if not eligible:
        raise ValueError(
            "No task cluster has a computed embedding centroid. "
            "Run `shiftgate init` to compute embeddings."
        )

    # Stack centroids into a matrix for vectorised dot product.
    centroid_matrix = np.array(
        [t.embedding_centroid for t in eligible], dtype=np.float32
    )  # shape: (n_tasks, dim)

    # L2-normalise the query vector.
    q_norm = np.linalg.norm(query_embedding)
    if q_norm == 0:
        raise ValueError("Query produced a zero-norm embedding.")
    q_unit = query_embedding / q_norm

    # Cosine similarity = dot(q_unit, centroid_unit) because centroids were
    # already L2-normalised at compute time (see task_registry.py).
    scores = centroid_matrix @ q_unit  # shape: (n_tasks,)

    # Grab top-K indices.
    k = min(k, len(eligible))
    top_indices = np.argsort(scores)[::-1][:k]

    return [(eligible[i], float(scores[i])) for i in top_indices]


def select_adapter(
    top_tasks: list[tuple[TaskCluster, float]],
    adapter_registry,  # AdapterRegistry — avoid circular import with string hint
) -> tuple[AdapterEntry, TaskCluster, float]:
    """Select the best adapter given the ranked task list.

    Strategy:
      1. Iterate top tasks in similarity order.
      2. For each task, try ``preferred_adapters`` then ``fallback_adapters``.
      3. Return the first adapter that exists in the registry.
      4. If no registered adapter matches any task, raise ``NoAdapterError``.

    Parameters
    ----------
    top_tasks:
        Output of ``top_k_tasks`` — list of (TaskCluster, score) descending.
    adapter_registry:
        ``AdapterRegistry`` instance to look up adapter IDs.

    Returns
    -------
    ``(AdapterEntry, TaskCluster, similarity_score)``
    """
    for task, score in top_tasks:
        candidates = list(task.preferred_adapters) + list(task.fallback_adapters)
        for adapter_id in candidates:
            adapter = adapter_registry.get_adapter(adapter_id)
            if adapter is not None:
                logger.debug(
                    "Selected adapter '%s' via task '%s' (score=%.4f)",
                    adapter.id,
                    task.id,
                    score,
                )
                return adapter, task, score

    # No adapter matched — surface a helpful error.
    task_ids = [t.id for t, _ in top_tasks]
    raise NoAdapterError(
        f"No registered adapter found for tasks {task_ids}. "
        "Add adapters with `shiftgate adapter add <hf_repo>`."
    )


class NoAdapterError(RuntimeError):
    """Raised when the matcher cannot find any registered adapter for a query."""
