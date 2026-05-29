"""
Cosine-similarity matcher: maps query embeddings to task clusters and adapters.

This module is deliberately stateless — all context (registries, embeddings)
is passed explicitly so the functions are easy to test in isolation.

Public surface
--------------
top_k_tasks        — rank task clusters by cosine similarity to a query vector
select_adapter     — pick the best registered adapter from a ranked task list
MatchResult        — structured result used by the router and the --explain view
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from shiftgate.registry.schemas import AdapterEntry, TaskCluster

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class TaskMatch:
    """One task cluster paired with its similarity score."""

    task: TaskCluster
    score: float
    # Adapters from this task that exist in the registry (populated by select_adapter)
    candidate_adapters: list[AdapterEntry] = field(default_factory=list)


@dataclass
class MatchResult:
    """Full structured result of a routing decision.

    Returned by ``select_adapter`` so that both the router and the
    ``--explain`` display have access to the complete decision tree.

    ``selected_adapter`` is ``None`` when the matched task has no adapter
    linked in the registry.  In that case ``selection_method`` is
    ``"no_adapter_for_task"`` and the router must NOT run inference.
    """

    selected_adapter: AdapterEntry | None
    matched_task: TaskCluster
    similarity_score: float
    # All top-K task matches including their candidate adapter lists
    all_task_matches: list[TaskMatch] = field(default_factory=list)
    # How the adapter was found: "preferred", "fallback", or "no_adapter_for_task"
    selection_method: str = "preferred"

    @property
    def has_adapter(self) -> bool:
        """True when an adapter was successfully selected."""
        return self.selected_adapter is not None


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def top_k_tasks(
    query_embedding: np.ndarray,
    task_clusters: list[TaskCluster],
    k: int = 3,
) -> list[TaskMatch]:
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
    List of ``TaskMatch`` objects sorted by score descending.
    """
    eligible = [t for t in task_clusters if t.embedding_centroid is not None]
    if not eligible:
        raise ValueError(
            "No task cluster has a computed embedding centroid. "
            "Run `shiftgate init` to compute embeddings."
        )

    centroid_matrix = np.array(
        [t.embedding_centroid for t in eligible], dtype=np.float32
    )  # shape: (n_tasks, dim)

    q_norm = np.linalg.norm(query_embedding)
    if q_norm == 0:
        raise ValueError("Query produced a zero-norm embedding.")
    q_unit = query_embedding / q_norm

    # Centroids were L2-normalised at compute time, so this is cosine similarity.
    scores = centroid_matrix @ q_unit  # shape: (n_tasks,)

    k = min(k, len(eligible))
    top_indices = np.argsort(scores)[::-1][:k]

    return [TaskMatch(task=eligible[i], score=float(scores[i])) for i in top_indices]


def select_adapter(
    top_tasks: list[TaskMatch],
    adapter_registry,  # AdapterRegistry — avoid circular import with string hint
) -> MatchResult:
    """Select the adapter linked to the best-matching task.

    Strategy
    --------
    For each top task (highest score first), walk ``preferred_adapters`` then
    ``fallback_adapters`` and collect the adapters that exist in the registry
    (populating ``TaskMatch.candidate_adapters`` for the ``--explain`` view).
    The first such adapter found, on the highest-scoring task, is selected.

    No silent fallback
    ------------------
    If the matched (top) task has **no** linked adapter in the registry, the
    router must NOT substitute an arbitrary adapter — doing so silently routes,
    e.g., a music query to a SQL adapter and destroys trust.  Instead this
    returns a ``MatchResult`` with ``selected_adapter=None`` and
    ``selection_method="no_adapter_for_task"``.

    Parameters
    ----------
    top_tasks:
        Output of ``top_k_tasks`` (sorted by score descending).
    adapter_registry:
        ``AdapterRegistry`` instance to look up adapter IDs.

    Returns
    -------
    ``MatchResult``.  ``selected_adapter`` is ``None`` when no adapter is
    linked to any of the ranked tasks.  The ``matched_task`` is always the
    top-scoring task so callers can still report what was matched.
    """
    # Populate candidate lists for every task (for the --explain view) and
    # find the first explicit match in score order.
    explicit_result: MatchResult | None = None

    for tm in top_tasks:
        preferred_ids = list(tm.task.preferred_adapters)
        fallback_ids = list(tm.task.fallback_adapters)

        for adapter_id in preferred_ids + fallback_ids:
            adapter = adapter_registry.get_adapter(adapter_id)
            if adapter is not None and adapter not in tm.candidate_adapters:
                tm.candidate_adapters.append(adapter)

        if explicit_result is None and tm.candidate_adapters:
            method = (
                "preferred"
                if tm.candidate_adapters[0].id in tm.task.preferred_adapters
                else "fallback"
            )
            explicit_result = MatchResult(
                selected_adapter=tm.candidate_adapters[0],
                matched_task=tm.task,
                similarity_score=tm.score,
                all_task_matches=top_tasks,
                selection_method=method,
            )

    if explicit_result is not None:
        logger.debug(
            "Selected adapter '%s' via task '%s' (score=%.4f, method=%s)",
            explicit_result.selected_adapter.id,
            explicit_result.matched_task.id,
            explicit_result.similarity_score,
            explicit_result.selection_method,
        )
        return explicit_result

    # No adapter linked to any ranked task — do NOT guess. Report the matched
    # task with no adapter so the caller can prompt the user to add one.
    top_task = top_tasks[0]
    logger.info(
        "No linked adapter for matched task '%s' — refusing to guess.",
        top_task.task.id,
    )
    return MatchResult(
        selected_adapter=None,
        matched_task=top_task.task,
        similarity_score=top_task.score,
        all_task_matches=top_tasks,
        selection_method="no_adapter_for_task",
    )


class NoAdapterError(RuntimeError):
    """Raised when the matcher cannot find any registered adapter for a query.

    Retained for backward compatibility; ``select_adapter`` no longer raises it
    (it returns a ``MatchResult`` with ``selected_adapter=None`` instead).
    """
