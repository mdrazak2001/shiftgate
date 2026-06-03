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
    available_runtimes: set[str] | None = None,
) -> MatchResult:
    """Select the adapter linked to the best-matching task.

    Strategy
    --------
    For each top task (highest score first), walk ``preferred_adapters`` then
    ``fallback_adapters`` and collect the adapters that exist in the registry
    (populating ``TaskMatch.candidate_adapters`` for the ``--explain`` view).
    The first viable adapter found, on the highest-scoring task, is selected.

    Backend-aware filtering
    -----------------------
    When ``available_runtimes`` is provided (the set of model/adapter names
    actually loaded on the active backend), only adapters whose
    ``effective_backend_name()`` is in that set are considered viable.  If a
    task's entire candidate list is filtered out, selection falls through to
    the next-best task.  When ``available_runtimes`` is ``None`` no filtering
    happens (the preview behaviour used by ``shiftgate route``).

    No silent fallback
    ------------------
    If the matched (top) task has **no** linked adapter in the registry, the
    router must NOT substitute an arbitrary adapter — doing so silently routes,
    e.g., a music query to a SQL adapter and destroys trust.  Instead this
    returns a ``MatchResult`` with ``selected_adapter=None``.

    Parameters
    ----------
    top_tasks:
        Output of ``top_k_tasks`` (sorted by score descending).
    adapter_registry:
        ``AdapterRegistry`` instance to look up adapter IDs.
    available_runtimes:
        Optional set of runtime names loaded on the active backend.  When set,
        adapters not in the set are skipped during selection.

    Returns
    -------
    ``MatchResult``.  ``selected_adapter`` is ``None`` when no viable adapter is
    found.  ``selection_method`` is ``"no_adapter_on_active_backend"`` when
    linked adapters exist but none are loaded on the active backend, otherwise
    ``"no_adapter_for_task"``.  The ``matched_task`` is always the top-scoring
    task so callers can still report what was matched.
    """
    def _is_viable(adapter) -> bool:
        if available_runtimes is None:
            return True
        return adapter.effective_backend_name() in available_runtimes

    explicit_result: MatchResult | None = None
    any_linked_adapter = False  # any task had at least one registered adapter

    for tm in top_tasks:
        preferred_ids = list(tm.task.preferred_adapters)
        fallback_ids = list(tm.task.fallback_adapters)

        # Populate candidate_adapters with every registered adapter (for the
        # --explain view, showing all candidates regardless of runtime).
        for adapter_id in preferred_ids + fallback_ids:
            adapter = adapter_registry.get_adapter(adapter_id)
            if adapter is not None and adapter not in tm.candidate_adapters:
                tm.candidate_adapters.append(adapter)

        if tm.candidate_adapters:
            any_linked_adapter = True

        if explicit_result is None:
            viable = [a for a in tm.candidate_adapters if _is_viable(a)]
            if viable:
                chosen = viable[0]
                method = (
                    "preferred"
                    if chosen.id in tm.task.preferred_adapters
                    else "fallback"
                )
                explicit_result = MatchResult(
                    selected_adapter=chosen,
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

    # No viable adapter across any ranked task. Distinguish "nothing linked at
    # all" from "linked but not loaded on the active backend".
    top_task = top_tasks[0]
    if available_runtimes is not None and any_linked_adapter:
        method = "no_adapter_on_active_backend"
        logger.info(
            "Linked adapter(s) for task '%s' exist but none are loaded on the "
            "active backend — refusing to guess.",
            top_task.task.id,
        )
    else:
        method = "no_adapter_for_task"
        logger.info(
            "No linked adapter for matched task '%s' — refusing to guess.",
            top_task.task.id,
        )

    return MatchResult(
        selected_adapter=None,
        matched_task=top_task.task,
        similarity_score=top_task.score,
        all_task_matches=top_tasks,
        selection_method=method,
    )


class NoAdapterError(RuntimeError):
    """Raised when the matcher cannot find any registered adapter for a query.

    Retained for backward compatibility; ``select_adapter`` no longer raises it
    (it returns a ``MatchResult`` with ``selected_adapter=None`` instead).
    """
