"""
Main routing orchestrator: query string → RoutingTrace.

This module ties together the embedder, matcher, and registries.
It is the single function that CLI commands and the runtime backend call.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone

from shiftgate.registry.adapter_registry import AdapterRegistry
from shiftgate.registry.schemas import RoutingTrace
from shiftgate.registry.task_registry import TaskRegistry
from shiftgate.router.embedder import Embedder
from shiftgate.router.matcher import MatchResult, select_adapter, top_k_tasks

logger = logging.getLogger(__name__)


def route(
    query: str,
    task_registry: TaskRegistry,
    adapter_registry: AdapterRegistry,
    embedder: Embedder,
    top_k: int = 3,
    available_runtimes: set[str] | None = None,
) -> tuple[RoutingTrace, MatchResult]:
    """Route a query string to the best matching adapter.

    Steps
    -----
    1. Embed the query with the frozen embedding model.
    2. Compute cosine similarity against all task centroid embeddings.
    3. Select the highest-ranked task whose preferred adapters exist.
    4. Build and return a ``RoutingTrace`` and the full ``MatchResult``.

    Parameters
    ----------
    query:
        The user's natural-language query or instruction.
    task_registry:
        Loaded ``TaskRegistry`` with pre-computed centroids.
    adapter_registry:
        Loaded ``AdapterRegistry``.
    embedder:
        ``Embedder`` instance (wraps fastembed singleton).
    top_k:
        Number of top task candidates to consider.  Defaults to 3.
    available_runtimes:
        Optional set of runtime names loaded on the active backend.  When set,
        adapters whose ``effective_backend_name()`` is not in the set are
        skipped, falling through to the next-best task.  If no viable adapter
        is found across all top-K tasks, the trace's ``selected_adapter_id`` is
        ``None`` and ``selection_method`` is ``"no_adapter_on_active_backend"``.

    Returns
    -------
    ``(RoutingTrace, MatchResult)`` — the trace for persistence/feedback and
    the full match result for detailed display (e.g. ``--explain``).

    Note
    ----
    When the matched task has no linked adapter, ``MatchResult.selected_adapter``
    is ``None`` and the trace's ``selected_adapter_id`` is ``None``.  The router
    never substitutes an arbitrary adapter.

    Raises
    ------
    ValueError
        If embeddings have not been computed (missing centroids).
    """
    if not task_registry.embeddings_ready():
        raise ValueError(
            "Task embeddings are not initialised. Run `shiftgate init` first."
        )

    query_embedding = embedder.embed(query)
    all_tasks = task_registry.get_all_tasks()
    ranked = top_k_tasks(query_embedding, all_tasks, k=top_k)

    if available_runtimes is not None:
        logger.debug(
            "filtering adapters to backend runtimes: %s",
            sorted(available_runtimes),
        )
    else:
        logger.debug("no active backend — adapter runtime filtering disabled")

    result = select_adapter(ranked, adapter_registry, available_runtimes=available_runtimes)

    selected_id = result.selected_adapter.id if result.selected_adapter else None

    trace = RoutingTrace(
        id=uuid.uuid4().hex,
        query=query,
        matched_task_id=result.matched_task.id,
        similarity_score=result.similarity_score,
        selected_adapter_id=selected_id,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )

    logger.info(
        "Routed '%s' → task='%s' (%.2f%%) → adapter='%s' [%s]",
        query[:60],
        result.matched_task.id,
        result.similarity_score * 100,
        selected_id or "<none>",
        result.selection_method,
    )
    return trace, result
