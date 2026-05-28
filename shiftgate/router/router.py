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
from shiftgate.router.matcher import NoAdapterError, select_adapter, top_k_tasks

logger = logging.getLogger(__name__)


def route(
    query: str,
    task_registry: TaskRegistry,
    adapter_registry: AdapterRegistry,
    embedder: Embedder,
    top_k: int = 3,
) -> RoutingTrace:
    """Route a query string to the best matching adapter.

    Steps
    -----
    1. Embed the query with the frozen embedding model.
    2. Compute cosine similarity against all task centroid embeddings.
    3. Select the highest-ranked task whose preferred adapters exist.
    4. Build and return a ``RoutingTrace``.

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
        Number of top task candidates to consider when walking the fallback
        chain.  Defaults to 3.

    Returns
    -------
    A ``RoutingTrace`` describing the decision.  The trace is **not**
    persisted here — call ``feedback.loop.record_trace(trace)`` separately.

    Raises
    ------
    NoAdapterError
        If no registered adapter matches any of the top-K tasks.
    ValueError
        If embeddings have not been computed (missing centroids).
    """
    if not task_registry.embeddings_ready():
        raise ValueError(
            "Task embeddings are not initialised. Run `shiftgate init` first."
        )

    # Step 1: embed the query
    query_embedding = embedder.embed(query)

    # Step 2: rank tasks by similarity
    all_tasks = task_registry.get_all_tasks()
    ranked = top_k_tasks(query_embedding, all_tasks, k=top_k)

    # Step 3: pick the best adapter
    adapter, matched_task, score = select_adapter(ranked, adapter_registry)

    # Step 4: assemble trace
    trace = RoutingTrace(
        id=uuid.uuid4().hex,
        query=query,
        matched_task_id=matched_task.id,
        similarity_score=score,
        selected_adapter_id=adapter.id,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )

    logger.info(
        "Routed '%s' → task='%s' (%.2f%%) → adapter='%s'",
        query[:60],
        matched_task.id,
        score * 100,
        adapter.id,
    )
    return trace
