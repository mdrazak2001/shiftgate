"""
Tests for the routing pipeline: embedder, matcher, and router.

The embedder tests use the real fastembed model (skipped if not installed).
The matcher and router tests use pre-computed synthetic embeddings so they
are fast, deterministic, and require no model download.
"""

from __future__ import annotations

import numpy as np
import pytest

from shiftgate.registry.adapter_registry import AdapterRegistry
from shiftgate.registry.schemas import AdapterEntry, TaskCluster
from shiftgate.registry.task_registry import TaskRegistry
from shiftgate.router.matcher import NoAdapterError, select_adapter, top_k_tasks


# ---------------------------------------------------------------------------
# Helpers: build a synthetic task registry with known geometry
# ---------------------------------------------------------------------------

def _unit(v: list[float]) -> list[float]:
    """Return an L2-normalised version of v."""
    arr = np.array(v, dtype=np.float32)
    return (arr / np.linalg.norm(arr)).tolist()


def _make_task(task_id: str, centroid: list[float], adapter_ids: list[str]) -> TaskCluster:
    return TaskCluster(
        id=task_id,
        name=task_id.replace("_", " ").title(),
        description=f"Synthetic task {task_id}",
        validation_examples=["placeholder"],
        embedding_centroid=_unit(centroid),
        preferred_adapters=adapter_ids,
    )


def _make_adapter(adapter_id: str) -> AdapterEntry:
    return AdapterEntry(
        id=adapter_id,
        name=adapter_id.replace("-", " ").title(),
        base_model="test/base-model",
        task_tags=[],
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def synthetic_tasks() -> list[TaskCluster]:
    """
    Three tasks whose centroids are axis-aligned unit vectors in 3D space.
    This makes similarity scores fully predictable.
    """
    return [
        _make_task("task_x", [1, 0, 0], ["adapter-x"]),
        _make_task("task_y", [0, 1, 0], ["adapter-y"]),
        _make_task("task_z", [0, 0, 1], ["adapter-z"]),
    ]


@pytest.fixture()
def task_reg(synthetic_tasks, tmp_path) -> TaskRegistry:
    reg = TaskRegistry(tasks=synthetic_tasks, source_path=tmp_path / "tasks.json")
    return reg


@pytest.fixture()
def adapter_reg(tmp_path) -> AdapterRegistry:
    adapters = [
        _make_adapter("adapter-x"),
        _make_adapter("adapter-y"),
        _make_adapter("adapter-z"),
    ]
    return AdapterRegistry(adapters=adapters, source_path=tmp_path / "adapters.json")


# ---------------------------------------------------------------------------
# top_k_tasks tests
# ---------------------------------------------------------------------------

class TestTopKTasks:
    def test_closest_task_is_first(self, synthetic_tasks):
        """A query aligned with task_y should rank task_y first."""
        query_emb = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        ranked = top_k_tasks(query_emb, synthetic_tasks, k=3)
        assert ranked[0][0].id == "task_y"

    def test_scores_descending(self, synthetic_tasks):
        """Returned scores must be sorted highest first."""
        query_emb = np.array([0.6, 0.8, 0.0], dtype=np.float32)
        ranked = top_k_tasks(query_emb, synthetic_tasks, k=3)
        scores = [s for _, s in ranked]
        assert scores == sorted(scores, reverse=True)

    def test_top_k_limits_results(self, synthetic_tasks):
        query_emb = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        ranked = top_k_tasks(query_emb, synthetic_tasks, k=2)
        assert len(ranked) == 2

    def test_no_centroid_tasks_are_skipped(self, synthetic_tasks):
        """Tasks missing a centroid should not appear in results."""
        synthetic_tasks[0].embedding_centroid = None  # remove task_x centroid
        query_emb = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        ranked = top_k_tasks(query_emb, synthetic_tasks, k=3)
        task_ids = [t.id for t, _ in ranked]
        assert "task_x" not in task_ids

    def test_raises_if_all_centroids_missing(self, synthetic_tasks):
        for t in synthetic_tasks:
            t.embedding_centroid = None
        query_emb = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        with pytest.raises(ValueError, match="No task cluster has a computed"):
            top_k_tasks(query_emb, synthetic_tasks, k=3)

    def test_query_aligned_with_task_x(self, synthetic_tasks):
        query_emb = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        ranked = top_k_tasks(query_emb, synthetic_tasks, k=1)
        assert ranked[0][0].id == "task_x"
        assert ranked[0][1] == pytest.approx(1.0, abs=1e-5)

    def test_diagonal_query_ranks_two_tasks_close(self, synthetic_tasks):
        """A query pointing between task_x and task_y should rank both near the top."""
        # 45-degree angle: equal similarity to task_x and task_y
        v = 1.0 / (2 ** 0.5)
        query_emb = np.array([v, v, 0.0], dtype=np.float32)
        ranked = top_k_tasks(query_emb, synthetic_tasks, k=2)
        top_ids = {t.id for t, _ in ranked}
        assert "task_x" in top_ids
        assert "task_y" in top_ids


# ---------------------------------------------------------------------------
# select_adapter tests
# ---------------------------------------------------------------------------

class TestSelectAdapter:
    def test_selects_preferred_adapter(self, synthetic_tasks, adapter_reg):
        query_emb = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        ranked = top_k_tasks(query_emb, synthetic_tasks, k=3)
        adapter, task, score = select_adapter(ranked, adapter_reg)
        assert adapter.id == "adapter-x"
        assert task.id == "task_x"
        assert score == pytest.approx(1.0, abs=1e-5)

    def test_falls_through_to_second_task_when_first_adapter_missing(
        self, synthetic_tasks, tmp_path
    ):
        """If the best task's adapter isn't in the registry, the next task's adapter wins."""
        # Only adapter-y and adapter-z exist
        adapter_reg_partial = AdapterRegistry(
            adapters=[_make_adapter("adapter-y"), _make_adapter("adapter-z")],
            source_path=tmp_path / "adapters.json",
        )
        query_emb = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        ranked = top_k_tasks(query_emb, synthetic_tasks, k=3)
        adapter, task, score = select_adapter(ranked, adapter_reg_partial)
        # task_x has no matching adapter → falls through to task_y or task_z
        assert adapter.id in {"adapter-y", "adapter-z"}

    def test_raises_no_adapter_error_when_nothing_matches(self, synthetic_tasks, tmp_path):
        empty_reg = AdapterRegistry(adapters=[], source_path=tmp_path / "adapters.json")
        query_emb = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        ranked = top_k_tasks(query_emb, synthetic_tasks, k=3)
        with pytest.raises(NoAdapterError):
            select_adapter(ranked, empty_reg)

    def test_fallback_adapters_are_tried(self, tmp_path):
        """preferred list empty but fallback_adapters has a match."""
        task_with_fallback = TaskCluster(
            id="fallback_task",
            name="Fallback Task",
            description="Tests fallback logic",
            validation_examples=["example"],
            embedding_centroid=_unit([1, 0, 0]),
            preferred_adapters=[],
            fallback_adapters=["fallback-adapter"],
        )
        fallback_adapter = _make_adapter("fallback-adapter")
        reg = AdapterRegistry(
            adapters=[fallback_adapter], source_path=tmp_path / "adapters.json"
        )
        query_emb = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        ranked = top_k_tasks(query_emb, [task_with_fallback], k=1)
        adapter, task, _ = select_adapter(ranked, reg)
        assert adapter.id == "fallback-adapter"


# ---------------------------------------------------------------------------
# router.route integration tests (no real embedder)
# ---------------------------------------------------------------------------

class MockEmbedder:
    """Deterministic stub: encodes queries as specific axis-aligned vectors."""

    _MAP = {
        "python": np.array([1.0, 0.0, 0.0], dtype=np.float32),
        "sql": np.array([0.0, 1.0, 0.0], dtype=np.float32),
        "generic": np.array([0.0, 0.0, 1.0], dtype=np.float32),
    }

    def embed(self, text: str) -> np.ndarray:
        for keyword, vec in self._MAP.items():
            if keyword in text.lower():
                return vec
        return self._MAP["generic"]

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        return np.stack([self.embed(t) for t in texts])


class TestRouteFunction:
    def test_route_returns_trace(self, task_reg, adapter_reg):
        from shiftgate.router.router import route

        trace = route("write python code", task_reg, adapter_reg, MockEmbedder())
        assert trace.matched_task_id == "task_x"
        assert trace.selected_adapter_id == "adapter-x"
        assert 0.0 <= trace.similarity_score <= 1.0

    def test_route_sql_query(self, task_reg, adapter_reg):
        from shiftgate.router.router import route

        trace = route("write a sql query", task_reg, adapter_reg, MockEmbedder())
        assert trace.matched_task_id == "task_y"
        assert trace.selected_adapter_id == "adapter-y"

    def test_route_raises_when_embeddings_missing(self, tmp_path, adapter_reg):
        from shiftgate.router.router import route

        task_no_centroid = TaskCluster(
            id="unready",
            name="Unready",
            description="No centroid",
            validation_examples=["x"],
        )
        task_reg_empty = TaskRegistry(tasks=[task_no_centroid], source_path=tmp_path / "t.json")
        with pytest.raises(ValueError, match="not initialised"):
            route("any query", task_reg_empty, adapter_reg, MockEmbedder())

    def test_route_trace_has_required_fields(self, task_reg, adapter_reg):
        from shiftgate.router.router import route

        trace = route("generic task", task_reg, adapter_reg, MockEmbedder())
        assert trace.id  # non-empty
        assert trace.timestamp
        assert trace.query == "generic task"
        assert trace.accepted is None
        assert trace.latency_ms is None
