"""
Tests for the routing pipeline: embedder, matcher, and router.

The matcher and router tests use pre-computed synthetic embeddings so they
are fast, deterministic, and require no model download.
"""

from __future__ import annotations

import numpy as np
import pytest

from shiftgate.registry.adapter_registry import AdapterRegistry, adapter_from_hf, adapter_from_runtime
from shiftgate.registry.schemas import AdapterEntry, TaskCluster
from shiftgate.registry.task_registry import TaskRegistry
from shiftgate.router.matcher import NoAdapterError, TaskMatch, select_adapter, top_k_tasks


# ---------------------------------------------------------------------------
# Helpers: build a synthetic task registry with known geometry
# ---------------------------------------------------------------------------

def _unit(v: list[float]) -> list[float]:
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
    """Three tasks whose centroids are axis-aligned unit vectors in 3D space."""
    return [
        _make_task("task_x", [1, 0, 0], ["adapter-x"]),
        _make_task("task_y", [0, 1, 0], ["adapter-y"]),
        _make_task("task_z", [0, 0, 1], ["adapter-z"]),
    ]


@pytest.fixture()
def task_reg(synthetic_tasks, tmp_path) -> TaskRegistry:
    return TaskRegistry(tasks=synthetic_tasks, source_path=tmp_path / "tasks.json")


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
        query_emb = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        ranked = top_k_tasks(query_emb, synthetic_tasks, k=3)
        assert ranked[0].task.id == "task_y"

    def test_returns_task_match_objects(self, synthetic_tasks):
        query_emb = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        ranked = top_k_tasks(query_emb, synthetic_tasks, k=1)
        assert isinstance(ranked[0], TaskMatch)
        assert isinstance(ranked[0].task, TaskCluster)
        assert isinstance(ranked[0].score, float)

    def test_scores_descending(self, synthetic_tasks):
        query_emb = np.array([0.6, 0.8, 0.0], dtype=np.float32)
        ranked = top_k_tasks(query_emb, synthetic_tasks, k=3)
        scores = [tm.score for tm in ranked]
        assert scores == sorted(scores, reverse=True)

    def test_top_k_limits_results(self, synthetic_tasks):
        query_emb = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        ranked = top_k_tasks(query_emb, synthetic_tasks, k=2)
        assert len(ranked) == 2

    def test_no_centroid_tasks_are_skipped(self, synthetic_tasks):
        synthetic_tasks[0].embedding_centroid = None
        query_emb = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        ranked = top_k_tasks(query_emb, synthetic_tasks, k=3)
        task_ids = [tm.task.id for tm in ranked]
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
        assert ranked[0].task.id == "task_x"
        assert ranked[0].score == pytest.approx(1.0, abs=1e-5)

    def test_diagonal_query_ranks_two_tasks_close(self, synthetic_tasks):
        v = 1.0 / (2 ** 0.5)
        query_emb = np.array([v, v, 0.0], dtype=np.float32)
        ranked = top_k_tasks(query_emb, synthetic_tasks, k=2)
        top_ids = {tm.task.id for tm in ranked}
        assert "task_x" in top_ids
        assert "task_y" in top_ids


# ---------------------------------------------------------------------------
# select_adapter tests
# ---------------------------------------------------------------------------

class TestSelectAdapter:
    def test_selects_preferred_adapter(self, synthetic_tasks, adapter_reg):
        query_emb = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        ranked = top_k_tasks(query_emb, synthetic_tasks, k=3)
        result = select_adapter(ranked, adapter_reg)
        assert result.selected_adapter.id == "adapter-x"
        assert result.matched_task.id == "task_x"
        assert result.similarity_score == pytest.approx(1.0, abs=1e-5)
        assert result.selection_method == "preferred"

    def test_match_result_has_all_task_matches(self, synthetic_tasks, adapter_reg):
        query_emb = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        ranked = top_k_tasks(query_emb, synthetic_tasks, k=3)
        result = select_adapter(ranked, adapter_reg)
        assert len(result.all_task_matches) == 3

    def test_candidate_adapters_populated_on_tasks(self, synthetic_tasks, adapter_reg):
        """Each TaskMatch should have candidate_adapters filled in after select_adapter."""
        query_emb = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        ranked = top_k_tasks(query_emb, synthetic_tasks, k=3)
        result = select_adapter(ranked, adapter_reg)
        # The winning task must have at least one candidate adapter.
        winning_tm = next(tm for tm in result.all_task_matches if tm.task.id == result.matched_task.id)
        assert len(winning_tm.candidate_adapters) >= 1

    def test_falls_through_to_second_task_when_first_adapter_missing(
        self, synthetic_tasks, tmp_path
    ):
        adapter_reg_partial = AdapterRegistry(
            adapters=[_make_adapter("adapter-y"), _make_adapter("adapter-z")],
            source_path=tmp_path / "adapters.json",
        )
        query_emb = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        ranked = top_k_tasks(query_emb, synthetic_tasks, k=3)
        result = select_adapter(ranked, adapter_reg_partial)
        assert result.selected_adapter.id in {"adapter-y", "adapter-z"}

    def test_no_adapter_when_registry_empty(self, synthetic_tasks, tmp_path):
        """An empty registry yields a None adapter, never an exception or a guess."""
        empty_reg = AdapterRegistry(adapters=[], source_path=tmp_path / "adapters.json")
        query_emb = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        ranked = top_k_tasks(query_emb, synthetic_tasks, k=3)
        result = select_adapter(ranked, empty_reg)
        assert result.selected_adapter is None
        assert result.has_adapter is False
        assert result.selection_method == "no_adapter_for_task"
        # The matched task is still reported so the caller can prompt the user.
        assert result.matched_task.id == "task_x"

    def test_no_silent_fallback_when_task_has_no_linked_adapter(self, tmp_path):
        """The bug fix: an unrelated adapter must NOT be picked for the matched task.

        A music-style query matches a task that has no linked adapter; even
        though a sql-lora exists in the registry, it must not be selected.
        """
        task_music = _make_task("audio_music", [0, 1, 0], adapter_ids=[])
        task_py    = _make_task("code_python", [1, 0, 0], adapter_ids=[])

        sql_adapter = AdapterEntry(
            id="sql-lora",
            name="SQL LoRA",
            base_model="llama3",
            task_tags=["sql", "code"],
        )
        reg = AdapterRegistry(adapters=[sql_adapter], source_path=tmp_path / "adapters.json")

        query_emb = np.array([0.0, 1.0, 0.0], dtype=np.float32)  # matches audio_music
        ranked = top_k_tasks(query_emb, [task_music, task_py], k=2)
        result = select_adapter(ranked, reg)

        assert result.selected_adapter is None
        assert result.selection_method == "no_adapter_for_task"
        assert result.matched_task.id == "audio_music"

    def test_fallback_adapters_are_tried(self, tmp_path):
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
        result = select_adapter(ranked, reg)
        assert result.selected_adapter.id == "fallback-adapter"
        assert result.selection_method == "fallback"


# ---------------------------------------------------------------------------
# router.route integration tests (no real embedder)
# ---------------------------------------------------------------------------

class MockEmbedder:
    """Deterministic stub: encodes queries as specific axis-aligned vectors."""

    _MAP = {
        "python": np.array([1.0, 0.0, 0.0], dtype=np.float32),
        "sql":    np.array([0.0, 1.0, 0.0], dtype=np.float32),
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
    def test_route_returns_trace_and_match_result(self, task_reg, adapter_reg):
        from shiftgate.router.matcher import MatchResult
        from shiftgate.router.router import route

        trace, result = route("write python code", task_reg, adapter_reg, MockEmbedder())
        assert trace.matched_task_id == "task_x"
        assert trace.selected_adapter_id == "adapter-x"
        assert isinstance(result, MatchResult)

    def test_route_sql_query(self, task_reg, adapter_reg):
        from shiftgate.router.router import route

        trace, result = route("write a sql query", task_reg, adapter_reg, MockEmbedder())
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

        trace, _ = route("generic task", task_reg, adapter_reg, MockEmbedder())
        assert trace.id
        assert trace.timestamp
        assert trace.query == "generic task"
        assert trace.accepted is None
        assert trace.latency_ms is None

    def test_route_match_result_has_all_tasks(self, task_reg, adapter_reg):
        from shiftgate.router.router import route

        _, result = route("python task", task_reg, adapter_reg, MockEmbedder())
        assert len(result.all_task_matches) == 3  # top_k default is 3

    def test_runtime_adapter_effective_name(self):
        """effective_backend_name() returns runtime_name when set."""
        adapter = adapter_from_runtime("vllm-sql", adapter_id="sql-lora")
        assert adapter.effective_backend_name() == "vllm-sql"

    def test_hf_adapter_effective_name_falls_back_to_id(self):
        adapter = adapter_from_hf("org/sql-lora", adapter_id="sql-lora")
        assert adapter.effective_backend_name() == "sql-lora"
