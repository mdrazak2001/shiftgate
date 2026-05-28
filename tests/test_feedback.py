"""
Tests for the feedback loop: recording traces, marking accepted/rejected,
and computing adapter acceptance scores.

All file I/O is redirected to a temporary directory.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pytest

from shiftgate.registry.schemas import RoutingTrace


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_trace(
    adapter_id: str = "test-adapter",
    task_id: str = "test_task",
    score: float = 0.85,
    accepted: bool | None = None,
) -> RoutingTrace:
    return RoutingTrace(
        id=uuid.uuid4().hex,
        query="test query",
        matched_task_id=task_id,
        similarity_score=score,
        selected_adapter_id=adapter_id,
        accepted=accepted,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def tmp_feedback(tmp_path, monkeypatch):
    """Redirect feedback loop I/O to a temporary directory."""
    import shiftgate.feedback.loop as loop_mod

    shiftgate_dir = tmp_path / ".shiftgate"
    shiftgate_dir.mkdir()
    traces_path = shiftgate_dir / "traces.jsonl"

    monkeypatch.setattr(loop_mod, "_SHIFTGATE_DIR", shiftgate_dir)
    monkeypatch.setattr(loop_mod, "_TRACES_PATH", traces_path)

    return traces_path


# ---------------------------------------------------------------------------
# record_trace
# ---------------------------------------------------------------------------

class TestRecordTrace:
    def test_creates_file_on_first_write(self, tmp_feedback):
        from shiftgate.feedback.loop import record_trace

        assert not tmp_feedback.exists()
        trace = _make_trace()
        record_trace(trace)
        assert tmp_feedback.exists()

    def test_appends_multiple_traces(self, tmp_feedback):
        from shiftgate.feedback.loop import record_trace

        for _ in range(5):
            record_trace(_make_trace())

        lines = [l for l in tmp_feedback.read_text().splitlines() if l.strip()]
        assert len(lines) == 5

    def test_each_line_is_valid_json(self, tmp_feedback):
        from shiftgate.feedback.loop import record_trace

        trace = _make_trace(adapter_id="my-adapter", score=0.73)
        record_trace(trace)

        line = tmp_feedback.read_text().strip()
        data = json.loads(line)
        assert data["selected_adapter_id"] == "my-adapter"
        assert data["similarity_score"] == pytest.approx(0.73)

    def test_trace_id_preserved(self, tmp_feedback):
        from shiftgate.feedback.loop import record_trace

        trace = _make_trace()
        record_trace(trace)
        data = json.loads(tmp_feedback.read_text().strip())
        assert data["id"] == trace.id


# ---------------------------------------------------------------------------
# get_last_trace
# ---------------------------------------------------------------------------

class TestGetLastTrace:
    def test_returns_none_if_no_file(self, tmp_feedback):
        from shiftgate.feedback.loop import get_last_trace

        assert get_last_trace() is None

    def test_returns_most_recent_trace(self, tmp_feedback):
        from shiftgate.feedback.loop import get_last_trace, record_trace

        first = _make_trace(adapter_id="adapter-1")
        second = _make_trace(adapter_id="adapter-2")
        record_trace(first)
        record_trace(second)

        last = get_last_trace()
        assert last is not None
        assert last.id == second.id
        assert last.selected_adapter_id == "adapter-2"


# ---------------------------------------------------------------------------
# mark_accepted
# ---------------------------------------------------------------------------

class TestMarkAccepted:
    def test_mark_accepted_true(self, tmp_feedback):
        from shiftgate.feedback.loop import mark_accepted, record_trace

        trace = _make_trace()
        record_trace(trace)
        result = mark_accepted(trace.id, True)
        assert result is True

        data = json.loads(tmp_feedback.read_text().strip())
        assert data["accepted"] is True

    def test_mark_accepted_false(self, tmp_feedback):
        from shiftgate.feedback.loop import mark_accepted, record_trace

        trace = _make_trace()
        record_trace(trace)
        result = mark_accepted(trace.id, False)
        assert result is True

        data = json.loads(tmp_feedback.read_text().strip())
        assert data["accepted"] is False

    def test_mark_accepted_returns_false_if_id_not_found(self, tmp_feedback):
        from shiftgate.feedback.loop import mark_accepted, record_trace

        record_trace(_make_trace())
        result = mark_accepted("nonexistent-id-hex", True)
        assert result is False

    def test_mark_last_accepted(self, tmp_feedback):
        from shiftgate.feedback.loop import mark_last_accepted, record_trace

        trace = _make_trace()
        record_trace(trace)
        updated = mark_last_accepted(True)
        assert updated is not None
        assert updated.accepted is True

    def test_mark_last_accepted_no_traces(self, tmp_feedback):
        from shiftgate.feedback.loop import mark_last_accepted

        assert mark_last_accepted(True) is None


# ---------------------------------------------------------------------------
# compute_adapter_scores
# ---------------------------------------------------------------------------

class TestComputeAdapterScores:
    def test_empty_file_returns_empty_dict(self, tmp_feedback):
        from shiftgate.feedback.loop import compute_adapter_scores

        scores = compute_adapter_scores()
        assert scores == {}

    def test_all_accepted(self, tmp_feedback):
        from shiftgate.feedback.loop import compute_adapter_scores, record_trace

        for _ in range(4):
            record_trace(_make_trace(adapter_id="good-adapter", accepted=True))

        scores = compute_adapter_scores()
        assert "good-adapter" in scores
        assert scores["good-adapter"] == pytest.approx(1.0)

    def test_all_rejected(self, tmp_feedback):
        from shiftgate.feedback.loop import compute_adapter_scores, record_trace

        for _ in range(3):
            record_trace(_make_trace(adapter_id="bad-adapter", accepted=False))

        scores = compute_adapter_scores()
        assert scores["bad-adapter"] == pytest.approx(0.0)

    def test_mixed_feedback(self, tmp_feedback):
        from shiftgate.feedback.loop import compute_adapter_scores, record_trace

        # 3 accepted, 1 rejected → 75%
        for _ in range(3):
            record_trace(_make_trace(adapter_id="mixed-adapter", accepted=True))
        record_trace(_make_trace(adapter_id="mixed-adapter", accepted=False))

        scores = compute_adapter_scores()
        assert scores["mixed-adapter"] == pytest.approx(0.75)

    def test_unrated_traces_excluded(self, tmp_feedback):
        from shiftgate.feedback.loop import compute_adapter_scores, record_trace

        # Two traces, neither rated
        record_trace(_make_trace(adapter_id="unrated-adapter", accepted=None))
        record_trace(_make_trace(adapter_id="unrated-adapter", accepted=None))

        scores = compute_adapter_scores()
        assert "unrated-adapter" not in scores

    def test_multiple_adapters(self, tmp_feedback):
        from shiftgate.feedback.loop import compute_adapter_scores, record_trace

        record_trace(_make_trace(adapter_id="adapter-a", accepted=True))
        record_trace(_make_trace(adapter_id="adapter-a", accepted=True))
        record_trace(_make_trace(adapter_id="adapter-b", accepted=False))

        scores = compute_adapter_scores()
        assert scores["adapter-a"] == pytest.approx(1.0)
        assert scores["adapter-b"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# get_trace_stats
# ---------------------------------------------------------------------------

class TestGetTraceStats:
    def test_correct_counts(self, tmp_feedback):
        from shiftgate.feedback.loop import get_trace_stats, record_trace

        record_trace(_make_trace(accepted=True))
        record_trace(_make_trace(accepted=True))
        record_trace(_make_trace(accepted=False))
        record_trace(_make_trace(accepted=None))

        stats = get_trace_stats()
        assert stats["total"] == 4
        assert stats["accepted"] == 2
        assert stats["rejected"] == 1
        assert stats["unrated"] == 1

    def test_empty_stats(self, tmp_feedback):
        from shiftgate.feedback.loop import get_trace_stats

        stats = get_trace_stats()
        assert stats["total"] == 0
