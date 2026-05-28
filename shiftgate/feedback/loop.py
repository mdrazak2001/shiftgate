"""
Feedback loop: persist routing traces and compute adapter acceptance scores.

Traces are stored as newline-delimited JSON in ``~/.shiftgate/traces.jsonl``.
Each line is a serialised ``RoutingTrace``.  This format is append-only and
easy to stream-process without loading the entire file into memory.

Workflow
--------
1. After every ``shiftgate route`` / ``shiftgate run``, call ``record_trace``.
2. User runs ``shiftgate feedback accept`` or ``shiftgate feedback reject``.
3. Call ``mark_accepted(trace_id, accepted)`` to annotate the trace.
4. ``compute_adapter_scores()`` aggregates acceptance rates per adapter.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from shiftgate.registry.schemas import RoutingTrace

logger = logging.getLogger(__name__)

_SHIFTGATE_DIR = Path.home() / ".shiftgate"
_TRACES_PATH = _SHIFTGATE_DIR / "traces.jsonl"

# How many recent traces to scan when ``mark_accepted`` searches by trace ID.
_RECENT_SCAN_LIMIT = 200


def record_trace(trace: RoutingTrace) -> None:
    """Append a ``RoutingTrace`` as a JSON line to the traces log.

    The file is created (along with its parent directory) on first write.
    """
    _SHIFTGATE_DIR.mkdir(parents=True, exist_ok=True)
    line = trace.model_dump_json()
    with _TRACES_PATH.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")
    logger.debug("Trace %s recorded.", trace.id)


def get_last_trace() -> RoutingTrace | None:
    """Return the most recently recorded trace, or None if no traces exist."""
    if not _TRACES_PATH.exists():
        return None
    last_line: str | None = None
    with _TRACES_PATH.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                last_line = line
    if last_line is None:
        return None
    return RoutingTrace.model_validate_json(last_line)


def mark_accepted(trace_id: str, accepted: bool) -> bool:
    """Set the ``accepted`` field on a specific trace.

    Rewrites the last ``_RECENT_SCAN_LIMIT`` lines of the traces file in-place
    (only those lines, prepending unchanged older lines).  Trades slight memory
    use for simplicity.

    Parameters
    ----------
    trace_id:
        The ``RoutingTrace.id`` hex string to update.
    accepted:
        True = good routing decision, False = bad routing decision.

    Returns
    -------
    True if the trace was found and updated; False if not found.
    """
    if not _TRACES_PATH.exists():
        logger.warning("No traces file found at %s.", _TRACES_PATH)
        return False

    lines = _TRACES_PATH.read_text(encoding="utf-8").splitlines()
    updated = False

    for i in range(len(lines) - 1, max(-1, len(lines) - _RECENT_SCAN_LIMIT - 1), -1):
        line = lines[i].strip()
        if not line:
            continue
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            continue
        if data.get("id") == trace_id:
            data["accepted"] = accepted
            lines[i] = json.dumps(data, ensure_ascii=False)
            updated = True
            break

    if updated:
        _TRACES_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
        logger.debug("Trace %s marked accepted=%s.", trace_id, accepted)

    return updated


def mark_last_accepted(accepted: bool) -> RoutingTrace | None:
    """Convenience: mark the most recent trace as accepted/rejected.

    Returns the updated trace, or None if no traces exist.
    """
    trace = get_last_trace()
    if trace is None:
        return None
    mark_accepted(trace.id, accepted)
    trace.accepted = accepted
    return trace


def load_all_traces() -> list[RoutingTrace]:
    """Load all traces from disk into memory.

    For large files prefer streaming with ``iter_traces()`` instead.
    """
    return list(iter_traces())


def iter_traces():
    """Yield ``RoutingTrace`` objects one at a time from the traces file."""
    if not _TRACES_PATH.exists():
        return
    with _TRACES_PATH.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                yield RoutingTrace.model_validate_json(line)
            except Exception as exc:
                logger.warning("Skipping malformed trace line: %s", exc)


def compute_adapter_scores() -> dict[str, float]:
    """Compute the acceptance rate for each adapter across all rated traces.

    Returns
    -------
    A dict mapping ``adapter_id`` → acceptance rate (0.0 – 1.0).
    Only adapters with at least one rated trace are included.
    Adapters with a 0 % acceptance rate are included with score 0.0.
    """
    totals: dict[str, int] = {}
    accepted_counts: dict[str, int] = {}

    for trace in iter_traces():
        if trace.accepted is None:
            continue
        aid = trace.selected_adapter_id
        totals[aid] = totals.get(aid, 0) + 1
        if trace.accepted:
            accepted_counts[aid] = accepted_counts.get(aid, 0) + 1

    return {
        aid: accepted_counts.get(aid, 0) / total
        for aid, total in totals.items()
    }


def get_trace_stats() -> dict[str, int]:
    """Return summary statistics about the traces file.

    Keys: ``total``, ``accepted``, ``rejected``, ``unrated``.
    """
    stats = {"total": 0, "accepted": 0, "rejected": 0, "unrated": 0}
    for trace in iter_traces():
        stats["total"] += 1
        if trace.accepted is True:
            stats["accepted"] += 1
        elif trace.accepted is False:
            stats["rejected"] += 1
        else:
            stats["unrated"] += 1
    return stats
