"""Operator signals for V2 execution-event writes and their delivery.

Each helper emits two kinds of signal for one occurrence: a single structured
log line that locates the task (and the tool attempt where relevant), and a
metric with bounded attributes for rates. Neither carries payload content or
exception text; exception messages can embed user data, so only type names,
identifiers, SQLSTATEs and numbers are reported. Task ids appear in logs only,
never as metric attributes.
"""

from __future__ import annotations

import logging

from ...core.agent.checkpoint import ExecutionEventPersistenceError
from ...core.runtime_performance import increment_counter
from .db_write_retry import cause_links, database_sqlstate

logger = logging.getLogger(__name__)

COMMIT_FAILURE_COUNTER = "xagent.execution_event.commit.failures"
REPLAY_CONSISTENCY_COUNTER = "xagent.websocket.replay.consistency"
BRIDGE_FRAMES_COUNTER = "xagent.task_event.bridge.frames"


def _failure_type(error: BaseException) -> str:
    """Class name of the first cause that is not the persistence wrapper."""
    for link in cause_links(error):
        if not isinstance(link, ExecutionEventPersistenceError):
            return type(link).__name__
    return type(error).__name__


def report_commit_failure(
    site: str,
    error: BaseException,
    *,
    task_id: int,
    run_id: str | None,
    retried: bool,
    tool_attempt_id: str | None = None,
) -> None:
    """Report one failed V2 execution-event commit, after retries are over.

    Call it once per failed write: after retries are exhausted or for a
    failure that is not retryable, never per attempt. Never raises, so a
    failure to report cannot mask the original error.
    """
    try:
        error_type = _failure_type(error)
        increment_counter(
            COMMIT_FAILURE_COUNTER,
            attributes={"site": site, "error.type": error_type, "storage": "v2"},
        )
        message = (
            "execution_event_commit_failed task_id=%s run_id=%s storage=v2 "
            "component=execution-events site=%s error_type=%s sqlstate=%s "
            "retried=%s"
        )
        args: list[object] = [
            task_id,
            run_id or "-",
            site,
            error_type,
            database_sqlstate(error) or "-",
            retried,
        ]
        if tool_attempt_id:
            message += " tool_attempt_id=%s"
            args.append(tool_attempt_id)
        logger.error(message, *args)
    except Exception:
        # Observability must never change the behavior of the measured path.
        return


def report_replay(
    task_id: int,
    *,
    horizon: int,
    events: int,
    cache_hit: bool,
    duration_ms: float,
) -> None:
    """Log one V2 historical replay: where it ends and how many events it sent."""
    try:
        logger.info(
            "execution_event_replay task_id=%s storage=v2 "
            "component=execution-events horizon=%s events=%s cache_hit=%s "
            "duration_ms=%.1f",
            task_id,
            horizon,
            events,
            cache_hit,
            duration_ms,
        )
    except Exception:
        return


def report_delivery_consistency(
    task_id: int,
    *,
    horizon: int,
    outcome: str,
    missing: int = 0,
    first_missing_sequence: int | None = None,
) -> None:
    """Report the delivery check of one closed V2 connection.

    ``outcome`` is one of:

    - ``complete``, ``missing``: the committed display frames after the
      replay horizon were compared with the frames sent. Only frames whose
      live delivery is unconditional are expected (accepted inputs and
      assistant chat messages are not), and the range ends at the highest
      sequence sent at least the reorder grace period before the disconnect.
      ``complete`` says nothing about frames committed after that point.
    - ``duplicate``: nothing missing, but the same frame after the horizon
      was sent live twice. Replay and reconcile passes are not counted.
    - ``idle``: nothing after the horizon was sent before the grace window,
      so there was nothing to compare.
    - ``skipped``: too many checks were pending; this one never ran.
    - ``overflow``: the socket was sent more frames than the ledger keeps.

    A WARNING locates the task only when committed display events were not
    sent; duplicates are informational because clients de-duplicate them.
    """
    try:
        increment_counter(
            REPLAY_CONSISTENCY_COUNTER,
            attributes={"outcome": outcome, "storage": "v2"},
        )
        if missing > 0:
            logger.warning(
                "execution_event_delivery_gap task_id=%s storage=v2 "
                "component=execution-events horizon=%s missing=%s "
                "first_missing_sequence=%s",
                task_id,
                horizon,
                missing,
                first_missing_sequence,
            )
    except Exception:
        return


def report_delivery_check_failure(
    task_id: int, *, horizon: int, error: BaseException
) -> None:
    """Report a delivery check that could not run, by exception type only."""
    try:
        logger.warning(
            "execution_event_delivery_check_failed task_id=%s storage=v2 "
            "component=execution-events horizon=%s error_type=%s",
            task_id,
            horizon,
            type(error).__name__,
        )
    except Exception:
        return


def report_bridge_frame(outcome: str) -> None:
    """Count one cross-process frame outcome on the task event bridge.

    ``outcome`` is ``duplicate_dropped``, ``gap_resync`` or ``publish_failed``.
    """
    increment_counter(BRIDGE_FRAMES_COUNTER, attributes={"outcome": outcome})


def report_bridge_gap(
    task_id: int, *, host: str, previous_sequence: int, sequence: int
) -> None:
    """Count and locate one bridge sequence gap that forces a client resync.

    Bridge sequences are per publishing host, not per task: the frames in the
    gap may belong to other tasks, and ``task_id`` is the first frame after it.
    """
    try:
        report_bridge_frame("gap_resync")
        logger.warning(
            "task_event_bridge_gap task_id=%s host=%s previous_sequence=%s sequence=%s",
            task_id,
            host,
            previous_sequence,
            sequence,
        )
    except Exception:
        return
