"""Operator signals for execution-event writes, unknown outcomes, recovery
reads and delivery.

Each helper emits two kinds of signal for one occurrence: a single structured
log line that locates the task (and the tool attempt where relevant), and a
metric with bounded attributes for rates. Neither carries payload content or
exception text; exception messages can embed user data, so only type names,
identifiers, SQLSTATEs, bounded reason names and numbers are reported. Task
ids appear in logs only, never as metric attributes.

``report_commit_failure`` covers failed event commits.
``report_unknown_outcome`` covers cases where the system does not know whether
something durable happened (a tool's external effect, a turn or command
acceptance, a finalize whose acknowledgement was lost, a settlement deferred
to lease recovery). One of its kinds, ``finalize_ack_reconciled``, is a
confirmed success counted for visibility only: alerts on the unknown-outcome
counter must exclude it.
``report_recovery_refusal`` covers recovery reads that refuse to resume a task.
Its counter counts refused reads, including classification probes, so it is
neither a per-task nor a per-resume count; see that helper for known gaps.
``report_replay``, ``report_delivery_consistency`` and the bridge helpers cover
reconnect replays and live delivery of V2 display frames.
"""

from __future__ import annotations

import logging

from ...core.agent.checkpoint import ExecutionEventPersistenceError
from ...core.runtime_performance import increment_counter
from .db_write_retry import cause_links, database_sqlstate

logger = logging.getLogger(__name__)

COMMIT_FAILURE_COUNTER = "xagent.execution_event.commit.failures"
UNKNOWN_OUTCOME_COUNTER = "xagent.execution_event.unknown_outcomes"
RECOVERY_REFUSAL_COUNTER = "xagent.execution_event.recovery.refusals"
REPLAY_CONSISTENCY_COUNTER = "xagent.websocket.replay.consistency"
BRIDGE_FRAMES_COUNTER = "xagent.task_event.bridge.frames"

_REFUSAL_LEVELS = {
    "lease_mismatch": logging.INFO,
    "task_ended": logging.INFO,
    "task_missing": logging.INFO,
    "page_limit": logging.WARNING,
    "unknown_tool_effect": logging.WARNING,
    "unavailable": logging.WARNING,
    "corrupt": logging.ERROR,
}


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


def report_recovery_refusal(
    reader: str,
    outcome: str,
    *,
    task_id: int,
    run_id: str | None = None,
    tool_attempt_id: str | None = None,
) -> None:
    """Report one V2 recovery read that refused to resume a task.

    ``reader`` names the read (``owner_check``, ``checkpoint``,
    ``tool_outcome``) and ``outcome`` why it refused; both are bounded names
    that become metric attributes. Call it where the refusal is first raised
    or converted, once per refusal, never again as it propagates. Never
    raises, so a failure to report cannot mask the refusal itself.

    The log level follows the outcome: normal fencing (another owner took
    over, or the task already ended) is routine and logs at INFO, a refusal
    that needs an operator logs at WARNING, and unreadable facts log at
    ERROR. Unlisted outcomes default to WARNING.

    The counter counts refused READS, not tasks and not resumes.
    Classification probes read the same facts as a resume (the lease recovery
    sweep, settlement classification, auto-recovery decisions), so one stuck
    task can count once per sweep for as long as it stays unresumable.

    Known gap: refusals raised by the shared ``DatabaseTraceHandler`` base on
    the V2 root checkpoint path (``_root_checkpoint_read_partition``:
    lease/run mismatch, ``active_run``, ``superseded_legacy``, a missing task
    without a lease; the stale-widening check; a failed run-tagged probe;
    session checkout) and the setup-time corrupt check in
    ``task_setup_snapshot.py`` are not reported by this signal yet.
    """
    try:
        level = _REFUSAL_LEVELS.get(outcome, logging.WARNING)
        increment_counter(
            RECOVERY_REFUSAL_COUNTER,
            attributes={"operation": reader, "outcome": outcome, "storage": "v2"},
        )
        message = (
            "execution_event_recovery_refused task_id=%s run_id=%s storage=v2 "
            "component=execution-events reader=%s outcome=%s"
        )
        args: list[object] = [task_id, run_id or "-", reader, outcome]
        if tool_attempt_id:
            message += " tool_attempt_id=%s"
            args.append(tool_attempt_id)
        logger.log(level, message, *args)
    except Exception:
        # Observability must never change the behavior of the measured path.
        return


def report_unknown_outcome(
    kind: str,
    *,
    task_id: int,
    run_id: str | None = None,
    ref: str | None = None,
    storage: str = "unknown",
    level: int = logging.WARNING,
) -> None:
    """Report one case where it is unknown whether something durable happened.

    ``kind`` names the case and is the counter's ``operation`` attribute, so
    it must stay a small fixed set: ``tool_effect``, ``turn_commit``,
    ``command_commit``, ``finalize_ack_unresolved``,
    ``resume_finalize_unreconciled``, ``settlement_deferred`` -- and
    ``finalize_ack_reconciled``, which is not unknown but a confirmed success
    (the read-back proved a lost acknowledgement's COMMIT applied), reported
    at INFO and counted for visibility only: alerts on this counter must
    exclude it. ``storage`` is ``v1``, ``v2`` or ``unknown``; ``turn_commit``,
    ``command_commit`` and ``settlement_deferred`` report ``unknown`` because
    the storage version is not in scope where they occur. ``ref`` is an
    identifier only (a turn id, a command id, a tool attempt id), never
    text. Call it once per occurrence. Never raises, so a failure to report
    cannot change the path being measured.
    """
    # Observability must never change the behavior of the measured path.
    # Two independent guards: a failing metric must not suppress the log line.
    try:
        increment_counter(
            UNKNOWN_OUTCOME_COUNTER,
            attributes={"operation": kind, "storage": storage},
        )
    except Exception:
        pass
    try:
        logger.log(
            level,
            "execution_event_outcome_unknown task_id=%s run_id=%s storage=%s "
            "component=execution-events kind=%s ref=%s",
            task_id,
            run_id or "-",
            storage,
            kind,
            ref or "-",
        )
    except Exception:
        pass


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
    task_id: int | None, *, host: str, previous_sequence: int, sequence: int
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
