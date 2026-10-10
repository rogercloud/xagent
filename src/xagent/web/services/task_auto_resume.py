"""Dispatch automatic resumes of interrupted tasks, and re-check them at claim.

The sweeper turns due ``scheduled`` rows of ``task_auto_recovery`` into
durable RESUME commands; ``resume_task`` re-checks each such command when it
is claimed (:func:`check_auto_resume_claim_sync`), because the task may have
moved on while the command waited. Recording an interruption does not
schedule yet, so in production every tick scans an empty index range; every
path here is reachable by inserting ``scheduled`` rows directly.

Every process that runs lease recovery runs a sweeper; correctness rests on
the compare-and-swaps below, not on a singleton.

Lock order, shared with every other writer of these rows: ``tasks`` ->
``task_auto_recovery`` -> ``trigger_runs``. Settlement and lease recovery
write the ``tasks`` row before they upsert the recovery row, so on
PostgreSQL the sweeper locks ``tasks`` first (``FOR UPDATE OF tasks SKIP
LOCKED``). Locking the recovery row first would deadlock with such a
settlement: the sweeper's command and event inserts take a key-share lock on
``tasks`` through their foreign keys. SQLite ignores row locks; there every
transition is a compare-and-swap that re-checks the task fence, the way
lease recovery's SQLite path does.

The fence is ``tasks.status = PAUSED``, ``control_state = 'paused'`` and the
row's ``run_id`` and ``paused_state_version``. It deliberately ignores
``runner_id``: in shared mode a PAUSED task may hold an idle coordinator's
owner lease, whose acquisition and release do not move ``state_version``;
``resume_task`` defers while another process holds a live lease.

A dispatch is one transaction: the CAS to ``dispatched``, its event, and the
command (staged last, as ``stage_task_command`` requires). The dispatcher is
notified after the commit. ``dispatched`` rows reuse ``next_attempt_at`` as
the time housekeeping next checks the dispatch, and clear it once the fence
has moved (the resume took effect or a person acted).
"""

from __future__ import annotations

import asyncio
import logging
import random
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Callable, Iterable

from sqlalchemy import and_, func, or_, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from ...config import (
    get_task_auto_resume_enabled,
    get_task_auto_resume_max_inflight,
    get_task_auto_resume_max_per_tick,
)
from ...core.runtime_performance import (
    increment_counter,
    observe_value,
    register_observable_gauge,
)
from ..models.task import Task, TaskStatus, task_status_predicate
from ..models.task_auto_recovery import (
    TaskAutoRecovery,
    TaskAutoRecoveryState,
    TaskRecoveryEvent,
    TaskRecoveryEventType,
)
from ..models.task_command import TaskExecutionCommand
from ..utils.db_timezone import format_datetime_for_api
from .db_runtime import is_database_pool_timeout, run_db_io_cancellation_safe
from .ops_signals import (
    TASK_AUTO_RESUME_UNAVAILABLE,
    clear_degradation,
    register_degradation,
)
from .task_auto_recovery import (
    AUTO_RESUME_COMMAND_PREFIX,
    TASK_INTERRUPTION_PAUSED_TRIGGER_ERROR,
    auto_recovery_eligibility,
    auto_resume_schedulable,
    is_auto_resume_command_id,
)
from .task_auto_recovery_policy import (
    auto_resume_limit,
    auto_resume_recovery_state,
    next_auto_resume_at,
    scheduled_trigger_superseded,
)
from .task_command_transport import (
    COMMAND_COMPLETED,
    COMMAND_FAILED,
    TaskCommandKind,
    notify_task_command_dispatcher,
    stage_task_command,
)
from .task_execution_admission import AdmissionQueueFull
from .task_execution_controller import TaskControlState
from .task_lease_service import utc_now

logger = logging.getLogger(__name__)

# A dispatched command is checked again this long after dispatch (and after
# every check that finds it still queued).
DISPATCH_CHECK_GRACE_SECONDS = 60
# A full admission queue pushes the candidate this far out.
ADMISSION_FULL_DELAY_SECONDS = 30
_COMMAND_ID_MAX_CHARS = 64
# Consecutive failed ticks before TASK_AUTO_RESUME_UNAVAILABLE is raised.
_UNAVAILABLE_AFTER_FAILURES = 3
_FAILURE_LOG_INTERVAL_SECONDS = 60.0
_MAX_FAILURE_BACKOFF_SECONDS = 60
_MAX_FAILURE_BACKOFF_EXPONENT = 6

# TriggerRun errors for a scheduled resume that automatic recovery stopped.
TASK_AUTO_RESUME_EXHAUSTED_TRIGGER_ERROR = (
    "Task paused after a system interruption; automatic resume stopped after "
    "repeated attempts. Manual resume is required."
)
TASK_AUTO_RESUME_EXPIRED_TRIGGER_ERROR = (
    "Task paused after a system interruption; it was not resumed within the "
    "automatic-resume window. Manual resume is required."
)
TASK_AUTO_RESUME_SUPERSEDED_TRIGGER_ERROR = (
    "Task paused after a system interruption and was superseded by a later "
    "scheduled run."
)
TASK_AUTO_RESUME_DISPATCH_FAILED_TRIGGER_ERROR = (
    "Task paused after a system interruption; automatic resume could not be "
    "started. Manual resume is required."
)
TASK_AUTO_RESUME_DISABLED_TRIGGER_ERROR = (
    "Task paused after a system interruption; automatic resume is disabled. "
    "Manual resume is required."
)

# ``state_detail`` of a row stopped because XAGENT_TASK_AUTO_RESUME_ENABLED is
# off, and the ``why`` of a claim skipped for it.
AUTO_RESUME_DISABLED_DETAIL = "auto_resume_disabled"
CHANNEL_HOLD_PENDING_DETAIL = "channel_hold_pending"

# Module-level so tests can seed it; drawn only by housekeeping reschedules.
_AUTO_RESUME_RNG = random.Random()

# Cached by each tick for the in-flight gauge.
_last_inflight = 0

_SCHEDULED = TaskAutoRecoveryState.SCHEDULED
_DISPATCHED = TaskAutoRecoveryState.DISPATCHED


def auto_resume_command_id(paused_state_version: int, attempt: int, run_id: str) -> str:
    """The id of the RESUME that dispatches ``attempt`` at one fence.

    ``(task_id, command_id)`` is unique. The fence version grows with every
    interruption and ``attempt`` with every dispatch at one fence, so two
    dispatches never share an id even after the counters restart. The unique
    part leads; truncation to the column only shortens the run id.
    """

    return f"{AUTO_RESUME_COMMAND_PREFIX}{int(paused_state_version)}:{int(attempt)}:{run_id}"[
        :_COMMAND_ID_MAX_CHARS
    ]


def _utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


# --------------------------------------------------------------------------
# Public notices (``task_recovery_update``)
# --------------------------------------------------------------------------

_STOP_REASONS = {
    TaskAutoRecoveryState.EXHAUSTED.value: "limit_reached",
    TaskAutoRecoveryState.EXPIRED.value: "expired",
    TaskAutoRecoveryState.SUPERSEDED.value: "superseded",
    TaskAutoRecoveryState.DISPATCH_FAILED.value: "failed",
    TaskAutoRecoveryState.INELIGIBLE.value: "ineligible",
}


def public_auto_resume_view(
    state: str,
    *,
    state_detail: str | None,
    reason: str,
    attempt: int | None,
    next_attempt_at: datetime | None,
) -> dict[str, Any] | None:
    """The client contract for one automatic-resume transition, or ``None``.

    Counters, ``state_detail``, ``last_error`` and command ids stay
    server-side. ``stale`` and ``manual`` (other than a disabled switch) are
    not announced: a person's own action produced them, or nothing changed
    for the client.
    """

    if state == _SCHEDULED.value:
        status, stop_reason = "scheduled", None
    elif state == _DISPATCHED.value:
        status, stop_reason = "resuming", None
    elif state in _STOP_REASONS:
        status, stop_reason = "stopped", _STOP_REASONS[state]
    elif (
        state == TaskAutoRecoveryState.MANUAL.value
        and state_detail == AUTO_RESUME_DISABLED_DETAIL
    ):
        status, stop_reason = "stopped", "disabled"
    else:
        return None
    return {
        "status": status,
        "stop_reason": stop_reason,
        "reason": reason,
        "attempt": attempt,
        "next_attempt_at": (
            format_datetime_for_api(next_attempt_at) if status == "scheduled" else None
        ),
    }


@dataclass(frozen=True)
class RecoveryNotice:
    """A committed transition to announce as ``task_recovery_update``."""

    task_id: int
    run_id: str
    # The row's fence; a client already past it ignores the notice.
    state_version: int
    auto_resume: dict[str, Any]

    def message(self) -> dict[str, Any]:
        return {
            "type": "task_recovery_update",
            "task_id": self.task_id,
            "run_id": self.run_id,
            "state_version": self.state_version,
            "auto_resume": self.auto_resume,
            "timestamp": datetime.now(timezone.utc).timestamp(),
        }


def _notice(
    row: TaskAutoRecovery,
    state: TaskAutoRecoveryState,
    *,
    state_detail: str | None,
    attempt: int | None,
    next_attempt_at: datetime | None = None,
) -> RecoveryNotice | None:
    view = public_auto_resume_view(
        state.value,
        state_detail=state_detail,
        reason=str(row.reason),
        attempt=attempt,
        next_attempt_at=next_attempt_at,
    )
    if view is None:
        return None
    return RecoveryNotice(
        task_id=int(row.task_id),
        run_id=str(row.run_id),
        state_version=int(row.paused_state_version),
        auto_resume=view,
    )


async def publish_recovery_notices(notices: Iterable[RecoveryNotice]) -> None:
    """Broadcast committed notices; best effort, a failure is only logged."""

    from .task_events import publish_task_event

    for notice in notices:
        try:
            await publish_task_event(notice.message(), notice.task_id)
        except Exception:
            logger.warning(
                "task_recovery_update committed but broadcast failed for task %s",
                notice.task_id,
                exc_info=True,
            )


# --------------------------------------------------------------------------
# Fence and shared transition helpers
# --------------------------------------------------------------------------


def _rowcount(result: Any) -> int:
    return int(getattr(result, "rowcount", 0) or 0)


def _fence_mismatch(task: Task, row: TaskAutoRecovery) -> str | None:
    """The first fence column that no longer matches ``row``, or ``None``."""

    if task.status != TaskStatus.PAUSED:
        return "status"
    if task.control_state != TaskControlState.PAUSED.value:
        return "control_state"
    if task.run_id != row.run_id:
        return "run_id"
    if int(task.state_version or 0) != int(row.paused_state_version):
        return "state_version"
    return None


def _task_at_fence(task_id: int, run_id: str, paused_state_version: int) -> Any:
    """``EXISTS`` over ``tasks`` for the fence, for compare-and-swaps."""

    return (
        select(Task.id)
        .where(
            Task.id == task_id,
            task_status_predicate.eq(TaskStatus.PAUSED),
            Task.control_state == TaskControlState.PAUSED.value,
            Task.run_id == run_id,
            Task.state_version == paused_state_version,
        )
        .exists()
    )


def _row_cas(
    row: TaskAutoRecovery,
    from_state: TaskAutoRecoveryState,
    *,
    fenced: bool,
) -> list[Any]:
    """The compare-and-swap predicate for a transition out of ``from_state``."""

    predicates: list[Any] = [
        TaskAutoRecovery.task_id == int(row.task_id),
        TaskAutoRecovery.state == from_state.value,
        TaskAutoRecovery.run_id == row.run_id,
        TaskAutoRecovery.paused_state_version == int(row.paused_state_version),
    ]
    if from_state is _DISPATCHED:
        predicates.append(TaskAutoRecovery.last_command_id == row.last_command_id)
    if fenced:
        predicates.append(
            _task_at_fence(
                int(row.task_id), str(row.run_id), int(row.paused_state_version)
            )
        )
    return predicates


_GIVE_UP_EVENTS = {
    TaskAutoRecoveryState.EXHAUSTED: TaskRecoveryEventType.EXHAUSTED,
    TaskAutoRecoveryState.EXPIRED: TaskRecoveryEventType.EXPIRED,
    TaskAutoRecoveryState.SUPERSEDED: TaskRecoveryEventType.SUPERSEDED,
    TaskAutoRecoveryState.INELIGIBLE: TaskRecoveryEventType.INELIGIBLE,
    TaskAutoRecoveryState.DISPATCH_FAILED: TaskRecoveryEventType.DISPATCH_FAILED,
    TaskAutoRecoveryState.STALE: TaskRecoveryEventType.STALE,
    # A dispatch-time ``manual`` (a reason with no policy, a channel task)
    # is the task being out of automatic recovery's reach; the event detail
    # names the state. The disabled switch passes SKIPPED_DISABLED itself.
    TaskAutoRecoveryState.MANUAL: TaskRecoveryEventType.INELIGIBLE,
}

_GIVE_UP_TRIGGER_ERRORS = {
    TaskAutoRecoveryState.EXHAUSTED: TASK_AUTO_RESUME_EXHAUSTED_TRIGGER_ERROR,
    TaskAutoRecoveryState.EXPIRED: TASK_AUTO_RESUME_EXPIRED_TRIGGER_ERROR,
    TaskAutoRecoveryState.SUPERSEDED: TASK_AUTO_RESUME_SUPERSEDED_TRIGGER_ERROR,
    TaskAutoRecoveryState.DISPATCH_FAILED: (
        TASK_AUTO_RESUME_DISPATCH_FAILED_TRIGGER_ERROR
    ),
}


def _trigger_error(state: TaskAutoRecoveryState, state_detail: str | None) -> str:
    if (
        state is TaskAutoRecoveryState.MANUAL
        and state_detail == AUTO_RESUME_DISABLED_DETAIL
    ):
        return TASK_AUTO_RESUME_DISABLED_TRIGGER_ERROR
    return _GIVE_UP_TRIGGER_ERRORS.get(state, TASK_INTERRUPTION_PAUSED_TRIGGER_ERROR)


class _Outcome(str, Enum):
    DISPATCHED = "dispatched"
    GAVE_UP = "gave_up"
    STALE = "stale"
    RESCHEDULED = "rescheduled"
    CONFIRMED = "confirmed"
    # Housekeeping found the command still queued; checked again later.
    REARMED = "rearmed"
    # Nothing written: the row moved, or a CAS lost.
    RACED = "raced"
    # Nothing written: a dispatch was due but the budget is spent.
    CAPACITY = "capacity"


@dataclass(frozen=True)
class _CandidateResult:
    outcome: _Outcome
    notice: RecoveryNotice | None = None
    staged: bool = False
    dispatch_delay_seconds: float | None = None

    @property
    def writes(self) -> bool:
        return self.outcome not in (_Outcome.RACED, _Outcome.CAPACITY)


_RACED = _CandidateResult(_Outcome.RACED)


def _give_up_no_commit(
    db: Session,
    task: Task,
    row: TaskAutoRecovery,
    new_state: TaskAutoRecoveryState,
    *,
    from_state: TaskAutoRecoveryState,
    at: str,
    state_detail: str | None = None,
    event: TaskRecoveryEventType | None = None,
    event_detail: dict[str, Any] | None = None,
) -> _CandidateResult:
    """Stop automatic recovery of ``row`` in ``new_state``; no commit.

    The CAS re-checks the fence except for ``stale``, which exists because
    the fence no longer holds. TriggerRun: a give-up fails a PENDING/RUNNING
    run with the state's message; ``stale`` only mirrors a task that already
    ended, since otherwise a person resumed or changed it and that run's own
    settlement reports the real outcome.
    """

    from .task_orchestrator import sync_trigger_run_status

    stale = new_state is TaskAutoRecoveryState.STALE
    updated = _rowcount(
        db.execute(
            update(TaskAutoRecovery)
            .where(*_row_cas(row, from_state, fenced=not stale))
            .values(
                state=new_state.value, state_detail=state_detail, next_attempt_at=None
            )
            .execution_options(synchronize_session=False)
        )
    )
    if updated != 1:
        increment_counter(
            "xagent.task.auto_resume.skipped", attributes={"outcome": "raced"}
        )
        return _RACED
    detail: dict[str, Any] = {"at": at, "state": new_state.value}
    if state_detail is not None:
        detail["state_detail"] = state_detail
    if event_detail:
        detail.update(event_detail)
    attempts = int(row.total_resumes or 0)
    db.add(
        TaskRecoveryEvent(
            task_id=int(row.task_id),
            run_id=row.run_id,
            event=(event or _GIVE_UP_EVENTS[new_state]).value,
            reason=row.reason,
            attempt=attempts,
            detail=detail,
        )
    )
    if stale:
        if task.status in (TaskStatus.COMPLETED, TaskStatus.FAILED):
            sync_trigger_run_status(db, task, TaskStatus(task.status))
    else:
        sync_trigger_run_status(
            db,
            task,
            TaskStatus.PAUSED,
            error_message=_trigger_error(new_state, state_detail),
        )
    db.flush()
    if stale:
        increment_counter("xagent.task.auto_resume.stale", attributes={"operation": at})
        logger.info(
            "component=auto-resume task_id=%s run_id=%s reason=%s state=%s "
            "attempt=%s next_attempt_at=None fence_mismatch=%s",
            row.task_id,
            row.run_id,
            row.reason,
            new_state.value,
            attempts,
            (event_detail or {}).get("mismatch"),
        )
        return _CandidateResult(_Outcome.STALE)
    increment_counter(
        "xagent.task.auto_resume.gave_up",
        attributes={"outcome": new_state.value, "operation": at},
    )
    logger.warning(
        "component=auto-resume task_id=%s run_id=%s reason=%s state=%s "
        "attempt=%s next_attempt_at=None state_detail=%s",
        row.task_id,
        row.run_id,
        row.reason,
        new_state.value,
        attempts,
        state_detail,
    )
    return _CandidateResult(
        _Outcome.GAVE_UP,
        _notice(row, new_state, state_detail=state_detail, attempt=attempts),
    )


def _load_locked_no_commit(
    db: Session, task_id: int
) -> tuple[Task | None, TaskAutoRecovery | None]:
    """Read the task, then the recovery row, in lock order.

    On PostgreSQL the caller already holds the ``tasks`` row lock; locking
    the recovery row then may wait, but only on a writer that also holds
    ``tasks`` first, so it cannot close a cycle.
    """

    task = db.get(Task, task_id, populate_existing=True)
    row = db.get(
        TaskAutoRecovery, task_id, with_for_update=True, populate_existing=True
    )
    return task, row


def _limit_state(
    row: TaskAutoRecovery, kind: str | None, now: datetime
) -> TaskAutoRecoveryState | None:
    limit = auto_resume_limit(
        reason=row.reason,
        kind=kind,
        interrupted_at=row.interrupted_at,
        episode_started_at=row.episode_started_at,
        no_progress_resumes=int(row.no_progress_resumes or 0),
        total_resumes=int(row.total_resumes or 0),
        now=now,
    )
    return None if limit is None else auto_resume_recovery_state(limit)


# --------------------------------------------------------------------------
# Dispatch
# --------------------------------------------------------------------------


class _AdmissionFull(Exception):
    """``AdmissionQueueFull`` carrying the fence the candidate was seen at."""

    def __init__(self, task_id: int, paused_state_version: int) -> None:
        super().__init__(task_id, paused_state_version)
        self.task_id = task_id
        self.paused_state_version = paused_state_version


def _process_due_candidate_no_commit(
    db: Session, task_id: int, *, now: datetime, can_dispatch: bool
) -> _CandidateResult:
    """Dispatch one due ``scheduled`` row, or stop its automatic recovery.

    Stopping never consumes the dispatch budget, so rows that only need to
    be closed keep moving while in-flight resumes are at the cap. Writes,
    in order: the CAS, its event, then the command, which must be the last
    write before the caller commits.
    """

    task, row = _load_locked_no_commit(db, task_id)
    if (
        task is None
        or row is None
        or row.state != _SCHEDULED.value
        or row.next_attempt_at is None
        or _utc(row.next_attempt_at) > now
    ):
        return _RACED
    mismatch = _fence_mismatch(task, row)
    if mismatch is not None:
        return _give_up_no_commit(
            db,
            task,
            row,
            TaskAutoRecoveryState.STALE,
            from_state=_SCHEDULED,
            at="dispatch",
            state_detail=f"dispatch:{mismatch}",
            event_detail={"mismatch": mismatch},
        )
    # Recomputed: the task's kind decides its window and may have changed.
    eligibility = auto_recovery_eligibility(db, task)
    if not eligibility.eligible:
        return _give_up_no_commit(
            db,
            task,
            row,
            TaskAutoRecoveryState.INELIGIBLE,
            from_state=_SCHEDULED,
            at="dispatch",
            state_detail=eligibility.detail,
        )
    if not auto_resume_schedulable(eligibility.kind):
        return _give_up_no_commit(
            db,
            task,
            row,
            TaskAutoRecoveryState.MANUAL,
            from_state=_SCHEDULED,
            at="dispatch",
            state_detail=CHANNEL_HOLD_PENDING_DETAIL,
        )
    if scheduled_trigger_superseded(db, task):
        return _give_up_no_commit(
            db,
            task,
            row,
            TaskAutoRecoveryState.SUPERSEDED,
            from_state=_SCHEDULED,
            at="dispatch",
        )
    limit_state = _limit_state(row, eligibility.kind, now)
    if limit_state is not None:
        return _give_up_no_commit(
            db, task, row, limit_state, from_state=_SCHEDULED, at="dispatch"
        )
    if not can_dispatch:
        return _CandidateResult(_Outcome.CAPACITY)

    run_id = str(row.run_id)
    paused_state_version = int(row.paused_state_version)
    attempt = int(row.total_resumes or 0) + 1
    command_id = auto_resume_command_id(paused_state_version, attempt, run_id)
    updated = _rowcount(
        db.execute(
            update(TaskAutoRecovery)
            .where(*_row_cas(row, _SCHEDULED, fenced=True))
            .values(
                state=_DISPATCHED.value,
                state_detail=None,
                no_progress_resumes=TaskAutoRecovery.no_progress_resumes + 1,
                total_resumes=TaskAutoRecovery.total_resumes + 1,
                last_command_id=command_id,
                next_attempt_at=now + timedelta(seconds=DISPATCH_CHECK_GRACE_SECONDS),
            )
            .execution_options(synchronize_session=False)
        )
    )
    if updated != 1:
        increment_counter(
            "xagent.task.auto_resume.skipped", attributes={"outcome": "raced"}
        )
        return _RACED
    db.add(
        TaskRecoveryEvent(
            task_id=task_id,
            run_id=run_id,
            event=TaskRecoveryEventType.AUTO_RESUMED.value,
            reason=row.reason,
            attempt=attempt,
            detail={"kind": eligibility.kind},
        )
    )
    db.flush()
    try:
        staged = stage_task_command(
            db,
            task_id=task_id,
            actor_user_id=int(task.user_id),
            command_id=command_id,
            kind=TaskCommandKind.RESUME,
            payload={
                "type": "resume_task",
                "auto_resume": {
                    "expected_run_id": run_id,
                    "expected_state_version": paused_state_version,
                    "reason": row.reason,
                    "attempt": attempt,
                },
            },
            target_run_id=run_id,
        )
    except AdmissionQueueFull as exc:
        raise _AdmissionFull(task_id, paused_state_version) from exc
    if not staged.created:
        # Only a race that dispatched this exact (fence, attempt) gets here;
        # that command exists and will run, so the CAS still commits.
        increment_counter(
            "xagent.task.auto_resume.skipped",
            attributes={"outcome": "command_exists"},
        )
        logger.warning(
            "component=auto-resume task_id=%s run_id=%s command %s already "
            "existed when dispatching attempt %s",
            task_id,
            run_id,
            command_id,
            attempt,
        )
    next_check_at = now + timedelta(seconds=DISPATCH_CHECK_GRACE_SECONDS)
    increment_counter(
        "xagent.task.auto_resume.dispatched",
        attributes={
            "operation": str(row.reason),
            "outcome": str(attempt) if attempt < 4 else "4+",
        },
    )
    logger.info(
        "component=auto-resume task_id=%s run_id=%s reason=%s state=%s "
        "attempt=%s next_attempt_at=%s",
        task_id,
        run_id,
        row.reason,
        _DISPATCHED.value,
        attempt,
        next_check_at,
    )
    return _CandidateResult(
        _Outcome.DISPATCHED,
        _notice(row, _DISPATCHED, state_detail=None, attempt=attempt),
        staged=staged.created,
        dispatch_delay_seconds=(now - _utc(row.next_attempt_at)).total_seconds(),
    )


def _postpone_after_admission_full(
    task_id: int, paused_state_version: int, now: datetime
) -> None:
    """Push a candidate the admission queue refused; a single-row CAS."""

    from ..models.database import get_session_local

    with get_session_local()() as db:
        try:
            db.execute(
                update(TaskAutoRecovery)
                .where(
                    TaskAutoRecovery.task_id == task_id,
                    TaskAutoRecovery.state == _SCHEDULED.value,
                    TaskAutoRecovery.paused_state_version == paused_state_version,
                )
                .values(
                    next_attempt_at=now
                    + timedelta(seconds=ADMISSION_FULL_DELAY_SECONDS)
                )
                .execution_options(synchronize_session=False)
            )
            db.commit()
        except Exception:
            db.rollback()
            raise


# --------------------------------------------------------------------------
# Housekeeping of dispatched rows
# --------------------------------------------------------------------------


def _housekeep_candidate_no_commit(
    db: Session, task_id: int, *, now: datetime
) -> _CandidateResult:
    """Check one ``dispatched`` row whose check time has come.

    - fence moved: the resume took effect or a person acted; confirm by
      clearing ``next_attempt_at`` (no event);
    - command failed or missing at an unchanged fence: ``dispatch_failed``;
    - command completed at an unchanged fence (e.g. ``already_in_progress``):
      the resume did not happen; schedule it again unless a limit stops it;
    - command still queued: check again after the grace period.
    """

    task, row = _load_locked_no_commit(db, task_id)
    if (
        task is None
        or row is None
        or row.state != _DISPATCHED.value
        or row.next_attempt_at is None
        or _utc(row.next_attempt_at) > now
    ):
        return _RACED
    command = (
        db.query(TaskExecutionCommand)
        .filter(
            TaskExecutionCommand.task_id == task_id,
            TaskExecutionCommand.command_id == row.last_command_id,
        )
        .one_or_none()
        if row.last_command_id is not None
        else None
    )
    if _fence_mismatch(task, row) is not None:
        return _update_dispatched_no_commit(
            db, row, fenced=False, outcome=_Outcome.CONFIRMED, next_attempt_at=None
        )
    command_result: dict[str, Any] = (
        command.result
        if command is not None and isinstance(command.result, dict)
        else {}
    )
    if command is None or command.status == COMMAND_FAILED:
        rejection = command_result.get("rejection_reason")
        return _give_up_no_commit(
            db,
            task,
            row,
            TaskAutoRecoveryState.DISPATCH_FAILED,
            from_state=_DISPATCHED,
            at="housekeeping",
            event_detail={
                "command": "missing" if command is None else "failed",
                "rejection_reason": (rejection if isinstance(rejection, str) else None),
            },
        )
    if command.status != COMMAND_COMPLETED:
        return _update_dispatched_no_commit(
            db,
            row,
            fenced=True,
            outcome=_Outcome.REARMED,
            next_attempt_at=now + timedelta(seconds=DISPATCH_CHECK_GRACE_SECONDS),
        )

    resume_outcome = command_result.get("resume_outcome")
    after = resume_outcome if isinstance(resume_outcome, str) else "unknown"
    eligibility = auto_recovery_eligibility(db, task)
    limit_state = _limit_state(row, eligibility.kind, now)
    if limit_state is not None:
        return _give_up_no_commit(
            db,
            task,
            row,
            limit_state,
            from_state=_DISPATCHED,
            at="housekeeping",
            event_detail={"after": after},
        )
    # The counters are not rolled back: the dispatch happened, and the limits
    # are what keep this from looping.
    next_attempt_at = next_auto_resume_at(
        row.reason, int(row.no_progress_resumes or 0), now, _AUTO_RESUME_RNG
    )
    updated = _rowcount(
        db.execute(
            update(TaskAutoRecovery)
            .where(*_row_cas(row, _DISPATCHED, fenced=True))
            .values(
                state=_SCHEDULED.value,
                state_detail=None,
                next_attempt_at=next_attempt_at,
            )
            .execution_options(synchronize_session=False)
        )
    )
    if updated != 1:
        return _RACED
    attempt = int(row.total_resumes or 0) + 1
    db.add(
        TaskRecoveryEvent(
            task_id=task_id,
            run_id=row.run_id,
            event=TaskRecoveryEventType.SCHEDULED.value,
            reason=row.reason,
            attempt=attempt,
            next_attempt_at=next_attempt_at,
            detail={"after": after},
        )
    )
    db.flush()
    increment_counter(
        "xagent.task.auto_resume.rescheduled", attributes={"outcome": after}
    )
    logger.info(
        "component=auto-resume task_id=%s run_id=%s reason=%s state=%s "
        "attempt=%s next_attempt_at=%s after=%s",
        task_id,
        row.run_id,
        row.reason,
        _SCHEDULED.value,
        attempt,
        next_attempt_at,
        after,
    )
    return _CandidateResult(
        _Outcome.RESCHEDULED,
        _notice(
            row,
            _SCHEDULED,
            state_detail=None,
            attempt=attempt,
            next_attempt_at=next_attempt_at,
        ),
    )


def _update_dispatched_no_commit(
    db: Session,
    row: TaskAutoRecovery,
    *,
    fenced: bool,
    outcome: _Outcome,
    next_attempt_at: datetime | None,
) -> _CandidateResult:
    updated = _rowcount(
        db.execute(
            update(TaskAutoRecovery)
            .where(*_row_cas(row, _DISPATCHED, fenced=fenced))
            .values(next_attempt_at=next_attempt_at)
            .execution_options(synchronize_session=False)
        )
    )
    if updated != 1:
        return _RACED
    if outcome is _Outcome.CONFIRMED:
        logger.info(
            "component=auto-resume task_id=%s run_id=%s reason=%s state=%s "
            "attempt=%s next_attempt_at=None confirmed",
            row.task_id,
            row.run_id,
            row.reason,
            _DISPATCHED.value,
            row.total_resumes,
        )
    return _CandidateResult(outcome)


# --------------------------------------------------------------------------
# Drain (switch off)
# --------------------------------------------------------------------------


def _drain_candidate_no_commit(
    db: Session, task_id: int, *, now: datetime
) -> _CandidateResult:
    """Stop one ``scheduled`` row because automatic resume is switched off.

    Not retroactive: a row stays stopped when the switch comes back. A row
    whose fence already moved is ``stale`` like at dispatch.
    """

    task, row = _load_locked_no_commit(db, task_id)
    if task is None or row is None or row.state != _SCHEDULED.value:
        return _RACED
    mismatch = _fence_mismatch(task, row)
    if mismatch is not None:
        return _give_up_no_commit(
            db,
            task,
            row,
            TaskAutoRecoveryState.STALE,
            from_state=_SCHEDULED,
            at="drain",
            state_detail=f"drain:{mismatch}",
            event_detail={"mismatch": mismatch},
        )
    return _give_up_no_commit(
        db,
        task,
        row,
        TaskAutoRecoveryState.MANUAL,
        from_state=_SCHEDULED,
        at="drain",
        state_detail=AUTO_RESUME_DISABLED_DETAIL,
        event=TaskRecoveryEventType.SKIPPED_DISABLED,
    )


# --------------------------------------------------------------------------
# Candidate selection
# --------------------------------------------------------------------------

_Cursor = tuple[datetime, int]


def _use_postgresql_partitioning() -> bool:
    """Return whether candidate row locking can partition sweeper workers."""

    from ..models.database import get_engine

    return get_engine().dialect.name == "postgresql"


def _candidates_statement(
    state: TaskAutoRecoveryState,
    *,
    now: datetime | None,
    after: _Cursor | None,
) -> Any:
    """Rows in ``state``, due by ``now`` when given, in ``(time, task)`` order.

    Joined to ``tasks`` so the PostgreSQL path can lock the task row (and
    skip a row whose task another transaction holds) before the recovery
    row.
    """

    statement = (
        select(TaskAutoRecovery.task_id, TaskAutoRecovery.next_attempt_at)
        .join(Task, Task.id == TaskAutoRecovery.task_id)
        .where(
            TaskAutoRecovery.state == state.value,
            TaskAutoRecovery.next_attempt_at.is_not(None),
        )
    )
    if now is not None:
        statement = statement.where(TaskAutoRecovery.next_attempt_at <= now)
    if after is not None:
        after_at, after_task_id = after
        statement = statement.where(
            or_(
                TaskAutoRecovery.next_attempt_at > after_at,
                and_(
                    TaskAutoRecovery.next_attempt_at == after_at,
                    TaskAutoRecovery.task_id > after_task_id,
                ),
            )
        )
    return statement.order_by(
        TaskAutoRecovery.next_attempt_at, TaskAutoRecovery.task_id
    )


def select_next_candidate_for_update(
    db: Session,
    state: TaskAutoRecoveryState,
    *,
    now: datetime | None,
    after: _Cursor | None = None,
) -> _Cursor | None:
    """Lock and return one candidate's ``tasks`` row, skipping held ones.

    PostgreSQL only: renders ``FOR UPDATE OF tasks SKIP LOCKED``, so peer
    sweepers partition the candidates and a row whose task a settlement
    holds is left for a later tick. The caller processes the candidate in
    this same transaction.
    """

    picked = db.execute(
        _candidates_statement(state, now=now, after=after)
        .with_for_update(of=Task, skip_locked=True)
        .limit(1)
    ).first()
    if picked is None:
        return None
    return picked.next_attempt_at, int(picked.task_id)


def _for_each_candidate(
    state: TaskAutoRecoveryState,
    *,
    now: datetime | None,
    limit: int,
    process: Callable[[Session, int], bool],
) -> None:
    """Hand candidates to ``process`` one transaction each, until it says stop.

    ``process`` owns the commit or rollback and returns True to stop.
    PostgreSQL locks each candidate's task row in its own transaction;
    SQLite scans one page and relies on the compare-and-swaps.
    """

    from ..models.database import get_session_local

    SessionLocal = get_session_local()
    if _use_postgresql_partitioning():
        cursor: _Cursor | None = None
        for _ in range(limit):
            with SessionLocal() as db:
                try:
                    picked = select_next_candidate_for_update(
                        db, state, now=now, after=cursor
                    )
                except Exception:
                    db.rollback()
                    raise
                if picked is None:
                    db.rollback()
                    return
                cursor = picked
                if process(db, picked[1]):
                    return
        return

    with SessionLocal() as scan_db:
        task_ids = [
            int(task_id)
            for task_id, _at in scan_db.execute(
                _candidates_statement(state, now=now, after=None).limit(limit)
            ).all()
        ]
    for task_id in task_ids:
        with SessionLocal() as db:
            if process(db, task_id):
                return


# --------------------------------------------------------------------------
# Tick
# --------------------------------------------------------------------------


@dataclass
class _TickTally:
    dispatched: int = 0
    gave_up: int = 0
    stale: int = 0
    rescheduled: int = 0
    confirmed: int = 0
    rearmed: int = 0
    raced: int = 0
    failed: int = 0
    staged_commands: int = 0
    inflight: int | None = None
    budget: int | None = None
    admission_full: bool = False
    notices: list[RecoveryNotice] = field(default_factory=list)

    def record(self, result: _CandidateResult) -> None:
        if result.outcome is _Outcome.DISPATCHED:
            self.dispatched += 1
        elif result.outcome is _Outcome.GAVE_UP:
            self.gave_up += 1
        elif result.outcome is _Outcome.STALE:
            self.stale += 1
        elif result.outcome is _Outcome.RESCHEDULED:
            self.rescheduled += 1
        elif result.outcome is _Outcome.CONFIRMED:
            self.confirmed += 1
        elif result.outcome is _Outcome.REARMED:
            self.rearmed += 1
        elif result.outcome is _Outcome.RACED:
            self.raced += 1
        if result.staged:
            self.staged_commands += 1
        if result.notice is not None:
            self.notices.append(result.notice)


@dataclass(frozen=True)
class AutoResumeTickReport:
    """What one sweeper tick committed."""

    dispatched: int = 0
    gave_up: int = 0
    stale: int = 0
    rescheduled: int = 0
    confirmed: int = 0
    rearmed: int = 0
    raced: int = 0
    failed: int = 0
    staged_commands: int = 0
    # Not measured when the switch is off.
    inflight: int | None = None
    budget: int | None = None
    admission_full: bool = False
    notices: tuple[RecoveryNotice, ...] = ()

    @classmethod
    def from_tally(cls, tally: _TickTally) -> AutoResumeTickReport:
        return cls(
            dispatched=tally.dispatched,
            gave_up=tally.gave_up,
            stale=tally.stale,
            rescheduled=tally.rescheduled,
            confirmed=tally.confirmed,
            rearmed=tally.rearmed,
            raced=tally.raced,
            failed=tally.failed,
            staged_commands=tally.staged_commands,
            inflight=tally.inflight,
            budget=tally.budget,
            admission_full=tally.admission_full,
            notices=tuple(tally.notices),
        )


def _settle_candidate(
    db: Session,
    task_id: int,
    tally: _TickTally,
    work: Callable[[], _CandidateResult],
) -> _CandidateResult | None:
    """Run one candidate's work and end its transaction.

    A failure is logged and skipped so one bad row cannot stall the rest,
    except a pool timeout, which ends the tick as lease recovery does.
    ``_AdmissionFull`` propagates for the dispatch path to handle.
    """

    try:
        result = work()
        if result.writes:
            db.commit()
        else:
            db.rollback()
    except _AdmissionFull:
        db.rollback()
        raise
    except IntegrityError:
        db.rollback()
        increment_counter(
            "xagent.task.auto_resume.skipped", attributes={"outcome": "raced"}
        )
        result = _RACED
    except Exception as exc:
        db.rollback()
        if is_database_pool_timeout(exc):
            raise
        tally.failed += 1
        increment_counter("xagent.task.auto_resume.candidate_failed")
        logger.exception(
            "component=auto-resume task_id=%s candidate failed; retrying next tick",
            task_id,
        )
        return None
    tally.record(result)
    return result


def housekeep_dispatched(now: datetime, tally: _TickTally, *, limit: int) -> None:
    """Check up to ``limit`` dispatched rows whose check time has come."""

    def process(db: Session, task_id: int) -> bool:
        _settle_candidate(
            db,
            task_id,
            tally,
            lambda: _housekeep_candidate_no_commit(db, task_id, now=now),
        )
        return False

    _for_each_candidate(_DISPATCHED, now=now, limit=limit, process=process)


def drain_disabled_rows(now: datetime, tally: _TickTally, *, limit: int) -> None:
    """Stop up to ``limit`` scheduled rows, due or not, with the switch off."""

    def process(db: Session, task_id: int) -> bool:
        _settle_candidate(
            db,
            task_id,
            tally,
            lambda: _drain_candidate_no_commit(db, task_id, now=now),
        )
        return False

    _for_each_candidate(_SCHEDULED, now=None, limit=limit, process=process)


def count_inflight(db: Session) -> int:
    """Automatic resumes in flight across every process.

    A: auto-resumed runs still RUNNING (driven from the task status index).
    B: dispatched commands whose fence has not moved yet (pending claim,
    waiting for admission, or not yet at RESUME_REQUESTED). The two never
    overlap: a running resumed run has moved past the fence version.
    """

    running = db.execute(
        select(func.count())
        .select_from(Task)
        .join(
            TaskAutoRecovery,
            and_(
                TaskAutoRecovery.task_id == Task.id,
                TaskAutoRecovery.run_id == Task.run_id,
            ),
        )
        .where(
            task_status_predicate.eq(TaskStatus.RUNNING),
            TaskAutoRecovery.state == _DISPATCHED.value,
        )
    ).scalar_one()
    queued = db.execute(
        select(func.count())
        .select_from(TaskAutoRecovery)
        .join(Task, Task.id == TaskAutoRecovery.task_id)
        .where(
            TaskAutoRecovery.state == _DISPATCHED.value,
            TaskAutoRecovery.next_attempt_at.is_not(None),
            Task.run_id == TaskAutoRecovery.run_id,
            Task.state_version == TaskAutoRecovery.paused_state_version,
        )
    ).scalar_one()
    return int(running) + int(queued)


def scan_due(
    now: datetime, tally: _TickTally, *, dispatch_budget: int, scan_limit: int
) -> None:
    """Dispatch or close due scheduled rows.

    Stops once a dispatch is due with the budget spent, after ``scan_limit``
    candidates, or when the admission queue is full (its buckets are shared,
    so the next candidate would be refused too).
    """

    def process(db: Session, task_id: int) -> bool:
        can_dispatch = tally.dispatched < dispatch_budget
        try:
            result = _settle_candidate(
                db,
                task_id,
                tally,
                lambda: _process_due_candidate_no_commit(
                    db, task_id, now=now, can_dispatch=can_dispatch
                ),
            )
        except _AdmissionFull as full:
            tally.admission_full = True
            increment_counter(
                "xagent.task.auto_resume.skipped",
                attributes={"outcome": "admission_full"},
            )
            _postpone_after_admission_full(full.task_id, full.paused_state_version, now)
            logger.info(
                "component=auto-resume task_id=%s admission queue is full; "
                "postponed %ss",
                full.task_id,
                ADMISSION_FULL_DELAY_SECONDS,
            )
            return True
        if result is None:
            return False
        if result.dispatch_delay_seconds is not None:
            observe_value(
                "xagent.task.auto_resume.dispatch_delay_seconds",
                max(0.0, result.dispatch_delay_seconds),
                unit="s",
            )
        return result.outcome is _Outcome.CAPACITY

    _for_each_candidate(_SCHEDULED, now=now, limit=scan_limit, process=process)


def run_auto_resume_tick(*, now: datetime | None = None) -> AutoResumeTickReport:
    """One sweeper pass; synchronous, for a worker thread.

    With the switch off it only drains ``scheduled`` rows to ``manual``.
    Otherwise housekeeping runs first (it frees in-flight slots), then due
    rows are dispatched within ``min(MAX_PER_TICK, MAX_INFLIGHT - inflight)``.
    The dispatcher is notified once, after every dispatch has committed.
    """

    global _last_inflight
    from ..models.database import get_session_local

    now = _utc(now or utc_now())
    tally = _TickTally()
    max_per_tick = get_task_auto_resume_max_per_tick()
    try:
        if not get_task_auto_resume_enabled():
            drain_disabled_rows(now, tally, limit=max_per_tick)
            return AutoResumeTickReport.from_tally(tally)
        housekeep_dispatched(now, tally, limit=max_per_tick)
        with get_session_local()() as db:
            inflight = count_inflight(db)
        _last_inflight = inflight
        budget = max(
            0, min(max_per_tick, get_task_auto_resume_max_inflight() - inflight)
        )
        tally.inflight, tally.budget = inflight, budget
        if budget == 0:
            increment_counter(
                "xagent.task.auto_resume.skipped",
                attributes={"outcome": "inflight_cap"},
            )
        scan_due(now, tally, dispatch_budget=budget, scan_limit=4 * max_per_tick)
        return AutoResumeTickReport.from_tally(tally)
    finally:
        if tally.staged_commands:
            notify_task_command_dispatcher()


def _clock() -> float:
    return time.monotonic()


def _failure_backoff_seconds(poll_interval_seconds: int, failures: int) -> int:
    if failures == 0:
        return poll_interval_seconds
    return min(
        poll_interval_seconds * int(2 ** min(failures, _MAX_FAILURE_BACKOFF_EXPONENT)),
        _MAX_FAILURE_BACKOFF_SECONDS,
    )


def _register_inflight_gauge() -> None:
    register_observable_gauge(
        "xagent.task.auto_resume.inflight",
        lambda: float(_last_inflight),
        unit="{task}",
        description="Automatic resumes in flight, as of this process's last tick",
    )


async def run_task_auto_resume_loop(*, poll_interval_seconds: int) -> None:
    """Run sweeper ticks until cancelled.

    A failed tick backs off exponentially (capped at 60s), is logged on the
    first failure and then at most once a minute, and after three in a row
    raises TASK_AUTO_RESUME_UNAVAILABLE until a tick succeeds.
    """

    _register_inflight_gauge()
    failures = 0
    last_failure_log = float("-inf")
    while True:
        try:
            report = await run_db_io_cancellation_safe(
                lambda: run_auto_resume_tick(now=utc_now())
            )
            await publish_recovery_notices(report.notices)
            if failures:
                logger.info(
                    "Auto-resume sweeper recovered after %s failed tick(s)", failures
                )
            failures = 0
            clear_degradation(TASK_AUTO_RESUME_UNAVAILABLE)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            failures += 1
            increment_counter("xagent.task.auto_resume.tick_failed")
            now_monotonic = _clock()
            if (
                failures == 1
                or now_monotonic - last_failure_log >= _FAILURE_LOG_INTERVAL_SECONDS
            ):
                last_failure_log = now_monotonic
                if is_database_pool_timeout(exc):
                    logger.warning(
                        "Auto-resume tick skipped after database pool timeout "
                        "(failures=%s)",
                        failures,
                    )
                else:
                    logger.warning(
                        "Auto-resume tick failed (failures=%s)",
                        failures,
                        exc_info=True,
                    )
            if failures >= _UNAVAILABLE_AFTER_FAILURES:
                register_degradation(
                    TASK_AUTO_RESUME_UNAVAILABLE,
                    f"{failures} consecutive failed auto-resume ticks",
                )
        await asyncio.sleep(_failure_backoff_seconds(poll_interval_seconds, failures))


# --------------------------------------------------------------------------
# Claim-time guard
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class AutoResumeClaimDecision:
    """Whether ``resume_task`` may go on with an automatic RESUME."""

    proceed: bool
    why: str | None = None
    notices: tuple[RecoveryNotice, ...] = ()


_PROCEED = AutoResumeClaimDecision(proceed=True)


def _valid_auto_resume_payload(auto_resume: Any) -> bool:
    if not isinstance(auto_resume, dict):
        return False
    run_id = auto_resume.get("expected_run_id")
    version = auto_resume.get("expected_state_version")
    return (
        isinstance(run_id, str)
        and bool(run_id)
        and isinstance(version, int)
        and not isinstance(version, bool)
    )


def _as_task_status(value: Any) -> TaskStatus | None:
    if isinstance(value, TaskStatus):
        return value
    try:
        return TaskStatus(str(value))
    except ValueError:
        return None


def check_auto_resume_claim_sync(
    *,
    task_id: int,
    command_id: Any,
    auto_resume: Any,
    status: Any,
    control_state: Any,
    run_id: str | None,
    state_version: int | None,
    attempt_count: Any,
) -> AutoResumeClaimDecision:
    """Decide, at claim, whether an automatic RESUME still applies.

    ``status`` .. ``state_version`` are the snapshot ``resume_task`` admits
    from; its RESUME_REQUESTED transition is fenced on the same version, so
    a row that moves after this check makes the transition defer and the
    retry lands here again.

    - Not the sweeper's command (no reserved id) or a malformed payload:
      ``ValueError``, which ``resume_task`` rejects as an invalid payload.
    - This command's own earlier attempt already wrote RESUME_REQUESTED and
      died: proceed, or the task would stay ``resume_requested``. Commands
      on a task are serialized, so only that attempt can have written it
      (exactly one version past the fence), whatever the switch says now.
    - At the fence with the switch on: proceed.
    - Otherwise skip: ``auto_resume_disabled`` at the fence (the row rests
      in ``manual`` and a RUNNING TriggerRun fails), else ``stale`` (a
      person moved the task on; WAITING_FOR_USER and PAUSE_REQUESTED land
      here). The row is updated only while it still names this command.
    """

    if not is_auto_resume_command_id(command_id) or not _valid_auto_resume_payload(
        auto_resume
    ):
        raise ValueError("invalid auto_resume payload")
    expected_run_id = str(auto_resume["expected_run_id"])
    expected_version = int(auto_resume["expected_state_version"])
    task_status = _as_task_status(status)
    version = int(state_version or 0)
    if (
        int(attempt_count or 0) > 1
        and task_status is TaskStatus.PAUSED
        and control_state == TaskControlState.RESUME_REQUESTED.value
        and run_id == expected_run_id
        and version == expected_version + 1
    ):
        return _PROCEED
    at_fence = (
        task_status is TaskStatus.PAUSED
        and control_state == TaskControlState.PAUSED.value
        and run_id == expected_run_id
        and version == expected_version
    )
    if at_fence and get_task_auto_resume_enabled():
        return _PROCEED
    why = AUTO_RESUME_DISABLED_DETAIL if at_fence else "stale"
    notices = _close_skipped_claim(task_id=task_id, command_id=command_id, why=why)
    logger.info(
        "auto resume skipped task_id=%s run_id=%s why=%s component=auto-resume",
        task_id,
        expected_run_id,
        why,
    )
    return AutoResumeClaimDecision(proceed=False, why=why, notices=notices)


def _close_skipped_claim(
    *, task_id: int, command_id: str, why: str
) -> tuple[RecoveryNotice, ...]:
    """Move the row a skipped claim belongs to out of ``dispatched``."""

    from ..models.database import get_session_local

    with get_session_local()() as db:
        try:
            if _use_postgresql_partitioning():
                # Lock order: tasks before the recovery row.
                db.execute(select(Task.id).where(Task.id == task_id).with_for_update())
            task, row = _load_locked_no_commit(db, task_id)
            if (
                task is None
                or row is None
                or row.state != _DISPATCHED.value
                or row.last_command_id != command_id
            ):
                db.rollback()
                return ()
            if why == AUTO_RESUME_DISABLED_DETAIL:
                result = _give_up_no_commit(
                    db,
                    task,
                    row,
                    TaskAutoRecoveryState.MANUAL,
                    from_state=_DISPATCHED,
                    at="claim",
                    state_detail=AUTO_RESUME_DISABLED_DETAIL,
                    event=TaskRecoveryEventType.SKIPPED_DISABLED,
                )
            else:
                mismatch = _fence_mismatch(task, row)
                result = _give_up_no_commit(
                    db,
                    task,
                    row,
                    TaskAutoRecoveryState.STALE,
                    from_state=_DISPATCHED,
                    at="claim",
                    state_detail=f"claim:{mismatch}" if mismatch else "claim",
                    event_detail={"mismatch": mismatch},
                )
            if result.writes:
                db.commit()
            else:
                db.rollback()
        except Exception:
            db.rollback()
            raise
    return (result.notice,) if result.notice is not None else ()
