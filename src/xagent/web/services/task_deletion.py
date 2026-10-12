"""Shared, foreign-key-safe deletion of task-owned database rows."""

from datetime import datetime, timezone

from sqlalchemy.orm import Session

from ..models.task import (
    DAGExecution,
    Task,
    TraceCheckpointBlob,
    TraceEvent,
    TraceMessageBlob,
)
from ..models.task_auto_recovery import (
    TaskAutoRecovery,
    TaskAutoRecoveryState,
    TaskRecoveryEvent,
)
from ..models.task_interaction import TaskInteractionRequest
from ..models.trigger import TriggerRun, TriggerRunStatus
from .task_file_lifecycle import (
    DetachmentReason,
    detach_task_files,
    lock_attachment_task,
)
from .task_interaction_schema import interaction_requests_table_exists

# TriggerRun error for a run whose task was deleted while automatic resume
# still owned its outcome.
TASK_DELETED_DURING_AUTO_RESUME_TRIGGER_ERROR = (
    "Task deleted while an automatic resume was pending."
)


def _fail_trigger_runs_awaiting_auto_resume(db: Session, task_id: int) -> None:
    """Fail the TriggerRun that a scheduled or dispatched resume kept RUNNING.

    Only automatic resume leaves a TriggerRun RUNNING across a pause, and
    only its resumed run would settle it. Deleting the task nulls
    ``trigger_runs.task_id`` (ON DELETE SET NULL), after which nothing could.
    A bulk statement: this session has autoflush disabled, and it must land
    before the task delete. Lock order holds: the task row is already locked.
    """

    awaiting = (
        db.query(TaskAutoRecovery.task_id)
        .filter(
            TaskAutoRecovery.task_id == task_id,
            TaskAutoRecovery.state.in_(
                (
                    TaskAutoRecoveryState.SCHEDULED.value,
                    TaskAutoRecoveryState.DISPATCHED.value,
                )
            ),
        )
        .first()
    )
    if awaiting is None:
        return
    db.query(TriggerRun).filter(
        TriggerRun.task_id == task_id,
        TriggerRun.status.in_(
            (TriggerRunStatus.PENDING.value, TriggerRunStatus.RUNNING.value)
        ),
    ).update(
        {
            TriggerRun.status: TriggerRunStatus.FAILED.value,
            TriggerRun.finished_at: datetime.now(timezone.utc),
            TriggerRun.error_message: TASK_DELETED_DURING_AUTO_RESUME_TRIGGER_ERROR,
        },
        synchronize_session=False,
    )


def purge_task_rows(
    db: Session,
    *,
    task_id: int,
    detached_reason: DetachmentReason,
) -> bool:
    """Delete one task and its non-cascading rows in a caller-owned transaction.

    ``detached_reason`` is required: ``task_deleted`` for deletion/expiry or
    ``task_create_failed`` for failed-create compensation. Marking shares this
    transaction with deletion; retained file bytes are reclaimed later.

    ``task_interaction_requests`` rows, unlike ``UploadedFile``, are deleted
    outright: the CASCADE on ``task_id`` would remove them anyway, and the
    delete must run before the ``trace_events`` delete below (see the
    ordering comment at that call).
    """

    task = lock_attachment_task(db, task_id, deleting=True)
    if task is None:
        return False
    detach_task_files(db, task_id=task_id, reason=detached_reason)

    # NULL the checkpoint pointer columns before the trace_events delete
    # below: last_checkpoint_trace_event_id FKs to trace_events.id, so a
    # task still pointing at a row would block (or, without DB-level
    # enforcement, orphan) that delete. A bulk statement, not an ORM
    # attribute assignment, because this session has autoflush disabled --
    # an attribute assignment would not reach the database until a later
    # flush, by which point the trace_events delete has already run.
    db.query(Task).filter(Task.id == task_id).update(
        {
            Task.last_checkpoint_event_id: None,
            Task.last_checkpoint_trace_event_id: None,
        },
        synchronize_session=False,
    )

    # Delete this task's interaction rows before the trace_events delete
    # below. task_interaction_requests.resume_trace_event_id FKs to
    # trace_events.id with ON DELETE SET NULL, and
    # ck_task_interaction_requests_active_anchor requires an active row to
    # hold a non-NULL anchor -- so deleting the anchored trace row first
    # makes the database NULL a column the CHECK forbids being NULL, and the
    # whole purge fails with an IntegrityError. Terminal rows are unaffected
    # (the CHECK's status <> 'active' branch already satisfies it); the
    # asymmetry is by design, and the ordering here is what makes the active
    # case work too.
    #
    # Plain delete, not terminate-then-delete: tasks.id cascades to these
    # rows anyway, so terminating first would cost an extra write and imply
    # -- falsely -- that an interaction row can outlive the purge of its
    # task.
    #
    # Gated on table presence: a deployment upgraded to a revision before
    # this table exists must still be able to delete tasks.
    if interaction_requests_table_exists(db):
        db.query(TaskInteractionRequest).filter(
            TaskInteractionRequest.task_id == task_id
        ).delete(synchronize_session=False)

    # Explicit rather than relying on ON DELETE CASCADE: SQLite only enforces
    # foreign keys when the pragma is on. No presence gate, unlike the
    # interaction table: an Alembic upgrade can stop short of these tables,
    # but ``Base.metadata.create_all`` runs right after ``try_upgrade_db`` on
    # every startup, so they exist before the process serves anything.
    _fail_trigger_runs_awaiting_auto_resume(db, task_id)
    db.query(TaskRecoveryEvent).filter(TaskRecoveryEvent.task_id == task_id).delete(
        synchronize_session=False
    )
    db.query(TaskAutoRecovery).filter(TaskAutoRecovery.task_id == task_id).delete(
        synchronize_session=False
    )

    db.query(TraceCheckpointBlob).filter(TraceCheckpointBlob.task_id == task_id).delete(
        synchronize_session=False
    )
    db.query(TraceMessageBlob).filter(TraceMessageBlob.task_id == task_id).delete(
        synchronize_session=False
    )
    db.query(TraceEvent).filter(TraceEvent.task_id == task_id).delete(
        synchronize_session=False
    )
    db.query(DAGExecution).filter(DAGExecution.task_id == task_id).delete(
        synchronize_session=False
    )
    db.delete(task)
    return True
