"""Exact-run journal for input whose checkpoint write must survive a restart."""

from __future__ import annotations

import logging
from typing import Any, cast

from pydantic import BaseModel, ConfigDict
from sqlalchemy import select

from ...core.agent.checkpoint import CheckpointReadError
from ...core.agent.runner import (
    UserMessageInjectionOutcome,
    UserMessageInjectionRejectedError,
)
from ..models.database import get_session_local
from ..models.task import Task
from .db_runtime import run_db_io_cancellation_safe
from .llm_utils import AutoModelUnavailableError
from .managed_file_ref import DurableStorageOperationError
from .task_lease_service import TaskLease, TaskLeaseLostError, lock_task_lease_no_commit

logger = logging.getLogger(__name__)


class PendingInjection(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    run_id: str
    owner_id: int
    turn_id: str
    execution_message: str
    display_message: str
    files: list[dict[str, Any]] | None = None
    interaction_id: int | None = None
    reply_command_id: str | None = None
    auto_resume: bool = True


def _journal(
    lease: TaskLease, pending: dict[str, Any] | None
) -> PendingInjection | None:
    with get_session_local()() as db, db.begin():
        if not lock_task_lease_no_commit(db, lease):
            raise TaskLeaseLostError("Injection owner changed")
        task = db.execute(select(Task).where(Task.id == lease.task_id)).scalar_one()
        saved = (
            PendingInjection.model_validate(task.pending_injection)
            if task.pending_injection
            else None
        )
        if saved is not None and (
            saved.run_id != lease.run_id or saved.owner_id != task.user_id
        ):
            raise TaskLeaseLostError(
                "Injection journal belongs to another run or owner"
            )
        if pending is None:
            return saved
        assert lease.run_id is not None
        candidate = PendingInjection(
            run_id=lease.run_id, owner_id=int(task.user_id), **pending
        )
        if (
            saved is not None
            and saved.model_copy(update={"auto_resume": True}) != candidate
        ):
            raise TaskLeaseLostError("An earlier input still requires settlement")
        setattr(task, "pending_injection", candidate.model_dump(mode="json"))
        return candidate


def clear_injection(lease: TaskLease, turn_id: str, *, rejected: bool = False) -> None:
    with get_session_local()() as db, db.begin():
        if not lock_task_lease_no_commit(db, lease):
            raise TaskLeaseLostError("Injection owner changed before settlement")
        task = db.get(Task, lease.task_id)
        assert task is not None
        if task.pending_injection is not None:
            pending = PendingInjection.model_validate(task.pending_injection)
            if (
                pending.run_id != lease.run_id
                or pending.turn_id != turn_id
                or pending.owner_id != task.user_id
            ):
                raise TaskLeaseLostError("Injection identity changed before settlement")
            if rejected:
                from ..models.chat_message import TaskChatMessage

                _settle_reply_command(db, task, pending, "not_resumable")

                db.query(TaskChatMessage).filter(
                    TaskChatMessage.task_id == lease.task_id,
                    TaskChatMessage.turn_id == turn_id,
                    TaskChatMessage.role == "user",
                    TaskChatMessage.delivery_status.in_(("pending", "outcome_unknown")),
                ).update(
                    {TaskChatMessage.delivery_status: "failed"},
                    synchronize_session=False,
                )
            setattr(task, "pending_injection", None)


async def load_pending_injection(lease: TaskLease) -> PendingInjection | None:
    return await run_db_io_cancellation_safe(lambda: _journal(lease, None))


async def post_journaled_message(
    agent_service: Any,
    lease: TaskLease,
    *,
    execution_message: str,
    display_message: str,
    turn_id: str,
    files: list[dict[str, Any]] | None = None,
    interaction_id: int | None = None,
    reply_command_id: str | None = None,
    request_interrupt: bool,
    reason: str,
) -> UserMessageInjectionOutcome:
    await run_db_io_cancellation_safe(
        lambda: _journal(
            lease,
            dict(
                execution_message=execution_message,
                display_message=display_message,
                turn_id=turn_id,
                files=files,
                interaction_id=interaction_id,
                reply_command_id=reply_command_id,
            ),
        )
    )
    try:
        outcome = await agent_service.post_user_message(
            str(lease.task_id),
            execution_message=execution_message,
            display_message=display_message,
            turn_id=turn_id,
            **({"files": files} if files is not None else {}),
            request_interrupt=request_interrupt,
            reason=reason,
        )
    except (
        CheckpointReadError,
        UserMessageInjectionRejectedError,
        AutoModelUnavailableError,
        DurableStorageOperationError,
    ):
        # These prove that this attempt never applied input. A cancellation,
        # lease loss or arbitrary write exception does not prove absence.
        await run_db_io_cancellation_safe(lambda: clear_injection(lease, turn_id))
        raise
    if not outcome or outcome is UserMessageInjectionOutcome.POSTED_REPLAY:
        await run_db_io_cancellation_safe(lambda: clear_injection(lease, turn_id))
    return cast(UserMessageInjectionOutcome, outcome)


async def settle_journaled_message(
    agent_service: Any, lease: TaskLease, pending: PendingInjection
) -> UserMessageInjectionOutcome:
    try:
        outcome = await agent_service.settle_injection_against_checkpoint(
            str(lease.task_id),
            execution_message=pending.execution_message,
            display_message=pending.display_message,
            turn_id=pending.turn_id,
            files=pending.files,
        )
    except UserMessageInjectionRejectedError:
        # Read-back proved absence, but this recovery write failed. Keep the
        # journal for the next owner rather than inviting a new message ID.
        return UserMessageInjectionOutcome.OUTCOME_UNKNOWN
    except CheckpointReadError:
        return UserMessageInjectionOutcome.OUTCOME_UNKNOWN
    # A replay here confirms THIS journal's original write. Its saved
    # interaction id can be closed, unlike a newly submitted replay.
    if outcome is UserMessageInjectionOutcome.POSTED_REPLAY:
        return UserMessageInjectionOutcome.POSTED_FRESH
    if not outcome:
        await run_db_io_cancellation_safe(
            lambda: clear_injection(lease, pending.turn_id, rejected=True)
        )
    return cast(UserMessageInjectionOutcome, outcome)


def complete_injection(lease: TaskLease, turn_id: str) -> None:
    """Resolve receipt, interaction and journal atomically under the resume owner."""
    from ..models.chat_message import TaskChatMessage
    from .chat_history_service import (
        DELIVERY_DISPATCHED,
        DELIVERY_OUTCOME_UNKNOWN,
        DELIVERY_PENDING,
    )
    from .task_interaction_close import close_legacy_resume_interaction
    from .task_interaction_schema import interaction_requests_table_exists

    with get_session_local()() as db, db.begin():
        if not lock_task_lease_no_commit(db, lease):
            raise TaskLeaseLostError("Injection owner changed before completion")
        task = db.get(Task, lease.task_id)
        assert task is not None
        if task.pending_injection is None:
            return
        pending = PendingInjection.model_validate(task.pending_injection)
        if (
            pending.run_id != lease.run_id
            or pending.turn_id != turn_id
            or pending.owner_id != task.user_id
            or not pending.auto_resume
            or task.control_state == "pause_requested"
        ):
            raise TaskLeaseLostError(
                "Injection changed or was paused before completion"
            )
        setattr(task, "input", pending.display_message)
        if interaction_requests_table_exists(db):
            close_legacy_resume_interaction(
                db,
                task_id=lease.task_id,
                run_id=pending.run_id,
                interaction_id=pending.interaction_id,
            )
        db.query(TaskChatMessage).filter(
            TaskChatMessage.task_id == lease.task_id,
            TaskChatMessage.turn_id == turn_id,
            TaskChatMessage.role == "user",
            TaskChatMessage.delivery_status.in_(
                (DELIVERY_PENDING, DELIVERY_OUTCOME_UNKNOWN)
            ),
        ).update(
            {TaskChatMessage.delivery_status: DELIVERY_DISPATCHED},
            synchronize_session=False,
        )
        _settle_reply_command(db, task, pending, "accepted")
        setattr(task, "pending_injection", None)


def _settle_reply_command(
    db: Any, task: Task, pending: PendingInjection, outcome: str
) -> None:
    from ..models.task_command import TaskExecutionCommand

    if pending.reply_command_id is None:
        return
    command = (
        db.query(TaskExecutionCommand)
        .filter_by(
            task_id=task.id,
            command_id=pending.reply_command_id,
            kind="resume_input",
            target_run_id=pending.run_id,
        )
        .first()
    )
    if command is not None:
        setattr(
            command,
            "result",
            {
                **(command.result or {}),
                "outcome": outcome,
                "run_id": pending.run_id,
                "state_version": task.state_version,
                "control_state": task.control_state,
            },
        )


def _pending_candidates(
    after_id: int, limit: int
) -> list[tuple[int, int, str, str | None]]:
    from ..models.task import TaskStatus

    with get_session_local()() as db:
        rows = db.execute(
            select(Task.id, Task.user_id, Task.run_id, Task.source)
            .where(
                Task.id > after_id,
                Task.pending_injection.is_not(None),
                Task.pending_injection["auto_resume"].as_boolean().is_(True),
                Task.pending_injection["run_id"].as_string() == Task.run_id,
                Task.pending_injection["owner_id"].as_integer() == Task.user_id,
                Task.status == TaskStatus.PAUSED,
            )
            .order_by(Task.id)
            .limit(limit)
        ).all()
        return [
            (int(tid), int(owner), str(run), source) for tid, owner, run, source in rows
        ]


async def recover_pending_injections(*, after_id: int, batch_size: int) -> int:
    """Schedule a bounded page through the existing coordinator and resume path."""
    import asyncio
    from dataclasses import dataclass

    from ...config import get_shared_task_execution_enabled
    from ...core.execution_scope import ExecutionScopeContext, resolve_execution_scope
    from ..user_isolated_memory import UserContext
    from .agent_service_manager import get_agent_manager
    from .task_command_transport import TaskCommandDeferred, TaskCommandKind
    from .task_coordinator_runtime import execute_coordinated_command
    from .task_execution import background_task_manager, execute_resume_background

    @dataclass(frozen=True)
    class Recovery:
        task_id: int
        kind: TaskCommandKind = TaskCommandKind.RESUME_INPUT

    rows = await run_db_io_cancellation_safe(
        lambda: _pending_candidates(after_id, batch_size)
    )
    for task_id, owner_id, run_id, task_source in rows:

        async def schedule() -> None:
            if not background_task_manager.reserve_resume(task_id):
                return
            child = None
            try:
                scope = await run_db_io_cancellation_safe(
                    lambda: resolve_execution_scope(task_id)
                )
                with UserContext(owner_id), ExecutionScopeContext(scope):
                    agent = await get_agent_manager().get_agent_for_task(
                        task_id, None, task_owner_user_id=owner_id
                    )
                    child = asyncio.create_task(
                        execute_resume_background(
                            task_id=task_id,
                            agent_service=agent,
                            task_owner_user_id=owner_id,
                            expected_run_id=run_id,
                            trusted_task_source=task_source,
                            recover_pending_injection=True,
                        )
                    )
                    background_task_manager.register_reserved_resume(
                        task_id, child, run_id=run_id
                    )
            except BaseException:
                if child is not None:
                    child.cancel()
                    from .db_runtime import drain_async_task_cancellation_safe

                    try:
                        await drain_async_task_cancellation_safe(child)
                    except asyncio.CancelledError:
                        pass
                background_task_manager.release_resume_reservation(task_id)
                raise

        try:
            if get_shared_task_execution_enabled():
                await execute_coordinated_command(Recovery(task_id), schedule)
            else:
                await schedule()
        except TaskCommandDeferred:
            pass  # Another exact owner won. The next sweep rechecks the journal.
        except Exception as exc:
            from .db_runtime import is_database_pool_timeout

            if is_database_pool_timeout(exc):
                raise
            logger.exception(
                "Pending input recovery could not start for task %s", task_id
            )
    return rows[-1][0] if len(rows) == batch_size else 0
