"""Durable RESUME admission and idempotency regressions for #1499."""

from __future__ import annotations

import asyncio
from contextlib import ExitStack
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from xagent.web.api import websocket as websocket_api
from xagent.web.api.websocket import (
    BackgroundTaskManager,
    ResumeCommandOutcome,
    ResumeReservationOutcome,
    _execute_durable_task_command,
)
from xagent.web.models.database import Base, get_db, get_engine, init_db
from xagent.web.models.task import Task, TaskStatus
from xagent.web.models.task_command import TaskExecutionCommand
from xagent.web.models.user import User
from xagent.web.services.task_command_transport import (
    COMMAND_PENDING,
    ClaimedTaskCommand,
    TaskCommandDeferred,
    TaskCommandKind,
    dispatch_one_task_command,
    enqueue_task_command,
)
from xagent.web.services.task_execution_controller import TaskControlState


@pytest.fixture()
def db_session(tmp_path):
    init_db(db_url=f"sqlite:///{tmp_path / 'durable_resume_contention.db'}")
    db = next(get_db())
    try:
        yield db
    finally:
        db.close()
        Base.metadata.drop_all(bind=get_engine())


def _user(db, username: str) -> User:
    user = User(username=username, password_hash="x", is_admin=False)
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


def _task(
    db,
    owner_id: int,
    *,
    status: TaskStatus = TaskStatus.PAUSED,
    control_state: TaskControlState = TaskControlState.PAUSED,
) -> Task:
    task = Task(
        user_id=owner_id,
        title="resume contention",
        description="resume contention",
        status=status,
        control_state=control_state.value,
        run_id="run-a",
        execution_mode="balanced",
        source="sdk",
    )
    db.add(task)
    db.commit()
    db.refresh(task)
    return task


def _command(task: Task, owner: User, command_id: str) -> ClaimedTaskCommand:
    return ClaimedTaskCommand(
        id=1,
        task_id=int(task.id),
        actor_user_id=int(owner.id),
        command_id=command_id,
        kind=TaskCommandKind.RESUME,
        payload={"type": "resume_task"},
        target_run_id="run-a",
        attempt_count=1,
    )


def _resume_runtime_patches(
    *, outcome: ResumeReservationOutcome
) -> tuple[ExitStack, MagicMock, MagicMock, MagicMock]:
    stack = ExitStack()
    agent = MagicMock()
    agent.supports_live_control.return_value = True
    agent_manager = MagicMock()
    agent_manager.get_agent_for_task = AsyncMock(return_value=agent)
    connection_manager = MagicMock()
    connection_manager.connections_for_task.return_value = []
    connection_manager.send_personal_message = AsyncMock()
    connection_manager.broadcast_to_task = AsyncMock()
    background_manager = MagicMock()
    background_manager.try_reserve_resume.return_value = outcome
    background_manager.resume_admission_state.return_value = (
        None if outcome is ResumeReservationOutcome.RESERVED else outcome
    )
    background_manager.running_tasks = {}

    stack.enter_context(
        patch("xagent.web.api.chat.get_agent_manager", return_value=agent_manager)
    )
    stack.enter_context(patch.object(websocket_api, "manager", connection_manager))
    stack.enter_context(
        patch.object(websocket_api, "background_task_manager", background_manager)
    )
    stack.enter_context(
        patch.object(
            websocket_api, "resolve_execution_scope_off_turn", return_value=None
        )
    )
    return stack, background_manager, connection_manager, agent_manager


@pytest.mark.asyncio
async def test_resume_reservation_classifies_all_admission_outcomes() -> None:
    manager = BackgroundTaskManager()

    assert manager.try_reserve_resume(1) is ResumeReservationOutcome.RESERVED
    assert manager.try_reserve_resume(1) is ResumeReservationOutcome.RESERVATION_HELD

    coordinator_gate = asyncio.get_running_loop().create_future()
    coordinator = asyncio.ensure_future(coordinator_gate)
    try:
        manager.register_reserved_resume(1, coordinator, run_id="run-a")
        assert (
            manager.try_reserve_resume(1, expected_run_id="run-a")
            is ResumeReservationOutcome.COORDINATOR_RUNNING
        )
        assert (
            manager.try_reserve_resume(1, expected_run_id="run-b")
            is ResumeReservationOutcome.RESERVATION_HELD
        )

        manager._shutting_down = True
        assert manager.try_reserve_resume(2) is ResumeReservationOutcome.SHUTTING_DOWN
    finally:
        coordinator_gate.cancel()
        await asyncio.gather(coordinator, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "outcome",
    [
        ResumeReservationOutcome.RESERVATION_HELD,
        ResumeReservationOutcome.SHUTTING_DOWN,
    ],
)
async def test_uncertain_resume_admission_defers_durable_command(
    db_session, outcome: ResumeReservationOutcome
) -> None:
    owner = _user(db_session, f"owner-{outcome.value}")
    task = _task(
        db_session,
        int(owner.id),
        control_state=(
            TaskControlState.RESUME_REQUESTED
            if outcome is ResumeReservationOutcome.RESERVATION_HELD
            else TaskControlState.PAUSED
        ),
    )
    stack, _background_manager, connection_manager, _agent_manager = (
        _resume_runtime_patches(outcome=outcome)
    )

    with stack, pytest.raises(TaskCommandDeferred, match="resume slot"):
        await _execute_durable_task_command(
            _command(task, owner, f"resume-{outcome.value}")
        )

    connection_manager.send_personal_message.assert_not_awaited()


@pytest.mark.asyncio
async def test_registered_coordinator_is_an_explicit_idempotent_success(
    db_session,
) -> None:
    owner = _user(db_session, "coordinator-owner")
    task = _task(db_session, int(owner.id))
    stack, background_manager, _connection_manager, _agent_manager = (
        _resume_runtime_patches(outcome=ResumeReservationOutcome.COORDINATOR_RUNNING)
    )

    with stack:
        result = await _execute_durable_task_command(
            _command(task, owner, "resume-already-running")
        )

    assert result is not None
    assert result["resume_outcome"] == ResumeCommandOutcome.ALREADY_IN_PROGRESS.value
    _agent_manager.get_agent_for_task.assert_not_awaited()
    background_manager.try_reserve_resume.assert_not_called()
    background_manager.register_reserved_resume.assert_not_called()


@pytest.mark.asyncio
async def test_running_control_state_is_an_explicit_idempotent_success(
    db_session,
) -> None:
    owner = _user(db_session, "running-owner")
    task = _task(
        db_session,
        int(owner.id),
        status=TaskStatus.RUNNING,
        control_state=TaskControlState.RUNNING,
    )
    task.runner_id = "runner-a"
    task.lease_expires_at = datetime.now(timezone.utc) + timedelta(minutes=1)
    db_session.commit()
    stack, background_manager, _connection_manager, _agent_manager = (
        _resume_runtime_patches(outcome=ResumeReservationOutcome.RESERVED)
    )

    with stack:
        result = await _execute_durable_task_command(
            _command(task, owner, "resume-running")
        )

    assert result is not None
    assert result["resume_outcome"] == ResumeCommandOutcome.ALREADY_IN_PROGRESS.value
    background_manager.try_reserve_resume.assert_not_called()


@pytest.mark.asyncio
async def test_running_without_a_live_lease_defers_for_recovery(db_session) -> None:
    owner = _user(db_session, "abandoned-running-owner")
    task = _task(
        db_session,
        int(owner.id),
        status=TaskStatus.RUNNING,
        control_state=TaskControlState.RUNNING,
    )
    stack, background_manager, _connection_manager, _agent_manager = (
        _resume_runtime_patches(outcome=ResumeReservationOutcome.RESERVED)
    )

    with stack, pytest.raises(TaskCommandDeferred, match="lease recovery"):
        await _execute_durable_task_command(
            _command(task, owner, "resume-abandoned-running")
        )

    background_manager.try_reserve_resume.assert_not_called()


@pytest.mark.asyncio
async def test_pending_pause_defers_resume_until_control_state_settles(
    db_session,
) -> None:
    owner = _user(db_session, "pause-requested-owner")
    task = _task(
        db_session,
        int(owner.id),
        status=TaskStatus.RUNNING,
        control_state=TaskControlState.PAUSE_REQUESTED,
    )
    stack, background_manager, _connection_manager, _agent_manager = (
        _resume_runtime_patches(outcome=ResumeReservationOutcome.RESERVED)
    )

    with stack, pytest.raises(TaskCommandDeferred, match="pending pause"):
        await _execute_durable_task_command(
            _command(task, owner, "resume-during-pause")
        )

    background_manager.try_reserve_resume.assert_not_called()


@pytest.mark.asyncio
async def test_contended_resume_dispatch_stays_pending_instead_of_completed(
    db_session,
) -> None:
    owner = _user(db_session, "dispatcher-owner")
    task = _task(db_session, int(owner.id))
    enqueued = enqueue_task_command(
        db_session,
        task_id=int(task.id),
        actor_user_id=int(owner.id),
        command_id="resume-dispatch-contention",
        kind=TaskCommandKind.RESUME,
        payload={"type": "resume_task"},
    )
    stack, _background_manager, _connection_manager, _agent_manager = (
        _resume_runtime_patches(outcome=ResumeReservationOutcome.RESERVATION_HELD)
    )

    with stack:
        assert await dispatch_one_task_command(
            _execute_durable_task_command,
            command_db_id=enqueued.command_id,
        )

    db_session.expire_all()
    stored = db_session.get(TaskExecutionCommand, enqueued.command_id)
    assert stored is not None
    assert stored.status == COMMAND_PENDING
    assert stored.defer_count == 1
    assert stored.result is None


@pytest.mark.asyncio
async def test_reserved_resume_records_scheduled_result(db_session) -> None:
    owner = _user(db_session, "scheduled-owner")
    task = _task(
        db_session,
        int(owner.id),
        control_state=TaskControlState.RESUME_REQUESTED,
    )
    stack, background_manager, _connection_manager, _agent_manager = (
        _resume_runtime_patches(outcome=ResumeReservationOutcome.RESERVED)
    )
    transition = AsyncMock(
        return_value=SimpleNamespace(run_id="run-a", status=TaskStatus.PAUSED)
    )
    resume_started = asyncio.Event()

    async def execute_resume_background(**_kwargs) -> None:
        resume_started.set()

    stack.enter_context(
        patch.object(
            websocket_api.task_execution_controller,
            "transition",
            new=transition,
        )
    )
    stack.enter_context(
        patch.object(
            websocket_api,
            "execute_resume_background",
            side_effect=execute_resume_background,
        )
    )

    with stack:
        result = await _execute_durable_task_command(
            _command(task, owner, "resume-scheduled")
        )
        await asyncio.wait_for(resume_started.wait(), timeout=1)

    assert result is not None
    assert result["resume_outcome"] == ResumeCommandOutcome.SCHEDULED.value
    background_manager.register_reserved_resume.assert_called_once()


@pytest.mark.asyncio
async def test_held_reservation_can_schedule_after_the_holder_releases(
    db_session,
) -> None:
    owner = _user(db_session, "released-holder-owner")
    task = _task(db_session, int(owner.id))
    stack, background_manager, _connection_manager, _agent_manager = (
        _resume_runtime_patches(outcome=ResumeReservationOutcome.RESERVATION_HELD)
    )
    background_manager.resume_admission_state.side_effect = [
        ResumeReservationOutcome.RESERVATION_HELD,
        None,
    ]
    background_manager.try_reserve_resume.return_value = (
        ResumeReservationOutcome.RESERVED
    )
    transition = AsyncMock(
        return_value=SimpleNamespace(run_id="run-a", status=TaskStatus.PAUSED)
    )
    resume_started = asyncio.Event()

    async def execute_resume_background(**_kwargs) -> None:
        resume_started.set()

    stack.enter_context(
        patch.object(
            websocket_api.task_execution_controller,
            "transition",
            new=transition,
        )
    )
    stack.enter_context(
        patch.object(
            websocket_api,
            "execute_resume_background",
            side_effect=execute_resume_background,
        )
    )

    with stack:
        with pytest.raises(TaskCommandDeferred):
            await _execute_durable_task_command(
                _command(task, owner, "resume-held-first-attempt")
            )
        result = await _execute_durable_task_command(
            _command(task, owner, "resume-held-second-attempt")
        )
        await asyncio.wait_for(resume_started.wait(), timeout=1)

    assert result is not None
    assert result["resume_outcome"] == ResumeCommandOutcome.SCHEDULED.value
    transition.assert_awaited_once()
    background_manager.register_reserved_resume.assert_called_once()
