"""Live-control writes are fenced on the acquisition driving the local run.

An expired RUNNING takeover keeps ``run_id`` and mints only a new
``lease_attempt_id``. A process whose lease was taken over while its local
run kept going (a zombie of the earlier attempt) therefore still matches a
run id fence, so PAUSE_REQUESTED and the live-message RESUME_REQUESTED
handoff must also match the exact acquisition. A late result must not
resurrect a row that already settled terminal either.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from sqlalchemy import update

from tests.web.api.test_durable_message_resume_contention import (
    _live_control_environment,
    _live_task,
    _message_command,
    _user,
)
from tests.web.api.test_durable_message_resume_contention import (
    db_session as db_session_fixture,
)
from tests.web.services.task_lease_shared import (
    live_task_lease as live_task_lease_fixture,
)
from xagent.core.agent.runner import UserMessageInjectionOutcome
from xagent.web.models.database import get_session_local
from xagent.web.models.task import Task, TaskStatus
from xagent.web.models.user import User
from xagent.web.services import agent_service_manager
from xagent.web.services import task_command_execution as commands
from xagent.web.services import task_coordinator_runtime
from xagent.web.services import task_execution as execution
from xagent.web.services import task_lease_service as leases
from xagent.web.services import task_setup_snapshot
from xagent.web.services.task_command_execution import (
    ClientVisibleTaskCommandDeferred,
)
from xagent.web.services.task_command_transport import (
    ClaimedTaskCommand,
    TaskCommandDeferred,
    TaskCommandKind,
)
from xagent.web.services.task_coordinator_service import TaskLease as CoordinatorLease
from xagent.web.services.task_execution_controller import (
    StaleTaskRunError,
    TaskControlState,
    transition_task_control_state_sync,
)
from xagent.web.services.task_lease_service import TaskLease, get_runner_id

db_session = db_session_fixture
live_task_lease = live_task_lease_fixture

SUCCESSOR_RUNNER = "successor-runner"
SUCCESSOR_ATTEMPT = "successor-attempt"


def _leased_task(db_session) -> Task:
    owner = _user(db_session, f"fence-owner-{datetime.now().timestamp()}")
    task = _live_task(db_session, int(owner.id))
    task.runner_id = get_runner_id()
    task.lease_expires_at = datetime.now(timezone.utc) + timedelta(minutes=1)
    task.control_state = TaskControlState.RUNNING.value
    task.state_version = 3
    db_session.commit()
    return task


def _take_over(task_id: int) -> None:
    """What ``acquire_task_lease_no_commit`` leaves after an expired takeover."""
    with get_session_local()() as db:
        db.execute(
            update(Task)
            .where(Task.id == task_id)
            .values(
                runner_id=SUCCESSOR_RUNNER,
                lease_attempt_id=SUCCESSOR_ATTEMPT,
                lease_expires_at=datetime.now(timezone.utc) + timedelta(minutes=5),
            )
        )
        db.commit()


def _row(db_session, task_id: int) -> Task:
    db_session.expire_all()
    task = db_session.get(Task, task_id)
    assert task is not None
    return task


def _patch_pause_runtime(monkeypatch, task: Task) -> AsyncMock:
    snapshot_task = SimpleNamespace(
        user_id=int(task.user_id), run_id=str(task.run_id), status=TaskStatus.RUNNING
    )
    monkeypatch.setattr(
        task_setup_snapshot,
        "load_task_setup_snapshot_sync",
        lambda *a, **k: SimpleNamespace(task=snapshot_task, runtime_user=object()),
    )
    monkeypatch.setattr(commands, "resolve_execution_scope_off_turn", lambda *a: None)
    pause_execution = AsyncMock(return_value=True)
    service = SimpleNamespace(pause_execution=pause_execution)
    monkeypatch.setattr(
        agent_service_manager,
        "get_agent_manager",
        lambda: SimpleNamespace(get_agent_for_task=AsyncMock(return_value=service)),
    )
    return pause_execution


def _pause_message(task: Task) -> dict:
    return {"user": SimpleNamespace(id=int(task.user_id), is_admin=False)}


# ---------------------------------------------------------------------------
# PAUSE_REQUESTED
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_current_owner_pause_is_applied(
    live_task_lease, db_session, monkeypatch
) -> None:
    task = _leased_task(db_session)
    live_task_lease(db_session, task)
    pause_execution = _patch_pause_runtime(monkeypatch, task)
    publish = AsyncMock()
    monkeypatch.setattr(commands, "publish_task_event", publish)
    reply = AsyncMock()
    message = _pause_message(task)

    try:
        await commands.pause_task(reply, int(task.id), message)
        assert execution._is_task_pause_accepted(int(task.id))
    finally:
        execution._clear_task_pause_accepted(int(task.id))

    pause_execution.assert_awaited_once_with()
    assert "_durable_command_error" not in message
    assert publish.await_args.args[0]["type"] == "task_pause_requested"
    stored = _row(db_session, int(task.id))
    assert stored.control_state == TaskControlState.PAUSE_REQUESTED.value
    assert stored.state_version == 4


@pytest.mark.asyncio
async def test_zombie_pause_interrupts_locally_and_defers_for_the_successor(
    live_task_lease, db_session, monkeypatch
) -> None:
    task = _leased_task(db_session)
    # This process still heartbeats the earlier attempt; its local run is
    # alive and accepts the interrupt.
    live_task_lease(db_session, task)
    _take_over(int(task.id))
    pause_execution = _patch_pause_runtime(monkeypatch, task)
    publish = AsyncMock()
    monkeypatch.setattr(commands, "publish_task_event", publish)
    reply = AsyncMock()
    message = _pause_message(task)

    with pytest.raises(
        ClientVisibleTaskCommandDeferred, match="waiting for the active task lease"
    ):
        await commands.pause_task(reply, int(task.id), message)

    # The zombie's run is still interrupted: its settlement is attempt-fenced.
    pause_execution.assert_awaited_once_with()
    # The user's intent is retried rather than answered or reported.
    assert not execution._is_task_pause_accepted(int(task.id))
    assert "_durable_command_error" not in message
    reply.assert_not_awaited()
    publish.assert_not_awaited()
    stored = _row(db_session, int(task.id))
    assert stored.control_state == TaskControlState.RUNNING.value
    assert stored.state_version == 3
    assert stored.runner_id == SUCCESSOR_RUNNER
    assert stored.lease_attempt_id == SUCCESSOR_ATTEMPT


@pytest.mark.asyncio
async def test_zombie_pause_defers_through_the_durable_dispatcher(
    live_task_lease, db_session, monkeypatch
) -> None:
    """The deferral is the retryable transport outcome, not a rejection."""
    task = _leased_task(db_session)
    live_task_lease(db_session, task)
    # Same runner, new acquisition: the foreign-runner pre-check cannot see
    # this, only the attempt fence can.
    with get_session_local()() as db:
        db.execute(
            update(Task)
            .where(Task.id == int(task.id))
            .values(lease_attempt_id=SUCCESSOR_ATTEMPT)
        )
        db.commit()
    _patch_pause_runtime(monkeypatch, task)
    monkeypatch.setattr(
        commands, "_load_command_actor", lambda _: SimpleNamespace(id=1, is_admin=False)
    )
    command = ClaimedTaskCommand(
        id=1,
        task_id=int(task.id),
        actor_user_id=int(task.user_id),
        command_id="pause-zombie",
        kind=TaskCommandKind.PAUSE,
        payload={},
        target_run_id=str(task.run_id),
        attempt_count=1,
    )

    with pytest.raises(ClientVisibleTaskCommandDeferred):
        await commands._execute_durable_task_command(command)

    assert _row(db_session, int(task.id)).control_state == "running"


def test_pause_fence_refuses_an_expired_own_lease(db_session) -> None:
    task = _leased_task(db_session)
    task.lease_attempt_id = "own-attempt"
    task.lease_expires_at = datetime.now(timezone.utc) - timedelta(seconds=1)
    db_session.commit()
    own = TaskLease(
        task_id=int(task.id),
        runner_id=str(task.runner_id),
        run_id=str(task.run_id),
        attempt_id="own-attempt",
    )

    with pytest.raises(ClientVisibleTaskCommandDeferred):
        commands._apply_pause_requested_isolated(
            int(task.id), expected_run_id=str(task.run_id), owner_leases=(own,)
        )

    assert _row(db_session, int(task.id)).control_state == "running"


def test_pause_fence_without_a_local_holder_defers(db_session) -> None:
    task = _leased_task(db_session)
    task.lease_attempt_id = "someone-else"
    db_session.commit()

    with pytest.raises(ClientVisibleTaskCommandDeferred):
        commands._apply_pause_requested_isolated(
            int(task.id), expected_run_id=str(task.run_id), owner_leases=()
        )

    assert _row(db_session, int(task.id)).state_version == 3


def test_pause_fence_still_reports_a_settled_run_as_finished(db_session) -> None:
    task = _leased_task(db_session)
    task.lease_attempt_id = "own-attempt"
    task.status = TaskStatus.PAUSED
    task.control_state = TaskControlState.PAUSED.value
    db_session.commit()

    applied = commands._apply_pause_requested_isolated(
        int(task.id),
        expected_run_id=str(task.run_id),
        owner_leases=(
            TaskLease(int(task.id), str(task.runner_id), str(task.run_id), "x"),
        ),
    )

    assert applied is False


class _FakeCoordinator:
    def __init__(self, lease: CoordinatorLease) -> None:
        self.lease = lease
        self.task_id = lease.task_id


@pytest.mark.asyncio
@pytest.mark.parametrize("taken_over", [False, True])
async def test_shared_mode_pause_is_fenced_on_the_coordinator_lease(
    db_session, monkeypatch, taken_over: bool
) -> None:
    task = _leased_task(db_session)
    task.lease_attempt_id = "coordinator-attempt"
    db_session.commit()
    coordinator = _FakeCoordinator(
        CoordinatorLease(int(task.id), get_runner_id(), "coordinator-attempt")
    )
    monkeypatch.setattr(
        task_coordinator_runtime,
        "current_task_coordinator",
        lambda task_id: coordinator if task_id == coordinator.task_id else None,
    )
    if taken_over:
        _take_over(int(task.id))
    _patch_pause_runtime(monkeypatch, task)
    monkeypatch.setattr(commands, "publish_task_event", AsyncMock())

    try:
        if taken_over:
            with pytest.raises(ClientVisibleTaskCommandDeferred):
                await commands.pause_task(
                    AsyncMock(), int(task.id), _pause_message(task)
                )
        else:
            await commands.pause_task(AsyncMock(), int(task.id), _pause_message(task))
    finally:
        execution._clear_task_pause_accepted(int(task.id))

    assert _row(db_session, int(task.id)).control_state == (
        TaskControlState.RUNNING.value
        if taken_over
        else TaskControlState.PAUSE_REQUESTED.value
    )


@pytest.mark.asyncio
async def test_local_holders_follow_registration_and_run(
    live_task_lease, db_session
) -> None:
    task = _leased_task(db_session)
    lease = live_task_lease(db_session, task)

    assert leases.local_task_lease_holders(int(task.id), str(task.run_id)) == (lease,)
    assert leases.local_task_lease_holders(int(task.id), "other-run") == ()
    assert leases.local_task_lease_holders(int(task.id) + 1, str(task.run_id)) == ()


# ---------------------------------------------------------------------------
# RESUME_REQUESTED (live message into a running run)
# ---------------------------------------------------------------------------


def test_owner_lease_transition_matches_only_the_exact_acquisition(
    db_session,
) -> None:
    task = _leased_task(db_session)
    task.lease_attempt_id = "own-attempt"
    db_session.commit()
    own = TaskLease(int(task.id), str(task.runner_id), str(task.run_id), "own-attempt")

    snapshot = transition_task_control_state_sync(
        int(task.id),
        TaskControlState.RESUME_REQUESTED,
        expected_run_id=str(task.run_id),
        owner_lease=own,
    )
    assert snapshot.control_state is TaskControlState.RESUME_REQUESTED

    _take_over(int(task.id))
    with pytest.raises(StaleTaskRunError, match="no longer owned"):
        transition_task_control_state_sync(
            int(task.id),
            TaskControlState.RUNNING,
            expected_run_id=str(task.run_id),
            owner_lease=own,
        )
    stored = _row(db_session, int(task.id))
    assert stored.control_state == TaskControlState.RESUME_REQUESTED.value
    assert stored.state_version == snapshot.state_version


@pytest.mark.asyncio
async def test_live_message_handoff_stays_off_a_successor_run(
    live_task_lease, db_session
) -> None:
    task = _leased_task(db_session)
    live_task_lease(db_session, task)
    owner = db_session.get(User, int(task.user_id))
    background_manager = execution.BackgroundTaskManager()
    with _live_control_environment(background_manager=background_manager) as (
        agent,
        _,
    ):

        async def inject_then_lose_lease(*args, **kwargs):
            # The takeover lands after routing resolved this process's lease
            # and after the injection, right before the handoff write.
            _take_over(int(task.id))
            return UserMessageInjectionOutcome.POSTED_FRESH

        agent.post_user_message = AsyncMock(side_effect=inject_then_lose_lease)

        # The refused handoff takes the same path as a rotated run: the
        # accepted injection keeps the delivery unresolved, so the command is
        # retried through recovery rather than reported delivered.
        with pytest.raises(TaskCommandDeferred, match="waiting for runtime injection"):
            await commands.execute_durable_task_command(
                _message_command(task, owner, "fenced-handoff")
            )
        agent.post_user_message.assert_awaited_once()
        # The failed handoff hands the reservation back.
        assert background_manager.resume_holder_age_seconds(int(task.id)) is None

    stored = _row(db_session, int(task.id))
    assert stored.control_state == TaskControlState.RUNNING.value
    assert stored.runner_id == SUCCESSOR_RUNNER
    assert stored.lease_attempt_id == SUCCESSOR_ATTEMPT


# ---------------------------------------------------------------------------
# Late results on a row that already settled terminal
# ---------------------------------------------------------------------------


def _owned_failed_task(db_session) -> tuple[Task, TaskLease]:
    """FAILED while the lease fence still matches, as in coordinator context."""
    task = _leased_task(db_session)
    task.lease_attempt_id = "own-attempt"
    task.status = TaskStatus.FAILED
    task.control_state = TaskControlState.FAILED.value
    task.error_message = "cancelled externally"
    db_session.commit()
    return task, TaskLease(
        int(task.id), str(task.runner_id), str(task.run_id), "own-attempt"
    )


_LATE_RESULTS = {
    "interrupted": {"status": "interrupted", "success": False, "output": "stopped"},
    "success": {"status": "completed", "success": True, "output": "late answer"},
}


@pytest.mark.parametrize("kind", sorted(_LATE_RESULTS))
def test_late_result_keeps_a_failed_row_failed(db_session, kind: str) -> None:
    task, lease = _owned_failed_task(db_session)

    finalized = execution._finalize_task_execution_result_isolated(
        task_id=int(task.id),
        task_user_id=int(task.user_id),
        pre_run_status=TaskStatus.RUNNING,
        result=dict(_LATE_RESULTS[kind]),
        expected_run_id=lease.run_id,
        task_lease=lease,
        resolved_scope_segments=(),
        prepared_outputs=execution._PreparedTaskFileOutputs((), (), ()),
    )

    assert finalized.final_task_status == TaskStatus.FAILED.value
    assert finalized.final_control_snapshot is None
    stored = _row(db_session, int(task.id))
    assert stored.status == TaskStatus.FAILED
    assert stored.control_state == TaskControlState.FAILED.value
    assert stored.error_message == "cancelled externally"
    assert stored.state_version == 3


@pytest.mark.parametrize("kind", sorted(_LATE_RESULTS))
def test_late_resumed_result_keeps_a_failed_row_failed(db_session, kind: str) -> None:
    task, lease = _owned_failed_task(db_session)
    result = dict(_LATE_RESULTS[kind])

    finalized = execution._finalize_resumed_task(
        int(task.id),
        status=result["status"],
        success=result["success"],
        output=result["output"],
        task_owner_user_id=int(task.user_id),
        result=result,
        task_lease=lease,
        prepared_outputs=execution._PreparedTaskFileOutputs((), (), ()),
    )

    assert finalized["late_result"] is True
    stored = _row(db_session, int(task.id))
    assert stored.status == TaskStatus.FAILED
    assert stored.control_state == TaskControlState.FAILED.value
    assert stored.error_message == "cancelled externally"
