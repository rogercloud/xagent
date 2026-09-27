"""A recovered delivery never resumes a run that recovery settled FAILED.

A durable MESSAGE command claimed its delivery row on the task's own run and
crashed before settling it. Lease recovery then found the run not recoverable
and settled the task FAILED. The retry finds the pending row as a recovered
claim on the command's own run, which is normally redriven through a resume
so the run replays the turn id against its checkpoint. A FAILED run must not
be resumed: the retry settles the command as accepted with an unknown outcome,
advances the row out of ``pending`` without running the turn, and leaves the
task FAILED with its status, control state, run and diagnostic untouched.

The routing snapshot can be stale, so the refusal is also enforced by the
RESUME_REQUESTED transition and by the resume lease claim, each as part of
its own conditional UPDATE.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from typing import Any, Iterator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tests.web.api.test_durable_message_resume_contention import (
    _live_control_environment,
    _live_task,
    _message_command,
    _user,
)
from tests.web.api.test_recovered_delivery_outcome_unknown import (
    TURN_ID,
    _assert_outcome_unknown_frames,
    _enqueue_settled_unknown_command,
    _outcome_unknown_result,
    _pending_row,
    _RecordingReply,
    _row_status,
    _user_rows,
)
from xagent.web.api import websocket as websocket_api
from xagent.web.models.database import (
    Base,
    get_db,
    get_engine,
    get_session_local,
    init_db,
)
from xagent.web.models.task import Task, TaskStatus
from xagent.web.services import task_command_execution as command_execution_service
from xagent.web.services import task_execution as task_execution_service
from xagent.web.services import task_execution_controller as controller_module
from xagent.web.services.chat_history_service import (
    DELIVERY_DISPATCHED,
    DELIVERY_OUTCOME_UNKNOWN,
)
from xagent.web.services.client_error_messages import ClientErrorCode
from xagent.web.services.task_command_execution import execute_durable_task_command
from xagent.web.services.task_command_transport import (
    TaskCommandDeferred,
    TaskCommandKind,
)
from xagent.web.services.task_execution import ResumeReservationOutcome
from xagent.web.services.task_execution_controller import (
    StaleTaskRunError,
    TaskControlState,
    TaskStatusRefusedError,
    apply_task_control_transition,
)
from xagent.web.services.task_lease_service import (
    get_expired_task_lease_candidates,
    recover_expired_task_lease_no_commit,
)
from xagent.web.services.task_orchestrator import TaskTurnOrchestrator, TurnKind

RECOVERY_ERROR = "not recoverable: checkpoint missing"


@pytest.fixture()
def db_session(tmp_path):
    init_db(db_url=f"sqlite:///{tmp_path / 'recovered_failed_run.db'}")
    db = next(get_db())
    try:
        yield db
    finally:
        db.close()
        Base.metadata.drop_all(bind=get_engine())


@pytest.fixture()
def recording_reply() -> Iterator[_RecordingReply]:
    reply = _RecordingReply()
    with patch.object(command_execution_service, "command_reply", return_value=reply):
        yield reply


def _expired_running_task(db, owner_id: int) -> Task:
    """A RUNNING task on the command's own run whose lease has expired."""

    task = _live_task(db, owner_id)
    task.runner_id = "dead-runner"
    task.lease_attempt_id = "dead-attempt"
    task.lease_expires_at = datetime.now(timezone.utc) - timedelta(minutes=5)
    task.control_state = TaskControlState.RUNNING.value
    db.commit()
    db.refresh(task)
    return task


def _recover_to_failed(task_id: int) -> None:
    """Run lease recovery's NOT_RECOVERABLE settlement in its own session."""

    with get_session_local()() as db:
        now = datetime.now(timezone.utc)
        candidate = next(
            candidate
            for candidate in get_expired_task_lease_candidates(db, cutoff=now, limit=10)
            if candidate.task_id == task_id
        )
        assert recover_expired_task_lease_no_commit(
            db,
            candidate,
            status=TaskStatus.FAILED,
            recovered_at=now,
            error_message=RECOVERY_ERROR,
        )
        db.commit()


def _failed_task(db, owner_id: int) -> Task:
    task = _expired_running_task(db, owner_id)
    _recover_to_failed(int(task.id))
    db.expire_all()
    task = db.get(Task, int(task.id))
    assert task.status == TaskStatus.FAILED
    return task


def _assert_still_failed(db, task_id: int, *, state_version: int) -> None:
    db.expire_all()
    stored = db.get(Task, task_id)
    assert stored.status == TaskStatus.FAILED
    assert stored.control_state == TaskControlState.FAILED.value
    assert stored.error_message == RECOVERY_ERROR
    assert stored.run_id == "live-run"
    assert stored.runner_id is None
    assert int(stored.state_version or 0) == state_version


@pytest.mark.asyncio
async def test_recovered_delivery_on_failed_run_settles_outcome_unknown(
    db_session,
    recording_reply: _RecordingReply,
) -> None:
    owner = _user(db_session, "failed-run-owner")
    task = _failed_task(db_session, int(owner.id))
    task_id = int(task.id)
    state_version = int(task.state_version or 0)
    _pending_row(db_session, task, int(owner.id))

    publish = AsyncMock()
    with (
        _live_control_environment(outcome=ResumeReservationOutcome.RESERVED) as (
            agent,
            background_manager,
        ),
        patch.object(command_execution_service, "publish_task_event", publish),
    ):
        result = await execute_durable_task_command(
            _message_command(task, owner, TURN_ID, attempt_count=2)
        )
        task_execution_service.execute_resume_background.assert_not_called()

    assert result == _outcome_unknown_result(task)
    agent.post_user_message.assert_not_awaited()
    background_manager.try_reserve_resume.assert_not_called()
    assert _row_status(db_session, task_id) == DELIVERY_DISPATCHED
    assert len(_user_rows(db_session, task_id)) == 1
    _assert_still_failed(db_session, task_id, state_version=state_version)
    _assert_outcome_unknown_frames(recording_reply)
    publish.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("fence", ["pre_check", "conditional_update"])
async def test_task_failed_after_routing_snapshot_is_caught_by_the_transition(
    db_session,
    recording_reply: _RecordingReply,
    fence: str,
) -> None:
    """Recovery commits FAILED after the handler routed on a RUNNING row.

    ``pre_check`` lands it before the transition loads the row;
    ``conditional_update`` lands it after the load, so only the UPDATE's own
    status predicate can refuse it.
    """

    owner = _user(db_session, f"transition-race-{fence}")
    task = _expired_running_task(db_session, int(owner.id))
    task_id = int(task.id)
    _pending_row(db_session, task, int(owner.id))

    if fence == "pre_check":
        real_sync = controller_module.transition_task_control_state_sync

        def recover_then_transition(*args: Any, **kwargs: Any):
            _recover_to_failed(task_id)
            return real_sync(*args, **kwargs)

        race = patch.object(
            controller_module,
            "transition_task_control_state_sync",
            side_effect=recover_then_transition,
        )
    else:
        real_apply = controller_module.apply_task_control_transition

        def recover_then_apply(task_row: Task, *args: Any, **kwargs: Any):
            assert task_row.status == TaskStatus.RUNNING
            _recover_to_failed(task_id)
            return real_apply(task_row, *args, **kwargs)

        race = patch.object(
            controller_module,
            "apply_task_control_transition",
            side_effect=recover_then_apply,
        )

    with (
        _live_control_environment(outcome=ResumeReservationOutcome.RESERVED) as (
            agent,
            background_manager,
        ),
        race as raced,
    ):
        result = await execute_durable_task_command(
            _message_command(task, owner, TURN_ID, attempt_count=2)
        )
        task_execution_service.execute_resume_background.assert_not_called()

    assert raced.call_count == 1
    assert raced.call_args.kwargs["refused_statuses"] == (TaskStatus.FAILED,)
    assert result == _outcome_unknown_result(task)
    background_manager.release_resume_reservation.assert_called_with(task_id)
    background_manager.register_reserved_resume.assert_not_called()
    assert _row_status(db_session, task_id) == DELIVERY_DISPATCHED
    db_session.expire_all()
    recovered_version = int(db_session.get(Task, task_id).state_version or 0)
    _assert_still_failed(db_session, task_id, state_version=recovered_version)
    _assert_outcome_unknown_frames(recording_reply)
    agent.post_user_message.assert_not_awaited()


@pytest.mark.asyncio
async def test_task_failed_after_the_transition_is_caught_by_the_lease_claim(
    db_session,
    recording_reply: _RecordingReply,
) -> None:
    """Recovery commits FAILED after RESUME_REQUESTED, before the claim."""

    owner = _user(db_session, "claim-race-owner")
    task = _expired_running_task(db_session, int(owner.id))
    task_id = int(task.id)
    _pending_row(db_session, task, int(owner.id))
    real_sync = controller_module.transition_task_control_state_sync

    def transition_then_recover(*args: Any, **kwargs: Any):
        snapshot = real_sync(*args, **kwargs)
        _recover_to_failed(task_id)
        return snapshot

    agent = MagicMock()
    agent.supports_live_control.return_value = True
    agent.post_user_message = AsyncMock()
    background_manager = task_execution_service.BackgroundTaskManager()
    resume_spy = AsyncMock(wraps=task_execution_service.execute_resume_background)
    with (
        patch(
            "xagent.web.services.agent_service_manager.get_agent_manager",
            return_value=MagicMock(get_agent_for_task=AsyncMock(return_value=agent)),
        ),
        patch.object(
            task_execution_service, "background_task_manager", background_manager
        ),
        patch.object(task_execution_service, "execute_resume_background", resume_spy),
        patch.object(task_execution_service, "publish_task_event", AsyncMock()),
        patch.object(
            controller_module,
            "transition_task_control_state_sync",
            side_effect=transition_then_recover,
        ),
    ):
        try:
            first = await execute_durable_task_command(
                _message_command(task, owner, TURN_ID, attempt_count=2)
            )
        except TaskCommandDeferred:
            # The handoff returned before the resume claimed; the row was
            # still pending when the command read it.
            first = None
        assert resume_spy.await_count == 1
        assert resume_spy.await_args.kwargs["refuse_failed_status"] is True
        for _ in range(200):
            if task_id not in background_manager.running_tasks:
                break
            await asyncio.sleep(0.01)
        assert task_id not in background_manager.running_tasks

        assert _row_status(db_session, task_id) == DELIVERY_OUTCOME_UNKNOWN
        recording_reply.frames.clear()
        result = (
            first
            if first is not None
            else await execute_durable_task_command(
                _message_command(task, owner, TURN_ID, attempt_count=3)
            )
        )

    assert result == _outcome_unknown_result(task)
    agent.post_user_message.assert_not_awaited()
    assert _row_status(db_session, task_id) == DELIVERY_OUTCOME_UNKNOWN
    db_session.expire_all()
    recovered_version = int(db_session.get(Task, task_id).state_version or 0)
    _assert_still_failed(db_session, task_id, state_version=recovered_version)
    if first is None:
        _assert_outcome_unknown_frames(recording_reply)


@pytest.mark.asyncio
async def test_fresh_message_on_failed_task_still_appends_a_new_run(
    db_session,
) -> None:
    owner = _user(db_session, "fresh-append-owner")
    task = _failed_task(db_session, int(owner.id))
    task_id = int(task.id)

    begin_turn = AsyncMock(wraps=TaskTurnOrchestrator.begin_turn)
    with (
        _live_control_environment(outcome=ResumeReservationOutcome.RESERVED) as (
            agent,
            _,
        ),
        patch.object(TaskTurnOrchestrator, "begin_turn", begin_turn),
    ):
        result = await execute_durable_task_command(
            _message_command(task, owner, "fresh-turn", attempt_count=1)
        )
        task_execution_service.execute_resume_background.assert_not_called()

    assert result == {
        "task_id": task_id,
        "command_id": "fresh-turn",
        "kind": TaskCommandKind.MESSAGE.value,
    }
    begin_turn.assert_awaited_once()
    assert begin_turn.await_args.kwargs["kind"] == TurnKind.APPEND
    agent.post_user_message.assert_not_awaited()
    db_session.expire_all()
    stored = db_session.get(Task, task_id)
    assert stored.status == TaskStatus.RUNNING
    assert stored.run_id not in {None, "live-run"}


@pytest.mark.asyncio
async def test_same_id_resend_after_failed_run_refusal_reports_outcome_unknown(
    db_session,
    recording_reply: _RecordingReply,
) -> None:
    owner = _user(db_session, "failed-resend-owner")
    task = _failed_task(db_session, int(owner.id))
    task_id = int(task.id)
    _pending_row(db_session, task, int(owner.id))
    command = _message_command(task, owner, TURN_ID, attempt_count=2)

    with _live_control_environment(outcome=ResumeReservationOutcome.RESERVED):
        result = await execute_durable_task_command(command)
    assert result == _outcome_unknown_result(task)
    _enqueue_settled_unknown_command(
        db_session, task, owner, dict(command.payload), result=result
    )

    enqueued = websocket_api._enqueue_websocket_task_command_sync(
        task_id=task_id,
        actor_user_id=int(owner.id),
        actor_is_admin=False,
        command_id=TURN_ID,
        kind=TaskCommandKind.MESSAGE,
        payload=dict(command.payload),
        allow_missing_task=True,
    )

    assert enqueued is not None
    assert enqueued.created is False
    assert enqueued.status == DELIVERY_OUTCOME_UNKNOWN
    assert enqueued.payload_matches is True
    assert result["kind"] == TaskCommandKind.MESSAGE.value
    assert ClientErrorCode.MESSAGE_OUTCOME_UNKNOWN.value in {
        frame.get("error_code") for frame in recording_reply.frames
    }


def test_refused_status_transition_leaves_the_row_untouched(db_session) -> None:
    owner = _user(db_session, "transition-unit-owner")
    task = _failed_task(db_session, int(owner.id))
    task_id = int(task.id)
    state_version = int(task.state_version or 0)

    with pytest.raises(TaskStatusRefusedError) as exc_info:
        apply_task_control_transition(
            task,
            TaskControlState.RESUME_REQUESTED,
            expected_run_id="live-run",
            refused_statuses=(TaskStatus.FAILED,),
        )
    assert exc_info.value.status == TaskStatus.FAILED
    db_session.rollback()
    _assert_still_failed(db_session, task_id, state_version=state_version)

    # A stale run is still reported as a stale run, not as a status refusal.
    task = db_session.get(Task, task_id)
    task.status = TaskStatus.PAUSED
    task.control_state = TaskControlState.PAUSED.value
    db_session.commit()
    with pytest.raises(StaleTaskRunError) as stale_info:
        apply_task_control_transition(
            task,
            TaskControlState.RESUME_REQUESTED,
            expected_run_id="another-run",
            refused_statuses=(TaskStatus.FAILED,),
        )
    assert not isinstance(stale_info.value, TaskStatusRefusedError)


@pytest.mark.parametrize("refuse", [False, True])
def test_resume_lease_claim_status_fence_is_opt_in(db_session, refuse: bool) -> None:
    owner = _user(db_session, f"claim-unit-owner-{refuse}")
    task = _failed_task(db_session, int(owner.id))
    task_id = int(task.id)
    state_version = int(task.state_version or 0)
    refused: list[bool] = []

    lease = task_execution_service._acquire_resume_task_lease(
        task_id,
        int(owner.id),
        "live-run",
        refuse_failed_status=refuse,
        status_refused_out=refused,
    )

    if refuse:
        assert lease is None
        assert refused == [True]
        _assert_still_failed(db_session, task_id, state_version=state_version)
    else:
        # Every other resume caller keeps the claim it had.
        assert lease is not None
        assert refused == []
        db_session.expire_all()
        assert db_session.get(Task, task_id).status == TaskStatus.RUNNING


def test_resume_lease_claim_refused_by_a_live_owner_is_not_a_status_refusal(
    db_session,
) -> None:
    owner = _user(db_session, "claim-live-owner")
    task = _live_task(db_session, int(owner.id))
    task.lease_expires_at = datetime.now(timezone.utc) + timedelta(minutes=5)
    db_session.commit()
    refused: list[bool] = []

    lease = task_execution_service._acquire_resume_task_lease(
        int(task.id),
        int(owner.id),
        "live-run",
        refuse_failed_status=True,
        status_refused_out=refused,
    )

    assert lease is None
    assert refused == []
