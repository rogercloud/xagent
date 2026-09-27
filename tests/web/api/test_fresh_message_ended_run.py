"""A fresh message whose run ends before its handoff opens a new turn.

A fresh (not retried) MESSAGE routed on a RUNNING or WAITING_FOR_USER row
takes the live path: it claims its delivery row, tries to inject into the
live run, requests a resume and hands off to a resume that claims the lease.
The run can end FAILED (lease recovery) or COMPLETED (its own runner) at any
point in that window. The RESUME_REQUESTED transition and the resume lease
claim refuse an ended run for every message, so it is never flipped back to
RUNNING.

A refused message that was never written into the run is not settled as
unknown: its row is withdrawn and the message is accepted as a new turn,
exactly as if the snapshot had already shown the ended run -- in the same
handler when the transition refuses it, on the durable retry when the lease
claim does. A message the live run already accepted before it ended keeps the
at-most-once answer: never resumed, never resent, outcome unknown.
"""

from __future__ import annotations

import asyncio
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Iterator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tests.web.api.test_durable_message_resume_contention import (
    _live_control_environment,
    _live_task,
    _message_command,
    _user,
    live_task_lease,
)
from tests.web.api.test_recovered_delivery_failed_run import (
    ENDED_STATUSES,
    RECOVERY_ERROR,
    _end_run,
    _end_run_directly,
    _expired_running_task,
    _real_resume_environment,
    _update_task,
)
from tests.web.api.test_recovered_delivery_outcome_unknown import (
    MESSAGE,
    TURN_ID,
    _assert_outcome_unknown_frames,
    _outcome_unknown_result,
    _RecordingReply,
    _row_status,
    _settled_task,
    _user_rows,
)
from xagent.web.models.chat_message import TaskChatMessage
from xagent.web.models.database import Base, get_db, get_engine, init_db
from xagent.web.models.task import Task, TaskStatus
from xagent.web.services import task_command_execution as command_execution_service
from xagent.web.services import task_execution as task_execution_service
from xagent.web.services import task_execution_controller as controller_module
from xagent.web.services.chat_history_service import (
    DELIVERY_DISPATCHED,
    DELIVERY_FAILED,
    DELIVERY_PENDING,
    withdraw_pending_user_message_delivery_sync,
)
from xagent.web.services.client_error_messages import ClientErrorCode
from xagent.web.services.task_command_execution import execute_durable_task_command
from xagent.web.services.task_command_transport import (
    TaskCommandDeferred,
    TaskCommandKind,
    get_runner_id,
)
from xagent.web.services.task_execution import ResumeReservationOutcome
from xagent.web.services.task_execution_controller import TaskControlState
from xagent.web.services.task_orchestrator import TaskTurnOrchestrator, TurnKind

# Re-exported fixture from the contention suite.
live_task_lease = live_task_lease


@pytest.fixture()
def db_session(tmp_path):
    init_db(db_url=f"sqlite:///{tmp_path / 'fresh_message_ended_run.db'}")
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


@pytest.fixture()
def begin_turn_spy() -> Iterator[AsyncMock]:
    spy = AsyncMock(wraps=TaskTurnOrchestrator.begin_turn)
    with patch.object(TaskTurnOrchestrator, "begin_turn", spy):
        yield spy


def _accepted_result(task_id: int) -> dict[str, Any]:
    return {
        "task_id": task_id,
        "command_id": TURN_ID,
        "kind": TaskCommandKind.MESSAGE.value,
    }


def _assert_started_a_new_turn(db, task_id: int, begin_turn_spy: AsyncMock) -> None:
    """The message ran as a new turn on a new run; the ended run is gone."""

    begin_turn_spy.assert_awaited_once()
    assert begin_turn_spy.await_args.kwargs["kind"] == TurnKind.APPEND
    rows = [row for row in _user_rows(db, task_id) if row.turn_id == TURN_ID]
    assert len(rows) == 1
    assert rows[0].content == MESSAGE
    db.expire_all()
    stored = db.get(Task, task_id)
    assert stored.status == TaskStatus.RUNNING
    assert stored.run_id not in {None, "live-run"}
    # A new turn resets the ended run's diagnostic, as any APPEND does.
    assert stored.error_message is None


def _assert_run_not_resumed(db, task_id: int, ended_status: TaskStatus) -> None:
    db.expire_all()
    stored = db.get(Task, task_id)
    assert stored.status == ended_status
    assert stored.control_state == ended_status.value.lower()
    assert stored.run_id == "live-run"
    assert stored.runner_id is None
    if ended_status == TaskStatus.FAILED:
        assert stored.error_message == RECOVERY_ERROR


def _no_outcome_unknown(reply: _RecordingReply) -> bool:
    return not any(
        frame.get("error_code") == ClientErrorCode.MESSAGE_OUTCOME_UNKNOWN.value
        for frame in reply.frames
    )


def _transition_race(task_id: int, ended_status: TaskStatus, fence: str):
    """End the run just before the RESUME_REQUESTED transition writes.

    ``pre_check`` lands it before the transition loads the row;
    ``conditional_update`` after the load, so only the UPDATE's own status
    predicate can refuse it.
    """

    if fence == "pre_check":
        real_sync = controller_module.transition_task_control_state_sync

        def end_then_transition(*args: Any, **kwargs: Any):
            _end_run(task_id, ended_status)
            return real_sync(*args, **kwargs)

        return patch.object(
            controller_module,
            "transition_task_control_state_sync",
            side_effect=end_then_transition,
        )
    real_apply = controller_module.apply_task_control_transition

    def end_then_apply(task_row: Task, *args: Any, **kwargs: Any):
        assert task_row.status == TaskStatus.RUNNING
        _end_run(task_id, ended_status)
        return real_apply(task_row, *args, **kwargs)

    return patch.object(
        controller_module,
        "apply_task_control_transition",
        side_effect=end_then_apply,
    )


def _end_after_transition(task_id: int, end: Any):
    real_sync = controller_module.transition_task_control_state_sync

    def transition_then_end(*args: Any, **kwargs: Any):
        snapshot = real_sync(*args, **kwargs)
        end()
        return snapshot

    return patch.object(
        controller_module,
        "transition_task_control_state_sync",
        side_effect=transition_then_end,
    )


async def _wait_for_resume_to_finish(background_manager: Any, task_id: int) -> None:
    """Wait for the handed-off resume, including one not yet promoted."""

    for _ in range(300):
        coordinator = background_manager.resume_tasks.get(task_id)
        if task_id not in background_manager.running_tasks and (
            coordinator is None or coordinator.done()
        ):
            return
        await asyncio.sleep(0.01)
    raise AssertionError("the resume coordinator never finished")


def _deferred_resume_agent() -> MagicMock:
    agent = MagicMock()
    agent.supports_live_control.return_value = True
    agent.post_user_message = AsyncMock()
    return agent


@pytest.mark.asyncio
@ENDED_STATUSES
@pytest.mark.parametrize("fence", ["pre_check", "conditional_update"])
async def test_run_ending_before_the_transition_appends_the_message(
    db_session,
    recording_reply: _RecordingReply,
    begin_turn_spy: AsyncMock,
    fence: str,
    ended_status: TaskStatus,
) -> None:
    owner = _user(db_session, f"fresh-transition-{fence}-{ended_status.value}")
    task = _expired_running_task(db_session, int(owner.id))
    task_id = int(task.id)

    with (
        _live_control_environment(outcome=ResumeReservationOutcome.RESERVED) as (
            agent,
            background_manager,
        ),
        _transition_race(task_id, ended_status, fence) as raced,
    ):
        result = await execute_durable_task_command(
            _message_command(task, owner, TURN_ID, attempt_count=1)
        )
        task_execution_service.execute_resume_background.assert_not_called()

    assert raced.call_count == 1
    assert raced.call_args.kwargs["refuse_terminal_status"] is True
    assert result == _accepted_result(task_id)
    background_manager.release_resume_reservation.assert_called_with(task_id)
    background_manager.register_reserved_resume.assert_not_called()
    agent.post_user_message.assert_not_awaited()
    _assert_started_a_new_turn(db_session, task_id, begin_turn_spy)
    assert _no_outcome_unknown(recording_reply)


@pytest.mark.asyncio
@ENDED_STATUSES
@pytest.mark.parametrize("read", ["before_the_resume", "after_the_withdrawal"])
async def test_run_ending_before_the_lease_claim_appends_on_retry(
    db_session,
    recording_reply: _RecordingReply,
    ended_status: TaskStatus,
    read: str,
) -> None:
    """The resume's claim is refused after the handler handed the turn off.

    The message was never injected, so the resume withdraws its row and the
    command's retry, finding no row, accepts it as a new turn. ``read`` pins
    whether the command read the row before the resume withdrew it (still
    pending) or after (absent, answered by the handoff marker); both defer.
    """

    owner = _user(db_session, f"fresh-claim-{read}-{ended_status.value}")
    task = _expired_running_task(db_session, int(owner.id))
    task_id = int(task.id)
    agent = _deferred_resume_agent()
    background_manager = task_execution_service.BackgroundTaskManager()
    resume_spy = AsyncMock(wraps=task_execution_service.execute_resume_background)
    real_status = command_execution_service._load_command_message_delivery_status
    status_reads = 0

    def status_after_withdrawal(task_id_arg: int, turn_id: str) -> str | None:
        nonlocal status_reads
        status_reads += 1
        status = real_status(task_id_arg, turn_id)
        # The first read precedes the handler; only the post-handoff one waits.
        deadline = time.monotonic() + 5
        while status_reads > 1 and status is not None and time.monotonic() < deadline:
            time.sleep(0.01)
            status = real_status(task_id_arg, turn_id)
        return status

    status_patch = (
        patch.object(
            command_execution_service,
            "_load_command_message_delivery_status",
            side_effect=status_after_withdrawal,
        )
        if read == "after_the_withdrawal"
        else patch.object(
            command_execution_service,
            "_load_command_message_delivery_status",
            side_effect=real_status,
        )
    )
    agent_patch, manager_patch = _real_resume_environment(agent, background_manager)
    publish = AsyncMock()
    with (
        agent_patch,
        manager_patch,
        patch.object(task_execution_service, "execute_resume_background", resume_spy),
        patch.object(task_execution_service, "publish_task_event", publish),
        _end_after_transition(task_id, lambda: _end_run(task_id, ended_status)),
        status_patch,
    ):
        with pytest.raises(TaskCommandDeferred) as deferred:
            await execute_durable_task_command(
                _message_command(task, owner, TURN_ID, attempt_count=1)
            )
        assert resume_spy.await_count == 1
        assert resume_spy.await_args.kwargs["refuse_terminal_status"] is True
        assert resume_spy.await_args.kwargs["delivery_claimed_fresh"] is True
        await _wait_for_resume_to_finish(background_manager, task_id)

    if read == "after_the_withdrawal":
        assert "new turn" in str(deferred.value)
    agent.post_user_message.assert_not_awaited()
    # Withdrawn, and the ended run was not resumed.
    assert [row for row in _user_rows(db_session, task_id)] == []
    _assert_run_not_resumed(db_session, task_id, ended_status)
    assert not any(
        call.args[0].get("error_code") == ClientErrorCode.TASK_BUSY.value
        for call in publish.await_args_list
    )

    begin_turn = AsyncMock(wraps=TaskTurnOrchestrator.begin_turn)
    with (
        _live_control_environment(outcome=ResumeReservationOutcome.RESERVED) as (
            retry_agent,
            _,
        ),
        patch.object(TaskTurnOrchestrator, "begin_turn", begin_turn),
    ):
        result = await execute_durable_task_command(
            _message_command(task, owner, TURN_ID, attempt_count=2)
        )
        task_execution_service.execute_resume_background.assert_not_called()

    assert result == _accepted_result(task_id)
    retry_agent.post_user_message.assert_not_awaited()
    _assert_started_a_new_turn(db_session, task_id, begin_turn)
    assert _no_outcome_unknown(recording_reply)


@pytest.mark.asyncio
@ENDED_STATUSES
async def test_direct_sender_is_told_to_resend_when_the_claim_finds_the_run_ended(
    db_session,
    ended_status: TaskStatus,
) -> None:
    """Without a durable command the sender's ack carries the answer."""

    owner = _user(db_session, f"fresh-direct-{ended_status.value}")
    task = _expired_running_task(db_session, int(owner.id))
    task_id = int(task.id)
    agent = _deferred_resume_agent()
    background_manager = task_execution_service.BackgroundTaskManager()
    reply = _RecordingReply()
    message_data = {
        "type": "chat_message",
        "message": MESSAGE,
        "client_message_id": TURN_ID,
        "files": [],
        "user": owner,
    }
    agent_patch, manager_patch = _real_resume_environment(agent, background_manager)
    with (
        agent_patch,
        manager_patch,
        patch.object(task_execution_service, "publish_task_event", AsyncMock()),
        _end_after_transition(task_id, lambda: _end_run(task_id, ended_status)),
    ):
        await command_execution_service.handle_task_message(
            reply, task_id, dict(message_data)
        )
        await _wait_for_resume_to_finish(background_manager, task_id)

    rejected = [f for f in reply.frames if f.get("type") == "message_rejected"]
    assert len(rejected) == 1
    assert rejected[0]["rejection_outcome"] == "not_accepted"
    assert rejected[0]["retry_with_new_id"] is True
    assert rejected[0]["error_code"] == ClientErrorCode.MESSAGE_DELIVERY_FAILED.value
    assert _user_rows(db_session, task_id) == []
    _assert_run_not_resumed(db_session, task_id, ended_status)

    # Even a same-id resend is safe: nothing of the first attempt remains.
    reply.frames.clear()
    begin_turn = AsyncMock(wraps=TaskTurnOrchestrator.begin_turn)
    with (
        _live_control_environment(outcome=ResumeReservationOutcome.RESERVED),
        patch.object(TaskTurnOrchestrator, "begin_turn", begin_turn),
    ):
        await command_execution_service.handle_task_message(
            reply, task_id, dict(message_data)
        )
    assert [f["type"] for f in reply.frames if f.get("turn_id") == TURN_ID] == [
        "message_accepted"
    ]
    _assert_started_a_new_turn(db_session, task_id, begin_turn)


def _live_owned_task(db_session, owner, live_task_lease) -> Task:
    task = _live_task(db_session, int(owner.id))
    task.runner_id = get_runner_id()
    task.lease_expires_at = datetime.now(timezone.utc) + timedelta(minutes=1)
    db_session.commit()
    live_task_lease(db_session, task)
    return task


@pytest.mark.asyncio
@ENDED_STATUSES
async def test_posted_message_whose_run_ends_before_the_transition_is_unknown(
    live_task_lease,
    db_session,
    recording_reply: _RecordingReply,
    begin_turn_spy: AsyncMock,
    ended_status: TaskStatus,
) -> None:
    """The live run accepted the message, then ended before the handoff.

    Whether it read the message is unknown: the run is not resumed, the
    message is not run again as a new turn, and the sender is told so.
    """

    owner = _user(db_session, f"posted-transition-{ended_status.value}")
    task = _live_owned_task(db_session, owner, live_task_lease)
    task_id = int(task.id)
    real_sync = controller_module.transition_task_control_state_sync

    def end_then_transition(*args: Any, **kwargs: Any):
        _end_run_directly(task_id, ended_status)
        return real_sync(*args, **kwargs)

    with (
        _live_control_environment(outcome=ResumeReservationOutcome.RESERVED) as (
            agent,
            background_manager,
        ),
        patch.object(
            controller_module,
            "transition_task_control_state_sync",
            side_effect=end_then_transition,
        ),
    ):
        result = await execute_durable_task_command(
            _message_command(task, owner, TURN_ID, attempt_count=1)
        )
        task_execution_service.execute_resume_background.assert_not_called()

    agent.post_user_message.assert_awaited_once()
    assert result == _outcome_unknown_result(task)
    begin_turn_spy.assert_not_awaited()
    background_manager.register_reserved_resume.assert_not_called()
    assert _row_status(db_session, task_id) == DELIVERY_DISPATCHED
    _assert_run_not_resumed(db_session, task_id, ended_status)
    _assert_outcome_unknown_frames(recording_reply)


@pytest.mark.asyncio
@ENDED_STATUSES
async def test_posted_message_whose_run_ends_before_the_claim_keeps_its_notice(
    live_task_lease,
    db_session,
    ended_status: TaskStatus,
) -> None:
    """Injected and handed off, then the run ended: the recovered semantics.

    The command already answered accepted; the ended run is not resumed and a
    task-wide outcome-unknown notice replaces the answer no resume will give.
    """

    owner = _user(db_session, f"posted-claim-{ended_status.value}")
    task = _live_owned_task(db_session, owner, live_task_lease)
    task_id = int(task.id)
    agent = MagicMock()
    agent.supports_live_control.return_value = True
    agent.post_user_message = AsyncMock(
        return_value=task_execution_service.UserMessageInjectionOutcome.POSTED_FRESH
    )
    background_manager = task_execution_service.BackgroundTaskManager()
    resume_spy = AsyncMock(wraps=task_execution_service.execute_resume_background)
    publish = AsyncMock()
    agent_patch, manager_patch = _real_resume_environment(agent, background_manager)
    with (
        agent_patch,
        manager_patch,
        patch.object(task_execution_service, "execute_resume_background", resume_spy),
        patch.object(task_execution_service, "publish_task_event", publish),
        _end_after_transition(
            task_id, lambda: _end_run_directly(task_id, ended_status)
        ),
    ):
        result = await execute_durable_task_command(
            _message_command(task, owner, TURN_ID, attempt_count=1)
        )
        assert resume_spy.await_args.kwargs["delivery_already_dispatched"] is True
        assert resume_spy.await_args.kwargs["delivery_claimed_fresh"] is True
        await _wait_for_resume_to_finish(background_manager, task_id)

    assert result == _accepted_result(task_id)
    assert _row_status(db_session, task_id) == DELIVERY_DISPATCHED
    _assert_run_not_resumed(db_session, task_id, ended_status)
    notices = [
        call.args[0]
        for call in publish.await_args_list
        if call.args[0].get("error_code")
        == ClientErrorCode.MESSAGE_OUTCOME_UNKNOWN.value
    ]
    assert len(notices) == 1
    assert notices[0]["turn_id"] == TURN_ID
    assert MESSAGE not in str(notices[0])


@pytest.mark.asyncio
async def test_fresh_message_to_a_live_run_still_injects(
    live_task_lease,
    db_session,
    recording_reply: _RecordingReply,
    begin_turn_spy: AsyncMock,
) -> None:
    owner = _user(db_session, "fresh-live-inject")
    task = _live_owned_task(db_session, owner, live_task_lease)
    task_id = int(task.id)

    with _live_control_environment(outcome=ResumeReservationOutcome.RESERVED) as (
        agent,
        background_manager,
    ):
        result = await execute_durable_task_command(
            _message_command(task, owner, TURN_ID, attempt_count=1)
        )
        resume = task_execution_service.execute_resume_background
        assert resume.call_count == 1
        assert resume.call_args.kwargs["refuse_terminal_status"] is True
        assert resume.call_args.kwargs["delivery_claimed_fresh"] is True

    agent.post_user_message.assert_awaited_once()
    assert result == _accepted_result(task_id)
    background_manager.register_reserved_resume.assert_called_once()
    begin_turn_spy.assert_not_awaited()
    assert _row_status(db_session, task_id) == DELIVERY_DISPATCHED
    db_session.expire_all()
    stored = db_session.get(Task, task_id)
    assert stored.status == TaskStatus.RUNNING
    assert stored.run_id == "live-run"
    assert stored.control_state == TaskControlState.RESUME_REQUESTED.value


@pytest.mark.asyncio
async def test_fresh_message_to_a_paused_resume_request_still_resumes(
    db_session,
    begin_turn_spy: AsyncMock,
) -> None:
    """A PAUSED run with a resume pending is resumable: the fences pass it."""

    owner = _user(db_session, "fresh-paused-resume")
    task = _settled_task(
        db_session, int(owner.id), status=TaskStatus.PAUSED, run_id="live-run"
    )
    task.control_state = TaskControlState.RESUME_REQUESTED.value
    db_session.commit()
    task_id = int(task.id)

    with _live_control_environment(outcome=ResumeReservationOutcome.RESERVED) as (
        _,
        background_manager,
    ):
        with pytest.raises(TaskCommandDeferred):
            await execute_durable_task_command(
                _message_command(task, owner, TURN_ID, attempt_count=1)
            )
        resume = task_execution_service.execute_resume_background
        assert resume.call_count == 1
        assert resume.call_args.kwargs["refuse_terminal_status"] is True

    background_manager.register_reserved_resume.assert_called_once()
    begin_turn_spy.assert_not_awaited()
    assert _row_status(db_session, task_id) == DELIVERY_PENDING

    ended: list[TaskStatus] = []
    lease = task_execution_service._acquire_resume_task_lease(
        task_id,
        int(owner.id),
        "live-run",
        refuse_terminal_status=True,
        ended_status_out=ended,
    )
    assert lease is not None
    assert ended == []
    db_session.expire_all()
    assert db_session.get(Task, task_id).status == TaskStatus.RUNNING


@pytest.mark.asyncio
@ENDED_STATUSES
async def test_resume_request_left_on_an_ended_run_routes_to_a_new_turn(
    db_session,
    recording_reply: _RecordingReply,
    begin_turn_spy: AsyncMock,
    ended_status: TaskStatus,
) -> None:
    """The snapshot already shows the run ended, with a resume still pending."""

    owner = _user(db_session, f"ended-resume-request-{ended_status.value}")
    task = _settled_task(
        db_session, int(owner.id), status=ended_status, run_id="live-run"
    )
    task.control_state = TaskControlState.RESUME_REQUESTED.value
    db_session.commit()
    task_id = int(task.id)

    with _live_control_environment(outcome=ResumeReservationOutcome.RESERVED) as (
        agent,
        background_manager,
    ):
        result = await execute_durable_task_command(
            _message_command(task, owner, TURN_ID, attempt_count=1)
        )
        task_execution_service.execute_resume_background.assert_not_called()

    assert result == _accepted_result(task_id)
    # Routed before the live path: no agent built, no resume slot taken.
    agent.supports_live_control.assert_not_called()
    background_manager.try_reserve_resume.assert_not_called()
    _assert_started_a_new_turn(db_session, task_id, begin_turn_spy)
    assert _no_outcome_unknown(recording_reply)


@pytest.mark.asyncio
async def test_claim_refused_by_a_live_owner_keeps_the_failed_delivery(
    db_session,
) -> None:
    """Only an ended run withdraws a fresh row; a live owner fails it as before."""

    owner = _user(db_session, "fresh-claim-live-owner")
    task = _expired_running_task(db_session, int(owner.id))
    task_id = int(task.id)
    agent = _deferred_resume_agent()
    background_manager = task_execution_service.BackgroundTaskManager()
    publish = AsyncMock()
    agent_patch, manager_patch = _real_resume_environment(agent, background_manager)
    with (
        agent_patch,
        manager_patch,
        patch.object(task_execution_service, "publish_task_event", publish),
        _end_after_transition(
            task_id,
            lambda: _update_task(
                task_id,
                runner_id="foreign-runner",
                lease_attempt_id="foreign-attempt",
                lease_expires_at=datetime.now(timezone.utc) + timedelta(minutes=5),
            ),
        ),
    ):
        with pytest.raises(TaskCommandDeferred):
            await execute_durable_task_command(
                _message_command(task, owner, TURN_ID, attempt_count=1)
            )
        await _wait_for_resume_to_finish(background_manager, task_id)

    agent.post_user_message.assert_not_awaited()
    assert _row_status(db_session, task_id) == DELIVERY_FAILED
    assert any(
        call.args[0].get("error_code") == ClientErrorCode.TASK_BUSY.value
        for call in publish.await_args_list
    )
    db_session.expire_all()
    stored = db_session.get(Task, task_id)
    assert stored.status == TaskStatus.RUNNING
    assert stored.runner_id == "foreign-runner"


def test_withdrawal_removes_only_a_pending_row(db_session) -> None:
    owner = _user(db_session, "withdraw-unit")
    task = _settled_task(
        db_session, int(owner.id), status=TaskStatus.FAILED, run_id="live-run"
    )
    task_id = int(task.id)
    for turn_id, status in (
        ("pending-turn", DELIVERY_PENDING),
        ("dispatched-turn", DELIVERY_DISPATCHED),
    ):
        db_session.add(
            TaskChatMessage(
                task_id=task_id,
                user_id=int(owner.id),
                role="user",
                message_type="user_message",
                content=MESSAGE,
                turn_id=turn_id,
                delivery_status=status,
            )
        )
    db_session.commit()

    assert withdraw_pending_user_message_delivery_sync(task_id, "pending-turn")
    assert not withdraw_pending_user_message_delivery_sync(task_id, "pending-turn")
    assert not withdraw_pending_user_message_delivery_sync(task_id, "dispatched-turn")
    assert [row.turn_id for row in _user_rows(db_session, task_id)] == [
        "dispatched-turn"
    ]


@ENDED_STATUSES
def test_resume_lease_claim_reports_the_ended_status(
    db_session, ended_status: TaskStatus
) -> None:
    owner = _user(db_session, f"claim-ended-status-{ended_status.value}")
    task = _settled_task(
        db_session, int(owner.id), status=ended_status, run_id="another-run"
    )
    ended: list[TaskStatus] = []
    not_resumable: list[bool] = []

    lease = task_execution_service._acquire_resume_task_lease(
        int(task.id),
        int(owner.id),
        "live-run",
        refuse_terminal_status=True,
        run_not_resumable_out=not_resumable,
        ended_status_out=ended,
    )

    assert lease is None
    # Reported whatever the run: a rotated run that ended is still ended.
    assert ended == [ended_status]
    assert not_resumable == [True]
