from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from xagent.core.agent.context import ContextManager
from xagent.core.agent.runner import AgentRunner, UserMessageInjectionOutcome
from xagent.web.models.database import Base, get_engine, get_session_local, init_db
from xagent.web.models.task import Task, TaskStatus
from xagent.web.models.user import User
from xagent.web.services import task_injection as module
from xagent.web.services.task_lease_service import TaskLease, TaskLeaseLostError


@pytest.fixture
def owned(tmp_path, monkeypatch):
    monkeypatch.setenv("XAGENT_SHARED_TASK_EXECUTION_ENABLED", "false")
    init_db(db_url=f"sqlite:///{tmp_path / 'injection.db'}")
    with get_session_local()() as db, db.begin():
        owner = User(username="owner", password_hash="unused")
        db.add(owner)
        db.flush()
        task = Task(
            user_id=owner.id,
            title="input",
            status=TaskStatus.RUNNING,
            run_id="run",
            runner_id="worker",
            lease_attempt_id="attempt",
            control_state="running",
            state_version=1,
            lease_expires_at=datetime.now(timezone.utc) + timedelta(minutes=5),
        )
        db.add(task)
        db.flush()
        lease = TaskLease(task.id, "worker", "run", "attempt")
    yield lease
    ContextManager()._contexts.clear()
    Base.metadata.drop_all(bind=get_engine())


async def post(service, lease):
    return await module.post_journaled_message(
        service,
        lease,
        execution_message="answer",
        display_message="answer",
        turn_id="turn",
        request_interrupt=False,
        reason="test",
    )


@pytest.mark.asyncio
async def test_journal_is_committed_before_injection_and_blocks_different_turn(owned):
    async def check(*args, **kwargs):
        with get_session_local()() as db:
            assert db.get(Task, owned.task_id).pending_injection["turn_id"] == "turn"
        return UserMessageInjectionOutcome.OUTCOME_UNKNOWN

    service = SimpleNamespace(post_user_message=AsyncMock(side_effect=check))
    assert await post(service, owned) is UserMessageInjectionOutcome.OUTCOME_UNKNOWN
    with pytest.raises(TaskLeaseLostError, match="earlier input"):
        await module.post_journaled_message(
            service,
            owned,
            execution_message="other",
            display_message="other",
            turn_id="new-id",
            request_interrupt=False,
            reason="test",
        )
    service.post_user_message.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("landed", [False, True])
async def test_fresh_process_settles_original_turn_without_duplicate(owned, landed):
    manager = ContextManager()
    context = manager.create_context(str(owned.task_id))
    context.add_user_message("initial")
    durable = {"context": context.to_dict()}
    tracer = SimpleNamespace(
        load_latest_checkpoint=AsyncMock(return_value=durable), checkpoint=AsyncMock()
    )
    runner = AgentRunner(SimpleNamespace(llm=None), tracer=tracer)

    async def ambiguous(**payload):
        nonlocal durable
        if landed:
            durable = payload
        tracer.load_latest_checkpoint.side_effect = OSError("offline")
        raise OSError("ack lost")

    tracer.checkpoint.side_effect = ambiguous

    async def inject(*args, **kwargs):
        return (await runner.inject_user_message(*args, **kwargs)).outcome

    assert (
        await post(SimpleNamespace(post_user_message=inject), owned)
        is UserMessageInjectionOutcome.OUTCOME_UNKNOWN
    )
    manager._contexts.clear()  # Process restart: no in-memory poison or message retained.
    new_runner = AgentRunner(SimpleNamespace(llm=None), tracer=tracer)
    tracer.load_latest_checkpoint.side_effect = None
    tracer.load_latest_checkpoint.return_value = durable
    tracer.checkpoint.side_effect = None

    async def settle(*args, **kwargs):
        return (
            await new_runner.settle_injection_against_checkpoint(*args, **kwargs)
        ).outcome

    pending = await module.load_pending_injection(owned)
    assert pending is not None
    outcome = await module.settle_journaled_message(
        SimpleNamespace(settle_injection_against_checkpoint=settle), owned, pending
    )
    assert outcome is UserMessageInjectionOutcome.POSTED_FRESH
    assert [m.content for m in manager.get_context(str(owned.task_id)).messages] == [
        "initial",
        "answer",
    ]
    module.complete_injection(owned, "turn")
    with get_session_local()() as db:
        assert db.get(Task, owned.task_id).pending_injection is None
    assert tracer.checkpoint.await_count == (1 if landed else 2)


@pytest.mark.asyncio
async def test_replacement_owner_cannot_be_cleared_by_old_attempt(owned):
    await post(
        SimpleNamespace(
            post_user_message=AsyncMock(
                return_value=UserMessageInjectionOutcome.OUTCOME_UNKNOWN
            )
        ),
        owned,
    )
    with get_session_local()() as db, db.begin():
        db.get(Task, owned.task_id).lease_attempt_id = "replacement"
    with pytest.raises(TaskLeaseLostError):
        module.complete_injection(owned, "turn")
    with get_session_local()() as db:
        assert db.get(Task, owned.task_id).pending_injection is not None


@pytest.mark.asyncio
async def test_expired_unknown_run_is_recoverable_even_without_checkpoint_pointer(
    owned,
):
    from xagent.web.services.task_lease_recovery import (
        recover_expired_task_leases_batch_isolated,
    )

    await post(
        SimpleNamespace(
            post_user_message=AsyncMock(
                return_value=UserMessageInjectionOutcome.OUTCOME_UNKNOWN
            )
        ),
        owned,
    )
    with get_session_local()() as db, db.begin():
        db.get(Task, owned.task_id).lease_expires_at = datetime.now(
            timezone.utc
        ) - timedelta(seconds=1)
    batch = recover_expired_task_leases_batch_isolated(
        cutoff=datetime.now(timezone.utc), batch_size=10, after=None
    )
    assert batch.recovered == 1
    with get_session_local()() as db:
        task = db.get(Task, owned.task_id)
        assert task.status == TaskStatus.PAUSED
        assert task.pending_injection["turn_id"] == "turn"
        assert task.runner_id is None
    assert module._pending_candidates(0, 10)[0][0] == owned.task_id


@pytest.mark.asyncio
async def test_explicit_pause_excludes_automatic_restart(owned):
    from xagent.web.services.task_command_execution import (
        _apply_pause_requested_isolated,
    )

    await post(
        SimpleNamespace(
            post_user_message=AsyncMock(
                return_value=UserMessageInjectionOutcome.OUTCOME_UNKNOWN
            )
        ),
        owned,
    )
    assert _apply_pause_requested_isolated(owned.task_id, expected_run_id="run")
    with get_session_local()() as db, db.begin():
        task = db.get(Task, owned.task_id)
        assert task.pending_injection["auto_resume"] is False
        task.status = TaskStatus.PAUSED
    assert module._pending_candidates(0, 10) == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "shared,landed,fault",
    [
        (False, False, None),
        (False, True, None),
        (True, False, None),
        (True, True, None),
        (False, True, "offline"),
        (True, True, "resume_read"),
    ],
)
async def test_expired_journal_runs_through_recovery_and_real_runner(
    owned, monkeypatch, shared, landed, fault
):
    import asyncio
    from unittest.mock import MagicMock

    from xagent.web.models.chat_message import TaskChatMessage
    from xagent.web.services import task_execution
    from xagent.web.services.agent_service_manager import get_agent_manager
    from xagent.web.services.task_coordinator_runtime import close_task_coordinators
    from xagent.web.services.task_lease_recovery import (
        recover_expired_task_leases_batch_isolated,
    )

    monkeypatch.setenv("XAGENT_SHARED_TASK_EXECUTION_ENABLED", str(shared).lower())
    await post(
        SimpleNamespace(
            post_user_message=AsyncMock(
                return_value=UserMessageInjectionOutcome.OUTCOME_UNKNOWN
            )
        ),
        owned,
    )
    context = ContextManager().create_context(str(owned.task_id))
    context.add_user_message("initial")
    context.metadata["task_source"] = "stale-source"
    if landed:
        context.add_user_message("answer", metadata={"turn_id": "turn"})
    durable = {"context": context.to_dict()}
    ContextManager()._contexts.clear()
    with get_session_local()() as db, db.begin():
        db.get(Task, owned.task_id).source = "slack"
        db.get(Task, owned.task_id).lease_expires_at = datetime.now(
            timezone.utc
        ) - timedelta(seconds=1)
        db.add(
            TaskChatMessage(
                task_id=owned.task_id,
                user_id=1,
                role="user",
                content="answer",
                message_type="user_message",
                turn_id="turn",
                delivery_status="outcome_unknown",
            )
        )
    recover_expired_task_leases_batch_isolated(
        cutoff=datetime.now(timezone.utc), batch_size=10, after=None
    )

    executed = []

    class Pattern:
        async def run(self, context, **kwargs):
            assert context.metadata["task_source"] == "slack"
            assert context.metadata["run_id"] == owned.run_id
            executed.append([m.content for m in context.messages])
            with get_session_local()() as db:
                assert db.get(Task, owned.task_id).pending_injection is None
            return {"success": True, "status": "completed", "output": "done"}

    reads = 0

    async def read(*args, **kwargs):
        nonlocal reads
        reads += 1
        if fault == "offline" or (fault == "resume_read" and reads == 2):
            from xagent.core.agent.checkpoint import CheckpointUnavailableError

            raise CheckpointUnavailableError("offline")
        return durable

    async def write(**payload):
        nonlocal durable
        durable = payload

    tracer = SimpleNamespace(load_latest_checkpoint=read, checkpoint=write)
    runner = AgentRunner(
        SimpleNamespace(llm=None, patterns=[Pattern()], tools=[]),
        tracer=tracer,
        workspace_enabled=False,
    )

    async def settle(*args, **kwargs):
        return (
            await runner.settle_injection_against_checkpoint(*args, **kwargs)
        ).outcome

    agent = MagicMock(
        settle_injection_against_checkpoint=AsyncMock(side_effect=settle),
        resume_execution_by_id=AsyncMock(side_effect=runner.resume),
    )
    monkeypatch.setattr(
        get_agent_manager(), "get_agent_for_task", AsyncMock(return_value=agent)
    )
    monkeypatch.setattr(task_execution, "publish_task_event", AsyncMock())
    try:
        assert await module.recover_pending_injections(after_id=0, batch_size=10) == 0
        child = task_execution.background_task_manager.resume_tasks[owned.task_id]
        await asyncio.wait_for(child, 10)
        if fault is not None:
            assert executed == []
            with get_session_local()() as db, db.begin():
                task = db.get(Task, owned.task_id)
                assert task.pending_injection is not None
                assert task.status == TaskStatus.RUNNING
                task.lease_expires_at = datetime.now(timezone.utc) - timedelta(
                    seconds=1
                )
            await close_task_coordinators()
            recover_expired_task_leases_batch_isolated(
                cutoff=datetime.now(timezone.utc), batch_size=10, after=None
            )
            fault = None
            await module.recover_pending_injections(after_id=0, batch_size=10)
            child = task_execution.background_task_manager.resume_tasks[owned.task_id]
            await asyncio.wait_for(child, 10)
        assert executed == [["initial", "answer"]]
        with get_session_local()() as db:
            task = db.get(Task, owned.task_id)
            assert task.pending_injection is None
            assert task.status == TaskStatus.COMPLETED
            assert db.query(TaskChatMessage).filter_by(
                turn_id="turn"
            ).one().delivery_status in ("dispatched", "completed")
        assert module._pending_candidates(0, 10) == []
    finally:
        await close_task_coordinators()


@pytest.mark.asyncio
async def test_queued_handoff_honors_pause_until_explicit_resume(owned):
    from xagent.web.services.task_command_execution import (
        _apply_pause_requested_isolated,
    )
    from xagent.web.services.task_execution import _acquire_resume_task_lease
    from xagent.web.services.task_execution_controller import (
        TaskControlState,
        transition_task_control_state_sync,
    )
    from xagent.web.services.task_lease_service import release_task_lease_no_commit

    await post(
        SimpleNamespace(
            post_user_message=AsyncMock(
                return_value=UserMessageInjectionOutcome.OUTCOME_UNKNOWN
            )
        ),
        owned,
    )
    assert _apply_pause_requested_isolated(owned.task_id, expected_run_id="run")
    with pytest.raises(TaskLeaseLostError, match="paused"):
        module.complete_injection(owned, "turn")
    with get_session_local()() as db, db.begin():
        release_task_lease_no_commit(db, owned, status=TaskStatus.PAUSED)
    assert _acquire_resume_task_lease(owned.task_id, 1, "run") is None
    transition_task_control_state_sync(
        owned.task_id,
        TaskControlState.RESUME_REQUESTED,
        expected_run_id="run",
        resume_pending_input=True,
    )
    resumed = _acquire_resume_task_lease(owned.task_id, 1, "run")
    assert resumed is not None
    module.complete_injection(resumed, "turn")
    with get_session_local()() as db:
        assert db.get(Task, owned.task_id).pending_injection is None


@pytest.mark.asyncio
@pytest.mark.parametrize("exit_before_claim", ["cancel", "busy", "setup_failure"])
async def test_unknown_handoff_never_becomes_failed_before_acquisition(
    owned, monkeypatch, exit_before_claim
):
    import asyncio
    from unittest.mock import MagicMock

    from xagent.web.models.chat_message import TaskChatMessage
    from xagent.web.services import task_execution

    await post(
        SimpleNamespace(
            post_user_message=AsyncMock(
                return_value=UserMessageInjectionOutcome.OUTCOME_UNKNOWN
            )
        ),
        owned,
    )
    with get_session_local()() as db, db.begin():
        db.add(
            TaskChatMessage(
                task_id=owned.task_id,
                user_id=1,
                role="user",
                content="answer",
                message_type="user_message",
                turn_id="turn",
                delivery_status="pending",
            )
        )
    monkeypatch.setattr(
        task_execution.background_task_manager, "promote_resume_task", MagicMock()
    )
    monkeypatch.setattr(
        task_execution, "_acquire_resume_task_lease", MagicMock(return_value=None)
    )
    monkeypatch.setattr(task_execution, "publish_task_event", AsyncMock())
    if exit_before_claim == "setup_failure":
        monkeypatch.setattr(
            task_execution,
            "resolve_execution_scope",
            MagicMock(side_effect=RuntimeError("unavailable")),
        )
    previous = (
        asyncio.create_task(asyncio.sleep(100))
        if exit_before_claim == "cancel"
        else None
    )
    if previous:
        previous.cancel()
    notifier = AsyncMock()
    work = task_execution.execute_resume_background(
        task_id=owned.task_id,
        agent_service=MagicMock(),
        task_owner_user_id=1,
        expected_run_id="run",
        previous_task=previous,
        delivery_turn_id="turn",
        delivery_outcome_unknown=True,
        delivery_notifier=notifier,
    )
    if previous:
        with pytest.raises(asyncio.CancelledError):
            await work
    else:
        await work
    assert notifier.await_count == 1
    assert notifier.call_args.kwargs["rejection_outcome"] == "outcome_unknown"
    assert notifier.call_args.kwargs["retry_with_new_id"] is False
    with get_session_local()() as db:
        assert (
            db.query(TaskChatMessage).filter_by(turn_id="turn").one().delivery_status
            == "outcome_unknown"
        )
        assert db.get(Task, owned.task_id).pending_injection is not None


@pytest.mark.asyncio
@pytest.mark.parametrize("accepted", [False, True])
async def test_recovery_settles_actual_reply_successor_only(owned, accepted):
    from xagent.web.models.task_command import TaskExecutionCommand

    with get_session_local()() as db, db.begin():
        for command_id, outcome in [
            ("original", "retryable_unavailable"),
            ("successor", "unknown"),
        ]:
            db.add(
                TaskExecutionCommand(
                    task_id=owned.task_id,
                    command_id=command_id,
                    kind="resume_input",
                    payload={},
                    target_run_id="run",
                    status="completed",
                    result={"outcome": outcome},
                )
            )
    await module.post_journaled_message(
        SimpleNamespace(
            post_user_message=AsyncMock(
                return_value=UserMessageInjectionOutcome.OUTCOME_UNKNOWN
            )
        ),
        owned,
        execution_message="answer",
        display_message="answer",
        turn_id="a2a:1:message",
        reply_command_id="successor",
        request_interrupt=False,
        reason="test",
    )
    if accepted:
        module.complete_injection(owned, "a2a:1:message")
    else:
        pending = await module.load_pending_injection(owned)
        service = SimpleNamespace(
            settle_injection_against_checkpoint=AsyncMock(
                return_value=UserMessageInjectionOutcome.NOT_POSTED
            )
        )
        assert not await module.settle_journaled_message(service, owned, pending)
    with get_session_local()() as db:
        outcomes = {
            c.command_id: c.result["outcome"] for c in db.query(TaskExecutionCommand)
        }
        assert outcomes == {
            "original": "retryable_unavailable",
            "successor": "accepted" if accepted else "not_resumable",
        }
        assert db.get(Task, owned.task_id).pending_injection is None


@pytest.mark.asyncio
@pytest.mark.parametrize("shared", [False, True])
async def test_append_rejects_unsettled_input_before_replacing_run(
    owned, monkeypatch, shared
):
    from xagent.web.services import task_orchestrator as orchestrator

    await post(
        SimpleNamespace(
            post_user_message=AsyncMock(
                return_value=UserMessageInjectionOutcome.OUTCOME_UNKNOWN
            )
        ),
        owned,
    )
    with get_session_local()() as db, db.begin():
        task = db.get(Task, owned.task_id)
        task.status = TaskStatus.PAUSED
        task.input = "original"
        task.last_checkpoint_event_id = "checkpoint"
    monkeypatch.setattr(orchestrator, "enqueues_task_turns", lambda: shared)
    with get_session_local()() as db:
        with pytest.raises(orchestrator.TaskTurnError, match="busy"):
            orchestrator._accept_turn_no_commit(
                db,
                owned.task_id,
                1,
                payload=orchestrator.TaskTurnPayload("replacement"),
                kind=orchestrator.TurnKind.APPEND,
            )
        db.commit()
    with get_session_local()() as db:
        task = db.get(Task, owned.task_id)
        assert task.run_id == "run"
        assert task.input == "original"
        assert task.last_checkpoint_event_id == "checkpoint"
        assert task.pending_injection["turn_id"] == "turn"
        assert task.status == TaskStatus.PAUSED


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [TaskStatus.RUNNING, TaskStatus.PAUSED])
async def test_public_pause_disables_recovery_without_live_runner(
    owned, monkeypatch, status
):
    from xagent.web.services import task_command_execution as commands
    from xagent.web.services import task_execution as execution
    from xagent.web.services.agent_service_manager import get_agent_manager

    await post(
        SimpleNamespace(
            post_user_message=AsyncMock(
                return_value=UserMessageInjectionOutcome.OUTCOME_UNKNOWN
            )
        ),
        owned,
    )
    with get_session_local()() as db, db.begin():
        task = db.get(Task, owned.task_id)
        task.status = status
        task.control_state = "paused" if status == TaskStatus.PAUSED else "running"
    monkeypatch.setattr(execution.background_task_manager, "running_tasks", {})
    build = AsyncMock(
        side_effect=AssertionError("pause must not construct an offline runner")
    )
    monkeypatch.setattr(get_agent_manager(), "get_agent_for_task", build)
    publish = AsyncMock()
    monkeypatch.setattr(commands, "publish_task_event", publish)
    reply = AsyncMock()
    data = {"user": SimpleNamespace(id=1, is_admin=False)}
    await commands.pause_task(reply, owned.task_id, data)
    assert "_durable_command_error" not in data
    reply.assert_not_awaited()
    build.assert_not_awaited()
    assert publish.call_args.args[0]["type"] == "task_pause_requested"
    with get_session_local()() as db:
        task = db.get(Task, owned.task_id)
        assert task.pending_injection["auto_resume"] is False
        assert task.run_id == "run"
        assert task.control_state == (
            "paused" if status == TaskStatus.PAUSED else "pause_requested"
        )
    assert module._pending_candidates(0, 10) == []


@pytest.mark.asyncio
async def test_conflicting_turn_rejects_without_leaving_recovery_journal(owned):
    from xagent.core.agent.runner import UserMessageInjectionConflictError

    context = ContextManager().create_context(str(owned.task_id))
    context.add_user_message("different", metadata={"turn_id": "turn"})
    runner = AgentRunner(SimpleNamespace(llm=None))

    async def inject(*args, **kwargs):
        return (await runner.inject_user_message(*args, **kwargs)).outcome

    with pytest.raises(UserMessageInjectionConflictError):
        await post(SimpleNamespace(post_user_message=inject), owned)
    assert await module.load_pending_injection(owned) is None

    # The same conflict in a previously unresolved journal is terminal too.
    await post(
        SimpleNamespace(
            post_user_message=AsyncMock(
                return_value=UserMessageInjectionOutcome.OUTCOME_UNKNOWN
            )
        ),
        owned,
    )
    runner.tracer = SimpleNamespace(
        load_latest_checkpoint=AsyncMock(return_value={"context": context.to_dict()})
    )

    async def settle(*args, **kwargs):
        return (
            await runner.settle_injection_against_checkpoint(*args, **kwargs)
        ).outcome

    pending = await module.load_pending_injection(owned)
    assert pending is not None
    assert (
        await module.settle_journaled_message(
            SimpleNamespace(settle_injection_against_checkpoint=settle), owned, pending
        )
        is UserMessageInjectionOutcome.NOT_POSTED
    )
    assert await module.load_pending_injection(owned) is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error_name", ["CheckpointCorruptError", "CheckpointAccessRefusedError"]
)
async def test_unreadable_checkpoint_suspends_automatic_recovery(owned, error_name):
    from xagent.core.agent import checkpoint
    from xagent.web.services.task_lease_recovery import (
        recover_expired_task_leases_batch_isolated,
    )

    await post(
        SimpleNamespace(
            post_user_message=AsyncMock(
                return_value=UserMessageInjectionOutcome.OUTCOME_UNKNOWN
            )
        ),
        owned,
    )
    pending = await module.load_pending_injection(owned)
    assert pending is not None
    service = SimpleNamespace(
        settle_injection_against_checkpoint=AsyncMock(
            side_effect=getattr(checkpoint, error_name)("unreadable")
        )
    )
    assert (
        await module.settle_journaled_message(service, owned, pending)
        is UserMessageInjectionOutcome.OUTCOME_UNKNOWN
    )
    with get_session_local()() as db, db.begin():
        db.get(Task, owned.task_id).lease_expires_at = datetime.now(
            timezone.utc
        ) - timedelta(seconds=1)
    recover_expired_task_leases_batch_isolated(
        cutoff=datetime.now(timezone.utc), batch_size=10, after=None
    )
    with get_session_local()() as db:
        task = db.get(Task, owned.task_id)
        assert task.status == TaskStatus.PAUSED
        assert task.pending_injection["turn_id"] == "turn"
        assert task.pending_injection["auto_resume"] is False
        assert "delivery remains unknown" in task.error_message
    assert module._pending_candidates(0, 10) == []


def test_recovery_observes_journal_written_after_snapshot(owned, monkeypatch):
    from xagent.web.services import task_lease_recovery as recovery

    with get_session_local()() as db, db.begin():
        db.get(Task, owned.task_id).lease_expires_at = datetime.now(
            timezone.utc
        ) - timedelta(seconds=1)
    original = recovery.recover_expired_task_lease_no_commit

    def land_journal_then_recover(db, candidate, **kwargs):
        assert kwargs["status"] == TaskStatus.FAILED
        module._journal(
            owned,
            dict(execution_message="answer", display_message="answer", turn_id="turn"),
        )
        return original(db, candidate, **kwargs)

    monkeypatch.setattr(
        recovery, "recover_expired_task_lease_no_commit", land_journal_then_recover
    )
    result = recovery.recover_expired_task_leases_batch_isolated(
        cutoff=datetime.now(timezone.utc), batch_size=10, after=None
    )
    assert result.recovered == 1
    with get_session_local()() as db:
        task = db.get(Task, owned.task_id)
        assert task.status == TaskStatus.PAUSED
        assert task.control_state == "paused"
        assert task.pending_injection is not None
        assert task.runner_id is None
    assert module._pending_candidates(0, 10)[0][0] == owned.task_id


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "status,success",
    [("completed", True), ("failed", False), ("waiting_for_user", False)],
)
async def test_resumed_finalizer_hands_pending_input_to_recovery(
    owned, status, success
):
    from xagent.web.services import task_execution as execution

    await post(
        SimpleNamespace(
            post_user_message=AsyncMock(
                return_value=UserMessageInjectionOutcome.OUTCOME_UNKNOWN
            )
        ),
        owned,
    )
    result = execution._finalize_resumed_task(
        owned.task_id,
        status=status,
        success=success,
        output="final answer" if success else "",
        task_owner_user_id=1,
        result={"status": status, "success": success},
        task_lease=owned,
        prepared_outputs=execution._PreparedTaskFileOutputs((), (), ()),
    )
    with get_session_local()() as db:
        task = db.get(Task, owned.task_id)
        assert task.status == TaskStatus.PAUSED
        assert task.pending_injection is not None
        assert task.runner_id is None
        if success:
            assert task.output == "final answer"
    assert result["final_status"] == "paused"
    assert result["lease_released"] is True
    assert module._pending_candidates(0, 10)[0][0] == owned.task_id


@pytest.mark.asyncio
async def test_first_finalizer_keeps_successful_answer_during_input_handoff(owned):
    from xagent.web.models.chat_message import TaskChatMessage
    from xagent.web.services import task_execution as execution

    await post(
        SimpleNamespace(
            post_user_message=AsyncMock(
                return_value=UserMessageInjectionOutcome.POSTED_FRESH
            )
        ),
        owned,
    )
    execution._finalize_task_execution_result_isolated(
        task_id=owned.task_id,
        task_user_id=1,
        pre_run_status=TaskStatus.RUNNING,
        result={"status": "completed", "success": True, "output": "final answer"},
        expected_run_id=owned.run_id,
        task_lease=owned,
        resolved_scope_segments=(),
        prepared_outputs=execution._PreparedTaskFileOutputs((), (), ()),
    )
    with get_session_local()() as db:
        task = db.get(Task, owned.task_id)
        assert task.status == TaskStatus.PAUSED
        assert task.output == "final answer"
        assert task.pending_injection is not None
        assert (
            db.query(TaskChatMessage)
            .filter_by(task_id=owned.task_id, role="assistant")
            .one()
            .content
            == "final answer"
        )
