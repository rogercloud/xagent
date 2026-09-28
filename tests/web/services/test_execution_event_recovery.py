"""V2 recovery must work without consulting compatibility content."""

import pytest
import sqlalchemy as sa

from tests.web.services.test_task_execution_event_writer import (
    canonical as canonical_fixture,
)
from tests.web.services.test_task_execution_event_writer import engine as engine_fixture
from tests.web.services.test_task_execution_event_writer import (
    facts,
)
from tests.web.services.test_task_execution_event_writer import (
    task_id as task_id_fixture,
)
from xagent.core.agent import ExecutionContext, PatternRuntime, ReActPattern
from xagent.core.agent.checkpoint import (
    CheckpointAccessRefusedError,
    CheckpointCorruptError,
    CheckpointUnavailableError,
    ExecutionEventPersistenceError,
    TraceCheckpointStore,
)
from xagent.core.agent.trace import Tracer
from xagent.web.models.task import Task
from xagent.web.models.task import TraceEvent as StoredTraceEvent
from xagent.web.models.task_execution_event import TaskExecutionEvent
from xagent.web.services.task_execution_event_writer import append_fact_no_commit
from xagent.web.tracing import ExecutionEventTraceAdapter

canonical = canonical_fixture
engine = engine_fixture
task_id = task_id_fixture


def tracer_for(task_id, scope=None):
    adapter = ExecutionEventTraceAdapter(task_id, build_id=scope)
    tracer = Tracer()
    tracer.handlers = [adapter]
    tracer.event_writer = adapter.commit_event
    return tracer


@pytest.mark.asyncio
@pytest.mark.parametrize("scope", [None, "child"])
async def test_checkpoint_reads_event_after_legacy_trace_deletion(canonical, scope):
    factory, tid = canonical
    tracer = tracer_for(tid, scope)
    store = TraceCheckpointStore(tracer)
    payload = {
        "execution_id": "execution",
        "context": ExecutionContext(execution_id="execution").to_dict(),
        "pattern_state": {},
        "label": "ready",
    }
    await store.save(payload)
    with factory() as db:
        db.execute(sa.delete(StoredTraceEvent).where(StoredTraceEvent.task_id == tid))
        db.commit()
    assert await store.load_latest_checkpoint("execution") == payload
    assert await store.load_latest_checkpoint("other-execution") is None
    if scope:
        assert (
            await TraceCheckpointStore(tracer_for(tid)).load_latest_checkpoint(
                "execution"
            )
            is None
        )


@pytest.mark.asyncio
async def test_corrupt_latest_state_does_not_fall_back_to_trace(canonical):
    factory, tid = canonical
    store = TraceCheckpointStore(tracer_for(tid))
    await store.save({"execution_id": "execution", "context": {"messages": []}})
    with factory() as db:
        event = db.query(TaskExecutionEvent).filter_by(kind="recovery_state").one()
        event.payload_version = 99
        db.commit()
    with pytest.raises(CheckpointCorruptError):
        await store.load_latest_checkpoint("execution")


@pytest.mark.asyncio
async def test_settled_execution_cannot_resume_older_checkpoint(canonical):
    factory, tid = canonical
    store = TraceCheckpointStore(tracer_for(tid))
    await store.save({"execution_id": "execution", "context": {"messages": []}})
    with factory() as db:
        append_fact_no_commit(
            db,
            task_id=tid,
            kind="execution_settled",
            key="settled",
            payload={"status": "failed", "result": {"status": "cancelled"}},
        )
        db.commit()
    with pytest.raises(CheckpointAccessRefusedError):
        await store.load_latest_checkpoint("execution")


@pytest.mark.asyncio
@pytest.mark.parametrize("scope", [None, "child"])
async def test_resume_reuses_tool_result_committed_before_next_checkpoint(
    canonical, scope, monkeypatch, tmp_path
):
    from tests.core.agent.test_react import FakeLLM, FakeTool
    from tests.core.agent.test_runner import FakeWorkspaceManager
    from xagent.core.agent import Agent, AgentRunner

    factory, tid = canonical
    tracer = tracer_for(tid, scope)
    store = TraceCheckpointStore(tracer)
    runtime = PatternRuntime(tracer=store)
    original_checkpoint = runtime.checkpoint

    async def crash_after_tool(label, **kwargs):
        if label in {"after_tool", "after_tool_batch"}:
            raise ExecutionEventPersistenceError("simulated crash after result commit")
        return await original_checkpoint(label, **kwargs)

    monkeypatch.setattr(runtime, "checkpoint", crash_after_tool)
    context = ExecutionContext(execution_id="tool-crash")
    context.add_user_message("2+2")
    tool = FakeTool()
    with pytest.raises(ExecutionEventPersistenceError):
        await ReActPattern(max_iterations=3).run(
            context=context,
            tools=[tool],
            runtime=runtime,
            llm=FakeLLM(
                responses=[
                    {
                        "content": "calculate",
                        "tool_calls": [
                            {
                                "id": "call",
                                "function": {
                                    "name": "calculator",
                                    "arguments": '{"expression":"2+2"}',
                                },
                            }
                        ],
                    }
                ]
            ),
        )
    assert len(tool.calls) == 1
    with factory() as db:
        db.execute(sa.delete(StoredTraceEvent).where(StoredTraceEvent.task_id == tid))
        db.commit()
    runner = AgentRunner(
        agent=Agent(
            name="recovered",
            patterns=[ReActPattern(max_iterations=3)],
            tools=[tool],
            llm=FakeLLM(responses=[{"content": "4", "done": True}]),
        ),
        tracer=store,
        workspace_manager=FakeWorkspaceManager(tmp_path),
    )
    result = await runner.resume("tool-crash", task="2+2")
    assert result["success"]
    assert len(tool.calls) == 1
    with factory() as db:
        assert (
            len([row for row in facts(db, tid) if row.kind == "tool_execution_start"])
            == 1
        )
        assert (
            len([row for row in facts(db, tid) if row.kind == "tool_execution_end"])
            == 1
        )


@pytest.mark.asyncio
async def test_unknown_tool_effect_is_not_reexecuted(canonical):
    from tests.core.agent.test_react import FakeTool

    _, tid = canonical
    runtime = PatternRuntime(tracer=TraceCheckpointStore(tracer_for(tid)))
    call = {
        "id": "call",
        "name": "calculator",
        "args": {"expression": "2+2"},
        "assistant_message_id": "batch",
        "tool_attempt_id": "attempt",
    }
    await runtime.on_tool_start(tool_call=call)
    tool = FakeTool()
    with pytest.raises(CheckpointUnavailableError, match="automatic replay is unsafe"):
        await ReActPattern()._execute_tool_safely(
            call, [tool], runtime, context=ExecutionContext()
        )
    assert tool.calls == []


@pytest.mark.asyncio
async def test_reader_preflights_all_pending_effects_before_scheduling(canonical):
    _, tid = canonical
    tracer = tracer_for(tid)
    store = TraceCheckpointStore(tracer)
    calls = [
        {
            "id": f"call-{i}",
            "name": "calculator",
            "args": {"expression": "2+2"},
            "assistant_message_id": "batch",
            "tool_attempt_id": f"attempt-{i}",
        }
        for i in range(2)
    ]
    await store.save(
        {
            "execution_id": "execution",
            "context": {"messages": []},
            "pattern_state": {"pending_tool_calls": calls},
        }
    )
    await PatternRuntime(tracer=store).on_tool_start(tool_call=calls[1])
    with pytest.raises(CheckpointUnavailableError, match="automatic replay is unsafe"):
        await store.load_latest_checkpoint("execution")


@pytest.mark.asyncio
async def test_interaction_anchor_and_question_do_not_read_legacy_content(
    canonical, monkeypatch
):
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    from xagent.web.services import task_execution
    from xagent.web.services.task_interaction_anchor import resolve_interaction_anchor
    from xagent.web.services.task_interaction_read import (
        get_pending_interaction_question,
    )
    from xagent.web.services.task_interaction_service import (
        _resolve_read_direction_anchor,
    )
    from xagent.web.services.task_lease_service import (
        acquire_task_lease,
        bind_task_lease_context,
    )
    from xagent.web.services.task_setup_snapshot import (
        load_task_reconstruction_snapshot_sync,
    )

    factory, tid = canonical
    monkeypatch.setattr(task_execution, "get_db", lambda: iter([factory()]))
    monkeypatch.setattr(task_execution, "publish_task_event", AsyncMock())
    with factory() as db:
        lease = acquire_task_lease(db, tid, new_run=True)
    with bind_task_lease_context(lease):
        await TraceCheckpointStore(tracer_for(tid)).save(
            {"execution_id": str(tid), "context": {"messages": []}}
        )
        await task_execution.make_agent_outbound_handler(tid, authoritative=True)(
            {
                "message": "Choose?",
                "expect_response": True,
                "metadata": {"interactions": [{"field": "choice", "type": "text"}]},
            }
        )
    with factory() as db:
        task = db.get(Task, tid)
        anchor = resolve_interaction_anchor(db, task)
        assert anchor is not None
        row = SimpleNamespace(
            id=1,
            task_id=tid,
            resume_trace_event_id=anchor.trace_event_id,
            resume_event_id=anchor.resume_event_id,
            resume_execution_id=anchor.resume_execution_id,
            resume_run_partition=anchor.resume_run_partition,
        )
        # Scrub compatibility content but keep identity/FK control storage.
        for legacy in db.query(StoredTraceEvent).all():
            legacy.data = {"unreadable": True}
        db.commit()

        def no_legacy_read(conn, cursor, statement, parameters, context, executemany):
            if statement.lstrip().lower().startswith("select") and (
                "trace_events" in statement.lower()
                or "task_chat_messages" in statement.lower()
                or "dag_executions" in statement.lower()
            ):
                raise AssertionError("Legacy content read")

        sa.event.listen(db.bind, "before_cursor_execute", no_legacy_read)
        try:
            assert _resolve_read_direction_anchor(db, row) is None
            assert resolve_interaction_anchor(db, task) == anchor
            assert get_pending_interaction_question(db, task) == (
                "Choose?",
                [{"field": "choice", "type": "text"}],
            )
            assert load_task_reconstruction_snapshot_sync(db, tid).has_history
            row.resume_execution_id = "wrong-execution"
            assert _resolve_read_direction_anchor(db, row).reason == "anchor_dangling"
        finally:
            sa.event.remove(db.bind, "before_cursor_execute", no_legacy_read)


@pytest.mark.asyncio
async def test_stale_lease_cannot_read_checkpoint_or_reuse_result(canonical):
    from xagent.web.services.task_lease_service import (
        acquire_task_lease,
        bind_task_lease_context,
    )

    factory, tid = canonical
    with factory() as db:
        lease = acquire_task_lease(db, tid, new_run=True)
    store = TraceCheckpointStore(tracer_for(tid))
    call = {
        "id": "call",
        "name": "calculator",
        "args": {},
        "assistant_message_id": "batch",
        "tool_attempt_id": "attempt",
    }
    with bind_task_lease_context(lease):
        await store.save({"execution_id": "execution", "context": {"messages": []}})
        await PatternRuntime(tracer=store).on_tool_start(tool_call=call)
        await PatternRuntime(tracer=store).on_tool_end(tool_call=call, result=4)
    with factory() as db:
        task = db.get(Task, tid)
        task.runner_id = "replacement-owner"
        db.commit()
    with bind_task_lease_context(lease):
        with pytest.raises(CheckpointAccessRefusedError):
            await store.load_latest_checkpoint("execution")
        with pytest.raises(CheckpointAccessRefusedError):
            await store.load_committed_tool_outcome(call)


@pytest.mark.asyncio
@pytest.mark.parametrize("scope", [None, "child"])
async def test_dag_resume_keeps_finished_step_and_reuses_child_result(
    canonical, scope, monkeypatch
):
    from tests.core.agent.test_dag import FakeTool, SequenceLLM, build_plan
    from xagent.core.agent import DAGPattern, PlanStep

    factory, tid = canonical
    store = TraceCheckpointStore(tracer_for(tid, scope))
    runtime = PatternRuntime(tracer=store)
    original_checkpoint = runtime.checkpoint

    async def crash_after_tool(label, **kwargs):
        if label in {"dag_after_tool", "dag_after_tool_batch"}:
            raise ExecutionEventPersistenceError("crash after child result")
        return await original_checkpoint(label, **kwargs)

    monkeypatch.setattr(runtime, "checkpoint", crash_after_tool)
    context = ExecutionContext(execution_id="dag-recovery")
    context.add_user_message("Prepare, calculate")
    plan = build_plan(
        PlanStep(id="prepare", task="Prepare"),
        PlanStep(id="calc", task="Calculate", dependencies=["prepare"]),
    )
    tool = FakeTool()
    with pytest.raises(ExecutionEventPersistenceError):
        await DAGPattern(lambda **_: plan, max_concurrency=1).run(
            context=context,
            tools=[tool],
            runtime=runtime,
            llm=SequenceLLM(
                [
                    {"content": "Prepared", "done": True},
                    {
                        "content": "Calculate",
                        "tool_calls": [
                            {
                                "id": "call",
                                "function": {
                                    "name": "calculator",
                                    "arguments": '{"expression":"6*7"}',
                                },
                            }
                        ],
                    },
                ]
            ),
        )
    with factory() as db:
        db.execute(sa.delete(StoredTraceEvent).where(StoredTraceEvent.task_id == tid))
        db.commit()
    snapshot = await store.load_latest_checkpoint("dag-recovery")
    pattern = DAGPattern(lambda **_: pytest.fail("must reuse adopted plan"))
    pattern.load_state(snapshot["pattern_state"])
    assert pattern.step_results == {"prepare": "Prepared"}
    result = await pattern.run(
        context=ExecutionContext.from_dict(snapshot["context"]),
        tools=[tool],
        runtime=PatternRuntime(tracer=store),
        llm=SequenceLLM([{"content": "42", "done": True}]),
    )
    assert result["success"]
    assert tool.calls == [{"expression": "6*7"}]
    assert result["step_results"] == {"prepare": "Prepared", "calc": "42"}


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ["failed", "waiting", "cancelled"])
async def test_tool_outcome_recovery_preserves_failure_and_waiting(canonical, outcome):
    from tests.core.agent.test_react import FakeTool

    _, tid = canonical
    runtime = PatternRuntime(tracer=TraceCheckpointStore(tracer_for(tid)))
    call = {
        "id": "call",
        "name": "calculator",
        "args": {"credential": "private"},
        "assistant_message_id": "batch",
        "tool_attempt_id": "attempt",
    }
    await runtime.on_tool_start(tool_call=call)
    tool = FakeTool()
    tool.sanitize_tool_args_for_trace = lambda args: {"credential": "redacted"}
    pattern = ReActPattern()
    if outcome == "cancelled":
        await runtime.on_tool_cancelled(tool_call=call)
        with pytest.raises(CheckpointUnavailableError):
            await pattern._execute_tool_safely(
                call, [tool], runtime, context=ExecutionContext()
            )
    else:
        result = (
            {"success": False, "error": "failed"}
            if outcome == "failed"
            else {"status": "waiting_for_user", "message": "Authorize"}
        )
        await runtime.on_tool_end(tool_call=call, result=result)
        assert (
            await pattern._execute_tool_safely(
                call, [tool], runtime, context=ExecutionContext()
            )
            == result
        )
        assert pattern.tool_ledger["call"].status == (
            "failed" if outcome == "failed" else "waiting_for_user"
        )
        assert pattern.tool_ledger["call"].args == {"credential": "redacted"}
    assert tool.calls == []


@pytest.mark.asyncio
async def test_owner_replaced_during_read_is_rechecked(canonical, monkeypatch):
    from xagent.web.services import task_execution_event_recovery as recovery
    from xagent.web.services.task_lease_service import (
        acquire_task_lease,
        bind_task_lease_context,
    )

    factory, tid = canonical
    with factory() as db:
        lease = acquire_task_lease(db, tid, new_run=True)
    store = TraceCheckpointStore(tracer_for(tid))
    original = recovery.read_event_checkpoint

    def replace_after_read(*args, **kwargs):
        result = original(*args, **kwargs)
        with factory() as db:
            db.execute(
                sa.update(Task).where(Task.id == tid).values(runner_id="new-owner")
            )
            db.commit()
        return result

    with bind_task_lease_context(lease):
        await store.save({"execution_id": str(tid), "context": {"messages": []}})
        monkeypatch.setattr(recovery, "read_event_checkpoint", replace_after_read)
        with pytest.raises(CheckpointAccessRefusedError):
            await store.load_latest_checkpoint(str(tid))


@pytest.mark.asyncio
async def test_cold_injection_preserves_compacted_application_and_pending_acceptance(
    canonical,
):
    from types import SimpleNamespace

    from xagent.core.agent.context import ContextManager
    from xagent.core.agent.runner import AgentRunner, UserMessageInjectionOutcome
    from xagent.web.models.chat_message import TaskChatMessage
    from xagent.web.services.chat_history_service import persist_user_message_no_commit

    factory, tid = canonical
    store = TraceCheckpointStore(tracer_for(tid))
    context = ExecutionContext(execution_id="input-recovery")
    manager = ContextManager()
    manager.set_context(context)
    runner = AgentRunner(
        SimpleNamespace(llm=None), tracer=store, context_manager=manager
    )
    with factory() as db:
        for turn, text in [("applied", "choose B"), ("pending", "choose C")]:
            persist_user_message_no_commit(
                db, tid, db.get(Task, tid).user_id, text, turn_id=turn
            )
        db.commit()
    posted = await runner.inject_user_message(
        context.execution_id, "choose B", turn_id="applied", request_interrupt=False
    )
    assert posted.outcome is UserMessageInjectionOutcome.POSTED_FRESH
    context.add_assistant_message("B selected")
    context.compact_with_llm_response("B selected")
    await store.save(
        {"execution_id": context.execution_id, "context": context.to_dict()}
    )
    with factory() as db:
        db.execute(sa.delete(StoredTraceEvent).where(StoredTraceEvent.task_id == tid))
        db.execute(sa.delete(TaskChatMessage).where(TaskChatMessage.task_id == tid))
        db.commit()
        before = len(facts(db, tid))
    cold = AgentRunner(SimpleNamespace(llm=None), tracer=store)
    replay = await cold.inject_user_message(
        context.execution_id, "choose B", turn_id="applied", request_interrupt=False
    )
    assert replay.outcome is UserMessageInjectionOutcome.POSTED_REPLAY
    with factory() as db:
        assert len(facts(db, tid)) == before
        assert [
            row.turn_id for row in facts(db, tid) if row.kind == "input_applied"
        ] == ["applied"]
    snapshot = await store.load_latest_checkpoint(context.execution_id)
    assert not any(
        message.get("content") == "choose C"
        for message in snapshot["context"]["messages"]
    )


@pytest.mark.asyncio
async def test_database_failure_is_unavailable_not_absence(canonical, monkeypatch):
    from xagent.web.services import task_execution_event_recovery as recovery

    _, tid = canonical
    store = TraceCheckpointStore(tracer_for(tid))
    await store.save({"execution_id": "execution", "context": {"messages": []}})

    def unavailable(*args, **kwargs):
        raise sa.exc.OperationalError("SELECT", {}, RuntimeError("offline"))

    monkeypatch.setattr(recovery, "read_event_checkpoint", unavailable)
    with pytest.raises(CheckpointUnavailableError):
        await store.load_latest_checkpoint("execution")
    monkeypatch.setattr(recovery, "read_committed_tool_outcome", unavailable)
    with pytest.raises(CheckpointUnavailableError):
        await store.load_committed_tool_outcome({"tool_attempt_id": "attempt"})


@pytest.mark.asyncio
@pytest.mark.parametrize("reader", ["checkpoint", "tool"])
async def test_terminal_status_after_read_blocks_return_with_same_lease(
    canonical, monkeypatch, reader
):
    from xagent.web.models.task import TaskStatus
    from xagent.web.services import task_execution_event_recovery as recovery
    from xagent.web.services.task_lease_service import (
        acquire_task_lease,
        bind_task_lease_context,
    )

    factory, tid = canonical
    with factory() as db:
        lease = acquire_task_lease(db, tid, new_run=True)
    store = TraceCheckpointStore(tracer_for(tid))
    runtime = PatternRuntime(tracer=store, execution_id=str(tid))
    call = {
        "id": "call",
        "name": "calculator",
        "arguments": {"expression": "2+2"},
        "assistant_message_id": "batch",
        "tool_attempt_id": "attempt",
    }
    original_check = recovery.check_recovery_owner
    checks = 0

    def cancel_at_return(db, task_id):
        nonlocal checks
        checks += 1
        if checks == 2:
            # Preserve the coordinator's lease identity, as external cancellation
            # can do. Its settlement is beyond the checkpoint reader's horizon.
            with factory() as writer:
                writer.execute(
                    sa.update(Task)
                    .where(Task.id == tid)
                    .values(status=TaskStatus.FAILED)
                )
                append_fact_no_commit(
                    writer,
                    task_id=tid,
                    kind="execution_settled",
                    key="late-cancel",
                    run_id=lease.run_id,
                    payload={"status": "failed", "result": {"status": "cancelled"}},
                )
                writer.commit()
        return original_check(db, task_id)

    with bind_task_lease_context(lease):
        await store.save({"execution_id": str(tid), "context": {"messages": []}})
        await runtime.on_tool_start(tool_call=call)
        await runtime.on_tool_end(
            tool_call=call, result={"success": True, "result": "4"}
        )
        monkeypatch.setattr(recovery, "check_recovery_owner", cancel_at_return)
        with pytest.raises(CheckpointAccessRefusedError, match="already ended"):
            if reader == "checkpoint":
                await store.load_latest_checkpoint(str(tid))
            else:
                await store.load_committed_tool_outcome(call)
    assert checks == 2
