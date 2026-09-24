from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from xagent.core.agent.checkpoint import CheckpointPersistenceError
from xagent.core.agent.context import ContextManager
from xagent.core.agent.runner import AgentRunner, UserMessageInjectionOutcome
from xagent.core.agent.runtime import PatternRuntime


@pytest.fixture
def live():
    manager = ContextManager()
    manager._contexts.clear()
    context = manager.create_context("atomic")
    context.add_user_message("original")
    tracer = SimpleNamespace(
        load_latest_checkpoint=AsyncMock(return_value=None), checkpoint=AsyncMock()
    )
    runner = AgentRunner(
        SimpleNamespace(llm=None), tracer=tracer, context_manager=manager
    )
    yield runner, context, tracer
    manager._contexts.clear()


@pytest.mark.asyncio
async def test_candidate_is_invisible_until_persisted(live):
    runner, context, tracer = live
    entered, release = asyncio.Event(), asyncio.Event()

    async def write(**payload):
        assert [m["content"] for m in payload["context"]["messages"]] == [
            "original",
            "new",
        ]
        entered.set()
        await release.wait()

    tracer.checkpoint.side_effect = write
    task = asyncio.create_task(
        runner.inject_user_message("atomic", "new", turn_id="turn")
    )
    await entered.wait()
    try:
        assert [m.content for m in context.messages] == ["original"]
        assert "_pending_user_message_trace_turn_id" not in context.metadata
    finally:
        release.set()
        await task
    assert [m.content for m in context.messages] == ["original", "new"]


@pytest.mark.asyncio
async def test_confirmed_failed_write_leaves_no_ghost(live):
    runner, context, tracer = live
    tracer.checkpoint.side_effect = RuntimeError("write failed")
    with pytest.raises(RuntimeError, match="write failed"):
        await runner.inject_user_message("atomic", "new", turn_id="turn")
    assert [m.content for m in context.messages] == ["original"]
    tracer.checkpoint.side_effect = None
    result = await runner.inject_user_message("atomic", "new", turn_id="turn")
    assert result.outcome is UserMessageInjectionOutcome.POSTED_FRESH
    assert tracer.checkpoint.await_count == 2


@pytest.mark.asyncio
async def test_lost_ack_readback_confirms_message(live):
    runner, context, tracer = live

    async def write(**payload):
        tracer.load_latest_checkpoint.return_value = payload
        raise RuntimeError("lost acknowledgement")

    tracer.checkpoint.side_effect = write
    result = await runner.inject_user_message("atomic", "new", turn_id="turn")
    assert result.outcome is UserMessageInjectionOutcome.POSTED_FRESH
    assert [m.content for m in context.messages] == ["original", "new"]


@pytest.mark.asyncio
async def test_uncertain_write_blocks_stale_checkpoint(live):
    runner, context, tracer = live

    async def write(**payload):
        tracer.load_latest_checkpoint.side_effect = RuntimeError("read unavailable")
        raise RuntimeError("write outcome unknown")

    tracer.checkpoint.side_effect = write
    result = await runner.inject_user_message("atomic", "new", turn_id="turn")
    assert result.outcome is UserMessageInjectionOutcome.OUTCOME_UNKNOWN
    assert [m.content for m in context.messages] == ["original"]
    runtime = PatternRuntime(execution_id="atomic", tracer=tracer)
    with pytest.raises(CheckpointPersistenceError):
        await runtime.checkpoint("late", context=context, pattern=SimpleNamespace())
    assert tracer.checkpoint.await_count == 1


@pytest.mark.asyncio
async def test_ordinary_checkpoint_waits_for_injection_publication(live):
    runner, context, tracer = live
    entered, release = asyncio.Event(), asyncio.Event()
    snapshots = []

    async def write(**payload):
        if payload["label"] == "user_message_injected":
            entered.set()
            await release.wait()
        snapshots.append(payload)

    tracer.checkpoint.side_effect = write
    injection = asyncio.create_task(
        runner.inject_user_message("atomic", "new", turn_id="turn")
    )
    await entered.wait()
    runtime = PatternRuntime(execution_id="atomic", tracer=tracer)
    ordinary = asyncio.create_task(
        runtime.checkpoint("ordinary", context=context, pattern=SimpleNamespace())
    )
    await asyncio.sleep(0)
    assert not ordinary.done()
    # The executing pattern can still append output while persistence awaits.
    context.add_assistant_message("concurrent output")
    release.set()
    await asyncio.gather(injection, ordinary)
    assert [m.content for m in context.messages] == [
        "original",
        "concurrent output",
        "new",
    ]
    assert snapshots[-1]["label"] == "ordinary"
    assert snapshots[-1]["context"]["messages"][-1]["content"] == "new"


@pytest.mark.asyncio
async def test_concurrent_duplicate_injection_writes_once(live):
    runner, context, tracer = live
    first, second = await asyncio.gather(
        runner.inject_user_message("atomic", "new", turn_id="turn"),
        runner.inject_user_message("atomic", "new", turn_id="turn"),
    )
    assert {first.outcome, second.outcome} == {
        UserMessageInjectionOutcome.POSTED_FRESH,
        UserMessageInjectionOutcome.POSTED_REPLAY,
    }
    assert tracer.checkpoint.await_count == 1
    assert len(context.messages) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel_at", ["write", "readback"])
async def test_cancelled_uncertain_injection_blocks_stale_writes(live, cancel_at):
    runner, context, tracer = live
    entered = asyncio.Event()

    async def block():
        entered.set()
        await asyncio.Event().wait()

    async def write(**payload):
        if cancel_at == "write":
            await block()
        else:
            tracer.load_latest_checkpoint.side_effect = lambda _: block()
            raise RuntimeError("uncertain")

    # AsyncMock's side_effect must itself await the suspended read.
    async def read(_):
        await block()

    async def fail_write(**payload):
        tracer.load_latest_checkpoint.side_effect = read
        raise RuntimeError("uncertain")

    tracer.checkpoint.side_effect = write if cancel_at == "write" else fail_write
    task = asyncio.create_task(
        runner.inject_user_message("atomic", "new", turn_id="turn")
    )
    await entered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert [m.content for m in context.messages] == ["original"]
    runtime = PatternRuntime(execution_id="atomic", tracer=tracer)
    with pytest.raises(CheckpointPersistenceError):
        await runtime.checkpoint("late", context=context, pattern=SimpleNamespace())


@pytest.mark.asyncio
async def test_concurrent_cold_injections_share_context(live):
    runner, context, tracer = live
    payload = {"context": context.to_dict()}
    runner.context_manager.remove_context("atomic")
    reads, both = 0, asyncio.Event()

    async def read(_):
        nonlocal reads
        reads += 1
        if reads == 2:
            both.set()
        await both.wait()
        return payload

    tracer.load_latest_checkpoint.side_effect = read
    a, b = await asyncio.gather(
        runner.inject_user_message("atomic", "one", turn_id="one"),
        runner.inject_user_message("atomic", "two", turn_id="two"),
    )
    assert a.context is b.context is runner.context_manager.get_context("atomic")
    assert {m.content for m in a.context.messages} == {"original", "one", "two"}
    assert {
        m["content"] for m in tracer.checkpoint.call_args.kwargs["context"]["messages"]
    } == {"original", "one", "two"}


@pytest.mark.asyncio
@pytest.mark.parametrize("landed", [False, True])
async def test_settlement_reloads_durable_state_before_retry(live, landed):
    runner, context, tracer = live
    durable = {"context": context.to_dict()}

    async def ambiguous_write(**payload):
        nonlocal durable
        if landed:
            durable = payload
        tracer.load_latest_checkpoint.side_effect = RuntimeError(
            "temporarily unavailable"
        )
        raise RuntimeError("lost response")

    tracer.checkpoint.side_effect = ambiguous_write
    result = await runner.inject_user_message("atomic", "new", turn_id="turn")
    assert result.outcome is UserMessageInjectionOutcome.OUTCOME_UNKNOWN
    tracer.load_latest_checkpoint.side_effect = None
    tracer.load_latest_checkpoint.return_value = durable
    tracer.checkpoint.side_effect = None
    settled = await runner.settle_injection_against_checkpoint(
        "atomic",
        execution_message="new",
        display_message="new",
        turn_id="turn",
    )
    assert settled.outcome is (
        UserMessageInjectionOutcome.POSTED_REPLAY
        if landed
        else UserMessageInjectionOutcome.POSTED_FRESH
    )
    assert [m.content for m in settled.context.messages] == ["original", "new"]
    assert settled.context is not context
    assert tracer.checkpoint.await_count == (1 if landed else 2)


@pytest.mark.asyncio
async def test_cancel_before_write_does_not_poison_context(live):
    from xagent.core.agent.context.execution import context_checkpoint_gate

    runner, context, tracer = live
    gate = context_checkpoint_gate(context)
    async with gate.shared():
        task = asyncio.create_task(
            runner.inject_user_message("atomic", "new", turn_id="turn")
        )
        await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    assert not gate.injection_uncertain
    tracer.checkpoint.assert_not_awaited()
    result = await runner.inject_user_message("atomic", "new", turn_id="turn")
    assert result.outcome is UserMessageInjectionOutcome.POSTED_FRESH


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "payload", ["malformed", {"context": {}}, {"context": {"messages": ["bad"]}}]
)
async def test_malformed_readback_cannot_prove_absence(live, payload):
    runner, context, tracer = live

    async def write(**_):
        tracer.load_latest_checkpoint.return_value = payload
        raise RuntimeError("unknown write")

    tracer.checkpoint.side_effect = write
    result = await runner.inject_user_message("atomic", "new", turn_id="turn")
    assert result.outcome is UserMessageInjectionOutcome.OUTCOME_UNKNOWN
    assert [m.content for m in context.messages] == ["original"]


@pytest.mark.asyncio
async def test_settlement_refuses_active_execution(live):
    from xagent.core.agent.runner import ExecutionControl

    runner, context, tracer = live
    runner._active_controls["atomic"] = ExecutionControl(
        PatternRuntime(execution_id="atomic"), "original"
    )
    with pytest.raises(RuntimeError, match="execution is active"):
        await runner.settle_injection_against_checkpoint(
            "atomic",
            execution_message="new",
            display_message="new",
            turn_id="turn",
        )
    tracer.load_latest_checkpoint.assert_not_awaited()
    tracer.checkpoint.assert_not_awaited()
    assert runner.context_manager.get_context("atomic") is context
