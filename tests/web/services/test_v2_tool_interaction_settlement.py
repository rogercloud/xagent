from __future__ import annotations

import pytest

from tests.core.agent.test_react import FakeLLM, SettlementApprovalTool
from tests.web.services.test_task_execution_event_writer import (
    canonical as canonical_fixture,
)
from tests.web.services.test_task_execution_event_writer import (
    engine as engine_fixture,
)
from tests.web.services.test_task_execution_event_writer import facts
from tests.web.services.test_task_execution_event_writer import (
    task_id as task_id_fixture,
)
from xagent.core.agent import ExecutionContext, ReActPattern
from xagent.core.agent.checkpoint import TraceCheckpointStore
from xagent.core.agent.runtime import PatternRuntime
from xagent.core.tools.user_interaction import ToolInteractionSettlement
from xagent.web.services.task_event_context_service import load_task_event_context
from xagent.web.services.task_event_display import load_event_display_snapshot
from xagent.web.tracing import ExecutionEventTraceAdapter, create_task_tracer

canonical = canonical_fixture
engine = engine_fixture
task_id = task_id_fixture


def _tool_call(call_id: str, name: str, arguments: str) -> dict:
    return {"id": call_id, "function": {"name": name, "arguments": arguments}}


@pytest.mark.asyncio
async def test_resumed_tool_interaction_settlement_is_readable_on_next_turn(
    canonical,
):
    """A V2 task must read its history after a tool interaction settles."""

    factory, task_id = canonical
    tracer = create_task_tracer(task_id)
    tracer.handlers = [
        h for h in tracer.handlers if isinstance(h, ExecutionEventTraceAdapter)
    ]
    runtime = PatternRuntime(tracer=TraceCheckpointStore(tracer))
    context = ExecutionContext(system_prompt="Publish")
    context.add_user_message("Publish the post.", metadata={"turn_id": "turn-1"})
    pattern = ReActPattern(max_iterations=3)
    waiting = await pattern.run(
        context=context,
        tools=[SettlementApprovalTool()],
        llm=FakeLLM(
            [
                {
                    "tool_calls": [
                        _tool_call(
                            "publish-call", "approval_gate", '{"expression":"go"}'
                        )
                    ]
                }
            ]
        ),
        runtime=runtime,
    )
    assert waiting["status"] == "waiting_for_user"

    context.add_user_message("Approve", metadata={"turn_id": "turn-2"})
    resumed = await pattern.run(
        context=context,
        tools=[
            SettlementApprovalTool(
                resume_result=ToolInteractionSettlement.succeeded(
                    {"success": True, "post_urn": "urn:li:share:123"}
                )
            )
        ],
        llm=FakeLLM(
            [
                {
                    "tool_calls": [
                        _tool_call(
                            "final-call",
                            "final_answer",
                            '{"response_language":"English",'
                            '"answer":"Published.","outcome":"completed"}',
                        )
                    ]
                }
            ]
        ),
        runtime=runtime,
    )
    assert resumed["success"] is True

    with factory() as db:
        settlement = [
            row
            for row in facts(db, task_id)
            if row.kind.startswith("tool_execution_")
            and row.payload["data"].get("settlement_delivery") is True
        ]
        # The settlement pair is committed under the original attempt.
        assert [row.kind for row in settlement] == [
            "tool_execution_start",
            "tool_execution_end",
        ]
        original = next(
            row
            for row in facts(db, task_id)
            if row.kind == "tool_execution_start" and row not in settlement
        )
        assert {row.tool_attempt_id for row in settlement} == {
            original.tool_attempt_id
        }
        assert {row.assistant_message_id for row in settlement} == {
            original.assistant_message_id
        }
        # The next turn sees one call whose observation is the settled result.
        loaded = load_task_event_context(db, task_id)
        calls = [m for m in loaded.messages if m.get("tool_calls")]
        assert [c["id"] for m in calls for c in m["tool_calls"]] == ["publish-call"]
        [observation] = [m for m in loaded.messages if m["role"] == "tool"]
        assert observation["raw_result"]["post_urn"] == "urn:li:share:123"
        load_event_display_snapshot(db, task_id)


@pytest.mark.asyncio
async def test_settlement_trace_replay_is_idempotent(canonical):
    """A settlement trace replayed after its facts committed adds none."""

    factory, task_id = canonical
    tracer = create_task_tracer(task_id)
    tracer.handlers = [
        h for h in tracer.handlers if isinstance(h, ExecutionEventTraceAdapter)
    ]
    runtime = PatternRuntime(tracer=TraceCheckpointStore(tracer))
    pattern = ReActPattern()
    [call] = pattern._normalize_tool_calls(
        [{"id": "gate-call", "function": {"name": "approval_gate", "arguments": "{}"}}]
    )
    call.update(tool_attempt_id="attempt-1", assistant_message_id="batch-1")
    await runtime.on_tool_start(tool_call=call)
    await runtime.on_tool_end(
        tool_call=call, result={"success": False, "status": "waiting_for_user"}
    )
    pattern._record_tool_call(
        call,
        status="completed",
        result={"success": True},
        settlement_status="succeeded",
        settlement_trace_pending=True,
    )
    [record] = pattern.tool_ledger.values()
    for _ in range(2):
        await pattern._trace_tool_interaction_settlement(
            record=record,
            status="succeeded",
            result={"success": True},
            error=None,
            runtime=runtime,
        )
    with factory() as db:
        rows = facts(db, task_id)
    assert [row.kind for row in rows] == [
        "tool_execution_start",
        "tool_execution_end",
        "tool_execution_start",
        "tool_execution_end",
    ]
    assert {row.tool_attempt_id for row in rows} == {"attempt-1"}
    assert not pattern.tool_ledger[next(iter(pattern.tool_ledger))].settlement_trace_pending
