"""Tests for the event-driven Langfuse trace handler."""

from __future__ import annotations

from typing import Any

import pytest

from tests.utils.mock_helpers import create_langfuse_mock
from xagent.core.agent.trace import (
    TraceCategory,
    Tracer,
    trace_action_end,
    trace_action_start,
    trace_task_completion,
    trace_task_start,
)
from xagent.core.tracing.langfuse import create_langfuse_trace_handler
from xagent.core.tracing.langfuse.client import get_langfuse_client


def _make_observation(mocker, trace_id: str, span_id: str) -> Any:
    observation = mocker.Mock()
    observation.trace_id = trace_id
    observation.id = span_id
    observation._otel_span = mocker.Mock()
    return observation


@pytest.mark.asyncio
async def test_langfuse_handler_records_task_and_tool_flow(
    mocker, monkeypatch, langfuse_client_reset
):
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "test-public")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "test-secret")
    monkeypatch.setenv("LANGFUSE_HOST", "https://langfuse.example")

    _, mock_langfuse = create_langfuse_mock(mocker)
    root = _make_observation(mocker, "trace-1", "root-1")
    task_event_observation = _make_observation(mocker, "trace-1", "event-1")
    tool_observation = _make_observation(mocker, "trace-1", "tool-1")
    mock_langfuse.start_observation.side_effect = [
        root,
        task_event_observation,
        tool_observation,
    ]

    handler = create_langfuse_trace_handler(
        task_id="task-1",
        user_id=7,
        trace_name="trace-name",
        session_id="session-1",
        tags=["xagent", "test"],
        metadata={"origin": "unit-test"},
    )
    assert handler is not None
    assert get_langfuse_client() is mock_langfuse

    tracer = Tracer()
    tracer.add_handler(handler)

    await trace_task_start(
        tracer,
        "task-1",
        TraceCategory.REACT,
        data={"message": "solve task"},
    )
    await trace_action_start(
        tracer,
        "task-1",
        "step-1",
        TraceCategory.TOOL,
        data={"tool_name": "calculator", "tool_args": {"expression": "1+1"}},
    )
    await trace_action_end(
        tracer,
        "task-1",
        "step-1",
        TraceCategory.TOOL,
        data={"tool_name": "calculator", "result": "2", "success": True},
    )
    await trace_task_completion(
        tracer,
        "task-1",
        {"answer": "2"},
        success=True,
    )

    assert mock_langfuse.start_observation.call_count == 3
    root.update_trace.assert_called()
    tool_observation.update.assert_called_once()
    tool_observation.end.assert_called_once()
    task_event_observation.update.assert_not_called()
    root.update.assert_called()
    root.end.assert_called_once()


@pytest.mark.asyncio
async def test_langfuse_handler_disabled_without_env(mocker, langfuse_client_reset):
    mocker.patch("xagent.core.tracing.langfuse.client.Langfuse")
    assert create_langfuse_trace_handler(task_id="task-2") is None
