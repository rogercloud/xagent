"""A retried turn stays a replay after compaction dropped its message.

LLM compaction keeps only the summary and the latest user message, so a
turn-id de-duplication that scans ``context.messages`` alone forgets every
older turn. These tests pin the metadata record that survives compaction.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from tests.core.agent.test_runner import (
    CheckpointingPattern,
    FakeWorkspaceManager,
    TracerCheckpointStore,
)
from xagent.core.agent import Agent, ContextManager, ExecutionContext
from xagent.core.agent.runner import (
    ACCEPTED_TURN_IDS_METADATA_KEY,
    MAX_ACCEPTED_TURN_IDS,
    AgentRunner,
    UserMessageInjectionConflictError,
    UserMessageInjectionOutcome,
    UserMessageInjectionRejectedError,
)
from xagent.core.agent.tracing import TRACE_TURN_IDS_KEY


def _runner(tracer: TracerCheckpointStore, tmp_path: Path) -> AgentRunner:
    return AgentRunner(
        agent=Agent(name="writer", patterns=[CheckpointingPattern({"success": True})]),
        tracer=tracer,
        workspace_manager=FakeWorkspaceManager(tmp_path),
    )


def _compact_stored_checkpoint(
    tracer: TracerCheckpointStore, execution_id: str
) -> ExecutionContext:
    """Compact the durable context as a pattern's LLM compaction would."""
    payload = tracer.by_execution_id[execution_id]
    context = ExecutionContext.from_dict(payload["context"])
    result = context.compact_with_llm_response("Summary of the earlier turns.")
    assert result.compacted
    payload["context"] = context.to_dict()
    return context


def _user_turn_ids(context: ExecutionContext) -> list[str | None]:
    return [
        message.metadata.get("turn_id")
        for message in context.messages
        if message.role == "user"
    ]


async def _inject_then_compact_past(
    runner: AgentRunner, tracer: TracerCheckpointStore, execution_id: str
) -> None:
    """Accept turn-t, accept a later turn, then compact turn-t out of messages."""
    await runner.run(task="Original task", execution_id=execution_id)
    first = await runner.inject_user_message(
        execution_id, "Choose B", turn_id="turn-t", request_interrupt=False
    )
    assert first.outcome is UserMessageInjectionOutcome.POSTED_FRESH
    later = await runner.inject_user_message(
        execution_id, "And then C", turn_id="turn-u", request_interrupt=False
    )
    assert later.outcome is UserMessageInjectionOutcome.POSTED_FRESH
    compacted = _compact_stored_checkpoint(tracer, execution_id)
    assert _user_turn_ids(compacted) == ["turn-u"]


@pytest.mark.asyncio
async def test_retry_of_compacted_turn_is_a_replay(tmp_path: Path) -> None:
    tracer = TracerCheckpointStore()
    execution_id = "exec-compacted-retry"
    runner = _runner(tracer, tmp_path)
    await _inject_then_compact_past(runner, tracer, execution_id)
    before = tracer.by_execution_id[execution_id]["context"]

    runner.pause = MagicMock(return_value=True)
    retry = await runner.inject_user_message(
        execution_id, "Choose B", turn_id="turn-t", request_interrupt=False
    )

    assert retry.outcome is UserMessageInjectionOutcome.POSTED_REPLAY
    # Nothing was persisted and the turn is not re-added to the context.
    assert tracer.by_execution_id[execution_id]["context"] is before
    assert retry.context is not None
    assert _user_turn_ids(retry.context) == ["turn-u"]
    runner.pause.assert_not_called()


@pytest.mark.asyncio
async def test_retry_of_compacted_turn_with_other_content_conflicts(
    tmp_path: Path,
) -> None:
    tracer = TracerCheckpointStore()
    execution_id = "exec-compacted-conflict"
    runner = _runner(tracer, tmp_path)
    await _inject_then_compact_past(runner, tracer, execution_id)

    with pytest.raises(UserMessageInjectionConflictError):
        await runner.inject_user_message(
            execution_id, "Choose Z", turn_id="turn-t", request_interrupt=False
        )


@pytest.mark.asyncio
async def test_new_turn_after_compaction_still_posts(tmp_path: Path) -> None:
    tracer = TracerCheckpointStore()
    execution_id = "exec-compacted-new-turn"
    runner = _runner(tracer, tmp_path)
    await _inject_then_compact_past(runner, tracer, execution_id)

    fresh = await runner.inject_user_message(
        execution_id, "Choose B", turn_id="turn-v", request_interrupt=False
    )

    assert fresh.outcome is UserMessageInjectionOutcome.POSTED_FRESH
    stored = ExecutionContext.from_dict(tracer.by_execution_id[execution_id]["context"])
    assert _user_turn_ids(stored) == ["turn-u", "turn-v"]
    assert list(stored.metadata[ACCEPTED_TURN_IDS_METADATA_KEY])[-3:] == [
        "turn-t",
        "turn-u",
        "turn-v",
    ]


@pytest.mark.asyncio
async def test_initial_run_turn_is_recognised_after_compaction(
    tmp_path: Path,
) -> None:
    tracer = TracerCheckpointStore()
    execution_id = "exec-compacted-initial"
    runner = _runner(tracer, tmp_path)
    await runner.run(
        task="Original task",
        execution_id=execution_id,
        metadata={"turn_id": "turn-start"},
    )
    await runner.inject_user_message(
        execution_id, "And then C", turn_id="turn-u", request_interrupt=False
    )
    _compact_stored_checkpoint(tracer, execution_id)

    retry = await runner.inject_user_message(
        execution_id, "Original task", turn_id="turn-start", request_interrupt=False
    )

    assert retry.outcome is UserMessageInjectionOutcome.POSTED_REPLAY


@pytest.mark.asyncio
async def test_live_context_compacted_in_memory_replays(tmp_path: Path) -> None:
    """The record on a cached live context survives in-memory compaction."""
    tracer = TracerCheckpointStore()
    execution_id = "exec-live-compaction"
    runner = _runner(tracer, tmp_path)
    await runner.run(task="Original task", execution_id=execution_id)
    live = ExecutionContext.from_dict(tracer.by_execution_id[execution_id]["context"])
    runner.context_manager.set_context(live)

    await runner.inject_user_message(
        execution_id, "Choose B", turn_id="turn-t", request_interrupt=False
    )
    await runner.inject_user_message(
        execution_id, "And then C", turn_id="turn-u", request_interrupt=False
    )
    assert runner.context_manager.get_context(execution_id) is live
    live.compact_with_llm_response("Summary of the earlier turns.")
    assert _user_turn_ids(live) == ["turn-u"]

    runner.pause = MagicMock(return_value=True)
    retry = await runner.inject_user_message(
        execution_id, "Choose B", turn_id="turn-t", request_interrupt=True
    )

    assert retry.outcome is UserMessageInjectionOutcome.POSTED_REPLAY
    assert retry.context is live
    assert _user_turn_ids(live) == ["turn-u"]
    # A replay still honours the caller's interrupt request, as before.
    runner.pause.assert_called_once_with(execution_id, reason="new user message")


def test_record_survives_checkpoint_round_trip_and_compaction() -> None:
    context = ExecutionContext(execution_id="exec-round-trip")
    for index in range(3):
        message = context.add_user_message(
            f"turn {index}", metadata={"turn_id": f"turn-{index}"}
        )
        AgentRunner._record_accepted_turn(context, message)

    restored = ExecutionContext.from_dict(context.to_dict())
    restored.compact_with_llm_response("Summary.")
    restored = ExecutionContext.from_dict(restored.to_dict())

    assert _user_turn_ids(restored) == ["turn-2"]
    for index in range(3):
        assert AgentRunner._accepted_turn_digest(
            restored, f"turn-{index}"
        ) == AgentRunner._turn_content_digest(f"turn {index}")
    assert AgentRunner._accepted_turn_digest(restored, "turn-9") is None


def test_record_is_bounded_to_most_recent_turns() -> None:
    context = ExecutionContext(execution_id="exec-bounded")
    for index in range(MAX_ACCEPTED_TURN_IDS + 5):
        message = context.add_user_message("x", metadata={"turn_id": f"t{index}"})
        AgentRunner._record_accepted_turn(context, message)

    accepted = context.metadata[ACCEPTED_TURN_IDS_METADATA_KEY]
    assert len(accepted) == MAX_ACCEPTED_TURN_IDS
    assert "t4" not in accepted
    assert "t5" in accepted
    assert list(accepted)[-1] == f"t{MAX_ACCEPTED_TURN_IDS + 4}"


@pytest.mark.asyncio
async def test_legacy_checkpoint_falls_back_to_traced_turn_ids(
    tmp_path: Path,
) -> None:
    """A checkpoint from before the accepted record keeps its traced ids."""
    tracer = TracerCheckpointStore()
    execution_id = "exec-legacy-traced"
    legacy = ExecutionContext(execution_id=execution_id)
    legacy.add_user_message("Summary.", metadata={"turn_id": "turn-u"})
    legacy.metadata[TRACE_TURN_IDS_KEY] = ["turn-t", "turn-u"]
    await tracer.checkpoint(
        type="checkpoint",
        execution_id=execution_id,
        label="waiting_for_user",
        context=legacy.to_dict(),
    )
    runner = _runner(tracer, tmp_path)

    retry = await runner.inject_user_message(
        execution_id, "Choose B", turn_id="turn-t", request_interrupt=False
    )

    assert retry.outcome is UserMessageInjectionOutcome.POSTED_REPLAY


@pytest.mark.asyncio
async def test_caller_metadata_cannot_seed_accepted_record(tmp_path: Path) -> None:
    tracer = TracerCheckpointStore()
    execution_id = "exec-forged-record"
    runner = _runner(tracer, tmp_path)
    await runner.run(
        task="Original task",
        execution_id=execution_id,
        metadata={
            ACCEPTED_TURN_IDS_METADATA_KEY: {"turn-forged": ""},
            "request_context": {ACCEPTED_TURN_IDS_METADATA_KEY: {"turn-forged": ""}},
        },
    )

    result = await runner.inject_user_message(
        execution_id, "Choose B", turn_id="turn-forged", request_interrupt=False
    )

    assert result.outcome is UserMessageInjectionOutcome.POSTED_FRESH


@pytest.fixture
def atomic_live():
    manager = ContextManager()
    context = manager.create_context("atomic")
    context.add_user_message("original")
    tracer = SimpleNamespace(
        load_latest_checkpoint=AsyncMock(return_value=None), checkpoint=AsyncMock()
    )
    runner = AgentRunner(
        SimpleNamespace(llm=None), tracer=tracer, context_manager=manager
    )
    yield runner, context, tracer


@pytest.mark.asyncio
async def test_record_is_written_in_the_accepting_checkpoint(atomic_live) -> None:
    runner, context, tracer = atomic_live
    entered, release = asyncio.Event(), asyncio.Event()
    persisted: list[dict[str, Any]] = []

    async def write(**payload: Any) -> None:
        persisted.append(payload)
        entered.set()
        await release.wait()

    tracer.checkpoint.side_effect = write
    task = asyncio.create_task(
        runner.inject_user_message("atomic", "new", turn_id="turn")
    )
    await entered.wait()
    try:
        # The accepting checkpoint carries message and record together ...
        stored = persisted[0]["context"]
        assert stored["messages"][-1]["metadata"]["turn_id"] == "turn"
        assert "turn" in stored["metadata"][ACCEPTED_TURN_IDS_METADATA_KEY]
        # ... while the live context sees neither until it is durable.
        assert ACCEPTED_TURN_IDS_METADATA_KEY not in context.metadata
    finally:
        release.set()
        await task
    assert "turn" in context.metadata[ACCEPTED_TURN_IDS_METADATA_KEY]


@pytest.mark.asyncio
async def test_rejected_write_leaves_no_record(atomic_live) -> None:
    runner, context, tracer = atomic_live
    tracer.checkpoint.side_effect = RuntimeError("write failed")

    with pytest.raises(UserMessageInjectionRejectedError):
        await runner.inject_user_message("atomic", "new", turn_id="turn")

    assert ACCEPTED_TURN_IDS_METADATA_KEY not in context.metadata
    tracer.checkpoint.side_effect = None
    result = await runner.inject_user_message("atomic", "new", turn_id="turn")
    assert result.outcome is UserMessageInjectionOutcome.POSTED_FRESH
