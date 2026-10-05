"""V2 settlement after an authoritative execution-event commit failure."""

from __future__ import annotations

from datetime import timedelta
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session

from tests.core.agent.test_react import FakeLLM, FakeTool
from tests.web.services.test_execution_event_recovery import tracer_for
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
    CheckpointUnavailableError,
    TraceCheckpointStore,
)
from xagent.web.models.task import Task, TaskStatus
from xagent.web.services import task_execution_event_recovery as event_recovery
from xagent.web.services import task_orchestrator as orchestrator
from xagent.web.services.managed_task_lease import start_managed_task_lease
from xagent.web.services.task_event_trace_handler import get_event_type_mapping
from xagent.web.services.task_lease_recovery import (
    TASK_UNKNOWN_TOOL_EFFECT_ERROR as LEASE_EXPIRED_UNKNOWN_EFFECT_ERROR,
)
from xagent.web.services.task_lease_recovery import (
    recover_task_lease_candidate_no_commit,
)
from xagent.web.services.task_lease_service import (
    TASK_UNKNOWN_TOOL_EFFECT_SETTLEMENT_ERROR,
    acquire_task_lease,
    bind_task_lease_context,
    get_expired_task_lease_candidates,
    utc_now,
)
from xagent.web.services.task_orchestrator import (
    TaskTurnPayload,
    _schedule_bg,
    settle_task_lease_isolated,
)
from xagent.web.tracing import ExecutionEventTraceAdapter

canonical = canonical_fixture
engine = engine_fixture
task_id = task_id_fixture


def _tool_turn_llm() -> FakeLLM:
    return FakeLLM(
        responses=[
            {
                "content": "calculate",
                "tool_calls": [
                    {
                        "id": "call1",
                        "function": {
                            "name": "calculator",
                            "arguments": '{"expression":"2+2"}',
                        },
                    }
                ],
            },
            {"content": "4", "done": True},
        ]
    )


def _fail_once(monkeypatch, kind: str) -> list[str]:
    """Fail the first commit of ``kind`` without writing it; later ones pass."""

    original = ExecutionEventTraceAdapter._save_to_database
    failed: list[str] = []

    async def save(self, event):
        if not failed and get_event_type_mapping(event) == kind:
            failed.append(kind)
            raise OperationalError("COMMIT", {}, Exception("connection reset"))
        return await original(self, event)

    monkeypatch.setattr(ExecutionEventTraceAdapter, "_save_to_database", save)
    return failed


def _react_execution(tid: int, tool: FakeTool):
    """A real ReAct turn writing V2 facts through the strict event writer."""

    async def execute(**kwargs):
        lease = kwargs.get("task_lease") or kwargs["lease"]
        with bind_task_lease_context(lease):
            runtime = PatternRuntime(tracer=TraceCheckpointStore(tracer_for(tid)))
            context = ExecutionContext(execution_id=str(tid), system_prompt="Calc")
            context.add_user_message("2+2", metadata={"turn_id": "turn"})
            await ReActPattern(max_iterations=3).run(
                context=context,
                tools=[tool],
                runtime=runtime,
                llm=_tool_turn_llm(),
            )

    return execute


async def _run_scheduled_turn(
    tid: int, lease, execute, *, channel: bool = False
) -> list[dict[str, Any]]:
    published: list[dict[str, Any]] = []

    async def publish(event, _task_id):
        published.append(event)

    with (
        patch.object(orchestrator, "run_task_lease_heartbeat", new=AsyncMock()),
        patch.object(
            orchestrator, "load_task_setup_snapshot_sync", return_value=MagicMock()
        ),
        patch.object(
            orchestrator, "resolve_execution_scope", return_value=None, create=True
        ),
        patch(
            "xagent.web.services.task_execution.execute_task_background",
            new=execute,
        ),
        patch(
            "xagent.web.services.shared_channel_execution.execute_channel_background",
            new=execute,
        ),
        patch(
            "xagent.web.services.task_event_display.publish_task_result", new=publish
        ),
        patch.object(orchestrator, "_get_agent_manager", return_value=MagicMock()),
    ):
        await _schedule_bg(
            task_id=tid,
            task_owner_user_id=1,
            task_source="sdk",
            task_lease=lease,
            payload=TaskTurnPayload("2+2"),
            force_fresh=False,
            context=None,
            channel_command=MagicMock() if channel else None,
        )
    return published


def _assert_unknown_effect_settlement(factory, tid: int, lease, error: str) -> None:
    with factory() as db:
        task = db.get(Task, tid)
        assert task.status == TaskStatus.FAILED
        assert task.runner_id is None
        assert task.error_message == error
        rows = facts(db, tid)
        kinds = [row.kind for row in rows]
        assert kinds.count("tool_execution_start") == 1
        assert "tool_execution_end" not in kinds
        settled = [row for row in rows if row.kind == "execution_settled"]
        assert len(settled) == 1
        assert settled[0].run_id == lease.run_id
        assert settled[0].payload == {"status": "failed", "result": {"error": error}}


@pytest.mark.asyncio
@pytest.mark.parametrize("channel", [False, True], ids=["web", "channel"])
async def test_lost_tool_result_settles_as_unknown_effect(
    canonical, monkeypatch, channel
):
    factory, tid = canonical
    with factory() as db:
        lease = acquire_task_lease(db, tid, new_run=True)
    failed = _fail_once(monkeypatch, "tool_execution_end")
    tool = FakeTool()

    published = await _run_scheduled_turn(
        tid, lease, _react_execution(tid, tool), channel=channel
    )

    # The tool ran once, its result commit failed, and the DB is fine again.
    assert failed == ["tool_execution_end"]
    assert len(tool.calls) == 1
    _assert_unknown_effect_settlement(
        factory, tid, lease, TASK_UNKNOWN_TOOL_EFFECT_SETTLEMENT_ERROR
    )
    # One client-safe terminal event; no success, no operator detail.
    assert [event["type"] for event in published] == ["task_error"]
    assert "Conversation event commit failed" not in str(published)


@pytest.mark.asyncio
async def test_unstarted_tool_keeps_generic_persistence_failure(canonical, monkeypatch):
    """No started attempt means no unknown effect: the tool never ran."""

    factory, tid = canonical
    with factory() as db:
        lease = acquire_task_lease(db, tid, new_run=True)
    failed = _fail_once(monkeypatch, "tool_execution_start")
    tool = FakeTool()

    await _run_scheduled_turn(tid, lease, _react_execution(tid, tool))

    assert failed == ["tool_execution_start"]
    assert tool.calls == []
    with factory() as db:
        task = db.get(Task, tid)
        assert task.status == TaskStatus.FAILED
        assert task.error_message.startswith(
            "setup/run error: ExecutionEventPersistenceError"
        )


@pytest.mark.asyncio
async def test_unreadable_classification_defers_to_lease_recovery(
    canonical, monkeypatch
):
    """Settlement that cannot classify keeps the lease; recovery classifies."""

    factory, tid = canonical
    with factory() as db:
        lease = acquire_task_lease(db, tid, new_run=True)
    _fail_once(monkeypatch, "tool_execution_end")
    tool = FakeTool()
    original_read = event_recovery.read_event_checkpoint

    def unavailable(*args, **kwargs):
        raise CheckpointUnavailableError("recovery read failed")

    monkeypatch.setattr(event_recovery, "read_event_checkpoint", unavailable)
    published = await _run_scheduled_turn(tid, lease, _react_execution(tid, tool))

    assert published == []
    with factory() as db:
        task = db.get(Task, tid)
        assert task.status == TaskStatus.RUNNING
        assert task.runner_id == lease.runner_id
        assert not [row for row in facts(db, tid) if row.kind == "execution_settled"]
        task.lease_expires_at = utc_now() - timedelta(seconds=5)
        db.commit()

    monkeypatch.setattr(event_recovery, "read_event_checkpoint", original_read)
    with factory() as db:
        candidate = get_expired_task_lease_candidates(db, cutoff=utc_now(), limit=1)[0]
        assert (
            recover_task_lease_candidate_no_commit(
                db, candidate, recovered_at=utc_now()
            )
            == TaskStatus.FAILED
        )
        db.commit()
    assert len(tool.calls) == 1
    _assert_unknown_effect_settlement(
        factory, tid, lease, LEASE_EXPIRED_UNKNOWN_EFFECT_ERROR
    )


@pytest.mark.asyncio
async def test_managed_lease_failure_settles_as_unknown_effect(canonical, monkeypatch):
    """Inline channel transports settle a raised run through the managed lease."""

    factory, tid = canonical
    with factory() as db:
        lease = acquire_task_lease(db, tid, new_run=True)
    _fail_once(monkeypatch, "tool_execution_end")
    tool = FakeTool()
    with patch(
        "xagent.web.services.managed_task_lease.run_task_lease_heartbeat",
        new=AsyncMock(),
    ):
        managed = start_managed_task_lease(lease)
        # The channel bots' error path: the run raised, no result to project.
        with pytest.raises(Exception) as caught:
            await _react_execution(tid, tool)(task_lease=lease)
        assert await managed.finalize_result(
            status=TaskStatus.FAILED, error_message=str(caught.value)
        )
    assert len(tool.calls) == 1
    _assert_unknown_effect_settlement(
        factory, tid, lease, TASK_UNKNOWN_TOOL_EFFECT_SETTLEMENT_ERROR
    )


def test_legacy_task_settlement_is_not_classified(canonical):
    factory, tid = canonical
    with factory() as db:
        db.get(Task, tid).conversation_storage_version = 1
        db.commit()
        lease = acquire_task_lease(db, tid, new_run=True)
    assert settle_task_lease_isolated(
        lease,
        error_message="setup/run error: RuntimeError: boom",
        classify_unknown_tool_effect=True,
    )
    with factory() as db:
        task = db.get(Task, tid)
        assert task.status == TaskStatus.FAILED
        assert task.error_message == "setup/run error: RuntimeError: boom"


async def _run_turn_with_broken_finalize_commit(
    canonical, monkeypatch, outcome: dict[str, Any], *, applied: bool
) -> tuple[Any, list[dict[str, Any]]]:
    """Run one real turn whose result COMMIT raises once.

    ``applied`` decides whether the server applied that COMMIT before the
    driver raised (a lost acknowledgement) or rolled it back.
    """

    from xagent.web.models import database
    from xagent.web.services import task_execution

    factory, tid = canonical
    monkeypatch.setattr(database, "_SessionLocal", factory)
    with factory() as db:
        lease = acquire_task_lease(db, tid, new_run=True)

    armed: list[bool] = []
    broken: list[str] = []
    original_commit = Session.commit

    def broken_commit(self):
        if not (armed and armed[0]):
            return original_commit(self)
        armed[0] = False
        broken.append("finalize")
        if applied:
            original_commit(self)
        raise OperationalError("COMMIT", {}, Exception("connection lost"))

    original_finalize = task_execution._finalize_task_execution_result_isolated

    def finalize(**kwargs):
        armed[:] = [True]
        try:
            return original_finalize(**kwargs)
        finally:
            armed.clear()

    monkeypatch.setattr(Session, "commit", broken_commit)
    monkeypatch.setattr(
        task_execution, "_finalize_task_execution_result_isolated", finalize
    )
    manager = MagicMock()
    manager.get_agent_for_task = AsyncMock(return_value=MagicMock())
    manager.execute_task = AsyncMock(return_value=dict(outcome))
    published: list[dict[str, Any]] = []

    async def publish(event, _task_id):
        published.append(event)

    with (
        patch.object(orchestrator, "run_task_lease_heartbeat", new=AsyncMock()),
        patch.object(
            orchestrator, "resolve_execution_scope", return_value=None, create=True
        ),
        patch.object(orchestrator, "_get_agent_manager", return_value=manager),
        patch("xagent.web.services.task_events.publish_task_event", new=publish),
        patch.object(task_execution, "publish_task_event", new=publish),
    ):
        await _schedule_bg(
            task_id=tid,
            task_owner_user_id=1,
            task_source="sdk",
            task_lease=lease,
            payload=TaskTurnPayload("2+2"),
            force_fresh=False,
            context=None,
        )

    assert broken == ["finalize"]
    manager.execute_task.assert_awaited_once()
    return lease, published


def _terminal_events(published: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        event
        for event in published
        if event.get("type") in {"task_completed", "task_error"}
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "outcome",
    [
        {"success": True, "output": "4"},
        {"success": False, "output": "", "error": "model refused"},
    ],
    ids=["completed", "failed"],
)
async def test_finalize_lost_ack_publishes_committed_terminal_once(
    canonical, monkeypatch, outcome
):
    """COMMIT reached the server but the driver raised: announce what committed."""

    factory, tid = canonical
    lease, published = await _run_turn_with_broken_finalize_commit(
        canonical, monkeypatch, outcome, applied=True
    )

    expected = TaskStatus.COMPLETED if outcome["success"] else TaskStatus.FAILED
    with factory() as db:
        task = db.get(Task, tid)
        assert task.status == expected
        assert task.runner_id is None
        settled = [row for row in facts(db, tid) if row.kind == "execution_settled"]
        assert len(settled) == 1
        state_version = int(task.state_version)
    terminal = _terminal_events(published)
    assert len(terminal) == 1
    assert terminal[0]["type"] == "task_completed"
    assert terminal[0]["task"]["status"] == expected.value
    assert terminal[0]["run_id"] == lease.run_id
    assert terminal[0]["state_version"] == state_version
    if outcome["success"]:
        # The V2 display path ran: the committed assistant frame precedes it.
        assert terminal[0]["message_id"] == published[-2]["data"]["message_id"]


@pytest.mark.asyncio
async def test_finalize_rolled_back_commit_is_never_reported_as_success(
    canonical, monkeypatch
):
    factory, tid = canonical
    lease, published = await _run_turn_with_broken_finalize_commit(
        canonical, monkeypatch, {"success": True, "output": "4"}, applied=False
    )

    with factory() as db:
        task = db.get(Task, tid)
        assert task.status == TaskStatus.FAILED
        assert task.runner_id is None
        assert task.error_message.startswith("setup/run error: OperationalError")
        settled = [row for row in facts(db, tid) if row.kind == "execution_settled"]
        assert [row.payload["status"] for row in settled] == ["failed"]
    terminal = _terminal_events(published)
    assert [event["type"] for event in terminal] == ["task_error"]
    assert terminal[0]["run_id"] == lease.run_id


@pytest.mark.asyncio
async def test_finalize_lost_ack_publishes_committed_wait_once(canonical, monkeypatch):
    factory, tid = canonical
    lease, published = await _run_turn_with_broken_finalize_commit(
        canonical,
        monkeypatch,
        {"success": True, "status": "waiting_for_user", "output": "Which file?"},
        applied=True,
    )

    with factory() as db:
        task = db.get(Task, tid)
        assert task.status == TaskStatus.WAITING_FOR_USER
        assert task.runner_id is None
        state_version = int(task.state_version)
    assert _terminal_events(published) == []
    info = [event for event in published if event.get("event_type") == "task_info"]
    assert len(info) == 1
    assert info[0]["data"]["status"] == TaskStatus.WAITING_FOR_USER.value
    assert info[0]["data"]["state_version"] == state_version
    assert info[0]["data"]["run_id"] == lease.run_id
