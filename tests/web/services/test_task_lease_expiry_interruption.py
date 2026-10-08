"""Lease-expiry recovery records why it stopped a run, and changes nothing else.

Runs on SQLite and, when ``XAGENT_TEST_POSTGRES_URL`` is set, on PostgreSQL
through the shared ``engine`` fixture. Every case drives the real recovery
batch, so SQLite exercises the compare-and-swap path and PostgreSQL the
row-lock path.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone
from typing import Any

import pytest
import sqlalchemy as sa
from sqlalchemy.orm import Session, sessionmaker

from tests.web.services.task_database_shared import engine as engine_fixture
from xagent.core.agent.checkpoint import CHECKPOINT_SCHEMA_VERSION, CHECKPOINT_TYPE
from xagent.web.models.agent import Agent
from xagent.web.models.database import Base
from xagent.web.models.task import Task, TaskStatus, TraceEvent
from xagent.web.models.task_auto_recovery import TaskAutoRecovery, TaskRecoveryEvent
from xagent.web.models.trigger import (
    AgentTrigger,
    TriggerRun,
    TriggerRunStatus,
    TriggerType,
)
from xagent.web.models.user import User
from xagent.web.models.user_channel import UserChannel
from xagent.web.models.workforce import Workforce, WorkforceRun
from xagent.web.services import task_auto_recovery, task_lease_recovery
from xagent.web.services.task_auto_recovery import auto_recovery_eligibility
from xagent.web.services.task_execution_event_writer import append_fact_no_commit
from xagent.web.services.task_lease_recovery import (
    TASK_LEASE_EXPIRED_ERROR,
    TASK_LEASE_PAUSED_TRIGGER_ERROR,
    TASK_UNKNOWN_TOOL_EFFECT_ERROR,
    recover_expired_task_leases_batch_isolated,
)
from xagent.web.services.task_lease_service import (
    TASK_RUN_ID_TRACE_FIELD,
    CheckpointRecoveryResolution,
    CheckpointRecoveryVerdict,
    utc_now,
)
from xagent.web.services.trace_message_storage import (
    encode_checkpoint_data_for_storage,
)

engine = engine_fixture


@pytest.fixture
def factory(engine, monkeypatch) -> sessionmaker:
    Base.metadata.create_all(engine)
    result = sessionmaker(engine)
    monkeypatch.setattr("xagent.web.models.database.get_session_local", lambda: result)
    # Selects the PostgreSQL row-lock path or the SQLite CAS path.
    monkeypatch.setattr("xagent.web.models.database.get_engine", lambda: engine)
    monkeypatch.setattr(
        task_lease_recovery, "invalidate_task_cache_best_effort", lambda _id: None
    )
    monkeypatch.delenv("XAGENT_TASK_INFRA_FAILURE_PAUSE_ENABLED", raising=False)
    return result


def _aware(value: datetime) -> datetime:
    return value if value.tzinfo else value.replace(tzinfo=timezone.utc)


def _user(db: Session) -> User:
    user = User(username=f"interruption-{uuid.uuid4().hex[:8]}", password_hash="x")
    db.add(user)
    db.flush()
    return user


def _snapshot(*, messages: int, iteration: int = 0) -> dict[str, Any]:
    return {
        "pattern": "ReActPattern",
        "context": {
            "messages": [
                {
                    "role": "user" if index % 2 == 0 else "assistant",
                    "content": f"m{index}",
                }
                for index in range(messages)
            ]
        },
        "pattern_state": {"current_iteration": iteration, "tool_ledger": {}},
    }


def _expire(task: Task, *, run_id: str) -> None:
    """Put ``task`` back in a RUNNING run whose lease has already expired."""
    task.status = TaskStatus.RUNNING
    task.control_state = "running"
    task.runner_id = f"dead-runner-{uuid.uuid4().hex[:6]}"
    task.run_id = run_id
    task.lease_expires_at = utc_now() - timedelta(seconds=5)
    task.last_heartbeat_at = utc_now() - timedelta(seconds=10)
    task.error_message = None


def _legacy_checkpoint(
    db: Session, task: Task, *, messages: int, iteration: int = 0, encode=False
) -> None:
    event_id = f"checkpoint-{uuid.uuid4().hex[:10]}"
    data: dict[str, Any] = {
        "checkpoint_type": CHECKPOINT_TYPE,
        "snapshot": _snapshot(messages=messages, iteration=iteration),
        TASK_RUN_ID_TRACE_FIELD: task.run_id,
    }
    if encode:
        data = encode_checkpoint_data_for_storage(
            db, task_id=int(task.id), data=data, use_v2=True
        )
        # The marker must come from decoded refs, not the stored markers.
        assert "__encoding" in data["snapshot"]["context"]["messages"]
    db.add(
        TraceEvent(
            task_id=task.id,
            event_id=event_id,
            event_type="system_update_general",
            timestamp=utc_now(),
            data=data,
        )
    )
    task.last_checkpoint_event_id = event_id
    task.last_checkpoint_trace_event_id = None


def _event_checkpoint(db: Session, task: Task, *, messages: int) -> None:
    task.conversation_storage_version = 2
    db.flush()
    append_fact_no_commit(
        db,
        task_id=int(task.id),
        kind="recovery_state",
        key=f"runtime:{uuid.uuid4()}",
        payload={
            "data": {
                "checkpoint_type": CHECKPOINT_TYPE,
                "snapshot_schema_version": CHECKPOINT_SCHEMA_VERSION,
                "execution_id": str(task.id),
                "snapshot": _snapshot(messages=messages, iteration=2),
            }
        },
        run_id=task.run_id,
    )


def _task(
    db: Session,
    user: User,
    *,
    checkpoint: str | None = "legacy",
    messages: int = 2,
    **fields: Any,
) -> Task:
    task = Task(
        user_id=user.id,
        title="Interrupted task",
        description="lease expiry interruption test",
        execution_mode="balanced",
        state_version=3,
        output="stale output",
        **fields,
    )
    _expire(task, run_id=f"run-{uuid.uuid4().hex[:8]}")
    if "control_state" in fields:
        task.control_state = fields["control_state"]
    db.add(task)
    db.flush()
    if checkpoint == "legacy":
        _legacy_checkpoint(db, task, messages=messages, iteration=1)
    elif checkpoint == "encoded":
        _legacy_checkpoint(db, task, messages=messages, iteration=1, encode=True)
    elif checkpoint == "events":
        _event_checkpoint(db, task, messages=messages)
    db.commit()
    return task


def _recover() -> int:
    return recover_expired_task_leases_batch_isolated(
        cutoff=utc_now(), batch_size=10, after=None
    ).recovered


def _state(factory: sessionmaker, task_id: int) -> tuple[Task, Any, list[Any]]:
    with factory() as db:
        task = db.get(Task, task_id)
        row = db.get(TaskAutoRecovery, task_id)
        events = list(
            db.scalars(
                sa.select(TaskRecoveryEvent)
                .where(TaskRecoveryEvent.task_id == task_id)
                .order_by(TaskRecoveryEvent.id)
            )
        )
        db.expunge_all()
    return task, row, events


@pytest.mark.parametrize("checkpoint", ["legacy", "encoded", "events"])
def test_recoverable_run_pauses_and_records_lease_expired(factory, checkpoint):
    with factory() as db:
        task_id = int(_task(db, _user(db), checkpoint=checkpoint, messages=3).id)
    before = utc_now()

    assert _recover() == 1

    task, row, events = _state(factory, task_id)
    # The task row is exactly what lease recovery writes without metadata.
    assert task.status == TaskStatus.PAUSED
    assert task.control_state == "paused"
    assert task.state_version == 4
    assert task.error_message is None
    assert task.runner_id is None and task.lease_expires_at is None
    assert row.run_id == task.run_id
    assert row.reason == "lease_expired"
    assert row.state == "manual"
    assert row.state_detail is None
    assert row.paused_state_version == task.state_version
    assert _aware(row.interrupted_at) >= before
    assert _aware(row.episode_started_at) == _aware(row.interrupted_at)
    assert (row.no_progress_resumes, row.total_resumes) == (0, 0)
    iterations = 2 if checkpoint == "events" else 1
    assert row.progress_marker == f"m3:i{iterations}:t0:p0:s0"
    assert row.next_attempt_at is None
    assert [(e.event, e.reason, e.run_id) for e in events] == [
        ("interrupted", "lease_expired", task.run_id)
    ]
    assert events[0].detail == {
        "task_status": "paused",
        "state": "manual",
        "kind": "normal",
    }


def test_pause_requested_at_crash_records_user_pause(factory):
    with factory() as db:
        task_id = int(_task(db, _user(db), control_state="pause_requested").id)

    assert _recover() == 1

    task, row, events = _state(factory, task_id)
    assert task.status == TaskStatus.PAUSED
    assert task.control_state == "paused"
    assert (row.reason, row.state) == ("user_pause", "manual")
    assert [e.reason for e in events] == ["user_pause"]


def test_run_without_checkpoint_fails_unchanged_and_records_not_recoverable(
    factory,
):
    with factory() as db:
        task_id = int(_task(db, _user(db), checkpoint=None).id)

    assert _recover() == 1

    task, row, events = _state(factory, task_id)
    assert task.status == TaskStatus.FAILED
    assert task.control_state == "failed"
    assert task.error_message == TASK_LEASE_EXPIRED_ERROR
    assert task.output is None
    assert (row.reason, row.state, row.progress_marker) == (
        "not_recoverable",
        "manual",
        None,
    )
    assert row.paused_state_version == task.state_version
    assert [(e.event, e.detail["task_status"]) for e in events] == [
        ("interrupted", "failed")
    ]


def test_unknown_tool_effect_fails_unchanged_and_records_its_reason(
    factory, monkeypatch
):
    with factory() as db:
        task_id = int(_task(db, _user(db), control_state="pause_requested").id)
    monkeypatch.setattr(
        task_lease_recovery,
        "resolve_checkpoint_recovery_with_data",
        lambda db, candidate: CheckpointRecoveryResolution(
            CheckpointRecoveryVerdict.UNKNOWN_TOOL_EFFECT
        ),
    )

    assert _recover() == 1

    task, row, _events = _state(factory, task_id)
    assert task.status == TaskStatus.FAILED
    assert task.error_message == TASK_UNKNOWN_TOOL_EFFECT_ERROR
    # A pause request does not outrank a terminal verdict.
    assert (row.reason, row.state) == ("unknown_tool_effect", "manual")


def test_indeterminate_verdict_writes_nothing(factory, monkeypatch):
    with factory() as db:
        task_id = int(_task(db, _user(db)).id)
    monkeypatch.setattr(
        task_lease_recovery,
        "resolve_checkpoint_recovery_with_data",
        lambda db, candidate: CheckpointRecoveryResolution(
            CheckpointRecoveryVerdict.INDETERMINATE
        ),
    )

    assert _recover() == 0

    task, row, events = _state(factory, task_id)
    assert task.status == TaskStatus.RUNNING
    assert row is None and events == []


def test_lost_recovery_race_writes_nothing(factory, monkeypatch):
    with factory() as db:
        task_id = int(_task(db, _user(db)).id)
    monkeypatch.setattr(
        task_lease_recovery,
        "recover_expired_task_lease_no_commit",
        lambda *args, **kwargs: False,
    )

    assert _recover() == 0

    _task_row, row, events = _state(factory, task_id)
    assert row is None and events == []


def test_disabled_infra_failure_pause_records_metadata_only(factory, monkeypatch):
    monkeypatch.setenv("XAGENT_TASK_INFRA_FAILURE_PAUSE_ENABLED", "false")
    with factory() as db:
        task_id = int(_task(db, _user(db)).id)

    assert _recover() == 1

    task, row, events = _state(factory, task_id)
    assert task.status == TaskStatus.PAUSED
    assert task.error_message is None
    assert (row.reason, row.state, row.state_detail) == (
        "lease_expired",
        "disabled",
        None,
    )
    assert [e.event for e in events] == ["interrupted"]


def _channel(db: Session, user: User) -> int:
    channel = UserChannel(
        user_id=user.id, channel_type="slack", channel_name="ops", config={}
    )
    db.add(channel)
    db.flush()
    return int(channel.id)


def _workforce_run(db: Session, user: User, *, is_preview: bool) -> int:
    manager = Agent(user_id=user.id, name="interruption manager")
    db.add(manager)
    db.flush()
    workforce = Workforce(
        owner_user_id=user.id,
        scope_type="user",
        scope_id=str(user.id),
        name=f"Interruption workforce {uuid.uuid4().hex[:6]}",
        manager_agent_id=manager.id,
        status="published",
    )
    db.add(workforce)
    db.flush()
    run = WorkforceRun(
        workforce_id=workforce.id,
        user_id=user.id,
        status="running",
        snapshot={},
        is_preview=is_preview,
    )
    db.add(run)
    db.flush()
    return int(run.id)


@pytest.mark.parametrize(
    ("case", "expected"),
    [
        ("sdk", "ineligible:source_sdk"),
        ("a2a", "ineligible:source_a2a"),
        ("external", "ineligible:source_external"),
        ("widget", "ineligible:source_widget"),
        ("shared_link", "ineligible:source_shared_link"),
        ("workforce_preview", "ineligible:preview"),
        ("internal_invisible", "ineligible:preview"),
        ("channel_combined", "ineligible:channel_inline"),
    ],
)
def test_ineligible_tasks_recover_unchanged_and_record_why(
    factory, monkeypatch, case, expected
):
    monkeypatch.setenv(
        "XAGENT_SHARED_TASK_EXECUTION_ENABLED",
        "false" if case == "channel_combined" else "true",
    )
    with factory() as db:
        user = _user(db)
        fields: dict[str, Any] = {}
        if case in {"sdk", "a2a", "external", "widget", "shared_link"}:
            fields["source"] = case
        elif case == "workforce_preview":
            fields["agent_config"] = {
                "workforce_run_id": _workforce_run(db, user, is_preview=True)
            }
        elif case == "internal_invisible":
            fields["is_visible"] = False
        else:
            fields["channel_id"] = _channel(db, user)
        task_id = int(_task(db, user, **fields).id)

    assert _recover() == 1

    task, row, events = _state(factory, task_id)
    assert task.status == TaskStatus.PAUSED
    assert task.error_message is None
    assert (row.reason, row.state, row.state_detail) == (
        "lease_expired",
        "ineligible",
        expected,
    )
    assert [(e.event, e.detail.get("state_detail")) for e in events] == [
        ("ineligible", expected)
    ]


@pytest.mark.parametrize(
    ("case", "kind"),
    [
        ("normal", "normal"),
        ("trigger", "trigger_webhook"),
        ("workforce", "workforce"),
        ("channel_shared", "channel"),
    ],
)
def test_eligible_task_kinds(factory, monkeypatch, case, kind):
    monkeypatch.setenv("XAGENT_SHARED_TASK_EXECUTION_ENABLED", "true")
    with factory() as db:
        user = _user(db)
        fields: dict[str, Any] = {}
        if case == "trigger":
            fields["source"] = "trigger"
            fields["agent_config"] = {"trigger_type": TriggerType.WEBHOOK.value}
        elif case == "workforce":
            fields["agent_config"] = {
                "workforce_run_id": _workforce_run(db, user, is_preview=False)
            }
        elif case == "channel_shared":
            fields["channel_id"] = _channel(db, user)
        task = _task(db, user, checkpoint=None, **fields)
        eligibility = auto_recovery_eligibility(db, task)

    assert eligibility.eligible is True
    assert eligibility.kind == kind
    assert eligibility.detail is None


def test_projections_match_plain_lease_recovery(factory):
    with factory() as db:
        user = _user(db)
        run_id = _workforce_run(db, user, is_preview=False)
        task = _task(db, user, source="trigger")
        task.agent_config = {"workforce_run_id": run_id}
        workforce_run = db.get(WorkforceRun, run_id)
        workforce_run.task_id = task.id
        trigger = AgentTrigger(
            user_id=user.id,
            workforce_id=workforce_run.workforce_id,
            type=TriggerType.SCHEDULED.value,
            name="Interruption trigger",
            config={},
        )
        db.add(trigger)
        db.flush()
        trigger_run = TriggerRun(
            trigger_id=trigger.id,
            task_id=task.id,
            status=TriggerRunStatus.RUNNING.value,
            idempotency_key=f"interruption-{task.id}",
        )
        db.add(trigger_run)
        db.commit()
        task_id, trigger_run_id = int(task.id), int(trigger_run.id)

    assert _recover() == 1

    task, row, _events = _state(factory, task_id)
    with factory() as db:
        workforce_run = db.get(WorkforceRun, run_id)
        trigger_run = db.get(TriggerRun, trigger_run_id)
        assert workforce_run.status == "paused"
        assert workforce_run.completed_at is None
        assert trigger_run.status == TriggerRunStatus.FAILED.value
        assert trigger_run.error_message == TASK_LEASE_PAUSED_TRIGGER_ERROR
        assert trigger_run.finished_at is not None
    assert task.status == TaskStatus.PAUSED
    assert (row.state, row.reason) == ("manual", "lease_expired")


def _rerun(factory: sessionmaker, task_id: int, *, run_id: str | None = None):
    """Resume ``task_id`` by hand and let the resumed run's lease expire."""
    with factory() as db:
        task = db.get(Task, task_id)
        _expire(task, run_id=run_id or task.run_id)
        db.commit()


def _progress(factory: sessionmaker, task_id: int, *, messages: int) -> None:
    with factory() as db:
        task = db.get(Task, task_id)
        _legacy_checkpoint(db, task, messages=messages, iteration=1)
        db.commit()


def test_repeated_interruptions_track_episodes_and_runs(factory):
    with factory() as db:
        task_id = int(_task(db, _user(db), messages=2).id)
    assert _recover() == 1
    _task_row, first, _events = _state(factory, task_id)
    # Stand-in for Phase 2 dispatches, which are what these counters count.
    with factory() as db:
        row = db.get(TaskAutoRecovery, task_id)
        row.no_progress_resumes, row.total_resumes = 2, 5
        row.last_command_id = "auto-resume:previous"
        db.commit()

    # Same run, same checkpoint: the no-progress episode continues.
    _rerun(factory, task_id)
    assert _recover() == 1
    task, same, events = _state(factory, task_id)
    assert same.run_id == first.run_id
    assert same.progress_marker == first.progress_marker
    assert _aware(same.episode_started_at) == _aware(first.episode_started_at)
    assert _aware(same.interrupted_at) > _aware(first.interrupted_at)
    assert same.paused_state_version == task.state_version
    assert same.paused_state_version > first.paused_state_version
    assert (same.no_progress_resumes, same.total_resumes) == (2, 5)
    assert same.last_command_id == "auto-resume:previous"
    assert len(events) == 2

    # Same run with a newer checkpoint: progress starts a new episode.
    _rerun(factory, task_id)
    _progress(factory, task_id, messages=4)
    assert _recover() == 1
    _task_row, progressed, _events = _state(factory, task_id)
    assert progressed.progress_marker == "m4:i1:t0:p0:s0"
    assert _aware(progressed.episode_started_at) == _aware(progressed.interrupted_at)
    assert _aware(progressed.episode_started_at) > _aware(same.episode_started_at)
    assert (progressed.no_progress_resumes, progressed.total_resumes) == (0, 5)
    assert progressed.last_command_id == "auto-resume:previous"

    # A new run resets every counter, even at an identical marker.
    with factory() as db:
        row = db.get(TaskAutoRecovery, task_id)
        row.no_progress_resumes = 1
        db.commit()
    _rerun(factory, task_id, run_id="run-next")
    with factory() as db:
        task = db.get(Task, task_id)
        _legacy_checkpoint(db, task, messages=4, iteration=1)
        db.commit()
    assert _recover() == 1
    _task_row, new_run, events = _state(factory, task_id)
    assert new_run.run_id == "run-next"
    assert new_run.progress_marker == progressed.progress_marker
    assert _aware(new_run.episode_started_at) == _aware(new_run.interrupted_at)
    assert (new_run.no_progress_resumes, new_run.total_resumes) == (0, 0)
    assert new_run.last_command_id is None
    assert [e.run_id for e in events] == [first.run_id] * 3 + ["run-next"]


@pytest.mark.parametrize("failure", ["python", "database"])
def test_metadata_failure_still_recovers_the_task(factory, monkeypatch, failure):
    with factory() as db:
        user = _user(db)
        task_id = int(_task(db, user).id)
    original = task_auto_recovery.record_interruption_no_commit

    def fail(db, **kwargs):
        # Leave real writes behind first: the SAVEPOINT must discard them.
        original(db, **kwargs)
        if failure == "database":
            db.execute(sa.text("SELECT * FROM no_such_table"))
        raise RuntimeError("metadata failed")

    monkeypatch.setattr(task_auto_recovery, "record_interruption_no_commit", fail)

    assert _recover() == 1

    task, row, events = _state(factory, task_id)
    assert task.status == TaskStatus.PAUSED
    assert task.state_version == 4
    assert row is None and events == []
