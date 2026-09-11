"""The START wire contract is durable but deliberately not executable yet."""

import json
from dataclasses import replace
from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from xagent.web.models.chat_message import TaskChatMessage
from xagent.web.models.database import Base, get_engine, get_session_local, init_db
from xagent.web.models.task import Task, TaskStatus
from xagent.web.models.task_command import TaskExecutionCommand
from xagent.web.models.user import User
from xagent.web.services.task_command_transport import (
    ClaimedTaskCommand,
    TaskCommandKind,
    claim_task_command,
    stage_task_command,
)
from xagent.web.services.task_start_protocol import (
    TaskStartPayload,
    read_task_start_command,
    stage_task_start_command,
)


@pytest.fixture
def db_session(tmp_path):
    init_db(db_url=f"sqlite:///{tmp_path / 'start-protocol.db'}")
    with get_session_local()() as db:
        yield db
    Base.metadata.drop_all(bind=get_engine())


def start_payload(**overrides):
    values = {
        "version": 1,
        "run_id": "run-1",
        "state_version": 1,
        "turn_id": "turn-1",
        "kind": "create",
        "message": "Read the attachment",
        "execution_message": "Read the attachment\n[file_id: uploaded-file-1]",
        "file_ids": ["uploaded-file-1"],
        "before_message_id": None,
        "timezone": "Asia/Taipei",
    }
    values.update(overrides)
    return TaskStartPayload(**values)


def accept_turn(db, start):
    """Stage a new accepted row and transcript without an execution lease."""
    user = User(username="start-owner", password_hash="unused")
    db.add(user)
    db.flush()
    task = Task(
        user_id=user.id,
        title="START protocol",
        source="sdk",
        status=TaskStatus.RUNNING,
        run_id=start.run_id,
        state_version=start.state_version,
        control_state="running",
        input=start.message,
    )
    db.add(task)
    db.flush()
    db.add(
        TaskChatMessage(
            task_id=task.id,
            user_id=user.id,
            role="user",
            message_type="user_message",
            content=start.message,
            turn_id=start.turn_id,
        )
    )
    return user, task


def claimed_command(payload):
    return ClaimedTaskCommand(
        id=1,
        task_id=1,
        actor_user_id=1,
        command_id=payload.turn_id,
        kind=TaskCommandKind.START,
        payload=payload.model_dump(mode="json"),
        target_run_id=payload.run_id,
        attempt_count=1,
    )


@pytest.mark.parametrize(
    "kind,force_fresh", [("create", False), ("append", False), ("append", True)]
)
def test_json_round_trip_preserves_execution_input_and_turn_identity(kind, force_fresh):
    start = start_payload(kind=kind, force_fresh=force_fresh, before_message_id=3)
    command = claimed_command(start)
    command = replace(command, payload=json.loads(json.dumps(command.payload)))
    decoded = read_task_start_command(command)
    assert decoded == start
    assert decoded.message != decoded.execution_message
    assert decoded.file_ids == ["uploaded-file-1"]
    assert decoded.timezone == "Asia/Taipei"


@pytest.mark.parametrize(
    "changes",
    [
        {"version": 2},
        {"version": True},
        {"version": 1.0},
        {"state_version": "1"},
        {"state_version": 0},
        {"turn_id": "turn with spaces"},
        {"kind": "resume"},
        {"force_fresh": True},
        {"file_ids": [""]},
        {"context": {"user": {"id": 1}}},
        {"secrets": {"token": "synthetic"}},
        {"task_lease": {"runner_id": "web"}},
    ],
)
def test_decode_rejects_unsupported_or_lossy_payload(changes):
    command = claimed_command(start_payload())
    with pytest.raises(ValidationError):
        read_task_start_command(
            replace(command, payload={**command.payload, **changes})
        )


def test_decode_requires_version():
    command = claimed_command(start_payload())
    del command.payload["version"]
    with pytest.raises(ValidationError):
        read_task_start_command(command)


@pytest.mark.parametrize(
    "changes",
    [
        {"kind": TaskCommandKind.MESSAGE},
        {"target_run_id": "another-run"},
        {"target_run_id": None},
        {"command_id": "another-turn"},
    ],
)
def test_decode_rejects_envelope_mismatch(changes):
    with pytest.raises(ValueError):
        read_task_start_command(replace(claimed_command(start_payload()), **changes))


def test_acceptance_and_command_are_visible_only_after_outer_commit(db_session):
    start = start_payload()
    user, task = accept_turn(db_session, start)
    staged = stage_task_start_command(
        db_session, task_id=task.id, actor_user_id=user.id, start=start
    )
    with get_session_local()() as observer:
        assert observer.get(Task, task.id) is None
        assert observer.get(TaskExecutionCommand, staged.staged_db_id) is None
    db_session.commit()
    with get_session_local()() as observer:
        row = observer.get(TaskExecutionCommand, staged.staged_db_id)
        assert row.kind == "start"
        assert row.target_run_id == start.run_id
        assert row.target_state_version == start.state_version
        assert row.target_runner_id is None
        assert row.task_owner_user_id == user.id
        assert row.payload == start.model_dump(mode="json")
        assert (
            observer.query(TaskChatMessage)
            .filter_by(task_id=task.id, turn_id=start.turn_id)
            .one()
            .content
            == start.message
        )
        accepted = observer.get(Task, task.id)
        assert accepted.runner_id is None
        assert accepted.lease_attempt_id is None


def test_outer_rollback_removes_task_message_and_start(db_session):
    start = start_payload()
    user, task = accept_turn(db_session, start)
    task_id = task.id
    staged = stage_task_start_command(
        db_session, task_id=task_id, actor_user_id=user.id, start=start
    )
    db_session.rollback()
    with get_session_local()() as observer:
        assert observer.get(Task, task_id) is None
        assert observer.get(TaskExecutionCommand, staged.staged_db_id) is None
        assert observer.query(TaskChatMessage).filter_by(task_id=task_id).count() == 0


def test_duplicate_turn_identity_compares_the_full_start_payload(db_session):
    start = start_payload()
    user, task = accept_turn(db_session, start)
    first = stage_task_start_command(
        db_session, task_id=task.id, actor_user_id=user.id, start=start
    )
    same = stage_task_start_command(
        db_session, task_id=task.id, actor_user_id=user.id, start=start
    )
    mismatch = stage_task_start_command(
        db_session,
        task_id=task.id,
        actor_user_id=user.id,
        start=start_payload(execution_message="different Agent input"),
    )
    assert same.staged_db_id == first.staged_db_id == mismatch.staged_db_id
    assert not same.created and same.payload_matches
    assert not mismatch.created and not mismatch.payload_matches
    assert (
        db_session.query(TaskExecutionCommand).filter_by(task_id=task.id).count() == 1
    )


@pytest.mark.parametrize(
    "changes",
    [
        {"status": TaskStatus.PENDING},
        {"run_id": "another-run"},
        {"state_version": 2},
        {"control_state": "pause_requested"},
        {"runner_id": "web-process"},
        {"lease_attempt_id": "old-attempt"},
        {"lease_expires_at": datetime.now(timezone.utc)},
        {"last_heartbeat_at": datetime.now(timezone.utc)},
    ],
)
def test_start_requires_exact_accepted_turn_without_execution_lease(
    db_session, changes
):
    start = start_payload()
    user, task = accept_turn(db_session, start)
    for field, value in changes.items():
        setattr(task, field, value)
    with pytest.raises(ValueError):
        stage_task_start_command(
            db_session, task_id=task.id, actor_user_id=user.id, start=start
        )
    assert db_session.query(TaskExecutionCommand).count() == 0


def test_existing_dispatcher_cannot_claim_start_or_overtake_it(db_session):
    start = start_payload()
    user, task = accept_turn(db_session, start)
    staged = stage_task_start_command(
        db_session, task_id=task.id, actor_user_id=user.id, start=start
    )
    db_session.commit()
    # Neither a targeted immediate dispatch nor the background scan may
    # consume a protocol whose execution semantics have not been wired yet.
    assert (
        claim_task_command(
            db_session, runner_id="runner", command_db_id=staged.staged_db_id
        )
        is None
    )
    assert claim_task_command(db_session, runner_id="runner") is None
    stage_task_command(
        db_session,
        task_id=task.id,
        actor_user_id=user.id,
        command_id="cancel-after-start",
        kind=TaskCommandKind.CANCEL,
        payload={},
    )
    other = Task(user_id=user.id, title="unrelated", status=TaskStatus.PENDING)
    db_session.add(other)
    db_session.flush()
    independent = stage_task_command(
        db_session,
        task_id=other.id,
        actor_user_id=user.id,
        command_id="independent-cancel",
        kind=TaskCommandKind.CANCEL,
        payload={},
    )
    db_session.commit()
    claimed = claim_task_command(db_session, runner_id="runner")
    assert claimed is not None and claimed.id == independent.staged_db_id
    assert db_session.get(TaskExecutionCommand, staged.staged_db_id).attempt_count == 0


def test_append_acceptance_rollback_preserves_previous_turn(db_session):
    start = start_payload(kind="append", force_fresh=True)
    user, task = accept_turn(db_session, start)
    task.status = TaskStatus.COMPLETED
    task.control_state = "completed"
    task.run_id = "previous-run"
    task.state_version = 0
    task.output = "previous output"
    db_session.flush()
    db_session.query(TaskChatMessage).filter_by(task_id=task.id).update(
        {
            TaskChatMessage.turn_id: "previous-turn",
            TaskChatMessage.content: "previous input",
        }
    )
    db_session.commit()
    task_id = task.id
    task.status = TaskStatus.RUNNING
    task.control_state = "running"
    task.run_id = start.run_id
    task.state_version = start.state_version
    task.output = None
    db_session.add(
        TaskChatMessage(
            task_id=task_id,
            user_id=user.id,
            role="user",
            message_type="user_message",
            content=start.message,
            turn_id=start.turn_id,
        )
    )
    staged = stage_task_start_command(
        db_session, task_id=task_id, actor_user_id=user.id, start=start
    )
    with get_session_local()() as observer:
        previous = observer.get(Task, task_id)
        assert previous.run_id == "previous-run"
        assert previous.status == TaskStatus.COMPLETED
    db_session.rollback()
    with get_session_local()() as observer:
        previous = observer.get(Task, task_id)
        assert previous.run_id == "previous-run"
        assert previous.output == "previous output"
        transcript = observer.query(TaskChatMessage).filter_by(task_id=task_id).one()
        assert transcript.turn_id == "previous-turn"
        assert transcript.content == "previous input"
        assert observer.get(TaskExecutionCommand, staged.staged_db_id) is None
