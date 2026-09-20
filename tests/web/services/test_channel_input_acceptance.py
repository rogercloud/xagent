"""Stable provider inputs cannot select a new task on retry."""

from dataclasses import replace
from unittest.mock import Mock

import pytest
from sqlalchemy.orm import sessionmaker

from tests.web.services.task_database_shared import engine as engine_fixture
from xagent.web.models.database import Base
from xagent.web.models.task import Task
from xagent.web.models.task_channel_delivery import TaskChannelDelivery
from xagent.web.models.task_command import TaskExecutionCommand
from xagent.web.models.task_input_receipt import TaskInputReceipt
from xagent.web.models.user import User
from xagent.web.models.user_channel import UserChannel
from xagent.web.services import channel_input_acceptance as inputs
from xagent.web.services import task_event_bridge
from xagent.web.services.task_orchestrator import TaskTurnError, TaskTurnPayload

engine = engine_fixture


@pytest.fixture
def ingress(engine, monkeypatch):
    Base.metadata.create_all(engine)
    sessions = sessionmaker(bind=engine)
    monkeypatch.setenv("XAGENT_SHARED_TASK_EXECUTION_ENABLED", "true")
    monkeypatch.setattr(inputs, "get_session_local", lambda: sessions)
    monkeypatch.setattr(task_event_bridge, "_bridge", Mock(host_id="ingress"))
    with sessions() as db:
        owner = User(username="owner", password_hash="unused")
        db.add(owner)
        db.flush()
        channel = UserChannel(
            user_id=owner.id,
            channel_type="slack",
            channel_name="test",
            config={"allowed_users": ["sender"]},
            is_active=True,
        )
        db.add(channel)
        db.commit()
        incoming = inputs.ChannelInput(
            int(channel.id),
            "sender",
            "slack",
            ("team", "chat"),
            "123.456",
            "hello",
            (),
            {"chat_id": "chat", "thread_ts": "123.456", "loading_ts": None},
        )
    yield incoming, sessions
    Base.metadata.drop_all(engine)


def accept(incoming, **changes):
    owner, _ = inputs.lookup_channel_input(incoming)
    return inputs.accept_channel_input(
        incoming,
        **(
            dict(
                owner_id=owner,
                active_task_id=None,
                channel_name="test",
                payload=TaskTurnPayload(incoming.text),
                staged_files=(),
                host_id="host",
            )
            | changes
        ),
    )


def test_duplicate_uses_original_task_and_delivery(ingress):
    incoming, sessions = ingress
    first = accept(incoming)
    replay = accept(incoming, active_task_id=-1)
    assert replay.replayed
    assert (replay.task_id, replay.command_db_id) == (
        first.task_id,
        first.command_db_id,
    )
    with sessions() as db:
        assert db.query(Task).count() == 1
        assert db.query(TaskExecutionCommand).count() == 1
        assert db.query(TaskChannelDelivery).count() == 1
        assert db.query(TaskInputReceipt).count() == 1


def test_content_conflict_and_distinct_identity(ingress):
    incoming, _ = ingress
    first = accept(incoming)
    with pytest.raises(TaskTurnError, match="input_conflict"):
        accept(replace(incoming, text="changed"))
    second = accept(replace(incoming, message_id="different"))
    assert second.task_id != first.task_id


def test_failed_start_rolls_back_receipt_and_new_task(ingress, monkeypatch):
    incoming, sessions = ingress
    monkeypatch.setattr(
        inputs,
        "accept_channel_turn_no_commit",
        Mock(side_effect=RuntimeError("failed START")),
    )
    with pytest.raises(RuntimeError, match="failed START"):
        accept(incoming)
    with sessions() as db:
        assert db.query(Task).count() == 0
        assert db.query(TaskInputReceipt).count() == 0


def test_deleted_target_does_not_resurrect(ingress):
    incoming, sessions = ingress
    accept(incoming)
    with sessions() as db:
        db.query(Task).delete(synchronize_session=False)
        db.commit()
    with pytest.raises(TaskTurnError, match="input_unavailable"):
        accept(incoming)


def test_simultaneous_acceptance_has_one_winner(ingress):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier

    incoming, sessions = ingress
    owner, _ = inputs.lookup_channel_input(incoming)
    barrier = Barrier(2)

    def submit():
        barrier.wait(timeout=10)
        return inputs.accept_channel_input(
            incoming,
            owner_id=owner,
            active_task_id=None,
            channel_name="test",
            payload=TaskTurnPayload("hello"),
            staged_files=(),
            host_id="host",
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(submit) for _ in range(2)]
        results = [future.result(timeout=20) for future in futures]
    assert sorted(result.replayed for result in results) == [False, True]
    assert len({result.command_db_id for result in results}) == 1
    with sessions() as db:
        assert db.query(Task).count() == db.query(TaskInputReceipt).count() == 1


def test_replay_rechecks_sender_authorization(ingress):
    from xagent.web.services.channel_runtime import ChannelAuthorizationError

    incoming, sessions = ingress
    accept(incoming)
    with sessions() as db:
        db.get(UserChannel, incoming.channel_id).config = {
            "allowed_users": ["someone-else"]
        }
        db.commit()
    with pytest.raises(ChannelAuthorizationError):
        inputs.lookup_channel_input(incoming)


def test_commit_acknowledgement_loss_replays_saved_input(ingress, monkeypatch):
    from sqlalchemy.orm import Session

    incoming, sessions = ingress
    original = Session.commit

    def commit(db):
        accepted = any(isinstance(item, TaskInputReceipt) for item in db.dirty)
        original(db)
        if accepted:
            raise ConnectionError("commit acknowledgement lost")

    monkeypatch.setattr(Session, "commit", commit)
    saved = accept(incoming)
    assert not saved.replayed
    assert saved.selection.is_new_task
    with sessions() as db:
        assert db.query(TaskExecutionCommand).count() == 1


def test_attachment_metadata_rolls_back_with_start(ingress, tmp_path, monkeypatch):
    from xagent.web.models.uploaded_file import UploadedFile
    from xagent.web.services.uploaded_file_store import StagedUploadedFile

    incoming, sessions = ingress
    owner, _ = inputs.lookup_channel_input(incoming)
    staged = StagedUploadedFile(
        "file",
        owner,
        None,
        "input.txt",
        str(tmp_path / "input.txt"),
        "local",
        "key",
        None,
        "checksum",
        None,
        None,
        None,
        "text/plain",
        8,
    )
    monkeypatch.setattr(
        inputs,
        "accept_channel_turn_no_commit",
        Mock(side_effect=RuntimeError("failed START")),
    )
    with pytest.raises(RuntimeError, match="failed START"):
        accept(incoming, staged_files=(staged,))
    with sessions() as db:
        assert (
            db.query(UploadedFile).count()
            == db.query(Task).count()
            == db.query(TaskInputReceipt).count()
            == 0
        )


def test_batch_receipts_survive_split_and_overlapping_retry(ingress):
    incoming, sessions = ingress
    second = replace(incoming, message_id="2", text="second")
    third = replace(incoming, message_id="3", text="third")
    first = accept(
        incoming, additional_inputs=(second,), payload=TaskTurnPayload("hello\nsecond")
    )
    _, pending, replays, unavailable = inputs.lookup_channel_inputs(
        (second, incoming, third, third)
    )
    assert not unavailable
    assert pending == (third,)
    assert [row.command_db_id for row in replays] == [first.command_db_id]
    assert accept(second).command_db_id == first.command_db_id
    with pytest.raises(inputs.ChannelInputBatchChanged):
        accept(second, additional_inputs=(third,))
    with sessions() as db:
        assert db.query(TaskInputReceipt).count() == 2
        assert db.query(TaskExecutionCommand).count() == 1


def test_batch_failure_rolls_back_every_alias(ingress, monkeypatch):
    incoming, sessions = ingress
    second = replace(incoming, message_id="2")
    monkeypatch.setattr(
        inputs,
        "accept_channel_turn_no_commit",
        Mock(side_effect=RuntimeError("failed")),
    )
    with pytest.raises(RuntimeError):
        accept(incoming, additional_inputs=(second,))
    with sessions() as db:
        assert db.query(TaskInputReceipt).count() == db.query(Task).count() == 0


def test_concurrent_overlapping_batches_accept_each_physical_input_once(ingress):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier

    incoming, sessions = ingress
    owner, _ = inputs.lookup_channel_input(incoming)
    a, b, c = (replace(incoming, message_id=str(i), text=str(i)) for i in range(3))
    barrier = Barrier(2)

    def submit(items):
        barrier.wait(timeout=10)
        while items:
            _, pending, _, _ = inputs.lookup_channel_inputs(items)
            if not pending:
                return
            try:
                inputs.accept_channel_input(
                    pending[0],
                    additional_inputs=pending[1:],
                    owner_id=owner,
                    active_task_id=None,
                    channel_name="test",
                    payload=TaskTurnPayload("\n".join(i.text for i in pending)),
                    staged_files=(),
                    host_id="host",
                )
                return
            except inputs.ChannelInputBatchChanged:
                items = pending

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(submit, items) for items in [(a, b), (c, b)]]
        for future in futures:
            future.result(timeout=20)
    with sessions() as db:
        assert db.query(TaskInputReceipt).count() == 3
        assert db.query(TaskExecutionCommand).count() == 2


@pytest.fixture
def second_channel(ingress):
    incoming, sessions = ingress
    with sessions() as db:
        owner = User(username="second-owner", password_hash="unused")
        db.add(owner)
        db.flush()
        channel = UserChannel(
            user_id=owner.id,
            channel_type="slack",
            channel_name="second",
            config={"allowed_users": ["sender"]},
            is_active=True,
        )
        db.add(channel)
        db.commit()
        return int(owner.id), replace(incoming, channel_id=int(channel.id))


def test_owner_change_between_lookup_and_acceptance_rejects_input(
    ingress, second_channel
):
    incoming, sessions = ingress
    previous_owner, _ = inputs.lookup_channel_input(incoming)
    other_owner, _ = second_channel
    with sessions() as db:
        db.get(UserChannel, incoming.channel_id).user_id = other_owner
        db.commit()
    with pytest.raises(TaskTurnError, match="owner_changed"):
        inputs.accept_channel_input(
            incoming,
            owner_id=previous_owner,
            active_task_id=None,
            channel_name="test",
            payload=TaskTurnPayload("hello"),
            staged_files=(),
            host_id="host",
        )
    with sessions() as db:
        assert db.query(TaskInputReceipt).count() == db.query(Task).count() == 0


def test_batch_cannot_accept_other_owners_channel(ingress, second_channel):
    incoming, sessions = ingress
    _, other = second_channel
    with pytest.raises(TaskTurnError, match="owner_changed"):
        accept(incoming, additional_inputs=(other,))
    with sessions() as db:
        assert db.query(TaskInputReceipt).count() == db.query(Task).count() == 0


@pytest.mark.parametrize(
    "binding", ["task_owner", "channel", "command_task", "command_actor"]
)
def test_replay_rejects_changed_persisted_binding(ingress, second_channel, binding):
    incoming, sessions = ingress
    other_owner, other = second_channel
    first, second = accept(incoming), accept(other)
    with sessions() as db:
        task = db.get(Task, first.task_id)
        command = db.get(TaskExecutionCommand, first.command_db_id)
        if binding == "task_owner":
            task.user_id = other_owner
        elif binding == "channel":
            task.channel_id = other.channel_id
        elif binding == "command_task":
            command.task_id = second.task_id
        else:
            command.actor_subject = db.get(User, other_owner).actor_subject
        db.commit()
    with pytest.raises(TaskTurnError, match="input_unavailable"):
        inputs.lookup_channel_input(incoming)


def test_raced_unavailable_alias_requests_repartition(ingress):
    incoming, sessions = ingress
    old = replace(incoming, message_id="old")
    previous = accept(old)
    with sessions() as db:
        db.delete(db.get(Task, previous.task_id))
        db.commit()
    with pytest.raises(inputs.ChannelInputBatchChanged):
        accept(incoming, additional_inputs=(old,))
    _, pending, replayed, unavailable = inputs.lookup_channel_inputs((incoming, old))
    assert pending == (incoming,)
    assert not replayed
    assert unavailable == (inputs.RejectedChannelInput(old, "input_unavailable"),)
    accept(pending[0])
    with sessions() as db:
        assert db.query(Task).count() == 1
        assert db.query(TaskInputReceipt).count() == 2


def test_raced_conflicting_alias_requests_repartition(ingress):
    incoming, sessions = ingress
    old = replace(incoming, message_id="old")
    accept(old)
    changed = replace(old, text="changed")
    with pytest.raises(inputs.ChannelInputBatchChanged):
        accept(incoming, additional_inputs=(changed,))
    _, pending, replayed, rejected = inputs.lookup_channel_inputs((incoming, changed))
    assert pending == (incoming,)
    assert not replayed
    assert rejected == (inputs.RejectedChannelInput(changed, "input_conflict"),)
    accept(pending[0])
    with sessions() as db:
        assert db.query(TaskInputReceipt).count() == 2


def test_batch_actor_identity_failure_is_not_a_rejected_receipt(ingress, monkeypatch):
    incoming, _ = ingress
    monkeypatch.setattr(inputs, "_resolve_actor_subject", lambda *args: None)
    with pytest.raises(TaskTurnError, match="input_unavailable"):
        inputs.lookup_channel_inputs((incoming,))


def test_acceptance_binds_agent_transcript_and_last_input_destination(ingress):
    from xagent.web.models.agent import Agent
    from xagent.web.models.chat_message import TaskChatMessage

    incoming, sessions = ingress
    owner, _ = inputs.lookup_channel_input(incoming)
    with sessions() as db:
        agent = Agent(user_id=owner, name="Selected agent", origin="user")
        db.add(agent)
        db.commit()
        agent_id = int(agent.id)
    last = replace(
        incoming,
        message_id="last",
        destination={"chat_id": "chat", "thread_ts": "last"},
    )
    accepted = accept(
        incoming,
        additional_inputs=(last,),
        agent_id=agent_id,
        payload=TaskTurnPayload("combined input", execution_message="worker input"),
    )
    with sessions() as db:
        task = db.get(Task, accepted.task_id)
        assert task.agent_id == agent_id
        transcript = db.query(TaskChatMessage).one()
        assert transcript.content == "combined input"
        assert transcript.turn_id == accepted.command_id
        command = db.get(TaskExecutionCommand, accepted.command_db_id)
        assert command.payload["message"] == "combined input"
        assert command.payload["execution_message"] == "worker input"
        assert command.payload["before_message_id"] == transcript.id
        delivery = db.query(TaskChannelDelivery).one()
        assert delivery.command_id == command.id
        assert delivery.destination == last.destination
        assert {r.command_db_id for r in db.query(TaskInputReceipt)} == {command.id}


@pytest.mark.parametrize("reason", ["input_unavailable", "input_conflict"])
def test_lookup_partitions_old_rejections_and_new_input(ingress, reason):
    incoming, sessions = ingress
    accepted = accept(incoming)
    if reason == "input_unavailable":
        with sessions() as db:
            db.delete(db.get(Task, accepted.task_id))
            db.commit()
        rejected = incoming
    else:
        rejected = replace(incoming, text="changed")
    fresh = replace(incoming, message_id="fresh")
    _, pending, replays, rejections = inputs.lookup_channel_inputs(
        (rejected, fresh, fresh)
    )
    assert pending == (fresh,)
    assert not replays
    assert rejections == (inputs.RejectedChannelInput(rejected, reason),)
    accept(pending[0])
    with sessions() as db:
        assert db.query(TaskInputReceipt).count() == 2


def test_unseen_conflicting_duplicates_do_not_accept_partial_input(ingress):
    incoming, sessions = ingress
    changed = replace(incoming, text="changed")
    with pytest.raises(TaskTurnError, match="input_conflict"):
        inputs.lookup_channel_inputs((incoming, changed))
    with pytest.raises(TaskTurnError, match="input_conflict"):
        accept(incoming, additional_inputs=(changed,))
    with sessions() as db:
        assert db.query(Task).count() == db.query(TaskInputReceipt).count() == 0


def test_failure_after_start_staging_rolls_back_entire_acceptance(ingress, monkeypatch):
    from xagent.web.models.chat_message import TaskChatMessage

    incoming, sessions = ingress
    stage = inputs.accept_channel_turn_no_commit

    def fail(db, *args):
        stage(db, *args)
        db.flush()
        raise RuntimeError("failed after START staging")

    monkeypatch.setattr(inputs, "accept_channel_turn_no_commit", fail)
    with pytest.raises(RuntimeError, match="failed after START staging"):
        accept(incoming)
    with sessions() as db:
        for model in (
            Task,
            TaskInputReceipt,
            TaskChatMessage,
            TaskExecutionCommand,
            TaskChannelDelivery,
        ):
            assert db.query(model).count() == 0


def test_attachment_binding_preserves_durable_key_and_uses_owner_cache(
    ingress, tmp_path, monkeypatch
):
    from xagent.core.workspace import scoped_user_root
    from xagent.web.models.uploaded_file import UploadedFile
    from xagent.web.services.uploaded_file_store import StagedUploadedFile

    incoming, sessions = ingress
    owner, _ = inputs.lookup_channel_input(incoming)
    uploads = tmp_path / "uploads"
    monkeypatch.setattr(inputs, "get_uploads_dir", lambda: uploads)
    staged = StagedUploadedFile(
        "file",
        owner,
        None,
        "input.txt",
        str(tmp_path / "ephemeral" / "input.txt"),
        "local",
        "durable-key",
        None,
        "checksum",
        None,
        None,
        None,
        "text/plain",
        8,
    )
    accepted = accept(
        incoming,
        staged_files=(staged,),
        payload=TaskTurnPayload("hello", file_ids=("file",)),
    )
    with sessions() as db:
        file = db.query(UploadedFile).one()
        assert file.task_id == accepted.task_id
        assert file.user_id == owner
        assert file.storage_key == "durable-key"
        assert file.checksum == "checksum"
        assert file.storage_path == str(
            scoped_user_root(uploads, owner) / "file" / "input.txt"
        )
        command = db.get(TaskExecutionCommand, accepted.command_db_id)
        assert command.payload["file_ids"] == ["file"]


def test_acceptance_rejects_run_changed_after_task_selection(ingress, monkeypatch):
    from sqlalchemy import update

    from xagent.web.models.task import TaskStatus

    incoming, sessions = ingress
    if sessions.kw["bind"].dialect.name != "postgresql":
        pytest.skip("Independent concurrent writer requires PostgreSQL")
    sessions.configure(autoflush=False)
    first = accept(incoming)
    with sessions() as db:
        task = db.get(Task, first.task_id)
        task.status = TaskStatus.COMPLETED
        task.run_id = "previous-run"
        db.get(TaskExecutionCommand, first.command_db_id).status = "completed"
        db.commit()
    prepare = inputs.prepare_channel_task_no_commit

    def interleave(db, **kwargs):
        selection = prepare(db, **kwargs)
        with sessions() as other:
            other.execute(
                update(Task)
                .where(Task.id == first.task_id)
                .values(run_id="newer-run", state_version=Task.state_version + 1)
            )
            other.commit()
        return selection

    monkeypatch.setattr(inputs, "prepare_channel_task_no_commit", interleave)
    with pytest.raises(TaskTurnError, match="busy"):
        accept(replace(incoming, message_id="next"), active_task_id=first.task_id)
    with sessions() as db:
        assert (
            db.query(TaskInputReceipt).count()
            == db.query(TaskExecutionCommand).count()
            == 1
        )
        assert db.get(Task, first.task_id).run_id == "newer-run"
