"""Shared handoff and deployment schema checks against disposable PostgreSQL."""

from unittest.mock import Mock

import pytest
from sqlalchemy.orm import sessionmaker

from tests.shared.postgres_disposable import disposable_database_factory
from tests.web.services.coordinator_command_shared import claim_for_owner
from xagent.web.models.database import Base
from xagent.web.models.task import Task, TaskStatus
from xagent.web.models.task_command import TaskExecutionCommand
from xagent.web.models.user import User
from xagent.web.services import task_coordinator_service as ownership
from xagent.web.services import task_event_bridge
from xagent.web.services import task_start_consumer as consumer
from xagent.web.services.task_command_transport import TaskCommandRejected
from xagent.web.services.task_orchestrator import TaskTurnOrchestrator, TaskTurnPayload

pytestmark = pytest.mark.postgresql


@pytest.mark.parametrize("fail_completion", [False, True])
def test_postgres_handoff_completion_and_lease_are_atomic(monkeypatch, fail_completion):
    monkeypatch.setenv("XAGENT_SHARED_TASK_EXECUTION_ENABLED", "true")
    monkeypatch.setattr(task_event_bridge, "get_task_event_bridge", lambda: Mock())
    with disposable_database_factory("shared_worker") as make_database:
        engine = make_database("handoff")
        Base.metadata.create_all(engine)
        sessions = sessionmaker(bind=engine)
        monkeypatch.setattr(consumer, "get_session_local", lambda: sessions)
        with sessions() as db:
            owner = User(username="owner", password_hash="unused")
            db.add(owner)
            db.flush()
            task = Task(
                user_id=owner.id,
                title="Shared",
                source="sdk",
                status=TaskStatus.PENDING,
            )
            db.add(task)
            db.flush()
            accepted = TaskTurnOrchestrator.claim_created_turn_no_commit(
                db,
                task_id=task.id,
                task_owner_user_id=owner.id,
                payload=TaskTurnPayload("hello"),
            )
            db.commit()
            task_id = task.id
            owner_lease = ownership.acquire_task_lease_no_commit(
                db, task_id, runner_id="worker"
            )
            db.commit()
            command = claim_for_owner(db, owner_lease, accepted.command_db_id)
        assert owner_lease is not None
        if fail_completion:
            monkeypatch.setattr(
                consumer, "finish_task_command_no_commit", lambda *args, **kwargs: False
            )
            with pytest.raises(TaskCommandRejected):
                consumer._commit_handoff(command, owner_lease)
        else:
            handoff = consumer._commit_handoff(command, owner_lease)
            with pytest.raises(TaskCommandRejected):
                consumer._commit_handoff(command, owner_lease)
        with sessions() as db:
            task = db.get(Task, task_id)
            row = db.get(TaskExecutionCommand, command.id)
            if fail_completion:
                assert task.lease_attempt_id == owner_lease.attempt_id
                assert task.status == TaskStatus.PENDING
                assert task.run_id is None
                assert row.status == "processing"
            else:
                assert task.lease_attempt_id == handoff.claimed.task_lease.attempt_id
                assert task.run_id == accepted.run_id
                assert task.status == TaskStatus.RUNNING
                assert row.status == "completed"
                assert row.result["lease_attempt_id"] == task.lease_attempt_id


def test_recovery_keeps_journal_committed_after_candidate_read(monkeypatch):
    from datetime import timedelta

    from xagent.web.services import task_injection, task_lease_recovery
    from xagent.web.services.task_lease_service import (
        TaskLease,
        get_expired_task_lease_candidates,
        utc_now,
    )

    with disposable_database_factory("input_recovery") as make_database:
        engine = make_database("journal_race")
        Base.metadata.create_all(engine)
        sessions = sessionmaker(bind=engine)
        monkeypatch.setattr(task_injection, "get_session_local", lambda: sessions)
        with sessions() as db, db.begin():
            owner = User(username="owner", password_hash="unused")
            db.add(owner)
            db.flush()
            task = Task(
                user_id=owner.id,
                title="Recovery",
                status=TaskStatus.RUNNING,
                run_id="run",
                runner_id="worker",
                lease_attempt_id="attempt",
                state_version=1,
                control_state="running",
                lease_expires_at=utc_now() - timedelta(seconds=1),
            )
            db.add(task)
            db.flush()
            lease = TaskLease(task.id, "worker", "run", "attempt")
        original = task_lease_recovery.recover_expired_task_lease_no_commit

        def inject_before_update(db, candidate, **kwargs):
            assert kwargs["status"] == TaskStatus.FAILED
            task_injection._journal(
                lease,
                dict(
                    execution_message="answer", display_message="answer", turn_id="turn"
                ),
            )
            return original(db, candidate, **kwargs)

        monkeypatch.setattr(
            task_lease_recovery,
            "recover_expired_task_lease_no_commit",
            inject_before_update,
        )
        with sessions() as db, db.begin():
            candidate = get_expired_task_lease_candidates(
                db, cutoff=utc_now(), limit=1
            )[0]
            status = task_lease_recovery.recover_task_lease_candidate_no_commit(
                db, candidate, recovered_at=utc_now()
            )
            assert status == TaskStatus.PAUSED
        with sessions() as db:
            task = db.get(Task, lease.task_id)
            assert task.status == TaskStatus.PAUSED
            assert task.control_state == "paused"
            assert task.runner_id is None
            assert task.pending_injection["turn_id"] == "turn"
