"""R1 (S5.2): injection atomicity against a real SQLite-backed checkpoint store.

Everything above ``tests/core/agent/test_runner.py`` exercises the lock and
the read-back disambiguation against in-memory test doubles. This file
drives the same protocol through the production ``Tracer`` ->
``DatabaseTraceHandler`` -> SQLite path, so the "commit succeeded, the
confirmation was lost" and "the write never committed" cases are proven
against a real database transaction, not a double that merely claims to
model one.
"""

from __future__ import annotations

import pytest

from xagent.core.agent import Agent, AgentRunner, ExecutionContext
from xagent.core.agent.checkpoint import (
    CheckpointPersistenceError,
    TraceCheckpointStore,
)
from xagent.core.agent.runner import UserMessageInjectionOutcome
from xagent.core.agent.trace import TraceEvent, TraceHandler, Tracer
from xagent.web.models.database import Base, get_db, get_engine, init_db
from xagent.web.models.task import Task, TaskStatus
from xagent.web.models.user import User
from xagent.web.services.task_lease_service import TaskLease, bind_task_lease_context
from xagent.web.services.trace_handlers import DatabaseTraceHandler


class _RaisingHandler(TraceHandler):
    """Fails every event after any earlier handler in the list has already
    run -- models a downstream handler failing after the database commit
    already landed (design S1.2's "commit succeeded, confirmation lost"
    source)."""

    async def handle_event(self, event: TraceEvent) -> None:
        raise RuntimeError("downstream trace handler failed after DB commit")


@pytest.fixture()
def db_session(tmp_path):
    init_db(db_url=f"sqlite:///{tmp_path / 'injection_atomicity.db'}")
    db = next(get_db())
    try:
        yield db
    finally:
        db.close()
        engine = get_engine()
        # FK enforcement is on for every connection this engine opens (see
        # xagent.db.sqlite); dropping tables that carry real trace_events
        # rows referencing this run's tasks needs it off on the specific
        # connection drop_all uses, or SQLite refuses the DROP.
        with engine.begin() as conn:
            conn.exec_driver_sql("PRAGMA foreign_keys=OFF")
            Base.metadata.drop_all(bind=conn)


def _running_task(db, owner_id: int, *, runner_id: str, run_id: str) -> Task:
    task = Task(
        user_id=owner_id,
        title="t",
        description="d",
        status=TaskStatus.RUNNING,
        execution_mode="balanced",
        source="sdk",
    )
    db.add(task)
    db.commit()
    db.refresh(task)
    task.runner_id = runner_id
    task.run_id = run_id
    task.lease_attempt_id = "attempt-1"
    db.commit()
    return task


def _user(db, username: str) -> User:
    user = User(username=username, password_hash="x", is_admin=False)
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


@pytest.mark.asyncio
async def test_commit_ack_lost_is_confirmed_by_a_real_sqlite_read_back(
    db_session,
) -> None:
    """F2 / confirmation-lost: the DB write commits, but a handler after it
    in the dispatch list raises, so ``Tracer.trace_event(require_persisted=
    True)`` raises even though the row is durably there. Injecting under a
    valid lease must resolve this as ``POSTED_FRESH`` by reading the real
    row back, and a second injection of the same turn id must replay
    without writing again."""
    owner = _user(db_session, "atomic-ack-lost-owner")
    task = _running_task(
        db_session, int(owner.id), runner_id="runner-a", run_id="run-a"
    )
    execution_id = str(task.id)
    lease = TaskLease(
        task_id=int(task.id),
        runner_id="runner-a",
        run_id="run-a",
        attempt_id="attempt-1",
    )

    tracer = Tracer()
    tracer.add_handler(DatabaseTraceHandler(int(task.id)))
    tracer.add_handler(_RaisingHandler())
    checkpoint_store = TraceCheckpointStore(tracer=tracer)

    runner = AgentRunner(
        agent=Agent(name="writer", patterns=[]),
        tracer=checkpoint_store,
    )
    context = ExecutionContext(execution_id=execution_id)
    runner.context_manager.set_context(context)

    with bind_task_lease_context(lease):
        result = await runner.inject_user_message(
            execution_id,
            "Hello from a real checkpoint store",
            turn_id="turn-ack-lost",
            request_interrupt=False,
        )

    assert result.outcome is UserMessageInjectionOutcome.POSTED_FRESH
    assert [
        message.content for message in result.context.messages if message.role == "user"
    ] == ["Hello from a real checkpoint store"]

    # The row this test cares about really is in the database, independent
    # of anything AgentRunner's in-memory context claims. The read must
    # stay inside the lease binding too: an unbound (or wrongly bound)
    # reader is refused outright while the task has an active run --
    # that is the same partition fencing exercised deliberately in
    # ``test_stale_lease_fence_miss_leaves_no_residue`` below.
    with bind_task_lease_context(lease):
        latest = await checkpoint_store.load_latest_checkpoint(execution_id)
    assert latest is not None
    stored_messages = latest["context"]["messages"]
    assert any(
        message.get("content") == "Hello from a real checkpoint store"
        and message.get("metadata", {}).get("turn_id") == "turn-ack-lost"
        for message in stored_messages
    )

    with bind_task_lease_context(lease):
        replay = await runner.inject_user_message(
            execution_id,
            "Hello from a real checkpoint store",
            turn_id="turn-ack-lost",
            request_interrupt=False,
        )

    assert replay.outcome is UserMessageInjectionOutcome.POSTED_REPLAY
    matches = [
        message
        for message in replay.context.messages
        if message.role == "user" and message.metadata.get("turn_id") == "turn-ack-lost"
    ]
    assert len(matches) == 1


@pytest.mark.asyncio
async def test_stale_lease_fence_miss_leaves_no_residue(db_session) -> None:
    """F3 / pointer fence miss: the lease bound to this call has already
    been superseded by a different attempt on the same run, so
    ``DatabaseTraceHandler`` refuses the write with a lease-changed
    ``RuntimeError`` before any row is committed. The read-back must
    therefore find the turn genuinely absent, so the injection re-raises
    with zero live-context residue -- exactly the "definitely did not
    commit" branch of the design's read-back disambiguation, exercised
    against a real stale-fence rejection rather than a synthetic one."""
    owner = _user(db_session, "atomic-fence-miss-owner")
    task = _running_task(
        db_session, int(owner.id), runner_id="runner-b", run_id="run-b"
    )
    execution_id = str(task.id)
    # Bound to an attempt id different from the task's current
    # ``lease_attempt_id`` ("attempt-1"): a real superseded-attempt fence
    # miss, not a mismatched runner/run id.
    stale_lease = TaskLease(
        task_id=int(task.id),
        runner_id="runner-b",
        run_id="run-b",
        attempt_id="attempt-0-superseded",
    )

    tracer = Tracer()
    tracer.add_handler(DatabaseTraceHandler(int(task.id)))
    checkpoint_store = TraceCheckpointStore(tracer=tracer)

    runner = AgentRunner(
        agent=Agent(name="writer", patterns=[]),
        tracer=checkpoint_store,
    )
    context = ExecutionContext(execution_id=execution_id)
    runner.context_manager.set_context(context)

    with bind_task_lease_context(stale_lease):
        with pytest.raises(CheckpointPersistenceError):
            await runner.inject_user_message(
                execution_id,
                "Should never land",
                turn_id="turn-fence-miss",
                request_interrupt=False,
            )

    assert context.messages == []
    # Verified through a lease that actually matches the task's real
    # current lease -- the stale one used for the injection attempt above
    # would itself be refused as an unbound/mismatched reader now that the
    # task has an active run (see the ack-lost test's read for the same
    # reasoning).
    valid_lease = TaskLease(
        task_id=int(task.id),
        runner_id="runner-b",
        run_id="run-b",
        attempt_id="attempt-1",
    )
    with bind_task_lease_context(valid_lease):
        latest = await checkpoint_store.load_latest_checkpoint(execution_id)
    assert latest is None
