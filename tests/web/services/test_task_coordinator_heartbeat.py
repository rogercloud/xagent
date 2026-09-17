"""Batch owner renewals, real row locks, and shutdown transaction ordering."""

import asyncio
from dataclasses import replace
from datetime import timedelta
from threading import Event
from unittest.mock import AsyncMock

import pytest
import sqlalchemy as sa
from sqlalchemy.orm import sessionmaker

from tests.web.services.test_task_execution_event_store import engine as engine_fixture
from tests.web.services.test_task_execution_event_store import (
    task_id as task_id_fixture,
)
from xagent.web.models.task import Task, TaskStatus
from xagent.web.services import task_coordinator_runtime as runtime
from xagent.web.services import task_coordinator_service as service

engine = engine_fixture
task_id = task_id_fixture


@pytest.fixture
def owners(engine, task_id):
    factory = sessionmaker(engine, expire_on_commit=False)
    with factory() as db, db.begin():
        first = db.get(Task, task_id)
        ids = [task_id]
        for status in (TaskStatus.WAITING_FOR_USER, TaskStatus.COMPLETED):
            task = Task(user_id=first.user_id, title=status.value, status=status)
            db.add(task)
            db.flush()
            ids.append(task.id)
        first.status = TaskStatus.PAUSED
        leases = tuple(
            service.acquire_task_lease_no_commit(db, tid, runner_id="owner")
            for tid in ids
        )
        db.execute(
            sa.update(Task)
            .where(Task.id.in_(ids))
            .values(last_heartbeat_at=service.utc_now() - timedelta(seconds=20))
        )
    return factory, leases


def times(factory, leases):
    with factory() as db:
        return dict(
            db.execute(
                sa.select(Task.id, Task.last_heartbeat_at).where(
                    Task.id.in_([lease.task_id for lease in leases])
                )
            ).all()
        )


def test_batch_keeps_non_running_owners_and_exact_attempts(owners):
    factory, leases = owners
    before = times(factory, leases)
    with factory() as db:
        originals = {
            t.id: (t.status, t.run_id, t.updated_at)
            for t in db.scalars(sa.select(Task))
        }
    stale = replace(leases[1], attempt_id="stale")
    with factory() as db, db.begin():
        states = service.renew_task_leases_no_commit(db, (leases[0], stale, leases[2]))
    assert states == {leases[0]: "renewed", stale: "lost", leases[2]: "renewed"}
    after = times(factory, leases)
    assert after[leases[0].task_id] > before[leases[0].task_id]
    assert after[leases[1].task_id] == before[leases[1].task_id]
    assert after[leases[2].task_id] > before[leases[2].task_id]
    with factory() as db:
        assert {
            t.id: (t.status, t.run_id, t.updated_at)
            for t in db.scalars(sa.select(Task))
        } == originals


def test_postgresql_skips_locked_owner_without_rolling_back_healthy_rows(
    engine, owners
):
    if engine.dialect.name != "postgresql":
        pytest.skip("PostgreSQL row locks")
    factory, leases = owners
    before = times(factory, leases)
    with factory() as blocker:
        blocker.execute(
            sa.update(Task)
            .where(Task.id == leases[1].task_id)
            .values(updated_at=Task.updated_at)
        )
        with factory() as db, db.begin():
            states = service.renew_task_leases_no_commit(db, leases)
        assert [states[lease] for lease in leases] == ["renewed", "deferred", "renewed"]
        after = times(factory, leases)
        assert after[leases[0].task_id] > before[leases[0].task_id]
        assert after[leases[1].task_id] == before[leases[1].task_id]
        assert after[leases[2].task_id] > before[leases[2].task_id]
        # Uncommitted replacement cannot be mistaken for a committed loss.
        blocker.execute(
            sa.update(Task)
            .where(Task.id == leases[1].task_id)
            .values(lease_attempt_id="successor")
        )
        with factory() as db, db.begin():
            assert (
                service.renew_task_leases_no_commit(db, (leases[1],))[leases[1]]
                == "deferred"
            )
        blocker.commit()
    with factory() as db, db.begin():
        assert (
            service.renew_task_leases_no_commit(db, (leases[1],))[leases[1]] == "lost"
        )


async def until(predicate):
    async with asyncio.timeout(5):
        while not predicate():
            await asyncio.sleep(0.005)


async def test_registry_single_connection_skips_lock_and_retries_subset(
    engine, task_id, monkeypatch
):
    if engine.dialect.name != "postgresql":
        pytest.skip("PostgreSQL row locks")
    monkeypatch.setattr(runtime, "get_task_lease_heartbeat_seconds", lambda: 0.4)
    app_engine = sa.create_engine(
        "postgresql://",
        creator=engine.pool._creator,
        pool_size=1,
        max_overflow=0,
        pool_timeout=0.1,
    )
    factory = sessionmaker(app_engine, expire_on_commit=False)
    admin = sessionmaker(engine)
    with admin() as db, db.begin():
        task = Task(user_id=db.get(Task, task_id).user_id, title="healthy")
        db.add(task)
        db.flush()
        healthy_id = task.id
    registry = runtime.TaskCoordinatorRegistry(factory)
    start_batch = asyncio.Event()
    run_batches = registry._run_heartbeats

    async def start_when_ready():
        await start_batch.wait()
        await run_batches()

    monkeypatch.setattr(registry, "_run_heartbeats", start_when_ready)
    batches = []
    original = registry._renew

    def renew(leases):
        result = original(leases)
        batches.append((leases, result))
        return result

    monkeypatch.setattr(registry, "_renew", renew)
    blocker = admin()
    try:
        blocked = await registry.ensure(task_id)
        healthy = await registry.ensure(healthy_id)
        blocker.execute(
            sa.update(Task).where(Task.id == task_id).values(updated_at=Task.updated_at)
        )
        start_batch.set()
        await until(lambda: not blocked._healthy)
        assert healthy._healthy
        assert healthy._heartbeat_error is None
        assert batches[0][1] == {blocked.lease: "deferred", healthy.lease: "renewed"}
        assert (
            blocked.submit_execution(
                admit=lambda *_: None, execute=AsyncMock(), settle=lambda *_: None
            )
            is None
        )
        blocker.rollback()
        await until(lambda: blocked._healthy)
        assert any(
            leases == (blocked.lease,) and results[blocked.lease] == "renewed"
            for leases, results in batches
        )
        assert blocked.state == runtime.CoordinatorState.ACTIVE
    finally:
        start_batch.set()
        blocker.close()
        await registry.close()
        app_engine.dispose()
    assert registry._heartbeats == {}
    assert registry._heartbeat_runner is None


async def test_close_waits_for_inflight_batch_without_cancelling_other_owner(
    engine, task_id, monkeypatch
):
    monkeypatch.setattr(runtime, "get_task_lease_heartbeat_seconds", lambda: 0.05)
    factory = sessionmaker(engine, expire_on_commit=False)
    with factory() as db, db.begin():
        task = Task(user_id=db.get(Task, task_id).user_id, title="survivor")
        db.add(task)
        db.flush()
        second_id = task.id
    registry = runtime.TaskCoordinatorRegistry(factory)
    start_batch = asyncio.Event()
    run_batches = registry._run_heartbeats

    async def start_when_ready():
        await start_batch.wait()
        await run_batches()

    monkeypatch.setattr(registry, "_run_heartbeats", start_when_ready)
    entered, unblock = Event(), Event()
    survivor_renewed = asyncio.Event()
    original = registry._renew
    calls = 0
    loop = asyncio.get_running_loop()

    def renew(leases):
        nonlocal calls
        calls += 1
        result = original(leases)
        if calls == 1:
            entered.set()
            assert unblock.wait(5)
        elif any(lease.task_id == second_id for lease in leases):
            loop.call_soon_threadsafe(survivor_renewed.set)
        return result

    monkeypatch.setattr(registry, "_renew", renew)
    closing = None
    try:
        first = await registry.ensure(task_id)
        second = await registry.ensure(second_id)
        start_batch.set()
        assert await asyncio.to_thread(entered.wait, 5)
        closing = asyncio.create_task(first.close())
        await until(lambda: first.lease not in registry._heartbeats)
        assert not first._heartbeat_done.done()
        assert not first._close_task.done()
        closing.cancel()
        with factory() as db:
            assert db.get(Task, task_id).lease_attempt_id == first.lease.attempt_id
        unblock.set()
        with pytest.raises(asyncio.CancelledError):
            await closing
        await first.close()
        await asyncio.wait_for(survivor_renewed.wait(), 5)
        assert second._healthy
        with factory() as db:
            assert db.get(Task, task_id).runner_id is None
            assert db.get(Task, second_id).lease_attempt_id == second.lease.attempt_id
    finally:
        start_batch.set()
        unblock.set()
        if closing is not None:
            await asyncio.gather(closing, return_exceptions=True)
        await registry.close()


async def test_postgresql_table_lock_defers_whole_batch(engine, owners):
    if engine.dialect.name != "postgresql":
        pytest.skip("PostgreSQL statement timeout")
    factory, leases = owners
    registry = runtime.TaskCoordinatorRegistry(factory)
    before = times(factory, leases)
    with factory() as blocker:
        blocker.execute(sa.text("LOCK TABLE tasks IN ACCESS EXCLUSIVE MODE"))
        states = await asyncio.to_thread(registry._renew, leases)
        assert all(state == "deferred" for state in states.values())
        blocker.rollback()
    assert times(factory, leases) == before
    assert all(state == "renewed" for state in registry._renew(leases).values())


async def test_batch_commit_failure_is_not_acknowledged(engine, task_id, monkeypatch):
    monkeypatch.setattr(runtime, "get_task_lease_heartbeat_seconds", lambda: 0.02)
    factory = sessionmaker(engine, expire_on_commit=False)
    registry = runtime.TaskCoordinatorRegistry(factory)
    try:
        owner = await registry.ensure(task_id)
        before = times(factory, (owner.lease,))

        def fail_commit(session):
            raise RuntimeError("commit failed")

        sa.event.listen(factory, "before_commit", fail_commit)
        try:
            await until(lambda: owner.state == runtime.CoordinatorState.CLOSED)
            assert owner._recovery_required
            assert not owner._healthy
            assert isinstance(owner._heartbeat_done.exception(), RuntimeError)
            assert times(factory, (owner.lease,)) == before
        finally:
            sa.event.remove(factory, "before_commit", fail_commit)
    finally:
        await registry.close()
