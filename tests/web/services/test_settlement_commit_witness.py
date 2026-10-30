"""Settlement COMMITs confirmed by their result fact's witness.

A settlement makes a fenced state transition, so a plain rerun after a lost
acknowledgement whose COMMIT applied would miss its own fence and report
nothing. Faults are raised at ``Session.commit`` (after or instead of the
real COMMIT) so each test knows whether the settlement applied; the read-back
must tell the two apart. Runs on SQLite and, when ``XAGENT_TEST_POSTGRES_URL``
is set, on PostgreSQL.
"""

from __future__ import annotations

import logging

import pytest
from sqlalchemy.exc import OperationalError

from tests.web.services.test_execution_event_signals import (
    metric_reader as metric_reader_fixture,
)
from tests.web.services.test_execution_event_unknown_outcomes import (
    _capture as _capture_fixture,
)
from tests.web.services.test_execution_event_unknown_outcomes import (
    _deadlock,
    assert_none,
    assert_one,
)
from tests.web.services.test_persistence_failure_pause import (
    AUTO_RESUME_SWITCH,
    PAUSE_SWITCH,
    RUN_ERROR,
    _assert_paused,
    _assistant_lines,
    _checkpoint,
    _persistence_interruption,
    _settle,
    _settled_facts,
    _start_run,
)
from tests.web.services.test_task_execution_event_writer import (
    canonical as canonical_fixture,
)
from tests.web.services.test_task_execution_event_writer import engine as engine_fixture
from tests.web.services.test_task_execution_event_writer import (
    task_id as task_id_fixture,
)
from tests.web.services.test_trace_fact_commit_retry import (
    fail_commits,
    invalidated,
)
from xagent.web.models.task import Task, TaskStatus
from xagent.web.services import db_write_retry
from xagent.web.services import managed_task_lease as managed
from xagent.web.services import task_execution_event_writer
from xagent.web.services import task_orchestrator as orchestrator
from xagent.web.services.task_execution_event_store import (
    lock_task_execution_events_no_commit,
)

canonical = canonical_fixture
engine = engine_fixture
task_id = task_id_fixture
metric_reader = metric_reader_fixture
_capture = _capture_fixture


@pytest.fixture(autouse=True)
def _settlement_retries(monkeypatch):
    monkeypatch.setenv("XAGENT_DB_WRITE_RETRY_ATTEMPTS", "3")
    monkeypatch.setenv("XAGENT_DB_WRITE_RETRY_BASE_DELAY_MS", "0")
    monkeypatch.setenv("XAGENT_DB_WRITE_RETRY_MAX_DELAY_MS", "0")
    # The pause branch decides as these suites do: pause, leave it manual.
    monkeypatch.setenv(PAUSE_SWITCH, "true")
    monkeypatch.setenv(AUTO_RESUME_SWITCH, "false")


@pytest.fixture
def settle_attempts(monkeypatch):
    """Count the transactions ``settle_task_lease_isolated`` runs."""
    calls: list[int] = []
    original = orchestrator._settle_task_lease_attempt

    def attempt(*args, **kwargs):
        calls.append(kwargs["attempt"].number)
        return original(*args, **kwargs)

    monkeypatch.setattr(orchestrator, "_settle_task_lease_attempt", attempt)
    return calls


@pytest.fixture
def read_backs(monkeypatch):
    """Record each witness read-back and its verdict."""
    verdicts: list[bool] = []
    original = task_execution_event_writer.fact_witness_committed

    def read_back(witness):
        verdict = original(witness)
        verdicts.append(verdict)
        return verdict

    monkeypatch.setattr(
        task_execution_event_writer, "fact_witness_committed", read_back
    )
    return verdicts


@pytest.fixture
def settles(monkeypatch):
    """Match the COMMIT of a transaction that staged a result fact."""
    staging: set[int] = set()
    stage = task_execution_event_writer.stage_result_fact_no_commit

    def staged(db, *args, **kwargs):
        staging.add(id(db))
        return stage(db, *args, **kwargs)

    monkeypatch.setattr(
        task_execution_event_writer, "stage_result_fact_no_commit", staged
    )

    def matches(db) -> bool:
        if id(db) not in staging:
            return False
        staging.discard(id(db))
        return True

    return matches


def _retries(metric_reader, site: str) -> dict[str, int]:
    """``outcome -> count`` of the retry counter for ``site``."""
    counts: dict[str, int] = {}
    data = metric_reader.get_metrics_data()
    for resource in data.resource_metrics if data else ():
        for scope in resource.scope_metrics:
            for metric in scope.metrics:
                if metric.name != "xagent.execution_event.write.retries":
                    continue
                for point in metric.data.data_points:
                    if point.attributes.get("site") == site:
                        outcome = str(point.attributes["outcome"])
                        counts[outcome] = counts.get(outcome, 0) + int(point.value)
    return counts


def _failed_once(factory, tid: int, lease) -> None:
    """One FAILED settlement with one failure transcript line and one fact."""
    with factory() as db:
        task = db.get(Task, tid)
        assert task.status == TaskStatus.FAILED
        assert task.error_message == RUN_ERROR
        assert task.runner_id is None
    (fact,) = _settled_facts(factory, tid)
    assert fact["status"] == "failed"
    assert len(_assistant_lines(factory, tid)) == 1


# ----------------------------------------------------------- the error branch


def test_a_lost_settlement_ack_whose_commit_applied_is_confirmed(
    canonical, settles, monkeypatch, settle_attempts, read_backs, caplog, metric_reader
):
    factory, tid = canonical
    lease = _start_run(factory, tid)
    raised = fail_commits(monkeypatch, settles, committed=True)

    settled, report = _settle(lease)

    assert len(raised) == 1
    # Confirmed by the read-back, not by a rerun that would miss its fence.
    assert settled
    assert settle_attempts == [1]
    assert read_backs == [True]
    _failed_once(factory, tid, lease)
    with factory() as db:
        task = db.get(Task, tid)
        assert report.control_state["state_version"] == task.state_version
        assert report.control_state["run_id"] == lease.run_id
    assert_one(
        caplog,
        metric_reader,
        kind="settlement_ack_reconciled",
        task=tid,
        run_id=lease.run_id,
        ref=None,
        storage="v2",
        level=logging.INFO,
    )


def test_without_the_read_back_a_lost_settlement_ack_is_not_confirmed(
    canonical, settles, monkeypatch, settle_attempts
):
    """Reverse check of the test above: with the read-back answering
    "absent", the rerun misses its own fence and reports nothing."""
    factory, tid = canonical
    lease = _start_run(factory, tid)
    monkeypatch.setattr(
        task_execution_event_writer, "fact_witness_committed", lambda _w: False
    )
    fail_commits(monkeypatch, settles, committed=True)

    settled, report = _settle(lease)

    assert not settled
    assert report.control_state == {}
    assert settle_attempts == [1, 2]
    _failed_once(factory, tid, lease)


def test_a_deadlocked_settlement_commit_is_rerun_once(
    canonical, settles, monkeypatch, settle_attempts, read_backs, caplog, metric_reader
):
    factory, tid = canonical
    lease = _start_run(factory, tid)
    raised = fail_commits(monkeypatch, settles, committed=False, error=_deadlock)

    settled, report = _settle(lease)

    assert len(raised) == 1
    assert settled
    assert settle_attempts == [1, 2]
    assert read_backs == [False]
    _failed_once(factory, tid, lease)
    assert report.control_state["run_id"] == lease.run_id
    # A rerun is a retry, not an unknown outcome.
    assert_none(caplog, metric_reader)
    assert _retries(metric_reader, "settlement") == {"retried": 1, "recovered": 1}


def test_exhausted_settlement_reruns_raise_the_last_error(
    canonical, settles, monkeypatch, settle_attempts
):
    factory, tid = canonical
    lease = _start_run(factory, tid)
    raised = fail_commits(
        monkeypatch, settles, committed=False, times=3, error=_deadlock
    )

    with pytest.raises(OperationalError) as caught:
        _settle(lease)

    assert caught.value is raised[-1]
    assert settle_attempts == [1, 2, 3]
    with factory() as db:
        task = db.get(Task, tid)
        # Retained for TTL recovery, as before.
        assert task.status == TaskStatus.RUNNING
        assert (task.runner_id, task.run_id) == (lease.runner_id, lease.run_id)
    assert _settled_facts(factory, tid) == []


@pytest.mark.parametrize("committed", [True, False], ids=["applied", "rolled-back"])
def test_an_undecidable_settlement_read_back_raises_and_reports_unknown(
    canonical, settles, monkeypatch, settle_attempts, caplog, metric_reader, committed
):
    factory, tid = canonical
    lease = _start_run(factory, tid)

    def read_back_fails(_witness):
        raise OperationalError("SELECT", {}, Exception("connection refused"))

    monkeypatch.setattr(
        task_execution_event_writer, "fact_witness_committed", read_back_fails
    )
    raised = fail_commits(monkeypatch, settles, committed=committed)

    with pytest.raises(OperationalError) as caught:
        _settle(lease)

    assert caught.value is raised[0]
    assert settle_attempts == [1]
    assert len(_settled_facts(factory, tid)) == (1 if committed else 0)
    assert_one(
        caplog,
        metric_reader,
        kind="settlement_ack_unresolved",
        task=tid,
        run_id=lease.run_id,
        ref=None,
        storage="v2",
    )


def test_a_legacy_settlement_is_not_retried(canonical, monkeypatch, settle_attempts):
    """No fact witness: the commit error propagates as before."""
    factory, tid = canonical
    lease = _start_run(factory, tid, conversation_storage_version=1)
    raised = fail_commits(monkeypatch, lambda _db: True, committed=False)

    with pytest.raises(OperationalError) as caught:
        _settle(lease)

    assert caught.value is raised[0]
    assert settle_attempts == [1]


# ----------------------------------------------------------- the pause branch


@pytest.mark.parametrize("committed", [True, False], ids=["lost-ack", "rolled-back"])
def test_a_failed_pause_commit_is_reconciled(
    canonical, settles, monkeypatch, settle_attempts, read_backs, committed
):
    factory, tid = canonical
    lease = _start_run(factory, tid)
    _checkpoint(factory, tid, lease.run_id)
    fail_commits(monkeypatch, settles, committed=committed)

    settled, report = _settle(lease, interruption=_persistence_interruption())

    assert settled
    assert report.paused_for is not None
    assert report.paused_for.value == "persistence_failure"
    assert report.control_state["status"] == "paused"
    assert report.recovery is not None
    assert settle_attempts == ([1] if committed else [1, 2])
    assert read_backs == [committed]
    row = _assert_paused(factory, tid, lease, reason="persistence_failure")
    assert report.control_state["state_version"] == row.paused_state_version
    (fact,) = _settled_facts(factory, tid)
    assert fact["status"] == "paused"
    assert _assistant_lines(factory, tid) == []


# ------------------------------------------------------- the managed _sync path


@pytest.mark.parametrize("committed", [True, False], ids=["lost-ack", "rolled-back"])
@pytest.mark.parametrize("path", ["finalize", "release"])
def test_a_failed_managed_settlement_commit_is_reconciled(
    canonical, settles, monkeypatch, read_backs, committed, path
):
    factory, tid = canonical
    lease = _start_run(factory, tid)
    fail_commits(monkeypatch, settles, committed=committed)

    if path == "finalize":
        settled = managed._finalize_managed_task_lease_result_sync(
            lease,
            status=TaskStatus.COMPLETED,
            assistant_content="done",
            execution_result={"success": True, "output": "done"},
        )
    else:
        settled = managed._release_managed_task_lease_sync(lease)

    assert settled
    assert read_backs == [committed]
    expected = TaskStatus.COMPLETED if path == "finalize" else TaskStatus.FAILED
    with factory() as db:
        task = db.get(Task, tid)
        assert task.status == expected
        assert task.runner_id is None
    (fact,) = _settled_facts(factory, tid)
    assert fact["status"] == expected.value
    assert len(_assistant_lines(factory, tid)) == 1


def test_a_managed_lost_ack_without_the_read_back_is_not_confirmed(
    canonical, settles, monkeypatch
):
    """Reverse check: rerun after an applied COMMIT misses the lease fence."""
    factory, tid = canonical
    lease = _start_run(factory, tid)
    monkeypatch.setattr(
        task_execution_event_writer, "fact_witness_committed", lambda _w: False
    )
    fail_commits(monkeypatch, settles, committed=True)

    assert not managed._finalize_managed_task_lease_result_sync(
        lease, status=TaskStatus.COMPLETED, assistant_content="done"
    )
    assert len(_settled_facts(factory, tid)) == 1


# --------------------------------------------------------------- PostgreSQL


@pytest.fixture
def abandoned_task_lock(canonical, engine, monkeypatch):
    """Hold the task row lock after the failed COMMIT, as a dropped
    connection's abandoned transaction can."""
    if engine.dialect.name != "postgresql":
        pytest.skip("lock_timeout is a PostgreSQL setting")
    factory, tid = canonical
    monkeypatch.setattr(
        task_execution_event_writer, "_WITNESS_READ_BACK_LOCK_TIMEOUT", "200ms"
    )
    monkeypatch.setattr(db_write_retry, "_RERUN_LOCK_TIMEOUT", "200ms")
    holder = factory()
    yield lambda: lock_task_execution_events_no_commit(holder, tid)
    holder.rollback()
    holder.close()


def test_a_read_back_blocked_by_an_abandoned_lock_is_undecided(
    canonical,
    settles,
    abandoned_task_lock,
    monkeypatch,
    settle_attempts,
    caplog,
    metric_reader,
):
    """The read-back waits a bounded time for the task row lock, then the
    settlement fails as unresolved instead of guessing."""
    factory, tid = canonical
    lease = _start_run(factory, tid)
    raised = fail_commits(
        monkeypatch,
        settles,
        committed=False,
        error=invalidated,
        between=abandoned_task_lock,
    )

    with pytest.raises(OperationalError) as caught:
        _settle(lease)

    assert caught.value is raised[0]
    assert settle_attempts == [1]
    assert_one(
        caplog,
        metric_reader,
        kind="settlement_ack_unresolved",
        task=tid,
        run_id=lease.run_id,
        ref=None,
        storage="v2",
    )
