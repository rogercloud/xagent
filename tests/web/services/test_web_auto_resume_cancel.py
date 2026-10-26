"""A person cancels a pending automatic resume from the web UI.

``cancel_auto_resume_for_web_sync`` locks the task, applies
``cancel_pending_auto_resume_no_commit`` for its owner or an admin, retries a
lost compare-and-swap at most three times, commits only a cancel, and then
drops the task's cached responses; the async wrapper publishes the very
message it returns. Runs on SQLite and, when ``XAGENT_TEST_POSTGRES_URL`` is
set, on PostgreSQL; the lock cases are PostgreSQL only and the
compare-and-swap cases SQLite only.
"""

from __future__ import annotations

import logging
import threading
from datetime import datetime, timezone
from typing import Any
from unittest.mock import AsyncMock

import pytest
import sqlalchemy as sa

from tests.web.services.channel_delivery_shared import (
    database_url as database_url_fixture,
)
from tests.web.services.channel_delivery_shared import selected as selected_fixture
from tests.web.services.task_database_shared import engine as engine_fixture
from tests.web.services.test_auto_resume_cancel import (
    _DISPATCHED,
    _interleave,
    _seeded,
    _sqlite_only,
    _state,
)
from tests.web.services.test_channel_reply_hold import (
    CID,
    PV,
    RUN,
    _postgresql_only,
    _record,
    _wait_for_a_lock_waiter,
)
from tests.web.services.test_channel_reply_hold import channel as channel_fixture
from tests.web.services.test_channel_reply_hold import factory as factory_fixture
from tests.web.services.test_channel_reply_hold import (
    short_lock_timeout as short_lock_timeout_fixture,
)
from tests.web.services.test_task_auto_resume import _trigger
from xagent.web.models.database import get_session_local
from xagent.web.models.task import Task, TaskStatus
from xagent.web.models.task_auto_recovery import (
    TaskAutoRecovery,
    TaskAutoRecoveryState,
    TaskRecoveryEvent,
)
from xagent.web.models.task_channel_delivery import TaskChannelDelivery
from xagent.web.models.task_command import TaskExecutionCommand
from xagent.web.models.trigger import TriggerRun, TriggerRunStatus
from xagent.web.models.user import User
from xagent.web.services import channel_delivery, task_auto_resume
from xagent.web.services import web_auto_resume_cancel as web_cancel
from xagent.web.services.execution_result_projection import interrupted_channel_result
from xagent.web.services.task_auto_resume import (
    AutoResumeCancel,
    cancel_pending_auto_resume_no_commit,
    check_auto_resume_claim_sync,
    select_next_candidate_for_update,
)
from xagent.web.services.task_execution_controller import TaskControlState
from xagent.web.services.task_lease_service import utc_now
from xagent.web.services.web_auto_resume_cancel import (
    WebAutoResumeCancelOutcome as Outcome,
)
from xagent.web.services.web_auto_resume_cancel import (
    cancel_auto_resume_for_web,
    cancel_auto_resume_for_web_sync,
)

engine = engine_fixture
factory = factory_fixture
short_lock_timeout = short_lock_timeout_fixture
database_url = database_url_fixture
selected = selected_fixture
channel = channel_fixture

STOPPED = {"status": "stopped", "stop_reason": "cancelled"}


@pytest.fixture(autouse=True)
def _quiet(monkeypatch):
    monkeypatch.setattr(
        task_auto_resume, "notify_task_command_dispatcher", lambda: None
    )
    monkeypatch.delenv("XAGENT_TASK_AUTO_RESUME_CHANNEL_WINDOW_SECONDS", raising=False)


@pytest.fixture
def invalidated(monkeypatch) -> list[int]:
    calls: list[int] = []
    monkeypatch.setattr(web_cancel, "invalidate_task_cache_best_effort", calls.append)
    return calls


def _owner(factory, tid: int) -> int:
    with factory() as db:
        return int(db.get(Task, tid).user_id)


def _cancel(factory, tid: int, *, actor: int | None = None, admin: bool = False, **kw):
    return cancel_auto_resume_for_web_sync(
        task_id=tid,
        actor_user_id=_owner(factory, tid) if actor is None else actor,
        actor_is_admin=admin,
        **kw,
    )


def _events(factory, tid: int) -> list[dict[str, Any]]:
    """The details of ``tid``'s ``skipped_user_intent`` events."""
    with factory() as db:
        return [
            dict(event.detail)
            for event in db.scalars(
                sa.select(TaskRecoveryEvent).where(
                    TaskRecoveryEvent.task_id == tid,
                    TaskRecoveryEvent.event == "skipped_user_intent",
                )
            )
        ]


# --------------------------------------------------------------------------
# W1. A pending resume is cancelled; the answer reads as GET reads it
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("seed", "fence"),
    [
        pytest.param({}, PV, id="scheduled"),
        pytest.param({**_DISPATCHED, "command_status": "pending"}, PV, id="pending"),
        pytest.param(
            {**_DISPATCHED, "command_status": "processing"}, PV, id="processing"
        ),
        pytest.param(
            {**_DISPATCHED, "command_status": "pending", "version": PV + 2},
            PV + 2,
            id="retry_after_a_rollback",
        ),
    ],
)
def test_a_pending_resume_is_cancelled(factory, invalidated, monkeypatch, seed, fence):
    def no_channel(*_args, **_kwargs):
        raise AssertionError("not a channel task")

    monkeypatch.setattr(
        web_cancel, "settle_channel_reply_after_cancel_no_commit", no_channel
    )
    tid = _seeded(factory, **seed)
    owner = _owner(factory, tid)

    result = _cancel(factory, tid)

    assert result.outcome is Outcome.CANCELLED
    assert _state(factory, tid) == (
        fence + 1,
        ("manual", "user_cancelled"),
        fence + 1,
        ["skipped_user_intent"],
    )
    (event,) = _events(factory, tid)
    assert event["actor_user_id"] == owner
    assert event["at"] == "web_cancel"
    (notice,) = result.notices
    assert (notice.task_id, notice.run_id, notice.state_version) == (
        tid,
        RUN,
        fence + 1,
    )
    assert result.task is not None
    assert {k: result.task[k] for k in ("status", "control_state", "run_id")} == {
        "status": "paused",
        "control_state": "paused",
        "run_id": RUN,
    }
    assert result.task["state_version"] == fence + 1
    view = result.task["auto_recovery"]
    assert (view["run_id"], view["state_version"], view["auto_resume"]) == (
        RUN,
        fence + 1,
        notice.auto_resume,
    )
    assert notice.auto_resume.items() >= STOPPED.items()
    assert invalidated == [tid]


def test_a_cancel_leaves_the_trigger_run_running(factory, invalidated):
    with factory() as db:
        tid = _seeded(factory)
        run_id, _ = _trigger(db, db.get(User, db.get(Task, tid).user_id), tid)
        db.commit()

    assert _cancel(factory, tid).outcome is Outcome.CANCELLED

    with factory() as db:
        assert db.get(TriggerRun, run_id).status == TriggerRunStatus.RUNNING.value


# --------------------------------------------------------------------------
# W2. Nothing to cancel: an outcome, never a write
# --------------------------------------------------------------------------


def _assert_unwritten(factory, tid: int, before, invalidated) -> None:
    assert _state(factory, tid) == before
    assert invalidated == []


def test_a_second_cancel_reads_already_cancelled(factory, invalidated):
    tid = _seeded(factory)
    assert _cancel(factory, tid).outcome is Outcome.CANCELLED
    before, invalidated[:] = _state(factory, tid), []

    result = _cancel(factory, tid)

    assert result.outcome is Outcome.ALREADY_CANCELLED
    assert result.notices == ()
    assert result.task["auto_recovery"]["auto_resume"].items() >= STOPPED.items()
    _assert_unwritten(factory, tid, before, invalidated)


def test_a_hand_off_reads_resume_in_progress(factory, invalidated):
    tid = _seeded(
        factory,
        **_DISPATCHED,
        command_status="processing",
        control_state="resume_requested",
        version=PV + 1,
    )
    before = _state(factory, tid)

    result = _cancel(factory, tid)

    assert result.outcome is Outcome.RESUME_IN_PROGRESS
    assert result.task["control_state"] == "resume_requested"
    _assert_unwritten(factory, tid, before, invalidated)


@pytest.mark.parametrize(
    "seed",
    [
        pytest.param(
            {"status": TaskStatus.RUNNING, "control_state": "running"}, id="running"
        ),
        pytest.param({"state": None}, id="no_row"),
        pytest.param({"state": "exhausted"}, id="exhausted"),
        pytest.param(
            {"state": "manual", "state_detail": "auto_resume_disabled"},
            id="disabled_by_switch",
        ),
        pytest.param(
            {**_DISPATCHED, "command_status": "completed", "version": PV + 2},
            id="rolled_back_command_ended",
        ),
    ],
)
def test_nothing_pending_reads_not_pending(factory, invalidated, seed):
    tid = _seeded(factory, **seed)
    before = _state(factory, tid)

    result = _cancel(factory, tid)

    assert result.outcome is Outcome.NOT_PENDING
    assert result.task is not None
    _assert_unwritten(factory, tid, before, invalidated)


def test_another_run_than_the_one_seen_is_not_cancelled(
    factory, invalidated, monkeypatch
):
    def unexpected(*_args, **_kwargs):
        raise AssertionError("the hook must not run")

    monkeypatch.setattr(web_cancel, "cancel_pending_auto_resume_no_commit", unexpected)
    tid = _seeded(factory)
    before = _state(factory, tid)

    result = _cancel(factory, tid, expected_run_id="an-earlier-run")

    assert result.outcome is Outcome.NOT_PENDING
    assert result.task["run_id"] == RUN
    _assert_unwritten(factory, tid, before, invalidated)


def test_the_run_seen_is_cancelled(factory, invalidated):
    tid = _seeded(factory)

    assert _cancel(factory, tid, expected_run_id=RUN).outcome is Outcome.CANCELLED


# --------------------------------------------------------------------------
# W3. Who may cancel; which tasks
# --------------------------------------------------------------------------


def _other_user(factory) -> int:
    with factory() as db:
        user = User(username="someone-else", password_hash="x")
        db.add(user)
        db.commit()
        return int(user.id)


def test_another_user_reads_missing(factory, invalidated):
    tid = _seeded(factory)
    before = _state(factory, tid)

    result = _cancel(factory, tid, actor=_other_user(factory))

    assert result.outcome is Outcome.MISSING
    assert result.task is None
    _assert_unwritten(factory, tid, before, invalidated)


def test_a_missing_task_reads_missing(factory, invalidated):
    tid = _seeded(factory)

    assert _cancel(factory, tid + 1000, actor=1).outcome is Outcome.MISSING


def test_an_admin_cancels_another_users_task_and_is_recorded(factory, invalidated):
    tid = _seeded(factory)
    admin = _other_user(factory)

    result = _cancel(factory, tid, actor=admin, admin=True)

    assert result.outcome is Outcome.CANCELLED
    (event,) = _events(factory, tid)
    assert event["actor_user_id"] == admin


def test_a_status_only_task_is_not_cancellable(factory, invalidated):
    tid = _seeded(factory)
    with factory() as db:
        db.get(Task, tid).source = "sdk"
        db.commit()
    before = _state(factory, tid)

    result = _cancel(factory, tid)

    assert result.outcome is Outcome.NOT_CANCELLABLE
    assert result.task is None
    _assert_unwritten(factory, tid, before, invalidated)


# --------------------------------------------------------------------------
# W4. A lost compare-and-swap (SQLite) is read again, three times at most
# --------------------------------------------------------------------------


def test_a_dispatch_that_commits_first_is_retried_then_cancelled(
    factory, invalidated, monkeypatch
):
    _sqlite_only()
    tid = _seeded(factory)
    written: list[int] = []

    def dispatch_once(other):
        if written:
            return
        written.append(1)
        row = other.get(TaskAutoRecovery, tid)
        row.state, row.last_command_id = "dispatched", CID

    _interleave(monkeypatch, dispatch_once)

    result = _cancel(factory, tid)

    assert result.outcome is Outcome.CANCELLED
    (event,) = _events(factory, tid)
    assert event["from"] == "dispatched"


def test_a_cancel_that_keeps_racing_reads_raced(
    factory, invalidated, monkeypatch, caplog
):
    tid = _seeded(factory)
    before = _state(factory, tid)
    calls: list[int] = []

    def always_raced(db, task, *, at, actor_user_id=None):
        calls.append(1)
        # A write staged before the race must not be kept.
        db.execute(
            sa.update(Task)
            .where(Task.id == task.id)
            .values(state_version=Task.state_version + 7)
        )
        return AutoResumeCancel(cancelled=False, raced=True)

    monkeypatch.setattr(
        web_cancel, "cancel_pending_auto_resume_no_commit", always_raced
    )

    with caplog.at_level(logging.WARNING):
        result = _cancel(factory, tid)

    assert result.outcome is Outcome.RACED
    assert result.task is None
    assert len(calls) == 3
    assert "kept losing a race" in caplog.text
    _assert_unwritten(factory, tid, before, invalidated)


def test_each_outcome_is_counted(factory, invalidated, monkeypatch):
    counted: list[tuple[str, dict[str, Any]]] = []
    monkeypatch.setattr(
        web_cancel,
        "increment_counter",
        lambda name, attributes=None: counted.append((name, attributes or {})),
    )
    tid = _seeded(factory)
    _cancel(factory, tid)
    _cancel(factory, tid)
    _cancel(factory, tid, actor=_other_user(factory))

    assert counted == [
        ("xagent.web.auto_resume_cancel", {"outcome": outcome})
        for outcome in ("cancelled", "already_cancelled", "missing")
    ]


# --------------------------------------------------------------------------
# W5. The async wrapper: published after the commit, the same message
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_the_message_returned_is_the_one_published_after_the_commit(
    factory, invalidated, monkeypatch
):
    tid = _seeded(factory)
    published: list[tuple[dict[str, Any], int, Any]] = []

    async def publish(message, task_id):
        with factory() as db:
            row = db.get(TaskAutoRecovery, task_id)
            published.append((message, task_id, (row.state, row.state_detail)))

    monkeypatch.setattr(web_cancel, "publish_task_event", publish)

    result = await cancel_auto_resume_for_web(
        task_id=tid, actor_user_id=_owner(factory, tid), actor_is_admin=False
    )

    ((message, task_id, committed),) = published
    assert (task_id, committed) == (tid, ("manual", "user_cancelled"))
    assert result.messages == (message,)
    assert message["type"] == "task_recovery_update"
    assert message["state_version"] == PV + 1

    again = await cancel_auto_resume_for_web(
        task_id=tid, actor_user_id=_owner(factory, tid), actor_is_admin=False
    )
    assert (again.outcome, again.messages, len(published)) == (
        Outcome.ALREADY_CANCELLED,
        (),
        1,
    )


@pytest.mark.asyncio
async def test_a_failed_broadcast_keeps_the_cancel(factory, invalidated, monkeypatch):
    tid = _seeded(factory)
    monkeypatch.setattr(
        web_cancel, "publish_task_event", AsyncMock(side_effect=RuntimeError("down"))
    )

    result = await cancel_auto_resume_for_web(
        task_id=tid, actor_user_id=_owner(factory, tid), actor_is_admin=False
    )

    assert result.outcome is Outcome.CANCELLED
    assert _state(factory, tid)[1] == ("manual", "user_cancelled")


# --------------------------------------------------------------------------
# W6. A channel task: its held reply is "stopped", deliverable now
# --------------------------------------------------------------------------


def _aware(value: datetime) -> datetime:
    return value if value.tzinfo else value.replace(tzinfo=timezone.utc)


@pytest.mark.asyncio
async def test_a_channel_tasks_held_reply_is_released_as_interrupted(
    channel, monkeypatch
):
    _record(channel)
    sender = AsyncMock()
    # The hold: a delivery attempt leaves the reply pending for 30s.
    assert not await channel_delivery.deliver_channel_result(channel.command_id, sender)
    monkeypatch.setattr(web_cancel, "invalidate_task_cache_best_effort", lambda _: None)
    with get_session_local()() as db:
        owner = int(db.get(Task, channel.task_id).user_id)

    result = cancel_auto_resume_for_web_sync(
        task_id=channel.task_id, actor_user_id=owner, actor_is_admin=False
    )

    assert result.outcome is Outcome.CANCELLED
    with get_session_local()() as db:
        start = db.get(TaskExecutionCommand, channel.command_id)
        assert (start.result or {}).get(
            "channel_result"
        ) == interrupted_channel_result()
        delivery = db.get(TaskChannelDelivery, channel.command_id)
        assert delivery.status == "pending"
        assert _aware(delivery.available_at) <= datetime.now(timezone.utc)
    assert await channel_delivery.deliver_channel_result(channel.command_id, sender)
    assert sender.await_args.args[1] == interrupted_channel_result()


# --------------------------------------------------------------------------
# P1-P4. PostgreSQL: the sweeper, the claim guard, a resume by hand
# --------------------------------------------------------------------------


def _cancel_in_thread(factory, tid: int, outcome: dict[str, Any]) -> threading.Thread:
    owner = _owner(factory, tid)

    def run() -> None:
        try:
            outcome["result"] = cancel_auto_resume_for_web_sync(
                task_id=tid, actor_user_id=owner, actor_is_admin=False
            )
        except BaseException as exc:  # pragma: no cover - reported below
            outcome["error"] = exc

    return threading.Thread(target=run)


def _joined(thread: threading.Thread, outcome: dict[str, Any]) -> Any:
    thread.join(timeout=10)
    assert not thread.is_alive()
    assert "error" not in outcome, outcome.get("error")
    return outcome["result"]


def test_postgresql_cancel_waits_for_a_dispatch_then_cancels_it(
    factory, invalidated, short_lock_timeout
):
    _postgresql_only()
    tid = _seeded(factory)
    outcome: dict[str, Any] = {}
    thread = _cancel_in_thread(factory, tid, outcome)
    sweeper = get_session_local()()
    try:
        now = utc_now()
        assert select_next_candidate_for_update(
            sweeper, TaskAutoRecoveryState.SCHEDULED, now=now
        ) == (sweeper.get(TaskAutoRecovery, tid).next_attempt_at, tid)
        assert task_auto_resume._process_due_candidate_no_commit(
            sweeper, tid, now=now, can_dispatch=True
        ).staged
        thread.start()
        with get_session_local()() as monitor:
            _wait_for_a_lock_waiter(monitor)
        sweeper.commit()
    finally:
        sweeper.close()

    assert _joined(thread, outcome).outcome is Outcome.CANCELLED
    (event,) = _events(factory, tid)
    assert event["from"] == "dispatched"


def _while_cancelling(monkeypatch, during) -> None:
    """Run ``during`` inside the cancel's transaction, after the hook."""

    def hook_then(db, task, **kwargs):
        result = cancel_pending_auto_resume_no_commit(db, task, **kwargs)
        during()
        return result

    monkeypatch.setattr(web_cancel, "cancel_pending_auto_resume_no_commit", hook_then)


def test_postgresql_sweeper_skips_a_task_a_cancel_holds(
    factory, invalidated, monkeypatch, short_lock_timeout
):
    _postgresql_only()
    tid = _seeded(factory)
    skipped: list[Any] = []

    def sweep() -> None:
        with get_session_local()() as sweeper:
            skipped.append(
                select_next_candidate_for_update(
                    sweeper, TaskAutoRecoveryState.SCHEDULED, now=utc_now()
                )
            )
            sweeper.rollback()

    _while_cancelling(monkeypatch, sweep)

    assert _cancel(factory, tid).outcome is Outcome.CANCELLED
    assert skipped == [None]


def test_postgresql_claim_guard_waits_for_a_cancel_and_skips(
    factory, invalidated, monkeypatch, short_lock_timeout
):
    _postgresql_only()
    tid = _seeded(factory, **_DISPATCHED, command_status="processing")
    outcome: dict[str, Any] = {}

    def guard() -> None:
        try:
            outcome["result"] = check_auto_resume_claim_sync(
                task_id=tid,
                command_id=CID,
                auto_resume={"expected_run_id": RUN, "expected_state_version": PV},
                status=TaskStatus.PAUSED,
                control_state=TaskControlState.PAUSED.value,
                run_id=RUN,
                state_version=PV,
                attempt_count=1,
            )
        except BaseException as exc:  # pragma: no cover - reported below
            outcome["error"] = exc

    thread = threading.Thread(target=guard)

    def start_the_guard() -> None:
        thread.start()
        with get_session_local()() as monitor:
            _wait_for_a_lock_waiter(monitor)

    _while_cancelling(monkeypatch, start_the_guard)

    assert _cancel(factory, tid).outcome is Outcome.CANCELLED
    assert _joined(thread, outcome).proceed is False


def test_postgresql_a_resume_by_hand_handing_off_first_wins(
    factory, invalidated, short_lock_timeout
):
    _postgresql_only()
    tid = _seeded(factory)
    outcome: dict[str, Any] = {}
    thread = _cancel_in_thread(factory, tid, outcome)
    resuming = get_session_local()()
    try:
        task = resuming.execute(
            sa.select(Task).where(Task.id == tid).with_for_update()
        ).scalar_one()
        # The RESUME's hand-off: RESUME_REQUESTED at the next version.
        task.control_state = TaskControlState.RESUME_REQUESTED.value
        task.state_version = PV + 1
        resuming.flush()
        thread.start()
        with get_session_local()() as monitor:
            _wait_for_a_lock_waiter(monitor)
        resuming.commit()
    finally:
        resuming.close()

    result = _joined(thread, outcome)
    assert result.outcome is Outcome.RESUME_IN_PROGRESS
    assert _state(factory, tid) == (PV + 1, ("scheduled", None), PV, [])
    assert invalidated == []
