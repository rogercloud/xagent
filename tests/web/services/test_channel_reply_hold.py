"""A channel reply held while an automatic resume of its run is due.

The hold is derived: ``_read_channel_result`` asks
``channel_reply_waits_for_auto_resume`` when the task rests PAUSED, the
shared channel leaf leaves the START without a ``channel_result`` only when
it recorded ``scheduled``, and the resumed run stores its answer on the
START. A new channel turn accepted before the resume hands off supersedes
it. Channel tasks are not scheduled yet, so every case seeds the recovery
row (or forces the recorded state) directly. Runs on SQLite and, when
``XAGENT_TEST_POSTGRES_URL`` is set, on PostgreSQL; the lock-order cases are
PostgreSQL only.
"""

from __future__ import annotations

import asyncio
import dataclasses
import threading
import time
import uuid
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
import sqlalchemy as sa
from sqlalchemy.orm import Session, sessionmaker

from tests.web.services.channel_delivery_shared import (
    database_url as database_url_fixture,
)
from tests.web.services.channel_delivery_shared import (
    expire_claim,
)
from tests.web.services.channel_delivery_shared import selected as selected_fixture
from tests.web.services.task_database_shared import engine as engine_fixture
from tests.web.services.task_database_shared import task_id as task_id_fixture
from tests.web.services.test_persistence_failure_pause import (
    _late_bound_sessions as _late_bound_sessions_fixture,
)
from tests.web.services.test_persistence_failure_pause import (
    _prepare_run,
    _projected_task,
)
from tests.web.services.test_provider_failure_pause import LLM_RESULT, _settle_channel
from tests.web.services.test_task_execution_event_writer import (
    canonical as canonical_fixture,
)
from tests.web.services.test_task_lease_expiry_interruption import _fail_recording
from xagent.web.models import database
from xagent.web.models.database import Base, get_engine, get_session_local
from xagent.web.models.task import Task, TaskStatus
from xagent.web.models.task_auto_recovery import TaskAutoRecovery, TaskRecoveryEvent
from xagent.web.models.task_channel_delivery import TaskChannelDelivery
from xagent.web.models.task_command import TaskExecutionCommand
from xagent.web.models.user import User
from xagent.web.models.user_channel import UserChannel

# Imported here, not lazily: a module first imported while ``canonical``
# patches ``database.get_session_local`` would keep that patched factory.
from xagent.web.services import agent_service_manager, channel_delivery
from xagent.web.services import shared_channel_execution as shared
from xagent.web.services import task_auto_recovery, task_auto_resume
from xagent.web.services import task_command_execution as commands
from xagent.web.services import task_execution as task_execution_service
from xagent.web.services.execution_result_projection import (
    interrupted_channel_result,
    project_execution_result_for_channel,
)
from xagent.web.services.task_auto_recovery_policy import staleness_window_seconds
from xagent.web.services.task_auto_resume import (
    CHANNEL_HOLD_CLOCK_GRACE_SECONDS,
    auto_resume_command_id,
    channel_reply_waits_for_auto_resume,
    check_auto_resume_claim_sync,
    select_next_candidate_for_update,
    supersede_pending_auto_resume_no_commit,
)
from xagent.web.services.task_command_execution import ResumeCommandOutcome
from xagent.web.services.task_command_transport import _claimable_query
from xagent.web.services.task_execution import (
    ResumeReservationOutcome,
    _finalize_resumed_task,
    _PreparedTaskFileOutputs,
)
from xagent.web.services.task_interaction_close import ActiveInteractionAbsent
from xagent.web.services.task_lease_service import utc_now
from xagent.web.services.task_orchestrator import TaskTurnPayload
from xagent.web.services.task_settlement_report import SettlementReport

engine = engine_fixture
task_id = task_id_fixture
canonical = canonical_fixture
database_url = database_url_fixture
selected = selected_fixture
_late_bound_sessions = _late_bound_sessions_fixture

RUN = "run-channel-hold"
PV = 9
CID = auto_resume_command_id(PV, 1, RUN)
AUTO_RESUME_SWITCH = "XAGENT_TASK_AUTO_RESUME_ENABLED"
PAUSE_SWITCH = "XAGENT_TASK_INFRA_FAILURE_PAUSE_ENABLED"
NO_OUTPUTS = _PreparedTaskFileOutputs((), (), ())


@pytest.fixture(autouse=True)
def _default_window(monkeypatch):
    monkeypatch.delenv("XAGENT_TASK_AUTO_RESUME_CHANNEL_WINDOW_SECONDS", raising=False)


def _limit() -> int:
    """Seconds after an interruption the hold ends."""
    return staleness_window_seconds("channel") + CHANNEL_HOLD_CLOCK_GRACE_SECONDS


@pytest.fixture
def factory(engine, monkeypatch) -> sessionmaker:
    Base.metadata.create_all(engine)
    result = sessionmaker(engine, autoflush=False)
    # Set the globals behind the accessors, as the auto-resume suites do.
    monkeypatch.setattr(database, "_SessionLocal", result)
    monkeypatch.setattr(database, "_engine", engine)
    monkeypatch.delenv(AUTO_RESUME_SWITCH, raising=False)
    return result


def _seed(
    db: Session,
    now: datetime,
    *,
    state: str | None = "scheduled",
    status: TaskStatus = TaskStatus.PAUSED,
    control_state: str = "paused",
    version: int = PV,
    command_status: str | None = None,
    interrupted_ago: float = 60,
    row_run: str = RUN,
) -> int:
    """A task resting on ``RUN`` and, unless ``state`` is None, its row
    recorded at ``PV`` (with its dispatched command, if ``command_status``)."""
    user = User(username=f"hold-{uuid.uuid4().hex[:8]}", password_hash="x")
    db.add(user)
    db.flush()
    task = Task(
        user_id=user.id,
        title="Hold",
        description="hold test",
        execution_mode="balanced",
        status=status,
        control_state=control_state,
        run_id=RUN,
        state_version=version,
    )
    db.add(task)
    db.flush()
    if state is not None:
        db.add(
            TaskAutoRecovery(
                task_id=task.id,
                run_id=row_run,
                reason="lease_expired",
                state=state,
                paused_state_version=PV,
                interrupted_at=now - timedelta(seconds=interrupted_ago),
                episode_started_at=now - timedelta(seconds=interrupted_ago),
                total_resumes=1 if state == "dispatched" else 0,
                no_progress_resumes=1 if state == "dispatched" else 0,
                next_attempt_at=now - timedelta(seconds=1),
                last_command_id=CID if state == "dispatched" else None,
            )
        )
    if command_status is not None:
        db.add(
            TaskExecutionCommand(
                task_id=task.id,
                actor_user_id=user.id,
                command_id=CID,
                kind="resume",
                payload={"type": "resume_task"},
                target_run_id=RUN,
                status=command_status,
            )
        )
    db.flush()
    return int(task.id)


# --------------------------------------------------------------------------
# 1. The decision table
# --------------------------------------------------------------------------

_PAST = "past"
_NEAR = "near"

_TABLE = [
    # (id, seed kwargs, expected)
    ("no_row", {"state": None}, False),
    ("other_run", {"row_run": "older-run"}, False),
    *[
        (f"{state}", {"state": state}, False)
        for state in (
            "manual",
            "exhausted",
            "expired",
            "ineligible",
            "dispatch_failed",
            "superseded",
            "disabled",
        )
    ],
    ("scheduled", {}, True),
    ("scheduled_inside_the_grace", {"interrupted_ago": _NEAR}, True),
    ("scheduled_past_the_deadline", {"interrupted_ago": _PAST}, False),
    ("scheduled_version_moved", {"version": PV + 1}, False),
    (
        "scheduled_handing_off",
        {"control_state": "resume_requested", "version": PV + 1},
        True,
    ),
    (
        "stale_handing_off",
        {"state": "stale", "control_state": "resume_requested", "version": PV + 1},
        True,
    ),
    (
        "handing_off_past_the_deadline",
        {
            "control_state": "resume_requested",
            "version": PV + 1,
            "interrupted_ago": _PAST,
        },
        False,
    ),
    (
        "dispatched_queued_after_a_rollback",
        {"state": "dispatched", "version": PV + 2, "command_status": "pending"},
        True,
    ),
    (
        "dispatched_processing_past_the_deadline",
        {
            "state": "dispatched",
            "version": PV + 2,
            "command_status": "processing",
            "interrupted_ago": _PAST,
        },
        True,
    ),
    (
        "dispatched_completed_at_the_fence",
        {"state": "dispatched", "command_status": "completed"},
        True,
    ),
    ("dispatched_command_missing", {"state": "dispatched"}, True),
    (
        "dispatched_completed_past_the_deadline",
        {
            "state": "dispatched",
            "command_status": "completed",
            "interrupted_ago": _PAST,
        },
        False,
    ),
    (
        "resumed_run_paused_by_its_user",
        {"state": "dispatched", "version": PV + 3, "command_status": "completed"},
        False,
    ),
    (
        "resume_failed",
        {"state": "dispatched", "version": PV + 2, "command_status": "failed"},
        False,
    ),
    ("stale_paused", {"state": "stale", "version": PV + 1}, False),
]


@pytest.mark.parametrize(
    ("seed", "expected"),
    [pytest.param(seed, expected, id=name) for name, seed, expected in _TABLE],
)
def test_hold_decision_table(factory, seed, expected):
    now = utc_now()
    seed = dict(seed)
    ago = seed.get("interrupted_ago")
    if ago == _PAST:
        seed["interrupted_ago"] = _limit() + 5
    elif ago == _NEAR:
        # Past the window itself, inside the clock grace.
        seed["interrupted_ago"] = _limit() - 5
    with factory() as db:
        tid = _seed(db, now, **seed)
        db.commit()
    with factory() as db:
        assert (
            channel_reply_waits_for_auto_resume(db, task_id=tid, run_id=RUN, now=now)
            is expected
        )


@pytest.mark.parametrize(
    ("change", "expected"),
    [
        pytest.param(
            {"status": TaskStatus.RUNNING, "control_state": "running"},
            True,
            id="running",
        ),
        pytest.param({"run_id": "rotated"}, True, id="run_rotated"),
        pytest.param(None, False, id="deleted"),
    ],
)
def test_a_task_that_moved_since_the_caller_read_it(factory, change, expected):
    """Anything but PAUSED on the run reads again (True); a deleted task is
    released (False) and the caller reads it as missing."""
    now = utc_now()
    with factory() as db:
        tid = _seed(db, now, state="manual")
        db.commit()
    with factory() as db:
        if change is None:
            db.delete(db.get(Task, tid))
        else:
            for name, value in change.items():
                setattr(db.get(Task, tid), name, value)
        db.commit()
    with factory() as db:
        assert (
            channel_reply_waits_for_auto_resume(db, task_id=tid, run_id=RUN, now=now)
            is expected
        )


# --------------------------------------------------------------------------
# 2. One consistent read
# --------------------------------------------------------------------------


def test_the_hold_snapshot_is_one_statement(factory, engine):
    now = utc_now()
    with factory() as db:
        tid = _seed(db, now, state="dispatched", command_status="pending")
        db.commit()
    statements: list[str] = []

    def count(_conn, _cursor, statement, *_args):
        statements.append(statement)

    with factory() as db:
        db.connection()  # BEGIN outside the count
        sa.event.listen(engine, "before_cursor_execute", count)
        try:
            assert channel_reply_waits_for_auto_resume(
                db, task_id=tid, run_id=RUN, now=now
            )
        finally:
            sa.event.remove(engine, "before_cursor_execute", count)
    assert len(statements) == 1, statements


# --------------------------------------------------------------------------
# 7. The supersede helper
# --------------------------------------------------------------------------


def _supersede(factory, tid: int, now: datetime) -> bool:
    with factory() as db:
        task = db.execute(
            sa.select(Task).where(Task.id == tid).with_for_update()
        ).scalar_one()
        superseded = supersede_pending_auto_resume_no_commit(db, task, now=now)
        db.commit()
    return superseded


@pytest.mark.parametrize(
    ("seed", "version"),
    [
        pytest.param({}, PV, id="scheduled"),
        pytest.param(
            {"state": "dispatched", "command_status": "pending"},
            PV,
            id="dispatched_queued",
        ),
        pytest.param(
            {"state": "dispatched", "command_status": "processing", "version": PV + 2},
            PV + 2,
            id="retrying_after_a_rollback",
        ),
    ],
)
def test_a_new_turn_supersedes_a_resume_that_has_not_handed_off(factory, seed, version):
    now = utc_now()
    with factory() as db:
        tid = _seed(db, now, **seed)
        db.commit()

    assert _supersede(factory, tid, now)

    with factory() as db:
        task = db.get(Task, tid)
        row = db.get(TaskAutoRecovery, tid)
        events = db.scalars(
            sa.select(TaskRecoveryEvent).where(TaskRecoveryEvent.task_id == tid)
        ).all()
        assert int(task.state_version) == version + 1
        assert (task.status, task.control_state) == (TaskStatus.PAUSED, "paused")
        assert (row.state, row.state_detail, row.next_attempt_at) == (
            "stale",
            "new_turn",
            None,
        )
        assert [(e.event, e.detail["at"], e.detail["mismatch"]) for e in events] == [
            ("stale", "new_turn", "new_turn")
        ]
        # The supersede ends the hold.
        assert not channel_reply_waits_for_auto_resume(
            db, task_id=tid, run_id=RUN, now=now
        )


@pytest.mark.parametrize(
    "seed",
    [
        pytest.param(
            {"control_state": "resume_requested", "version": PV + 1},
            id="handing_off",
        ),
        pytest.param(
            {
                "state": "dispatched",
                "command_status": "processing",
                "control_state": "resume_requested",
                "version": PV + 1,
            },
            id="dispatched_handing_off",
        ),
        pytest.param(
            {
                "status": TaskStatus.RUNNING,
                "control_state": "running",
                "version": PV + 2,
            },
            id="running",
        ),
        pytest.param({"state": "manual"}, id="manual"),
        pytest.param({"state": "expired"}, id="expired"),
        pytest.param({"interrupted_ago": "past"}, id="past_the_deadline"),
        pytest.param({"version": PV + 1}, id="version_moved"),
    ],
)
def test_nothing_to_supersede_leaves_the_task_alone(factory, seed):
    now = utc_now()
    seed = dict(seed)
    if seed.get("interrupted_ago") == "past":
        seed["interrupted_ago"] = _limit() + 5
    with factory() as db:
        tid = _seed(db, now, **seed)
        before = (
            int(db.get(Task, tid).state_version),
            db.get(TaskAutoRecovery, tid).state,
        )
        db.commit()

    assert not _supersede(factory, tid, now)

    with factory() as db:
        assert (
            int(db.get(Task, tid).state_version),
            db.get(TaskAutoRecovery, tid).state,
        ) == before
        assert (
            db.scalar(
                sa.select(sa.func.count())
                .select_from(TaskRecoveryEvent)
                .where(TaskRecoveryEvent.task_id == tid)
            )
            == 0
        )


# --------------------------------------------------------------------------
# Channel fixtures: a real accepted channel turn whose run rests PAUSED
# --------------------------------------------------------------------------


@pytest.fixture
def channel(selected):
    """M1 accepted; its run started and then rested PAUSED on its run."""
    selected.delivery_destination = {"chat_id": "conversation"}
    command_id = shared._accept_channel_turn(
        selected, TaskTurnPayload("hello"), "ingress"
    )
    run_id = selected.run_id
    with get_session_local()() as db:
        command = db.get(TaskExecutionCommand, command_id)
        command.status = "completed"
        command.result = {"run_id": run_id, "lease_attempt_id": "attempt"}
        task = db.get(Task, command.task_id)
        task.status = TaskStatus.PAUSED
        task.control_state = "paused"
        task.run_id = run_id
        task.runner_id = None
        task.state_version = int(task.state_version or 0) + 2
        version = int(task.state_version)
        db.commit()
        tid = int(task.id)
    return SimpleNamespace(
        turn=selected,
        command_id=command_id,
        task_id=tid,
        run_id=run_id,
        version=version,
        channel_id=selected.selection.channel_id,
    )


def _record(
    ch, *, state: str = "scheduled", dispatched_status: str | None = None
) -> str | None:
    """The recovery row of ``ch``'s run; ``dispatched_status`` also stages
    the sweeper's RESUME. Returns its command id."""
    now = utc_now()
    command_id = (
        auto_resume_command_id(ch.version, 1, ch.run_id)
        if dispatched_status is not None
        else None
    )
    with get_session_local()() as db:
        db.add(
            TaskAutoRecovery(
                task_id=ch.task_id,
                run_id=ch.run_id,
                reason="lease_expired",
                state="dispatched" if dispatched_status is not None else state,
                paused_state_version=ch.version,
                interrupted_at=now - timedelta(seconds=30),
                episode_started_at=now - timedelta(seconds=30),
                total_resumes=1 if dispatched_status is not None else 0,
                no_progress_resumes=1 if dispatched_status is not None else 0,
                next_attempt_at=now - timedelta(seconds=1),
                last_command_id=command_id,
            )
        )
        if dispatched_status is not None:
            db.add(
                TaskExecutionCommand(
                    task_id=ch.task_id,
                    actor_user_id=db.get(Task, ch.task_id).user_id,
                    command_id=command_id,
                    kind="resume",
                    payload={
                        "type": "resume_task",
                        "auto_resume": {
                            "expected_run_id": ch.run_id,
                            "expected_state_version": ch.version,
                            "reason": "lease_expired",
                            "attempt": 1,
                        },
                    },
                    target_run_id=ch.run_id,
                    status=dispatched_status,
                )
            )
        db.commit()
    return command_id


def _set_row(ch, **values: Any) -> None:
    with get_session_local()() as db:
        row = db.get(TaskAutoRecovery, ch.task_id)
        for name, value in values.items():
            setattr(row, name, value)
        db.commit()


def _delivery(command_id: int) -> TaskChannelDelivery:
    with get_session_local()() as db:
        row = db.get(TaskChannelDelivery, command_id)
        db.expunge_all()
        return row


def _second_turn(ch, *, version: int | None = None) -> shared.SharedChannelTurn:
    selection = dataclasses.replace(
        ch.turn.selection,
        is_new_task=False,
        previous_run_id=ch.run_id,
        state_version=ch.version if version is None else version,
    )
    turn = shared.SharedChannelTurn(selection, workspace=SimpleNamespace())
    turn.origin = "origin-token"
    turn.delivery_destination = {"chat_id": "conversation"}
    return turn


def _command(command_id: int) -> TaskExecutionCommand:
    with get_session_local()() as db:
        row = db.get(TaskExecutionCommand, command_id)
        db.expunge_all()
        return row


# --------------------------------------------------------------------------
# 3. The read path
# --------------------------------------------------------------------------


def test_read_holds_while_a_resume_is_due(channel):
    _record(channel)
    assert shared._read_channel_result(channel.command_id, channel.run_id) is None


def test_read_is_interrupted_once_automatic_resume_stops(channel):
    _record(channel, state="exhausted")
    assert (
        shared._read_channel_result(channel.command_id, channel.run_id)
        == interrupted_channel_result()
    )


def test_read_without_a_recovery_row_is_interrupted_as_before(channel):
    assert (
        shared._read_channel_result(channel.command_id, channel.run_id)
        == interrupted_channel_result()
    )


def test_a_stored_channel_result_wins_over_the_hold(channel):
    _record(channel)
    answer = {"success": True, "status": "completed", "output": "the answer"}
    with get_session_local()() as db:
        command = db.get(TaskExecutionCommand, channel.command_id)
        command.result = {**command.result, "channel_result": answer}
        db.commit()
    assert shared._read_channel_result(channel.command_id, channel.run_id) == answer


# --------------------------------------------------------------------------
# 6. Delivery under the hold
# --------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("via", ["deliver", "recover"])
async def test_a_held_delivery_stays_pending_without_a_failure(channel, via):
    _record(channel)
    sender = AsyncMock()
    before = datetime.now(timezone.utc)

    if via == "deliver":
        assert not await channel_delivery.deliver_channel_result(
            channel.command_id, sender
        )
    else:
        await channel_delivery.recover_channel_results(channel.channel_id, sender)

    sender.assert_not_awaited()
    row = _delivery(channel.command_id)
    assert (row.status, row.failure_count, row.claim_token) == ("pending", 0, None)
    available_at = row.available_at
    if available_at.tzinfo is None:
        available_at = available_at.replace(tzinfo=timezone.utc)
    assert before + timedelta(seconds=25) <= available_at
    assert available_at <= datetime.now(timezone.utc) + timedelta(seconds=35)

    # Released: the next attempt delivers "interrupted" exactly once.
    _set_row(channel, state="expired")
    expire_claim(channel.command_id)
    first, second = await asyncio.gather(
        channel_delivery.deliver_channel_result(channel.command_id, sender),
        channel_delivery.deliver_channel_result(channel.command_id, sender),
    )
    assert sorted([first, second]) == [False, True]
    sender.assert_awaited_once()
    assert sender.await_args.args[1] == interrupted_channel_result()
    assert _delivery(channel.command_id).status == "delivered"


@pytest.mark.asyncio
async def test_a_pending_notice_during_the_hold_keeps_the_reply_pending(channel):
    """The original wait times out into the "still processing" notice."""
    _record(channel)
    sender = AsyncMock()

    assert not await channel_delivery.deliver_channel_result(
        channel.command_id, sender, pending_notice=True
    )

    sender.assert_awaited_once()
    assert sender.await_args.args[1] == channel_delivery.PENDING_CHANNEL_RESULT
    assert _delivery(channel.command_id).status == "pending"


# --------------------------------------------------------------------------
# 8. Acceptance supersedes
# --------------------------------------------------------------------------


@pytest.mark.parametrize("dispatched_status", [None, "pending"])
def test_a_turn_accepted_during_the_hold_supersedes_the_resume(
    channel, dispatched_status
):
    _record(channel, dispatched_status=dispatched_status)

    second = shared._accept_channel_turn(
        _second_turn(channel), TaskTurnPayload("again"), "ingress"
    )

    start = _command(second)
    with get_session_local()() as db:
        task = db.get(Task, channel.task_id)
        row = db.get(TaskAutoRecovery, channel.task_id)
        assert int(task.state_version) == channel.version + 1
        assert start.payload["state_version"] == channel.version + 1
        assert start.target_state_version == channel.version + 1
        assert (row.state, row.state_detail) == ("stale", "new_turn")
    # M1 now reads "interrupted"; M2 waits for its own run.
    assert (
        shared._read_channel_result(channel.command_id, channel.run_id)
        == interrupted_channel_result()
    )


@pytest.mark.parametrize("state", ["manual", "expired"])
def test_a_turn_accepted_without_a_hold_leaves_the_version(channel, state):
    """9a is inert: a channel row that is never scheduled changes nothing."""
    _record(channel, state=state)

    second = shared._accept_channel_turn(
        _second_turn(channel), TaskTurnPayload("again"), "ingress"
    )

    assert _command(second).payload["state_version"] == channel.version
    with get_session_local()() as db:
        assert int(db.get(Task, channel.task_id).state_version) == channel.version
        assert db.get(TaskAutoRecovery, channel.task_id).state == state


# --------------------------------------------------------------------------
# 9. A claimed automatic RESUME loses to the accepted turn
# --------------------------------------------------------------------------


@pytest.fixture
def handler(monkeypatch):
    """Stub everything past the admission decisions of ``resume_task``."""

    def snapshot(task_id, _owner, *, actor_user_id=None, actor_is_admin=False):
        with get_session_local()() as db:
            task = db.get(Task, task_id)
            return SimpleNamespace(
                task=SimpleNamespace(
                    user_id=task.user_id,
                    status=task.status,
                    control_state=task.control_state,
                    run_id=task.run_id,
                    state_version=task.state_version,
                    source=task.source,
                ),
                runtime_user=SimpleNamespace(id=task.user_id, is_admin=False),
            )

    monkeypatch.setattr(
        "xagent.web.services.task_setup_snapshot.load_task_setup_snapshot_sync",
        snapshot,
    )
    interaction = MagicMock(return_value=ActiveInteractionAbsent())
    monkeypatch.setattr(commands, "active_interaction_id_sync", interaction)
    monkeypatch.setattr(
        commands, "resolve_execution_scope_off_turn", lambda _task_id: None
    )
    manager = MagicMock()
    manager.running_tasks = {}
    manager.resume_admission_state.return_value = None
    manager.try_reserve_resume.return_value = ResumeReservationOutcome.RESERVED
    monkeypatch.setattr(task_execution_service, "background_task_manager", manager)
    background = AsyncMock()
    monkeypatch.setattr(task_execution_service, "execute_resume_background", background)
    agent_service = MagicMock()
    agent_service.supports_live_control = MagicMock(return_value=True)
    agents = MagicMock()
    agents.get_agent_for_task = AsyncMock(return_value=agent_service)
    monkeypatch.setattr(agent_service_manager, "get_agent_manager", lambda: agents)
    return SimpleNamespace(interaction=interaction, background=background)


async def _auto_resume(ch, command_id: str, *, attempt_count: int):
    with get_session_local()() as db:
        user_id = db.get(Task, ch.task_id).user_id
    return await commands.resume_task(
        AsyncMock(),
        ch.task_id,
        {
            "type": "resume_task",
            "auto_resume": {
                "expected_run_id": ch.run_id,
                "expected_state_version": ch.version,
                "reason": "lease_expired",
                "attempt": 1,
            },
            "user": SimpleNamespace(id=user_id, is_admin=False),
            "_durable_command_id": command_id,
            "_durable_attempt_count": attempt_count,
        },
    )


@pytest.mark.asyncio
async def test_a_turn_accepted_after_the_guard_defers_the_resume_and_runs(
    channel, handler
):
    auto_command_id = _record(channel, dispatched_status="processing")
    accepted: list[int] = []

    def accept_between_guard_and_transition(_task_id):
        # M2 lands after the claim guard let the RESUME through, before its
        # RESUME_REQUESTED transition.
        accepted.append(
            shared._accept_channel_turn(
                _second_turn(channel), TaskTurnPayload("again"), "ingress"
            )
        )
        return ActiveInteractionAbsent()

    handler.interaction.side_effect = accept_between_guard_and_transition
    first = await _auto_resume(channel, auto_command_id, attempt_count=1)

    assert first.outcome is ResumeCommandOutcome.DEFERRED
    with get_session_local()() as db:
        task = db.get(Task, channel.task_id)
        assert (task.status, task.control_state) == (TaskStatus.PAUSED, "paused")
        assert int(task.state_version) == channel.version + 1

    handler.interaction.side_effect = None
    retry = await _auto_resume(channel, auto_command_id, attempt_count=2)

    assert retry.outcome is ResumeCommandOutcome.AUTO_SKIPPED
    handler.background.assert_not_called()
    with get_session_local()() as db:
        row = db.get(TaskAutoRecovery, channel.task_id)
        assert (row.state, row.state_detail) == ("stale", "new_turn")
        # Serialized behind the RESUME until it completes (auto_skipped).
        claimable = [
            c.id for c in _claimable_query(db, runner_id="worker", command_db_id=None)
        ]
        assert accepted[0] not in claimable
        auto = db.scalars(
            sa.select(TaskExecutionCommand).where(
                TaskExecutionCommand.command_id == auto_command_id
            )
        ).one()
        auto.status = "completed"
        auto.result = {"resume_outcome": "auto_skipped"}
        db.commit()
    with get_session_local()() as db:
        claimable = [
            c.id for c in _claimable_query(db, runner_id="worker", command_db_id=None)
        ]
        assert claimable == [accepted[0]]
        # START's own admission checks (task_start_consumer) pass.
        task = db.get(Task, channel.task_id)
        start = db.get(TaskExecutionCommand, accepted[0])
        assert start.target_state_version == start.payload["state_version"]
        assert task.run_id == start.payload["expected_run_id"] == channel.run_id
        assert int(task.state_version) == start.payload["state_version"]
        assert task.status != TaskStatus.RUNNING
    assert (
        shared._read_channel_result(channel.command_id, channel.run_id)
        == interrupted_channel_result()
    )


# --------------------------------------------------------------------------
# 4. The shared channel leaf (S4)
# --------------------------------------------------------------------------


@pytest.mark.parametrize("recorded", ["scheduled", "exhausted", "manual", "raises"])
def test_the_leaf_holds_its_reply_only_for_a_scheduled_resume(
    canonical, monkeypatch, recorded
):
    monkeypatch.setenv(PAUSE_SWITCH, "true")
    monkeypatch.setenv(AUTO_RESUME_SWITCH, "true")
    if recorded == "raises":
        _fail_recording(monkeypatch, "python")
    else:
        original = task_auto_recovery.record_interruption_no_commit

        def force(db, **kwargs):
            row = original(db, **kwargs)
            row.state = recorded
            if recorded != "scheduled":
                row.next_attempt_at = None
            db.flush()
            return row

        monkeypatch.setattr(task_auto_recovery, "record_interruption_no_commit", force)
    factory, _tid = canonical
    ids = _projected_task(factory)
    lease = _prepare_run(factory, ids)
    report = SettlementReport()

    _settle_channel(factory, ids, lease, LLM_RESULT, report=report)

    with factory() as db:
        task = db.get(Task, ids["task"])
        result = db.get(TaskExecutionCommand, ids["command"]).result
        assert task.status == TaskStatus.PAUSED
        assert task.output == "previous answer"
    assert report.paused
    if recorded == "scheduled":
        assert report.recovery.state == "scheduled"
        assert result == {"run_id": lease.run_id, "lease_attempt_id": lease.attempt_id}
    else:
        assert result["channel_result"] == interrupted_channel_result()
        assert (report.recovery is None) is (recorded == "raises")


# --------------------------------------------------------------------------
# 5. The resumed run stores the channel answer
# --------------------------------------------------------------------------

_COMPLETED = {
    "success": True,
    "status": "completed",
    "output": "The resumed answer",
    "completion_outcome": "completed",
}
_EXCEL_FAILED = {
    "success": False,
    "status": "failed",
    "output": "workbook locked",
    "error": "workbook locked",
    "failure_code": "excel_workbook_locked",
}
_QUESTION = {
    "success": True,
    "status": "waiting_for_user",
    "output": "Which one?",
    "chat_response": {
        "message": "Which one?",
        "interactions": [
            {
                "type": "select_one",
                "field": "choice",
                "label": "Pick",
                "options": ["left", "right"],
            }
        ],
    },
}


def _channel_task_run(
    factory, *, channel_payload: bool = True, start: bool = True, stored=None
):
    ids = _projected_task(factory)
    with factory() as db:
        user_channel = UserChannel(
            user_id=ids["user"],
            channel_type="feishu",
            channel_name="hold",
            config={},
            is_active=True,
        )
        db.add(user_channel)
        db.flush()
        db.get(Task, ids["task"]).channel_id = user_channel.id
        db.commit()
        channel_id = int(user_channel.id)
    lease = _prepare_run(factory, ids)
    if start:
        result: dict[str, Any] = {
            "run_id": lease.run_id,
            "lease_attempt_id": "first-segment",
        }
        if stored is not None:
            result["channel_result"] = stored
        with factory() as db:
            command = TaskExecutionCommand(
                task_id=ids["task"],
                command_id=f"start-{ids['turn_id']}",
                kind="start",
                payload={
                    "channel": (
                        {"channel_id": channel_id, "external_user_id": "sender"}
                        if channel_payload
                        else None
                    )
                },
                target_run_id=lease.run_id,
                status="completed",
                result=result,
            )
            db.add(command)
            db.commit()
            ids["command"] = int(command.id)
    return ids, lease


def _finalize(ids, lease, result: dict[str, Any]) -> dict[str, Any]:
    return _finalize_resumed_task(
        ids["task"],
        status=str(result.get("status") or ""),
        success=bool(result.get("success")),
        output=str(result.get("output") or result.get("error") or ""),
        task_owner_user_id=ids["user"],
        result=dict(result),
        task_lease=lease,
        prepared_outputs=NO_OUTPUTS,
    )


def _start_result(factory, ids) -> dict[str, Any]:
    with factory() as db:
        return dict(db.get(TaskExecutionCommand, ids["command"]).result)


@pytest.mark.parametrize(
    ("result", "status"),
    [
        pytest.param(_COMPLETED, TaskStatus.COMPLETED, id="completed"),
        pytest.param(_EXCEL_FAILED, TaskStatus.FAILED, id="failed_coded"),
        pytest.param(_QUESTION, TaskStatus.WAITING_FOR_USER, id="waiting_for_user"),
    ],
)
def test_the_resumed_run_stores_the_leaf_projection(canonical, result, status):
    factory, _tid = canonical
    ids, lease = _channel_task_run(factory)

    finalized = _finalize(ids, lease, result)

    assert finalized["lease_released"]
    assert finalized["final_status"] == status.value
    stored = _start_result(factory, ids)["channel_result"]
    projection = project_execution_result_for_channel(dict(result))
    with factory() as db:
        task_output = db.get(Task, ids["task"]).output
    transcript = task_output if result["success"] else None
    assert stored == shared.channel_durable_result(
        dict(result), projection, transcript=transcript
    )
    if status == TaskStatus.COMPLETED:
        # The text the transcript stored (file links rewritten there).
        assert stored["output"] == task_output == "The resumed answer"
        assert stored["status"] == "completed"
    elif status == TaskStatus.FAILED:
        assert stored["failure_code"] == "excel_workbook_locked"
        assert stored["success"] is False
    else:
        assert (
            stored["chat_response"]["interactions"]
            == (_QUESTION["chat_response"]["interactions"])
        )
        assert stored["status"] == "waiting_for_user"


@pytest.mark.parametrize("result", ["interrupted", "paused_again"])
def test_a_resumed_run_that_pauses_stores_nothing(canonical, monkeypatch, result):
    monkeypatch.setenv(PAUSE_SWITCH, "true")
    monkeypatch.setenv(AUTO_RESUME_SWITCH, "true")
    factory, _tid = canonical
    ids, lease = _channel_task_run(factory)
    payload = (
        {"success": False, "status": "interrupted", "output": ""}
        if result == "interrupted"
        else LLM_RESULT
    )

    finalized = _finalize(ids, lease, payload)

    assert finalized["final_status"] == TaskStatus.PAUSED.value
    assert "channel_result" not in _start_result(factory, ids)


def test_a_stored_channel_result_is_never_replaced(canonical):
    factory, _tid = canonical
    ids, lease = _channel_task_run(factory, stored=interrupted_channel_result())

    assert _finalize(ids, lease, _COMPLETED)["lease_released"]

    assert _start_result(factory, ids)["channel_result"] == (
        interrupted_channel_result()
    )


def test_a_start_that_is_not_a_channel_turn_is_left_alone(canonical):
    factory, _tid = canonical
    ids, lease = _channel_task_run(factory, channel_payload=False)

    assert _finalize(ids, lease, _COMPLETED)["lease_released"]

    assert "channel_result" not in _start_result(factory, ids)


def test_a_run_without_its_start_still_settles(canonical):
    factory, _tid = canonical
    ids, lease = _channel_task_run(factory, start=False)

    finalized = _finalize(ids, lease, _COMPLETED)

    assert finalized["lease_released"]
    assert finalized["final_status"] == TaskStatus.COMPLETED.value


def test_a_rolled_back_settlement_stores_nothing(canonical, monkeypatch):
    factory, _tid = canonical
    ids, lease = _channel_task_run(factory)
    monkeypatch.setattr(
        task_execution_service,
        "release_task_lease_no_commit",
        lambda *_args, **_kwargs: False,
    )

    finalized = _finalize(ids, lease, _COMPLETED)

    assert finalized["late_result"]
    assert "channel_result" not in _start_result(factory, ids)


# --------------------------------------------------------------------------
# 16-18. PostgreSQL: lock order with the sweeper, the claim guard and W
# --------------------------------------------------------------------------


def _postgresql_only() -> None:
    if get_engine().dialect.name != "postgresql":
        pytest.skip("needs two real PostgreSQL transactions")


@pytest.fixture
def short_lock_timeout():
    """Every new connection waits at most 5s on a lock: a deadlock or a
    wrong lock order fails instead of hanging."""
    engine = get_engine()
    engine.dispose()

    def apply(connection, _record):
        connection.autocommit = True
        with connection.cursor() as cursor:
            cursor.execute("SET lock_timeout = '5s'")
        connection.autocommit = False

    sa.event.listen(engine, "connect", apply)
    yield
    sa.event.remove(engine, "connect", apply)
    engine.dispose()


def test_postgresql_sweeper_skips_a_task_an_acceptance_holds(
    channel, short_lock_timeout
):
    _postgresql_only()
    _record(channel)
    scheduled = task_auto_resume.TaskAutoRecoveryState.SCHEDULED
    now = utc_now()
    with get_session_local()() as db:
        assert select_next_candidate_for_update(db, scheduled, now=now) is not None
        db.rollback()

    accepting = get_session_local()()
    try:
        shared.accept_channel_turn_no_commit(
            accepting, _second_turn(channel), TaskTurnPayload("again"), "ingress"
        )
        with get_session_local()() as sweeper:
            # SKIP LOCKED: the acceptance holds the task row.
            assert select_next_candidate_for_update(sweeper, scheduled, now=now) is None
            sweeper.rollback()
        accepting.commit()
    finally:
        accepting.close()

    with get_session_local()() as db:
        # Superseded: nothing left for the next tick.
        assert select_next_candidate_for_update(db, scheduled, now=now) is None
        assert db.get(TaskAutoRecovery, channel.task_id).state == "stale"


@pytest.mark.parametrize("snapshot", ["at_fence", "moved"])
def test_postgresql_claim_guard_waits_for_an_acceptance_without_deadlock(
    channel, short_lock_timeout, snapshot
):
    """At the fence the guard locks the recovery row; past it, the task row
    first. Either way it waits for the acceptance (which holds both, tasks
    first) and then finds the row superseded."""
    _postgresql_only()
    command_id = _record(channel, dispatched_status="processing")
    outcome: dict[str, Any] = {}

    def guard() -> None:
        try:
            outcome["decision"] = check_auto_resume_claim_sync(
                task_id=channel.task_id,
                command_id=command_id,
                auto_resume={
                    "expected_run_id": channel.run_id,
                    "expected_state_version": channel.version,
                },
                status=TaskStatus.PAUSED,
                control_state="paused",
                run_id=channel.run_id,
                state_version=(
                    channel.version if snapshot == "at_fence" else channel.version + 5
                ),
                attempt_count=1,
            )
        except BaseException as exc:  # pragma: no cover - reported below
            outcome["error"] = exc

    accepting = get_session_local()()
    try:
        shared.accept_channel_turn_no_commit(
            accepting, _second_turn(channel), TaskTurnPayload("again"), "ingress"
        )
        thread = threading.Thread(target=guard)
        thread.start()
        time.sleep(0.5)
        assert thread.is_alive(), "the guard did not wait for the acceptance"
        accepting.commit()
    finally:
        accepting.close()
    thread.join(timeout=10)
    assert not thread.is_alive()
    assert "error" not in outcome, outcome.get("error")
    assert outcome["decision"].proceed is False
    with get_session_local()() as db:
        row = db.get(TaskAutoRecovery, channel.task_id)
        assert (row.state, row.state_detail) == ("stale", "new_turn")


def _running_channel_run(ch) -> None:
    """The run resumed: RUNNING on the same run under a lease."""
    with get_session_local()() as db:
        task = db.get(Task, ch.task_id)
        task.status = TaskStatus.RUNNING
        task.control_state = "running"
        task.runner_id = "worker"
        task.lease_expires_at = utc_now() + timedelta(minutes=5)
        db.commit()


def _stamp_locked(db: Session, ch, answer: dict[str, Any]) -> None:
    """``_finalize_resumed_task``'s order: the task row, then the START."""
    task = db.execute(
        sa.select(Task).where(Task.id == ch.task_id).with_for_update()
    ).scalar_one()
    task.status = TaskStatus.COMPLETED
    task.control_state = "completed"
    task.runner_id = None
    task.lease_expires_at = None
    db.flush()
    assert shared.stamp_channel_result_no_commit(
        db, task_id=ch.task_id, run_id=ch.run_id, channel_result=answer
    )


ANSWER = {"success": True, "status": "completed", "output": "resumed answer"}


def test_postgresql_stamp_and_a_delivery_claim_do_not_block_each_other(
    channel, short_lock_timeout
):
    _postgresql_only()
    _record(channel, dispatched_status="completed")
    _running_channel_run(channel)

    # W holds the task and START rows; the claim neither waits nor sees it.
    settling = get_session_local()()
    try:
        _stamp_locked(settling, channel, ANSWER)
        outcome: dict[str, Any] = {}

        def claim() -> None:
            try:
                outcome["claimed"] = channel_delivery._claim(channel.command_id)
            except BaseException as exc:  # pragma: no cover - reported below
                outcome["error"] = exc

        thread = threading.Thread(target=claim)
        thread.start()
        thread.join(timeout=10)
        assert not thread.is_alive()
        assert "error" not in outcome, outcome.get("error")
        _delivery_row, result = outcome["claimed"]
        assert result is None  # still RUNNING as committed
        settling.commit()
    finally:
        settling.close()

    # A claim held open does not block the stamp's locks either.
    expire_claim(channel.command_id)
    claiming = get_session_local()()
    try:
        claiming.execute(
            sa.update(TaskChannelDelivery)
            .where(TaskChannelDelivery.command_id == channel.command_id)
            .values(claim_token="held")
        )
        assert shared._read_channel_result(channel.command_id, channel.run_id) == ANSWER
        with get_session_local()() as other:
            # Already stored: first writer wins, and it took the locks freely.
            other.execute(
                sa.select(Task).where(Task.id == channel.task_id).with_for_update()
            ).scalar_one()
            assert not shared.stamp_channel_result_no_commit(
                other,
                task_id=channel.task_id,
                run_id=channel.run_id,
                channel_result={"success": False},
            )
            other.rollback()
        claiming.rollback()
    finally:
        claiming.close()
