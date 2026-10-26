"""Reconnect/replay delivery consistency for V2 task sockets.

Each V2 socket records the display frames it was actually sent. On disconnect
the frames committed after its replay horizon are compared with what was
sent, off the event loop. Bridge duplicates and gaps are counted where the
receiver drops or resyncs. No signal line carries payload content.
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest
from sqlalchemy.exc import OperationalError

from tests.web.services.test_execution_event_signals import (
    exported,
    key,
)
from tests.web.services.test_execution_event_signals import (
    metric_reader as metric_reader_fixture,
)
from tests.web.services.test_task_execution_event_writer import (
    canonical as canonical_fixture,
)
from tests.web.services.test_task_execution_event_writer import engine as engine_fixture
from tests.web.services.test_task_execution_event_writer import (
    task_id as task_id_fixture,
)
from xagent.web.api import websocket as websocket_api
from xagent.web.api.websocket import ConnectionManager, _HistoricalStreamSnapshot
from xagent.web.models.task import Task
from xagent.web.services import task_event_bridge
from xagent.web.services.task_event_display import (
    convert_display_event,
    load_event_display_snapshot,
)
from xagent.web.services.task_execution_event_writer import append_fact_no_commit

canonical = canonical_fixture
engine = engine_fixture
task_id = task_id_fixture
metric_reader = metric_reader_fixture

SENTINEL = "T3A-SENTINEL"
CHECK_TIMEOUT = 10


class Socket:
    def __init__(self) -> None:
        self.frames: list[dict] = []

    async def send_text(self, text: str) -> None:
        self.frames.append(json.loads(text))

    async def close(self, code: int = 1000) -> None:
        return None


@pytest.fixture
def manager(monkeypatch):
    """A fresh process-wide manager with control-state enrichment disabled."""
    monkeypatch.setenv("XAGENT_SHARED_TASK_EXECUTION_ENABLED", "false")

    async def unchanged(message, **_kwargs):
        return message

    monkeypatch.setattr(websocket_api, "_with_current_task_control_state", unchanged)
    fresh = ConnectionManager()
    monkeypatch.setattr(websocket_api, "manager", fresh)
    return fresh


def fact(db, task_id, kind, key, data, *, flat=False, **kwargs):
    return append_fact_no_commit(
        db,
        task_id=task_id,
        kind=kind,
        key=key,
        payload=data if flat else {"data": data, "protocol_event_id": key},
        occurred_at=datetime(2026, 9, 30, tzinfo=timezone.utc),
        **kwargs,
    )


def write_history(factory, task_id):
    """Display facts interleaved with facts that never produce a frame.

    Returns the committed rows in sequence order.
    """
    with factory() as db:
        rows = [
            fact(
                db,
                task_id,
                "input_accepted",
                "turn-1",
                {"content": f"hello {SENTINEL}", "message_type": "user"},
                flat=True,
                turn_id="turn-1",
            ),
            fact(db, task_id, "agent_message", "before", {"message": SENTINEL}),
            # Horizon of the replay below ends here.
            fact(db, task_id, "recovery_state", "state", {"snapshot": SENTINEL}),
            fact(db, task_id, "agent_message", "live-1", {"message": SENTINEL}),
            fact(
                db,
                task_id,
                "agent_message",
                "child",
                {"message": SENTINEL},
                scope_id="child",
            ),
            fact(
                db,
                task_id,
                "llm_call_end",
                "audit",
                {"__audit_only__": True, "response": SENTINEL},
            ),
            fact(db, task_id, "control_state_changed", "ctl", {"state": "x"}),
            fact(db, task_id, "agent_message", "live-2", {"message": SENTINEL}),
            fact(db, task_id, "agent_message", "live-3", {"message": SENTINEL}),
        ]
        db.commit()
        for row in rows:
            db.refresh(row)
        db.expunge_all()
        return rows


def by_key(rows):
    return {row.idempotency_key: row for row in rows}


def live_frame(row):
    frame = convert_display_event(row)
    assert frame is not None
    return frame


def replay_snapshot(factory, task_id, horizon):
    with factory() as db:
        events = load_event_display_snapshot(
            db, task_id, through_sequence=horizon
        ).events
    return _HistoricalStreamSnapshot(
        events=tuple(events), horizon=horizon, display_events=len(events)
    )


async def replay(monkeypatch, socket, task_id, snapshot):
    monkeypatch.setattr(
        websocket_api, "_load_historical_stream_snapshot_sync", lambda *a, **k: snapshot
    )
    await websocket_api.send_historical_data_as_stream(
        socket, task_id, SimpleNamespace(id=1, is_admin=True)
    )


async def settle_checks(manager):
    pending = list(manager._delivery_checks)
    if pending:
        await asyncio.wait_for(asyncio.gather(*pending), timeout=CHECK_TIMEOUT)


def consistency(reader):
    return exported(reader, "xagent.websocket.replay.consistency")


def signal_lines(caplog, prefix="execution_event_"):
    return [r.getMessage() for r in caplog.records if r.getMessage().startswith(prefix)]


def fields(line: str) -> dict[str, str]:
    return dict(part.split("=", 1) for part in line.split()[1:])


def horizon_of(rows):
    return by_key(rows)["state"].sequence


async def connect_and_replay(manager, monkeypatch, factory, task_id, rows, socket):
    manager.register_connection(socket, task_id)
    await replay(
        monkeypatch,
        socket,
        task_id,
        replay_snapshot(factory, task_id, horizon_of(rows)),
    )


@pytest.mark.asyncio
async def test_replay_and_live_frames_covering_everything_are_complete(
    canonical, manager, monkeypatch, caplog, metric_reader
):
    factory, task_id = canonical
    rows = write_history(factory, task_id)
    socket = Socket()
    await connect_and_replay(manager, monkeypatch, factory, task_id, rows, socket)
    rows_by_key = by_key(rows)
    # Alternate the send paths so each one delivers a frame below the
    # highest sequence another path delivered.
    await manager.broadcast_to_task(live_frame(rows_by_key["live-1"]), task_id)
    await manager.send_personal_message(live_frame(rows_by_key["live-2"]), socket)
    await manager.broadcast_to_task(live_frame(rows_by_key["live-3"]), task_id)
    ledger = manager._delivery_ledgers[socket]
    assert ledger.horizon == horizon_of(rows)
    assert ledger.max_sequence == rows_by_key["live-3"].sequence

    caplog.set_level(logging.INFO)
    manager.disconnect(socket)
    await settle_checks(manager)

    assert consistency(metric_reader) == {key(outcome="complete", storage="v2"): 1}
    assert not signal_lines(caplog, "execution_event_delivery_gap")


@pytest.mark.asyncio
async def test_internal_kinds_and_other_scopes_are_not_missing(
    canonical, manager, monkeypatch, caplog, metric_reader
):
    """Sequence gaps between delivered frames are normal and never missing."""
    factory, task_id = canonical
    rows = write_history(factory, task_id)
    rows_by_key = by_key(rows)
    socket = Socket()
    await connect_and_replay(manager, monkeypatch, factory, task_id, rows, socket)
    await manager.broadcast_to_task(live_frame(rows_by_key["live-1"]), task_id)
    # Child scope, audit-only and control facts sit between live-1 and live-2.
    await manager.broadcast_to_task(live_frame(rows_by_key["live-2"]), task_id)
    await manager.broadcast_to_task(live_frame(rows_by_key["live-3"]), task_id)
    assert rows_by_key["live-2"].sequence - rows_by_key["live-1"].sequence == 4
    assert rows_by_key["live-1"].sequence - horizon_of(rows) == 1

    manager.disconnect(socket)
    await settle_checks(manager)

    assert consistency(metric_reader) == {key(outcome="complete", storage="v2"): 1}


@pytest.mark.asyncio
async def test_withheld_frame_is_missing_with_its_sequence(
    canonical, manager, monkeypatch, caplog, metric_reader
):
    factory, task_id = canonical
    rows = write_history(factory, task_id)
    rows_by_key = by_key(rows)
    socket = Socket()
    other = Socket()
    manager.register_connection(other, task_id)
    await connect_and_replay(manager, monkeypatch, factory, task_id, rows, socket)
    await manager.broadcast_to_task(live_frame(rows_by_key["live-1"]), task_id)
    # A lost send: live-2 reaches the other socket only.
    manager._remove_from_task(socket, task_id)
    await manager.broadcast_to_task(live_frame(rows_by_key["live-2"]), task_id)
    manager.active_connections[task_id].append(socket)
    await manager.broadcast_to_task(live_frame(rows_by_key["live-3"]), task_id)

    caplog.set_level(logging.INFO)
    manager.disconnect(socket)
    await settle_checks(manager)

    assert consistency(metric_reader) == {key(outcome="missing", storage="v2"): 1}
    (line,) = signal_lines(caplog, "execution_event_delivery_gap")
    assert fields(line) == {
        "task_id": str(task_id),
        "storage": "v2",
        "component": "execution-events",
        "horizon": str(horizon_of(rows)),
        "missing": "1",
        "first_missing_sequence": str(rows_by_key["live-2"].sequence),
    }
    assert [r.levelno for r in caplog.records if r.getMessage() == line] == [
        logging.WARNING
    ]
    for line in signal_lines(caplog):
        assert SENTINEL not in line
    assert SENTINEL in json.dumps(socket.frames)


@pytest.mark.asyncio
async def test_replay_and_live_overlap_is_duplicate_without_warning(
    canonical, manager, monkeypatch, caplog, metric_reader
):
    factory, task_id = canonical
    rows = write_history(factory, task_id)
    rows_by_key = by_key(rows)
    socket = Socket()
    manager.register_connection(socket, task_id)
    # Registered before the replay: a live frame below the horizon arrives
    # first and the replay sends it again.
    await manager.broadcast_to_task(live_frame(rows_by_key["before"]), task_id)
    await replay(
        monkeypatch,
        socket,
        task_id,
        replay_snapshot(factory, task_id, horizon_of(rows)),
    )
    for name in ("live-1", "live-2", "live-3"):
        await manager.broadcast_to_task(live_frame(rows_by_key[name]), task_id)

    caplog.set_level(logging.INFO)
    manager.disconnect(socket)
    await settle_checks(manager)

    assert consistency(metric_reader) == {key(outcome="duplicate", storage="v2"): 1}
    assert not signal_lines(caplog, "execution_event_delivery_gap")


@pytest.mark.asyncio
async def test_repeated_replay_on_request_is_not_a_duplicate(
    canonical, manager, monkeypatch, metric_reader
):
    factory, task_id = canonical
    rows = write_history(factory, task_id)
    socket = Socket()
    await connect_and_replay(manager, monkeypatch, factory, task_id, rows, socket)
    for name in ("live-1", "live-2", "live-3"):
        await manager.broadcast_to_task(live_frame(by_key(rows)[name]), task_id)
    # A status request replays the whole history up to a later horizon.
    await replay(
        monkeypatch,
        socket,
        task_id,
        replay_snapshot(factory, task_id, rows[-1].sequence),
    )

    manager.disconnect(socket)
    await settle_checks(manager)

    assert consistency(metric_reader) == {key(outcome="complete", storage="v2"): 1}


@pytest.mark.asyncio
async def test_frames_missed_live_but_sent_by_a_later_replay_are_complete(
    canonical, manager, monkeypatch, metric_reader
):
    factory, task_id = canonical
    rows = write_history(factory, task_id)
    socket = Socket()
    await connect_and_replay(manager, monkeypatch, factory, task_id, rows, socket)
    await replay(
        monkeypatch,
        socket,
        task_id,
        replay_snapshot(factory, task_id, rows[-1].sequence),
    )

    manager.disconnect(socket)
    await settle_checks(manager)

    assert consistency(metric_reader) == {key(outcome="complete", storage="v2"): 1}


@pytest.mark.asyncio
async def test_public_audience_projection_is_not_missing(
    canonical, manager, monkeypatch, metric_reader
):
    """Guest sockets receive a projection of every frame, with the same ids.

    Broadcasts filter no V2 frame per recipient; facts that produce no frame
    for anyone (audit-only, internal kinds, child scope) are not expected.
    """
    factory, task_id = canonical
    rows = write_history(factory, task_id)
    socket = Socket()
    manager.register_connection(socket, task_id, public_audience=True)
    await replay(
        monkeypatch,
        socket,
        task_id,
        replay_snapshot(factory, task_id, horizon_of(rows)),
    )
    for name in ("live-1", "live-2", "live-3"):
        await manager.broadcast_to_task(live_frame(by_key(rows)[name]), task_id)

    manager.disconnect(socket)
    await settle_checks(manager)

    assert consistency(metric_reader) == {key(outcome="complete", storage="v2"): 1}


@pytest.mark.asyncio
async def test_shared_mode_records_frames_when_the_writer_sends_them(
    canonical, manager, monkeypatch, metric_reader
):
    factory, task_id = canonical
    rows = write_history(factory, task_id)
    monkeypatch.setenv("XAGENT_SHARED_TASK_EXECUTION_ENABLED", "true")
    monkeypatch.setattr(
        task_event_bridge,
        "_bridge",
        SimpleNamespace(discard_recipient=lambda _socket: None),
    )
    socket = Socket()
    await connect_and_replay(manager, monkeypatch, factory, task_id, rows, socket)
    writer = manager._writers[socket]
    rows_by_key = by_key(rows)
    await manager.deliver_shared_event(live_frame(rows_by_key["live-1"]), task_id)
    # Queued is not sent: nothing is recorded until the writer sends.
    ledger = manager._delivery_ledgers[socket]
    assert ledger.max_sequence < rows_by_key["live-1"].sequence
    # A personal message waits for the writer, so live-1 is sent by now.
    await manager.send_personal_message(live_frame(rows_by_key["live-2"]), socket)
    assert ledger.max_sequence == rows_by_key["live-2"].sequence
    await manager.deliver_shared_event(live_frame(rows_by_key["live-3"]), task_id)
    while not writer.queue.empty():
        await asyncio.sleep(0)
    await asyncio.sleep(0)
    assert ledger.max_sequence == rows_by_key["live-3"].sequence
    assert set(ledger.sent) >= {
        live_frame(rows_by_key[name])["event_id"] for name in ("live-1", "live-2")
    }

    manager.disconnect(socket)
    await asyncio.gather(writer.task, return_exceptions=True)
    await settle_checks(manager)

    assert consistency(metric_reader) == {key(outcome="complete", storage="v2"): 1}


@pytest.mark.asyncio
async def test_more_frames_than_the_cap_overflow_without_a_check(
    manager, monkeypatch, metric_reader
):
    def unexpected(*_args):
        raise AssertionError("an overflowed ledger must not be checked")

    monkeypatch.setattr(websocket_api, "_expected_display_frames_sync", unexpected)
    socket = Socket()
    manager.register_connection(socket, 7)
    await replay(
        monkeypatch,
        socket,
        7,
        _HistoricalStreamSnapshot(events=({"type": "task_info"},), horizon=0),
    )
    for sequence in range(1, websocket_api._DELIVERY_LEDGER_CAP + 2):
        await manager.send_personal_message(
            {
                "type": "trace_event",
                "event_id": f"e{sequence}",
                "execution_sequence": sequence,
            },
            socket,
        )
    ledger = manager._delivery_ledgers[socket]
    assert ledger.overflow and not ledger.sent

    manager.disconnect(socket)
    assert not manager._delivery_checks
    assert consistency(metric_reader) == {key(outcome="overflow", storage="v2"): 1}


@pytest.mark.asyncio
async def test_v1_socket_records_nothing(manager, monkeypatch, metric_reader):
    socket = Socket()
    manager.register_connection(socket, 7)
    await replay(
        monkeypatch,
        socket,
        7,
        _HistoricalStreamSnapshot(
            events=(
                {"type": "task_info"},
                {"type": "trace_event", "event_id": "v1", "data": {}},
            )
        ),
    )
    await manager.broadcast_to_task(
        {"type": "trace_event", "event_id": "v1-live", "data": {}}, 7
    )
    assert manager._delivery_ledgers == {}

    manager.disconnect(socket)
    assert not manager._delivery_checks
    assert consistency(metric_reader) == {}


@pytest.mark.asyncio
async def test_check_runs_off_loop_and_a_failing_check_only_warns(
    manager, monkeypatch, caplog, metric_reader
):
    loop_thread = threading.get_ident()
    threads: list[int] = []
    release = threading.Event()

    def failing(*_args):
        threads.append(threading.get_ident())
        release.wait(CHECK_TIMEOUT)
        raise OperationalError("SELECT", {}, Exception(SENTINEL))

    monkeypatch.setattr(websocket_api, "_expected_display_frames_sync", failing)
    socket = Socket()
    manager.register_connection(socket, 7)
    await replay(
        monkeypatch,
        socket,
        7,
        _HistoricalStreamSnapshot(events=({"type": "task_info"},), horizon=3),
    )
    await manager.send_personal_message(
        {"type": "trace_event", "event_id": "e4", "execution_sequence": 4}, socket
    )

    caplog.set_level(logging.INFO)
    # Disconnect returns at once while the check is still blocked in a thread.
    manager.disconnect(socket)
    assert not manager.has_connections_for_task(7)
    assert socket not in manager._delivery_ledgers
    (check,) = manager._delivery_checks
    assert not check.done()
    release.set()
    await settle_checks(manager)

    assert threads and loop_thread not in threads
    assert consistency(metric_reader) == {}
    (line,) = signal_lines(caplog, "execution_event_delivery_check_failed")
    assert fields(line) == {
        "task_id": "7",
        "storage": "v2",
        "component": "execution-events",
        "horizon": "3",
        "error_type": "OperationalError",
    }
    assert SENTINEL not in caplog.text


@pytest.mark.asyncio
async def test_pending_checks_are_cancelled_on_shutdown(manager, monkeypatch):
    started = threading.Event()
    release = threading.Event()

    def blocked(*_args):
        started.set()
        release.wait(CHECK_TIMEOUT)
        return {}

    monkeypatch.setattr(websocket_api, "_expected_display_frames_sync", blocked)
    socket = Socket()
    manager.register_connection(socket, 7)
    await replay(
        monkeypatch,
        socket,
        7,
        _HistoricalStreamSnapshot(events=({"type": "task_info"},), horizon=0),
    )
    await manager.send_personal_message(
        {"type": "trace_event", "event_id": "e1", "execution_sequence": 1}, socket
    )
    manager.disconnect(socket)
    (check,) = manager._delivery_checks
    await asyncio.to_thread(started.wait, CHECK_TIMEOUT)
    stopping = asyncio.create_task(manager.stop_delivery_checks())
    await asyncio.sleep(0)
    release.set()
    await asyncio.wait_for(stopping, timeout=CHECK_TIMEOUT)
    assert check.cancelled()
    assert not manager._delivery_checks


@pytest.mark.asyncio
async def test_replay_logs_one_payload_free_line(
    canonical, manager, monkeypatch, caplog
):
    factory, task_id = canonical
    rows = write_history(factory, task_id)
    with factory() as db:
        task = db.get(Task, task_id)
        owner = int(task.user_id)
    socket = Socket()
    manager.register_connection(socket, task_id)
    caplog.set_level(logging.INFO)
    user = SimpleNamespace(id=owner, is_admin=False)
    await websocket_api.send_historical_data_as_stream(socket, task_id, user)
    await websocket_api.send_historical_data_as_stream(socket, task_id, user)

    lines = signal_lines(caplog, "execution_event_replay ")
    assert len(lines) == 2
    first = fields(lines[0])
    assert {k: v for k, v in first.items() if k != "duration_ms"} == {
        "task_id": str(task_id),
        "storage": "v2",
        "component": "execution-events",
        "horizon": str(rows[-1].sequence),
        # input, before, live-1, live-2, live-3 (child/audit/internal excluded)
        "events": "5",
        "cache_hit": "False",
    }
    assert float(first["duration_ms"]) >= 0
    assert fields(lines[1])["cache_hit"] in {"True", "False"}
    assert manager._delivery_ledgers[socket].horizon == rows[-1].sequence
    for line in signal_lines(caplog):
        assert SENTINEL not in line
    assert SENTINEL in json.dumps(socket.frames)


def bridge_frames(reader):
    return exported(reader, "xagent.task_event.bridge.frames")


@pytest.fixture
def bridge(monkeypatch):
    monkeypatch.setenv("XAGENT_REDIS_URL", "redis://localhost:6379/0")
    return task_event_bridge.TaskEventBridge()


@pytest.mark.asyncio
async def test_bridge_counts_duplicates_and_gaps_by_ids_only(
    bridge, caplog, metric_reader
):
    async def deliver(message, task_id):
        return None

    async def status(kind):
        return None

    bridge.deliver = deliver
    bridge.stream_status = status
    body = {
        "version": 1,
        "kind": "event",
        "host": "worker-1",
        "sequence": 1,
        "task_id": 42,
        "message": {"type": "trace_event", "data": {"message": SENTINEL}},
    }
    await bridge._receive(body)
    await bridge._receive(body)
    await bridge._receive({**body, "sequence": 4, "task_id": 43})
    await bridge.close()

    assert bridge_frames(metric_reader) == {
        key(outcome="duplicate_dropped"): 1,
        key(outcome="gap_resync"): 1,
    }
    (record,) = [
        r for r in caplog.records if r.getMessage().startswith("task_event_bridge_gap")
    ]
    assert record.levelno == logging.WARNING
    assert fields(record.getMessage()) == {
        "task_id": "43",
        "host": "worker-1",
        "previous_sequence": "1",
        "sequence": "4",
    }
    assert SENTINEL not in caplog.text


@pytest.mark.asyncio
async def test_bridge_counts_publish_failures(bridge, monkeypatch, metric_reader):
    async def fail(channel, body):
        raise ConnectionError("redis down")

    monkeypatch.setattr(bridge, "_publish", fail)
    await bridge.publish({"type": "heartbeat"}, 42)
    await bridge.close()

    assert bridge_frames(metric_reader) == {key(outcome="publish_failed"): 1}
