import json
from datetime import datetime, timedelta, timezone

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from xagent.core.agent.result import CONTROL_TOOL_NAMES
from xagent.core.context_ref import CONTEXT_REFS_KEY
from xagent.web.models.chat_message import TaskChatMessage
from xagent.web.models.database import Base
from xagent.web.models.task import Task, TaskStatus, TraceEvent
from xagent.web.models.user import User
from xagent.web.services.chat_history_service import _MAX_HISTORICAL_IMAGE_CONTEXT_REFS
from xagent.web.services.task_conversation_context_service import (
    load_task_conversation_context_sync,
)

# Serialized size matters here: the old buggy path blind-sliced tool results
# at 240 chars, and this fixture must land comfortably past that boundary so
# the test still catches a truncation regression after future edits. The
# original 7-field version of this fixture serialized (as ``{"handle": ...}``)
# to 241 chars -- a single character of margin over the old 240-char cutoff,
# which the old buggy code would have slipped past undetected on nearly any
# edit. ``commit_sha``/``container_image`` are added purely to push this to
# ~368 chars, well clear of that boundary; don't shrink this fixture without
# re-checking it stays well above 240.
STRUCTURED_HANDLE = {
    "workspace": "4b33784773d5",
    "branch": "review-pr-1392",
    "provider": "omp",
    "profile": "omp",
    "provider_session_id": "01a00aff-558e-7000-8ebf-ec547e8018b0",
    "last_run_id": "76ff56bf27554b899e74bce25f9dc769",
    "node_id": "node-0",
    "commit_sha": "a1b2c3d4e5f60718293a4b5c6d7e8f9012345678",
    "container_image": "ghcr.io/xagent/sandbox-runner:2026.08.17-py311",
}


def _create_db_session():
    engine = create_engine("sqlite:///:memory:")
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    Base.metadata.create_all(bind=engine)
    return SessionLocal()


def _create_task(db_session):
    user = User(username="tester", password_hash="hashed_password", is_admin=False)
    db_session.add(user)
    db_session.commit()
    db_session.refresh(user)

    task = Task(
        user_id=int(user.id),
        title="Conversation context task",
        description="Task conversation context",
        status=TaskStatus.COMPLETED,
    )
    db_session.add(task)
    db_session.commit()
    db_session.refresh(task)
    return task


def _add_chat_message(
    db_session, task, *, role, content, created_at, turn_id=None, attachments=None
):
    message = TaskChatMessage(
        task_id=int(task.id),
        user_id=int(task.user_id),
        role=role,
        content=content,
        message_type=role,
        turn_id=turn_id,
        created_at=created_at,
        attachments=attachments,
    )
    db_session.add(message)
    db_session.commit()
    db_session.refresh(message)
    return message


def _add_trace_event(
    db_session,
    task,
    *,
    event_type,
    data,
    timestamp,
    event_id=None,
    step_id=None,
):
    event = TraceEvent(
        task_id=int(task.id),
        build_id=None,
        event_id=event_id or f"{event_type}-{timestamp.isoformat()}",
        event_type=event_type,
        timestamp=timestamp,
        step_id=step_id,
        parent_event_id=None,
        data=data,
    )
    db_session.add(event)
    db_session.commit()
    db_session.refresh(event)
    return event


def _ts(seconds_offset, base=None, tz=timezone.utc):
    base = base or datetime(2026, 1, 1, tzinfo=tz)
    return base + timedelta(seconds=seconds_offset)


def test_structured_handle_survives_reconstruction_byte_for_byte():
    db_session = _create_db_session()
    try:
        task = _create_task(db_session)
        _add_trace_event(
            db_session,
            task,
            event_type="tool_execution_start",
            timestamp=_ts(0),
            data={
                "tool_name": "resume_task",
                "tool_params": {"workspace": "4b33784773d5"},
                "tool_call_id": "call-handle-1",
                "assistant_content": "Resuming previous workspace.",
            },
        )
        _add_trace_event(
            db_session,
            task,
            event_type="tool_execution_end",
            timestamp=_ts(1),
            data={
                "tool_name": "resume_task",
                "tool_call_id": "call-handle-1",
                "success": True,
                "result": {"handle": dict(STRUCTURED_HANDLE)},
            },
        )

        messages = load_task_conversation_context_sync(db_session, int(task.id))

        tool_messages = [m for m in messages if m["role"] == "tool"]
        assert len(tool_messages) == 1
        assert tool_messages[0]["raw_result"]["handle"] == STRUCTURED_HANDLE

        serialized = json.dumps(messages, ensure_ascii=False, default=str)
        # Discriminating check: the old bug did a blind fixed-length
        # character slice, which would corrupt whichever field that offset
        # happened to land inside. A field near the FRONT of the object
        # (like "branch", checked by an earlier version of this test) would
        # still survive most such slices and give a false pass -- the
        # 237-char legacy boundary sat well past "branch"'s offset. A field
        # placed at the very END of the object, like "container_image"
        # here, is what a fixed-length slice would actually clip first;
        # its exact, untruncated presence is the meaningful signal.
        assert (
            '"container_image": "ghcr.io/xagent/sandbox-runner:2026.08.17-py311"'
            in serialized
        )
        assert json.loads(serialized) == messages  # round-trips byte-for-byte
    finally:
        db_session.close()


def test_tool_call_pairing_never_orphaned():
    db_session = _create_db_session()
    try:
        task = _create_task(db_session)

        # Normal start + end.
        _add_trace_event(
            db_session,
            task,
            event_type="tool_execution_start",
            timestamp=_ts(0),
            data={
                "tool_name": "list_files",
                "tool_params": {"path": "."},
                "tool_call_id": "call-1",
            },
        )
        _add_trace_event(
            db_session,
            task,
            event_type="tool_execution_end",
            timestamp=_ts(1),
            data={
                "tool_name": "list_files",
                "tool_call_id": "call-1",
                "success": True,
                "result": {"files": ["a.txt"]},
            },
        )

        # End with no start.
        _add_trace_event(
            db_session,
            task,
            event_type="tool_execution_end",
            timestamp=_ts(2),
            data={
                "tool_name": "read_file",
                "tool_call_id": "call-orphan-end",
                "success": True,
                "result": {"content": "hello"},
            },
        )

        # Start with no end -> should yield nothing.
        _add_trace_event(
            db_session,
            task,
            event_type="tool_execution_start",
            timestamp=_ts(3),
            data={
                "tool_name": "write_file",
                "tool_params": {"path": "b.txt"},
                "tool_call_id": "call-orphan-start",
            },
        )

        # Missing tool_call_id entirely.
        _add_trace_event(
            db_session,
            task,
            event_type="tool_execution_start",
            timestamp=_ts(4),
            data={
                "tool_name": "web_search",
                "tool_params": {"query": "xagent"},
            },
        )
        _add_trace_event(
            db_session,
            task,
            event_type="tool_execution_end",
            timestamp=_ts(5),
            data={
                "tool_name": "web_search",
                "success": True,
                "result": {"top_result": "https://example.com"},
            },
        )

        # tool_execution_failed.
        _add_trace_event(
            db_session,
            task,
            event_type="tool_execution_start",
            timestamp=_ts(6),
            data={
                "tool_name": "run_command",
                "tool_params": {"cmd": "false"},
                "tool_call_id": "call-fail",
            },
        )
        _add_trace_event(
            db_session,
            task,
            event_type="tool_execution_failed",
            timestamp=_ts(7),
            data={
                "tool_name": "run_command",
                "tool_call_id": "call-fail",
                "error_type": "agent_tool_error",
                "error": "boom",
                "error_message": "boom",
            },
        )

        messages = load_task_conversation_context_sync(db_session, int(task.id))

        tool_calls_by_id = {}
        for index, message in enumerate(messages):
            if message["role"] != "assistant" or not message.get("tool_calls"):
                continue
            for call in message["tool_calls"]:
                tool_calls_by_id[call["id"]] = index

        for index, message in enumerate(messages):
            if message["role"] != "tool":
                continue
            call_id = message["tool_call_id"]
            assert call_id in tool_calls_by_id
            assert tool_calls_by_id[call_id] == index - 1

        tool_names_emitted = [m["tool_name"] for m in messages if m["role"] == "tool"]
        assert "write_file" not in tool_names_emitted  # start-with-no-end -> nothing
        assert "list_files" in tool_names_emitted
        assert "read_file" in tool_names_emitted
        assert "run_command" in tool_names_emitted

        run_command_tool = next(
            m
            for m in messages
            if m["role"] == "tool" and m["tool_name"] == "run_command"
        )
        assert run_command_tool["raw_result"]["error"] == "boom"
    finally:
        db_session.close()


def test_multi_turn_history_is_chronological():
    db_session = _create_db_session()
    try:
        task = _create_task(db_session)
        base = datetime(2026, 1, 1, tzinfo=timezone.utc)

        for turn in range(3):
            offset_minutes = turn * 10
            _add_chat_message(
                db_session,
                task,
                role="user",
                content=f"user turn {turn}",
                created_at=base + timedelta(minutes=offset_minutes),
            )
            for tool_index in range(2):
                call_id = f"turn{turn}-call{tool_index}"
                ts_start = base + timedelta(
                    minutes=offset_minutes, seconds=1 + tool_index * 2
                )
                ts_end = base + timedelta(
                    minutes=offset_minutes, seconds=2 + tool_index * 2
                )
                _add_trace_event(
                    db_session,
                    task,
                    event_type="tool_execution_start",
                    timestamp=ts_start,
                    data={
                        "tool_name": "list_files",
                        "tool_params": {"turn": turn, "tool_index": tool_index},
                        "tool_call_id": call_id,
                    },
                )
                _add_trace_event(
                    db_session,
                    task,
                    event_type="tool_execution_end",
                    timestamp=ts_end,
                    data={
                        "tool_name": "list_files",
                        "tool_call_id": call_id,
                        "success": True,
                        "result": {"turn": turn, "tool_index": tool_index},
                    },
                )
            _add_chat_message(
                db_session,
                task,
                role="assistant",
                content=f"assistant answer {turn}",
                created_at=base + timedelta(minutes=offset_minutes, seconds=9),
            )

        messages = load_task_conversation_context_sync(db_session, int(task.id))
        roles = [m["role"] for m in messages]

        assert roles.count("user") == 3
        assert (
            roles.count("assistant") == 3 + 3 * 2
        )  # turn answers + tool-call assistants
        assert roles.count("tool") == 6

        for turn in range(3):
            user_index = messages.index(
                {"role": "user", "content": f"user turn {turn}"}
            )
            answer_index = next(
                index
                for index, m in enumerate(messages)
                if m["role"] == "assistant"
                and m.get("content") == f"assistant answer {turn}"
            )
            assert user_index < answer_index
            # every tool exchange for this turn sits strictly between the two
            for index in range(user_index + 1, answer_index):
                message = messages[index]
                if message["role"] == "tool":
                    assert message["raw_result"]["turn"] == turn
    finally:
        db_session.close()


def test_naive_and_aware_timestamps_merge_without_error():
    db_session = _create_db_session()
    try:
        task = _create_task(db_session)

        # created_at written as naive (as SQLite round-trips it) while trace
        # timestamps are aware -- must not raise TypeError when merged.
        _add_chat_message(
            db_session,
            task,
            role="user",
            content="naive user message",
            created_at=datetime(2026, 1, 1, 0, 0, 0),  # naive
        )
        _add_trace_event(
            db_session,
            task,
            event_type="tool_execution_start",
            timestamp=datetime(2026, 1, 1, 0, 0, 1, tzinfo=timezone.utc),
            data={
                "tool_name": "list_files",
                "tool_params": {},
                "tool_call_id": "call-aware",
            },
        )
        _add_trace_event(
            db_session,
            task,
            event_type="tool_execution_end",
            timestamp=datetime(2026, 1, 1, 0, 0, 2, tzinfo=timezone.utc),
            data={
                "tool_name": "list_files",
                "tool_call_id": "call-aware",
                "success": True,
                "result": {"ok": True},
            },
        )
        _add_chat_message(
            db_session,
            task,
            role="assistant",
            content="naive assistant reply",
            created_at=datetime(2026, 1, 1, 0, 0, 3),  # naive
        )

        messages = load_task_conversation_context_sync(db_session, int(task.id))
        roles = [m["role"] for m in messages]
        assert roles == ["user", "assistant", "tool", "assistant"]
        assert messages[0]["content"] == "naive user message"
        assert messages[-1]["content"] == "naive assistant reply"
    finally:
        db_session.close()


def test_as_aware_utc_normalizes_naive_and_preserves_aware():
    """Direct unit test of ``_as_aware_utc`` itself, independent of any DB
    round trip. A naive datetime gets stamped UTC; an already-aware datetime
    (in a non-UTC zone) passes through unchanged rather than being re-based.
    """
    from xagent.web.services.task_conversation_context_service import _as_aware_utc

    naive = datetime(2026, 1, 1, 12, 0, 0)
    normalized = _as_aware_utc(naive)
    assert normalized.tzinfo is timezone.utc
    assert normalized == datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)

    aware_non_utc = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone(timedelta(hours=5)))
    normalized_aware = _as_aware_utc(aware_non_utc)
    assert normalized_aware.tzinfo == timezone(timedelta(hours=5))
    assert normalized_aware == aware_non_utc


def test_merge_chronologically_orders_naive_and_aware_without_typeerror():
    """Direct unit test of the merge path (R4) with hand-built naive/aware
    datetimes, bypassing the DB round trip entirely.

    ``test_naive_and_aware_timestamps_merge_without_error`` above looks like
    it covers the naive-vs-aware hazard, but it doesn't: SQLite strips
    tzinfo from *both* ``TaskChatMessage.created_at`` and
    ``TraceEvent.timestamp`` on read-back, so by the time
    ``_load_transcript_rows``/``_load_tool_exchanges`` see them, nothing
    naive is ever compared against anything aware -- that test would pass
    identically even if ``_as_aware_utc`` were replaced by the identity
    function. This test builds a ``_TranscriptRow`` whose ``created_at`` is
    genuinely naive-turned-aware and a ``_ToolExchange`` whose ``sort_key``
    is genuinely aware, so a broken (identity) ``_as_aware_utc`` would raise
    ``TypeError`` on the ``entries.sort`` call inside
    ``_merge_chronologically`` -- proving the normalization is load-bearing.
    """
    from xagent.web.services.task_conversation_context_service import (
        _as_aware_utc,
        _merge_chronologically,
        _ToolExchange,
        _TranscriptRow,
    )

    naive_user = _TranscriptRow(
        row_id=1,
        role="user",
        content="naive user",
        created_at=_as_aware_utc(datetime(2026, 1, 1, 0, 0, 0)),  # naive input
    )
    aware_reply = _TranscriptRow(
        row_id=2,
        role="assistant",
        content="aware assistant",
        created_at=_as_aware_utc(datetime(2026, 1, 1, 0, 0, 5, tzinfo=timezone.utc)),
    )
    exchange = _ToolExchange(
        call_id="call-1",
        tool_name="list_files",
        tool_params={},
        result={"ok": True},
        assistant_content="",
        sort_key=(
            _as_aware_utc(datetime(2026, 1, 1, 0, 0, 2, tzinfo=timezone.utc)),
            10,
        ),
    )

    entries = _merge_chronologically([naive_user, aware_reply], [exchange])
    kinds_in_order = [entry.kind for entry in entries]
    assert kinds_in_order == ["transcript", "exchange", "transcript"]
    assert entries[1].exchange is exchange
    assert entries[0].transcript is naive_user
    assert entries[2].transcript is aware_reply


def test_control_tools_are_excluded():
    """Every name in ``CONTROL_TOOL_NAMES`` -- not just a couple of them --
    must be excluded from reconstructed tool exchanges. This is checked
    against the real ``CONTROL_TOOL_NAMES`` set rather than a hardcoded
    tuple so a future addition to that set is automatically covered here."""
    db_session = _create_db_session()
    try:
        task = _create_task(db_session)

        for index, tool_name in enumerate(sorted(CONTROL_TOOL_NAMES)):
            call_id = f"call-control-{index}"
            _add_trace_event(
                db_session,
                task,
                event_type="tool_execution_start",
                timestamp=_ts(index * 10),
                data={
                    "tool_name": tool_name,
                    "tool_params": {},
                    "tool_call_id": call_id,
                },
            )
            _add_trace_event(
                db_session,
                task,
                event_type="tool_execution_end",
                timestamp=_ts(index * 10 + 1),
                data={
                    "tool_name": tool_name,
                    "tool_call_id": call_id,
                    "success": True,
                    "result": {"ok": True},
                },
            )

        _add_trace_event(
            db_session,
            task,
            event_type="tool_execution_start",
            timestamp=_ts(1000),
            data={
                "tool_name": "list_files",
                "tool_params": {},
                "tool_call_id": "call-real",
            },
        )
        _add_trace_event(
            db_session,
            task,
            event_type="tool_execution_end",
            timestamp=_ts(1001),
            data={
                "tool_name": "list_files",
                "tool_call_id": "call-real",
                "success": True,
                "result": {"files": []},
            },
        )

        messages = load_task_conversation_context_sync(db_session, int(task.id))
        tool_names = [m["tool_name"] for m in messages if m["role"] == "tool"]
        for control_tool_name in CONTROL_TOOL_NAMES:
            assert control_tool_name not in tool_names
        assert tool_names == ["list_files"]
    finally:
        db_session.close()


def test_parallel_batch_assistant_content_not_duplicated():
    db_session = _create_db_session()
    try:
        task = _create_task(db_session)
        shared_prose = "Running two lookups in parallel."

        for index in range(2):
            call_id = f"parallel-call-{index}"
            _add_trace_event(
                db_session,
                task,
                event_type="tool_execution_start",
                timestamp=_ts(index),
                data={
                    "tool_name": "list_files",
                    "tool_params": {"index": index},
                    "tool_call_id": call_id,
                    "assistant_content": shared_prose,
                },
            )
            _add_trace_event(
                db_session,
                task,
                event_type="tool_execution_end",
                timestamp=_ts(index + 10),
                data={
                    "tool_name": "list_files",
                    "tool_call_id": call_id,
                    "success": True,
                    "result": {"index": index},
                },
            )

        messages = load_task_conversation_context_sync(db_session, int(task.id))
        assistant_contents = [
            m["content"]
            for m in messages
            if m["role"] == "assistant" and m.get("tool_calls")
        ]
        assert assistant_contents.count(shared_prose) == 1
        assert assistant_contents.count("") == 1
    finally:
        db_session.close()


def test_concurrent_dag_steps_with_colliding_tool_call_ids_are_not_mispaired():
    """Two concurrent DAG branches whose tool-call ids collide (both fall
    back to ``tool_call_0`` because the provider omitted a real id) must NOT
    have their starts/ends cross-paired.

    This reproduces the real corruption: DAG runs steps concurrently
    (``asyncio.create_task`` in ``dag.py``), and ``_normalize_tool_calls``
    (react.py) synthesizes ``tool_call_{index}`` where ``index`` is only
    unique within a single LLM response -- so two concurrent steps can each
    emit id "tool_call_0". Interleaving start A, start B, end A, end B with a
    bare-``tool_call_id`` pending map would let B's start overwrite A's, and
    A's end would then pop B's start, attaching branch B's tool_name /
    assistant_content to branch A's result (and vice versa).

    Each step carries its own ``step_id`` (DAG's ``_with_step`` stamps a
    unique step id per concurrent branch, persisted on
    ``TraceEvent.step_id``), which is exactly the discriminator this test
    checks is actually used to disambiguate.
    """
    db_session = _create_db_session()
    try:
        task = _create_task(db_session)

        # Branch A start (step "step-a"), then branch B start (step "step-b"),
        # both using the colliding id "tool_call_0" -- interleaved as
        # start A, start B, end A, end B, matching real concurrent arrival.
        _add_trace_event(
            db_session,
            task,
            event_type="tool_execution_start",
            timestamp=_ts(0),
            step_id="step-a",
            data={
                "tool_name": "read_file",
                "tool_params": {"path": "a.txt"},
                "tool_call_id": "tool_call_0",
                "assistant_content": "Branch A: reading a.txt.",
            },
        )
        _add_trace_event(
            db_session,
            task,
            event_type="tool_execution_start",
            timestamp=_ts(1),
            step_id="step-b",
            data={
                "tool_name": "list_files",
                "tool_params": {"path": "b/"},
                "tool_call_id": "tool_call_0",
                "assistant_content": "Branch B: listing b/.",
            },
        )
        _add_trace_event(
            db_session,
            task,
            event_type="tool_execution_end",
            timestamp=_ts(2),
            step_id="step-a",
            data={
                "tool_name": "read_file",
                "tool_call_id": "tool_call_0",
                "success": True,
                "result": {"content": "contents of a.txt"},
            },
        )
        _add_trace_event(
            db_session,
            task,
            event_type="tool_execution_end",
            timestamp=_ts(3),
            step_id="step-b",
            data={
                "tool_name": "list_files",
                "tool_call_id": "tool_call_0",
                "success": True,
                "result": {"files": ["b1.txt", "b2.txt"]},
            },
        )

        messages = load_task_conversation_context_sync(db_session, int(task.id))

        assistant_messages = [
            m for m in messages if m["role"] == "assistant" and m.get("tool_calls")
        ]
        tool_messages = [m for m in messages if m["role"] == "tool"]
        assert len(assistant_messages) == 2
        assert len(tool_messages) == 2

        by_result = {}
        for assistant_message, tool_message in zip(assistant_messages, tool_messages):
            call = assistant_message["tool_calls"][0]
            by_result[tool_message["raw_result"].get("content") or "listing"] = (
                assistant_message["content"],
                call["function"]["name"],
                tool_message["tool_name"],
            )

        content_a, tool_name_a, result_tool_name_a = by_result["contents of a.txt"]
        assert content_a == "Branch A: reading a.txt."
        assert tool_name_a == "read_file"
        assert result_tool_name_a == "read_file"

        content_b, tool_name_b, result_tool_name_b = by_result["listing"]
        assert content_b == "Branch B: listing b/."
        assert tool_name_b == "list_files"
        assert result_tool_name_b == "list_files"
    finally:
        db_session.close()


def test_react_shaped_events_without_step_discriminator_still_pair_correctly():
    """ReAct never sets a colliding id within one turn, and older rows
    persisted before ``TraceEvent.step_id`` was populated carry
    ``step_id=None`` for every event. Both must still pair by bare
    ``tool_call_id`` exactly as before this fix -- the step-id
    discriminator must be a no-op here, not a behavior change.
    """
    db_session = _create_db_session()
    try:
        task = _create_task(db_session)

        _add_trace_event(
            db_session,
            task,
            event_type="tool_execution_start",
            timestamp=_ts(0),
            step_id=None,
            data={
                "tool_name": "search_web",
                "tool_params": {"query": "xagent"},
                "tool_call_id": "call-1",
                "assistant_content": "Searching the web.",
            },
        )
        _add_trace_event(
            db_session,
            task,
            event_type="tool_execution_end",
            timestamp=_ts(1),
            step_id=None,
            data={
                "tool_name": "search_web",
                "tool_call_id": "call-1",
                "success": True,
                "result": {"top_result": "https://example.com"},
            },
        )

        messages = load_task_conversation_context_sync(db_session, int(task.id))

        assistant_messages = [
            m for m in messages if m["role"] == "assistant" and m.get("tool_calls")
        ]
        tool_messages = [m for m in messages if m["role"] == "tool"]
        assert len(assistant_messages) == 1
        assert len(tool_messages) == 1
        assert assistant_messages[0]["content"] == "Searching the web."
        assert assistant_messages[0]["tool_calls"][0]["function"]["name"] == (
            "search_web"
        )
        assert tool_messages[0]["raw_result"] == {"top_result": "https://example.com"}
    finally:
        db_session.close()


# ---------------------------------------------------------------------------
# Part C -- coverage for previously-untested paths
# ---------------------------------------------------------------------------


def test_clock_skew_guard_relocates_misplaced_exchange_atomically():
    """A tool exchange timestamped BEFORE the first transcript user message
    (DB clock vs app clock skew) must be repositioned after it in the merged
    output, and the relocated assistant/tool block must stay atomic (the
    tool message immediately follows its declaring assistant message)."""
    db_session = _create_db_session()
    try:
        task = _create_task(db_session)
        base = datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)

        # The transcript's first user message, created_at = base.
        _add_chat_message(
            db_session,
            task,
            role="user",
            content="first turn",
            created_at=base,
        )

        # This exchange's trace timestamps predate the first user message --
        # simulating clock skew between the DB clock (created_at) and the
        # app clock (trace timestamps).
        _add_trace_event(
            db_session,
            task,
            event_type="tool_execution_start",
            timestamp=base - timedelta(seconds=50),
            data={
                "tool_name": "list_files",
                "tool_params": {},
                "tool_call_id": "call-skewed",
                "assistant_content": "Looking things up.",
            },
        )
        _add_trace_event(
            db_session,
            task,
            event_type="tool_execution_end",
            timestamp=base - timedelta(seconds=49),
            data={
                "tool_name": "list_files",
                "tool_call_id": "call-skewed",
                "success": True,
                "result": {"ok": True},
            },
        )

        _add_chat_message(
            db_session,
            task,
            role="assistant",
            content="done",
            created_at=base + timedelta(seconds=10),
        )

        messages = load_task_conversation_context_sync(db_session, int(task.id))
        roles = [m["role"] for m in messages]

        # Repositioned after the first user message, not before it.
        assert roles[0] == "user"
        assert messages[0]["content"] == "first turn"
        assert roles[1] == "assistant" and messages[1].get("tool_calls")
        assert roles[2] == "tool"
        assert messages[2]["tool_call_id"] == "call-skewed"
        # Atomic: the tool message is immediately preceded by its assistant.
        assert messages[1]["tool_calls"][0]["id"] == messages[2]["tool_call_id"]
        assert roles[3] == "assistant" and messages[3]["content"] == "done"
    finally:
        db_session.close()


def test_before_message_id_excludes_current_turn_user_message():
    db_session = _create_db_session()
    try:
        task = _create_task(db_session)
        base = datetime(2026, 1, 1, tzinfo=timezone.utc)

        prior = _add_chat_message(
            db_session,
            task,
            role="user",
            content="prior turn message",
            created_at=base,
        )
        current = _add_chat_message(
            db_session,
            task,
            role="user",
            content="current turn message",
            created_at=base + timedelta(seconds=10),
        )
        assert prior.id < current.id

        messages = load_task_conversation_context_sync(
            db_session, int(task.id), before_message_id=int(current.id)
        )
        contents = [m.get("content") for m in messages]
        assert "prior turn message" in contents
        assert "current turn message" not in contents
    finally:
        db_session.close()


def test_resolve_tool_result_interrupted_branch_without_result():
    """A ``tool_execution_end`` marked ``interrupted`` with no ``result`` key
    must produce the degraded interrupted dict, not fall through to the
    "missing result" branch or a bare ``None``."""
    from xagent.web.services.task_conversation_context_service import (
        _resolve_tool_result,
    )

    resolved = _resolve_tool_result(
        "tool_execution_end",
        {
            "tool_name": "run_command",
            "interrupted": True,
            "interrupt_reason": "user cancelled the run",
        },
    )
    assert resolved == {
        "success": False,
        "interrupted": True,
        "error": "user cancelled the run",
    }


def test_final_pairing_sweep_drops_orphaned_tool_message():
    """Direct unit test of ``_final_pairing_sweep``: no fixture in this suite
    naturally produces an orphaned ``tool`` message (construction is
    designed to never emit one), so this exercises the safety-net sweep
    itself with a hand-built orphan."""
    from xagent.web.services.task_conversation_context_service import (
        _final_pairing_sweep,
    )

    messages = [
        {"role": "user", "content": "hi"},
        {
            "role": "tool",
            "tool_call_id": "orphan-1",
            "content": "",
            "tool_name": "orphan_tool",
            "raw_result": {},
        },
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {"name": "real_tool", "arguments": "{}"},
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call-1",
            "content": "",
            "tool_name": "real_tool",
            "raw_result": {},
        },
    ]

    sanitized = _final_pairing_sweep(messages)
    assert sanitized == [messages[0], messages[2], messages[3]]
    tool_call_ids = [m["tool_call_id"] for m in sanitized if m["role"] == "tool"]
    assert tool_call_ids == ["call-1"]


def test_final_pairing_sweep_drops_orphan_when_both_ids_are_none():
    """Review regression (PR #1601): the old comparison was
    ``str(call.get("id")) == str(message.get("tool_call_id"))``, so a
    ``tool`` message with ``tool_call_id: None`` preceded by an assistant
    whose declared call also has ``id: None`` rendered as
    ``"None" == "None"`` and was kept -- an orphan slipping through the
    exact defense meant to catch it. This must fail against that old
    ``str(...) == str(...)`` form and pass with ``_ids_match``."""
    from xagent.web.services.task_conversation_context_service import (
        _final_pairing_sweep,
    )

    messages = [
        {"role": "user", "content": "hi"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": None,
                    "type": "function",
                    "function": {"name": "real_tool", "arguments": "{}"},
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": None,
            "content": "",
            "tool_name": "real_tool",
            "raw_result": {},
        },
    ]

    sanitized = _final_pairing_sweep(messages)
    assert sanitized == [messages[0], messages[1]]
    assert all(m.get("role") != "tool" for m in sanitized)


def test_final_pairing_sweep_keeps_tool_message_with_matching_real_id():
    """Normal path, unchanged: a tool message with a real id preceded by
    the assistant that declared that same id is kept."""
    from xagent.web.services.task_conversation_context_service import (
        _final_pairing_sweep,
    )

    messages = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call-42",
                    "type": "function",
                    "function": {"name": "real_tool", "arguments": "{}"},
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call-42",
            "content": "",
            "tool_name": "real_tool",
            "raw_result": {},
        },
    ]

    sanitized = _final_pairing_sweep(messages)
    assert sanitized == messages


def test_final_pairing_sweep_drops_tool_message_with_mismatched_real_id():
    """A tool message with a real id whose preceding assistant declares a
    different id is still dropped as an orphan."""
    from xagent.web.services.task_conversation_context_service import (
        _final_pairing_sweep,
    )

    messages = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {"name": "real_tool", "arguments": "{}"},
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call-2",
            "content": "",
            "tool_name": "real_tool",
            "raw_result": {},
        },
    ]

    sanitized = _final_pairing_sweep(messages)
    assert sanitized == [messages[0]]


def test_ids_match_ignores_malformed_tool_call_entries():
    """Malformed ``tool_calls`` entries -- a non-dict item, or a dict with
    no ``id`` key -- must not raise and must not produce a false match."""
    from xagent.web.services.task_conversation_context_service import (
        _ids_match,
    )

    assert _ids_match("call-1", ["not-a-dict", {"type": "function"}]) is False
    assert _ids_match(None, ["not-a-dict", {"type": "function"}]) is False
    assert (
        _ids_match(
            "call-1",
            ["not-a-dict", {"type": "function"}, {"id": "call-1"}],
        )
        is True
    )


# ---------------------------------------------------------------------------
# Part D -- regression tests for the fixes just made
# ---------------------------------------------------------------------------


def test_dedup_positional_after_sort_keeps_different_turns_distinct_prose():
    """Fix 5 regression: two DIFFERENT turns emitting byte-identical
    assistant prose must BOTH keep their prose. The old code tracked
    ``last_assistant_content`` across trace rows in *end-event-arrival*
    order; when an unrelated exchange's END event lands before an earlier
    exchange's own (slow-running) END event, positional dedup by end-order
    could blank a later, unrelated turn's coincidentally identical prose.
    Sorting by *start* time before deduping (the fix) keeps these turns
    correctly separated by the truly-intervening exchange."""
    db_session = _create_db_session()
    try:
        task = _create_task(db_session)
        shared_prose = "Investigating the failure."

        # Turn A: starts first, but is slow -- its end event arrives LAST.
        _add_trace_event(
            db_session,
            task,
            event_type="tool_execution_start",
            timestamp=_ts(0),
            data={
                "tool_name": "tool_a",
                "tool_params": {},
                "tool_call_id": "call-a",
                "assistant_content": shared_prose,
            },
        )
        # Turn B: starts after A, but is fast -- its end event arrives
        # before A's, and carries different prose.
        _add_trace_event(
            db_session,
            task,
            event_type="tool_execution_start",
            timestamp=_ts(1),
            data={
                "tool_name": "tool_b",
                "tool_params": {},
                "tool_call_id": "call-b",
                "assistant_content": "Checking a different thing.",
            },
        )
        _add_trace_event(
            db_session,
            task,
            event_type="tool_execution_end",
            timestamp=_ts(2),  # B's end lands before A's end below.
            data={
                "tool_name": "tool_b",
                "tool_call_id": "call-b",
                "success": True,
                "result": {"ok": True},
            },
        )
        _add_trace_event(
            db_session,
            task,
            event_type="tool_execution_end",
            timestamp=_ts(100),  # A finishes much later.
            data={
                "tool_name": "tool_a",
                "tool_call_id": "call-a",
                "success": True,
                "result": {"ok": True},
            },
        )
        # Turn C: a much later, unrelated turn that coincidentally repeats
        # A's exact prose.
        _add_trace_event(
            db_session,
            task,
            event_type="tool_execution_start",
            timestamp=_ts(200),
            data={
                "tool_name": "tool_c",
                "tool_params": {},
                "tool_call_id": "call-c",
                "assistant_content": shared_prose,
            },
        )
        _add_trace_event(
            db_session,
            task,
            event_type="tool_execution_end",
            timestamp=_ts(201),
            data={
                "tool_name": "tool_c",
                "tool_call_id": "call-c",
                "success": True,
                "result": {"ok": True},
            },
        )

        messages = load_task_conversation_context_sync(db_session, int(task.id))
        assistant_by_tool = {}
        for index, message in enumerate(messages):
            if message["role"] != "assistant" or not message.get("tool_calls"):
                continue
            tool_message = messages[index + 1]
            assistant_by_tool[tool_message["tool_name"]] = message["content"]

        assert assistant_by_tool["tool_a"] == shared_prose
        assert assistant_by_tool["tool_c"] == shared_prose
        assert assistant_by_tool["tool_b"] == "Checking a different thing."
    finally:
        db_session.close()


def test_dedup_still_blanks_immediately_following_parallel_duplicate():
    """Fix 5 regression, other half: a genuine parallel batch (same start
    time, no other exchange between them in start order) must still have
    its duplicate prose blanked for the immediately-following exchange."""
    db_session = _create_db_session()
    try:
        task = _create_task(db_session)
        shared_prose = "Running parallel lookups."

        for index in range(3):
            call_id = f"batch-call-{index}"
            _add_trace_event(
                db_session,
                task,
                event_type="tool_execution_start",
                timestamp=_ts(index),
                data={
                    "tool_name": "list_files",
                    "tool_params": {"index": index},
                    "tool_call_id": call_id,
                    "assistant_content": shared_prose,
                },
            )
            _add_trace_event(
                db_session,
                task,
                event_type="tool_execution_end",
                timestamp=_ts(index + 10),
                data={
                    "tool_name": "list_files",
                    "tool_call_id": call_id,
                    "success": True,
                    "result": {"index": index},
                },
            )

        messages = load_task_conversation_context_sync(db_session, int(task.id))
        assistant_contents = [
            m["content"]
            for m in messages
            if m["role"] == "assistant" and m.get("tool_calls")
        ]
        assert len(assistant_contents) == 3
        assert assistant_contents.count(shared_prose) == 1
        assert assistant_contents.count("") == 2
    finally:
        db_session.close()


def test_tool_execution_end_missing_result_yields_degraded_dict_not_none():
    """Fix 6 regression: a ``tool_execution_end`` with no ``result`` key and
    not marked ``interrupted`` must produce a degraded placeholder dict,
    never a bare ``None`` flowing into ``raw_result``."""
    db_session = _create_db_session()
    try:
        task = _create_task(db_session)
        _add_trace_event(
            db_session,
            task,
            event_type="tool_execution_start",
            timestamp=_ts(0),
            data={
                "tool_name": "flaky_tool",
                "tool_params": {},
                "tool_call_id": "call-degraded",
            },
        )
        _add_trace_event(
            db_session,
            task,
            event_type="tool_execution_end",
            timestamp=_ts(1),
            data={
                "tool_name": "flaky_tool",
                "tool_call_id": "call-degraded",
                "success": True,
                # No "result" key, and not interrupted -- the degraded case.
            },
        )

        messages = load_task_conversation_context_sync(db_session, int(task.id))
        tool_message = next(m for m in messages if m["role"] == "tool")
        assert tool_message["raw_result"] is not None
        assert tool_message["raw_result"] == {
            "success": False,
            "status": "unknown",
            "error": "tool result missing from persisted trace event",
        }
    finally:
        db_session.close()


# ---------------------------------------------------------------------------
# Uploaded-image context refs (regression: our resume path replaced
# ``load_task_transcript`` with ``load_task_conversation_context_sync``, but
# this module never read ``attachments`` -- resumed conversations silently
# dropped uploaded images even though ``load_task_transcript`` still carries
# them. These tests pin the fix.)
# ---------------------------------------------------------------------------


def test_uploaded_images_on_transcript_rows_survive_reconstruction():
    db_session = _create_db_session()
    try:
        task = _create_task(db_session)
        base = _ts(0)
        _add_chat_message(
            db_session,
            task,
            role="user",
            content="What is shown?",
            created_at=base,
            attachments=[
                {
                    "file_id": "image-id",
                    "name": "diagram.png",
                    "size": 321,
                    "type": "image/png",
                },
                {
                    "file_id": "pdf-id",
                    "name": "notes.pdf",
                    "size": 654,
                    "type": "application/pdf",
                },
            ],
        )
        _add_chat_message(
            db_session,
            task,
            role="assistant",
            content="It's a diagram.",
            created_at=base + timedelta(seconds=1),
        )

        messages = load_task_conversation_context_sync(db_session, int(task.id))

        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "What is shown?"
        references = messages[0][CONTEXT_REFS_KEY]
        # Only the image attachment (not the pdf) becomes a context ref,
        # matching ``build_image_context_references``'s image-only filter.
        assert len(references) == 1
        assert references[0]["file_ref"]["file_id"] == "image-id"
        assert references[0]["metadata"] == {"source": "user_upload"}

        # No image attachment on the assistant row -- no key at all, not an
        # empty list, matching ``load_task_transcript``'s shape exactly.
        assert CONTEXT_REFS_KEY not in messages[1]
    finally:
        db_session.close()


def test_historical_image_budget_keeps_only_the_newest_n_across_messages():
    db_session = _create_db_session()
    try:
        task = _create_task(db_session)
        base = _ts(0)
        total_images = _MAX_HISTORICAL_IMAGE_CONTEXT_REFS + 2
        for index in range(total_images):
            _add_chat_message(
                db_session,
                task,
                role="user",
                content=f"Image {index}",
                created_at=base + timedelta(seconds=index),
                attachments=[
                    {
                        "file_id": f"image-{index}",
                        "name": f"image-{index}.png",
                        "type": "image/png",
                    }
                ],
            )

        messages = load_task_conversation_context_sync(db_session, int(task.id))

        assert len(messages) == total_images
        # The oldest two messages, over budget, carry no context refs at all.
        assert CONTEXT_REFS_KEY not in messages[0]
        assert CONTEXT_REFS_KEY not in messages[1]
        retained_ids = [
            message[CONTEXT_REFS_KEY][0]["file_ref"]["file_id"]
            for message in messages
            if CONTEXT_REFS_KEY in message
        ]
        assert retained_ids == [f"image-{index}" for index in range(2, total_images)]
    finally:
        db_session.close()
