import json
from datetime import datetime, timedelta, timezone

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from xagent.web.models.chat_message import TaskChatMessage
from xagent.web.models.database import Base
from xagent.web.models.task import Task, TaskStatus, TraceEvent
from xagent.web.models.user import User
from xagent.web.services.task_conversation_context_service import (
    load_task_conversation_context_sync,
)

STRUCTURED_HANDLE = {
    "workspace": "4b33784773d5",
    "branch": "review-pr-1392",
    "provider": "omp",
    "profile": "omp",
    "provider_session_id": "01a00aff-558e-7000-8ebf-ec547e8018b0",
    "last_run_id": "76ff56bf27554b899e74bce25f9dc769",
    "node_id": "node-0",
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


def _add_chat_message(db_session, task, *, role, content, created_at, turn_id=None):
    message = TaskChatMessage(
        task_id=int(task.id),
        user_id=int(task.user_id),
        role=role,
        content=content,
        message_type=role,
        turn_id=turn_id,
        created_at=created_at,
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
        # The original bug clipped mid-field, e.g. '"bran...' -- the full,
        # untruncated field must be present instead.
        assert '"branch": "review-pr-1392"' in serialized
        assert '"bran...' not in serialized
        assert json.loads(serialized)  # round-trips cleanly
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


def test_oversized_result_is_replaced_with_valid_json_marker():
    db_session = _create_db_session()
    try:
        task = _create_task(db_session)
        huge_payload = {"data": "x" * 1000}

        _add_trace_event(
            db_session,
            task,
            event_type="tool_execution_start",
            timestamp=_ts(0),
            data={
                "tool_name": "big_tool",
                "tool_params": {},
                "tool_call_id": "call-big",
            },
        )
        _add_trace_event(
            db_session,
            task,
            event_type="tool_execution_end",
            timestamp=_ts(1),
            data={
                "tool_name": "big_tool",
                "tool_call_id": "call-big",
                "success": True,
                "result": huge_payload,
            },
        )

        messages = load_task_conversation_context_sync(
            db_session, int(task.id), max_single_result_chars=100
        )

        tool_message = next(m for m in messages if m["role"] == "tool")
        raw_result = tool_message["raw_result"]
        assert raw_result["__omitted__"] is True
        assert raw_result["tool_name"] == "big_tool"
        assert raw_result["original_chars"] > 100

        round_tripped = json.loads(json.dumps(raw_result))
        assert round_tripped == raw_result
    finally:
        db_session.close()


def test_budget_drops_oldest_whole_exchanges_and_emits_notice():
    db_session = _create_db_session()
    try:
        task = _create_task(db_session)
        base = datetime(2026, 1, 1, tzinfo=timezone.utc)

        _add_chat_message(
            db_session,
            task,
            role="user",
            content="do things",
            created_at=base,
        )

        for index in range(5):
            call_id = f"call-{index}"
            _add_trace_event(
                db_session,
                task,
                event_type="tool_execution_start",
                timestamp=base + timedelta(seconds=1 + index * 2),
                data={
                    "tool_name": f"tool_{index}",
                    "tool_params": {"payload": "p" * 50},
                    "tool_call_id": call_id,
                },
            )
            _add_trace_event(
                db_session,
                task,
                event_type="tool_execution_end",
                timestamp=base + timedelta(seconds=2 + index * 2),
                data={
                    "tool_name": f"tool_{index}",
                    "tool_call_id": call_id,
                    "success": True,
                    "result": {"payload": "r" * 50, "index": index},
                },
            )

        _add_chat_message(
            db_session,
            task,
            role="assistant",
            content="done",
            created_at=base + timedelta(seconds=20),
        )

        full_messages = load_task_conversation_context_sync(db_session, int(task.id))
        full_tool_names = [m["tool_name"] for m in full_messages if m["role"] == "tool"]
        assert full_tool_names == [f"tool_{i}" for i in range(5)]

        # A budget that only fits the newest couple of exchanges plus the
        # transcript rows.
        budgeted_messages = load_task_conversation_context_sync(
            db_session, int(task.id), max_chars=250
        )

        # Every transcript message must survive regardless of the budget.
        transcript_contents = [
            m["content"]
            for m in budgeted_messages
            if m["role"] in ("user",)
            or (m["role"] == "assistant" and not m.get("tool_calls"))
        ]
        assert "do things" in transcript_contents
        assert "done" in transcript_contents

        # A leading system notice must name the dropped tools.
        assert budgeted_messages[0]["role"] == "system"
        notice = budgeted_messages[0]["content"]
        assert "dropped" in notice.lower() or "truncated" in notice.lower()

        kept_tool_names = [
            m["tool_name"] for m in budgeted_messages if m["role"] == "tool"
        ]
        assert kept_tool_names, "expected at least one surviving tool exchange"
        # Whole blocks: every surviving tool result matches one of the newest names.
        assert set(kept_tool_names).issubset(set(full_tool_names))
        # Oldest exchanges are the ones dropped.
        dropped = set(full_tool_names) - set(kept_tool_names)
        assert dropped
        for name in dropped:
            assert name in notice

        # Never a half-block: every surviving tool message is preceded by its assistant.
        for index, message in enumerate(budgeted_messages):
            if message["role"] != "tool":
                continue
            previous = budgeted_messages[index - 1]
            assert previous["role"] == "assistant"
            assert previous["tool_calls"][0]["id"] == message["tool_call_id"]
    finally:
        db_session.close()


def test_control_tools_are_excluded():
    db_session = _create_db_session()
    try:
        task = _create_task(db_session)

        for tool_name, call_id in (
            ("final_answer", "call-final"),
            ("ask_user_question", "call-ask"),
        ):
            _add_trace_event(
                db_session,
                task,
                event_type="tool_execution_start",
                timestamp=_ts(0 if tool_name == "final_answer" else 10),
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
                timestamp=_ts(1 if tool_name == "final_answer" else 11),
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
            timestamp=_ts(20),
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
            timestamp=_ts(21),
            data={
                "tool_name": "list_files",
                "tool_call_id": "call-real",
                "success": True,
                "result": {"files": []},
            },
        )

        messages = load_task_conversation_context_sync(db_session, int(task.id))
        tool_names = [m["tool_name"] for m in messages if m["role"] == "tool"]
        assert "final_answer" not in tool_names
        assert "ask_user_question" not in tool_names
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
