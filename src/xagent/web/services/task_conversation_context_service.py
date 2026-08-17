"""Faithfully reconstruct a task's prior conversation from persisted trace events.

Historically, ``task_execution_context_service.load_task_execution_context_messages``
collapsed all prior turns into a single synthetic ``system`` message: it kept only
the last few ``tool_execution_end`` events and clipped each result to a fixed
character budget with a blind string slice. That slicing could land mid-JSON and
destroy structured data (for example, cutting a continuation handle apart as
``"branch": "review-pr-1392", ...`` -> ``"bran``), which then fed a syntactically
broken fragment back to the model as "context" and caused it to hallucinate.

This module instead reconstructs the true ``assistant``/``tool`` message pairs a
live run would have produced, interleaved chronologically with the persisted
``user``/``assistant``/``system`` transcript rows. Any size limiting is done by
dropping whole exchange blocks (or replacing a whole oversized result with a
JSON-safe marker), never by slicing a value in the middle.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional, cast

from sqlalchemy import asc
from sqlalchemy.orm import Session

from ...config import (
    get_context_reconstruction_max_single_result_chars,
    get_context_reconstruction_max_tool_exchanges,
)
from ...core.agent.result import CONTROL_TOOL_NAMES
from ..models.chat_message import TaskChatMessage
from ..models.task import TraceEvent

logger = logging.getLogger(__name__)

# Tool-side trace event types persisted by PatternRuntime (runtime.py).
_TOOL_EVENT_TYPES = (
    "tool_execution_start",
    "tool_execution_end",
    "tool_execution_failed",
)

# Transcript roles kept from ``task_chat_messages``, mirroring
# ``normalize_transcript_messages`` (core/agent/transcript.py).
_TRANSCRIPT_ROLES = {"user", "assistant", "system"}

# Sort tiebreak when a tool exchange and a transcript row share the exact same
# timestamp: the exchange (0) sorts before the transcript row (1), since tool
# work logically precedes the turn's final answer.
_TIEBREAK_EXCHANGE = 0
_TIEBREAK_TRANSCRIPT = 1


@dataclass
class _TranscriptRow:
    """A single normalized ``task_chat_messages`` row."""

    row_id: int
    role: str
    content: str
    created_at: datetime


@dataclass
class _PendingToolStart:
    """A ``tool_execution_start`` awaiting its matching end/failure event."""

    assistant_content: str
    tool_name: str
    tool_params: Any
    sort_key: tuple[datetime, int]


@dataclass
class _ToolExchange:
    """A reconstructed assistant/tool message pair."""

    call_id: str
    tool_name: str
    tool_params: Any
    result: Any
    assistant_content: str
    sort_key: tuple[datetime, int]
    result_chars: int = field(default=0, compare=False)
    omitted: bool = field(default=False, compare=False)


@dataclass
class _MergeEntry:
    """One item queued for the chronological merge (R4)."""

    sort_key: tuple[datetime, int, int]
    kind: str  # "exchange" | "transcript"
    exchange: Optional[_ToolExchange] = None
    transcript: Optional[_TranscriptRow] = None


def load_task_conversation_context_sync(
    db: Session,
    task_id: int,
    *,
    before_message_id: int | None = None,
    max_tool_exchanges: int | None = None,
    max_chars: int | None = None,
    max_single_result_chars: int | None = None,
) -> list[dict[str, Any]]:
    """Reconstruct a task's prior conversation as planner-visible messages.

    Returns a chronological list of ``{"role": ..., ...}`` dicts covering the
    full persisted history (transcript rows plus reconstructed tool exchanges),
    subject to the safety limits below. This is a pure read: it never mutates
    the database and performs no async I/O, so it can be called from a worker
    thread that owns a live ``Session``.

    Args:
        db: An open, caller-owned database session.
        task_id: The task whose history to reconstruct.
        before_message_id: If given, only ``task_chat_messages`` rows with
            ``id < before_message_id`` are included (used when reconstructing
            context as of a specific point in the conversation).
        max_tool_exchanges: Maximum number of tool exchange pairs to keep;
            older whole exchanges are dropped first. Defaults to
            :func:`get_context_reconstruction_max_tool_exchanges`.
        max_chars: Total character budget across all emitted messages. Whole
            exchange blocks are dropped from the head (oldest first) until the
            budget is met. ``None`` disables this cap. Transcript messages are
            never dropped for budget reasons.
        max_single_result_chars: Per-tool-result size cap; a result whose
            serialized JSON exceeds this is replaced wholesale by an
            ``__omitted__`` marker. Defaults to
            :func:`get_context_reconstruction_max_single_result_chars`.
    """
    resolved_max_tool_exchanges = (
        max_tool_exchanges
        if max_tool_exchanges is not None
        else get_context_reconstruction_max_tool_exchanges()
    )
    resolved_max_single_result_chars = (
        max_single_result_chars
        if max_single_result_chars is not None
        else get_context_reconstruction_max_single_result_chars()
    )

    transcript_rows = _load_transcript_rows(
        db, task_id, before_message_id=before_message_id
    )
    exchanges = _load_tool_exchanges(db, task_id)

    dropped_result_tool_names: list[str] = []
    for exchange in exchanges:
        _apply_single_result_cap(
            exchange,
            max_single_result_chars=resolved_max_single_result_chars,
            dropped_names=dropped_result_tool_names,
        )

    exchanges, dropped_by_count = _apply_exchange_count_cap(
        exchanges, max_tool_exchanges=resolved_max_tool_exchanges
    )

    merged = _merge_chronologically(transcript_rows, exchanges)

    merged, dropped_by_budget = _apply_char_budget(merged, max_chars=max_chars)

    dropped_tool_names = dropped_by_count + dropped_by_budget
    messages = _render_messages(merged)
    messages = _final_pairing_sweep(messages)

    if dropped_tool_names or dropped_result_tool_names:
        notice = _build_dropped_notice(dropped_tool_names, dropped_result_tool_names)
        if notice:
            messages.insert(0, {"role": "system", "content": notice})

    return messages


# ---------------------------------------------------------------------------
# R1 -- transcript rows
# ---------------------------------------------------------------------------


def _load_transcript_rows(
    db: Session,
    task_id: int,
    *,
    before_message_id: int | None,
) -> list[_TranscriptRow]:
    """Load ``task_chat_messages`` rows, preserving ``created_at`` for merging.

    ``load_task_transcript`` is deliberately not reused here: it normalizes
    away ``created_at``, which this module needs for the chronological merge
    against trace events.
    """
    query = db.query(
        TaskChatMessage.id,
        TaskChatMessage.role,
        TaskChatMessage.content,
        TaskChatMessage.created_at,
    ).filter(TaskChatMessage.task_id == task_id)
    if before_message_id is not None:
        query = query.filter(TaskChatMessage.id < before_message_id)
    query = query.order_by(asc(TaskChatMessage.id))

    rows: list[_TranscriptRow] = []
    for row_id, role, content, created_at in query.all():
        normalized_role = str(role or "").strip().lower()
        normalized_content = str(content or "").strip()
        if normalized_role not in _TRANSCRIPT_ROLES or not normalized_content:
            continue
        rows.append(
            _TranscriptRow(
                row_id=int(row_id),
                role=normalized_role,
                content=normalized_content,
                created_at=_as_aware_utc(created_at),
            )
        )
    return rows


# ---------------------------------------------------------------------------
# R2/R3 -- tool events and exchange pairing
# ---------------------------------------------------------------------------


def _load_tool_exchanges(db: Session, task_id: int) -> list[_ToolExchange]:
    trace_rows = (
        db.query(TraceEvent)
        .filter(
            TraceEvent.task_id == task_id,
            TraceEvent.build_id.is_(None),
            TraceEvent.event_type.in_(_TOOL_EVENT_TYPES),
        )
        .order_by(asc(TraceEvent.timestamp), asc(TraceEvent.id))
        .all()
    )

    pending: dict[str, _PendingToolStart] = {}
    exchanges: list[_ToolExchange] = []
    last_assistant_content: str | None = None

    for trace_row in trace_rows:
        data: dict[str, Any] = (
            trace_row.data if isinstance(trace_row.data, dict) else {}
        )
        row_timestamp = _as_aware_utc(cast(datetime, trace_row.timestamp))
        row_id = int(trace_row.id)

        if trace_row.event_type == "tool_execution_start":
            call_id = str(data.get("tool_call_id") or "") or f"recon-{row_id}"
            assistant_content = str(data.get("assistant_content") or "").strip()
            pending[call_id] = _PendingToolStart(
                assistant_content=assistant_content,
                tool_name=str(data.get("tool_name") or ""),
                tool_params=data.get("tool_params"),
                sort_key=(row_timestamp, row_id),
            )
            continue

        # tool_execution_end / tool_execution_failed: pop the matching start so
        # each start is consumed by exactly one end.
        raw_call_id = str(data.get("tool_call_id") or "")
        start = None
        call_id = raw_call_id
        if raw_call_id:
            start = pending.pop(raw_call_id, None)
        if start is None:
            # No id, or id present but no matching start recorded (e.g. the
            # matching start fell outside this query, or lost its id). Fall
            # back to a synthesized key so pairing always has an identity.
            call_id = raw_call_id or f"recon-{row_id}"

        tool_name = str(data.get("tool_name") or "").strip()
        if not tool_name and start is not None:
            tool_name = start.tool_name
        if not tool_name:
            # Cannot identify the tool this exchange belongs to; skip rather
            # than emit an anonymous exchange the model can't reason about.
            continue

        if tool_name in CONTROL_TOOL_NAMES:
            # final_answer / send_message / ask_user_question are pseudo-tools
            # whose observable effect already exists as a TaskChatMessage row.
            # Re-injecting them as tool exchanges would duplicate that content,
            # and replaying a completed final_answer risks the model believing
            # the *current* turn is already answered.
            continue

        tool_params = data.get("tool_params")
        if tool_params is None and start is not None:
            tool_params = start.tool_params
        if tool_params is None:
            tool_params = {}

        result = _resolve_tool_result(cast(str, trace_row.event_type), data)

        assistant_content = start.assistant_content if start is not None else ""
        if assistant_content and assistant_content == last_assistant_content:
            # Same prose repeated across a parallel tool-call batch (the
            # runtime's active_react_step_id can't delimit iterations, so
            # concurrent calls all carry the same assistant_content). Emit it
            # once, not N times.
            assistant_content = ""
        elif assistant_content:
            last_assistant_content = assistant_content

        sort_key = start.sort_key if start is not None else (row_timestamp, row_id)

        exchanges.append(
            _ToolExchange(
                call_id=call_id,
                tool_name=tool_name,
                tool_params=tool_params,
                result=result,
                assistant_content=assistant_content,
                sort_key=sort_key,
            )
        )

    return exchanges


def _resolve_tool_result(event_type: str, data: dict[str, Any]) -> Any:
    if event_type == "tool_execution_end":
        if bool(data.get("interrupted")) and "result" not in data:
            return {
                "success": False,
                "interrupted": True,
                "error": data.get("interrupt_reason"),
            }
        return data.get("result")

    # tool_execution_failed
    if "result" in data:
        return data.get("result")
    return {
        "success": False,
        "status": "error",
        "error": data.get("error") or data.get("error_message"),
    }


# ---------------------------------------------------------------------------
# R5.1 -- per-result size cap
# ---------------------------------------------------------------------------


def _apply_single_result_cap(
    exchange: _ToolExchange,
    *,
    max_single_result_chars: int,
    dropped_names: list[str],
) -> None:
    try:
        serialized = json.dumps(exchange.result, ensure_ascii=False, default=str)
    except Exception:
        serialized = str(exchange.result)
    exchange.result_chars = len(serialized)
    if exchange.result_chars <= max_single_result_chars:
        return
    exchange.omitted = True
    exchange.result = {
        "__omitted__": True,
        "reason": "tool result exceeded reconstruction budget",
        "tool_name": exchange.tool_name,
        "original_chars": exchange.result_chars,
    }
    dropped_names.append(exchange.tool_name)


# ---------------------------------------------------------------------------
# R5.2 -- exchange-count cap
# ---------------------------------------------------------------------------


def _apply_exchange_count_cap(
    exchanges: list[_ToolExchange],
    *,
    max_tool_exchanges: int,
) -> tuple[list[_ToolExchange], list[str]]:
    if len(exchanges) <= max_tool_exchanges:
        return exchanges, []
    # Exchanges are already in chronological (start-time) order from the R2/R3
    # pass; keep the newest N, drop older whole blocks.
    keep_from = len(exchanges) - max_tool_exchanges
    dropped = exchanges[:keep_from]
    kept = exchanges[keep_from:]
    return kept, [exchange.tool_name for exchange in dropped]


# ---------------------------------------------------------------------------
# R4 -- chronological merge
# ---------------------------------------------------------------------------


def _merge_chronologically(
    transcript_rows: list[_TranscriptRow],
    exchanges: list[_ToolExchange],
) -> list[_MergeEntry]:
    entries: list[_MergeEntry] = []
    for row in transcript_rows:
        entries.append(
            _MergeEntry(
                sort_key=(row.created_at, _TIEBREAK_TRANSCRIPT, row.row_id),
                kind="transcript",
                transcript=row,
            )
        )
    for index, exchange in enumerate(exchanges):
        timestamp, event_id = exchange.sort_key
        entries.append(
            _MergeEntry(
                sort_key=(timestamp, _TIEBREAK_EXCHANGE, event_id),
                kind="exchange",
                exchange=exchange,
            )
        )

    entries.sort(key=lambda entry: entry.sort_key)

    # Clock-skew guard: created_at uses the DB clock while trace timestamps use
    # the app clock, so an exchange can sort before the conversation's first
    # user message purely due to skew. Move any such exchange to just after it.
    first_user_index = next(
        (
            index
            for index, entry in enumerate(entries)
            if entry.kind == "transcript"
            and entry.transcript is not None
            and entry.transcript.role == "user"
        ),
        None,
    )
    if first_user_index is not None:
        misplaced = [
            entry
            for index, entry in enumerate(entries[:first_user_index])
            if entry.kind == "exchange"
        ]
        if misplaced:
            remaining = [
                entry
                for index, entry in enumerate(entries)
                if not (index < first_user_index and entry.kind == "exchange")
            ]
            # first_user_index shifts left by however many exchange entries
            # were pulled out ahead of it.
            new_first_user_index = first_user_index - len(misplaced)
            entries = (
                remaining[: new_first_user_index + 1]
                + misplaced
                + remaining[new_first_user_index + 1 :]
            )

    return entries


# ---------------------------------------------------------------------------
# R5.3 -- total character budget
# ---------------------------------------------------------------------------


def _entry_char_estimate(entry: _MergeEntry) -> int:
    if entry.kind == "transcript":
        assert entry.transcript is not None
        return len(entry.transcript.content)
    assert entry.exchange is not None
    exchange = entry.exchange
    tool_calls_json = json.dumps(
        [
            {
                "id": exchange.call_id,
                "type": "function",
                "function": {
                    "name": exchange.tool_name,
                    "arguments": json.dumps(
                        exchange.tool_params or {}, ensure_ascii=False, default=str
                    ),
                },
            }
        ],
        ensure_ascii=False,
    )
    return len(exchange.assistant_content) + len(tool_calls_json)


def _apply_char_budget(
    entries: list[_MergeEntry],
    *,
    max_chars: int | None,
) -> tuple[list[_MergeEntry], list[str]]:
    if max_chars is None:
        return entries, []

    total = sum(_entry_char_estimate(entry) for entry in entries)
    if total <= max_chars:
        return entries, []

    # Accumulate from the tail backwards; drop whole exchange blocks from the
    # head once the budget is exceeded. Transcript messages are never dropped.
    kept_reversed: list[_MergeEntry] = []
    running = 0
    dropped_names: list[str] = []
    over_budget = False

    for entry in reversed(entries):
        size = _entry_char_estimate(entry)
        if entry.kind == "transcript":
            kept_reversed.append(entry)
            running += size
            continue
        if not over_budget and running + size <= max_chars:
            kept_reversed.append(entry)
            running += size
        else:
            over_budget = True
            assert entry.exchange is not None
            dropped_names.append(entry.exchange.tool_name)

    kept_reversed.reverse()
    return kept_reversed, dropped_names


# ---------------------------------------------------------------------------
# Rendering + notice + final pairing sweep
# ---------------------------------------------------------------------------


def _render_messages(entries: list[_MergeEntry]) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    for entry in entries:
        if entry.kind == "transcript":
            assert entry.transcript is not None
            messages.append(
                {"role": entry.transcript.role, "content": entry.transcript.content}
            )
            continue

        assert entry.exchange is not None
        exchange = entry.exchange
        messages.append(
            {
                "role": "assistant",
                "content": exchange.assistant_content or "",
                "tool_calls": [
                    {
                        "id": exchange.call_id,
                        "type": "function",
                        "function": {
                            "name": exchange.tool_name,
                            "arguments": json.dumps(
                                exchange.tool_params or {},
                                ensure_ascii=False,
                                default=str,
                            ),
                        },
                    }
                ],
            }
        )
        messages.append(
            {
                "role": "tool",
                "tool_call_id": exchange.call_id,
                "content": "",
                "tool_name": exchange.tool_name,
                "raw_result": exchange.result,
            }
        )
    return messages


def _final_pairing_sweep(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Drop any ``tool`` message not immediately preceded by its declaring assistant.

    Construction should never produce this, but this mirrors the invariant
    ``ExecutionContext._sanitize_tool_message_pairs`` protects at the live
    context layer.
    """
    sanitized: list[dict[str, Any]] = []
    for index, message in enumerate(messages):
        if message.get("role") != "tool":
            sanitized.append(message)
            continue
        previous = messages[index - 1] if index > 0 else None
        if (
            previous is not None
            and previous.get("role") == "assistant"
            and any(
                str(call.get("id")) == str(message.get("tool_call_id"))
                for call in previous.get("tool_calls") or []
            )
        ):
            sanitized.append(message)
        # else: orphaned tool message, drop it.
    return sanitized


def _build_dropped_notice(
    dropped_by_size_or_count: list[str],
    dropped_by_result_size: list[str],
) -> str:
    """Describe dropped/omitted tool observations, modeled on
    ``ExecutionContext._dropped_tool_results_notice``.
    """
    all_names = [*dropped_by_size_or_count, *dropped_by_result_size]
    if not all_names:
        return ""
    counts: dict[str, int] = {}
    for name in all_names:
        counts[name] = counts.get(name, 0) + 1

    total = sum(counts.values())
    call_label = "call was" if total == 1 else "calls were"
    lines = [
        f"Raw observations from {total} tool {call_label} dropped or truncated "
        "during context reconstruction. Their exact values are no longer in "
        "context. Treat any figure not literally present elsewhere in this "
        "conversation as unavailable, and re-read or re-query rather than "
        "reconstructing it from memory. Tools affected:",
    ]
    for name, count in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
        label = name or "unnamed tool"
        lines.append(f"- {label} ({count})" if count > 1 else f"- {label}")
    return "\n".join(lines)


def _as_aware_utc(value: datetime) -> datetime:
    """Normalize a possibly-naive datetime to aware UTC for safe comparison.

    ``TaskChatMessage.created_at`` and ``TraceEvent.timestamp`` are both
    declared ``DateTime(timezone=True)``, but SQLite (used by tests) returns
    naive datetimes while Postgres returns aware ones. Comparing a naive and
    an aware datetime raises ``TypeError``, so every value is normalized here
    before it is ever used as a sort key.
    """
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value
