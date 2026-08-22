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
``user``/``assistant``/``system`` transcript rows.

This version reconstructs the full history with no size limiting: it is
deliberately not wired into any caller yet. Bounds (a per-result size cap, a
total exchange-count cap, and a total character budget) and the observability
around which of them fired ship in a follow-up change, alongside the wiring
that puts this service on the hot path.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional, cast

from sqlalchemy import asc
from sqlalchemy.orm import Session

from ...core.agent.attachments import build_image_context_references
from ...core.agent.result import CONTROL_TOOL_NAMES
from ...core.context_ref import CONTEXT_REFS_KEY, ContextReference
from ..models.chat_message import TaskChatMessage
from ..models.task import TraceEvent
from .chat_history_service import _MAX_HISTORICAL_IMAGE_CONTEXT_REFS

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
    # Raw ``TaskChatMessage.attachments`` payload, kept only long enough for
    # ``_attach_historical_image_context_refs`` to project it into
    # ``context_refs`` below; never read after ``_load_transcript_rows``
    # returns.
    attachments: Any = field(default=None, compare=False)
    # Uploaded-image context refs surviving the historical-image budget (see
    # ``_attach_historical_image_context_refs``). Populated after filtering,
    # so this is empty for any row that never reaches ``_render_messages``.
    context_refs: tuple[ContextReference, ...] = field(default=())


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
    # Raw ``TraceEvent.step_id`` discriminator (see ``_load_tool_exchanges``),
    # empty string when the column was unset. Used only to group exchanges
    # produced by the same turn for the R4 merge below -- never for pairing,
    # which already has its own step-scoped keying.
    step_id: str = ""


@dataclass
class _MergeEntry:
    """One item queued for the chronological merge (R4)."""

    sort_key: tuple[datetime, int, int]
    kind: str  # "group" | "transcript"
    group: Optional[list[_ToolExchange]] = None
    transcript: Optional[_TranscriptRow] = None


def load_task_conversation_context_sync(
    db: Session,
    task_id: int,
    *,
    before_message_id: int | None = None,
) -> list[dict[str, Any]]:
    """Reconstruct a task's prior conversation as planner-visible messages.

    Returns a chronological list of ``{"role": ..., ...}`` dicts covering the
    full persisted history: transcript rows plus reconstructed tool exchanges.
    This is a pure read: it never mutates the database and performs no async
    I/O, so it can be called from a worker thread that owns a live
    ``Session``.

    Args:
        db: An open, caller-owned database session.
        task_id: The task whose history to reconstruct.
        before_message_id: If given, only ``task_chat_messages`` rows with
            ``id < before_message_id`` are included (used when reconstructing
            context as of a specific point in the conversation).
    """
    transcript_rows = _load_transcript_rows(
        db, task_id, before_message_id=before_message_id
    )
    exchanges = _load_tool_exchanges(db, task_id)

    merged = _merge_chronologically(transcript_rows, exchanges)

    messages = _render_messages(merged)
    messages = _final_pairing_sweep(messages)

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

    ``chat_history_service``'s row-loading helpers normalize away
    ``created_at``, which this module needs for the chronological merge
    against trace events, so the rows are queried directly here instead.
    ``attachments`` is selected alongside the existing narrow column set
    (rather than loading the whole ORM row) so the historical-image budget
    below can reach uploaded-image metadata without widening this query's
    result shape beyond what image support actually needs.
    """
    query = db.query(
        TaskChatMessage.id,
        TaskChatMessage.role,
        TaskChatMessage.content,
        TaskChatMessage.created_at,
        TaskChatMessage.attachments,
    ).filter(TaskChatMessage.task_id == task_id)
    if before_message_id is not None:
        query = query.filter(TaskChatMessage.id < before_message_id)
    query = query.order_by(asc(TaskChatMessage.id))

    rows: list[_TranscriptRow] = []
    for row_id, role, content, created_at, attachments in query.all():
        normalized_role = str(role or "").strip().lower()
        if normalized_role not in _TRANSCRIPT_ROLES:
            continue
        normalized_content = str(content or "").strip()
        rows.append(
            _TranscriptRow(
                row_id=int(row_id),
                role=normalized_role,
                content=normalized_content,
                created_at=_as_aware_utc(created_at),
                attachments=attachments,
            )
        )
    # Attach image context refs *before* dropping blank-content rows:
    # ``persist_user_message_no_commit`` (chat_history_service.py) documents
    # that "a row with empty content but non-empty attachments is still
    # persisted" -- an image-only user turn. Filtering on content alone
    # before this call would discard that row, and its attachment, together
    # -- the ref would never get the chance to be built. The retention test
    # below (content OR context refs) mirrors
    # ``normalize_transcript_messages`` (core/agent/transcript.py).
    _attach_historical_image_context_refs(rows)
    return [row for row in rows if row.content or row.context_refs]


def _attach_historical_image_context_refs(rows: list[_TranscriptRow]) -> None:
    """Mirror ``chat_history_service.load_task_transcript``'s image budget.

    Same reverse scan, same global cap: walk the transcript newest-first
    and keep attaching image context refs to each row until
    ``_MAX_HISTORICAL_IMAGE_CONTEXT_REFS`` (shared with
    ``load_task_transcript`` so the two paths can never silently diverge)
    is exhausted, so only the most recent uploaded images survive. This is
    the only bound this module currently applies; the tool-exchange caps
    live in a follow-up change (see the module docstring).
    """
    remaining = _MAX_HISTORICAL_IMAGE_CONTEXT_REFS
    for index in range(len(rows) - 1, -1, -1):
        if remaining <= 0:
            break
        references = build_image_context_references(rows[index].attachments)
        kept = references[:remaining]
        rows[index].context_refs = kept
        remaining -= len(kept)
        rows[index].attachments = None


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

    # ``tool_call_id`` alone is not a safe pending-map key: the DAG pattern
    # runs steps CONCURRENTLY (``dag.py``'s ``asyncio.create_task`` batch),
    # and when a provider omits real tool-call ids, ``_normalize_tool_calls``
    # (react.py) falls back to ``f"tool_call_{index}"`` where ``index`` is
    # only unique WITHIN one LLM response -- two concurrent DAG steps can
    # each emit "tool_call_0". Without a per-step discriminator, step A's
    # pending start can be overwritten by step B's, and A's end then pops
    # B's start: the WRONG tool_name/assistant_content gets attached to a
    # result, corrupting history rather than merely losing it.
    #
    # ``TraceEvent.step_id`` is that discriminator: ``PatternRuntime`` always
    # threads a step id through ``_step_id_from_payload`` (runtime.py) into
    # the dedicated ``step_id`` COLUMN (not ``data``) via
    # ``stage_trace_event_row`` (trace_event_staging.py). The DAG's
    # ``_with_step`` (dag.py) stamps each concurrent step's tool calls with
    # its own unique ``step_id``/``dag_step_id``, so concurrent branches get
    # different keys here. ReAct sets the same step id on both the start and
    # end of a given call (one LLM turn = one step), so this is a no-op for
    # ReAct's already-unique-per-turn ids -- and rows written before this
    # column was populated simply carry ``step_id=None``, which normalizes
    # to the same empty-string discriminator for both events, reproducing
    # the old bare-``tool_call_id`` keying exactly.
    pending: dict[tuple[str, str], _PendingToolStart] = {}
    exchanges: list[_ToolExchange] = []

    for trace_row in trace_rows:
        data: dict[str, Any] = (
            trace_row.data if isinstance(trace_row.data, dict) else {}
        )
        row_timestamp = _as_aware_utc(cast(datetime, trace_row.timestamp))
        row_id = int(trace_row.id)
        step_discriminator = str(trace_row.step_id or "")

        if trace_row.event_type == "tool_execution_start":
            call_id = str(data.get("tool_call_id") or "") or f"recon-{row_id}"
            assistant_content = str(data.get("assistant_content") or "").strip()
            pending[(step_discriminator, call_id)] = _PendingToolStart(
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
            start = pending.pop((step_discriminator, raw_call_id), None)
        if start is None:
            # No id, or id present but no matching start recorded (e.g. the
            # matching start fell outside this query, or lost its id, or its
            # step id doesn't match this end's). Fall back to a synthesized
            # key so pairing always has an identity.
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

        # Dedup of prose repeated across a parallel tool-call batch happens
        # below, in final sort-key order -- not here. Trace rows are iterated
        # in the order the *end* events arrive, which for a parallel batch
        # can differ from the *start* order (whichever call finishes first
        # gets processed first), so deduping inline here would attach the
        # prose to the wrong exchange and could blank a duplicate that lands
        # far from its match in final order. Keep the raw value for now.
        assistant_content = start.assistant_content if start is not None else ""

        sort_key = start.sort_key if start is not None else (row_timestamp, row_id)

        exchanges.append(
            _ToolExchange(
                call_id=call_id,
                tool_name=tool_name,
                tool_params=tool_params,
                result=result,
                assistant_content=assistant_content,
                sort_key=sort_key,
                step_id=step_discriminator,
            )
        )

    # Sort by the *start* time (sort_key), matching the order exchanges will
    # ultimately appear in once merged with the transcript (R4).
    exchanges.sort(key=lambda exchange: exchange.sort_key)
    _dedupe_parallel_batch_prose(exchanges)
    return exchanges


def _dedupe_parallel_batch_prose(exchanges: list[_ToolExchange]) -> None:
    """Blank assistant prose that exactly repeats the immediately preceding
    exchange, in final chronological (sort-key) order.

    Parallel tool-call batches share one ``assistant_content`` value (the
    runtime's ``active_react_step_id`` can't delimit iterations, so every
    concurrent call in the batch is stamped with the same prose); this
    blanks all but the first occurrence in a run of adjacent duplicates so
    the model doesn't see the same paragraph N times.

    Deliberately positional rather than "seen anywhere before": comparing
    against everything seen earlier in the stream would also blank prose
    from a genuinely different, later turn that happens to be byte-identical
    (e.g. a retried step re-emitting the same message) -- that is real
    content, not a duplicate, and must not be dropped.
    """
    previous: str | None = None
    for exchange in exchanges:
        content = exchange.assistant_content
        if not content:
            continue
        if content == previous:
            exchange.assistant_content = ""
            continue
        previous = content


def _resolve_tool_result(event_type: str, data: dict[str, Any]) -> Any:
    if event_type == "tool_execution_end":
        if bool(data.get("interrupted")) and "result" not in data:
            return {
                "success": False,
                "interrupted": True,
                "error": data.get("interrupt_reason"),
            }
        result = data.get("result")
        if result is None:
            # A "tool_execution_end" with neither a "result" key nor the
            # interrupted marker above -- degraded data, but still a
            # completed call. Mirror the shape of the other degraded cases
            # here rather than letting a bare ``None`` flow into
            # ``add_tool_result``.
            return {
                "success": False,
                "status": "unknown",
                "error": "tool result missing from persisted trace event",
            }
        return result

    # tool_execution_failed
    if "result" in data:
        return data.get("result")
    return {
        "success": False,
        "status": "error",
        "error": data.get("error") or data.get("error_message"),
    }


# ---------------------------------------------------------------------------
# R4 -- chronological merge
# ---------------------------------------------------------------------------


def _group_exchanges_by_turn(
    exchanges: list[_ToolExchange],
) -> list[list[_ToolExchange]]:
    """Group tool exchanges by the turn that produced them.

    ``TraceEvent.step_id`` is the grouping key. For ReAct tasks it carries
    ``PatternRuntime.active_react_step_id`` (runtime.py), which is minted
    fresh (``f"react_{uuid4().hex[:8]}"``) in ``on_pattern_start`` and cleared
    in ``on_pattern_end``/``on_pattern_error`` -- so every tool event produced
    while handling one user turn shares one step id, and no two turns ever
    share one. For DAG-produced events, ``_with_step`` (pattern/dag/dag.py)
    stamps each concurrently-scheduled DAG step with its own ``step_id``, so
    a single turn that fans out over several DAG steps yields *several*
    groups rather than one -- that's fine here: each such group still carries
    its own real timestamps and gets positioned (and guarded, see
    ``_relocate_groups_before_their_user_message`` below) independently, so
    the atomicity and ordering invariants below hold per group regardless of
    how many groups one turn happens to produce.

    Rows written before ``TraceEvent.step_id`` existed (or otherwise missing
    it) carry ``step_id=None``, which ``_load_tool_exchanges`` normalizes to
    ``""``. Grouping *all* such exchanges under one empty-string key would
    silently merge unrelated turns back together -- exactly the bug this
    function exists to avoid. So each exchange with no step id instead gets
    its own singleton group, keyed by its position in the (already
    time-sorted) ``exchanges`` list: this reproduces the pre-grouping,
    per-exchange behavior exactly for that degraded case, rather than
    inventing a new failure mode for it.
    """
    groups: dict[str, list[_ToolExchange]] = {}
    order: list[str] = []
    for index, exchange in enumerate(exchanges):
        key = exchange.step_id or f"__no_step_id__{index}"
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(exchange)
    return [groups[key] for key in order]


def _relocate_groups_before_their_user_message(
    entries: list[_MergeEntry],
) -> list[_MergeEntry]:
    """Generalize the clock-skew guard from the first user message to every one.

    ``created_at`` (``TaskChatMessage``, DB clock via ``server_default``)
    and the trace ``timestamp`` (``TraceEvent``, app clock) are different
    clocks, so a turn's tool-exchange group can sort a hair before the very
    user message that triggered it purely from sub-second skew. The old
    guard only relocated exchanges that sorted before the conversation's
    very first user message -- a case that's easy to make a hard rule for
    ("nothing can precede the task's opening message"), but it left every
    later turn unprotected.

    This generalizes that rule to every user message by walking the merged,
    timestamp-sorted entries and buffering consecutive ``"group"`` entries.
    When a user-role transcript row is reached, any buffered groups are
    flushed *after* it instead of before -- exactly what the old guard did
    for the first user message, just repeated at each one. When a
    non-user-role transcript row is reached first (typically that group's
    own turn's assistant answer), the buffered groups are flushed in place,
    unchanged: a group is only ever adjacent-before a user row (with no
    other transcript row between them) when either skew misplaced it or it's
    trailing, unanswered content at the very end of the merge, and in the
    normal well-timestamped case a turn's own answer row already sits
    between its group and the next user message, so this never fires there.

    Grouping (see ``_group_exchanges_by_turn``) makes this correct at whole
    turn granularity: a group is placed as one unit, so it can never
    straddle a user message, and within a group the members keep their
    relative (single-clock, timestamp) order from ``_load_tool_exchanges``.

    Residual gap (deliberately not closed here): which group belongs to
    which user message is still inferred from adjacency in timestamp order,
    not from a durable link. ``TaskChatMessage.turn_id`` is an indexed,
    populated column, but ``TraceEvent`` has no ``turn_id`` column, and the
    only code that projects ``turn_id`` into trace ``data``
    (``PatternRuntime._with_dag_turn_id``, runtime.py) is called from the
    DAG hooks only -- never from ``on_tool_start``/``on_tool_end``/
    ``on_tool_error``. Threading ``turn_id`` through those hooks and onto
    ``TraceEvent`` would let this function match a group to its user message
    exactly instead of by adjacency; that plumbing is deferred to a
    follow-up change.
    """
    result: list[_MergeEntry] = []
    pending_groups: list[_MergeEntry] = []
    for entry in entries:
        if entry.kind == "group":
            pending_groups.append(entry)
            continue
        is_user_row = (
            entry.kind == "transcript"
            and entry.transcript is not None
            and entry.transcript.role == "user"
        )
        if is_user_row:
            result.append(entry)
            result.extend(pending_groups)
            pending_groups = []
            continue
        result.extend(pending_groups)
        pending_groups = []
        result.append(entry)
    result.extend(pending_groups)
    return result


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

    for group in _group_exchanges_by_turn(exchanges):
        # Position the group by its earliest member's timestamp -- still a
        # timestamp comparison, but the unit that can be misplaced by
        # cross-clock skew is now a whole turn's worth of exchanges, not a
        # single event, and distinct turns are separated by human thinking
        # time (seconds to minutes), not the sub-second skew between the two
        # clocks. Within the group, members keep the timestamp order already
        # established in ``_load_tool_exchanges`` (one turn, one clock).
        anchor = min(
            (timestamp, _TIEBREAK_EXCHANGE, event_id)
            for timestamp, event_id in (exchange.sort_key for exchange in group)
        )
        entries.append(
            _MergeEntry(
                sort_key=anchor,
                kind="group",
                group=group,
            )
        )

    entries.sort(key=lambda entry: entry.sort_key)

    return _relocate_groups_before_their_user_message(entries)


# ---------------------------------------------------------------------------
# Rendering + final pairing sweep
# ---------------------------------------------------------------------------


def _render_messages(entries: list[_MergeEntry]) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    for entry in entries:
        if entry.kind == "transcript":
            assert entry.transcript is not None
            item: dict[str, Any] = {
                "role": entry.transcript.role,
                "content": entry.transcript.content,
            }
            if entry.transcript.context_refs:
                item[CONTEXT_REFS_KEY] = [
                    reference.durable_dict()
                    for reference in entry.transcript.context_refs
                ]
            messages.append(item)
            continue

        assert entry.group is not None
        for exchange in entry.group:
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
            and _ids_match(message.get("tool_call_id"), previous.get("tool_calls"))
        ):
            sanitized.append(message)
        # else: orphaned tool message, drop it.
    return sanitized


def _ids_match(tool_call_id: Any, tool_calls: Any) -> bool:
    """Whether ``tool_calls`` declares ``tool_call_id``.

    Compares only present ids. ``str()`` on two ``None``s would render
    ``"None" == "None"`` and report a match, which would let this sweep --
    whose whole job is catching pairs construction should never have
    produced -- pass an orphan through in exactly the case it exists for.
    """
    if tool_call_id is None:
        return False
    target = str(tool_call_id)
    for call in tool_calls or []:
        call_id = call.get("id") if isinstance(call, dict) else None
        if call_id is not None and str(call_id) == target:
            return True
    return False


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
