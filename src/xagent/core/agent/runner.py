from __future__ import annotations

import asyncio
import contextlib
import inspect
import logging
from dataclasses import dataclass, replace
from enum import Enum
from pathlib import Path
from typing import Any, Literal, cast
from uuid import uuid4

from ...config import (
    COMPACT_THRESHOLD_DEFAULT,
    get_compact_threshold_default,
)
from ..context_materializer import WorkspaceContextReferenceResolver
from ..context_ref import CONTEXT_REFS_KEY, ContextReference
from ..inline_file_delivery import InlineFileDelivery
from ..model.intent import enter_goal, exit_goal
from ..task_runtime import (
    PREFERRED_INPUT_MODALITIES_METADATA_KEY,
    normalize_input_modalities,
)
from ..workspace import WorkspaceManager
from .attachments import build_image_context_references
from .checkpoint import (
    CheckpointCorruptError,
    CheckpointReadError,
    read_latest_checkpoint_payload,
)
from .context import ContextManager, ExecutionContext
from .context.execution import (
    COMPACT_THRESHOLD_SOURCE_DEFAULT,
    TOOL_EVIDENCE_REMOVED_METADATA_KEY,
    Message,
    context_write_lock,
    derive_compact_threshold,
    serialize_message,
)
from .language import reset_output_language_to_request_context
from .result import extract_assistant_message, set_assistant_message
from .runtime import (
    THRESHOLD_WARNING_KEY_PREFIX,
    ExecutionInterrupted,
    PatternRuntime,
    compact_model_key,
    load_pattern_checkpoint,
    warn_once_per_model,
    warn_restored_compact_threshold,
)

logger = logging.getLogger(__name__)

# Metadata keys the engine writes as run facts. Client input reaches
# ``context.metadata`` verbatim through two surfaces -- the top-level metadata
# dict and the nested request_context -- so both are filtered at the single
# merge point rather than defended again at each reader. The restored branch of
# that merge point reaches neither surface: it carries only the modality
# preference over and drops the rest of the current run's metadata, so no client
# key reaches a context rebuilt from a checkpoint.
RESERVED_ENGINE_METADATA_KEYS = frozenset({TOOL_EVIDENCE_REMOVED_METADATA_KEY})


@dataclass
class ExecutionControl:
    """In-memory control state for an active execution."""

    runtime: PatternRuntime
    task: str | None


class UserMessageInjectionOutcome(str, Enum):
    """What ``AgentRunner.inject_user_message`` actually did on a given
    call, threaded unmodified through every layer that forwards its
    result (``ExecutionRegistry``, ``AgentExecutionAdapter``,
    ``AgentService.post_user_message``).

    ``POSTED_FRESH`` and ``POSTED_REPLAY`` both mean a live, durable user
    turn answers the caller's message -- a short-circuited repeat
    ``turn_id`` (``POSTED_REPLAY``) returns the same context the first
    attempt did without writing anything new, while ``POSTED_FRESH``
    persisted one. ``NOT_POSTED`` is the empty string and the other two
    members are not, so an unmodified ``if not posted`` / ``bool(posted)``
    caller keeps asking exactly the question it always asked -- "did this
    hand back a usable context at all" -- unaffected by the fresh/replay
    split. Telling a replay apart from a fresh write requires comparing
    identity against ``POSTED_REPLAY``; truthiness alone cannot and must
    not be used for that.

    ``OUTCOME_UNKNOWN`` is deliberately truthy too. It means the candidate
    checkpoint write raised, and the read-back taken to disambiguate it
    could not itself confirm or rule out the write, so whether the message
    is durably persisted is genuinely unknown -- it was NOT applied to the
    live context either way. Existing ``if not posted`` / ``bool(posted)``
    callers therefore treat it the same as a real post: "handed off,
    caller must not silently retry or re-open for reply", which is the
    safe direction for something that might already be committed. The one
    place this is wrong is a raw "did this succeed" boolean -- callers that
    need to distinguish success from unknown must compare identity against
    ``POSTED_FRESH`` (already true of every existing ``is POSTED_FRESH``
    guard) or explicitly check for ``OUTCOME_UNKNOWN``. See ``runner.py``'s
    ``inject_user_message`` docstring and the R1 design for the persistence
    protocol this disambiguates.
    """

    NOT_POSTED = ""
    POSTED_FRESH = "posted_fresh"
    POSTED_REPLAY = "posted_replay"
    OUTCOME_UNKNOWN = "outcome_unknown"


class InjectionSettleRefusedError(RuntimeError):
    """``AgentRunner.settle_injection_against_checkpoint`` refused to run
    because one of its preconditions does not hold (a run of the execution
    is active on this runner). Nothing was read or written: the caller's
    outcome is exactly as unknown as it was before the call."""


@dataclass(frozen=True)
class UserMessageInjectionResult:
    """Bundles the execution context an injection attempt produced (or
    found already holding the answered turn) with what kind of write, if
    any, produced it.

    ``context`` is ``None`` only when there was no execution context to
    inject into at all (``outcome`` is then ``NOT_POSTED``); every layer
    downstream of the registry drops the raw context and forwards only
    ``outcome``, since nothing past that boundary reads the context
    object itself today.
    """

    context: ExecutionContext | None
    outcome: UserMessageInjectionOutcome


class AgentRunner:
    """Execute an agent by materializing an execution context and invoking patterns."""

    # Total attempts ``inject_user_message`` makes at persisting one
    # candidate before giving up and reporting ``OUTCOME_UNKNOWN``: the
    # first attempt plus this many retries triggered by an ambiguous
    # (persist-write-exception, read-back-also-inconclusive) failure.
    _INJECTION_UNKNOWN_RETRY_ATTEMPTS = 3
    # Backoff between retries, multiplied by the (1-based) retry index so
    # it increases slightly on each additional attempt.
    _INJECTION_UNKNOWN_RETRY_BACKOFF_SECONDS = 0.05

    def __init__(
        self,
        agent: Any,
        *,
        workspace_manager: WorkspaceManager | None = None,
        memory_manager: Any | None = None,
        tracer: Any | None = None,
        callbacks: list[Any] | None = None,
        context_manager: ContextManager | None = None,
        workspace_base_dir: str = "workspace",
        workspace_enabled: bool = True,
        scope_segments: tuple[str, ...] = (),
        outbound_message_handler: Any | None = None,
    ) -> None:
        self.agent = agent
        self.workspace_manager = workspace_manager or WorkspaceManager()
        self.memory_manager = memory_manager
        self.tracer = tracer
        self.callbacks = callbacks or []
        self.context_manager = context_manager or ContextManager()
        self.workspace_base_dir = workspace_base_dir
        self.workspace_enabled = workspace_enabled
        self.scope_segments = scope_segments
        self.outbound_message_handler = outbound_message_handler
        self._active_controls: dict[str, ExecutionControl] = {}

    async def run(
        self,
        task: str | None,
        user_id: str | None = None,
        execution_id: str | None = None,
        *,
        session_id: str | None = None,
        workspace_id: str | None = None,
        allowed_external_dirs: list[str] | None = None,
        base_dir: str | None = None,
        resume: bool = False,
        checkpoint: dict[str, Any] | None = None,
        runtime: PatternRuntime | None = None,
        interrupt_checker: Any | None = None,
        outbound_message_handler: Any | None = None,
        streaming_handler: Any | None = None,
        extra_tools: list[Any] | None = None,
        metadata: dict[str, Any] | None = None,
        initial_messages: list[dict[str, Any]] | None = None,
        task_context_refs: tuple[ContextReference, ...] = (),
    ) -> dict[str, Any]:
        execution_id = execution_id or str(uuid4())
        checkpoint = checkpoint or (
            await self._load_latest_checkpoint(execution_id) if resume else None
        )
        if (
            resume
            and checkpoint is not None
            and not isinstance(checkpoint.get("context"), dict)
        ):
            raise CheckpointCorruptError(
                "Resume checkpoint exists but carries no execution context."
            )
        if task is None:
            task = self._resolve_task(
                task=task,
                checkpoint=checkpoint,
                execution_id=execution_id,
            )
        if checkpoint and isinstance(checkpoint.get("context"), dict):
            reset_output_language_to_request_context(checkpoint)
            context = ExecutionContext.from_dict(checkpoint["context"])
            warn_restored_compact_threshold(context, getattr(self.agent, "llm", None))
            self._merge_context_metadata(context, metadata, restored=True)
            self.context_manager.set_context(context)
            execution_id = context.execution_id
            workspace = None
        else:
            context, workspace = await self._build_context(
                task=task,
                execution_id=execution_id,
                user_id=user_id,
                session_id=session_id,
                workspace_id=workspace_id,
                allowed_external_dirs=allowed_external_dirs,
                base_dir=base_dir,
                metadata=metadata,
            )
            replay_messages = initial_messages or []
            if (
                replay_messages
                and str(replay_messages[0].get("role") or "").strip() == "assistant"
            ):
                # A task's persisted history can begin with an
                # assistant-role message today only via a marketplace Hire
                # flow's seeded persona greeting - there is no other path
                # that persists an assistant row before any user message.
                # Anthropic's Messages API (and every claude_compatible
                # provider routed through it) rejects a request whose first
                # message isn't role "user", so correct it once here,
                # before this history is ever replayed into context. This
                # is deliberately not done in get_messages_for_llm(), which
                # also serves truncated windows and tool-call/tool-result
                # pairs that legitimately start mid-conversation.
                context.add_user_message(
                    "(conversation start)",
                    metadata={"_xagent_synthetic": "leading_user_turn"},
                )
            for message in replay_messages:
                role = str(message.get("role") or "").strip()
                content = str(message.get("content") or "").strip()
                context_refs = message.get(
                    CONTEXT_REFS_KEY, message.get("context_refs", ())
                )
                if not role:
                    continue
                if not (
                    content
                    or context_refs
                    or message.get("tool_calls")
                    or role == "tool"
                ):
                    continue
                if (
                    role == "tool"
                    and message.get("tool_name") is not None
                    and "raw_result" in message
                ):
                    # Replay through add_tool_result so it gets the same
                    # sanitization/formatting and metadata (raw_result,
                    # tool_name) as a live tool observation would.
                    context.add_tool_result(
                        tool_name=str(message["tool_name"]),
                        result=message["raw_result"],
                        tool_call_id=message.get("tool_call_id"),
                        context_refs=context_refs,
                    )
                elif role == "assistant" and message.get("tool_calls"):
                    context.add_assistant_message(
                        content,
                        tool_calls=message["tool_calls"],
                        context_refs=context_refs,
                    )
                else:
                    context.add_message(
                        role,
                        content,
                        context_refs=context_refs,
                        tool_calls=message.get("tool_calls"),
                        tool_call_id=message.get("tool_call_id"),
                    )
            if task:
                context.add_user_message(
                    task,
                    metadata=self._initial_user_message_metadata(context),
                    context_refs=task_context_refs,
                )

        # A runner registered by post_user_message before the host installed
        # its handler would otherwise resume handler-less (#1328); an
        # explicit handler passed to the resumed run wins over the
        # constructor-time one. Passing None here means "inherit the
        # constructor-time handler," not "clear it"; no caller clears a
        # handler today, so there is deliberately no sentinel for that.
        runtime = runtime or PatternRuntime(
            tracer=self.tracer,
            execution_id=execution_id,
            interrupt_checker=interrupt_checker,
            outbound_message_handler=(
                outbound_message_handler
                if outbound_message_handler is not None
                else self.outbound_message_handler
            ),
        )
        if self.workspace_enabled and runtime.context_ref_resolver is None:
            if workspace is None:
                workspace_base = base_dir or self.workspace_base_dir
                if context.workspace_path:
                    workspace_base = str(Path(context.workspace_path).parent)
                workspace = self.workspace_manager.get_or_create_workspace(
                    base_dir=workspace_base,
                    task_id=context.workspace_id or workspace_id or execution_id,
                    allowed_external_dirs=allowed_external_dirs,
                    scope_segments=self.scope_segments,
                )
                if inspect.isawaitable(workspace):
                    workspace = await workspace
            runtime.context_ref_resolver = WorkspaceContextReferenceResolver(workspace)
        if workspace is not None and callable(
            getattr(workspace, "register_delivery_file", None)
        ):
            runtime.inline_file_delivery = InlineFileDelivery(workspace)
        self._active_controls[execution_id] = ExecutionControl(
            runtime=runtime,
            task=task,
        )

        # Establish the user's request as the turn's goal. The "auto" model
        # routes on this rather than on the scaffolded sub-prompt a given LLM
        # call carries; finer units (DAG steps) override it with their own goal.
        goal_token = enter_goal(task)

        await self._dispatch_callback(
            "on_run_start",
            runner=self,
            context=context,
            resume=resume,
            checkpoint=checkpoint,
        )

        try:
            patterns = list(getattr(self.agent, "patterns", []))
            if not patterns:
                result = {
                    "success": False,
                    "error": "Agent has no execution patterns configured.",
                    "execution_id": execution_id,
                    "context": context,
                }
                await self._dispatch_callback(
                    "on_run_end", runner=self, context=context, result=result
                )
                return result

            tools = [*getattr(self.agent, "tools", []), *(extra_tools or [])]
            pattern_errors: list[dict[str, Any]] = []
            teardown_status: str | None = "failed"

            try:
                await self._setup_tools(tools, task_id=execution_id)
                for pattern in patterns:
                    load_pattern_checkpoint(pattern, checkpoint)
                    teardown_status = "failed"
                    try:
                        result = await pattern.run(
                            **self._build_pattern_kwargs(
                                pattern=pattern,
                                task=task or "",
                                context=context,
                                tools=tools,
                                runtime=runtime,
                                streaming_handler=streaming_handler,
                            )
                        )
                    except ExecutionInterrupted as exc:
                        teardown_status = "interrupted"
                        normalized = {
                            "success": False,
                            "status": "interrupted",
                            "error": str(exc),
                            "execution_id": execution_id,
                            "context": context,
                            "pattern": pattern.__class__.__name__,
                        }
                        await self._dispatch_callback(
                            "on_run_end",
                            runner=self,
                            context=context,
                            result=normalized,
                        )
                        return normalized
                    except Exception as exc:  # noqa: BLE001
                        teardown_status = "failed"
                        logger.exception(
                            "Pattern %s failed", pattern.__class__.__name__
                        )
                        pattern_errors.append(
                            {
                                "pattern": pattern.__class__.__name__,
                                "error": str(exc),
                                "exception_type": exc.__class__.__name__,
                            }
                        )
                        continue

                    # Normalize user-facing answer text, never tool arguments,
                    # input files or reasoning. Streaming and buffered patterns
                    # share this delivery owner and its registered-file cache.
                    raw_answer = (
                        extract_assistant_message(result)
                        if isinstance(result, dict)
                        else result
                    )
                    if isinstance(raw_answer, str):
                        delivered_answer = await runtime.prepare_final_answer(
                            raw_answer
                        )
                        if delivered_answer != raw_answer:
                            if isinstance(result, dict):
                                result = dict(result)
                                set_assistant_message(result, delivered_answer)
                            else:
                                result = {"success": True, "output": delivered_answer}
                            for index in range(len(context.messages) - 1, -1, -1):
                                context_message = context.messages[index]
                                if (
                                    context_message.role == "assistant"
                                    and context_message.content == raw_answer
                                ):
                                    context.messages[index] = replace(
                                        context_message, content=delivered_answer
                                    )
                                    break
                    normalized = self._normalize_result(
                        result=result,
                        pattern=pattern,
                        context=context,
                        execution_id=execution_id,
                    )
                    if normalized.get("success"):
                        teardown_status = str(normalized.get("status") or "completed")
                        await self._dispatch_callback(
                            "on_run_end",
                            runner=self,
                            context=context,
                            result=normalized,
                        )
                        return normalized
                    normalized_status = str(
                        normalized.get("status") or "failed"
                    ).strip()
                    teardown_status = normalized_status or "failed"
                    if (
                        normalized_status == "waiting_for_user"
                        and normalized.get("clarification_draft") is not None
                    ):
                        # A pattern that hands back a question to answer must
                        # not still have other step tasks running -- those
                        # would keep mutating shared state (self.status,
                        # active step bookkeeping) after this run() call has
                        # already told the caller it is safe to resume from
                        # a single waiting step. Only DAGPattern implements
                        # this today; any other pattern is read as having no
                        # live tasks (auto.py's top-level case included --
                        # its run() cannot return while a nested DAG child
                        # still has tasks in flight, so that path is a known,
                        # accepted gap rather than a real miss).
                        has_live_step_tasks = getattr(
                            pattern, "has_live_step_tasks", None
                        )
                        if callable(has_live_step_tasks) and has_live_step_tasks():
                            raise AssertionError(
                                f"{pattern.__class__.__name__} for execution "
                                f"{execution_id} returned a waiting_for_user "
                                "result while it still has live step tasks."
                            )
                    if normalized_status in {"interrupted", "waiting_for_user"}:
                        await self._dispatch_callback(
                            "on_run_end",
                            runner=self,
                            context=context,
                            result=normalized,
                        )
                        return normalized

                    pattern_errors.append(
                        {
                            "pattern": pattern.__class__.__name__,
                            "error": normalized.get(
                                "error", "Pattern failed without a detailed error."
                            ),
                            "result": normalized,
                        }
                    )
            finally:
                await self._teardown_tools(
                    tools,
                    task_id=execution_id,
                    execution_status=teardown_status,
                )

            if len(pattern_errors) == 1:
                single_result = pattern_errors[0].get("result")
                if isinstance(single_result, dict):
                    await self._dispatch_callback(
                        "on_run_end", runner=self, context=context, result=single_result
                    )
                    return single_result

            result = {
                "success": False,
                "error": f"All {len(patterns)} patterns failed or returned unsuccessful results.",
                "pattern_errors": pattern_errors,
                "patterns_attempted": len(patterns),
                "execution_id": execution_id,
                "context": context,
            }
            await self._dispatch_callback(
                "on_run_end", runner=self, context=context, result=result
            )
            return result
        finally:
            runtime.discard_inline_file_streams()
            self._active_controls.pop(execution_id, None)
            exit_goal(goal_token)

    def pause(self, execution_id: str, reason: str | None = None) -> bool:
        control = self._active_controls.get(execution_id)
        if control is None:
            return False
        control.runtime.request_interrupt(reason or "paused by runner")
        return True

    def cancel(self, execution_id: str, reason: str | None = None) -> bool:
        control = self._active_controls.get(execution_id)
        if control is None:
            return False
        control.runtime.request_interrupt(reason or "cancelled by runner")
        return True

    async def resume(
        self,
        execution_id: str,
        *,
        task: str | None = None,
        user_id: str | None = None,
        session_id: str | None = None,
        workspace_id: str | None = None,
        allowed_external_dirs: list[str] | None = None,
        base_dir: str | None = None,
        streaming_handler: Any | None = None,
        extra_tools: list[Any] | None = None,
        metadata: dict[str, Any] | None = None,
        interrupt_checker: Any | None = None,
        outbound_message_handler: Any | None = None,
    ) -> dict[str, Any]:
        checkpoint = await self._load_latest_checkpoint(execution_id)
        resolved_task = self._resolve_task(
            task=task,
            checkpoint=checkpoint,
            execution_id=execution_id,
        )
        return await self.run(
            task=resolved_task,
            user_id=user_id,
            execution_id=execution_id,
            session_id=session_id,
            workspace_id=workspace_id,
            allowed_external_dirs=allowed_external_dirs,
            base_dir=base_dir,
            resume=True,
            checkpoint=checkpoint,
            streaming_handler=streaming_handler,
            extra_tools=extra_tools,
            metadata=metadata,
            interrupt_checker=interrupt_checker,
            outbound_message_handler=outbound_message_handler,
        )

    async def _resolve_injection_context(
        self, execution_id: str
    ) -> tuple[ExecutionContext, dict[str, Any] | None] | None:
        """Resolve the live context an injection should write into.

        Returns ``(context, cold_start_checkpoint)``: the second element is
        the checkpoint a context was just rebuilt from (``None`` when a
        live context was already registered), reused below as the merge
        baseline so the same window is not read twice. Returns ``None``
        when there is nothing to inject into at all -- no live context and
        no checkpoint to cold-start from.

        This is the single place that decides which context object an
        injection targets. It is the intended seam for R3 (recognizing a
        cached context as stale and rebuilding it); R1 keeps today's
        semantics unchanged except for using ``set_context_if_absent``
        below.
        """
        context = self.context_manager.get_context(execution_id)
        if context is not None:
            return context, None

        checkpoint = await self._load_latest_checkpoint(execution_id)
        if checkpoint is None:
            return None
        if not isinstance(checkpoint.get("context"), dict):
            raise CheckpointCorruptError(
                "Stored checkpoint carries no execution context to restore."
            )
        reset_output_language_to_request_context(checkpoint)
        context = ExecutionContext.from_dict(checkpoint["context"])
        warn_restored_compact_threshold(context, getattr(self.agent, "llm", None))
        # ``set_context_if_absent``, not ``set_context``: two concurrent
        # cold starts for the same execution must converge on one context
        # object -- and therefore one ``context_write_lock`` -- instead of
        # each installing its own, mutually oblivious object.
        context = self.context_manager.set_context_if_absent(context)
        return context, checkpoint

    @staticmethod
    def _find_replayed_turn(
        context: ExecutionContext,
        turn_id: str,
        execution_message: str,
    ) -> bool:
        """True if ``turn_id`` is already applied to ``context`` with the
        same content -- the caller should short-circuit to
        ``POSTED_REPLAY`` rather than writing again. Raises ``ValueError``
        when the turn id is already associated with different content: a
        genuine conflict, not a replay.

        Must be called under ``context_write_lock``: a same-turn injection
        racing this one may commit while this call was waiting for the
        lock, and the dedupe check has to see that write.
        """
        for existing in context.messages:
            existing_metadata = getattr(existing, "metadata", None)
            if (
                getattr(existing, "role", None) != "user"
                or not isinstance(existing_metadata, dict)
                or existing_metadata.get("turn_id") != turn_id
            ):
                continue
            if existing.content != execution_message:
                raise ValueError(
                    "turn_id is already associated with a different user message"
                )
            return True
        return False

    async def _resolve_injection_baseline(
        self,
        execution_id: str,
        cold_start_checkpoint: dict[str, Any] | None,
    ) -> dict[str, Any] | None:
        """Resolve the checkpoint-merge baseline before any write.

        A live runtime's cached checkpoint wins over a fresh read; a
        cold-start read that already happened while resolving the context
        is reused instead of reading the same window twice. A read
        failure here propagates as-is, before anything has been written,
        so a retry after it sees zero residue and genuinely persists
        rather than finding a ghost confirmation from the rejected
        attempt.
        """
        control = self._active_controls.get(execution_id)
        if control is not None and control.runtime.last_checkpoint is not None:
            return control.runtime.last_checkpoint
        if cold_start_checkpoint is not None:
            return cold_start_checkpoint
        return await self._load_latest_checkpoint(execution_id)

    async def _confirm_injected_turn(
        self,
        execution_id: str,
        turn_id: str,
        execution_message: str,
    ) -> Literal["found", "absent", "unknown"]:
        """Disambiguate a persist-write exception by reading back the
        latest checkpoint, still under ``context_write_lock``.

        - ``"found"``: the turn is in the latest checkpoint with matching
          content -- the write committed but the confirmation was lost.
          The caller applies it and reports ``POSTED_FRESH``.
        - ``"absent"``: the read succeeded and the turn is not there --
          the write definitely did not commit. The caller re-raises the
          original exception with zero residue.
        - ``"unknown"``: the read itself failed, returned a malformed
          payload, or (unexpectedly) found the same turn id with
          different content. Whether the write committed cannot be
          determined; the caller does not apply the message and reports
          ``OUTCOME_UNKNOWN``.

        Reads via the raw ``read_latest_checkpoint_payload`` loader rather
        than ``self._load_latest_checkpoint`` -- the latter collapses any
        non-``dict`` payload into ``None``, which would misreport a
        malformed read as a confirmed ``"absent"`` (write definitely did
        not commit) instead of the ``"unknown"`` this ambiguous case
        actually is. Here, only a genuine ``None`` (no checkpoint at all)
        means ``"absent"``; any other non-``dict`` result is ``"unknown"``.
        """
        if self.tracer is None:
            return "absent"
        try:
            payload = await read_latest_checkpoint_payload(self.tracer, execution_id)
        except Exception:
            logger.warning(
                "read-back after injection persist failure could not confirm "
                "turn %s for %s",
                turn_id,
                execution_id,
                exc_info=True,
            )
            return "unknown"

        if payload is None:
            return "absent"
        if not isinstance(payload, dict):
            logger.warning(
                "read-back after injection persist failure returned a "
                "non-dict payload (%s) for %s (turn %s)",
                type(payload).__name__,
                execution_id,
                turn_id,
            )
            return "unknown"

        context_payload = payload.get("context") if isinstance(payload, dict) else None
        messages = (
            context_payload.get("messages")
            if isinstance(context_payload, dict)
            else None
        )
        if not isinstance(messages, list):
            logger.warning(
                "read-back after injection persist failure returned a "
                "malformed payload for %s (turn %s)",
                execution_id,
                turn_id,
            )
            return "unknown"

        for entry in messages:
            if not isinstance(entry, dict) or entry.get("role") != "user":
                continue
            entry_metadata = entry.get("metadata")
            if (
                not isinstance(entry_metadata, dict)
                or entry_metadata.get("turn_id") != turn_id
            ):
                continue
            if entry.get("content") == execution_message:
                return "found"
            logger.warning(
                "read-back after injection persist failure found turn %s "
                "with different content for %s",
                turn_id,
                execution_id,
            )
            return "unknown"
        return "absent"

    async def inject_user_message(
        self,
        execution_id: str,
        message: str | None = None,
        *,
        execution_message: str | None = None,
        display_message: str | None = None,
        files: list[dict[str, Any]] | None = None,
        turn_id: str | None = None,
        request_interrupt: bool = True,
        reason: str | None = None,
    ) -> UserMessageInjectionResult:
        # Pure parameter validation up front, no side effects yet.
        #
        # Display-vs-execution split: ``execution_message`` is the prompt
        # the agent runtime sees (may be enriched with file refs / system
        # context); ``display_message`` is what the chat bubble should
        # show. Both fall back to ``message`` for legacy callers.
        resolved_execution_message = (
            execution_message if execution_message is not None else message
        )
        if resolved_execution_message is None:
            raise ValueError(
                "inject_user_message requires message or execution_message"
            )
        if display_message is None and message is None:
            raise ValueError(
                "inject_user_message requires display_message when "
                "execution_message is provided without legacy message"
            )
        resolved_display_message = (
            display_message if display_message is not None else message
        )
        requested_turn_id = turn_id.strip() if turn_id and turn_id.strip() else None

        resolved = await self._resolve_injection_context(execution_id)
        if resolved is None:
            return UserMessageInjectionResult(
                context=None,
                outcome=UserMessageInjectionOutcome.NOT_POSTED,
            )
        context, cold_start_checkpoint = resolved

        # Attach files + display text to the new Message so they survive
        # checkpoint round-trips: Message.metadata is serialized by
        # ExecutionContext. The on_user_message_posted callback reads
        # display_message back from metadata so the chat bubble shows the
        # user-typed text rather than the LLM-augmented prompt.
        metadata: dict[str, Any] = {"display_message": resolved_display_message}
        if files is not None:
            metadata["files"] = files
        if requested_turn_id is not None:
            metadata["turn_id"] = requested_turn_id
        tid = self._ensure_user_message_turn_id(metadata)

        outcome = UserMessageInjectionOutcome.POSTED_FRESH
        added: Any = None

        # Snapshot, write and confirm the candidate all happen under the
        # context's write lock -- see ``context_write_lock`` for the
        # invariant this protects: live pattern code must never observe
        # this message before its persist is confirmed, and a concurrent
        # ``PatternRuntime.checkpoint`` for the same context must never
        # interleave with this write. Callbacks and ``pause`` stay OUTSIDE
        # the lock (below), so this block never dispatches one.
        #
        # Bounded retry for a persist-write exception whose read-back
        # confirmation itself cannot disambiguate ("unknown"): the write
        # may actually have committed, so dropping straight to
        # OUTCOME_UNKNOWN on the very first ambiguous failure needlessly
        # loses the message on a merely flaky store. Content is idempotent
        # by turn_id -- ``_find_replayed_turn``/``_confirm_injected_turn``
        # both match on (turn_id, content) -- so retrying the same
        # candidate is always safe: it either lands once or is recognized
        # as already landed. Only if every attempt's confirmation comes
        # back "unknown" is OUTCOME_UNKNOWN reported.
        #
        # Each attempt takes the EXCLUSIVE lock on its own and the backoff
        # sleep runs with it released, so a slow retry never stalls every
        # pattern checkpoint for this context through the sleep. Because
        # the lock is dropped in between, every attempt re-runs the dedupe
        # check and rebuilds its baseline and candidate from the CURRENT
        # live context: a same-turn injection may have committed, or a
        # pattern checkpoint may have landed, while it was released.
        #
        # Residual risk: the store calls inside one attempt carry no
        # timeout of their own (no existing config helper bounds a single
        # checkpoint write/read here), so a hung store still holds the
        # EXCLUSIVE lock -- and blocks this context's pattern checkpoints
        # -- for as long as that one call hangs. Lease loss still cancels
        # the whole injection through ``run_while_task_lease_owned``.
        for attempt in range(self._INJECTION_UNKNOWN_RETRY_ATTEMPTS):
            if attempt > 0:
                await asyncio.sleep(
                    self._INJECTION_UNKNOWN_RETRY_BACKOFF_SECONDS * attempt
                )
            async with context_write_lock(context):
                # Waiting for the lock may have let a same-turn injection
                # that was already in flight commit, so the dedupe check has
                # to run here, not before acquiring the lock. On a retry it
                # runs against ``tid`` even for a generated id (harmless:
                # nothing else can have applied that id) so every retry
                # re-checks the live context it is about to snapshot.
                if (
                    requested_turn_id is not None or attempt > 0
                ) and self._find_replayed_turn(
                    context, tid, resolved_execution_message
                ):
                    outcome = UserMessageInjectionOutcome.POSTED_REPLAY
                    break
                try:
                    checkpoint_baseline = await self._resolve_injection_baseline(
                        execution_id, cold_start_checkpoint if attempt == 0 else None
                    )
                except CheckpointReadError:
                    if attempt == 0:
                        # Nothing written yet: propagate with zero residue
                        # (see ``_resolve_injection_baseline``).
                        raise
                    # An earlier attempt of THIS injection may already have
                    # committed, so a failed re-read must not surface as
                    # "not injected" (callers read a read error that way);
                    # it leaves this attempt unknown, like its predecessor.
                    outcome = UserMessageInjectionOutcome.OUTCOME_UNKNOWN
                    continue
                new_message = Message(
                    role="user",
                    content=resolved_execution_message,
                    metadata=metadata,
                    context_refs=build_image_context_references(files),
                )
                # Candidate is a snapshot + the new message, not the live
                # context -- the live context is not touched until the
                # write below is confirmed (see the exception handling).
                candidate = context.to_dict()
                candidate["messages"].append(serialize_message(new_message))
                self._pending_marker_into(candidate["metadata"], new_message)
                try:
                    # Persist BEFORE applying to the live context or
                    # emitting the trace, so the message is durable even if
                    # the trace dispatch fails -- the resume path's catch-up
                    # logic in TraceEventCallback.on_run_start will replay
                    # the marked turn.
                    await self._persist_injected_context(
                        execution_id=execution_id,
                        context_payload=candidate,
                        label="user_message_injected",
                        baseline=checkpoint_baseline,
                    )
                except asyncio.CancelledError:
                    # Cancellation carries no outcome for the caller, and
                    # the persisted store may or may not have committed
                    # before the cancellation landed -- do not guess. Never
                    # applied to the live context either way, and never
                    # retried.
                    raise
                except Exception:
                    verdict = await self._confirm_injected_turn(
                        execution_id, tid, resolved_execution_message
                    )
                    if verdict == "absent":
                        # Definitely not committed: re-raise with zero
                        # residue, exactly like a rejected attempt today.
                        # Not retried -- a confirmed absence is a genuine
                        # rejection, not an ambiguity a retry could
                        # resolve.
                        raise
                    if verdict == "unknown":
                        # Whether it committed cannot be determined from
                        # this attempt alone. Try again (after releasing
                        # the lock for the backoff) unless attempts are
                        # exhausted.
                        outcome = UserMessageInjectionOutcome.OUTCOME_UNKNOWN
                        continue
                    # verdict == "found": committed, only the confirmation
                    # was lost. Apply as POSTED_FRESH below.
                # Linearization point: only now, still holding the lock,
                # does the message become visible to live pattern code
                # reading ``context.messages``.
                outcome = UserMessageInjectionOutcome.POSTED_FRESH
                added = context.append_message(new_message)
                self._pending_marker_into(context.metadata, new_message)
                break

        if outcome is not UserMessageInjectionOutcome.POSTED_FRESH:
            if request_interrupt:
                self.pause(execution_id, reason=reason or "new user message")
            return UserMessageInjectionResult(context=context, outcome=outcome)

        # ---- lock released; the rest mirrors pre-R1 behavior ----
        # Snapshot the watermark BEFORE the callback so we can detect a
        # change (see comment below) and persist it.
        watermark_before = self._read_trace_watermark(context)
        traced_turn_ids_before = self._read_traced_turn_ids(context)
        try:
            await self._dispatch_callback(
                "on_user_message_posted",
                runner=self,
                context=context,
                message=added,
                files=files,
            )
        except Exception:
            # The injected-context checkpoint above is the durable acceptance
            # boundary. Trace projection can be replayed from its pending
            # marker, so it must not turn an accepted user message into a
            # rejected delivery.
            logger.warning(
                "user-message callback failed after injection checkpoint for %s",
                execution_id,
                exc_info=True,
            )
        # Re-persist when the trace callback advanced the watermark —
        # without this a worker crash between trace emission and the next
        # checkpoint would let the resume path replay the same user_message
        # event because the watermark was still living only in memory.
        # The same persist also clears the pending marker since the trace
        # has now been emitted; doing both in one persist keeps the
        # invariant {pending => never traced yet} on every durable state.
        watermark_after = self._read_trace_watermark(context)
        traced_turn_ids_after = self._read_traced_turn_ids(context)
        if (
            watermark_after and watermark_after != watermark_before
        ) or traced_turn_ids_after != traced_turn_ids_before:
            async with context_write_lock(context):
                self._clear_pending_user_message_marker(context)
                try:
                    # Re-resolve the baseline HERE, inside the lock, rather
                    # than reusing ``checkpoint_baseline`` captured before
                    # ``on_user_message_posted`` ran: a pattern checkpoint
                    # can land (SHARED-mode, concurrently with the callback
                    # since the exclusive lock was released for it above)
                    # and advance ``control.runtime.last_checkpoint`` in
                    # the meantime. Merging this watermark write onto the
                    # stale pre-callback baseline would silently regress
                    # whatever pattern_state that newer checkpoint wrote.
                    watermark_baseline = await self._resolve_injection_baseline(
                        execution_id, None
                    )
                    await self._persist_injected_context(
                        execution_id=execution_id,
                        context_payload=context.to_dict(),
                        label="user_message_trace_watermark",
                        baseline=watermark_baseline,
                    )
                except Exception:
                    # Declared swallow point: the message was already durably
                    # accepted by the injection checkpoint above, so a failure
                    # here (including a checkpoint read/write error) only
                    # costs the watermark re-persist and must not turn an
                    # accepted message into a rejected delivery. Preserve the
                    # pending marker for resume catch-up when the trace
                    # watermark cannot be persisted after acceptance.
                    logger.warning(
                        "user-message watermark checkpoint failed after injection for %s",
                        execution_id,
                        exc_info=True,
                    )
        if request_interrupt:
            self.pause(execution_id, reason=reason or "new user message")
        return UserMessageInjectionResult(
            context=context,
            outcome=UserMessageInjectionOutcome.POSTED_FRESH,
        )

    async def settle_injection_against_checkpoint(
        self,
        execution_id: str,
        message: str | None = None,
        *,
        execution_message: str | None = None,
        display_message: str | None = None,
        files: list[dict[str, Any]] | None = None,
        turn_id: str,
    ) -> UserMessageInjectionResult:
        """Settle an earlier ``OUTCOME_UNKNOWN`` injection of ``turn_id``
        against the latest durable checkpoint, WITHOUT touching the
        ``ContextManager``.

        ``OUTCOME_UNKNOWN`` is never applied to the registered context, so
        that object cannot say whether the turn committed; the latest
        checkpoint -- the state the caller's subsequent ``resume`` rebuilds
        from anyway -- can. Each attempt reads it, rebuilds a DETACHED
        ``ExecutionContext`` from it, and:

        - the turn is there with the same content -> ``POSTED_REPLAY`` (an
          earlier attempt committed; only its confirmation was lost);
        - it is absent -> persists a candidate (the detached context plus
          the message and its pending-trace marker) merged onto that same
          checkpoint, disambiguating a write exception by read-back exactly
          like ``inject_user_message``: ``"found"`` -> ``POSTED_FRESH``,
          ``"absent"`` -> the write exception propagates, ``"unknown"`` ->
          retried (bounded), then ``OUTCOME_UNKNOWN``;
        - there is no checkpoint at all -> ``NOT_POSTED`` (nothing ever
          committed); a stale registered context is ignored and left as is.

        No callback, trace watermark persist or ``pause`` runs: nothing is
        live to show the message to. The durable pending marker hands the
        trace to the resume's catch-up (``TraceEventCallback.on_run_start``).
        ``result.context`` is the detached object, never a registered one.

        Preconditions: the caller holds the execution's lease, no run of it
        is active (enforced here for this runner, see
        ``InjectionSettleRefusedError``), and no other injection of it is in
        flight.

        Serialization: every attempt's read -> dedupe -> write -> confirm
        runs under the EXCLUSIVE ``context_write_lock`` of whatever context
        is registered for ``execution_id`` at that moment (the detached
        object's own gate would serialize against nothing). The
        ``ContextManager`` is process-wide while ``_active_controls`` is
        per-runner, so a pattern still checkpointing through that object
        from another runner cannot interleave with this write and have its
        state regressed by it. Residual (R3) caveat: a writer holding a
        DIFFERENT object for the same execution -- one replaced in, or never
        installed into, the registry, or another process -- is not
        serialized by that gate; only the lease preconditions exclude it.
        """
        resolved_execution_message = (
            execution_message if execution_message is not None else message
        )
        if resolved_execution_message is None:
            raise ValueError(
                "settle_injection_against_checkpoint requires message or "
                "execution_message"
            )
        if display_message is None and message is None:
            raise ValueError(
                "settle_injection_against_checkpoint requires display_message "
                "when execution_message is provided without legacy message"
            )
        resolved_display_message = (
            display_message if display_message is not None else message
        )
        tid = turn_id.strip() if isinstance(turn_id, str) else ""
        if not tid:
            # Dedupe is by turn id; without the caller's original id there
            # is nothing to settle against.
            raise ValueError("settle_injection_against_checkpoint requires turn_id")

        metadata: dict[str, Any] = {
            "display_message": resolved_display_message,
            "turn_id": tid,
        }
        if files is not None:
            metadata["files"] = files

        outcome = UserMessageInjectionOutcome.OUTCOME_UNKNOWN
        detached: ExecutionContext | None = None
        for attempt in range(self._INJECTION_UNKNOWN_RETRY_ATTEMPTS):
            if attempt > 0:
                await asyncio.sleep(
                    self._INJECTION_UNKNOWN_RETRY_BACKOFF_SECONDS * attempt
                )
            # Checked on every attempt, not only up front: a run may have
            # started on this runner while the gate was released for the
            # backoff.
            if execution_id in self._active_controls:
                raise InjectionSettleRefusedError(
                    f"cannot settle an injection for {execution_id} while a "
                    "run of it is active on this runner"
                )
            registered = self.context_manager.get_context(execution_id)
            gate: Any = (
                context_write_lock(registered)
                if registered is not None
                else contextlib.nullcontext()
            )
            async with gate:
                try:
                    checkpoint = await self._load_latest_checkpoint(execution_id)
                except CheckpointReadError:
                    if attempt == 0:
                        # Nothing written yet: the caller's outcome is as
                        # unknown as before, with zero residue.
                        raise
                    # An earlier attempt of THIS call may have committed.
                    outcome = UserMessageInjectionOutcome.OUTCOME_UNKNOWN
                    continue
                if checkpoint is None:
                    if attempt == 0:
                        return UserMessageInjectionResult(
                            context=None,
                            outcome=UserMessageInjectionOutcome.NOT_POSTED,
                        )
                    # Our own earlier attempt may have committed; a missing
                    # checkpoint now cannot rule that out.
                    outcome = UserMessageInjectionOutcome.OUTCOME_UNKNOWN
                    continue
                if not isinstance(checkpoint.get("context"), dict):
                    raise CheckpointCorruptError(
                        "Stored checkpoint carries no execution context to restore."
                    )
                detached = ExecutionContext.from_dict(checkpoint["context"])
                if self._find_replayed_turn(detached, tid, resolved_execution_message):
                    outcome = UserMessageInjectionOutcome.POSTED_REPLAY
                    break
                new_message = Message(
                    role="user",
                    content=resolved_execution_message,
                    metadata=metadata,
                    context_refs=build_image_context_references(files),
                )
                candidate = detached.to_dict()
                candidate["messages"].append(serialize_message(new_message))
                self._pending_marker_into(candidate["metadata"], new_message)
                try:
                    await self._persist_injected_context(
                        execution_id=execution_id,
                        context_payload=candidate,
                        label="user_message_injected",
                        baseline=checkpoint,
                    )
                except asyncio.CancelledError:
                    raise
                except Exception:
                    verdict = await self._confirm_injected_turn(
                        execution_id, tid, resolved_execution_message
                    )
                    if verdict == "absent":
                        raise
                    if verdict == "unknown":
                        outcome = UserMessageInjectionOutcome.OUTCOME_UNKNOWN
                        continue
                    # "found": committed, only the confirmation was lost.
                detached.append_message(new_message)
                self._pending_marker_into(detached.metadata, new_message)
                outcome = UserMessageInjectionOutcome.POSTED_FRESH
                break

        return UserMessageInjectionResult(context=detached, outcome=outcome)

    @staticmethod
    def _read_trace_watermark(context: ExecutionContext) -> str | None:
        """Read the user-message trace watermark off context metadata, if any.

        Kept in-runner so we don't import the tracing module (which would
        create a cycle) — the key is a stable contract spelled out in
        ``core.agent.tracing.TRACE_WATERMARK_KEY``.
        """
        metadata = getattr(context, "metadata", None)
        if not isinstance(metadata, dict):
            return None
        value = metadata.get("_user_message_trace_watermark")
        return value if isinstance(value, str) and value else None

    @staticmethod
    def _read_traced_turn_ids(context: ExecutionContext) -> tuple[str, ...]:
        metadata = getattr(context, "metadata", None)
        if not isinstance(metadata, dict):
            return ()
        value = metadata.get("_user_message_trace_turn_ids")
        if not isinstance(value, list):
            return ()
        return tuple(item for item in value if isinstance(item, str) and item)

    @staticmethod
    def _pending_marker_into(metadata: dict[str, Any], message: Any) -> None:
        """Stamp ``_pending_user_message_trace_timestamp`` into ``metadata``.

        Mirrored in ``core.agent.tracing.PENDING_MARKER_KEY``; kept in-runner
        to avoid an import cycle. The timestamp is the just-added message's
        normalized ISO-UTC timestamp so the catch-up loop can replay this
        specific turn rather than scanning history.

        Takes a plain ``dict`` rather than a context, so the candidate
        checkpoint snapshot and the live context can each get the marker
        stamped into their own (independent) metadata dict.
        """
        ts = AgentRunner._message_iso_timestamp(message)
        if ts is None:
            return
        metadata["_pending_user_message_trace_timestamp"] = ts
        turn_id = AgentRunner._message_turn_id(message)
        if turn_id:
            metadata["_pending_user_message_trace_turn_id"] = turn_id

    @staticmethod
    def _clear_pending_user_message_marker(context: ExecutionContext) -> None:
        metadata = getattr(context, "metadata", None)
        if isinstance(metadata, dict):
            metadata.pop("_pending_user_message_trace_timestamp", None)
            metadata.pop("_pending_user_message_trace_turn_id", None)

    @staticmethod
    def _message_turn_id(message: Any) -> str | None:
        metadata = getattr(message, "metadata", None)
        if not isinstance(metadata, dict):
            return None
        value = metadata.get("turn_id")
        return value if isinstance(value, str) and value else None

    @staticmethod
    def _message_iso_timestamp(message: Any) -> str | None:
        """ISO-UTC string of ``message.timestamp`` — same normalization the
        tracing module uses for its watermark, kept here to avoid a cycle.
        """
        from datetime import datetime, timezone

        ts = getattr(message, "timestamp", None)
        if isinstance(ts, datetime):
            aware = ts.replace(tzinfo=timezone.utc) if ts.tzinfo is None else ts
            return aware.astimezone(timezone.utc).isoformat()
        if isinstance(ts, str) and ts:
            return ts
        return None

    async def post_user_message(
        self,
        execution_id: str,
        message: str | None = None,
        *,
        execution_message: str | None = None,
        display_message: str | None = None,
        files: list[dict[str, Any]] | None = None,
        turn_id: str | None = None,
        request_interrupt: bool = True,
        reason: str | None = None,
    ) -> UserMessageInjectionResult:
        """Alias for external callers to inject a user message into an execution.

        `send_message` is an agent-side tool (`agent -> user`).
        `post_user_message` is a runner-side control API (`user/system -> execution`).
        """
        return await self.inject_user_message(
            execution_id,
            message,
            execution_message=execution_message,
            display_message=display_message,
            files=files,
            turn_id=turn_id,
            request_interrupt=request_interrupt,
            reason=reason,
        )

    async def _build_context(
        self,
        *,
        task: str | None,
        execution_id: str,
        user_id: str | None,
        session_id: str | None,
        workspace_id: str | None,
        allowed_external_dirs: list[str] | None,
        base_dir: str | None,
        metadata: dict[str, Any] | None,
    ) -> tuple[ExecutionContext, Any | None]:
        workspace = None
        context_kwargs: dict[str, Any] = {}
        if self.workspace_enabled:
            workspace = self.workspace_manager.get_or_create_workspace(
                base_dir=base_dir or self.workspace_base_dir,
                task_id=workspace_id or execution_id,
                allowed_external_dirs=allowed_external_dirs,
                scope_segments=self.scope_segments,
            )
            if inspect.isawaitable(workspace):
                workspace = await workspace
            context_kwargs = {
                "workspace_id": workspace.id,
                "workspace_path": str(workspace.workspace_dir),
                "cwd": str(workspace.workspace_dir),
                "workspace_state": self._workspace_state(workspace),
            }
        context = self.context_manager.create_context(
            execution_id=execution_id,
            user_id=user_id,
            session_id=session_id,
            system_prompt=getattr(self.agent, "system_prompt", None),
            **context_kwargs,
        )
        # Snapshotted at task start. On resume the context (and this threshold)
        # is restored verbatim from the checkpoint, so a context-window or ratio
        # change made after checkpointing only affects newly started tasks.
        (
            context.compact_config.threshold,
            context.compact_config.threshold_source,
        ) = self._resolve_compact_threshold()
        self._merge_context_metadata(context, metadata)
        if task:
            context.metadata.setdefault("task", task)

        memory_session = await self._resolve_memory_session(
            execution_id=execution_id,
            user_id=user_id,
            session_id=session_id,
        )
        if memory_session is not None:
            memory_id, snapshot = memory_session
            context.attach_memory_session(memory_id, snapshot)

        return context, workspace

    def _merge_context_metadata(
        self,
        context: ExecutionContext,
        metadata: dict[str, Any] | None,
        *,
        restored: bool = False,
    ) -> None:
        """Overlay current-run metadata on new or checkpoint-restored context."""

        if metadata is None:
            return
        if restored:
            # Symmetric with the fresh-context branch below: the current run's
            # metadata is authoritative for the modality preference, so an
            # absent key clears any checkpointed value rather than keeping it.
            preferred_modalities = normalize_input_modalities(
                metadata.get(PREFERRED_INPUT_MODALITIES_METADATA_KEY, ())
            )
            if preferred_modalities:
                context.metadata[PREFERRED_INPUT_MODALITIES_METADATA_KEY] = list(
                    preferred_modalities
                )
            else:
                context.metadata.pop(
                    PREFERRED_INPUT_MODALITIES_METADATA_KEY,
                    None,
                )
            # A resumed checkpoint may carry stale or legacy caller-influenced
            # values. The resume boundary supplies these two server-owned
            # identities from the current task row and exact execution lease.
            #
            # ``is not None`` rather than ``in metadata``: several resume
            # entry points pass the key unconditionally with a None value
            # (they have no trusted source of their own to supply). Treating
            # that as authoritative would erase the checkpointed real source
            # and permanently deny a pending approval that was gated under it.
            # An absent value means "I do not know", never "there is none".
            for key in ("task_source", "run_id"):
                if metadata.get(key) is not None:
                    context.metadata[key] = metadata[key]
            return

        current_metadata = dict(metadata)
        for reserved_key in RESERVED_ENGINE_METADATA_KEYS:
            current_metadata.pop(reserved_key, None)
        preferred_modalities = normalize_input_modalities(
            current_metadata.pop(PREFERRED_INPUT_MODALITIES_METADATA_KEY, ())
        )
        if preferred_modalities:
            context.metadata[PREFERRED_INPUT_MODALITIES_METADATA_KEY] = list(
                preferred_modalities
            )
        else:
            context.metadata.pop(PREFERRED_INPUT_MODALITIES_METADATA_KEY, None)
        context.metadata.update(current_metadata)
        request_context = metadata.get("request_context")
        if isinstance(request_context, dict):
            self._apply_request_context(context, request_context)

    def _resolve_compact_threshold(self) -> tuple[int, str]:
        """Derive the context-compaction threshold from the model's context window.

        When the model declares a context window, compact at
        ``context_window * ratio`` tokens; otherwise fall back to the configured
        default (preserving the historical 32000 behaviour) and warn once per
        model, because that default is far too low for long-context models and
        the resulting early compaction is otherwise invisible.

        Returns the threshold and its provenance (a ``COMPACT_THRESHOLD_SOURCE_*``
        value) for the compaction trace metadata.
        """
        llm = getattr(self.agent, "llm", None)
        context_window = getattr(llm, "context_window", None)
        # context_window is typed int | None end to end (DB Integer -> Pydantic
        # Optional[int]); bool is not a valid value, so a plain int check suffices.
        derived = derive_compact_threshold(context_window)
        if derived is not None:
            return derived
        threshold = get_compact_threshold_default()
        # A virtual model resolves its concrete window per call and
        # ``prepare_llm_for_context`` recomputes the threshold then, so its
        # missing window at task start is expected and not worth a warning.
        if llm is not None and not callable(getattr(llm, "prepare_for_call", None)):
            model_key = compact_model_key(llm)
            warn_once_per_model(
                THRESHOLD_WARNING_KEY_PREFIX + model_key,
                "Model %s has no context_window; the context compaction "
                "threshold falls back to %d tokens (%s). Set context_window "
                "on the model so compaction triggers at a fraction of its "
                "real window instead.",
                model_key,
                threshold,
                COMPACT_THRESHOLD_DEFAULT,
            )
        return threshold, COMPACT_THRESHOLD_SOURCE_DEFAULT

    def _initial_user_message_metadata(
        self, context: ExecutionContext
    ) -> dict[str, Any]:
        metadata: dict[str, Any] = {}
        context_metadata = (
            context.metadata if isinstance(context.metadata, dict) else {}
        )
        request_context = context_metadata.get("request_context")

        candidates = []
        if isinstance(request_context, dict):
            candidates.append(request_context)
        candidates.append(context_metadata)

        for candidate in candidates:
            turn_id = candidate.get("turn_id")
            if isinstance(turn_id, str) and turn_id:
                metadata["turn_id"] = turn_id
                break
        self._ensure_user_message_turn_id(metadata)

        for candidate in candidates:
            if "display_message" in candidate:
                display_message = candidate.get("display_message")
                metadata["display_message"] = (
                    display_message if isinstance(display_message, str) else ""
                )
                break
            if "display_user_message" in candidate:
                display_message = candidate.get("display_user_message")
                metadata["display_message"] = (
                    display_message if isinstance(display_message, str) else ""
                )
                break

        for candidate in candidates:
            files = candidate.get("files")
            if isinstance(files, list):
                metadata["files"] = files
                break
            attachments = candidate.get("attachments")
            if isinstance(attachments, list):
                metadata["files"] = attachments
                break

        return metadata

    @staticmethod
    def _ensure_user_message_turn_id(metadata: dict[str, Any]) -> str:
        value = metadata.get("turn_id")
        if isinstance(value, str) and value:
            return value
        turn_id = str(uuid4())
        metadata["turn_id"] = turn_id
        return turn_id

    def _apply_request_context(
        self,
        context: ExecutionContext,
        request_context: dict[str, Any],
    ) -> None:
        system_prompt = request_context.get("system_prompt")
        if isinstance(system_prompt, str) and system_prompt.strip():
            prompt = system_prompt.strip()
            if context.system_prompt and context.system_prompt.strip():
                existing = context.system_prompt.strip()
                if prompt not in existing:
                    context.system_prompt = f"{existing}\n\n{prompt}"
            else:
                context.system_prompt = prompt

        for key, value in request_context.items():
            if key == "system_prompt" or key in RESERVED_ENGINE_METADATA_KEYS:
                continue
            context.metadata[key] = value

    def _resolve_task(
        self,
        *,
        task: str | None,
        checkpoint: dict[str, Any] | None,
        execution_id: str,
    ) -> str | None:
        if task:
            return task

        if isinstance(checkpoint, dict):
            context_payload = checkpoint.get("context")
            if isinstance(context_payload, dict):
                metadata = context_payload.get("metadata")
                if isinstance(metadata, dict):
                    saved_task = metadata.get("task")
                    if isinstance(saved_task, str) and saved_task:
                        return saved_task
                messages = context_payload.get("messages")
                if isinstance(messages, list):
                    for message in reversed(messages):
                        if (
                            isinstance(message, dict)
                            and message.get("role") == "user"
                            and isinstance(message.get("content"), str)
                            and message["content"]
                        ):
                            content = cast(str, message["content"])
                            return content

        control = self._active_controls.get(execution_id)
        if control is not None:
            return control.task

        return None

    def _workspace_state(self, workspace: Any) -> dict[str, Any]:
        state: dict[str, Any] = {
            "input_dir": str(getattr(workspace, "input_dir", "")),
            "output_dir": str(getattr(workspace, "output_dir", "")),
            "temp_dir": str(getattr(workspace, "temp_dir", "")),
        }
        allowed_dirs = getattr(workspace, "allowed_external_dirs", None)
        if allowed_dirs is not None:
            state["allowed_external_dirs"] = [str(path) for path in allowed_dirs]
        return state

    async def _resolve_memory_session(
        self,
        *,
        execution_id: str,
        user_id: str | None,
        session_id: str | None,
    ) -> tuple[str | None, dict[str, Any] | None] | None:
        if self.memory_manager is None:
            return None

        for method_name in (
            "get_or_create_session",
            "create_session",
            "get_session",
            "load_session",
        ):
            method = getattr(self.memory_manager, method_name, None)
            if method is None:
                continue
            payload = self._call_with_supported_kwargs(
                method,
                execution_id=execution_id,
                user_id=user_id,
                session_id=session_id,
            )
            if inspect.isawaitable(payload):
                payload = await payload
            return self._normalize_memory_session(payload, session_id=session_id)

        return None

    def _normalize_memory_session(
        self,
        payload: Any,
        *,
        session_id: str | None,
    ) -> tuple[str | None, dict[str, Any] | None]:
        if payload is None:
            return session_id, None
        if isinstance(payload, tuple) and len(payload) == 2:
            return payload[0], payload[1]
        if isinstance(payload, str):
            return payload, None
        if isinstance(payload, dict):
            resolved_id = payload.get("session_id") or payload.get("id") or session_id
            snapshot = payload.get("snapshot")
            if snapshot is None:
                snapshot = {
                    key: value
                    for key, value in payload.items()
                    if key not in {"session_id", "id"}
                }
            return resolved_id, snapshot

        resolved_id = (
            getattr(payload, "session_id", None)
            or getattr(payload, "id", None)
            or session_id
        )
        snapshot = getattr(payload, "snapshot", None)
        if snapshot is None and hasattr(payload, "to_dict"):
            snapshot = payload.to_dict()
        return resolved_id, snapshot

    def _build_pattern_kwargs(
        self,
        *,
        pattern: Any,
        task: str,
        context: ExecutionContext,
        tools: list[Any],
        runtime: PatternRuntime,
        streaming_handler: Any | None,
    ) -> dict[str, Any]:
        return self._call_signature_kwargs(
            pattern.run,
            agent=self.agent,
            task=task,
            context=context,
            llm=getattr(self.agent, "llm", None),
            compact_llm=getattr(self.agent, "compact_llm", None),
            tools=tools,
            tracer=self.tracer,
            runtime=runtime,
            callbacks=self.callbacks,
            streaming_handler=streaming_handler,
            memory_store=getattr(self.agent, "memory_store", None),
            memory_similarity_threshold=getattr(
                self.agent, "memory_similarity_threshold", None
            ),
            skill_manager=getattr(self.agent, "skill_manager", None),
            allowed_skills=getattr(self.agent, "allowed_skills", None),
        )

    async def _setup_tools(self, tools: list[Any], *, task_id: str) -> None:
        for tool in tools:
            setup = getattr(tool, "setup", None)
            if not callable(setup):
                continue
            result = setup(task_id=task_id)
            if inspect.isawaitable(result):
                await result

    async def _teardown_tools(
        self,
        tools: list[Any],
        *,
        task_id: str,
        execution_status: str | None = None,
    ) -> None:
        for tool in reversed(tools):
            teardown = getattr(tool, "teardown", None)
            if not callable(teardown):
                continue
            try:
                result = self._call_with_supported_kwargs(
                    teardown,
                    task_id=task_id,
                    execution_status=execution_status,
                )
                if inspect.isawaitable(result):
                    await result
            except Exception:
                logger.exception(
                    "Tool teardown failed for %s", getattr(tool, "name", tool)
                )

    async def _load_latest_checkpoint(
        self,
        execution_id: str,
    ) -> dict[str, Any] | None:
        if self.tracer is None:
            return None

        payload = await read_latest_checkpoint_payload(self.tracer, execution_id)
        return payload if isinstance(payload, dict) else None

    async def _persist_injected_context(
        self,
        *,
        execution_id: str,
        context_payload: dict[str, Any],
        label: str,
        baseline: dict[str, Any] | None,
    ) -> None:
        """Persist ``context_payload`` (already-serialized ``to_dict()``
        output, plus one candidate message for the injection path) merged
        onto ``baseline``.

        Takes a plain dict rather than an ``ExecutionContext`` so a
        candidate snapshot -- built before the live context is touched --
        can be persisted without ever calling ``to_dict()`` on the live
        object.
        """
        if self.tracer is None:
            return

        payload = dict(baseline or {})
        payload.update(
            {
                "type": "checkpoint",
                "label": label,
                "execution_id": execution_id,
                "context": context_payload,
            }
        )

        checkpoint = getattr(self.tracer, "checkpoint", None)
        if callable(checkpoint):
            result = checkpoint(**payload)
            if inspect.isawaitable(result):
                await result
            return

        write_checkpoint = getattr(self.tracer, "write_checkpoint", None)
        if callable(write_checkpoint):
            result = write_checkpoint(payload)
            if inspect.isawaitable(result):
                await result
            return

    def _normalize_result(
        self,
        *,
        result: Any,
        pattern: Any,
        context: ExecutionContext,
        execution_id: str,
    ) -> dict[str, Any]:
        if isinstance(result, dict):
            normalized = dict(result)
        else:
            normalized = {"success": True, "output": result}

        normalized.setdefault("success", True)
        normalized.setdefault("execution_id", execution_id)
        normalized.setdefault("context", context)
        normalized.setdefault("pattern", pattern.__class__.__name__)

        assistant_message = extract_assistant_message(normalized)
        if assistant_message:
            set_assistant_message(normalized, assistant_message)
        if assistant_message and not self._has_assistant_message(
            context, assistant_message
        ):
            context.add_assistant_message(assistant_message)

        return normalized

    def _has_assistant_message(self, context: ExecutionContext, content: str) -> bool:
        return any(
            message.role == "assistant" and message.content == content
            for message in context.messages
        )

    async def _dispatch_callback(self, event: str, **payload: Any) -> None:
        for callback in self.callbacks:
            handler = getattr(callback, event, None)
            if handler is None:
                continue
            maybe_coroutine = handler(**payload)
            if inspect.isawaitable(maybe_coroutine):
                await maybe_coroutine

    def _call_with_supported_kwargs(self, fn: Any, **kwargs: Any) -> Any:
        return fn(**self._call_signature_kwargs(fn, **kwargs))

    def _call_signature_kwargs(self, fn: Any, **kwargs: Any) -> dict[str, Any]:
        try:
            signature = inspect.signature(fn)
        except (TypeError, ValueError):
            return kwargs
        parameters = signature.parameters.values()
        if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in parameters):
            return kwargs
        return {
            name: value
            for name, value in kwargs.items()
            if name in signature.parameters
        }
