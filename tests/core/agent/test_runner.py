from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest

from xagent.core.agent import (
    Agent,
    ContextManager,
    ExecutionContext,
    PatternRuntime,
    TraceEventCallback,
)
from xagent.core.agent import runtime as runtime_module
from xagent.core.agent.attachments import build_image_context_references
from xagent.core.agent.checkpoint import (
    CheckpointCorruptError,
    CheckpointPersistenceError,
    CheckpointUnavailableError,
)
from xagent.core.agent.context.execution import (
    COMPACT_THRESHOLD_SOURCE_CONTEXT_WINDOW,
    COMPACT_THRESHOLD_SOURCE_DEFAULT,
    TOOL_EVIDENCE_REMOVED_METADATA_KEY,
    context_write_lock,
    tool_evidence_state,
)
from xagent.core.agent.language import (
    OUTPUT_LANGUAGE_METADATA_KEY,
    OUTPUT_LANGUAGE_SOURCE_METADATA_KEY,
    reset_output_language_to_request_context,
)
from xagent.core.agent.runner import (
    AgentRunner,
    ExecutionControl,
    InjectionSettleRefusedError,
    UserMessageInjectionOutcome,
    UserMessageInjectionRejectedError,
)
from xagent.core.agent.runtime import LLMCallInterrupted
from xagent.core.task_runtime import PREFERRED_INPUT_MODALITIES_METADATA_KEY


@pytest.fixture(autouse=True)
def reset_context_manager() -> None:
    manager = ContextManager()
    manager._contexts.clear()  # type: ignore[attr-defined]
    yield
    manager._contexts.clear()  # type: ignore[attr-defined]


@pytest.fixture(autouse=True)
def reset_compact_warnings(monkeypatch: pytest.MonkeyPatch) -> None:
    # The once-per-model warning set is process-global; isolate it per test.
    monkeypatch.setattr(runtime_module, "_COMPACT_WINDOW_WARNED_MODELS", set())


@dataclass
class FakeWorkspace:
    id: str
    workspace_dir: Path
    input_dir: Path
    output_dir: Path
    temp_dir: Path
    allowed_external_dirs: list[Path]


class FakeWorkspaceManager:
    def __init__(self, tmp_path: Path) -> None:
        self.tmp_path = tmp_path
        self.calls: list[dict[str, Any]] = []

    def get_or_create_workspace(
        self,
        base_dir: str,
        task_id: str,
        allowed_external_dirs: list[str] | None = None,
        scope_segments: tuple[str, ...] = (),
    ) -> FakeWorkspace:
        self.calls.append(
            {
                "base_dir": base_dir,
                "task_id": task_id,
                "allowed_external_dirs": allowed_external_dirs,
            }
        )
        workspace_dir = self.tmp_path / task_id
        return FakeWorkspace(
            id=task_id,
            workspace_dir=workspace_dir,
            input_dir=workspace_dir / "input",
            output_dir=workspace_dir / "output",
            temp_dir=workspace_dir / "temp",
            allowed_external_dirs=[Path(path) for path in allowed_external_dirs or []],
        )


class FakeMemoryManager:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def get_or_create_session(
        self,
        *,
        execution_id: str,
        user_id: str | None = None,
        session_id: str | None = None,
    ) -> dict[str, Any]:
        self.calls.append(
            {
                "execution_id": execution_id,
                "user_id": user_id,
                "session_id": session_id,
            }
        )
        return {
            "session_id": session_id or f"memory-{execution_id}",
            "snapshot": {"summary": f"resume {execution_id}"},
        }


class AsyncMemoryManager(FakeMemoryManager):
    async def get_or_create_session(
        self,
        *,
        execution_id: str,
        user_id: str | None = None,
        session_id: str | None = None,
    ) -> dict[str, Any]:
        return super().get_or_create_session(
            execution_id=execution_id,
            user_id=user_id,
            session_id=session_id,
        )


class FakePattern:
    def __init__(self, result: dict[str, Any]) -> None:
        self.result = result
        self.calls: list[dict[str, Any]] = []

    async def run(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(kwargs)
        return dict(self.result)


class FailingPattern:
    def __init__(self, error: str) -> None:
        self.error = error

    async def run(self, **_: Any) -> dict[str, Any]:
        return {"success": False, "error": self.error}


class LLMInterruptedPattern:
    async def run(self, **_: Any) -> dict[str, Any]:
        raise LLMCallInterrupted("paused during LLM call")


class StatefulPattern:
    def __init__(self) -> None:
        self.state: dict[str, Any] = {}
        self.calls: list[dict[str, Any]] = []

    def load_state(self, state: dict[str, Any]) -> None:
        self.state = state

    async def run(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(kwargs)
        return {
            "success": True,
            "output": self.state["output"],
            "message_count": len(kwargs["context"].messages),
        }


class InjectingPattern:
    def __init__(self, runner: AgentRunner, execution_id: str) -> None:
        self.runner = runner
        self.execution_id = execution_id

    async def run(self, *, context: ExecutionContext, **_: Any) -> dict[str, Any]:
        injected = await self.runner.inject_user_message(
            self.execution_id,
            "Injected while resumed.",
            request_interrupt=False,
        )
        return {
            "success": True,
            "same_context": injected.context is context,
            "messages": [message.content for message in context.messages],
        }


class TrackingCallback:
    def __init__(self) -> None:
        self.events: list[tuple[str, str]] = []

    async def on_run_start(self, **payload: Any) -> None:
        context = payload["context"]
        self.events.append(("start", context.execution_id))

    async def on_run_end(self, **payload: Any) -> None:
        context = payload["context"]
        self.events.append(("end", context.execution_id))


class FailingUserMessageCallback:
    async def on_user_message_posted(self, **_: Any) -> None:
        raise RuntimeError("trace callback failed")


class StatusAwareTeardownTool:
    def __init__(self) -> None:
        self.teardown_calls: list[tuple[str | None, str | None]] = []

    async def setup(self, task_id: str | None = None) -> None:
        return None

    async def teardown(
        self,
        task_id: str | None = None,
        execution_status: str | None = None,
    ) -> None:
        self.teardown_calls.append((task_id, execution_status))


class RecordingTraceEventTracer:
    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []

    async def trace_event(
        self,
        event_type: Any,
        *,
        task_id: str | None = None,
        step_id: str | None = None,
        data: dict[str, Any] | None = None,
    ) -> str:
        self.events.append(
            {
                "event_type": getattr(event_type, "value", str(event_type)),
                "task_id": task_id,
                "step_id": step_id,
                "data": data or {},
            }
        )
        return str(len(self.events))


class InterruptingPattern:
    def __init__(
        self,
        runner: AgentRunner,
        execution_id: str,
        *,
        before_interrupt_check: Any | None = None,
    ) -> None:
        self.runner = runner
        self.execution_id = execution_id
        self.before_interrupt_check = before_interrupt_check

    async def run(
        self,
        *,
        context: ExecutionContext,
        runtime: PatternRuntime,
        **_: Any,
    ) -> dict[str, Any]:
        if callable(self.before_interrupt_check):
            maybe_result = self.before_interrupt_check()
            if maybe_result is not None:
                await maybe_result
        else:
            self.runner.pause(self.execution_id, reason="pause before step")

        if await runtime.should_interrupt():
            await runtime.checkpoint(
                "interrupted",
                context=context,
                pattern=self,
                status="interrupted",
                metadata={"safe_point": "during_pattern"},
            )
            return {
                "success": False,
                "status": "interrupted",
                "error": runtime.interrupt_reason or "interrupted",
            }

        return {"success": True, "output": "continued"}


class TracerCheckpointStore:
    def __init__(self) -> None:
        self.by_execution_id: dict[str, dict[str, Any]] = {}
        self.write_calls = 0

    async def checkpoint(self, **payload: Any) -> None:
        self.write_calls += 1
        self.by_execution_id[str(payload["execution_id"])] = dict(payload)

    async def load_latest_checkpoint(self, execution_id: str) -> dict[str, Any] | None:
        payload = self.by_execution_id.get(execution_id)
        return dict(payload) if payload is not None else None


class EmptyCanonicalCheckpointStore:
    def __init__(self) -> None:
        self.legacy_reads = 0

    async def load_latest_checkpoint(self, execution_id: str) -> dict[str, Any] | None:
        del execution_id
        return None

    def get_latest_checkpoint(self, execution_id: str) -> dict[str, Any] | None:
        del execution_id
        self.legacy_reads += 1
        raise AssertionError("canonical empty result must end checkpoint lookup")


def test_user_message_injection_outcome_truthiness_contract() -> None:
    """``NOT_POSTED`` is the empty string, and the other two members are
    not. That is the whole reason an unmodified ``if not posted`` /
    ``bool(posted)`` caller keeps asking exactly the question it always
    asked -- "did this hand back a usable context at all" -- across the
    fresh/replay split. Roughly a dozen call sites in ``websocket.py``,
    ``a2a.py`` and ``task_reply.py`` rest on it, and none of them names
    the enum, so an edit to these values would break them all silently.
    Assert the contract here instead, where the values live.
    """
    assert UserMessageInjectionOutcome.NOT_POSTED == ""
    assert not UserMessageInjectionOutcome.NOT_POSTED
    assert UserMessageInjectionOutcome.POSTED_FRESH
    assert UserMessageInjectionOutcome.POSTED_REPLAY
    # OUTCOME_UNKNOWN is deliberately truthy too (R1): an unmodified
    # ``bool(posted)`` caller must treat "write outcome undetermined" the
    # same as a real post -- the safe direction when the write might
    # already be committed. See the enum's docstring.
    assert UserMessageInjectionOutcome.OUTCOME_UNKNOWN


def test_user_message_injection_outcome_member_set_has_not_drifted() -> None:
    """A fifth member added here falls through the ``is
    UserMessageInjectionOutcome.POSTED_FRESH`` guards in ``a2a.py`` and
    ``websocket.py`` silently -- see ``task_interaction_close.py`` for what
    that means for an interaction row left open. The three guard sites are
    not equally exposed to it, though: the deferred WebSocket guard is
    documented defense-in-depth there, since it can only ever re-name a row
    an earlier attempt already retired.

    Relative to ``test_user_message_injection_outcome_truthiness_contract``
    above, this test's only unique catch is a member being added -- a
    rename or removal already raises ``AttributeError`` there. A member's
    value changing is the reverse case: caught there, not here.
    """
    assert {member.name for member in UserMessageInjectionOutcome} == {
        "NOT_POSTED",
        "POSTED_FRESH",
        "POSTED_REPLAY",
        "OUTCOME_UNKNOWN",
    }


@pytest.mark.asyncio
async def test_runner_treats_canonical_empty_checkpoint_as_authoritative() -> None:
    checkpoint_store = EmptyCanonicalCheckpointStore()
    runner = AgentRunner(
        agent=Agent(name="checkpoint-reader", patterns=[], llm=None),
        tracer=checkpoint_store,
    )

    result = await runner.inject_user_message(
        "missing-execution",
        "Continue",
        request_interrupt=False,
    )

    assert result.context is None
    assert result.outcome is UserMessageInjectionOutcome.NOT_POSTED
    assert checkpoint_store.legacy_reads == 0


class ContextlessCheckpointStore:
    """Returns a recognized checkpoint payload that carries no context.

    Every production checkpoint writer persists a ``context`` dict; a
    stored payload without one is malformed. The runner must classify it
    as corrupt rather than silently building fresh state on resume."""

    async def load_latest_checkpoint(self, execution_id: str) -> dict[str, Any]:
        return {"type": "checkpoint", "execution_id": execution_id}


class UnavailableCheckpointStore:
    """Every read fails -- distinct from ``EmptyCanonicalCheckpointStore``,
    whose ``None`` is an authoritative "no checkpoint" the runner may act
    on by building fresh state."""

    async def load_latest_checkpoint(self, execution_id: str) -> dict[str, Any]:
        del execution_id
        raise CheckpointUnavailableError("checkpoint store unavailable")


class FlakyOnceCheckpointStore:
    """Fails the first read, then behaves like a normal checkpoint store.

    Models a transient failure during ``inject_user_message``'s baseline
    read: the retry after the failure must actually persist the message,
    not find a "ghost" confirmation left behind by the rejected attempt.
    """

    def __init__(self) -> None:
        self.by_execution_id: dict[str, dict[str, Any]] = {}
        self.load_calls = 0

    async def checkpoint(self, **payload: Any) -> None:
        self.by_execution_id[str(payload["execution_id"])] = dict(payload)

    async def load_latest_checkpoint(self, execution_id: str) -> dict[str, Any] | None:
        self.load_calls += 1
        if self.load_calls == 1:
            raise CheckpointUnavailableError("transient")
        payload = self.by_execution_id.get(execution_id)
        return dict(payload) if payload is not None else None


class WriteFailsOnceStore:
    """The first checkpoint write raises WITHOUT storing anything, so a
    read-back correctly reports the turn as absent; every write after that
    succeeds normally. Models a transient write failure distinct from
    ``FlakyOnceCheckpointStore`` above, which fails the baseline *read*
    instead -- this one fails the write itself, exercising the R1
    read-back disambiguation path rather than the pre-write baseline path.
    """

    def __init__(self) -> None:
        self.by_execution_id: dict[str, dict[str, Any]] = {}
        self.write_calls = 0

    async def checkpoint(self, **payload: Any) -> None:
        self.write_calls += 1
        if self.write_calls == 1:
            raise CheckpointPersistenceError("transient write failure")
        self.by_execution_id[str(payload["execution_id"])] = dict(payload)

    async def load_latest_checkpoint(self, execution_id: str) -> dict[str, Any] | None:
        payload = self.by_execution_id.get(execution_id)
        return dict(payload) if payload is not None else None


class CommitAckLostStore:
    """Durably commits the payload, then raises -- models "the write
    succeeded but the confirmation was lost" (e.g. a downstream trace
    handler failing after the database commit already landed). A
    read-back must find the just-committed turn.
    """

    def __init__(self) -> None:
        self.by_execution_id: dict[str, dict[str, Any]] = {}
        self.write_calls = 0

    async def checkpoint(self, **payload: Any) -> None:
        self.write_calls += 1
        self.by_execution_id[str(payload["execution_id"])] = dict(payload)
        raise CheckpointPersistenceError("commit acknowledgement lost")

    async def load_latest_checkpoint(self, execution_id: str) -> dict[str, Any] | None:
        payload = self.by_execution_id.get(execution_id)
        return dict(payload) if payload is not None else None


class ReadBackFailingStore:
    """While ``fail`` is True, both the checkpoint write and the
    ``load_latest_checkpoint`` read (used for the injection's read-back,
    and for any baseline read that is not shortcut by a cached checkpoint)
    fail -- an infrastructure outage that prevents both. Flipping ``fail``
    off models the store recovering, e.g. for a retry.
    """

    def __init__(self) -> None:
        self.by_execution_id: dict[str, dict[str, Any]] = {}
        self.fail = True
        self.write_calls = 0
        self.read_calls = 0

    async def checkpoint(self, **payload: Any) -> None:
        self.write_calls += 1
        if self.fail:
            raise CheckpointPersistenceError("write refused")
        self.by_execution_id[str(payload["execution_id"])] = dict(payload)

    async def load_latest_checkpoint(self, execution_id: str) -> dict[str, Any] | None:
        self.read_calls += 1
        if self.fail:
            raise CheckpointUnavailableError("read-back unavailable")
        payload = self.by_execution_id.get(execution_id)
        return dict(payload) if payload is not None else None


class MalformedReadBackStore:
    """``load_latest_checkpoint`` returns a non-``None``, non-``dict``
    payload -- the malformed-read case ``_confirm_injected_turn`` must
    report as ``"unknown"`` (cannot tell whether the write committed),
    never ``"absent"`` (which would incorrectly claim the write is
    confirmed to not have happened)."""

    async def load_latest_checkpoint(self, execution_id: str) -> Any:
        del execution_id
        return "not-a-checkpoint-dict"


@pytest.mark.asyncio
async def test_confirm_injected_turn_treats_non_dict_payload_as_unknown(
    tmp_path: Path,
) -> None:
    runner = AgentRunner(
        agent=Agent(name="writer", patterns=[]),
        tracer=MalformedReadBackStore(),
        workspace_manager=FakeWorkspaceManager(tmp_path),
    )

    verdict = await runner._confirm_injected_turn(
        "exec-malformed-readback", "turn-1", "Hello"
    )

    assert verdict == "unknown"


@pytest.mark.asyncio
async def test_confirm_injected_turn_treats_none_payload_as_absent(
    tmp_path: Path,
) -> None:
    runner = AgentRunner(
        agent=Agent(name="writer", patterns=[]),
        tracer=TracerCheckpointStore(),
        workspace_manager=FakeWorkspaceManager(tmp_path),
    )

    verdict = await runner._confirm_injected_turn(
        "exec-no-checkpoint", "turn-1", "Hello"
    )

    assert verdict == "absent"


class BlockingCheckpointStore:
    """Checkpoint writer that blocks until released, for tests that need
    to observe state while a write is in flight (and holding the context's
    write lock, since the write happens inside it).

    ``entered`` is set as soon as a write call starts (before it blocks),
    so a test can wait for "the write has begun" without a race. ``release``
    must be set for the blocked write to proceed. ``fail_mode``:

    - ``"none"``: commits normally once released.
    - ``"raise_before_commit"``: raises without storing -- the write never
      happened, so a read-back must report "absent".
    - ``"commit_then_raise"``: stores the payload (so a read-back sees it)
      and then raises -- "committed, confirmation lost".
    """

    def __init__(
        self,
        entered: asyncio.Event | None = None,
        release: asyncio.Event | None = None,
        fail_mode: str = "none",
    ) -> None:
        self.entered = entered if entered is not None else asyncio.Event()
        self.release = release if release is not None else asyncio.Event()
        self.fail_mode = fail_mode
        self.by_execution_id: dict[str, dict[str, Any]] = {}
        self.write_calls = 0

    async def checkpoint(self, **payload: Any) -> None:
        self.write_calls += 1
        self.entered.set()
        await self.release.wait()
        if self.fail_mode == "raise_before_commit":
            raise CheckpointPersistenceError("blocked write refused")
        self.by_execution_id[str(payload["execution_id"])] = dict(payload)
        if self.fail_mode == "commit_then_raise":
            raise CheckpointPersistenceError("commit acknowledgement lost")

    async def load_latest_checkpoint(self, execution_id: str) -> dict[str, Any] | None:
        payload = self.by_execution_id.get(execution_id)
        return dict(payload) if payload is not None else None


class BlockingReadCheckpointStore:
    """Checkpoint store whose ``load_latest_checkpoint`` blocks until
    released, so two concurrent cold-start injections can be forced to
    both reach the point where they have read the same seed checkpoint
    and are about to race on installing a context for it -- deterministic
    interleaving instead of hoping ``asyncio.gather`` happens to schedule
    it that way. ``entered`` counts how many callers are currently
    blocked in the read so a test can wait for "both are racing" without
    a sleep-based guess. Writes are instant and recorded normally.
    """

    def __init__(self, seed_payload: dict[str, Any]) -> None:
        self.seed_payload = seed_payload
        self.entered = 0
        self.both_entered = asyncio.Event()
        self.release = asyncio.Event()
        self.by_execution_id: dict[str, dict[str, Any]] = {}
        self.write_calls = 0

    async def load_latest_checkpoint(self, execution_id: str) -> dict[str, Any] | None:
        del execution_id
        self.entered += 1
        if self.entered >= 2:
            self.both_entered.set()
        await self.release.wait()
        return dict(self.seed_payload)

    async def checkpoint(self, **payload: Any) -> None:
        self.write_calls += 1
        self.by_execution_id[str(payload["execution_id"])] = dict(payload)


@pytest.mark.asyncio
async def test_run_resume_does_not_build_fresh_context_on_unavailable() -> None:
    runner = AgentRunner(
        agent=Agent(name="checkpoint-reader", patterns=[], llm=None),
        tracer=UnavailableCheckpointStore(),
    )
    build_context_calls: list[Any] = []
    original_build_context = runner._build_context

    async def spy_build_context(*args: Any, **kwargs: Any) -> Any:
        build_context_calls.append((args, kwargs))
        return await original_build_context(*args, **kwargs)

    runner._build_context = spy_build_context  # type: ignore[method-assign]

    with pytest.raises(CheckpointUnavailableError):
        await runner.run(
            task=None,
            execution_id="exec-resume-unavailable",
            resume=True,
        )

    assert build_context_calls == []
    assert runner.context_manager.get_context("exec-resume-unavailable") is None


@pytest.mark.asyncio
async def test_inject_user_message_propagates_unavailable() -> None:
    """Distinct from the canonical-empty-checkpoint case above: a read
    failure must not be swallowed into the same ``None`` "no checkpoint"
    result -- the caller cannot tell a real failure from genuine absence."""
    runner = AgentRunner(
        agent=Agent(name="checkpoint-reader", patterns=[], llm=None),
        tracer=UnavailableCheckpointStore(),
    )

    with pytest.raises(CheckpointUnavailableError):
        await runner.inject_user_message(
            "missing-execution-unavailable",
            "Continue",
            request_interrupt=False,
        )


@pytest.mark.asyncio
async def test_inject_rejection_leaves_no_dedupe_residue() -> None:
    tracer = FlakyOnceCheckpointStore()
    runner = AgentRunner(
        agent=Agent(name="checkpoint-reader", patterns=[], llm=None),
        tracer=tracer,
    )
    context = ExecutionContext(execution_id="exec-residue")
    runner.context_manager.set_context(context)

    with pytest.raises(CheckpointUnavailableError):
        await runner.inject_user_message(
            "exec-residue",
            "Continue",
            turn_id="turn-1",
            request_interrupt=False,
        )

    # The rejected attempt must not have mutated the in-memory context.
    assert context.messages == []

    # A retry must actually persist the message -- it must not find a
    # ghost confirmation left behind by the failed attempt above.
    result = await runner.inject_user_message(
        "exec-residue",
        "Continue",
        turn_id="turn-1",
        request_interrupt=False,
    )

    assert result.context is context
    assert result.outcome is UserMessageInjectionOutcome.POSTED_FRESH
    assert len(context.messages) == 1
    assert tracer.by_execution_id["exec-residue"]["context"]["messages"]


# --- R1: candidate snapshot + write lock + read-back on failure ----------


@pytest.mark.asyncio
async def test_inject_persist_failure_leaves_no_residue_and_retry_is_fresh(
    tmp_path: Path,
) -> None:
    """The reverse of the probe this design fixes: a write failure whose
    read-back finds the turn genuinely absent must raise with zero
    residue, and a retry of the same turn must actually write (not find a
    ghost confirmation) -- ending with exactly one copy of the turn."""
    tracer = WriteFailsOnceStore()
    runner = AgentRunner(
        agent=Agent(name="writer", patterns=[]),
        tracer=tracer,
        workspace_manager=FakeWorkspaceManager(tmp_path),
    )
    context = ExecutionContext(execution_id="exec-write-fail-once")
    runner.context_manager.set_context(context)

    with pytest.raises(UserMessageInjectionRejectedError):
        await runner.inject_user_message(
            "exec-write-fail-once",
            "Hello",
            turn_id="turn-1",
            request_interrupt=False,
        )
    assert context.messages == []
    assert tracer.write_calls == 1

    result = await runner.inject_user_message(
        "exec-write-fail-once",
        "Hello",
        turn_id="turn-1",
        request_interrupt=False,
    )

    assert result.outcome is UserMessageInjectionOutcome.POSTED_FRESH
    assert tracer.write_calls == 2
    live_matches = [
        message
        for message in result.context.messages
        if message.role == "user" and message.metadata.get("turn_id") == "turn-1"
    ]
    assert len(live_matches) == 1
    stored_matches = [
        message
        for message in tracer.by_execution_id["exec-write-fail-once"]["context"][
            "messages"
        ]
        if message.get("metadata", {}).get("turn_id") == "turn-1"
    ]
    assert len(stored_matches) == 1


@pytest.mark.asyncio
async def test_live_pattern_cannot_observe_injection_before_persist_confirms(
    tmp_path: Path,
) -> None:
    tracer = BlockingCheckpointStore()
    runner = AgentRunner(
        agent=Agent(name="writer", patterns=[]),
        tracer=tracer,
        workspace_manager=FakeWorkspaceManager(tmp_path),
    )
    context = ExecutionContext(execution_id="exec-visibility")
    runner.context_manager.set_context(context)

    inject_task = asyncio.create_task(
        runner.inject_user_message("exec-visibility", "Hello", request_interrupt=False)
    )
    try:
        await asyncio.wait_for(tracer.entered.wait(), timeout=5)
        assert context.messages == []

        tracer.release.set()
        result = await asyncio.wait_for(inject_task, timeout=5)
    finally:
        if not inject_task.done():
            inject_task.cancel()

    assert result.outcome is UserMessageInjectionOutcome.POSTED_FRESH
    assert [message.content for message in context.messages] == ["Hello"]


@pytest.mark.asyncio
async def test_pattern_checkpoint_waits_for_inflight_injection(
    tmp_path: Path,
) -> None:
    tracer = BlockingCheckpointStore()
    runner = AgentRunner(
        agent=Agent(name="writer", patterns=[]),
        tracer=tracer,
        workspace_manager=FakeWorkspaceManager(tmp_path),
    )
    context = ExecutionContext(execution_id="exec-checkpoint-waits")
    runner.context_manager.set_context(context)

    inject_task = asyncio.create_task(
        runner.inject_user_message(
            "exec-checkpoint-waits", "Hello", request_interrupt=False
        )
    )
    try:
        await asyncio.wait_for(tracer.entered.wait(), timeout=5)
        assert tracer.write_calls == 1

        pattern_runtime = PatternRuntime(
            execution_id="exec-checkpoint-waits", tracer=tracer
        )
        checkpoint_task = asyncio.create_task(
            pattern_runtime.checkpoint(
                "step",
                context=context,
                pattern=FakePattern({"success": True}),
            )
        )
        try:
            await asyncio.sleep(0.05)
            # Still waiting on the context's write lock -- must not have
            # reached the store at all yet.
            assert tracer.write_calls == 1

            tracer.release.set()
            inject_result = await asyncio.wait_for(inject_task, timeout=5)
            payload = await asyncio.wait_for(checkpoint_task, timeout=5)
        finally:
            if not checkpoint_task.done():
                checkpoint_task.cancel()
    finally:
        if not inject_task.done():
            inject_task.cancel()

    assert inject_result.outcome is UserMessageInjectionOutcome.POSTED_FRESH
    payload_messages = payload["context"]["messages"]
    assert any(message["content"] == "Hello" for message in payload_messages)


@pytest.mark.asyncio
async def test_injection_waits_for_inflight_pattern_checkpoint(
    tmp_path: Path,
) -> None:
    """The reverse ordering of the previous test: a pattern checkpoint
    blocks first, an injection arrives while it's in flight, and once both
    complete the latest checkpoint must contain both the pattern's own
    message and the injected one -- an injection built from a stale
    pre-checkpoint snapshot must never overwrite the pattern's write."""
    tracer = BlockingCheckpointStore()
    runner = AgentRunner(
        agent=Agent(name="writer", patterns=[]),
        tracer=tracer,
        workspace_manager=FakeWorkspaceManager(tmp_path),
    )
    context = ExecutionContext(execution_id="exec-reverse-order")
    context.add_assistant_message("Pattern output")
    runner.context_manager.set_context(context)

    pattern_runtime = PatternRuntime(execution_id="exec-reverse-order", tracer=tracer)
    checkpoint_task = asyncio.create_task(
        pattern_runtime.checkpoint(
            "step", context=context, pattern=FakePattern({"success": True})
        )
    )
    try:
        await asyncio.wait_for(tracer.entered.wait(), timeout=5)
        assert tracer.write_calls == 1

        inject_task = asyncio.create_task(
            runner.inject_user_message(
                "exec-reverse-order", "Hello", request_interrupt=False
            )
        )
        try:
            await asyncio.sleep(0.05)
            # Injection is waiting on the lock the pattern checkpoint
            # holds -- must not have reached the store yet.
            assert tracer.write_calls == 1

            tracer.release.set()
            await asyncio.wait_for(checkpoint_task, timeout=5)
            inject_result = await asyncio.wait_for(inject_task, timeout=5)
        finally:
            if not inject_task.done():
                inject_task.cancel()
    finally:
        if not checkpoint_task.done():
            checkpoint_task.cancel()

    assert inject_result.outcome is UserMessageInjectionOutcome.POSTED_FRESH
    final_messages = tracer.by_execution_id["exec-reverse-order"]["context"]["messages"]
    assert any(
        message["role"] == "assistant" and message["content"] == "Pattern output"
        for message in final_messages
    )
    assert any(
        message["role"] == "user" and message["content"] == "Hello"
        for message in final_messages
    )


class _OverlapCountingBlockingStore:
    """Like ``BlockingCheckpointStore`` but tracks how many ``checkpoint``
    calls are in flight at once, so a test can assert two SHARED holders
    were genuinely concurrent (both entered the store before either
    returned) rather than merely both eventually completing."""

    def __init__(self) -> None:
        self.in_flight = 0
        self.max_in_flight = 0
        self.both_entered = asyncio.Event()
        self.release = asyncio.Event()

    async def checkpoint(self, **payload: Any) -> None:
        del payload
        self.in_flight += 1
        self.max_in_flight = max(self.max_in_flight, self.in_flight)
        if self.in_flight >= 2:
            self.both_entered.set()
        await self.release.wait()
        self.in_flight -= 1

    async def load_latest_checkpoint(self, execution_id: str) -> dict[str, Any] | None:
        del execution_id
        return None


@pytest.mark.asyncio
async def test_two_concurrent_pattern_checkpoints_on_one_context_overlap() -> None:
    """SHARED holders must run in parallel with each other: two pattern
    checkpoints for the same context both enter the store before either
    returns, exactly as on base (pre-gate) behavior."""
    tracer = _OverlapCountingBlockingStore()
    context = ExecutionContext(execution_id="exec-shared-overlap")
    runtime_a = PatternRuntime(execution_id="exec-shared-overlap", tracer=tracer)
    runtime_b = PatternRuntime(execution_id="exec-shared-overlap", tracer=tracer)

    task_a = asyncio.create_task(
        runtime_a.checkpoint("step-a", context=context, pattern=FakePattern({}))
    )
    task_b = asyncio.create_task(
        runtime_b.checkpoint("step-b", context=context, pattern=FakePattern({}))
    )
    try:
        await asyncio.wait_for(tracer.both_entered.wait(), timeout=5)
        assert tracer.max_in_flight == 2
        tracer.release.set()
        await asyncio.wait_for(asyncio.gather(task_a, task_b), timeout=5)
    finally:
        if not task_a.done():
            task_a.cancel()
        if not task_b.done():
            task_b.cancel()


@pytest.mark.asyncio
async def test_concurrent_same_turn_injections_write_once(tmp_path: Path) -> None:
    """Uses a blocking store to force true interleaving rather than
    hoping ``asyncio.gather`` happens to schedule it that way: the first
    injection is held INSIDE the store, still holding the exclusive
    write-lock, while the second is started. The second must not reach
    the store at all (its dedupe check, run only after it acquires the
    now-held lock, must find the first's write once that commits) --
    exactly the race the lock and the in-lock dedupe check exist to
    prevent."""
    tracer = BlockingCheckpointStore()
    runner = AgentRunner(
        agent=Agent(name="writer", patterns=[]),
        tracer=tracer,
        workspace_manager=FakeWorkspaceManager(tmp_path),
    )
    context = ExecutionContext(execution_id="exec-concurrent-same-turn")
    runner.context_manager.set_context(context)

    first_task = asyncio.create_task(
        runner.inject_user_message(
            "exec-concurrent-same-turn",
            "Hello",
            turn_id="turn-x",
            request_interrupt=False,
        )
    )
    try:
        await asyncio.wait_for(tracer.entered.wait(), timeout=5)
        assert tracer.write_calls == 1

        second_task = asyncio.create_task(
            runner.inject_user_message(
                "exec-concurrent-same-turn",
                "Hello",
                turn_id="turn-x",
                request_interrupt=False,
            )
        )
        try:
            # The first call is still blocked inside the store, holding
            # the exclusive lock. The second must be stuck waiting for
            # that lock -- it must not have reached the store (its dedupe
            # check runs only once it holds the lock).
            await asyncio.sleep(0.05)
            assert not second_task.done()
            assert tracer.write_calls == 1

            tracer.release.set()
            results = await asyncio.wait_for(
                asyncio.gather(first_task, second_task), timeout=5
            )
        finally:
            if not second_task.done():
                second_task.cancel()
    finally:
        if not first_task.done():
            first_task.cancel()

    outcomes = {result.outcome for result in results}
    assert outcomes == {
        UserMessageInjectionOutcome.POSTED_FRESH,
        UserMessageInjectionOutcome.POSTED_REPLAY,
    }
    assert tracer.write_calls == 1
    matches = [
        message
        for message in context.messages
        if message.role == "user" and message.metadata.get("turn_id") == "turn-x"
    ]
    assert len(matches) == 1


@pytest.mark.asyncio
async def test_concurrent_cold_start_injections_share_one_context(
    tmp_path: Path,
) -> None:
    """Uses a blocking read store to force both cold starts to genuinely
    race at the ``set_context_if_absent`` seam: both are held inside the
    checkpoint read, each having independently rebuilt an
    ``ExecutionContext`` from the same seed, until both have entered --
    only then are they released to race on installing one. Without
    ``set_context_if_absent`` (i.e. with a bare ``set_context``), each
    would instead install its OWN object, and the two injections would
    end up on two different, mutually oblivious contexts (and therefore
    two different write locks) instead of converging on one."""
    execution_id = "exec-concurrent-cold-start"
    seed_context = ExecutionContext(execution_id=execution_id)
    seed_context.add_user_message("Original task")
    seed_payload = {
        "type": "checkpoint",
        "execution_id": execution_id,
        "pattern": "FakePattern",
        "label": "waiting_for_user",
        "status": "waiting_for_user",
        "context": seed_context.to_dict(),
        "pattern_state": {},
        "metadata": {},
    }
    tracer = BlockingReadCheckpointStore(seed_payload)
    runner = AgentRunner(
        agent=Agent(name="writer", patterns=[]),
        tracer=tracer,
        workspace_manager=FakeWorkspaceManager(tmp_path),
    )

    first_task = asyncio.create_task(
        runner.inject_user_message(
            execution_id, "First follow-up", turn_id="turn-a", request_interrupt=False
        )
    )
    second_task = asyncio.create_task(
        runner.inject_user_message(
            execution_id, "Second follow-up", turn_id="turn-b", request_interrupt=False
        )
    )
    try:
        await asyncio.wait_for(tracer.both_entered.wait(), timeout=5)
        tracer.release.set()
        results = await asyncio.wait_for(
            asyncio.gather(first_task, second_task), timeout=5
        )
    finally:
        if not first_task.done():
            first_task.cancel()
        if not second_task.done():
            second_task.cancel()

    assert results[0].context is results[1].context
    contents = [
        message.content
        for message in results[0].context.messages
        if message.role == "user"
    ]
    assert "First follow-up" in contents
    assert "Second follow-up" in contents
    # Both writes landed serialized through the one shared context's lock,
    # not raced onto two independent (and therefore corrupting) objects.
    assert tracer.write_calls == 2


@pytest.mark.asyncio
async def test_inject_cancelled_mid_persist_does_not_apply(tmp_path: Path) -> None:
    tracer = BlockingCheckpointStore()
    runner = AgentRunner(
        agent=Agent(name="writer", patterns=[]),
        tracer=tracer,
        workspace_manager=FakeWorkspaceManager(tmp_path),
    )
    context = ExecutionContext(execution_id="exec-cancel-mid-persist")
    runner.context_manager.set_context(context)

    task = asyncio.create_task(
        runner.inject_user_message(
            "exec-cancel-mid-persist",
            "Hello",
            turn_id="turn-cancel",
            request_interrupt=False,
        )
    )
    await asyncio.wait_for(tracer.entered.wait(), timeout=5)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    # Cancelled before the blocked store call ever committed: zero residue.
    assert context.messages == []
    assert "exec-cancel-mid-persist" not in tracer.by_execution_id

    # Let the store accept writes again and retry the same turn.
    tracer.release.set()
    result = await runner.inject_user_message(
        "exec-cancel-mid-persist",
        "Hello",
        turn_id="turn-cancel",
        request_interrupt=False,
    )

    assert result.outcome is UserMessageInjectionOutcome.POSTED_FRESH
    matches = [
        message
        for message in result.context.messages
        if message.role == "user" and message.metadata.get("turn_id") == "turn-cancel"
    ]
    assert len(matches) == 1


class CommitThenHangOnceStore:
    """The first ``checkpoint`` call stores the payload (so it genuinely
    committed) and then hangs indefinitely -- distinct from
    ``BlockingCheckpointStore``, whose ``entered`` fires BEFORE the store
    commits. Used to model "cancelled strictly after the write landed":
    the caller cancels while the coroutine is stuck past the commit, so
    ``inject_user_message`` never gets a chance to run its own confirm
    logic at all (cancellation is re-raised immediately, per its
    docstring) even though the store already durably has the turn. Every
    call after the first commits and returns immediately, so a retry
    proceeds normally.
    """

    def __init__(self) -> None:
        self.by_execution_id: dict[str, dict[str, Any]] = {}
        self.write_calls = 0
        self.committed = asyncio.Event()
        self.hang_forever = asyncio.Event()

    async def checkpoint(self, **payload: Any) -> None:
        self.write_calls += 1
        self.by_execution_id[str(payload["execution_id"])] = dict(payload)
        if self.write_calls == 1:
            self.committed.set()
            await self.hang_forever.wait()

    async def load_latest_checkpoint(self, execution_id: str) -> dict[str, Any] | None:
        payload = self.by_execution_id.get(execution_id)
        return dict(payload) if payload is not None else None


@pytest.mark.asyncio
async def test_inject_cancelled_after_commit_then_retry_yields_fresh_once(
    tmp_path: Path,
) -> None:
    """Finding P: cancellation strictly AFTER the write already committed
    to the store (not before, like the sibling test above). The cancelled
    call's own outcome is undefined (it never returns -- cancellation
    propagates), and nothing was applied to the live context either way.
    A retry with the SAME turn_id must still converge on exactly one
    committed message -- POSTED_FRESH exactly once, no duplicate -- since
    the live context (what the retry's dedupe check reads) never saw the
    cancelled attempt's write."""
    tracer = CommitThenHangOnceStore()
    runner = AgentRunner(
        agent=Agent(name="writer", patterns=[]),
        tracer=tracer,
        workspace_manager=FakeWorkspaceManager(tmp_path),
    )
    context = ExecutionContext(execution_id="exec-cancel-after-commit")
    runner.context_manager.set_context(context)

    task = asyncio.create_task(
        runner.inject_user_message(
            "exec-cancel-after-commit",
            "Hello",
            turn_id="turn-cancel-after-commit",
            request_interrupt=False,
        )
    )
    await asyncio.wait_for(tracer.committed.wait(), timeout=5)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    # The store DID commit the cancelled attempt's candidate, but it was
    # never applied to the live context -- the linearization point (only
    # reached after a successful, uncancelled persist) never ran.
    assert context.messages == []
    stored_payload = tracer.by_execution_id["exec-cancel-after-commit"]
    stored_user_messages = [
        message
        for message in stored_payload["context"]["messages"]
        if message["role"] == "user"
    ]
    assert len(stored_user_messages) == 1

    result = await runner.inject_user_message(
        "exec-cancel-after-commit",
        "Hello",
        turn_id="turn-cancel-after-commit",
        request_interrupt=False,
    )

    assert result.outcome is UserMessageInjectionOutcome.POSTED_FRESH
    assert tracer.write_calls == 2
    matches = [
        message
        for message in result.context.messages
        if message.role == "user"
        and message.metadata.get("turn_id") == "turn-cancel-after-commit"
    ]
    assert len(matches) == 1
    final_stored_user_messages = [
        message
        for message in tracer.by_execution_id["exec-cancel-after-commit"]["context"][
            "messages"
        ]
        if message["role"] == "user"
    ]
    assert len(final_stored_user_messages) == 1


@pytest.mark.asyncio
async def test_inject_commit_ack_lost_is_confirmed_by_read_back(
    tmp_path: Path,
) -> None:
    tracer = CommitAckLostStore()
    dispatched: list[dict[str, Any]] = []

    class _RecordingCallback:
        async def on_user_message_posted(self, **kwargs: Any) -> None:
            dispatched.append(kwargs)

    runner = AgentRunner(
        agent=Agent(name="writer", patterns=[]),
        tracer=tracer,
        callbacks=[_RecordingCallback()],
        workspace_manager=FakeWorkspaceManager(tmp_path),
    )
    context = ExecutionContext(execution_id="exec-ack-lost")
    runner.context_manager.set_context(context)
    runner.pause = MagicMock(return_value=True)

    result = await runner.inject_user_message(
        "exec-ack-lost", "Hello", request_interrupt=True
    )

    assert result.outcome is UserMessageInjectionOutcome.POSTED_FRESH
    assert [
        message.content for message in result.context.messages if message.role == "user"
    ] == ["Hello"]
    assert len(dispatched) == 1
    runner.pause.assert_called_once()


class _FakePatternWithState:
    def __init__(self, state: dict[str, Any]) -> None:
        self._state = state

    def get_state(self) -> dict[str, Any]:
        return self._state


@pytest.mark.asyncio
async def test_inject_watermark_repersist_does_not_regress_concurrent_checkpoint(
    tmp_path: Path,
) -> None:
    """A pattern checkpoint that lands during ``on_user_message_posted``
    (the callback runs OUTSIDE the injection's exclusive lock, so a
    SHARED-mode pattern checkpoint can genuinely interleave with it) must
    not be clobbered by the watermark re-persist that follows it: the
    watermark write's baseline must be re-resolved AFTER the callback
    runs, not reused from before it, or it would silently overwrite the
    newer pattern_state with the stale pre-callback one."""
    from xagent.core.agent.tracing import TRACE_WATERMARK_KEY

    tracer = TracerCheckpointStore()
    execution_id = "exec-watermark-baseline"
    pattern_runtime = PatternRuntime(execution_id=execution_id, tracer=tracer)

    class _WatermarkAdvancingCallback:
        async def on_user_message_posted(self, **kwargs: Any) -> None:
            context = kwargs["context"]
            # A pattern checkpoint lands concurrently with this callback,
            # advancing the runtime's cached baseline with fresh
            # pattern_state.
            await pattern_runtime.checkpoint(
                "concurrent-step",
                context=context,
                pattern=_FakePatternWithState({"progress": "fresh-from-callback"}),
            )
            context.metadata[TRACE_WATERMARK_KEY] = "watermark-after-callback"

    runner = AgentRunner(
        agent=Agent(name="writer", patterns=[]),
        tracer=tracer,
        callbacks=[_WatermarkAdvancingCallback()],
        workspace_manager=FakeWorkspaceManager(tmp_path),
    )
    context = ExecutionContext(execution_id=execution_id)
    runner.context_manager.set_context(context)
    runner._active_controls[execution_id] = ExecutionControl(
        runtime=pattern_runtime, task=None
    )
    runner.pause = MagicMock(return_value=True)

    result = await runner.inject_user_message(
        execution_id, "Hello", request_interrupt=True
    )

    assert result.outcome is UserMessageInjectionOutcome.POSTED_FRESH
    final_payload = tracer.by_execution_id[execution_id]
    assert final_payload["label"] == "user_message_trace_watermark"
    assert final_payload["pattern_state"] == {"progress": "fresh-from-callback"}


@pytest.mark.asyncio
async def test_inject_unconfirmable_write_returns_outcome_unknown(
    tmp_path: Path,
) -> None:
    tracer = ReadBackFailingStore()
    runner = AgentRunner(
        agent=Agent(name="writer", patterns=[]),
        tracer=tracer,
        workspace_manager=FakeWorkspaceManager(tmp_path),
    )
    context = ExecutionContext(execution_id="exec-unknown-outcome")
    runner.context_manager.set_context(context)
    # Cache a baseline on an active control so the pre-write baseline read
    # (which would itself fail while ``tracer.fail`` is True) is skipped --
    # isolating the assertion to the write-then-read-back failure this test
    # actually targets.
    baseline_checkpoint = {
        "type": "checkpoint",
        "execution_id": "exec-unknown-outcome",
        "context": context.to_dict(),
    }
    runner._active_controls["exec-unknown-outcome"] = ExecutionControl(
        runtime=SimpleNamespace(last_checkpoint=baseline_checkpoint),
        task=None,
    )
    runner.pause = MagicMock(return_value=True)

    result = await runner.inject_user_message(
        "exec-unknown-outcome",
        "Hello",
        turn_id="turn-unknown",
        request_interrupt=True,
    )

    assert result.outcome is UserMessageInjectionOutcome.OUTCOME_UNKNOWN
    assert result.context is context
    assert context.messages == []
    runner.pause.assert_called_once()

    tracer.fail = False
    retry = await runner.inject_user_message(
        "exec-unknown-outcome",
        "Hello",
        turn_id="turn-unknown",
        request_interrupt=False,
    )

    assert retry.outcome is UserMessageInjectionOutcome.POSTED_FRESH
    matches = [
        message
        for message in retry.context.messages
        if message.role == "user" and message.metadata.get("turn_id") == "turn-unknown"
    ]
    assert len(matches) == 1


class IntermittentUnknownStore:
    """Checkpoint writes and reads are unavailable for the first fail_count writes."""

    def __init__(self, fail_count: int) -> None:
        self.fail_count = fail_count
        self.write_calls = 0
        self.read_calls = 0
        self.by_execution_id: dict[str, dict[str, Any]] = {}

    async def checkpoint(self, **payload: Any) -> None:
        self.write_calls += 1
        if self.write_calls <= self.fail_count:
            raise CheckpointPersistenceError("transient store outage")
        self.by_execution_id[str(payload["execution_id"])] = dict(payload)

    async def load_latest_checkpoint(self, execution_id: str) -> dict[str, Any] | None:
        self.read_calls += 1
        if self.write_calls <= self.fail_count:
            raise CheckpointUnavailableError("transient store outage")
        payload = self.by_execution_id.get(execution_id)
        return dict(payload) if payload is not None else None


def _cache_injection_baseline(
    runner: AgentRunner, execution_id: str, context: Any
) -> None:
    """Seed ``_active_controls`` with a cached baseline so
    ``_resolve_injection_baseline`` short-circuits to it instead of
    calling into a deliberately-failing store for the pre-write baseline
    read -- isolating these tests to the write/read-back behavior
    they actually target, same trick as
    ``test_inject_unconfirmable_write_returns_outcome_unknown`` above."""
    baseline_checkpoint = {
        "type": "checkpoint",
        "execution_id": execution_id,
        "context": context.to_dict(),
    }
    runner._active_controls[execution_id] = ExecutionControl(
        runtime=SimpleNamespace(last_checkpoint=baseline_checkpoint),
        task=None,
    )


@pytest.mark.asyncio
async def test_inject_unknown_returns_without_rewriting_live_snapshot(
    tmp_path: Path,
) -> None:
    tracer = IntermittentUnknownStore(fail_count=1)
    runner = AgentRunner(
        agent=Agent(name="writer", patterns=[]),
        tracer=tracer,
        workspace_manager=FakeWorkspaceManager(tmp_path),
    )
    context = ExecutionContext(execution_id="exec-no-retry")
    runner.context_manager.set_context(context)
    _cache_injection_baseline(runner, context.execution_id, context)
    runner.pause = MagicMock(return_value=True)

    result = await runner.inject_user_message(
        context.execution_id, "Hello", turn_id="turn-no-retry", request_interrupt=True
    )

    assert result.outcome is UserMessageInjectionOutcome.OUTCOME_UNKNOWN
    assert tracer.write_calls == 1
    assert context.messages == []
    assert not context_write_lock(context).locked()
    runner.pause.assert_called_once()


class FirstReadOnlyStore:
    """Only the very first read (the pre-write baseline) succeeds; every
    write commits and then raises, and every later read fails."""

    def __init__(self) -> None:
        self.read_calls = 0
        self.write_calls = 0
        self.by_execution_id: dict[str, dict[str, Any]] = {}

    async def checkpoint(self, **payload: Any) -> None:
        self.write_calls += 1
        self.by_execution_id[str(payload["execution_id"])] = dict(payload)
        raise CheckpointPersistenceError("ack lost")

    async def load_latest_checkpoint(self, execution_id: str) -> dict[str, Any] | None:
        self.read_calls += 1
        if self.read_calls > 1:
            raise CheckpointUnavailableError("read unavailable")
        payload = self.by_execution_id.get(execution_id)
        return dict(payload) if payload is not None else None


@pytest.mark.asyncio
async def test_inject_readback_failure_after_possible_commit_is_unknown(
    tmp_path: Path,
) -> None:
    """A failed readback after a possible commit must not imply rejection."""
    tracer = FirstReadOnlyStore()
    runner = AgentRunner(
        agent=Agent(name="writer", patterns=[]),
        tracer=tracer,
        workspace_manager=FakeWorkspaceManager(tmp_path),
    )
    context = ExecutionContext(execution_id="exec-retry-baseline-read")
    runner.context_manager.set_context(context)
    runner.pause = MagicMock(return_value=True)

    result = await runner.inject_user_message(
        "exec-retry-baseline-read",
        "Hello",
        turn_id="turn-retry-baseline-read",
        request_interrupt=False,
    )

    assert result.outcome is UserMessageInjectionOutcome.OUTCOME_UNKNOWN
    assert tracer.write_calls == 1
    assert context.messages == []


class CommitThenUnreadableStore:
    """While ``fail`` is True every write durably commits and THEN raises,
    and every read fails -- so the injection's read-back cannot tell that
    the write landed and reports ``OUTCOME_UNKNOWN`` although the turn IS
    durable. With ``commit_while_failing=False`` the failing writes store
    nothing instead (the turn is genuinely absent)."""

    def __init__(self, *, commit_while_failing: bool) -> None:
        self.commit_while_failing = commit_while_failing
        self.fail = True
        self.write_calls = 0
        self.by_execution_id: dict[str, dict[str, Any]] = {}

    async def checkpoint(self, **payload: Any) -> None:
        self.write_calls += 1
        if self.fail:
            if self.commit_while_failing:
                self.by_execution_id[str(payload["execution_id"])] = dict(payload)
            raise CheckpointPersistenceError("ack lost")
        self.by_execution_id[str(payload["execution_id"])] = dict(payload)

    async def load_latest_checkpoint(self, execution_id: str) -> dict[str, Any] | None:
        if self.fail:
            raise CheckpointUnavailableError("read unavailable")
        payload = self.by_execution_id.get(execution_id)
        return dict(payload) if payload is not None else None


async def _inject_until_unknown(
    runner: AgentRunner, execution_id: str, turn_id: str
) -> ExecutionContext:
    context = ExecutionContext(execution_id=execution_id)
    runner.context_manager.set_context(context)
    _cache_injection_baseline(runner, execution_id, context)
    runner.pause = MagicMock(return_value=True)
    result = await runner.inject_user_message(
        execution_id, "Hello", turn_id=turn_id, request_interrupt=False
    )
    assert result.outcome is UserMessageInjectionOutcome.OUTCOME_UNKNOWN
    assert context.messages == []
    # The run that owned this context is over by the time a resume owner
    # settles the unknown outcome.
    runner._active_controls.pop(execution_id)
    return context


def _settle_runner(tmp_path: Path, tracer: Any) -> AgentRunner:
    runner = AgentRunner(
        agent=Agent(name="writer", patterns=[]),
        tracer=tracer,
        workspace_manager=FakeWorkspaceManager(tmp_path),
    )
    runner.pause = MagicMock(return_value=True)
    return runner


def _durable_turn_ids(tracer: Any, execution_id: str) -> list[Any]:
    messages = tracer.by_execution_id[execution_id]["context"]["messages"]
    return [m["metadata"].get("turn_id") for m in messages if m["role"] == "user"]


@pytest.mark.asyncio
async def test_settle_replays_turn_committed_under_unknown(tmp_path: Path) -> None:
    tracer = CommitThenUnreadableStore(commit_while_failing=True)
    runner = _settle_runner(tmp_path, tracer)
    stale = await _inject_until_unknown(runner, "exec-settle-found", "turn-rb")
    tracer.fail = False
    writes_before = tracer.write_calls

    # A plain retry would dedupe against the stale registered context,
    # which never saw the turn; the settle dedupes against the durable
    # checkpoint, which did.
    result = await runner.settle_injection_against_checkpoint(
        "exec-settle-found", "Hello", turn_id="turn-rb"
    )

    assert result.outcome is UserMessageInjectionOutcome.POSTED_REPLAY
    assert tracer.write_calls == writes_before
    assert result.context is not None and result.context is not stale
    assert [m.metadata.get("turn_id") for m in result.context.messages] == ["turn-rb"]
    # Detached: the registry still holds the very same (stale) object.
    assert runner.context_manager.get_context("exec-settle-found") is stale
    assert stale.messages == []
    runner.pause.assert_not_called()


@pytest.mark.asyncio
async def test_settle_persists_turn_absent_under_unknown(tmp_path: Path) -> None:
    tracer = CommitThenUnreadableStore(commit_while_failing=False)
    runner = _settle_runner(tmp_path, tracer)
    # Seed a durable checkpoint the settle can read.
    tracer.fail = False
    seed = ExecutionContext(execution_id="exec-settle-absent")
    await tracer.checkpoint(
        type="checkpoint", execution_id="exec-settle-absent", context=seed.to_dict()
    )
    tracer.fail = True
    stale = await _inject_until_unknown(runner, "exec-settle-absent", "turn-ra")
    tracer.fail = False
    callback = MagicMock()
    runner.callbacks = [SimpleNamespace(on_user_message_posted=callback)]

    result = await runner.settle_injection_against_checkpoint(
        "exec-settle-absent",
        execution_message="Hello",
        display_message="Hi",
        turn_id="turn-ra",
    )

    assert result.outcome is UserMessageInjectionOutcome.POSTED_FRESH
    assert _durable_turn_ids(tracer, "exec-settle-absent") == ["turn-ra"]
    durable = tracer.by_execution_id["exec-settle-absent"]["context"]
    assert durable["messages"][-1]["metadata"]["display_message"] == "Hi"
    # The trace is handed to the resume's catch-up via the pending marker;
    # nothing live is notified.
    assert durable["metadata"]["_pending_user_message_trace_turn_id"] == "turn-ra"
    callback.assert_not_called()
    runner.pause.assert_not_called()
    assert runner.context_manager.get_context("exec-settle-absent") is stale
    assert stale.messages == []


class CommitThenLoseAckStore:
    """Reads always work; the first write durably commits and then raises
    (the ack is lost), later writes succeed."""

    def __init__(self) -> None:
        self.write_calls = 0
        self.read_calls = 0
        self.by_execution_id: dict[str, dict[str, Any]] = {}

    async def checkpoint(self, **payload: Any) -> None:
        self.write_calls += 1
        self.by_execution_id[str(payload["execution_id"])] = dict(payload)
        if self.write_calls == 1:
            raise CheckpointPersistenceError("ack lost")

    async def load_latest_checkpoint(self, execution_id: str) -> dict[str, Any] | None:
        self.read_calls += 1
        payload = self.by_execution_id.get(execution_id)
        return dict(payload) if payload is not None else None


@pytest.mark.asyncio
async def test_settle_write_error_confirmed_by_read_back_is_posted_fresh(
    tmp_path: Path,
) -> None:
    """The "found" branch: the settle's own write raises, the read-back
    sees the turn committed, so it is reported POSTED_FRESH with no second
    write."""
    tracer = CommitThenLoseAckStore()
    tracer.by_execution_id["exec-settle-readback"] = {
        "type": "checkpoint",
        "execution_id": "exec-settle-readback",
        "context": ExecutionContext(execution_id="exec-settle-readback").to_dict(),
    }
    runner = _settle_runner(tmp_path, tracer)

    result = await runner.settle_injection_against_checkpoint(
        "exec-settle-readback", "Hello", turn_id="turn-readback"
    )

    assert result.outcome is UserMessageInjectionOutcome.POSTED_FRESH
    assert tracer.write_calls == 1
    # Pre-write read + read-back.
    assert tracer.read_calls == 2
    assert _durable_turn_ids(tracer, "exec-settle-readback") == ["turn-readback"]
    assert runner.context_manager.get_context("exec-settle-readback") is None


@pytest.mark.asyncio
async def test_settle_without_checkpoint_ignores_stale_context(
    tmp_path: Path,
) -> None:
    tracer = TracerCheckpointStore()
    runner = _settle_runner(tmp_path, tracer)
    stale = ExecutionContext(execution_id="exec-settle-none")
    stale.add_user_message("older turn", metadata={"turn_id": "turn-old"})
    runner.context_manager.set_context(stale)
    try:
        result = await runner.settle_injection_against_checkpoint(
            "exec-settle-none", "Hello", turn_id="turn-new"
        )

        assert result.outcome is UserMessageInjectionOutcome.NOT_POSTED
        assert result.context is None
        assert tracer.write_calls == 0
        assert runner.context_manager.get_context("exec-settle-none") is stale
        assert [m.metadata.get("turn_id") for m in stale.messages] == ["turn-old"]
    finally:
        runner.context_manager.remove_context("exec-settle-none")


class PatternCheckpointRacingStore:
    """The settle's first write commits and loses its ack, and the
    read-back fails (``unknown``). Meanwhile a pattern checkpoint of the
    REGISTERED context -- as another runner's still-running pattern would
    take it -- is queued on that object's gate; it must not run inside the
    settle's attempt, and lands only after confirmation finishes."""

    def __init__(self, live: ExecutionContext) -> None:
        self.live = live
        self.labels: list[str] = []
        self.by_execution_id: dict[str, dict[str, Any]] = {}
        self.pattern_task: asyncio.Task[None] | None = None
        self.pattern_interleaved = False
        self.fail_next_read = False

    async def _pattern_checkpoint(self) -> None:
        async with context_write_lock(self.live).shared():
            self.labels.append("pattern_step")
            self.by_execution_id[self.live.execution_id] = {
                "type": "checkpoint",
                "label": "pattern_step",
                "execution_id": self.live.execution_id,
                "context": self.live.to_dict(),
            }

    async def checkpoint(self, **payload: Any) -> None:
        self.labels.append(str(payload.get("label")))
        self.by_execution_id[str(payload["execution_id"])] = dict(payload)
        if self.pattern_task is None:
            self.pattern_task = asyncio.create_task(self._pattern_checkpoint())
            for _ in range(3):
                await asyncio.sleep(0)
            self.pattern_interleaved = "pattern_step" in self.labels
            self.fail_next_read = True
            raise CheckpointPersistenceError("ack lost")

    async def load_latest_checkpoint(self, execution_id: str) -> dict[str, Any] | None:
        if self.fail_next_read:
            self.fail_next_read = False
            raise CheckpointUnavailableError("read-back unavailable")
        payload = self.by_execution_id.get(execution_id)
        return dict(payload) if payload is not None else None


@pytest.mark.asyncio
async def test_settle_gate_excludes_pattern_checkpoint_until_confirmation(
    tmp_path: Path,
) -> None:
    execution_id = "exec-settle-race"
    live = ExecutionContext(execution_id=execution_id)
    live.add_message("assistant", "step done")
    tracer = PatternCheckpointRacingStore(live)
    tracer.by_execution_id[execution_id] = {
        "type": "checkpoint",
        "execution_id": execution_id,
        "context": ExecutionContext(execution_id=execution_id).to_dict(),
    }
    runner = _settle_runner(tmp_path, tracer)
    runner.context_manager.set_context(live)
    try:
        result = await runner.settle_injection_against_checkpoint(
            execution_id, "Hello", turn_id="turn-race"
        )
        assert tracer.pattern_task is not None
        await tracer.pattern_task

        assert result.outcome is UserMessageInjectionOutcome.OUTCOME_UNKNOWN
        assert tracer.pattern_interleaved is False
        assert tracer.labels == ["user_message_injected", "pattern_step"]
        assert runner.context_manager.get_context(execution_id) is live
        assert [m.role for m in live.messages] == ["assistant"]
    finally:
        runner.context_manager.remove_context(execution_id)


@pytest.mark.asyncio
async def test_settle_refused_while_a_run_is_active(tmp_path: Path) -> None:
    tracer = CommitThenLoseAckStore()
    runner = _settle_runner(tmp_path, tracer)
    context = ExecutionContext(execution_id="exec-settle-active")
    _cache_injection_baseline(runner, "exec-settle-active", context)

    with pytest.raises(InjectionSettleRefusedError):
        await runner.settle_injection_against_checkpoint(
            "exec-settle-active", "Hello", turn_id="turn-active"
        )

    assert tracer.read_calls == 0
    assert tracer.write_calls == 0


@pytest.mark.asyncio
async def test_settle_requires_turn_id(tmp_path: Path) -> None:
    tracer = CommitThenLoseAckStore()
    runner = _settle_runner(tmp_path, tracer)

    with pytest.raises(ValueError, match="turn_id"):
        await runner.settle_injection_against_checkpoint(
            "exec-settle-no-turn", "Hello", turn_id="  "
        )

    assert tracer.read_calls == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("registered", [True, False], ids=["stale", "none"])
async def test_settle_never_changes_registered_context_identity(
    tmp_path: Path, registered: bool
) -> None:
    execution_id = f"exec-settle-identity-{registered}"
    tracer = TracerCheckpointStore()
    tracer.by_execution_id[execution_id] = {
        "type": "checkpoint",
        "execution_id": execution_id,
        "context": ExecutionContext(execution_id=execution_id).to_dict(),
    }
    runner = _settle_runner(tmp_path, tracer)
    stale = ExecutionContext(execution_id=execution_id) if registered else None
    if stale is not None:
        runner.context_manager.set_context(stale)
    try:
        result = await runner.settle_injection_against_checkpoint(
            execution_id, "Hello", turn_id="turn-identity"
        )

        assert result.outcome is UserMessageInjectionOutcome.POSTED_FRESH
        assert runner.context_manager.get_context(execution_id) is stale
        assert result.context is not stale
        assert _durable_turn_ids(tracer, execution_id) == ["turn-identity"]
    finally:
        runner.context_manager.remove_context(execution_id)


def test_context_write_lock_rebinds_across_event_loops() -> None:
    """Companion to the ``test_context.py`` coverage of the same function,
    kept here too per the R1 test plan: a lock idle when a different loop
    asks for it is rebound silently; a lock held across loops raises."""
    context = ExecutionContext(execution_id="exec-cross-loop-runner")

    async def acquire_once() -> asyncio.Lock:
        lock = context_write_lock(context)
        async with lock:
            pass
        return lock

    first_loop = asyncio.new_event_loop()
    try:
        first_lock = first_loop.run_until_complete(acquire_once())
    finally:
        first_loop.close()

    second_loop = asyncio.new_event_loop()
    try:
        second_lock = second_loop.run_until_complete(acquire_once())
    finally:
        second_loop.close()

    assert first_lock is not second_lock

    third_loop = asyncio.new_event_loop()
    fourth_loop = asyncio.new_event_loop()
    try:

        async def acquire_and_hold() -> asyncio.Lock:
            lock = context_write_lock(context)
            await lock.acquire()
            return lock

        held_lock = third_loop.run_until_complete(acquire_and_hold())

        async def try_rebind() -> None:
            context_write_lock(context)

        with pytest.raises(RuntimeError, match="two event loops"):
            fourth_loop.run_until_complete(try_rebind())

        async def release(lock: asyncio.Lock) -> None:
            lock.release()

        third_loop.run_until_complete(release(held_lock))
    finally:
        third_loop.close()
        fourth_loop.close()


@pytest.mark.asyncio
async def test_runner_builds_context_and_invokes_pattern(tmp_path: Path) -> None:
    workspace_manager = FakeWorkspaceManager(tmp_path)
    memory_manager = FakeMemoryManager()
    callback = TrackingCallback()
    pattern = FakePattern({"success": True, "output": "done"})
    agent = Agent(
        name="writer",
        patterns=[pattern],
        tools=["local-tool"],
        llm="fake-llm",
        system_prompt="System prompt",
    )
    runner = AgentRunner(
        agent=agent,
        workspace_manager=workspace_manager,
        memory_manager=memory_manager,
        callbacks=[callback],
        workspace_base_dir=str(tmp_path / "workspaces"),
    )

    result = await runner.run(
        task="Write a summary",
        execution_id="exec-1",
        user_id="user-1",
        session_id="session-1",
        allowed_external_dirs=[str(tmp_path / "kb")],
        extra_tools=["extra-tool"],
        metadata={"source": "test"},
    )

    assert result["success"] is True
    assert result["execution_id"] == "exec-1"
    context = result["context"]
    assert isinstance(context, ExecutionContext)
    assert context.system_prompt == "System prompt"
    assert context.user_id == "user-1"
    assert context.session_id == "session-1"
    assert context.workspace_id == "exec-1"
    assert context.memory_session_id == "session-1"
    assert context.memory_snapshot == {"summary": "resume exec-1"}
    assert context.metadata["task"] == "Write a summary"
    assert context.metadata["source"] == "test"
    assert [message.role for message in context.messages] == ["user", "assistant"]
    assert context.messages[0].content == "Write a summary"
    assert context.messages[1].content == "done"
    assert ContextManager().get_context("exec-1") is context

    pattern_call = pattern.calls[0]
    assert pattern_call["task"] == "Write a summary"
    assert pattern_call["context"] is context
    assert pattern_call["tools"] == ["local-tool", "extra-tool"]
    assert pattern_call["llm"] == "fake-llm"
    assert isinstance(pattern_call["runtime"], PatternRuntime)
    assert workspace_manager.calls[0]["task_id"] == "exec-1"
    assert callback.events == [("start", "exec-1"), ("end", "exec-1")]


@pytest.mark.asyncio
async def test_runner_inserts_synthetic_user_turn_before_a_leading_assistant_initial_message(
    tmp_path: Path,
) -> None:
    # The marketplace Hire flow seeds a persona greeting as a task's very
    # first persisted message (see seed_assistant_message in
    # src/xagent/web/api/chat.py) - initial_messages then starts with role
    # "assistant" and no prior user turn. Anthropic's Messages API (and
    # every claude_compatible provider routed through it) rejects a request
    # whose first message isn't role "user", so the runner must correct
    # this before it's ever replayed into context.
    workspace_manager = FakeWorkspaceManager(tmp_path)
    memory_manager = FakeMemoryManager()
    pattern = FakePattern({"success": True, "output": "done"})
    agent = Agent(name="writer", patterns=[pattern], tools=[], llm="fake-llm")
    runner = AgentRunner(
        agent=agent,
        workspace_manager=workspace_manager,
        memory_manager=memory_manager,
        workspace_base_dir=str(tmp_path / "workspaces"),
    )

    result = await runner.run(
        task="Let's get started",
        execution_id="exec-seed",
        user_id="user-1",
        initial_messages=[
            {
                "role": "assistant",
                "content": "Hi - I'm Maya, your Social Media Content Manager.",
            }
        ],
    )

    context = result["context"]
    assert [message.role for message in context.messages] == [
        "user",
        "assistant",
        "user",
        "assistant",
    ]
    assert context.messages[0].content == "(conversation start)"
    assert context.messages[0].metadata.get("_xagent_synthetic") == "leading_user_turn"
    assert (
        context.messages[1].content
        == "Hi - I'm Maya, your Social Media Content Manager."
    )
    assert context.messages[2].content == "Let's get started"


@pytest.mark.asyncio
async def test_runner_does_not_insert_synthetic_turn_for_user_first_initial_messages(
    tmp_path: Path,
) -> None:
    workspace_manager = FakeWorkspaceManager(tmp_path)
    memory_manager = FakeMemoryManager()
    pattern = FakePattern({"success": True, "output": "done"})
    agent = Agent(name="writer", patterns=[pattern], tools=[], llm="fake-llm")
    runner = AgentRunner(
        agent=agent,
        workspace_manager=workspace_manager,
        memory_manager=memory_manager,
        workspace_base_dir=str(tmp_path / "workspaces"),
    )

    result = await runner.run(
        task="Follow up",
        execution_id="exec-normal",
        user_id="user-1",
        initial_messages=[
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello, how can I help?"},
        ],
    )

    context = result["context"]
    assert [message.role for message in context.messages] == [
        "user",
        "assistant",
        "user",
        "assistant",
    ]
    assert context.messages[0].content == "Hi"


@pytest.mark.asyncio
async def test_runner_passes_waiting_status_to_tool_teardown(tmp_path: Path) -> None:
    tool = StatusAwareTeardownTool()
    agent = Agent(
        name="interactive",
        patterns=[
            FakePattern(
                {
                    "success": False,
                    "status": "waiting_for_user",
                    "message": "Provide the missing value.",
                }
            )
        ],
    )
    runner = AgentRunner(
        agent=agent,
        workspace_manager=FakeWorkspaceManager(tmp_path),
    )

    result = await runner.run(
        task="Run an interactive tool",
        execution_id="interaction-task",
        extra_tools=[tool],
    )

    assert result["status"] == "waiting_for_user"
    assert tool.teardown_calls == [("interaction-task", "waiting_for_user")]


class LiveStepTasksPattern:
    """A pattern that reports a waiting_for_user exit while its
    has_live_step_tasks() predicate is under test control, to pin the
    runner-side guard independently of any real pattern implementation."""

    def __init__(self, *, live_step_tasks: bool) -> None:
        self._live_step_tasks = live_step_tasks

    async def run(self, **kwargs: Any) -> dict[str, Any]:
        return {
            "success": False,
            "status": "waiting_for_user",
            "message": "Pick one.",
            "clarification_draft": {"source": "test"},
        }

    def has_live_step_tasks(self) -> bool:
        return self._live_step_tasks


@pytest.mark.asyncio
async def test_runner_raises_when_waiting_exit_still_has_live_step_tasks(
    tmp_path: Path,
) -> None:
    agent = Agent(
        name="interactive",
        patterns=[LiveStepTasksPattern(live_step_tasks=True)],
    )
    runner = AgentRunner(agent=agent, workspace_manager=FakeWorkspaceManager(tmp_path))

    with pytest.raises(AssertionError, match="live step tasks"):
        await runner.run(task="Run an interactive tool", execution_id="live-tasks-task")


@pytest.mark.asyncio
async def test_runner_allows_waiting_exit_once_step_tasks_are_clear(
    tmp_path: Path,
) -> None:
    agent = Agent(
        name="interactive",
        patterns=[LiveStepTasksPattern(live_step_tasks=False)],
    )
    runner = AgentRunner(agent=agent, workspace_manager=FakeWorkspaceManager(tmp_path))

    result = await runner.run(
        task="Run an interactive tool", execution_id="no-live-tasks-task"
    )

    assert result["status"] == "waiting_for_user"


@pytest.mark.asyncio
async def test_runner_passes_failed_status_when_tool_setup_raises(
    tmp_path: Path,
) -> None:
    class SetupFailingTool(StatusAwareTeardownTool):
        async def setup(self, task_id: str | None = None) -> None:
            raise RuntimeError("setup failed")

    tool = SetupFailingTool()
    runner = AgentRunner(
        agent=Agent(
            name="setup-failure",
            patterns=[FakePattern({"success": True, "output": "unused"})],
        ),
        workspace_manager=FakeWorkspaceManager(tmp_path),
    )

    with pytest.raises(RuntimeError, match="setup failed"):
        await runner.run(
            task="Initialize tools",
            execution_id="setup-failure-task",
            extra_tools=[tool],
        )

    assert tool.teardown_calls == [("setup-failure-task", "failed")]


@pytest.mark.asyncio
async def test_runner_awaits_async_memory_manager(tmp_path: Path) -> None:
    memory_manager = AsyncMemoryManager()
    pattern = FakePattern({"success": True, "output": "done"})
    agent = Agent(name="writer", patterns=[pattern])
    runner = AgentRunner(
        agent=agent,
        workspace_manager=FakeWorkspaceManager(tmp_path),
        memory_manager=memory_manager,
    )

    result = await runner.run(
        task="Write a summary",
        execution_id="exec-async-memory",
        session_id="session-async",
    )

    assert result["success"] is True
    assert result["context"].memory_session_id == "session-async"
    assert result["context"].memory_snapshot == {"summary": "resume exec-async-memory"}


@pytest.mark.asyncio
async def test_runner_tries_multiple_patterns_and_collects_failures(
    tmp_path: Path,
) -> None:
    first = FailingPattern("first failed")
    second = FakePattern({"success": True, "message": "second worked"})
    agent = Agent(name="writer", patterns=[first, second])
    runner = AgentRunner(
        agent=agent,
        workspace_manager=FakeWorkspaceManager(tmp_path),
    )

    result = await runner.run(task="Recover", execution_id="exec-2")

    assert result["success"] is True
    assert result["pattern"] == "FakePattern"
    context = result["context"]
    assert [message.content for message in context.messages] == [
        "Recover",
        "second worked",
    ]


@pytest.mark.asyncio
async def test_runner_returns_aggregate_error_when_all_patterns_fail(
    tmp_path: Path,
) -> None:
    agent = Agent(
        name="writer",
        patterns=[FailingPattern("first failed"), FailingPattern("second failed")],
    )
    runner = AgentRunner(
        agent=agent,
        workspace_manager=FakeWorkspaceManager(tmp_path),
    )

    result = await runner.run(task="Impossible", execution_id="exec-3")

    assert result["success"] is False
    assert result["patterns_attempted"] == 2
    assert len(result["pattern_errors"]) == 2
    assert result["context"].messages[0].content == "Impossible"


@pytest.mark.asyncio
async def test_runner_returns_single_pattern_failure_result(tmp_path: Path) -> None:
    agent = Agent(
        name="writer",
        patterns=[
            FakePattern(
                {
                    "success": False,
                    "status": "failed",
                    "failure_reason": "structured_failure",
                    "error": "failed with details",
                }
            )
        ],
    )
    runner = AgentRunner(
        agent=agent,
        workspace_manager=FakeWorkspaceManager(tmp_path),
    )

    result = await runner.run(task="Impossible", execution_id="exec-single-fail")

    assert result["success"] is False
    assert result["status"] == "failed"
    assert result["failure_reason"] == "structured_failure"
    assert result["error"] == "failed with details"
    assert "pattern_errors" not in result


@pytest.mark.asyncio
async def test_runner_does_not_add_empty_user_message_for_missing_task(
    tmp_path: Path,
) -> None:
    agent = Agent(name="writer", patterns=[FakePattern({"success": True})])
    runner = AgentRunner(
        agent=agent,
        workspace_manager=FakeWorkspaceManager(tmp_path),
    )

    result = await runner.run(task=None, execution_id="exec-empty-task")

    assert result["success"] is True
    assert result["context"].messages == []


@pytest.mark.asyncio
async def test_initial_messages_replay_tool_pairs(tmp_path: Path) -> None:
    agent = Agent(name="writer", patterns=[FakePattern({"success": True})])
    runner = AgentRunner(
        agent=agent,
        workspace_manager=FakeWorkspaceManager(tmp_path),
    )

    initial_messages = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {"name": "read_file", "arguments": "{}"},
                }
            ],
        },
        {
            "role": "tool",
            "tool_name": "read_file",
            "tool_call_id": "call-1",
            "raw_result": {"output": "file contents"},
        },
    ]

    result = await runner.run(
        task=None,
        execution_id="exec-replay",
        initial_messages=initial_messages,
    )

    assert result["success"] is True
    messages = result["context"].messages
    # initial_messages[0] is role "assistant", so AgentRunner.run prepends a
    # synthetic leading user turn (Anthropic's Messages API rejects a request
    # whose first message isn't role "user") before replaying the
    # assistant/tool pair.
    assert [message.role for message in messages] == ["user", "assistant", "tool"]

    synthetic_message = messages[0]
    assert synthetic_message.content == "(conversation start)"
    assert synthetic_message.metadata["_xagent_synthetic"] == "leading_user_turn"

    assistant_message = messages[1]
    assert assistant_message.content == ""
    assert assistant_message.tool_calls == initial_messages[0]["tool_calls"]

    tool_message = messages[2]
    assert tool_message.content == "Tool read_file returned: file contents"
    assert tool_message.tool_call_id == "call-1"
    assert tool_message.metadata["raw_result"] == {"output": "file contents"}
    assert tool_message.metadata["tool_name"] == "read_file"
    # The synthetic turn is prepended, not interleaved, so tool-call pairing
    # is undisturbed: every "tool" message is still immediately preceded by
    # the assistant message declaring its tool_call_id.
    for index, message in enumerate(messages):
        if message.role == "tool":
            assert messages[index - 1].role == "assistant"
            assert message.tool_call_id in {
                call["id"] for call in (messages[index - 1].tool_calls or [])
            }


@pytest.mark.asyncio
async def test_initial_assistant_with_empty_content_and_tool_calls_survives(
    tmp_path: Path,
) -> None:
    agent = Agent(name="writer", patterns=[FakePattern({"success": True})])
    runner = AgentRunner(
        agent=agent,
        workspace_manager=FakeWorkspaceManager(tmp_path),
    )

    initial_messages = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call-2",
                    "type": "function",
                    "function": {"name": "list_files", "arguments": "{}"},
                }
            ],
        },
    ]

    result = await runner.run(
        task=None,
        execution_id="exec-empty-content-tool-calls",
        initial_messages=initial_messages,
    )

    assert result["success"] is True
    messages = result["context"].messages
    # The original defect this test guards: an assistant message with
    # content="" and non-empty tool_calls must not be dropped by the replay
    # loop's "is there anything to keep" check. The leading synthetic user
    # turn (added because initial_messages[0] is role "assistant") must not
    # mask that — the assistant message must still be present right after it.
    assert [message.role for message in messages] == ["user", "assistant"]

    synthetic_message = messages[0]
    assert synthetic_message.content == "(conversation start)"
    assert synthetic_message.metadata["_xagent_synthetic"] == "leading_user_turn"

    assistant_message = messages[1]
    assert assistant_message.content == ""
    assert assistant_message.tool_calls == initial_messages[0]["tool_calls"]


@pytest.mark.asyncio
async def test_initial_messages_starting_with_user_no_synthetic_turn(
    tmp_path: Path,
) -> None:
    """A realistically-shaped reconstruction that already starts with a user
    transcript message, followed by an assistant/tool pair, must NOT trigger
    the synthetic leading-user-turn correction: it is only needed when the
    replay would otherwise start with role "assistant".
    """
    agent = Agent(name="writer", patterns=[FakePattern({"success": True})])
    runner = AgentRunner(
        agent=agent,
        workspace_manager=FakeWorkspaceManager(tmp_path),
    )

    initial_messages = [
        {"role": "user", "content": "Please read the file"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call-3",
                    "type": "function",
                    "function": {"name": "read_file", "arguments": "{}"},
                }
            ],
        },
        {
            "role": "tool",
            "tool_name": "read_file",
            "tool_call_id": "call-3",
            "raw_result": {"output": "file contents"},
        },
    ]

    result = await runner.run(
        task=None,
        execution_id="exec-replay-user-first",
        initial_messages=initial_messages,
    )

    assert result["success"] is True
    messages = result["context"].messages
    assert [message.role for message in messages] == ["user", "assistant", "tool"]
    assert messages[0].content == "Please read the file"
    assert not any(
        message.metadata.get("_xagent_synthetic") == "leading_user_turn"
        for message in messages
    )


@pytest.mark.asyncio
async def test_initial_messages_plain_roles_unchanged(tmp_path: Path) -> None:
    agent = Agent(name="writer", patterns=[FakePattern({"success": True})])
    runner = AgentRunner(
        agent=agent,
        workspace_manager=FakeWorkspaceManager(tmp_path),
    )

    initial_messages = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Hello there"},
        {"role": "user", "content": ""},  # dropped: no content/context_refs
    ]

    result = await runner.run(
        task=None,
        execution_id="exec-plain-initial",
        initial_messages=initial_messages,
    )

    assert result["success"] is True
    messages = result["context"].messages
    assert [(message.role, message.content) for message in messages] == [
        ("system", "You are helpful."),
        ("user", "Hello there"),
    ]


@pytest.mark.asyncio
async def test_runner_stops_on_llm_call_interrupt(tmp_path: Path) -> None:
    fallback = FakePattern({"success": True, "output": "should not run"})
    agent = Agent(name="writer", patterns=[LLMInterruptedPattern(), fallback])
    runner = AgentRunner(
        agent=agent,
        workspace_manager=FakeWorkspaceManager(tmp_path),
    )

    result = await runner.run(task="Pause me", execution_id="exec-llm-interrupt")

    assert result["success"] is False
    assert result["status"] == "interrupted"
    assert result["error"] == "paused during LLM call"
    assert result["pattern"] == "LLMInterruptedPattern"
    assert fallback.calls == []


@pytest.mark.asyncio
async def test_runner_restores_context_and_pattern_from_checkpoint(
    tmp_path: Path,
) -> None:
    checkpoint_context = ExecutionContext(execution_id="exec-resume")
    checkpoint_context.add_user_message("Original task")
    checkpoint_context.metadata[PREFERRED_INPUT_MODALITIES_METADATA_KEY] = ["text"]
    checkpoint = {
        "context": checkpoint_context.to_dict(),
        "pattern": "StatefulPattern",
        "pattern_state": {"output": "restored"},
    }
    pattern = StatefulPattern()
    agent = Agent(name="writer", patterns=[pattern])
    runner = AgentRunner(
        agent=agent,
        workspace_manager=FakeWorkspaceManager(tmp_path),
    )

    result = await runner.run(
        task="Should not be appended",
        execution_id="exec-resume",
        checkpoint=checkpoint,
        metadata={PREFERRED_INPUT_MODALITIES_METADATA_KEY: ["image"]},
    )

    assert result["success"] is True
    assert result["output"] == "restored"
    assert result["message_count"] == 1
    assert pattern.state == {"output": "restored"}
    assert result["context"].metadata[PREFERRED_INPUT_MODALITIES_METADATA_KEY] == [
        "image"
    ]
    assert [message.content for message in result["context"].messages] == [
        "Original task",
        "restored",
    ]


@pytest.mark.asyncio
async def test_runner_clears_checkpointed_modality_preference(
    tmp_path: Path,
) -> None:
    checkpoint_context = ExecutionContext(execution_id="exec-clear-modality")
    checkpoint_context.add_user_message("Original task")
    checkpoint_context.metadata[PREFERRED_INPUT_MODALITIES_METADATA_KEY] = ["image"]
    checkpoint = {
        "context": checkpoint_context.to_dict(),
        "pattern": "StatefulPattern",
        "pattern_state": {"output": "restored"},
    }
    pattern = StatefulPattern()
    runner = AgentRunner(
        agent=Agent(name="writer", patterns=[pattern]),
        workspace_manager=FakeWorkspaceManager(tmp_path),
    )

    result = await runner.run(
        task="Should not be appended",
        execution_id="exec-clear-modality",
        checkpoint=checkpoint,
        metadata={PREFERRED_INPUT_MODALITIES_METADATA_KEY: []},
    )

    assert PREFERRED_INPUT_MODALITIES_METADATA_KEY not in result["context"].metadata


def test_merge_context_metadata_restored_clears_absent_modality_key(
    tmp_path: Path,
) -> None:
    """Restored merges clear the modality key just like fresh-context merges."""

    runner = AgentRunner(
        agent=Agent(name="writer", patterns=[StatefulPattern()]),
        workspace_manager=FakeWorkspaceManager(tmp_path),
    )
    context = ExecutionContext(execution_id="exec-merge-absent")
    context.metadata[PREFERRED_INPUT_MODALITIES_METADATA_KEY] = ["image"]
    context.metadata["execution_type"] = "checkpointed"

    runner._merge_context_metadata(context, {}, restored=True)

    assert PREFERRED_INPUT_MODALITIES_METADATA_KEY not in context.metadata
    assert context.metadata["execution_type"] == "checkpointed"


def test_merge_context_metadata_restored_overlays_execution_identity(
    tmp_path: Path,
) -> None:
    runner = AgentRunner(
        agent=Agent(name="writer", patterns=[StatefulPattern()]),
        workspace_manager=FakeWorkspaceManager(tmp_path),
    )
    context = ExecutionContext(execution_id="exec-trusted-identity")
    context.metadata.update(
        {"task_source": "spoofed", "run_id": "stale", "other": "checkpointed"}
    )

    runner._merge_context_metadata(
        context,
        {"task_source": "slack", "run_id": "run-current"},
        restored=True,
    )

    assert context.metadata["task_source"] == "slack"
    assert context.metadata["run_id"] == "run-current"
    assert context.metadata["other"] == "checkpointed"


@pytest.mark.parametrize("key", ["task_source", "run_id"])
def test_merge_context_metadata_restored_keeps_identity_against_none(
    tmp_path: Path, key: str
) -> None:
    """A resume entry point with no trusted source must not erase the real one.

    Several resume paths pass these keys unconditionally with a None value.
    Overwriting the checkpointed identity with None would permanently deny a
    pending approval that was gated under it.
    """

    runner = AgentRunner(
        agent=Agent(name="writer", patterns=[StatefulPattern()]),
        workspace_manager=FakeWorkspaceManager(tmp_path),
    )
    context = ExecutionContext(execution_id="exec-trusted-identity")
    context.metadata.update({"task_source": "slack", "run_id": "run-original"})

    runner._merge_context_metadata(context, {key: None}, restored=True)

    assert context.metadata["task_source"] == "slack"
    assert context.metadata["run_id"] == "run-original"


@pytest.mark.asyncio
async def test_runner_empty_resume_metadata_preserves_non_modality_metadata(
    tmp_path: Path,
) -> None:
    """Resume metadata is authoritative for the modality key only.

    Every other checkpointed metadata entry survives an empty resume metadata
    mapping; the modality preference is cleared because the current run did not
    declare one.
    """

    checkpoint_context = ExecutionContext(execution_id="exec-preserve-metadata")
    checkpoint_context.add_user_message("Original task")
    checkpoint_context.metadata.update(
        {
            PREFERRED_INPUT_MODALITIES_METADATA_KEY: ["image"],
            "execution_type": "checkpointed",
        }
    )
    checkpoint = {
        "context": checkpoint_context.to_dict(),
        "pattern": "StatefulPattern",
        "pattern_state": {"output": "restored"},
    }
    runner = AgentRunner(
        agent=Agent(name="writer", patterns=[StatefulPattern()]),
        workspace_manager=FakeWorkspaceManager(tmp_path),
    )

    result = await runner.run(
        task="Should not be appended",
        execution_id="exec-preserve-metadata",
        checkpoint=checkpoint,
        metadata={},
    )

    assert PREFERRED_INPUT_MODALITIES_METADATA_KEY not in result["context"].metadata
    assert result["context"].metadata["execution_type"] == "checkpointed"


@pytest.mark.asyncio
async def test_runner_registers_restored_context_for_live_message_injection(
    tmp_path: Path,
) -> None:
    checkpoint_context = ExecutionContext(execution_id="exec-restore-inject")
    checkpoint_context.add_user_message("Original task")
    checkpoint = {
        "context": checkpoint_context.to_dict(),
        "pattern": "InjectingPattern",
        "pattern_state": {},
    }
    agent = Agent(name="writer", patterns=[])
    runner = AgentRunner(
        agent=agent,
        workspace_manager=FakeWorkspaceManager(tmp_path),
    )
    agent.patterns = [InjectingPattern(runner, "exec-restore-inject")]

    result = await runner.run(
        task=None,
        execution_id="exec-restore-inject",
        checkpoint=checkpoint,
    )

    assert result["success"] is True
    assert result["same_context"] is True
    assert result["messages"] == ["Original task", "Injected while resumed."]


@pytest.mark.asyncio
async def test_runner_pause_requests_interrupt_for_active_execution(
    tmp_path: Path,
) -> None:
    tracer = TracerCheckpointStore()
    agent = Agent(name="writer", patterns=[])
    runner = AgentRunner(
        agent=agent,
        tracer=tracer,
        workspace_manager=FakeWorkspaceManager(tmp_path),
    )
    agent.patterns = [InterruptingPattern(runner, "exec-pause")]

    result = await runner.run(task="Calculate 6*7", execution_id="exec-pause")

    assert result["success"] is False
    assert result["status"] == "interrupted"
    assert tracer.by_execution_id["exec-pause"]["label"] == "interrupted"
    assert (
        tracer.by_execution_id["exec-pause"]["metadata"]["safe_point"]
        == "during_pattern"
    )


@pytest.mark.asyncio
async def test_runner_inject_user_message_updates_live_context_and_requests_interrupt(
    tmp_path: Path,
) -> None:
    tracer = TracerCheckpointStore()
    agent = Agent(name="writer", patterns=[])
    runner = AgentRunner(
        agent=agent,
        tracer=tracer,
        workspace_manager=FakeWorkspaceManager(tmp_path),
    )
    agent.patterns = [
        InterruptingPattern(
            runner,
            "exec-inject",
            before_interrupt_check=lambda: runner.inject_user_message(
                "exec-inject",
                "Use metric units.",
                reason="new user message",
            ),
        )
    ]

    result = await runner.run(task="Calculate 6*7", execution_id="exec-inject")
    context = result["context"]

    assert result["success"] is False
    assert result["status"] == "interrupted"
    user_messages = [msg.content for msg in context.messages if msg.role == "user"]
    assert user_messages == ["Calculate 6*7", "Use metric units."]
    checkpoint_messages = tracer.by_execution_id["exec-inject"]["context"]["messages"]
    assert any(
        message["role"] == "user" and message["content"] == "Use metric units."
        for message in checkpoint_messages
    )


@pytest.mark.asyncio
async def test_runner_resume_restores_from_latest_checkpoint_after_restart(
    tmp_path: Path,
) -> None:
    tracer = TracerCheckpointStore()
    execution_id = "exec-restart"
    first_agent = Agent(name="writer", patterns=[])
    first_runner = AgentRunner(
        agent=first_agent,
        tracer=tracer,
        workspace_manager=FakeWorkspaceManager(tmp_path),
    )
    first_agent.patterns = [InterruptingPattern(first_runner, execution_id)]

    interrupted = await first_runner.run(
        task="Calculate 6*7",
        execution_id=execution_id,
    )

    assert interrupted["status"] == "interrupted"
    await first_runner.inject_user_message(
        execution_id,
        "Reply with only the number.",
        request_interrupt=False,
    )

    agent = Agent(
        name="writer",
        patterns=[FakePattern({"success": True, "response": "42"})],
    )
    resumed_runner = AgentRunner(
        agent=agent,
        tracer=tracer,
        workspace_manager=FakeWorkspaceManager(tmp_path),
    )

    resumed = await resumed_runner.resume(execution_id)

    assert resumed["success"] is True
    assert resumed["response"] == "42"
    resumed_contents = [message.content for message in resumed["context"].messages]
    assert "Reply with only the number." in resumed_contents
    assert resumed_contents.index(
        "Reply with only the number."
    ) < resumed_contents.index("42")


@pytest.mark.asyncio
async def test_runner_inject_user_message_with_files_dispatches_trace_callback(
    tmp_path: Path,
) -> None:
    """End-to-end coverage of the continuation chip path: a websocket-style
    ``post_user_message`` call with attachments must (a) attach the files to
    the new Message so they survive checkpoints and (b) fire the trace
    callback so the chip is broadcast live (instead of only appearing after
    a page reload via historical replay)."""
    tracer = RecordingTraceEventTracer()
    agent = Agent(name="writer", patterns=[FakePattern({"success": True})])
    runner = AgentRunner(
        agent=agent,
        tracer=tracer,
        callbacks=[TraceEventCallback()],
        workspace_manager=FakeWorkspaceManager(tmp_path),
    )
    await runner.run(task="Original task", execution_id="exec-cont-files")

    files = [
        {
            "file_id": "fid-cont",
            "name": "follow-up.pdf",
            "size": 2048,
            "type": "application/pdf",
        }
    ]
    result = await runner.post_user_message(
        "exec-cont-files",
        "Use the attached PDF.",
        request_interrupt=False,
        files=files,
    )

    assert result.context is not None
    new_user_message = next(
        msg for msg in reversed(result.context.messages) if msg.role == "user"
    )
    assert new_user_message.metadata.get("files") == files
    turn_id = new_user_message.metadata.get("turn_id")
    assert isinstance(turn_id, str) and turn_id

    user_message_events = [
        event
        for event in tracer.events
        if event["event_type"] == "task_start_message"
        and event["data"].get("message") == "Use the attached PDF."
    ]
    assert len(user_message_events) == 1
    assert user_message_events[0]["data"]["turn_id"] == turn_id
    assert user_message_events[0]["data"]["files"] == files
    assert user_message_events[0]["data"]["attachments"] == files


@pytest.mark.asyncio
async def test_runner_post_user_message_alias_matches_inject_behavior(
    tmp_path: Path,
) -> None:
    tracer = TracerCheckpointStore()
    agent = Agent(name="writer", patterns=[FakePattern({"success": True})])
    runner = AgentRunner(
        agent=agent,
        tracer=tracer,
        workspace_manager=FakeWorkspaceManager(tmp_path),
    )
    checkpoint_context = ExecutionContext(execution_id="exec-alias")
    checkpoint_context.add_user_message("Original task")
    await tracer.checkpoint(
        type="checkpoint",
        execution_id="exec-alias",
        pattern="FakePattern",
        label="before_llm",
        status="interrupted",
        context=checkpoint_context.to_dict(),
        pattern_state={},
        metadata={},
    )

    result = await runner.post_user_message(
        "exec-alias",
        "Follow-up from user.",
        request_interrupt=False,
    )

    assert result.context is not None
    user_messages = [
        message.content for message in result.context.messages if message.role == "user"
    ]
    assert user_messages == ["Original task", "Follow-up from user."]


@pytest.mark.asyncio
async def test_runner_post_user_message_deduplicates_explicit_turn_id_after_failure(
    tmp_path: Path,
) -> None:
    tracer = TracerCheckpointStore()
    execution_id = "exec-idempotent-message"
    checkpoint_context = ExecutionContext(execution_id=execution_id)
    checkpoint_context.add_user_message("Original task")
    await tracer.checkpoint(
        type="checkpoint",
        execution_id=execution_id,
        pattern="FakePattern",
        label="waiting_for_user",
        status="waiting_for_user",
        context=checkpoint_context.to_dict(),
        pattern_state={},
        metadata={},
    )
    failing_runner = AgentRunner(
        agent=Agent(name="writer", patterns=[FakePattern({"success": True})]),
        tracer=tracer,
        callbacks=[FailingUserMessageCallback()],
        workspace_manager=FakeWorkspaceManager(tmp_path),
    )
    failing_runner.pause = MagicMock(return_value=True)

    accepted = await failing_runner.post_user_message(
        execution_id,
        "Choose B",
        turn_id="a2a:42:msg-1",
        request_interrupt=True,
    )

    assert accepted.context is not None
    assert accepted.outcome is UserMessageInjectionOutcome.POSTED_FRESH
    failing_runner.pause.assert_called_once_with(
        execution_id,
        reason="new user message",
    )

    failing_runner.context_manager.remove_context(execution_id)
    retry_runner = AgentRunner(
        agent=Agent(name="writer", patterns=[FakePattern({"success": True})]),
        tracer=tracer,
        workspace_manager=FakeWorkspaceManager(tmp_path),
    )
    result = await retry_runner.post_user_message(
        execution_id,
        "Choose B",
        turn_id="a2a:42:msg-1",
        request_interrupt=False,
    )

    # The retry cold-starts from the checkpoint the first (pre-callback-
    # failure) attempt already persisted, so this is the short-circuit
    # replaying an already-seen turn id, not a second fresh write.
    assert result.context is not None
    assert result.outcome is UserMessageInjectionOutcome.POSTED_REPLAY
    retried_messages = [
        message
        for message in result.context.messages
        if message.role == "user" and message.metadata.get("turn_id") == "a2a:42:msg-1"
    ]
    assert len(retried_messages) == 1
    assert retried_messages[0].content == "Choose B"


@pytest.mark.asyncio
async def test_runner_rejects_reused_turn_id_with_different_content(
    tmp_path: Path,
) -> None:
    tracer = TracerCheckpointStore()
    execution_id = "exec-conflicting-message"
    checkpoint_context = ExecutionContext(execution_id=execution_id)
    checkpoint_context.add_user_message(
        "Choose A",
        metadata={"turn_id": "a2a:42:msg-1"},
    )
    await tracer.checkpoint(
        type="checkpoint",
        execution_id=execution_id,
        pattern="FakePattern",
        label="waiting_for_user",
        status="waiting_for_user",
        context=checkpoint_context.to_dict(),
        pattern_state={},
        metadata={},
    )
    runner = AgentRunner(
        agent=Agent(name="writer", patterns=[FakePattern({"success": True})]),
        tracer=tracer,
        workspace_manager=FakeWorkspaceManager(tmp_path),
    )

    with pytest.raises(ValueError, match="different user message"):
        await runner.post_user_message(
            execution_id,
            "Choose B",
            turn_id="a2a:42:msg-1",
            request_interrupt=False,
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("scenario", ["fresh", "replay", "conflicting_content"])
async def test_runner_inject_user_message_reports_fresh_vs_replay(
    tmp_path: Path, scenario: str
) -> None:
    """The three states this contract exists to name: a first write reports
    POSTED_FRESH, a repeat of the same turn id with the same content
    short-circuits and reports POSTED_REPLAY without persisting anything
    new, and a repeat with different content still raises -- unchanged from
    before this contract existed."""
    tracer = TracerCheckpointStore()
    execution_id = "exec-fresh-replay-grid"
    agent = Agent(name="writer", patterns=[FakePattern({"success": True})])
    runner = AgentRunner(
        agent=agent,
        tracer=tracer,
        workspace_manager=FakeWorkspaceManager(tmp_path),
    )
    await runner.run(task="Original task", execution_id=execution_id)

    first = await runner.inject_user_message(
        execution_id,
        "Choose B",
        turn_id="turn-fresh-replay-grid",
        request_interrupt=False,
    )
    assert first.outcome is UserMessageInjectionOutcome.POSTED_FRESH
    assert first.context is not None

    if scenario == "fresh":
        return

    if scenario == "replay":
        second = await runner.inject_user_message(
            execution_id,
            "Choose B",
            turn_id="turn-fresh-replay-grid",
            request_interrupt=False,
        )
        assert second.outcome is UserMessageInjectionOutcome.POSTED_REPLAY
        assert second.context is first.context
        matching = [
            message
            for message in second.context.messages
            if message.role == "user"
            and message.metadata.get("turn_id") == "turn-fresh-replay-grid"
        ]
        assert len(matching) == 1
        return

    assert scenario == "conflicting_content"
    with pytest.raises(ValueError, match="different user message"):
        await runner.inject_user_message(
            execution_id,
            "Choose C",
            turn_id="turn-fresh-replay-grid",
            request_interrupt=False,
        )


@pytest.mark.asyncio
async def test_runner_post_user_message_preserves_display_and_execution_contract(
    tmp_path: Path,
) -> None:
    tracer = TracerCheckpointStore()
    agent = Agent(name="writer", patterns=[FakePattern({"success": True})])
    runner = AgentRunner(
        agent=agent,
        tracer=tracer,
        workspace_manager=FakeWorkspaceManager(tmp_path),
    )
    checkpoint_context = ExecutionContext(execution_id="exec-display-contract")
    checkpoint_context.add_user_message("Original task")
    await tracer.checkpoint(
        type="checkpoint",
        execution_id="exec-display-contract",
        pattern="FakePattern",
        label="before_llm",
        status="interrupted",
        context=checkpoint_context.to_dict(),
        pattern_state={},
        metadata={},
    )

    execution_message = "Read file\n\n## UPLOADED FILES\nfile_id=file-123"
    files = [{"file_id": "file-123", "name": "notes.txt"}]
    result = await runner.post_user_message(
        "exec-display-contract",
        execution_message=execution_message,
        display_message="Read file",
        files=files,
        turn_id="client-turn-123",
        request_interrupt=False,
    )

    assert result.context is not None
    latest_user = [
        message for message in result.context.messages if message.role == "user"
    ][-1]
    assert latest_user.content == execution_message
    assert latest_user.metadata["display_message"] == "Read file"
    assert latest_user.metadata["files"] == files
    turn_id = latest_user.metadata.get("turn_id")
    assert turn_id == "client-turn-123"

    checkpoint_messages = tracer.by_execution_id["exec-display-contract"]["context"][
        "messages"
    ]
    latest_checkpoint_user = [
        message for message in checkpoint_messages if message["role"] == "user"
    ][-1]
    assert latest_checkpoint_user["content"] == execution_message
    assert latest_checkpoint_user["metadata"]["display_message"] == "Read file"
    assert latest_checkpoint_user["metadata"]["files"] == files
    assert latest_checkpoint_user["metadata"]["turn_id"] == turn_id


@pytest.mark.asyncio
async def test_runner_initial_user_message_preserves_display_metadata(
    tmp_path: Path,
) -> None:
    tracer = RecordingTraceEventTracer()
    agent = Agent(
        name="writer",
        patterns=[FakePattern({"success": True, "response": "Done"})],
    )
    runner = AgentRunner(
        agent=agent,
        tracer=tracer,
        callbacks=[TraceEventCallback()],
        workspace_manager=FakeWorkspaceManager(tmp_path),
    )

    execution_message = "Read file\n\n## UPLOADED FILES\nfile_id=file-123"
    files = [{"file_id": "file-123", "name": "notes.txt"}]
    result = await runner.run(
        task=execution_message,
        execution_id="exec-initial-display",
        metadata={"request_context": {"display_message": "Read file", "files": files}},
    )

    first_user = next(
        message for message in result["context"].messages if message.role == "user"
    )
    assert first_user.content == execution_message
    assert first_user.metadata["display_message"] == "Read file"
    assert first_user.metadata["files"] == files
    assert result["context"].current_user_request_text(prefer_display=True) == (
        "Read file"
    )
    turn_id = first_user.metadata.get("turn_id")
    assert isinstance(turn_id, str) and turn_id
    user_event = next(
        event for event in tracer.events if event["event_type"] == "task_start_message"
    )
    assert user_event["data"]["message"] == "Read file"
    assert user_event["data"]["turn_id"] == turn_id


@pytest.mark.parametrize(
    ("request_context", "expected"),
    [
        pytest.param({}, None, id="missing"),
        pytest.param({"display_message": None}, "", id="null"),
        pytest.param({"display_message": 17}, "", id="non-string"),
        pytest.param({"display_message": ""}, "", id="blank"),
        pytest.param({"display_message": "  \n\t"}, "  \n\t", id="whitespace"),
        pytest.param({"display_message": "Read file"}, "Read file", id="text"),
    ],
)
def test_runner_normalizes_initial_display_message_state(
    request_context: dict[str, Any], expected: str | None
) -> None:
    runner = AgentRunner(agent=Agent(name="writer", patterns=[]))
    context = ExecutionContext(metadata={"request_context": request_context})

    metadata = runner._initial_user_message_metadata(context)

    if expected is None:
        assert "display_message" not in metadata
    else:
        assert metadata["display_message"] == expected


@pytest.mark.asyncio
async def test_runner_attaches_uploaded_image_refs_to_initial_user_message(
    tmp_path: Path,
) -> None:
    agent = Agent(
        name="vision",
        patterns=[FakePattern({"success": True, "response": "Done"})],
    )
    runner = AgentRunner(
        agent=agent,
        workspace_manager=FakeWorkspaceManager(tmp_path),
    )
    references = build_image_context_references(
        [{"file_id": "image-123", "name": "diagram.png", "type": "image/png"}]
    )

    result = await runner.run(
        task="What is shown?",
        execution_id="exec-initial-image",
        task_context_refs=references,
    )

    first_user = next(
        message for message in result["context"].messages if message.role == "user"
    )
    assert first_user.context_refs == references


@pytest.mark.asyncio
async def test_runner_attaches_uploaded_image_refs_to_injected_user_message(
    tmp_path: Path,
) -> None:
    tracer = TracerCheckpointStore()
    agent = Agent(name="vision", patterns=[FakePattern({"success": True})])
    runner = AgentRunner(
        agent=agent,
        tracer=tracer,
        workspace_manager=FakeWorkspaceManager(tmp_path),
    )
    await runner.run(task="Start", execution_id="exec-injected-image")

    result = await runner.inject_user_message(
        "exec-injected-image",
        "Inspect the new image",
        files=[{"file_id": "image-456", "name": "screen.jpg", "type": "image/jpeg"}],
        request_interrupt=False,
    )

    assert result.context is not None
    assert result.context.messages[-1].context_refs[0].file_id == "image-456"


@pytest.mark.asyncio
async def test_runner_post_user_message_rejects_execution_without_display(
    tmp_path: Path,
) -> None:
    tracer = TracerCheckpointStore()
    agent = Agent(name="writer", patterns=[FakePattern({"success": True})])
    runner = AgentRunner(
        agent=agent,
        tracer=tracer,
        workspace_manager=FakeWorkspaceManager(tmp_path),
    )
    checkpoint_context = ExecutionContext(execution_id="exec-display-required")
    checkpoint_context.add_user_message("Original task")
    await tracer.checkpoint(
        type="checkpoint",
        execution_id="exec-display-required",
        pattern="FakePattern",
        label="before_llm",
        status="interrupted",
        context=checkpoint_context.to_dict(),
        pattern_state={},
        metadata={},
    )

    with pytest.raises(ValueError, match="requires display_message"):
        await runner.post_user_message(
            "exec-display-required",
            execution_message="Read file\n\n## UPLOADED FILES\nfile_id=file-123",
            request_interrupt=False,
        )


@pytest.mark.asyncio
async def test_trace_callback_does_not_emit_completion_for_interrupted_run(
    tmp_path: Path,
) -> None:
    tracer = RecordingTraceEventTracer()
    agent = Agent(
        name="paused",
        patterns=[
            FakePattern(
                {
                    "success": False,
                    "status": "interrupted",
                    "error": "Paused by user.",
                }
            )
        ],
    )
    runner = AgentRunner(
        agent=agent,
        tracer=tracer,
        callbacks=[TraceEventCallback()],
        workspace_manager=FakeWorkspaceManager(tmp_path),
    )

    result = await runner.run(task="Pause this", execution_id="exec-paused")

    assert result["status"] == "interrupted"
    event_types = [event["event_type"] for event in tracer.events]
    assert event_types == ["task_start_message"]
    assert "task_end_general" not in event_types


@pytest.mark.asyncio
async def test_trace_callback_unwraps_final_answer_and_omits_success_context(
    tmp_path: Path,
) -> None:
    tracer = RecordingTraceEventTracer()
    agent = Agent(
        name="writer",
        patterns=[
            FakePattern(
                {
                    "success": True,
                    "output": (
                        '```json\n{"action":"final_answer",'
                        '"action_input":"Done cleanly."}\n```'
                    ),
                    "message": (
                        '```json\n{"action":"final_answer",'
                        '"action_input":"Done cleanly."}\n```'
                    ),
                }
            )
        ],
    )
    runner = AgentRunner(
        agent=agent,
        tracer=tracer,
        callbacks=[TraceEventCallback()],
        workspace_manager=FakeWorkspaceManager(tmp_path),
    )

    result = await runner.run(task="Finish", execution_id="exec-success")

    assert result["output"] == "Done cleanly."
    assert result["message"] == "Done cleanly."
    ai_event = next(
        event for event in tracer.events if event["event_type"] == "task_end_message"
    )
    assert ai_event["data"]["content"] == "Done cleanly."
    assert "context" not in ai_event["data"]


class _FakeLLM:
    def __init__(self, context_window: Any, model_name: str = "fake-model") -> None:
        self.context_window = context_window
        self.model_name = model_name


def _threshold_runner(
    context_window: Any, model_name: str = "fake-model"
) -> AgentRunner:
    agent = Agent(
        name="t",
        patterns=[FakePattern({})],
        llm=_FakeLLM(context_window, model_name),
    )
    return AgentRunner(agent=agent)


def test_resolve_compact_threshold_uses_window_ratio(monkeypatch) -> None:
    monkeypatch.delenv("XAGENT_COMPACT_THRESHOLD_RATIO", raising=False)
    # 128000 * 0.75
    assert _threshold_runner(128000)._resolve_compact_threshold() == (
        96000,
        COMPACT_THRESHOLD_SOURCE_CONTEXT_WINDOW,
    )


def test_resolve_compact_threshold_respects_ratio_env(monkeypatch) -> None:
    monkeypatch.setenv("XAGENT_COMPACT_THRESHOLD_RATIO", "0.8")
    assert _threshold_runner(200000)._resolve_compact_threshold() == (
        160000,
        COMPACT_THRESHOLD_SOURCE_CONTEXT_WINDOW,
    )


@pytest.mark.parametrize("window", [None, 0, -1, "128000"])
def test_resolve_compact_threshold_falls_back_to_default(monkeypatch, window) -> None:
    monkeypatch.delenv("XAGENT_COMPACT_THRESHOLD_DEFAULT", raising=False)
    # None / non-positive / non-int all fall back to the global default.
    assert _threshold_runner(window)._resolve_compact_threshold() == (
        32000,
        COMPACT_THRESHOLD_SOURCE_DEFAULT,
    )


def test_resolve_compact_threshold_default_env_override(monkeypatch) -> None:
    monkeypatch.setenv("XAGENT_COMPACT_THRESHOLD_DEFAULT", "50000")
    assert _threshold_runner(None)._resolve_compact_threshold() == (
        50000,
        COMPACT_THRESHOLD_SOURCE_DEFAULT,
    )


def test_resolve_compact_threshold_missing_llm() -> None:
    agent = Agent(name="t", patterns=[FakePattern({})], llm=None)
    assert AgentRunner(agent=agent)._resolve_compact_threshold() == (
        32000,
        COMPACT_THRESHOLD_SOURCE_DEFAULT,
    )


def test_resolve_compact_threshold_warns_once_per_model_on_fallback(
    monkeypatch, caplog
) -> None:
    monkeypatch.delenv("XAGENT_COMPACT_THRESHOLD_DEFAULT", raising=False)
    runner = _threshold_runner(None, model_name="moonshotai.kimi-k2.5")

    with caplog.at_level(logging.WARNING, logger="xagent.core.agent.runtime"):
        runner._resolve_compact_threshold()
        runner._resolve_compact_threshold()
        _threshold_runner(None, model_name="other-model")._resolve_compact_threshold()
        _threshold_runner(128000, model_name="sized")._resolve_compact_threshold()

    fallback_records = [
        record
        for record in caplog.records
        if "context_window" in record.getMessage()
        and "compaction threshold" in record.getMessage()
    ]
    assert len(fallback_records) == 2
    assert "moonshotai.kimi-k2.5" in fallback_records[0].getMessage()
    assert "32000" in fallback_records[0].getMessage()
    assert "other-model" in fallback_records[1].getMessage()


def test_resolve_compact_threshold_does_not_warn_for_virtual_models(
    monkeypatch, caplog
) -> None:
    monkeypatch.delenv("XAGENT_COMPACT_THRESHOLD_DEFAULT", raising=False)

    class VirtualLLM:
        model_name = "auto"
        context_window = None

        async def prepare_for_call(self, messages: Any, **_: Any) -> Any:
            return self

    agent = Agent(name="t", patterns=[FakePattern({})], llm=VirtualLLM())
    with caplog.at_level(logging.WARNING, logger="xagent.core.agent.runtime"):
        resolved = AgentRunner(agent=agent)._resolve_compact_threshold()

    # The window is resolved per call and the threshold recomputed then.
    assert resolved == (32000, COMPACT_THRESHOLD_SOURCE_DEFAULT)
    assert not any("compaction threshold" in r.getMessage() for r in caplog.records)


@pytest.mark.asyncio
async def test_run_records_compact_threshold_source_on_context(monkeypatch) -> None:
    monkeypatch.delenv("XAGENT_COMPACT_THRESHOLD_RATIO", raising=False)
    captured: dict[str, Any] = {}

    class CapturingPattern(FakePattern):
        async def run(self, **kwargs: Any) -> dict[str, Any]:
            context = kwargs["context"]
            captured["threshold"] = context.compact_config.threshold
            captured["source"] = context.compact_config.threshold_source
            return await super().run(**kwargs)

    agent = Agent(name="t", patterns=[CapturingPattern({})], llm=_FakeLLM(128000))
    await AgentRunner(agent=agent).run("hello")

    assert captured == {
        "threshold": 96000,
        "source": COMPACT_THRESHOLD_SOURCE_CONTEXT_WINDOW,
    }


@pytest.mark.asyncio
async def test_run_resume_warns_when_restored_threshold_is_the_default(
    tmp_path: Path, caplog
) -> None:
    """A task resumed in a fresh process has no in-memory record of why its
    compaction threshold is what it is; ``AgentRunner.run`` re-issues the
    fallback warning from the restored checkpoint so the missing
    ``context_window`` column is still visible in this process's log."""
    tracer = TracerCheckpointStore()
    execution_id = "exec-resume-warn"
    checkpoint_context = ExecutionContext(execution_id=execution_id)
    checkpoint_context.add_user_message("Original task")
    assert (
        checkpoint_context.compact_config.threshold_source
        == COMPACT_THRESHOLD_SOURCE_DEFAULT
    )
    tracer.by_execution_id[execution_id] = {
        "execution_id": execution_id,
        "context": checkpoint_context.to_dict(),
    }

    agent = Agent(
        name="writer",
        patterns=[FakePattern({"success": True, "message": "ok"})],
        llm=_FakeLLM(None, "moonshotai.kimi-k2.5"),
    )
    runner = AgentRunner(
        agent=agent,
        tracer=tracer,
        workspace_manager=FakeWorkspaceManager(tmp_path),
    )

    with caplog.at_level(logging.WARNING, logger="xagent.core.agent.runtime"):
        result = await runner.run(task=None, execution_id=execution_id, resume=True)

    assert result["success"] is True
    warnings = [
        record.getMessage()
        for record in caplog.records
        if "resumed task" in record.getMessage()
    ]
    assert len(warnings) == 1
    assert "moonshotai.kimi-k2.5" in warnings[0]


@pytest.mark.asyncio
async def test_run_resume_stays_silent_when_model_now_has_a_window(
    tmp_path: Path, caplog
) -> None:
    """The model row backing this resumed task now has a ``context_window``
    (populated after the checkpoint was written, or simply since a fresh
    process last saw it), so the restored default threshold is not running
    blind and must not be re-warned about."""
    tracer = TracerCheckpointStore()
    execution_id = "exec-resume-silent"
    checkpoint_context = ExecutionContext(execution_id=execution_id)
    checkpoint_context.add_user_message("Original task")
    tracer.by_execution_id[execution_id] = {
        "execution_id": execution_id,
        "context": checkpoint_context.to_dict(),
    }

    agent = Agent(
        name="writer",
        patterns=[FakePattern({"success": True, "message": "ok"})],
        llm=_FakeLLM(256_000, "sized"),
    )
    runner = AgentRunner(
        agent=agent,
        tracer=tracer,
        workspace_manager=FakeWorkspaceManager(tmp_path),
    )

    with caplog.at_level(logging.WARNING, logger="xagent.core.agent.runtime"):
        result = await runner.run(task=None, execution_id=execution_id, resume=True)

    assert result["success"] is True
    assert not any("resumed task" in record.getMessage() for record in caplog.records)


@pytest.mark.asyncio
async def test_run_resume_raises_corrupt_on_contextless_checkpoint() -> None:
    runner = AgentRunner(
        agent=Agent(name="checkpoint-reader", patterns=[], llm=None),
        tracer=ContextlessCheckpointStore(),
    )
    build_context_calls: list[Any] = []
    original_build_context = runner._build_context

    async def spy_build_context(*args: Any, **kwargs: Any) -> Any:
        build_context_calls.append((args, kwargs))
        return await original_build_context(*args, **kwargs)

    runner._build_context = spy_build_context  # type: ignore[method-assign]

    with pytest.raises(CheckpointCorruptError):
        await runner.run(
            task=None,
            execution_id="exec-resume-contextless",
            resume=True,
        )

    assert build_context_calls == []
    assert runner.context_manager.get_context("exec-resume-contextless") is None


@pytest.mark.asyncio
async def test_inject_user_message_raises_corrupt_on_contextless_checkpoint() -> None:
    """A found checkpoint without a context dict is malformed, not absent:
    returning ``None`` here would be indistinguishable from "no checkpoint"
    and the caller would defer forever against a row that can never resume."""
    runner = AgentRunner(
        agent=Agent(name="checkpoint-reader", patterns=[], llm=None),
        tracer=ContextlessCheckpointStore(),
    )

    with pytest.raises(CheckpointCorruptError):
        await runner.inject_user_message(
            "exec-inject-contextless",
            message="hello",
        )


@pytest.mark.asyncio
async def test_resume_drops_legacy_router_output_language(tmp_path: Path) -> None:
    checkpoint_context = ExecutionContext(execution_id="exec-legacy-router-language")
    checkpoint_context.metadata["pattern"] = "auto"
    checkpoint_context.metadata[OUTPUT_LANGUAGE_METADATA_KEY] = "Simplified Chinese"
    checkpoint_context.metadata[OUTPUT_LANGUAGE_SOURCE_METADATA_KEY] = "auto_router"
    checkpoint_context.add_user_message("Summarize the release notes in one paragraph.")
    child_context = ExecutionContext(execution_id="exec-legacy-router-language_child")
    child_context.metadata[OUTPUT_LANGUAGE_METADATA_KEY] = "Simplified Chinese"
    child_context.metadata[OUTPUT_LANGUAGE_SOURCE_METADATA_KEY] = "auto_router"
    checkpoint = {
        "context": checkpoint_context.to_dict(),
        "pattern": "StatefulPattern",
        "pattern_state": {
            "output": "restored",
            "active_step_contexts": {"step_1": child_context.to_dict()},
        },
    }
    pattern = StatefulPattern()
    runner = AgentRunner(
        agent=Agent(name="writer", patterns=[pattern]),
        workspace_manager=FakeWorkspaceManager(tmp_path),
    )

    result = await runner.run(
        task=None,
        execution_id="exec-legacy-router-language",
        checkpoint=checkpoint,
    )

    assert result["success"] is True
    metadata = result["context"].metadata
    assert OUTPUT_LANGUAGE_METADATA_KEY not in metadata
    assert OUTPUT_LANGUAGE_SOURCE_METADATA_KEY not in metadata
    restored_child = pattern.state["active_step_contexts"]["step_1"]["metadata"]
    assert OUTPUT_LANGUAGE_METADATA_KEY not in restored_child
    assert OUTPUT_LANGUAGE_SOURCE_METADATA_KEY not in restored_child
    system_content = result["context"].get_messages_for_llm()[0]["content"]
    assert "Output language: Simplified Chinese" not in system_content
    assert "Summarize the release notes in one paragraph." in system_content


@pytest.mark.asyncio
async def test_resume_drops_legacy_plan_output_language(tmp_path: Path) -> None:
    checkpoint_context = ExecutionContext(execution_id="exec-legacy-plan-language")
    checkpoint_context.metadata["pattern"] = "dag_plan_execute"
    checkpoint_context.metadata[OUTPUT_LANGUAGE_METADATA_KEY] = "Simplified Chinese"
    checkpoint_context.metadata[OUTPUT_LANGUAGE_SOURCE_METADATA_KEY] = "dag_plan"
    checkpoint_context.add_user_message("Summarize the release notes in one paragraph.")
    child_context = ExecutionContext(execution_id="exec-legacy-plan-language_child")
    child_context.metadata[OUTPUT_LANGUAGE_METADATA_KEY] = "Simplified Chinese"
    child_context.metadata[OUTPUT_LANGUAGE_SOURCE_METADATA_KEY] = "dag_plan"
    checkpoint = {
        "context": checkpoint_context.to_dict(),
        "pattern": "StatefulPattern",
        "pattern_state": {
            "output": "restored",
            "active_step_contexts": {"step_1": child_context.to_dict()},
        },
    }
    pattern = StatefulPattern()
    runner = AgentRunner(
        agent=Agent(name="writer", patterns=[pattern]),
        workspace_manager=FakeWorkspaceManager(tmp_path),
    )

    result = await runner.run(
        task=None,
        execution_id="exec-legacy-plan-language",
        checkpoint=checkpoint,
    )

    assert result["success"] is True
    metadata = result["context"].metadata
    assert OUTPUT_LANGUAGE_METADATA_KEY not in metadata
    assert OUTPUT_LANGUAGE_SOURCE_METADATA_KEY not in metadata
    restored_child = pattern.state["active_step_contexts"]["step_1"]["metadata"]
    assert OUTPUT_LANGUAGE_METADATA_KEY not in restored_child
    assert OUTPUT_LANGUAGE_SOURCE_METADATA_KEY not in restored_child
    system_content = result["context"].get_messages_for_llm()[0]["content"]
    assert "Output language: Simplified Chinese" not in system_content
    assert "Summarize the release notes in one paragraph." in system_content


@pytest.mark.asyncio
async def test_resume_keeps_caller_supplied_output_language(tmp_path: Path) -> None:
    checkpoint_context = ExecutionContext(execution_id="exec-caller-language")
    checkpoint_context.metadata["request_context"] = {
        OUTPUT_LANGUAGE_METADATA_KEY: "French"
    }
    checkpoint_context.metadata[OUTPUT_LANGUAGE_METADATA_KEY] = "French"
    checkpoint_context.add_user_message("Summarize the release notes.")
    checkpoint = {
        "context": checkpoint_context.to_dict(),
        "pattern": "StatefulPattern",
        "pattern_state": {"output": "restored"},
    }
    runner = AgentRunner(
        agent=Agent(name="writer", patterns=[StatefulPattern()]),
        workspace_manager=FakeWorkspaceManager(tmp_path),
    )

    result = await runner.run(
        task=None,
        execution_id="exec-caller-language",
        checkpoint=checkpoint,
    )

    assert result["success"] is True
    assert result["context"].metadata[OUTPUT_LANGUAGE_METADATA_KEY] == "French"
    system_content = result["context"].get_messages_for_llm()[0]["content"]
    assert "Output language: French" in system_content


class _StoredContextCheckpointStore:
    def __init__(self, context: ExecutionContext) -> None:
        self.payload = {"type": "checkpoint", "context": context.to_dict()}

    async def load_latest_checkpoint(self, execution_id: str) -> dict[str, Any]:
        del execution_id
        return self.payload


def _cold_start_runner(context: ExecutionContext) -> AgentRunner:
    return AgentRunner(
        agent=Agent(name="writer", patterns=[], llm=None),
        tracer=_StoredContextCheckpointStore(context),
    )


@pytest.mark.asyncio
async def test_inject_user_message_cold_start_drops_legacy_output_language() -> None:
    stored = ExecutionContext(execution_id="exec-inject-legacy-language")
    stored.metadata["pattern"] = "auto"
    stored.metadata[OUTPUT_LANGUAGE_METADATA_KEY] = "Simplified Chinese"
    stored.metadata[OUTPUT_LANGUAGE_SOURCE_METADATA_KEY] = "auto_router"
    stored.add_user_message("Summarize the release notes.")

    result = await _cold_start_runner(stored).inject_user_message(
        "exec-inject-legacy-language",
        message="continue",
    )

    assert result.context is not None
    assert OUTPUT_LANGUAGE_METADATA_KEY not in result.context.metadata
    assert OUTPUT_LANGUAGE_SOURCE_METADATA_KEY not in result.context.metadata


@pytest.mark.asyncio
async def test_inject_user_message_cold_start_keeps_caller_output_language() -> None:
    stored = ExecutionContext(execution_id="exec-inject-caller-language")
    stored.metadata["request_context"] = {OUTPUT_LANGUAGE_METADATA_KEY: "French"}
    stored.metadata[OUTPUT_LANGUAGE_METADATA_KEY] = "French"
    stored.metadata[OUTPUT_LANGUAGE_SOURCE_METADATA_KEY] = "auto_router"
    stored.add_user_message("Summarize the release notes.")

    result = await _cold_start_runner(stored).inject_user_message(
        "exec-inject-caller-language",
        message="continue",
    )

    assert result.context is not None
    assert result.context.metadata[OUTPUT_LANGUAGE_METADATA_KEY] == "French"
    assert OUTPUT_LANGUAGE_SOURCE_METADATA_KEY not in result.context.metadata


def test_resume_migration_only_touches_execution_context_nodes() -> None:
    """The migration owns ExecutionContext metadata and nothing else: a
    ``metadata`` dict inside a message, a tool argument, or a step result is
    someone else's payload, and a cold start would persist a silent edit."""
    root = ExecutionContext(execution_id="exec-migration-ownership")
    root.metadata["pattern"] = "dag_plan_execute"
    root.metadata[OUTPUT_LANGUAGE_METADATA_KEY] = "Simplified Chinese"
    root.metadata[OUTPUT_LANGUAGE_SOURCE_METADATA_KEY] = "dag_plan"
    root.add_user_message(
        "Translate the attached note.",
        metadata={OUTPUT_LANGUAGE_METADATA_KEY: "French"},
    )
    child = ExecutionContext(execution_id="exec-migration-ownership_child")
    child.metadata[OUTPUT_LANGUAGE_METADATA_KEY] = "Simplified Chinese"
    child.metadata[OUTPUT_LANGUAGE_SOURCE_METADATA_KEY] = "dag_plan"

    checkpoint = {
        "context": root.to_dict(),
        "pattern": "DAGPattern",
        "metadata": {OUTPUT_LANGUAGE_METADATA_KEY: "French"},
        "pattern_state": {
            "active_step_contexts": {"step_1": child.to_dict()},
            "step_results": {
                "step_1": {"metadata": {OUTPUT_LANGUAGE_METADATA_KEY: "French"}}
            },
            "active_step_pattern_states": {
                "step_1": {
                    "last_response": {
                        "tool_calls": [
                            {
                                "arguments": {
                                    "metadata": {OUTPUT_LANGUAGE_METADATA_KEY: "French"}
                                }
                            }
                        ]
                    }
                }
            },
        },
    }

    reset_output_language_to_request_context(checkpoint)

    assert OUTPUT_LANGUAGE_METADATA_KEY not in checkpoint["context"]["metadata"]
    assert OUTPUT_LANGUAGE_SOURCE_METADATA_KEY not in checkpoint["context"]["metadata"]
    restored_child = checkpoint["pattern_state"]["active_step_contexts"]["step_1"]
    assert OUTPUT_LANGUAGE_METADATA_KEY not in restored_child["metadata"]
    assert OUTPUT_LANGUAGE_SOURCE_METADATA_KEY not in restored_child["metadata"]

    message_metadata = checkpoint["context"]["messages"][0]["metadata"]
    assert message_metadata[OUTPUT_LANGUAGE_METADATA_KEY] == "French"
    assert checkpoint["metadata"][OUTPUT_LANGUAGE_METADATA_KEY] == "French"
    step_result = checkpoint["pattern_state"]["step_results"]["step_1"]
    assert step_result["metadata"][OUTPUT_LANGUAGE_METADATA_KEY] == "French"
    tool_arguments = checkpoint["pattern_state"]["active_step_pattern_states"][
        "step_1"
    ]["last_response"]["tool_calls"][0]["arguments"]
    assert tool_arguments["metadata"][OUTPUT_LANGUAGE_METADATA_KEY] == "French"


def test_resume_migration_reaches_a_nested_auto_pattern_child_context() -> None:
    child = ExecutionContext(execution_id="exec-migration-nested_child")
    child.metadata[OUTPUT_LANGUAGE_METADATA_KEY] = "Simplified Chinese"
    child.metadata[OUTPUT_LANGUAGE_SOURCE_METADATA_KEY] = "auto_router"
    checkpoint = {
        "context": ExecutionContext(execution_id="exec-migration-nested").to_dict(),
        "pattern_state": {
            "dag_state": {"active_step_contexts": {"step_1": child.to_dict()}}
        },
    }

    reset_output_language_to_request_context(checkpoint)

    nested = checkpoint["pattern_state"]["dag_state"]["active_step_contexts"]["step_1"]
    assert OUTPUT_LANGUAGE_METADATA_KEY not in nested["metadata"]


@pytest.mark.parametrize(
    "client_value, engine_value",
    [(True, False), ("true", False), (1, False), (False, True)],
    ids=["client_true", "client_string", "client_one", "client_false_on_latched_run"],
)
@pytest.mark.parametrize("surface", ["top_level", "request_context"])
def test_a_client_cannot_set_the_tool_evidence_marker(
    tmp_path: Path, surface: str, client_value: object, engine_value: bool
) -> None:
    """Both surfaces that carry client input into metadata refuse this key.

    Each cell sends the opposite of what the engine currently holds, so a cell
    goes red if the client's value lands -- including the dangerous direction,
    a client sending False to clear a run that really did lose observations.
    """
    runner = AgentRunner(
        agent=Agent(name="writer", patterns=[StatefulPattern()]),
        workspace_manager=FakeWorkspaceManager(tmp_path),
    )
    context = ContextManager().create_context(execution_id="exec-reserved-key")
    context.metadata[TOOL_EVIDENCE_REMOVED_METADATA_KEY] = engine_value
    client_keys = {
        TOOL_EVIDENCE_REMOVED_METADATA_KEY: client_value,
        "some_client_key": "kept",
    }
    metadata = (
        dict(client_keys)
        if surface == "top_level"
        else {"request_context": dict(client_keys)}
    )

    runner._merge_context_metadata(context, metadata)

    assert context.metadata[TOOL_EVIDENCE_REMOVED_METADATA_KEY] is engine_value
    # Proves the merge actually ran; without it, the line above could pass
    # simply because nothing was merged at all.
    assert context.metadata["some_client_key"] == "kept"


def test_a_restored_context_takes_no_client_metadata_at_all(tmp_path: Path) -> None:
    """The restore branch returns before both filters because it merges nothing.

    A run rebuilt from a checkpoint keeps what it latched: the current turn's
    metadata contributes only the modality preference, so neither surface that
    carries client input reaches it.
    """
    runner = AgentRunner(
        agent=Agent(name="writer", patterns=[StatefulPattern()]),
        workspace_manager=FakeWorkspaceManager(tmp_path),
    )
    context = ContextManager().create_context(execution_id="exec-restored-key")
    context.metadata[TOOL_EVIDENCE_REMOVED_METADATA_KEY] = True
    client_keys = {
        TOOL_EVIDENCE_REMOVED_METADATA_KEY: False,
        "some_client_key": "kept",
    }

    runner._merge_context_metadata(
        context,
        {**client_keys, "request_context": dict(client_keys)},
        restored=True,
    )

    assert context.metadata[TOOL_EVIDENCE_REMOVED_METADATA_KEY] is True
    assert "some_client_key" not in context.metadata
    assert "request_context" not in context.metadata


def test_a_restored_context_with_no_marker_key_is_never_backfilled(
    tmp_path: Path,
) -> None:
    """A checkpoint written before this key existed must stay keyless on resume.

    Backfilling either value here would erase the distinction the third state
    exists to carry: False would tell a run that really did lose observations
    that nothing was removed, and True would tell a run that lost nothing
    that something was. The restore branch returns before either client-input
    filter runs, so nothing it does can write this key in either direction.
    """
    runner = AgentRunner(
        agent=Agent(name="writer", patterns=[StatefulPattern()]),
        workspace_manager=FakeWorkspaceManager(tmp_path),
    )
    context = ContextManager().create_context(execution_id="exec-restored-no-key")
    context.metadata.pop(TOOL_EVIDENCE_REMOVED_METADATA_KEY, None)
    client_keys = {
        TOOL_EVIDENCE_REMOVED_METADATA_KEY: True,
        "some_client_key": "kept",
    }

    runner._merge_context_metadata(
        context,
        {**client_keys, "request_context": dict(client_keys)},
        restored=True,
    )

    assert TOOL_EVIDENCE_REMOVED_METADATA_KEY not in context.metadata
    assert tool_evidence_state(context) == "unknown"


@pytest.mark.parametrize("stored", [True, False], ids=["removed", "intact"])
def test_a_restored_context_with_a_marker_key_keeps_its_stored_value(
    tmp_path: Path, stored: bool
) -> None:
    """A checkpoint that does carry the key is never recomputed on restore."""
    runner = AgentRunner(
        agent=Agent(name="writer", patterns=[StatefulPattern()]),
        workspace_manager=FakeWorkspaceManager(tmp_path),
    )
    context = ContextManager().create_context(execution_id="exec-restored-has-key")
    context.metadata[TOOL_EVIDENCE_REMOVED_METADATA_KEY] = stored
    client_keys = {
        TOOL_EVIDENCE_REMOVED_METADATA_KEY: not stored,
        "some_client_key": "kept",
    }

    runner._merge_context_metadata(
        context,
        {**client_keys, "request_context": dict(client_keys)},
        restored=True,
    )

    assert context.metadata[TOOL_EVIDENCE_REMOVED_METADATA_KEY] is stored
    assert tool_evidence_state(context) == ("removed" if stored else "intact")


def test_the_marker_is_stamped_before_any_client_metadata_is_merged(
    tmp_path: Path,
) -> None:
    """The refusal must not depend on the stamp happening to win a race.

    ``create_context`` writes False before the merge runs, so the ordering is
    asserted here and a later refactor that moves the stamp cannot pass quietly.
    """
    runner = AgentRunner(
        agent=Agent(name="writer", patterns=[StatefulPattern()]),
        workspace_manager=FakeWorkspaceManager(tmp_path),
    )
    context = ContextManager().create_context(execution_id="exec-reserved-order")
    seen: list[object] = []
    original = runner._apply_request_context

    def record(ctx: ExecutionContext, request_context: dict[str, Any]) -> None:
        seen.append(ctx.metadata.get(TOOL_EVIDENCE_REMOVED_METADATA_KEY))
        original(ctx, request_context)

    runner._apply_request_context = record  # type: ignore[method-assign]
    runner._merge_context_metadata(context, {"request_context": {"a": 1}})
    assert seen == [False]
