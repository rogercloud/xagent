"""A run whose model provider stayed unavailable, or whose model kept producing
unusable output, rests PAUSED instead of FAILED.

``llm_unavailable`` follows ``XAGENT_TASK_INFRA_FAILURE_PAUSE_ENABLED`` like
``persistence_failure``; ``model_output_invalid`` also needs
``XAGENT_TASK_AUTO_RESUME_ENABLED``. Both reach settlement as an unsuccessful
result, so the result paths carry them: a new run's
(``_finalize_task_execution_result_isolated``) and a resumed run's
(``_finalize_resumed_task``). Runs on SQLite and, when
``XAGENT_TEST_POSTGRES_URL`` is set, on PostgreSQL through the shared
``canonical`` fixture.
"""

from __future__ import annotations

from typing import Any

import pytest

from tests.core.agent.test_auto import decision_tool_response, plan_tool_response
from tests.core.agent.test_react import FakeLLM, FakeTool
from tests.core.agent.test_runner import FakeWorkspaceManager
from tests.web.services.test_execution_event_recovery import tracer_for
from tests.web.services.test_persistence_failure_pause import (
    _assert_no_failure_in_model_context,
    _assert_paused,
    _assistant_lines,
    _break_decision,
    _checkpoint,
    _expire_and_recover,
    _finalize,
    _finalize_and_release,
    _finalize_resumed,
)
from tests.web.services.test_persistence_failure_pause import (
    _late_bound_sessions as _late_bound_sessions_fixture,
)
from tests.web.services.test_persistence_failure_pause import (
    _prepare_run,
    _projected_task,
    _projection,
    _schedule_failing_turn,
    _settled_facts,
    _start_run,
    _state,
    _without_interruption,
)
from tests.web.services.test_task_execution_event_writer import (
    canonical as canonical_fixture,
)
from tests.web.services.test_task_execution_event_writer import engine as engine_fixture
from tests.web.services.test_task_execution_event_writer import (
    task_id as task_id_fixture,
)
from tests.web.services.test_task_lease_expiry_interruption import (
    _fail_recording,
)
from xagent.core.agent import (
    Agent,
    AgentRunner,
    AutoPattern,
    ReActPattern,
)
from xagent.core.agent.checkpoint import (
    TraceCheckpointStore,
)
from xagent.core.agent.interruption import InterruptionReason
from xagent.core.agent.pattern.auto.auto import DECISION_TOOL_NAME
from xagent.core.agent.service import AgentService
from xagent.core.model.chat.error import retry_on
from xagent.core.model.chat.exceptions import LLMEmptyContentError, LLMTimeoutError
from xagent.core.retry import RetryWrapper
from xagent.core.retry.strategy import FixedDelay
from xagent.web.models.task import Task, TaskStatus
from xagent.web.services import task_auto_recovery, task_execution
from xagent.web.services.task_auto_recovery import (
    TASK_INTERRUPTION_PAUSED_TRIGGER_ERROR,
    InterruptionSettlementDeferred,
    settlement_interruption_for_failure,
    settlement_interruption_for_result,
    settlement_pause_enabled,
)
from xagent.web.services.task_execution import _acquire_resume_task_lease
from xagent.web.services.task_lease_recovery import TASK_LEASE_PAUSED_TRIGGER_ERROR
from xagent.web.services.task_lease_service import (
    TASK_UNKNOWN_TOOL_EFFECT_SETTLEMENT_ERROR,
    CheckpointRecoveryResolution,
    CheckpointRecoveryVerdict,
    TaskLease,
    bind_task_lease_context,
)
from xagent.web.services.task_orchestrator import (
    settle_task_lease_isolated,
)

canonical = canonical_fixture
engine = engine_fixture
task_id = task_id_fixture
_late_bound_sessions = _late_bound_sessions_fixture

PAUSE_SWITCH = "XAGENT_TASK_INFRA_FAILURE_PAUSE_ENABLED"
AUTO_SWITCH = "XAGENT_TASK_AUTO_RESUME_ENABLED"
PAUSED_FACT = {"status": "paused", "result": {"error": None}}


@pytest.fixture(autouse=True)
def _switch_defaults(monkeypatch):
    monkeypatch.delenv(PAUSE_SWITCH, raising=False)
    monkeypatch.delenv(AUTO_SWITCH, raising=False)


# ------------------------------------------------- real runs, then resumed


def _calculator_call() -> dict[str, Any]:
    return {
        "content": "calculate",
        "tool_calls": [
            {
                "id": "call1",
                "function": {
                    "name": "calculator",
                    "arguments": '{"expression":"2+2"}',
                },
            }
        ],
    }


def _empty_final_answer() -> dict[str, Any]:
    """A ``final_answer`` call without an answer: twice ends the run."""
    return {
        "tool_calls": [
            {
                "id": "call_empty",
                "function": {"name": "final_answer", "arguments": '{"answer": ""}'},
            }
        ]
    }


def _final_answer(answer: str) -> dict[str, Any]:
    return {
        "tool_calls": [
            {
                "id": "call_final",
                "function": {
                    "name": "final_answer",
                    "arguments": '{"answer": "%s"}' % answer,
                },
            }
        ]
    }


class _UnavailableProvider:
    """A provider that keeps timing out, behind the real retry loop.

    Each ``chat`` call retries the request ``attempts`` times through
    ``RetryWrapper`` with the production retry predicate, then raises what
    the last attempt raised -- what a pattern sees once retries exhaust.
    """

    def __init__(self, before: list[Any], *, attempts: int = 3) -> None:
        self.before = list(before)
        self.attempts = attempts
        self.provider_requests = 0
        self.calls: list[dict[str, Any]] = []

    async def chat(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        if self.before:
            return self.before.pop(0)
        provider = self

        class _Target:
            def invoke(self, *_args: Any, **_kwargs: Any) -> Any:
                raise AssertionError("async only")

            async def ainvoke(self, *_args: Any, **_kwargs: Any) -> Any:
                provider.provider_requests += 1
                raise LLMTimeoutError("provider timed out")

        return await RetryWrapper(
            _Target(),
            strategy=FixedDelay(0),
            max_retries=self.attempts,
            retry_on=retry_on,
        ).ainvoke()


def _runner(tid: int, pattern: Any, llm: Any, tool: FakeTool, tmp_path) -> AgentRunner:
    return AgentRunner(
        agent=Agent(name="provider", patterns=[pattern], tools=[tool], llm=llm),
        tracer=TraceCheckpointStore(tracer_for(tid)),
        workspace_manager=FakeWorkspaceManager(tmp_path),
    )


def _service_result(result: dict[str, Any]) -> dict[str, Any]:
    """The top-level fields ``AgentExecutionAdapter`` forwards to settlement."""
    return {
        key: result[key]
        for key in ("success", "status", "error", "output", "interruption_reason")
        if key in result
    }


def _react() -> ReActPattern:
    return ReActPattern(max_iterations=4)


def _auto() -> AutoPattern:
    return AutoPattern(react_pattern=_react())


async def _resume(factory, tid: int, lease: TaskLease, pattern, llm, tool, tmp_path):
    """Manual resume: a resume lease on the same run, then the runner."""
    with factory() as db:
        user_id = int(db.get(Task, tid).user_id)
    resume_lease = _acquire_resume_task_lease(
        tid, user_id, lease.run_id, refuse_terminal_status=True
    )
    assert resume_lease is not None and resume_lease.run_id == lease.run_id
    with bind_task_lease_context(resume_lease):
        return await _runner(tid, pattern, llm, tool, tmp_path).resume(
            str(tid), task="2+2"
        )


def _settle_new_run(factory, tid: int, lease: TaskLease, result: dict[str, Any]):
    finalized = _finalize(factory, tid, lease, _service_result(result))
    # The scheduler's settlement releases the paused lease.
    assert settle_task_lease_isolated(lease)
    return finalized


@pytest.mark.asyncio
@pytest.mark.parametrize("wrapped", ["react", "auto"])
async def test_invalid_tool_protocol_resume_samples_the_model_again(
    canonical, tmp_path, wrapped
):
    """The run gives up on the tool protocol after a tool call; resuming the
    paused run continues from its checkpoint with a NEW model call -- it does
    not replay the cached failure (Auto caches the child's result in its
    ``auto_after_child`` checkpoint) and does not re-run the tool."""

    factory, tid = canonical
    lease = _start_run(factory, tid)
    tool = FakeTool()
    decision = [decision_tool_response("react", "Needs a tool.")]
    first_llm = FakeLLM(
        responses=[
            *(decision if wrapped == "auto" else []),
            _calculator_call(),
            _empty_final_answer(),
            _empty_final_answer(),
        ]
    )
    pattern = _auto() if wrapped == "auto" else _react()
    with bind_task_lease_context(lease):
        result = await _runner(tid, pattern, first_llm, tool, tmp_path).run(
            task="2+2", execution_id=str(tid)
        )
    assert result["status"] == "invalid_tool_protocol"
    assert result["interruption_reason"] == "model_output_invalid"
    assert len(tool.calls) == 1
    # The checkpoint a resume loads: Auto's carries the cached child result.
    with bind_task_lease_context(lease):
        latest = await TraceCheckpointStore(tracer_for(tid)).load_latest_checkpoint(
            str(tid)
        )
    if wrapped == "auto":
        assert latest["label"] == "auto_after_child"
        cached = latest["pattern_state"]["last_result"]
        assert (cached["success"], cached["status"]) == (
            False,
            "invalid_tool_protocol",
        )
    else:
        assert latest["label"] == "invalid_tool_protocol"

    finalized = _settle_new_run(factory, tid, lease, result)

    assert finalized.interruption_pause_reason is (
        InterruptionReason.MODEL_OUTPUT_INVALID
    )
    _assert_paused(factory, tid, lease, reason="model_output_invalid")
    assert _settled_facts(factory, tid) == [PAUSED_FACT]
    assert _assistant_lines(factory, tid) == []
    _assert_no_failure_in_model_context(factory, tid)

    resume_llm = FakeLLM(responses=[_final_answer("4")])
    resumed = await _resume(
        factory,
        tid,
        lease,
        _auto() if wrapped == "auto" else _react(),
        resume_llm,
        tool,
        tmp_path,
    )

    assert resumed["success"], resumed
    assert resumed["output"] == "4"
    # One new ReAct call: no replayed failure and no fresh routing decision.
    assert len(resume_llm.calls) == 1
    offered = {schema["function"]["name"] for schema in resume_llm.calls[0]["tools"]}
    assert "final_answer" in offered and DECISION_TOOL_NAME not in offered
    assert len(tool.calls) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("wrapped", ["react", "auto"])
async def test_unavailable_provider_pauses_and_the_run_resumes(
    canonical, tmp_path, wrapped
):
    """The provider times out until the retry loop gives up; the run pauses
    as ``llm_unavailable`` and a manual resume against a healthy provider
    finishes it from its checkpoint without re-running the tool."""

    factory, tid = canonical
    lease = _start_run(factory, tid)
    tool = FakeTool()
    decision = [decision_tool_response("react", "Needs a tool.")]
    provider = _UnavailableProvider(
        [*(decision if wrapped == "auto" else []), _calculator_call()]
    )
    pattern = _auto() if wrapped == "auto" else _react()
    with bind_task_lease_context(lease):
        result = await _runner(tid, pattern, provider, tool, tmp_path).run(
            task="2+2", execution_id=str(tid)
        )
    assert result["success"] is False
    assert result["interruption_reason"] == "llm_unavailable"
    assert provider.provider_requests == provider.attempts
    assert len(tool.calls) == 1

    finalized = _settle_new_run(factory, tid, lease, result)

    assert finalized.interruption_pause_reason is InterruptionReason.LLM_UNAVAILABLE
    row = _assert_paused(factory, tid, lease, reason="llm_unavailable")
    assert row.last_error == result["error"]
    assert _settled_facts(factory, tid) == [PAUSED_FACT]
    _assert_no_failure_in_model_context(factory, tid)

    healthy = FakeLLM(responses=[_final_answer("4")])
    resumed = await _resume(
        factory,
        tid,
        lease,
        _auto() if wrapped == "auto" else _react(),
        healthy,
        tool,
        tmp_path,
    )

    assert resumed["success"], resumed
    assert resumed["output"] == "4"
    assert len(healthy.calls) == 1
    assert len(tool.calls) == 1


# ------------------------------------------------------ DAG runs never pause


class _ProviderGoesDown:
    """Answers its scripted responses, then times out on every call."""

    model_name = "fake-llm"

    def __init__(self, responses: list[Any]) -> None:
        self.responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    async def chat(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        if self.responses:
            return self.responses.pop(0)
        raise LLMTimeoutError("provider timed out")


@pytest.mark.asyncio
@pytest.mark.parametrize("pattern", ["dag_plan_execute", "auto"])
@pytest.mark.parametrize("failing", ["planning", "step"])
async def test_dag_runs_never_pause_for_provider_failures(canonical, pattern, failing):
    """A DAG run (think mode, or Auto routed to it) swallows its provider
    failures into an ordinary failed result without an interruption reason,
    so settlement fails it as before (v1 leaves DAG LLM failures out)."""

    llm = _ProviderGoesDown(
        [
            *(
                [decision_tool_response("plan_execute", "Plan it.")]
                if pattern == "auto"
                else []
            ),
            *(
                [plan_tool_response([{"id": "answer", "task": "Answer directly"}])]
                if failing == "step"
                else []
            ),
        ]
    )
    service = AgentService(
        name=f"dag-{pattern}-{failing}",
        id=f"dag-{pattern}-{failing}",
        pattern=pattern,
        llm=llm,
        tools=[FakeTool()],
        tool_config=None,
    )
    service.allowed_skills = []

    result = await service.execute_task("Plan then answer", task_id="dag-run")

    assert result["success"] is False
    assert result["agent_result"]["failure_reason"] == (
        "step_failed" if failing == "step" else "plan_generation_error"
    )
    assert "interruption_reason" not in result
    assert settlement_interruption_for_result(result) is None

    factory, tid = canonical
    lease = _start_run(factory, tid)
    _checkpoint(factory, tid, lease.run_id)
    settled = {
        key: value
        for key, value in result.items()
        if key in {"success", "status", "error", "output"}
    }
    finalized = _settle_new_run(factory, tid, lease, settled)

    assert finalized.interruption_pause_reason is None
    task, row, _events = _state(factory, tid)
    assert task.status == TaskStatus.FAILED
    assert row is None


# --------------------------------------------------------- switches, unit

LLM_RESULT = {
    "success": False,
    "status": "failed",
    "output": "All patterns failed",
    "error": "All 1 patterns failed or returned unsuccessful results.",
    "interruption_reason": "llm_unavailable",
}
MODEL_RESULT = {
    "success": False,
    "status": "invalid_tool_protocol",
    "output": "The model returned an invalid tool protocol response.",
    "error": (
        "The model returned an invalid tool protocol response after one repair attempt."
    ),
    "interruption_reason": "model_output_invalid",
}
RESULTS = {"llm_unavailable": LLM_RESULT, "model_output_invalid": MODEL_RESULT}


def test_settlement_acts_on_provider_and_model_output_failures():
    assert settlement_interruption_for_result(LLM_RESULT) is (
        InterruptionReason.LLM_UNAVAILABLE
    )
    assert settlement_interruption_for_result(MODEL_RESULT) is (
        InterruptionReason.MODEL_OUTPUT_INVALID
    )
    # The status alone names unusable output, with or without the runner's key.
    bare = {k: v for k, v in MODEL_RESULT.items() if k != "interruption_reason"}
    assert settlement_interruption_for_result(bare) is (
        InterruptionReason.MODEL_OUTPUT_INVALID
    )
    assert settlement_interruption_for_failure(LLMTimeoutError("timed out")) is (
        InterruptionReason.LLM_UNAVAILABLE
    )
    assert settlement_interruption_for_failure(LLMEmptyContentError("empty")) is (
        InterruptionReason.MODEL_OUTPUT_INVALID
    )
    # Quota wins over either.
    for result in (LLM_RESULT, MODEL_RESULT):
        assert (
            settlement_interruption_for_result({**result, "status": "quota_exceeded"})
            is None
        )
    # Still never: a lease expiry is TTL recovery's, not a settlement's.
    assert (
        settlement_interruption_for_result(
            {"success": False, "interruption_reason": "lease_expired"}
        )
        is None
    )


@pytest.mark.parametrize(
    ("infra", "auto", "enabled"),
    [
        ("true", "true", {"persistence_failure", "llm_unavailable", "model"}),
        ("true", "false", {"persistence_failure", "llm_unavailable"}),
        ("false", "true", set()),
        ("false", "false", set()),
    ],
)
def test_settlement_switches_per_reason(monkeypatch, infra, auto, enabled):
    monkeypatch.setenv(PAUSE_SWITCH, infra)
    monkeypatch.setenv(AUTO_SWITCH, auto)
    names = {
        "persistence_failure": InterruptionReason.PERSISTENCE_FAILURE,
        "llm_unavailable": InterruptionReason.LLM_UNAVAILABLE,
        "model": InterruptionReason.MODEL_OUTPUT_INVALID,
        "user_pause": InterruptionReason.USER_PAUSE,
    }
    assert {
        name for name, reason in names.items() if settlement_pause_enabled(reason)
    } == enabled | ({"user_pause"} if infra == "true" else set())


# --------------------------------------------- result paths: S1 and S3

_PATH_REASONS = [
    (path, reason) for path in ("result", "resumed_result") for reason in RESULTS
]
CASES = pytest.mark.parametrize(("path", "reason"), _PATH_REASONS)


def _settle_result(path: str, factory, ids, lease, result) -> None:
    if path == "result":
        _finalize_and_release(factory, ids, lease, dict(result))
    else:
        assert _finalize_resumed(ids, lease, result)["lease_released"]


def _without_result_interruption(m: pytest.MonkeyPatch) -> None:
    """Settle as before this series: no result path sees an interruption."""
    m.setattr(task_execution, "settlement_interruption_for_result", lambda _r: None)
    m.setattr(task_auto_recovery, "settlement_interruption_for_result", lambda _r: None)


def _without_settled_reason(projection: dict[str, Any], reason: str):
    """A quota stop's settled fact keeps the result, reason included."""
    for fact in projection["settled"]:
        assert fact["result"].pop("interruption_reason") == reason
    return projection


@CASES
def test_pause_projects_like_lease_recovery(canonical, path, reason):
    factory, _tid = canonical
    paused, recovered = _projected_task(factory), _projected_task(factory)
    paused_lease = _prepare_run(factory, paused)
    recovered_lease = _prepare_run(factory, recovered)

    _settle_result(path, factory, paused, paused_lease, RESULTS[reason])
    assert _expire_and_recover(factory, recovered) == TaskStatus.PAUSED

    left = _projection(factory, paused, paused_lease)
    right = _projection(factory, recovered, recovered_lease)
    assert left.pop("trigger_run")[1] == TASK_INTERRUPTION_PAUSED_TRIGGER_ERROR
    assert right.pop("trigger_run")[1] == TASK_LEASE_PAUSED_TRIGGER_ERROR
    # Only lease recovery reconciles orphaned delivery rows.
    left.pop("delivery")
    right.pop("delivery")
    assert left == right
    assert left["status"] == TaskStatus.PAUSED
    assert left["output"] == "previous answer"
    assert left["assistant"] == []
    assert left["settled"] == [PAUSED_FACT]
    task, row, events = _state(factory, paused["task"])
    assert (row.reason, row.state) == (reason, "manual")
    assert row.paused_state_version == task.state_version
    assert row.last_error == RESULTS[reason]["error"]
    assert [(e.event, e.detail["task_status"]) for e in events] == [
        ("interrupted", "paused")
    ]
    _assert_no_failure_in_model_context(factory, paused["task"])


def _switch_case(m: pytest.MonkeyPatch, case: str) -> str:
    """Apply ``case``; return the task source it settles."""
    if case == "infra_off":
        m.setenv(PAUSE_SWITCH, "false")
    elif case == "auto_resume_off":
        m.setenv(AUTO_SWITCH, "false")
    return "sdk" if case == "ineligible" else "trigger"


@pytest.mark.parametrize(
    ("path", "reason", "case"),
    [
        (path, reason, case)
        for path, reason in _PATH_REASONS
        for case in ("infra_off", "auto_resume_off", "ineligible")
        # Automatic resume gates model_output_invalid only.
        if case != "auto_resume_off" or reason == "model_output_invalid"
    ],
)
def test_unpaused_runs_settle_exactly_as_before(
    canonical, monkeypatch, path, reason, case
):
    """A switch the reason depends on is off, or the task is ineligible: the
    same FAILED settlement as without an interruption, plus a ``disabled``
    (or ``ineligible``) row. Auto-resume gates ``model_output_invalid`` only."""

    source = _switch_case(monkeypatch, case)
    factory, _tid = canonical
    gated = _projected_task(factory, source=source)
    baseline = _projected_task(factory, source=source)
    gated_lease = _prepare_run(factory, gated)
    baseline_lease = _prepare_run(factory, baseline)

    _settle_result(path, factory, gated, gated_lease, RESULTS[reason])
    with monkeypatch.context() as m:
        _without_result_interruption(m)
        _settle_result(path, factory, baseline, baseline_lease, RESULTS[reason])

    gated_projection = _projection(factory, gated, gated_lease)
    assert gated_projection == _projection(factory, baseline, baseline_lease)
    assert gated_projection["status"] == TaskStatus.FAILED
    _task, row, events = _state(factory, gated["task"])
    expected = "ineligible" if case == "ineligible" else "disabled"
    assert (row.reason, row.state) == (reason, expected)
    assert [e.detail["task_status"] for e in events] == ["failed"]
    assert _state(factory, baseline["task"])[1] is None


@pytest.mark.parametrize(
    ("path", "reason"),
    [case for case in _PATH_REASONS if case[1] != "model_output_invalid"],
)
def test_auto_resume_switch_gates_only_model_output(
    canonical, monkeypatch, path, reason
):
    """With automatic resume off, the infrastructure reasons still pause:
    the task can be resumed by hand."""

    monkeypatch.setenv(AUTO_SWITCH, "false")
    factory, _tid = canonical
    ids = _projected_task(factory)
    lease = _prepare_run(factory, ids)

    _settle_result(path, factory, ids, lease, RESULTS[reason])

    task, row, _events = _state(factory, ids["task"])
    assert (task.status, task.control_state) == (TaskStatus.PAUSED, "paused")
    assert (row.reason, row.state) == (reason, "manual")


@CASES
@pytest.mark.parametrize(
    "case",
    ["unknown_tool_effect", "not_recoverable", "pause_requested", "resume_requested"],
)
def test_paths_decide_every_verdict(canonical, monkeypatch, path, reason, case):
    factory, _tid = canonical
    ids = _projected_task(factory)
    lease = _prepare_run(factory, ids, checkpoint=case != "not_recoverable")
    if case == "unknown_tool_effect":
        monkeypatch.setattr(
            task_auto_recovery,
            "resolve_checkpoint_recovery_with_data",
            lambda _db, _candidate: CheckpointRecoveryResolution(
                CheckpointRecoveryVerdict.UNKNOWN_TOOL_EFFECT
            ),
        )
    if case in {"pause_requested", "resume_requested"}:
        with factory() as db:
            db.get(Task, ids["task"]).control_state = case
            db.commit()

    _settle_result(path, factory, ids, lease, RESULTS[reason])

    task, row, _events = _state(factory, ids["task"])
    projection = _projection(factory, ids, lease)
    if case in {"unknown_tool_effect", "not_recoverable"}:
        assert task.status == TaskStatus.FAILED
        assert task.error_message == (
            TASK_UNKNOWN_TOOL_EFFECT_SETTLEMENT_ERROR
            if case == "unknown_tool_effect"
            else RESULTS[reason]["error"]
        )
        assert (row.reason, row.state) == (case, "manual")
        assert [fact["status"] for fact in projection["settled"]] == ["failed"]
    else:
        assert (task.status, task.control_state) == (TaskStatus.PAUSED, "paused")
        assert task.runner_id is None
        assert (row.reason, row.state) == (
            "user_pause" if case == "pause_requested" else reason,
            "manual",
        )
        assert projection["settled"] == [PAUSED_FACT]
        assert projection["assistant"] == []


@CASES
@pytest.mark.parametrize("switch", ["on", "off"])
def test_quota_stop_wins_over_the_reason(canonical, monkeypatch, path, reason, switch):
    if switch == "off":
        monkeypatch.setenv(PAUSE_SWITCH, "false")
    factory, _tid = canonical
    quota = {**RESULTS[reason], "status": "quota_exceeded", "error_code": "q"}
    plain_quota = {k: v for k, v in quota.items() if k != "interruption_reason"}
    gated, baseline = _projected_task(factory), _projected_task(factory)
    gated_lease = _prepare_run(factory, gated)
    baseline_lease = _prepare_run(factory, baseline)

    _settle_result(path, factory, gated, gated_lease, quota)
    _settle_result(path, factory, baseline, baseline_lease, plain_quota)

    gated_projection = _without_settled_reason(
        _projection(factory, gated, gated_lease), reason
    )
    assert gated_projection == _projection(factory, baseline, baseline_lease)
    assert gated_projection["status"] == TaskStatus.FAILED
    assert _state(factory, gated["task"])[1] is None


@CASES
@pytest.mark.parametrize("failure", ["python", "database", "marker"])
def test_recording_failure_does_not_change_the_pause(
    canonical, monkeypatch, path, reason, failure
):
    factory, _tid = canonical
    recorded, unrecorded = _projected_task(factory), _projected_task(factory)
    recorded_lease = _prepare_run(factory, recorded)
    unrecorded_lease = _prepare_run(factory, unrecorded)

    _settle_result(path, factory, recorded, recorded_lease, RESULTS[reason])
    with monkeypatch.context() as m:
        _fail_recording(m, failure)
        _settle_result(path, factory, unrecorded, unrecorded_lease, RESULTS[reason])

    left = _projection(factory, recorded, recorded_lease)
    assert left == _projection(factory, unrecorded, unrecorded_lease)
    assert left["status"] == TaskStatus.PAUSED
    assert left["trigger_run"][1] == TASK_INTERRUPTION_PAUSED_TRIGGER_ERROR
    assert _state(factory, unrecorded["task"])[1:] == (None, [])


@CASES
@pytest.mark.parametrize("failure", ["eligibility", "checkpoint_read"])
def test_undecidable_interruption_keeps_the_lease_for_ttl(
    canonical, monkeypatch, path, reason, failure
):
    """Any error deciding the interruption defers it: nothing settles, the run
    stays RUNNING under its lease, and TTL recovery pauses it later."""

    factory, _tid = canonical
    ids = _projected_task(factory)
    lease = _prepare_run(factory, ids)

    with monkeypatch.context() as m:
        _break_decision(m, failure)
        with pytest.raises(InterruptionSettlementDeferred) as deferred:
            if path == "result":
                _finalize(factory, ids["task"], lease, dict(RESULTS[reason]))
            else:
                _finalize_resumed(ids, lease, RESULTS[reason])
        assert deferred.value.__cause__ is not None

    task, row, _events = _state(factory, ids["task"])
    assert task.status == TaskStatus.RUNNING
    assert (task.runner_id, task.run_id) == (lease.runner_id, lease.run_id)
    assert row is None
    assert _projection(factory, ids, lease)["settled"] == []

    assert _expire_and_recover(factory, ids) == TaskStatus.PAUSED
    _task, row, _events = _state(factory, ids["task"])
    assert (row.reason, row.state) == ("lease_expired", "manual")


# ------------------------------------------- S2: the rare raised provider error


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("error", "reason"),
    [
        (LLMTimeoutError("provider timed out"), "llm_unavailable"),
        (LLMEmptyContentError("empty response"), "model_output_invalid"),
    ],
)
async def test_raised_provider_failure_pauses_on_the_exception_path(
    canonical, error, reason
):
    """The runner turns pattern failures into results, so these rarely raise
    past it; when one does, the exception path decides it the same way."""

    factory, _tid = canonical
    ids = _projected_task(factory)
    lease = _prepare_run(factory, ids)

    frames = await _schedule_failing_turn(factory, ids, lease, error)

    assert frames == ["task_paused"]
    task, row, _events = _state(factory, ids["task"])
    assert (task.status, task.control_state) == (TaskStatus.PAUSED, "paused")
    assert (row.reason, row.state) == (reason, "manual")


@pytest.mark.asyncio
async def test_raised_model_output_failure_without_auto_resume_fails_as_before(
    canonical, monkeypatch
):
    monkeypatch.setenv(AUTO_SWITCH, "false")
    factory, _tid = canonical
    gated, baseline = _projected_task(factory), _projected_task(factory)
    gated_lease = _prepare_run(factory, gated)
    baseline_lease = _prepare_run(factory, baseline)

    gated_frames = await _schedule_failing_turn(
        factory, gated, gated_lease, LLMEmptyContentError("empty response")
    )
    with monkeypatch.context() as m:
        _without_interruption(m)
        baseline_frames = await _schedule_failing_turn(
            factory, baseline, baseline_lease, LLMEmptyContentError("empty response")
        )

    gated_projection = _projection(factory, gated, gated_lease)
    assert gated_projection == _projection(factory, baseline, baseline_lease)
    assert gated_projection["status"] == TaskStatus.FAILED
    assert gated_frames == baseline_frames == ["task_error"]
    _task, row, _events = _state(factory, gated["task"])
    assert (row.reason, row.state) == ("model_output_invalid", "disabled")
