"""Automatic resume after model provider faults, through real hosts.

The hosts' scripted provider fails on demand (see ``MODEL_FAULT_FILE``): a
real ``openai.InternalServerError`` (503) for an outage, or an empty
``final_answer`` for unusable output. Faults are consumed per provider
attempt, so the moment the provider "recovers" is fixed by call count, never
by the clock. The production tier-1 retry layer sits in front of the provider
with exactly three attempts per model call, so an outage pauses the run only
after that layer gave up, as it would in production.

Backoff and polling are shortened so each resume is due within a few seconds.
"""

from __future__ import annotations

import json
import time

import pytest

from tests.e2e.shared_execution_harness import (
    SETTLE_TIMEOUT,
    answered_model_calls,
    assistant_messages,
    auto_resume_commands,
    clear_model_fault,
    collect_events,
    model_faults,
    receive_event,
    recovery_state,
    set_model_fault,
    start_task,
    task_detail,
    task_state,
    wait_for,
    ws_url,
)

pytestmark = pytest.mark.e2e

# Provider attempts per model call (E2E_MODEL_RETRY_ATTEMPTS).
ATTEMPTS = 3
# Longer than any resume this module can schedule: an LLM backoff of at most
# 2s plus 1s jitter and a 1s sweeper poll, with a second to spare.
LLM_QUIET_SECONDS = 5
# Unusable output is resumed after up to 5s of jitter, then a 1s poll.
MODEL_OUTPUT_QUIET_SECONDS = 7
ANSWER = "Shared E2E answer"
TWO_RESUMES = [
    "interrupted",
    "scheduled",
    "auto_resumed",
    "interrupted",
    "scheduled",
    "auto_resumed",
]


@pytest.fixture(autouse=True)
def _fast_auto_resume(monkeypatch):
    # Read by the hosts the shared_app fixture spawns after this.
    monkeypatch.setenv("XAGENT_TASK_AUTO_RESUME_POLL_SECONDS", "1")
    monkeypatch.setenv("XAGENT_TASK_AUTO_RESUME_SHORT_INITIAL_BACKOFF_SECONDS", "1")
    monkeypatch.setenv("XAGENT_TASK_AUTO_RESUME_LLM_INITIAL_BACKOFF_SECONDS", "1")
    monkeypatch.setenv("XAGENT_TASK_AUTO_RESUME_LLM_MAX_BACKOFF_SECONDS", "2")
    monkeypatch.setenv("XAGENT_TASK_AUTO_RESUME_MODEL_OUTPUT_MAX_NO_PROGRESS", "2")
    monkeypatch.setenv("E2E_MODEL_RETRY_ATTEMPTS", str(ATTEMPTS))
    # The attempt count, not the deadline, bounds each call.
    monkeypatch.setenv("XAGENT_LLM_RETRY_DEADLINE_SECONDS", "30")
    for name in (
        "XAGENT_TASK_AUTO_RESUME_ENABLED",
        "XAGENT_TASK_INFRA_FAILURE_PAUSE_ENABLED",
        "XAGENT_TASK_AUTO_RESUME_MAX_TOTAL_PER_RUN",
        "XAGENT_TASK_AUTO_RESUME_LLM_MAX_ELAPSED_SECONDS",
    ):
        monkeypatch.delenv(name, raising=False)


# Per-test settings: requested before ``shared_app`` so its hosts see them.


@pytest.fixture
def two_resumes_per_run(monkeypatch):
    # A count limit, not elapsed time: the dispatch count stays exact.
    monkeypatch.setenv("XAGENT_TASK_AUTO_RESUME_MAX_TOTAL_PER_RUN", "2")


@pytest.fixture
def four_resumes_per_run(monkeypatch):
    # A safety net only: the no-progress limit (2) is expected to stop first,
    # and if progress were misread the failure still ends in bounded time.
    monkeypatch.setenv("XAGENT_TASK_AUTO_RESUME_MAX_TOTAL_PER_RUN", "4")


@pytest.fixture
def switches_off(monkeypatch):
    monkeypatch.setenv("XAGENT_TASK_AUTO_RESUME_ENABLED", "false")
    monkeypatch.setenv("XAGENT_TASK_INFRA_FAILURE_PAUSE_ENABLED", "false")


def _run_id(app, task_id: int) -> str:
    return wait_for(lambda: task_state(task_id)["run_id"], app, "the run")


def _row(task_id: int) -> dict:
    return recovery_state(task_id)[0] or {}


def _assert_answered_once(task_id: int) -> None:
    """One answer reached the history, and no failure bubble."""
    messages = assistant_messages(task_id)
    assert [m for m in messages if ANSWER in m[1]], messages
    assert len([m for m in messages if ANSWER in m[1]]) == 1, messages
    assert not [m for m in messages if m[0] == "task_failure"], messages


def _stopped(reason: str) -> dict:
    return {
        "status": "stopped",
        "stop_reason": "limit_reached",
        "reason": reason,
        "attempt": 2,
        "next_attempt_at": None,
    }


# --------------------------------------------------------------------------
# E4a: an outage the provider recovers from
# --------------------------------------------------------------------------


def test_provider_outage_pauses_then_resumes_when_provider_recovers(shared_app):
    app = shared_app
    # Two whole model calls fail; the third call's first attempt answers.
    set_model_fault(app, "unavailable", count=2 * ATTEMPTS)
    task_id = start_task(app, "e2e outage", gate=False)
    run_id = _run_id(app, task_id)

    def first_pause():
        if (_row(task_id).get("state"), _row(task_id).get("total_resumes")) != (
            "scheduled",
            0,
        ):
            return None
        detail = task_detail(app, task_id)
        recovery = detail["auto_recovery"] or {}
        if (recovery.get("auto_resume") or {}).get("status") != "scheduled":
            return None
        return task_state(task_id), _row(task_id), recovery

    # The first resume is due a second or so after the pause.
    state, row, recovery = wait_for(first_pause, app, "the first scheduled resume")
    assert state["status"] == "paused"
    assert row["reason"] == "llm_unavailable"
    assert (recovery["reason"], recovery["run_id"]) == ("llm_unavailable", run_id)
    assert recovery["auto_resume"]["attempt"] == 1

    settled = app.wait_task(task_id, run_id=run_id, timeout=SETTLE_TIMEOUT)
    assert ANSWER in (settled["output"] or "")
    assert settled["error"] is None
    # Every failed call was retried exactly ATTEMPTS times before it paused
    # the run, and the recovered provider was asked once.
    assert len(model_faults(app, "unavailable")) == 2 * ATTEMPTS
    assert len(answered_model_calls(app)) == 1
    row, events = recovery_state(task_id)
    assert row is not None
    assert (row["reason"], row["run_id"], row["total_resumes"], row["state"]) == (
        "llm_unavailable",
        run_id,
        2,
        "dispatched",
    )
    assert events == TWO_RESUMES
    commands = auto_resume_commands(task_id)
    assert len(commands) == len(set(commands)) == 2
    _assert_answered_once(task_id)


# --------------------------------------------------------------------------
# E4b: a provider that stays down exhausts the limit; the user can resume
# --------------------------------------------------------------------------


def test_provider_that_stays_down_exhausts_then_manual_resume_completes(
    two_resumes_per_run, shared_app
):
    from websockets.sync.client import connect

    app = shared_app
    set_model_fault(app, "unavailable", count=None)
    task_id = start_task(app, "e2e outage", gate=False)
    run_id = _run_id(app, task_id)

    wait_for(lambda: _row(task_id).get("state") == "exhausted", app, "exhaustion")
    assert task_state(task_id)["status"] == "paused"
    row, events = recovery_state(task_id)
    assert (row["reason"], row["total_resumes"]) == ("llm_unavailable", 2)
    assert events == TWO_RESUMES + ["interrupted", "exhausted"]
    assert len(model_faults(app, "unavailable")) == 3 * ATTEMPTS
    assert task_detail(app, task_id)["auto_recovery"]["auto_resume"] == _stopped(
        "llm_unavailable"
    )

    # Nothing resumes an exhausted run on its own.
    time.sleep(LLM_QUIET_SECONDS)
    assert task_state(task_id)["status"] == "paused"
    assert recovery_state(task_id)[1] == events
    assert len(auto_resume_commands(task_id)) == 2
    assert len(model_faults(app)) == 3 * ATTEMPTS

    clear_model_fault(app)
    with connect(ws_url(app, task_id)) as ws:
        receive_event(ws, "historical_data_complete")
        ws.send(
            json.dumps({"type": "resume_task", "command_id": "manual-after-exhausted"})
        )
        receive_event(ws, "task_command_accepted")
    settled = app.wait_task(task_id, run_id=run_id, timeout=SETTLE_TIMEOUT)
    assert ANSWER in (settled["output"] or "")
    assert len(answered_model_calls(app)) == 1
    assert len(auto_resume_commands(task_id)) == 2
    assert task_detail(app, task_id)["auto_recovery"] is None
    _assert_answered_once(task_id)


# --------------------------------------------------------------------------
# E5a: output that stays unusable is resampled twice, then left to the user
# --------------------------------------------------------------------------


def test_unusable_model_output_resumes_twice_then_stops(
    four_resumes_per_run, shared_app
):
    from websockets.sync.client import connect

    app = shared_app
    set_model_fault(app, "invalid_output", count=None)
    response = app.client.post(
        "/api/chat/task/create",
        headers=app.headers,
        json={
            "title": "Unusable output",
            "description": "e2e unusable output",
            "execution_mode": "balanced",
        },
    )
    assert response.status_code == 200, response.text
    task_id = response.json()["task_id"]

    def stopped(event):
        return (
            event.get("type") == "task_recovery_update"
            and event["auto_resume"]["status"] == "stopped"
        )

    with connect(ws_url(app, task_id)) as ws:
        receive_event(ws, "historical_data_complete")
        ws.send(json.dumps({"type": "execute_task"}))
        events_seen = collect_events(ws, stopped)
    wait_for(lambda: _row(task_id).get("state") == "exhausted", app, "exhaustion")

    assert task_state(task_id)["status"] == "paused"
    row, events = recovery_state(task_id)
    # Resampling made no progress: the no-progress limit stopped it, well
    # before the total-resume safety net.
    assert (row["reason"], row["total_resumes"], row["no_progress_resumes"]) == (
        "model_output_invalid",
        2,
        2,
    )
    assert events == TWO_RESUMES + ["interrupted", "exhausted"]
    # Each run asks twice: the answer and its one repair retry.
    assert len(model_faults(app, "invalid_output")) == 2 * 3
    assert answered_model_calls(app) == []
    assert len(auto_resume_commands(task_id)) == 2
    assert task_detail(app, task_id)["auto_recovery"]["auto_resume"] == _stopped(
        "model_output_invalid"
    )

    updates = [
        (event["auto_resume"]["status"], event["auto_resume"]["attempt"])
        for event in events_seen
        if event.get("type") == "task_recovery_update"
    ]
    expected = iter(
        [("scheduled", 1), ("resuming", 1), ("scheduled", 2), ("resuming", 2)]
    )
    pending = next(expected)
    for update in updates:
        if pending is not None and update == pending:
            pending = next(expected, None)
    assert pending is None, updates
    assert updates[-1] == ("stopped", 2)
    assert events_seen[-1]["auto_resume"]["stop_reason"] == "limit_reached"
    paused = [e for e in events_seen if e.get("type") == "task_paused"]
    assert len(paused) == 3, events_seen
    assert {e["interruption_reason"] for e in paused} == {"model_output_invalid"}
    assert not [m for m in assistant_messages(task_id) if m[0] == "task_failure"]

    time.sleep(MODEL_OUTPUT_QUIET_SECONDS)
    assert len(auto_resume_commands(task_id)) == 2
    assert len(model_faults(app)) == 2 * 3


# --------------------------------------------------------------------------
# E5b: one unusable answer, then a resampled one that is fine
# --------------------------------------------------------------------------


def test_unusable_model_output_recovers_on_resample(shared_app):
    app = shared_app
    # One run's answer and its repair retry are unusable.
    set_model_fault(app, "invalid_output", count=2)
    task_id = start_task(app, "e2e unusable output", gate=False)
    run_id = _run_id(app, task_id)

    settled = app.wait_task(task_id, run_id=run_id, timeout=SETTLE_TIMEOUT)
    assert ANSWER in (settled["output"] or "")
    # The resumed run asked the model again rather than replaying the
    # unusable result.
    assert len(model_faults(app, "invalid_output")) == 2
    assert len(answered_model_calls(app)) == 1
    row, events = recovery_state(task_id)
    assert (row["reason"], row["run_id"], row["total_resumes"]) == (
        "model_output_invalid",
        run_id,
        1,
    )
    assert events == ["interrupted", "scheduled", "auto_resumed"]
    assert len(auto_resume_commands(task_id)) == 1
    _assert_answered_once(task_id)


# --------------------------------------------------------------------------
# E10b-i: with both switches off an outage fails the task, as before
# --------------------------------------------------------------------------


def test_switches_off_llm_failure_fails_the_task(switches_off, shared_app):
    app = shared_app
    set_model_fault(app, "unavailable", count=None)
    task_id = start_task(app, "e2e outage", gate=False)

    settled = app.wait_task(task_id, status="failed", timeout=SETTLE_TIMEOUT)
    assert settled["error"]
    # One model call, retried by tier 1 only.
    assert len(model_faults(app, "unavailable")) == ATTEMPTS
    row, events = recovery_state(task_id)
    assert (row["state"], row["reason"]) == ("disabled", "llm_unavailable")
    assert events == ["interrupted"]
    assert auto_resume_commands(task_id) == []
    assert task_detail(app, task_id)["auto_recovery"] is None
    # The legacy failure path still writes its failure bubble.
    assert [m for m in assistant_messages(task_id) if m[0] == "task_failure"]

    time.sleep(3)
    assert task_state(task_id)["status"] == "failed"
    assert len(model_faults(app)) == ATTEMPTS
    assert recovery_state(task_id)[1] == ["interrupted"]
    assert auto_resume_commands(task_id) == []
