"""Automatic resume of SDK tasks, as their caller sees it.

An SDK caller only polls a task's status, and PAUSED is not a final one, so
a run interrupted by a lost lease is resumed for it like any other task and
the caller's poll ends on the resumed run's result. A settlement
interruption (here a provider outage) still fails the task, as it always
has: a pause that automatic resume later gives up on would leave the caller
polling a PAUSED task for good.
"""

from __future__ import annotations

import time

import pytest

from tests.e2e.shared_execution_harness import (
    SETTLE_TIMEOUT,
    auto_resume_commands,
    create_sdk_task,
    kill_host,
    model_calls,
    model_faults,
    recovery_state,
    set_model_fault,
    task_state,
    wait_for,
)

pytestmark = pytest.mark.e2e

# Provider attempts per model call (E2E_MODEL_RETRY_ATTEMPTS).
ATTEMPTS = 3
# Longer than any resume this module could schedule (see
# test_shared_provider_outage.py).
LLM_QUIET_SECONDS = 5
ANSWER = "Shared E2E answer"
FINAL_STATUSES = {"completed", "failed"}


@pytest.fixture(autouse=True)
def _fast_auto_resume(monkeypatch):
    # Read by the hosts the shared_app fixture spawns after this.
    monkeypatch.setenv("XAGENT_TASK_AUTO_RESUME_POLL_SECONDS", "1")
    monkeypatch.setenv("XAGENT_TASK_AUTO_RESUME_SHORT_INITIAL_BACKOFF_SECONDS", "1")
    monkeypatch.setenv("XAGENT_TASK_AUTO_RESUME_LLM_INITIAL_BACKOFF_SECONDS", "1")
    monkeypatch.setenv("XAGENT_TASK_AUTO_RESUME_LLM_MAX_BACKOFF_SECONDS", "2")
    monkeypatch.setenv("E2E_MODEL_RETRY_ATTEMPTS", str(ATTEMPTS))
    monkeypatch.setenv("XAGENT_LLM_RETRY_DEADLINE_SECONDS", "30")
    for name in (
        "XAGENT_TASK_AUTO_RESUME_ENABLED",
        "XAGENT_TASK_INFRA_FAILURE_PAUSE_ENABLED",
        "XAGENT_TASK_AUTO_RESUME_MAX_TOTAL_PER_RUN",
        "XAGENT_TASK_AUTO_RESUME_LLM_MAX_ELAPSED_SECONDS",
    ):
        monkeypatch.delenv(name, raising=False)


def poll_until_final(app, task_id: int, headers) -> tuple[dict, list[str]]:
    """Poll as an SDK caller does: the final view and every status seen."""
    seen: list[str] = []
    deadline = time.monotonic() + SETTLE_TIMEOUT
    while time.monotonic() < deadline:
        response = app.client.get(f"/v1/chat/tasks/{task_id}", headers=headers)
        assert response.status_code == 200, response.text
        view = response.json()
        if not seen or seen[-1] != view["status"]:
            seen.append(view["status"])
        if view["status"] in FINAL_STATUSES:
            return view, seen
        time.sleep(0.1)
    pytest.fail(f"SDK task stayed {seen}\n{app.diagnostics()}")


def test_sdk_task_resumes_after_its_worker_is_killed(shared_app):
    app = shared_app
    task_id, run_id, headers = create_sdk_task(app, "e2e:gate")
    wait_for(lambda: (app.root / "model-entered").exists(), app, "model call")

    kill_host(app, app.processes[1])
    # The resumed call may answer; the killed one never returns.
    (app.root / "model-release").touch()
    app.start("worker")

    view, seen = poll_until_final(app, task_id, headers)
    # The caller never saw the run fail: it kept polling through the pause
    # and got the resumed run's answer.
    assert "failed" not in seen, seen
    assert view["status"] == "completed", view
    assert ANSWER in (view["output"] or ""), view
    assert view["error"] is None, view
    events = app.client.get(f"/v1/chat/tasks/{task_id}/events", headers=headers)
    assert events.status_code == 200, events.text
    assert "event: task.completed" in events.text
    assert ANSWER in events.text

    settled = app.wait_task(task_id, run_id=run_id, timeout=SETTLE_TIMEOUT)
    assert settled["run_id"] == run_id
    # The killed call and the resumed one: nothing ran twice.
    assert len(model_calls(app)) == 2
    row, recovery_events = recovery_state(task_id)
    assert row is not None
    assert (row["reason"], row["run_id"], row["total_resumes"]) == (
        "lease_expired",
        run_id,
        1,
    )
    assert recovery_events == ["interrupted", "scheduled", "auto_resumed"]
    assert len(auto_resume_commands(task_id)) == 1


def test_sdk_task_fails_on_a_provider_outage_without_pausing(shared_app):
    app = shared_app
    set_model_fault(app, "unavailable", count=None)
    task_id, run_id, headers = create_sdk_task(app, "e2e outage")

    view, seen = poll_until_final(app, task_id, headers)
    assert "paused" not in seen, seen
    assert view["status"] == "failed", view
    assert view["error"], view
    assert task_state(task_id)["run_id"] == run_id
    # One model call, retried by tier 1 only, and never resumed.
    assert len(model_faults(app, "unavailable")) == ATTEMPTS
    row, events = recovery_state(task_id)
    assert row is not None
    # Recorded like any failed run: not ineligible, and not "disabled"
    # either, since the infrastructure-failure switch would not pause it.
    assert (row["state"], row["state_detail"], row["reason"]) == (
        "manual",
        None,
        "llm_unavailable",
    )
    assert events == ["interrupted"]
    assert auto_resume_commands(task_id) == []

    time.sleep(LLM_QUIET_SECONDS)
    assert task_state(task_id)["status"] == "failed"
    assert len(model_faults(app)) == ATTEMPTS
    assert recovery_state(task_id)[1] == ["interrupted"]
    assert auto_resume_commands(task_id) == []
