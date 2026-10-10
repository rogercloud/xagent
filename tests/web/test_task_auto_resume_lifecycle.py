"""Backend lifecycle wiring for the auto-resume sweeper."""

from __future__ import annotations

import asyncio

import pytest
from fastapi import FastAPI

from xagent.web import app as app_module


@pytest.mark.asyncio
async def test_auto_resume_start_and_stop_owns_background_task(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    started = asyncio.Event()
    stopped = asyncio.Event()

    async def fake_loop(*, poll_interval_seconds: int) -> None:
        assert poll_interval_seconds == 11
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            stopped.set()

    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    monkeypatch.setattr(app_module, "get_task_auto_resume_poll_seconds", lambda: 11)
    monkeypatch.setattr(app_module, "run_task_auto_resume_loop", fake_loop)
    app = FastAPI()

    task = app_module.start_task_auto_resume_task(app)
    assert task is app.state.task_auto_resume_task
    # Idempotent while running.
    assert app_module.start_task_auto_resume_task(app) is task
    await asyncio.wait_for(started.wait(), timeout=1)

    await app_module.stop_task_auto_resume_task(app)

    assert stopped.is_set()
    assert task.cancelled()
    assert app.state.task_auto_resume_task is None


@pytest.mark.asyncio
async def test_auto_resume_stop_consumes_completed_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    failure = RuntimeError("sweeper failed")

    async def failed() -> None:
        raise failure

    task = asyncio.create_task(failed())
    await asyncio.sleep(0)
    app = FastAPI()
    app.state.task_auto_resume_task = task
    logged: list[BaseException] = []
    monkeypatch.setattr(
        app_module.logger,
        "error",
        lambda *_a, exc_info=None, **_k: logged.append(exc_info),
    )

    await app_module.stop_task_auto_resume_task(app)

    assert logged == [failure]
    assert app.state.task_auto_resume_task is None


def test_auto_resume_start_skips_under_pytest() -> None:
    app = FastAPI()
    assert app_module.start_task_auto_resume_task(app) is None
    assert getattr(app.state, "task_auto_resume_task", None) is None


def test_web_role_does_not_run_the_sweeper(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    monkeypatch.setenv("XAGENT_SHARED_TASK_EXECUTION_ENABLED", "true")
    monkeypatch.setenv("XAGENT_TASK_EXECUTION_ROLE", "web")
    assert app_module.start_task_auto_resume_task(FastAPI()) is None


@pytest.mark.asyncio
async def test_application_shutdown_stops_the_sweeper(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    order: list[str] = []

    async def stop_sweeper(app_instance) -> None:
        assert app_instance is app_module.app
        order.append("auto_resume")

    async def stop_recovery(app_instance) -> None:
        order.append("lease_recovery")

    monkeypatch.setattr(app_module, "stop_task_auto_resume_task", stop_sweeper)
    monkeypatch.setattr(app_module, "stop_task_lease_recovery_task", stop_recovery)
    # Reuse the full shutdown scaffold of the lease-recovery lifecycle test.
    from tests.web import test_task_lease_recovery_lifecycle as lease_lifecycle

    await lease_lifecycle.test_application_shutdown_stops_task_lease_recovery(
        monkeypatch
    )
    assert "auto_resume" in order
