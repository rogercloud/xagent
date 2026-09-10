"""The task runtime can load and publish events without a Web API host."""

import subprocess
import sys
import textwrap
from unittest.mock import AsyncMock

import pytest

from xagent.web.services import task_events


def test_execution_services_and_tracer_load_without_api_routes() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            textwrap.dedent(
                """
                import importlib.abc
                import sys

                class RejectRoutes(importlib.abc.MetaPathFinder):
                    def find_spec(self, fullname, path=None, target=None):
                        if fullname == "xagent.web.api" or fullname.startswith("xagent.web.api."):
                            raise AssertionError(f"Execution imported API route: {fullname}")

                sys.meta_path.insert(0, RejectRoutes())
                from xagent.web.services import agent_service_manager, task_execution, task_orchestrator
                from xagent.web.tracing import create_task_tracer
                create_task_tracer(1, user_id=1)
                """
            ),
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.asyncio
async def test_event_delivery_uses_the_host_sink(monkeypatch) -> None:
    monkeypatch.setattr(task_events, "_task_event_sink", None)
    event = {"type": "task_completed", "task_id": 42}
    await task_events.publish_task_event(event, 42)

    sink = AsyncMock()
    task_events.set_task_event_sink(sink)
    await task_events.publish_task_event(event, 42)
    sink.assert_awaited_once_with(event, 42)

    sink.side_effect = RuntimeError("delivery failed")
    with pytest.raises(RuntimeError, match="delivery failed"):
        await task_events.publish_task_event(event, 42)
