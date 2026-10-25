"""Runtime performance telemetry for one app or worker process.

Both entry points start it: the app in its lifespan, the shared-execution
worker in ``run_worker``. A worker is where tasks run once web and worker
roles are deployed apart, so without it every execution-side metric would
be recorded into a no-op meter.
"""

from __future__ import annotations

import asyncio
import logging

from ..core.runtime_performance import (
    initialize_runtime_performance_telemetry,
    register_observable_gauge,
    shutdown_runtime_performance_telemetry,
    start_event_loop_lag_monitor,
    stop_event_loop_lag_monitor,
)

logger = logging.getLogger(__name__)


def start_process_runtime_telemetry() -> asyncio.Task[None] | None:
    """Enable OTLP export, register process gauges and sample loop lag.

    Returns the lag sampler, or ``None`` when telemetry is disabled or could
    not start. Never raises: telemetry is observational and must not block a
    process from starting.
    """

    try:
        if not initialize_runtime_performance_telemetry():
            return None

        from .api.websocket import manager
        from .services.task_execution import background_task_manager

        register_observable_gauge(
            "xagent.agent_tasks.running",
            lambda: len(background_task_manager.running_tasks),
            unit="{task}",
            description="Current locally running agent tasks",
        )
        register_observable_gauge(
            "xagent.agent_tasks.resuming",
            lambda: len(background_task_manager.resume_tasks),
            unit="{task}",
            description="Current local resume coordinators",
        )
        register_observable_gauge(
            "xagent.websocket.connections",
            manager.connection_count,
            unit="{connection}",
            description="Current task WebSocket connections",
        )
        return start_event_loop_lag_monitor()
    except Exception:
        shutdown_runtime_performance_telemetry()
        logger.warning("Could not start runtime performance monitor", exc_info=True)
        return None


async def stop_process_runtime_telemetry(task: asyncio.Task[None] | None) -> None:
    """Stop the lag sampler, then flush and shut down metric export."""

    await stop_event_loop_lag_monitor(task)
    await asyncio.to_thread(shutdown_runtime_performance_telemetry)
