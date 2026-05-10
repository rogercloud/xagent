"""Web tracer factory helpers."""

from __future__ import annotations

from typing import Optional

from ..core.agent.trace import ConsoleTraceHandler, TraceHandler, Tracer
from ..core.tracing import create_agent_tracer
from .api.trace_handlers import DatabaseTraceHandler
from .models.user import User


def create_task_tracer(
    task_id: int,
    user: Optional[User] = None,
    user_id: Optional[int] = None,
) -> Tracer:
    """Build the standard tracer stack for persisted web task execution."""
    from .api.ws_trace_handlers import WebSocketTraceHandler

    resolved_user_id = user_id
    if user is not None and user.id is not None:
        resolved_user_id = int(user.id)

    return create_agent_tracer(
        handlers=[
            ConsoleTraceHandler(),
            DatabaseTraceHandler(task_id),
            WebSocketTraceHandler(task_id),
        ],
        task_id=str(task_id),
        user_id=resolved_user_id,
        trace_name=f"xagent-web-task-{task_id}",
        session_id=f"task:{task_id}",
        tags=["xagent", "web", "task"],
        metadata={
            "source": "xagent-web",
            "task_id": task_id,
            "is_preview": False,
        },
    )


def create_ephemeral_tracer(
    *,
    task_id: str,
    websocket_handler: TraceHandler,
    user: Optional[User] = None,
    is_preview: bool = False,
) -> Tracer:
    """Build a tracer for websocket-only flows such as builder preview."""
    return create_agent_tracer(
        handlers=[websocket_handler],
        task_id=task_id,
        user_id=int(user.id) if user and user.id is not None else None,
        trace_name=f"xagent-web-{task_id}",
        session_id=task_id,
        tags=["xagent", "web", "preview" if is_preview else "builder"],
        metadata={
            "source": "xagent-web",
            "task_id": task_id,
            "is_preview": is_preview,
        },
    )
