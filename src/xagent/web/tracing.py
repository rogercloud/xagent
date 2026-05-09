"""Web tracer factory helpers."""

from __future__ import annotations

from typing import Optional

from ..core.agent.trace import ConsoleTraceHandler, TraceHandler, Tracer
from ..core.tracing.langfuse import create_langfuse_trace_handler
from .api.trace_handlers import DatabaseTraceHandler
from .models.user import User


def create_task_tracer(task_id: int, user: Optional[User] = None) -> Tracer:
    """Build the standard tracer stack for persisted web task execution."""
    from .api.ws_trace_handlers import WebSocketTraceHandler

    tracer = Tracer()
    tracer.add_handler(ConsoleTraceHandler())
    tracer.add_handler(DatabaseTraceHandler(task_id))
    tracer.add_handler(WebSocketTraceHandler(task_id))

    langfuse_handler = create_langfuse_trace_handler(
        task_id=str(task_id),
        user_id=int(user.id) if user and user.id is not None else None,
        trace_name=f"xagent-web-task-{task_id}",
        session_id=f"task:{task_id}",
        tags=["xagent", "web", "task"],
        metadata={"task_id": task_id, "is_preview": False},
    )
    if langfuse_handler is not None:
        tracer.add_handler(langfuse_handler)

    return tracer


def create_ephemeral_tracer(
    *,
    task_id: str,
    websocket_handler: TraceHandler,
    user: Optional[User] = None,
    is_preview: bool = False,
) -> Tracer:
    """Build a tracer for websocket-only flows such as builder preview."""
    tracer = Tracer()
    tracer.add_handler(websocket_handler)

    langfuse_handler = create_langfuse_trace_handler(
        task_id=task_id,
        user_id=int(user.id) if user and user.id is not None else None,
        trace_name=f"xagent-web-{task_id}",
        session_id=task_id,
        tags=["xagent", "web", "preview" if is_preview else "builder"],
        metadata={"task_id": task_id, "is_preview": is_preview},
    )
    if langfuse_handler is not None:
        tracer.add_handler(langfuse_handler)

    return tracer
