"""Langfuse sink for xagent trace events."""

from __future__ import annotations

import logging
from typing import Any, Optional, cast

from langfuse import Langfuse
from langfuse._client.attributes import LangfuseOtelSpanAttributes

from ...agent.trace import (
    TraceAction,
    TraceCategory,
    TraceEvent,
    TraceHandler,
    TraceScope,
)
from .client import get_langfuse_client
from .serialization import coerce_usage_details, serialize_for_langfuse

logger = logging.getLogger(__name__)


class LangfuseTraceHandler(TraceHandler):
    """Forward xagent trace events into Langfuse observations."""

    def __init__(
        self,
        *,
        task_id: str,
        user_id: Optional[int] = None,
        trace_name: Optional[str] = None,
        session_id: Optional[str] = None,
        tags: Optional[list[str]] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        self.task_id = task_id
        self.user_id = str(user_id) if user_id is not None else None
        self.trace_name = trace_name or f"xagent-task-{task_id}"
        self.session_id = session_id or f"task:{task_id}"
        self.tags = list(tags or ["xagent", "web"])
        self.metadata = {
            "source": "xagent-web",
            "task_id": task_id,
            **(metadata or {}),
        }
        self._root_observation: Optional[Any] = None
        self._step_observations: dict[str, Any] = {}
        self._action_observations: dict[str, Any] = {}
        self._task_llm_observations: dict[str, Any] = {}
        self._closed = False

    async def handle_event(self, event: TraceEvent) -> None:
        if self._closed:
            return

        client = get_langfuse_client()
        if client is None:
            return

        try:
            root = self._ensure_root_observation(client, event)
            if root is None:
                return

            if event.event_type.scope == TraceScope.TASK:
                self._handle_task_event(client, root, event)
            elif event.event_type.scope == TraceScope.STEP:
                self._handle_step_event(client, root, event)
            elif event.event_type.scope == TraceScope.ACTION:
                self._handle_action_event(client, root, event)
            else:
                self._record_event(client, root, event)
        except Exception as exc:
            logger.warning(f"Failed to forward trace event to Langfuse: {exc}")

    def _ensure_root_observation(
        self, client: Langfuse, event: TraceEvent
    ) -> Optional[Any]:
        if self._root_observation is not None:
            return self._root_observation

        trace_input = self._extract_trace_input(event)
        root_metadata = {
            **self.metadata,
            "initial_event_type": event.event_type.value,
        }

        root = cast(
            Any,
            client.start_observation(
                name=self.trace_name,
                as_type="agent",
                input=trace_input,
                metadata=root_metadata,
            ),
        )
        root.update_trace(
            name=self.trace_name,
            user_id=self.user_id,
            session_id=self.session_id,
            input=trace_input,
            metadata=root_metadata,
            tags=self.tags,
        )
        self._root_observation = root
        return root

    def _handle_task_event(
        self, client: Langfuse, root: Any, event: TraceEvent
    ) -> None:
        data = serialize_for_langfuse(event.data or {})
        metadata = self._event_metadata(event)

        if (
            event.event_type.category == TraceCategory.MESSAGE
            and event.event_type.action == TraceAction.START
        ):
            trace_input = self._extract_trace_input(event)
            root.update_trace(input=trace_input, metadata=metadata)
            self._record_event(client, root, event)
            return

        if event.event_type.category == TraceCategory.LLM:
            self._handle_task_llm_event(client, root, event)
            return

        if (
            event.event_type.category == TraceCategory.GENERAL
            and event.event_type.action == TraceAction.END
        ):
            output = self._extract_trace_output(data)
            success = (
                bool(data.get("success", True)) if isinstance(data, dict) else True
            )
            level = None if success else "ERROR"
            status_message = None if success else "Task completed with failure state"
            root.update(
                output=output,
                metadata=metadata,
                level=level,
                status_message=status_message,
            )
            root.update_trace(output=output, metadata=metadata)
            self._close_open_observations()
            root.end()
            self._closed = True
            return

        if event.event_type.action == TraceAction.ERROR:
            root.update(
                output=data,
                metadata=metadata,
                level="ERROR",
                status_message=self._error_message(data),
            )
            root.update_trace(metadata=metadata)
            self._close_open_observations()
            root.end()
            self._closed = True
            return

        self._record_event(client, root, event)

    def _handle_task_llm_event(
        self, client: Langfuse, root: Any, event: TraceEvent
    ) -> None:
        data = serialize_for_langfuse(event.data or {})
        metadata = self._event_metadata(event)
        observation_key = str(event.task_id or event.id)
        parent = self._resolve_task_event_parent(root, data)

        if event.event_type.action == TraceAction.START:
            observation_kwargs = self._start_observation_kwargs(event, data, metadata)
            observation = self._start_child_observation(
                client, parent, **observation_kwargs
            )
            self._task_llm_observations[observation_key] = observation
            return

        observation = self._task_llm_observations.get(observation_key)
        if observation is None:
            self._record_event(client, parent, event)
            return

        update_kwargs = self._update_observation_kwargs(event, data, metadata)
        if self._is_event_error(data):
            update_kwargs["level"] = "ERROR"
            update_kwargs["status_message"] = self._error_message(data)

        observation.update(**update_kwargs)
        observation.end()
        self._task_llm_observations.pop(observation_key, None)

    def _handle_step_event(
        self, client: Langfuse, root: Any, event: TraceEvent
    ) -> None:
        data = serialize_for_langfuse(event.data or {})
        metadata = self._event_metadata(event)
        step_id = event.step_id or "unknown-step"

        if event.event_type.action == TraceAction.START:
            observation = self._start_child_observation(
                client,
                root,
                name=self._observation_name(event),
                as_type="span",
                input=data,
                metadata=metadata,
            )
            self._step_observations[step_id] = observation
            return

        observation = self._step_observations.get(step_id)
        if observation is None:
            self._record_event(client, root, event)
            return

        if event.event_type.action == TraceAction.END:
            observation.update(output=data, metadata=metadata)
            observation.end()
            self._step_observations.pop(step_id, None)
            return

        if event.event_type.action == TraceAction.ERROR:
            observation.update(
                metadata=metadata,
                output={"last_error": data},
            )
            self._record_event(client, observation, event)
            return

        self._record_event(client, observation, event)

    def _handle_action_event(
        self, client: Langfuse, root: Any, event: TraceEvent
    ) -> None:
        data = serialize_for_langfuse(event.data or {})
        metadata = self._event_metadata(event)
        key = self._action_key(event, data)
        parent = self._step_observations.get(event.step_id or "", root)

        if event.event_type.action == TraceAction.START:
            observation_kwargs = self._start_observation_kwargs(event, data, metadata)
            observation = self._start_child_observation(
                client, parent, **observation_kwargs
            )
            self._action_observations[key] = observation
            return

        observation = self._action_observations.get(key)
        if observation is None:
            self._record_event(client, parent, event)
            return

        if event.event_type.action == TraceAction.END:
            update_kwargs = self._update_observation_kwargs(event, data, metadata)
            observation.update(**update_kwargs)
            observation.end()
            self._action_observations.pop(key, None)
            return

        if event.event_type.action == TraceAction.ERROR:
            update_kwargs = self._update_observation_kwargs(event, data, metadata)
            update_kwargs["level"] = "ERROR"
            update_kwargs["status_message"] = self._error_message(data)
            observation.update(**update_kwargs)
            observation.end()
            self._action_observations.pop(key, None)
            return

        self._record_event(client, observation, event)

    def _record_event(self, client: Langfuse, parent: Any, event: TraceEvent) -> None:
        data = serialize_for_langfuse(event.data or {})
        metadata = self._event_metadata(event)
        input_value = data if event.event_type.action == TraceAction.START else None
        output_value = None if event.event_type.action == TraceAction.START else data
        self._start_child_observation(
            client,
            parent,
            name=self._observation_name(event),
            as_type="event",
            input=input_value,
            output=output_value,
            metadata=metadata,
            level="ERROR" if event.event_type.action == TraceAction.ERROR else None,
            status_message=self._error_message(data)
            if event.event_type.action == TraceAction.ERROR
            else None,
        )

    def _start_child_observation(
        self,
        client: Langfuse,
        parent: Any,
        **kwargs: Any,
    ) -> Any:
        observation = client.start_observation(
            trace_context={"trace_id": parent.trace_id, "parent_span_id": parent.id},
            **kwargs,
        )
        otel_span = getattr(observation, "_otel_span", None)
        if otel_span is not None:
            try:
                otel_span.set_attribute(LangfuseOtelSpanAttributes.AS_ROOT, False)
            except Exception:
                pass
        return observation

    def _start_observation_kwargs(
        self,
        event: TraceEvent,
        data: Any,
        metadata: dict[str, Any],
    ) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "name": self._observation_name(event),
            "as_type": self._observation_type(event),
            "input": self._extract_observation_input(event, data),
            "metadata": metadata,
        }

        if event.event_type.category == TraceCategory.LLM and isinstance(data, dict):
            kwargs["model"] = data.get("model_name")
            usage_details = coerce_usage_details(data.get("usage"))
            if usage_details:
                kwargs["usage_details"] = usage_details

        return kwargs

    def _update_observation_kwargs(
        self,
        event: TraceEvent,
        data: Any,
        metadata: dict[str, Any],
    ) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "output": self._extract_observation_output(event, data),
            "metadata": metadata,
        }

        if event.event_type.category == TraceCategory.LLM and isinstance(data, dict):
            kwargs["model"] = data.get("model_name")
            usage_details = coerce_usage_details(data.get("usage"))
            if usage_details:
                kwargs["usage_details"] = usage_details

        return kwargs

    def _observation_name(self, event: TraceEvent) -> str:
        if event.event_type.scope == TraceScope.STEP and event.step_id:
            return f"step_{event.step_id}"

        category = event.event_type.category.value
        action = event.event_type.action.value
        if category == TraceCategory.TOOL.value and isinstance(event.data, dict):
            tool_name = event.data.get("tool_name")
            if tool_name:
                return f"tool_{tool_name}_{action}"
        if category == TraceCategory.LLM.value and isinstance(event.data, dict):
            task_type = event.data.get("task_type")
            if task_type:
                return f"llm_{task_type}_{action}"
            model_name = event.data.get("model_name")
            if model_name:
                return f"llm_{model_name}_{action}"
        return event.event_type.value

    def _observation_type(self, event: TraceEvent) -> str:
        if event.event_type.category == TraceCategory.LLM:
            return "generation"
        if event.event_type.category == TraceCategory.TOOL:
            return "tool"
        if event.event_type.category == TraceCategory.MEMORY_RETRIEVE:
            return "retriever"
        return "span"

    def _extract_trace_input(self, event: TraceEvent) -> Any:
        data = serialize_for_langfuse(event.data or {})
        if isinstance(data, dict):
            if "message" in data:
                return data["message"]
            if "task" in data:
                return data["task"]
            if "task_preview" in data:
                return data["task_preview"]
        return data

    def _extract_trace_output(self, data: Any) -> Any:
        if isinstance(data, dict) and "result" in data:
            return data["result"]
        return data

    def _extract_observation_input(self, event: TraceEvent, data: Any) -> Any:
        if not isinstance(data, dict):
            return data

        if event.event_type.category == TraceCategory.LLM:
            if data.get("messages"):
                return {
                    "messages": data.get("messages"),
                    "model_name": data.get("model_name"),
                    "tools": data.get("tools"),
                    "tool_choice": data.get("tool_choice"),
                    "task_type": data.get("task_type"),
                    "step_id": data.get("step_id"),
                    "step_name": data.get("step_name"),
                }
            return {
                "model_name": data.get("model_name"),
                "task_type": data.get("task_type"),
                "step_id": data.get("step_id"),
                "step_name": data.get("step_name"),
            }

        if event.event_type.category == TraceCategory.TOOL:
            return {
                "tool_name": data.get("tool_name"),
                "tool_args": data.get("tool_args"),
            }

        return data

    def _extract_observation_output(self, event: TraceEvent, data: Any) -> Any:
        if not isinstance(data, dict):
            return data

        if event.event_type.category == TraceCategory.LLM:
            if data.get("response") is None and data.get("content") is None:
                return data
            return {
                "response": data.get("response"),
                "content": data.get("content"),
                "is_tool_call": data.get("is_tool_call"),
                "usage": data.get("usage"),
                "success": data.get("success"),
                "error": data.get("error"),
                "task_type": data.get("task_type"),
            }

        if event.event_type.category == TraceCategory.TOOL:
            return {
                "tool_name": data.get("tool_name"),
                "result": data.get("result"),
                "success": data.get("success"),
            }

        return data

    def _event_metadata(self, event: TraceEvent) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "event_id": event.id,
            "event_type": event.event_type.value,
            "scope": event.event_type.scope.value,
            "action": event.event_type.action.value,
            "category": event.event_type.category.value,
            "step_id": event.step_id,
            "parent_event_id": event.parent_id,
            "data": serialize_for_langfuse(event.data or {}),
        }

    def _action_key(self, event: TraceEvent, data: Any) -> str:
        tool_name = data.get("tool_name") if isinstance(data, dict) else None
        return f"{event.step_id}:{event.event_type.category.value}:{tool_name or ''}"

    def _error_message(self, data: Any) -> Optional[str]:
        if isinstance(data, dict):
            message = data.get("error_message") or data.get("error")
            if message:
                return str(message)
        return None

    def _is_event_error(self, data: Any) -> bool:
        if not isinstance(data, dict):
            return False
        if data.get("success") is False:
            return True
        return bool(data.get("error") or data.get("error_message"))

    def _resolve_task_event_parent(self, root: Any, data: Any) -> Any:
        if isinstance(data, dict):
            step_id = data.get("step_id")
            if step_id and step_id in self._step_observations:
                return self._step_observations[step_id]
        return root

    def _close_open_observations(self) -> None:
        for observation in list(self._action_observations.values()):
            try:
                observation.end()
            except Exception:
                pass
        self._action_observations.clear()

        for observation in list(self._task_llm_observations.values()):
            try:
                observation.end()
            except Exception:
                pass
        self._task_llm_observations.clear()

        for observation in list(self._step_observations.values()):
            try:
                observation.end()
            except Exception:
                pass
        self._step_observations.clear()
