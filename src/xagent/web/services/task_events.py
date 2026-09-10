"""Task event publication, with delivery supplied by the hosting process."""

from collections.abc import Awaitable, Callable
from typing import Any, Literal, Protocol


class DeliveryNotifier(Protocol):
    """Transport callback for acknowledging a deferred user message."""

    async def __call__(
        self,
        *,
        turn_id: str | None,
        accepted: bool,
        message: str | None = None,
        error_code: str | None = None,
        retry_with_new_id: bool = False,
        rejection_outcome: Literal["not_accepted", "outcome_unknown"] | None = None,
    ) -> None: ...


TaskEventSink = Callable[[dict[str, Any], int], Awaitable[None]]
_task_event_sink: TaskEventSink | None = None


def set_task_event_sink(sink: TaskEventSink | None) -> None:
    """Attach the host's event delivery adapter; None means no live listeners."""
    global _task_event_sink
    _task_event_sink = sink


async def publish_task_event(message: dict[str, Any], task_id: int) -> None:
    """Publish a live event without depending on connections or API handlers."""
    if _task_event_sink is not None:
        await _task_event_sink(message, task_id)
