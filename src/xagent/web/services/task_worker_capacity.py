"""Process-local execution slots shared by command claims and running tasks."""

from __future__ import annotations

import asyncio
from contextvars import ContextVar
from threading import Lock
from typing import Any

from ...config import get_task_worker_max_concurrent_tasks


class TaskWorkerCapacity:
    """Count distinct tasks, including claims whose execution is still starting."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._references: dict[int, int] = {}

    def admitted_tasks_when_full(self) -> tuple[int, ...] | None:
        with self._lock:
            if len(self._references) < get_task_worker_max_concurrent_tasks():
                return None
            return tuple(self._references)

    def reserve(self, task_id: int) -> bool:
        # Claims run on DB threads; their check and reservation must be atomic.
        with self._lock:
            if (
                task_id not in self._references
                and len(self._references) >= get_task_worker_max_concurrent_tasks()
            ):
                return False
            self._references[task_id] = self._references.get(task_id, 0) + 1
            return True

    def release(self, task_id: int) -> None:
        with self._lock:
            remaining = self._references[task_id] - 1
            if remaining:
                self._references[task_id] = remaining
                return
            del self._references[task_id]
        from .task_command_transport import notify_task_command_dispatcher

        notify_task_command_dispatcher()


class TaskExecutionReservation:
    """Transfer a command's slot to its registered outer execution handles."""

    def __init__(self, capacity: TaskWorkerCapacity) -> None:
        self.capacity = capacity
        self.task_id: int | None = None

    def reserve(self, task_id: int) -> bool:
        if not self.capacity.reserve(task_id):
            return False
        self.task_id = task_id
        return True

    def track_execution(self, handle: asyncio.Task[Any]) -> None:
        assert self.task_id is not None
        task_id = self.task_id
        reserved = self.capacity.reserve(task_id)
        assert reserved
        handle.add_done_callback(lambda finished: self.capacity.release(task_id))

    def release(self) -> None:
        if self.task_id is not None:
            self.capacity.release(self.task_id)


worker_capacity = TaskWorkerCapacity()
current_execution_reservation: ContextVar[TaskExecutionReservation | None] = ContextVar(
    "current_execution_reservation", default=None
)
