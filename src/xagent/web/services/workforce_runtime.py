from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from sqlalchemy.orm import Session

from xagent.web.models.task import TaskStatus

from ..models.workforce import WorkforceRun
from .workforce_snapshot import build_agent_tool_overrides


@dataclass(frozen=True)
class WorkforceTaskRuntime:
    workforce_run_id: int
    workforce_id: int
    snapshot: dict[str, Any]
    allowed_agent_ids: list[int]
    agent_tool_overrides: dict[int, dict[str, Any]]
    worker_tool_names: set[str]
    manager_system_prompt: str | None
    manager_agent_id: int | None
    enable_global_agent_tools: bool = False
    allow_cross_user_agent_ids: bool = True

    @property
    def agent_call_stack(self) -> list[int]:
        return [self.manager_agent_id] if self.manager_agent_id is not None else []


def extract_workforce_run_id(task: Any) -> int | None:
    agent_config = getattr(task, "agent_config", None)
    if not isinstance(agent_config, dict):
        return None
    workforce_run_id = agent_config.get("workforce_run_id")
    return workforce_run_id if isinstance(workforce_run_id, int) else None


def is_workforce_task(task: Any) -> bool:
    agent_config = getattr(task, "agent_config", None)
    return isinstance(agent_config, dict) and isinstance(
        agent_config.get("workforce_run_id"), int
    )


def resolve_workforce_task_runtime(
    db: Session,
    task: Any,
) -> WorkforceTaskRuntime | None:
    workforce_run_id = extract_workforce_run_id(task)
    if workforce_run_id is None:
        return None

    task_id = getattr(task, "id", None)
    user_id = getattr(task, "user_id", None)
    if task_id is None or user_id is None:
        return None

    run = (
        db.query(WorkforceRun)
        .filter(
            WorkforceRun.id == workforce_run_id,
            WorkforceRun.task_id == int(task_id),
            WorkforceRun.user_id == int(user_id),
        )
        .first()
    )
    if run is None or not isinstance(run.snapshot, dict):
        return None

    snapshot = run.snapshot
    workforce_data = snapshot.get("workforce")
    manager_data = snapshot.get("manager")
    workers_data = snapshot.get("workers")
    if not isinstance(workforce_data, dict) or not isinstance(manager_data, dict):
        return None
    if not isinstance(workers_data, list):
        return None

    allowed_agent_ids: list[int] = []
    for worker in workers_data:
        if not isinstance(worker, dict) or worker.get("enabled") is False:
            continue
        agent_id = worker.get("agent_id")
        if isinstance(agent_id, int):
            allowed_agent_ids.append(agent_id)

    if not allowed_agent_ids:
        return None

    overrides = {
        agent_id: override
        for agent_id, override in build_agent_tool_overrides(
            snapshot, workforce_run_id=workforce_run_id
        ).items()
        if agent_id in set(allowed_agent_ids)
    }
    worker_tool_names = {
        str(override["tool_name"])
        for override in overrides.values()
        if isinstance(override.get("tool_name"), str)
    }
    workforce_id = workforce_data.get("id")
    manager_agent_id = manager_data.get("agent_id")
    manager_system_prompt = manager_data.get("runtime_prompt")

    return WorkforceTaskRuntime(
        workforce_run_id=workforce_run_id,
        workforce_id=int(workforce_id) if isinstance(workforce_id, int) else 0,
        snapshot=snapshot,
        allowed_agent_ids=allowed_agent_ids,
        agent_tool_overrides=overrides,
        worker_tool_names=worker_tool_names,
        manager_system_prompt=manager_system_prompt
        if isinstance(manager_system_prompt, str)
        else None,
        manager_agent_id=manager_agent_id
        if isinstance(manager_agent_id, int)
        else None,
    )


def _map_task_status(status: Any) -> str | None:
    if isinstance(status, str):
        try:
            status = TaskStatus(status)
        except ValueError:
            return None
    if status == TaskStatus.PENDING:
        return "pending"
    if status == TaskStatus.RUNNING:
        return "running"
    if status in {TaskStatus.PAUSED, TaskStatus.WAITING_FOR_USER}:
        return "paused"
    if status == TaskStatus.COMPLETED:
        return "completed"
    if status == TaskStatus.FAILED:
        return "failed"
    return None


def sync_workforce_run_status(
    db: Session, task: Any, status: Any | None = None
) -> bool:
    workforce_run_id = extract_workforce_run_id(task)
    mapped_status = _map_task_status(status if status is not None else task.status)
    if workforce_run_id is None or mapped_status is None:
        return False

    task_id = getattr(task, "id", None)
    user_id = getattr(task, "user_id", None)
    if task_id is None or user_id is None:
        return False

    run = (
        db.query(WorkforceRun)
        .filter(
            WorkforceRun.id == workforce_run_id,
            WorkforceRun.task_id == int(task_id),
            WorkforceRun.user_id == int(user_id),
        )
        .first()
    )
    if run is None:
        return False

    changed = False
    if run.status != mapped_status:
        setattr(run, "status", mapped_status)
        changed = True

    if mapped_status in {"completed", "failed", "cancelled"}:
        if run.completed_at is None:
            setattr(run, "completed_at", datetime.now(timezone.utc))
            changed = True
    elif run.completed_at is not None:
        setattr(run, "completed_at", None)
        changed = True

    return changed
