from datetime import datetime, timezone
from typing import Any

from sqlalchemy import inspect, text
from sqlalchemy.orm import Session
from xagent.web.models.task import TaskStatus


def extract_workforce_run_id(task: Any) -> int | None:
    agent_config = getattr(task, "agent_config", None)
    if not isinstance(agent_config, dict):
        return None
    workforce_run_id = agent_config.get("workforce_run_id")
    return workforce_run_id if isinstance(workforce_run_id, int) else None


def has_workforce_runs_table(db: Session) -> bool:
    try:
        inspector = inspect(db.bind)
        return bool(inspector is not None and inspector.has_table("workforce_runs"))
    except Exception:
        return False


def is_verified_workforce_task(db: Session, task: Any, workforce_run_id: Any) -> bool:
    if not isinstance(workforce_run_id, int):
        return False
    if not has_workforce_runs_table(db):
        return False
    try:
        row = db.execute(
            text(
                "SELECT 1 FROM workforce_runs "
                "WHERE id = :run_id AND task_id = :task_id AND user_id = :user_id"
            ),
            {
                "run_id": workforce_run_id,
                "task_id": int(task.id),
                "user_id": int(task.user_id),
            },
        ).first()
        return row is not None
    except Exception:
        return False


def _map_task_status(status: Any) -> str | None:
    if status == TaskStatus.PENDING:
        return "pending"
    if status == TaskStatus.RUNNING:
        return "running"
    if status == TaskStatus.COMPLETED:
        return "completed"
    if status == TaskStatus.FAILED:
        return "failed"
    return None


def sync_workforce_run_status(db: Session, task: Any, status: Any) -> None:
    workforce_run_id = extract_workforce_run_id(task)
    mapped_status = _map_task_status(status)
    if workforce_run_id is None or mapped_status is None:
        return
    if not has_workforce_runs_table(db):
        return

    completed_at = None
    if mapped_status in {"completed", "failed"}:
        completed_at = datetime.now(timezone.utc)

    db.execute(
        text(
            "UPDATE workforce_runs "
            "SET status = :status, completed_at = :completed_at "
            "WHERE id = :run_id AND task_id = :task_id"
        ),
        {
            "status": mapped_status,
            "completed_at": completed_at,
            "run_id": workforce_run_id,
            "task_id": int(task.id),
        },
    )
