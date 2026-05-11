from typing import Any, cast

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import or_
from sqlalchemy.orm import Session
from xagent.web.auth_dependencies import get_current_user
from xagent.web.models.agent import Agent
from xagent.web.models.database import get_db
from xagent.web.models.task import Task, TaskStatus
from xagent.web.models.user import User

from ..models.workforce import (
    Workforce,
    WorkforceAgent,
    WorkforceBuilderMessage,
    WorkforceRun,
)
from ..services.workforce_access import (
    can_create_workforce,
    can_view_workforce,
    ensure_agent_access,
    ensure_workforce_access,
    get_workforce_policy,
    resolve_create_scope,
)
from ..services.workforce_builder import (
    apply_builder_patch,
    generate_builder_patch,
    list_builder_messages,
    serialize_builder_message,
)
from ..services.workforce_snapshot import (
    build_workforce_snapshot,
    build_workforce_task_config,
    normalize_text,
    normalize_workforce_status,
)
from ..services.workforce_workers import (
    create_workforce_worker,
    ensure_supported_source_type,
    list_template_summaries,
)

router = APIRouter(prefix="/api/workforces", tags=["workforces"])


class WorkforceNewAgentInput(BaseModel):
    name: str = Field(..., min_length=1, max_length=200)
    description: str | None = None
    instructions: str | None = None
    execution_mode: str | None = "balanced"
    models: dict[str, Any] | None = None
    knowledge_bases: list[str] = Field(default_factory=list)
    skills: list[str] = Field(default_factory=list)
    tool_categories: list[str] = Field(default_factory=list)
    suggested_prompts: list[str] = Field(default_factory=list)


class WorkforceWorkerInput(BaseModel):
    source_type: str = Field(default="existing")
    agent_id: int | None = None
    alias: str | None = None
    assignment_instructions: str = Field(..., min_length=1)
    template_id: str | None = None
    agent: WorkforceNewAgentInput | None = None
    enabled: bool = True
    sort_order: int | None = None
    canvas_position: dict[str, Any] | None = None


class WorkforceCreateRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=200)
    description: str | None = None
    manager_agent_id: int
    manager_instructions: str | None = None
    status: str | None = "draft"
    canvas_layout: dict[str, Any] | None = None
    workers: list[WorkforceWorkerInput] = Field(default_factory=list)


class WorkforceUpdateRequest(BaseModel):
    name: str | None = Field(None, min_length=1, max_length=200)
    description: str | None = None
    manager_agent_id: int | None = None
    manager_instructions: str | None = None
    status: str | None = None
    canvas_layout: dict[str, Any] | None = None


class WorkforceWorkerUpdateRequest(BaseModel):
    alias: str | None = None
    assignment_instructions: str | None = None
    enabled: bool | None = None
    sort_order: int | None = None
    canvas_position: dict[str, Any] | None = None


class WorkforceRunRequest(BaseModel):
    message: str = Field(..., min_length=1)
    files: list[str] = Field(default_factory=list)
    execution_mode: str | None = None


class WorkforceBuilderProposeRequest(BaseModel):
    message: str = Field(..., min_length=1)
    context: dict[str, Any] | None = None


class WorkforceBuilderApplyRequest(BaseModel):
    message_id: int
    proposed_patch: dict[str, Any]


def _agent_status_value(agent: Agent) -> str:
    status = getattr(agent, "status", None)
    value = getattr(status, "value", None)
    if isinstance(value, str):
        return value
    return str(status or "")


def _serialize_agent(agent: Agent) -> dict[str, Any]:
    return {
        "id": agent.id,
        "name": agent.name,
        "description": agent.description,
        "logo_url": agent.logo_url,
        "status": _agent_status_value(agent),
    }


def _serialize_worker(worker: WorkforceAgent) -> dict[str, Any]:
    return {
        "id": worker.id,
        "agent": _serialize_agent(worker.agent),
        "alias": worker.alias,
        "assignment_instructions": worker.assignment_instructions,
        "source_type": worker.source_type,
        "template_id": worker.template_id,
        "enabled": worker.enabled,
        "sort_order": worker.sort_order,
        "canvas_position": worker.canvas_position,
    }


def _serialize_workforce_detail(workforce: Workforce) -> dict[str, Any]:
    workers = sorted(workforce.workers, key=lambda item: (item.sort_order or 0, item.id or 0))
    return {
        "id": workforce.id,
        "name": workforce.name,
        "description": workforce.description,
        "status": workforce.status,
        "manager": _serialize_agent(workforce.manager_agent),
        "manager_instructions": workforce.manager_instructions,
        "workers": [_serialize_worker(worker) for worker in workers],
        "canvas_layout": workforce.canvas_layout,
        "scope_type": workforce.scope_type,
        "scope_id": workforce.scope_id,
        "owner_user_id": workforce.owner_user_id,
        "created_at": workforce.created_at.isoformat() if workforce.created_at else None,
        "updated_at": workforce.updated_at.isoformat() if workforce.updated_at else None,
    }


def _serialize_workforce_list_item(db: Session, workforce: Workforce) -> dict[str, Any]:
    last_run = (
        db.query(WorkforceRun)
        .filter(WorkforceRun.workforce_id == workforce.id)
        .order_by(WorkforceRun.created_at.desc(), WorkforceRun.id.desc())
        .first()
    )
    return {
        "id": workforce.id,
        "name": workforce.name,
        "description": workforce.description,
        "status": workforce.status,
        "manager": {
            "id": workforce.manager_agent.id,
            "name": workforce.manager_agent.name,
            "logo_url": workforce.manager_agent.logo_url,
        },
        "worker_count": len(workforce.workers),
        "last_run": (
            {
                "id": last_run.id,
                "task_id": last_run.task_id,
                "status": last_run.status,
                "created_at": last_run.created_at.isoformat() if last_run.created_at else None,
            }
            if last_run
            else None
        ),
        "created_at": workforce.created_at.isoformat() if workforce.created_at else None,
        "updated_at": workforce.updated_at.isoformat() if workforce.updated_at else None,
    }


def _check_duplicate_workforce_name(
    db: Session,
    scope_type: str,
    scope_id: str,
    name: str,
    exclude_workforce_id: int | None = None,
) -> None:
    query = db.query(Workforce).filter(
        Workforce.scope_type == scope_type,
        Workforce.scope_id == scope_id,
        Workforce.name == name,
    )
    if exclude_workforce_id is not None:
        query = query.filter(Workforce.id != exclude_workforce_id)
    if query.first():
        raise HTTPException(status_code=409, detail="Workforce name already exists")


def _validate_worker_agent_ids(workers: list[WorkforceWorkerInput], manager_agent_id: int) -> None:
    seen_agent_ids: set[int] = set()
    for worker in workers:
        ensure_supported_source_type(worker.source_type)
        if worker.source_type == "existing":
            if worker.agent_id is None:
                raise HTTPException(status_code=400, detail="agent_id is required")
            if worker.agent_id == manager_agent_id:
                raise HTTPException(status_code=400, detail="Manager agent cannot also be a worker")
            if worker.agent_id in seen_agent_ids:
                raise HTTPException(status_code=409, detail="Duplicate worker agent in workforce")
            seen_agent_ids.add(worker.agent_id)
        elif worker.source_type == "template":
            if not worker.template_id:
                raise HTTPException(status_code=400, detail="template_id is required")
        elif worker.source_type == "new":
            if worker.agent is None:
                raise HTTPException(
                    status_code=400, detail="agent is required for source_type='new'"
                )
            new_agent_name = normalize_text(worker.agent.name, "agent.name", required=True)
            if new_agent_name is None:
                raise HTTPException(status_code=400, detail="agent.name is required")


def _ensure_can_activate(status: str, workforce: Workforce | None, workers: list[Any]) -> None:
    if status != "active":
        return
    enabled_count = 0
    if workforce is not None:
        enabled_count = sum(1 for worker in workforce.workers if worker.enabled)
    else:
        enabled_count = sum(1 for worker in workers if getattr(worker, "enabled", True))
    if enabled_count == 0:
        raise HTTPException(
            status_code=400, detail="Active workforce requires at least one enabled worker"
        )


def _create_worker_row(
    db: Session,
    workforce: Workforce,
    worker_input: WorkforceWorkerInput,
    user: User,
) -> WorkforceAgent:
    return create_workforce_worker(
        db,
        workforce,
        user,
        source_type=worker_input.source_type,
        assignment_instructions=worker_input.assignment_instructions,
        alias=worker_input.alias,
        agent_id=worker_input.agent_id,
        template_id=worker_input.template_id,
        agent_payload=worker_input.agent.model_dump() if worker_input.agent else None,
        enabled=worker_input.enabled,
        sort_order=worker_input.sort_order,
        canvas_position=worker_input.canvas_position,
    )


@router.get("")
async def list_workforces(
    search: str = "",
    page: int = 1,
    size: int = 20,
    status: str | None = None,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> dict[str, Any]:
    if page < 1 or size < 1 or size > 100:
        raise HTTPException(status_code=400, detail="Invalid pagination parameters")

    query = db.query(Workforce)
    if search:
        query = query.filter(
            or_(Workforce.name.ilike(f"%{search}%"), Workforce.description.ilike(f"%{search}%"))
        )
    if status:
        query = query.filter(Workforce.status == normalize_workforce_status(status))

    items = query.order_by(Workforce.updated_at.desc(), Workforce.id.desc()).all()
    if not user.is_admin:
        items = [workforce for workforce in items if can_view_workforce(db, user, workforce)]
    total = len(items)
    paged_items = items[(page - 1) * size : (page - 1) * size + size]
    return {
        "items": [_serialize_workforce_list_item(db, workforce) for workforce in paged_items],
        "total": total,
        "page": page,
        "size": size,
        "pages": (total + size - 1) // size,
    }


@router.get("/templates")
async def list_workforce_templates(
    user: User = Depends(get_current_user),
) -> list[dict[str, Any]]:
    _ = user
    return list_template_summaries()


@router.post("")
async def create_workforce(
    request: WorkforceCreateRequest,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> dict[str, Any]:
    name = normalize_text(request.name, "name", required=True)
    if name is None:
        raise HTTPException(status_code=400, detail="name is required")

    scope_type, scope_id = resolve_create_scope(db, user)
    if not can_create_workforce(db, user, scope_type, scope_id):
        raise HTTPException(status_code=403, detail="Access denied")

    manager_agent = ensure_agent_access(
        db.query(Agent).filter(Agent.id == request.manager_agent_id).first(),
        user,
        db,
    )
    _check_duplicate_workforce_name(db, scope_type, scope_id, name)
    _validate_worker_agent_ids(request.workers, manager_agent.id)
    status = normalize_workforce_status(request.status)
    _ensure_can_activate(status, None, request.workers)

    workforce = Workforce(
        owner_user_id=user.id,
        scope_type=scope_type,
        scope_id=scope_id,
        name=name,
        description=normalize_text(request.description, "description"),
        manager_agent_id=manager_agent.id,
        manager_instructions=normalize_text(request.manager_instructions, "manager_instructions"),
        status=status,
        canvas_layout=request.canvas_layout,
    )
    db.add(workforce)
    db.flush()

    for worker_input in request.workers:
        _create_worker_row(db, workforce, worker_input, user)

    db.commit()
    db.refresh(workforce)
    return _serialize_workforce_detail(workforce)


@router.get("/{workforce_id}")
async def get_workforce(
    workforce_id: int,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> dict[str, Any]:
    workforce = db.query(Workforce).filter(Workforce.id == workforce_id).first()
    workforce = ensure_workforce_access(db, user, workforce, action="view")
    return _serialize_workforce_detail(workforce)


@router.get("/{workforce_id}/builder/messages")
async def get_workforce_builder_messages(
    workforce_id: int,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> dict[str, Any]:
    workforce = db.query(Workforce).filter(Workforce.id == workforce_id).first()
    workforce = ensure_workforce_access(db, user, workforce, action="view")
    messages = list_builder_messages(db, cast(int, workforce.id))
    return {"items": [serialize_builder_message(message) for message in messages]}


@router.patch("/{workforce_id}")
async def update_workforce(
    workforce_id: int,
    request: WorkforceUpdateRequest,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> dict[str, Any]:
    workforce = db.query(Workforce).filter(Workforce.id == workforce_id).first()
    workforce = ensure_workforce_access(db, user, workforce, action="edit")

    if request.name is not None:
        name = normalize_text(request.name, "name", required=True)
        if name is None:
            raise HTTPException(status_code=400, detail="name is required")
        if name != workforce.name:
            _check_duplicate_workforce_name(
                db,
                cast(str, workforce.scope_type),
                cast(str, workforce.scope_id),
                name,
                cast(int, workforce.id),
            )
            workforce.name = name

    if request.description is not None:
        workforce.description = normalize_text(request.description, "description")
    if request.manager_instructions is not None:
        workforce.manager_instructions = normalize_text(
            request.manager_instructions,
            "manager_instructions",
        )
    if request.canvas_layout is not None:
        workforce.canvas_layout = request.canvas_layout
    if request.manager_agent_id is not None:
        manager_agent = ensure_agent_access(
            db.query(Agent).filter(Agent.id == request.manager_agent_id).first(),
            user,
            db,
        )
        if any(worker.agent_id == manager_agent.id for worker in workforce.workers):
            raise HTTPException(status_code=400, detail="Manager agent cannot also be a worker")
        workforce.manager_agent_id = manager_agent.id
    if request.status is not None:
        status = normalize_workforce_status(request.status)
        _ensure_can_activate(status, workforce, [])
        workforce.status = status

    db.commit()
    db.refresh(workforce)
    return _serialize_workforce_detail(workforce)


@router.delete("/{workforce_id}")
async def archive_workforce(
    workforce_id: int,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> dict[str, Any]:
    workforce = db.query(Workforce).filter(Workforce.id == workforce_id).first()
    workforce = ensure_workforce_access(db, user, workforce, action="edit")
    workforce.status = "archived"
    db.commit()
    return {"id": workforce.id, "status": workforce.status}


@router.post("/{workforce_id}/agents")
async def add_workforce_agent(
    workforce_id: int,
    request: WorkforceWorkerInput,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> dict[str, Any]:
    workforce = db.query(Workforce).filter(Workforce.id == workforce_id).first()
    workforce = ensure_workforce_access(db, user, workforce, action="edit")
    worker = _create_worker_row(db, workforce, request, user)
    workforce_status = cast(str, workforce.status)
    if workforce_status == "active" and not worker.enabled:
        _ensure_can_activate(workforce_status, workforce, [])
    db.commit()
    db.refresh(worker)
    return _serialize_worker(worker)


@router.patch("/{workforce_id}/agents/{member_id}")
async def update_workforce_agent(
    workforce_id: int,
    member_id: int,
    request: WorkforceWorkerUpdateRequest,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> dict[str, Any]:
    workforce = db.query(Workforce).filter(Workforce.id == workforce_id).first()
    workforce = ensure_workforce_access(db, user, workforce, action="edit")
    worker = (
        db.query(WorkforceAgent)
        .filter(WorkforceAgent.id == member_id, WorkforceAgent.workforce_id == workforce.id)
        .first()
    )
    if worker is None:
        raise HTTPException(status_code=404, detail="Workforce worker not found")

    if request.alias is not None:
        worker.alias = normalize_text(request.alias, "alias")
    if request.assignment_instructions is not None:
        worker.assignment_instructions = (
            normalize_text(
                request.assignment_instructions,
                "assignment_instructions",
                required=True,
            )
            or ""
        )
    if request.enabled is not None:
        worker.enabled = bool(request.enabled)
    if request.sort_order is not None:
        worker.sort_order = request.sort_order
    if request.canvas_position is not None:
        worker.canvas_position = request.canvas_position

    _ensure_can_activate(cast(str, workforce.status), workforce, [])
    db.commit()
    db.refresh(worker)
    return _serialize_worker(worker)


@router.delete("/{workforce_id}/agents/{member_id}")
async def remove_workforce_agent(
    workforce_id: int,
    member_id: int,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> dict[str, Any]:
    workforce = db.query(Workforce).filter(Workforce.id == workforce_id).first()
    workforce = ensure_workforce_access(db, user, workforce, action="edit")
    worker = (
        db.query(WorkforceAgent)
        .filter(WorkforceAgent.id == member_id, WorkforceAgent.workforce_id == workforce.id)
        .first()
    )
    if worker is None:
        raise HTTPException(status_code=404, detail="Workforce worker not found")

    db.delete(worker)
    db.flush()
    _ensure_can_activate(cast(str, workforce.status), workforce, [])
    db.commit()
    return {"status": "deleted"}


@router.post("/{workforce_id}/runs")
async def create_workforce_run(
    workforce_id: int,
    request: WorkforceRunRequest,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> dict[str, Any]:
    workforce = db.query(Workforce).filter(Workforce.id == workforce_id).first()
    workforce = ensure_workforce_access(db, user, workforce, action="run")
    get_workforce_policy().before_workforce_run(db, user, workforce)

    snapshot = build_workforce_snapshot(db, user, workforce)
    message = normalize_text(request.message, "message", required=True)
    if message is None:
        raise HTTPException(status_code=400, detail="message is required")

    task_title = f"{workforce.name}: {message}"
    if len(task_title) > 50:
        task_title = task_title[:50] + "..."

    task = Task(
        user_id=user.id,
        title=task_title,
        description=message,
        status=TaskStatus.PENDING,
        agent_id=workforce.manager_agent_id,
        agent_config=build_workforce_task_config(snapshot, selected_file_ids=request.files),
        execution_mode=request.execution_mode
        or workforce.manager_agent.execution_mode
        or "balanced",
    )
    db.add(task)
    db.flush()

    workforce_run = WorkforceRun(
        workforce_id=workforce.id,
        task_id=task.id,
        user_id=user.id,
        status="pending",
        snapshot=snapshot,
    )
    db.add(workforce_run)
    db.flush()

    task.agent_config = build_workforce_task_config(
        snapshot,
        selected_file_ids=request.files,
        workforce_run_id=cast(int, workforce_run.id),
    )
    get_workforce_policy().after_workforce_run_created(
        db,
        user,
        workforce,
        workforce_run,
        task,
    )
    db.commit()
    db.refresh(workforce_run)
    db.refresh(task)

    return {
        "workforce_run_id": workforce_run.id,
        "task_id": task.id,
        "status": workforce_run.status,
        "redirect_url": f"/task/{task.id}",
    }


@router.post("/{workforce_id}/builder/propose")
async def propose_workforce_changes(
    workforce_id: int,
    request: WorkforceBuilderProposeRequest,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> dict[str, Any]:
    workforce = db.query(Workforce).filter(Workforce.id == workforce_id).first()
    workforce = ensure_workforce_access(db, user, workforce, action="edit")
    user_message = normalize_text(request.message, "message", required=True)
    if user_message is None:
        raise HTTPException(status_code=400, detail="message is required")

    db.add(
        WorkforceBuilderMessage(
            workforce_id=workforce.id,
            user_id=user.id,
            role="user",
            content=user_message,
            status="message",
        )
    )
    assistant_message, patch = await generate_builder_patch(db, user, workforce, user_message)
    assistant_row = WorkforceBuilderMessage(
        workforce_id=workforce.id,
        user_id=user.id,
        role="assistant",
        content=assistant_message,
        proposed_patch=patch,
        status="proposed",
    )
    db.add(assistant_row)
    db.commit()
    db.refresh(assistant_row)

    return {
        "message_id": assistant_row.id,
        "assistant_message": assistant_row.content,
        "proposed_patch": assistant_row.proposed_patch,
        "requires_confirmation": True,
    }


@router.post("/{workforce_id}/builder/apply")
async def apply_workforce_changes(
    workforce_id: int,
    request: WorkforceBuilderApplyRequest,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> dict[str, Any]:
    workforce = db.query(Workforce).filter(Workforce.id == workforce_id).first()
    workforce = ensure_workforce_access(db, user, workforce, action="edit")
    message = (
        db.query(WorkforceBuilderMessage)
        .filter(
            WorkforceBuilderMessage.id == request.message_id,
            WorkforceBuilderMessage.workforce_id == workforce.id,
        )
        .first()
    )
    if message is None:
        raise HTTPException(status_code=404, detail="Builder message not found")
    if message.role != "assistant":
        raise HTTPException(status_code=400, detail="Builder message is not applicable")
    if message.status != "proposed" or not isinstance(message.proposed_patch, dict):
        raise HTTPException(status_code=400, detail="Builder message has no pending patch")
    if request.proposed_patch != message.proposed_patch:
        raise HTTPException(status_code=400, detail="Proposed patch does not match message")

    workforce = apply_builder_patch(db, user, workforce, request.proposed_patch)
    message.status = "applied"
    message.proposed_patch = request.proposed_patch
    db.commit()
    db.refresh(workforce)
    db.refresh(message)
    return {
        "status": "applied",
        "message_id": message.id,
        "workforce": _serialize_workforce_detail(workforce),
    }


@router.get("/{workforce_id}/canvas")
async def get_workforce_canvas(
    workforce_id: int,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> dict[str, Any]:
    workforce = db.query(Workforce).filter(Workforce.id == workforce_id).first()
    workforce = ensure_workforce_access(db, user, workforce, action="view")
    workers = sorted(workforce.workers, key=lambda item: (item.sort_order or 0, item.id or 0))

    nodes = [
        {"id": "human", "type": "human", "label": "Human"},
        {
            "id": f"manager-{workforce.manager_agent.id}",
            "type": "manager",
            "agent_id": workforce.manager_agent.id,
            "label": workforce.manager_agent.name,
        },
    ]
    edges = [
        {
            "id": "human-manager",
            "source": "human",
            "target": f"manager-{workforce.manager_agent.id}",
        }
    ]

    for worker in workers:
        label = worker.alias or worker.agent.name
        nodes.append(
            {
                "id": f"worker-{worker.id}",
                "type": "worker",
                "agent_id": worker.agent_id,
                "label": label,
                "position": worker.canvas_position,
                "enabled": worker.enabled,
            }
        )
        edges.append(
            {
                "id": f"manager-worker-{worker.id}",
                "source": f"manager-{workforce.manager_agent.id}",
                "target": f"worker-{worker.id}",
            }
        )

    return {"nodes": nodes, "edges": edges, "layout": workforce.canvas_layout or {}}
