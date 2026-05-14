import re
from pathlib import Path
from typing import Any, cast

from fastapi import HTTPException
from sqlalchemy.orm import Session

from xagent.templates.utils import create_template_manager
from xagent.web.models.agent import Agent, AgentStatus
from xagent.web.models.user import User

from ..models.workforce import Workforce, WorkforceAgent
from .workforce_access import ensure_agent_access
from .workforce_snapshot import normalize_text

_TEMPLATE_MANAGER = create_template_manager()

_SAFE_TEMPLATE_ID_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_.-]*$")
_MAX_TEMPLATE_ID_LENGTH = 200


def _validate_template_id(template_id: str) -> None:
    if len(template_id) > _MAX_TEMPLATE_ID_LENGTH:
        raise HTTPException(status_code=400, detail="template_id too long")
    if not _SAFE_TEMPLATE_ID_RE.fullmatch(template_id):
        raise HTTPException(
            status_code=400, detail=f"Invalid template_id: {template_id}"
        )


def ensure_supported_source_type(source_type: str) -> None:
    if source_type != "existing":
        raise HTTPException(
            status_code=400,
            detail="source_type must be existing; publish an agent before adding it to a workforce",
        )


def normalize_execution_mode(value: str | None) -> str:
    normalized = str(value or "balanced").strip().lower()
    if normalized not in {"flash", "balanced", "think"}:
        raise HTTPException(
            status_code=400, detail="execution_mode must be flash, balanced, or think"
        )
    return normalized


def normalize_string_list(values: list[str] | None) -> list[str]:
    normalized: list[str] = []
    for value in values or []:
        text = str(value or "").strip()
        if text:
            normalized.append(text)
    return normalized


def _resolve_unique_agent_name(db: Session, user: User, name: str) -> str:
    normalized_name = normalize_text(name, "name", required=True)
    if normalized_name is None:
        raise HTTPException(status_code=400, detail="name is required")

    existing = (
        db.query(Agent)
        .filter(Agent.user_id == user.id, Agent.name == normalized_name)
        .first()
    )
    if existing is None:
        return normalized_name

    base_name = normalized_name
    suffix = 2
    while True:
        suffix_text = f" {suffix}"
        candidate_base = base_name[: max(1, 200 - len(suffix_text))].rstrip()
        candidate = f"{candidate_base}{suffix_text}"
        conflict = (
            db.query(Agent)
            .filter(Agent.user_id == user.id, Agent.name == candidate)
            .first()
        )
        if conflict is None:
            return candidate
        suffix += 1


def create_agent_record(
    db: Session,
    user: User,
    *,
    name: str,
    description: str | None = None,
    instructions: str | None = None,
    execution_mode: str | None = "balanced",
    models: dict[str, Any] | None = None,
    knowledge_bases: list[str] | None = None,
    skills: list[str] | None = None,
    tool_categories: list[str] | None = None,
    suggested_prompts: list[str] | None = None,
    ensure_unique_name: bool = False,
    status: AgentStatus = AgentStatus.DRAFT,
) -> Agent:
    normalized_name = normalize_text(name, "name", required=True)
    if normalized_name is None:
        raise HTTPException(status_code=400, detail="name is required")

    if ensure_unique_name:
        final_name = _resolve_unique_agent_name(db, user, normalized_name)
    else:
        existing = (
            db.query(Agent)
            .filter(Agent.user_id == user.id, Agent.name == normalized_name)
            .first()
        )
        if existing:
            raise HTTPException(
                status_code=409, detail="Agent with this name already exists"
            )
        final_name = normalized_name

    agent = Agent(
        user_id=user.id,
        name=final_name,
        description=normalize_text(description, "description"),
        instructions=normalize_text(instructions, "instructions"),
        execution_mode=normalize_execution_mode(execution_mode),
        models=models,
        knowledge_bases=normalize_string_list(knowledge_bases),
        skills=normalize_string_list(skills),
        tool_categories=normalize_string_list(tool_categories),
        suggested_prompts=normalize_string_list(suggested_prompts),
        status=status,
        widget_enabled=True,
        allowed_domains=[],
    )
    db.add(agent)
    db.flush()
    return agent


def load_template_detail(template_id: str) -> dict[str, Any]:
    _validate_template_id(template_id)
    yaml_path = Path(_TEMPLATE_MANAGER.templates_root) / f"{template_id}.yaml"
    if not yaml_path.exists():
        raise HTTPException(status_code=404, detail="Template not found")

    template = _TEMPLATE_MANAGER._parse_yaml_file(yaml_path)
    enriched = _TEMPLATE_MANAGER._enrich_template(template)
    descriptions = enriched.get("descriptions") or {}
    if isinstance(descriptions, dict):
        description = descriptions.get("en") or next(iter(descriptions.values()), None)
    else:
        description = descriptions

    return {
        **enriched,
        "description": normalize_text(description, "description"),
    }


def list_template_summaries() -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    templates_root = Path(_TEMPLATE_MANAGER.templates_root)
    for yaml_path in sorted(templates_root.glob("*.yaml")):
        template = _TEMPLATE_MANAGER._parse_yaml_file(yaml_path)
        enriched = _TEMPLATE_MANAGER._enrich_template(template)
        descriptions = enriched.get("descriptions") or {}
        if isinstance(descriptions, dict):
            description = descriptions.get("en") or next(
                iter(descriptions.values()), None
            )
        else:
            description = descriptions
        results.append(
            {
                "id": enriched.get("id"),
                "name": enriched.get("name"),
                "description": normalize_text(description, "description"),
            }
        )
    return results


def create_agent_from_template(
    db: Session,
    user: User,
    template_id: str,
) -> Agent:
    template = load_template_detail(template_id)
    agent_config = template.get("agent_config") or {}
    return create_agent_record(
        db,
        user,
        name=template.get("name") or template_id,
        description=template.get("description"),
        instructions=agent_config.get("instructions"),
        execution_mode=agent_config.get("execution_mode"),
        models=agent_config.get("models"),
        knowledge_bases=agent_config.get("knowledge_bases") or [],
        skills=agent_config.get("skills") or [],
        tool_categories=agent_config.get("tool_categories") or [],
        suggested_prompts=agent_config.get("suggested_prompts") or [],
        ensure_unique_name=True,
    )


def next_worker_sort_order(db: Session, workforce_id: int) -> int:
    max_sort_order = (
        db.query(WorkforceAgent.sort_order)
        .filter(WorkforceAgent.workforce_id == workforce_id)
        .order_by(WorkforceAgent.sort_order.desc(), WorkforceAgent.id.desc())
        .first()
    )
    return (
        int(max_sort_order[0]) + 1
        if max_sort_order and max_sort_order[0] is not None
        else 1
    )


def create_workforce_worker(
    db: Session,
    workforce: Workforce,
    user: User,
    *,
    source_type: str,
    assignment_instructions: str,
    alias: str | None = None,
    agent_id: int | None = None,
    template_id: str | None = None,
    agent_payload: dict[str, Any] | None = None,
    enabled: bool = True,
    sort_order: int | None = None,
    canvas_position: dict[str, Any] | None = None,
) -> WorkforceAgent:
    ensure_supported_source_type(source_type)

    normalized_assignment = normalize_text(
        assignment_instructions,
        "assignment_instructions",
        required=True,
    )
    if normalized_assignment is None:
        raise HTTPException(
            status_code=400, detail="assignment_instructions is required"
        )

    if source_type == "existing":
        if agent_id is None:
            raise HTTPException(status_code=400, detail="agent_id is required")
        agent = ensure_agent_access(
            db.query(Agent).filter(Agent.id == agent_id).first(),
            user,
            db,
            require_published=True,
        )
        existing = (
            db.query(WorkforceAgent)
            .filter(
                WorkforceAgent.workforce_id == workforce.id,
                WorkforceAgent.agent_id == agent.id,
            )
            .first()
        )
        if existing:
            raise HTTPException(
                status_code=409, detail="Agent already added to workforce"
            )
    elif source_type == "template":
        if not template_id:
            raise HTTPException(status_code=400, detail="template_id is required")
        agent = create_agent_from_template(db, user, template_id)
    else:
        if not isinstance(agent_payload, dict):
            raise HTTPException(
                status_code=400, detail="agent is required for source_type='new'"
            )
        agent = create_agent_record(
            db,
            user,
            name=str(agent_payload.get("name") or ""),
            description=agent_payload.get("description"),
            instructions=agent_payload.get("instructions"),
            execution_mode=agent_payload.get("execution_mode"),
            models=agent_payload.get("models"),
            knowledge_bases=agent_payload.get("knowledge_bases"),
            skills=agent_payload.get("skills"),
            tool_categories=agent_payload.get("tool_categories"),
            suggested_prompts=agent_payload.get("suggested_prompts"),
            ensure_unique_name=True,
        )

    agent_id_value = cast(int, agent.id)
    workforce_manager_id = cast(int, workforce.manager_agent_id)
    if agent_id_value == workforce_manager_id:
        raise HTTPException(
            status_code=400, detail="Manager agent cannot also be a worker"
        )

    workforce_id = cast(int, workforce.id)
    worker = WorkforceAgent(
        workforce_id=workforce_id,
        agent_id=agent_id_value,
        alias=normalize_text(alias, "alias"),
        assignment_instructions=normalized_assignment,
        source_type=source_type,
        template_id=template_id,
        enabled=bool(enabled),
        sort_order=sort_order
        if sort_order is not None
        else next_worker_sort_order(db, workforce_id),
        canvas_position=canvas_position,
    )
    db.add(worker)
    db.flush()
    return worker
