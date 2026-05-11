import json
import logging
import os
import re
from typing import Any, cast

from fastapi import HTTPException
from sqlalchemy.orm import Session
from xagent.web.models.agent import Agent
from xagent.web.models.user import User
from xagent.web.services.llm_utils import UserAwareModelStorage

from ..models.workforce import Workforce, WorkforceAgent, WorkforceBuilderMessage
from .workforce_access import ensure_agent_access, ensure_workforce_access
from .workforce_snapshot import (
    normalize_text,
    normalize_workforce_status,
)
from .workforce_workers import create_workforce_worker, list_template_summaries

logger = logging.getLogger(__name__)

SUPPORTED_BUILDER_OPS = {
    "update_workforce",
    "add_existing_worker",
    "add_worker_from_template",
    "create_worker_agent",
    "update_worker",
    "remove_worker",
}

PLACEHOLDER_PATCH_WARNING = (
    "Could not confidently infer a structured change. "
    "Please refine the instruction or edit manually."
)


def serialize_builder_message(message: WorkforceBuilderMessage) -> dict[str, Any]:
    return {
        "id": message.id,
        "role": message.role,
        "content": message.content,
        "status": message.status,
        "proposed_patch": message.proposed_patch,
        "created_at": message.created_at.isoformat() if message.created_at else None,
    }


def list_builder_messages(db: Session, workforce_id: int) -> list[WorkforceBuilderMessage]:
    return (
        db.query(WorkforceBuilderMessage)
        .filter(WorkforceBuilderMessage.workforce_id == workforce_id)
        .order_by(WorkforceBuilderMessage.id.asc())
        .all()
    )


def _serialize_workers_for_prompt(workforce: Workforce) -> list[dict[str, Any]]:
    workers = sorted(workforce.workers, key=lambda item: (item.sort_order or 0, item.id or 0))
    return [
        {
            "member_id": worker.id,
            "agent_id": worker.agent_id,
            "agent_name": worker.agent.name,
            "alias": worker.alias,
            "assignment_instructions": worker.assignment_instructions,
            "enabled": worker.enabled,
            "sort_order": worker.sort_order,
        }
        for worker in workers
    ]


def _make_builder_context(workforce: Workforce) -> dict[str, Any]:
    return {
        "workforce": {
            "id": workforce.id,
            "name": workforce.name,
            "description": workforce.description,
            "status": workforce.status,
            "manager_agent_id": workforce.manager_agent_id,
            "manager_agent_name": workforce.manager_agent.name if workforce.manager_agent else None,
            "manager_instructions": workforce.manager_instructions,
        },
        "workers": _serialize_workers_for_prompt(workforce),
    }


def _clean_patch(candidate: dict[str, Any]) -> dict[str, Any]:
    summary = str(candidate.get("summary") or "").strip() or "Update workforce configuration."
    warnings = candidate.get("warnings")
    if not isinstance(warnings, list):
        warnings = []
    clean_warnings = [str(item).strip() for item in warnings if str(item).strip()]

    operations = candidate.get("operations")
    if not isinstance(operations, list):
        operations = []

    clean_operations: list[dict[str, Any]] = []
    for operation in operations:
        if not isinstance(operation, dict):
            continue
        op_name = str(operation.get("op") or "").strip()
        if op_name not in SUPPORTED_BUILDER_OPS:
            continue
        clean_operations.append(operation)

    return {
        "summary": summary,
        "operations": clean_operations,
        "warnings": clean_warnings,
    }


def _has_meaningful_operations(patch: dict[str, Any]) -> bool:
    operations = patch.get("operations")
    if not isinstance(operations, list) or not operations:
        return False
    if len(operations) != 1:
        return True

    only_operation = operations[0]
    if only_operation.get("op") != "update_workforce":
        return True

    fields = only_operation.get("fields")
    return isinstance(fields, dict) and bool(fields)


def _fallback_patch_from_message(workforce: Workforce, message: str) -> dict[str, Any]:
    text = message.strip()
    lower = text.lower()
    operations: list[dict[str, Any]] = []
    warnings: list[str] = []
    summary_parts: list[str] = []

    quoted_texts = re.findall(r'"([^"]+)"', text)
    workforce_name_target = quoted_texts[0].strip() if quoted_texts else None

    rename_match = re.search(r"\brename\b|\brenamed\b|\bname it\b|\bcall it\b", lower)
    if rename_match and workforce_name_target:
        operations.append(
            {
                "op": "update_workforce",
                "fields": {"name": workforce_name_target},
            }
        )
        summary_parts.append(f'Rename workforce to "{workforce_name_target}"')

    if (
        ("description" in lower or "desc" in lower)
        and workforce_name_target
        and len(quoted_texts) >= 2
    ):
        operations.append(
            {
                "op": "update_workforce",
                "fields": {"description": quoted_texts[1].strip()},
            }
        )
        summary_parts.append("Update workforce description")

    manager_instructions_match = re.search(
        r"(manager instructions?|manager prompt)\s*(?:to|as|=)?\s*\"([^\"]+)\"",
        text,
        flags=re.IGNORECASE,
    )
    if manager_instructions_match:
        operations.append(
            {
                "op": "update_workforce",
                "fields": {"manager_instructions": manager_instructions_match.group(2).strip()},
            }
        )
        summary_parts.append("Update manager instructions")

    remove_match = re.search(r"\bremove\b\s+([a-zA-Z0-9 _-]+)", text, flags=re.IGNORECASE)
    if remove_match:
        raw_target = remove_match.group(1).strip().rstrip(".")
        matched_worker = _find_worker_by_name(workforce, raw_target)
        if matched_worker is not None:
            operations.append({"op": "remove_worker", "member_id": matched_worker.id})
            warnings.append(
                f'Removing worker "{matched_worker.alias or matched_worker.agent.name}".'
            )
            summary_parts.append(
                f'Remove worker "{matched_worker.alias or matched_worker.agent.name}"'
            )

    update_match = re.search(
        (
            r"(?:make|update|change)\s+([a-zA-Z0-9 _-]+?)\s+"
            r"(?:focus on|to handle|to work on|to)\s+(.+)"
        ),
        text,
        flags=re.IGNORECASE,
    )
    if update_match:
        target_name = update_match.group(1).strip()
        instructions = update_match.group(2).strip().rstrip(".")
        matched_worker = _find_worker_by_name(workforce, target_name)
        if matched_worker is not None and instructions:
            operations.append(
                {
                    "op": "update_worker",
                    "member_id": matched_worker.id,
                    "assignment_instructions": instructions,
                }
            )
            summary_parts.append(
                f'Update worker "{matched_worker.alias or matched_worker.agent.name}"'
            )

    if "add" in lower and "worker" in lower and ("template" in lower or "based on" in lower):
        template = _find_template_candidate_for_message(text)
        instructions = _extract_assignment_instructions(text)
        if template is not None and instructions:
            alias = _extract_alias_after_add_worker(text)
            operations.append(
                {
                    "op": "add_worker_from_template",
                    "template_id": template["id"],
                    "alias": alias or template["name"],
                    "assignment_instructions": instructions,
                }
            )
            summary_parts.append(f'Add worker from template "{template["name"]}"')

    if "add" in lower and "worker" in lower and ("new agent" in lower or "create agent" in lower):
        instructions = _extract_assignment_instructions(text)
        agent_name = _extract_new_agent_name(text)
        if agent_name and instructions:
            operations.append(
                {
                    "op": "create_worker_agent",
                    "agent": {
                        "name": agent_name,
                        "description": f"{agent_name} created by Workforce Builder.",
                        "instructions": instructions,
                        "execution_mode": "balanced",
                        "tool_categories": ["basic"],
                    },
                    "alias": agent_name,
                    "assignment_instructions": instructions,
                }
            )
            summary_parts.append(f'Create new worker agent "{agent_name}"')

    if "add" in lower and "worker" in lower:
        agent = _find_agent_candidate_for_message(workforce, text)
        instructions = _extract_assignment_instructions(text)
        if agent is not None and instructions:
            operations.append(
                {
                    "op": "add_existing_worker",
                    "agent_id": agent.id,
                    "alias": agent.name,
                    "assignment_instructions": instructions,
                }
            )
            summary_parts.append(f'Add worker "{agent.name}"')

    if not operations:
        operations.append(
            {
                "op": "update_workforce",
                "fields": {},
            }
        )
        warnings.append(PLACEHOLDER_PATCH_WARNING)
        summary_parts.append("No safe structured changes inferred")

    return {
        "summary": ". ".join(summary_parts) + ".",
        "operations": operations,
        "warnings": warnings,
    }


def _find_worker_by_name(workforce: Workforce, raw_name: str) -> WorkforceAgent | None:
    target = raw_name.strip().lower()
    if not target:
        return None
    for raw_worker in workforce.workers:
        worker = cast(WorkforceAgent, raw_worker)
        candidates = [
            worker.alias or "",
            worker.agent.name if worker.agent else "",
        ]
        for candidate in candidates:
            if candidate and candidate.lower() == target:
                return worker
    for raw_worker in workforce.workers:
        worker = cast(WorkforceAgent, raw_worker)
        candidates = [
            worker.alias or "",
            worker.agent.name if worker.agent else "",
        ]
        for candidate in candidates:
            if candidate and target in candidate.lower():
                return worker
    return None


def _find_agent_candidate_for_message(workforce: Workforce, message: str) -> Agent | None:
    lower = message.lower()
    owner_id = workforce.owner_user_id
    db = workforce._sa_instance_state.session
    if db is None:
        return None
    agents = db.query(Agent).filter(Agent.user_id == owner_id).order_by(Agent.id.asc()).all()
    existing_agent_ids = {worker.agent_id for worker in workforce.workers}
    manager_id = workforce.manager_agent_id
    for agent in agents:
        if agent.id in existing_agent_ids or agent.id == manager_id:
            continue
        if agent.name.lower() in lower:
            return agent
    return None


def _find_template_candidate_for_message(message: str) -> dict[str, Any] | None:
    lower = message.lower()
    for template in list_template_summaries():
        template_name = str(template.get("name") or "")
        template_id = str(template.get("id") or "")
        if template_name.lower() in lower or template_id.lower() in lower:
            return template
    return None


def _extract_alias_after_add_worker(message: str) -> str | None:
    match = re.search(
        r"add\s+(?:a\s+)?worker\s+(?:named|called)\s+\"?([a-zA-Z0-9 _-]+)\"?",
        message,
        flags=re.IGNORECASE,
    )
    if match:
        alias = match.group(1).strip()
        return alias or None
    return None


def _extract_new_agent_name(message: str) -> str | None:
    quoted = [str(item).strip() for item in re.findall(r'"([^"]+)"', message)]
    if quoted:
        return quoted[0] or None
    match = re.search(
        r"(?:new agent|create agent)\s+([a-zA-Z0-9 _-]+?)(?:\s+to\s+|\s+for\s+|$)",
        message,
        flags=re.IGNORECASE,
    )
    if match:
        candidate = match.group(1).strip()
        return candidate or None
    return None


def _extract_assignment_instructions(message: str) -> str | None:
    quoted = [str(item).strip() for item in re.findall(r'"([^"]+)"', message)]
    if quoted:
        return quoted[-1] or None

    match = re.search(
        r"(?:to|for)\s+(?:handle|focus on|work on|cover)\s+(.+)$",
        message,
        flags=re.IGNORECASE,
    )
    if match:
        return match.group(1).strip().rstrip(".")
    return None


async def _llm_generate_patch(
    db: Session,
    user: User,
    workforce: Workforce,
    message: str,
) -> dict[str, Any] | None:
    try:
        storage = UserAwareModelStorage(db)
        user_id = int(user.id)
        llm = None
        default_llm, _, _, _ = storage.get_configured_defaults(user_id)
        llm = default_llm
        if not llm:
            default_llm, _, _, _ = storage.get_configured_defaults(None)
            llm = default_llm
        if not llm:
            return None

        system_prompt = (
            "You are a Workforce Builder assistant. "
            "Team means human organization and must never be modified. "
            "Workforce means AI orchestration and may be modified only at the relationship layer. "
            "You may only output JSON for a proposed patch. "
            "You can modify workforce name, description, manager instructions, "
            "worker membership, worker alias, worker assignment instructions, "
            "and worker order. "
            "Do not modify underlying agent instructions, models, tools, skills, "
            "or knowledge bases. "
            "Supported operations are: update_workforce, add_existing_worker, "
            "add_worker_from_template, create_worker_agent, update_worker, remove_worker. "
            "For destructive operations like remove_worker, include a warning. "
            "Return a JSON object with keys summary, operations, warnings. No markdown fences."
        )
        user_prompt = json.dumps(
            {
                "request": message,
                "current_state": _make_builder_context(workforce),
            },
            ensure_ascii=False,
        )
        response = await llm.chat(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_object"},
        )
        content = (
            response["content"]
            if isinstance(response, dict) and "content" in response
            else response
        )
        if not isinstance(content, str):
            content = str(content)
        parsed = json.loads(content)
        if not isinstance(parsed, dict):
            return None
        return _clean_patch(parsed)
    except Exception as exc:
        logger.warning("Failed to generate builder patch with LLM: %s", exc)
        return None


async def generate_builder_patch(
    db: Session,
    user: User,
    workforce: Workforce,
    message: str,
) -> tuple[str, dict[str, Any]]:
    normalized_message = normalize_text(message, "message", required=True)
    if normalized_message is None:
        raise HTTPException(status_code=400, detail="message is required")

    fallback_patch = _clean_patch(_fallback_patch_from_message(workforce, normalized_message))
    if _has_meaningful_operations(fallback_patch):
        return (
            f"I prepared {len(fallback_patch['operations'])} change(s) using rule-based parsing.",
            fallback_patch,
        )

    if os.getenv("XAGENT_WORKFORCE_BUILDER_ENABLE_LLM") == "1":
        llm_patch = await _llm_generate_patch(db, user, workforce, normalized_message)
        if llm_patch is not None and _has_meaningful_operations(llm_patch):
            return (
                f"I prepared {len(llm_patch['operations'])} change(s) for review.",
                llm_patch,
            )

    return (
        "I could not confidently translate the request into a safe structured change. "
        "Review the warning and refine the prompt if needed.",
        fallback_patch,
    )


def _apply_update_workforce(workforce: Workforce, operation: dict[str, Any], db: Session) -> None:
    fields = operation.get("fields")
    if not isinstance(fields, dict):
        return
    if "name" in fields:
        name = normalize_text(fields.get("name"), "name", required=True)
        if name:
            duplicate = (
                db.query(Workforce)
                .filter(
                    Workforce.id != workforce.id,
                    Workforce.scope_type == workforce.scope_type,
                    Workforce.scope_id == workforce.scope_id,
                    Workforce.name == name,
                )
                .first()
            )
            if duplicate:
                raise HTTPException(status_code=409, detail="Workforce name already exists")
            workforce.name = name
    if "description" in fields:
        workforce.description = normalize_text(fields.get("description"), "description")
    if "manager_instructions" in fields:
        workforce.manager_instructions = normalize_text(
            fields.get("manager_instructions"),
            "manager_instructions",
        )
    if "status" in fields:
        workforce.status = normalize_workforce_status(fields.get("status"))


def _apply_add_existing_worker(
    workforce: Workforce,
    operation: dict[str, Any],
    db: Session,
    user: User,
) -> None:
    agent_id = operation.get("agent_id")
    if not isinstance(agent_id, int):
        raise HTTPException(status_code=400, detail="agent_id is required for add_existing_worker")

    agent = ensure_agent_access(
        db.query(Agent).filter(Agent.id == agent_id).first(),
        user,
        db,
    )
    if agent.id == workforce.manager_agent_id:
        raise HTTPException(status_code=400, detail="Manager agent cannot also be a worker")
    existing = (
        db.query(WorkforceAgent)
        .filter(
            WorkforceAgent.workforce_id == workforce.id,
            WorkforceAgent.agent_id == agent.id,
        )
        .first()
    )
    if existing:
        raise HTTPException(status_code=409, detail="Agent already added to workforce")

    assignment_instructions = normalize_text(
        operation.get("assignment_instructions"),
        "assignment_instructions",
        required=True,
    )
    if assignment_instructions is None:
        raise HTTPException(status_code=400, detail="assignment_instructions is required")

    create_workforce_worker(
        db,
        workforce,
        user,
        source_type="existing",
        assignment_instructions=assignment_instructions,
        alias=operation.get("alias"),
        agent_id=agent.id,
        enabled=bool(operation.get("enabled", True)),
        sort_order=operation.get("sort_order")
        if isinstance(operation.get("sort_order"), int)
        else None,
    )


def _apply_add_worker_from_template(
    workforce: Workforce,
    operation: dict[str, Any],
    db: Session,
    user: User,
) -> None:
    template_id = operation.get("template_id")
    if not isinstance(template_id, str) or not template_id.strip():
        raise HTTPException(
            status_code=400, detail="template_id is required for add_worker_from_template"
        )

    assignment_instructions = normalize_text(
        operation.get("assignment_instructions"),
        "assignment_instructions",
        required=True,
    )
    if assignment_instructions is None:
        raise HTTPException(status_code=400, detail="assignment_instructions is required")

    create_workforce_worker(
        db,
        workforce,
        user,
        source_type="template",
        assignment_instructions=assignment_instructions,
        alias=operation.get("alias"),
        template_id=template_id.strip(),
        enabled=bool(operation.get("enabled", True)),
        sort_order=operation.get("sort_order")
        if isinstance(operation.get("sort_order"), int)
        else None,
    )


def _apply_create_worker_agent(
    workforce: Workforce,
    operation: dict[str, Any],
    db: Session,
    user: User,
) -> None:
    agent_payload = operation.get("agent")
    if not isinstance(agent_payload, dict):
        raise HTTPException(status_code=400, detail="agent is required for create_worker_agent")

    assignment_instructions = normalize_text(
        operation.get("assignment_instructions"),
        "assignment_instructions",
        required=True,
    )
    if assignment_instructions is None:
        raise HTTPException(status_code=400, detail="assignment_instructions is required")

    create_workforce_worker(
        db,
        workforce,
        user,
        source_type="new",
        assignment_instructions=assignment_instructions,
        alias=operation.get("alias"),
        agent_payload=agent_payload,
        enabled=bool(operation.get("enabled", True)),
        sort_order=operation.get("sort_order")
        if isinstance(operation.get("sort_order"), int)
        else None,
    )


def _apply_update_worker(workforce: Workforce, operation: dict[str, Any], db: Session) -> None:
    member_id = operation.get("member_id")
    if not isinstance(member_id, int):
        raise HTTPException(status_code=400, detail="member_id is required for update_worker")
    worker = (
        db.query(WorkforceAgent)
        .filter(
            WorkforceAgent.id == member_id,
            WorkforceAgent.workforce_id == workforce.id,
        )
        .first()
    )
    if worker is None:
        raise HTTPException(status_code=404, detail="Workforce worker not found")

    if "alias" in operation:
        worker.alias = normalize_text(operation.get("alias"), "alias")
    if "assignment_instructions" in operation:
        assignment_instructions = normalize_text(
            operation.get("assignment_instructions"),
            "assignment_instructions",
            required=True,
        )
        if assignment_instructions is None:
            raise HTTPException(status_code=400, detail="assignment_instructions is required")
        worker.assignment_instructions = assignment_instructions
    if "enabled" in operation:
        worker.enabled = bool(operation.get("enabled"))
    if "sort_order" in operation and isinstance(operation.get("sort_order"), int):
        worker.sort_order = int(operation["sort_order"])


def _apply_remove_worker(workforce: Workforce, operation: dict[str, Any], db: Session) -> None:
    member_id = operation.get("member_id")
    if not isinstance(member_id, int):
        raise HTTPException(status_code=400, detail="member_id is required for remove_worker")
    worker = (
        db.query(WorkforceAgent)
        .filter(
            WorkforceAgent.id == member_id,
            WorkforceAgent.workforce_id == workforce.id,
        )
        .first()
    )
    if worker is None:
        raise HTTPException(status_code=404, detail="Workforce worker not found")
    db.delete(worker)
    db.flush()


def apply_builder_patch(
    db: Session,
    user: User,
    workforce: Workforce,
    patch: dict[str, Any],
) -> Workforce:
    workforce = ensure_workforce_access(db, user, workforce, action="edit")
    clean_patch = _clean_patch(patch)
    operations = clean_patch["operations"]

    for operation in operations:
        op_name = operation["op"]
        if op_name == "update_workforce":
            _apply_update_workforce(workforce, operation, db)
        elif op_name == "add_existing_worker":
            _apply_add_existing_worker(workforce, operation, db, user)
        elif op_name == "add_worker_from_template":
            _apply_add_worker_from_template(workforce, operation, db, user)
        elif op_name == "create_worker_agent":
            _apply_create_worker_agent(workforce, operation, db, user)
        elif op_name == "update_worker":
            _apply_update_worker(workforce, operation, db)
        elif op_name == "remove_worker":
            _apply_remove_worker(workforce, operation, db)
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported builder operation: {op_name}")

    db.flush()
    if workforce.status == "active":
        enabled_count = (
            db.query(WorkforceAgent)
            .filter(
                WorkforceAgent.workforce_id == workforce.id,
                WorkforceAgent.enabled.is_(True),
            )
            .count()
        )
        if enabled_count == 0:
            raise HTTPException(
                status_code=400,
                detail="Active workforce requires at least one enabled worker",
            )
    return workforce
