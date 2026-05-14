import json
import logging
import re
from typing import Any, cast

from sqlalchemy.orm import Session

from xagent.web.models.agent import Agent, AgentStatus
from xagent.web.models.user import User
from xagent.web.services.llm_utils import UserAwareModelStorage

logger = logging.getLogger(__name__)


def _available_published_agents(db: Session, user: User) -> list[Agent]:
    query = db.query(Agent).filter(cast(Any, Agent.status) == AgentStatus.PUBLISHED)
    if not user.is_admin:
        query = query.filter(Agent.user_id == user.id)
    return [cast(Agent, item) for item in query.order_by(Agent.id.asc()).all()]


def _serialize_available_agents(agents: list[Agent]) -> list[dict[str, Any]]:
    return [
        {
            "agent_id": agent.id,
            "name": agent.name,
            "description": agent.description,
            "status": getattr(agent.status, "value", str(agent.status)),
        }
        for agent in agents
    ]


def _short_name_from_prompt(prompt: str) -> str:
    words = re.findall(r"[\w-]+", prompt, flags=re.UNICODE)
    if not words:
        return "New Workforce"
    name = " ".join(words[:6]).strip()
    if len(name) > 80:
        name = name[:80].rstrip()
    return f"{name} Workforce" if "workforce" not in name.lower() else name


def _clean_creation_plan(
    candidate: dict[str, Any],
    available_agent_ids: set[int],
    prompt: str,
) -> dict[str, Any]:
    name = str(candidate.get("name") or "").strip() or _short_name_from_prompt(prompt)
    description = str(candidate.get("description") or "").strip() or prompt.strip()

    manager = candidate.get("manager")
    if not isinstance(manager, dict):
        manager = {}
    manager_name = str(manager.get("name") or "").strip() or f"{name} Manager"
    manager_description = (
        str(manager.get("description") or "").strip()
        or "Coordinates this Workforce and synthesizes worker outputs."
    )
    manager_instructions = (
        str(manager.get("instructions") or "").strip()
        or (
            "Understand the user's goal, delegate focused work to the available "
            "workers, compare their outputs, and return one coherent final answer."
        )
    )

    workers = candidate.get("workers")
    if not isinstance(workers, list):
        workers = []
    clean_workers: list[dict[str, Any]] = []
    seen_agent_ids: set[int] = set()
    for item in workers:
        if not isinstance(item, dict):
            continue
        agent_id = item.get("agent_id")
        if not isinstance(agent_id, int):
            continue
        if agent_id not in available_agent_ids or agent_id in seen_agent_ids:
            continue
        instructions = str(item.get("assignment_instructions") or "").strip()
        if not instructions:
            continue
        clean_workers.append(
            {
                "agent_id": agent_id,
                "alias": str(item.get("alias") or "").strip() or None,
                "assignment_instructions": instructions,
                "enabled": bool(item.get("enabled", True)),
            }
        )
        seen_agent_ids.add(agent_id)

    warnings = candidate.get("warnings")
    if not isinstance(warnings, list):
        warnings = []
    clean_warnings = [str(item).strip() for item in warnings if str(item).strip()]
    if not clean_workers:
        clean_warnings.append(
            "No published worker agents were selected. Add workers before running."
        )

    return {
        "name": name[:200],
        "description": description,
        "manager": {
            "name": manager_name[:200],
            "description": manager_description,
            "instructions": manager_instructions,
        },
        "manager_instructions": manager_instructions,
        "workers": clean_workers,
        "warnings": clean_warnings,
    }


def _fallback_creation_plan(prompt: str, agents: list[Agent]) -> dict[str, Any]:
    lower_prompt = prompt.lower()
    selected: list[Agent] = []
    for agent in agents:
        haystack = f"{agent.name} {agent.description or ''}".lower()
        if any(token and token in haystack for token in re.findall(r"[a-z0-9]+", lower_prompt)):
            selected.append(agent)
        if len(selected) >= 3:
            break
    if not selected:
        selected = agents[: min(3, len(agents))]

    name = _short_name_from_prompt(prompt)
    return _clean_creation_plan(
        {
            "name": name,
            "description": prompt,
            "manager": {
                "name": f"{name} Manager",
                "description": "Coordinates this Workforce and synthesizes worker outputs.",
                "instructions": (
                    "Break the user request into worker assignments, call the "
                    "right workers, reconcile their outputs, and provide a concise final answer."
                ),
            },
            "workers": [
                {
                    "agent_id": cast(int, agent.id),
                    "alias": agent.name,
                    "assignment_instructions": (
                        f"Contribute to the Workforce goal using the strengths of {agent.name}."
                    ),
                    "enabled": True,
                }
                for agent in selected
            ],
            "warnings": [
                "Created from a rule-based fallback because no LLM plan was available."
            ],
        },
        {cast(int, agent.id) for agent in agents},
        prompt,
    )


async def generate_workforce_creation_plan(
    db: Session,
    user: User,
    prompt: str,
) -> dict[str, Any]:
    agents = _available_published_agents(db, user)
    available_agent_ids = {cast(int, agent.id) for agent in agents}

    try:
        storage = UserAwareModelStorage(db)
        user_id = int(user.id)
        default_llm, _, _, _ = storage.get_configured_defaults(user_id)
        llm = default_llm
        if not llm:
            default_llm, _, _, _ = storage.get_configured_defaults(None)
            llm = default_llm
        if not llm:
            return _fallback_creation_plan(prompt, agents)

        system_prompt = (
            "You create initial AI Workforce drafts from a user's goal. "
            "A Workforce is AI orchestration, not a human team. "
            "Create a manager spec that can coordinate workers and synthesize results. "
            "Select workers only from available_published_agents. "
            "Do not create worker agents. If no worker fits, return no workers and add a warning. "
            "Return JSON only with keys: name, description, manager, manager_instructions, workers, warnings. "
            "manager has name, description, instructions. "
            "Each worker has agent_id, alias, assignment_instructions, enabled. "
            "Keep instructions concrete and task-oriented."
        )
        user_prompt = json.dumps(
            {
                "request": prompt,
                "available_published_agents": _serialize_available_agents(agents),
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
            return _fallback_creation_plan(prompt, agents)
        return _clean_creation_plan(parsed, available_agent_ids, prompt)
    except Exception as exc:
        logger.warning("Failed to generate Workforce creation plan with LLM: %s", exc)
        return _fallback_creation_plan(prompt, agents)
