"""Integration tests for agent management endpoints."""

from typing import Any

import pytest

from xagent.web.models.agent import Agent, AgentStatus
from xagent.web.models.agent_api_key import AgentApiKey
from xagent.web.models.task import Task, TaskStatus
from xagent.web.models.user import User
from xagent.web.services.workforce_access import WorkforcePolicy, set_workforce_policy

from .conftest import (
    _admin_headers,
    _direct_db_session,
    _register_second_user,
    client,
)

pytestmark = pytest.mark.usefixtures("_test_db")


@pytest.fixture(autouse=True)
def _reset_workforce_policy() -> None:
    set_workforce_policy(WorkforcePolicy())
    yield
    set_workforce_policy(WorkforcePolicy())


def _create_agent(headers: dict[str, str], name: str = "Test Agent") -> int:
    resp = client.post(
        "/api/agents",
        headers=headers,
        json={
            "name": name,
            "description": "test",
            "instructions": "You are a test agent.",
            "execution_mode": "balanced",
        },
    )
    assert resp.status_code == 200, resp.text
    return resp.json()["id"]


def _create_agent_row(
    *,
    user_id: int,
    name: str,
    status: AgentStatus = AgentStatus.DRAFT,
) -> int:
    db = _direct_db_session()
    try:
        agent = Agent(
            user_id=user_id,
            name=name,
            description=f"{name} description",
            instructions=f"{name} instructions",
            execution_mode="balanced",
            status=status,
        )
        db.add(agent)
        db.commit()
        db.refresh(agent)
        return int(agent.id)
    finally:
        db.close()


def _user_id(username: str) -> int:
    db = _direct_db_session()
    try:
        user = db.query(User).filter(User.username == username).first()
        assert user is not None
        return int(user.id)
    finally:
        db.close()


class _VisibleAgentPolicy(WorkforcePolicy):
    def __init__(self, visible_agent_ids: set[int]) -> None:
        self.visible_agent_ids = visible_agent_ids

    def get_visible_agent_ids(
        self,
        db: Any,
        user: User,
        purpose: str,
    ) -> set[int]:
        del db, user, purpose
        return self.visible_agent_ids


def test_list_agents_includes_owned_agents_and_policy_visible_agents() -> None:
    _admin_headers()
    bob_headers = _register_second_user()
    admin_id = _user_id("admin")
    bob_id = _user_id("bob")

    bob_draft_id = _create_agent_row(user_id=bob_id, name="Bob Draft")
    bob_published_id = _create_agent_row(
        user_id=bob_id,
        name="Bob Published",
        status=AgentStatus.PUBLISHED,
    )
    shared_published_id = _create_agent_row(
        user_id=admin_id,
        name="Shared Published",
        status=AgentStatus.PUBLISHED,
    )
    shared_draft_id = _create_agent_row(
        user_id=admin_id,
        name="Shared Draft",
        status=AgentStatus.DRAFT,
    )
    set_workforce_policy(_VisibleAgentPolicy({shared_published_id, shared_draft_id}))

    response = client.get("/api/agents", headers=bob_headers)
    assert response.status_code == 200, response.text
    items_by_id = {item["id"]: item for item in response.json()}

    assert {
        bob_draft_id,
        bob_published_id,
        shared_published_id,
        shared_draft_id,
    }.issubset(items_by_id)

    assert items_by_id[bob_draft_id]["access"] == "owner"
    assert items_by_id[bob_draft_id]["readonly"] is False
    assert items_by_id[bob_draft_id]["can_edit"] is True
    assert items_by_id[bob_draft_id]["can_publish"] is True
    assert items_by_id[bob_draft_id]["can_delete"] is True

    assert items_by_id[shared_published_id]["access"] == "policy"
    assert items_by_id[shared_published_id]["readonly"] is True
    assert items_by_id[shared_published_id]["can_edit"] is False
    assert items_by_id[shared_published_id]["can_publish"] is False
    assert items_by_id[shared_published_id]["can_delete"] is False
    assert items_by_id[shared_draft_id]["access"] == "policy"
    assert items_by_id[shared_draft_id]["status"] == "draft"
    assert items_by_id[shared_draft_id]["readonly"] is True


class TestDeleteAgent:
    """DELETE /api/agents/{agent_id} - remove an agent."""

    def test_with_tasks_keeps_tasks_and_nulls_agent_id(self):
        headers = _admin_headers()
        agent_id = _create_agent(headers)
        client.post(f"/api/agents/{agent_id}/api-key", headers=headers)

        db = _direct_db_session()
        try:
            admin_user = db.query(User).filter(User.username == "admin").first()
            assert admin_user is not None
            task = Task(
                user_id=admin_user.id,
                title="task tied to agent",
                description="task tied to agent",
                status=TaskStatus.PENDING,
                agent_id=agent_id,
            )
            db.add(task)
            db.commit()
            db.refresh(task)
            task_id = task.id
        finally:
            db.close()

        delete_resp = client.delete(f"/api/agents/{agent_id}", headers=headers)
        assert delete_resp.status_code == 200, delete_resp.text

        db = _direct_db_session()
        try:
            task = db.query(Task).filter(Task.id == task_id).first()
            assert task is not None
            assert task.agent_id is None
            assert (
                db.query(AgentApiKey).filter(AgentApiKey.agent_id == agent_id).all()
                == []
            )
        finally:
            db.close()
