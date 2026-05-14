import os
import tempfile

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from xagent.core.tools.adapters.vibe.agent_tool import (
    AgentTool,
    create_agent_tools,
    get_published_agents_tools,
)
from xagent.web.models.agent import Agent, AgentStatus
from xagent.web.models.database import Base
from xagent.web.models.model import Model
from xagent.web.models.user import User
from xagent.web.tools.config import WebToolConfig


def _create_session() -> tuple[Session, str]:
    temp_db = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    temp_db.close()
    db_url = f"sqlite:///{temp_db.name}"
    engine = create_engine(db_url)
    Base.metadata.create_all(bind=engine)
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    return SessionLocal(), temp_db.name


def _remove_db(path: str) -> None:
    try:
        os.remove(path)
    except OSError:
        pass


def test_non_owner_cannot_see_other_users_published_agent_tools() -> None:
    db, db_path = _create_session()
    try:
        owner = User(username="owner", password_hash="x", is_admin=False)
        other_user = User(username="other", password_hash="x", is_admin=False)
        db.add_all([owner, other_user])
        db.commit()
        db.refresh(owner)
        db.refresh(other_user)

        published_agent = Agent(
            user_id=owner.id,
            name="Owner Published Agent",
            status=AgentStatus.PUBLISHED,
        )
        db.add(published_agent)
        db.commit()

        tools_for_other = get_published_agents_tools(db=db, user_id=2)
        tool_names = {tool.name for tool in tools_for_other}

        assert "call_agent_owner_published_agent" not in tool_names
    finally:
        db.close()
        _remove_db(db_path)


def test_owner_sees_only_own_published_agents_not_drafts() -> None:
    db, db_path = _create_session()
    try:
        owner = User(username="owner2", password_hash="x", is_admin=False)
        db.add(owner)
        db.commit()
        db.refresh(owner)

        published_agent = Agent(
            user_id=owner.id,
            name="Owner Published Agent",
            status=AgentStatus.PUBLISHED,
        )
        draft_agent = Agent(
            user_id=owner.id,
            name="Owner Draft Agent",
            status=AgentStatus.DRAFT,
        )
        db.add_all([published_agent, draft_agent])
        db.commit()

        tools_for_owner = get_published_agents_tools(db=db, user_id=1)
        tool_names = {tool.name for tool in tools_for_owner}

        assert "call_agent_owner_published_agent" in tool_names
        assert "call_agent_owner_draft_agent" not in tool_names
    finally:
        db.close()
        _remove_db(db_path)


def test_allowed_agent_ids_include_only_selected_published_user_agents() -> None:
    db, db_path = _create_session()
    try:
        owner = User(username="owner3", password_hash="x", is_admin=False)
        other_user = User(username="other3", password_hash="x", is_admin=False)
        db.add_all([owner, other_user])
        db.commit()
        db.refresh(owner)
        db.refresh(other_user)

        selected_published = Agent(
            user_id=owner.id,
            name="Selected Published Agent",
            status=AgentStatus.PUBLISHED,
        )
        selected_draft = Agent(
            user_id=owner.id,
            name="Selected Draft Agent",
            status=AgentStatus.DRAFT,
        )
        unselected_published = Agent(
            user_id=owner.id,
            name="Unselected Published Agent",
            status=AgentStatus.PUBLISHED,
        )
        other_users_agent = Agent(
            user_id=other_user.id,
            name="Other Users Agent",
            status=AgentStatus.PUBLISHED,
        )
        db.add_all(
            [
                selected_published,
                selected_draft,
                unselected_published,
                other_users_agent,
            ]
        )
        db.commit()

        tools = get_published_agents_tools(
            db=db,
            user_id=owner.id,
            allowed_agent_ids=[
                selected_published.id,
                selected_draft.id,
                other_users_agent.id,
            ],
        )
        tool_names = {tool.name for tool in tools}

        assert "call_agent_selected_published_agent" in tool_names
        assert "call_agent_selected_draft_agent" not in tool_names
        assert "call_agent_unselected_published_agent" not in tool_names
        assert "call_agent_other_users_agent" not in tool_names
    finally:
        db.close()
        _remove_db(db_path)


def test_allowed_agent_ids_can_cross_users_only_when_explicitly_enabled() -> None:
    db, db_path = _create_session()
    try:
        owner = User(username="owner_cross", password_hash="x", is_admin=False)
        runner = User(username="runner", password_hash="x", is_admin=False)
        db.add_all([owner, runner])
        db.commit()
        db.refresh(owner)
        db.refresh(runner)

        published_agent = Agent(
            user_id=owner.id,
            name="Shared Workforce Worker",
            status=AgentStatus.PUBLISHED,
        )
        db.add(published_agent)
        db.commit()
        db.refresh(published_agent)

        blocked_tools = get_published_agents_tools(
            db=db,
            user_id=runner.id,
            allowed_agent_ids=[published_agent.id],
            enable_global_agent_tools=False,
        )
        assert "call_agent_shared_workforce_worker" not in {
            tool.name for tool in blocked_tools
        }

        allowed_tools = get_published_agents_tools(
            db=db,
            user_id=runner.id,
            allowed_agent_ids=[published_agent.id],
            enable_global_agent_tools=False,
            allow_cross_user_agent_ids=True,
        )
        assert "call_agent_shared_workforce_worker" in {
            tool.name for tool in allowed_tools
        }
    finally:
        db.close()
        _remove_db(db_path)


@pytest.mark.asyncio
async def test_create_agent_tools_treats_empty_delegate_allowlist_as_unrestricted() -> (
    None
):
    db, db_path = _create_session()
    try:
        owner = User(username="owner4", password_hash="x", is_admin=False)
        db.add(owner)
        db.commit()
        db.refresh(owner)

        published_agent = Agent(
            user_id=owner.id,
            name="Default Published Agent",
            status=AgentStatus.PUBLISHED,
        )
        db.add(published_agent)
        db.commit()

        config = WebToolConfig(db=db, request=None, user_id=owner.id, user=owner)
        config._delegate_agent_ids = []

        tools = await create_agent_tools(config)
        tool_names = {tool.name for tool in tools}

        assert "call_agent_default_published_agent" in tool_names
    finally:
        db.close()
        _remove_db(db_path)


def test_agent_call_stack_excludes_recursive_agent_tools() -> None:
    db, db_path = _create_session()
    try:
        owner = User(username="stack-owner", password_hash="x", is_admin=False)
        db.add(owner)
        db.commit()
        db.refresh(owner)

        agent_a = Agent(
            user_id=owner.id,
            name="Agent A",
            status=AgentStatus.PUBLISHED,
        )
        agent_b = Agent(
            user_id=owner.id,
            name="Agent B",
            status=AgentStatus.PUBLISHED,
        )
        agent_c = Agent(
            user_id=owner.id,
            name="Agent C",
            status=AgentStatus.PUBLISHED,
        )
        db.add_all([agent_a, agent_b, agent_c])
        db.commit()
        db.refresh(agent_a)
        db.refresh(agent_b)
        db.refresh(agent_c)

        tools = get_published_agents_tools(
            db=db,
            user_id=owner.id,
            agent_call_stack=[agent_a.id, agent_b.id],
        )
        tool_names = {tool.name for tool in tools}

        assert "call_agent_agent_a" not in tool_names
        assert "call_agent_agent_b" not in tool_names
        assert "call_agent_agent_c" in tool_names
    finally:
        db.close()
        _remove_db(db_path)


@pytest.mark.asyncio
async def test_agent_tool_emits_workforce_delegation_trace_events(monkeypatch) -> None:
    db, db_path = _create_session()
    try:
        user = User(username="trace-user", password_hash="x", is_admin=False)
        db.add(user)
        db.commit()
        db.refresh(user)

        model = Model(
            model_id="fake-general",
            model_provider="openai",
            model_name="fake-general",
            category="llm",
        )
        model.api_key = "fake-key"
        db.add(model)
        db.commit()
        db.refresh(model)

        agent = Agent(
            user_id=user.id,
            name="Trace Worker",
            status=AgentStatus.PUBLISHED,
            models={"general": model.id},
        )
        db.add(agent)
        db.commit()
        db.refresh(agent)

        class FakeStorage:
            def __init__(self, db):
                self.db = db

            def get_llm_by_name_with_access(self, model_id, user_id):
                return object()

        class FakeAgentService:
            init_kwargs = None

            def __init__(self, *args, **kwargs):
                self.__class__.init_kwargs = kwargs

            async def execute_task(self, task, context=None, task_id=None):
                return {"output": "worker output"}

        class FakeTracer:
            def __init__(self):
                self.events = []

            async def trace_event(
                self, event_type, task_id=None, step_id=None, data=None, parent_id=None
            ):
                self.events.append(
                    {
                        "event_type": event_type.value,
                        "task_id": task_id,
                        "data": data or {},
                    }
                )

        monkeypatch.setattr(
            "xagent.web.services.llm_utils.UserAwareModelStorage",
            FakeStorage,
        )
        monkeypatch.setattr(
            "xagent.core.agent.service.AgentService",
            FakeAgentService,
        )

        tracer = FakeTracer()
        tool = AgentTool(
            agent_id=agent.id,
            agent_name=agent.name,
            agent_description="Trace worker",
            db=db,
            user_id=user.id,
            parent_task_id=123,
            parent_tracer=tracer,
            workforce_context={
                "workforce_run_id": 456,
                "worker_alias": "Researcher",
            },
        )

        result = await tool.run_json_async({"task": "research"})

        assert result["response"] == "worker output"
        assert FakeAgentService.init_kwargs["tracer"] is tracer
        nested_tool_config = FakeAgentService.init_kwargs["tool_config"]
        assert nested_tool_config.get_parent_tracer() is tracer
        assert nested_tool_config.get_parent_task_id() == 123
        assert nested_tool_config.get_agent_call_stack() == [agent.id]
        assert [event["data"]["event_type"] for event in tracer.events] == [
            "workforce_delegation_start",
            "workforce_delegation_end",
        ]
        assert tracer.events[0]["task_id"] == "123"
        assert tracer.events[0]["data"]["workforce_run_id"] == 456
        assert tracer.events[0]["data"]["worker_alias"] == "Researcher"
        assert tracer.events[1]["data"]["output"] == "worker output"
    finally:
        db.close()
        _remove_db(db_path)
