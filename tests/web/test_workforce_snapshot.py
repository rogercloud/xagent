import tempfile

from fastapi import HTTPException
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from xagent.web.models.agent import Agent, AgentStatus
from xagent.web.models.database import Base
from xagent.web.models.user import User
from xagent.web.models.workforce import Workforce, WorkforceAgent
from xagent.web.services.workforce_snapshot import (
    build_agent_tool_overrides,
    build_workforce_snapshot,
)


def _create_session() -> tuple[Session, str]:
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as temp_db:
        db_path = temp_db.name
    db_url = f"sqlite:///{db_path}"
    engine = create_engine(db_url)
    Base.metadata.create_all(bind=engine)
    session_local = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    return session_local(), db_path


def _create_user(db_session: Session, username: str) -> User:
    user = User(username=username, password_hash="x", is_admin=False)
    db_session.add(user)
    db_session.commit()
    db_session.refresh(user)
    return user


def _create_agent(db_session, user, name, execution_mode="balanced"):
    agent = Agent(
        user_id=user.id,
        name=name,
        description=f"{name} description",
        instructions=f"{name} instructions",
        execution_mode=execution_mode,
        models={"general": 1},
        status=AgentStatus.PUBLISHED,
    )
    db_session.add(agent)
    db_session.commit()
    db_session.refresh(agent)
    return agent


def test_build_workforce_snapshot():
    db_session, db_path = _create_session()
    try:
        regular_user = _create_user(db_session, "snapshot-user")
        manager = _create_agent(
            db_session, regular_user, "Manager", execution_mode="think"
        )
        worker = _create_agent(db_session, regular_user, "Research Agent")

        workforce = Workforce(
            scope_type="user",
            scope_id=str(regular_user.id),
            owner_user_id=regular_user.id,
            name="Launch Workforce",
            description="Coordinate launch tasks",
            manager_agent_id=manager.id,
            manager_instructions="Keep the answer concise.",
            status="active",
        )
        db_session.add(workforce)
        db_session.commit()
        db_session.refresh(workforce)

        db_session.add(
            WorkforceAgent(
                workforce_id=workforce.id,
                agent_id=worker.id,
                alias="Researcher",
                assignment_instructions="Research competitors.",
                enabled=True,
                sort_order=1,
            )
        )
        db_session.commit()
        db_session.refresh(workforce)

        snapshot = build_workforce_snapshot(db_session, regular_user, workforce)
        assert snapshot["workforce"]["name"] == "Launch Workforce"
        assert snapshot["workforce"]["scope_type"] == "user"
        assert snapshot["workforce"]["scope_id"] == str(regular_user.id)
        assert snapshot["manager"]["agent_id"] == manager.id
        assert snapshot["workers"][0]["alias"] == "Researcher"
        assert snapshot["workers"][0]["tool_name"].startswith("call_workforce_worker_")
        assert "You are the Workforce Manager" in snapshot["manager"]["runtime_prompt"]

        overrides = build_agent_tool_overrides(snapshot, workforce_run_id=88)
        assert overrides[worker.id]["tool_name"] == snapshot["workers"][0]["tool_name"]
        assert "Research competitors." in overrides[worker.id]["extra_system_prompt"]
        assert overrides[worker.id]["workforce_run_id"] == 88
        assert overrides[worker.id]["worker_alias"] == "Researcher"
    finally:
        db_session.close()
        try:
            import os

            os.remove(db_path)
        except OSError:
            pass


def test_build_workforce_snapshot_requires_enabled_worker():
    db_session, db_path = _create_session()
    try:
        regular_user = _create_user(db_session, "snapshot-user-2")
        manager = _create_agent(db_session, regular_user, "Manager Two")
        worker = _create_agent(db_session, regular_user, "Writer Agent")

        workforce = Workforce(
            scope_type="user",
            scope_id=str(regular_user.id),
            owner_user_id=regular_user.id,
            name="No Enabled Workers",
            manager_agent_id=manager.id,
            status="draft",
        )
        db_session.add(workforce)
        db_session.commit()
        db_session.refresh(workforce)
        db_session.add(
            WorkforceAgent(
                workforce_id=workforce.id,
                agent_id=worker.id,
                alias="Writer",
                assignment_instructions="Write copy.",
                enabled=False,
            )
        )
        db_session.commit()
        db_session.refresh(workforce)

        try:
            build_workforce_snapshot(db_session, regular_user, workforce)
        except HTTPException as exc:
            assert exc.status_code == 400
            assert "enabled worker" in exc.detail
            return
        raise AssertionError("Expected HTTPException")
    finally:
        db_session.close()
        try:
            import os

            os.remove(db_path)
        except OSError:
            pass
