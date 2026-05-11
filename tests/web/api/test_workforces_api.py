import tempfile

import pytest
from fastapi import HTTPException
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from xagent.web.api.agents import delete_agent
from xagent.web.api.workforces import (
    WorkforceBuilderApplyRequest,
    WorkforceBuilderProposeRequest,
    WorkforceCreateRequest,
    WorkforceRunRequest,
    WorkforceUpdateRequest,
    WorkforceWorkerInput,
    WorkforceWorkerUpdateRequest,
    add_workforce_agent,
    apply_workforce_changes,
    create_workforce,
    create_workforce_run,
    get_workforce,
    get_workforce_canvas,
    list_workforces,
    propose_workforce_changes,
    remove_workforce_agent,
    update_workforce,
    update_workforce_agent,
)
from xagent.web.models.agent import Agent, AgentStatus
from xagent.web.models.database import Base
from xagent.web.models.user import User
from xagent.web.models.workforce import Workforce, WorkforceAgent


def _create_session() -> tuple[Session, str]:
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as temp_db:
        db_path = temp_db.name
    db_url = f"sqlite:///{db_path}"
    engine = create_engine(db_url)
    Base.metadata.create_all(bind=engine)
    session_local = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    return session_local(), db_path


def _create_user(db_session: Session, username: str, is_admin: bool = False) -> User:
    user = User(username=username, password_hash="x", is_admin=is_admin)
    db_session.add(user)
    db_session.commit()
    db_session.refresh(user)
    return user


def _create_agent(
    db_session: Session, user: User, name: str, execution_mode: str = "balanced"
) -> Agent:
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


@pytest.mark.asyncio
async def test_create_and_get_workforce() -> None:
    db_session, db_path = _create_session()
    try:
        regular_user = _create_user(db_session, "regular-user")
        manager = _create_agent(db_session, regular_user, "Manager")
        worker = _create_agent(db_session, regular_user, "Research Agent")

        result = await create_workforce(
            WorkforceCreateRequest(
                name="Launch Workforce",
                description="Coordinate launch",
                manager_agent_id=manager.id,
                manager_instructions="Coordinate workers.",
                status="active",
                workers=[
                    WorkforceWorkerInput(
                        agent_id=worker.id,
                        alias="Researcher",
                        assignment_instructions="Research competitors.",
                        sort_order=1,
                    )
                ],
            ),
            db_session,
            regular_user,
        )
        assert result["name"] == "Launch Workforce"
        assert result["status"] == "active"
        assert result["manager"]["id"] == manager.id
        assert result["scope_type"] == "user"
        assert result["scope_id"] == str(regular_user.id)
        assert len(result["workers"]) == 1
        workforce_id = result["id"]

        detail = await get_workforce(workforce_id, db_session, regular_user)
        assert detail["manager_instructions"] == "Coordinate workers."
        assert detail["workers"][0]["alias"] == "Researcher"
    finally:
        db_session.close()
        try:
            import os

            os.remove(db_path)
        except OSError:
            pass


@pytest.mark.asyncio
async def test_list_workforces_filters_to_visible_items() -> None:
    db_session, db_path = _create_session()
    try:
        regular_user = _create_user(db_session, "owner-user")
        other_user = _create_user(db_session, "other-user")
        manager_a = _create_agent(db_session, regular_user, "Manager A")
        worker_a = _create_agent(db_session, regular_user, "Worker A")
        manager_b = _create_agent(db_session, other_user, "Manager B")
        worker_b = _create_agent(db_session, other_user, "Worker B")

        await create_workforce(
            WorkforceCreateRequest(
                name="Visible Workforce",
                manager_agent_id=manager_a.id,
                workers=[
                    WorkforceWorkerInput(
                        agent_id=worker_a.id,
                        alias="Writer",
                        assignment_instructions="Write copy.",
                    )
                ],
            ),
            db_session,
            regular_user,
        )

        await create_workforce(
            WorkforceCreateRequest(
                name="Hidden Workforce",
                manager_agent_id=manager_b.id,
                workers=[
                    WorkforceWorkerInput(
                        agent_id=worker_b.id,
                        alias="Analyst",
                        assignment_instructions="Analyze data.",
                    )
                ],
            ),
            db_session,
            other_user,
        )

        owner_result = await list_workforces(db=db_session, user=regular_user)
        assert owner_result["total"] == 1
        assert owner_result["items"][0]["name"] == "Visible Workforce"

        other_result = await list_workforces(db=db_session, user=other_user)
        assert other_result["total"] == 1
        assert other_result["items"][0]["name"] == "Hidden Workforce"
    finally:
        db_session.close()
        try:
            import os

            os.remove(db_path)
        except OSError:
            pass


@pytest.mark.asyncio
async def test_update_worker_and_canvas() -> None:
    db_session, db_path = _create_session()
    try:
        regular_user = _create_user(db_session, "edit-user")
        manager = _create_agent(db_session, regular_user, "Manager Three")
        worker = _create_agent(db_session, regular_user, "Researcher Agent")

        created = await create_workforce(
            WorkforceCreateRequest(
                name="Editable Workforce",
                manager_agent_id=manager.id,
                workers=[
                    WorkforceWorkerInput(
                        agent_id=worker.id,
                        alias="Researcher",
                        assignment_instructions="Research market.",
                    )
                ],
            ),
            db_session,
            regular_user,
        )

        updated = await update_workforce(
            created["id"],
            WorkforceUpdateRequest(
                name="Editable Workforce V2",
                description="Updated description",
                manager_instructions="Use worker results carefully.",
                canvas_layout={"zoom": 1.2},
            ),
            db_session,
            regular_user,
        )
        assert updated["name"] == "Editable Workforce V2"
        assert updated["canvas_layout"] == {"zoom": 1.2}

        member_id = updated["workers"][0]["id"]
        worker_updated = await update_workforce_agent(
            created["id"],
            member_id,
            WorkforceWorkerUpdateRequest(
                alias="Market Researcher",
                assignment_instructions="Focus on pricing.",
                sort_order=3,
                canvas_position={"x": 20, "y": 30},
            ),
            db_session,
            regular_user,
        )
        assert worker_updated["alias"] == "Market Researcher"
        assert worker_updated["assignment_instructions"] == "Focus on pricing."
        assert worker_updated["sort_order"] == 3

        canvas = await get_workforce_canvas(created["id"], db_session, regular_user)
        worker_node = next(node for node in canvas["nodes"] if node["type"] == "worker")
        assert worker_node["position"] == {"x": 20, "y": 30}
    finally:
        db_session.close()
        try:
            import os

            os.remove(db_path)
        except OSError:
            pass


@pytest.mark.asyncio
async def test_builder_apply_and_run_workforce() -> None:
    db_session, db_path = _create_session()
    try:
        regular_user = _create_user(db_session, "builder-user")
        manager = _create_agent(db_session, regular_user, "Manager Four")
        worker = _create_agent(db_session, regular_user, "Analyst Agent")
        extra_worker = _create_agent(db_session, regular_user, "Writer Agent")

        created = await create_workforce(
            WorkforceCreateRequest(
                name="Builder Workforce",
                manager_agent_id=manager.id,
                workers=[
                    WorkforceWorkerInput(
                        agent_id=worker.id,
                        alias="Analyst",
                        assignment_instructions="Analyze data.",
                    )
                ],
            ),
            db_session,
            regular_user,
        )

        propose = await propose_workforce_changes(
            created["id"],
            WorkforceBuilderProposeRequest(
                message='Rename this workforce to "Launch Crew".',
            ),
            db_session,
            regular_user,
        )
        proposed_patch = propose["proposed_patch"]
        assert proposed_patch["operations"]

        applied = await apply_workforce_changes(
            created["id"],
            WorkforceBuilderApplyRequest(
                message_id=propose["message_id"],
                proposed_patch=proposed_patch,
            ),
            db_session,
            regular_user,
        )
        assert applied["workforce"]["name"] == "Launch Crew"

        added_worker = await add_workforce_agent(
            created["id"],
            WorkforceWorkerInput(
                agent_id=extra_worker.id,
                alias="Writer",
                assignment_instructions="Draft the summary.",
            ),
            db_session,
            regular_user,
        )
        assert added_worker["alias"] == "Writer"

        canvas = await get_workforce_canvas(created["id"], db_session, regular_user)
        assert len(canvas["nodes"]) == 4
        assert len(canvas["edges"]) == 3

        run_response = await create_workforce_run(
            created["id"],
            WorkforceRunRequest(message="Coordinate a launch brief"),
            db_session,
            regular_user,
        )
        assert run_response["status"] == "pending"
        assert run_response["redirect_url"].startswith("/task/")
    finally:
        db_session.close()
        try:
            import os

            os.remove(db_path)
        except OSError:
            pass


@pytest.mark.asyncio
async def test_delete_agent_blocked_when_referenced_by_workforce() -> None:
    db_session, db_path = _create_session()
    try:
        regular_user = _create_user(db_session, "delete-user")
        manager = _create_agent(db_session, regular_user, "Manager Five")
        worker = _create_agent(db_session, regular_user, "Worker Five")

        await create_workforce(
            WorkforceCreateRequest(
                name="Protected Workforce",
                manager_agent_id=manager.id,
                workers=[
                    WorkforceWorkerInput(
                        agent_id=worker.id,
                        alias="Worker",
                        assignment_instructions="Handle analysis.",
                    )
                ],
            ),
            db_session,
            regular_user,
        )

        try:
            await delete_agent(manager.id, regular_user, db_session)
        except HTTPException as exc:
            assert exc.status_code == 409
            assert "manager agent" in exc.detail
        else:
            raise AssertionError("Expected manager delete to fail")

        try:
            await delete_agent(worker.id, regular_user, db_session)
        except HTTPException as exc:
            assert exc.status_code == 409
            assert "worker agent" in exc.detail
        else:
            raise AssertionError("Expected worker delete to fail")
    finally:
        db_session.close()
        try:
            import os

            os.remove(db_path)
        except OSError:
            pass


@pytest.mark.asyncio
async def test_delete_last_enabled_worker_from_active_workforce_fails() -> None:
    db_session, db_path = _create_session()
    try:
        regular_user = _create_user(db_session, "delete-worker-user")
        manager = _create_agent(db_session, regular_user, "Manager")
        worker = _create_agent(db_session, regular_user, "Solo Worker")

        workforce = Workforce(
            scope_type="user",
            scope_id=str(regular_user.id),
            owner_user_id=regular_user.id,
            name="Solo Workforce",
            manager_agent_id=manager.id,
            status="active",
        )
        db_session.add(workforce)
        db_session.commit()
        db_session.refresh(workforce)

        worker_link = WorkforceAgent(
            workforce_id=workforce.id,
            agent_id=worker.id,
            alias="Solo",
            assignment_instructions="Do everything.",
            enabled=True,
        )
        db_session.add(worker_link)
        db_session.commit()
        db_session.refresh(worker_link)

        try:
            await remove_workforce_agent(
                workforce_id=workforce.id,
                member_id=worker_link.id,
                db=db_session,
                user=regular_user,
            )
        except HTTPException as exc:
            assert exc.status_code == 400
            assert "enabled worker" in exc.detail.lower()
            return
        raise AssertionError(
            "Expected HTTPException when deleting last enabled worker from active workforce"
        )
    finally:
        db_session.close()
        try:
            import os

            os.remove(db_path)
        except OSError:
            pass
