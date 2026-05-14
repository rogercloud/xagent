import tempfile

from fastapi import HTTPException
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from xagent.web.models.agent import Agent, AgentStatus
from xagent.web.models.database import Base
from xagent.web.models.task import Task, TaskStatus
from xagent.web.models.user import User
from xagent.web.models.workforce import Workforce, WorkforceAgent, WorkforceRun
from xagent.web.services.workforce_runtime import (
    _map_task_status,
    sync_workforce_run_status,
)
from xagent.web.services.workforce_snapshot import (
    _build_worker_tool_name,
    build_agent_tool_overrides,
    build_workforce_snapshot,
)
from xagent.web.services.workforce_workers import load_template_detail


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
            status="active",
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


def test_map_task_status_maps_paused() -> None:
    assert _map_task_status(TaskStatus.PENDING) == "pending"
    assert _map_task_status(TaskStatus.RUNNING) == "running"
    assert _map_task_status(TaskStatus.PAUSED) == "paused"
    assert _map_task_status(TaskStatus.COMPLETED) == "completed"
    assert _map_task_status(TaskStatus.FAILED) == "failed"
    assert _map_task_status(None) is None
    assert _map_task_status("unknown") is None


def test_sync_workforce_run_status_tracks_task_lifecycle() -> None:
    db_session, db_path = _create_session()
    try:
        regular_user = _create_user(db_session, "run-sync-user")
        manager = _create_agent(db_session, regular_user, "Run Sync Manager")
        workforce = Workforce(
            scope_type="user",
            scope_id=str(regular_user.id),
            owner_user_id=regular_user.id,
            name="Run Sync Workforce",
            manager_agent_id=manager.id,
            status="active",
        )
        db_session.add(workforce)
        db_session.flush()
        task = Task(
            user_id=regular_user.id,
            title="Run sync task",
            description="Run sync task",
            status=TaskStatus.PENDING,
            agent_id=manager.id,
            agent_config={},
        )
        db_session.add(task)
        db_session.flush()
        run = WorkforceRun(
            workforce_id=workforce.id,
            task_id=task.id,
            user_id=regular_user.id,
            status="pending",
            snapshot={"version": 1},
        )
        db_session.add(run)
        db_session.flush()
        task.agent_config = {"workforce_run_id": run.id}
        db_session.commit()

        sync_workforce_run_status(db_session, task, TaskStatus.RUNNING)
        db_session.commit()
        db_session.refresh(run)
        assert run.status == "running"
        assert run.completed_at is None

        sync_workforce_run_status(db_session, task, TaskStatus.COMPLETED)
        db_session.commit()
        db_session.refresh(run)
        assert run.status == "completed"
        assert run.completed_at is not None
    finally:
        db_session.close()
        try:
            import os

            os.remove(db_path)
        except OSError:
            pass


def test_build_worker_tool_name_truncates_long_alias() -> None:
    normal = _build_worker_tool_name(1, "researcher")
    assert normal == "call_workforce_worker_1_researcher"
    assert len(normal) <= 64

    long_alias = "a" * 100
    truncated = _build_worker_tool_name(1, long_alias)
    assert len(truncated) <= 64
    assert truncated.startswith("call_workforce_worker_1_")

    same_alias_second = _build_worker_tool_name(2, long_alias)
    assert same_alias_second != truncated
    assert same_alias_second.startswith("call_workforce_worker_2_")


def test_load_template_detail_rejects_path_traversal() -> None:
    traversal_ids = [
        "../archived/foo",
        "foo/bar",
        "../../etc/passwd",
        "foo\\bar",
    ]
    safe_ids = [
        "valid_name",
        "valid.name",
    ]
    for tid in traversal_ids:
        try:
            load_template_detail(tid)
        except HTTPException as exc:
            assert exc.status_code == 400, (
                f"Expected 400 for {tid}, got {exc.status_code}"
            )
        else:
            raise AssertionError(f"Expected HTTPException for template_id={tid}")

    for tid in safe_ids:
        try:
            load_template_detail(tid)
        except HTTPException as exc:
            if exc.status_code == 400:
                raise AssertionError(f"Safe template_id {tid} was rejected with 400")
            # 404 is expected for non-existent template files


def test_load_template_detail_rejects_overly_long_id() -> None:
    long_id = "a" * 255
    try:
        load_template_detail(long_id)
    except HTTPException as exc:
        assert exc.status_code == 400
    else:
        raise AssertionError("Expected HTTPException for overly long template_id")
