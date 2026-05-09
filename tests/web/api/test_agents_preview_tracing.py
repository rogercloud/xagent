from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tests.utils.mock_helpers import create_langfuse_mock
from xagent.core.tracing.langfuse.handler import LangfuseTraceHandler
from xagent.web.api.agents import AgentPreviewRequest, preview_agent
from xagent.web.models.user import User


@pytest.mark.asyncio
async def test_preview_agent_injects_langfuse_tracer(
    mocker, monkeypatch, langfuse_client_reset
):
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "test-public")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "test-secret")
    create_langfuse_mock(mocker)

    current_user = User()
    current_user.id = 7
    current_user.is_admin = False

    db = MagicMock()
    model_record = MagicMock()
    model_record.model_id = "test-model"
    db.query.return_value.filter.return_value.first.return_value = model_record

    request = AgentPreviewRequest(
        instructions="preview instructions",
        execution_mode="balanced",
        models={"general": 1},
        knowledge_bases=[],
        skills=[],
        tool_categories=[],
        message="hello",
    )

    with (
        patch("xagent.web.api.agents.UserAwareModelStorage") as mock_storage_class,
        patch("xagent.web.api.agents.InMemoryMemoryStore"),
        patch("xagent.web.api.agents.AgentService") as mock_agent_service_class,
    ):
        mock_storage = MagicMock()
        mock_llm = MagicMock()
        mock_storage.get_llm_by_name_with_access.return_value = mock_llm
        mock_storage_class.return_value = mock_storage

        mock_agent_service = mock_agent_service_class.return_value
        mock_agent_service.execute_task = AsyncMock(
            return_value={"output": "preview response", "status": "completed"}
        )

        response = await preview_agent(
            request=request, current_user=current_user, db=db
        )

    assert response.response == "preview response"
    tracer = mock_agent_service_class.call_args.kwargs["tracer"]
    assert any(isinstance(handler, LangfuseTraceHandler) for handler in tracer.handlers)
