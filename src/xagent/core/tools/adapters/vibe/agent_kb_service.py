import logging
from typing import TYPE_CHECKING

from ...core.RAG_tools.core.schemas import IngestionConfig

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from ...core.RAG_tools.kb import KBToolCompatibilityFacade


class AgentKnowledgeBaseError(RuntimeError):
    """Raised when agent-triggered knowledge base setup cannot be completed."""


def _get_tool_compatibility_facade() -> "KBToolCompatibilityFacade":
    """Return the coordinator-owned KB tool compatibility facade."""
    from ...core.RAG_tools.kb import get_kb_coordinator

    return get_kb_coordinator().tool_compatibility


class AgentKnowledgeBaseService:
    """Shared collection setup/refresh flow for agent-triggered KB creation."""

    def __init__(self, user_id: int, is_admin: bool = False) -> None:
        self.user_id = user_id
        self.is_admin = is_admin

    async def prepare_collection(
        self,
        collection_name: str,
        ingestion_config: IngestionConfig,
    ) -> str:
        return await _get_tool_compatibility_facade().prepare_agent_collection(
            collection_name=collection_name,
            ingestion_config=ingestion_config,
            user_id=self.user_id,
            is_admin=self.is_admin,
        )

    async def refresh_collection_metadata(self, collection_name: str) -> None:
        await _get_tool_compatibility_facade().refresh_agent_collection_metadata(
            collection_name,
            user_id=self.user_id,
            is_admin=self.is_admin,
        )


async def _prepare_collection_impl(
    *,
    collection_name: str,
    ingestion_config: IngestionConfig,
    user_id: int,
) -> str:
    from .....web.config import sanitize_path_component
    from ...core.RAG_tools.storage.factory import get_metadata_store

    safe_collection = sanitize_path_component(collection_name, "collection")
    metadata_store = get_metadata_store()

    try:
        await metadata_store.save_collection_config(
            collection=safe_collection,
            config_json=ingestion_config.model_dump_json(exclude_unset=True),
            user_id=user_id,
        )
    except Exception as exc:
        logger.error(
            "Failed to save collection config for agent knowledge base %s: %s",
            safe_collection,
            exc,
        )
        raise AgentKnowledgeBaseError(
            f"Failed to save collection config for knowledge base '{safe_collection}'"
        ) from exc

    return safe_collection


async def _refresh_collection_metadata_impl(
    *,
    collection_name: str,
    user_id: int,
    is_admin: bool = False,
) -> None:
    from ...core.RAG_tools.management.collections import list_collections

    if not is_admin:
        # Non-admin realtime refreshes do not persist metadata and only add scan cost.
        return

    try:
        # Refresh metadata cache so agent-created KBs are visible like API-created ones.
        await list_collections(
            user_id=user_id,
            is_admin=is_admin,
            force_realtime=True,
        )
    except Exception as exc:
        logger.error(
            "Failed to refresh collection metadata after agent ingestion for %s: %s",
            collection_name,
            exc,
        )
        raise AgentKnowledgeBaseError(
            f"Failed to refresh knowledge base metadata for '{collection_name}'"
        ) from exc
