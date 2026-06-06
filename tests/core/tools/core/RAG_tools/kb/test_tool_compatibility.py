"""Tests for the KB tool compatibility facade."""

from __future__ import annotations

import inspect
from typing import Optional
from unittest.mock import MagicMock

import pytest

from xagent.core.tools.adapters.vibe.document_search import (
    KnowledgeSearchTool,
    ListKnowledgeBasesTool,
)
from xagent.core.tools.core.RAG_tools.core.schemas import (
    CollectionInfo,
    IngestionConfig,
)
from xagent.core.tools.core.RAG_tools.kb import (
    KBCoordinator,
    KBToolCompatibilityFacade,
    get_kb_coordinator,
    reset_kb_coordinator_for_tests,
)


class _FakeMetadataStore:
    def __init__(self, collection: Optional[CollectionInfo]) -> None:
        self.collection = collection
        self.saved: list[CollectionInfo] = []

    async def get_collection(self, collection: str) -> CollectionInfo:
        if self.collection is None or self.collection.name != collection:
            raise ValueError(f"Collection {collection!r} not found")
        return self.collection

    async def save_collection(self, collection: CollectionInfo) -> None:
        self.saved.append(collection)
        self.collection = collection


class _FakeStorageShim:
    def __init__(self, metadata_store: _FakeMetadataStore) -> None:
        self.metadata_store = metadata_store

    def get_metadata_store(self) -> _FakeMetadataStore:
        return self.metadata_store


def test_kb_tool_facade_public_surface_imports() -> None:
    import xagent.core.tools.core.RAG_tools.kb as kb

    assert hasattr(kb, "KBToolCompatibilityFacade")
    reset_kb_coordinator_for_tests()
    assert isinstance(
        get_kb_coordinator().tool_compatibility, KBToolCompatibilityFacade
    )
    assert get_kb_coordinator().tools is get_kb_coordinator().tool_compatibility


@pytest.mark.asyncio
async def test_public_list_knowledge_bases_routes_through_tool_facade(monkeypatch):
    from xagent.core.tools.core import document_search

    sentinel = object()
    calls: list[tuple[object, int, bool]] = []

    class Facade:
        async def list_knowledge_bases(
            self,
            tool_args: object,
            user_id: Optional[int] = None,
            is_admin: bool = False,
        ) -> object:
            calls.append((tool_args, user_id or 0, is_admin))
            return sentinel

    args = document_search.ListKnowledgeBasesArgs()
    monkeypatch.setattr(document_search, "_get_tool_compatibility_facade", Facade)

    result = await document_search.list_knowledge_bases(
        args,
        user_id=7,
        is_admin=True,
    )

    assert result is sentinel
    assert calls == [(args, 7, True)]


@pytest.mark.asyncio
async def test_ensure_agent_collection_backend_binding_creates_missing_metadata():
    metadata_store = _FakeMetadataStore(None)
    facade = KBToolCompatibilityFacade(storage_shim=_FakeStorageShim(metadata_store))

    updated = await facade.ensure_agent_collection_backend_binding("demo")

    assert updated.name == "demo"
    assert updated.owners == []
    assert updated.extra_metadata["kb_storage"] == {"backend": "lancedb"}
    assert metadata_store.saved == [updated]


@pytest.mark.asyncio
async def test_prepare_agent_collection_saves_user_config_before_backend_binding(
    monkeypatch,
):
    from xagent.core.tools.adapters.vibe import agent_kb_service

    metadata_store = _FakeMetadataStore(None)
    facade = KBToolCompatibilityFacade(storage_shim=_FakeStorageShim(metadata_store))
    prepare_calls: list[int] = []

    async def fake_prepare_collection_impl(
        *,
        collection_name: str,
        ingestion_config: IngestionConfig,
        user_id: int,
    ) -> str:
        prepare_calls.append(user_id)
        return collection_name

    monkeypatch.setattr(
        agent_kb_service,
        "_prepare_collection_impl",
        fake_prepare_collection_impl,
    )

    collection = await facade.prepare_agent_collection(
        collection_name="demo",
        ingestion_config=IngestionConfig(),
        user_id=7,
    )

    assert collection == "demo"
    assert prepare_calls == [7]
    assert metadata_store.saved[-1].owners == []
    assert metadata_store.saved[-1].extra_metadata["kb_storage"] == {
        "backend": "lancedb"
    }


@pytest.mark.asyncio
async def test_ensure_agent_collection_backend_binding_preserves_existing_binding():
    existing = CollectionInfo(
        name="demo",
        extra_metadata={"kb_storage": {"backend": "postgresql"}, "other": "kept"},
    )
    metadata_store = _FakeMetadataStore(existing)
    facade = KBToolCompatibilityFacade(storage_shim=_FakeStorageShim(metadata_store))

    result = await facade.ensure_agent_collection_backend_binding("demo")

    assert result is existing
    assert existing.extra_metadata["kb_storage"] == {"backend": "postgresql"}
    assert existing.extra_metadata["other"] == "kept"
    assert metadata_store.saved == []


@pytest.mark.asyncio
async def test_ensure_agent_collection_backend_binding_preserves_existing_owners():
    existing = CollectionInfo(
        name="demo",
        owners=[3],
        extra_metadata={"kb_storage": {"backend": "postgresql"}, "other": "kept"},
    )
    metadata_store = _FakeMetadataStore(existing)
    facade = KBToolCompatibilityFacade(storage_shim=_FakeStorageShim(metadata_store))

    result = await facade.ensure_agent_collection_backend_binding("demo")

    assert result is existing
    assert result.owners == [3]
    assert result.extra_metadata["kb_storage"] == {"backend": "postgresql"}
    assert result.extra_metadata["other"] == "kept"
    assert metadata_store.saved == []


def test_tool_factories_keep_names_models_and_async_only_sync_errors() -> None:
    facade = KBToolCompatibilityFacade()

    list_tool = facade.get_list_knowledge_bases_tool(allowed_collections=["kb1"])
    search_tool = facade.get_knowledge_search_tool(allowed_collections=["kb1"])

    assert isinstance(list_tool, ListKnowledgeBasesTool)
    assert list_tool.name == "list_knowledge_bases"
    assert list_tool.args_type().__name__ == "ListKnowledgeBasesArgs"
    assert list_tool.return_type().__name__ == "ListKnowledgeBasesResult"
    assert isinstance(search_tool, KnowledgeSearchTool)
    assert search_tool.name == "knowledge_search"
    assert search_tool.args_type().__name__ == "KnowledgeSearchArgs"
    assert search_tool.return_type().__name__ == "KnowledgeSearchResult"
    assert not inspect.iscoroutinefunction(list_tool.run_json_sync)
    assert not inspect.iscoroutinefunction(search_tool.run_json_sync)

    with pytest.raises(
        NotImplementedError,
        match="ListKnowledgeBasesTool only supports async execution.",
    ):
        list_tool.run_json_sync({})
    with pytest.raises(
        NotImplementedError,
        match="KnowledgeSearchTool only supports async execution.",
    ):
        search_tool.run_json_sync({})


@pytest.mark.asyncio
async def test_ingestion_tool_factories_keep_tool_names_and_sync_errors() -> None:
    facade = KBToolCompatibilityFacade()
    config = MagicMock()
    config.get_user_id.return_value = 7
    config.is_admin.return_value = False

    file_tools = await facade.create_file_ingestion_tools(config)
    web_tools = await facade.create_web_ingestion_tools(config)

    assert [tool.name for tool in file_tools] == ["create_knowledge_base_from_file"]
    assert [tool.name for tool in web_tools] == ["create_knowledge_base_from_url"]
    with pytest.raises(NotImplementedError, match="Only supports async execution."):
        file_tools[0].run_json_sync({})
    with pytest.raises(NotImplementedError, match="Only supports async execution."):
        web_tools[0].run_json_sync({})


def test_coordinator_accepts_injected_tool_facade() -> None:
    facade = KBToolCompatibilityFacade()
    coordinator = KBCoordinator(tool_compatibility=facade)

    assert coordinator.tool_compatibility is facade
    assert coordinator.tools is facade
