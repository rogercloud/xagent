"""Tests for the KB API compatibility facade."""

from __future__ import annotations

import inspect
from typing import Optional

import pytest

from xagent.core.tools.core.RAG_tools.core.schemas import (
    CollectionInfo,
    IngestionResult,
)
from xagent.core.tools.core.RAG_tools.kb import (
    KBApiCompatibilityFacade,
    KBCoordinator,
    KBOperationCompatibilityFacade,
    RollbackStatus,
    get_kb_coordinator,
    reset_kb_coordinator_for_tests,
)


class _FakeMetadataStore:
    def __init__(self, collection: Optional[CollectionInfo]) -> None:
        self.collection = collection
        self.saved_configs: list[tuple[str, str, int]] = []
        self.saved_collections: list[CollectionInfo] = []

    async def get_collection(self, collection: str) -> CollectionInfo:
        if self.collection is None or self.collection.name != collection:
            raise ValueError(f"Collection {collection!r} not found")
        return self.collection

    async def save_collection(self, collection: CollectionInfo) -> None:
        self.saved_collections.append(collection)
        self.collection = collection

    async def save_collection_config(
        self,
        *,
        collection: str,
        config_json: str,
        user_id: int,
    ) -> None:
        self.saved_configs.append((collection, config_json, user_id))


class _ConfigOnlyMetadataStore:
    def __init__(self) -> None:
        self.saved_configs: list[tuple[str, str, int]] = []

    async def save_collection_config(
        self,
        *,
        collection: str,
        config_json: str,
        user_id: int,
    ) -> None:
        self.saved_configs.append((collection, config_json, user_id))


class _FakeStorageShim:
    def __init__(self, metadata_store: object) -> None:
        self.metadata_store = metadata_store

    def get_metadata_store(self) -> object:
        return self.metadata_store


def test_kb_api_facade_public_surface_imports() -> None:
    import xagent.core.tools.core.RAG_tools.kb as kb

    assert hasattr(kb, "KBApiCompatibilityFacade")
    reset_kb_coordinator_for_tests()
    assert isinstance(get_kb_coordinator().api_compatibility, KBApiCompatibilityFacade)
    assert get_kb_coordinator().api is get_kb_coordinator().api_compatibility


@pytest.mark.asyncio
async def test_save_collection_config_creates_owner_neutral_backend_binding() -> None:
    metadata_store = _FakeMetadataStore(None)
    facade = KBApiCompatibilityFacade(storage_shim=_FakeStorageShim(metadata_store))

    await facade.save_collection_config(
        collection="demo",
        config_json="{}",
        user_id=7,
    )

    assert metadata_store.saved_configs == [("demo", "{}", 7)]
    assert metadata_store.saved_collections == [metadata_store.collection]
    assert metadata_store.collection is not None
    assert metadata_store.collection.owners == []
    assert metadata_store.collection.extra_metadata["kb_storage"] == {
        "backend": "lancedb"
    }


@pytest.mark.asyncio
async def test_save_collection_config_preserves_existing_backend_binding() -> None:
    existing = CollectionInfo(
        name="demo",
        extra_metadata={"kb_storage": {"backend": "postgresql"}, "other": "kept"},
    )
    metadata_store = _FakeMetadataStore(existing)
    facade = KBApiCompatibilityFacade(storage_shim=_FakeStorageShim(metadata_store))

    await facade.save_collection_config(
        collection="demo",
        config_json='{"chunk_size": 1000}',
        user_id=7,
    )

    assert metadata_store.saved_configs == [("demo", '{"chunk_size": 1000}', 7)]
    assert existing.extra_metadata["kb_storage"] == {"backend": "postgresql"}
    assert existing.extra_metadata["other"] == "kept"
    assert metadata_store.saved_collections == []


@pytest.mark.asyncio
async def test_save_collection_config_tolerates_config_only_test_stores() -> None:
    metadata_store = _ConfigOnlyMetadataStore()
    facade = KBApiCompatibilityFacade(storage_shim=_FakeStorageShim(metadata_store))

    await facade.save_collection_config(
        collection="demo",
        config_json="{}",
        user_id=7,
    )

    assert metadata_store.saved_configs == [("demo", "{}", 7)]


def test_coordinator_accepts_injected_api_facade() -> None:
    facade = KBApiCompatibilityFacade()
    coordinator = KBCoordinator(api_compatibility=facade)

    assert coordinator.api_compatibility is facade
    assert coordinator.api is facade


def test_api_operation_result_consumes_new_operation_outcome() -> None:
    operation_facade = KBOperationCompatibilityFacade()
    coordinator = KBCoordinator(operation_compatibility=operation_facade)
    facade = coordinator.api_compatibility

    def operation() -> IngestionResult:
        with operation_facade.start_operation(
            operation_type="document_ingestion",
            collection="demo",
        ) as active_operation:
            active_operation.finish(
                status="error",
                rollback_status=RollbackStatus.INCOMPLETE,
                side_effects_may_remain=True,
            )
        return IngestionResult(status="error", message="failed")

    api_result = facade.run_with_operation_outcome(
        operation,
        operation_type="document_ingestion",
        collection="demo",
    )

    assert api_result.result.status == "error"
    assert api_result.operation_outcome is operation_facade.last_outcome
    cleanup_decision = facade.failed_ingest_cleanup_decision(api_result)
    assert cleanup_decision.successful_documents == 0
    assert cleanup_decision.side_effects_may_remain is True

    completed_rollback = facade.with_rollback_complete(api_result, True)
    cleanup_after_rollback = facade.failed_ingest_cleanup_decision(completed_rollback)
    assert cleanup_after_rollback.side_effects_may_remain is False


def test_api_operation_result_ignores_stale_operation_outcome() -> None:
    operation_facade = KBOperationCompatibilityFacade()
    coordinator = KBCoordinator(operation_compatibility=operation_facade)
    facade = coordinator.api_compatibility

    with operation_facade.start_operation(
        operation_type="document_ingestion",
        collection="demo",
    ) as active_operation:
        active_operation.finish(status="success")
    assert operation_facade.last_outcome is not None

    api_result = facade.run_with_operation_outcome(
        lambda: IngestionResult(status="error", message="patched failure"),
        operation_type="document_ingestion",
        collection="demo",
    )

    assert api_result.operation_outcome is None
    cleanup_decision = facade.failed_ingest_cleanup_decision(api_result)
    assert cleanup_decision.side_effects_may_remain is False


def test_list_document_records_omits_none_max_results(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from xagent.core.tools.core.RAG_tools.storage import factory

    calls: list[dict[str, object]] = []

    class VectorStore:
        def list_document_records(self, **kwargs: object) -> list[str]:
            calls.append(kwargs)
            return ["record"]

    monkeypatch.setattr(factory, "get_vector_index_store", lambda: VectorStore())

    records = KBApiCompatibilityFacade().list_document_records(
        collection_name="demo",
        user_id=7,
        is_admin=False,
    )

    assert records == ["record"]
    assert calls == [
        {
            "collection_name": "demo",
            "user_id": 7,
            "is_admin": False,
        }
    ]


@pytest.mark.asyncio
async def test_rename_collection_routes_storage_metadata_and_status(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from xagent.core.tools.core.RAG_tools.storage import factory

    calls: list[tuple[str, str, int, bool]] = []

    class VectorStore:
        def rename_collection_data(
            self,
            *,
            collection_name: str,
            new_name: str,
            user_id: int,
            is_admin: bool,
        ) -> list[str]:
            calls.append((collection_name, new_name, user_id, is_admin))
            return ["vector warning"]

    class MetadataStore:
        async def rename_collection(
            self,
            *,
            old_name: str,
            new_name: str,
            user_id: int,
            is_admin: bool,
        ) -> None:
            calls.append((old_name, new_name, user_id, is_admin))

    class StatusStore:
        def rename_collection_status(
            self,
            *,
            old_name: str,
            new_name: str,
            user_id: int,
            is_admin: bool,
        ) -> list[str]:
            calls.append((old_name, new_name, user_id, is_admin))
            return ["status warning"]

    monkeypatch.setattr(factory, "get_vector_index_store", lambda: VectorStore())
    monkeypatch.setattr(factory, "get_metadata_store", lambda: MetadataStore())
    monkeypatch.setattr(factory, "get_ingestion_status_store", lambda: StatusStore())

    facade = KBApiCompatibilityFacade()

    assert facade.rename_collection_data(
        collection_name="old",
        new_name="new",
        user_id=7,
        is_admin=False,
    ) == ["vector warning"]
    await facade.rename_collection_metadata(
        old_name="old",
        new_name="new",
        user_id=7,
        is_admin=False,
    )
    assert facade.rename_collection_status(
        old_name="old",
        new_name="new",
        user_id=7,
        is_admin=False,
    ) == ["status warning"]
    assert calls == [
        ("old", "new", 7, False),
        ("old", "new", 7, False),
        ("old", "new", 7, False),
    ]


def test_web_api_search_wrapper_routes_through_api_facade(monkeypatch) -> None:
    from xagent.web.api import kb as kb_api

    sentinel = object()
    calls: list[tuple[str, str, int, bool]] = []

    class Facade:
        def run_document_search(
            self,
            collection: str,
            query_text: str,
            **kwargs: object,
        ) -> object:
            calls.append(
                (
                    collection,
                    query_text,
                    int(kwargs["user_id"]),
                    bool(kwargs["is_admin"]),
                )
            )
            return sentinel

    monkeypatch.setattr(
        kb_api,
        "_get_api_compatibility_facade",
        lambda: Facade(),
    )

    result = kb_api.run_document_search(
        collection="demo",
        query_text="question",
        user_id=7,
        is_admin=True,
    )

    assert result is sentinel
    assert calls == [("demo", "question", 7, True)]


def test_web_api_delete_document_wrapper_routes_through_api_facade(
    monkeypatch,
) -> None:
    from xagent.web.api import kb as kb_api

    sentinel = object()
    calls: list[tuple[str, str, int, bool]] = []

    class Facade:
        def delete_document(
            self,
            collection: str,
            doc_id: str,
            user_id: int,
            is_admin: bool,
        ) -> object:
            calls.append((collection, doc_id, user_id, is_admin))
            return sentinel

    monkeypatch.setattr(
        kb_api,
        "_get_api_compatibility_facade",
        lambda: Facade(),
    )

    result = kb_api.delete_document(
        collection="demo",
        doc_id="doc-1",
        user_id=7,
        is_admin=True,
    )

    assert result is sentinel
    assert calls == [("demo", "doc-1", 7, True)]


def test_delete_document_api_does_not_shadow_api_facade_wrapper() -> None:
    from xagent.web.api import kb as kb_api

    source = inspect.getsource(kb_api.delete_document_api)

    assert "management.collections import delete_document" not in source
