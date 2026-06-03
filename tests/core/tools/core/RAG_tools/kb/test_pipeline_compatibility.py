"""Tests for KB pipeline compatibility facade."""

from __future__ import annotations

from typing import Optional

from xagent.core.tools.core.RAG_tools.core.schemas import CollectionInfo
from xagent.core.tools.core.RAG_tools.kb import KBPipelineCompatibilityFacade


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


def test_ensure_collection_backend_binding_sets_lancedb_when_missing() -> None:
    metadata_store = _FakeMetadataStore(CollectionInfo(name="demo"))
    facade = KBPipelineCompatibilityFacade(
        storage_shim=_FakeStorageShim(metadata_store)
    )

    updated = facade.ensure_collection_backend_binding("demo")

    assert updated is not None
    assert updated.extra_metadata["kb_storage"] == {"backend": "lancedb"}
    assert metadata_store.saved == [updated]


def test_ensure_collection_backend_binding_preserves_existing_binding() -> None:
    existing_binding = {"backend": "postgresql", "dsn": "kept"}
    metadata_store = _FakeMetadataStore(
        CollectionInfo(
            name="demo",
            extra_metadata={"kb_storage": existing_binding, "other": "value"},
        )
    )
    facade = KBPipelineCompatibilityFacade(
        storage_shim=_FakeStorageShim(metadata_store)
    )

    existing = facade.ensure_collection_backend_binding("demo")

    assert existing is metadata_store.collection
    assert existing is not None
    assert existing.extra_metadata["kb_storage"] == existing_binding
    assert existing.extra_metadata["other"] == "value"
    assert metadata_store.saved == []


def test_ensure_collection_backend_binding_ignores_missing_collection() -> None:
    metadata_store = _FakeMetadataStore(None)
    facade = KBPipelineCompatibilityFacade(
        storage_shim=_FakeStorageShim(metadata_store)
    )

    assert facade.ensure_collection_backend_binding("missing") is None
    assert metadata_store.saved == []
