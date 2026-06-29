"""Tests for the main-pointer lifecycle inline in LanceDBCollectionHandle (#513).

The handle owns collection-scoped main-pointer mechanics: get, set, list, delete.
These tests mirror the assertions in
version_management/test_main_pointer_manager.py:63,165,224.

A lightweight ``_FakeMainPointerStore`` is injected directly into the
``KBCollectionContext`` to avoid real LanceDB I/O (same isolation strategy used
by test_version_compatibility.py).
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from xagent.core.tools.core.RAG_tools.kb.collection_handle import (
    LanceDBCollectionHandle,
)
from xagent.core.tools.core.RAG_tools.kb.models import (
    KBAccessMode,
    KBBackendCapabilities,
    KBCollectionContext,
    KBStorageBackend,
    KBUserScope,
)
from xagent.core.tools.core.RAG_tools.storage.contracts import MainPointerStore
from xagent.core.tools.core.RAG_tools.storage.factory import (
    get_ingestion_status_store,
    get_metadata_store,
    get_vector_index_store,
)


class _FakeMainPointerStore(MainPointerStore):
    """In-memory MainPointerStore for handle-level tests."""

    def __init__(self) -> None:
        self.rows: dict[tuple[str, str, str, Optional[str]], dict[str, Any]] = {}

    def _key(
        self,
        collection: str,
        doc_id: str,
        step_type: str,
        model_tag: Optional[str],
    ) -> tuple[str, str, str, Optional[str]]:
        return (collection, doc_id, step_type, model_tag)

    def get_main_pointer(
        self,
        collection: str,
        doc_id: str,
        step_type: str,
        model_tag: Optional[str] = None,
        user_id: Optional[int] = None,
    ) -> Optional[Dict[str, Any]]:
        row = self.rows.get(self._key(collection, doc_id, step_type, model_tag))
        return dict(row) if row is not None else None

    def set_main_pointer(
        self,
        collection: str,
        doc_id: str,
        step_type: str,
        semantic_id: str,
        technical_id: str,
        model_tag: Optional[str] = None,
        operator: Optional[str] = None,
        user_id: Optional[int] = None,
    ) -> None:
        key = self._key(collection, doc_id, step_type, model_tag)
        existing = self.rows.get(key)
        now = datetime.now(timezone.utc)
        created_at = existing["created_at"] if existing else now
        self.rows[key] = {
            "collection": collection,
            "doc_id": doc_id,
            "step_type": step_type,
            "model_tag": model_tag if model_tag is not None else "",
            "semantic_id": semantic_id,
            "technical_id": technical_id,
            "operator": operator,
            "created_at": created_at,
            "updated_at": now,
        }

    def list_main_pointers(
        self,
        collection: str,
        doc_id: Optional[str] = None,
        user_id: Optional[int] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        rows = [
            dict(row)
            for (row_coll, row_doc, _, _), row in self.rows.items()
            if row_coll == collection and (doc_id is None or row_doc == doc_id)
        ]
        return rows[:limit]

    def delete_main_pointer(
        self,
        collection: str,
        doc_id: str,
        step_type: str,
        model_tag: Optional[str] = None,
        user_id: Optional[int] = None,
    ) -> bool:
        return (
            self.rows.pop(self._key(collection, doc_id, step_type, model_tag), None)
            is not None
        )

    async def get_main_pointer_async(
        self,
        collection: str,
        doc_id: str,
        step_type: str,
        model_tag: Optional[str] = None,
        user_id: Optional[int] = None,
    ) -> Optional[Dict[str, Any]]:
        return self.get_main_pointer(
            collection, doc_id, step_type, model_tag=model_tag, user_id=user_id
        )

    async def set_main_pointer_async(
        self,
        collection: str,
        doc_id: str,
        step_type: str,
        semantic_id: str,
        technical_id: str,
        model_tag: Optional[str] = None,
        operator: Optional[str] = None,
        user_id: Optional[int] = None,
    ) -> None:
        self.set_main_pointer(
            collection,
            doc_id,
            step_type,
            semantic_id,
            technical_id,
            model_tag=model_tag,
            operator=operator,
            user_id=user_id,
        )

    async def list_main_pointers_async(
        self,
        collection: str,
        doc_id: Optional[str] = None,
        user_id: Optional[int] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        return self.list_main_pointers(
            collection, doc_id=doc_id, user_id=user_id, limit=limit
        )

    async def delete_main_pointer_async(
        self,
        collection: str,
        doc_id: str,
        step_type: str,
        model_tag: Optional[str] = None,
        user_id: Optional[int] = None,
    ) -> bool:
        return self.delete_main_pointer(
            collection, doc_id, step_type, model_tag=model_tag, user_id=user_id
        )


def make_handle(
    collection: str = "coll",
    store: Optional[MainPointerStore] = None,
) -> LanceDBCollectionHandle:
    context = KBCollectionContext(
        collection=collection,
        user_scope=KBUserScope(user_id=None, is_admin=True),
        access_mode=KBAccessMode.WRITE,
        allow_create=True,
        hide_missing=True,
        metadata_store=get_metadata_store(),
        vector_index_store=get_vector_index_store(),
        ingestion_status_store=get_ingestion_status_store(),
        main_pointer_store=store if store is not None else _FakeMainPointerStore(),
        backend=KBStorageBackend.LANCEDB,
        capabilities=KBBackendCapabilities.lancedb(),
        collection_info=None,
    )
    return LanceDBCollectionHandle(context)


def test_set_get_roundtrip() -> None:
    """set_main_pointer + get_main_pointer round-trips the pointer correctly.

    Mirrors test_main_pointer_manager.py:63.
    """
    handle = make_handle("roundtrip_coll")
    handle.set_main_pointer(
        "doc-1",
        "parse",
        semantic_id="parse_123",
        technical_id="hash_456",
    )
    result = handle.get_main_pointer("doc-1", "parse")
    assert result is not None
    assert result["semantic_id"] == "parse_123"
    assert result["technical_id"] == "hash_456"
    assert result["collection"] == "roundtrip_coll"
    assert result["doc_id"] == "doc-1"
    assert result["step_type"] == "parse"


def test_delete_returns_bool() -> None:
    """delete_main_pointer returns True when deleted, False when already absent.

    Mirrors test_main_pointer_manager.py:165.
    """
    handle = make_handle("delete_coll")
    handle.set_main_pointer(
        "doc-1",
        "parse",
        semantic_id="parse_1",
        technical_id="hash_1",
    )
    # First delete returns True
    result = handle.delete_main_pointer("doc-1", "parse")
    assert result is True
    # Second delete (already absent) returns False
    result2 = handle.delete_main_pointer("doc-1", "parse")
    assert result2 is False


def test_get_missing_returns_none() -> None:
    """get_main_pointer returns None for a row that doesn't exist."""
    handle = make_handle("missing_coll")
    result = handle.get_main_pointer("nonexistent-doc", "parse")
    assert result is None


def test_model_tag_none_normalized() -> None:
    """model_tag=None passes through to the store (store normalizes to '').

    Mirrors test_main_pointer_manager.py:165 backward-compat check.
    """
    handle = make_handle("modeltag_coll")
    handle.set_main_pointer(
        "doc-1",
        "embed",
        semantic_id="embed_123",
        technical_id="embed_hash",
        model_tag=None,
    )
    # Get with model_tag=None should find the row (None passed straight through)
    result = handle.get_main_pointer("doc-1", "embed", model_tag=None)
    assert result is not None
    assert result["semantic_id"] == "embed_123"


def test_set_preserves_created_at() -> None:
    """set_main_pointer preserves created_at on existing rows (store behavior).

    Mirrors test_main_pointer_manager.py:224.
    """
    handle = make_handle("preserve_coll")
    handle.set_main_pointer(
        "doc-1",
        "parse",
        semantic_id="old_parse",
        technical_id="old_hash",
    )
    first = handle.get_main_pointer("doc-1", "parse")
    assert first is not None
    created_at_first = first.get("created_at")

    # Update the pointer
    handle.set_main_pointer(
        "doc-1",
        "parse",
        semantic_id="new_parse",
        technical_id="new_hash",
    )
    second = handle.get_main_pointer("doc-1", "parse")
    assert second is not None
    assert second["semantic_id"] == "new_parse"
    assert second["technical_id"] == "new_hash"
    # created_at preserved from first write
    assert second.get("created_at") == created_at_first


def test_list_main_pointers() -> None:
    """list_main_pointers returns all pointers for the collection."""
    handle = make_handle("list_coll")
    handle.set_main_pointer(
        "doc-1",
        "parse",
        semantic_id="parse_1",
        technical_id="hash_1",
    )
    handle.set_main_pointer(
        "doc-2",
        "chunk",
        semantic_id="chunk_2",
        technical_id="hash_2",
    )
    pointers = handle.list_main_pointers()
    assert len(pointers) == 2
    doc_ids = {p["doc_id"] for p in pointers}
    assert doc_ids == {"doc-1", "doc-2"}


def test_list_main_pointers_filtered_by_doc_id() -> None:
    """list_main_pointers(doc_id=...) returns only that document's rows."""
    handle = make_handle("list_filter_coll")
    handle.set_main_pointer(
        "doc-1",
        "parse",
        semantic_id="parse_1",
        technical_id="hash_1",
    )
    handle.set_main_pointer(
        "doc-2",
        "parse",
        semantic_id="parse_2",
        technical_id="hash_2",
    )
    pointers = handle.list_main_pointers(doc_id="doc-1")
    assert len(pointers) == 1
    assert pointers[0]["doc_id"] == "doc-1"


# ── Task 4: list_candidates handle tests ─────────────────────────────────────


class _FakeVectorIndexStore:
    """Minimal in-memory VectorIndexStore stub for list_candidates handle tests."""

    def __init__(self, candidates: list) -> None:
        self._candidates = candidates

    def list_version_candidate_rows(
        self,
        collection: str,
        doc_id: str,
        step_type: str,
        model_tag=None,
    ):
        return list(self._candidates)

    async def list_version_candidate_rows_async(
        self,
        collection: str,
        doc_id: str,
        step_type: str,
        model_tag=None,
    ):
        return self.list_version_candidate_rows(
            collection, doc_id, step_type, model_tag
        )


def make_handle_with_vector_store(
    collection: str, candidates: list
) -> "LanceDBCollectionHandle":
    """Make a handle with a fake VectorIndexStore for list_candidates testing."""
    from xagent.core.tools.core.RAG_tools.kb.models import (
        KBAccessMode,
        KBBackendCapabilities,
        KBCollectionContext,
        KBStorageBackend,
        KBUserScope,
    )
    from xagent.core.tools.core.RAG_tools.storage.factory import (
        get_ingestion_status_store,
        get_metadata_store,
    )

    fake_vis = _FakeVectorIndexStore(candidates)
    context = KBCollectionContext(
        collection=collection,
        user_scope=KBUserScope(user_id=None, is_admin=True),
        access_mode=KBAccessMode.WRITE,
        allow_create=True,
        hide_missing=True,
        metadata_store=get_metadata_store(),
        vector_index_store=fake_vis,  # type: ignore[arg-type]
        ingestion_status_store=get_ingestion_status_store(),
        main_pointer_store=_FakeMainPointerStore(),
        backend=KBStorageBackend.LANCEDB,
        capabilities=KBBackendCapabilities.lancedb(),
        collection_info=None,
    )
    return LanceDBCollectionHandle(context)


def test_list_candidates_sort_before_limit() -> None:
    """Sorting must happen before limit (get correct top-N). Mirrors test_list_candidates.py:408."""

    base_time = datetime(2024, 1, 1)
    candidates = [
        {
            "semantic_id": "s1",
            "technical_id": "hash_oldest",
            "params_brief": {},
            "stats": {},
            "state": "candidate",
            "created_at": base_time,
            "operator": "unknown",
        },
        {
            "semantic_id": "s2",
            "technical_id": "hash_middle",
            "params_brief": {},
            "stats": {},
            "state": "candidate",
            "created_at": base_time + timedelta(days=5),
            "operator": "unknown",
        },
        {
            "semantic_id": "s3",
            "technical_id": "hash_newer",
            "params_brief": {},
            "stats": {},
            "state": "candidate",
            "created_at": base_time + timedelta(days=7),
            "operator": "unknown",
        },
        {
            "semantic_id": "s4",
            "technical_id": "hash_newest",
            "params_brief": {},
            "stats": {},
            "state": "candidate",
            "created_at": base_time + timedelta(days=10),
            "operator": "unknown",
        },
        {
            "semantic_id": "s5",
            "technical_id": "hash_second_newest",
            "params_brief": {},
            "stats": {},
            "state": "candidate",
            "created_at": base_time + timedelta(days=8),
            "operator": "unknown",
        },
    ]

    handle = make_handle_with_vector_store("sort_coll", candidates)
    result = handle.list_candidates(
        "doc-1", "parse", limit=3, order_by="created_at desc"
    )

    assert len(result["candidates"]) == 3
    assert result["total_count"] == 5
    assert result["returned_count"] == 3

    technical_ids = [c["technical_id"] for c in result["candidates"]]
    assert technical_ids[0] == "hash_newest"
    assert technical_ids[1] == "hash_second_newest"
    assert technical_ids[2] == "hash_newer"
    assert "hash_oldest" not in technical_ids
    assert "hash_middle" not in technical_ids


def test_list_candidates_state_filter() -> None:
    """State filter narrows candidates. Mirrors test_list_candidates.py:296."""
    candidates = [
        {
            "semantic_id": "s1",
            "technical_id": "hash1",
            "params_brief": {},
            "stats": {},
            "state": "candidate",
            "created_at": datetime(2024, 1, 1),
            "operator": "unknown",
        },
        {
            "semantic_id": "s2",
            "technical_id": "hash2",
            "params_brief": {},
            "stats": {},
            "state": "main",
            "created_at": datetime(2024, 1, 2),
            "operator": "unknown",
        },
    ]
    handle = make_handle_with_vector_store("state_coll", candidates)

    result = handle.list_candidates("doc-1", "parse", state="candidate")
    assert (
        result["total_count"] == 1
    )  # total AFTER state filter (matches _list_candidates_impl behavior)
    assert result["returned_count"] == 1
    assert len(result["candidates"]) == 1
    assert result["candidates"][0]["state"] == "candidate"
    assert result["filters"]["state"] == "candidate"


def test_list_candidates_model_tag_filter() -> None:
    """model_tag is passed through to the store. Mirrors test_list_candidates.py:369."""
    candidates_bge = [
        {
            "semantic_id": "embed_BAAI/bge-large-zh-v1.5_parse_ha",
            "technical_id": "parse_hash1",
            "params_brief": {
                "model": "BAAI/bge-large-zh-v1.5",
                "model_tag": "bge_large",
            },
            "stats": {"upsert_count": 1, "vector_dim": 3},
            "state": "candidate",
            "created_at": datetime(2024, 1, 1),
            "operator": "unknown",
        },
    ]

    handle = make_handle_with_vector_store("mtag_coll", candidates_bge)
    result = handle.list_candidates("doc-1", "embed", model_tag="bge_large")

    assert len(result["candidates"]) == 1
    assert result["total_count"] == 1
    assert result["model_tag"] == "bge_large"


def test_list_candidates_result_dict_shape() -> None:
    """Result dict has all required keys. Mirrors test_list_candidates.py:296."""
    candidates = [
        {
            "semantic_id": "s1",
            "technical_id": "h1",
            "params_brief": {},
            "stats": {},
            "state": "candidate",
            "created_at": datetime(2024, 1, 1),
            "operator": "unknown",
        },
    ]
    handle = make_handle_with_vector_store("shape_coll", candidates)
    result = handle.list_candidates(
        "doc-1",
        "parse",
        model_tag=None,
        state=None,
        limit=50,
        order_by="created_at desc",
    )

    assert "candidates" in result
    assert "total_count" in result
    assert "returned_count" in result
    assert "step_type" in result
    assert "model_tag" in result
    assert "filters" in result
    assert result["step_type"] == "parse"
    assert result["filters"]["state"] is None
    assert result["filters"]["limit"] == 50
    assert result["filters"]["order_by"] == "created_at desc"
