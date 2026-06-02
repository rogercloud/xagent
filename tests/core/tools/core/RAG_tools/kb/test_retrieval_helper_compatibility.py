"""Tests for the KB retrieval helper compatibility facade."""

from __future__ import annotations

import inspect
from datetime import datetime, timezone
from typing import Any, Optional

import pytest

from xagent.core.tools.core.RAG_tools.core.schemas import IndexResult, SearchResult


class _FakeVectorStore:
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self._rows = rows
        self.create_index_calls: list[tuple[str, bool]] = []
        self.sync_search_calls: list[dict[str, Any]] = []
        self.async_search_calls: list[dict[str, Any]] = []

    def create_index(self, model_tag: str, readonly: bool) -> IndexResult:
        self.create_index_calls.append((model_tag, readonly))
        return IndexResult(
            status="readonly" if readonly else "index_ready",
            advice="Readonly mode - no index operations" if readonly else None,
            fts_enabled=False,
        )

    def search_vectors_by_model(self, **kwargs: Any) -> list[dict[str, Any]]:
        self.sync_search_calls.append(kwargs)
        return self._matching_rows(
            filters=kwargs["filters"],
            user_id=kwargs["user_id"],
            is_admin=kwargs["is_admin"],
        )

    async def search_vectors_by_model_async(
        self, **kwargs: Any
    ) -> list[dict[str, Any]]:
        self.async_search_calls.append(kwargs)
        return self._matching_rows(
            filters=kwargs["filters"],
            user_id=kwargs["user_id"],
            is_admin=kwargs["is_admin"],
        )

    def _matching_rows(
        self,
        filters: Any,
        user_id: Optional[int],
        is_admin: bool,
    ) -> list[dict[str, Any]]:
        rows = [row for row in self._rows if _matches_filter_expr(row, filters)]
        if is_admin:
            return rows
        return [row for row in rows if row.get("user_id") == user_id]


class _FakeStorageShim:
    def __init__(self, vector_store: _FakeVectorStore) -> None:
        self.vector_store = vector_store

    def get_vector_index_store(self) -> _FakeVectorStore:
        return self.vector_store


def _matches_filter_expr(record: dict[str, Any], expr: Any) -> bool:
    if expr is None:
        return True
    if isinstance(expr, (tuple, list)):
        return all(_matches_filter_expr(record, item) for item in expr)

    operator = getattr(expr, "operator", None)
    value = getattr(expr, "value", None)
    field = getattr(expr, "field", "")
    operator_value = getattr(operator, "value", operator)
    record_value = record.get(field)

    if operator_value == "eq":
        return record_value == value
    if operator_value == "gte":
        return record_value >= value
    return False


def _filter_conditions(expr: Any) -> list[tuple[str, str, Any]]:
    if expr is None:
        return []
    if isinstance(expr, (tuple, list)):
        conditions: list[tuple[str, str, Any]] = []
        for item in expr:
            conditions.extend(_filter_conditions(item))
        return conditions
    operator = getattr(expr, "operator", None)
    return [
        (
            getattr(expr, "field"),
            getattr(operator, "value", operator),
            getattr(expr, "value"),
        )
    ]


def _search_row(
    *,
    collection: str = "docs",
    doc_id: str = "doc-1",
    chunk_id: str = "chunk-1",
    user_id: int = 7,
    page_number: int = 3,
    distance: float = 3.0,
) -> dict[str, Any]:
    return {
        "collection": collection,
        "doc_id": doc_id,
        "chunk_id": chunk_id,
        "text": f"text for {doc_id}",
        "_distance": distance,
        "parse_hash": f"parse-{doc_id}",
        "created_at": datetime(2026, 1, 1, tzinfo=timezone.utc),
        "metadata": '{"page": 3, "section": "intro"}',
        "user_id": user_id,
        "page_number": page_number,
    }


def _signature_shape(callable_obj: Any) -> list[tuple[str, Any, Any]]:
    return [
        (name, parameter.kind, parameter.default)
        for name, parameter in inspect.signature(callable_obj).parameters.items()
    ]


def test_kb_retrieval_helper_facade_public_surface_imports() -> None:
    """Given the KB package, the retrieval helper facade is publicly importable."""
    import xagent.core.tools.core.RAG_tools.kb as kb
    from xagent.core.tools.core.RAG_tools.kb import (
        KBRetrievalHelperCompatibilityFacade,
        get_kb_coordinator,
        reset_kb_coordinator_for_tests,
    )

    assert hasattr(kb, "KBRetrievalHelperCompatibilityFacade")
    reset_kb_coordinator_for_tests()
    coordinator = get_kb_coordinator()
    assert isinstance(
        coordinator.retrieval_helper_compatibility,
        KBRetrievalHelperCompatibilityFacade,
    )
    assert coordinator.retrieval_helper is coordinator.retrieval_helper_compatibility


def test_retrieval_facade_methods_match_public_helper_signatures() -> None:
    """Given legacy helpers, facade methods preserve their call signatures."""
    from xagent.core.tools.core.RAG_tools.kb import (
        KBRetrievalHelperCompatibilityFacade,
    )
    from xagent.core.tools.core.RAG_tools.retrieval import format_context, search_engine

    facade = KBRetrievalHelperCompatibilityFacade(
        storage_shim=_FakeStorageShim(_FakeVectorStore([]))
    )

    assert _signature_shape(facade.search_dense_engine) == _signature_shape(
        search_engine.search_dense_engine
    )
    assert _signature_shape(facade.search_dense_engine_async) == _signature_shape(
        search_engine.search_dense_engine_async
    )
    assert _signature_shape(facade.format_search_results_for_llm) == _signature_shape(
        format_context.format_search_results_for_llm
    )


@pytest.mark.asyncio
async def test_public_retrieval_helpers_keep_sync_async_shapes_and_route_through_facade(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Given public retrieval helper calls, they route through the facade."""
    from xagent.core.tools.core.RAG_tools.retrieval import format_context, search_engine

    calls: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []

    class _FakeFacade:
        def search_dense_engine(self, *args: Any, **kwargs: Any):
            calls.append(("sync", args, kwargs))
            return ([], "index_ready", None)

        async def search_dense_engine_async(self, *args: Any, **kwargs: Any):
            calls.append(("async", args, kwargs))
            return ([], "readonly", "advice")

        def format_search_results_for_llm(self, *args: Any, **kwargs: Any) -> str:
            calls.append(("format", args, kwargs))
            return "formatted"

    facade = _FakeFacade()
    monkeypatch.setattr(
        search_engine,
        "_get_retrieval_helper_compatibility_facade",
        lambda: facade,
    )
    monkeypatch.setattr(
        format_context,
        "_get_retrieval_helper_compatibility_facade",
        lambda: facade,
    )

    assert not inspect.iscoroutinefunction(search_engine.search_dense_engine)
    assert inspect.iscoroutinefunction(search_engine.search_dense_engine_async)
    assert not inspect.iscoroutinefunction(format_context.format_search_results_for_llm)

    assert search_engine.search_dense_engine(
        "docs",
        "model-a",
        [0.1, 0.2],
        top_k=2,
        filters={"doc_id": "doc-1"},
        readonly=True,
        nprobes=4,
        refine_factor=8,
        user_id=7,
        is_admin=False,
    ) == ([], "index_ready", None)
    assert await search_engine.search_dense_engine_async(
        "docs",
        "model-a",
        [0.1, 0.2],
        top_k=2,
        user_id=7,
        is_admin=True,
    ) == ([], "readonly", "advice")
    assert format_context.format_search_results_for_llm([], top_k=1) == "formatted"

    assert calls == [
        (
            "sync",
            (),
            {
                "collection": "docs",
                "model_tag": "model-a",
                "query_vector": [0.1, 0.2],
                "top_k": 2,
                "filters": {"doc_id": "doc-1"},
                "readonly": True,
                "nprobes": 4,
                "refine_factor": 8,
                "user_id": 7,
                "is_admin": False,
            },
        ),
        (
            "async",
            (),
            {
                "collection": "docs",
                "model_tag": "model-a",
                "query_vector": [0.1, 0.2],
                "top_k": 2,
                "filters": None,
                "readonly": False,
                "nprobes": None,
                "refine_factor": None,
                "user_id": 7,
                "is_admin": True,
            },
        ),
        (
            "format",
            ([],),
            {"top_k": 1, "include_metadata": False, "separator": "\n---\n"},
        ),
    ]


def test_retrieval_facade_preserves_sync_tuple_filter_scope_and_conversion() -> None:
    """Given sync dense helper calls, existing tuple and filtering behavior stays."""
    from xagent.core.tools.core.RAG_tools.kb import (
        KBRetrievalHelperCompatibilityFacade,
    )

    vector_store = _FakeVectorStore(
        [
            _search_row(),
            _search_row(collection="other", doc_id="doc-1", user_id=7),
            _search_row(doc_id="doc-1", user_id=8),
            _search_row(doc_id="doc-1", user_id=7, page_number=1),
        ]
    )
    facade = KBRetrievalHelperCompatibilityFacade(
        storage_shim=_FakeStorageShim(vector_store)
    )

    results, index_status, index_advice = facade.search_dense_engine(
        "docs",
        "model-a",
        [0.5, 0.25],
        top_k=5,
        filters={"doc_id": "doc-1", "page_number": {"operator": "gte", "value": 2}},
        readonly=True,
        user_id=7,
        is_admin=False,
    )

    assert index_status == "readonly"
    assert index_advice == "Readonly mode - no index operations"
    assert vector_store.create_index_calls == [("model-a", True)]
    assert len(results) == 1
    assert isinstance(results[0], SearchResult)
    assert results[0].doc_id == "doc-1"
    assert results[0].score == 0.25
    assert results[0].metadata == {"page": 3, "section": "intro"}

    search_call = vector_store.sync_search_calls[0]
    assert search_call["model_tag"] == "model-a"
    assert search_call["query_vector"] == [0.5, 0.25]
    assert search_call["top_k"] == 5
    assert search_call["vector_column_name"] == "vector"
    assert search_call["user_id"] == 7
    assert search_call["is_admin"] is False
    assert _filter_conditions(search_call["filters"]) == [
        ("collection", "eq", "docs"),
        ("doc_id", "eq", "doc-1"),
        ("page_number", "gte", 2),
    ]


def test_retrieval_facade_falls_back_to_request_user_scope() -> None:
    """Given omitted user scope, facade search uses request-scoped values."""
    from xagent.core.tools.core.RAG_tools.kb import (
        KBRetrievalHelperCompatibilityFacade,
    )
    from xagent.core.tools.core.RAG_tools.utils.user_scope import user_scope_context

    vector_store = _FakeVectorStore(
        [
            _search_row(doc_id="doc-1", user_id=7),
            _search_row(doc_id="doc-1", user_id=8),
        ]
    )
    facade = KBRetrievalHelperCompatibilityFacade(
        storage_shim=_FakeStorageShim(vector_store)
    )

    with user_scope_context(user_id=7, is_admin=False):
        results, _, _ = facade.search_dense_engine(
            "docs",
            "model-a",
            [0.5],
            top_k=5,
            filters={"doc_id": "doc-1"},
            readonly=True,
        )

    assert [result.doc_id for result in results] == ["doc-1"]
    search_call = vector_store.sync_search_calls[0]
    assert search_call["user_id"] == 7
    assert search_call["is_admin"] is False


@pytest.mark.asyncio
async def test_retrieval_facade_async_falls_back_to_request_user_scope() -> None:
    """Given omitted user scope, facade async search uses request-scoped values."""
    from xagent.core.tools.core.RAG_tools.kb import (
        KBRetrievalHelperCompatibilityFacade,
    )
    from xagent.core.tools.core.RAG_tools.utils.user_scope import user_scope_context

    vector_store = _FakeVectorStore(
        [
            _search_row(doc_id="doc-1", user_id=7),
            _search_row(doc_id="doc-1", user_id=8),
        ]
    )
    facade = KBRetrievalHelperCompatibilityFacade(
        storage_shim=_FakeStorageShim(vector_store)
    )

    with user_scope_context(user_id=7, is_admin=False):
        results, _, _ = await facade.search_dense_engine_async(
            "docs",
            "model-a",
            [0.5],
            top_k=5,
            filters={"doc_id": "doc-1"},
            readonly=True,
        )

    assert [result.doc_id for result in results] == ["doc-1"]
    search_call = vector_store.async_search_calls[0]
    assert search_call["user_id"] == 7
    assert search_call["is_admin"] is False


def test_public_retrieval_helper_falls_back_to_request_user_scope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Given omitted user scope, public helper lets facade read request context."""
    from xagent.core.tools.core.RAG_tools.retrieval import search_engine
    from xagent.core.tools.core.RAG_tools.utils.user_scope import user_scope_context

    vector_store = _FakeVectorStore(
        [
            _search_row(doc_id="doc-1", user_id=7),
            _search_row(doc_id="doc-1", user_id=8),
        ]
    )
    facade = search_engine._get_retrieval_helper_compatibility_facade()
    monkeypatch.setattr(
        facade,
        "_storage_shim",
        _FakeStorageShim(vector_store),
    )

    with user_scope_context(user_id=7, is_admin=False):
        results, _, _ = search_engine.search_dense_engine(
            "docs",
            "model-a",
            [0.5],
            top_k=5,
            filters={"doc_id": "doc-1"},
            readonly=True,
        )

    assert [result.doc_id for result in results] == ["doc-1"]
    search_call = vector_store.sync_search_calls[0]
    assert search_call["user_id"] == 7
    assert search_call["is_admin"] is False


@pytest.mark.asyncio
async def test_public_retrieval_helper_async_falls_back_to_request_user_scope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Given omitted user scope, public async helper lets facade read context."""
    from xagent.core.tools.core.RAG_tools.retrieval import search_engine
    from xagent.core.tools.core.RAG_tools.utils.user_scope import user_scope_context

    vector_store = _FakeVectorStore(
        [
            _search_row(doc_id="doc-1", user_id=7),
            _search_row(doc_id="doc-1", user_id=8),
        ]
    )
    facade = search_engine._get_retrieval_helper_compatibility_facade()
    monkeypatch.setattr(
        facade,
        "_storage_shim",
        _FakeStorageShim(vector_store),
    )

    with user_scope_context(user_id=7, is_admin=False):
        results, _, _ = await search_engine.search_dense_engine_async(
            "docs",
            "model-a",
            [0.5],
            top_k=5,
            filters={"doc_id": "doc-1"},
            readonly=True,
        )

    assert [result.doc_id for result in results] == ["doc-1"]
    search_call = vector_store.async_search_calls[0]
    assert search_call["user_id"] == 7
    assert search_call["is_admin"] is False


def test_retrieval_search_after_rollback_cleanup_returns_no_removed_artifacts() -> None:
    """Given rollback-cleaned artifacts, search does not return removed rows."""
    from xagent.core.tools.core.RAG_tools.kb import (
        KBRetrievalHelperCompatibilityFacade,
    )

    vector_store = _FakeVectorStore(
        [
            _search_row(collection="other", doc_id="doc-rolled-back", user_id=7),
            _search_row(collection="docs", doc_id="doc-rolled-back", user_id=8),
            _search_row(collection="docs", doc_id="doc-active", user_id=7),
        ]
    )
    facade = KBRetrievalHelperCompatibilityFacade(
        storage_shim=_FakeStorageShim(vector_store)
    )

    results, _, _ = facade.search_dense_engine(
        "docs",
        "model-a",
        [0.5],
        top_k=5,
        filters={"doc_id": "doc-rolled-back"},
        readonly=True,
        user_id=7,
        is_admin=False,
    )

    assert results == []
    assert _filter_conditions(vector_store.sync_search_calls[0]["filters"]) == [
        ("collection", "eq", "docs"),
        ("doc_id", "eq", "doc-rolled-back"),
    ]


def test_retrieval_incomplete_rollback_artifacts_keep_existing_filters() -> None:
    """Given rollback leftovers, retrieval still applies collection and user scope."""
    from xagent.core.tools.core.RAG_tools.kb import (
        KBRetrievalHelperCompatibilityFacade,
    )

    vector_store = _FakeVectorStore(
        [
            _search_row(
                collection="docs",
                doc_id="doc-rollback-leftover",
                user_id=8,
                distance=0.0,
            ),
            _search_row(
                collection="other",
                doc_id="doc-rollback-leftover",
                user_id=7,
                distance=1.0,
            ),
            _search_row(collection="docs", doc_id="doc-visible", user_id=7),
        ]
    )
    facade = KBRetrievalHelperCompatibilityFacade(
        storage_shim=_FakeStorageShim(vector_store)
    )

    non_admin_results, _, _ = facade.search_dense_engine(
        "docs",
        "model-a",
        [0.5],
        top_k=5,
        filters={"doc_id": "doc-rollback-leftover"},
        readonly=True,
        user_id=7,
        is_admin=False,
    )
    admin_results, _, _ = facade.search_dense_engine(
        "docs",
        "model-a",
        [0.5],
        top_k=5,
        filters={"doc_id": "doc-rollback-leftover"},
        readonly=True,
        user_id=None,
        is_admin=True,
    )

    assert non_admin_results == []
    assert [result.doc_id for result in admin_results] == ["doc-rollback-leftover"]
    assert [result.score for result in admin_results] == [1.0]


def test_retrieval_facade_preserves_invalid_legacy_filter_errors() -> None:
    """Given invalid legacy filters, facade search keeps the existing error."""
    from xagent.core.tools.core.RAG_tools.kb import (
        KBRetrievalHelperCompatibilityFacade,
    )

    facade = KBRetrievalHelperCompatibilityFacade(
        storage_shim=_FakeStorageShim(_FakeVectorStore([]))
    )

    with pytest.raises(ValueError, match="Unknown filter operator: between"):
        facade.search_dense_engine(
            "docs",
            "model-a",
            [0.5],
            top_k=5,
            filters={
                "page_number": {
                    "operator": "between",
                    "value": [1, 3],
                }
            },
            readonly=True,
            user_id=7,
            is_admin=False,
        )


def test_retrieval_facade_preserves_filter_depth_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Given too-deep parsed filters, facade search keeps the depth guard."""
    from xagent.core.tools.core.RAG_tools.kb import (
        KBRetrievalHelperCompatibilityFacade,
    )
    from xagent.core.tools.core.RAG_tools.retrieval import search_engine
    from xagent.core.tools.core.RAG_tools.storage.contracts import (
        FilterCondition,
        FilterOperator,
    )

    too_deep_filter: Any = FilterCondition(
        field="doc_id",
        operator=FilterOperator.EQ,
        value="doc-1",
    )
    for _ in range(12):
        too_deep_filter = [too_deep_filter]

    monkeypatch.setattr(
        search_engine,
        "parse_legacy_filters",
        lambda _filters: too_deep_filter,
    )

    facade = KBRetrievalHelperCompatibilityFacade(
        storage_shim=_FakeStorageShim(_FakeVectorStore([]))
    )

    with pytest.raises(ValueError, match="Filter expression depth exceeds"):
        facade.search_dense_engine(
            "docs",
            "model-a",
            [0.5],
            top_k=5,
            filters={"doc_id": "doc-1"},
            readonly=True,
            user_id=7,
            is_admin=False,
        )


@pytest.mark.asyncio
async def test_retrieval_facade_preserves_async_tuple_shape_and_admin_filtering() -> (
    None
):
    """Given async dense helper calls, admin scope and tuple shape stay stable."""
    from xagent.core.tools.core.RAG_tools.kb import (
        KBRetrievalHelperCompatibilityFacade,
    )

    vector_store = _FakeVectorStore(
        [
            _search_row(doc_id="doc-1", user_id=7, distance=0.0),
            _search_row(doc_id="doc-2", user_id=8, distance=1.0),
            _search_row(collection="other", doc_id="doc-3", user_id=8, distance=2.0),
        ]
    )
    facade = KBRetrievalHelperCompatibilityFacade(
        storage_shim=_FakeStorageShim(vector_store)
    )

    results, index_status, index_advice = await facade.search_dense_engine_async(
        "docs",
        "model-a",
        [1.0, 0.0],
        top_k=3,
        readonly=False,
        user_id=None,
        is_admin=True,
    )

    assert index_status == "index_ready"
    assert index_advice is None
    assert vector_store.create_index_calls == [("model-a", False)]
    assert [result.doc_id for result in results] == ["doc-1", "doc-2"]
    assert [result.score for result in results] == [1.0, 0.5]

    search_call = vector_store.async_search_calls[0]
    assert search_call["user_id"] is None
    assert search_call["is_admin"] is True
    assert _filter_conditions(search_call["filters"]) == [("collection", "eq", "docs")]


def test_retrieval_facade_preserves_llm_context_formatting() -> None:
    """Given formatting calls through the facade, context strings stay unchanged."""
    from xagent.core.tools.core.RAG_tools.kb import (
        KBRetrievalHelperCompatibilityFacade,
    )

    result = SearchResult(
        doc_id="doc-1",
        chunk_id="chunk-1",
        text="retrieved text",
        score=0.75,
        parse_hash="parse-1",
        model_tag="model-a",
        created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        metadata={"page": 1},
    )

    assert KBRetrievalHelperCompatibilityFacade().format_search_results_for_llm(
        [result],
        include_metadata=True,
    ) == (
        "[1]\n"
        "Document ID: doc-1, Chunk ID: chunk-1, Score: 0.7500, "
        "Metadata: {'page': 1}\n"
        "retrieved text"
    )
