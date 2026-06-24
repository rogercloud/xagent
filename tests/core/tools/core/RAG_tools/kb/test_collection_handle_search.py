"""Tests for the collection handle dense search lifecycle (#511).

The handle owns collection-scoped dense search mechanics: capability guard,
index creation, filter building, score conversion, and result assembly.
Search provider calls (vector store) are mocked.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from xagent.core.tools.core.RAG_tools.core.schemas import (
    DenseSearchResponse,
    IndexStatus,
)
from xagent.core.tools.core.RAG_tools.kb.collection_handle import (
    LanceDBCollectionHandle,
)


def _make_handle(*, supports_search: bool = True):
    """Create a LanceDBCollectionHandle with mocked context/store/capabilities.

    LanceDBCollectionHandle is a frozen dataclass, so we use object.__setattr__
    to inject mocks into the underlying ``context`` field.
    """
    handle = LanceDBCollectionHandle.__new__(LanceDBCollectionHandle)
    ctx = MagicMock()
    ctx.collection = "col1"
    # frozen dataclass — must use object.__setattr__
    object.__setattr__(handle, "context", ctx)

    store = MagicMock()
    ctx.vector_index_store = store

    caps = MagicMock()
    caps.supports_search = supports_search
    ctx.capabilities = caps

    return handle, ctx, store, caps


def _index_result(status="index_ready", advice=None):
    obj = MagicMock()
    obj.status = status
    obj.advice = advice
    obj.fts_enabled = True
    return obj


def test_search_dense_success_score_and_filters():
    handle, ctx, store, _ = _make_handle()
    store.create_index.return_value = _index_result()
    store.search_vectors_by_model.return_value = [
        {
            "doc_id": "d1",
            "chunk_id": "c1",
            "text": "t",
            "parse_hash": "h",
            "created_at": "2026-01-01",
            "metadata": None,
            "_distance": 1.0,
        },
    ]
    resp = handle.search_dense(
        "model-x", [0.1, 0.2], top_k=5, user_id=7, is_admin=False
    )
    assert isinstance(resp, DenseSearchResponse)
    assert resp.status == "success"
    assert resp.total_count == 1
    assert resp.results[0].score == pytest.approx(0.5)  # 1/(1+1.0)
    # collection filter + user scope reached the store
    kwargs = store.search_vectors_by_model.call_args.kwargs
    assert kwargs["model_tag"] == "model-x"
    assert kwargs["user_id"] == 7 and kwargs["is_admin"] is False


def test_search_dense_failure_returns_failed_response():
    handle, _, store, _ = _make_handle()
    store.create_index.side_effect = RuntimeError("boom")
    resp = handle.search_dense("model-x", [0.1])
    assert resp.status == "failed"
    assert resp.results == [] and resp.total_count == 0
    assert resp.index_status == IndexStatus.NO_INDEX
    assert any(w.code == "DENSE_SEARCH_FAILED" for w in resp.warnings)


def test_search_dense_capability_unsupported():
    handle, _, store, _ = _make_handle(supports_search=False)
    resp = handle.search_dense("model-x", [0.1])
    assert resp.status == "failed"
    assert any(w.code == "SEARCH_NOT_SUPPORTED" for w in resp.warnings)
    store.create_index.assert_not_called()  # guard is before any store access


@pytest.mark.asyncio
async def test_search_dense_async_capability_unsupported():
    handle, _, store, _ = _make_handle(supports_search=False)
    resp = await handle.search_dense_async("model-x", [0.1])
    assert resp.status == "failed"
    assert any(w.code == "SEARCH_NOT_SUPPORTED" for w in resp.warnings)
