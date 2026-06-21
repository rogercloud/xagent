"""Tests for the collection handle document-row lifecycle (#508).

The abstract ``KBCollectionHandle`` owns collection-scoped document-row
operations; ``LanceDBCollectionHandle`` is the first implementation. These
tests assert equivalence with the behavior locked by the file-level
characterization oracle.

Storage isolation/reset is provided by the autouse ``isolate_rag_storage``
fixture in ``tests/conftest.py``.
"""

from pathlib import Path

import pytest

from xagent.core.tools.core.RAG_tools.core.schemas import RegisterDocumentRequest
from xagent.core.tools.core.RAG_tools.kb.collection_handle import (
    KBCollectionHandle,
    LanceDBCollectionHandle,
)
from xagent.core.tools.core.RAG_tools.kb.models import (
    KBAccessMode,
    KBBackendCapabilities,
    KBCollectionContext,
    KBStorageBackend,
    KBUserScope,
)
from xagent.core.tools.core.RAG_tools.storage.factory import (
    get_metadata_store,
    get_vector_index_store,
)
from xagent.core.tools.core.RAG_tools.utils.string_utils import (
    generate_deterministic_doc_id,
)


def make_handle(collection: str = "coll") -> LanceDBCollectionHandle:
    """Build a LanceDB-backed handle bound to the current test stores."""
    context = KBCollectionContext(
        collection=collection,
        user_scope=KBUserScope(user_id=None, is_admin=True),
        access_mode=KBAccessMode.WRITE,
        allow_create=True,
        hide_missing=True,
        metadata_store=get_metadata_store(),
        vector_index_store=get_vector_index_store(),
        backend=KBStorageBackend.LANCEDB,
        capabilities=KBBackendCapabilities.lancedb(),
        collection_info=None,
    )
    return LanceDBCollectionHandle(context)


class TestHandleAbstractness:
    def test_base_is_abstract_and_lancedb_implements_it(self) -> None:
        assert issubclass(LanceDBCollectionHandle, KBCollectionHandle)
        with pytest.raises(TypeError):
            KBCollectionHandle()  # type: ignore[abstract]


class TestHandleRegisterDocument:
    def test_register_new_document_matches_oracle(self, tmp_path: Path) -> None:
        src = tmp_path / "report.txt"
        src.write_text("hello world")
        handle = make_handle("coll")

        response = handle.register_document(
            RegisterDocumentRequest(
                collection="coll",
                source_path=str(src),
                doc_id="doc-1",
                user_id=7,
                file_id="file-9",
            )
        )

        assert response.doc_id == "doc-1"
        assert response.created is True
        assert len(response.content_hash) == 64

        store = get_vector_index_store()
        assert (
            store.count_rows(
                "documents", {"collection": "coll", "doc_id": "doc-1"}, is_admin=True
            )
            == 1
        )

    def test_deterministic_doc_id_and_idempotency(self, tmp_path: Path) -> None:
        src = tmp_path / "report.docx"
        src.write_text("content")
        handle = make_handle("my_kb")

        first = handle.register_document(
            RegisterDocumentRequest(collection="my_kb", source_path=str(src))
        )
        second = handle.register_document(
            RegisterDocumentRequest(collection="my_kb", source_path=str(src))
        )

        # Deterministic doc_id derived from (collection, source_path).
        assert first.doc_id == generate_deterministic_doc_id("my_kb", str(src))
        assert first.doc_id == second.doc_id
        assert first.created is True
        assert second.created is False  # idempotent re-register is an update
        assert first.content_hash == second.content_hash

    def test_auto_file_type_detection(self, tmp_path: Path) -> None:
        src = tmp_path / "notes.md"
        src.write_text("# Title")
        handle = make_handle("coll")

        response = handle.register_document(
            RegisterDocumentRequest(
                collection="coll", source_path=str(src), doc_id="md-doc"
            )
        )

        store = get_vector_index_store()
        for batch in store.iter_batches(
            table_name="documents",
            filters={"collection": "coll", "doc_id": "md-doc"},
            is_admin=True,
        ):
            row = batch.to_pandas().iloc[0]
            assert row["file_type"] == "md"
            assert row["content_hash"] == response.content_hash
            break

    def test_missing_source_path_raises(self, tmp_path: Path) -> None:
        from xagent.core.tools.core.RAG_tools.core.exceptions import (
            DocumentValidationError,
        )

        handle = make_handle("coll")
        with pytest.raises(DocumentValidationError, match="Source path does not exist"):
            handle.register_document(
                RegisterDocumentRequest(
                    collection="coll", source_path="/no/such/file.txt"
                )
            )


def _register(handle: LanceDBCollectionHandle, src: Path, doc_id: str, **kwargs):
    src.write_text(kwargs.pop("content", f"content of {doc_id}"))
    return handle.register_document(
        RegisterDocumentRequest(
            collection=handle.context.collection,
            source_path=str(src),
            doc_id=doc_id,
            **kwargs,
        )
    )


class TestHandleLoadDocument:
    def test_admin_scope_returns_detail(self, tmp_path: Path) -> None:
        handle = make_handle("coll")
        _register(handle, tmp_path / "a.txt", "doc-1", user_id=7, file_id="f9")

        detail = handle.load_document("doc-1", user_id=None, is_admin=True)

        assert detail is not None
        assert detail.collection == "coll"
        assert detail.doc_id == "doc-1"
        assert detail.file_type == "txt"
        assert detail.user_id == 7
        assert detail.content_hash is not None
        assert len(detail.content_hash) == 64
        # Lossless legacy shape: full 10-column row dict.
        assert set(detail.to_legacy_dict().keys()) == {
            "collection",
            "doc_id",
            "file_id",
            "source_path",
            "file_type",
            "content_hash",
            "uploaded_at",
            "title",
            "language",
            "user_id",
        }

    def test_anonymous_scope_returns_none(self, tmp_path: Path) -> None:
        handle = make_handle("coll")
        _register(handle, tmp_path / "a.txt", "doc-1")  # user_id=None

        # Default (anonymous) scope sees no rows -> None (matches legacy
        # get_document behavior locked by the characterization oracle).
        assert handle.load_document("doc-1") is None

    def test_missing_returns_none(self, tmp_path: Path) -> None:
        handle = make_handle("coll")
        _register(handle, tmp_path / "a.txt", "exists")
        assert handle.load_document("nope", is_admin=True) is None


class TestHandleListDocuments:
    def test_returns_result_and_maps_to_legacy(self, tmp_path: Path) -> None:
        handle = make_handle("coll_a")
        _register(handle, tmp_path / "a1.txt", "a1")
        _register(handle, tmp_path / "a2.txt", "a2")
        # A doc in a different collection must not leak in.
        other = make_handle("coll_b")
        _register(other, tmp_path / "b1.txt", "b1")

        result = handle.list_documents(is_admin=True, limit=100)

        assert result.total_count == 2
        legacy = result.to_legacy_dicts()
        assert {row["doc_id"] for row in legacy} == {"a1", "a2"}
        for row in legacy:
            assert row["collection"] == "coll_a"
            assert set(row.keys()) == {
                "collection",
                "doc_id",
                "file_id",
                "source_path",
                "file_type",
                "content_hash",
                "uploaded_at",
                "title",
                "language",
                "user_id",
            }

    def test_limit_is_honored(self, tmp_path: Path) -> None:
        handle = make_handle("coll")
        _register(handle, tmp_path / "a1.txt", "a1")
        _register(handle, tmp_path / "a2.txt", "a2")

        result = handle.list_documents(is_admin=True, limit=1)
        assert result.total_count == 1
        assert len(result.documents) == 1

    def test_empty_collection_returns_empty_result(self, tmp_path: Path) -> None:
        handle = make_handle("coll")
        result = handle.list_documents(is_admin=True, limit=10)
        assert result.total_count == 0
        assert result.documents == []
        assert result.to_legacy_dicts() == []
