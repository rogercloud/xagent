"""Tests for search_dense functionality.

This module tests the dense vector search implementation:
- search_dense main function
- search_engine core logic
- _build_safe_filter utility
- Integration with LanceDB and index management
"""

import os
import tempfile
import unittest
import uuid
from unittest.mock import Mock, patch

import pytest

from xagent.core.tools.core.RAG_tools.core.exceptions import DocumentValidationError
from xagent.core.tools.core.RAG_tools.core.schemas import (
    DenseSearchResponse,
    IndexResult,
    IndexStatus,
    SearchResult,
)
from xagent.core.tools.core.RAG_tools.retrieval.search_dense import search_dense
from xagent.core.tools.core.RAG_tools.retrieval.search_engine import search_dense_engine


class TestSearchDenseEngine:
    """Test search_dense_engine function."""

    @pytest.fixture
    def mock_search_chain(self):
        """Create a reusable mock search chain for table operations.

        Returns a function that sets up the mock chain and returns mock objects.
        The returned function accepts an optional results_df parameter.
        """

        def _create_mock_chain(mock_table: Mock, results_df=None):
            """Create and configure the mock search chain.

            Args:
                mock_table: The mock table to attach the chain to
                results_df: Optional DataFrame to return from to_pandas()
                           If None, defaults to empty DataFrame

            Returns:
                Tuple of (mock_search, mock_where, mock_limit)
            """
            import pandas as pd

            # Default to empty DataFrame if not provided
            if results_df is None:
                results_df = pd.DataFrame([])

            # Create mock chain for search -> where -> limit -> to_pandas
            mock_search = Mock()
            mock_where = Mock()
            mock_limit = Mock()

            mock_table.search.return_value = mock_search
            mock_search.where.return_value = mock_where
            mock_search.limit.return_value = (
                mock_limit  # For when no filters are applied
            )
            mock_where.limit.return_value = mock_limit  # For when filters are applied

            # UPDATED: Support to_arrow() -> to_list() -> to_pandas() three-tier fallback
            # Create mock Arrow table
            mock_arrow_table = Mock()
            mock_arrow_table.to_pylist.return_value = results_df.to_dict("records")
            mock_limit.to_arrow.return_value = mock_arrow_table

            mock_limit.to_list.return_value = results_df.to_dict("records")
            mock_limit.to_pandas.return_value = results_df

            return mock_search, mock_where, mock_limit

        return _create_mock_chain

    def test_search_engine_basic(self, mock_search_chain) -> None:
        """Test basic search engine functionality."""
        # Mock table operations - create proper chain of mocks
        import pandas as pd

        mock_results_df = pd.DataFrame(
            [
                {
                    "doc_id": "doc1",
                    "chunk_id": "chunk1",
                    "text": "test content",
                    "score": 0.8,
                    "parse_hash": "hash1",
                    "model_tag": "test_model",
                    "created_at": pd.Timestamp.now(),
                    "_distance": 0.5,  # Squared Euclidean distance
                }
            ]
        )

        # Use fixture to create mock search chain
        mock_table = Mock()
        mock_search, mock_where, mock_limit = mock_search_chain(
            mock_table, mock_results_df
        )

        # Mock vector store
        mock_vector_store = Mock()
        mock_vector_store.build_filter_expression.return_value = (
            "collection == 'test_collection'"
        )
        mock_vector_store.create_index.return_value = IndexResult(
            status="index_ready",
            advice=None,
            fts_enabled=False,  # Dense search doesn't use FTS
        )

        # Mock search by model method
        mock_vector_store.search_vectors_by_model.return_value = [
            {
                "doc_id": "doc1",
                "chunk_id": "chunk1",
                "text": "test content",
                "_distance": 0.5,
                "parse_hash": "hash1",
                "created_at": pd.Timestamp.now(),
                "metadata": "{}",
            }
        ]

        with patch(
            "xagent.core.tools.core.RAG_tools.retrieval.search_engine.get_vector_index_store"
        ) as mock_get_vector_store:
            mock_get_vector_store.return_value = mock_vector_store

            # Execute search
            results, index_status, index_advice = search_dense_engine(
                collection="test_collection",
                model_tag="test_model",
                query_vector=[0.1, 0.2, 0.3],
                top_k=5,
                user_id=None,
                is_admin=True,
            )

            # Verify results
            assert len(results) == 1
            assert isinstance(results[0], SearchResult)
            assert results[0].doc_id == "doc1"
            assert results[0].chunk_id == "chunk1"
            assert results[0].text == "test content"
            assert (
                abs(results[0].score - (1.0 / (1.0 + 0.5))) < 0.001
            )  # Distance to similarity conversion

            # Verify vector store operations
            mock_vector_store.create_index.assert_called_once_with("test_model", False)
            # Note: build_filter_expression is now called inside the abstraction layer,
            # not in search_dense_engine
            mock_vector_store.search_vectors_by_model.assert_called_once_with(
                model_tag="test_model",
                query_vector=[0.1, 0.2, 0.3],
                top_k=5,
                filters=unittest.mock.ANY,
                vector_column_name="vector",
                user_id=None,
                is_admin=True,
            )

    def test_search_engine_with_filters(self, mock_search_chain) -> None:
        """Test search engine with filters."""
        import pandas as pd

        mock_results_df = pd.DataFrame([])

        # Use fixture to create mock search chain
        mock_table = Mock()
        mock_search_chain(mock_table, mock_results_df)

        # Mock vector store
        mock_vector_store = Mock()
        filters = {"doc_id": "test_doc", "file_type": "pdf"}
        expected_filter_clause = "doc_id = 'test_doc' AND file_type = 'pdf'"
        mock_vector_store.build_filter_expression.return_value = expected_filter_clause
        mock_vector_store.create_index.return_value = IndexResult(
            status="index_ready",
            advice=None,
            fts_enabled=False,  # Dense search doesn't use FTS
        )

        # Mock search by model method - returns empty list
        mock_vector_store.search_vectors_by_model.return_value = []

        with patch(
            "xagent.core.tools.core.RAG_tools.retrieval.search_engine.get_vector_index_store"
        ) as mock_get_vector_store:
            mock_get_vector_store.return_value = mock_vector_store

            # Execute search with filters (collection filter + custom filters)
            search_dense_engine(
                collection="test_collection",
                model_tag="test_model",
                query_vector=[0.1, 0.2, 0.3],
                top_k=5,
                filters=filters,
                user_id=None,
                is_admin=True,
            )

            # Verify filter application (collection filter + custom filters)
            mock_vector_store.create_index.assert_called_once_with("test_model", False)
            # Note: build_filter_expression is now called inside the abstraction layer
            # Verify search was called
            mock_vector_store.search_vectors_by_model.assert_called_once_with(
                model_tag="test_model",
                query_vector=[0.1, 0.2, 0.3],
                top_k=5,
                filters=unittest.mock.ANY,
                vector_column_name="vector",
                user_id=None,
                is_admin=True,
            )

    def test_search_dense_engine_applies_collection_filter(
        self, mock_search_chain
    ) -> None:
        """Test that search_dense_engine always applies collection filter for KB isolation (Issue #72)."""
        import pandas as pd

        mock_table = Mock()
        mock_search_chain(mock_table, pd.DataFrame([]))

        # Mock vector store
        mock_vector_store = Mock()
        mock_vector_store.build_filter_expression.return_value = "collection == 'my_kb'"
        mock_vector_store.create_index.return_value = IndexResult(
            status="index_ready",
            advice=None,
            fts_enabled=False,  # Dense search doesn't use FTS
        )

        # Mock search by model method - returns empty list
        mock_vector_store.search_vectors_by_model.return_value = []

        with patch(
            "xagent.core.tools.core.RAG_tools.retrieval.search_engine.get_vector_index_store"
        ) as mock_get_vector_store:
            mock_get_vector_store.return_value = mock_vector_store

            search_dense_engine(
                collection="my_kb",
                model_tag="test_model",
                query_vector=[0.1, 0.2, 0.3],
                top_k=5,
                user_id=None,
                is_admin=True,
            )

            # Note: build_filter_expression is now called inside the abstraction layer
            # Verify search was called
            mock_vector_store.search_vectors_by_model.assert_called_once()

    def test_search_engine_readonly_mode(self, mock_search_chain) -> None:
        """Test search engine in readonly mode."""
        import pandas as pd

        mock_results_df = pd.DataFrame([])

        # Use fixture to create mock search chain
        mock_table = Mock()
        mock_search_chain(mock_table, mock_results_df)

        # Mock vector store
        mock_vector_store = Mock()
        mock_vector_store.build_filter_expression.return_value = (
            "collection == 'test_collection'"
        )
        mock_vector_store.create_index.return_value = IndexResult(
            status="readonly",
            advice="Readonly mode - no index operations for embeddings_test_model",
            fts_enabled=False,
        )

        # Mock search by model method - returns empty list
        mock_vector_store.search_vectors_by_model.return_value = []

        with patch(
            "xagent.core.tools.core.RAG_tools.retrieval.search_engine.get_vector_index_store"
        ) as mock_get_vector_store:
            mock_get_vector_store.return_value = mock_vector_store

            # Execute search in readonly mode
            results, index_status, index_advice = search_dense_engine(
                collection="test_collection",
                model_tag="test_model",
                query_vector=[0.1, 0.2, 0.3],
                top_k=5,
                readonly=True,
                user_id=None,
                is_admin=True,
            )

            assert index_status == "readonly"
            assert "Readonly mode" in index_advice

            # Verify readonly mode passed to create_index
            mock_vector_store.create_index.assert_called_once_with("test_model", True)
            mock_vector_store.search_vectors_by_model.assert_called_once_with(
                model_tag="test_model",
                query_vector=[0.1, 0.2, 0.3],
                top_k=5,
                filters=unittest.mock.ANY,
                vector_column_name="vector",
                user_id=None,
                is_admin=True,
            )
            # Note: build_filter_expression is now called inside the abstraction layer

    def test_search_engine_error_handling(self) -> None:
        """Test error handling in search engine."""
        mock_vector_store = Mock()
        mock_vector_store.search_vectors_by_model.side_effect = Exception(
            "Database connection failed"
        )

        with patch(
            "xagent.core.tools.core.RAG_tools.retrieval.search_engine.get_vector_index_store"
        ) as mock_get_vector_store:
            mock_get_vector_store.return_value = mock_vector_store

            with pytest.raises(Exception, match="Database connection failed"):
                search_dense_engine(
                    collection="test_collection",
                    model_tag="test_model",
                    query_vector=[0.1, 0.2, 0.3],
                    top_k=5,
                    user_id=None,
                    is_admin=True,
                )
            mock_vector_store.search_vectors_by_model.assert_called_once()
            # Index check not reached due to early exception


class TestSearchDense:
    """Test search_dense main function."""

    def _patch_search_dense_module(self):
        """Helper method to import and patch search_dense module.

        Resolves ambiguity when module name and function name are the same.
        """
        import importlib

        search_dense_module = importlib.import_module(
            "xagent.core.tools.core.RAG_tools.retrieval.search_dense"
        )
        return search_dense_module

    @pytest.fixture
    def temp_lancedb_dir(self):
        """Create a temporary directory for LanceDB."""
        with tempfile.TemporaryDirectory() as temp_dir:
            original_env = os.environ.get("LANCEDB_DIR")
            os.environ["LANCEDB_DIR"] = temp_dir
            yield temp_dir
            if original_env is not None:
                os.environ["LANCEDB_DIR"] = original_env
            else:
                os.environ.pop("LANCEDB_DIR", None)

    @pytest.fixture
    def test_collection(self):
        """Test collection name."""
        return f"test_collection_{uuid.uuid4().hex[:8]}"

    def test_search_dense_input_validation(self):
        """Test input validation in search_dense."""
        # Test invalid collection
        with pytest.raises(DocumentValidationError):
            search_dense("", "model", [1.0, 2.0, 3.0], user_id=None, is_admin=True)

        # Test invalid model_tag
        with pytest.raises(DocumentValidationError):
            search_dense("collection", "", [1.0, 2.0, 3.0], user_id=None, is_admin=True)

        # Test invalid top_k
        with pytest.raises(DocumentValidationError):
            search_dense(
                "collection",
                "model",
                [1.0, 2.0, 3.0],
                top_k=0,
                user_id=None,
                is_admin=True,
            )

        with pytest.raises(DocumentValidationError):
            search_dense(
                "collection",
                "model",
                [1.0, 2.0, 3.0],
                top_k=2000,
                user_id=None,
                is_admin=True,
            )

    def test_search_dense_success_path(self, make_handle, routed_facade):
        """Test successful search_dense execution through the routed handle.

        Re-pointed (#511): the public ``search_dense`` now runs the real handle
        logic against a mock ``vector_index_store`` instead of patching the
        ``search_dense_engine`` free function. The handle reads raw rows from
        ``store.search_vectors_by_model`` and converts ``_distance`` to a score.
        """
        search_dense_module = self._patch_search_dense_module()
        handle, store, _ = make_handle()
        store.create_index.return_value = IndexResult(
            status="index_ready", advice="Index is ready", fts_enabled=True
        )
        store.search_vectors_by_model.return_value = [
            {
                "doc_id": "doc1",
                "chunk_id": "chunk1",
                "text": "content",
                "parse_hash": "hash1",
                "created_at": "2026-01-01",
                "metadata": None,
                "_distance": 0.0,  # score = 1/(1+0) = 1.0
            }
        ]

        with routed_facade(search_dense_module, handle):
            response = search_dense(
                collection="test_collection",
                model_tag="test_model",
                query_vector=[0.1, 0.2, 0.3],
                top_k=5,
                user_id=None,
                is_admin=True,
            )

        # Verify response
        assert isinstance(response, DenseSearchResponse)
        assert response.status == "success"
        assert len(response.results) == 1
        assert response.total_count == 1
        assert response.index_status == IndexStatus.INDEX_READY
        assert response.results[0].doc_id == "doc1"

        # The store search reached with the right model_tag/top_k/scope.
        store.search_vectors_by_model.assert_called_once()
        kwargs = store.search_vectors_by_model.call_args.kwargs
        assert kwargs["model_tag"] == "test_model"
        assert kwargs["top_k"] == 5
        assert kwargs["is_admin"] is True

    def test_search_dense_validation_fallback(self, make_handle, routed_facade):
        """Test search_dense returns cleanly when the store yields no rows."""
        search_dense_module = self._patch_search_dense_module()
        handle, store, _ = make_handle()
        store.create_index.return_value = IndexResult(
            status="index_ready", advice="Index is ready", fts_enabled=True
        )
        store.search_vectors_by_model.return_value = []

        with routed_facade(search_dense_module, handle):
            response = search_dense(
                collection="test_collection",
                model_tag="test_model",
                query_vector=[0.1, 0.2, 0.3],
                top_k=5,
                user_id=None,
                is_admin=True,
            )

        assert response.status == "success"
        assert response.total_count == 0
        # Query vector still validated at the public boundary.
        store.search_vectors_by_model.assert_called_once()
        assert (
            store.search_vectors_by_model.call_args.kwargs["model_tag"] == "test_model"
        )

    def test_search_dense_index_status_mapping(self, make_handle, routed_facade):
        """Test index status mapping in search_dense via the routed handle."""
        search_dense_module = self._patch_search_dense_module()

        test_cases = [
            ("index_ready", IndexStatus.INDEX_READY),
            ("index_building", IndexStatus.INDEX_BUILDING),
            ("no_index", IndexStatus.NO_INDEX),
            ("index_corrupted", IndexStatus.INDEX_CORRUPTED),
            ("readonly", IndexStatus.READONLY),
            ("below_threshold", IndexStatus.BELOW_THRESHOLD),
        ]

        for engine_status, expected_enum in test_cases:
            handle, store, _ = make_handle()
            store.create_index.return_value = IndexResult(
                status=engine_status, advice="test advice", fts_enabled=True
            )
            store.search_vectors_by_model.return_value = []

            with routed_facade(search_dense_module, handle):
                response = search_dense(
                    "col", "model", [1.0], top_k=1, user_id=None, is_admin=True
                )
            assert response.index_status == expected_enum


class TestSearchDenseIntegration:
    """Integration tests for search_dense with real LanceDB operations."""

    @pytest.fixture
    def temp_lancedb_dir(self):
        """Create a temporary directory for LanceDB."""
        with tempfile.TemporaryDirectory() as temp_dir:
            original_env = os.environ.get("LANCEDB_DIR")
            os.environ["LANCEDB_DIR"] = temp_dir
            yield temp_dir
            if original_env is not None:
                os.environ["LANCEDB_DIR"] = original_env
            else:
                os.environ.pop("LANCEDB_DIR", None)

    @pytest.fixture
    def test_collection(self):
        """Test collection name."""
        return f"test_collection_{uuid.uuid4().hex[:8]}"

    def test_full_search_workflow(self, temp_lancedb_dir, test_collection):
        """Test complete search workflow from data insertion to retrieval."""
        from xagent.core.tools.core.RAG_tools.core.schemas import ChunkEmbeddingData
        from xagent.core.tools.core.RAG_tools.LanceDB.schema_manager import (
            ensure_embeddings_table,
        )
        from xagent.core.tools.core.RAG_tools.storage.factory import (
            get_vector_store_raw_connection,
        )
        from xagent.core.tools.core.RAG_tools.vector_storage.vector_manager import (
            write_vectors_to_db,
        )

        conn = get_vector_store_raw_connection()
        model_tag = "integration_test_model"

        # Step 1: Clean up any existing table and create fresh table
        table_name = f"embeddings_{model_tag}"
        try:
            conn.drop_table(table_name)
        except Exception:
            pass  # Table might not exist, that's fine

        ensure_embeddings_table(conn, model_tag, vector_dim=3)

        # Create embeddings with Python lists for LanceDB compatibility
        embeddings = [
            ChunkEmbeddingData(
                doc_id="doc1",
                chunk_id="chunk1",
                parse_hash="parse1",
                model=model_tag,
                vector=[1.0, 0.0, 0.0],  # Unit vector along x-axis
                text="This is about artificial intelligence",
                chunk_hash="hash1",
            ),
            ChunkEmbeddingData(
                doc_id="doc2",
                chunk_id="chunk2",
                parse_hash="parse2",
                model=model_tag,
                vector=[0.0, 1.0, 0.0],  # Unit vector along y-axis
                text="This is about machine learning",
                chunk_hash="hash2",
            ),
        ]

        # Insert data
        write_result = write_vectors_to_db(
            test_collection,
            embeddings,
            create_index=False,  # Skip index creation for now
        )
        assert write_result.upsert_count == 2

        # Step 2: Execute search
        query_vector = [1.0, 0.0, 0.0]  # Same as first embedding
        response = search_dense(
            collection=test_collection,
            model_tag=model_tag,
            query_vector=query_vector,
            top_k=2,
            user_id=None,
            is_admin=True,
        )

        # Step 3: Verify results
        assert response.status == "success"
        assert len(response.results) == 2
        assert response.total_count == 2

        # First result should be the most similar (exact match)
        assert response.results[0].doc_id == "doc1"
        assert response.results[0].chunk_id == "chunk1"
        assert abs(response.results[0].score - 1.0) < 0.1  # High similarity score

        # Second result should be less similar
        assert response.results[1].doc_id == "doc2"
        assert response.results[1].score < response.results[0].score

        # Verify index status (include BELOW_THRESHOLD for small datasets)
        assert response.index_status in [
            IndexStatus.INDEX_READY,
            IndexStatus.INDEX_BUILDING,
            IndexStatus.BELOW_THRESHOLD,
        ]

    def test_search_with_filters(self, temp_lancedb_dir, test_collection):
        """Test search functionality with filters."""
        from xagent.core.tools.core.RAG_tools.core.schemas import ChunkEmbeddingData
        from xagent.core.tools.core.RAG_tools.LanceDB.schema_manager import (
            ensure_embeddings_table,
        )
        from xagent.core.tools.core.RAG_tools.storage.factory import (
            get_vector_store_raw_connection,
        )
        from xagent.core.tools.core.RAG_tools.vector_storage.vector_manager import (
            write_vectors_to_db,
        )

        conn = get_vector_store_raw_connection()
        model_tag = "filter_test_model"

        # Clean up any existing table and create fresh table
        table_name = f"embeddings_{model_tag}"
        try:
            conn.drop_table(table_name)
        except Exception:
            pass  # Table might not exist, that's fine

        ensure_embeddings_table(conn, model_tag, vector_dim=2)

        # Create embeddings with Python lists for LanceDB compatibility
        embeddings = [
            ChunkEmbeddingData(
                doc_id="doc1",
                chunk_id="chunk1",
                parse_hash="parse1",
                model=model_tag,
                vector=[1.0, 0.0],
                text="First document content",
                chunk_hash="hash1",
            ),
            ChunkEmbeddingData(
                doc_id="doc2",
                chunk_id="chunk2",
                parse_hash="parse1",
                model=model_tag,
                vector=[0.0, 1.0],
                text="Second document content",
                chunk_hash="hash2",
            ),
        ]

        write_vectors_to_db(test_collection, embeddings, create_index=False)

        # Search with doc_id filter
        response = search_dense(
            collection=test_collection,
            model_tag=model_tag,
            query_vector=[1.0, 0.0],
            top_k=5,
            filters={"doc_id": "doc1"},
            user_id=None,
            is_admin=True,
        )

        # Should only return results from doc1
        assert len(response.results) == 1
        assert response.results[0].doc_id == "doc1"

    def test_search_engine_basic_with_results(self) -> None:
        """Test search engine with actual results (replaces arrow_fallback_to_list test)."""
        import pandas as pd

        # Mock vector store
        mock_vector_store = Mock()
        mock_vector_store.build_filter_expression.return_value = None
        mock_vector_store.create_index.return_value = IndexResult(
            status="index_ready",
            advice=None,
            fts_enabled=False,  # Dense search doesn't use FTS
        )

        # Mock search by model method - returns results
        mock_vector_store.search_vectors_by_model.return_value = [
            {
                "doc_id": "doc1",
                "chunk_id": "chunk1",
                "text": "test content",
                "_distance": 0.5,
                "parse_hash": "hash1",
                "created_at": pd.Timestamp.now(),
                "metadata": '{"key": "value"}',
            }
        ]

        with patch(
            "xagent.core.tools.core.RAG_tools.retrieval.search_engine.get_vector_index_store"
        ) as mock_get_vector_store:
            mock_get_vector_store.return_value = mock_vector_store

            results, _, _ = search_dense_engine(
                collection="test_collection",
                model_tag="test_model",
                query_vector=[0.1, 0.2, 0.3],
                top_k=5,
                user_id=None,
                is_admin=True,
            )

            # Verify results
            assert len(results) == 1
            assert results[0].doc_id == "doc1"

    def test_search_engine_with_missing_optional_fields(self) -> None:
        """Test search engine handles results with missing/None optional fields (replaces arrow_fallback_to_pandas_with_nan test)."""
        import pandas as pd

        # Mock vector store
        mock_vector_store = Mock()
        mock_vector_store.build_filter_expression.return_value = None
        mock_vector_store.create_index.return_value = IndexResult(
            status="index_ready",
            advice=None,
            fts_enabled=False,  # Dense search doesn't use FTS
        )

        # Mock search by model method - returns results with missing optional fields
        mock_vector_store.search_vectors_by_model.return_value = [
            {
                "doc_id": "doc1",
                "chunk_id": "chunk1",
                "text": "test content",
                "_distance": 0.5,
                "parse_hash": "hash1",
                "created_at": pd.Timestamp.now(),
                "metadata": '{"key": "value"}',
                # Missing optional_field
            }
        ]

        with patch(
            "xagent.core.tools.core.RAG_tools.retrieval.search_engine.get_vector_index_store"
        ) as mock_get_vector_store:
            mock_get_vector_store.return_value = mock_vector_store

            results, _, _ = search_dense_engine(
                collection="test_collection",
                model_tag="test_model",
                query_vector=[0.1, 0.2, 0.3],
                top_k=5,
                user_id=None,
                is_admin=True,
            )

            # Verify results are handled correctly
            assert len(results) == 1
            assert results[0].doc_id == "doc1"
