from __future__ import annotations

import json
import logging
from typing import Any, List, Optional, Union
from uuid import uuid4

from ...providers.vector_store.lancedb import (
    LanceDBConnectionManager,
    LanceDBVectorStore,
)
from ..model.embedding import BaseEmbedding, DashScopeEmbedding
from ..model.embedding.adapter import create_embedding_adapter
from ..model.model import EmbeddingModelConfig
from ..tools.core.RAG_tools.LanceDB.schema_manager import _safe_close_table
from .base import MemoryStore
from .core import MemoryNote, MemoryResponse

logger = logging.getLogger(__name__)


class LanceDBMemoryStore(MemoryStore):
    """LanceDB-based memory store implementation with vector search capabilities."""

    _embedding_model: Optional[BaseEmbedding]

    def __init__(
        self,
        db_dir: str,
        collection_name: str = "memories",
        embedding_model: Optional[Union[BaseEmbedding, EmbeddingModelConfig]] = None,
        similarity_threshold: float = 1.0,
        **embedding_kwargs: Any,
    ):
        """
        Initialize LanceDB memory store.

        Args:
            db_dir: Database directory path
            collection_name: Collection name for storing memories
            embedding_model: Optional BaseEmbedding instance or EmbeddingModel config
            similarity_threshold: Cosine distance threshold for vector search (lower = more strict)
            **embedding_kwargs: Additional arguments for embedding model
        """
        self._collection_name = collection_name

        # Handle different types of embedding_model input
        if embedding_model is None:
            # Try to create a default embedding model only if embedding_kwargs are provided
            if embedding_kwargs:
                try:
                    self._embedding_model = DashScopeEmbedding(**embedding_kwargs)
                except Exception:
                    # If embedding model creation fails, set to None (will use fallback)
                    self._embedding_model = None
                    logger.warning(
                        "Failed to create embedding model, will use fallback text search"
                    )
            else:
                self._embedding_model = None
                logger.info(
                    "No embedding model provided, will use fallback text search"
                )
        elif isinstance(embedding_model, BaseEmbedding):
            self._embedding_model = embedding_model
        elif isinstance(embedding_model, EmbeddingModelConfig):
            self._embedding_model = create_embedding_adapter(embedding_model)
        else:
            raise ValueError(
                f"Unsupported embedding model type: {type(embedding_model)}"
            )
        self._similarity_threshold = similarity_threshold
        self._vector_store = LanceDBVectorStore(db_dir, collection_name)
        self._conn_manager = LanceDBConnectionManager()
        self._ensure_table_schema()

    def _ensure_table_schema(self) -> None:
        """Ensure the table has the correct schema for memory storage."""
        table = None
        try:
            conn = self._vector_store.get_raw_connection()
            table = conn.open_table(self._collection_name)

            # Check if table exists and has basic structure
            df = table.search().limit(1).to_pandas()

            # Table exists, check if it has required columns
            if not all(col in df.columns for col in ["id", "text", "metadata"]):
                # Schema is incompatible, drop and recreate
                logger.warning(
                    f"Table {self._collection_name} has incompatible schema, recreating"
                )
                inner_table = None
                try:
                    # Try to drop the table if the method exists
                    if hasattr(conn, "drop_table"):
                        conn.drop_table(self._collection_name)
                    else:
                        # Alternative: try to use delete all records instead
                        inner_table = conn.open_table(self._collection_name)
                        inner_table.delete()
                except Exception:
                    # If drop fails, continue with recreation
                    pass
                finally:
                    _safe_close_table(inner_table)
                self._create_empty_table()

        except Exception:
            # Table doesn't exist, create it with basic schema
            logger.info(f"Creating table {self._collection_name} with basic schema")
            self._create_empty_table()
        finally:
            _safe_close_table(table)

    def _create_empty_table(self) -> None:
        """Create an empty table with the correct schema."""
        conn = self._vector_store.get_raw_connection()

        # Check if we have an embedding model
        if self._embedding_model:
            # Create table with vector support
            try:
                # Generate a sample embedding to get dimension
                sample_embedding = self._get_embedding("sample")
                if sample_embedding:
                    # Create sample data with vector
                    sample_data = [
                        {
                            "id": "sample",
                            "text": "sample",
                            "metadata": "{}",
                            "vector": sample_embedding,
                        }
                    ]
                else:
                    # Fallback to non-vector schema
                    sample_data = [{"id": "sample", "text": "sample", "metadata": "{}"}]
            except Exception:
                # If embedding fails, use non-vector schema
                sample_data = [{"id": "sample", "text": "sample", "metadata": "{}"}]
        else:
            # No embedding model, create without vector column
            sample_data = [{"id": "sample", "text": "sample", "metadata": "{}"}]

        # Create table with appropriate schema
        table = conn.create_table(self._collection_name, data=sample_data)
        # Remove sample data
        table.delete("id = 'sample'")

    def _get_embedding(self, text: str) -> Optional[list[float]]:
        """Get embedding for text using the configured embedding model."""
        if not self._embedding_model or not text.strip():
            return None

        try:
            result = self._embedding_model.encode(text)
            # encode should return list[float] for single text input
            if isinstance(result, list):
                if len(result) > 0 and isinstance(result[0], list):
                    # Got list[list[float]], return the first embedding
                    return result[0]
                elif len(result) > 0 and isinstance(result[0], (int, float)):
                    # Got list[float], return as is
                    return result  # type: ignore[return-value]
            logger.warning(f"Unexpected embedding result format: {type(result)}")
            return None
        except Exception as e:
            logger.error(f"Failed to generate embedding for text '{text[:50]}...': {e}")
            return None

    def _memory_note_to_dict(self, note: MemoryNote) -> dict[str, Any]:
        """Convert MemoryNote to dictionary for storage."""
        # Get embedding for the content
        content_text = (
            note.content.decode() if isinstance(note.content, bytes) else note.content
        )
        embedding = self._get_embedding(content_text)

        # Prepare metadata
        metadata = {
            "content": note.content,
            "keywords": note.keywords,
            "tags": note.tags,
            "category": note.category,
            "timestamp": note.timestamp.isoformat(),
            "mime_type": note.mime_type,
            **note.metadata,
        }

        return {
            "id": note.id,
            "vector": embedding,
            "text": note.content,
            "metadata": json.dumps(metadata, ensure_ascii=False),
        }

    def _dict_to_memory_note(self, data: dict[str, Any]) -> MemoryNote:
        """Convert dictionary from storage to MemoryNote."""
        try:
            metadata = json.loads(data.get("metadata", "{}"))
        except (json.JSONDecodeError, TypeError):
            metadata = {}

        return MemoryNote(
            id=data.get("id"),
            content=metadata.pop("content", data.get("text", "")),
            keywords=metadata.pop("keywords", []),
            tags=metadata.pop("tags", []),
            category=metadata.pop("category", "general"),
            timestamp=metadata.pop("timestamp", None),
            mime_type=metadata.pop("mime_type", "text/plain"),
            metadata=metadata,
        )

    def _apply_filters(self, note: MemoryNote, filters: dict[str, Any]) -> bool:
        """Apply filters to a MemoryNote for vector search results."""
        for key, value in filters.items():
            # Special handling for category - check note.category first
            if key == "category":
                if str(note.category) != str(value):
                    return False
            elif key == "metadata":
                # Handle nested metadata filters
                if not self._apply_metadata_filters(note.metadata, value):
                    return False
            else:
                # For other fields, check metadata
                if str(note.metadata.get(key, "")) != str(value):
                    return False
        return True

    def _apply_text_search_filters(
        self, metadata_dict: dict[str, Any], filters: dict[str, Any]
    ) -> bool:
        """Apply filters to metadata dict for text search results."""
        for key, value in filters.items():
            if key == "metadata":
                # Handle nested metadata filters
                if not self._apply_metadata_filters(metadata_dict, value):
                    return False
            else:
                # Direct field comparison
                if str(metadata_dict.get(key, "")) != str(value):
                    return False
        return True

    def _apply_metadata_filters(
        self, metadata: dict[str, Any], metadata_filters: dict[str, Any]
    ) -> bool:
        """Apply nested metadata filters."""
        for key, value in metadata_filters.items():
            if str(metadata.get(key, "")) != str(value):
                return False
        return True

    def add(self, note: MemoryNote) -> MemoryResponse:
        """Add a memory note to the store."""
        try:
            # Generate ID if not provided
            if not note.id:
                note.id = str(uuid4())

            # Convert to storage format
            data = self._memory_note_to_dict(note)

            # Add to vector store - use a consistent approach
            conn = self._vector_store.get_raw_connection()
            table = None
            try:
                table = conn.open_table(self._collection_name)

                # Prepare record for insertion
                record = {
                    "id": data["id"],
                    "text": data["text"],
                    "metadata": data["metadata"],
                }

                # Add vector if available
                if data["vector"]:
                    record["vector"] = data["vector"]

                # Try to add the record, recreate table if schema mismatch
                try:
                    table.add([record])
                except Exception as add_error:
                    logger.warning(
                        f"Failed to add record due to schema mismatch: {add_error}"
                    )
                    # Recreate table and try again
                    logger.info("Recreating table with fresh schema")
                    inner_table = None
                    try:
                        # Try to drop the table if the method exists
                        if hasattr(conn, "drop_table"):
                            conn.drop_table(self._collection_name)
                        else:
                            # Alternative: try to use delete all records instead
                            inner_table = conn.open_table(self._collection_name)
                            inner_table.delete()
                    except Exception:
                        # If drop fails, continue with recreation
                        pass
                    finally:
                        _safe_close_table(inner_table)
                    self._create_empty_table()
                    _safe_close_table(table)
                    table = conn.open_table(self._collection_name)

                    # After recreating, check if we should include vector
                    # Get current table schema
                    df = table.search().limit(1).to_pandas()
                    if not df.empty and "vector" not in df.columns:
                        # New table doesn't have vector column, remove vector from record
                        record_without_vector = {
                            k: v for k, v in record.items() if k != "vector"
                        }
                        table.add([record_without_vector])
                    else:
                        table.add([record])
            finally:
                _safe_close_table(table)

            return MemoryResponse(success=True, memory_id=data["id"])

        except Exception as e:
            logger.error(f"Failed to add memory note {note.id}: {e}")
            return MemoryResponse(
                success=False,
                error=f"Failed to add memory: {str(e)}",
                memory_id=note.id,
            )

    def get(self, note_id: str) -> MemoryResponse:
        """Retrieve a memory note by its ID."""
        table = None
        try:
            table = self._vector_store.get_raw_connection().open_table(
                self._collection_name
            )

            # Search by ID
            results = table.search().where(f"id = '{note_id}'").to_pandas()

            if len(results) == 0:
                return MemoryResponse(
                    success=False,
                    error="Memory not found",
                    memory_id=note_id,
                )

            # Convert to MemoryNote
            data = results.iloc[0].to_dict()
            note = self._dict_to_memory_note(data)

            return MemoryResponse(
                success=True,
                memory_id=note_id,
                content=note,
            )

        except Exception as e:
            logger.error(f"Failed to get memory note {note_id}: {e}")
            return MemoryResponse(
                success=False,
                error=f"Failed to get memory: {str(e)}",
                memory_id=note_id,
            )
        finally:
            _safe_close_table(table)

    def update(self, note: MemoryNote) -> MemoryResponse:
        """Update an existing memory note."""
        try:
            # Check if memory exists
            get_response = self.get(note.id)
            if not get_response.success:
                return MemoryResponse(
                    success=False,
                    error="Memory not found",
                    memory_id=note.id,
                )

            # Delete old record
            self.delete(note.id)

            # Add updated record
            return self.add(note)

        except Exception as e:
            logger.error(f"Failed to update memory note {note.id}: {e}")
            return MemoryResponse(
                success=False,
                error=f"Failed to update memory: {str(e)}",
                memory_id=note.id,
            )

    def delete(self, note_id: str) -> MemoryResponse:
        """Delete a memory note by its ID."""
        try:
            success = self._vector_store.delete_vectors([note_id])

            if success:
                return MemoryResponse(success=True, memory_id=note_id)
            else:
                return MemoryResponse(
                    success=False,
                    error="Failed to delete memory",
                    memory_id=note_id,
                )

        except Exception as e:
            logger.error(f"Failed to delete memory note {note_id}: {e}")
            return MemoryResponse(
                success=False,
                error=f"Failed to delete memory: {str(e)}",
                memory_id=note_id,
            )

    def search(
        self,
        query: str,
        k: int = 5,
        filters: Optional[dict[str, Any]] = None,
        similarity_threshold: Optional[float] = None,
    ) -> list[MemoryNote]:
        """Search memory notes by query text with optional filters."""
        table = None
        try:
            table = self._vector_store.get_raw_connection().open_table(
                self._collection_name
            )
            results = []

            # Try vector search first
            try:
                query_embedding = self._get_embedding(query)
                if query_embedding:
                    # Check if vector column exists and has the right dimension
                    sample_df = table.search().limit(1).to_pandas()
                    if not sample_df.empty and "vector" in sample_df.columns:
                        # Try vector search
                        try:
                            vector_df = (
                                table.search(
                                    query_embedding, vector_column_name="vector"
                                )
                                .limit(k)
                                .to_pandas()
                            )

                            for _, row in vector_df.iterrows():
                                # Check similarity threshold
                                threshold = (
                                    similarity_threshold
                                    if similarity_threshold is not None
                                    else self._similarity_threshold
                                )
                                distance = row.get("_distance", float("inf"))
                                if distance > threshold:
                                    logger.info(
                                        f"Skipping result with distance {distance} > threshold {threshold}"
                                    )
                                    continue

                                logger.info(
                                    f"Accepting result with distance {distance} <= threshold {threshold}"
                                )

                                note_data = {
                                    "id": row.get("id", ""),
                                    "text": row.get("text", ""),
                                    "metadata": row.get("metadata", "{}"),
                                }
                                note = self._dict_to_memory_note(note_data)

                                # Apply metadata filters to vector search results too
                                if filters:
                                    filter_match = self._apply_filters(note, filters)
                                    if not filter_match:
                                        continue

                                results.append(note)
                        except Exception as vector_error:
                            logger.warning(
                                f"Vector search failed, falling back to text search: {vector_error}"
                            )
            except Exception as embedding_error:
                logger.warning(
                    f"Embedding generation failed, using text search: {embedding_error}"
                )

            # Fallback to text search if no vector results or vector search failed
            if not results:
                # Text search
                df = table.search().to_pandas()

                # Filter by query text and apply filters
                for _, row in df.iterrows():
                    text = row.get("text", "")
                    metadata = row.get("metadata", "{}")

                    try:
                        metadata_dict = json.loads(metadata)
                    except (json.JSONDecodeError, TypeError):
                        metadata_dict = {}

                    # Apply metadata filters if specified
                    if filters:
                        filter_match = self._apply_text_search_filters(
                            metadata_dict, filters
                        )
                        if not filter_match:
                            continue

                    # Simple text matching
                    if not query or query.lower() in text.lower():
                        note_data = {
                            "id": row.get("id", ""),
                            "text": text,
                            "metadata": metadata,
                        }
                        note = self._dict_to_memory_note(note_data)
                        results.append(note)

                        if len(results) >= k:
                            break

            return results[:k]

        except Exception as e:
            logger.error(f"Failed to search memories with query '{query[:50]}...': {e}")
            return []
        finally:
            _safe_close_table(table)

    def clear(self) -> None:
        """Clear all memory notes from the store."""
        try:
            self._vector_store.clear()
        except Exception as e:
            logger.error(f"Failed to clear memory store: {e}")

    def list_all(self, filters: Optional[dict[str, Any]] = None) -> List[MemoryNote]:
        """List all memory notes with optional filtering."""
        try:
            # Use empty query to get all results
            return self.search(query="", k=10000, filters=filters or {})
        except Exception as e:
            logger.error(f"Failed to list all memories: {e}")
            return []

    def get_stats(self) -> dict[str, Any]:
        """Get statistics about the memory store."""
        try:
            # Get all memories to calculate stats
            all_memories = self.list_all()

            total_count = len(all_memories)
            category_counts: dict[str, int] = {}
            tag_counts: dict[str, int] = {}

            for note in all_memories:
                # Count by category
                category_counts[note.category] = (
                    category_counts.get(note.category, 0) + 1
                )

                # Count tags
                for tag in note.tags:
                    tag_counts[tag] = tag_counts.get(tag, 0) + 1

            return {
                "total_count": total_count,
                "category_counts": category_counts,
                "tag_counts": tag_counts,
                "memory_store_type": "lancedb",
            }
        except Exception as e:
            logger.error(f"Failed to get memory stats: {e}")
            return {
                "total_count": 0,
                "category_counts": {},
                "tag_counts": {},
                "memory_store_type": "lancedb",
                "error": str(e),
            }
