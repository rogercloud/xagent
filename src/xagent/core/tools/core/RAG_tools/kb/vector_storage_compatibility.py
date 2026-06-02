"""Vector storage compatibility facade."""

from __future__ import annotations

import logging
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from ..LanceDB.model_tag_utils import to_model_tag
from ..LanceDB.schema_manager import _safe_close_table
from ..utils.lancedb_query_utils import _safe_count_rows, list_embeddings_table_names
from ..utils.string_utils import (
    build_lancedb_filter_expression,
    build_user_id_filter_for_table,
)

if TYPE_CHECKING:
    from ..core.schemas import (
        ChunkEmbeddingData,
        EmbeddingReadResponse,
        EmbeddingWriteResponse,
    )
    from .coordinator import KBCoordinator
    from .storage_shim import KBStorageShimCompatibilityFacade

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class KBVectorStorageCleanupResult:
    """Outcome for vector-storage rollback cleanup actions."""

    collection: str
    status: str
    deleted_count: int = 0
    table_counts: dict[str, int] = field(default_factory=dict)
    model_tag: Optional[str] = None
    preview_only: bool = True
    warnings: tuple[str, ...] = ()
    side_effects_may_remain: bool = False


class KBVectorStorageCompatibilityFacade:
    """Compatibility boundary for legacy vector-storage helpers.

    Public vector-storage functions keep their historical synchronous shape.
    The facade binds coordinator-owned storage access, then delegates to the
    current vector manager implementation so model-tag routing, dimension
    checks, merge error mapping, and result models remain unchanged.
    """

    def __init__(
        self,
        coordinator: "KBCoordinator | None" = None,
        storage_shim: "KBStorageShimCompatibilityFacade | None" = None,
    ) -> None:
        self._coordinator = coordinator
        self._storage_shim = storage_shim

    def _active_storage_shim(self) -> "KBStorageShimCompatibilityFacade | None":
        if self._storage_shim is not None:
            return self._storage_shim
        if self._coordinator is not None:
            return self._coordinator.storage_shim
        return None

    @contextmanager
    def _storage_context(self) -> Iterator[None]:
        storage_shim = self._active_storage_shim()
        if storage_shim is None:
            yield
            return

        from ..storage.factory import bind_storage_shim_for_current_context

        with bind_storage_shim_for_current_context(storage_shim):
            yield

    def validate_query_vector(
        self,
        query_vector: List[float],
        model_tag: Optional[str] = None,
        conn: Any = None,
        user_id: Optional[int] = None,
        is_admin: bool = False,
    ) -> None:
        from ..vector_storage.vector_manager import _validate_query_vector_impl

        with self._storage_context():
            _validate_query_vector_impl(
                query_vector,
                model_tag=model_tag,
                conn=conn,
                user_id=user_id,
                is_admin=is_admin,
            )

    def read_chunks_for_embedding(
        self,
        collection: str,
        doc_id: str,
        parse_hash: str,
        model: str,
        filters: Optional[Dict[str, Any]] = None,
        user_id: Optional[int] = None,
        is_admin: bool = False,
    ) -> "EmbeddingReadResponse":
        from ..vector_storage.vector_manager import _read_chunks_for_embedding_impl

        with self._storage_context():
            return _read_chunks_for_embedding_impl(
                collection=collection,
                doc_id=doc_id,
                parse_hash=parse_hash,
                model=model,
                filters=filters,
                user_id=user_id,
                is_admin=is_admin,
            )

    def write_vectors_to_db(
        self,
        collection: str,
        embeddings: List["ChunkEmbeddingData"],
        create_index: bool = True,
        user_id: Optional[int] = None,
    ) -> "EmbeddingWriteResponse":
        from ..vector_storage.vector_manager import _write_vectors_to_db_impl

        with self._storage_context():
            return _write_vectors_to_db_impl(
                collection=collection,
                embeddings=embeddings,
                create_index=create_index,
                user_id=user_id,
            )

    def cleanup_vectors_for_document(
        self,
        *,
        collection: str,
        doc_id: str,
        model_tag: Optional[str] = None,
        user_id: Optional[int] = None,
        is_admin: bool = False,
        preview_only: bool = True,
        confirm: bool = False,
    ) -> KBVectorStorageCleanupResult:
        """Delete or preview all vectors for one document."""
        return self.cleanup_vectors_for_operation(
            collection=collection,
            doc_id=doc_id,
            model_tag=model_tag,
            user_id=user_id,
            is_admin=is_admin,
            preview_only=preview_only,
            confirm=confirm,
        )

    def cleanup_vectors_for_chunks(
        self,
        *,
        collection: str,
        doc_id: str,
        chunk_ids: Sequence[str],
        parse_hash: Optional[str] = None,
        model_tag: Optional[str] = None,
        user_id: Optional[int] = None,
        is_admin: bool = False,
        preview_only: bool = True,
        confirm: bool = False,
    ) -> KBVectorStorageCleanupResult:
        """Delete or preview vectors for an explicit chunk set."""
        return self.cleanup_vectors_for_operation(
            collection=collection,
            doc_id=doc_id,
            parse_hash=parse_hash,
            chunk_ids=chunk_ids,
            model_tag=model_tag,
            user_id=user_id,
            is_admin=is_admin,
            preview_only=preview_only,
            confirm=confirm,
        )

    def cleanup_vectors_for_operation(
        self,
        *,
        collection: str,
        doc_id: Optional[str] = None,
        parse_hash: Optional[str] = None,
        chunk_ids: Optional[Sequence[str]] = None,
        model_tag: Optional[str] = None,
        user_id: Optional[int] = None,
        is_admin: bool = False,
        preview_only: bool = True,
        confirm: bool = False,
    ) -> KBVectorStorageCleanupResult:
        """Delete or preview vectors created by a failed compatibility operation.

        Current embeddings rows do not carry an operation ID, so rollback callers
        scope cleanup through the operation's known document, parse, chunk set,
        and model tag. Future handle-level embedding rollback can replace this
        adapter without changing coordinator rollback code.
        """
        normalized_collection = collection.strip() if isinstance(collection, str) else ""
        if not normalized_collection:
            raise ValueError("collection must be a non-empty string")
        normalized_chunk_ids = _normalize_chunk_ids(chunk_ids)
        if not any([doc_id, parse_hash, normalized_chunk_ids]):
            raise ValueError(
                "At least one of doc_id, parse_hash, or chunk_ids is required"
            )

        with self._storage_context():
            return self._cleanup_vectors_for_operation_impl(
                collection=normalized_collection,
                doc_id=doc_id,
                parse_hash=parse_hash,
                chunk_ids=normalized_chunk_ids,
                model_tag=model_tag,
                user_id=user_id,
                is_admin=is_admin,
                preview_only=preview_only,
                confirm=confirm,
            )

    def _cleanup_vectors_for_operation_impl(
        self,
        *,
        collection: str,
        doc_id: Optional[str],
        parse_hash: Optional[str],
        chunk_ids: tuple[str, ...],
        model_tag: Optional[str],
        user_id: Optional[int],
        is_admin: bool,
        preview_only: bool,
        confirm: bool,
    ) -> KBVectorStorageCleanupResult:
        storage_shim = self._active_storage_shim()
        if storage_shim is not None:
            conn = storage_shim.get_vector_store_raw_connection()
        else:
            from ..storage.factory import get_vector_store_raw_connection

            conn = get_vector_store_raw_connection()

        table_counts: dict[str, int] = {}
        warnings: list[str] = []
        side_effects_may_remain = False
        should_delete = bool(confirm and not preview_only)
        target_tables = _target_embedding_tables(conn, model_tag=model_tag)

        if not target_tables:
            return KBVectorStorageCleanupResult(
                collection=collection,
                status="skipped",
                deleted_count=0,
                table_counts={},
                model_tag=model_tag,
                preview_only=preview_only,
            )

        filter_inputs = _build_filter_inputs(
            collection=collection,
            doc_id=doc_id,
            parse_hash=parse_hash,
            chunk_ids=chunk_ids,
        )

        for table_name in target_tables:
            table = None
            try:
                table = conn.open_table(table_name)
                count = 0
                for filter_values in filter_inputs:
                    filter_expr = build_lancedb_filter_expression(
                        filter_values,
                        user_id=user_id,
                        is_admin=is_admin,
                        skip_user_filter=True,
                    )
                    filter_expr = _append_table_user_filter(
                        table=table,
                        filter_expr=filter_expr,
                        user_id=user_id,
                        is_admin=is_admin,
                    )
                    matched = _safe_count_rows(table, filter_expr, on_error="raise")
                    if should_delete and matched > 0:
                        table.delete(filter_expr)
                    count += matched
                table_counts[table_name] = count
            except Exception as exc:  # noqa: BLE001 - report rollback cleanup state
                side_effects_may_remain = True
                message = f"{table_name}: {exc}"
                warnings.append(message)
                logger.warning("Vector cleanup failed for %s: %s", table_name, exc)
            finally:
                _safe_close_table(table)

        deleted_count = sum(table_counts.values())
        if warnings:
            status = "incomplete"
        elif should_delete:
            status = "complete"
        else:
            status = "planned"

        return KBVectorStorageCleanupResult(
            collection=collection,
            status=status,
            deleted_count=deleted_count,
            table_counts=table_counts,
            model_tag=model_tag,
            preview_only=preview_only,
            warnings=tuple(warnings),
            side_effects_may_remain=side_effects_may_remain,
        )


def _normalize_chunk_ids(chunk_ids: Optional[Sequence[str]]) -> tuple[str, ...]:
    if not chunk_ids:
        return ()
    return tuple(sorted({str(chunk_id) for chunk_id in chunk_ids if str(chunk_id)}))


def _build_filter_inputs(
    *,
    collection: str,
    doc_id: Optional[str],
    parse_hash: Optional[str],
    chunk_ids: tuple[str, ...],
) -> list[dict[str, Any]]:
    base: dict[str, Any] = {"collection": collection}
    if doc_id is not None:
        base["doc_id"] = doc_id
    if parse_hash is not None:
        base["parse_hash"] = parse_hash
    if not chunk_ids:
        return [base]
    return [dict(base, chunk_id=chunk_id) for chunk_id in chunk_ids]


def _target_embedding_tables(conn: Any, model_tag: Optional[str]) -> list[str]:
    table_names = list_embeddings_table_names(conn)
    if model_tag is None:
        return table_names

    candidate_tags = {model_tag, to_model_tag(model_tag)}
    candidate_tables = {f"embeddings_{tag}" for tag in candidate_tags}
    return [table_name for table_name in table_names if table_name in candidate_tables]


def _append_table_user_filter(
    *,
    table: Any,
    filter_expr: str,
    user_id: Optional[int],
    is_admin: bool,
) -> str:
    if is_admin or user_id is None:
        return filter_expr
    if not _table_has_column(table, "user_id"):
        return filter_expr
    user_filter = build_user_id_filter_for_table(table, int(user_id))
    if not filter_expr:
        return user_filter
    return f"{filter_expr} AND {user_filter}"


def _table_has_column(table: Any, column_name: str) -> bool:
    try:
        schema = getattr(table, "schema", None)
        if schema is None:
            return False
        names = getattr(schema, "names", None)
        if names is not None:
            return str(column_name) in {str(name) for name in names}
        field = schema.field(column_name)
        return field is not None
    except Exception:
        return False
