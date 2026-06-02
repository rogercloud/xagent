"""Shared KB cleanup scope and predicate helpers."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Optional

from ..LanceDB.model_tag_utils import to_model_tag
from ..LanceDB.schema_manager import _safe_close_table
from ..utils.lancedb_query_utils import list_embeddings_table_names
from ..utils.string_utils import (
    build_lancedb_filter_expression,
    build_user_id_filter_for_table,
    escape_lancedb_string,
)
from ..utils.user_permissions import UserPermissions
from ..utils.user_scope import resolve_user_scope


@dataclass(frozen=True)
class KBCleanupScope:
    """Resolved scope for destructive KB cleanup operations."""

    collection: str
    user_id: Optional[int]
    is_admin: bool
    doc_id: Optional[str] = None
    parse_hash: Optional[str] = None
    chunk_ids: tuple[str, ...] = ()
    model_tag: Optional[str] = None


def resolve_cleanup_scope(
    *,
    collection: str,
    doc_id: Optional[str] = None,
    parse_hash: Optional[str] = None,
    chunk_ids: Optional[Sequence[object]] = None,
    model_tag: Optional[str] = None,
    user_id: Optional[int] = None,
    is_admin: Optional[bool] = None,
    require_target: bool = True,
) -> KBCleanupScope:
    """Normalize and validate a cleanup scope with request-context fallback."""
    normalized_collection = collection.strip() if isinstance(collection, str) else ""
    if not normalized_collection:
        raise ValueError("collection must be a non-empty string")

    normalized_doc_id = _normalize_optional_string(doc_id)
    normalized_parse_hash = _normalize_optional_string(parse_hash)
    normalized_chunk_ids = normalize_cleanup_chunk_ids(chunk_ids)
    if normalized_chunk_ids and normalized_doc_id is None:
        raise ValueError("doc_id is required when chunk_ids are provided")
    if require_target and not any(
        [normalized_doc_id, normalized_parse_hash, normalized_chunk_ids]
    ):
        raise ValueError("At least one of doc_id, parse_hash, or chunk_ids is required")

    user_scope = resolve_user_scope(user_id=user_id, is_admin=is_admin)
    return KBCleanupScope(
        collection=normalized_collection,
        doc_id=normalized_doc_id,
        parse_hash=normalized_parse_hash,
        chunk_ids=normalized_chunk_ids,
        model_tag=model_tag,
        user_id=user_scope.user_id,
        is_admin=user_scope.is_admin,
    )


def normalize_cleanup_chunk_ids(
    chunk_ids: Optional[Sequence[object]],
) -> tuple[str, ...]:
    """Return a stable deduplicated chunk-id tuple."""
    if not chunk_ids:
        return ()
    normalized: set[str] = set()
    for chunk_id in chunk_ids:
        if chunk_id is None:
            continue
        value = str(chunk_id)
        if value:
            normalized.add(value)
    return tuple(sorted(normalized))


def select_embedding_tables(conn: Any, model_tag: Optional[str] = None) -> list[str]:
    """List embeddings tables, optionally scoped to one model tag."""
    table_names = list_embeddings_table_names(conn)
    if model_tag is None:
        return table_names

    candidate_tags = {model_tag, to_model_tag(model_tag)}
    candidate_tables = {f"embeddings_{tag}" for tag in candidate_tags}
    return [table_name for table_name in table_names if table_name in candidate_tables]


def build_embedding_cleanup_filters(
    conn: Any,
    scope: KBCleanupScope,
) -> dict[str, list[str]]:
    """Build per-embeddings-table cleanup predicates for a resolved scope."""
    table_filters: dict[str, list[str]] = {}
    for table_name in select_embedding_tables(conn, model_tag=scope.model_tag):
        table = None
        try:
            table = conn.open_table(table_name)
            table_filters[table_name] = _build_filters_for_table(table, scope)
        except Exception:
            table_filters[table_name] = _build_filters_without_schema(scope)
        finally:
            _safe_close_table(table)
    return table_filters


def build_embedding_cleanup_filters_from_base(
    conn: Any,
    *,
    base_filter: str,
    user_id: Optional[int],
    is_admin: bool,
    model_tag: Optional[str] = None,
) -> dict[str, list[str]]:
    """Build per-embeddings-table predicate lists from an existing base filter."""
    table_filters: dict[str, list[str]] = {}
    for table_name in select_embedding_tables(conn, model_tag=model_tag):
        table = None
        try:
            table = conn.open_table(table_name)
            table_filters[table_name] = [
                append_user_filter_for_table(
                    table=table,
                    filter_expr=base_filter,
                    user_id=user_id,
                    is_admin=is_admin,
                )
            ]
        except Exception:
            table_filters[table_name] = [
                append_user_filter_without_schema(
                    filter_expr=base_filter,
                    user_id=user_id,
                    is_admin=is_admin,
                )
            ]
        finally:
            _safe_close_table(table)
    return table_filters


def append_user_filter_for_table(
    *,
    table: Any,
    filter_expr: str,
    user_id: Optional[int],
    is_admin: bool,
) -> str:
    """Append a schema-aware user filter for tenant-aware cleanup."""
    if is_admin:
        return filter_expr
    if user_id is None:
        return _and_filter(filter_expr, UserPermissions.get_no_access_filter())
    if not table_has_column(table, "user_id"):
        return filter_expr
    user_filter = build_user_id_filter_for_table(table, int(user_id))
    return _and_filter(filter_expr, user_filter)


def append_user_filter_without_schema(
    *,
    filter_expr: str,
    user_id: Optional[int],
    is_admin: bool,
) -> str:
    """Append a conservative user filter when table schema is unavailable."""
    if is_admin:
        return filter_expr
    if user_id is None:
        return _and_filter(filter_expr, UserPermissions.get_no_access_filter())
    return _and_filter(filter_expr, f"user_id == {int(user_id)}")


def table_has_column(table: Any, column_name: str) -> bool:
    """Return whether a backend table schema includes a column."""
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


def _build_filters_for_table(table: Any, scope: KBCleanupScope) -> list[str]:
    base_filter = _build_base_filter(scope)
    filter_expr = _append_chunk_id_filter_if_needed(base_filter, scope.chunk_ids)
    return [
        append_user_filter_for_table(
            table=table,
            filter_expr=filter_expr,
            user_id=scope.user_id,
            is_admin=scope.is_admin,
        )
    ]


def _build_filters_without_schema(scope: KBCleanupScope) -> list[str]:
    base_filter = _build_base_filter(scope)
    filter_expr = _append_chunk_id_filter_if_needed(base_filter, scope.chunk_ids)
    return [
        append_user_filter_without_schema(
            filter_expr=filter_expr,
            user_id=scope.user_id,
            is_admin=scope.is_admin,
        )
    ]


def _and_filter(base_filter: str, extra_filter: str) -> str:
    if not base_filter:
        return extra_filter
    return f"{base_filter} AND {extra_filter}"


def _build_base_filter(scope: KBCleanupScope) -> str:
    filter_values: dict[str, Any] = {"collection": scope.collection}
    if scope.doc_id is not None:
        filter_values["doc_id"] = scope.doc_id
    if scope.parse_hash is not None:
        filter_values["parse_hash"] = scope.parse_hash
    return build_lancedb_filter_expression(
        filter_values,
        user_id=scope.user_id,
        is_admin=scope.is_admin,
        skip_user_filter=True,
    )


def _append_chunk_id_filter_if_needed(
    base_filter: str, chunk_ids: tuple[str, ...]
) -> str:
    if not chunk_ids:
        return base_filter
    return _and_filter(base_filter, _chunk_ids_filter(chunk_ids))


def _chunk_ids_filter(chunk_ids: tuple[str, ...]) -> str:
    if len(chunk_ids) == 1:
        return f"chunk_id == '{escape_lancedb_string(chunk_ids[0])}'"
    values = ", ".join(f"'{escape_lancedb_string(chunk_id)}'" for chunk_id in chunk_ids)
    return f"chunk_id IN ({values})"


def _normalize_optional_string(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    normalized = str(value)
    return normalized if normalized else None
