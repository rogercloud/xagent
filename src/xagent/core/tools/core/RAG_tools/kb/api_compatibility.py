"""API compatibility facade for KB route-facing operations."""

from __future__ import annotations

import inspect
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Optional

from ..core.schemas import (
    CollectionInfo,
    CollectionOperationResult,
    DocumentListResult,
    DocumentOperationResult,
    IngestionResult,
    ListCollectionsResult,
    SearchConfig,
    SearchPipelineResult,
    WebCrawlConfig,
    WebIngestionResult,
)
from .models import KBStorageBackend
from .pipeline_compatibility import KB_STORAGE_METADATA_KEY

if TYPE_CHECKING:
    from .coordinator import KBCoordinator
    from .storage_shim import KBStorageShimCompatibilityFacade


async def _maybe_await(value: Any) -> Any:
    if inspect.isawaitable(value):
        return await value
    return value


def _has_store_method(metadata_store: object, name: str) -> bool:
    """Return True for real MetadataStore implementations, not loose mocks."""
    return callable(getattr(type(metadata_store), name, None))


class KBApiCompatibilityFacade:
    """Compatibility boundary for KB API route semantics.

    FastAPI request parsing, dependency handling, response wrappers, and HTTP
    error mapping stay in ``web.api.kb``. This facade owns the normalized KB
    operations that routes call after those API-layer concerns are resolved.
    """

    def __init__(
        self,
        coordinator: KBCoordinator | None = None,
        storage_shim: KBStorageShimCompatibilityFacade | None = None,
    ) -> None:
        self._coordinator = coordinator
        self._storage_shim = storage_shim

    def _active_storage_shim(self) -> KBStorageShimCompatibilityFacade | None:
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

    async def save_collection_config(
        self,
        *,
        collection: str,
        config_json: str,
        user_id: int,
        metadata_store: Any | None = None,
    ) -> None:
        """Save tenant-scoped config and ensure owner-neutral backend binding."""
        with self._storage_context():
            store = metadata_store
            if store is None:
                from ..storage.factory import get_metadata_store

                store = get_metadata_store()

            await _maybe_await(
                store.save_collection_config(
                    collection=collection,
                    config_json=config_json,
                    user_id=user_id,
                )
            )
            await self.ensure_collection_backend_binding(
                collection,
                metadata_store=store,
            )

    async def ensure_collection_backend_binding(
        self,
        collection: str,
        *,
        metadata_store: Any | None = None,
    ) -> CollectionInfo | None:
        """Create a collection-level backend binding without changing owners."""
        with self._storage_context():
            store = metadata_store
            if store is None:
                from ..storage.factory import get_metadata_store

                store = get_metadata_store()

            if not _has_store_method(store, "save_collection"):
                return None

            collection_info: CollectionInfo | None = None
            if _has_store_method(store, "get_collection"):
                try:
                    loaded = store.get_collection(collection)
                    loaded = await _maybe_await(loaded)
                except ValueError:
                    collection_info = CollectionInfo(name=collection)
                else:
                    if isinstance(loaded, CollectionInfo):
                        collection_info = loaded
            else:
                collection_info = CollectionInfo(name=collection)

            if collection_info is None:
                return None

            extra_metadata = dict(collection_info.extra_metadata or {})
            if extra_metadata.get(KB_STORAGE_METADATA_KEY) is not None:
                return collection_info

            extra_metadata[KB_STORAGE_METADATA_KEY] = {
                "backend": KBStorageBackend.LANCEDB.value
            }
            updated_collection = collection_info.model_copy(
                update={"extra_metadata": extra_metadata}
            )
            await _maybe_await(store.save_collection(updated_collection))
            return updated_collection

    def get_collection_sync(self, collection_name: str) -> CollectionInfo:
        if self._coordinator is not None:
            return self._coordinator.maintenance_compatibility.get_collection_sync(
                collection_name
            )

        from ..management.collection_manager import get_collection_sync

        with self._storage_context():
            return get_collection_sync(collection_name)

    def delete_collection_metadata_sync(
        self,
        *,
        collection_name: str,
        user_id: Optional[int],
        is_admin: bool = False,
        delete_orphaned_metadata: bool = False,
    ) -> dict[str, int]:
        if self._coordinator is not None:
            return (
                self._coordinator.maintenance_compatibility.delete_collection_metadata_sync(
                    collection_name=collection_name,
                    user_id=user_id,
                    is_admin=is_admin,
                    delete_orphaned_metadata=delete_orphaned_metadata,
                )
            )

        from ..management.collection_manager import delete_collection_metadata_sync

        with self._storage_context():
            return delete_collection_metadata_sync(
                collection_name=collection_name,
                user_id=user_id,
                is_admin=is_admin,
                delete_orphaned_metadata=delete_orphaned_metadata,
            )

    async def list_collections(
        self,
        user_id: Optional[int] = None,
        is_admin: Optional[bool] = None,
        force_realtime: bool = False,
    ) -> ListCollectionsResult:
        if self._coordinator is not None:
            return await self._coordinator.management.list_collections(
                user_id=user_id,
                is_admin=is_admin,
                force_realtime=force_realtime,
            )

        from ..management.collections import list_collections

        with self._storage_context():
            return await list_collections(
                user_id=user_id,
                is_admin=is_admin,
                force_realtime=force_realtime,
            )

    def list_documents(
        self,
        collection: str,
        user_id: Optional[int] = None,
        is_admin: bool = False,
    ) -> DocumentListResult:
        if self._coordinator is not None:
            return self._coordinator.management.list_documents(
                collection=collection,
                user_id=user_id,
                is_admin=is_admin,
            )

        from ..management.collections import list_documents

        with self._storage_context():
            return list_documents(
                collection=collection,
                user_id=user_id,
                is_admin=is_admin,
            )

    def list_document_records(
        self,
        *,
        collection_name: Optional[str],
        user_id: Optional[int],
        is_admin: bool = False,
        max_results: Optional[int] = None,
    ) -> list[Any]:
        from ..storage.factory import get_vector_index_store

        with self._storage_context():
            kwargs: dict[str, Any] = {
                "collection_name": collection_name,
                "user_id": user_id,
                "is_admin": is_admin,
            }
            if max_results is not None:
                kwargs["max_results"] = max_results
            return get_vector_index_store().list_document_records(**kwargs)

    def delete_document(
        self,
        collection: str,
        doc_id: str,
        user_id: int,
        is_admin: bool = False,
    ) -> DocumentOperationResult:
        if self._coordinator is not None:
            return self._coordinator.management.delete_document(
                collection=collection,
                doc_id=doc_id,
                user_id=user_id,
                is_admin=is_admin,
            )

        from ..management.collections import delete_document

        with self._storage_context():
            return delete_document(
                collection,
                doc_id,
                user_id,
                is_admin,
            )

    def delete_collection(
        self,
        collection: str,
        user_id: Optional[int] = None,
        is_admin: bool = False,
    ) -> CollectionOperationResult:
        if self._coordinator is not None:
            return self._coordinator.management.delete_collection(
                collection=collection,
                user_id=user_id,
                is_admin=is_admin,
            )

        from ..management.collections import delete_collection

        with self._storage_context():
            return delete_collection(collection, user_id, is_admin)

    def rename_collection_data(
        self,
        *,
        collection_name: str,
        new_name: str,
        user_id: Optional[int],
        is_admin: bool = False,
    ) -> list[str]:
        from ..storage.factory import get_vector_index_store

        with self._storage_context():
            return get_vector_index_store().rename_collection_data(
                collection_name=collection_name,
                new_name=new_name,
                user_id=user_id,
                is_admin=is_admin,
            )

    async def rename_collection_metadata(
        self,
        *,
        old_name: str,
        new_name: str,
        user_id: Optional[int],
        is_admin: bool = False,
    ) -> None:
        from ..storage.factory import get_metadata_store

        with self._storage_context():
            await get_metadata_store().rename_collection(
                old_name=old_name,
                new_name=new_name,
                user_id=user_id,
                is_admin=is_admin,
            )

    def rename_collection_status(
        self,
        *,
        old_name: str,
        new_name: str,
        user_id: Optional[int],
        is_admin: bool = False,
    ) -> list[str]:
        from ..storage.factory import get_ingestion_status_store

        with self._storage_context():
            return get_ingestion_status_store().rename_collection_status(
                old_name=old_name,
                new_name=new_name,
                user_id=user_id,
                is_admin=is_admin,
            )

    def run_document_ingestion(
        self,
        collection: str,
        source_path: str,
        *,
        ingestion_config: Optional[Any] = None,
        progress_manager: Optional[Any] = None,
        user_id: Optional[int] = None,
        is_admin: Optional[bool] = None,
        file_id: Optional[str] = None,
        metadata_source_path: Optional[str] = None,
        commit_gate: Optional[Any] = None,
    ) -> IngestionResult:
        if self._coordinator is not None:
            return self._coordinator.pipeline_compatibility.run_document_ingestion(
                collection=collection,
                source_path=source_path,
                ingestion_config=ingestion_config,
                progress_manager=progress_manager,
                user_id=user_id,
                is_admin=is_admin,
                file_id=file_id,
                metadata_source_path=metadata_source_path,
                commit_gate=commit_gate,
            )

        from ..pipelines.document_ingestion import run_document_ingestion

        with self._storage_context():
            return run_document_ingestion(
                collection=collection,
                source_path=source_path,
                ingestion_config=ingestion_config,
                progress_manager=progress_manager,
                user_id=user_id,
                is_admin=is_admin,
                file_id=file_id,
                metadata_source_path=metadata_source_path,
                commit_gate=commit_gate,
            )

    def run_document_search(
        self,
        collection: str,
        query_text: str,
        *,
        config: Optional[SearchConfig | Mapping[str, Any]] = None,
        progress_manager: Optional[Any] = None,
        user_id: Optional[int] = None,
        is_admin: Optional[bool] = None,
    ) -> SearchPipelineResult:
        if self._coordinator is not None:
            return self._coordinator.pipeline_compatibility.run_document_search(
                collection=collection,
                query_text=query_text,
                config=config,
                progress_manager=progress_manager,
                user_id=user_id,
                is_admin=is_admin,
            )

        from ..pipelines.document_search import run_document_search

        with self._storage_context():
            return run_document_search(
                collection=collection,
                query_text=query_text,
                config=config,
                progress_manager=progress_manager,
                user_id=user_id,
                is_admin=is_admin,
            )

    async def run_web_ingestion(
        self,
        collection: str,
        crawl_config: WebCrawlConfig,
        *,
        ingestion_config: Optional[Any] = None,
        progress_callback: Optional[Any] = None,
        user_id: Optional[int] = None,
        is_admin: Optional[bool] = None,
        file_handler: Optional[Any] = None,
    ) -> WebIngestionResult:
        if self._coordinator is not None:
            return await self._coordinator.pipeline_compatibility.run_web_ingestion(
                collection=collection,
                crawl_config=crawl_config,
                ingestion_config=ingestion_config,
                progress_callback=progress_callback,
                user_id=user_id,
                is_admin=is_admin,
                file_handler=file_handler,
            )

        from ..pipelines.web_ingestion import run_web_ingestion

        with self._storage_context():
            return await run_web_ingestion(
                collection=collection,
                crawl_config=crawl_config,
                ingestion_config=ingestion_config,
                progress_callback=progress_callback,
                user_id=user_id,
                is_admin=is_admin,
                file_handler=file_handler,
            )

    def reconstruct_parse_result_from_db(
        self,
        collection: str,
        doc_id: str,
        parse_hash: Optional[str] = None,
        *,
        user_id: Optional[int] = None,
        is_admin: bool = False,
    ) -> tuple[list[dict[str, Any]], str | None]:
        if self._coordinator is not None:
            return self._coordinator.parse_display_compatibility.reconstruct_parse_result_from_db(
                collection=collection,
                doc_id=doc_id,
                parse_hash=parse_hash,
                user_id=user_id,
                is_admin=is_admin,
            )

        from ..parse.parse_display import reconstruct_parse_result_from_db

        with self._storage_context():
            return reconstruct_parse_result_from_db(
                collection,
                doc_id,
                parse_hash,
                user_id=user_id,
                is_admin=is_admin,
            )

    def paginate_parse_results(
        self,
        elements: list[dict[str, Any]],
        page: int,
        page_size: int,
    ) -> tuple[list[Any], dict[str, Any]]:
        if self._coordinator is not None:
            return self._coordinator.parse_display_compatibility.paginate_parse_results(
                elements,
                page,
                page_size,
            )

        from ..parse.parse_display import paginate_parse_results

        return paginate_parse_results(elements, page, page_size)
