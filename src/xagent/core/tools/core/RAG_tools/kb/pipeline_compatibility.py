"""Pipeline compatibility facade."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Callable, Mapping, Optional

from ..core.schemas import (
    IngestionConfig,
    IngestionResult,
    SearchConfig,
    SearchPipelineResult,
    WebCrawlConfig,
    WebIngestionResult,
)
from .models import KBStorageBackend

if TYPE_CHECKING:
    from ..core.schemas import CollectionInfo
    from .coordinator import KBCoordinator
    from .storage_shim import KBStorageShimCompatibilityFacade

KB_STORAGE_METADATA_KEY = "kb_storage"


class KBPipelineCompatibilityFacade:
    """Compatibility boundary for high-level KB pipeline entry points.

    Pipeline modules keep their historical import paths and response contracts.
    The facade centralizes coordinator-owned storage binding and collection
    backend binding while delegating parser, chunker, embedding, crawler,
    progress, and rerank behavior to the existing pipeline implementations.
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

    def ensure_collection_backend_binding(
        self, collection: str
    ) -> CollectionInfo | None:
        """Ensure direct pipeline-created collections carry a backend binding."""
        storage_shim = self._active_storage_shim()
        if storage_shim is None:
            return None

        from .coordinator import _run_in_separate_loop

        metadata_store = storage_shim.get_metadata_store()
        try:
            collection_info = _run_in_separate_loop(
                metadata_store.get_collection(collection)
            )
        except ValueError:
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
        _run_in_separate_loop(metadata_store.save_collection(updated_collection))
        return updated_collection

    def process_document(
        self,
        collection: str,
        source_path: str,
        *,
        config: Optional[IngestionConfig] = None,
        progress_manager: Optional[Any] = None,
        user_id: Optional[int] = None,
        is_admin: bool = False,
        file_id: Optional[str] = None,
        metadata_source_path: Optional[str] = None,
        commit_gate: Optional[Callable[[], None]] = None,
    ) -> IngestionResult:
        from ..pipelines.document_ingestion import _process_document_impl

        with self._storage_context():
            result = _process_document_impl(
                collection=collection,
                source_path=source_path,
                config=config,
                progress_manager=progress_manager,
                user_id=user_id,
                is_admin=is_admin,
                file_id=file_id,
                metadata_source_path=metadata_source_path,
                commit_gate=commit_gate,
            )
            self.ensure_collection_backend_binding(collection)
            return result

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
        commit_gate: Optional[Callable[[], None]] = None,
    ) -> IngestionResult:
        from ..pipelines.document_ingestion import _run_document_ingestion_impl

        with self._storage_context():
            result = _run_document_ingestion_impl(
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
            self.ensure_collection_backend_binding(collection)
            return result

    def search_documents(
        self,
        collection: str,
        query_text: str,
        *,
        config: Optional[SearchConfig] = None,
        progress_manager: Optional[Any] = None,
        user_id: Optional[int] = None,
        is_admin: Optional[bool] = None,
    ) -> SearchPipelineResult:
        from ..pipelines.document_search import _search_documents_impl

        with self._storage_context():
            return _search_documents_impl(
                collection=collection,
                query_text=query_text,
                config=config,
                progress_manager=progress_manager,
                user_id=user_id,
                is_admin=is_admin,
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
        from ..pipelines.document_search import _run_document_search_impl

        with self._storage_context():
            return _run_document_search_impl(
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
        ingestion_config: Optional[IngestionConfig] = None,
        progress_callback: Optional[Callable[[str, int, int], None]] = None,
        user_id: Optional[int] = None,
        is_admin: Optional[bool] = None,
        file_handler: Optional[Callable[..., Any]] = None,
    ) -> WebIngestionResult:
        from ..pipelines.web_ingestion import _run_web_ingestion_impl

        with self._storage_context():
            result = await _run_web_ingestion_impl(
                collection=collection,
                crawl_config=crawl_config,
                ingestion_config=ingestion_config,
                progress_callback=progress_callback,
                user_id=user_id,
                is_admin=is_admin,
                file_handler=file_handler,
            )
            self.ensure_collection_backend_binding(collection)
            return result
