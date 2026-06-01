"""Parse display compatibility facade."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from .models import KBAccessMode, KBContextRequest

if TYPE_CHECKING:
    from ..core.schemas import ParsedElementDisplay
    from ..storage.contracts import VectorIndexStore
    from .coordinator import KBCoordinator
    from .storage_shim import KBStorageShimCompatibilityFacade


class KBParseDisplayCompatibilityFacade:
    """Compatibility boundary for legacy parse display helpers.

    Parse display helpers are synchronous read-only APIs. The facade resolves
    coordinator-owned context and storage access while preserving the legacy
    helper names, signatures, tuple shapes, and conversion behavior.
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

    def _resolve_vector_store(
        self,
        collection: str,
        user_id: Optional[int],
        is_admin: bool,
    ) -> VectorIndexStore:
        if self._coordinator is not None:
            context = self._coordinator.get_context_sync(
                KBContextRequest(
                    collection=collection,
                    user_id=user_id,
                    is_admin=is_admin,
                    access_mode=KBAccessMode.READ,
                    hide_missing=True,
                )
            )
            return context.vector_index_store

        storage_shim = self._active_storage_shim()
        if storage_shim is not None:
            return storage_shim.get_vector_index_store()

        from ..storage.factory import get_vector_index_store

        return get_vector_index_store()

    def reconstruct_parse_result_from_db(
        self,
        collection: str,
        doc_id: str,
        parse_hash: Optional[str] = None,
        user_id: Optional[int] = None,
        is_admin: bool = False,
    ) -> Tuple[List[Dict[str, Any]], Optional[str]]:
        from ..parse.parse_display import _reconstruct_parse_result_from_db_impl

        vector_store = self._resolve_vector_store(collection, user_id, is_admin)
        return _reconstruct_parse_result_from_db_impl(
            collection,
            doc_id,
            parse_hash=parse_hash,
            user_id=user_id,
            is_admin=is_admin,
            vector_store=vector_store,
        )

    def paginate_parse_results(
        self,
        elements: List[Dict[str, Any]],
        page: int = 1,
        page_size: int = 20,
    ) -> Tuple[List[ParsedElementDisplay], Dict[str, Any]]:
        from ..parse.parse_display import _paginate_parse_results_impl

        return _paginate_parse_results_impl(elements, page=page, page_size=page_size)
