"""Retrieval helper compatibility facade."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from ..core.schemas import SearchResult
from ..utils.user_scope import resolve_user_scope

if TYPE_CHECKING:
    from .coordinator import KBCoordinator
    from .storage_shim import KBStorageShimCompatibilityFacade


class KBRetrievalHelperCompatibilityFacade:
    """Compatibility boundary for low-level retrieval helper functions.

    Retrieval helpers keep their historical import paths, sync/async shapes,
    tuple return contracts, filter parsing, score conversion, index advice, and
    prompt-context formatting. The facade gives coordinator-owned code one
    retrieval boundary while delegating to the current helper implementations.
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

    def search_dense_engine(
        self,
        collection: str,
        model_tag: str,
        query_vector: List[float],
        *,
        top_k: int,
        filters: Optional[Dict[str, Any]] = None,
        readonly: bool = False,
        nprobes: Optional[int] = None,
        refine_factor: Optional[int] = None,
        user_id: Optional[int] = None,
        is_admin: Optional[bool] = None,
    ) -> Tuple[List[SearchResult], str, Optional[str]]:
        from ..retrieval.search_engine import _search_dense_engine_impl

        user_scope = resolve_user_scope(user_id=user_id, is_admin=is_admin)
        with self._storage_context():
            return _search_dense_engine_impl(
                collection=collection,
                model_tag=model_tag,
                query_vector=query_vector,
                top_k=top_k,
                filters=filters,
                readonly=readonly,
                nprobes=nprobes,
                refine_factor=refine_factor,
                user_id=user_scope.user_id,
                is_admin=user_scope.is_admin,
            )

    async def search_dense_engine_async(
        self,
        collection: str,
        model_tag: str,
        query_vector: List[float],
        *,
        top_k: int,
        filters: Optional[Dict[str, Any]] = None,
        readonly: bool = False,
        nprobes: Optional[int] = None,
        refine_factor: Optional[int] = None,
        user_id: Optional[int] = None,
        is_admin: Optional[bool] = None,
    ) -> Tuple[List[SearchResult], str, Optional[str]]:
        from ..retrieval.search_engine import _search_dense_engine_async_impl

        user_scope = resolve_user_scope(user_id=user_id, is_admin=is_admin)
        with self._storage_context():
            return await _search_dense_engine_async_impl(
                collection=collection,
                model_tag=model_tag,
                query_vector=query_vector,
                top_k=top_k,
                filters=filters,
                readonly=readonly,
                nprobes=nprobes,
                refine_factor=refine_factor,
                user_id=user_scope.user_id,
                is_admin=user_scope.is_admin,
            )

    def format_search_results_for_llm(
        self,
        search_results: List[SearchResult],
        top_k: Optional[int] = None,
        include_metadata: bool = False,
        separator: str = "\n---\n",
    ) -> str:
        from ..retrieval.format_context import _format_search_results_for_llm_impl

        return _format_search_results_for_llm_impl(
            search_results,
            top_k=top_k,
            include_metadata=include_metadata,
            separator=separator,
        )
