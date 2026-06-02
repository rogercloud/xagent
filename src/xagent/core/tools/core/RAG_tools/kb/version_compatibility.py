"""Version management compatibility facade."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

from ..core.schemas import StepType
from typing_extensions import Literal

if TYPE_CHECKING:
    from .coordinator import KBCoordinator
    from .storage_shim import KBStorageShimCompatibilityFacade


@dataclass(frozen=True)
class KBMainPointerSnapshot:
    """Snapshot of one main-pointer row before a version mutation."""

    collection: str
    doc_id: str
    step_type: str
    model_tag: Optional[str]
    pointer: Optional[Dict[str, Any]]


class KBVersionCompatibilityFacade:
    """Compatibility boundary for legacy version-management helpers.

    Version-management functions remain synchronous and keep their historical
    import paths while coordinator-owned code gets one stable surface for
    candidate listing, promotion, main-pointer operations, and cascade cleanup.
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

    def list_candidates(
        self,
        collection: str,
        doc_id: str,
        step_type: Union[StepType, str],
        model_tag: Optional[str] = None,
        state: Optional[str] = None,
        limit: int = 50,
        order_by: str = "created_at desc",
    ) -> Dict[str, Any]:
        from ..version_management.list_candidates import _list_candidates_impl

        with self._storage_context():
            return _list_candidates_impl(
                collection=collection,
                doc_id=doc_id,
                step_type=step_type,
                model_tag=model_tag,
                state=state,
                limit=limit,
                order_by=order_by,
            )

    def promote_version_main(
        self,
        collection: str,
        doc_id: str,
        step_type: Union[StepType, str],
        selected_id: str,
        operator: Optional[str] = None,
        preview_only: bool = False,
        confirm: bool = False,
        model_tag: Optional[str] = None,
    ) -> Dict[str, Any]:
        from ..version_management.promote_version_main import _promote_version_main_impl

        with self._storage_context():
            return _promote_version_main_impl(
                collection=collection,
                doc_id=doc_id,
                step_type=step_type,
                selected_id=selected_id,
                operator=operator,
                preview_only=preview_only,
                confirm=confirm,
                model_tag=model_tag,
            )

    def get_main_pointer(
        self,
        collection: str,
        doc_id: str,
        step_type: str,
        model_tag: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        from ..version_management.main_pointer_manager import _get_main_pointer_impl

        with self._storage_context():
            return _get_main_pointer_impl(
                collection=collection,
                doc_id=doc_id,
                step_type=step_type,
                model_tag=model_tag,
            )

    def set_main_pointer(
        self,
        lancedb_dir: str,
        collection: str,
        doc_id: str,
        step_type: str,
        semantic_id: str,
        technical_id: str,
        model_tag: Optional[str] = None,
        operator: Optional[str] = None,
    ) -> None:
        from ..version_management.main_pointer_manager import _set_main_pointer_impl

        with self._storage_context():
            _set_main_pointer_impl(
                lancedb_dir=lancedb_dir,
                collection=collection,
                doc_id=doc_id,
                step_type=step_type,
                semantic_id=semantic_id,
                technical_id=technical_id,
                model_tag=model_tag,
                operator=operator,
            )

    def list_main_pointers(
        self, collection: str, doc_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        from ..version_management.main_pointer_manager import _list_main_pointers_impl

        with self._storage_context():
            return _list_main_pointers_impl(collection=collection, doc_id=doc_id)

    def delete_main_pointer(
        self,
        collection: str,
        doc_id: str,
        step_type: str,
        model_tag: Optional[str] = None,
    ) -> bool:
        from ..version_management.main_pointer_manager import _delete_main_pointer_impl

        with self._storage_context():
            return _delete_main_pointer_impl(
                collection=collection,
                doc_id=doc_id,
                step_type=step_type,
                model_tag=model_tag,
            )

    def capture_main_pointer_snapshot(
        self,
        collection: str,
        doc_id: str,
        step_type: str,
        model_tag: Optional[str] = None,
    ) -> KBMainPointerSnapshot:
        return KBMainPointerSnapshot(
            collection=collection,
            doc_id=doc_id,
            step_type=step_type,
            model_tag=model_tag,
            pointer=self.get_main_pointer(collection, doc_id, step_type, model_tag),
        )

    def restore_main_pointer_snapshot(
        self,
        snapshot: KBMainPointerSnapshot,
        *,
        lancedb_dir: str = "",
        operator: Optional[str] = None,
    ) -> bool:
        if snapshot.pointer is None:
            return self.delete_main_pointer(
                snapshot.collection,
                snapshot.doc_id,
                snapshot.step_type,
                snapshot.model_tag,
            )

        self.set_main_pointer(
            lancedb_dir,
            snapshot.collection,
            snapshot.doc_id,
            snapshot.step_type,
            snapshot.pointer["semantic_id"],
            snapshot.pointer["technical_id"],
            model_tag=snapshot.model_tag,
            operator=operator,
        )
        return True

    def cascade_delete(
        self,
        *,
        target: Literal["collection", "document"],
        collection: str,
        doc_id: Optional[str] = None,
        user_id: Optional[int] = None,
        is_admin: Optional[bool] = None,
        model_tag: Optional[str] = None,
        preview_only: bool = True,
        confirm: bool = False,
        conn: Any | None = None,
    ) -> Dict[str, int]:
        from ..version_management.cascade_cleaner import _cascade_delete_impl

        with self._storage_context():
            return _cascade_delete_impl(
                target=target,
                collection=collection,
                doc_id=doc_id,
                user_id=user_id,
                is_admin=is_admin,
                model_tag=model_tag,
                preview_only=preview_only,
                confirm=confirm,
                conn=conn,
            )

    def cleanup_cascade(
        self,
        collection: str,
        doc_id: str,
        scope: str,
        new_parse_hash: Optional[str] = None,
        old_parse_hash: Optional[str] = None,
        model_tag: Optional[str] = None,
        user_id: Optional[int] = None,
        is_admin: Optional[bool] = None,
        preview_only: bool = True,
        confirm: bool = False,
    ) -> Dict[str, int]:
        from ..version_management.cascade_cleaner import _cleanup_cascade_impl

        with self._storage_context():
            return _cleanup_cascade_impl(
                collection=collection,
                doc_id=doc_id,
                scope=scope,
                new_parse_hash=new_parse_hash,
                old_parse_hash=old_parse_hash,
                model_tag=model_tag,
                user_id=user_id,
                is_admin=is_admin,
                preview_only=preview_only,
                confirm=confirm,
            )

    def cleanup_document_cascade(
        self,
        collection: str,
        doc_id: str,
        model_tag: Optional[str] = None,
        user_id: Optional[int] = None,
        is_admin: bool = True,
        preview_only: bool = True,
        confirm: bool = False,
    ) -> Dict[str, int]:
        from ..version_management.cascade_cleaner import _cleanup_document_cascade_impl

        with self._storage_context():
            return _cleanup_document_cascade_impl(
                collection=collection,
                doc_id=doc_id,
                model_tag=model_tag,
                user_id=user_id,
                is_admin=is_admin,
                preview_only=preview_only,
                confirm=confirm,
            )

    def cleanup_parse_cascade(
        self,
        collection: str,
        doc_id: str,
        old_parse_hash: Optional[str] = None,
        new_parse_hash: Optional[str] = None,
        user_id: Optional[int] = None,
        is_admin: bool = True,
        preview_only: bool = True,
        confirm: bool = False,
    ) -> Dict[str, int]:
        from ..version_management.cascade_cleaner import _cleanup_parse_cascade_impl

        with self._storage_context():
            return _cleanup_parse_cascade_impl(
                collection=collection,
                doc_id=doc_id,
                old_parse_hash=old_parse_hash,
                new_parse_hash=new_parse_hash,
                user_id=user_id,
                is_admin=is_admin,
                preview_only=preview_only,
                confirm=confirm,
            )

    def cleanup_chunk_cascade(
        self,
        collection: str,
        doc_id: str,
        old_parse_hash: Optional[str] = None,
        new_parse_hash: Optional[str] = None,
        user_id: Optional[int] = None,
        is_admin: bool = True,
        preview_only: bool = True,
        confirm: bool = False,
    ) -> Dict[str, int]:
        from ..version_management.cascade_cleaner import _cleanup_chunk_cascade_impl

        with self._storage_context():
            return _cleanup_chunk_cascade_impl(
                collection=collection,
                doc_id=doc_id,
                old_parse_hash=old_parse_hash,
                new_parse_hash=new_parse_hash,
                user_id=user_id,
                is_admin=is_admin,
                preview_only=preview_only,
                confirm=confirm,
            )

    def cleanup_embed_cascade(
        self,
        collection: str,
        doc_id: str,
        model_tag: Optional[str] = None,
        old_technical_id: Optional[str] = None,
        new_technical_id: Optional[str] = None,
        user_id: Optional[int] = None,
        is_admin: bool = True,
        preview_only: bool = True,
        confirm: bool = False,
    ) -> Dict[str, int]:
        from ..version_management.cascade_cleaner import _cleanup_embed_cascade_impl

        with self._storage_context():
            return _cleanup_embed_cascade_impl(
                collection=collection,
                doc_id=doc_id,
                model_tag=model_tag,
                old_technical_id=old_technical_id,
                new_technical_id=new_technical_id,
                user_id=user_id,
                is_admin=is_admin,
                preview_only=preview_only,
                confirm=confirm,
            )
