"""Collection metadata maintenance compatibility facade."""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from ..core.schemas import CollectionInfo


@dataclass(frozen=True)
class CollectionConfigSnapshot:
    """Snapshot of the tenant-scoped collection config row before mutation."""

    collection_name: str
    user_id: Optional[int]
    config_user_id: int
    config_json: Optional[str]
    existed: bool


@dataclass(frozen=True)
class CollectionRollbackMaintenanceResult:
    """Outcome for collection-level rollback maintenance actions."""

    collection_name: str
    status: str
    skipped: bool = False
    reason: Optional[str] = None
    cleanup_counts: dict[str, int] = field(default_factory=dict)
    warnings: tuple[str, ...] = ()
    side_effects_may_remain: bool = False


class KBMaintenanceCompatibilityFacade:
    """Compatibility boundary for legacy collection maintenance helpers.

    The legacy manager and module-level helper implementations remain in
    ``management.collection_manager``. This facade gives coordinator-owned code
    a stable maintenance entry point while preserving public helper names,
    signatures, and sync/async behavior.
    """

    def get_collection_sync(self, collection_name: str) -> "CollectionInfo":
        from ..management.collection_manager import _get_collection_sync_impl

        return _get_collection_sync_impl(collection_name)

    def initialize_collection_embedding_sync(
        self, collection_name: str, embedding_model_id: str
    ) -> "CollectionInfo":
        from ..management.collection_manager import (
            _initialize_collection_embedding_sync_impl,
        )

        return _initialize_collection_embedding_sync_impl(
            collection_name, embedding_model_id
        )

    def validate_document_processing_sync(
        self,
        collection_name: str,
        file_path: str,
        parsing_method: str,
        chunking_method: str,
    ) -> None:
        from ..management.collection_manager import (
            _validate_document_processing_sync_impl,
        )

        _validate_document_processing_sync_impl(
            collection_name, file_path, parsing_method, chunking_method
        )

    def update_collection_stats_sync(
        self,
        collection_name: str,
        documents_delta: int = 0,
        processed_documents_delta: int = 0,
        parses_delta: int = 0,
        chunks_delta: int = 0,
        embeddings_delta: int = 0,
        document_name: Optional[str] = None,
    ) -> "CollectionInfo":
        from ..management.collection_manager import _update_collection_stats_sync_impl

        return _update_collection_stats_sync_impl(
            collection_name,
            documents_delta,
            processed_documents_delta,
            parses_delta,
            chunks_delta,
            embeddings_delta,
            document_name,
        )

    def mark_collection_accessed_sync(self, collection_name: str) -> None:
        from ..management.collection_manager import _mark_collection_accessed_sync_impl

        _mark_collection_accessed_sync_impl(collection_name)

    def delete_collection_metadata_sync(
        self,
        collection_name: str,
        user_id: Optional[int],
        is_admin: bool = False,
        delete_orphaned_metadata: bool = False,
    ) -> dict[str, int]:
        from ..management.collection_manager import (
            _delete_collection_metadata_sync_impl,
        )

        return _delete_collection_metadata_sync_impl(
            collection_name,
            user_id,
            is_admin,
            delete_orphaned_metadata,
        )

    async def capture_collection_config_snapshot(
        self, collection_name: str, user_id: Optional[int]
    ) -> "CollectionConfigSnapshot":
        from ..management.collection_manager import (
            _capture_collection_config_snapshot_impl,
        )

        return await _capture_collection_config_snapshot_impl(collection_name, user_id)

    def capture_collection_config_snapshot_sync(
        self, collection_name: str, user_id: Optional[int]
    ) -> "CollectionConfigSnapshot":
        from ..management.collection_manager import (
            _capture_collection_config_snapshot_sync_impl,
        )

        return _capture_collection_config_snapshot_sync_impl(collection_name, user_id)

    async def restore_collection_config_snapshot(
        self,
        snapshot: "CollectionConfigSnapshot",
        *,
        rollback_complete: bool,
        side_effects_may_remain: bool = False,
    ) -> "CollectionRollbackMaintenanceResult":
        from ..management.collection_manager import (
            _restore_collection_config_snapshot_impl,
        )

        return await _restore_collection_config_snapshot_impl(
            snapshot,
            rollback_complete=rollback_complete,
            side_effects_may_remain=side_effects_may_remain,
        )

    def restore_collection_config_snapshot_sync(
        self,
        snapshot: "CollectionConfigSnapshot",
        *,
        rollback_complete: bool,
        side_effects_may_remain: bool = False,
    ) -> "CollectionRollbackMaintenanceResult":
        from ..management.collection_manager import (
            _restore_collection_config_snapshot_sync_impl,
        )

        return _restore_collection_config_snapshot_sync_impl(
            snapshot,
            rollback_complete=rollback_complete,
            side_effects_may_remain=side_effects_may_remain,
        )

    async def cleanup_collection_metadata_after_rollback(
        self,
        collection_name: str,
        user_id: Optional[int],
        is_admin: bool = False,
        *,
        rollback_complete: bool,
        side_effects_may_remain: bool = False,
        delete_orphaned_metadata: bool = True,
    ) -> "CollectionRollbackMaintenanceResult":
        from ..management.collection_manager import (
            _cleanup_collection_metadata_after_rollback_impl,
        )

        return await _cleanup_collection_metadata_after_rollback_impl(
            collection_name,
            user_id,
            is_admin,
            rollback_complete=rollback_complete,
            side_effects_may_remain=side_effects_may_remain,
            delete_orphaned_metadata=delete_orphaned_metadata,
        )

    def cleanup_collection_metadata_after_rollback_sync(
        self,
        collection_name: str,
        user_id: Optional[int],
        is_admin: bool = False,
        *,
        rollback_complete: bool,
        side_effects_may_remain: bool = False,
        delete_orphaned_metadata: bool = True,
    ) -> "CollectionRollbackMaintenanceResult":
        from ..management.collection_manager import (
            _cleanup_collection_metadata_after_rollback_sync_impl,
        )

        return _cleanup_collection_metadata_after_rollback_sync_impl(
            collection_name,
            user_id,
            is_admin,
            rollback_complete=rollback_complete,
            side_effects_may_remain=side_effects_may_remain,
            delete_orphaned_metadata=delete_orphaned_metadata,
        )

    async def rebuild_collection_stats(
        self,
        collection_name: str,
    ) -> Optional["CollectionInfo"]:
        from ..management.collection_manager import _rebuild_collection_stats_impl

        return await _rebuild_collection_stats_impl(collection_name)

    def rebuild_collection_stats_sync(
        self,
        collection_name: str,
    ) -> Optional["CollectionInfo"]:
        from ..management.collection_manager import _rebuild_collection_stats_sync_impl

        return _rebuild_collection_stats_sync_impl(collection_name)

    def resolve_effective_embedding_model_sync(
        self, collection_name: str, config_model_id: Optional[str] = None
    ) -> str:
        from ..management.collection_manager import (
            _resolve_effective_embedding_model_sync_impl,
        )

        return _resolve_effective_embedding_model_sync_impl(
            collection_name, config_model_id
        )

    async def rebuild_collection_metadata(self) -> None:
        from ..management.collection_manager import _rebuild_collection_metadata_impl

        await _rebuild_collection_metadata_impl()
