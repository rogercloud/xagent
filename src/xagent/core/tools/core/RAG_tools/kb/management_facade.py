"""Core management compatibility facade for KB operations."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from ..core.schemas import (
    CollectionOperationResult,
    DocumentListResult,
    DocumentOperationResult,
    DocumentStatsResult,
    ListCollectionsResult,
)


class KBCoreManagementCompatibilityFacade:
    """Compatibility boundary for legacy management module functions.

    Public management modules keep their historical import paths and signatures,
    while coordinator-owned code gets one stable surface for list, delete,
    retry/cancel, and ingestion-status operations.
    """

    async def list_collections(
        self,
        user_id: Optional[int] = None,
        is_admin: Optional[bool] = None,
        force_realtime: bool = False,
    ) -> ListCollectionsResult:
        from ..management import collections as management_collections

        return await management_collections._list_collections_impl(
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
        from ..management import collections as management_collections

        return management_collections._list_documents_impl(
            collection=collection,
            user_id=user_id,
            is_admin=is_admin,
        )

    def get_document_stats(
        self,
        collection: str,
        doc_id: str,
        model_tag: Optional[str] = None,
        user_id: Optional[int] = None,
        is_admin: bool = False,
    ) -> DocumentStatsResult:
        from ..management import collections as management_collections

        return management_collections._get_document_stats_impl(
            collection=collection,
            doc_id=doc_id,
            model_tag=model_tag,
            user_id=user_id,
            is_admin=is_admin,
        )

    def delete_document(
        self,
        collection: str,
        doc_id: str,
        user_id: int,
        is_admin: bool = False,
    ) -> DocumentOperationResult:
        from ..management import collections as management_collections

        return management_collections._delete_document_impl(
            collection=collection,
            doc_id=doc_id,
            user_id=user_id,
            is_admin=is_admin,
        )

    def delete_collection(
        self,
        collection: str,
        user_id: Optional[int] = None,
        is_admin: bool = False,
    ) -> CollectionOperationResult:
        from ..management import collections as management_collections

        return management_collections._delete_collection_impl(
            collection=collection,
            user_id=user_id,
            is_admin=is_admin,
        )

    def retry_document(
        self,
        collection: str,
        doc_id: str,
        user_id: int,
        is_admin: bool = False,
    ) -> DocumentOperationResult:
        from ..management import collections as management_collections

        return management_collections._retry_document_impl(
            collection=collection,
            doc_id=doc_id,
            user_id=user_id,
            is_admin=is_admin,
        )

    def cancel_document(
        self,
        collection: str,
        doc_id: str,
        user_id: int,
        is_admin: bool = False,
        reason: Optional[str] = None,
    ) -> DocumentOperationResult:
        from ..management import collections as management_collections

        return management_collections._cancel_document_impl(
            collection=collection,
            doc_id=doc_id,
            user_id=user_id,
            is_admin=is_admin,
            reason=reason,
        )

    def cancel_collection(
        self,
        collection: str,
        reason: Optional[str] = None,
        user_id: Optional[int] = None,
        is_admin: bool = False,
    ) -> CollectionOperationResult:
        from ..management import collections as management_collections

        return management_collections._cancel_collection_impl(
            collection=collection,
            reason=reason,
            user_id=user_id,
            is_admin=is_admin,
        )

    def get_document_status(self, collection: str, doc_id: str) -> Dict[str, Any]:
        from ..management import collections as management_collections

        return management_collections._get_document_status_impl(
            collection=collection,
            doc_id=doc_id,
        )

    def write_ingestion_status(
        self,
        collection: str,
        doc_id: str,
        *,
        status: str,
        message: Optional[str] = None,
        parse_hash: Optional[str] = None,
        user_id: Optional[int] = None,
    ) -> None:
        from ..management import status as management_status

        management_status._write_ingestion_status_impl(
            collection=collection,
            doc_id=doc_id,
            status=status,
            message=message,
            parse_hash=parse_hash,
            user_id=user_id,
        )

    def load_ingestion_status(
        self,
        collection: Optional[str] = None,
        doc_id: Optional[str] = None,
        user_id: Optional[int] = None,
        is_admin: bool = False,
    ) -> List[Dict[str, Any]]:
        from ..management import status as management_status

        return management_status._load_ingestion_status_impl(
            collection=collection,
            doc_id=doc_id,
            user_id=user_id,
            is_admin=is_admin,
        )

    def clear_ingestion_status(
        self,
        collection: str,
        doc_id: str,
        user_id: Optional[int] = None,
        is_admin: bool = False,
    ) -> None:
        from ..management import status as management_status

        management_status._clear_ingestion_status_impl(
            collection=collection,
            doc_id=doc_id,
            user_id=user_id,
            is_admin=is_admin,
        )

    async def write_ingestion_status_async(
        self,
        collection: str,
        doc_id: str,
        *,
        status: str,
        message: Optional[str] = None,
        parse_hash: Optional[str] = None,
        user_id: Optional[int] = None,
    ) -> None:
        from ..management import status as management_status

        await management_status._write_ingestion_status_async_impl(
            collection=collection,
            doc_id=doc_id,
            status=status,
            message=message,
            parse_hash=parse_hash,
            user_id=user_id,
        )

    async def load_ingestion_status_async(
        self,
        collection: Optional[str] = None,
        doc_id: Optional[str] = None,
        user_id: Optional[int] = None,
        is_admin: bool = False,
    ) -> List[Dict[str, Any]]:
        from ..management import status as management_status

        return await management_status._load_ingestion_status_async_impl(
            collection=collection,
            doc_id=doc_id,
            user_id=user_id,
            is_admin=is_admin,
        )

    async def clear_ingestion_status_async(
        self,
        collection: str,
        doc_id: str,
        user_id: Optional[int] = None,
        is_admin: bool = False,
    ) -> None:
        from ..management import status as management_status

        await management_status._clear_ingestion_status_async_impl(
            collection=collection,
            doc_id=doc_id,
            user_id=user_id,
            is_admin=is_admin,
        )
