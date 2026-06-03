"""Tests for KB operation rollback compatibility outcomes."""

from __future__ import annotations

import pytest

from xagent.core.tools.core.RAG_tools.kb import (
    KBOperationCompatibilityFacade,
    RollbackStatus,
    SideEffectPlane,
)


def test_operation_compensation_steps_are_idempotent_and_lifo() -> None:
    facade = KBOperationCompatibilityFacade()

    with facade.start_operation(
        operation_type="document_ingestion",
        collection="demo",
    ) as operation:
        operation.record_side_effect(
            name="remove_document",
            plane=SideEffectPlane.DOCUMENT,
            payload={"doc_id": "doc-1"},
            idempotency_key="document:doc-1",
        )
        operation.record_side_effect(
            name="remove_parse",
            plane=SideEffectPlane.PARSE,
            payload={"parse_hash": "parse-1"},
            idempotency_key="parse:parse-1",
        )
        operation.record_side_effect(
            name="remove_document",
            plane=SideEffectPlane.DOCUMENT,
            payload={"doc_id": "doc-1"},
            idempotency_key="document:doc-1",
        )
        operation.finish(
            status="partial",
            rollback_status=RollbackStatus.INCOMPLETE,
            side_effects_may_remain=True,
        )

    outcome = facade.last_outcome

    assert outcome is not None
    assert [step.name for step in outcome.compensation_steps] == [
        "remove_document",
        "remove_parse",
    ]
    assert [step.name for step in outcome.compensation_plan] == [
        "remove_parse",
        "remove_document",
    ]
    assert outcome.rollback_status is RollbackStatus.INCOMPLETE
    assert outcome.side_effects_may_remain is True


class _OperationCancelled(BaseException):
    pass


def test_operation_base_exception_records_error_outcome() -> None:
    facade = KBOperationCompatibilityFacade()

    with pytest.raises(_OperationCancelled):
        with facade.start_operation(
            operation_type="document_ingestion",
            collection="demo",
        ) as operation:
            operation.record_side_effect(
                name="remove_document",
                plane=SideEffectPlane.DOCUMENT,
                payload={"doc_id": "doc-1"},
                idempotency_key="document:doc-1",
            )
            raise _OperationCancelled("cancelled")

    outcome = facade.last_outcome
    assert outcome is not None
    assert outcome.status == "error"
    assert outcome.rollback_status is RollbackStatus.INCOMPLETE
    assert outcome.side_effects_may_remain is True
    assert outcome.warnings == ("cancelled",)
