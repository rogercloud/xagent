"""Final platform sends are fenced separately from Agent execution."""

# Pytest fixture imports are intentionally shadowed by test parameters.
# ruff: noqa: F811

import asyncio
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock

import pytest

from tests.web.services.test_shared_channel_execution import (
    database_url as database_url,
)
from tests.web.services.test_shared_channel_execution import selected as selected
from xagent.web.models.database import get_session_local
from xagent.web.models.task import Task, TaskStatus
from xagent.web.models.task_channel_delivery import TaskChannelDelivery
from xagent.web.models.task_command import TaskExecutionCommand
from xagent.web.models.user_channel import UserChannel
from xagent.web.services import channel_delivery as delivery
from xagent.web.services import shared_channel_execution as shared
from xagent.web.services.task_orchestrator import TaskTurnPayload


@pytest.fixture
def accepted(selected):
    selected.delivery_destination = {
        "chat_id": "conversation",
        "loading_message_id": "loading",
    }
    command_id = shared._accept_channel_turn(
        selected, TaskTurnPayload("hello"), "old-ingress"
    )
    return command_id


def complete(command_id):
    with get_session_local()() as db:
        command = db.get(TaskExecutionCommand, command_id)
        command.status = "completed"
        command.result = {
            "channel_result": {
                "success": True,
                "status": "completed",
                "output": "saved answer",
            }
        }
        task = db.get(Task, command.task_id)
        task.run_id = command.target_run_id
        task.status = TaskStatus.COMPLETED
        db.commit()


def expire_claim(command_id):
    with get_session_local()() as db:
        db.get(TaskChannelDelivery, command_id).available_at = datetime.now(
            timezone.utc
        ) - timedelta(seconds=1)
        db.commit()


@pytest.mark.asyncio
async def test_recovery_sends_saved_result_once_to_saved_destination(
    accepted, selected
):
    complete(accepted)
    sender = AsyncMock()
    await delivery.recover_channel_results(selected.selection.channel_id, sender)
    await delivery.recover_channel_results(selected.selection.channel_id, sender)
    sender.assert_awaited_once()
    record, result = sender.await_args.args
    assert record.destination == selected.delivery_destination
    assert result["output"] == "saved answer"
    with get_session_local()() as db:
        assert db.get(TaskChannelDelivery, accepted).status == "delivered"
        assert db.query(TaskExecutionCommand).count() == 1


@pytest.mark.asyncio
async def test_concurrent_recovery_cannot_duplicate_active_send(accepted):
    complete(accepted)
    entered, release = asyncio.Event(), asyncio.Event()

    async def send(*args):
        entered.set()
        await release.wait()

    sender = AsyncMock(side_effect=send)
    first = asyncio.create_task(delivery.deliver_channel_result(accepted, sender))
    try:
        await entered.wait()
        await delivery.deliver_channel_result(accepted, sender)
        sender.assert_awaited_once()
    finally:
        release.set()
        await first


@pytest.mark.asyncio
async def test_platform_failure_retries_only_delivery(accepted):
    complete(accepted)
    sender = AsyncMock(side_effect=[ConnectionError("platform unavailable"), None])
    await delivery.deliver_channel_result(accepted, sender)
    await delivery.deliver_channel_result(accepted, sender)
    sender.assert_awaited_once()
    expire_claim(accepted)
    await delivery.deliver_channel_result(accepted, sender)
    assert sender.await_count == 2
    with get_session_local()() as db:
        assert db.get(TaskChannelDelivery, accepted).status == "delivered"
        assert db.query(TaskExecutionCommand).count() == 1


@pytest.mark.asyncio
async def test_send_ack_commit_failure_can_repeat_send_but_never_agent(
    accepted, monkeypatch
):
    complete(accepted)
    settle = delivery._settle

    def fail_success_once(record, *, status="pending"):
        if status == "delivered":
            monkeypatch.setattr(delivery, "_settle", settle)
            raise ConnectionError("completion commit unavailable")
        settle(record, status=status)

    monkeypatch.setattr(delivery, "_settle", fail_success_once)
    sender = AsyncMock()
    await delivery.deliver_channel_result(accepted, sender)
    expire_claim(accepted)
    await delivery.deliver_channel_result(accepted, sender)
    assert sender.await_count == 2
    with get_session_local()() as db:
        assert db.query(TaskExecutionCommand).count() == 1
        assert db.get(TaskChannelDelivery, accepted).status == "delivered"


@pytest.mark.asyncio
async def test_abandoned_claim_recovers_after_expiry(accepted):
    complete(accepted)
    claimed = delivery._claim(accepted)
    assert claimed is not None
    sender = AsyncMock()
    await delivery.deliver_channel_result(accepted, sender)
    sender.assert_not_awaited()
    expire_claim(accepted)
    await delivery.deliver_channel_result(accepted, sender)
    sender.assert_awaited_once()


@pytest.mark.asyncio
async def test_pending_notice_does_not_complete_delivery(accepted):
    sender = AsyncMock()
    await delivery.deliver_channel_result(accepted, sender, pending_notice=True)
    assert sender.await_args.args[1]["status"] == "accepted"
    with get_session_local()() as db:
        assert db.get(TaskChannelDelivery, accepted).status == "pending"
    complete(accepted)
    expire_claim(accepted)
    await delivery.deliver_channel_result(accepted, sender)
    assert sender.await_args.args[1]["output"] == "saved answer"


@pytest.mark.asyncio
async def test_revoked_channel_cannot_receive_saved_answer(accepted, selected):
    complete(accepted)
    with get_session_local()() as db:
        db.get(UserChannel, selected.selection.channel_id).config = {
            "allowed_users": ["someone-else"]
        }
        db.commit()
    sender = AsyncMock()
    await delivery.deliver_channel_result(accepted, sender)
    sender.assert_not_awaited()
    with get_session_local()() as db:
        assert db.get(TaskChannelDelivery, accepted).status == "discarded"


@pytest.mark.asyncio
async def test_terminal_command_with_replaced_run_reports_interruption(accepted):
    with get_session_local()() as db:
        command = db.get(TaskExecutionCommand, accepted)
        command.status = "completed"
        db.get(Task, command.task_id).run_id = "replacement"
        db.commit()
    sender = AsyncMock()
    await delivery.deliver_channel_result(accepted, sender)
    assert sender.await_args.args[1]["status"] == "interrupted"


@pytest.mark.asyncio
async def test_lost_delivery_claim_cancels_inflight_sender(accepted, monkeypatch):
    complete(accepted)
    monkeypatch.setattr(delivery, "_DELIVERY_LEASE_SECONDS", 0.03)
    monkeypatch.setattr(delivery, "_renew", lambda record: False)
    cancelled = asyncio.Event()

    async def send(*args):
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    assert not await asyncio.wait_for(
        delivery.deliver_channel_result(accepted, send), 1
    )
    assert cancelled.is_set()
    with get_session_local()() as db:
        assert db.get(TaskChannelDelivery, accepted).status == "pending"
