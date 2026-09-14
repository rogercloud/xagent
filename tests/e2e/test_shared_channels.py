"""Platform callbacks use durable START and real worker execution.

Only platform network I/O is replaced; selection, authorization, transcript,
Redis trace forwarding, AgentService and final rendering use production code.
"""

import asyncio
import importlib
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

pytestmark = pytest.mark.e2e


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "platform,with_file",
    [("slack", False), ("feishu", False), ("telegram", False), ("telegram", True)],
)
async def test_channel_callback_runs_remotely_and_returns_answer(
    shared_app, monkeypatch, platform, with_file, tmp_path
):
    from xagent.web.models.database import get_session_local
    from xagent.web.models.task import Task
    from xagent.web.models.user_channel import UserChannel
    from xagent.web.services.task_event_bridge import (
        start_task_event_bridge,
        stop_task_event_bridge,
    )

    app = shared_app
    monkeypatch.chdir(tmp_path)
    with get_session_local()() as db:
        channel = UserChannel(
            user_id=app.user_id,
            channel_type=platform,
            channel_name="E2E",
            is_active=True,
        )
        channel.config = {"allowed_users": ["sender", "123"]}
        db.add(channel)
        db.commit()
        channel_id = channel.id
    forwarded = []
    module = importlib.import_module(f"xagent.web.channels.{platform}.bot")
    handler_type = getattr(
        module,
        {
            "slack": "SlackTraceHandler",
            "feishu": "FeishuTraceHandler",
            "telegram": "TelegramTraceHandler",
        }[platform],
    )
    handle_event = handler_type.handle_event

    async def observe_event(self, event):
        forwarded.append(event)
        await handle_event(self, event)

    monkeypatch.setattr(handler_type, "handle_event", observe_event)
    await start_task_event_bridge()
    bot = None
    try:
        if platform == "slack":
            from xagent.web.channels.slack.bot import SlackBotInstance

            bot = SlackBotInstance("test-token", None, "e2e", channel_id, "E2E", "bot")
            bot.web_client = SimpleNamespace(
                chat_postMessage=AsyncMock(return_value={"ts": "loading"}),
                chat_update=AsyncMock(return_value={"ok": True}),
            )
            await asyncio.wait_for(
                bot._process_event(
                    "conversation",
                    {},
                    {
                        "type": "message",
                        "channel_type": "im",
                        "channel": "D1",
                        "user": "sender",
                        "ts": "1.0",
                        "text": "Channel question",
                    },
                ),
                30,
            )
            assert "Shared E2E answer" in str(bot.web_client.chat_update.call_args_list)
        elif platform == "feishu":
            from xagent.web.channels.feishu.bot import FeishuBotInstance

            bot = FeishuBotInstance("test-id", "test-secret", "e2e", channel_id, "E2E")
            bot.api_client = Mock()
            bot.api_client.im.v1.message.patch.return_value.success.return_value = True
            bot._send_text = AsyncMock(return_value="loading")
            bot._update_text = AsyncMock()
            message = SimpleNamespace(
                event=SimpleNamespace(
                    message=SimpleNamespace(
                        chat_id="chat",
                        message_id="message",
                        message_type="text",
                        content='{"text":"Channel question"}',
                    )
                )
            )
            await asyncio.wait_for(bot._process_messages_batch("sender", [message]), 30)
            assert "Shared E2E answer" in str(bot._update_text.call_args_list)
        else:
            from xagent.web.channels.telegram.bot import TelegramBotInstance

            bot = TelegramBotInstance(
                "123456:ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghi", "e2e", channel_id, "E2E"
            )
            await bot.bot.session.close()

            async def download_file(_path, *, destination):
                Path(destination).write_bytes(b"unique shared input\n")

            bot.bot = SimpleNamespace(
                edit_message_text=AsyncMock(),
                send_message=AsyncMock(),
                get_file=AsyncMock(
                    return_value=SimpleNamespace(file_path="source.txt")
                ),
                download_file=download_file,
            )
            delivered = []

            async def answer_document(document, **kwargs):
                delivered.append(Path(document.path).read_bytes())
                return SimpleNamespace(delete=AsyncMock())

            loading = SimpleNamespace(
                message_id=77, edit_text=AsyncMock(), delete=AsyncMock()
            )
            message = SimpleNamespace(
                from_user=SimpleNamespace(id=123),
                chat=SimpleNamespace(id=456),
                answer=AsyncMock(return_value=loading),
                answer_document=answer_document,
                text="e2e:files" if with_file else "Channel question",
                caption=None,
                document=SimpleNamespace(
                    file_id="platform-file",
                    file_name="source.txt",
                    mime_type="text/plain",
                    file_size=20,
                )
                if with_file
                else None,
                photo=None,
                audio=None,
                voice=None,
                video=None,
            )
            await asyncio.wait_for(bot._process_user_messages_batch(123, [message]), 30)
            assert "Shared E2E answer" in str(loading.edit_text.call_args_list)
            if with_file:
                assert delivered == [b"UNIQUE SHARED INPUT\n"]
        with get_session_local()() as db:
            tasks = db.query(Task).all()
            assert len(tasks) == 1
            task_id = tasks[0].id
        app.wait_task(task_id)
        calls = [
            json.loads(line)
            for path in app.root.glob("model-*.jsonl")
            for line in path.read_text().splitlines()
        ]
        assert calls and {call["role"] for call in calls} == {"worker"}
        assert forwarded
        assert all(
            event.task_id is None or str(event.task_id) == str(task_id)
            for event in forwarded
        )
    finally:
        await stop_task_event_bridge()
        from xagent.web.services.task_events import (
            set_task_command_delivery,
            set_task_event_sink,
        )

        set_task_command_delivery(None)
        set_task_event_sink(None)


@pytest.mark.asyncio
@pytest.mark.parametrize("action", ["stop", "deactivate"])
async def test_queued_channel_control_is_applied_by_worker(shared_app, action):
    from xagent.web.models.database import get_session_local
    from xagent.web.models.task_command import TaskExecutionCommand
    from xagent.web.models.user_channel import UserChannel
    from xagent.web.services.shared_channel_execution import prepare_shared_channel_turn
    from xagent.web.services.task_event_bridge import (
        start_task_event_bridge,
        stop_task_event_bridge,
    )
    from xagent.web.services.task_events import (
        set_task_command_delivery,
        set_task_event_sink,
    )
    from xagent.web.services.task_orchestrator import TaskTurnPayload

    app = shared_app
    worker, pipe = app.processes[-1], app.pipes[-1]
    pipe.send("stop")
    worker.join(30)
    assert worker.exitcode == 0, app.diagnostics()
    with get_session_local()() as db:
        channel = UserChannel(
            user_id=app.user_id,
            channel_type="telegram",
            channel_name="Queued",
            is_active=True,
        )
        channel.config = {"allowed_users": ["123"]}
        db.add(channel)
        db.commit()
        channel_id = channel.id
    await start_task_event_bridge()
    turn = None
    execution = None
    try:
        turn = await prepare_shared_channel_turn(
            channel_id=channel_id,
            external_user_id="123",
            active_task_id=None,
            text="e2e:gate",
            channel_name="Queued",
        )
        assert turn is not None
        execution = asyncio.create_task(
            turn.execute(
                TaskTurnPayload(
                    transcript_message="e2e:gate", execution_message="e2e:gate"
                ),
                None,
            )
        )
        async with asyncio.timeout(10):
            while not turn.accepted:
                await asyncio.sleep(0.02)
        if action == "stop":
            await turn.stop()
        else:
            with get_session_local()() as db:
                db.get(UserChannel, channel_id).is_active = False
                db.commit()
        await asyncio.to_thread(app.start, "worker")
        result = await asyncio.wait_for(execution, 30)
        expected = "paused" if action == "stop" else "failed"
        await asyncio.to_thread(app.wait_task, turn.selection.task_id, status=expected)
        if action == "stop":
            assert result["status"] in {"paused", "interrupted"}
            with get_session_local()() as db:
                stop = db.query(TaskExecutionCommand).filter_by(kind="pause").one()
                assert stop.status == "completed"
                assert stop.target_run_id == turn.run_id
        else:
            assert not list(app.root.glob("model-*.jsonl"))
    finally:
        if execution is not None and not execution.done():
            execution.cancel()
            await asyncio.gather(execution, return_exceptions=True)
        if turn is not None:
            await turn.close()
        await stop_task_event_bridge()
        set_task_command_delivery(None)
        set_task_event_sink(None)
