"""Durable channel input identity, before task selection or attachment binding."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, cast

from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from ..models.database import get_session_local
from ..models.task import Task
from ..models.task_command import TaskExecutionCommand
from ..models.task_input_receipt import TaskInputReceipt
from .channel_runtime import (
    SelectedChannelTask,
    _load_channel_owner_sync,
    prepare_channel_task_no_commit,
)
from .shared_channel_execution import SharedChannelTurn, accept_channel_turn_no_commit
from .task_command_transport import (
    _resolve_actor_subject,
    command_identity_matches_task,
)
from .task_orchestrator import TaskTurnError, TaskTurnPayload
from .uploaded_file_store import StagedUploadedFile, UploadedFileStore


@dataclass(frozen=True)
class ChannelInput:
    channel_id: int
    external_user_id: str
    source: str
    scope: tuple[str, ...]
    message_id: str
    text: str
    source_file_ids: tuple[str, ...]
    destination: dict[str, Any]

    def payload_hash(self) -> str:
        return _hash([self.text, self.source_file_ids, self.destination])


def _hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _identity(db: Session, incoming: ChannelInput) -> tuple[int, str]:
    owner = _load_channel_owner_sync(
        db, channel_id=incoming.channel_id, external_user_id=incoming.external_user_id
    )
    subject = _resolve_actor_subject(db, owner.user_id)
    if subject is None:
        raise TaskTurnError("input_unavailable")
    return owner.user_id, _hash(
        [
            "channel/input/v1",
            subject,
            incoming.channel_id,
            incoming.external_user_id,
            incoming.source,
            incoming.scope,
            incoming.message_id,
        ]
    )


@dataclass(frozen=True)
class AcceptedChannelInput:
    task_id: int
    command_db_id: int
    command_id: str
    run_id: str
    user_id: int
    replayed: bool
    selection: SelectedChannelTask

    def as_turn(self) -> SharedChannelTurn:
        return SharedChannelTurn(
            self.selection,
            workspace=None,
            run_id=self.run_id,
            command_id=self.command_id,
            accepted=True,
            command_db_id=self.command_db_id,
        )


def _replay(
    db: Session, receipt: TaskInputReceipt, incoming: ChannelInput, owner_id: int
) -> AcceptedChannelInput:
    task = db.get(Task, receipt.task_id) if receipt.task_id is not None else None
    command = (
        db.get(TaskExecutionCommand, receipt.command_db_id)
        if receipt.command_db_id is not None
        else None
    )
    if (
        task is None
        or command is None
        or task.user_id != owner_id
        or task.channel_id != incoming.channel_id
        or command.task_id != task.id
        or not command_identity_matches_task(db, task, command)
    ):
        raise TaskTurnError("input_unavailable")
    if receipt.payload_hash != incoming.payload_hash():
        raise TaskTurnError("input_conflict")
    return AcceptedChannelInput(
        int(task.id),
        int(command.id),
        str(command.command_id),
        str(command.target_run_id),
        owner_id,
        True,
        SelectedChannelTask(
            owner_id,
            int(task.id),
            False,
            incoming.channel_id,
            incoming.external_user_id,
            cast(str | None, task.run_id),
            int(task.state_version),
        ),
    )


def lookup_channel_input(
    incoming: ChannelInput,
) -> tuple[int, AcceptedChannelInput | None]:
    """Authorize before downloading, without selecting or creating a task."""
    with get_session_local()() as db:
        owner_id, identity = _identity(db, incoming)
        receipt = db.get(TaskInputReceipt, identity)
        result = (
            _replay(db, receipt, incoming, owner_id) if receipt is not None else None
        )
        # Legacy owner subject initialization belongs to this short transaction.
        db.commit()
        return owner_id, result


def accept_channel_input(
    incoming: ChannelInput,
    *,
    owner_id: int,
    active_task_id: int | None,
    channel_name: str | None,
    payload: TaskTurnPayload,
    staged_files: tuple[StagedUploadedFile, ...],
    host_id: str,
) -> AcceptedChannelInput:
    """Commit receipt, selection, files, transcript, START and reply mapping once."""
    with get_session_local()() as db:
        current_owner_id, identity = _identity(db, incoming)
        if current_owner_id != owner_id:
            raise TaskTurnError("owner_changed")
        existing = db.get(TaskInputReceipt, identity)
        if existing is not None:
            return _replay(db, existing, incoming, owner_id)
        receipt = TaskInputReceipt(
            identity_hash=identity, payload_hash=incoming.payload_hash()
        )
        db.add(receipt)
        try:
            db.flush()
        except IntegrityError:
            db.rollback()
            current_owner_id, _ = _identity(db, incoming)
            existing = db.get(TaskInputReceipt, identity)
            if existing is None:
                raise
            return _replay(db, existing, incoming, current_owner_id)
        selection = prepare_channel_task_no_commit(
            db,
            channel_id=incoming.channel_id,
            external_user_id=incoming.external_user_id,
            active_task_id=active_task_id,
            text=payload.transcript_message,
            channel_name=channel_name,
            expected_owner_user_id=owner_id,
            defer_execution=True,
        )
        if selection is None:
            raise TaskTurnError("busy")
        assert isinstance(selection, SelectedChannelTask)
        for staged in staged_files:
            if staged.user_id != owner_id or staged.task_id is not None:
                raise TaskTurnError("file_unavailable")
            record = staged.to_record()
            setattr(record, "task_id", selection.task_id)
            UploadedFileStore(db).add_already_durable(record)
        db.flush()
        turn = SharedChannelTurn(selection, workspace=None)
        turn.delivery_destination = dict(incoming.destination)
        command_id = accept_channel_turn_no_commit(db, turn, payload, host_id)
        receipt.task_id = selection.task_id
        receipt.command_db_id = command_id
        try:
            db.commit()
        except Exception:
            db.close()
            with get_session_local()() as check:
                current_owner_id, _ = _identity(check, incoming)
                saved = check.get(TaskInputReceipt, identity)
                if saved is None:
                    raise
                recovered = _replay(check, saved, incoming, current_owner_id)
                if recovered.command_id != turn.command_id:
                    return recovered
                # Our own commit succeeded. Preserve its selection so ingress
                # installs the new conversation mapping just as on a known commit.
        return AcceptedChannelInput(
            selection.task_id,
            command_id,
            turn.command_id,
            turn.run_id,
            owner_id,
            False,
            selection,
        )
