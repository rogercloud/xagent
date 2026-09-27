"""add partial index on task_chat_messages for pending deliveries

The R6 lease-recovery loop (task_lease_recovery.py) queries
task_chat_messages rows with delivery_status = 'pending' every tick, both to
page orphaned task_ids and to close one task's rows inside its own recovery
transaction. There was no index on delivery_status, so each tick scanned the
whole table. Almost every row settles out of "pending" quickly, so a partial
index scoped to that one value stays small regardless of table growth.

(task_id, created_at) are carried alongside the predicate: task_id backs the
per-task equality lookup both call sites use, and created_at backs the
sweep's additional "created_at < created_before" filter over the same rows.

Revision ID: 20260927_pending_delivery_index
Revises: 20260916_durable_create_operations
Create Date: 2026-09-27

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "20260927_pending_delivery_index"
down_revision: Union[str, None] = "20260916_durable_create_operations"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

TABLE = "task_chat_messages"
INDEX = "ix_task_chat_messages_pending_delivery"
INDEX_COLUMNS = ("task_id", "created_at")
# The predicate column itself, so a database whose task_chat_messages
# predates 20260710_add_chat_message_delivery_state (which introduced
# delivery_status) is recognized as not ready for this index yet, same as
# the two index columns below.
REQUIRED_COLUMNS = (*INDEX_COLUMNS, "delivery_status")
PREDICATE = sa.text("delivery_status = 'pending'")


def _columns() -> set[str]:
    inspector = sa.inspect(op.get_bind())
    if TABLE not in inspector.get_table_names():
        return set()
    return {column["name"] for column in inspector.get_columns(TABLE)}


def _indexes() -> set[str]:
    inspector = sa.inspect(op.get_bind())
    if TABLE not in inspector.get_table_names():
        return set()
    return {index["name"] for index in inspector.get_indexes(TABLE)}


def upgrade() -> None:
    columns = _columns()
    if not columns:
        return
    # A database whose task_chat_messages predates one of these columns (an
    # Alembic-only install stopped short of the migration that added it, or a
    # legacy fixture in a test) cannot support this predicate or these index
    # columns yet; skip rather than fail the whole chain, matching the
    # guard-clause convention 20260922_task_last_activity_at.py and
    # 20260710_add_chat_message_delivery_state.py use for the same table.
    if not set(REQUIRED_COLUMNS) <= columns:
        return
    if INDEX in _indexes():
        return
    dialect = op.get_bind().dialect.name
    kwargs: dict[str, object] = {}
    if dialect == "sqlite":
        kwargs["sqlite_where"] = PREDICATE
    elif dialect == "postgresql":
        kwargs["postgresql_where"] = PREDICATE
    op.create_index(INDEX, TABLE, list(INDEX_COLUMNS), **kwargs)


def downgrade() -> None:
    if INDEX in _indexes():
        op.drop_index(INDEX, table_name=TABLE)
