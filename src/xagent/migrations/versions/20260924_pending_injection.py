"""Persist unresolved input across execution-owner restarts."""

import sqlalchemy as sa
from alembic import op

revision = "20260924_pending_injection"
down_revision = "20260923_seed_freshdesk_mcp_app"
branch_labels = None
depends_on = None


def upgrade() -> None:
    inspector = sa.inspect(op.get_bind())
    # Core tables are metadata-owned and may be absent in Alembic-only runs.
    if not inspector.has_table("tasks"):
        return
    if "pending_injection" not in {c["name"] for c in inspector.get_columns("tasks")}:
        op.add_column(
            "tasks",
            sa.Column("pending_injection", sa.JSON(none_as_null=True), nullable=True),
        )
    if "ix_tasks_pending_injection" not in {
        i["name"] for i in inspector.get_indexes("tasks")
    }:
        op.create_index(
            "ix_tasks_pending_injection",
            "tasks",
            ["id"],
            sqlite_where=sa.text("pending_injection IS NOT NULL"),
            postgresql_where=sa.text("pending_injection IS NOT NULL"),
        )


def downgrade() -> None:
    if not sa.inspect(op.get_bind()).has_table("tasks"):
        return
    op.drop_index("ix_tasks_pending_injection", table_name="tasks")
    op.drop_column("tasks", "pending_injection")
