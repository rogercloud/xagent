"""add workforce core tables

Revision ID: 20260511_add_workforces_core
Revises: 7f4d2c9a1b58
Create Date: 2026-05-11
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "20260511_add_workforces_core"
down_revision: str | None = "7f4d2c9a1b58"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _table_exists(inspector: sa.Inspector, table_name: str) -> bool:
    return table_name in inspector.get_table_names()


def upgrade() -> None:
    from alembic import context

    bind = context.get_bind()
    inspector = sa.inspect(bind)

    if not _table_exists(inspector, "workforces"):
        op.create_table(
            "workforces",
            sa.Column("id", sa.Integer(), nullable=False),
            sa.Column("owner_user_id", sa.Integer(), nullable=False),
            sa.Column("scope_type", sa.String(length=50), nullable=False),
            sa.Column("scope_id", sa.String(length=200), nullable=False),
            sa.Column("name", sa.String(length=200), nullable=False),
            sa.Column("description", sa.Text(), nullable=True),
            sa.Column("manager_agent_id", sa.Integer(), nullable=False),
            sa.Column("manager_instructions", sa.Text(), nullable=True),
            sa.Column("status", sa.String(length=20), nullable=False),
            sa.Column("canvas_layout", sa.JSON(), nullable=True),
            sa.Column(
                "created_at", sa.DateTime(timezone=True), server_default=sa.func.now()
            ),
            sa.Column(
                "updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()
            ),
            sa.ForeignKeyConstraint(
                ["manager_agent_id"], ["agents.id"], ondelete="RESTRICT"
            ),
            sa.ForeignKeyConstraint(
                ["owner_user_id"], ["users.id"], ondelete="CASCADE"
            ),
            sa.PrimaryKeyConstraint("id"),
            sa.UniqueConstraint(
                "scope_type", "scope_id", "name", name="uq_workforce_scope_name"
            ),
        )
        op.create_index(op.f("ix_workforces_id"), "workforces", ["id"], unique=False)
        op.create_index(
            op.f("ix_workforces_owner_user_id"),
            "workforces",
            ["owner_user_id"],
            unique=False,
        )
        op.create_index(
            op.f("ix_workforces_scope_id"), "workforces", ["scope_id"], unique=False
        )
        op.create_index(
            op.f("ix_workforces_scope_type"), "workforces", ["scope_type"], unique=False
        )
        op.create_index(
            op.f("ix_workforces_manager_agent_id"),
            "workforces",
            ["manager_agent_id"],
            unique=False,
        )
        op.create_index(
            op.f("ix_workforces_status"), "workforces", ["status"], unique=False
        )

    if not _table_exists(inspector, "workforce_agents"):
        op.create_table(
            "workforce_agents",
            sa.Column("id", sa.Integer(), nullable=False),
            sa.Column("workforce_id", sa.Integer(), nullable=False),
            sa.Column("agent_id", sa.Integer(), nullable=False),
            sa.Column("alias", sa.String(length=200), nullable=True),
            sa.Column("assignment_instructions", sa.Text(), nullable=False),
            sa.Column("source_type", sa.String(length=20), nullable=False),
            sa.Column("template_id", sa.String(length=200), nullable=True),
            sa.Column("enabled", sa.Boolean(), nullable=False),
            sa.Column("sort_order", sa.Integer(), nullable=False),
            sa.Column("canvas_position", sa.JSON(), nullable=True),
            sa.Column(
                "created_at", sa.DateTime(timezone=True), server_default=sa.func.now()
            ),
            sa.Column(
                "updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()
            ),
            sa.ForeignKeyConstraint(["agent_id"], ["agents.id"], ondelete="RESTRICT"),
            sa.ForeignKeyConstraint(
                ["workforce_id"], ["workforces.id"], ondelete="CASCADE"
            ),
            sa.PrimaryKeyConstraint("id"),
            sa.UniqueConstraint("workforce_id", "agent_id", name="uq_workforce_agent"),
        )
        op.create_index(
            op.f("ix_workforce_agents_id"), "workforce_agents", ["id"], unique=False
        )
        op.create_index(
            op.f("ix_workforce_agents_workforce_id"),
            "workforce_agents",
            ["workforce_id"],
            unique=False,
        )
        op.create_index(
            op.f("ix_workforce_agents_agent_id"),
            "workforce_agents",
            ["agent_id"],
            unique=False,
        )

    if not _table_exists(inspector, "workforce_runs"):
        op.create_table(
            "workforce_runs",
            sa.Column("id", sa.Integer(), nullable=False),
            sa.Column("workforce_id", sa.Integer(), nullable=False),
            sa.Column("task_id", sa.Integer(), nullable=True),
            sa.Column("user_id", sa.Integer(), nullable=False),
            sa.Column("status", sa.String(length=20), nullable=False),
            sa.Column("snapshot", sa.JSON(), nullable=False),
            sa.Column(
                "created_at", sa.DateTime(timezone=True), server_default=sa.func.now()
            ),
            sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
            sa.ForeignKeyConstraint(["task_id"], ["tasks.id"], ondelete="SET NULL"),
            sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
            sa.ForeignKeyConstraint(
                ["workforce_id"], ["workforces.id"], ondelete="CASCADE"
            ),
            sa.PrimaryKeyConstraint("id"),
            sa.UniqueConstraint("task_id"),
        )
        op.create_index(
            op.f("ix_workforce_runs_id"), "workforce_runs", ["id"], unique=False
        )
        op.create_index(
            op.f("ix_workforce_runs_workforce_id"),
            "workforce_runs",
            ["workforce_id"],
            unique=False,
        )
        op.create_index(
            op.f("ix_workforce_runs_user_id"),
            "workforce_runs",
            ["user_id"],
            unique=False,
        )
        op.create_index(
            op.f("ix_workforce_runs_status"), "workforce_runs", ["status"], unique=False
        )

    if not _table_exists(inspector, "workforce_builder_messages"):
        op.create_table(
            "workforce_builder_messages",
            sa.Column("id", sa.Integer(), nullable=False),
            sa.Column("workforce_id", sa.Integer(), nullable=False),
            sa.Column("user_id", sa.Integer(), nullable=False),
            sa.Column("role", sa.String(length=20), nullable=False),
            sa.Column("content", sa.Text(), nullable=False),
            sa.Column("proposed_patch", sa.JSON(), nullable=True),
            sa.Column("status", sa.String(length=20), nullable=False),
            sa.Column(
                "created_at", sa.DateTime(timezone=True), server_default=sa.func.now()
            ),
            sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
            sa.ForeignKeyConstraint(
                ["workforce_id"], ["workforces.id"], ondelete="CASCADE"
            ),
            sa.PrimaryKeyConstraint("id"),
        )
        op.create_index(
            op.f("ix_workforce_builder_messages_id"),
            "workforce_builder_messages",
            ["id"],
            unique=False,
        )
        op.create_index(
            op.f("ix_workforce_builder_messages_workforce_id"),
            "workforce_builder_messages",
            ["workforce_id"],
            unique=False,
        )
        op.create_index(
            op.f("ix_workforce_builder_messages_user_id"),
            "workforce_builder_messages",
            ["user_id"],
            unique=False,
        )


def downgrade() -> None:
    from alembic import context

    bind = context.get_bind()
    inspector = sa.inspect(bind)

    if _table_exists(inspector, "workforce_builder_messages"):
        op.drop_index(
            op.f("ix_workforce_builder_messages_user_id"),
            table_name="workforce_builder_messages",
        )
        op.drop_index(
            op.f("ix_workforce_builder_messages_workforce_id"),
            table_name="workforce_builder_messages",
        )
        op.drop_index(
            op.f("ix_workforce_builder_messages_id"),
            table_name="workforce_builder_messages",
        )
        op.drop_table("workforce_builder_messages")

    if _table_exists(inspector, "workforce_runs"):
        op.drop_index(op.f("ix_workforce_runs_status"), table_name="workforce_runs")
        op.drop_index(op.f("ix_workforce_runs_user_id"), table_name="workforce_runs")
        op.drop_index(
            op.f("ix_workforce_runs_workforce_id"), table_name="workforce_runs"
        )
        op.drop_index(op.f("ix_workforce_runs_id"), table_name="workforce_runs")
        op.drop_table("workforce_runs")

    if _table_exists(inspector, "workforce_agents"):
        op.drop_index(
            op.f("ix_workforce_agents_agent_id"), table_name="workforce_agents"
        )
        op.drop_index(
            op.f("ix_workforce_agents_workforce_id"),
            table_name="workforce_agents",
        )
        op.drop_index(op.f("ix_workforce_agents_id"), table_name="workforce_agents")
        op.drop_table("workforce_agents")

    if _table_exists(inspector, "workforces"):
        op.drop_index(op.f("ix_workforces_status"), table_name="workforces")
        op.drop_index(op.f("ix_workforces_manager_agent_id"), table_name="workforces")
        op.drop_index(op.f("ix_workforces_scope_type"), table_name="workforces")
        op.drop_index(op.f("ix_workforces_scope_id"), table_name="workforces")
        op.drop_index(op.f("ix_workforces_owner_user_id"), table_name="workforces")
        op.drop_index(op.f("ix_workforces_id"), table_name="workforces")
        op.drop_table("workforces")
