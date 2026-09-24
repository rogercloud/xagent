"""The input journal is nullable for existing tasks and bootstrap-safe."""

import importlib

import sqlalchemy as sa
from alembic.migration import MigrationContext
from alembic.operations import Operations

from tests.web.services.task_database_shared import engine as engine_fixture

engine = engine_fixture


def test_pending_input_upgrade_preserves_old_tasks(engine):
    migration = importlib.import_module(
        "xagent.migrations.versions.20260924_pending_injection"
    )
    metadata = sa.MetaData()
    table = sa.Table("tasks", metadata, sa.Column("id", sa.Integer, primary_key=True))
    metadata.create_all(engine)
    with (
        engine.begin() as connection,
        Operations.context(MigrationContext.configure(connection)),
    ):
        connection.execute(table.insert().values(id=7))
        migration.upgrade()
        migration.upgrade()
        assert connection.execute(
            sa.text("SELECT id, pending_injection FROM tasks")
        ).all() == [(7, None)]
        assert "ix_tasks_pending_injection" in {
            i["name"] for i in sa.inspect(connection).get_indexes("tasks")
        }
        migration.downgrade()
        assert connection.execute(table.select()).all() == [(7,)]


def test_pending_input_migration_without_tasks_table(engine):
    migration = importlib.import_module(
        "xagent.migrations.versions.20260924_pending_injection"
    )
    with (
        engine.begin() as connection,
        Operations.context(MigrationContext.configure(connection)),
    ):
        migration.upgrade()
        migration.downgrade()
        assert not sa.inspect(connection).has_table("tasks")
