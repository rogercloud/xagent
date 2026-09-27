"""Partial index on task_chat_messages for the R6 lease-recovery sweep.

The sweep (task_lease_recovery.py) filters on delivery_status = 'pending' on
every tick; this migration adds a partial index scoped to that predicate so
the query stays cheap regardless of table size. Both dialects the app
actually runs on are exercised: the partial-index syntax and the query
planner's willingness to use it differ enough between them that a SQLite-only
check would not catch a PostgreSQL-side regression (and vice versa).
"""

from __future__ import annotations

from pathlib import Path

import pytest
import sqlalchemy as sa
from alembic import command
from alembic.migration import MigrationContext
from alembic.operations import Operations
from sqlalchemy import create_engine, inspect, text

from tests.shared.postgres_disposable import (
    disposable_database_factory,
    load_migration_module,
)
from xagent.db.config import create_alembic_config

REVISION = "20260927_pending_delivery_index"
DOWN_REVISION = "20260916_durable_create_operations"
TABLE = "task_chat_messages"
INDEX = "ix_task_chat_messages_pending_delivery"
MIGRATION_PATH = (
    Path(__file__).parent.parent.parent
    / "src/xagent/migrations/versions/20260927_pending_delivery_index.py"
)


def _create_legacy_table(connection, *, timestamp_type: str) -> None:
    connection.execute(
        text(
            f"CREATE TABLE {TABLE} ("
            "id INTEGER PRIMARY KEY, task_id INTEGER NOT NULL, "
            "role VARCHAR(32) NOT NULL, turn_id VARCHAR(64), "
            "delivery_status VARCHAR(32), "
            f"created_at {timestamp_type})"
        )
    )


@pytest.fixture
def postgresql_engine_factory():
    with disposable_database_factory("xagent_pending_delivery_index") as make:
        yield make


def test_sqlite_upgrade_adds_partial_index_and_downgrade_removes_it() -> None:
    engine = create_engine("sqlite:///:memory:")
    config = create_alembic_config(engine)

    with engine.begin() as connection:
        connection.execute(
            text("CREATE TABLE alembic_version (version_num VARCHAR(255) NOT NULL)")
        )
        connection.execute(
            text("INSERT INTO alembic_version (version_num) VALUES (:revision)"),
            {"revision": DOWN_REVISION},
        )
        _create_legacy_table(connection, timestamp_type="DATETIME")
        connection.execute(
            text(
                "INSERT INTO task_chat_messages "
                "(id, task_id, role, turn_id, delivery_status, created_at) VALUES "
                "(1, 7, 'user', 't1', 'pending', '2026-09-01 00:00:00'), "
                "(2, 7, 'assistant', 't2', NULL, '2026-09-02 00:00:00'), "
                "(3, 8, 'user', 't3', 'dispatched', '2026-09-03 00:00:00')"
            )
        )
        config.attributes["connection"] = connection

        command.upgrade(config, REVISION)

        indexes = {
            index["name"]: index for index in inspect(connection).get_indexes(TABLE)
        }
        assert INDEX in indexes
        assert indexes[INDEX]["column_names"] == ["task_id", "created_at"]

        # The predicate must actually be used by the sweep's own shape of
        # query: filter on delivery_status = 'pending' plus a task_id/
        # created_at predicate, exactly as _orphaned_pending_delivery_predicates
        # composes it.
        plan = connection.execute(
            text(
                "EXPLAIN QUERY PLAN SELECT * FROM task_chat_messages "
                "WHERE delivery_status = 'pending' AND task_id = 7"
            )
        ).all()
        assert any(INDEX in str(row) for row in plan), plan

        command.downgrade(config, DOWN_REVISION)
        assert INDEX not in {
            index["name"] for index in inspect(connection).get_indexes(TABLE)
        }

        # Repeatable: re-running upgrade/downgrade must not error.
        command.upgrade(config, REVISION)
        assert INDEX in {
            index["name"] for index in inspect(connection).get_indexes(TABLE)
        }
        command.downgrade(config, DOWN_REVISION)


def test_sqlite_upgrade_skips_without_the_table() -> None:
    engine = create_engine("sqlite:///:memory:")
    config = create_alembic_config(engine)

    with engine.begin() as connection:
        connection.execute(
            text("CREATE TABLE alembic_version (version_num VARCHAR(255) NOT NULL)")
        )
        connection.execute(
            text("INSERT INTO alembic_version (version_num) VALUES (:revision)"),
            {"revision": DOWN_REVISION},
        )
        config.attributes["connection"] = connection

        command.upgrade(config, REVISION)

        assert TABLE not in inspect(connection).get_table_names()
        assert (
            connection.execute(
                text("SELECT version_num FROM alembic_version")
            ).scalar_one()
            == REVISION
        )


@pytest.mark.postgresql
def test_postgresql_upgrade_adds_partial_index_used_by_the_sweep_query(
    postgresql_engine_factory,
) -> None:
    """Drive the revision module directly (not ``command.upgrade``).

    ``env.py`` commits the connection's transaction itself before configuring
    the migration context on PostgreSQL (see ``transaction_per_migration``),
    which conflicts with holding the same connection inside an outer
    ``engine.begin()`` block the way ``command.upgrade(config, ...)`` would
    need here. Driving ``migration.upgrade()``/``downgrade()`` through
    ``Operations.context`` directly -- the pattern
    tests/alembic/test_20260809_add_task_interaction_requests.py's PostgreSQL
    tests use -- keeps this test's own transaction in charge instead.
    """
    migration = load_migration_module(MIGRATION_PATH, "pending_delivery_index_pg")
    engine = postgresql_engine_factory("upgrade")

    with engine.begin() as connection:
        _create_legacy_table(connection, timestamp_type="TIMESTAMP WITH TIME ZONE")
        connection.execute(
            text(
                "INSERT INTO task_chat_messages "
                "(id, task_id, role, turn_id, delivery_status, created_at) "
                "SELECT g, 7, 'user', 't' || g, "
                "CASE WHEN g % 50 = 0 THEN 'pending' ELSE 'dispatched' END, "
                "TIMESTAMP '2026-09-01 00:00:00' + (g || ' seconds')::interval "
                "FROM generate_series(1, 4000) AS g"
            )
        )
        connection.execute(sa.text("ANALYZE task_chat_messages"))

        context = MigrationContext.configure(connection)
        with Operations.context(context):
            migration.upgrade()

        indexes = {
            index["name"]: index for index in inspect(connection).get_indexes(TABLE)
        }
        assert INDEX in indexes
        assert indexes[INDEX]["column_names"] == ["task_id", "created_at"]

        plan = "\n".join(
            row[0]
            for row in connection.execute(
                text(
                    "EXPLAIN (FORMAT TEXT) SELECT * FROM task_chat_messages "
                    "WHERE delivery_status = 'pending' AND task_id = 7"
                )
            ).all()
        )
        assert INDEX in plan, plan

        with Operations.context(context):
            migration.downgrade()
        assert INDEX not in {
            index["name"] for index in inspect(connection).get_indexes(TABLE)
        }
