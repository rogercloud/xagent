"""Partial index on task_chat_messages for the R6 lease-recovery sweep.

The sweep (task_lease_recovery.py) filters on delivery_status = 'pending' on
every tick; this migration adds a partial index scoped to that predicate so
the query stays cheap regardless of table size. Both dialects the app
actually runs on are exercised: the partial-index syntax and the query
planner's willingness to use it differ enough between them that a SQLite-only
check would not catch a PostgreSQL-side regression (and vice versa).
"""

from __future__ import annotations

from contextlib import nullcontext
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
from xagent.web.models.chat_message import TaskChatMessage

REVISION = "20260927_pending_delivery_index"
DOWN_REVISION = "20260916_durable_create_operations"
TABLE = "task_chat_messages"
INDEX = "ix_task_chat_messages_pending_delivery"
INDEX_COLUMNS = ("task_id", "created_at")
MIGRATION_PATH = (
    Path(__file__).parent.parent.parent
    / "src/xagent/migrations/versions/20260927_pending_delivery_index.py"
)


def _create_legacy_table(
    connection, *, timestamp_type: str, with_delivery_status: bool = True
) -> None:
    delivery_column = "delivery_status VARCHAR(32), " if with_delivery_status else ""
    connection.execute(
        text(
            f"CREATE TABLE {TABLE} ("
            "id INTEGER PRIMARY KEY, task_id INTEGER NOT NULL, "
            "role VARCHAR(32) NOT NULL, turn_id VARCHAR(64), "
            f"{delivery_column}"
            f"created_at {timestamp_type})"
        )
    )


def _create_competing_indexes(connection) -> None:
    """The two indexes that already exist on this table in production.

    A plain index on ``task_id`` alone (``task_id = Column(..., index=True)``)
    and the unique ``(task_id, role, turn_id)`` index both cover ``task_id``
    equality, so a test that only ever creates the partial index in isolation
    cannot show the planner actually preferring it over an index it would
    also see in production.
    """

    connection.execute(
        text(f"CREATE INDEX ix_task_chat_messages_task_id ON {TABLE} (task_id)")
    )
    connection.execute(
        text(
            "CREATE UNIQUE INDEX uq_task_chat_messages_task_role_turn_id "
            f"ON {TABLE} (task_id, role, turn_id)"
        )
    )


def _seed_rows(connection, *, dialect: str) -> None:
    """Rows spread across several task_ids, mixing pending/settled delivery.

    A single task_id (as the previous version of this test used) lets a
    planner "use the index" trivially because it is the only index that
    mentions any predicate column at all; spreading rows across many
    task_ids and mixing delivery_status values is what actually exercises
    whether the partial index is preferred over the competing ones above for
    the sweep's real access pattern.
    """

    if dialect == "postgresql":
        connection.execute(
            text(
                "INSERT INTO task_chat_messages "
                "(id, task_id, role, turn_id, delivery_status, created_at) "
                "SELECT g, (g % 20) + 1, 'user', 't' || g, "
                "CASE WHEN g % 7 = 0 THEN 'pending' ELSE 'dispatched' END, "
                "TIMESTAMP '2026-09-01 00:00:00' + (g || ' seconds')::interval "
                "FROM generate_series(1, 4000) AS g"
            )
        )
        connection.execute(sa.text("ANALYZE task_chat_messages"))
    else:
        rows = []
        for g in range(1, 201):
            task_id = (g % 20) + 1
            delivery_status = "pending" if g % 7 == 0 else "dispatched"
            rows.append(
                f"({g}, {task_id}, 'user', 't{g}', '{delivery_status}', "
                f"'2026-09-01 00:00:{g % 60:02d}')"
            )
        connection.execute(
            text(
                "INSERT INTO task_chat_messages "
                "(id, task_id, role, turn_id, delivery_status, created_at) VALUES "
                + ", ".join(rows)
            )
        )


# The sweep issues two shapes of query against this predicate
# (task_lease_recovery.py):
#
# * ``reconcile_orphaned_pending_deliveries_no_commit`` closes one task's
#   rows with an equality lookup: ``delivery_status = 'pending' AND
#   task_id = :id`` (plus the role/turn_id/EXISTS predicates below).
# * ``reconcile_orphaned_pending_deliveries_isolated`` pages task_ids with
#   ``SELECT DISTINCT task_id ... WHERE role = 'user' AND
#   delivery_status = 'pending' AND turn_id IS NOT NULL AND created_at <
#   :cutoff ORDER BY task_id LIMIT :n`` (see
#   ``_orphaned_pending_delivery_predicates`` composed in that function).
#
# Both are reproduced below. Neither reproduces the three correlated
# EXISTS/NOT EXISTS subqueries against ``tasks``/``task_execution_commands``
# that ``_orphaned_pending_delivery_predicates`` also adds: those constrain
# which rows qualify but are evaluated per-row against the outer scan, so
# they do not change which index the planner picks for scanning
# ``task_chat_messages`` itself. This asserts the partial index is used for
# that outer scan, not that the two-table joins are separately optimal.
EQUALITY_LOOKUP_SQL = (
    "SELECT * FROM task_chat_messages WHERE delivery_status = 'pending' AND task_id = 7"
)
TICK_SCAN_SQL = (
    "SELECT DISTINCT task_id FROM task_chat_messages "
    "WHERE role = 'user' AND delivery_status = 'pending' "
    "AND turn_id IS NOT NULL AND created_at < :cutoff "
    "ORDER BY task_id LIMIT :batch"
)


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
        _create_competing_indexes(connection)
        _seed_rows(connection, dialect="sqlite")
        config.attributes["connection"] = connection

        command.upgrade(config, REVISION)

        indexes = {
            index["name"]: index for index in inspect(connection).get_indexes(TABLE)
        }
        assert INDEX in indexes
        assert indexes[INDEX]["column_names"] == ["task_id", "created_at"]

        # The equality lookup used to close one task's rows.
        plan = connection.execute(
            text(f"EXPLAIN QUERY PLAN {EQUALITY_LOOKUP_SQL}")
        ).all()
        assert any(INDEX in str(row) for row in plan), plan

        # The tick scan used to page task_ids across the whole table.
        plan = connection.execute(
            text(f"EXPLAIN QUERY PLAN {TICK_SCAN_SQL}"),
            {"cutoff": "2026-12-01 00:00:00", "batch": 50},
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


def test_sqlite_upgrade_skips_without_delivery_status_column() -> None:
    """A legacy table that predates delivery_status must not error the chain.

    Mirrors the guard-clause convention this migration and
    20260710_add_chat_message_delivery_state.py share for the same table:
    the predicate/index columns are checked as a group before touching DDL.
    """

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
        _create_legacy_table(
            connection, timestamp_type="DATETIME", with_delivery_status=False
        )
        config.attributes["connection"] = connection

        command.upgrade(config, REVISION)

        assert INDEX not in {
            index["name"] for index in inspect(connection).get_indexes(TABLE)
        }


def test_sqlite_upgrade_is_a_noop_when_the_index_already_exists() -> None:
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
                f"CREATE INDEX {INDEX} ON {TABLE} (task_id, created_at) "
                "WHERE delivery_status = 'pending'"
            )
        )
        config.attributes["connection"] = connection

        command.upgrade(config, REVISION)

        indexes = {
            index["name"]: index for index in inspect(connection).get_indexes(TABLE)
        }
        assert INDEX in indexes
        assert indexes[INDEX]["column_names"] == ["task_id", "created_at"]


def test_migration_column_list_and_predicate_match_the_model() -> None:
    """Parity check: the migration must build the same index the ORM model
    declares in ``TaskChatMessage.__table_args__``, or a fresh database built
    via ``create_all`` (tests, or an Alembic-only-stamped install) and one
    built by running every migration end up with different indexes."""

    migration = load_migration_module(MIGRATION_PATH, "pending_delivery_index_parity")

    model_index = next(
        index for index in TaskChatMessage.__table__.indexes if index.name == INDEX
    )
    model_columns = tuple(column.name for column in model_index.columns)
    assert model_columns == migration.INDEX_COLUMNS == INDEX_COLUMNS

    model_predicate = model_index.dialect_options["postgresql"]["where"]
    assert str(model_predicate) == str(migration.PREDICATE)
    sqlite_predicate = model_index.dialect_options["sqlite"]["where"]
    assert str(sqlite_predicate) == str(migration.PREDICATE)


@pytest.fixture
def postgresql_engine_factory():
    with disposable_database_factory("xagent_pending_delivery_index") as make:
        yield make


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
    tests use -- keeps this test's own transaction in charge instead. This
    also exercises the real online path: ``upgrade()`` builds the index with
    ``CREATE INDEX CONCURRENTLY`` inside ``autocommit_block()`` because the
    bind's dialect is genuinely PostgreSQL here.

    ``autocommit_block()`` requires the ``MigrationContext`` to be tracking
    its own per-migration transaction (``self._transaction is not None``) --
    exactly what ``env.py``'s real online run gives it, via
    ``context.begin_transaction(_per_migration=True)`` inside
    ``run_migrations()`` (see ``transaction_per_migration`` in env.py). A
    bare ``Operations.context(context)`` with no such transaction leaves
    ``self._transaction`` at ``None``, so ``autocommit_block()`` asserts as
    soon as anything (our own inspection queries included) has caused
    SQLAlchemy's connection-level autobegin to kick in. Calling
    ``context.begin_transaction(_per_migration=True)`` around each
    ``migration.upgrade()``/``downgrade()`` call reproduces that real
    per-migration transaction so the assertion holds the same way it does
    under a real ``alembic upgrade``.
    """
    migration = load_migration_module(MIGRATION_PATH, "pending_delivery_index_pg")
    engine = postgresql_engine_factory("upgrade")

    with engine.connect() as connection:
        _create_legacy_table(connection, timestamp_type="TIMESTAMP WITH TIME ZONE")
        _create_competing_indexes(connection)
        _seed_rows(connection, dialect="postgresql")
        connection.commit()

        context = MigrationContext.configure(
            connection, opts={"transaction_per_migration": True}
        )
        with Operations.context(context):
            with context.begin_transaction(_per_migration=True):
                migration.upgrade()

        indexes = {
            index["name"]: index for index in inspect(connection).get_indexes(TABLE)
        }
        assert INDEX in indexes
        assert indexes[INDEX]["column_names"] == ["task_id", "created_at"]

        plan = "\n".join(
            row[0]
            for row in connection.execute(
                text(f"EXPLAIN (FORMAT TEXT) {EQUALITY_LOOKUP_SQL}")
            ).all()
        )
        assert INDEX in plan, plan

        plan = "\n".join(
            row[0]
            for row in connection.execute(
                text(f"EXPLAIN (FORMAT TEXT) {TICK_SCAN_SQL}"),
                {"cutoff": "2026-12-01 00:00:00+00", "batch": 50},
            ).all()
        )
        assert INDEX in plan, plan
        connection.commit()

        # Idempotent: re-running upgrade() against an index that already
        # exists and validates must not raise or attempt another build.
        with Operations.context(context):
            with context.begin_transaction(_per_migration=True):
                migration.upgrade()
        indexes = {index["name"] for index in inspect(connection).get_indexes(TABLE)}
        assert INDEX in indexes

        with Operations.context(context):
            with context.begin_transaction(_per_migration=True):
                migration.downgrade()
        assert INDEX not in {
            index["name"] for index in inspect(connection).get_indexes(TABLE)
        }


def test_postgresql_online_upgrade_retries_an_invalid_concurrent_index(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Mirrors
    test_20260725_add_task_lease_recovery_index.py::test_postgresql_online_upgrade_retries_an_invalid_concurrent_index:
    a ``CREATE INDEX CONCURRENTLY`` interrupted partway leaves an ``INVALID``
    index behind rather than rolling back, so a re-run must drop and rebuild
    it rather than treating its mere presence as "already done".
    """
    migration = load_migration_module(
        MIGRATION_PATH, "pending_delivery_index_invalid_retry"
    )
    context = MigrationContext.configure(dialect_name="postgresql")
    operations = Operations(context)
    calls: list[tuple[str, dict[str, object]]] = []

    monkeypatch.setattr(migration, "_inspector", lambda: object())
    monkeypatch.setattr(
        migration, "_columns", lambda _inspector: set(migration.REQUIRED_COLUMNS)
    )
    monkeypatch.setattr(migration, "_postgres_index_validity", lambda: False)
    monkeypatch.setattr(
        migration,
        "_index_columns",
        lambda _inspector, _name: migration.INDEX_COLUMNS,
    )
    monkeypatch.setattr(context, "autocommit_block", nullcontext)
    monkeypatch.setattr(
        operations,
        "drop_index",
        lambda *args, **kwargs: calls.append(("drop", kwargs)),
    )
    monkeypatch.setattr(
        operations,
        "create_index",
        lambda *args, **kwargs: calls.append(("create", kwargs)),
    )

    with Operations.context(context):
        monkeypatch.setattr(migration, "op", operations)
        migration.upgrade()

    assert calls == [
        (
            "drop",
            {
                "table_name": TABLE,
                "if_exists": True,
                "postgresql_concurrently": True,
            },
        ),
        (
            "create",
            {
                "if_not_exists": True,
                "postgresql_concurrently": True,
                "postgresql_where": migration.PREDICATE,
            },
        ),
    ]


def test_postgresql_online_upgrade_replaces_valid_wrong_index_definition(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    migration = load_migration_module(
        MIGRATION_PATH, "pending_delivery_index_wrong_columns"
    )
    context = MigrationContext.configure(dialect_name="postgresql")
    operations = Operations(context)
    calls: list[str] = []

    monkeypatch.setattr(migration, "_inspector", lambda: object())
    monkeypatch.setattr(
        migration, "_columns", lambda _inspector: set(migration.REQUIRED_COLUMNS)
    )
    monkeypatch.setattr(migration, "_postgres_index_validity", lambda: True)
    monkeypatch.setattr(
        migration, "_index_columns", lambda _inspector, _name: ("task_id",)
    )
    monkeypatch.setattr(context, "autocommit_block", nullcontext)
    monkeypatch.setattr(
        operations,
        "drop_index",
        lambda *_args, **_kwargs: calls.append("drop"),
    )
    monkeypatch.setattr(
        operations,
        "create_index",
        lambda *_args, **_kwargs: calls.append("create"),
    )

    with Operations.context(context):
        monkeypatch.setattr(migration, "op", operations)
        migration.upgrade()

    assert calls == ["drop", "create"]


def test_postgresql_offline_upgrade_and_downgrade_emit_concurrent_index_sql() -> None:
    migration = load_migration_module(
        MIGRATION_PATH, "pending_delivery_index_offline_sql"
    )
    from io import StringIO

    def _offline_sql(operation: str) -> str:
        output = StringIO()
        context = MigrationContext.configure(
            dialect_name="postgresql",
            opts={"as_sql": True, "output_buffer": output},
        )
        with Operations.context(context):
            getattr(migration, operation)()
        return output.getvalue()

    upgrade_sql = _offline_sql("upgrade")
    downgrade_sql = _offline_sql("downgrade")

    assert f"CREATE INDEX CONCURRENTLY {INDEX} ON {TABLE}" in upgrade_sql
    assert "WHERE delivery_status = 'pending'" in upgrade_sql
    assert f"DROP INDEX CONCURRENTLY {INDEX}" in downgrade_sql


def test_offline_sql_rendering_does_not_inspect_the_bind() -> None:
    """``alembic upgrade --sql`` uses a ``MockConnection`` that cannot be
    inspected; the offline branch must not call ``_columns()``/``_indexes()``
    (which call ``sa.inspect(op.get_bind())``) at all."""
    migration = load_migration_module(
        MIGRATION_PATH, "pending_delivery_index_offline_no_inspect"
    )
    from io import StringIO

    def _boom(*_args, **_kwargs):
        raise AssertionError("offline rendering must not inspect the bind")

    context = MigrationContext.configure(
        dialect_name="sqlite",
        opts={"as_sql": True, "output_buffer": StringIO()},
    )
    original_inspector = migration._inspector
    migration._inspector = _boom
    try:
        with Operations.context(context):
            migration.upgrade()
    finally:
        migration._inspector = original_inspector
