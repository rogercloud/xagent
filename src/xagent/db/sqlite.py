"""SQLite engine hardening for concurrent access.

Enables WAL journaling (concurrent readers alongside a single writer) and a
``busy_timeout`` (a writer waits for the lock instead of failing immediately
with "database is locked") on SQLite engines. This lets the shared engine
tolerate the concurrent writes that in-turn tool concurrency can produce.

No effect on non-SQLite engines (e.g. Postgres handles this with MVCC and
row-level locks).
"""

from __future__ import annotations

import logging

from sqlalchemy import Engine, event

logger = logging.getLogger(__name__)

# 5s gives concurrent writers room to wait out a short-lived write lock without
# unbounded blocking.
DEFAULT_BUSY_TIMEOUT_MS = 5000


def apply_sqlite_concurrency_pragmas(
    engine: Engine, *, busy_timeout_ms: int = DEFAULT_BUSY_TIMEOUT_MS
) -> None:
    """Register a connect hook that sets WAL + busy_timeout on SQLite engines.

    Args:
        engine: The SQLAlchemy engine to harden. Non-SQLite engines are ignored.
        busy_timeout_ms: How long (ms) a blocked writer waits for the lock.
    """
    if engine.dialect.name != "sqlite":
        return

    timeout_ms = max(0, int(busy_timeout_ms))

    @event.listens_for(engine, "connect")
    def _set_sqlite_pragmas(dbapi_connection, _connection_record):  # type: ignore[no-untyped-def]
        cursor = dbapi_connection.cursor()
        try:
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.execute(f"PRAGMA busy_timeout={timeout_ms}")
        finally:
            cursor.close()
