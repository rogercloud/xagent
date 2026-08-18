"""Pin that the production migration path never runs ``fileConfig``.

This is a companion to ``test_env_logging_config.py``, not a substitute for
it. That file proves the *fix* is correct for the code path it actually
changes (a file-backed Alembic ``Config``, used by the standalone ``alembic
upgrade head`` CLI and by the test suite).

This file proves something different and narrower: that production startup
(``xagent.web.models.database`` -> ``xagent.db.migration.try_upgrade_db`` ->
``xagent.db.config.create_alembic_config``) never reaches the ``fileConfig``
branch in ``env.py`` at all, because ``create_alembic_config`` builds its
``Config`` in memory with no filename. Since ``config.config_file_name`` is
``None`` there, the ``if config.config_file_name is not None:`` guard skips
``fileConfig`` entirely -- with or without this PR's
``disable_existing_loggers=False`` fix.

In other words: this test does NOT demonstrate that the fix changed
production behavior (it didn't -- production was never affected by the
bug). It exists so that if ``create_alembic_config`` is ever changed to
pass a filename, the process-wide logger-disabling regression this PR fixed
for the test suite cannot silently reappear in production without a test
failing here first.
"""

from __future__ import annotations

import logging
from collections.abc import Generator
from pathlib import Path

import pytest
from sqlalchemy import create_engine

from xagent.db.migration import try_upgrade_db


@pytest.fixture
def preexisting_logger() -> Generator[logging.Logger, None, None]:
    """A logger that exists before the migration runs, restored afterward."""
    name = "xagent.test.env_logging_config_production.preexisting"
    logger = logging.getLogger(name)
    original_disabled = logger.disabled
    try:
        yield logger
    finally:
        logger.disabled = original_disabled


@pytest.mark.integration
def test_try_upgrade_db_does_not_disable_preexisting_loggers(
    preexisting_logger: logging.Logger,
    tmp_path: Path,
) -> None:
    """``try_upgrade_db`` (the production startup path) must not touch loggers.

    ``create_alembic_config`` builds its ``Config`` with no filename, so
    ``env.py``'s ``fileConfig`` branch is skipped for this path regardless of
    the ``disable_existing_loggers`` argument. This test pins that fact: it
    is a guard against a future regression (e.g. someone adding a
    ``config_file_name`` to ``create_alembic_config``), not evidence that
    this PR's fix altered production behavior -- it did not, because
    production never executed the buggy line in the first place.
    """
    # An empty database exercises `command.stamp(alembic_cfg, "head")`,
    # `try_upgrade_db`'s path for a brand-new database. `env.py` runs either
    # way (stamp and upgrade both trigger it), so this covers the same
    # fileConfig-skipping guard as a populated database would.
    db_path = tmp_path / "try_upgrade_db_logging.db"
    db_url = f"sqlite:///{db_path}"
    engine = create_engine(db_url)

    assert not preexisting_logger.disabled, "sanity check before the upgrade"

    try_upgrade_db(engine)

    assert not preexisting_logger.disabled, (
        "the production migration path (try_upgrade_db) disabled a "
        "pre-existing logger; create_alembic_config() must not pass a "
        "config_file_name that would route env.py through fileConfig()"
    )

    engine.dispose()
