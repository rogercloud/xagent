"""Regression test for the alembic ``env.py`` logging configuration.

``env.py`` calls ``logging.config.fileConfig(config.config_file_name)`` to
apply ``alembic.ini``'s ``[loggers]`` section. ``fileConfig`` defaults
``disable_existing_loggers`` to ``True``, which disables *every* logger
already registered in ``logging.Logger.manager.loggerDict`` that is not one
of the names explicitly configured in ``alembic.ini`` (``root``,
``sqlalchemy``, ``alembic``).

Because migrations run in-process during normal application startup (see
``xagent.web.models.database._initialize_database_schema`` ->
``xagent.db.migration.try_upgrade_db`` -> ``command.upgrade`` -> this
``env.py``), that default would silently and permanently disable
application loggers such as everything under ``xagent.*`` for the lifetime
of the process, contradicting the project's own logging setup
(``xagent.web.logging_config.setup_logging`` explicitly passes
``disable_existing_loggers=False``).

This test drives a real Alembic upgrade (exercising the actual ``env.py``)
against a throwaway SQLite database and asserts that a logger which existed
before the upgrade is not disabled afterward.
"""

from __future__ import annotations

import logging
import os
import tempfile
from collections.abc import Generator
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config

project_root = Path(__file__).parent.parent.parent


@pytest.fixture
def preexisting_logger() -> Generator[logging.Logger, None, None]:
    """A logger that exists before the migration runs, restored afterward."""
    name = "xagent.test.env_logging_config.preexisting"
    logger = logging.getLogger(name)
    original_disabled = logger.disabled
    try:
        yield logger
    finally:
        logger.disabled = original_disabled


@pytest.fixture
def sqlite_alembic_cfg() -> Generator[Config, None, None]:
    """An Alembic config pointed at a throwaway SQLite database."""
    old_database_url = os.environ.get("DATABASE_URL")
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    db_url = f"sqlite:///{path}"
    os.environ["DATABASE_URL"] = db_url

    cfg = Config(str(project_root / "alembic.ini"))
    cfg.set_main_option("sqlalchemy.url", db_url)

    try:
        yield cfg
    finally:
        if old_database_url is None:
            os.environ.pop("DATABASE_URL", None)
        else:
            os.environ["DATABASE_URL"] = old_database_url
        os.unlink(path)


@pytest.mark.integration
def test_alembic_env_does_not_disable_preexisting_loggers(
    preexisting_logger: logging.Logger,
    sqlite_alembic_cfg: Config,
) -> None:
    """Running the real migration env.py must not disable existing loggers.

    ``preexisting_logger`` stands in for any ``xagent.*`` logger created by
    normal module imports before migrations run during app startup. Alembic
    only explicitly configures ``root``, ``sqlalchemy``, and ``alembic`` in
    ``alembic.ini``; every other pre-existing logger must be left alone.
    """
    # Base tables are created by SQLAlchemy in production before migrations
    # run; mirror that so `command.upgrade` exercises the same path as
    # `tests/migrations/test_migration_integration.py`.
    from sqlalchemy import create_engine

    from xagent.web.models.database import Base

    sqlalchemy_url = sqlite_alembic_cfg.get_main_option("sqlalchemy.url")
    assert sqlalchemy_url is not None
    engine = create_engine(sqlalchemy_url)
    Base.metadata.create_all(bind=engine)
    engine.dispose()

    assert not preexisting_logger.disabled, "sanity check before the upgrade"

    command.upgrade(sqlite_alembic_cfg, "head")

    assert not preexisting_logger.disabled, (
        "alembic's env.py disabled a pre-existing logger; fileConfig() must "
        "be called with disable_existing_loggers=False"
    )
