"""Observe an isolated three-arm Milvus CI experiment; preserve whole modules."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import pytest

GROUP_A = frozenset(
    {
        "tests/core/tools/core/RAG_tools/kb/test_collection_handle_contract.py",
        "tests/core/tools/core/RAG_tools/kb/test_milvus_commit.py",
        "tests/core/tools/core/RAG_tools/kb/test_milvus_cleanup.py",
        "tests/core/tools/core/RAG_tools/kb/test_milvus_storage.py",
    }
)
STATE = pytest.StashKey[dict[str, Any]]()


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption("--milvus-benchmark-arm", choices=("serial", "a", "b"))
    parser.addoption("--milvus-benchmark-output")


def pytest_configure(config: pytest.Config) -> None:
    if not config.getoption("--milvus-benchmark-arm") or not config.getoption(
        "--milvus-benchmark-output"
    ):
        raise pytest.UsageError("The Milvus benchmark requires an arm and output path")
    config.stash[STATE] = {"reports": []}


def pytest_sessionstart(session: pytest.Session) -> None:
    session.config.stash[STATE]["started"] = time.perf_counter()


@pytest.hookimpl(wrapper=True, tryfirst=True)
def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]):
    # Observe the original selection after marker and project collection hooks.
    result = yield
    arm = config.getoption("--milvus-benchmark-arm")
    selected, omitted = [], []
    for item in items:
        in_a = item.nodeid.split("::", 1)[0] in GROUP_A
        belongs = arm == "serial" or (in_a if arm == "a" else not in_a)
        (selected if belongs else omitted).append(item)
    state = config.stash[STATE]
    state["all_nodes"] = [item.nodeid for item in items]
    state["selected_nodes"] = [item.nodeid for item in selected]
    items[:] = selected
    config.hook.pytest_deselected(items=omitted)
    return result


@pytest.hookimpl(wrapper=True, tryfirst=True)
def pytest_runtest_makereport(item: pytest.Item, call: pytest.CallInfo):
    report = yield
    reason = getattr(report, "wasxfail", "")
    if report.skipped and not reason:
        reason = (
            report.longrepr[2]
            if isinstance(report.longrepr, tuple)
            else str(report.longrepr)
        )
    item.config.stash[STATE]["reports"].append(
        {
            "nodeid": report.nodeid,
            "phase": report.when,
            "outcome": report.outcome,
            "reason": reason,
            "duration": report.duration,
        }
    )
    return report


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    config = session.config
    state = config.stash[STATE]
    state["pytest_session_seconds"] = time.perf_counter() - state.pop("started")
    state["exitstatus"] = int(exitstatus)
    state["commit"] = os.environ.get("GITHUB_SHA")
    arm = config.getoption("--milvus-benchmark-arm")
    state["arm"] = arm
    output = Path(config.getoption("--milvus-benchmark-output"))
    output.mkdir(parents=True, exist_ok=True)
    (output / f"{arm}-results.json").write_text(json.dumps(state, indent=2))
