"""Operator signals for V2 execution-event writes and reads.

Each write helper emits two kinds of signal for one occurrence: a single structured
log line that locates the task (and the tool attempt where relevant), and a
metric with bounded attributes for rates. Neither carries payload content or
exception text; exception messages can embed user data, so only type names,
identifiers, SQLSTATEs and numbers are reported. Task ids appear in logs only,
never as metric attributes.

Reads of the event log are measured by ``measure_execution_event_read``: the
cost of each of the five operations (model context, display, reconnect replay,
checkpoint recovery, tool outcome) grows with history. Optional budgets
(``XAGENT_EXECUTION_EVENT_READ_*_BUDGETS*``) only report; they never refuse or
truncate a read.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from collections.abc import Callable, Generator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass

from ...config import (
    EXECUTION_EVENT_READ_DEFAULT_BUDGET_KEY,
    EXECUTION_EVENT_READ_DURATION_BUDGETS_MS,
    EXECUTION_EVENT_READ_ROW_BUDGETS,
    get_execution_event_read_duration_budgets_ms,
    get_execution_event_read_row_budgets,
)
from ...core.agent.checkpoint import ExecutionEventPersistenceError
from ...core.runtime_performance import increment_counter, observe_value
from .db_write_retry import cause_links, database_sqlstate

logger = logging.getLogger(__name__)

COMMIT_FAILURE_COUNTER = "xagent.execution_event.commit.failures"
READ_DURATION_METRIC = "xagent.execution_event.read.duration"
READ_ROWS_METRIC = "xagent.execution_event.read.rows"
READ_BUDGET_EXCEEDED_COUNTER = "xagent.execution_event.read.budget_exceeded"


def _failure_type(error: BaseException) -> str:
    """Class name of the first cause that is not the persistence wrapper."""
    for link in cause_links(error):
        if not isinstance(link, ExecutionEventPersistenceError):
            return type(link).__name__
    return type(error).__name__


def report_commit_failure(
    site: str,
    error: BaseException,
    *,
    task_id: int,
    run_id: str | None,
    retried: bool,
    tool_attempt_id: str | None = None,
) -> None:
    """Report one failed V2 execution-event commit, after retries are over.

    Call it once per failed write: after retries are exhausted or for a
    failure that is not retryable, never per attempt. Never raises, so a
    failure to report cannot mask the original error.
    """
    try:
        error_type = _failure_type(error)
        increment_counter(
            COMMIT_FAILURE_COUNTER,
            attributes={"site": site, "error.type": error_type, "storage": "v2"},
        )
        message = (
            "execution_event_commit_failed task_id=%s run_id=%s storage=v2 "
            "component=execution-events site=%s error_type=%s sqlstate=%s "
            "retried=%s"
        )
        args: list[object] = [
            task_id,
            run_id or "-",
            site,
            error_type,
            database_sqlstate(error) or "-",
            retried,
        ]
        if tool_attempt_id:
            message += " tool_attempt_id=%s"
            args.append(tool_attempt_id)
        logger.error(message, *args)
    except Exception:
        # Observability must never change the behavior of the measured path.
        return


@dataclass(slots=True)
class ExecutionEventReadMeasurement:
    """Row count and outcome of one read, set by the reader as it learns them."""

    rows: int | None = None
    outcome: str | None = None


# Set while a measured read is running in this context (thread or task), so a
# measured reader called from another measured reader is counted once, as the
# outer operation. Context-local, so concurrent reads never suppress each other.
_READ_MEASUREMENT_ACTIVE: ContextVar[bool] = ContextVar(
    "execution_event_read_measurement_active", default=False
)
_reported_invalid_budget_configs: set[tuple[str, str | None]] = set()
_reported_invalid_budget_configs_lock = threading.Lock()


def _read_budget(
    env_var: str,
    getter: Callable[[], Mapping[str, int]],
    operation: str,
) -> int | None:
    """The operation's budget, or None; invalid config is warned once and ignored."""
    try:
        budgets = getter()
    except ValueError as error:
        # Warn once per distinct invalid value, not per read: reads are hot and
        # the getter re-raises on every call (failures are never cached).
        # Changing the value to another invalid one warns again.
        marker = (env_var, os.getenv(env_var))
        with _reported_invalid_budget_configs_lock:
            first = marker not in _reported_invalid_budget_configs
            _reported_invalid_budget_configs.add(marker)
        if first:
            logger.warning(
                "execution_event_read_budget_config_invalid component=execution-events "
                "env=%s error=%s; reads are measured without this budget",
                env_var,
                error,
            )
        return None
    budget = budgets.get(operation)
    return (
        budget
        if budget is not None
        else budgets.get(EXECUTION_EVENT_READ_DEFAULT_BUDGET_KEY)
    )


def _record_execution_event_read(
    operation: str,
    task_id: int,
    measurement: ExecutionEventReadMeasurement,
    duration_ms: float,
) -> None:
    attributes = {"operation": operation, "storage": "v2"}
    if measurement.outcome:
        attributes["outcome"] = measurement.outcome
    observe_value(READ_DURATION_METRIC, duration_ms, unit="ms", attributes=attributes)
    rows = measurement.rows
    if rows is not None:
        observe_value(READ_ROWS_METRIC, rows, unit="{row}", attributes=attributes)

    row_budget = _read_budget(
        EXECUTION_EVENT_READ_ROW_BUDGETS,
        get_execution_event_read_row_budgets,
        operation,
    )
    duration_budget = _read_budget(
        EXECUTION_EVENT_READ_DURATION_BUDGETS_MS,
        get_execution_event_read_duration_budgets_ms,
        operation,
    )
    exceeded = []
    # A cache-hit reconnect replays a stored snapshot and never reads the event
    # log, so the budgets (which guard event-log reads) do not apply; its
    # histograms above are still recorded.
    if measurement.outcome != "cache_hit":
        if rows is not None and row_budget is not None and rows > row_budget:
            exceeded.append("rows")
        if duration_budget is not None and duration_ms > duration_budget:
            exceeded.append("duration")
    for kind in exceeded:
        increment_counter(
            READ_BUDGET_EXCEEDED_COUNTER,
            attributes={"operation": operation, "outcome": kind, "storage": "v2"},
        )
    if exceeded:
        # One line per exceeded read. Identifiers and numbers only: never
        # payload content or exception text.
        logger.warning(
            "execution_event_read_budget_exceeded task_id=%s storage=v2 "
            "component=execution-events operation=%s rows=%s duration_ms=%.1f "
            "budget_rows=%s budget_ms=%s",
            task_id,
            operation,
            "-" if rows is None else rows,
            duration_ms,
            "-" if row_budget is None else row_budget,
            "-" if duration_budget is None else duration_budget,
        )


@contextmanager
def measure_execution_event_read(
    operation: str,
    *,
    task_id: int,
) -> Generator[ExecutionEventReadMeasurement, None, None]:
    """Measure one whole V2 execution-event read (every page, once per call).

    ``rows`` is what the operation read or produced, set by the reader from a
    list it already built:

    - ``model``: model events loaded.
    - ``display``: display facts loaded.
    - ``reconnect``: frames replayed to the client (the info frame, the
      converted events, and the completion and status frames). Its duration
      includes cache I/O and projection work, not just database time.
    - ``checkpoint``: ``recovery_state`` rows scanned. Pending-tool outcome
      reads nested inside it are not counted in rows, but their time is in
      the duration.
    - ``tool_outcome``: rows read.

    A reader may also set ``measurement.outcome`` when the read has more than
    one shape (reconnect: ``loaded`` or ``cache_hit``). A ``cache_hit`` replay
    never reads the event log, so it is recorded in the histograms but never
    compared with the budgets. On exit the duration and, if set, the row count
    are recorded and compared with the operation's optional budgets, which only
    report. If the read raises, the duration is still recorded and the error
    propagates unchanged; rows are recorded only if the reader had set them.
    Never raises itself, never changes what the reader returns, and works in
    any thread: no event loop is involved. Nested inside another measured read
    it records nothing, so the outer operation owns the whole cost.
    """
    measurement = ExecutionEventReadMeasurement()
    if _READ_MEASUREMENT_ACTIVE.get():
        yield measurement
        return
    token = _READ_MEASUREMENT_ACTIVE.set(True)
    started = time.perf_counter()
    try:
        yield measurement
    finally:
        duration_ms = (time.perf_counter() - started) * 1_000.0
        _READ_MEASUREMENT_ACTIVE.reset(token)
        try:
            _record_execution_event_read(operation, task_id, measurement, duration_ms)
        except Exception:
            # Observability must never change the behavior of the measured path.
            pass
