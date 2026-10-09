"""Pure auto-resume policy: how often, how long and whether to retry a task.

Nothing here touches the database except :func:`scheduled_trigger_superseded`
(read-only) and nothing changes behavior until an executor calls it. The
executor calls :func:`plan_auto_resume` twice: once after recording an
interruption (to decide ``scheduled`` / ``exhausted`` / ``expired`` and the
first ``next_attempt_at``) and again at dispatch time, because the clock, the
counters and the env limits may have moved while the row waited. Config is
read at call time so an env change applies without a restart.

All datetimes are timezone-aware UTC; a naive one is a programming error.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum

from sqlalchemy.orm import Session

from ...config import (
    get_task_auto_resume_channel_window_seconds,
    get_task_auto_resume_llm_initial_backoff_seconds,
    get_task_auto_resume_llm_max_backoff_seconds,
    get_task_auto_resume_llm_max_elapsed_seconds,
    get_task_auto_resume_max_total_per_run,
    get_task_auto_resume_model_output_max_no_progress,
    get_task_auto_resume_short_initial_backoff_seconds,
    get_task_auto_resume_short_max_no_progress,
    get_task_auto_resume_window_seconds,
)
from ...core.agent.interruption import InterruptionReason
from ..models.task import Task
from ..models.trigger import TriggerRun, TriggerType

# Backoff ceiling for the short-backoff reasons (lease expiry, persistence).
_SHORT_MAX_BACKOFF_SECONDS = 60.0
_SHORT_JITTER_CAP_SECONDS = 30.0
_LLM_JITTER_CAP_SECONDS = 30.0
_MODEL_OUTPUT_JITTER_CAP_SECONDS = 5.0

# ``2 ** attempt`` is bounded past this point: the cap always wins long before.
_MAX_BACKOFF_EXPONENT = 62


@dataclass(frozen=True)
class AutoResumePolicy:
    """Retry limits for one interruption reason."""

    initial_backoff_seconds: float
    # Backoff doubles per no-progress resume up to this ceiling.
    max_backoff_seconds: float
    max_no_progress: int | None
    # Measured from the episode start, not from the latest interruption.
    max_elapsed_seconds: float | None
    jitter_cap_seconds: float


def auto_resume_policy(reason: InterruptionReason) -> AutoResumePolicy | None:
    """The policy for ``reason``; ``None`` means it is never auto-resumed."""

    if reason in (
        InterruptionReason.LEASE_EXPIRED,
        InterruptionReason.PERSISTENCE_FAILURE,
    ):
        return AutoResumePolicy(
            initial_backoff_seconds=get_task_auto_resume_short_initial_backoff_seconds(),
            max_backoff_seconds=_SHORT_MAX_BACKOFF_SECONDS,
            max_no_progress=get_task_auto_resume_short_max_no_progress(),
            max_elapsed_seconds=None,
            jitter_cap_seconds=_SHORT_JITTER_CAP_SECONDS,
        )
    if reason is InterruptionReason.LLM_UNAVAILABLE:
        return AutoResumePolicy(
            initial_backoff_seconds=get_task_auto_resume_llm_initial_backoff_seconds(),
            max_backoff_seconds=get_task_auto_resume_llm_max_backoff_seconds(),
            max_no_progress=None,
            max_elapsed_seconds=get_task_auto_resume_llm_max_elapsed_seconds(),
            jitter_cap_seconds=_LLM_JITTER_CAP_SECONDS,
        )
    if reason is InterruptionReason.MODEL_OUTPUT_INVALID:
        # Retried immediately (a fresh sample usually fixes it), so a short
        # run-level limit stands in for backoff.
        return AutoResumePolicy(
            initial_backoff_seconds=0,
            max_backoff_seconds=0,
            max_no_progress=get_task_auto_resume_model_output_max_no_progress(),
            max_elapsed_seconds=None,
            jitter_cap_seconds=_MODEL_OUTPUT_JITTER_CAP_SECONDS,
        )
    # ``shutdown`` is reserved for a future graceful-shutdown hand-off and is
    # not auto-resumed yet. user_pause, input_outcome_unknown,
    # unknown_tool_effect and not_recoverable need a human by definition.
    return None


def backoff_seconds(policy: AutoResumePolicy, attempt: int) -> float:
    """Doubling backoff, capped; ``attempt`` counts no-progress resumes so far."""

    exponent = min(max(attempt, 0), _MAX_BACKOFF_EXPONENT)
    return min(
        policy.initial_backoff_seconds * 2.0**exponent, policy.max_backoff_seconds
    )


def jitter_seconds(
    policy: AutoResumePolicy, backoff: float, rng: random.Random
) -> float:
    """Random spread so a burst of interruptions does not retry in lockstep.

    A zero-backoff policy is retried "immediately" and spreads over the whole
    jitter cap; otherwise jitter is at most half the (capped) backoff.
    """

    if policy.initial_backoff_seconds == 0:
        return rng.uniform(0, policy.jitter_cap_seconds)
    return rng.uniform(0, min(backoff, policy.jitter_cap_seconds) * 0.5)


def staleness_window_seconds(kind: str | None) -> int:
    """How long after an interruption a task of ``kind`` may still be resumed."""

    if kind == "channel":
        return get_task_auto_resume_channel_window_seconds()
    return get_task_auto_resume_window_seconds()


class AutoResumeOutcome(str, Enum):
    SCHEDULE = "schedule"
    EXHAUSTED = "exhausted"
    EXPIRED = "expired"
    NEVER = "never"


@dataclass(frozen=True)
class AutoResumeVerdict:
    outcome: AutoResumeOutcome
    # Set only for SCHEDULE.
    next_attempt_at: datetime | None = None


def _require_aware(name: str, value: datetime) -> None:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{name} must be timezone-aware")


def plan_auto_resume(
    *,
    reason: InterruptionReason,
    kind: str | None,
    interrupted_at: datetime,
    episode_started_at: datetime,
    no_progress_resumes: int,
    total_resumes: int,
    now: datetime,
    rng: random.Random,
) -> AutoResumeVerdict:
    """Decide whether and when to auto-resume; checks apply in order.

    1. no policy for ``reason`` -> NEVER;
    2. interrupted longer ago than the staleness window -> EXPIRED;
    3. total resume cap reached -> EXHAUSTED;
    4. no-progress limit reached -> EXHAUSTED;
    5. episode elapsed limit reached -> EXHAUSTED;
    6. otherwise SCHEDULE at ``now + backoff + jitter``.

    A scheduled attempt that would land past the staleness window or the
    elapsed limit is still scheduled: the dispatch-time call re-checks with
    the then-current clock and settles it as EXPIRED / EXHAUSTED.
    """

    _require_aware("interrupted_at", interrupted_at)
    _require_aware("episode_started_at", episode_started_at)
    _require_aware("now", now)

    policy = auto_resume_policy(reason)
    if policy is None:
        return AutoResumeVerdict(AutoResumeOutcome.NEVER)
    if now - interrupted_at > timedelta(seconds=staleness_window_seconds(kind)):
        return AutoResumeVerdict(AutoResumeOutcome.EXPIRED)
    if total_resumes >= get_task_auto_resume_max_total_per_run():
        return AutoResumeVerdict(AutoResumeOutcome.EXHAUSTED)
    if (
        policy.max_no_progress is not None
        and no_progress_resumes >= policy.max_no_progress
    ):
        return AutoResumeVerdict(AutoResumeOutcome.EXHAUSTED)
    if policy.max_elapsed_seconds is not None and now - episode_started_at >= timedelta(
        seconds=policy.max_elapsed_seconds
    ):
        return AutoResumeVerdict(AutoResumeOutcome.EXHAUSTED)

    backoff = backoff_seconds(policy, no_progress_resumes)
    delay = backoff + jitter_seconds(policy, backoff, rng)
    return AutoResumeVerdict(
        AutoResumeOutcome.SCHEDULE, next_attempt_at=now + timedelta(seconds=delay)
    )


def scheduled_trigger_superseded(db: Session, task: Task) -> bool:
    """Whether a newer run of the task's scheduled trigger has already started.

    A scheduled run is the latest tick of a recurring job; resuming an old
    tick once the next one is running would duplicate its work. Webhook and
    gmail runs each carry a distinct event, so they are never superseded.
    ``TriggerRun.task_id`` is the link; a missing run row is not superseded.
    """

    if task.source != "trigger":
        return False
    agent_config = task.agent_config
    if not isinstance(agent_config, dict):
        return False
    if agent_config.get("trigger_type") != TriggerType.SCHEDULED.value:
        return False

    run = (
        db.query(TriggerRun)
        .filter(TriggerRun.task_id == task.id)
        .order_by(TriggerRun.id.desc())
        .first()
    )
    if run is None:
        return False
    later = (
        db.query(TriggerRun.id)
        .filter(
            TriggerRun.trigger_id == run.trigger_id,
            TriggerRun.id > run.id,
            TriggerRun.started_at.isnot(None),
        )
        .first()
    )
    return later is not None
