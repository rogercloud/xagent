"""Result holder for what a task settlement committed."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ...core.agent.interruption import InterruptionReason


@dataclass
class SettlementReport:
    """What a settlement committed, filled only after its commit succeeds.

    Callers create an empty report, pass it in, and read it after the call.
    It stays empty when the settlement did not commit (fence missed, commit
    failed, nothing to settle).

    ``control_state`` is the committed V2 control identity, so the caller can
    publish without re-reading the latest state. ``paused_for`` is the
    recorded reason of a committed interruption pause.
    """

    control_state: dict[str, Any] = field(default_factory=dict)
    paused_for: InterruptionReason | None = None

    @property
    def paused(self) -> bool:
        return self.paused_for is not None
