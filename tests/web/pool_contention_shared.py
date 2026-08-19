"""Wall-clock budgets for tests that assert on one-slot QueuePool contention.

These tests are timing-shaped by nature: they hold the only slot of a one-slot
``QueuePool`` and assert on what the runtime does while it waits. The web test
leg runs ``pytest -n 4`` on a shared CI runner, where a descheduled worker
thread can eat several hundred milliseconds on its own — enough to blow any
budget picked to make the test feel snappy. Both failure modes have been
observed in CI as spurious reds.

The budgets are therefore split by what each one actually guards:

``EXHAUSTION_POOL_TIMEOUT``
    Semantic. Tests that assert on pool-*exhaustion* behaviour need the checkout
    to give up quickly, and a slow runner only makes it fire sooner — which is
    the outcome those tests already expect.

``CONTENTION_POOL_TIMEOUT``
    The opposite. Tests that assert work *waits its turn* must never see the
    checkout give up, so this sits far above any plausible scheduling hiccup. It
    still has to stay finite and reasonably tight, because it — not
    ``LOOP_LIVENESS_TIMEOUT`` — is what bounds the on-loop regression these
    tests exist to catch: a checkout that blocks the loop also blocks the
    coroutine that would notice, so nothing can be evaluated until the checkout
    gives up. Keep it well above ``LOOP_LIVENESS_TIMEOUT`` and well below
    anything a human would call a hang.

``LOOP_LIVENESS_TIMEOUT`` / ``LOOP_LIVENESS_TICKS``
    How long to wait to observe that the event loop is still turning *while the
    contended operation is in flight*. A loop blocked by a synchronous checkout
    never ticks at all, so this only has to out-wait a descheduled runner — it
    is not measuring throughput.

``GUARD_TIMEOUT``
    A hang detector, not an assertion. Once the pool slot is released the awaited
    work should finish immediately; the only failure worth reporting here is "it
    never finished at all", so the ceiling is deliberately generous.
"""

from __future__ import annotations

import asyncio
from typing import Callable

EXHAUSTION_POOL_TIMEOUT = 0.4
CONTENTION_POOL_TIMEOUT = 15.0
LOOP_LIVENESS_TIMEOUT = 2.0
LOOP_LIVENESS_TICKS = 3
GUARD_TIMEOUT = 30.0


async def wait_for_ticks(
    read_ticks: Callable[[], int],
    *,
    minimum: int = LOOP_LIVENESS_TICKS,
    timeout: float = LOOP_LIVENESS_TIMEOUT,
) -> int:
    """Give the event loop up to ``timeout`` to turn ``minimum`` more times.

    Counts from a baseline taken on entry, so only progress made *after* the
    call is credited. Callers must therefore wait for the contended operation to
    start before calling this — otherwise ticks banked while nothing was
    contending could satisfy the assertion, and the test would release the held
    connection without ever observing the loop during contention.

    Returns the number of ticks observed since entry, so the caller keeps the
    assertion and its message. A loop blocked by a synchronous checkout stops
    ticking altogether, so a regression still fails here no matter how long we
    are willing to wait — waiting longer only buys tolerance for a runner that
    is merely slow. Note that such a regression cannot be *detected* until the
    blocking checkout returns, so it is bounded by the pool timeout in force,
    not by ``timeout``.
    """

    loop = asyncio.get_running_loop()
    baseline = read_ticks()
    deadline = loop.time() + timeout
    while read_ticks() - baseline < minimum and loop.time() < deadline:
        await asyncio.sleep(0.01)
    return read_ticks() - baseline
