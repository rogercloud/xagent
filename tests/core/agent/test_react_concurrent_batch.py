"""Inc.3 — concurrent batch execution + ordered backfill (design §4.2.2, §5.5).

``_run_concurrent_batch`` runs a segment of concurrency-safe tool calls under a
Semaphore via ``asyncio.gather`` and back-fills their results into the context
in the original tool-call order. Invariants pinned here:

- I1: ``add_tool_result`` order == input order, even when tools finish out of
  order.
- I2: every ``tool_call_id`` gets exactly one result (including failures).
- real concurrency: the batch overlaps (peak == batch size) rather than running
  serially, and the Semaphore caps the peak.
- exception isolation: one failing tool yields an error result for that call
  while the rest succeed.
"""

from __future__ import annotations

import asyncio

from tests.core.agent.concurrency_harness import (
    ConcurrencyTracker,
    FakeRuntime,
    FakeTool,
    RecordingContext,
    make_react,
    make_tool_call,
)


async def test_ordered_backfill_despite_out_of_order_completion() -> None:
    names = ["s1", "s2", "s3"]
    tracker = ConcurrencyTracker()
    gates = {name: asyncio.Event() for name in names}
    tools = [
        FakeTool(name, concurrency_safe=True, gate=gates[name], tracker=tracker)
        for name in names
    ]
    pattern = make_react(parallel=True, max_concurrency=3)
    runtime = FakeRuntime()
    context = RecordingContext()
    batch = [make_tool_call(name) for name in names]

    task = asyncio.create_task(
        pattern._run_concurrent_batch(batch, tools, runtime, context)
    )
    # Wait until all three are in-flight, then release them in REVERSE order so
    # completion order is the opposite of input order.
    while tracker.active < 3:
        await asyncio.sleep(0)
    for index, name in enumerate(["s3", "s2", "s1"], start=1):
        gates[name].set()
        while len(tracker.leave_order) < index:
            await asyncio.sleep(0)
    await task

    # Completion was reverse, but backfill preserves input order (I1).
    assert tracker.leave_order == ["s3", "s2", "s1"]
    assert [r["tool_name"] for r in context.tool_results] == names
    assert [r["tool_call_id"] for r in context.tool_results] == [
        tc["id"] for tc in batch
    ]


async def test_every_tool_call_id_gets_exactly_one_result() -> None:
    names = ["s1", "s2", "s3"]
    tools = [FakeTool(name, concurrency_safe=True) for name in names]
    pattern = make_react(parallel=True, max_concurrency=3)
    context = RecordingContext()
    batch = [make_tool_call(name) for name in names]

    await pattern._run_concurrent_batch(batch, tools, FakeRuntime(), context)

    ids = [r["tool_call_id"] for r in context.tool_results]
    assert sorted(ids) == sorted(tc["id"] for tc in batch)
    assert len(ids) == len(set(ids)) == 3


async def test_batch_runs_concurrently() -> None:
    names = ["s1", "s2", "s3"]
    tracker = ConcurrencyTracker()
    tools = [
        FakeTool(name, concurrency_safe=True, delay=0.02, tracker=tracker)
        for name in names
    ]
    pattern = make_react(parallel=True, max_concurrency=3)
    context = RecordingContext()
    batch = [make_tool_call(name) for name in names]

    await pattern._run_concurrent_batch(batch, tools, FakeRuntime(), context)

    # All three overlapped (a serial run would never exceed 1).
    assert tracker.peak == 3


async def test_semaphore_caps_concurrency() -> None:
    names = ["s1", "s2", "s3", "s4"]
    tracker = ConcurrencyTracker()
    tools = [
        FakeTool(name, concurrency_safe=True, delay=0.02, tracker=tracker)
        for name in names
    ]
    pattern = make_react(parallel=True, max_concurrency=2)
    context = RecordingContext()
    batch = [make_tool_call(name) for name in names]

    await pattern._run_concurrent_batch(batch, tools, FakeRuntime(), context)

    assert tracker.peak <= 2
    assert len(context.tool_results) == 4


async def test_exception_isolation_within_batch() -> None:
    tools = [
        FakeTool("s1", concurrency_safe=True),
        FakeTool("boom", concurrency_safe=True, raises=RuntimeError("kaboom")),
        FakeTool("s3", concurrency_safe=True),
    ]
    pattern = make_react(parallel=True, max_concurrency=3)
    context = RecordingContext()
    batch = [make_tool_call(name) for name in ["s1", "boom", "s3"]]

    results = await pattern._run_concurrent_batch(batch, tools, FakeRuntime(), context)

    # All three back-filled in order; the middle one is an error result.
    assert [r["tool_name"] for r in context.tool_results] == ["s1", "boom", "s3"]
    assert results[0]["success"] is True
    assert results[1]["success"] is False
    assert "kaboom" in results[1]["error"]
    assert results[2]["success"] is True
