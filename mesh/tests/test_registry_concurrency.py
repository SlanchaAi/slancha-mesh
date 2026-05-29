"""Concurrency regression tests for `MeshRegistry`.

The push topology mounts `create_mesh_app` into slancha-api and lets N
nodes hit `POST /heartbeat` concurrently. FastAPI's path-operation
handlers are *synchronous* `def` (see `mesh.registry_app.post_heartbeat`),
which Starlette dispatches on its anyio threadpool — so even with a
single uvicorn worker, two heartbeats can land on
`MeshRegistry.record_heartbeat` from two different threads at once.

The lost-update bug these tests pin down:

    `_compact_heartbeats` reads `self._events` in pass 1 to compute the
    set of indices to keep, then rebinds `self._events` to a filtered
    list in pass 2. If another thread appends a fresh heartbeat between
    the two passes, the appended event sits at an index that was NOT in
    `keep` (computed before the append) AND IS a `HeartbeatEvent` — so
    pass 2's predicate filters it out. The new node got a `200 ack=True`,
    but its heartbeat is silently dropped from the log → the node will
    surface as `unreachable` until its next beat.

Run on `main` *without* the lock fix and `test_concurrent_heartbeat_
during_compaction_is_not_lost` fails by losing node-C; with the fix it
passes.
"""

from __future__ import annotations

import threading
import time
from datetime import timedelta

import pytest

from mesh.models import NodeProbe
from mesh.registry import HeartbeatPostRequest, MeshRegistry
from mesh.tests.conftest import make_heartbeat


def _node(node_id: str) -> NodeProbe:
    """A minimum-viable NodeProbe — keeps tests focused on registry semantics."""
    return NodeProbe(
        node_id=node_id,
        friendly_name=node_id,
        chip="NVIDIA GB10",
        arch="aarch64",
        cuda_capability="12.1",
        fp4_tops=3800.0,
        fp16_tops=250.0,
        ram_total_gb=128.0,
        ram_available_gb=110.0,
        vram_total_gb=None,
        vram_available_gb=None,
        unified_memory=True,
        memory_bandwidth_gbs=273.0,
        available_backends=["vllm"],
        disk_free_gb=500.0,
        rtt_to_master_ms=2.0,
    )


def test_concurrent_heartbeat_during_compaction_is_not_lost(catalog, fresh_now):
    """Pass-1/pass-2 window in `_compact_heartbeats` must not drop a
    heartbeat that lands inside it.

    Deterministic: monkey-patches the instance's `_compact_heartbeats`
    to expose the window as a `threading.Event`, drives one writer
    (node-B) into the window, and races a second writer (node-C) into
    the window before pass 2 rebinds the list. The expected outcome is
    that node-C's beat survives, which requires the registry to
    serialize mutations across threads.
    """
    node_a, node_b, node_c = _node("node-a"), _node("node-b"), _node("node-c")
    sid = "qwen3-math-7b-q4"

    # max_events=1 so the next append after seeding triggers compaction.
    reg = MeshRegistry(catalog=catalog, max_events=1)

    # Seed node-A so node-B's append in T1 is the one that triggers compaction.
    reg.record_heartbeat(
        HeartbeatPostRequest(
            heartbeat=make_heartbeat(node_a, fresh_now, [sid], catalog),
            node_url="http://a:8003/v1",
        )
    )

    pass1_done = threading.Event()
    resume_pass2 = threading.Event()

    def slow_compact() -> None:
        # Pass 1 — read indices.
        from mesh.registry import HeartbeatEvent

        latest_hb_idx: dict[str, int] = {}
        for i, ev in enumerate(reg._events):
            if isinstance(ev, HeartbeatEvent):
                latest_hb_idx[ev.heartbeat.node_id] = i
        keep = set(latest_hb_idx.values())

        # Window — let the racer (T2) attempt its append.
        pass1_done.set()
        resume_pass2.wait(timeout=2.0)

        # Pass 2 — rebind.
        reg._events = [
            ev
            for i, ev in enumerate(reg._events)
            if not isinstance(ev, HeartbeatEvent) or i in keep
        ]

    reg._compact_heartbeats = slow_compact  # type: ignore[method-assign]

    def t1() -> None:
        # Triggers compaction (events == 1 == max_events, append pushes to 2).
        reg.record_heartbeat(
            HeartbeatPostRequest(
                heartbeat=make_heartbeat(node_b, fresh_now + timedelta(seconds=1), [sid], catalog),
                node_url="http://b:8003/v1",
            )
        )

    def t2() -> None:
        # Wait until T1 is inside the compaction window, then race an append.
        pass1_done.wait(timeout=2.0)
        reg.record_heartbeat(
            HeartbeatPostRequest(
                heartbeat=make_heartbeat(node_c, fresh_now + timedelta(seconds=2), [sid], catalog),
                node_url="http://c:8003/v1",
            )
        )

    th1 = threading.Thread(target=t1)
    th2 = threading.Thread(target=t2)
    th1.start()
    th2.start()

    # Wait until T1 hits the window, give T2 time to attempt its append, then
    # release T1's pass 2.
    assert pass1_done.wait(timeout=2.0), "T1 never reached the compaction window"
    time.sleep(0.05)
    resume_pass2.set()
    th1.join(timeout=2.0)
    th2.join(timeout=2.0)
    assert not th1.is_alive() and not th2.is_alive()

    snap = reg.snapshot(now=fresh_now + timedelta(seconds=10))
    # Both racing nodes' heartbeats must survive in the final snapshot.
    # Without registry-level locking, node-C's heartbeat lands at an index
    # outside `keep` and is dropped by pass 2 — the silent ack-but-lose bug.
    assert node_a.node_id in snap.nodes
    assert node_b.node_id in snap.nodes
    assert node_c.node_id in snap.nodes, (
        "node-C's heartbeat was acked but dropped by concurrent compaction — "
        "the documented `asyncio.Lock` mitigation does not cover sync handlers "
        "in Starlette's threadpool; a `threading` lock is required."
    )


def test_record_and_snapshot_under_thread_storm_preserve_all_nodes(catalog, fresh_now):
    """Sanity stress: M nodes × K beats with `max_events` tiny enough that
    compaction fires constantly. Every node's latest beat must end up in
    the snapshot. This catches the same lost-update bug probabilistically,
    independent of the monkey-patched determinism above.
    """
    n_nodes = 12
    n_beats = 25
    reg = MeshRegistry(catalog=catalog, max_events=4)
    sid = "qwen3-math-7b-q4"
    nodes = [_node(f"node-{i:02d}") for i in range(n_nodes)]

    def hammer(idx: int) -> None:
        node = nodes[idx]
        for k in range(n_beats):
            reg.record_heartbeat(
                HeartbeatPostRequest(
                    heartbeat=make_heartbeat(
                        node, fresh_now + timedelta(seconds=k), [sid], catalog
                    ),
                    node_url=f"http://node-{idx:02d}:8003/v1",
                )
            )

    threads = [threading.Thread(target=hammer, args=(i,)) for i in range(n_nodes)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=5.0)
    assert all(not t.is_alive() for t in threads)

    snap = reg.snapshot(now=fresh_now + timedelta(seconds=n_beats + 10))
    missing = [n.node_id for n in nodes if n.node_id not in snap.nodes]
    assert not missing, f"lost heartbeats for {len(missing)}/{n_nodes} nodes: {missing[:5]}"


@pytest.mark.parametrize("n_threads", [4, 8])
def test_concurrent_record_and_snapshot_dont_crash(catalog, fresh_now, n_threads):
    """Snapshot iterates `self._events` while writers may compact-and-rebind it.

    A reader holds a reference to the original list, so a rebind on the
    writer side can't *crash* the reader — but if `snapshot()` and
    `record_heartbeat` are not co-locked, the reader may walk a list that
    is concurrently being filtered/grown, returning a stale view. This
    test asserts the no-crash baseline; correctness of the latest-beat
    semantics is covered by the deterministic test above.
    """
    sid = "qwen3-math-7b-q4"
    reg = MeshRegistry(catalog=catalog, max_events=8)
    stop = threading.Event()

    def writers(idx: int) -> None:
        node = _node(f"w-{idx}")
        k = 0
        while not stop.is_set():
            reg.record_heartbeat(
                HeartbeatPostRequest(
                    heartbeat=make_heartbeat(
                        node, fresh_now + timedelta(seconds=k), [sid], catalog
                    ),
                    node_url=f"http://w-{idx}:8003/v1",
                )
            )
            k += 1

    def reader() -> None:
        while not stop.is_set():
            _ = reg.snapshot(now=fresh_now + timedelta(seconds=1))

    ws = [threading.Thread(target=writers, args=(i,)) for i in range(n_threads)]
    rs = [threading.Thread(target=reader) for _ in range(2)]
    for t in ws + rs:
        t.start()
    time.sleep(0.25)
    stop.set()
    for t in ws + rs:
        t.join(timeout=3.0)
    assert all(not t.is_alive() for t in ws + rs)
