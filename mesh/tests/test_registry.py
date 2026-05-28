"""Registry tests — heartbeat ingestion + snapshot replay."""

from __future__ import annotations

from datetime import timedelta

import pytest

from mesh.registry import (
    HeartbeatPostRequest,
    MeshRegistry,
    NODE_UNREACHABLE_AFTER,
    build_ranked_routes,
)
from mesh.tests.conftest import make_heartbeat


def test_heartbeat_ingest_creates_node(spark_node, catalog, fresh_now):
    reg = MeshRegistry(catalog=catalog)
    hb = make_heartbeat(spark_node, fresh_now, ["qwen3-math-7b-q4"], catalog)
    resp = reg.record_heartbeat(
        HeartbeatPostRequest(heartbeat=hb, node_url="http://spark-1:8000/v1")
    )
    assert resp.ack is True
    snap = reg.snapshot(now=fresh_now)
    assert spark_node.node_id in snap.nodes
    assert snap.nodes[spark_node.node_id].node_url == "http://spark-1:8000/v1"
    assert "qwen3-math-7b-q4" in snap.specialists
    assert snap.specialists["qwen3-math-7b-q4"][0].node_id == spark_node.node_id


def test_heartbeat_replay_takes_latest(spark_node, catalog, fresh_now):
    reg = MeshRegistry(catalog=catalog)
    hb1 = make_heartbeat(spark_node, fresh_now, ["qwen3-math-7b-q4"], catalog, queue_depth=0)
    hb2 = make_heartbeat(
        spark_node,
        fresh_now + timedelta(seconds=5),
        ["qwen3-math-7b-q4"],
        catalog,
        queue_depth=3,
    )
    reg.record_heartbeat(HeartbeatPostRequest(heartbeat=hb1, node_url="http://spark-1:8000/v1"))
    reg.record_heartbeat(HeartbeatPostRequest(heartbeat=hb2))
    snap = reg.snapshot(now=fresh_now + timedelta(seconds=6))
    # Latest queue_depth wins
    assert snap.nodes[spark_node.node_id].queue_depth == 3


def test_node_unreachable_after_5_min(spark_node, catalog, fresh_now):
    reg = MeshRegistry(catalog=catalog)
    hb = make_heartbeat(spark_node, fresh_now, ["qwen3-math-7b-q4"], catalog)
    reg.record_heartbeat(HeartbeatPostRequest(heartbeat=hb, node_url="http://spark-1:8000/v1"))
    snap = reg.snapshot(now=fresh_now + NODE_UNREACHABLE_AFTER + timedelta(seconds=1))
    assert snap.nodes[spark_node.node_id].health == "unreachable"


def test_node_left_event_drops_node(spark_node, catalog, fresh_now):
    reg = MeshRegistry(catalog=catalog)
    hb = make_heartbeat(spark_node, fresh_now, ["qwen3-math-7b-q4"], catalog)
    reg.record_heartbeat(HeartbeatPostRequest(heartbeat=hb, node_url="http://spark-1:8000/v1"))
    reg.record_node_left(spark_node.node_id, reason="graceful")
    snap = reg.snapshot(now=fresh_now + timedelta(seconds=10))
    assert spark_node.node_id not in snap.nodes


def test_coverage_built_from_loaded_models(spark_node, catalog, fresh_now):
    reg = MeshRegistry(catalog=catalog)
    hb = make_heartbeat(
        spark_node,
        fresh_now,
        ["qwen3-math-7b-q4", "qwen3-coder-7b-q4"],
        catalog,
    )
    reg.record_heartbeat(HeartbeatPostRequest(heartbeat=hb, node_url="http://spark-1:8000/v1"))
    snap = reg.snapshot(now=fresh_now)
    assert "math" in snap.coverage
    assert "code" in snap.coverage
    assert spark_node.node_id in snap.coverage["math"]


def test_build_ranked_routes(spark_node, catalog, fresh_now):
    reg = MeshRegistry(catalog=catalog)
    hb = make_heartbeat(spark_node, fresh_now, ["qwen3-math-7b-q4"], catalog, queue_depth=1)
    reg.record_heartbeat(HeartbeatPostRequest(heartbeat=hb, node_url="http://spark-1:8000/v1"))
    snap = reg.snapshot(now=fresh_now)
    ranked = build_ranked_routes(snap)
    # qwen3-math has difficulty_tiers ["medium", "hard"]
    assert "math|medium" in ranked
    assert "math|hard" in ranked
    routes = ranked["math|medium"]
    assert routes[0].node_id == spark_node.node_id
    assert routes[0].estimated_queue_ms == 500  # 1 * 500ms


def test_event_log_is_append_only(spark_node, catalog, fresh_now):
    reg = MeshRegistry(catalog=catalog)
    hb = make_heartbeat(spark_node, fresh_now, ["qwen3-math-7b-q4"], catalog)
    reg.record_heartbeat(HeartbeatPostRequest(heartbeat=hb, node_url="http://spark-1:8000/v1"))
    assert len(reg.events) == 1
    reg.record_heartbeat(HeartbeatPostRequest(heartbeat=hb))
    assert len(reg.events) == 2
    # Snapshot calls do NOT mutate the log
    reg.snapshot()
    assert len(reg.events) == 2


def test_run_allocator_uses_latest_hardware(spark_node, catalog, fresh_now):
    reg = MeshRegistry(catalog=catalog)
    hb = make_heartbeat(spark_node, fresh_now, [], catalog)
    reg.record_heartbeat(HeartbeatPostRequest(heartbeat=hb, node_url="http://spark-1:8000/v1"))
    out = reg.run_allocator(strategy="tiered")
    assert spark_node.node_id in out
    # GB10 should fit something
    assert out[spark_node.node_id] is not None


# ─── heartbeat-log compaction (unbounded-growth fix) ─────────────────────────


def _spam_heartbeats(reg, spark_node, catalog, fresh_now, n: int):
    for i in range(n):
        hb = make_heartbeat(
            spark_node,
            fresh_now + timedelta(seconds=5 * i),
            ["qwen3-math-7b-q4"],
            catalog,
            queue_depth=i,
        )
        reg.record_heartbeat(
            HeartbeatPostRequest(heartbeat=hb, node_url="http://spark-1:8000/v1")
        )


def test_heartbeat_log_compacts_past_max_events(spark_node, catalog, fresh_now):
    """Past max_events, superseded heartbeats are dropped — the log stays
    bounded instead of growing one entry per 5s tick forever."""
    reg = MeshRegistry(catalog=catalog, max_events=5)
    _spam_heartbeats(reg, spark_node, catalog, fresh_now, n=40)
    assert len(reg.events) <= 5  # not 40


def test_compaction_preserves_latest_heartbeat(spark_node, catalog, fresh_now):
    """Dropping older heartbeats must not change what the snapshot reports —
    snapshot only reads the latest heartbeat per node."""
    reg = MeshRegistry(catalog=catalog, max_events=5)
    _spam_heartbeats(reg, spark_node, catalog, fresh_now, n=40)
    snap = reg.snapshot(now=fresh_now + timedelta(seconds=5 * 40))
    assert snap.nodes[spark_node.node_id].queue_depth == 39  # the latest tick
    assert snap.nodes[spark_node.node_id].node_url == "http://spark-1:8000/v1"


def test_compaction_preserves_node_left_drop(spark_node, catalog, fresh_now):
    """A node_left after compaction still drops the node — the snapshot's
    left-handling compares against the (retained) latest heartbeat."""
    reg = MeshRegistry(catalog=catalog, max_events=3)
    _spam_heartbeats(reg, spark_node, catalog, fresh_now, n=10)
    reg.record_node_left(spark_node.node_id)
    snap = reg.snapshot(now=fresh_now + timedelta(seconds=60))
    assert spark_node.node_id not in snap.nodes


def test_compaction_preserves_allocator(spark_node, catalog, fresh_now):
    """run_allocator uses the latest heartbeat's hardware per node, which
    compaction retains."""
    reg = MeshRegistry(catalog=catalog, max_events=3)
    _spam_heartbeats(reg, spark_node, catalog, fresh_now, n=10)
    out = reg.run_allocator(strategy="tiered")
    assert spark_node.node_id in out
    assert out[spark_node.node_id] is not None
