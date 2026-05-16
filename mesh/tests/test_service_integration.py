"""Service integration tests — locks contract under realistic mesh shapes.

Unit tests in `test_service.py` cover endpoint plumbing + auth in isolation.
This module exercises the HTTP boundary under multi-node / multi-event
sequences a real mesh sees: heterogeneous fleet, stale-node detection,
re-join, multi-call audit trail, JSON round-trip fidelity, concurrent
heartbeats.

All tests run in dev-mode (no auth) to keep the focus on the service
contract, not the auth surface (already covered).
"""

from __future__ import annotations

import threading
from datetime import datetime, timedelta, timezone
from typing import Any

import pytest
from fastapi.testclient import TestClient

from mesh.models import NetworkLink, NodeHeartbeat
from mesh.registry import (
    AllocationEvent,
    HeartbeatPostRequest,
    MeshRegistry,
    NODE_UNREACHABLE_AFTER,
)
from mesh.service import NODE_TOKEN_ENV, create_mesh_app
from mesh.tests.conftest import make_heartbeat


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def app_client(monkeypatch, catalog):
    """Dev-mode TestClient with a fresh catalog-loaded MeshRegistry.

    The catalog is pre-loaded so /registry snapshots include it — the
    router downstream depends on `snapshot.catalog` for specialist
    metadata (domain, difficulty_tiers, etc).
    """
    monkeypatch.delenv(NODE_TOKEN_ENV, raising=False)
    reg = MeshRegistry(catalog=catalog)
    return TestClient(create_mesh_app(registry=reg)), reg


def _post_hb(client: TestClient, node, ts: datetime, catalog, loaded: list[str], **hb_kw: Any) -> None:
    hb = make_heartbeat(node, ts, loaded, catalog, **hb_kw)
    req = HeartbeatPostRequest(heartbeat=hb, node_url=f"http://{node.node_id}:8001/v1")
    resp = client.post("/heartbeat", json=req.model_dump(mode="json"))
    assert resp.status_code == 200, resp.text


# ---------------------------------------------------------------------------
# Heterogeneous fleet — spark + mac-mini + tiny all heartbeat
# ---------------------------------------------------------------------------


def test_heterogeneous_fleet_visible_in_snapshot(
    app_client, spark_node, mac_mini_node, tiny_node, catalog, fresh_now
):
    client, _ = app_client
    # Pick a specialist each node can actually serve. spark_node has vllm,
    # mac_mini has llamacpp/mlx, tiny has llamacpp.
    _post_hb(client, spark_node, fresh_now, catalog, ["qwen3-coder-30b-a3b-fp8"])
    _post_hb(client, mac_mini_node, fresh_now, catalog, ["qwen3-math-7b-q4"])
    _post_hb(client, tiny_node, fresh_now, catalog, ["qwen3-math-7b-q4"])

    snap = client.get("/registry").json()["snapshot"]
    node_ids = set(snap["nodes"].keys())
    assert node_ids == {spark_node.node_id, mac_mini_node.node_id, tiny_node.node_id}
    # Each node's reported specialist is in the snapshot specialists map
    assert "qwen3-coder-30b-a3b-fp8" in snap["specialists"]
    assert "qwen3-math-7b-q4" in snap["specialists"]
    # qwen3-math is on both mac-mini and tiny
    math_bindings = snap["specialists"]["qwen3-math-7b-q4"]
    binding_node_ids = {b["node_id"] for b in math_bindings}
    assert binding_node_ids == {mac_mini_node.node_id, tiny_node.node_id}


# ---------------------------------------------------------------------------
# Stale-node detection through HTTP
# ---------------------------------------------------------------------------


def test_stale_node_flips_unreachable_in_snapshot(app_client, spark_node, mac_mini_node, catalog, fresh_now):
    """Per spec §3.4: nodes silent >5 min should show unreachable.

    The /registry endpoint uses datetime.now(UTC) internally — we can't
    monkey-patch that across the HTTP boundary, so we drive staleness by
    posting a stale-ts heartbeat then a fresh one and asserting the stale
    one is marked unreachable.
    """
    client, reg = app_client
    # spark posts a heartbeat 10 minutes in the past
    stale_ts = datetime.now(timezone.utc) - timedelta(minutes=10)
    _post_hb(client, spark_node, stale_ts, catalog, ["qwen3-coder-30b-a3b-fp8"])
    # mac-mini posts fresh
    _post_hb(client, mac_mini_node, datetime.now(timezone.utc), catalog, ["qwen3-math-7b-q4"])

    snap = client.get("/registry").json()["snapshot"]
    assert snap["nodes"][spark_node.node_id]["health"] == "unreachable"
    assert snap["nodes"][mac_mini_node.node_id]["health"] == "healthy"


# ---------------------------------------------------------------------------
# Allocator sees latest hardware via HTTP
# ---------------------------------------------------------------------------


def test_allocate_reflects_latest_heartbeats(app_client, spark_node, mac_mini_node, catalog, fresh_now):
    client, _ = app_client
    _post_hb(client, spark_node, fresh_now, catalog, [])  # registry-only mode
    _post_hb(client, mac_mini_node, fresh_now, catalog, [])

    resp = client.post("/allocate", json={"strategy": "tiered"})
    assert resp.status_code == 200
    body = resp.json()
    # Both nodes appear in suggestions, each either gets a specialist or None
    assert set(body["suggestions"].keys()) == {spark_node.node_id, mac_mini_node.node_id}


def test_allocate_traffic_mix_passed_through(app_client, spark_node, catalog, fresh_now):
    client, _ = app_client
    _post_hb(client, spark_node, fresh_now, catalog, [])

    resp = client.post(
        "/allocate",
        json={"strategy": "tiered", "traffic_mix": {"math": 0.7, "code": 0.3}},
    )
    assert resp.status_code == 200
    assert resp.json()["strategy"] == "tiered"


# ---------------------------------------------------------------------------
# Audit trail — multiple allocate calls leave AllocationEvents
# ---------------------------------------------------------------------------


def test_allocation_audit_trail_in_event_log(app_client, spark_node, catalog, fresh_now):
    client, reg = app_client
    _post_hb(client, spark_node, fresh_now, catalog, [])

    for _ in range(3):
        client.post("/allocate", json={"strategy": "tiered"})

    alloc_events = [e for e in reg.events if isinstance(e, AllocationEvent)]
    assert len(alloc_events) == 3
    assert all(e.strategy == "tiered" for e in alloc_events)


# ---------------------------------------------------------------------------
# Re-join — node_left then heartbeat again
# ---------------------------------------------------------------------------


def test_node_rejoin_clears_left_flag(app_client, spark_node, catalog, fresh_now):
    client, reg = app_client
    _post_hb(client, spark_node, fresh_now, catalog, ["qwen3-coder-30b-a3b-fp8"])

    # Force a node_left at the registry layer — no HTTP endpoint for this
    # in v0.0.3, so we drive it via the registry directly. The contract
    # we're locking: a fresh heartbeat clears the left flag.
    reg.record_node_left(spark_node.node_id, reason="graceful")
    snap_after_left = client.get("/registry").json()["snapshot"]
    assert spark_node.node_id not in snap_after_left["nodes"]

    # Re-heartbeat
    _post_hb(
        client,
        spark_node,
        fresh_now + timedelta(seconds=30),
        catalog,
        ["qwen3-coder-30b-a3b-fp8"],
    )
    snap_after_rejoin = client.get("/registry").json()["snapshot"]
    assert spark_node.node_id in snap_after_rejoin["nodes"]


# ---------------------------------------------------------------------------
# JSON round-trip fidelity — RegistrySnapshot survives HTTP serialization
# ---------------------------------------------------------------------------


def test_snapshot_json_roundtrip_preserves_specialist_metadata(
    app_client, spark_node, catalog, fresh_now
):
    client, _ = app_client
    _post_hb(client, spark_node, fresh_now, catalog, ["qwen3-coder-30b-a3b-fp8"])

    snap = client.get("/registry").json()["snapshot"]
    # Catalog field must survive — routers depend on it
    assert "catalog" in snap
    assert "qwen3-coder-30b-a3b-fp8" in snap["catalog"]
    card = snap["catalog"]["qwen3-coder-30b-a3b-fp8"]
    # Spot-check fields the router/select.py reads
    assert "domain" in card
    assert "difficulty_tiers" in card
    assert isinstance(card["difficulty_tiers"], list)
    # NodeSummary's queue_depth + last_seen survive
    summary = snap["nodes"][spark_node.node_id]
    assert "queue_depth" in summary
    assert "last_seen" in summary
    assert "loaded_specialist_ids" in summary


def test_probe_network_json_roundtrip_preserves_links(app_client, spark_node, mac_mini_node, catalog, fresh_now):
    client, reg = app_client
    hb = make_heartbeat(spark_node, fresh_now, ["qwen3-coder-30b-a3b-fp8"], catalog)
    hb_with_view = NodeHeartbeat(
        node_id=hb.node_id,
        ts=hb.ts,
        hardware=hb.hardware,
        loaded_models=hb.loaded_models,
        util=hb.util,
        health=hb.health,
        network_view={mac_mini_node.node_id: NetworkLink(rtt_ms=5.0, bandwidth_mbps=1000.0)},
    )
    reg.record_heartbeat(HeartbeatPostRequest(heartbeat=hb_with_view, node_url="http://spark:8001/v1"))

    body = client.post("/probe-network").json()
    link = body["network_views"][spark_node.node_id][mac_mini_node.node_id]
    # Both numeric fields preserved through JSON
    assert link["rtt_ms"] == 5.0
    assert link["bandwidth_mbps"] == 1000.0


# ---------------------------------------------------------------------------
# Concurrent heartbeats — registry sequencing under load
# ---------------------------------------------------------------------------


def test_concurrent_heartbeats_all_recorded(app_client, spark_node, mac_mini_node, tiny_node, catalog, fresh_now):
    """N threads POST heartbeats in parallel. Final snapshot must include
    all N nodes; the event log must include exactly N+ HeartbeatEvents.

    MeshRegistry isn't thread-safe per its own docstring, so this exists
    to FLAG if the contract ever needs locking — under TestClient + the
    GIL we expect serialization to keep it green; under uvicorn workers
    > 1 the test would surface the gap.
    """
    client, reg = app_client

    nodes = [spark_node, mac_mini_node, tiny_node]
    loaded_per_node = [
        ["qwen3-coder-30b-a3b-fp8"],
        ["qwen3-math-7b-q4"],
        ["qwen3-math-7b-q4"],
    ]

    def post(node, loaded):
        _post_hb(client, node, fresh_now, catalog, loaded)

    threads = [
        threading.Thread(target=post, args=(node, loaded))
        for node, loaded in zip(nodes, loaded_per_node)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    snap = client.get("/registry").json()["snapshot"]
    assert set(snap["nodes"].keys()) == {n.node_id for n in nodes}
    # Exactly 3 heartbeat events (one per node)
    from mesh.registry import HeartbeatEvent

    hb_events = [e for e in reg.events if isinstance(e, HeartbeatEvent)]
    assert len(hb_events) == 3


# ---------------------------------------------------------------------------
# Idempotency-ish — repeated heartbeats from same node
# ---------------------------------------------------------------------------


def test_repeated_heartbeats_take_latest(app_client, spark_node, catalog, fresh_now):
    client, _ = app_client
    # Three heartbeats from the same node with increasing queue_depth
    _post_hb(client, spark_node, fresh_now, catalog, ["qwen3-coder-30b-a3b-fp8"], queue_depth=0)
    _post_hb(
        client,
        spark_node,
        fresh_now + timedelta(seconds=5),
        catalog,
        ["qwen3-coder-30b-a3b-fp8"],
        queue_depth=2,
    )
    _post_hb(
        client,
        spark_node,
        fresh_now + timedelta(seconds=10),
        catalog,
        ["qwen3-coder-30b-a3b-fp8"],
        queue_depth=5,
    )

    snap = client.get("/registry").json()["snapshot"]
    # Snapshot reflects the LAST heartbeat's queue depth
    assert snap["nodes"][spark_node.node_id]["queue_depth"] == 5
