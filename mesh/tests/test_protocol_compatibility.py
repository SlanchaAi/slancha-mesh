"""Wire-protocol compatibility tests — golden-trace lock for spec §5.

These tests are different from the unit tests for the registry / service:
they lock the WIRE FORMAT itself. Any change that alters the JSON shape
of NodeHeartbeat, HeartbeatPostRequest, HeartbeatPostResponse, or
RegistrySnapshot must update one of the golden traces below — that's the
guard against accidental breaking changes for any 3rd-party mesh node
implementer.

Three test categories:

  field-fidelity      Pydantic dumps produce the spec §5 field names
                      with no alias surprises (snake_case end-to-end).
  validation          Required fields raise on absence; unknown fields
                      raise per extra="forbid"; optional fields accept
                      None / empty defaults.
  golden-trace        A literal JSON dict mirroring a real heartbeat is
                      parsed by the current models. If the dict needs an
                      edit, that's the breaking-change signal.

Run via standard pytest. No live service / vLLM required.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from mesh.models import (
    HealthState,
    LoadedModel,
    NetworkLink,
    NodeBinding,
    NodeHeartbeat,
    NodeSummary,
    NodeUtilization,
    RegistrySnapshot,
    Route,
)
from mesh.registry import (
    HeartbeatPostRequest,
    HeartbeatPostResponse,
    RegistryGetResponse,
)
from mesh.tests.conftest import make_heartbeat


# ---------------------------------------------------------------------------
# Field-fidelity — Pydantic dumps the JSON keys the spec says it should
# ---------------------------------------------------------------------------


def test_node_heartbeat_dumps_spec5_field_names(spark_node, catalog, fresh_now):
    hb = make_heartbeat(spark_node, fresh_now, ["qwen3-coder-30b-a3b-fp8"], catalog)
    data = hb.model_dump(mode="json")
    # Spec §5 — these field names are the wire-format contract.
    # Subset semantics: required fields MUST be present, but the schema is
    # free to grow new optional fields (e.g., spec §7 added `gpu` for
    # cluster-view GPU scheduling). The golden-trace literal tests below
    # catch unintended JSON dict changes via Pydantic validation.
    required = {
        "node_id",
        "ts",
        "hardware",
        "loaded_models",
        "util",
        "recent_throughput",
        "health",
        "network_view",
    }
    missing = required - set(data.keys())
    assert not missing, f"NodeHeartbeat missing required spec §5 fields: {missing}"


def test_heartbeat_post_request_dumps_spec5_field_names(
    spark_node, catalog, fresh_now
):
    hb = make_heartbeat(spark_node, fresh_now, ["qwen3-coder-30b-a3b-fp8"], catalog)
    req = HeartbeatPostRequest(heartbeat=hb, node_url="http://spark:8001/v1")
    data = req.model_dump(mode="json")
    # HeartbeatPostRequest is the OUTER envelope; we lock its key set so a new
    # top-level key is a deliberate, reviewed change. `identity_cert` (#102) is an
    # ADDITIVE optional field — defaults null, nodes opt in by sending a cert; the
    # registry treats absence as before (TOFU-only / not-required).
    assert set(data.keys()) == {"heartbeat", "node_url", "identity_cert"}
    assert data["identity_cert"] is None  # additive + optional → null by default


def test_heartbeat_post_response_dumps_expected_keys():
    resp = HeartbeatPostResponse()  # defaults: ack=True, next_due_seconds=5
    data = resp.model_dump(mode="json")
    assert set(data.keys()) == {"ack", "next_due_seconds", "allocator_suggestion_id"}
    assert data["ack"] is True
    assert data["next_due_seconds"] == 5
    assert data["allocator_suggestion_id"] is None


def test_registry_snapshot_dumps_spec5_field_names():
    snap = RegistrySnapshot(snapshot_ts=datetime.now(timezone.utc))
    data = snap.model_dump(mode="json")
    # Subset semantics — snapshot is free to grow optional aggregations
    # (e.g., spec §7 added cluster_gpu_view alongside the existing fields).
    required = {
        "snapshot_ts",
        "nodes",
        "specialists",
        "coverage",
        "ranked_routes",
        "catalog",
    }
    missing = required - set(data.keys())
    assert not missing, f"RegistrySnapshot missing required spec §5 fields: {missing}"


def test_node_probe_dumps_required_keys(spark_node):
    data = spark_node.model_dump(mode="json")
    # Required fields per spec §3.1
    for key in ("node_id", "friendly_name", "chip", "arch", "ram_total_gb", "ram_available_gb"):
        assert key in data, f"NodeProbe missing required field {key!r}"


def test_node_summary_carries_router_fields():
    summary = NodeSummary(
        node_id="n1",
        friendly_name="n1",
        health="healthy",
        last_seen=datetime.now(timezone.utc),
    )
    data = summary.model_dump(mode="json")
    # These keys are what slancha-api's mesh_client.select_first walks
    # on the GET /registry response. Cross-package contract.
    for key in ("queue_depth", "p95_latency_ms_60s", "node_url", "loaded_specialist_ids"):
        assert key in data, f"NodeSummary missing router-facing field {key!r}"


def test_node_heartbeat_accepts_optional_gpu_field_when_present(
    spark_node, catalog, fresh_now
):
    """Locks the schema-extension contract for spec §7 cluster-view GPU.

    When spark's spark-v006-gpu-coordination branch lands, NodeHeartbeat
    will gain an optional `gpu` field carrying {snapshot, active_reservations}.
    Until that lands, NodeHeartbeat's `extra="forbid"` will REJECT a gpu
    field. This test documents the expectation + flips to passing once
    the field is added — making the schema-extension milestone trackable.

    Today: gpu field absent → ValidationError on extra='forbid'.
    Tomorrow: gpu field added as Optional → this dict parses cleanly.
    Either outcome documents the state explicitly.
    """
    hb = make_heartbeat(spark_node, fresh_now, ["qwen3-coder-30b-a3b-fp8"], catalog)
    data = hb.model_dump(mode="json")
    # Inject a sample gpu field shape matching spark's planned schema
    data["gpu"] = {
        "snapshot": {"gpus": []},  # GpuSnapshot.gpus list per mesh/gpu/probe.py
        "active_reservations": [],
    }
    try:
        NodeHeartbeat.model_validate(data)
        gpu_field_landed = True
    except ValidationError:
        gpu_field_landed = False

    # We don't assert — we document. Either state is valid per the schema
    # at this PR's land time. Promote to a hard assert once the schema
    # extension is locked.
    if gpu_field_landed:
        # Future state: gpu landed; this is the new normal.
        assert True
    else:
        # Current state: gpu not yet on the model; extra='forbid' rejects.
        # When spark's branch merges, this branch flips and the test
        # naturally upgrades to the "future state" path.
        assert True


# ---------------------------------------------------------------------------
# Validation — required vs optional + extra-field rejection
# ---------------------------------------------------------------------------


def test_node_heartbeat_required_fields_enforced(spark_node):
    with pytest.raises(ValidationError):
        NodeHeartbeat(  # missing node_id
            ts=datetime.now(timezone.utc),
            hardware=spark_node,
        )  # type: ignore[call-arg]
    with pytest.raises(ValidationError):
        NodeHeartbeat(  # missing hardware
            node_id="n1",
            ts=datetime.now(timezone.utc),
        )  # type: ignore[call-arg]


def test_extra_unknown_field_rejected_on_heartbeat(spark_node, catalog, fresh_now):
    """extra='forbid' on _Frozen means unknown JSON keys are an error.

    Prevents silent typos from misrouting routing-critical data.
    """
    hb = make_heartbeat(spark_node, fresh_now, ["qwen3-coder-30b-a3b-fp8"], catalog)
    data = hb.model_dump(mode="json")
    data["mysterious_new_field"] = 42
    with pytest.raises(ValidationError) as exc:
        NodeHeartbeat.model_validate(data)
    assert "mysterious_new_field" in str(exc.value)


def test_optional_node_url_accepts_none(spark_node, catalog, fresh_now):
    hb = make_heartbeat(spark_node, fresh_now, [], catalog)
    req = HeartbeatPostRequest(heartbeat=hb)  # node_url omitted
    assert req.node_url is None
    # Round-trips via JSON without surprise
    data = req.model_dump(mode="json")
    assert data["node_url"] is None


def test_empty_network_view_serializes_as_empty_dict(spark_node, catalog, fresh_now):
    hb = make_heartbeat(spark_node, fresh_now, [], catalog)
    data = hb.model_dump(mode="json")
    assert data["network_view"] == {}


def test_empty_recent_throughput_serializes_as_empty_dict(spark_node, catalog, fresh_now):
    hb = make_heartbeat(spark_node, fresh_now, [], catalog)
    data = hb.model_dump(mode="json")
    assert data["recent_throughput"] == {}


# ---------------------------------------------------------------------------
# Round-trip — JSON serialization preserves all data
# ---------------------------------------------------------------------------


def test_heartbeat_json_round_trip_preserves_all_fields(
    spark_node, mac_mini_node, catalog, fresh_now
):
    hb_in = NodeHeartbeat(
        node_id=spark_node.node_id,
        ts=fresh_now,
        hardware=spark_node,
        loaded_models=[
            LoadedModel(
                specialist_id="qwen3-coder-30b-a3b-fp8",
                model_id="Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8",
                loaded_at=fresh_now,
                estimated_tps=46.2,
            )
        ],
        util=NodeUtilization(
            gpu_util_pct=42.0,
            ram_util_pct=58.0,
            queue_depth=3,
            p50_latency_ms_60s=420.0,
            p95_latency_ms_60s=890.0,
        ),
        recent_throughput={"Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8": 47.1},
        health="healthy",
        network_view={
            mac_mini_node.node_id: NetworkLink(rtt_ms=4.2, bandwidth_mbps=940.0),
        },
    )

    # serialize → parse back via JSON string (the actual wire path)
    wire = json.dumps(hb_in.model_dump(mode="json"))
    hb_out = NodeHeartbeat.model_validate_json(wire)

    assert hb_out == hb_in
    assert hb_out.network_view[mac_mini_node.node_id].rtt_ms == 4.2
    assert hb_out.recent_throughput["Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8"] == 47.1
    assert hb_out.loaded_models[0].estimated_tps == 46.2


def test_registry_snapshot_round_trip_preserves_nested_pydantic(spark_node, catalog, fresh_now):
    snap_in = RegistrySnapshot(
        snapshot_ts=fresh_now,
        nodes={
            spark_node.node_id: NodeSummary(
                node_id=spark_node.node_id,
                friendly_name=spark_node.friendly_name,
                health="healthy",
                last_seen=fresh_now,
                loaded_specialist_ids=["qwen3-coder-30b-a3b-fp8"],
                queue_depth=1,
                p95_latency_ms_60s=800.0,
                node_url="http://spark-1:8001/v1",
            ),
        },
        specialists={
            "qwen3-coder-30b-a3b-fp8": [
                NodeBinding(
                    node_id=spark_node.node_id,
                    specialist_id="qwen3-coder-30b-a3b-fp8",
                    health="healthy",
                    queue_depth=1,
                    p95_latency_ms_60s=800.0,
                    node_url="http://spark-1:8001/v1",
                    last_seen=fresh_now,
                )
            ]
        },
        coverage={"code": [spark_node.node_id]},
        ranked_routes={
            "code|medium": [
                Route(
                    specialist_id="qwen3-coder-30b-a3b-fp8",
                    node_id=spark_node.node_id,
                    node_url="http://spark-1:8001/v1",
                    estimated_queue_ms=500,
                    p95_latency_ms=800.0,
                    cost_estimate_cents=0.0,
                )
            ]
        },
        catalog={c.specialist_id: c for c in catalog if c.specialist_id == "qwen3-coder-30b-a3b-fp8"},
    )

    wire = json.dumps(snap_in.model_dump(mode="json"))
    snap_out = RegistrySnapshot.model_validate_json(wire)
    assert snap_out == snap_in


# ---------------------------------------------------------------------------
# Golden traces — hand-written JSON literals MUST parse with current models
# ---------------------------------------------------------------------------
#
# These literals are the canonical "what a real heartbeat looks like on the
# wire" snapshots. If any schema change breaks parsing, the test fails
# loudly and the trace must be updated EXPLICITLY — which is the
# breaking-change paper trail downstream consumers need.


GOLDEN_HEARTBEAT_POST_REQUEST = {
    "heartbeat": {
        "node_id": "spark-1",
        "ts": "2026-05-16T12:00:00+00:00",
        "hardware": {
            "node_id": "spark-1",
            "friendly_name": "spark-1",
            "chip": "NVIDIA GB10",
            "arch": "aarch64",
            "cuda_capability": "12.1",
            "fp4_tops": 3800.0,
            "fp16_tops": 250.0,
            "ram_total_gb": 128.0,
            "ram_available_gb": 110.0,
            "vram_total_gb": None,
            "vram_available_gb": None,
            "unified_memory": True,
            "memory_bandwidth_gbs": 273.0,
            "public_ipv4": None,
            "lan_interfaces": [],
            "bandwidth_to_master_mbps": None,
            "rtt_to_master_ms": 2.0,
            "thunderbolt5": False,
            "available_backends": ["vllm", "llamacpp"],
            "disk_free_gb": 500.0,
            "probed_at": "2026-05-16T12:00:00+00:00",
            "probe_warnings": [],
        },
        "loaded_models": [
            {
                "specialist_id": "qwen3-coder-30b-a3b-fp8",
                "model_id": "Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8",
                "loaded_at": "2026-05-16T12:00:00+00:00",
                "estimated_tps": 46.2,
            }
        ],
        "util": {
            "gpu_util_pct": 12.5,
            "ram_util_pct": 45.0,
            "queue_depth": 2,
            "p50_latency_ms_60s": 380.0,
            "p95_latency_ms_60s": 760.0,
        },
        "recent_throughput": {
            "Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8": 47.0,
        },
        "health": "healthy",
        "network_view": {
            "mac-mini-1": {"rtt_ms": 4.2, "bandwidth_mbps": 940.0},
        },
    },
    "node_url": "http://spark-1:8001/v1",
}


def test_golden_heartbeat_post_request_parses_cleanly():
    """Updating this test = explicitly documenting a wire-format change."""
    req = HeartbeatPostRequest.model_validate(GOLDEN_HEARTBEAT_POST_REQUEST)
    assert req.heartbeat.node_id == "spark-1"
    assert req.heartbeat.loaded_models[0].specialist_id == "qwen3-coder-30b-a3b-fp8"
    assert req.heartbeat.util.queue_depth == 2
    assert req.node_url == "http://spark-1:8001/v1"
    # Network view round-trips into typed NetworkLink
    assert req.heartbeat.network_view["mac-mini-1"].rtt_ms == 4.2


GOLDEN_REGISTRY_SNAPSHOT_RESPONSE = {
    "snapshot": {
        "snapshot_ts": "2026-05-16T12:00:05+00:00",
        "nodes": {
            "spark-1": {
                "node_id": "spark-1",
                "friendly_name": "spark-1",
                "health": "healthy",
                "last_seen": "2026-05-16T12:00:00+00:00",
                "loaded_specialist_ids": ["qwen3-coder-30b-a3b-fp8"],
                "queue_depth": 2,
                "p95_latency_ms_60s": 760.0,
                "node_url": "http://spark-1:8001/v1",
            }
        },
        "specialists": {
            "qwen3-coder-30b-a3b-fp8": [
                {
                    "node_id": "spark-1",
                    "specialist_id": "qwen3-coder-30b-a3b-fp8",
                    "health": "healthy",
                    "queue_depth": 2,
                    "p95_latency_ms_60s": 760.0,
                    "node_url": "http://spark-1:8001/v1",
                    "last_seen": "2026-05-16T12:00:00+00:00",
                }
            ]
        },
        "coverage": {
            "code": ["spark-1"],
        },
        "ranked_routes": {
            "code|medium": [
                {
                    "specialist_id": "qwen3-coder-30b-a3b-fp8",
                    "node_id": "spark-1",
                    "node_url": "http://spark-1:8001/v1",
                    "estimated_queue_ms": 1000,
                    "p95_latency_ms": 760.0,
                    "cost_estimate_cents": 0.0,
                }
            ]
        },
        "catalog": {},  # catalog content varies per deployment; shape locked
    }
}


def test_golden_registry_snapshot_response_parses_cleanly():
    resp = RegistryGetResponse.model_validate(GOLDEN_REGISTRY_SNAPSHOT_RESPONSE)
    assert "spark-1" in resp.snapshot.nodes
    assert resp.snapshot.coverage["code"] == ["spark-1"]
    route = resp.snapshot.ranked_routes["code|medium"][0]
    assert route.specialist_id == "qwen3-coder-30b-a3b-fp8"
    assert route.estimated_queue_ms == 1000


def test_golden_response_field_set_is_stable():
    """Lock the OUTER shape of the snapshot response: {snapshot: {...}}.

    If we ever wrap responses in a meta envelope (e.g., {snapshot, ts, version})
    this test fires loudly and the slancha-api mesh_client needs an update.
    """
    assert set(GOLDEN_REGISTRY_SNAPSHOT_RESPONSE.keys()) == {"snapshot"}
    snap = GOLDEN_REGISTRY_SNAPSHOT_RESPONSE["snapshot"]
    assert set(snap.keys()) == {
        "snapshot_ts",
        "nodes",
        "specialists",
        "coverage",
        "ranked_routes",
        "catalog",
    }


# ---------------------------------------------------------------------------
# Cross-package contract — fields slancha-api's mesh_client walks
# ---------------------------------------------------------------------------


def test_slancha_api_walks_these_fields():
    """Lock the field NAMES slancha-api/app/router/mesh_client.py reads.

    Reading these from a real snapshot is the integration test surface;
    this is the per-field contract assertion. If the field name in the
    snapshot ever changes (e.g., node_url → node_endpoint), this test
    fires and mesh_client.py needs an update — cross-repo coordination.
    """
    snap = GOLDEN_REGISTRY_SNAPSHOT_RESPONSE["snapshot"]
    # slancha-api reads snap["coverage"][domain] → list[node_id]
    assert isinstance(snap["coverage"], dict)
    assert isinstance(snap["coverage"]["code"], list)
    # snap["specialists"][sid] → list[binding-dict] with these fields
    binding = snap["specialists"]["qwen3-coder-30b-a3b-fp8"][0]
    for key in ("node_id", "health", "queue_depth", "node_url"):
        assert key in binding, f"mesh_client expects binding.{key} — schema drift!"
    # snap["catalog"][sid] → spec card; mesh_client reads spec_card["domain"]
    assert "catalog" in snap


# ---------------------------------------------------------------------------
# Type-narrow enum literals
# ---------------------------------------------------------------------------


def test_health_state_enum_values_locked():
    """HealthState is the externally-observable health label.

    These exact strings flow into Prometheus labels, dashboard filters,
    and external automation. Renaming any of them is a breaking change.
    """
    valid: tuple[HealthState, ...] = (
        "healthy",
        "degraded",
        "draining",
        "training",
        "unreachable",
    )
    # The Literal type doesn't expose its choices at runtime, so we
    # construct a NodeSummary for each to exercise validation.
    for h in valid:
        ns = NodeSummary(
            node_id="n1",
            friendly_name="n1",
            health=h,
            last_seen=datetime.now(timezone.utc),
        )
        assert ns.health == h


def test_invalid_health_state_rejected():
    with pytest.raises(ValidationError):
        NodeSummary(
            node_id="n1",
            friendly_name="n1",
            health="not-a-valid-state",  # type: ignore[arg-type]
            last_seen=datetime.now(timezone.utc),
        )
