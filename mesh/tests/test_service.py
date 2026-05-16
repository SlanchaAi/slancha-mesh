"""FastAPI subapp tests — auth, endpoints, registry round-trip."""

from __future__ import annotations

from datetime import datetime

import pytest
from fastapi.testclient import TestClient

from mesh.models import NetworkLink, NodeHeartbeat
from mesh.registry import HeartbeatPostRequest, MeshRegistry
from mesh.service import NODE_TOKEN_ENV, create_mesh_app
from mesh.tests.conftest import make_heartbeat


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _new_client(monkeypatch, token: str | None = None, registry: MeshRegistry | None = None) -> TestClient:
    """Build a TestClient with the env primed for the desired auth posture."""
    if token is None:
        monkeypatch.delenv(NODE_TOKEN_ENV, raising=False)
    else:
        monkeypatch.setenv(NODE_TOKEN_ENV, token)
    return TestClient(create_mesh_app(registry=registry))


def _hb_payload(node, ts: datetime, catalog) -> dict:
    hb = make_heartbeat(node, ts, ["qwen3-math-7b-q4"], catalog)
    req = HeartbeatPostRequest(heartbeat=hb, node_url=f"http://{node.node_id}:8001/v1")
    return req.model_dump(mode="json")


# ---------------------------------------------------------------------------
# /health — no auth ever
# ---------------------------------------------------------------------------


def test_health_no_auth_required_dev_mode(monkeypatch):
    client = _new_client(monkeypatch)
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["auth_required"] is False


def test_health_no_auth_required_even_when_token_set(monkeypatch):
    client = _new_client(monkeypatch, token="secret-123")
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["auth_required"] is True


# ---------------------------------------------------------------------------
# Auth — disabled (dev mode)
# ---------------------------------------------------------------------------


def test_dev_mode_heartbeat_no_token_accepted(monkeypatch, spark_node, catalog, fresh_now):
    client = _new_client(monkeypatch)
    resp = client.post("/heartbeat", json=_hb_payload(spark_node, fresh_now, catalog))
    assert resp.status_code == 200
    assert resp.json()["ack"] is True


def test_dev_mode_empty_env_treated_as_disabled(monkeypatch, spark_node, catalog, fresh_now):
    client = _new_client(monkeypatch, token="")
    resp = client.post("/heartbeat", json=_hb_payload(spark_node, fresh_now, catalog))
    assert resp.status_code == 200


# ---------------------------------------------------------------------------
# Auth — enforcement (env set)
# ---------------------------------------------------------------------------


def test_auth_required_missing_header_401(monkeypatch, spark_node, catalog, fresh_now):
    client = _new_client(monkeypatch, token="secret-123")
    resp = client.post("/heartbeat", json=_hb_payload(spark_node, fresh_now, catalog))
    assert resp.status_code == 401
    assert "WWW-Authenticate" in resp.headers
    assert resp.headers["WWW-Authenticate"].startswith("Bearer")


def test_auth_required_malformed_header_401(monkeypatch, spark_node, catalog, fresh_now):
    client = _new_client(monkeypatch, token="secret-123")
    resp = client.post(
        "/heartbeat",
        json=_hb_payload(spark_node, fresh_now, catalog),
        headers={"Authorization": "Token secret-123"},
    )
    assert resp.status_code == 401


def test_auth_required_wrong_token_403(monkeypatch, spark_node, catalog, fresh_now):
    client = _new_client(monkeypatch, token="secret-123")
    resp = client.post(
        "/heartbeat",
        json=_hb_payload(spark_node, fresh_now, catalog),
        headers={"Authorization": "Bearer wrong-token"},
    )
    assert resp.status_code == 403


def test_auth_required_valid_token_200(monkeypatch, spark_node, catalog, fresh_now):
    client = _new_client(monkeypatch, token="secret-123")
    resp = client.post(
        "/heartbeat",
        json=_hb_payload(spark_node, fresh_now, catalog),
        headers={"Authorization": "Bearer secret-123"},
    )
    assert resp.status_code == 200


# ---------------------------------------------------------------------------
# /registry — round-trip
# ---------------------------------------------------------------------------


def test_registry_empty(monkeypatch):
    client = _new_client(monkeypatch)
    resp = client.get("/registry")
    assert resp.status_code == 200
    snap = resp.json()["snapshot"]
    assert snap["nodes"] == {}
    assert snap["specialists"] == {}


def test_registry_sees_posted_heartbeat(monkeypatch, spark_node, catalog, fresh_now):
    client = _new_client(monkeypatch)
    client.post("/heartbeat", json=_hb_payload(spark_node, fresh_now, catalog))

    resp = client.get("/registry")
    assert resp.status_code == 200
    snap = resp.json()["snapshot"]
    assert spark_node.node_id in snap["nodes"]
    assert snap["nodes"][spark_node.node_id]["node_url"] == f"http://{spark_node.node_id}:8001/v1"
    assert "qwen3-math-7b-q4" in snap["specialists"]


# ---------------------------------------------------------------------------
# /probe-network — aggregates network_view from heartbeats
# ---------------------------------------------------------------------------


def test_probe_network_empty(monkeypatch):
    client = _new_client(monkeypatch)
    resp = client.post("/probe-network")
    assert resp.status_code == 200
    body = resp.json()
    assert body["nodes_observed"] == 0
    assert body["network_views"] == {}


def test_probe_network_returns_views_from_heartbeats(
    monkeypatch, spark_node, mac_mini_node, catalog, fresh_now
):
    reg = MeshRegistry(catalog=catalog)
    hb_spark = make_heartbeat(spark_node, fresh_now, ["qwen3-math-7b-q4"], catalog)
    hb_spark_with_view = NodeHeartbeat(
        node_id=hb_spark.node_id,
        ts=hb_spark.ts,
        hardware=hb_spark.hardware,
        loaded_models=hb_spark.loaded_models,
        util=hb_spark.util,
        health=hb_spark.health,
        network_view={mac_mini_node.node_id: NetworkLink(rtt_ms=4.2, bandwidth_mbps=940.0)},
    )
    reg.record_heartbeat(
        HeartbeatPostRequest(heartbeat=hb_spark_with_view, node_url="http://spark-1:8001/v1")
    )

    client = TestClient(create_mesh_app(registry=reg))
    monkeypatch.delenv(NODE_TOKEN_ENV, raising=False)
    resp = client.post("/probe-network")
    assert resp.status_code == 200
    body = resp.json()
    assert body["nodes_observed"] == 1
    assert spark_node.node_id in body["network_views"]
    peer_view = body["network_views"][spark_node.node_id][mac_mini_node.node_id]
    assert peer_view["rtt_ms"] == 4.2
    assert peer_view["bandwidth_mbps"] == 940.0


# ---------------------------------------------------------------------------
# /allocate — runs cluster allocator
# ---------------------------------------------------------------------------


def test_allocate_empty_registry(monkeypatch):
    client = _new_client(monkeypatch)
    resp = client.post("/allocate", json={"strategy": "tiered"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["strategy"] == "tiered"
    assert body["suggestions"] == {}


def test_allocate_with_heartbeats(monkeypatch, spark_node, catalog, fresh_now):
    client = _new_client(monkeypatch)
    client.post("/heartbeat", json=_hb_payload(spark_node, fresh_now, catalog))
    resp = client.post("/allocate", json={"strategy": "tiered"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["strategy"] == "tiered"
    assert spark_node.node_id in body["suggestions"]


def test_allocate_default_strategy(monkeypatch):
    client = _new_client(monkeypatch)
    resp = client.post("/allocate", json={})
    assert resp.status_code == 200
    assert resp.json()["strategy"] == "tiered"


# ---------------------------------------------------------------------------
# Registry injection — shared state across mounts
# ---------------------------------------------------------------------------


def test_injected_registry_shared(monkeypatch, spark_node, catalog, fresh_now):
    shared = MeshRegistry(catalog=catalog)
    monkeypatch.delenv(NODE_TOKEN_ENV, raising=False)

    client_a = TestClient(create_mesh_app(registry=shared))
    client_b = TestClient(create_mesh_app(registry=shared))

    client_a.post("/heartbeat", json=_hb_payload(spark_node, fresh_now, catalog))

    resp_b = client_b.get("/registry")
    assert resp_b.status_code == 200
    assert spark_node.node_id in resp_b.json()["snapshot"]["nodes"]


def test_default_registry_isolated(monkeypatch, spark_node, catalog, fresh_now):
    """create_mesh_app() with no registry gets a fresh one per call."""
    monkeypatch.delenv(NODE_TOKEN_ENV, raising=False)
    client_a = TestClient(create_mesh_app())
    client_b = TestClient(create_mesh_app())

    client_a.post("/heartbeat", json=_hb_payload(spark_node, fresh_now, catalog))

    resp_b = client_b.get("/registry")
    assert resp_b.json()["snapshot"]["nodes"] == {}
