"""Per-specialist tailnet URLs: serve → heartbeat → registry → routing.

Two things this locks:
  1. Each loaded specialist advertises the URL of *its own* port (the
     multi-port mis-routing fix) — a node on MagicDNS with vLLM :8003 +
     HF :8004 must hand the gateway the right port per specialist.
  2. The advertised host is a tailnet name, not loopback, when a daemon
     has an advertise_host. With no advertise_host, behavior is unchanged
     (loopback) — back-compat for non-tailnet dev.
"""

from __future__ import annotations

from datetime import datetime, timezone

from mesh.backends import NullBackend, VLLMBackend
from mesh.models import LoadedModel, NodeHeartbeat, NodeProbe, NodeUtilization, SpecialistCard
from mesh.registry import HeartbeatPostRequest, MeshRegistry
from mesh.select import ClassifierSignals, select_mesh_route
from mesh.serve import ServeDaemon, build_backend
from mesh.tailnet import TailnetConfig


def _probe() -> NodeProbe:
    return NodeProbe(
        node_id="gb10-1",
        friendly_name="gb10-1",
        chip="NVIDIA GB10",
        arch="aarch64",
        cuda_capability="12.1",
        ram_total_gb=128.0,
        ram_available_gb=110.0,
        unified_memory=True,
        memory_bandwidth_gbs=273.0,
        available_backends=["vllm"],
        disk_free_gb=500.0,
    )


def _card(spec_id: str, domain: str) -> SpecialistCard:
    return SpecialistCard(
        model_id=f"test/{spec_id}",
        specialist_id=spec_id,
        domain=domain,
        difficulty_tiers=["easy", "medium", "hard"],
        required_backend="vllm",
        storage_gb=10.0,
        runtime_gb=20.0,
        min_vram_gb=10.0,
        context_window=8192,
        n_layers=32,
        estimated_tps_at={"gb10": 40.0},
    )


# ---------------------------------------------------------------------------
# Registry: per-loaded-model node_url honored, with node-level fallback
# ---------------------------------------------------------------------------


def test_registry_binds_per_loaded_model_url():
    """Each loaded model's own node_url lands on its binding — distinct
    ports on the same MagicDNS host route to the right backend."""
    now = datetime(2026, 5, 25, 12, 0, 0, tzinfo=timezone.utc)
    code = _card("paul-voice", "code")
    voice = _card("paul-voice-v8", "writing")
    reg = MeshRegistry(catalog=[code, voice])
    host = "gb10-1.tnet-example.ts.net"
    hb = NodeHeartbeat(
        node_id="gb10-1",
        ts=now,
        hardware=_probe(),
        loaded_models=[
            LoadedModel(specialist_id="paul-voice", model_id="test/paul-voice",
                        loaded_at=now, node_url=f"http://{host}:8003"),
            LoadedModel(specialist_id="paul-voice-v8", model_id="test/paul-voice-v8",
                        loaded_at=now, node_url=f"http://{host}:8004"),
        ],
        util=NodeUtilization(queue_depth=0),
    )
    reg.record_heartbeat(HeartbeatPostRequest(heartbeat=hb, node_url=f"http://{host}:8003"))
    snap = reg.snapshot(now=now)

    assert snap.specialists["paul-voice"][0].node_url == f"http://{host}:8003"
    assert snap.specialists["paul-voice-v8"][0].node_url == f"http://{host}:8004"


def test_registry_falls_back_to_node_level_url_when_loaded_model_url_absent():
    """Old nodes that don't set per-model node_url still resolve via the
    node-level HeartbeatPostRequest.node_url (back-compat)."""
    now = datetime(2026, 5, 25, 12, 0, 0, tzinfo=timezone.utc)
    card = _card("paul-voice", "code")
    reg = MeshRegistry(catalog=[card])
    hb = NodeHeartbeat(
        node_id="gb10-1",
        ts=now,
        hardware=_probe(),
        loaded_models=[
            LoadedModel(specialist_id="paul-voice", model_id="test/paul-voice", loaded_at=now)
        ],  # node_url omitted
        util=NodeUtilization(queue_depth=0),
    )
    reg.record_heartbeat(HeartbeatPostRequest(heartbeat=hb, node_url="http://gb10-1.ts.net:8003"))
    snap = reg.snapshot(now=now)
    assert snap.specialists["paul-voice"][0].node_url == "http://gb10-1.ts.net:8003"


# ---------------------------------------------------------------------------
# Serve: advertise_host rewrites loopback bind URLs to MagicDNS
# ---------------------------------------------------------------------------


def test_daemon_advertises_magicdns_per_specialist():
    """Two backends on distinct ports + advertise_host → heartbeat carries
    a MagicDNS URL per specialist with the correct port."""
    host = "gb10-1.tnet-example.ts.net"
    code = _card("paul-voice", "code")
    voice = _card("paul-voice-v8", "writing")
    be_code = NullBackend(card=code, base_url="http://0.0.0.0:8003")
    be_voice = NullBackend(card=voice, base_url="http://0.0.0.0:8004")
    reg = MeshRegistry(catalog=[code, voice])
    daemon = ServeDaemon(
        backends=[be_code, be_voice], probe=_probe(), registry=reg, advertise_host=host
    )
    daemon.start(wait_ready=True, ready_timeout=1.0)
    daemon.post_heartbeat()
    snap = reg.snapshot()

    assert snap.specialists["paul-voice"][0].node_url == f"http://{host}:8003"
    assert snap.specialists["paul-voice-v8"][0].node_url == f"http://{host}:8004"

    # Real routing hands the gateway the right tailnet URL.
    result = select_mesh_route(
        signals=ClassifierSignals(domain="code", difficulty="easy"),
        registry_snapshot=snap,
    )
    assert result.node_url == f"http://{host}:8003"
    daemon.stop()


def test_daemon_without_advertise_host_keeps_loopback():
    """Back-compat: no advertise_host → loopback URL unchanged."""
    card = _card("paul-voice", "code")
    be = NullBackend(card=card, base_url="http://127.0.0.1:8003")
    reg = MeshRegistry(catalog=[card])
    daemon = ServeDaemon(backends=[be], probe=_probe(), registry=reg)  # no advertise_host
    daemon.start(wait_ready=True, ready_timeout=1.0)
    daemon.post_heartbeat()
    snap = reg.snapshot()
    assert snap.specialists["paul-voice"][0].node_url == "http://127.0.0.1:8003"
    daemon.stop()


# ---------------------------------------------------------------------------
# build_backend: bind host plumbing (0.0.0.0 for tailnet reachability)
# ---------------------------------------------------------------------------


def test_build_backend_binds_to_configured_host():
    card = _card("paul-voice", "code")
    be = build_backend(card, port=8003, bind_host="0.0.0.0")
    assert isinstance(be, VLLMBackend)
    assert be.host == "0.0.0.0"
    assert be.base_url == "http://0.0.0.0:8003"


def test_build_backend_defaults_to_loopback():
    card = _card("paul-voice", "code")
    be = build_backend(card, port=8003)
    assert isinstance(be, VLLMBackend)
    assert be.host == "127.0.0.1"
