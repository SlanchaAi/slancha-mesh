"""ServeDaemon tests — heartbeat construction + failure handling.

Real vLLM lifecycle is in `test_integration_vllm.py` (guarded). Here we
test the daemon's contract against `NullBackend` only.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone

import pytest

from mesh.backends import NullBackend
from mesh.models import NodeProbe, SpecialistCard
from mesh.registry import MeshRegistry
from mesh.select import ClassifierSignals, select_mesh_route
from mesh.serve import ServeDaemon, build_backend, build_daemon


def _probe() -> NodeProbe:
    return NodeProbe(
        node_id="test-node",
        friendly_name="test-node",
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


def _card(spec_id: str = "qwen3-coder-30b-a3b-fp8", domain: str = "code") -> SpecialistCard:
    return SpecialistCard(
        model_id=f"test/{spec_id}",
        specialist_id=spec_id,
        domain=domain,
        difficulty_tiers=["easy", "medium", "hard"],
        required_backend="vllm",
        storage_gb=30.0,
        runtime_gb=70.0,
        min_vram_gb=30.0,
        context_window=8192,
        n_layers=48,
        estimated_tps_at={"gb10": 30.0},
    )


def test_daemon_starts_and_heartbeats_with_null_backend():
    card = _card()
    be = NullBackend(card=card, base_url="http://127.0.0.1:8001")
    daemon = ServeDaemon(backends=[be], probe=_probe())
    assert daemon.start(wait_ready=True, ready_timeout=1.0)
    hb = daemon.heartbeat()
    assert hb.node_id == "test-node"
    assert hb.health == "healthy"
    assert [lm.specialist_id for lm in hb.loaded_models] == [card.specialist_id]
    daemon.stop()
    assert not be.is_alive()


def test_daemon_serves_multiple_backends_on_distinct_ports():
    """Multi-specialist coexistence (spec §3.3): GB10 with 128GB unified mem
    runs code AND math/general simultaneously, each on its own port.

    Heartbeat reports both specialists loaded; queue depth aggregates
    across backends. Stopping the daemon stops both backends.
    """
    code_card = _card(spec_id="qwen3-coder-30b-a3b-fp8", domain="code")
    math_card = _card(spec_id="qwen3-math-7b-q4", domain="math")
    be_code = NullBackend(card=code_card, base_url="http://127.0.0.1:8001")
    be_math = NullBackend(card=math_card, base_url="http://127.0.0.1:8002")

    daemon = ServeDaemon(backends=[be_code, be_math], probe=_probe())
    assert daemon.start(wait_ready=True, ready_timeout=1.0)

    hb = daemon.heartbeat()
    loaded_ids = {lm.specialist_id for lm in hb.loaded_models}
    assert loaded_ids == {"qwen3-coder-30b-a3b-fp8", "qwen3-math-7b-q4"}
    assert hb.health == "healthy"

    # Both backends still alive
    assert be_code.is_alive()
    assert be_math.is_alive()

    daemon.stop()
    assert not be_code.is_alive()
    assert not be_math.is_alive()


def test_daemon_partial_failure_keeps_surviving_backend():
    """If one of two backends dies, daemon stays healthy on the survivor.

    Spec §6.6 fallback contract: router falls through to next route in
    chain on per-specialist failure. Daemon should NOT crash the whole
    node when ONE backend dies — heartbeat reflects reduced capacity.
    """
    code_card = _card(spec_id="qwen3-coder-30b-a3b-fp8", domain="code")
    math_card = _card(spec_id="qwen3-math-7b-q4", domain="math")
    be_code = NullBackend(card=code_card, base_url="http://127.0.0.1:8001")
    be_math = NullBackend(card=math_card, base_url="http://127.0.0.1:8002")

    daemon = ServeDaemon(backends=[be_code, be_math], probe=_probe())
    daemon.start(wait_ready=True, ready_timeout=1.0)

    be_code.stop()  # simulate code-backend OOM mid-session

    hb = daemon.heartbeat()
    loaded_ids = {lm.specialist_id for lm in hb.loaded_models}
    assert loaded_ids == {"qwen3-math-7b-q4"}, (
        f"expected only math after code died; got {loaded_ids}"
    )
    assert hb.health == "healthy", "daemon should NOT mark whole node degraded on partial death"

    daemon.stop()


def test_daemon_heartbeat_degraded_when_no_backend_alive():
    """If a backend dies, the next heartbeat reports degraded — router falls through."""
    card = _card()
    be = NullBackend(card=card)
    daemon = ServeDaemon(backends=[be], probe=_probe())
    daemon.start(wait_ready=True, ready_timeout=1.0)
    be.stop()  # simulate vLLM crash
    hb = daemon.heartbeat()
    assert hb.health == "degraded"
    assert hb.loaded_models == []


def test_daemon_posts_to_registry_and_router_routes_to_it():
    """End-to-end: spawn daemon → heartbeat → registry snapshot → router pick."""
    card = _card()
    be = NullBackend(card=card, base_url="http://127.0.0.1:8001")
    registry = MeshRegistry(catalog=[card])
    daemon = ServeDaemon(backends=[be], probe=_probe(), registry=registry)
    daemon.start(wait_ready=True, ready_timeout=1.0)
    daemon.post_heartbeat()
    daemon.post_heartbeat()
    snap = registry.snapshot()
    assert "test-node" in snap.nodes
    assert "qwen3-coder-30b-a3b-fp8" in snap.specialists

    # Real routing decision: code/easy → our backend
    result = select_mesh_route(
        signals=ClassifierSignals(domain="code", difficulty="easy"),
        registry_snapshot=snap,
    )
    assert result.cluster_coverage_used is True
    assert result.node_id == "test-node"
    assert result.specialist_id == "qwen3-coder-30b-a3b-fp8"
    assert result.node_url == "http://127.0.0.1:8001"
    daemon.stop()


def test_daemon_routes_to_cloud_when_backend_dies_mid_session():
    """Spec §6.6: 5xx / dead backend → next in fallback chain (cloud in v0.0.2)."""
    card = _card()
    be = NullBackend(card=card, base_url="http://127.0.0.1:8001")
    registry = MeshRegistry(catalog=[card])
    daemon = ServeDaemon(backends=[be], probe=_probe(), registry=registry)
    daemon.start(wait_ready=True, ready_timeout=1.0)
    daemon.post_heartbeat()

    # Backend dies; daemon heartbeats degraded
    be.stop()
    daemon.post_heartbeat()
    snap = registry.snapshot()

    # Node is marked degraded → router drops it from healthy routes → cloud
    result = select_mesh_route(
        signals=ClassifierSignals(domain="code", difficulty="easy"),
        registry_snapshot=snap,
    )
    assert result.cluster_coverage_used is False
    assert result.node_id is None
    assert "cloud" in result.reason or "no mesh route" in result.reason
    daemon.stop()


def test_daemon_routes_to_cloud_when_no_specialist_loaded():
    """Heartbeat with `loaded_models=[]` should not get routed traffic."""
    daemon = ServeDaemon(backends=[], probe=_probe())
    registry = MeshRegistry(catalog=[_card()])
    daemon.registry = registry
    daemon.post_heartbeat()
    snap = registry.snapshot()
    result = select_mesh_route(
        signals=ClassifierSignals(domain="code", difficulty="easy"),
        registry_snapshot=snap,
    )
    assert result.cluster_coverage_used is False
    assert result.node_id is None


def test_daemon_routes_around_queue_full():
    """Spec §6.3: queue > max_queue_ms drops the route, fallback to cloud."""
    card = _card()

    class BusyBackend(NullBackend):
        def utilization(self) -> dict:
            return {"queue_depth": 100, "running": 1}  # 100 * 500ms = 50s queue

    be = BusyBackend(card=card, base_url="http://127.0.0.1:8001")
    registry = MeshRegistry(catalog=[card])
    daemon = ServeDaemon(backends=[be], probe=_probe(), registry=registry)
    daemon.start(wait_ready=True, ready_timeout=1.0)
    daemon.post_heartbeat()
    snap = registry.snapshot()
    result = select_mesh_route(
        signals=ClassifierSignals(domain="code", difficulty="easy"),
        registry_snapshot=snap,
        max_queue_ms=2000,
    )
    # Queue (50s) exceeds 2s budget → cloud
    assert result.cluster_coverage_used is False
    daemon.stop()


def test_build_backend_falls_back_to_null_for_unsupported_backend():
    """v0.0.2 ships vLLM only; non-vllm cards should not crash the daemon."""
    card = SpecialistCard(
        model_id="test/llama",
        specialist_id="llama-test",
        domain="general",
        difficulty_tiers=["easy"],
        required_backend="llamacpp",  # not yet implemented
        storage_gb=4.0,
        runtime_gb=6.0,
        min_vram_gb=8.0,
        context_window=8192,
        n_layers=32,
        estimated_tps_at={"gb10": 50.0},
    )
    be = build_backend(card, port=9001)
    assert be.name == "null"
    assert be.card.specialist_id == "llama-test"


def test_build_daemon_raises_on_unknown_specialist():
    with pytest.raises(KeyError):
        build_daemon(specialist_ids=["nonexistent"], probe=_probe())


def test_heartbeat_loop_runs_in_thread_and_stops():
    """Smoke test the threaded heartbeat path used by serve.py daemons."""
    card = _card()
    be = NullBackend(card=card, base_url="http://127.0.0.1:8001")
    registry = MeshRegistry(catalog=[card])
    daemon = ServeDaemon(
        backends=[be],
        probe=_probe(),
        registry=registry,
        heartbeat_interval_s=0.05,
    )
    daemon.start(wait_ready=True, ready_timeout=1.0)
    daemon.run_in_thread()
    time.sleep(0.2)
    daemon.stop()
    # We should have sent at least 2 heartbeats in 200ms with 50ms interval.
    assert daemon.heartbeats_sent >= 2
