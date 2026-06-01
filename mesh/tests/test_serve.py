"""ServeDaemon tests — heartbeat construction + failure handling.

Real vLLM lifecycle is in `test_integration_vllm.py` (guarded). Here we
test the daemon's contract against `NullBackend` only.
"""

from __future__ import annotations

import time

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


def test_daemon_idle_detector_reports_training_health_via_heartbeat():
    """Spec §7 + ServeDaemon integration: when the detector transitions
    to TRAINING, heartbeat.health flips to 'training' so the router
    drops hot-interactive traffic.
    """
    from datetime import datetime as _dt, timedelta as _td, timezone as _tz

    from mesh.idle import IdleDetector
    from mesh.models import NodeUtilization as _NU

    card = _card()
    be = NullBackend(card=card)
    detector = IdleDetector()

    # Drive detector to TRAINING via synthetic clock BEFORE wiring it to
    # the daemon. heartbeat() calls observe() with real `now`, which would
    # otherwise overwrite our synthetic _idle_since. The integration test
    # boundary is "heartbeat reads detector.health()"; the state transition
    # is unit-tested in test_idle.py.
    anchor = _dt(2026, 5, 16, 12, 0, 0, tzinfo=_tz.utc)
    detector.observe(_NU(gpu_util_pct=0.0, queue_depth=0), anchor)
    detector.observe(_NU(gpu_util_pct=0.0, queue_depth=0), anchor + _td(seconds=61))
    assert detector.should_start_training()
    detector.mark_training_started(anchor + _td(seconds=61))

    daemon = ServeDaemon(backends=[be], probe=_probe(), idle_detector=detector)
    daemon.start(wait_ready=True, ready_timeout=1.0)

    # Idle utilization observed in heartbeat → detector stays TRAINING
    # (observe during TRAINING does not transition; spec §7 contract).
    hb = daemon.heartbeat()
    assert hb.health == "training"
    daemon.stop()


def test_daemon_without_idle_detector_is_backwards_compatible():
    """v0.0.3-shape callers (no idle_detector) get unchanged behavior."""
    card = _card()
    be = NullBackend(card=card)
    daemon = ServeDaemon(backends=[be], probe=_probe())  # no idle_detector
    daemon.start(wait_ready=True, ready_timeout=1.0)
    hb = daemon.heartbeat()
    assert hb.health == "healthy"
    daemon.stop()


def test_daemon_spawns_training_thread_on_ready_edge(tmp_path):
    """Integration (#39): detector READY_TO_TRAIN + training config →
    daemon spawns TrainingPass thread on next heartbeat. Thread runs
    to completion; subsequent heartbeat reaps it → COOLDOWN."""
    from datetime import datetime as _dt, timedelta as _td, timezone as _tz
    import time as _time

    from mesh.idle import IdleDetector
    from mesh.models import NodeUtilization as _NU
    from mesh.replay_store import TrafficReplayStore

    card = _card()
    be = NullBackend(card=card)
    detector = IdleDetector()
    store = TrafficReplayStore(max_size=10)
    for i in range(3):
        store.add(f"p{i}", f"r{i}", "code", "easy")

    daemon = ServeDaemon(
        backends=[be],
        probe=_probe(),
        idle_detector=detector,
        training_replay_store=store,
        training_checkpoint_dir=tmp_path,
        training_kwargs={"n_steps_planned": 3, "per_step_sleep_s": 0.001},
    )
    daemon.start(wait_ready=True, ready_timeout=1.0)

    anchor = _dt(2026, 5, 16, 12, 0, 0, tzinfo=_tz.utc)
    detector.observe(_NU(gpu_util_pct=0.0, queue_depth=0), anchor)
    detector.observe(_NU(gpu_util_pct=0.0, queue_depth=0), anchor + _td(seconds=61))
    assert detector.should_start_training()

    # First heartbeat: spawns training thread, transitions to TRAINING.
    hb1 = daemon.heartbeat()
    assert hb1.health == "training"
    assert daemon._training_thread is not None

    _time.sleep(0.1)
    assert not daemon._training_thread.is_alive()

    # Second heartbeat: reaps thread → COOLDOWN.
    hb2 = daemon.heartbeat()
    assert detector.state.value == "cooldown"
    assert hb2.health == "healthy"
    assert daemon.last_checkpoint_path is not None
    assert daemon.last_checkpoint_path.exists()

    daemon.stop()


def test_daemon_preempts_training_when_traffic_returns(tmp_path):
    """Backend reports queue_depth > 0 mid-training → daemon signals
    preempt → training thread yields with preempted=True."""
    from datetime import datetime as _dt, timedelta as _td, timezone as _tz
    import time as _time
    import json

    from mesh.idle import IdleDetector
    from mesh.models import NodeUtilization as _NU
    from mesh.replay_store import TrafficReplayStore

    card = _card()

    class BusyBackend(NullBackend):
        def utilization(self) -> dict:
            return {"queue_depth": 5}

    be_idle = NullBackend(card=card)
    detector = IdleDetector()
    store = TrafficReplayStore(max_size=10)
    for i in range(3):
        store.add(f"p{i}", f"r{i}", "code", "easy")

    daemon = ServeDaemon(
        backends=[be_idle],
        probe=_probe(),
        idle_detector=detector,
        training_replay_store=store,
        training_checkpoint_dir=tmp_path,
        training_kwargs={"n_steps_planned": 5000, "per_step_sleep_s": 0.001},
    )
    daemon.start(wait_ready=True, ready_timeout=1.0)

    anchor = _dt(2026, 5, 16, 12, 0, 0, tzinfo=_tz.utc)
    detector.observe(_NU(gpu_util_pct=0.0, queue_depth=0), anchor)
    detector.observe(_NU(gpu_util_pct=0.0, queue_depth=0), anchor + _td(seconds=61))
    daemon.heartbeat()  # spawns
    assert daemon._training_thread.is_alive()

    # Swap to busy backend → next heartbeat sees queue_depth=5 → preempts.
    daemon.backends = [BusyBackend(card=card, base_url="http://x")]
    daemon.backends[0].start()
    _time.sleep(0.02)
    daemon.heartbeat()  # observes busy → signal_preempt

    daemon._training_thread.join(timeout=2.0)
    assert not daemon._training_thread.is_alive()

    daemon.heartbeat()  # reap → COOLDOWN
    assert detector.state.value == "cooldown"

    ck = daemon.last_checkpoint_path
    assert ck is not None
    meta = json.loads((ck / "meta.json").read_text())
    assert meta["preempted"] is True
    assert 0 < meta["n_steps_completed"] < 5000

    daemon.stop()


def test_daemon_training_disabled_without_config():
    """idle_detector set + training config NOT set → detector observes,
    but daemon never spawns training thread (back-compat)."""
    from datetime import datetime as _dt, timedelta as _td, timezone as _tz

    from mesh.idle import IdleDetector
    from mesh.models import NodeUtilization as _NU

    card = _card()
    be = NullBackend(card=card)
    detector = IdleDetector()
    daemon = ServeDaemon(
        backends=[be],
        probe=_probe(),
        idle_detector=detector,
        # training_replay_store + checkpoint_dir intentionally None
    )
    daemon.start(wait_ready=True, ready_timeout=1.0)

    anchor = _dt(2026, 5, 16, 12, 0, 0, tzinfo=_tz.utc)
    detector.observe(_NU(gpu_util_pct=0.0, queue_depth=0), anchor)
    detector.observe(_NU(gpu_util_pct=0.0, queue_depth=0), anchor + _td(seconds=61))
    assert detector.should_start_training()

    hb = daemon.heartbeat()
    assert detector.state.value == "ready_to_train"  # never advanced
    assert hb.health == "healthy"
    assert daemon._training_thread is None

    daemon.stop()


def test_daemon_stop_preempts_inflight_training(tmp_path):
    """daemon.stop() during training → signals preempt, joins cleanly."""
    from datetime import datetime as _dt, timedelta as _td, timezone as _tz

    from mesh.idle import IdleDetector
    from mesh.models import NodeUtilization as _NU
    from mesh.replay_store import TrafficReplayStore

    card = _card()
    be = NullBackend(card=card)
    detector = IdleDetector()
    store = TrafficReplayStore(max_size=10)
    store.add("p", "r", "code", "easy")

    daemon = ServeDaemon(
        backends=[be],
        probe=_probe(),
        idle_detector=detector,
        training_replay_store=store,
        training_checkpoint_dir=tmp_path,
        training_kwargs={"n_steps_planned": 50000, "per_step_sleep_s": 0.001},
    )
    daemon.start(wait_ready=True, ready_timeout=1.0)
    anchor = _dt(2026, 5, 16, 12, 0, 0, tzinfo=_tz.utc)
    detector.observe(_NU(gpu_util_pct=0.0, queue_depth=0), anchor)
    detector.observe(_NU(gpu_util_pct=0.0, queue_depth=0), anchor + _td(seconds=61))
    daemon.heartbeat()  # spawns training (would take 50s without preempt)

    assert daemon._training_thread.is_alive()
    daemon.stop(timeout=5.0)
    assert daemon._training_thread is None


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


def test_build_backend_llamacpp_without_gguf_falls_back_to_null():
    """A `llamacpp` card missing `gguf_path` falls back to NullBackend with a
    hint, so a mixed catalog still boots (mirrors the ollama_tag-missing path).
    """
    card = SpecialistCard(
        model_id="test/llama",
        specialist_id="llama-test",
        domain="general",
        difficulty_tiers=["easy"],
        required_backend="llamacpp",  # wired, but needs gguf_path
        storage_gb=4.0,
        runtime_gb=6.0,
        min_vram_gb=8.0,
        context_window=8192,
        n_layers=32,
        estimated_tps_at={"gb10": 50.0},
        # gguf_path intentionally unset
    )
    be = build_backend(card, port=9001)
    assert be.name == "null"
    assert be.card.specialist_id == "llama-test"


def test_build_backend_llamacpp_with_gguf_returns_llamacpp_backend():
    """A `llamacpp` card with `gguf_path` set wires to LlamaCppBackend."""
    card = SpecialistCard(
        model_id="test/llama",
        specialist_id="llama-test",
        domain="general",
        difficulty_tiers=["easy"],
        required_backend="llamacpp",
        storage_gb=4.0,
        runtime_gb=6.0,
        min_vram_gb=8.0,
        context_window=8192,
        n_layers=32,
        estimated_tps_at={"gb10": 50.0},
        gguf_path="/models/llama.gguf",
    )
    be = build_backend(card, port=9001)
    assert be.name == "llamacpp"
    assert be.base_url == "http://127.0.0.1:9001"


def test_build_backend_mlx_without_repo_falls_back_to_null():
    """An `mlx` card missing `mlx_repo` falls back to NullBackend with a hint."""
    card = SpecialistCard(
        model_id="test/qwen",
        specialist_id="qwen-mlx-test",
        domain="general",
        difficulty_tiers=["easy"],
        required_backend="mlx",
        storage_gb=4.0,
        runtime_gb=6.0,
        min_vram_gb=8.0,
        context_window=8192,
        n_layers=32,
        estimated_tps_at={"m4_pro": 50.0},
        # mlx_repo intentionally unset
    )
    be = build_backend(card, port=9002)
    assert be.name == "null"


def test_build_backend_mlx_with_repo_returns_mlx_backend():
    """An `mlx` card with `mlx_repo` set wires to MLXBackend (no process spawned)."""
    card = SpecialistCard(
        model_id="test/qwen",
        specialist_id="qwen-mlx-test",
        domain="general",
        difficulty_tiers=["easy"],
        required_backend="mlx",
        storage_gb=4.0,
        runtime_gb=6.0,
        min_vram_gb=8.0,
        context_window=8192,
        n_layers=32,
        estimated_tps_at={"m4_pro": 50.0},
        mlx_repo="mlx-community/Qwen2.5-Coder-7B-Instruct-4bit",
    )
    be = build_backend(card, port=9002)
    assert be.name == "mlx"
    assert be.base_url == "http://127.0.0.1:9002"


def test_build_backend_ollama_route_with_tag_returns_ollama_backend():
    """An ollama card with `ollama_tag` set wires to `OllamaBackend` —
    the per-specialist port is informational (Ollama multiplexes on one port).
    """
    card = SpecialistCard(
        model_id="Qwen/Qwen2.5-Coder-7B-Instruct",
        specialist_id="qwen2.5-coder-7b",
        domain="code",
        difficulty_tiers=["medium"],
        required_backend="ollama",
        storage_gb=5.0,
        runtime_gb=6.0,
        min_vram_gb=8.0,
        context_window=32768,
        n_layers=28,
        estimated_tps_at={"gb10": 40.0},
        ollama_tag="qwen2.5-coder:7b",
    )
    be = build_backend(card, port=8013)  # 8013 is informational for Ollama
    assert be.name == "ollama"
    # Default Ollama daemon port — overridable via OLLAMA_PORT env (covered below).
    assert be.base_url.endswith(":11434")


def test_build_backend_ollama_route_without_tag_falls_back_to_null():
    """A card declaring ollama but missing `ollama_tag` shouldn't crash the
    daemon — fall back to NullBackend so a mixed catalog still boots and the
    operator gets a hint to add the tag.
    """
    card = SpecialistCard(
        model_id="Qwen/Qwen2.5-Coder-7B-Instruct",
        specialist_id="qwen2.5-coder-7b-untagged",
        domain="code",
        difficulty_tiers=["medium"],
        required_backend="ollama",
        storage_gb=5.0,
        runtime_gb=6.0,
        min_vram_gb=8.0,
        context_window=32768,
        n_layers=28,
        estimated_tps_at={"gb10": 40.0},
        # ollama_tag intentionally unset
    )
    be = build_backend(card, port=8013)
    assert be.name == "null"


def test_build_backend_ollama_respects_OLLAMA_PORT_env(monkeypatch):
    """Operators with a non-default Ollama bind shouldn't have to hand-edit cards."""
    monkeypatch.setenv("OLLAMA_PORT", "11500")
    card = SpecialistCard(
        model_id="Qwen/Qwen2.5-Coder-7B-Instruct",
        specialist_id="qwen2.5-coder-7b",
        domain="code",
        difficulty_tiers=["medium"],
        required_backend="ollama",
        storage_gb=5.0,
        runtime_gb=6.0,
        min_vram_gb=8.0,
        context_window=32768,
        n_layers=28,
        estimated_tps_at={"gb10": 40.0},
        ollama_tag="qwen2.5-coder:7b",
    )
    be = build_backend(card, port=8013)
    assert be.base_url.endswith(":11500")


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
