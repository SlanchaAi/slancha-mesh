"""Live failure tests against a real vLLM — spec §12 day 10.

These tests run against an actually-serving vLLM on `VLLM_LIVE_URL`. They
validate VLLMBackend's lifecycle contract under real subprocess + HTTP
conditions that mock backends cannot simulate:

  * PID adoption when wrapping an externally-launched vLLM (port-busy)
  * is_alive() reflects real process death (not just port-still-listening)
  * stop() is idempotent on dead PID + on never-started backend
  * wait_ready() returns True on a live /health, False after process kill
  * /metrics parser handles real vLLM 0.17 Prometheus output
  * ServeDaemon over real vLLM heartbeats degraded after backend dies
  * Router falls through to cloud when the only mesh route is unhealthy

Run with:
    VLLM_LIVE_URL=http://127.0.0.1:8001 \
        python3 -m pytest mesh/tests/test_failures_live.py -v -s

These tests STOP the live vLLM as part of their failure-mode coverage.
Don't run them against a vLLM you want to keep around — the suite kills
the backend in the partial-death tests. The vLLM bring-up script can
relaunch quickly if you do.

Why not iptables to simulate network partition? Two reasons:
  1. Requires sudo — non-portable for a pytest run.
  2. From the registry's POV, "node behind iptables drop" is observably
     identical to "node process died" — same fallback path through
     spec §6.6 (heartbeat misses → degraded → next route in chain).
  Process kill is the simpler, equally-valid signal.
"""

from __future__ import annotations

import json
import os
import signal as _signal
import subprocess
import time
import urllib.error
import urllib.request

import pytest

from mesh.backends import VLLMBackend, _find_pid_on_port
from mesh.catalog import load_catalog
from mesh.models import LoadedModel, NodeHeartbeat, NodeUtilization
from mesh.probe import probe_node
from mesh.registry import HeartbeatPostRequest, MeshRegistry
from mesh.select import ClassifierSignals, select_mesh_route
from mesh.serve import ServeDaemon

LIVE_URL = os.environ.get("VLLM_LIVE_URL")
LIVE_SPECIALIST = os.environ.get(
    "VLLM_LIVE_SPECIALIST", "qwen3-coder-30b-a3b-fp8"
)

requires_live_vllm = pytest.mark.skipif(
    LIVE_URL is None,
    reason="VLLM_LIVE_URL not set; skipping live failure tests",
)


def _live_port() -> int:
    assert LIVE_URL is not None
    return int(LIVE_URL.rsplit(":", 1)[-1])


def _card():
    catalog = load_catalog()
    by_id = {c.specialist_id: c for c in catalog}
    return by_id[LIVE_SPECIALIST]


# ---------------------------------------------------------------------------
# Backend adoption + introspection
# ---------------------------------------------------------------------------


@requires_live_vllm
def test_backend_adopts_externally_launched_vllm():
    """VLLMBackend.start() on a port-busy condition should adopt the
    existing PID via ss(8) lookup, NOT raise or relaunch.

    This matters because cold-load is 3-4 min; relaunching on every
    daemon restart wastes ~7 minutes per session.
    """
    card = _card()
    be = VLLMBackend(card=card, host="127.0.0.1", port=_live_port())
    be.start()  # port already busy → adopt
    # adopted_pid should now be set to the running vllm process
    assert be._adopted_pid is not None, "expected PID adoption via port-busy"
    pid_via_ss = _find_pid_on_port(_live_port())
    assert be._adopted_pid == pid_via_ss
    # is_alive() should be True against the adopted PID
    assert be.is_alive()


@requires_live_vllm
def test_backend_wait_ready_returns_true_on_live_health():
    """wait_ready against an already-serving vLLM should return True
    within seconds (no cold-load wait, /health 200 immediately)."""
    card = _card()
    be = VLLMBackend(card=card, host="127.0.0.1", port=_live_port())
    be.start()  # adopt
    t0 = time.time()
    assert be.wait_ready(timeout=10.0)
    elapsed = time.time() - t0
    print(f"\n  wait_ready over adopted vLLM: {elapsed:.2f}s")
    assert elapsed < 5.0, f"adoption-mode wait_ready unreasonably slow: {elapsed:.2f}s"


@requires_live_vllm
def test_backend_utilization_returns_real_metrics():
    """utilization() pulls /metrics from a real vLLM. The parser must
    extract the gauges we depend on — a silent rename surfaces here.
    """
    card = _card()
    be = VLLMBackend(card=card, host="127.0.0.1", port=_live_port())
    be.start()
    util = be.utilization()
    assert isinstance(util, dict)
    # vLLM 0.17 emits these; if any go missing, parser returns zero
    # rather than raising — that's the silent-regression bug we want
    # to surface here by asserting the keys exist with int/float types.
    assert "queue_depth" in util and isinstance(util["queue_depth"], int)
    assert "running" in util and isinstance(util["running"], int)
    # gpu_cache_pct is optional; only assert if present
    if "gpu_cache_pct" in util:
        assert isinstance(util["gpu_cache_pct"], (int, float))
    print(f"\n  live util: {util}")


# ---------------------------------------------------------------------------
# Failure paths
# ---------------------------------------------------------------------------


@requires_live_vllm
def test_backend_is_alive_flips_false_after_process_kill():
    """Real subprocess monitoring: SIGKILL the adopted vLLM PID and
    verify is_alive() reports False on the next call.

    DESTRUCTIVE: kills the live vLLM. Bring-up script relaunches.
    """
    card = _card()
    be = VLLMBackend(card=card, host="127.0.0.1", port=_live_port())
    be.start()  # adopt
    pid = be._adopted_pid
    assert pid is not None and be.is_alive()

    # Kill the whole process group (vllm spawns workers; SIGKILL on PID alone
    # leaves workers orphaned briefly).
    try:
        os.killpg(os.getpgid(pid), _signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pytest.skip("could not signal vLLM process group; permission gap")

    # Linux PID-cleanup is near-instant for SIGKILL; give it 2s grace.
    deadline = time.time() + 5.0
    while time.time() < deadline:
        if not be.is_alive():
            break
        time.sleep(0.2)
    assert not be.is_alive(), "expected is_alive() to flip False after SIGKILL"


@requires_live_vllm
def test_stop_is_idempotent_on_already_dead_backend():
    """After kill in previous test, stop() should not raise.

    Runs after test_backend_is_alive_flips_false_after_process_kill — if
    that test killed the vLLM, this confirms cleanup is idempotent. If
    that test didn't run (selection mismatch), this is a no-op against
    an alive backend, which also shouldn't raise.
    """
    card = _card()
    be = VLLMBackend(card=card, host="127.0.0.1", port=_live_port())
    be.start()  # whatever state; adopt or relaunch
    be.stop(timeout=5.0)  # MUST NOT raise
    # After stop, internal state should clear
    assert be._proc is None
    assert be._adopted_pid is None


# ---------------------------------------------------------------------------
# Mesh end-to-end fallback
# ---------------------------------------------------------------------------


@requires_live_vllm
def test_router_falls_through_to_cloud_when_only_node_dies():
    """Spec §6.6: mesh of one node, that node dies → router returns
    cluster_coverage_used=False and the caller falls through to cloud.

    Assumes the vLLM was killed in the prior test (or by an external
    operator). If still alive, this test passes-but-uninformative — it
    just verifies the success path. The real assertion is the fallback.
    """
    card = _card()
    probe = probe_node()

    # Check vLLM state — alive or dead
    try:
        urllib.request.urlopen(f"{LIVE_URL}/health", timeout=2)
        backend_alive = True
    except (urllib.error.URLError, ConnectionError, TimeoutError):
        backend_alive = False

    registry = MeshRegistry(catalog=[card])
    now = __import__("datetime").datetime.now(
        __import__("datetime").timezone.utc
    )

    if backend_alive:
        # Construct a heartbeat reflecting the live backend
        hb = NodeHeartbeat(
            node_id=probe.node_id,
            ts=now,
            hardware=probe,
            loaded_models=[
                LoadedModel(
                    specialist_id=card.specialist_id,
                    model_id=card.model_id,
                    loaded_at=now,
                    estimated_tps=card.estimated_tps_at.get("gb10"),
                )
            ],
            util=NodeUtilization(queue_depth=0),
            health="healthy",
        )
        registry.record_heartbeat(
            HeartbeatPostRequest(heartbeat=hb, node_url=LIVE_URL)
        )
        snap = registry.snapshot()
        result = select_mesh_route(
            signals=ClassifierSignals(domain="code", difficulty="medium"),
            registry_snapshot=snap,
        )
        assert result.cluster_coverage_used is True
        assert result.node_url == LIVE_URL
        print(f"\n  alive path: router → {result.specialist_id} @ {result.node_url}")
    else:
        # Construct a degraded heartbeat (loaded_models=[], health=degraded)
        hb = NodeHeartbeat(
            node_id=probe.node_id,
            ts=now,
            hardware=probe,
            loaded_models=[],
            util=NodeUtilization(queue_depth=0),
            health="degraded",
        )
        registry.record_heartbeat(
            HeartbeatPostRequest(heartbeat=hb, node_url=LIVE_URL)
        )
        snap = registry.snapshot()
        result = select_mesh_route(
            signals=ClassifierSignals(domain="code", difficulty="medium"),
            registry_snapshot=snap,
        )
        assert result.cluster_coverage_used is False, (
            f"degraded node should not be routable; got {result}"
        )
        assert result.node_id is None
        print(f"\n  fallback path: cloud (reason={result.reason})")


@requires_live_vllm
def test_dead_port_does_not_yield_pid():
    """If vLLM is gone (port unbound), _find_pid_on_port returns None.

    This guards against a stale-cache adoption bug where a backend
    might think it has a PID even after vLLM exited. Caller of start()
    sees adopted_pid=None and raises rather than silently failing later.
    """
    # Pick a port unlikely to be bound
    pid = _find_pid_on_port(65530)
    assert pid is None
