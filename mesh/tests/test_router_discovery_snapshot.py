"""Tests for the pull-mode glue in `mesh.router_app`:

- `discovery_to_snapshot` — translates a `DiscoveryResult` into a
  `RegistrySnapshot` the router can consume.
- `_RefreshingSnapshot` — caches a snapshot + refreshes it on a fixed
  cadence in a daemon thread.

These two together are what lets `slancha-mesh router` run without a
central registry: discovery is the source of truth, the router is its
OpenAI-compatible surface.
"""

from __future__ import annotations

import threading
import time

from mesh.discovery import DiscoveredSpecialist, DiscoveryResult
from mesh.models import SpecialistCard
from mesh.router_app import _RefreshingSnapshot, discovery_to_snapshot


def _card(specialist_id: str, *, ollama_tag: str | None = "qwen2.5-coder:7b") -> SpecialistCard:
    return SpecialistCard(
        model_id="Qwen/Qwen2.5-Coder-7B-Instruct",
        specialist_id=specialist_id,
        domain="code",
        difficulty_tiers=["medium"],
        required_backend="ollama",
        ollama_tag=ollama_tag,
        storage_gb=5.0,
        runtime_gb=6.0,
        min_vram_gb=8.0,
        context_window=32768,
        n_layers=28,
        estimated_tps_at={"gb10": 40.0},
    )


def test_discovery_to_snapshot_empty_returns_empty_snapshot():
    snap = discovery_to_snapshot(DiscoveryResult())
    assert snap.specialists == {}
    assert snap.nodes == {}
    assert snap.catalog == {}


def test_discovery_to_snapshot_synthesizes_bindings_per_node_url():
    """Each node_url in a discovered specialist becomes one NodeBinding."""
    spec = DiscoveredSpecialist(
        specialist_id="qwen2.5-coder-7b-q4-ollama",
        model_id="Qwen/Qwen2.5-Coder-7B-Instruct",
        domain="code",
        node_urls=(
            "http://192.168.1.10:11434",
            "http://192.168.1.20:11434",
        ),
    )
    result = DiscoveryResult(
        specialists={spec.specialist_id: spec},
        reachable=["192.168.1.10", "192.168.1.20"],
    )
    snap = discovery_to_snapshot(result)
    bindings = snap.specialists[spec.specialist_id]
    assert len(bindings) == 2
    assert {b.node_url for b in bindings} == {
        "http://192.168.1.10:11434",
        "http://192.168.1.20:11434",
    }
    # Node ids must be deterministic + human-meaningful (host:port).
    assert {b.node_id for b in bindings} == {"192.168.1.10:11434", "192.168.1.20:11434"}
    for b in bindings:
        assert b.health == "healthy"  # we just successfully fetched /models
        assert b.queue_depth == 0
        assert b.p95_latency_ms_60s is None


def test_discovery_to_snapshot_carries_local_catalog_for_rewrite():
    """The local catalog is what gives the router an `ollama_tag` to
    rewrite `model` against — otherwise `model = specialist_id` flows
    through unchanged.
    """
    spec = DiscoveredSpecialist(
        specialist_id="qwen2.5-coder-7b-q4-ollama",
        node_urls=("http://10.0.0.5:11434",),
    )
    result = DiscoveryResult(specialists={spec.specialist_id: spec})
    catalog = [_card("qwen2.5-coder-7b-q4-ollama")]
    snap = discovery_to_snapshot(result, catalog=catalog)
    assert "qwen2.5-coder-7b-q4-ollama" in snap.catalog
    assert snap.catalog["qwen2.5-coder-7b-q4-ollama"].ollama_tag == "qwen2.5-coder:7b"


def test_discovery_to_snapshot_keeps_node_summary_per_node():
    """One NodeSummary per derived node_id, even when several specialists
    share the same node URL."""
    spec_a = DiscoveredSpecialist(specialist_id="a", node_urls=("http://h:11434",))
    spec_b = DiscoveredSpecialist(specialist_id="b", node_urls=("http://h:11434",))
    result = DiscoveryResult(specialists={"a": spec_a, "b": spec_b})
    snap = discovery_to_snapshot(result)
    assert list(snap.nodes) == ["h:11434"]
    assert snap.nodes["h:11434"].health == "healthy"


# ---------------------------------------------------------------------------
# _RefreshingSnapshot
# ---------------------------------------------------------------------------


def test_refreshing_snapshot_cold_get_runs_refresher_synchronously():
    """First .get() before background loop runs must still return data."""
    calls = []

    def refresher() -> DiscoveryResult:
        calls.append(1)
        return DiscoveryResult(
            specialists={"x": DiscoveredSpecialist(specialist_id="x", node_urls=("http://h:80",))}
        )

    holder = _RefreshingSnapshot(refresher, refresh_s=60.0)  # tall interval — no auto-refresh
    snap = holder.get()
    assert "x" in snap.specialists
    assert len(calls) == 1


def test_refreshing_snapshot_caches_after_first_get():
    """Subsequent .get() before the next refresh tick returns the cached
    snapshot — does NOT call the refresher again."""
    calls = []

    def refresher() -> DiscoveryResult:
        calls.append(1)
        return DiscoveryResult()

    holder = _RefreshingSnapshot(refresher, refresh_s=60.0)
    holder.get()
    holder.get()
    holder.get()
    assert len(calls) == 1


def test_refreshing_snapshot_background_loop_swaps_snapshot():
    """Start the loop; the snapshot the router sees must reflect new
    discovery results without anyone calling .get() again."""
    state = {"version": 0}
    ready = threading.Event()

    def refresher() -> DiscoveryResult:
        state["version"] += 1
        sid = f"spec-v{state['version']}"
        if state["version"] >= 2:
            ready.set()
        return DiscoveryResult(
            specialists={sid: DiscoveredSpecialist(specialist_id=sid, node_urls=("http://h:80",))}
        )

    holder = _RefreshingSnapshot(refresher, refresh_s=0.05)
    holder.start()
    try:
        assert ready.wait(timeout=2.0), "refresher loop never reached version 2"
        snap = holder.get()
        assert any(sid.startswith("spec-v") for sid in snap.specialists)
    finally:
        holder.stop()


def test_refreshing_snapshot_swallows_refresher_exceptions():
    """A flaky refresh must not crash the daemon thread — the router has
    to keep serving the most recent snapshot it has."""
    state = {"first": True}

    def refresher() -> DiscoveryResult:
        if state["first"]:
            state["first"] = False
            return DiscoveryResult(
                specialists={
                    "good": DiscoveredSpecialist(specialist_id="good", node_urls=("http://h:80",))
                }
            )
        raise RuntimeError("transient discovery failure")

    holder = _RefreshingSnapshot(refresher, refresh_s=0.02)
    snap0 = holder.get()
    assert "good" in snap0.specialists
    holder.start()
    # Let the background loop fail a couple of times.
    time.sleep(0.15)
    holder.stop()
    # The snapshot from the cold .get() must still be present — failures
    # didn't overwrite it.
    snap1 = holder.get()
    assert "good" in snap1.specialists


def test_refreshing_snapshot_stop_terminates_thread():
    """Hygienic shutdown for `slancha-mesh router` SIGINT handling."""
    holder = _RefreshingSnapshot(lambda: DiscoveryResult(), refresh_s=0.05)
    holder.start()
    holder.stop(timeout=1.0)
    assert holder._thread is not None
    assert not holder._thread.is_alive()
