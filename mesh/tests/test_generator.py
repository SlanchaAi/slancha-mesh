"""Tests for mesh.generator (issue #87 — ignition stage / traffic→spec).

All tests inject a fake clustering seam (`cluster_fn`) so they run with no
numpy / scikit-learn / slancha-local installed — the adapter and its tests are
heavy-dep-free, the same discipline as the lazy-torch training path.
"""

from __future__ import annotations

from pathlib import Path

from mesh.generator import (
    MAX_DRIFT,
    MIN_STABLE_WINDOWS,
    MIN_TRACES,
    GateBindingDefaults,
    _Cluster,
    build_spec,
    centroid_ref,
    evaluate_ignition,
    generate,
    load_drift_state,
    save_drift_state,
    update_drift,
)
from mesh.loop_runner import _pick_pending, read_queue


# ───────────────────────────── ignition gate ─────────────────────────────────


def test_ignition_all_conditions_met():
    d = evaluate_ignition(
        n_traces=600, drift=0.05, stable_windows=3, has_healthy_champion=False
    )
    assert d.ignite
    assert d.reasons == ()


def test_ignition_rejects_low_volume():
    d = evaluate_ignition(
        n_traces=499, drift=0.05, stable_windows=3, has_healthy_champion=False
    )
    assert not d.ignite
    assert any("volume" in r for r in d.reasons)


def test_ignition_rejects_high_drift():
    d = evaluate_ignition(
        n_traces=600, drift=0.2, stable_windows=3, has_healthy_champion=False
    )
    assert not d.ignite
    assert any("drift" in r for r in d.reasons)


def test_ignition_rejects_too_few_stable_windows():
    d = evaluate_ignition(
        n_traces=600, drift=0.05, stable_windows=2, has_healthy_champion=False
    )
    assert not d.ignite
    assert any("window" in r for r in d.reasons)


def test_ignition_rejects_when_healthy_champion_exists():
    """Never retrain a winning specialist."""
    d = evaluate_ignition(
        n_traces=600, drift=0.05, stable_windows=3, has_healthy_champion=True
    )
    assert not d.ignite
    assert any("champion" in r for r in d.reasons)


# ─────────────────────────────── drift state ─────────────────────────────────


def test_update_drift_first_sighting_not_settled():
    state: dict = {}
    drift, stable = update_drift(state, "code:0", [1.0, 0.0, 0.0])
    assert drift == 1.0  # no prior → max drift
    assert stable == 1


def test_update_drift_settles_over_consecutive_low_drift_windows():
    state: dict = {}
    c = [1.0, 0.0, 0.0]
    update_drift(state, "code:0", c)              # window 1: stable=1
    d2, s2 = update_drift(state, "code:0", c)     # window 2: same centroid
    d3, s3 = update_drift(state, "code:0", c)     # window 3
    assert d2 < MAX_DRIFT and s2 == 2
    assert d3 < MAX_DRIFT and s3 == MIN_STABLE_WINDOWS


def test_update_drift_resets_streak_on_high_drift():
    state: dict = {}
    update_drift(state, "code:0", [1.0, 0.0, 0.0])
    update_drift(state, "code:0", [1.0, 0.0, 0.0])  # stable=2
    drift, stable = update_drift(state, "code:0", [0.0, 1.0, 0.0])  # orthogonal flip
    assert drift >= MAX_DRIFT
    assert stable == 1  # streak broken, current window is the new baseline


def test_drift_state_persists_round_trip(tmp_path: Path):
    p = tmp_path / "drift.json"
    state = {"code:0": {"centroid": [1.0, 0.0], "stable": 2}}
    save_drift_state(p, state)
    assert load_drift_state(p) == state
    assert load_drift_state(tmp_path / "missing.json") == {}


# ───────────────────────────── centroid ref / hash ───────────────────────────


def test_centroid_ref_is_frozen_content_hash():
    r = centroid_ref([1.0, 2.0, 3.0])
    assert r.startswith("frozen://sha256:")
    assert r == centroid_ref([1.0, 2.0, 3.0])          # deterministic
    assert r != centroid_ref([1.0, 2.0, 3.5])          # content-sensitive


# ─────────────────────────────── spec emission ───────────────────────────────


def test_build_spec_matches_gate_contract_schema():
    spec = build_spec(
        route="code",
        cluster_id=7,
        n_traces=1840,
        drift=0.11,
        exemplar_trace_ids=["t1", "t2"],
        centroid=[1.0, 0.0, 0.0],
        base_model_id="Qwen/Qwen3-8B",
    )
    # Top-level GATE-CONTRACT fields.
    for k in ("id", "type", "priority", "source", "generator", "cmd", "gate", "env", "status"):
        assert k in spec, f"missing spec field {k}"
    assert spec["type"] == "train"
    assert spec["source"] == "traffic_cluster"
    assert spec["status"] == "pending"
    # Generator (ignition) payload.
    g = spec["generator"]
    assert g["cluster_id"] == "code:7"
    assert g["centroid_ref"].startswith("frozen://sha256:")
    assert g["n_traces"] == 1840
    # Gate binding: centroid IS the frozen judge (binding #7).
    gate = spec["gate"]
    assert gate["task"] == "cluster:code:7"
    assert gate["holdout_ref"] == g["centroid_ref"]
    short = g["centroid_ref"].rsplit(":", 1)[-1][:12]
    assert gate["judge_model"].endswith("@" + short)
    # id embeds the centroid hash → idempotent enqueue.
    assert spec["id"].endswith("@" + short)
    # env carries the single-token alloc form (torch 2.11 safe).
    assert spec["env"]["PYTORCH_CUDA_ALLOC_CONF"] == "expandable_segments:True"


def test_build_spec_respects_gate_defaults_override():
    spec = build_spec(
        route="math", cluster_id=1, n_traces=600, drift=0.0,
        exemplar_trace_ids=[], centroid=[1.0],
        base_model_id="b",
        gate_defaults=GateBindingDefaults(min_n=500, judge_grader="custom-judge"),
    )
    assert spec["gate"]["min_n"] == 500
    assert spec["gate"]["judge_model"].startswith("custom-judge@")


# ─────────────────────── end-to-end generation (fake clusterer) ──────────────


def _big_cluster(centroid):
    return _Cluster(
        route="code", cluster_id=0, trace_indices=list(range(MIN_TRACES + 100)),
        centroid=centroid,
    )


def _traces(n):
    return [{"id": f"t{i}"} for i in range(n)]


def test_generate_ignites_only_after_cluster_settles(tmp_path: Path):
    """A big, stable cluster needs MIN_STABLE_WINDOWS consecutive low-drift
    windows before a spec is emitted — never ignite on a transient spike."""
    q = tmp_path / "queue.jsonl"
    ds = tmp_path / "drift.json"
    traces = _traces(MIN_TRACES + 100)
    centroid = [1.0, 0.0, 0.0]
    fake = lambda _t: [_big_cluster(centroid)]  # noqa: E731

    emitted = []
    for _ in range(MIN_STABLE_WINDOWS):
        r = generate(traces, queue_path=q, drift_state_path=ds, cluster_fn=fake)
        emitted.append(len(r.specs))
    # windows 1..(N-1) settle silently; only the N-th ignites.
    assert emitted == [0] * (MIN_STABLE_WINDOWS - 1) + [1]
    specs = read_queue(q)
    assert len(specs) == 1
    assert specs[0]["generator"]["cluster_id"] == "code:0"


def test_generate_is_idempotent_after_ignition(tmp_path: Path):
    """Re-running on an unchanged cluster enqueues nothing new (enqueue dedups
    by the centroid-hashed id)."""
    q = tmp_path / "queue.jsonl"
    ds = tmp_path / "drift.json"
    traces = _traces(MIN_TRACES + 100)
    fake = lambda _t: [_big_cluster([1.0, 0.0, 0.0])]  # noqa: E731
    for _ in range(MIN_STABLE_WINDOWS + 2):
        generate(traces, queue_path=q, drift_state_path=ds, cluster_fn=fake)
    assert len(read_queue(q)) == 1  # one spec total, despite repeated passes


def test_generate_skips_small_cluster(tmp_path: Path):
    q = tmp_path / "queue.jsonl"
    ds = tmp_path / "drift.json"
    small = _Cluster(route="code", cluster_id=0, trace_indices=list(range(10)),
                     centroid=[1.0, 0.0])
    fake = lambda _t: [small]  # noqa: E731
    for _ in range(MIN_STABLE_WINDOWS + 1):
        r = generate(_traces(10), queue_path=q, drift_state_path=ds, cluster_fn=fake)
    assert r.specs == []
    assert not q.exists() or read_queue(q) == []
    # The rejection is still recorded for audit.
    assert r.decisions and not r.decisions[0][1].ignite


def test_generate_skips_cluster_with_healthy_champion(tmp_path: Path):
    q = tmp_path / "queue.jsonl"
    ds = tmp_path / "drift.json"
    traces = _traces(MIN_TRACES + 100)
    fake = lambda _t: [_big_cluster([1.0, 0.0, 0.0])]  # noqa: E731
    for _ in range(MIN_STABLE_WINDOWS + 1):
        r = generate(
            traces, queue_path=q, drift_state_path=ds, cluster_fn=fake,
            has_healthy_champion=lambda _k: True,
        )
    assert r.specs == []
    assert not q.exists() or read_queue(q) == []


def test_emitted_spec_is_loop_runner_consumable(tmp_path: Path):
    """The whole point: the loop-runner (#82) picks the emitted spec unmodified."""
    q = tmp_path / "queue.jsonl"
    ds = tmp_path / "drift.json"
    traces = _traces(MIN_TRACES + 100)
    fake = lambda _t: [_big_cluster([1.0, 0.0, 0.0])]  # noqa: E731
    for _ in range(MIN_STABLE_WINDOWS):
        generate(traces, queue_path=q, drift_state_path=ds, cluster_fn=fake)
    specs = read_queue(q)
    idx = _pick_pending(specs)
    assert idx == 0
    assert specs[idx]["type"] == "train"
    assert specs[idx]["status"] == "pending"
