"""Traffic-cluster → experiment-spec generator — the ignition stage (issue #87).

The self-improving loop's fusion interface (`docs/GATE-CONTRACT.md`) decouples
three stages: **ignition (generator) ⊥ execution+safety (runner, #82) ⊥
promotion (gate)**. The runner and gate live here; this module is the missing
ignition stage — the piece that turns a window of live, graded traffic into a
queued, gated experiment spec the loop-runner executes unmodified.

**Where the substrate lives (issue #87 decision: bundle, don't vendor).** The
clustering substrate — the mmBERT embeddings on each trace, KMeans-per-route,
stable cluster identity + centroid matching — already exists in the **public
`slancha-local`** package (`slancha_local.train.cluster.cluster_by_route`). The
operator's call was "everything in the container": rather than copy ~1000 lines
of ML substrate into this repo (a second parity-mirror to drift, after the gate
mirror), slancha-local is installed *into the same container* as a dependency
(the ``[loop]`` extra) and this module is a **thin adapter** over it. Single
source of truth for the generator; one deployable unit.

This adapter does three things on top of slancha-local's clustering:
  1. applies the **ignition gate** (GATE-CONTRACT: ``n_traces >= 500``,
     centroid ``drift < 0.15`` across ``>= 3`` consecutive windows, and
     **no healthy champion** for the cluster — never retrain a winner);
  2. emits a **GATE-CONTRACT experiment spec** per qualifying cluster, with the
     cluster centroid as the frozen judge anchor (binding #7: "the cluster
     centroid *is* the frozen judge" — demand defines the eval, Goodhart-
     resistant by construction); and
  3. enqueues it via :func:`mesh.loop_runner.enqueue` (which dedups by spec id,
     so re-running generation on an unchanged cluster is a no-op).

Heavy deps (numpy / scikit-learn / slancha-local) are imported **lazily** inside
the default clustering seam, and the clustering function is **injectable**
(``cluster_fn``), so this module and its tests import and run with none of them
installed — the same discipline as the lazy ``torch`` import in
``mesh.training`` and the pure-Python gate/spotcheck.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from mesh.loop_runner import enqueue

# ───────────────────────────── ignition thresholds ───────────────────────────
# GATE-CONTRACT "Ignition gate" defaults (field-tested from forge throughput).
MIN_TRACES = 500          # rolling-window volume floor; below → route to base
MAX_DRIFT = 0.15          # centroid cosine drift ceiling for "settled"
MIN_STABLE_WINDOWS = 3    # consecutive low-drift windows required to ignite


# ─────────────────────────────── small math (pure) ───────────────────────────


def _as_floats(vec: Sequence[float] | Any) -> list[float]:
    """Coerce a centroid (np.ndarray, list, tuple, …) to a plain float list —
    keeps this module numpy-free; an ndarray is iterable so list() suffices."""
    return [float(x) for x in vec]


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine similarity of two equal-length vectors; 0.0 if either is zero or
    lengths differ (treated as 'no match' → maximal drift)."""
    av, bv = _as_floats(a), _as_floats(b)
    if len(av) != len(bv) or not av:
        return 0.0
    dot = sum(x * y for x, y in zip(av, bv))
    na = math.sqrt(sum(x * x for x in av))
    nb = math.sqrt(sum(y * y for y in bv))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def _drift(prev: Sequence[float] | None, cur: Sequence[float]) -> float:
    """Centroid drift = 1 − cosine(prev, cur). No prior → 1.0 (max drift): a
    cluster's first sighting is never 'settled'."""
    if prev is None:
        return 1.0
    return 1.0 - _cosine(prev, cur)


def centroid_ref(centroid: Sequence[float]) -> str:
    """Content-hash of a centroid → ``frozen://sha256:<64hex>`` (GATE-CONTRACT
    "frozen refs are content-hashes"). Rounded before hashing so the same
    centroid hashes equal across float-repr noise; this ref keys judge-match on
    the frozen bytes, so a swapped judge = a new ref = judge-match fires."""
    payload = json.dumps([round(x, 8) for x in _as_floats(centroid)])
    return "frozen://sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


# ─────────────────────────────── the ignition gate ───────────────────────────


@dataclass(frozen=True)
class IgnitionDecision:
    """Why a cluster did (or didn't) earn a train slot — auditable like a
    PromotionVerdict, but for the *ignition* stage."""

    ignite: bool
    reasons: tuple[str, ...]
    n_traces: int
    drift: float
    stable_windows: int


def evaluate_ignition(
    *,
    n_traces: int,
    drift: float,
    stable_windows: int,
    has_healthy_champion: bool,
) -> IgnitionDecision:
    """Apply the three GATE-CONTRACT ignition thresholds. Ignite iff ALL hold.

    A cluster ignites only when it is high-volume, settled, and not already
    served by a winning specialist — mirroring the promotion gate's
    "clear every check or don't" posture, one stage earlier.
    """
    reasons: list[str] = []
    if n_traces < MIN_TRACES:
        reasons.append(f"volume {n_traces} below floor {MIN_TRACES}")
    if drift >= MAX_DRIFT:
        reasons.append(f"drift {drift:.3f} >= {MAX_DRIFT} (not settled)")
    if stable_windows < MIN_STABLE_WINDOWS:
        reasons.append(
            f"only {stable_windows} consecutive low-drift window(s), "
            f"need {MIN_STABLE_WINDOWS}"
        )
    if has_healthy_champion:
        reasons.append("cluster already has a healthy champion (never retrain a winner)")
    return IgnitionDecision(
        ignite=not reasons,
        reasons=tuple(reasons),
        n_traces=n_traces,
        drift=drift,
        stable_windows=stable_windows,
    )


# ─────────────────────────── drift state (persisted) ─────────────────────────


def load_drift_state(path: Path) -> dict[str, Any]:
    """Per-cluster centroid history → `{key: {centroid: [...], stable: int}}`.
    Missing/empty file → empty state."""
    if not path.exists():
        return {}
    text = path.read_text().strip()
    return json.loads(text) if text else {}


def save_drift_state(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, ensure_ascii=False, indent=2))


def update_drift(
    state: dict[str, Any], key: str, centroid: Sequence[float]
) -> tuple[float, int]:
    """Advance one cluster's drift history by one window; return (drift,
    stable_windows). First sighting seeds the baseline (stable=1, drift=1.0 so
    it can't ignite yet). A low-drift window increments the consecutive count;
    a high-drift window resets it to 1 (this window restarts the streak)."""
    cur = _as_floats(centroid)
    prior = state.get(key)
    prev_centroid = prior.get("centroid") if prior else None
    drift = _drift(prev_centroid, cur)
    if prev_centroid is None:
        stable = 1
    elif drift < MAX_DRIFT:
        stable = int(prior.get("stable", 0)) + 1
    else:
        stable = 1  # streak broken; current window is the new baseline window
    state[key] = {"centroid": cur, "stable": stable}
    return drift, stable


# ───────────────────────────────── spec emit ─────────────────────────────────


@dataclass(frozen=True)
class GateBindingDefaults:
    """Defaults for the spec's GATE-CONTRACT `gate` binding. Tunable per
    deployment; the runner uses its own `GateThresholds` at decision time, so
    these are the portability + audit binding (GATE-CONTRACT: the `gate` block
    is authoritative for audit)."""

    primary: str = "mean_holdout_score"
    axes: tuple[str, ...] = ("per_domain_score", "coherence")
    floor: float = 1.0
    min_gain: float = 0.0
    min_n: int = 100
    judge_grader: str = "qwen3-8b"
    min_champion_lifetime_s: int = 3600
    decisive_gain: float = 2.0


def _short(ref: str) -> str:
    """Last 12 hex of a `frozen://sha256:` / `sha256:` ref, for ids + judge key."""
    return ref.rsplit(":", 1)[-1][:12]


def build_spec(
    *,
    route: str,
    cluster_id: int,
    n_traces: int,
    drift: float,
    exemplar_trace_ids: list[Any],
    centroid: Sequence[float],
    base_model_id: str,
    gate_defaults: GateBindingDefaults = GateBindingDefaults(),
    priority: int = 5,
) -> dict[str, Any]:
    """Build one GATE-CONTRACT experiment-spec line for a qualifying cluster.

    The spec id embeds the centroid content-hash, so an unchanged cluster emits
    an identical id and :func:`mesh.loop_runner.enqueue` dedups it — generation
    is idempotent. The cluster centroid is both `centroid_ref` and the
    `holdout_ref` and is folded into `judge_model` (binding #7: the centroid IS
    the frozen judge — keying judge-match on the frozen bytes).
    """
    cref = centroid_ref(centroid)
    short = _short(cref)
    task = f"cluster:{route}:{cluster_id}"
    return {
        "id": f"ft_{route}-c{cluster_id}@{short}",
        "type": "train",
        "priority": priority,
        "source": "traffic_cluster",
        "generator": {
            "cluster_id": f"{route}:{cluster_id}",
            "centroid_ref": cref,
            "n_traces": n_traces,
            "drift": round(drift, 4),
            "exemplar_trace_ids": list(exemplar_trace_ids),
        },
        "cmd": (
            f"slancha-mesh train --task {task} --base {base_model_id} "
            f"--corpus-from-cluster {route}:{cluster_id}"
        ),
        "gate": {
            "task": task,
            "primary": gate_defaults.primary,
            "axes": list(gate_defaults.axes),
            "floor": gate_defaults.floor,
            "min_gain": gate_defaults.min_gain,
            "min_n": gate_defaults.min_n,
            "judge_model": f"{gate_defaults.judge_grader}@{short}",
            "min_champion_lifetime_s": gate_defaults.min_champion_lifetime_s,
            "decisive_gain": gate_defaults.decisive_gain,
            "holdout_ref": cref,
        },
        "env": {"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"},
        "status": "pending",
    }


# ─────────────────────────── the clustering seam ──────────────────────────────


@dataclass
class _Cluster:
    """Minimal cluster shape the adapter consumes. slancha-local's
    ``TraceCluster`` is duck-compatible (same attrs); tests use this directly."""

    route: str
    cluster_id: int
    trace_indices: list[int]
    centroid: Sequence[float]


class ClusterFn(Protocol):
    """The clustering seam: (traces) → clusters. Default wraps slancha-local's
    ``cluster_by_route``; tests inject a fake so no numpy/sklearn is needed."""

    def __call__(self, traces: list[dict[str, Any]]) -> list[Any]: ...


def _default_cluster_fn(
    *, node_capacity: int | None, snapshot_path: Path | None
) -> ClusterFn:
    """Build the production clustering seam: slancha-local's KMeans-per-route
    with stable cluster identity. Imported lazily so this module loads without
    the ``[loop]`` extra; a clear error names the fix if it's missing."""

    def _run(traces: list[dict[str, Any]]) -> list[Any]:
        try:
            from slancha_local.train.cluster import (  # type: ignore
                ClusterSnapshot,
                cluster_by_route,
            )
        except ImportError as e:  # pragma: no cover - exercised only sans extra
            raise RuntimeError(
                "the traffic-cluster generator needs the slancha-local substrate. "
                'Install the loop extra:  pip install -e ".[loop]"  (bundles the '
                "public slancha-local package — issue #87 decision).  Original "
                f"import error: {e}"
            ) from e
        prior = None
        if snapshot_path is not None and snapshot_path.exists():
            prior = ClusterSnapshot.load(snapshot_path)
        return cluster_by_route(traces, prior=prior, node_capacity=node_capacity)

    return _run


# ───────────────────────────────── orchestration ─────────────────────────────


class ChampionPredicate(Protocol):
    """`(cluster_key) -> True if a HEALTHY champion already serves it`. The
    default binding reads a champion registry; injected in tests. 'Healthy'
    folds the GATE-CONTRACT 'no champion OR champion regressing' rule into one
    predicate: return False when there is no champion *or* the existing one is
    regressing on a fresh cluster eval (→ re-ignite)."""

    def __call__(self, cluster_key: str) -> bool: ...


def _no_champion(cluster_key: str) -> bool:
    """Conservative default: assume no healthy champion (always eligible). A
    real deployment injects a registry-backed predicate."""
    return False


@dataclass
class GenerationResult:
    """What one generation pass produced — emitted specs + per-cluster ignition
    decisions (including the rejected ones, for audit)."""

    specs: list[dict[str, Any]] = field(default_factory=list)
    decisions: list[tuple[str, IgnitionDecision]] = field(default_factory=list)


def generate(
    traces: list[dict[str, Any]],
    *,
    queue_path: Path,
    drift_state_path: Path,
    base_model_id: str = "Qwen/Qwen3-8B",
    has_healthy_champion: ChampionPredicate = _no_champion,
    cluster_fn: ClusterFn | None = None,
    node_capacity: int | None = None,
    snapshot_path: Path | None = None,
    gate_defaults: GateBindingDefaults = GateBindingDefaults(),
    n_exemplars: int = 8,
) -> GenerationResult:
    """One ignition pass: cluster a graded-traffic window, ignition-gate each
    cluster, and enqueue a GATE-CONTRACT spec for every cluster that qualifies.

    `traces` are graded-trace dicts in slancha-local's shape (each carrying
    `embedding_b64` + `classifier.route`); the default `cluster_fn` clusters
    them via slancha-local. Returns every cluster's ignition decision (emitted
    or not) for audit; only ignited clusters are enqueued.

    Idempotent: the spec id embeds the centroid hash and `enqueue` dedups by id,
    so re-running on an unchanged window enqueues nothing new.
    """
    cf = cluster_fn or _default_cluster_fn(
        node_capacity=node_capacity, snapshot_path=snapshot_path
    )
    clusters = cf(traces)
    drift_state = load_drift_state(drift_state_path)
    result = GenerationResult()

    for c in clusters:
        if c.centroid is None:  # a degenerate single-member cluster w/o centroid
            continue
        key = f"{c.route}:{c.cluster_id}"
        drift, stable = update_drift(drift_state, key, c.centroid)
        n_traces = len(c.trace_indices)
        decision = evaluate_ignition(
            n_traces=n_traces,
            drift=drift,
            stable_windows=stable,
            has_healthy_champion=has_healthy_champion(key),
        )
        result.decisions.append((key, decision))
        if not decision.ignite:
            continue
        exemplars = [
            traces[i].get("id", i) for i in c.trace_indices[:n_exemplars]
        ]
        spec = build_spec(
            route=c.route,
            cluster_id=c.cluster_id,
            n_traces=n_traces,
            drift=drift,
            exemplar_trace_ids=exemplars,
            centroid=c.centroid,
            base_model_id=base_model_id,
            gate_defaults=gate_defaults,
        )
        if enqueue(queue_path, spec):
            result.specs.append(spec)

    save_drift_state(drift_state_path, drift_state)
    return result


__all__ = [
    "MAX_DRIFT",
    "MIN_STABLE_WINDOWS",
    "MIN_TRACES",
    "ChampionPredicate",
    "ClusterFn",
    "GateBindingDefaults",
    "GenerationResult",
    "IgnitionDecision",
    "build_spec",
    "centroid_ref",
    "evaluate_ignition",
    "generate",
    "load_drift_state",
    "save_drift_state",
    "update_drift",
]
