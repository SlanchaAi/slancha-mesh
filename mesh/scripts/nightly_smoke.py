"""Nightly smoke — replay a fixed corpus, diff vs prev day, alert on drift.

Run from systemd timer (see mesh/deploy/mesh-nightly-smoke.{service,timer}).
Steps:
  1. Invoke mesh.scripts.mesh_replay against `--corpus` → today's JSONL.
  2. Look up the most-recent prior run in `--history-dir`.
  3. Diff today's mesh-hit-rate + fallback-domain set vs that prior run.
  4. If Δhit-rate > alert-threshold OR new fallback-domain → write
     alert JSONL + non-zero exit code; else exit 0.
  5. Move today's replay JSONL into the history dir as `YYYY-MM-DD.jsonl`.

The diff logic is testable without invoking mesh_replay — the
detect/alert functions take parsed records, so unit tests bypass
subprocess.

Exit codes:
  0 — replay green, no alert
  1 — replay green, alert (drift detected)
  2 — replay subprocess failed (smoke itself broken; investigate)

Usage:
  python -m mesh.scripts.nightly_smoke \\
      --corpus     /path/to/preclassified.jsonl \\
      --registry-url http://localhost:8088 \\
      --history-dir ~/.local/state/slancha-mesh/replay-history \\
      [--alert-threshold-pct 10] \\
      [--token TOKEN]
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import subprocess
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from mesh.dashboard.panels import (
    DecisionRecord,
    fallback_chain_shape_histogram,
    load_replay_records,
    summary_stats,
)

DEFAULT_ALERT_THRESHOLD_PCT = 10.0
# Per-domain hit-rate drift uses the SAME threshold as aggregate by default.
# Override via run_nightly_smoke(per_domain_alert_threshold_pct=...) when a
# noisy single-domain workload should be tolerated.
DEFAULT_PER_DOMAIN_THRESHOLD_PCT = 10.0
# KL divergence over the fallback-chain shape distribution. Empirically
# stable distributions sit < 0.1 even with mild noise; > 0.3 indicates a
# real shape shift (new chain shapes appearing, large redistribution).
# Laplace smoothing constant keeps zero-prob shapes from blowing the
# logarithm to infinity.
DEFAULT_KL_THRESHOLD = 0.3
_KL_SMOOTHING = 0.5
# Per-domain drift signals require a minimum sample size in BOTH today's
# and prior's slice for the domain. Below this, the domain's hit-rate is
# too noisy to alarm on — we note it in `per_domain_drift` but skip the
# alert. Tunable per deployment.
PER_DOMAIN_MIN_SAMPLES = 5


# ---------------------------------------------------------------------------
# Drift detection
# ---------------------------------------------------------------------------


@dataclass
class DriftReport:
    """Result of comparing today's replay to a prior run.

    `alert=True` indicates the run should fire an operator-visible alert.
    `reasons` is a list of human-readable strings explaining each
    contributing signal.

    Beyond the aggregate hit-rate signal, the report carries two finer
    drift readouts (added in M2):
    - `per_domain_drift`: {domain: signed-Δpp} for every domain present
      in either run. Domains with <PER_DOMAIN_MIN_SAMPLES samples in
      either run are still reported but do NOT trigger alerts (noise
      floor protection).
    - `fallback_shape_kl`: KL(today || prior) over the fallback-chain
      shape distribution. > DEFAULT_KL_THRESHOLD signals a redistribution
      of WHERE traffic falls back to cloud (e.g., a specialist died →
      its cloud-fallback shape now dominates).
    """

    alert: bool
    today_hit_rate: float
    prior_hit_rate: float
    delta_hit_rate_pct: float
    new_fallback_domains: list[str]
    reasons: list[str]
    per_domain_drift: dict[str, float] = field(default_factory=dict)
    fallback_shape_kl: float = 0.0


def _fallback_domains(records: list[DecisionRecord]) -> set[str]:
    """Domains whose decisions fell back to cloud (mesh_hit=False)."""
    out: set[str] = set()
    for r in records:
        if r["decision"].get("mesh_hit") is False:
            domain = r.get("signals", {}).get("domain", "unknown")
            out.add(domain)
    return out


def _per_domain_hit_rates(
    records: list[DecisionRecord],
) -> dict[str, tuple[float, int]]:
    """Hit-rate + sample count per domain.

    Returns `{domain: (hit_rate, n)}` so callers can apply min-sample
    floors before alerting. Domains absent from the records are absent
    from the result; "unknown" is used when a record's signals.domain
    is missing.
    """
    counts: dict[str, list[bool]] = defaultdict(list)
    for r in records:
        domain = r.get("signals", {}).get("domain", "unknown")
        counts[domain].append(bool(r["decision"].get("mesh_hit")))
    return {
        d: (sum(hits) / len(hits) if hits else 0.0, len(hits))
        for d, hits in counts.items()
    }


def _fallback_shape_distribution(
    records: list[DecisionRecord],
) -> dict[str, float]:
    """Probability mass over fallback-chain shapes.

    Reuses `fallback_chain_shape_histogram` for shape keying so today's
    and prior's distributions are computed identically. Returns a
    {shape: probability} map summing to ~1.0; empty input → empty dict.
    """
    histogram = fallback_chain_shape_histogram(records)
    total = sum(c for _, c in histogram)
    if total == 0:
        return {}
    return {shape: count / total for shape, count in histogram}


def _kl_divergence(p: dict[str, float], q: dict[str, float]) -> float:
    """KL(p || q) over a shape distribution, with Laplace smoothing.

    Both distributions are smoothed by `_KL_SMOOTHING` over the UNION of
    their support so the log-ratio doesn't blow up on shapes present in
    only one. Returns 0.0 if both are empty (no fallback data either
    run); returns the actual divergence in nats otherwise.
    """
    support = set(p) | set(q)
    if not support:
        return 0.0
    # Smooth: each shape gets (count + smoothing) / (total + smoothing * |support|).
    # We don't have raw counts here, so reverse-engineer: a probability
    # entry of 0 becomes _KL_SMOOTHING / (|support| * _KL_SMOOTHING) after
    # smoothing, which simplifies to 1 / |support|. To make the math
    # tractable, blend each input with a uniform distribution over the
    # support at weight `_KL_SMOOTHING`. This preserves the high-prob
    # shapes' ranking while flooring zero-prob shapes.
    n = len(support)
    uniform = 1.0 / n
    smoothed_p = {
        s: (1.0 - _KL_SMOOTHING) * p.get(s, 0.0) + _KL_SMOOTHING * uniform
        for s in support
    }
    smoothed_q = {
        s: (1.0 - _KL_SMOOTHING) * q.get(s, 0.0) + _KL_SMOOTHING * uniform
        for s in support
    }
    divergence = 0.0
    for s in support:
        pi = smoothed_p[s]
        qi = smoothed_q[s]
        if pi > 0.0:
            divergence += pi * math.log(pi / qi)
    return max(0.0, divergence)


def detect_drift(
    today: list[DecisionRecord],
    prior: list[DecisionRecord] | None,
    *,
    alert_threshold_pct: float = DEFAULT_ALERT_THRESHOLD_PCT,
    per_domain_alert_threshold_pct: float = DEFAULT_PER_DOMAIN_THRESHOLD_PCT,
    fallback_shape_kl_threshold: float = DEFAULT_KL_THRESHOLD,
) -> DriftReport:
    """Compare today's replay against the prior run.

    No prior run → no alert (first-night), but report still emitted so
    history captures today's baseline.

    Aggregate Δhit-rate: today_pct - prior_pct (signed). Alerts on
    absolute > alert_threshold_pct.

    New fallback domains: domains in today's cloud-fallback set that were
    NOT in prior's set. (Removed-from-fallback domains are also relevant
    but less worrying; they're noted, not alerted.)

    Per-domain drift (M2): |Δhit-rate| > per_domain_alert_threshold_pct
    in any domain with ≥PER_DOMAIN_MIN_SAMPLES in BOTH runs. Catches
    "aggregate looks fine but one workload class collapsed" failures
    that aggregate Δ hides.

    Fallback-chain shape KL (M2): KL(today || prior) over the
    distribution of distinct fallback-chain shapes. >
    fallback_shape_kl_threshold (default 0.3 nats) signals that traffic
    is taking materially different fallback paths than yesterday — e.g.,
    a specialist died and its cloud-fallback chain now dominates.
    """
    today_stats = summary_stats(today)
    today_rate = today_stats["mesh_hit_rate"]
    today_fb = _fallback_domains(today)
    today_per_domain = _per_domain_hit_rates(today)

    if not prior:
        return DriftReport(
            alert=False,
            today_hit_rate=today_rate,
            prior_hit_rate=0.0,
            delta_hit_rate_pct=0.0,
            new_fallback_domains=[],
            reasons=["no prior run; baseline captured today"],
            per_domain_drift={d: 0.0 for d in today_per_domain},
            fallback_shape_kl=0.0,
        )

    prior_stats = summary_stats(prior)
    prior_rate = prior_stats["mesh_hit_rate"]
    delta_pct = (today_rate - prior_rate) * 100.0

    prior_fb = _fallback_domains(prior)
    new_fb = sorted(today_fb - prior_fb)

    prior_per_domain = _per_domain_hit_rates(prior)
    per_domain_drift: dict[str, float] = {}
    for domain in set(today_per_domain) | set(prior_per_domain):
        today_rate_d, _ = today_per_domain.get(domain, (0.0, 0))
        prior_rate_d, _ = prior_per_domain.get(domain, (0.0, 0))
        per_domain_drift[domain] = (today_rate_d - prior_rate_d) * 100.0

    today_dist = _fallback_shape_distribution(today)
    prior_dist = _fallback_shape_distribution(prior)
    shape_kl = _kl_divergence(today_dist, prior_dist)

    reasons: list[str] = []
    alert = False
    if abs(delta_pct) > alert_threshold_pct:
        reasons.append(
            f"|Δhit-rate| {delta_pct:+.1f}pp exceeds threshold "
            f"{alert_threshold_pct:.1f}pp "
            f"(today={today_rate:.0%}, prior={prior_rate:.0%})"
        )
        alert = True
    if new_fb:
        reasons.append(f"new fallback domains: {', '.join(new_fb)}")
        alert = True

    # M2: per-domain drift — only alert on domains with enough samples
    # in BOTH runs (otherwise it's noise).
    significant_domain_alerts: list[str] = []
    for domain, drift_pp in per_domain_drift.items():
        _, n_today = today_per_domain.get(domain, (0.0, 0))
        _, n_prior = prior_per_domain.get(domain, (0.0, 0))
        if (
            n_today < PER_DOMAIN_MIN_SAMPLES
            or n_prior < PER_DOMAIN_MIN_SAMPLES
        ):
            continue
        if abs(drift_pp) > per_domain_alert_threshold_pct:
            significant_domain_alerts.append(
                f"{domain}: {drift_pp:+.1f}pp"
            )
    if significant_domain_alerts:
        reasons.append(
            f"per-domain drift (|Δ|>{per_domain_alert_threshold_pct:.0f}pp): "
            + ", ".join(sorted(significant_domain_alerts))
        )
        alert = True

    # M2: fallback-chain-shape KL divergence — alerts when WHERE we fall
    # back redistributes materially, even if hit-rate looks unchanged.
    if shape_kl > fallback_shape_kl_threshold:
        reasons.append(
            f"fallback-shape KL {shape_kl:.2f} > threshold "
            f"{fallback_shape_kl_threshold:.2f} — chain mix shifted"
        )
        alert = True

    if not alert:
        reasons.append(
            f"stable (Δhit-rate {delta_pct:+.1f}pp, "
            f"shape-KL {shape_kl:.2f}, "
            f"no new fallback domains)"
        )

    return DriftReport(
        alert=alert,
        today_hit_rate=today_rate,
        prior_hit_rate=prior_rate,
        delta_hit_rate_pct=delta_pct,
        new_fallback_domains=new_fb,
        reasons=reasons,
        per_domain_drift=per_domain_drift,
        fallback_shape_kl=shape_kl,
    )


# ---------------------------------------------------------------------------
# History storage
# ---------------------------------------------------------------------------


def _most_recent_history(history_dir: Path) -> Path | None:
    """Find the lexicographically-latest history file (YYYY-MM-DD.jsonl).

    Returns None if no files matching the pattern exist.
    """
    if not history_dir.exists():
        return None
    candidates = sorted(history_dir.glob("[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9].jsonl"))
    return candidates[-1] if candidates else None


def _today_history_path(history_dir: Path, today: dt.date | None = None) -> Path:
    today = today or dt.date.today()
    return history_dir / f"{today.isoformat()}.jsonl"


def _atomic_copy(src: Path, dst: Path) -> None:
    """Write src → dst atomically via tempfile + rename."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".tmp")
    tmp.write_bytes(src.read_bytes())
    tmp.replace(dst)


# ---------------------------------------------------------------------------
# Alert + report emission
# ---------------------------------------------------------------------------


def _write_alert(alert_path: Path, report: DriftReport, today_path: Path) -> None:
    """Emit an alert JSONL line (one record, line-delimited for log aggregation)."""
    alert_path.parent.mkdir(parents=True, exist_ok=True)
    rec = {
        "ts": dt.datetime.now(dt.timezone.utc).isoformat(),
        "today_replay": str(today_path),
        "alert": report.alert,
        "today_hit_rate": report.today_hit_rate,
        "prior_hit_rate": report.prior_hit_rate,
        "delta_hit_rate_pct": report.delta_hit_rate_pct,
        "new_fallback_domains": report.new_fallback_domains,
        "per_domain_drift": report.per_domain_drift,
        "fallback_shape_kl": report.fallback_shape_kl,
        "reasons": report.reasons,
    }
    with alert_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")


# ---------------------------------------------------------------------------
# Replay subprocess invocation
# ---------------------------------------------------------------------------


def _run_replay(
    corpus: Path,
    output: Path,
    registry_url: str,
    token: str | None,
) -> int:
    """Invoke mesh_replay as subprocess; return exit code."""
    cmd = [
        sys.executable,
        "-m",
        "mesh.scripts.mesh_replay",
        "--corpus",
        str(corpus),
        "--output",
        str(output),
        "--registry-url",
        registry_url,
    ]
    if token:
        cmd.extend(["--token", token])
    result = subprocess.run(cmd, check=False)
    return result.returncode


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def run_nightly_smoke(
    corpus: Path,
    history_dir: Path,
    registry_url: str,
    token: str | None = None,
    alert_threshold_pct: float = DEFAULT_ALERT_THRESHOLD_PCT,
    today: dt.date | None = None,
    _replay_runner=_run_replay,  # injectable for tests
) -> int:
    """End-to-end nightly smoke. Returns exit code per module docstring.

    `_replay_runner` lets tests bypass the subprocess by injecting a
    stub that writes a synthetic replay JSONL and returns 0.
    """
    today = today or dt.date.today()
    history_dir.mkdir(parents=True, exist_ok=True)
    today_path = _today_history_path(history_dir, today)
    # Stage to a `.tmp` while replay runs; only promote on success.
    staging_path = today_path.with_suffix(today_path.suffix + ".staging")

    rc = _replay_runner(corpus, staging_path, registry_url, token)
    if rc != 0:
        # Replay itself broken — don't promote, don't diff. Exit 2.
        if staging_path.exists():
            staging_path.unlink()
        return 2

    today_records = load_replay_records(staging_path)
    prior_path = _most_recent_history(history_dir)
    prior_records = load_replay_records(prior_path) if prior_path else None
    report = detect_drift(
        today_records,
        prior_records,
        alert_threshold_pct=alert_threshold_pct,
    )

    # Promote staging → today's history file (atomic rename within same dir)
    staging_path.replace(today_path)

    # Alert file is appended (line per night), kept next to history
    alert_path = history_dir / "alerts.jsonl"
    _write_alert(alert_path, report, today_path)

    return 1 if report.alert else 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Nightly mesh-replay smoke + drift alert.")
    ap.add_argument(
        "--corpus",
        type=Path,
        default=Path(os.environ.get("MESH_NIGHTLY_CORPUS", "")),
        help="Path to pre-classified prompt corpus (or MESH_NIGHTLY_CORPUS env).",
    )
    ap.add_argument(
        "--history-dir",
        type=Path,
        default=Path(
            os.environ.get(
                "XDG_STATE_HOME",
                Path.home() / ".local" / "state",
            )
        )
        / "slancha-mesh"
        / "replay-history",
    )
    ap.add_argument("--registry-url", default="http://localhost:8088")
    ap.add_argument("--token", default=None)
    ap.add_argument(
        "--alert-threshold-pct",
        type=float,
        default=DEFAULT_ALERT_THRESHOLD_PCT,
        help="Absolute Δmesh-hit-rate (percentage points) above which to alert.",
    )
    args = ap.parse_args(argv)

    if not args.corpus or str(args.corpus) == "" or not args.corpus.exists():
        print(
            f"corpus not found: {args.corpus} "
            f"(set --corpus or MESH_NIGHTLY_CORPUS env)",
            file=sys.stderr,
        )
        return 2

    return run_nightly_smoke(
        corpus=args.corpus,
        history_dir=args.history_dir,
        registry_url=args.registry_url,
        token=args.token,
        alert_threshold_pct=args.alert_threshold_pct,
    )


if __name__ == "__main__":
    sys.exit(main())
