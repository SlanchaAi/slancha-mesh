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
import os
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mesh.dashboard.panels import (
    DecisionRecord,
    fallback_chain_shape_histogram,
    load_replay_records,
    summary_stats,
)

DEFAULT_ALERT_THRESHOLD_PCT = 10.0


# ---------------------------------------------------------------------------
# Drift detection
# ---------------------------------------------------------------------------


@dataclass
class DriftReport:
    """Result of comparing today's replay to a prior run.

    `alert=True` indicates the run should fire an operator-visible alert.
    `reasons` is a list of human-readable strings explaining each
    contributing signal.
    """

    alert: bool
    today_hit_rate: float
    prior_hit_rate: float
    delta_hit_rate_pct: float
    new_fallback_domains: list[str]
    reasons: list[str]


def _fallback_domains(records: list[DecisionRecord]) -> set[str]:
    """Domains whose decisions fell back to cloud (mesh_hit=False)."""
    out: set[str] = set()
    for r in records:
        if r["decision"].get("mesh_hit") is False:
            domain = r.get("signals", {}).get("domain", "unknown")
            out.add(domain)
    return out


def detect_drift(
    today: list[DecisionRecord],
    prior: list[DecisionRecord] | None,
    *,
    alert_threshold_pct: float = DEFAULT_ALERT_THRESHOLD_PCT,
) -> DriftReport:
    """Compare today's replay against the prior run.

    No prior run → no alert (first-night), but report still emitted so
    history captures today's baseline.

    Δhit-rate: today_pct - prior_pct (signed). Alerts on absolute > threshold.
    New fallback domains: domains in today's cloud-fallback set that were
    NOT in prior's set. (Removed-from-fallback domains are also relevant
    but less worrying; they're noted, not alerted.)
    """
    today_stats = summary_stats(today)
    today_rate = today_stats["mesh_hit_rate"]
    today_fb = _fallback_domains(today)

    if not prior:
        return DriftReport(
            alert=False,
            today_hit_rate=today_rate,
            prior_hit_rate=0.0,
            delta_hit_rate_pct=0.0,
            new_fallback_domains=[],
            reasons=["no prior run; baseline captured today"],
        )

    prior_stats = summary_stats(prior)
    prior_rate = prior_stats["mesh_hit_rate"]
    delta_pct = (today_rate - prior_rate) * 100.0

    prior_fb = _fallback_domains(prior)
    new_fb = sorted(today_fb - prior_fb)

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
    if not alert:
        reasons.append(
            f"stable (Δhit-rate {delta_pct:+.1f}pp, no new fallback domains)"
        )

    return DriftReport(
        alert=alert,
        today_hit_rate=today_rate,
        prior_hit_rate=prior_rate,
        delta_hit_rate_pct=delta_pct,
        new_fallback_domains=new_fb,
        reasons=reasons,
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
