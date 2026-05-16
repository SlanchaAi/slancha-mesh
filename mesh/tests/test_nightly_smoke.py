"""Tests for mesh.scripts.nightly_smoke — drift detection + orchestrator.

The orchestrator's subprocess call to mesh_replay is replaced with an
injected stub `_replay_runner` so tests are hermetic + sub-second.
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

from mesh.scripts.nightly_smoke import (
    DEFAULT_ALERT_THRESHOLD_PCT,
    DriftReport,
    _most_recent_history,
    _today_history_path,
    detect_drift,
    run_nightly_smoke,
)


# ---------------------------------------------------------------------------
# Helpers — synthetic decision records
# ---------------------------------------------------------------------------


def _rec(
    prompt_id: str,
    *,
    mesh_hit: bool,
    domain: str = "code",
    specialist: str | None = None,
) -> dict:
    return {
        "prompt_id": prompt_id,
        "prompt_hash": None,
        "signals": {"domain": domain, "difficulty": "medium"},
        "decision": {
            "chosen_specialist": specialist if mesh_hit else None,
            "chosen_node": "n1" if mesh_hit else None,
            "node_url": None,
            "model": specialist or "claude-sonnet-4-7",
            "reason": "synthetic",
            "queue_ms": 0,
            "fallback_chain": [],
            "mesh_hit": mesh_hit,
            "vs_cloud_baseline_cost": 0.0,
        },
        "snapshot_ts": "2026-05-16T03:00:00+00:00",
    }


def _make_replay_runner(records_to_write):
    """Build an injectable _replay_runner that writes `records` then returns 0."""
    def runner(corpus, output, registry_url, token):
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8") as f:
            for r in records_to_write:
                f.write(json.dumps(r) + "\n")
        return 0
    return runner


def _failing_replay_runner(corpus, output, registry_url, token):
    return 2  # simulate mesh_replay exit-code-2 (corpus not found, etc.)


# ---------------------------------------------------------------------------
# detect_drift
# ---------------------------------------------------------------------------


def test_drift_no_prior_no_alert():
    today = [_rec("p1", mesh_hit=True), _rec("p2", mesh_hit=False)]
    report = detect_drift(today, prior=None)
    assert report.alert is False
    assert report.today_hit_rate == 0.5
    assert report.prior_hit_rate == 0.0
    assert "no prior" in report.reasons[0]


def test_drift_stable_no_alert():
    today = [_rec(f"p{i}", mesh_hit=(i < 7)) for i in range(10)]  # 70% hit
    prior = [_rec(f"p{i}", mesh_hit=(i < 7)) for i in range(10)]  # 70% hit
    report = detect_drift(today, prior, alert_threshold_pct=10.0)
    assert report.alert is False
    assert abs(report.delta_hit_rate_pct) < 1e-6


def test_drift_hit_rate_drop_alerts():
    """20% absolute hit-rate drop with 10% threshold → alert."""
    today = [_rec(f"p{i}", mesh_hit=(i < 5)) for i in range(10)]  # 50% hit
    prior = [_rec(f"p{i}", mesh_hit=(i < 7)) for i in range(10)]  # 70% hit
    report = detect_drift(today, prior, alert_threshold_pct=10.0)
    assert report.alert is True
    assert report.delta_hit_rate_pct < -10.0
    assert any("Δhit-rate" in r for r in report.reasons)


def test_drift_hit_rate_rise_alerts():
    """Big improvements also alert — could be a bench-corpus accident."""
    today = [_rec(f"p{i}", mesh_hit=(i < 9)) for i in range(10)]  # 90% hit
    prior = [_rec(f"p{i}", mesh_hit=(i < 5)) for i in range(10)]  # 50% hit
    report = detect_drift(today, prior, alert_threshold_pct=10.0)
    assert report.alert is True
    assert report.delta_hit_rate_pct > 10.0


def test_drift_new_fallback_domain_alerts():
    today = [
        _rec("p1", mesh_hit=False, domain="code"),
        _rec("p2", mesh_hit=False, domain="multilingual"),  # NEW domain falling back
    ]
    prior = [_rec("p1", mesh_hit=False, domain="code")]
    report = detect_drift(today, prior, alert_threshold_pct=99.0)  # threshold high so only domain signal fires
    assert report.alert is True
    assert "multilingual" in report.new_fallback_domains
    assert any("new fallback" in r for r in report.reasons)


def test_drift_both_signals_compound_alert():
    today = [
        _rec("p1", mesh_hit=False, domain="multilingual"),
        _rec("p2", mesh_hit=False, domain="code"),
    ]
    prior = [_rec("p1", mesh_hit=True, specialist="coder", domain="code")]
    report = detect_drift(today, prior, alert_threshold_pct=10.0)
    assert report.alert is True
    # Both reasons present
    assert any("Δhit-rate" in r for r in report.reasons)
    assert any("new fallback" in r for r in report.reasons)


# ---------------------------------------------------------------------------
# History storage helpers
# ---------------------------------------------------------------------------


def test_most_recent_history_empty(tmp_path):
    assert _most_recent_history(tmp_path) is None


def test_most_recent_history_nonexistent_dir(tmp_path):
    assert _most_recent_history(tmp_path / "does-not-exist") is None


def test_most_recent_history_picks_latest(tmp_path):
    (tmp_path / "2026-05-14.jsonl").write_text("")
    (tmp_path / "2026-05-15.jsonl").write_text("")
    (tmp_path / "2026-05-10.jsonl").write_text("")
    # Non-conforming filenames are ignored
    (tmp_path / "alerts.jsonl").write_text("")
    (tmp_path / "today.jsonl").write_text("")

    latest = _most_recent_history(tmp_path)
    assert latest is not None
    assert latest.name == "2026-05-15.jsonl"


def test_today_history_path_uses_iso_date(tmp_path):
    p = _today_history_path(tmp_path, today=dt.date(2026, 5, 16))
    assert p == tmp_path / "2026-05-16.jsonl"


# ---------------------------------------------------------------------------
# Orchestrator — end to end with injected runner
# ---------------------------------------------------------------------------


def test_run_first_night_exits_zero_and_writes_history(tmp_path):
    """No prior run → exit 0, today's JSONL promoted into history dir."""
    history = tmp_path / "history"
    runner = _make_replay_runner([_rec("p1", mesh_hit=True, specialist="coder")])
    rc = run_nightly_smoke(
        corpus=tmp_path / "corpus.jsonl",  # not actually read by stub
        history_dir=history,
        registry_url="http://stub",
        today=dt.date(2026, 5, 16),
        _replay_runner=runner,
    )
    assert rc == 0
    promoted = history / "2026-05-16.jsonl"
    assert promoted.exists()
    # Alert file appended (line count = 1)
    alerts = (history / "alerts.jsonl").read_text().splitlines()
    assert len(alerts) == 1
    record = json.loads(alerts[0])
    assert record["alert"] is False
    assert "no prior" in record["reasons"][0]


def test_run_second_night_drift_alerts_exits_one(tmp_path):
    history = tmp_path / "history"
    history.mkdir()
    # Seed yesterday at 70% hit-rate
    (history / "2026-05-15.jsonl").write_text(
        "\n".join(json.dumps(_rec(f"p{i}", mesh_hit=(i < 7))) for i in range(10)) + "\n"
    )
    # Today at 30% hit-rate → 40pp drop, alerts
    today_runner = _make_replay_runner(
        [_rec(f"p{i}", mesh_hit=(i < 3)) for i in range(10)]
    )
    rc = run_nightly_smoke(
        corpus=tmp_path / "corpus.jsonl",
        history_dir=history,
        registry_url="http://stub",
        today=dt.date(2026, 5, 16),
        _replay_runner=today_runner,
    )
    assert rc == 1
    assert (history / "2026-05-16.jsonl").exists()
    alerts = (history / "alerts.jsonl").read_text().splitlines()
    record = json.loads(alerts[0])
    assert record["alert"] is True
    assert record["delta_hit_rate_pct"] < -10.0


def test_run_replay_failure_exits_two_no_promotion(tmp_path):
    """Replay subprocess fails → exit 2, no history mutation, no alert."""
    history = tmp_path / "history"
    history.mkdir()
    rc = run_nightly_smoke(
        corpus=tmp_path / "corpus.jsonl",
        history_dir=history,
        registry_url="http://stub",
        today=dt.date(2026, 5, 16),
        _replay_runner=_failing_replay_runner,
    )
    assert rc == 2
    assert not (history / "2026-05-16.jsonl").exists()
    assert not (history / "alerts.jsonl").exists()
    # Staging file cleaned up
    staging = history / "2026-05-16.jsonl.staging"
    assert not staging.exists()


def test_run_stable_night_exits_zero_no_alert_in_log(tmp_path):
    history = tmp_path / "history"
    history.mkdir()
    stable_records = [_rec(f"p{i}", mesh_hit=(i < 7)) for i in range(10)]
    (history / "2026-05-15.jsonl").write_text(
        "\n".join(json.dumps(r) for r in stable_records) + "\n"
    )
    rc = run_nightly_smoke(
        corpus=tmp_path / "corpus.jsonl",
        history_dir=history,
        registry_url="http://stub",
        today=dt.date(2026, 5, 16),
        _replay_runner=_make_replay_runner(stable_records),
    )
    assert rc == 0
    record = json.loads((history / "alerts.jsonl").read_text().splitlines()[0])
    assert record["alert"] is False
    assert "stable" in record["reasons"][0]


def test_drift_report_dataclass_shape():
    """Smoke that DriftReport carries the fields the alert writer reads."""
    r = DriftReport(
        alert=False,
        today_hit_rate=0.5,
        prior_hit_rate=0.5,
        delta_hit_rate_pct=0.0,
        new_fallback_domains=[],
        reasons=["stable"],
    )
    assert r.alert is False
    assert r.today_hit_rate == 0.5


def test_default_alert_threshold_pct_is_ten():
    assert DEFAULT_ALERT_THRESHOLD_PCT == 10.0


# ---------------------------------------------------------------------------
# M2 — per-domain drift + fallback-shape KL
# ---------------------------------------------------------------------------


from mesh.scripts.nightly_smoke import (  # noqa: E402 — keep grouped with M2 tests
    DEFAULT_KL_THRESHOLD,
    DEFAULT_PER_DOMAIN_THRESHOLD_PCT,
    PER_DOMAIN_MIN_SAMPLES,
    _fallback_shape_distribution,
    _kl_divergence,
    _per_domain_hit_rates,
)


def test_per_domain_hit_rates_groups_correctly():
    recs = [
        _rec("p1", mesh_hit=True, domain="code"),
        _rec("p2", mesh_hit=True, domain="code"),
        _rec("p3", mesh_hit=False, domain="code"),
        _rec("p4", mesh_hit=False, domain="math"),
    ]
    out = _per_domain_hit_rates(recs)
    assert out["code"] == (2 / 3, 3)
    assert out["math"] == (0.0, 1)


def test_per_domain_hit_rates_empty():
    assert _per_domain_hit_rates([]) == {}


def test_fallback_shape_distribution_sums_to_one():
    recs = [
        _rec("p1", mesh_hit=False, domain="code"),
        _rec("p2", mesh_hit=False, domain="code"),
        _rec("p3", mesh_hit=False, domain="math"),
    ]
    dist = _fallback_shape_distribution(recs)
    assert abs(sum(dist.values()) - 1.0) < 1e-9


def test_fallback_shape_distribution_empty():
    assert _fallback_shape_distribution([]) == {}


def test_kl_divergence_identical_distributions_is_zero():
    d = {"a": 0.5, "b": 0.5}
    assert _kl_divergence(d, d) == 0.0


def test_kl_divergence_disjoint_distributions_positive():
    p = {"a": 1.0}
    q = {"b": 1.0}
    assert _kl_divergence(p, q) > 0.0


def test_kl_divergence_empty_inputs_zero():
    assert _kl_divergence({}, {}) == 0.0


def test_drift_per_domain_alerts_when_one_domain_collapses(monkeypatch):
    """Aggregate hit-rate stable, but one domain's hit-rate collapses → alert."""
    today = (
        [_rec(f"c{i}", mesh_hit=True, domain="code") for i in range(10)]
        + [_rec(f"m{i}", mesh_hit=False, domain="math") for i in range(10)]
    )
    prior = (
        [_rec(f"c{i}", mesh_hit=True, domain="code") for i in range(10)]
        + [_rec(f"m{i}", mesh_hit=True, domain="math") for i in range(10)]
    )
    # Aggregate: today=50%, prior=100% → 50pp drop. That hits the aggregate
    # alert too; but we also assert the per-domain detail.
    report = detect_drift(today, prior, alert_threshold_pct=99.0)  # aggregate disabled
    assert report.alert is True
    assert any("per-domain drift" in r and "math" in r for r in report.reasons)
    assert report.per_domain_drift["code"] == 0.0
    assert report.per_domain_drift["math"] == -100.0
    # New-fallback-domain signal independently fires
    assert "math" in report.new_fallback_domains


def test_drift_per_domain_skips_small_sample_domains():
    """Domain with <PER_DOMAIN_MIN_SAMPLES samples doesn't trigger alert."""
    # 4 samples in code: below threshold → noise floor protection
    today = [_rec(f"c{i}", mesh_hit=False, domain="code") for i in range(4)]
    prior = [_rec(f"c{i}", mesh_hit=True, domain="code") for i in range(4)]
    report = detect_drift(today, prior, alert_threshold_pct=99.0)
    # per_domain_drift is reported regardless (for visibility), but no
    # per-domain alert reason fires because samples < threshold.
    assert report.per_domain_drift["code"] == -100.0
    assert not any("per-domain drift" in r for r in report.reasons)


def test_drift_fallback_shape_kl_alerts_on_distribution_shift():
    """All-cloud yesterday → all-cloud-via-different-shape today → KL > threshold."""
    prior = [_rec(f"p{i}", mesh_hit=False, domain="code") for i in range(20)]
    # Today: same domain falls back BUT via a different chain shape
    today_recs: list[dict] = []
    for i in range(20):
        rec = _rec(f"t{i}", mesh_hit=False, domain="code")
        # Override the fallback chain to a distinct shape
        rec["decision"]["fallback_chain"] = [["new-model", None]]
        today_recs.append(rec)
    report = detect_drift(today_recs, prior, alert_threshold_pct=99.0)
    assert report.fallback_shape_kl > DEFAULT_KL_THRESHOLD
    assert any("fallback-shape KL" in r for r in report.reasons)
    assert report.alert is True


def test_drift_fallback_shape_kl_stable_no_alert():
    """Identical shape distributions → KL ≈ 0 → no alert from this signal."""
    recs = [_rec(f"p{i}", mesh_hit=False, domain="code") for i in range(20)]
    report = detect_drift(recs, recs, alert_threshold_pct=99.0)
    assert report.fallback_shape_kl < DEFAULT_KL_THRESHOLD


def test_drift_report_carries_m2_fields():
    """First-night DriftReport always carries the M2 fields, defaulted."""
    today = [_rec(f"p{i}", mesh_hit=(i < 7), domain="code") for i in range(10)]
    report = detect_drift(today, prior=None)
    assert "code" in report.per_domain_drift
    assert report.per_domain_drift["code"] == 0.0  # no prior to drift from
    assert report.fallback_shape_kl == 0.0


def test_m2_thresholds_have_documented_defaults():
    assert DEFAULT_PER_DOMAIN_THRESHOLD_PCT == 10.0
    assert DEFAULT_KL_THRESHOLD == 0.3
    assert PER_DOMAIN_MIN_SAMPLES == 5
