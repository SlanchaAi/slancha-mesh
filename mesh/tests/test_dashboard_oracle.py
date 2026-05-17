"""Tests for mesh.dashboard.oracle — pure-function panels over the
oracle JSONL spark emits during the 472e judge pass.

Mirrors the test pattern in test_dashboard_panels.py and
test_dashboard_live_run.py: small in-memory fixtures, deterministic
assertions.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mesh.dashboard.oracle import (
    alt_recommended_rate_by_domain,
    load_oracle_records,
    oracle_summary,
    per_domain_quality_matrix,
    quality_over_time,
    quality_score_histogram,
    quality_vs_latency_scatter,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _rec(
    score: int | None,
    *,
    ts: str | None = "2026-05-17T01:30:00Z",
    domain: str = "code",
    model: str = "codestral:22b",
    better_model: str | None = None,
    latency_ms: int | None = 100,
    judge_model: str | None = "qwen3-coder-30b-a3b-fp8",
) -> dict:
    return {
        "prompt_id": "p-x",
        "signals": {"domain": domain},
        "route_decision": {"backend": "ollama", "model": model},
        "latency_ms": latency_ms,
        "oracle": {
            "judge_score": score,
            "judge_reason": "stub",
            "better_model": better_model,
            "judge_model": judge_model,
            "judge_ts": ts,
        },
    }


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------


def test_load_oracle_records_skips_blank_lines(tmp_path: Path):
    p = tmp_path / "x.jsonl"
    p.write_text("\n" + json.dumps(_rec(4)) + "\n\n", encoding="utf-8")
    rows = load_oracle_records(p)
    assert len(rows) == 1


def test_load_oracle_records_raises_on_bad_json(tmp_path: Path):
    p = tmp_path / "x.jsonl"
    p.write_text("{bad json\n", encoding="utf-8")
    with pytest.raises(ValueError, match="line 1"):
        load_oracle_records(p)


# ---------------------------------------------------------------------------
# Quality histogram
# ---------------------------------------------------------------------------


def test_quality_score_histogram_basic():
    recs = [_rec(5), _rec(4), _rec(4), _rec(2), _rec(1)]
    assert quality_score_histogram(recs) == {5: 1, 4: 2, 2: 1, 1: 1}


def test_quality_score_histogram_skips_missing():
    recs = [_rec(5), _rec(None), _rec(3)]
    assert quality_score_histogram(recs) == {5: 1, 3: 1}


def test_quality_score_histogram_skips_malformed():
    rec_bad = _rec(5)
    rec_bad["oracle"]["judge_score"] = "not-a-number"
    out = quality_score_histogram([_rec(5), rec_bad])
    assert out == {5: 1}


# ---------------------------------------------------------------------------
# Quality over time
# ---------------------------------------------------------------------------


def test_quality_over_time_buckets_and_means():
    recs = [
        _rec(5, ts="2026-05-17T01:00:00Z"),
        _rec(3, ts="2026-05-17T01:00:30Z"),
        _rec(4, ts="2026-05-17T01:05:30Z"),
    ]
    out = quality_over_time(recs, bucket_seconds=300)
    assert len(out) == 2
    assert out[0][1] == pytest.approx(4.0)
    assert out[0][2] == 2
    assert out[1][1] == 4.0
    assert out[1][2] == 1


def test_quality_over_time_skips_records_missing_ts_or_score():
    recs = [_rec(5, ts=None), _rec(None, ts="2026-05-17T01:00:00Z"), _rec(4)]
    out = quality_over_time(recs, bucket_seconds=300)
    assert len(out) == 1
    assert out[0][1] == 4.0


def test_quality_over_time_handles_z_suffix_ts():
    rec = _rec(5, ts="2026-05-17T01:00:00Z")
    out = quality_over_time([rec], bucket_seconds=60)
    assert len(out) == 1


# ---------------------------------------------------------------------------
# Per-domain × per-model heatmap
# ---------------------------------------------------------------------------


def test_per_domain_quality_matrix_basic():
    recs = [
        _rec(5, domain="code", model="codestral:22b"),
        _rec(3, domain="code", model="codestral:22b"),
        _rec(4, domain="code", model="phi4:14b"),
        _rec(2, domain="math", model="codestral:22b"),
    ]
    m = per_domain_quality_matrix(recs)
    assert m["code"]["codestral:22b"]["mean_score"] == pytest.approx(4.0)
    assert m["code"]["codestral:22b"]["n"] == 2
    assert m["code"]["phi4:14b"]["mean_score"] == 4.0
    assert m["math"]["codestral:22b"]["n"] == 1


def test_per_domain_quality_matrix_skips_records_missing_score():
    recs = [_rec(None, domain="code"), _rec(5, domain="code")]
    m = per_domain_quality_matrix(recs)
    assert m["code"]["codestral:22b"]["n"] == 1


# ---------------------------------------------------------------------------
# Alt-model recommendation rate
# ---------------------------------------------------------------------------


def test_alt_recommended_rate_by_domain():
    recs = [
        _rec(3, domain="code", better_model="phi4:14b"),
        _rec(3, domain="code", better_model="phi4:14b"),
        _rec(5, domain="code", better_model=None),
        _rec(2, domain="math", better_model="gemma2:9b"),
    ]
    out = alt_recommended_rate_by_domain(recs)
    assert out["code"]["rate"] == pytest.approx(2 / 3, abs=1e-3)
    assert out["code"]["n_recommended"] == 2
    assert out["code"]["top_alt_model"] == "phi4:14b"
    assert out["math"]["rate"] == 1.0
    assert out["math"]["top_alt_model"] == "gemma2:9b"


def test_alt_recommended_rate_empty_recommendations():
    recs = [_rec(5, domain="code"), _rec(5, domain="code")]
    out = alt_recommended_rate_by_domain(recs)
    assert out["code"]["rate"] == 0.0
    assert out["code"]["top_alt_model"] is None


# ---------------------------------------------------------------------------
# Quality vs latency scatter
# ---------------------------------------------------------------------------


def test_quality_vs_latency_scatter_yields_triples():
    recs = [
        _rec(5, latency_ms=100, domain="code"),
        _rec(3, latency_ms=200, domain="math"),
    ]
    out = quality_vs_latency_scatter(recs)
    assert (100, 5, "code") in out
    assert (200, 3, "math") in out


def test_quality_vs_latency_scatter_skips_missing():
    recs = [_rec(5, latency_ms=None), _rec(None, latency_ms=100), _rec(4, latency_ms=50)]
    out = quality_vs_latency_scatter(recs)
    assert out == [(50, 4, "code")]


def test_quality_vs_latency_scatter_caps_at_max_points():
    recs = [_rec(5, latency_ms=i) for i in range(1, 20)]
    out = quality_vs_latency_scatter(recs, max_points=5)
    assert len(out) == 5


# ---------------------------------------------------------------------------
# Top-card summary
# ---------------------------------------------------------------------------


def test_oracle_summary_empty():
    s = oracle_summary([])
    assert s["total"] == 0
    assert s["labeled"] == 0
    assert s["mean_score"] == 0.0


def test_oracle_summary_all_labeled():
    recs = [_rec(5), _rec(4), _rec(4), _rec(2), _rec(1)]
    s = oracle_summary(recs)
    assert s["total"] == 5
    assert s["labeled"] == 5
    assert s["mean_score"] == pytest.approx(3.2)
    assert s["pct_acceptable"] == pytest.approx(0.6)
    assert s["pct_failure"] == pytest.approx(0.4)


def test_oracle_summary_partial_labeled():
    recs = [_rec(5), _rec(None), _rec(None), _rec(3)]
    s = oracle_summary(recs)
    assert s["total"] == 4
    assert s["labeled"] == 2
    assert s["mean_score"] == pytest.approx(4.0)


def test_oracle_summary_alt_rate():
    recs = [
        _rec(3, better_model="alt-a"),
        _rec(3, better_model="alt-b"),
        _rec(5, better_model=None),
    ]
    s = oracle_summary(recs)
    assert s["alt_rate"] == pytest.approx(2 / 3, abs=1e-3)


def test_oracle_summary_counts_distinct_judges():
    recs = [
        _rec(5, judge_model="qwen3-coder-30b-a3b-fp8"),
        _rec(5, judge_model="qwen3-coder-30b-a3b-fp8"),
        _rec(5, judge_model="phi4:14b"),
    ]
    s = oracle_summary(recs)
    assert s["distinct_judges"] == 2
