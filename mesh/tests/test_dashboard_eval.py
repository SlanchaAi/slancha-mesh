"""Tests for mesh.dashboard.eval — eval-results loader + time-series."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mesh.dashboard.eval import (
    eval_summary,
    load_eval_results,
    mean_score_over_time,
    per_domain_score_over_time,
    per_version_summary,
)


def _rec(
    ts: str,
    mean_score: float,
    *,
    router_version: str = "fast_head_v1 + overrides_v1",
    fast_head_version: int = 1,
    overrides_version: int = 1,
    n_eval: int = 500,
    per_domain_mean: dict | None = None,
    per_model_mean: dict | None = None,
    pct_acceptable: float = 0.8,
    pct_failure: float = 0.05,
    median_score: float | None = None,
    elapsed_seconds: float = 100.0,
) -> dict:
    # `is None` check (not `or`) so callers can explicitly pass {} to test the empty case
    if per_domain_mean is None:
        per_domain_mean = {"code": mean_score + 0.1, "general": mean_score}
    if per_model_mean is None:
        per_model_mean = {"qwen3-coder-30b": mean_score, "qwen3-8b": mean_score - 0.1}
    return {
        "ts": ts,
        "router_version": router_version,
        "fast_head_version": fast_head_version,
        "overrides_version": overrides_version,
        "holdout_version": 1,
        "n_eval": n_eval,
        "judge_model": "qwen3-coder-30b-a3b-fp8",
        "mean_score": mean_score,
        "median_score": median_score if median_score is not None else mean_score,
        "pct_acceptable": pct_acceptable,
        "pct_failure": pct_failure,
        "per_domain_mean": per_domain_mean,
        "per_model_mean": per_model_mean,
        "elapsed_seconds": elapsed_seconds,
    }


# ---------------------------------------------------------------------------
# load_eval_results
# ---------------------------------------------------------------------------


def test_load_returns_empty_when_file_missing(tmp_path: Path):
    assert load_eval_results(tmp_path / "missing.jsonl") == []


def test_load_skips_blanks_and_sorts_by_ts(tmp_path: Path):
    p = tmp_path / "e.jsonl"
    p.write_text(
        json.dumps(_rec("2026-05-17T03:00:00Z", 4.1)) + "\n" +
        "\n" +
        json.dumps(_rec("2026-05-17T01:00:00Z", 3.5)) + "\n" +
        json.dumps(_rec("2026-05-17T02:00:00Z", 3.8)) + "\n",
        encoding="utf-8",
    )
    recs = load_eval_results(p)
    assert [r["mean_score"] for r in recs] == [3.5, 3.8, 4.1]


def test_load_raises_on_bad_json(tmp_path: Path):
    p = tmp_path / "e.jsonl"
    p.write_text("{not json}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="line 1"):
        load_eval_results(p)


# ---------------------------------------------------------------------------
# mean_score_over_time
# ---------------------------------------------------------------------------


def test_mean_score_over_time_basic():
    recs = [
        _rec("2026-05-17T01:00:00Z", 3.5),
        _rec("2026-05-17T02:00:00Z", 3.8),
        _rec("2026-05-17T03:00:00Z", 4.1),
    ]
    out = mean_score_over_time(recs)
    assert len(out) == 3
    assert [pt[1] for pt in out] == [3.5, 3.8, 4.1]
    assert [pt[2] for pt in out] == [500, 500, 500]


def test_mean_score_over_time_empty():
    assert mean_score_over_time([]) == []


def test_mean_score_over_time_skips_missing_fields():
    recs = [
        _rec("2026-05-17T01:00:00Z", 3.5),
        {"ts": None, "mean_score": 4.0},   # no ts
        {"ts": "2026-05-17T02:00:00Z"},     # no score
    ]
    out = mean_score_over_time(recs)
    assert len(out) == 1


# ---------------------------------------------------------------------------
# per_domain_score_over_time
# ---------------------------------------------------------------------------


def test_per_domain_score_over_time_buckets():
    recs = [
        _rec("2026-05-17T01:00:00Z", 3.5,
             per_domain_mean={"code": 4.0, "general": 3.0}),
        _rec("2026-05-17T02:00:00Z", 3.8,
             per_domain_mean={"code": 4.5, "general": 3.1}),
    ]
    out = per_domain_score_over_time(recs)
    assert set(out.keys()) == {"code", "general"}
    assert len(out["code"]) == 2
    assert [pt[1] for pt in out["code"]] == [4.0, 4.5]


def test_per_domain_score_over_time_skips_empty():
    recs = [
        _rec("2026-05-17T01:00:00Z", 3.5, per_domain_mean={}),
    ]
    out = per_domain_score_over_time(recs)
    assert out == {}


# ---------------------------------------------------------------------------
# per_version_summary
# ---------------------------------------------------------------------------


def test_per_version_summary_takes_latest_per_version():
    recs = [
        _rec("2026-05-17T01:00:00Z", 3.5, router_version="v1"),
        _rec("2026-05-17T02:00:00Z", 3.6, router_version="v1"),  # later v1
        _rec("2026-05-17T03:00:00Z", 4.0, router_version="v2"),
    ]
    rows = per_version_summary(recs)
    assert len(rows) == 2
    v1_row = next(r for r in rows if r["router_version"] == "v1")
    assert v1_row["mean_score"] == 3.6  # took the later one


def test_per_version_summary_sorted_by_ts():
    recs = [
        _rec("2026-05-17T03:00:00Z", 4.0, router_version="v2"),
        _rec("2026-05-17T01:00:00Z", 3.5, router_version="v1"),
    ]
    rows = per_version_summary(recs)
    assert rows[0]["router_version"] == "v1"
    assert rows[1]["router_version"] == "v2"


# ---------------------------------------------------------------------------
# eval_summary (top-card)
# ---------------------------------------------------------------------------


def test_eval_summary_empty():
    s = eval_summary([])
    assert s["n_passes"] == 0
    assert s["improvement_pct"] == 0.0


def test_eval_summary_improvement_pp_and_pct():
    recs = [
        _rec("2026-05-17T01:00:00Z", 3.5),
        _rec("2026-05-17T05:00:00Z", 4.2),
    ]
    s = eval_summary(recs)
    assert s["first_score"] == 3.5
    assert s["latest_score"] == 4.2
    assert s["improvement_pp"] == pytest.approx(0.7, abs=0.01)
    assert s["improvement_pct"] == pytest.approx(20.0, abs=0.1)
    assert s["n_passes"] == 2


def test_eval_summary_handles_single_pass():
    recs = [_rec("2026-05-17T01:00:00Z", 3.5)]
    s = eval_summary(recs)
    assert s["n_passes"] == 1
    assert s["improvement_pp"] == 0.0
    assert s["improvement_pct"] == 0.0


def test_eval_summary_distinct_router_versions():
    recs = [
        _rec("2026-05-17T01:00:00Z", 3.5, router_version="v1"),
        _rec("2026-05-17T02:00:00Z", 3.7, router_version="v2"),
        _rec("2026-05-17T03:00:00Z", 3.9, router_version="v3"),
    ]
    s = eval_summary(recs)
    assert s["distinct_router_versions"] == 3


def test_eval_summary_total_routings():
    recs = [
        _rec("2026-05-17T01:00:00Z", 3.5, n_eval=500),
        _rec("2026-05-17T02:00:00Z", 3.6, n_eval=500),
        _rec("2026-05-17T03:00:00Z", 4.0, n_eval=1000),
    ]
    s = eval_summary(recs)
    assert s["total_held_out_routings"] == 2000
