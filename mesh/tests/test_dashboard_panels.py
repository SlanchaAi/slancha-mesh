"""Tests for mesh.dashboard.panels — pure-function chart-data computers.

No streamlit dependency; runs on mac/CI without the dashboard runtime.
Inputs are synthetic decision records mirroring the JSONL shape that
mesh.scripts.mesh_replay emits.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from mesh.dashboard.panels import (
    fallback_chain_shape_histogram,
    load_replay_records,
    mesh_hit_rate_over_time,
    per_specialist_invocation_counts,
    summary_stats,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ts(hours_from_base: int = 0) -> str:
    base = datetime(2026, 5, 16, 12, 0, 0, tzinfo=timezone.utc)
    return (base + timedelta(hours=hours_from_base)).isoformat()


def _rec(
    prompt_id: str,
    *,
    mesh_hit: bool,
    specialist: str | None = None,
    node: str | None = None,
    domain: str = "code",
    queue_ms: int = 0,
    fallback_chain: list[list] | None = None,
    ts: str | None = None,
) -> dict:
    return {
        "prompt_id": prompt_id,
        "prompt_hash": None,
        "signals": {"domain": domain, "difficulty": "medium"},
        "decision": {
            "chosen_specialist": specialist,
            "chosen_node": node,
            "node_url": None,
            "model": specialist or "claude-sonnet-4-7",
            "reason": "synthetic",
            "queue_ms": queue_ms,
            "fallback_chain": fallback_chain or [],
            "mesh_hit": mesh_hit,
            "vs_cloud_baseline_cost": 0.0,
        },
        "snapshot_ts": ts or _ts(0),
    }


# ---------------------------------------------------------------------------
# load_replay_records
# ---------------------------------------------------------------------------


def test_load_replay_records_basic(tmp_path):
    p = tmp_path / "replay.jsonl"
    p.write_text(
        json.dumps(_rec("p1", mesh_hit=True, specialist="s1", node="n1")) + "\n" +
        "\n" +
        json.dumps(_rec("p2", mesh_hit=False)) + "\n"
    )
    recs = load_replay_records(p)
    assert len(recs) == 2
    assert recs[0]["prompt_id"] == "p1"
    assert recs[1]["decision"]["mesh_hit"] is False


def test_load_replay_records_invalid_json_raises(tmp_path):
    p = tmp_path / "bad.jsonl"
    p.write_text("{not json}\n")
    with pytest.raises(ValueError, match="line 1: invalid JSON"):
        load_replay_records(p)


def test_load_replay_records_empty(tmp_path):
    p = tmp_path / "empty.jsonl"
    p.write_text("")
    assert load_replay_records(p) == []


# ---------------------------------------------------------------------------
# mesh_hit_rate_over_time
# ---------------------------------------------------------------------------


def test_hit_rate_empty():
    assert mesh_hit_rate_over_time([]) == []


def test_hit_rate_all_cloud():
    recs = [_rec(f"p{i}", mesh_hit=False) for i in range(5)]
    out = mesh_hit_rate_over_time(recs)
    assert len(out) == 1
    bucket, rate, n = out[0]
    assert rate == 0.0
    assert n == 5


def test_hit_rate_mixed_single_bucket():
    recs = [
        _rec("p1", mesh_hit=True, specialist="s", node="n"),
        _rec("p2", mesh_hit=True, specialist="s", node="n"),
        _rec("p3", mesh_hit=False),
        _rec("p4", mesh_hit=False),
    ]
    out = mesh_hit_rate_over_time(recs)
    assert len(out) == 1
    _, rate, n = out[0]
    assert rate == 0.5
    assert n == 4


def test_hit_rate_multi_bucket_sorted():
    # 3 buckets across 3 hours, default bucket = 3600s
    recs = [
        _rec("p1", mesh_hit=True, ts=_ts(0)),
        _rec("p2", mesh_hit=False, ts=_ts(0)),  # bucket 1: 0.5
        _rec("p3", mesh_hit=True, ts=_ts(1)),
        _rec("p4", mesh_hit=True, ts=_ts(1)),   # bucket 2: 1.0
        _rec("p5", mesh_hit=False, ts=_ts(2)),  # bucket 3: 0.0
    ]
    out = mesh_hit_rate_over_time(recs, bucket_seconds=3600)
    rates = [pt[1] for pt in out]
    counts = [pt[2] for pt in out]
    buckets = [pt[0] for pt in out]
    assert rates == [0.5, 1.0, 0.0]
    assert counts == [2, 2, 1]
    # Sorted ascending by bucket start
    assert buckets == sorted(buckets)


def test_hit_rate_custom_bucket_seconds():
    recs = [
        _rec("p1", mesh_hit=True, ts=_ts(0)),
        _rec("p2", mesh_hit=True, ts=_ts(2)),  # 2 hours later
    ]
    # 1-hour bucket → 2 separate buckets
    out_1h = mesh_hit_rate_over_time(recs, bucket_seconds=3600)
    assert len(out_1h) == 2
    # 24-hour bucket → 1 bucket
    out_24h = mesh_hit_rate_over_time(recs, bucket_seconds=86400)
    assert len(out_24h) == 1


# ---------------------------------------------------------------------------
# fallback_chain_shape_histogram
# ---------------------------------------------------------------------------


def test_fallback_shape_empty_records():
    assert fallback_chain_shape_histogram([]) == []


def test_fallback_shape_empty_chain_marked():
    recs = [_rec("p1", mesh_hit=True, fallback_chain=[])]
    out = fallback_chain_shape_histogram(recs)
    assert out == [("(empty)", 1)]


def test_fallback_shape_groups_identical_chains():
    chain_a = [["claude-sonnet-4-7", None]]
    chain_b = [["qwen3-coder", "n2"], ["claude-sonnet-4-7", None]]
    recs = [
        _rec("p1", mesh_hit=False, fallback_chain=chain_a),
        _rec("p2", mesh_hit=False, fallback_chain=chain_a),
        _rec("p3", mesh_hit=True, fallback_chain=chain_b),
    ]
    out = fallback_chain_shape_histogram(recs)
    # chain_a appears 2× (sort first), chain_b appears 1×
    assert out[0][1] == 2
    assert out[1][1] == 1
    assert "cloud" in out[0][0]  # node=None → "cloud" label
    assert "n2" in out[1][0]


def test_fallback_shape_sort_stability():
    """Ties on count break by shape string ascending — determinism for tests."""
    recs = [
        _rec("p1", mesh_hit=False, fallback_chain=[["mZ", None]]),
        _rec("p2", mesh_hit=False, fallback_chain=[["mA", None]]),
    ]
    out = fallback_chain_shape_histogram(recs)
    assert out[0][0].startswith("mA@cloud")
    assert out[1][0].startswith("mZ@cloud")


# ---------------------------------------------------------------------------
# per_specialist_invocation_counts
# ---------------------------------------------------------------------------


def test_invocation_counts_empty():
    assert per_specialist_invocation_counts([]) == {}


def test_invocation_counts_groups_by_specialist_and_domain():
    recs = [
        _rec("p1", mesh_hit=True, specialist="coder", node="n1", domain="code"),
        _rec("p2", mesh_hit=True, specialist="coder", node="n1", domain="code"),
        _rec("p3", mesh_hit=True, specialist="coder", node="n1", domain="general"),
        _rec("p4", mesh_hit=True, specialist="math", node="n2", domain="math"),
        _rec("p5", mesh_hit=False, domain="multilingual"),  # cloud
    ]
    out = per_specialist_invocation_counts(recs)
    assert out["coder"] == {"code": 2, "general": 1}
    assert out["math"] == {"math": 1}
    assert out["cloud"] == {"multilingual": 1}


def test_invocation_counts_excludes_cloud_when_asked():
    recs = [
        _rec("p1", mesh_hit=True, specialist="coder", node="n1"),
        _rec("p2", mesh_hit=False),
    ]
    out = per_specialist_invocation_counts(recs, include_cloud=False)
    assert "cloud" not in out
    assert "coder" in out


def test_invocation_counts_handles_missing_signals():
    rec = _rec("p1", mesh_hit=True, specialist="coder", node="n1")
    rec["signals"] = {}  # drop the signals dict
    out = per_specialist_invocation_counts([rec])
    assert out["coder"] == {"unknown": 1}


# ---------------------------------------------------------------------------
# summary_stats
# ---------------------------------------------------------------------------


def test_summary_stats_empty():
    out = summary_stats([])
    assert out["total"] == 0
    assert out["mesh_hit_rate"] == 0.0
    assert out["mean_queue_ms"] == 0.0


def test_summary_stats_basic():
    recs = [
        _rec("p1", mesh_hit=True, specialist="coder", node="n1", queue_ms=100),
        _rec("p2", mesh_hit=True, specialist="coder", node="n1", queue_ms=200),
        _rec("p3", mesh_hit=True, specialist="math", node="n2", queue_ms=50),
        _rec("p4", mesh_hit=False),
    ]
    out = summary_stats(recs)
    assert out["total"] == 4
    assert out["mesh_hits"] == 3
    assert out["cloud_fallbacks"] == 1
    assert out["mesh_hit_rate"] == 0.75
    assert out["distinct_specialists"] == 2
    assert out["distinct_nodes"] == 2
    # mean_queue_ms over mesh-hit rows: (100+200+50)/3 = 116.66...
    assert abs(out["mean_queue_ms"] - 350 / 3) < 1e-6


def test_summary_stats_all_cloud_no_div_zero():
    recs = [_rec(f"p{i}", mesh_hit=False) for i in range(5)]
    out = summary_stats(recs)
    assert out["mesh_hits"] == 0
    assert out["mean_queue_ms"] == 0.0
    assert out["mesh_hit_rate"] == 0.0
