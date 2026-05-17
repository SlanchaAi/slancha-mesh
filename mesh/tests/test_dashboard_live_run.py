"""Tests for mesh.dashboard.live_run — pure-function panels over the
100K-corpus run ledger.

Mirrors the pattern in test_dashboard_panels.py: build small in-memory
record fixtures, assert chart-data outputs are correct + deterministic.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from mesh.dashboard.live_run import (
    cost_and_latency_summary,
    error_rate_over_time,
    kl_divergence_vs_uniform,
    live_run_summary,
    load_ledger_records,
    model_mix,
    throughput_over_time,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _rec(
    ts: str,
    *,
    model: str = "codestral:22b",
    backend: str = "ollama",
    ok: bool = True,
    in_tokens: int = 10,
    out_tokens: int = 20,
    latency_ms: int = 100,
    cost_usd: float = 0.0,
    error_code: str | None = None,
) -> dict:
    return {
        "ts": ts,
        "prompt_id": "p-x",
        "source": "test",
        "classifier_signals": {"domain": "code", "difficulty": "medium", "language": "en"},
        "route_decision": {"backend": backend, "model": model, "node": "spark-1"},
        "response": {"ok": ok, "error_code": error_code},
        "tokens": {"input": in_tokens, "output": out_tokens},
        "latency_ms": latency_ms,
        "cost_usd": cost_usd,
    }


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------


def test_load_ledger_records_skips_blank_lines(tmp_path: Path):
    p = tmp_path / "ledger.jsonl"
    p.write_text("\n" + json.dumps(_rec("2026-05-17T00:00:00Z")) + "\n\n", encoding="utf-8")
    records = load_ledger_records(p)
    assert len(records) == 1


def test_load_ledger_records_raises_on_bad_json(tmp_path: Path):
    p = tmp_path / "ledger.jsonl"
    p.write_text("{not json}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="line 1"):
        load_ledger_records(p)


# ---------------------------------------------------------------------------
# Throughput
# ---------------------------------------------------------------------------


def test_throughput_empty():
    assert throughput_over_time([], bucket_seconds=60) == []


def test_throughput_buckets_correctly():
    # 3 in bucket A (00:00), 2 in bucket B (00:01)
    records = [
        _rec("2026-05-17T00:00:01Z"),
        _rec("2026-05-17T00:00:30Z"),
        _rec("2026-05-17T00:00:59Z"),
        _rec("2026-05-17T00:01:05Z"),
        _rec("2026-05-17T00:01:55Z"),
    ]
    out = throughput_over_time(records, bucket_seconds=60)
    assert len(out) == 2
    assert out[0][2] == 3 and out[0][1] == pytest.approx(3 / 60)
    assert out[1][2] == 2 and out[1][1] == pytest.approx(2 / 60)


def test_throughput_sorted_ascending():
    # Out-of-order insertion → sorted output
    records = [
        _rec("2026-05-17T00:05:00Z"),
        _rec("2026-05-17T00:01:00Z"),
        _rec("2026-05-17T00:03:00Z"),
    ]
    out = throughput_over_time(records, bucket_seconds=60)
    timestamps = [pt[0] for pt in out]
    assert timestamps == sorted(timestamps)


# ---------------------------------------------------------------------------
# Error rate
# ---------------------------------------------------------------------------


def test_error_rate_empty():
    assert error_rate_over_time([], bucket_seconds=60) == []


def test_error_rate_computed_per_bucket():
    records = [
        _rec("2026-05-17T00:00:01Z", ok=True),
        _rec("2026-05-17T00:00:02Z", ok=False, error_code="timeout"),
        _rec("2026-05-17T00:00:03Z", ok=False, error_code="500"),
        _rec("2026-05-17T00:01:00Z", ok=True),
    ]
    out = error_rate_over_time(records, bucket_seconds=60)
    # bucket 1: 2/3 errors
    assert out[0][1] == pytest.approx(2 / 3)
    assert out[0][2] == 2 and out[0][3] == 3
    # bucket 2: 0/1 errors
    assert out[1][1] == 0.0


# ---------------------------------------------------------------------------
# Model mix + KL
# ---------------------------------------------------------------------------


def test_model_mix_counts():
    records = [
        _rec("2026-05-17T00:00:00Z", model="codestral:22b"),
        _rec("2026-05-17T00:00:00Z", model="codestral:22b"),
        _rec("2026-05-17T00:00:00Z", model="phi4:14b"),
    ]
    assert model_mix(records) == {"codestral:22b": 2, "phi4:14b": 1}


def test_kl_uniform_is_zero():
    counts = {"a": 10, "b": 10, "c": 10}
    assert kl_divergence_vs_uniform(counts) == pytest.approx(0.0, abs=1e-9)


def test_kl_skewed_is_positive():
    counts = {"a": 90, "b": 5, "c": 5}
    assert kl_divergence_vs_uniform(counts) > 0.5


def test_kl_single_model_is_zero():
    """One model used → no diversity dimension → KL = 0 by convention."""
    assert kl_divergence_vs_uniform({"only": 100}) == 0.0


def test_kl_empty_counts_is_zero():
    assert kl_divergence_vs_uniform({}) == 0.0


def test_kl_max_skew_equals_log_k():
    """Maximally skewed = all mass on one of k buckets → KL = log(k)."""
    counts = {"a": 100, "b": 0, "c": 0}
    # Only `a` has non-zero count → effective k = 1 → KL = 0
    assert kl_divergence_vs_uniform(counts) == 0.0
    # With two non-zero models, one heavy:
    counts2 = {"a": 999, "b": 1}
    kl = kl_divergence_vs_uniform(counts2)
    assert kl < math.log(2)  # below the max but positive
    assert kl > 0.5


# ---------------------------------------------------------------------------
# Cost + latency summary
# ---------------------------------------------------------------------------


def test_cost_latency_summary_empty():
    s = cost_and_latency_summary([])
    assert s["total_cost_usd"] == 0.0
    assert s["p95_latency_ms"] == 0
    assert s["per_backend_cost_usd"] == {}


def test_cost_latency_summary_aggregates():
    records = [
        _rec("2026-05-17T00:00:00Z", backend="ollama", latency_ms=100, cost_usd=0.0),
        _rec("2026-05-17T00:00:01Z", backend="ollama", latency_ms=200, cost_usd=0.0),
        _rec("2026-05-17T00:00:02Z", backend="openrouter", latency_ms=500, cost_usd=0.001),
        _rec("2026-05-17T00:00:03Z", backend="openrouter", latency_ms=900, cost_usd=0.002),
    ]
    s = cost_and_latency_summary(records)
    assert s["total_cost_usd"] == pytest.approx(0.003)
    assert s["per_backend_cost_usd"]["ollama"] == 0.0
    assert s["per_backend_cost_usd"]["openrouter"] == pytest.approx(0.003)
    assert s["p50_latency_ms"] in (200, 500)  # sort-rank dependent
    # p95 should be at least the second-highest
    assert s["p95_latency_ms"] >= 500


def test_cost_latency_summary_handles_missing_fields():
    """Missing tokens / latency / cost fields → fall back to 0 without raising."""
    rec = {
        "ts": "2026-05-17T00:00:00Z",
        "route_decision": {"backend": "ollama"},
        "response": {"ok": True},
    }
    s = cost_and_latency_summary([rec])
    assert s["total_cost_usd"] == 0.0
    assert s["total_input_tokens"] == 0
    assert s["p50_latency_ms"] == 0


# ---------------------------------------------------------------------------
# Live run summary
# ---------------------------------------------------------------------------


def test_live_run_summary_empty():
    s = live_run_summary([])
    assert s["total"] == 0
    assert s["error_rate"] == 0.0
    assert s["distinct_models"] == 0


def test_live_run_summary_counts():
    records = [
        _rec("2026-05-17T00:00:00Z", model="codestral:22b", ok=True),
        _rec("2026-05-17T00:00:00Z", model="codestral:22b", ok=False),
        _rec("2026-05-17T00:00:00Z", model="phi4:14b", ok=True),
    ]
    s = live_run_summary(records)
    assert s["total"] == 3
    assert s["errors"] == 1
    assert s["error_rate"] == pytest.approx(1 / 3)
    assert s["distinct_models"] == 2
    # KL > 0 since 2/3 vs 1/3 skewed
    assert s["model_mix_kl"] > 0.0
