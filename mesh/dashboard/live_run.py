"""Live-run panel computers — pure functions over the 100K-corpus ledger.

Companion to `panels.py`, which targets mesh_replay JSONL. This module
targets the ledger format spark writes during a live route-through run
of the 100K v3 corpus (see corpus/training/v3/manifest.json).

Each record is one prompt → response, with classifier signals, the route
decision, response status, token counts, latency, and cost. Pure-function
chart computers below; streamlit wrapper lives at `streamlit_app.py`.

Ledger record shape (proposed — confirm with spark before locking):

    {"ts":                "2026-05-17T00:30:00.123456Z",
     "prompt_id":         "wildchat-0000123",
     "source":            "allenai/WildChat-1M",
     "classifier_signals": {"domain": "code", "difficulty": "medium",
                              "language": "en", "needs_tools": false,
                              "route_class": "standard"},
     "route_decision":    {"backend":   "ollama"|"openrouter"|"cloud",
                              "model":      "codestral:22b",
                              "node":       "spark-1"|null,
                              "fallback":   ["primary_model", ...]},
     "response":          {"ok":        true|false,
                              "content":   "...",         # may be omitted for ok=false
                              "error_code": null|"timeout"|"500"|"context_overflow"|...},
     "tokens":            {"input": 42, "output": 128},
     "latency_ms":        850,
     "cost_usd":          0.0}

If spark settles on a different schema, the adapters below should be the
only place that knows the shape — every panel function operates on
LedgerRecord dicts.
"""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TypeAlias

LedgerRecord: TypeAlias = dict[str, Any]


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------


def load_ledger_records(path: Path | str) -> list[LedgerRecord]:
    """Parse a live-run ledger JSONL.

    Blank lines skipped. Invalid lines raise ValueError with line number,
    matching the pattern in `mesh.dashboard.panels.load_replay_records`.
    """
    path = Path(path)
    out: list[LedgerRecord] = []
    with path.open("r", encoding="utf-8") as f:
        for i, raw in enumerate(f, start=1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                out.append(json.loads(raw))
            except json.JSONDecodeError as exc:
                raise ValueError(f"ledger JSONL line {i}: invalid JSON ({exc})") from exc
    return out


# ---------------------------------------------------------------------------
# Time helpers (mirror panels._parse_ts / _bucket but on `ts` field)
# ---------------------------------------------------------------------------


def _parse_ts(s: str) -> datetime:
    dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _bucket(dt: datetime, seconds: int) -> datetime:
    epoch = dt.timestamp()
    floored = (int(epoch) // seconds) * seconds
    return datetime.fromtimestamp(floored, tz=timezone.utc)


# ---------------------------------------------------------------------------
# Throughput — req/s in time buckets
# ---------------------------------------------------------------------------


def throughput_over_time(
    records: list[LedgerRecord],
    bucket_seconds: int = 60,
) -> list[tuple[datetime, float, int]]:
    """Time-bucketed throughput.

    Returns `(bucket_start, req_per_s, sample_count)` per non-empty bucket,
    sorted ascending by bucket_start. `req_per_s = sample_count /
    bucket_seconds`. Empty `records` → empty list.
    """
    if not records:
        return []
    by_bucket: dict[datetime, int] = defaultdict(int)
    for r in records:
        ts = _parse_ts(r["ts"])
        by_bucket[_bucket(ts, bucket_seconds)] += 1
    out: list[tuple[datetime, float, int]] = []
    for bucket in sorted(by_bucket):
        n = by_bucket[bucket]
        out.append((bucket, n / bucket_seconds, n))
    return out


# ---------------------------------------------------------------------------
# Error rate — non-ok responses / total per bucket
# ---------------------------------------------------------------------------


def error_rate_over_time(
    records: list[LedgerRecord],
    bucket_seconds: int = 60,
) -> list[tuple[datetime, float, int, int]]:
    """Time-bucketed error rate.

    Returns `(bucket_start, error_rate, errors, total)` per non-empty
    bucket. `error_rate` is in [0,1]. A record counts as an error when
    its `response.ok` field is falsy.
    """
    if not records:
        return []
    by_bucket: dict[datetime, list[bool]] = defaultdict(list)
    for r in records:
        ts = _parse_ts(r["ts"])
        ok = bool(r.get("response", {}).get("ok"))
        by_bucket[_bucket(ts, bucket_seconds)].append(ok)
    out: list[tuple[datetime, float, int, int]] = []
    for bucket in sorted(by_bucket):
        oks = by_bucket[bucket]
        errors = sum(1 for ok in oks if not ok)
        total = len(oks)
        rate = errors / total if total else 0.0
        out.append((bucket, rate, errors, total))
    return out


# ---------------------------------------------------------------------------
# Model-mix diversity — per-model counts + KL vs uniform
# ---------------------------------------------------------------------------


def model_mix(records: list[LedgerRecord]) -> dict[str, int]:
    """Counts per chosen model id. Records missing a model are counted
    under the empty string ("") so caller can see omissions."""
    out: Counter[str] = Counter()
    for r in records:
        model = r.get("route_decision", {}).get("model") or ""
        out[model] += 1
    return dict(out)


def kl_divergence_vs_uniform(counts: dict[str, int]) -> float:
    """KL(P || U) where P is the empirical model distribution and U is
    uniform over the observed model set.

    Returns 0.0 when only one model was used (no diversity dimension to
    measure), or when the input is empty. Higher value = more skewed.
    Useful as a single-number "model-mix diversity" gauge for the
    dashboard headline metric.

    Formula:
        KL(P || U) = sum_i p_i * log(p_i / (1/k))
                   = sum_i p_i * (log p_i + log k)
                   = log k - H(P)
    """
    keys = [k for k, v in counts.items() if v > 0]
    if len(keys) <= 1:
        return 0.0
    total = sum(counts[k] for k in keys)
    if total == 0:
        return 0.0
    k = len(keys)
    entropy = 0.0
    for key in keys:
        p = counts[key] / total
        if p > 0:
            entropy -= p * math.log(p)
    return math.log(k) - entropy


# ---------------------------------------------------------------------------
# Cost / latency / token rollups
# ---------------------------------------------------------------------------


def cost_and_latency_summary(records: list[LedgerRecord]) -> dict[str, Any]:
    """Aggregate cost + latency stats across the run.

    Returns:
      - total_cost_usd: float (sum of cost_usd, missing → 0)
      - total_input_tokens, total_output_tokens: int
      - p50_latency_ms, p95_latency_ms, p99_latency_ms: int
      - per_backend_cost_usd: {backend: float}
    """
    if not records:
        return {
            "total_cost_usd": 0.0,
            "total_input_tokens": 0,
            "total_output_tokens": 0,
            "p50_latency_ms": 0,
            "p95_latency_ms": 0,
            "p99_latency_ms": 0,
            "per_backend_cost_usd": {},
        }
    latencies: list[int] = []
    total_cost = 0.0
    in_toks = 0
    out_toks = 0
    per_backend: dict[str, float] = defaultdict(float)
    for r in records:
        cost = float(r.get("cost_usd") or 0.0)
        total_cost += cost
        toks = r.get("tokens") or {}
        in_toks += int(toks.get("input") or 0)
        out_toks += int(toks.get("output") or 0)
        latency = r.get("latency_ms")
        if latency is not None:
            latencies.append(int(latency))
        backend = r.get("route_decision", {}).get("backend") or "unknown"
        per_backend[backend] += cost
    latencies.sort()

    def _pct(p: float) -> int:
        if not latencies:
            return 0
        idx = min(len(latencies) - 1, int(len(latencies) * p))
        return latencies[idx]

    return {
        "total_cost_usd": round(total_cost, 6),
        "total_input_tokens": in_toks,
        "total_output_tokens": out_toks,
        "p50_latency_ms": _pct(0.5),
        "p95_latency_ms": _pct(0.95),
        "p99_latency_ms": _pct(0.99),
        "per_backend_cost_usd": {k: round(v, 6) for k, v in per_backend.items()},
    }


def live_run_summary(records: list[LedgerRecord]) -> dict[str, Any]:
    """Top-card summary for the dashboard headline row.

    Returns a compact dict combining throughput-derived totals and the
    cost+latency rollup. `model_mix_kl` is the KL-vs-uniform diversity
    score (0 = single model used, higher = more skewed)."""
    if not records:
        return {
            "total": 0,
            "errors": 0,
            "error_rate": 0.0,
            "distinct_models": 0,
            "model_mix_kl": 0.0,
        }
    total = len(records)
    errors = sum(1 for r in records if not r.get("response", {}).get("ok"))
    counts = model_mix(records)
    return {
        "total": total,
        "errors": errors,
        "error_rate": errors / total if total else 0.0,
        "distinct_models": sum(1 for v in counts.values() if v > 0),
        "model_mix_kl": round(kl_divergence_vs_uniform(counts), 4),
    }
