"""Oracle-label dashboard panel computers — pure functions over the
oracle JSONL spark writes during the 472e judge pass.

Companion to `live_run.py` (live throughput / error / cost) and `panels.py`
(mesh_replay post-hoc). This module targets the *labeled* output: each
floodgate row decorated with a judge_score (1-5) + reasoning + an optional
"better_model" recommendation from the 472e Qwen3-Coder-30B-A3B-FP8 judge.

Oracle record shape (proposed — confirm with spark before locking):

    {"prompt_id":          "wildchat-0000123",
     "prompt_text":        "...",
     "source":             "allenai/WildChat-1M",
     "signals":            {"domain": "code", "difficulty": "medium",
                              "language": "en", ...},
     "route_decision":     {"backend": "ollama"|"openrouter",
                              "model":     "codestral:22b",
                              "node":      "spark-1"},
     "response":           {"ok": true, "content": "...", "error_code": null},
     "tokens":             {"input": 42, "output": 128},
     "latency_ms":         850,
     "cost_usd":           0.0,
     "oracle": {"judge_score":    4,             # 1-5 int
                  "judge_reason":   "Correct but verbose. ...",
                  "better_model":   "phi4:14b"|null,    # spark's "alt-model better" flag
                  "judge_model":    "qwen3-coder-30b-a3b-fp8",
                  "judge_ts":       "2026-05-17T01:30:00Z"}}

If spark's shape differs the only place that knows is `_oracle_score()` +
`_oracle_better_model()`; one fix lands everywhere.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TypeAlias

OracleRecord: TypeAlias = dict[str, Any]


# ---------------------------------------------------------------------------
# I/O + shape adapters
# ---------------------------------------------------------------------------


def load_oracle_records(path: Path | str) -> list[OracleRecord]:
    """Parse an oracle JSONL. Blank lines skipped. Invalid lines raise
    ValueError with line number."""
    path = Path(path)
    out: list[OracleRecord] = []
    with path.open("r", encoding="utf-8") as f:
        for i, raw in enumerate(f, start=1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                out.append(json.loads(raw))
            except json.JSONDecodeError as exc:
                raise ValueError(f"oracle JSONL line {i}: invalid JSON ({exc})") from exc
    return out


def _oracle_score(rec: OracleRecord) -> int | None:
    """Extract the judge score. Returns None if missing or malformed."""
    o = rec.get("oracle") or {}
    s = o.get("judge_score")
    try:
        return int(s) if s is not None else None
    except (TypeError, ValueError):
        return None


def _oracle_better_model(rec: OracleRecord) -> str | None:
    o = rec.get("oracle") or {}
    bm = o.get("better_model")
    if not bm or not isinstance(bm, str):
        return None
    return bm


def _oracle_ts(rec: OracleRecord) -> datetime | None:
    o = rec.get("oracle") or {}
    ts = o.get("judge_ts")
    if not ts:
        return None
    try:
        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _bucket(dt: datetime, seconds: int) -> datetime:
    epoch = dt.timestamp()
    floored = (int(epoch) // seconds) * seconds
    return datetime.fromtimestamp(floored, tz=timezone.utc)


# ---------------------------------------------------------------------------
# 1. Quality score histogram
# ---------------------------------------------------------------------------


def quality_score_histogram(records: list[OracleRecord]) -> dict[int, int]:
    """Count of judge_score values across all records.

    Returns a dict keyed by score 1-5 (and possibly outliers — never silently
    drops bad data). Records missing a usable score are not counted.
    """
    counts: Counter[int] = Counter()
    for r in records:
        s = _oracle_score(r)
        if s is not None:
            counts[s] += 1
    return dict(counts)


# ---------------------------------------------------------------------------
# 2. Quality over time
# ---------------------------------------------------------------------------


def quality_over_time(
    records: list[OracleRecord],
    bucket_seconds: int = 300,
) -> list[tuple[datetime, float, int]]:
    """Time-bucketed mean judge_score.

    Returns `(bucket_start, mean_score, sample_count)` per non-empty bucket,
    sorted ascending. Records without a judge_score or judge_ts are skipped.
    """
    by_bucket: dict[datetime, list[int]] = defaultdict(list)
    for r in records:
        s = _oracle_score(r)
        ts = _oracle_ts(r)
        if s is None or ts is None:
            continue
        by_bucket[_bucket(ts, bucket_seconds)].append(s)
    out: list[tuple[datetime, float, int]] = []
    for bucket in sorted(by_bucket):
        scores = by_bucket[bucket]
        out.append((bucket, sum(scores) / len(scores), len(scores)))
    return out


# ---------------------------------------------------------------------------
# 3. Per-domain × per-model quality heatmap
# ---------------------------------------------------------------------------


def per_domain_quality_matrix(
    records: list[OracleRecord],
) -> dict[str, dict[str, dict[str, float | int]]]:
    """{domain: {model: {mean_score, n}}}.

    Cell value is a nested dict so caller can choose to render the mean,
    the sample count, or both (a streamlit heatmap can color by mean,
    annotate by n). Missing models grouped under the empty string.
    """
    bucket: dict[tuple[str, str], list[int]] = defaultdict(list)
    for r in records:
        s = _oracle_score(r)
        if s is None:
            continue
        domain = r.get("signals", {}).get("domain", "unknown")
        model = r.get("route_decision", {}).get("model") or ""
        bucket[(domain, model)].append(s)
    out: dict[str, dict[str, dict[str, float | int]]] = defaultdict(dict)
    for (domain, model), scores in bucket.items():
        out[domain][model] = {
            "mean_score": round(sum(scores) / len(scores), 3),
            "n": len(scores),
        }
    return {d: dict(v) for d, v in out.items()}


# ---------------------------------------------------------------------------
# 4. Alt-model-recommended rate by domain
# ---------------------------------------------------------------------------


def alt_recommended_rate_by_domain(
    records: list[OracleRecord],
) -> dict[str, dict[str, float | int]]:
    """{domain: {rate, n, n_recommended, top_alt_model}}.

    `rate` is the fraction of records in that domain where the judge
    flagged a different model as better. `top_alt_model` is the single
    most-recommended alternative for that domain.
    """
    by_domain: dict[str, list[str | None]] = defaultdict(list)
    for r in records:
        domain = r.get("signals", {}).get("domain", "unknown")
        alt = _oracle_better_model(r)
        by_domain[domain].append(alt)
    out: dict[str, dict[str, float | int]] = {}
    for domain, alts in by_domain.items():
        n_total = len(alts)
        recs = [a for a in alts if a]
        top_alt = Counter(recs).most_common(1)
        out[domain] = {
            "rate": round(len(recs) / n_total, 4) if n_total else 0.0,
            "n": n_total,
            "n_recommended": len(recs),
            "top_alt_model": top_alt[0][0] if top_alt else None,
        }
    return out


# ---------------------------------------------------------------------------
# 5. Quality-vs-latency scatter
# ---------------------------------------------------------------------------


def quality_vs_latency_scatter(
    records: list[OracleRecord],
    max_points: int = 5000,
) -> list[tuple[int, int, str]]:
    """`(latency_ms, judge_score, domain)` triples.

    Capped at `max_points` (deterministic head-of-iteration sample) so the
    streamlit scatter stays responsive on 100K-row ledgers. Records missing
    score or latency are skipped.
    """
    out: list[tuple[int, int, str]] = []
    for r in records:
        if len(out) >= max_points:
            break
        s = _oracle_score(r)
        latency = r.get("latency_ms")
        if s is None or latency is None:
            continue
        domain = r.get("signals", {}).get("domain", "unknown")
        out.append((int(latency), s, domain))
    return out


# ---------------------------------------------------------------------------
# 6. Top-card summary
# ---------------------------------------------------------------------------


def oracle_summary(records: list[OracleRecord]) -> dict[str, Any]:
    """Single-glance summary for the dashboard top-card row.

    Returns:
      - total:          all rows (whether labeled or not)
      - labeled:        rows with a usable judge_score
      - mean_score:     mean across labeled rows (0.0 if none)
      - pct_acceptable: share of labeled with score >= 4
      - pct_failure:    share of labeled with score <= 2
      - alt_rate:       share of labeled where better_model was recommended
      - distinct_judges: count of distinct judge_model values
    """
    total = len(records)
    if total == 0:
        return {
            "total": 0,
            "labeled": 0,
            "mean_score": 0.0,
            "pct_acceptable": 0.0,
            "pct_failure": 0.0,
            "alt_rate": 0.0,
            "distinct_judges": 0,
        }
    scores: list[int] = []
    alt_count = 0
    judges: set[str] = set()
    for r in records:
        s = _oracle_score(r)
        if s is not None:
            scores.append(s)
        if _oracle_better_model(r):
            alt_count += 1
        o = r.get("oracle") or {}
        jm = o.get("judge_model")
        if jm:
            judges.add(jm)
    labeled = len(scores)
    if labeled == 0:
        return {
            "total": total,
            "labeled": 0,
            "mean_score": 0.0,
            "pct_acceptable": 0.0,
            "pct_failure": 0.0,
            "alt_rate": 0.0,
            "distinct_judges": len(judges),
        }
    return {
        "total": total,
        "labeled": labeled,
        "mean_score": round(sum(scores) / labeled, 3),
        "pct_acceptable": round(sum(1 for s in scores if s >= 4) / labeled, 4),
        "pct_failure": round(sum(1 for s in scores if s <= 2) / labeled, 4),
        "alt_rate": round(alt_count / labeled, 4),
        "distinct_judges": len(judges),
    }
