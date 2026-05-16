"""Chart-data computers — pure functions over mesh_replay JSONL output.

Each function consumes a list of `DecisionRecord` dicts (parsed from the
JSONL emitted by `mesh.scripts.mesh_replay.replay_corpus`) and returns
plain Python data structures (lists / dicts / tuples). No streamlit, no
pandas, no plotting library. The Streamlit wrapper in `streamlit_app.py`
turns these returns into charts.

Why pure functions: testable on any platform (no GUI runtime), reusable
in non-Streamlit consumers (e.g., the nightly_smoke diff harness in
v0.0.4 #37 reads the same JSONL).

Record shape (see mesh.scripts.mesh_replay.replay_one):
    {prompt_id, prompt_hash, signals,
     decision: {chosen_specialist, chosen_node, node_url, model,
                reason, queue_ms, fallback_chain, mesh_hit,
                vs_cloud_baseline_cost},
     snapshot_ts: ISO-8601 UTC}
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TypeAlias

# Decision records are loosely-typed dicts from JSONL; we operate over
# them by key access rather than reconstructing typed objects, so panel
# code stays a thin transformation layer.
DecisionRecord: TypeAlias = dict[str, Any]


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------


def load_replay_records(path: Path | str) -> list[DecisionRecord]:
    """Parse a mesh_replay JSONL into a list of decision records.

    Blank lines are skipped. Invalid JSON lines raise ValueError with the
    line number, matching the pattern used by `mesh.scripts.mesh_replay.iter_corpus`.
    """
    path = Path(path)
    records: list[DecisionRecord] = []
    with path.open("r", encoding="utf-8") as f:
        for i, raw in enumerate(f, start=1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                records.append(json.loads(raw))
            except json.JSONDecodeError as exc:
                raise ValueError(f"replay JSONL line {i}: invalid JSON ({exc})") from exc
    return records


# ---------------------------------------------------------------------------
# Time-bucketed mesh-hit-rate
# ---------------------------------------------------------------------------


def _parse_ts(s: str) -> datetime:
    """Parse a snapshot_ts ISO string into an aware datetime.

    mesh_replay emits `snap.snapshot_ts.isoformat()` which is timezone-aware
    on Pydantic v2; fallback to UTC if naive (defensive).
    """
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _bucket(dt: datetime, seconds: int) -> datetime:
    """Floor a datetime to the previous `seconds`-aligned boundary."""
    epoch = dt.timestamp()
    floored = (int(epoch) // seconds) * seconds
    return datetime.fromtimestamp(floored, tz=timezone.utc)


def mesh_hit_rate_over_time(
    records: list[DecisionRecord],
    bucket_seconds: int = 3600,
) -> list[tuple[datetime, float, int]]:
    """Time-bucketed mesh-hit-rate.

    Returns a sorted list of `(bucket_start, hit_rate, sample_count)`
    tuples — one per non-empty bucket. `hit_rate` is in [0, 1].

    Empty `records` → empty list. Buckets with zero samples are not
    emitted (we don't fabricate zero-rate points where no data exists).
    The Streamlit caller turns this into a line/area chart.
    """
    if not records:
        return []
    by_bucket: dict[datetime, list[bool]] = defaultdict(list)
    for r in records:
        ts = _parse_ts(r["snapshot_ts"])
        bucket = _bucket(ts, bucket_seconds)
        by_bucket[bucket].append(bool(r["decision"]["mesh_hit"]))
    out: list[tuple[datetime, float, int]] = []
    for bucket in sorted(by_bucket):
        hits = by_bucket[bucket]
        rate = sum(hits) / len(hits)
        out.append((bucket, rate, len(hits)))
    return out


# ---------------------------------------------------------------------------
# Fallback-chain shape histogram
# ---------------------------------------------------------------------------


def fallback_chain_shape_histogram(
    records: list[DecisionRecord],
) -> list[tuple[str, int]]:
    """Count distinct fallback-chain shapes across the replay.

    A "shape" is the sequence of `(model, node_id_or_cloud)` strings in
    the decision's fallback_chain, joined by `→`. We summarize the
    *shape* of the fallback graph traffic took, not just the primary
    choice — gives a fast read of "which routes are still cloud-only
    when the primary fails."

    Returns descending-sorted `[(shape_string, count), ...]`. Stable
    secondary sort by shape string for determinism in tests.
    """
    counter: Counter[str] = Counter()
    for r in records:
        chain = r["decision"].get("fallback_chain", [])
        if not chain:
            shape = "(empty)"
        else:
            parts = []
            for pair in chain:
                if isinstance(pair, list) and len(pair) == 2:
                    model, node_id = pair
                else:
                    model, node_id = (str(pair), None)
                node_label = node_id if node_id else "cloud"
                parts.append(f"{model}@{node_label}")
            shape = " → ".join(parts)
        counter[shape] += 1
    # Descending by count, tiebreak by shape string ascending
    return sorted(counter.items(), key=lambda kv: (-kv[1], kv[0]))


# ---------------------------------------------------------------------------
# Per-specialist invocation counts (heatmap input)
# ---------------------------------------------------------------------------


def per_specialist_invocation_counts(
    records: list[DecisionRecord],
    include_cloud: bool = True,
) -> dict[str, dict[str, int]]:
    """Counts of `{specialist_or_cloud: {domain: count}}` across records.

    Output is a nested dict keyed by chosen_specialist (or "cloud" when
    `chosen_specialist is None`) → request domain → invocation count.
    Used as a heatmap input: rows = specialists, cols = domains.

    `include_cloud=False` drops the cloud-fallback row — useful for
    "mesh-only utilization" views where cloud noise hides the per-mesh
    spec distribution.
    """
    out: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for r in records:
        spec = r["decision"].get("chosen_specialist")
        domain = r.get("signals", {}).get("domain", "unknown")
        key = spec if spec else "cloud"
        if key == "cloud" and not include_cloud:
            continue
        out[key][domain] += 1
    # Convert nested defaultdicts to plain dicts for predictable JSON
    return {k: dict(v) for k, v in out.items()}


# ---------------------------------------------------------------------------
# Summary stats — single-glance read
# ---------------------------------------------------------------------------


def summary_stats(records: list[DecisionRecord]) -> dict[str, Any]:
    """Single-glance summary of a replay run.

    Returns:
      - total: int
      - mesh_hits: int
      - cloud_fallbacks: int
      - mesh_hit_rate: float in [0, 1]; 0.0 if empty
      - distinct_specialists: int (count of unique non-null chosen_specialist)
      - distinct_nodes: int (count of unique non-null chosen_node)
      - mean_queue_ms: float (mean over mesh-hit rows; 0.0 if no hits)

    Designed for the dashboard's top-card row.
    """
    if not records:
        return {
            "total": 0,
            "mesh_hits": 0,
            "cloud_fallbacks": 0,
            "mesh_hit_rate": 0.0,
            "distinct_specialists": 0,
            "distinct_nodes": 0,
            "mean_queue_ms": 0.0,
        }
    mesh_hits = 0
    queue_sum = 0
    specialists: set[str] = set()
    nodes: set[str] = set()
    for r in records:
        d = r["decision"]
        if d.get("mesh_hit"):
            mesh_hits += 1
            queue_sum += int(d.get("queue_ms", 0))
        if d.get("chosen_specialist"):
            specialists.add(d["chosen_specialist"])
        if d.get("chosen_node"):
            nodes.add(d["chosen_node"])
    total = len(records)
    return {
        "total": total,
        "mesh_hits": mesh_hits,
        "cloud_fallbacks": total - mesh_hits,
        "mesh_hit_rate": mesh_hits / total,
        "distinct_specialists": len(specialists),
        "distinct_nodes": len(nodes),
        "mean_queue_ms": (queue_sum / mesh_hits) if mesh_hits else 0.0,
    }
