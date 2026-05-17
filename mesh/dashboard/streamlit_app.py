"""Streamlit renderer for mesh-replay + live-run dashboards.

Two modes:

    # Replay-mode (mesh_replay JSONL, post-hoc analysis)
    streamlit run mesh/dashboard/streamlit_app.py -- --replay PATH/TO/replay.jsonl

    # Live-run mode (100K-corpus route-through ledger, near-real-time)
    streamlit run mesh/dashboard/streamlit_app.py -- --ledger PATH/TO/ledger.jsonl

The wrapper is intentionally thin: it imports panel computers from
`mesh.dashboard.panels` + `mesh.dashboard.live_run` (pure functions,
testable without streamlit) and turns each return value into a streamlit
widget. Add new panels by adding a pure function to the relevant module
and a `render_*` call here.

Mac doesn't have streamlit installed; this module imports it lazily so
the rest of `mesh.dashboard` (panels + tests) keeps working without it.
Spark's slancha-test/dashboard mounts this app.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from mesh.dashboard.live_run import (
    cost_and_latency_summary,
    error_rate_over_time,
    live_run_summary,
    load_ledger_records,
    model_mix,
    throughput_over_time,
)
from mesh.dashboard.panels import (
    fallback_chain_shape_histogram,
    load_replay_records,
    mesh_hit_rate_over_time,
    per_specialist_invocation_counts,
    summary_stats,
)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse args from streamlit's `-- ...` tail.

    Streamlit forwards everything after `--` to the script; we just
    consume our flags from sys.argv[1:].
    """
    ap = argparse.ArgumentParser(description="Mesh-replay + live-run dashboard.")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--replay", type=Path, help="Path to a mesh_replay JSONL.")
    g.add_argument("--ledger", type=Path, help="Path to a live-run ledger JSONL (100K-corpus route-through).")
    ap.add_argument(
        "--bucket-seconds",
        type=int,
        default=3600,
        help="Hit-rate time-bucket for replay mode (default 3600).",
    )
    ap.add_argument(
        "--throughput-bucket-seconds",
        type=int,
        default=60,
        help="Throughput time-bucket for live-run mode (default 60).",
    )
    return ap.parse_args(argv)


def _render_replay(st, args) -> None:  # pragma: no cover — UI runtime
    """Replay-mode page (mesh_replay JSONL → post-hoc analysis)."""
    records = load_replay_records(args.replay)

    st.set_page_config(page_title="Slancha-Mesh Replay", layout="wide")
    st.title("Slancha-Mesh — Replay Dashboard")
    st.caption(f"Source: `{args.replay}` · {len(records)} decisions")

    # --- summary row ---
    stats = summary_stats(records)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total prompts", stats["total"])
    c2.metric("Mesh hits", f"{stats['mesh_hits']} ({stats['mesh_hit_rate']:.0%})")
    c3.metric("Distinct specialists", stats["distinct_specialists"])
    c4.metric("Mean queue (mesh-hit)", f"{stats['mean_queue_ms']:.0f} ms")

    # --- mesh-hit-rate over time ---
    st.subheader("Mesh-hit rate over time")
    hr_data = mesh_hit_rate_over_time(records, bucket_seconds=args.bucket_seconds)
    if hr_data:
        st.line_chart(
            {"hit_rate": [pt[1] for pt in hr_data], "sample_count": [pt[2] for pt in hr_data]},
            x_label=f"bucket ({args.bucket_seconds}s)",
        )
    else:
        st.info("No bucketed data — replay is empty.")

    # --- fallback-chain shape histogram ---
    st.subheader("Fallback-chain shape distribution")
    shapes = fallback_chain_shape_histogram(records)
    if shapes:
        # Top 15 to keep the chart legible
        top = shapes[:15]
        st.bar_chart({s: c for s, c in top})
    else:
        st.info("No fallback chains observed.")

    # --- per-specialist invocations (heatmap-like) ---
    st.subheader("Per-specialist × per-domain invocations")
    counts = per_specialist_invocation_counts(records, include_cloud=True)
    if counts:
        # Pivot to a 2-d dict-of-rows for st.dataframe
        all_domains = sorted({d for row in counts.values() for d in row})
        table = {
            spec: [counts[spec].get(d, 0) for d in all_domains]
            for spec in sorted(counts)
        }
        st.dataframe({"domain": all_domains, **table})
    else:
        st.info("No invocations recorded.")


def _render_live(st, args) -> None:  # pragma: no cover — UI runtime
    """Live-run mode page (100K-corpus route-through ledger)."""
    records = load_ledger_records(args.ledger)

    st.set_page_config(page_title="Slancha-Mesh Live Run", layout="wide")
    st.title("Slancha-Mesh — Live Run (100K corpus)")
    st.caption(f"Ledger: `{args.ledger}` · {len(records)} requests routed")

    # --- top-card summary ---
    s = live_run_summary(records)
    cost = cost_and_latency_summary(records)
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total requests", s["total"])
    c2.metric("Errors", f"{s['errors']} ({s['error_rate']:.1%})")
    c3.metric("Distinct models", s["distinct_models"])
    c4.metric("Model-mix KL", f"{s['model_mix_kl']:.3f}",
              help="KL vs uniform. 0 = even spread. Higher = skewed toward one model.")
    c5.metric("Total cost", f"${cost['total_cost_usd']:.4f}")

    # --- throughput over time ---
    st.subheader("Throughput (requests / s)")
    tp = throughput_over_time(records, bucket_seconds=args.throughput_bucket_seconds)
    if tp:
        st.line_chart(
            {"req_per_s": [pt[1] for pt in tp], "samples": [pt[2] for pt in tp]},
            x_label=f"bucket ({args.throughput_bucket_seconds}s)",
        )
    else:
        st.info("No throughput data — ledger is empty.")

    # --- error rate over time ---
    st.subheader("Error rate over time")
    er = error_rate_over_time(records, bucket_seconds=args.throughput_bucket_seconds)
    if er:
        st.line_chart(
            {"error_rate": [pt[1] for pt in er], "errors": [pt[2] for pt in er]},
            x_label=f"bucket ({args.throughput_bucket_seconds}s)",
        )
    else:
        st.info("No error-rate data.")

    # --- model mix ---
    st.subheader("Model invocation counts")
    mix = model_mix(records)
    if mix:
        # Descending by count
        sorted_mix = sorted(mix.items(), key=lambda kv: -kv[1])
        st.bar_chart({m or "(none)": c for m, c in sorted_mix})
    else:
        st.info("No model invocations recorded.")

    # --- latency percentiles + per-backend cost ---
    st.subheader("Latency + cost by backend")
    p1, p2, p3 = st.columns(3)
    p1.metric("p50 latency", f"{cost['p50_latency_ms']} ms")
    p2.metric("p95 latency", f"{cost['p95_latency_ms']} ms")
    p3.metric("p99 latency", f"{cost['p99_latency_ms']} ms")
    if cost["per_backend_cost_usd"]:
        st.bar_chart(cost["per_backend_cost_usd"])
    else:
        st.info("No per-backend cost data.")


def render(argv: list[str] | None = None) -> None:  # pragma: no cover — UI runtime
    """Entry point — invoked by `streamlit run mesh/dashboard/streamlit_app.py`."""
    import streamlit as st  # lazy: tests + panels.py work without streamlit installed

    args = _parse_args(argv)
    if args.replay is not None:
        _render_replay(st, args)
    else:
        _render_live(st, args)


if __name__ == "__main__":  # pragma: no cover — UI runtime
    render(sys.argv[1:])
