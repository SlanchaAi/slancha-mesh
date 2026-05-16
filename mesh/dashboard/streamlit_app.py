"""Streamlit renderer for the mesh-replay dashboard.

Run from a host with streamlit installed:
    streamlit run mesh/dashboard/streamlit_app.py -- --replay PATH/TO/replay.jsonl

The wrapper is intentionally thin: it imports panel computers from
`mesh.dashboard.panels` (pure functions, testable without streamlit) and
turns each return value into a streamlit widget. Add new panels by
adding a pure function to panels.py and a `render_*` call here.

Mac doesn't have streamlit installed; this module imports it lazily so
the rest of `mesh.dashboard` (panels + tests) keeps working without it.
Spark's slancha-test/dashboard mounts this app.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

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
    ap = argparse.ArgumentParser(description="Mesh-replay dashboard.")
    ap.add_argument("--replay", type=Path, required=True, help="Path to a mesh_replay JSONL.")
    ap.add_argument("--bucket-seconds", type=int, default=3600, help="Hit-rate time-bucket.")
    return ap.parse_args(argv)


def render(argv: list[str] | None = None) -> None:  # pragma: no cover — UI runtime
    """Entry point — invoked by `streamlit run mesh/dashboard/streamlit_app.py`."""
    import streamlit as st  # lazy: tests + panels.py work without streamlit installed

    args = _parse_args(argv)
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


if __name__ == "__main__":  # pragma: no cover — UI runtime
    render(sys.argv[1:])
