"""mesh.dashboard — chart-data computers + Streamlit wrapper.

`panels.py` holds pure functions: parsed-JSONL → chart data. No streamlit
import, no plotting. Unit-tested on every platform.

`streamlit_app.py` is the thin renderer: imports streamlit + panels, wires
records → panels → charts. Run from Spark (or anywhere streamlit is
installed) via `streamlit run mesh/dashboard/streamlit_app.py -- --replay PATH`.
"""

from mesh.dashboard.panels import (
    DecisionRecord,
    fallback_chain_shape_histogram,
    load_replay_records,
    mesh_hit_rate_over_time,
    per_specialist_invocation_counts,
    summary_stats,
)

__all__ = [
    "DecisionRecord",
    "fallback_chain_shape_histogram",
    "load_replay_records",
    "mesh_hit_rate_over_time",
    "per_specialist_invocation_counts",
    "summary_stats",
]
