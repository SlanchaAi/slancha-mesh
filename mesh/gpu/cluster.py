"""Cluster-wide GPU scheduling — heterogeneous-network placement.

Operator wants GPU scheduling that works across the whole mesh, not
just one Spark. Pattern stolen from
`exo/src/exo/master/placement.py` — Topology + filter-by-memory +
pick-best. We strip the model-sharding logic (exo splits one model
across many devices; we place whole workloads on one node) and add
reservation-awareness.

Flow:
  1. Each mesh node reports gpu_snapshot + active_reservations via
     heartbeat (see mesh/models.py NodeHeartbeat extension)
  2. Registry aggregates → ClusterGpuView per snapshot
  3. `pick_best_node(view, gb_requested, hard_filters...)` scores each
     node by free headroom (after subtracting active reservations) +
     hardware fit
  4. CLI / service POST /gpu/reserve uses the result + writes a
     reservation to the chosen node's local store

This module is pure — no I/O, no HTTP. The CLI + service wrap it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

from mesh.gpu.probe import GpuSnapshot
from mesh.gpu.reservations import Reservation


@dataclass(frozen=True)
class NodeGpuView:
    """One node's GPU state as seen by the registry.

    Reservation-aware: `free_gb_after_reservations` is the effective
    headroom a new workload can claim. Cluster scheduler picks based
    on this, not raw nvidia-smi free.
    """

    node_id: str
    friendly_name: str
    snapshot: GpuSnapshot
    active_reservations: list[Reservation] = field(default_factory=list)
    # Operator/admin-set; node-level total memory (e.g. 128 for GB10 unified).
    # Falls back to snapshot.mem_total_gb when not provided.
    declared_total_gb: Optional[float] = None
    # Hardware tags that affect placement decisions (e.g. "blackwell",
    # "apple-silicon", "fp4"). Populated by node hardware probe.
    hardware_tags: list[str] = field(default_factory=list)

    @property
    def total_gb(self) -> Optional[float]:
        if self.declared_total_gb is not None:
            return self.declared_total_gb
        return self.snapshot.mem_total_gb

    @property
    def reserved_gb(self) -> float:
        return sum(r.gb_requested for r in self.active_reservations)

    @property
    def used_gb(self) -> float:
        """Currently held by running CUDA processes, per nvidia-smi."""
        return self.snapshot.total_proc_memory_gb

    @property
    def free_gb_after_reservations(self) -> Optional[float]:
        """Effective headroom: total - max(used, reserved).

        We take MAX of used + reserved (not sum) because once a reserved
        workload starts, its memory shows up in `used` AND the reservation
        is still active — double-counting would block placement. The
        steady state is `used ≈ reserved`; transient state is one or
        the other.
        """
        total = self.total_gb
        if total is None:
            return None
        held = max(self.used_gb, self.reserved_gb)
        return max(0.0, total - held)


@dataclass(frozen=True)
class ClusterGpuView:
    """Aggregated GPU state across all mesh nodes."""

    snapshot_ts: datetime
    nodes: dict[str, NodeGpuView] = field(default_factory=dict)

    @property
    def total_free_gb(self) -> float:
        return sum(
            n.free_gb_after_reservations or 0.0 for n in self.nodes.values()
        )

    @property
    def total_used_gb(self) -> float:
        return sum(n.used_gb for n in self.nodes.values())

    @property
    def total_reserved_gb(self) -> float:
        return sum(n.reserved_gb for n in self.nodes.values())


@dataclass(frozen=True)
class PlacementResult:
    """Outcome of pick_best_node — either a winning node + score, or
    a structured rejection reason for the caller to surface."""

    chosen_node_id: Optional[str]
    fit_score: float = -math.inf
    reason: str = ""
    rejected: dict[str, str] = field(default_factory=dict)  # node_id → why

    @property
    def ok(self) -> bool:
        return self.chosen_node_id is not None


def filter_nodes_by_memory(
    view: ClusterGpuView,
    gb_required: float,
) -> list[NodeGpuView]:
    """Steals exo's `filter_cycles_by_memory` pattern — drop nodes that
    can't fit the workload after subtracting active reservations.

    Nodes whose total_gb is unknown (snapshot returned [N/A] AND no
    declared_total_gb) are EXCLUDED — we can't promise placement on a
    box whose capacity we don't know.
    """
    eligible: list[NodeGpuView] = []
    for n in view.nodes.values():
        free = n.free_gb_after_reservations
        if free is None:
            continue
        if free >= gb_required:
            eligible.append(n)
    return eligible


def pick_best_node(
    view: ClusterGpuView,
    gb_requested: float,
    require_hardware_tags: Optional[list[str]] = None,
    avoid_nodes: Optional[set[str]] = None,
) -> PlacementResult:
    """Cluster-wide placement decision.

    Filters in order:
      1. avoid_nodes — explicit exclusion list (e.g. node currently
         draining, or the requester's own node when --avoid-local).
      2. require_hardware_tags — workload-declared hard requirement
         (e.g. needs FP4 → exclude non-Blackwell).
      3. memory headroom — filter_nodes_by_memory.

    Among survivors, score by:
      - more free headroom = better (lets multiple workloads coexist)
      - fewer active reservations = simpler scheduling
      - lower current util = warmer cache, lower contention

    Returns PlacementResult with `chosen_node_id` set on success,
    None + populated `rejected` dict on failure.
    """
    rejected: dict[str, str] = {}
    avoid = avoid_nodes or set()
    require_tags = set(require_hardware_tags or [])

    candidates: list[NodeGpuView] = []
    for n in view.nodes.values():
        if n.node_id in avoid:
            rejected[n.node_id] = "in avoid_nodes"
            continue
        if require_tags and not require_tags.issubset(set(n.hardware_tags)):
            missing = require_tags - set(n.hardware_tags)
            rejected[n.node_id] = f"missing hardware tags: {sorted(missing)}"
            continue
        free = n.free_gb_after_reservations
        if free is None:
            rejected[n.node_id] = "total memory unknown"
            continue
        if free < gb_requested:
            rejected[n.node_id] = (
                f"insufficient free: {free:.1f} GB available, "
                f"{gb_requested:.1f} GB requested"
            )
            continue
        candidates.append(n)

    if not candidates:
        return PlacementResult(
            chosen_node_id=None,
            reason=(
                f"no eligible node for {gb_requested:.1f} GB across "
                f"{len(view.nodes)} nodes (rejected {len(rejected)})"
            ),
            rejected=rejected,
        )

    # Score: higher headroom + lower reservation count + lower util.
    def _score(n: NodeGpuView) -> float:
        free = n.free_gb_after_reservations or 0.0
        n_reservations = len(n.active_reservations)
        util = n.snapshot.util_pct or 0.0
        # Headroom dominates; reservation count is a tiebreaker; util
        # is a tertiary signal (lower = warmer cache).
        return free - 2.0 * n_reservations - 0.1 * util

    candidates.sort(key=_score, reverse=True)
    best = candidates[0]
    best_score = _score(best)
    return PlacementResult(
        chosen_node_id=best.node_id,
        fit_score=best_score,
        reason=(
            f"{best.friendly_name} ({best.node_id}): "
            f"free={best.free_gb_after_reservations:.1f} GB, "
            f"used={best.used_gb:.1f} GB, "
            f"reserved={best.reserved_gb:.1f} GB, "
            f"util={best.snapshot.util_pct or 0:.0f}%"
        ),
        rejected=rejected,
    )


def build_cluster_view_from_heartbeats(
    heartbeats: list[dict],
    snapshot_ts: Optional[datetime] = None,
) -> ClusterGpuView:
    """Aggregate per-node heartbeats (raw dicts as the registry sees them)
    into a ClusterGpuView. Heartbeats lacking a `gpu` field are skipped
    (older mesh nodes / non-GPU nodes / nodes with the v0.0.6 extension
    disabled).

    Wire format (NodeHeartbeat.gpu — see mesh/models.py extension):
        {
          "snapshot": {...},          # serialized GpuSnapshot
          "reservations": [...],      # list of serialized Reservation
          "declared_total_gb": float,
          "hardware_tags": [str, ...]
        }
    """
    snapshot_ts = snapshot_ts or datetime.now(timezone.utc)
    nodes: dict[str, NodeGpuView] = {}
    for hb in heartbeats:
        node_id = hb.get("node_id") or hb.get("heartbeat", {}).get("node_id")
        if not node_id:
            continue
        gpu_payload = hb.get("gpu") or hb.get("heartbeat", {}).get("gpu")
        if not gpu_payload:
            continue
        snap_dict = gpu_payload.get("snapshot")
        if not snap_dict:
            continue
        snap = _deserialize_snapshot(snap_dict)
        reservations = [
            Reservation.from_json(r) for r in gpu_payload.get("reservations", [])
        ]
        friendly = (
            hb.get("friendly_name")
            or hb.get("hardware", {}).get("friendly_name")
            or hb.get("heartbeat", {}).get("hardware", {}).get("friendly_name")
            or node_id
        )
        nodes[node_id] = NodeGpuView(
            node_id=node_id,
            friendly_name=friendly,
            snapshot=snap,
            active_reservations=reservations,
            declared_total_gb=gpu_payload.get("declared_total_gb"),
            hardware_tags=list(gpu_payload.get("hardware_tags", [])),
        )
    return ClusterGpuView(snapshot_ts=snapshot_ts, nodes=nodes)


def _deserialize_snapshot(d: dict) -> GpuSnapshot:
    """Inverse of GpuSnapshot serialization (planned alongside heartbeat ext).

    Tolerant of missing fields — defaults match GpuSnapshot's None
    semantics for unsupported probes.
    """
    from mesh.gpu.probe import GpuProcess

    probed_at_raw = d.get("probed_at")
    if isinstance(probed_at_raw, str):
        probed_at = datetime.fromisoformat(probed_at_raw)
    else:
        probed_at = datetime.now(timezone.utc)
    procs = [
        GpuProcess(
            pid=p["pid"],
            process_name=p["process_name"],
            used_memory_mib=p["used_memory_mib"],
            user=p.get("user"),
            cmdline=p.get("cmdline"),
            runtime_s=p.get("runtime_s"),
        )
        for p in d.get("processes", [])
    ]
    return GpuSnapshot(
        probed_at=probed_at,
        util_pct=d.get("util_pct"),
        mem_used_mib=d.get("mem_used_mib"),
        mem_free_mib=d.get("mem_free_mib"),
        mem_total_mib=d.get("mem_total_mib"),
        processes=procs,
        nvidia_smi_available=d.get("nvidia_smi_available", True),
    )


__all__ = [
    "ClusterGpuView",
    "NodeGpuView",
    "PlacementResult",
    "build_cluster_view_from_heartbeats",
    "filter_nodes_by_memory",
    "pick_best_node",
]
