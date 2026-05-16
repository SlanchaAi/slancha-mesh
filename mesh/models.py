"""Pydantic data models for Slancha-Mesh v0.

All shapes are `frozen=True` per spec §2 examples — once a heartbeat or
probe is constructed it doesn't mutate, so caching + event-sourcing
replay both work safely.

Type aliases use `str` newtypes (Pydantic doesn't need NewType-level
nominal typing; we keep the names for self-documentation).
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

# Newtype-ish aliases — kept as plain str for Pydantic compat.
NodeId = str
SpecialistId = str
DomainId = str
ModelId = str

DifficultyTier = Literal["easy", "medium", "hard"]
Backend = Literal["vllm", "llamacpp", "ollama", "mlx"]
HealthState = Literal["healthy", "degraded", "draining", "training", "unreachable"]
Arch = Literal["aarch64", "x86_64", "apple-silicon"]


class _Frozen(BaseModel):
    """Base for immutable Pydantic records."""

    model_config = ConfigDict(frozen=True, extra="forbid")


# ---------------------------------------------------------------------------
# NodeProbe — output of mesh/probe.py
# ---------------------------------------------------------------------------


class NodeProbe(_Frozen):
    """Hardware + network probe of a single mesh node.

    See spec §3.1. Fields that the probe cannot determine on a given
    platform are `None` (e.g., GB10 does not expose memory_bandwidth via
    nvidia-smi today — we record None and the catalog table fills in via
    `estimated_tps_at` instead).
    """

    node_id: NodeId
    friendly_name: str

    # Compute
    chip: str
    arch: Arch
    cuda_capability: str | None = None
    fp4_tops: float | None = None
    fp16_tops: float | None = None

    # Memory
    ram_total_gb: float
    ram_available_gb: float
    vram_total_gb: float | None = None
    vram_available_gb: float | None = None
    unified_memory: bool = False
    memory_bandwidth_gbs: float | None = None  # spec says required, but GB10 hides it

    # Network
    public_ipv4: str | None = None
    lan_interfaces: list[str] = Field(default_factory=list)
    bandwidth_to_master_mbps: float | None = None
    rtt_to_master_ms: float | None = None
    thunderbolt5: bool = False

    # Backends + storage
    available_backends: list[Backend] = Field(default_factory=list)
    disk_free_gb: float = 0.0

    # Provenance
    probed_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    probe_warnings: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# SpecialistCard — loaded from mesh/catalog/*.toml
# ---------------------------------------------------------------------------


class SpecialistCard(_Frozen):
    """A model card extending exo's TOML schema with Slancha routing fields.

    See spec §3.2. `estimated_tps_at` is a sparse lookup table keyed by
    hardware family (gb10, m4_pro, l40, ...) used when the live probe
    doesn't know `memory_bandwidth_gbs`.
    """

    model_id: ModelId
    specialist_id: SpecialistId  # local handle, e.g., "qwen3-math-7b-q4"
    domain: DomainId
    difficulty_tiers: list[DifficultyTier]
    languages: list[str] = Field(default_factory=lambda: ["en"])

    required_backend: Backend
    requires_fp4: bool = False

    storage_gb: float
    runtime_gb: float
    min_vram_gb: float
    context_window: int
    n_layers: int
    hidden_size: int | None = None

    estimated_tps_at: dict[str, float] = Field(default_factory=dict)
    supports_lora_finetune: bool = False
    upstream_model_card: str | None = None

    # Slancha-internal: coverage tier used by the tiered allocator.
    # Tier 1 = essentials (math/code/general); Tier 2 = important
    # (multilingual/tool_use/summarization); Tier 3 = specialized.
    coverage_tier: int = 1


# ---------------------------------------------------------------------------
# Allocator output
# ---------------------------------------------------------------------------


class NodeSuggestion(_Frozen):
    """Per-spec §3.3: what the allocator tells a node to host.

    `primary` is the always-loaded specialist on the node. `secondaries`
    are additional specialists the allocator decides this node can also
    host concurrently — populated only when the node's effective memory
    has 2× headroom past primary's runtime budget. Each secondary gets
    its own backend on its own port (heartbeat reports all loaded).

    Why `secondaries` lives alongside `alternates` rather than replacing:
    - `alternates`: ranked fallbacks if primary fails to load (one-of-many)
    - `secondaries`: also-loaded concurrent specialists (all-load-together)
    Different semantic; concurrent coexistence requires both fields.
    """

    node_id: NodeId
    primary: SpecialistCard | None  # None = registry-only node (no inference)
    secondaries: list[SpecialistCard] = Field(default_factory=list)
    alternates: list[SpecialistCard] = Field(default_factory=list)
    rationale: str = ""
    sticky: bool = False
    fit_score: float = 0.0


# ---------------------------------------------------------------------------
# Heartbeat / utilization
# ---------------------------------------------------------------------------


class LoadedModel(_Frozen):
    specialist_id: SpecialistId
    model_id: ModelId
    loaded_at: datetime
    estimated_tps: float | None = None


class NodeUtilization(_Frozen):
    gpu_util_pct: float = 0.0
    ram_util_pct: float = 0.0
    queue_depth: int = 0
    p50_latency_ms_60s: float | None = None
    p95_latency_ms_60s: float | None = None


class NetworkLink(_Frozen):
    rtt_ms: float | None = None
    bandwidth_mbps: float | None = None


class NodeHeartbeat(_Frozen):
    """Per spec §5. Posted to `POST /mesh/v1/heartbeat` every ~5s.

    `gpu` field (v0.0.6) carries cluster-wide GPU coordination payload:
    nvidia-smi snapshot + active local reservations + declared total
    + hardware tags. Optional + omitted on non-GPU nodes. Registry
    aggregates these into the cluster GPU view served at /gpu/cluster.

    Field shape (intentionally loose dict so this module stays decoupled
    from mesh.gpu — production callers serialize via
    GpuHeartbeatExtension below; mesh.gpu.cluster.build_cluster_view_from_heartbeats
    parses the dict back into typed objects).
    """

    node_id: NodeId
    ts: datetime
    hardware: NodeProbe
    loaded_models: list[LoadedModel] = Field(default_factory=list)
    util: NodeUtilization = Field(default_factory=NodeUtilization)
    recent_throughput: dict[ModelId, float] = Field(default_factory=dict)
    health: HealthState = "healthy"
    network_view: dict[NodeId, NetworkLink] = Field(default_factory=dict)
    gpu: dict | None = None  # v0.0.6: cluster GPU coordination payload


# ---------------------------------------------------------------------------
# Registry snapshot used by router
# ---------------------------------------------------------------------------


class NodeSummary(_Frozen):
    node_id: NodeId
    friendly_name: str
    health: HealthState
    last_seen: datetime
    loaded_specialist_ids: list[SpecialistId] = Field(default_factory=list)
    queue_depth: int = 0
    p95_latency_ms_60s: float | None = None
    node_url: str | None = None  # OpenAI-compatible base URL


class NodeBinding(_Frozen):
    """One (specialist, node) binding the registry knows about."""

    node_id: NodeId
    specialist_id: SpecialistId
    health: HealthState
    queue_depth: int
    p95_latency_ms_60s: float | None = None
    node_url: str | None = None
    last_seen: datetime


class Route(_Frozen):
    """A concrete routing candidate (specialist, node) the router can pick."""

    specialist_id: SpecialistId
    node_id: NodeId
    node_url: str | None
    estimated_queue_ms: int
    p95_latency_ms: float | None
    cost_estimate_cents: float = 0.0  # mesh-local nodes ≈ 0; cloud routes set this


class RegistrySnapshot(_Frozen):
    snapshot_ts: datetime
    nodes: dict[NodeId, NodeSummary] = Field(default_factory=dict)
    specialists: dict[SpecialistId, list[NodeBinding]] = Field(default_factory=dict)
    coverage: dict[DomainId, list[NodeId]] = Field(default_factory=dict)
    # router-facing pre-ranked routes per (domain, difficulty)
    ranked_routes: dict[str, list[Route]] = Field(default_factory=dict)
    catalog: dict[SpecialistId, SpecialistCard] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Router result
# ---------------------------------------------------------------------------


class MeshSelectionResult(_Frozen):
    """Output of `select_mesh_route`. Mirrors slancha-api SelectionResult.

    Spec §6: extends SelectionResult with node_id + node_url. None means
    the router fell through to cloud (no mesh node matched).
    """

    model: ModelId
    specialist_id: SpecialistId | None
    node_id: NodeId | None
    node_url: str | None
    reason: str
    queue_ms_estimated: int
    cluster_coverage_used: bool
    fallback_chain: list[tuple[ModelId, NodeId | None]] = Field(default_factory=list)
