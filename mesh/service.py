"""FastAPI subapp wrapping MeshRegistry — spec §5 + §11.

Mountable into slancha-api or run standalone. Endpoints:

- `POST /heartbeat`     — node-side ingest of `HeartbeatPostRequest`
- `GET  /registry`      — read the latest `RegistrySnapshot`
- `POST /probe-network` — aggregate the network views from recent heartbeats
- `POST /allocate`      — re-run the cluster allocator, return suggestions
- `GET  /health`        — liveness check (no auth required)

Auth (spec §11): Bearer token via `SLANCHA_NODE_TOKEN` env var. Unset or
empty env = auth disabled (dev mode); set = enforced on every endpoint
except `/health`. Constant-time compare; missing / malformed / wrong
header all 401/403 per RFC 6750.

Standalone:
    SLANCHA_NODE_TOKEN=... uvicorn mesh.service:app --port 8088

Mounted (slancha-api):
    from mesh.service import create_mesh_app
    app.mount("/mesh/v1", create_mesh_app(registry=shared_registry))
"""

from __future__ import annotations

import hmac
import os
from datetime import datetime
from typing import Annotated

from fastapi import Depends, FastAPI, Header, HTTPException, status
from pydantic import BaseModel, ConfigDict, Field

from mesh.models import DomainId, NetworkLink, NodeId, SpecialistId
from mesh.quality_probe import detect_drift
from mesh.registry import (
    HeartbeatEvent,
    HeartbeatPostRequest,
    HeartbeatPostResponse,
    MeshRegistry,
    RegistryGetResponse,
)

NODE_TOKEN_ENV = "SLANCHA_NODE_TOKEN"


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------


def _expected_token() -> str | None:
    """Configured bearer token, or None if auth is disabled.

    Empty-string and unset env both treated as disabled (dev mode).
    """
    tok = os.environ.get(NODE_TOKEN_ENV, "").strip()
    return tok or None


def verify_node_token(
    authorization: Annotated[str | None, Header()] = None,
) -> None:
    """Validate Bearer token against `SLANCHA_NODE_TOKEN`.

    Returns silently if auth is disabled OR the token matches.
    Raises 401 on missing/malformed header, 403 on wrong token.
    """
    expected = _expected_token()
    if expected is None:
        return
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="missing or malformed Authorization header",
            headers={"WWW-Authenticate": 'Bearer realm="slancha-mesh"'},
        )
    received = authorization[len("Bearer ") :].strip()
    if not hmac.compare_digest(received, expected):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="invalid bearer token",
        )


# ---------------------------------------------------------------------------
# Request / response schemas for endpoints that don't already have shapes
# ---------------------------------------------------------------------------


class AllocateRequest(BaseModel):
    """Body for `POST /allocate`."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    strategy: str = "tiered"
    traffic_mix: dict[DomainId, float] | None = None


class AllocateResponse(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    strategy: str
    suggestions: dict[NodeId, SpecialistId | None]


class ProbeNetworkResponse(BaseModel):
    """Aggregate of per-node `network_view` data from the latest heartbeats."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    snapshot_ts: datetime
    nodes_observed: int
    network_views: dict[NodeId, dict[NodeId, NetworkLink]] = Field(default_factory=dict)


class QualityObservationPostRequest(BaseModel):
    """Body for `POST /quality_observation` — Phase 6."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    specialist_id: SpecialistId
    score: float = Field(..., ge=0.0, le=5.0)
    sample_count: int = Field(..., ge=1)
    observation_source: str = Field(..., pattern="^(synthetic|shadow_traffic|real_traffic)$")
    observed_at: datetime | None = None


class QualityObservationResponse(BaseModel):
    """Ack + optional DriftEvent for `POST /quality_observation`."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    ack: bool
    prior_score: float | None
    drift: dict | None = None


class HealthResponse(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    status: str = "ok"
    auth_required: bool


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------


def create_mesh_app(registry: MeshRegistry | None = None) -> FastAPI:
    """Build a FastAPI subapp wrapping a `MeshRegistry`.

    Pass `registry` to share state across the host app (e.g., slancha-api
    long-lived registry) or for test injection. When `None`, a fresh
    `MeshRegistry` is created with the on-disk catalog auto-loaded so
    /registry exposes specialist cards from app boot (no separate
    bootstrap script needed; replaces the v0.0.x serve_with_catalog.py
    pattern).
    """
    if registry is not None:
        reg = registry
    else:
        from mesh.catalog import load_catalog

        try:
            cards = load_catalog()
        except Exception:
            # Surface load failures in logs but don't crash the service —
            # heartbeats can still populate the registry, and an empty
            # catalog is recoverable (matches prior v0.0.x behavior).
            import logging

            logging.getLogger(__name__).exception(
                "Catalog auto-load failed; starting with empty catalog"
            )
            cards = []
        reg = MeshRegistry(catalog=cards)

    app = FastAPI(
        title="Slancha-Mesh Registry",
        version="0.0.3",
        description="Node-side registry: heartbeats, snapshots, allocation.",
    )
    app.state.registry = reg

    @app.post(
        "/heartbeat",
        response_model=HeartbeatPostResponse,
        summary="Ingest a node heartbeat",
    )
    def post_heartbeat(
        req: HeartbeatPostRequest,
        _: Annotated[None, Depends(verify_node_token)],
    ) -> HeartbeatPostResponse:
        return reg.record_heartbeat(req)

    @app.get(
        "/registry",
        response_model=RegistryGetResponse,
        summary="Read the latest registry snapshot",
    )
    def get_registry(
        _: Annotated[None, Depends(verify_node_token)],
    ) -> RegistryGetResponse:
        return RegistryGetResponse(snapshot=reg.snapshot())

    @app.post(
        "/probe-network",
        response_model=ProbeNetworkResponse,
        summary="Aggregate the network views from recent heartbeats",
    )
    def post_probe_network(
        _: Annotated[None, Depends(verify_node_token)],
    ) -> ProbeNetworkResponse:
        snap = reg.snapshot()
        views: dict[NodeId, dict[NodeId, NetworkLink]] = {}
        latest_by_node: dict[NodeId, HeartbeatEvent] = {}
        for ev in reg.events:
            if isinstance(ev, HeartbeatEvent):
                latest_by_node[ev.heartbeat.node_id] = ev
        for node_id, ev in latest_by_node.items():
            if ev.heartbeat.network_view:
                views[node_id] = dict(ev.heartbeat.network_view)
        return ProbeNetworkResponse(
            snapshot_ts=snap.snapshot_ts,
            nodes_observed=len(snap.nodes),
            network_views=views,
        )

    @app.post(
        "/allocate",
        response_model=AllocateResponse,
        summary="Re-run the cluster allocator",
    )
    def post_allocate(
        body: AllocateRequest,
        _: Annotated[None, Depends(verify_node_token)],
    ) -> AllocateResponse:
        suggestions = reg.run_allocator(
            strategy=body.strategy,
            traffic_mix=body.traffic_mix,
        )
        return AllocateResponse(strategy=body.strategy, suggestions=suggestions)

    @app.get(
        "/health",
        response_model=HealthResponse,
        summary="Liveness probe (no auth)",
    )
    def get_health() -> HealthResponse:
        return HealthResponse(status="ok", auth_required=_expected_token() is not None)

    # ------------------------------------------------------------------
    # Phase 6 — router-observed quality observations + drift alerts
    # ------------------------------------------------------------------

    @app.post(
        "/quality_observation",
        response_model=QualityObservationResponse,
        summary="Record one router-observed quality score for a specialist",
    )
    def post_quality_observation(
        body: QualityObservationPostRequest,
        _: Annotated[None, Depends(verify_node_token)],
    ) -> QualityObservationResponse:
        """Write a probe-set-derived score into a specialist's card.

        Updates SpecialistCard.quality_router_observed +
        quality_sample_count + quality_observation_source on the matching
        catalog entry. Compares against the prior router_observed
        score; emits a `mesh.quality.drift` log line + returns a
        DriftEvent in the response when |delta| > 0.5.

        Caller is typically `python -m mesh.quality_probe` (CLI) but any
        operator-controlled probe runner can write here. Auth: same
        SLANCHA_NODE_TOKEN as the rest of the registry.
        """
        prior, _ev = reg.record_quality_observation(
            specialist_id=body.specialist_id,
            score=body.score,
            sample_count=body.sample_count,
            observation_source=body.observation_source,
            observed_at=body.observed_at,
        )
        drift = detect_drift(
            prior=prior,
            current=body.score,
            specialist_id=body.specialist_id,
        )
        if drift is not None:
            # Structured log line so CloudWatch / Langfuse can alert on it.
            import logging

            logging.getLogger("mesh.quality").warning(
                "mesh.quality.drift specialist=%s prior=%.3f new=%.3f delta=%.3f direction=%s threshold=%.3f",
                drift.specialist_id,
                drift.prior_score,
                drift.new_score,
                drift.delta,
                drift.direction,
                drift.threshold,
            )

        return QualityObservationResponse(
            ack=True,
            prior_score=prior,
            drift=(
                {
                    "specialist_id": drift.specialist_id,
                    "prior_score": drift.prior_score,
                    "new_score": drift.new_score,
                    "delta": drift.delta,
                    "direction": drift.direction,
                    "threshold": drift.threshold,
                }
                if drift is not None
                else None
            ),
        )

    # ------------------------------------------------------------------
    # v0.0.6 — cluster-wide GPU coordination (#46)
    # ------------------------------------------------------------------

    @app.get(
        "/gpu/cluster",
        summary="Aggregate GPU state across all mesh nodes",
    )
    def get_gpu_cluster(
        _: Annotated[None, Depends(verify_node_token)],
    ) -> dict:
        """Returns a cluster GPU view assembled from the most-recent
        heartbeat per node that carries a `gpu` payload. Nodes without
        a gpu field (non-GPU hosts, older clients) are omitted.

        Shape (consumed by mesh.gpu.cluster.build_cluster_view_from_heartbeats):
          {
            "snapshot_ts": "<iso8601>",
            "nodes": {
              "<node_id>": {
                "friendly_name": str,
                "snapshot": {...GpuSnapshot serialized...},
                "reservations": [...Reservation serialized...],
                "declared_total_gb": float | null,
                "hardware_tags": [str, ...],
                "free_gb_after_reservations": float | null,
                "used_gb": float,
                "reserved_gb": float
              }
            }
          }
        """
        from mesh.gpu.cluster import build_cluster_view_from_heartbeats
        from mesh.registry import HeartbeatEvent

        # Walk the event log for the most-recent heartbeat per node.
        latest: dict[str, dict] = {}
        for ev in reg.events:
            if isinstance(ev, HeartbeatEvent):
                hb = ev.heartbeat
                gpu = getattr(hb, "gpu", None)
                if gpu is None:
                    continue
                latest[hb.node_id] = {
                    "node_id": hb.node_id,
                    "friendly_name": hb.hardware.friendly_name,
                    "gpu": gpu,
                }

        view = build_cluster_view_from_heartbeats(list(latest.values()))
        return {
            "snapshot_ts": view.snapshot_ts.isoformat(),
            "nodes": {
                nid: {
                    "friendly_name": n.friendly_name,
                    "snapshot": _serialize_snapshot(n.snapshot),
                    "reservations": [r.to_json() for r in n.active_reservations],
                    "declared_total_gb": n.declared_total_gb,
                    "hardware_tags": list(n.hardware_tags),
                    "free_gb_after_reservations": n.free_gb_after_reservations,
                    "used_gb": n.used_gb,
                    "reserved_gb": n.reserved_gb,
                }
                for nid, n in view.nodes.items()
            },
            "total_free_gb": view.total_free_gb,
            "total_used_gb": view.total_used_gb,
            "total_reserved_gb": view.total_reserved_gb,
        }

    @app.post(
        "/gpu/reserve",
        summary="Cluster-aware GPU reservation (registers intent)",
    )
    def post_gpu_reserve(
        body: dict,
        _: Annotated[None, Depends(verify_node_token)],
    ) -> dict:
        """Register a cluster-level reservation intent.

        Body: {node_id?, gb_requested, duration_s, purpose?, hardware_tags?}

        If node_id provided, the reservation is recorded as "intended for
        that node" and SHOULD be claimed via `mesh-gpu reserve` on that
        node within the duration window. If omitted, the registry picks
        best-fit + returns the chosen node_id; caller is expected to ssh
        to that node + claim locally.

        This is a CLUSTER-LEVEL intent; the actual GPU memory hold lives
        in the chosen node's local ReservationStore. The registry tracks
        outstanding intents for visibility but doesn't enforce them at
        the kernel.
        """
        from mesh.gpu.cluster import build_cluster_view_from_heartbeats, pick_best_node
        from mesh.registry import HeartbeatEvent

        # Rebuild view from heartbeats
        latest: dict[str, dict] = {}
        for ev in reg.events:
            if isinstance(ev, HeartbeatEvent):
                hb = ev.heartbeat
                gpu = getattr(hb, "gpu", None)
                if gpu is None:
                    continue
                latest[hb.node_id] = {
                    "node_id": hb.node_id,
                    "friendly_name": hb.hardware.friendly_name,
                    "gpu": gpu,
                }
        view = build_cluster_view_from_heartbeats(list(latest.values()))

        gb = float(body.get("gb_requested", 0))
        if gb <= 0:
            raise HTTPException(status_code=400, detail="gb_requested must be > 0")

        target_node = body.get("node_id")
        require_tags = body.get("hardware_tags")

        if target_node is None:
            result = pick_best_node(
                view, gb_requested=gb, require_hardware_tags=require_tags,
            )
            if not result.ok:
                raise HTTPException(
                    status_code=409, detail={
                        "reason": result.reason,
                        "rejected": result.rejected,
                    },
                )
            target_node = result.chosen_node_id

        # v0.0.6: respond with the chosen target + remind caller to claim
        # locally. v0.0.7 will POST the reservation through to the chosen
        # node's local store via the mesh substrate (libp2p or HTTP fanout).
        return {
            "chosen_node_id": target_node,
            "gb_requested": gb,
            "duration_s": body.get("duration_s"),
            "claim_command": (
                f"ssh {target_node} -- mesh-gpu reserve "
                f"--gb {gb} --duration {body.get('duration_s', 3600)}s "
                f"--purpose {body.get('purpose', '')!r}"
            ),
            "note": "v0.0.6 returns intent only; v0.0.7 will fan out to node",
        }

    return app


def _serialize_snapshot(snap) -> dict:
    """Inverse of mesh.gpu.cluster._deserialize_snapshot."""
    return {
        "probed_at": snap.probed_at.isoformat(),
        "util_pct": snap.util_pct,
        "mem_used_mib": snap.mem_used_mib,
        "mem_free_mib": snap.mem_free_mib,
        "mem_total_mib": snap.mem_total_mib,
        "nvidia_smi_available": snap.nvidia_smi_available,
        "processes": [
            {
                "pid": p.pid,
                "process_name": p.process_name,
                "used_memory_mib": p.used_memory_mib,
                "user": p.user,
                "cmdline": p.cmdline,
                "runtime_s": p.runtime_s,
            }
            for p in snap.processes
        ],
    }


# Convenience module-level app for `uvicorn mesh.service:app`.
app = create_mesh_app()


__all__ = [
    "AllocateRequest",
    "AllocateResponse",
    "HealthResponse",
    "NODE_TOKEN_ENV",
    "ProbeNetworkResponse",
    "app",
    "create_mesh_app",
    "verify_node_token",
]
