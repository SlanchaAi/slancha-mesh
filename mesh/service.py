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
    empty `MeshRegistry` is created.
    """
    reg = registry if registry is not None else MeshRegistry()

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

    return app


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
