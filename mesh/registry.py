"""Event-sourced registry (in-memory) + FastAPI route shapes.

v0.0.1 scope:
- Heartbeat ingestion (append-only event log).
- Snapshot construction (replay events → RegistrySnapshot).
- Pure-Python; no network server required to test.

The actual FastAPI app lives on slancha-api; we expose the route
dependencies + schemas here so they can be imported there without
re-implementation. The `MeshRegistry` class is the in-memory store.

Out of scope for v0.0.1: persistence to disk, multi-tenant scoping,
auth (spec §11 will add bearer tokens), libp2p replication.
"""

from __future__ import annotations

from collections import OrderedDict
from datetime import datetime, timedelta, timezone
from typing import Literal

from pydantic import BaseModel, Field

from mesh.allocator import allocate_cluster
from mesh.models import (
    DomainId,
    NodeBinding,
    NodeHeartbeat,
    NodeId,
    NodeSummary,
    RegistrySnapshot,
    Route,
    SpecialistCard,
    SpecialistId,
)

# Spec §3.4: nodes unreachable >5 min are treated as left.
NODE_UNREACHABLE_AFTER = timedelta(minutes=5)


# ---------------------------------------------------------------------------
# Event types — event-sourced log
# ---------------------------------------------------------------------------


class _Event(BaseModel):
    """Base envelope for registry events."""

    model_config = {"frozen": True, "extra": "forbid"}
    kind: Literal["heartbeat", "node_left", "allocation"]
    ts: datetime


class HeartbeatEvent(_Event):
    kind: Literal["heartbeat"] = "heartbeat"
    node_url: str | None = None
    heartbeat: NodeHeartbeat


class NodeLeftEvent(_Event):
    kind: Literal["node_left"] = "node_left"
    node_id: NodeId
    reason: str = "graceful"


class AllocationEvent(_Event):
    kind: Literal["allocation"] = "allocation"
    strategy: str
    suggestions: dict[NodeId, SpecialistId | None]


class QualityObservationEvent(_Event):
    """Phase 6 — one router-observed quality score for a specialist."""

    kind: Literal["quality_observation"] = "quality_observation"
    specialist_id: SpecialistId
    score: float = Field(..., ge=0.0, le=5.0)
    sample_count: int = Field(..., ge=0)
    observation_source: Literal["synthetic", "shadow_traffic", "real_traffic"]


Event = HeartbeatEvent | NodeLeftEvent | AllocationEvent | QualityObservationEvent


# ---------------------------------------------------------------------------
# Request/response schemas (FastAPI handlers on slancha-api consume these)
# ---------------------------------------------------------------------------


class HeartbeatPostRequest(BaseModel):
    """Body for POST /mesh/v1/heartbeat."""

    model_config = {"frozen": True, "extra": "forbid"}
    heartbeat: NodeHeartbeat
    node_url: str | None = Field(
        default=None,
        description="The node's OpenAI-compatible base URL. Required on first heartbeat.",
    )


class HeartbeatPostResponse(BaseModel):
    model_config = {"frozen": True, "extra": "forbid"}
    ack: bool = True
    next_due_seconds: int = 5
    allocator_suggestion_id: SpecialistId | None = None


class RegistryGetResponse(BaseModel):
    model_config = {"frozen": True, "extra": "forbid"}
    snapshot: RegistrySnapshot


# ---------------------------------------------------------------------------
# MeshRegistry — the event-sourced store
# ---------------------------------------------------------------------------


class MeshRegistry:
    """Append-only event log with snapshot replay.

    Not thread-safe; FastAPI handlers should wrap mutations in an
    asyncio.Lock or rely on uvicorn's single-worker mode in dev. The
    contract is intentionally narrow: ingest events, replay to build a
    snapshot. No mutation API beyond `record_heartbeat` / `record_node_left`.
    """

    def __init__(self, catalog: list[SpecialistCard] | None = None) -> None:
        self._events: list[Event] = []
        # specialist_id -> SpecialistCard, used by the snapshot to enrich
        # NodeBindings with card metadata.
        self._catalog: dict[SpecialistId, SpecialistCard] = {
            c.specialist_id: c for c in (catalog or [])
        }
        # node_id -> known node_url (last reported)
        self._node_urls: dict[NodeId, str] = {}

    # --- ingestion ---

    def record_heartbeat(self, req: HeartbeatPostRequest) -> HeartbeatPostResponse:
        if req.node_url:
            self._node_urls[req.heartbeat.node_id] = req.node_url
        self._events.append(
            HeartbeatEvent(ts=req.heartbeat.ts, node_url=req.node_url, heartbeat=req.heartbeat)
        )
        return HeartbeatPostResponse(ack=True, next_due_seconds=5)

    def record_node_left(self, node_id: NodeId, reason: str = "graceful") -> None:
        self._events.append(
            NodeLeftEvent(ts=datetime.now(timezone.utc), node_id=node_id, reason=reason)
        )

    def record_allocation(
        self, strategy: str, suggestions: dict[NodeId, SpecialistId | None]
    ) -> None:
        self._events.append(
            AllocationEvent(ts=datetime.now(timezone.utc), strategy=strategy, suggestions=suggestions)
        )

    def record_quality_observation(
        self,
        *,
        specialist_id: SpecialistId,
        score: float,
        sample_count: int,
        observation_source: Literal["synthetic", "shadow_traffic", "real_traffic"],
        observed_at: datetime | None = None,
    ) -> tuple[float | None, "QualityObservationEvent"]:
        """Append a quality observation + update the specialist's card.

        Returns (prior_score_or_None, event). Callers (the service-side
        admin endpoint) use prior_score to compute a DriftEvent via
        mesh.quality_probe.detect_drift.

        Cards are frozen=True, so we replace the existing card with a
        new instance carrying the updated quality fields. Snapshot
        consumers see the new values from the next /registry call.
        """
        when = observed_at or datetime.now(timezone.utc)
        ev = QualityObservationEvent(
            ts=when,
            specialist_id=specialist_id,
            score=score,
            sample_count=sample_count,
            observation_source=observation_source,
        )
        self._events.append(ev)

        prior = None
        old_card = self._catalog.get(specialist_id)
        if old_card is not None:
            prior = old_card.quality_router_observed
            self._catalog[specialist_id] = old_card.model_copy(
                update={
                    "quality_router_observed": score,
                    "quality_sample_count": (old_card.quality_sample_count or 0) + sample_count,
                    "quality_observation_source": observation_source,
                }
            )
        return prior, ev

    @property
    def events(self) -> list[Event]:
        """Read-only view of the append-only log (for tests + debugging)."""
        return list(self._events)

    def register_catalog(self, cards: list[SpecialistCard]) -> None:
        for c in cards:
            self._catalog[c.specialist_id] = c

    # --- snapshot construction ---

    def snapshot(self, now: datetime | None = None) -> RegistrySnapshot:
        """Replay the event log into a RegistrySnapshot.

        Strategy: walk events in order, keeping only the latest heartbeat
        per node_id. Nodes whose latest heartbeat is older than
        NODE_UNREACHABLE_AFTER are marked unreachable; nodes with a
        subsequent NodeLeftEvent are dropped entirely.
        """
        now = now or datetime.now(timezone.utc)
        latest: OrderedDict[NodeId, HeartbeatEvent] = OrderedDict()
        left: set[NodeId] = set()
        for ev in self._events:
            if isinstance(ev, HeartbeatEvent):
                # A re-join after `node_left` clears the left flag.
                left.discard(ev.heartbeat.node_id)
                latest[ev.heartbeat.node_id] = ev
            elif isinstance(ev, NodeLeftEvent):
                left.add(ev.node_id)
                latest.pop(ev.node_id, None)
            # AllocationEvent is informational; doesn't affect snapshot

        nodes: dict[NodeId, NodeSummary] = {}
        specialists: dict[SpecialistId, list[NodeBinding]] = {}
        coverage: dict[DomainId, list[NodeId]] = {}

        for node_id, ev in latest.items():
            if node_id in left:
                continue
            hb = ev.heartbeat
            age = now - hb.ts
            health = hb.health
            if age > NODE_UNREACHABLE_AFTER:
                health = "unreachable"

            node_url = ev.node_url or self._node_urls.get(node_id)
            nodes[node_id] = NodeSummary(
                node_id=node_id,
                friendly_name=hb.hardware.friendly_name,
                health=health,
                last_seen=hb.ts,
                loaded_specialist_ids=[lm.specialist_id for lm in hb.loaded_models],
                queue_depth=hb.util.queue_depth,
                p95_latency_ms_60s=hb.util.p95_latency_ms_60s,
                node_url=node_url,
            )
            for lm in hb.loaded_models:
                binding = NodeBinding(
                    node_id=node_id,
                    specialist_id=lm.specialist_id,
                    health=health,
                    queue_depth=hb.util.queue_depth,
                    p95_latency_ms_60s=hb.util.p95_latency_ms_60s,
                    node_url=node_url,
                    last_seen=hb.ts,
                )
                specialists.setdefault(lm.specialist_id, []).append(binding)
                card = self._catalog.get(lm.specialist_id)
                if card is not None:
                    coverage.setdefault(card.domain, []).append(node_id)

        # ranked_routes is filled by the router (mesh.select) when it
        # consumes the snapshot. We pre-populate empty dict; the
        # snapshot is otherwise router-agnostic.
        return RegistrySnapshot(
            snapshot_ts=now,
            nodes=nodes,
            specialists=specialists,
            coverage=coverage,
            ranked_routes={},
            catalog=dict(self._catalog),
        )

    # --- ops: re-run allocator ---

    def run_allocator(
        self,
        strategy: str = "tiered",
        traffic_mix: dict[DomainId, float] | None = None,
    ) -> dict[NodeId, SpecialistId | None]:
        """Re-run the cluster allocator using the latest hardware snapshot.

        Returns a `{node_id: specialist_id | None}` map; persists an
        AllocationEvent so the event log stays the canonical source.
        """
        latest_hardware = {}
        for ev in self._events:
            if isinstance(ev, HeartbeatEvent):
                latest_hardware[ev.heartbeat.node_id] = ev.heartbeat.hardware
        nodes = list(latest_hardware.values())
        catalog = list(self._catalog.values())
        suggestions = allocate_cluster(
            nodes=nodes, catalog=catalog, traffic_mix=traffic_mix, strategy=strategy  # type: ignore[arg-type]
        )
        result = {
            nid: (s.primary.specialist_id if s.primary else None)
            for nid, s in suggestions.items()
        }
        self.record_allocation(strategy=strategy, suggestions=result)
        return result


# ---------------------------------------------------------------------------
# Router glue: pre-rank routes for a snapshot
# ---------------------------------------------------------------------------


def build_ranked_routes(snapshot: RegistrySnapshot) -> dict[str, list[Route]]:
    """Construct `(domain, difficulty) -> [Route]` from a snapshot.

    Routes are ranked by a simple composite: lower queue + lower p95 +
    healthy. The router consumes this in `mesh.select`; we materialize
    it here so snapshots cache cheaply.

    Key format: `"{domain}|{difficulty}"`. Difficulty is taken from the
    specialist card's `difficulty_tiers` — one route entry per tier the
    specialist supports.
    """

    ranked: dict[str, list[Route]] = {}

    for spec_id, bindings in snapshot.specialists.items():
        card = snapshot.catalog.get(spec_id)
        if card is None:
            continue
        for tier in card.difficulty_tiers:
            key = f"{card.domain}|{tier}"
            routes: list[Route] = []
            for b in bindings:
                if b.health != "healthy":
                    continue
                queue_ms = b.queue_depth * 500  # crude: 500ms / queued req
                routes.append(
                    Route(
                        specialist_id=spec_id,
                        node_id=b.node_id,
                        node_url=b.node_url,
                        estimated_queue_ms=queue_ms,
                        p95_latency_ms=b.p95_latency_ms_60s,
                        cost_estimate_cents=0.0,
                    )
                )
            routes.sort(
                key=lambda r: (
                    r.estimated_queue_ms,
                    r.p95_latency_ms or 99999,
                )
            )
            if routes:
                ranked.setdefault(key, []).extend(routes)

    return ranked


__all__ = [
    "AllocationEvent",
    "HeartbeatEvent",
    "HeartbeatPostRequest",
    "HeartbeatPostResponse",
    "MeshRegistry",
    "NODE_UNREACHABLE_AFTER",
    "NodeLeftEvent",
    "RegistryGetResponse",
    "build_ranked_routes",
]
