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

import threading
from collections import OrderedDict
from datetime import datetime, timedelta, timezone
from collections.abc import Callable
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator

from mesh.identity import NodeIdentityError, verify_node_cert
from mesh.url_guard import validate_node_url

from mesh.allocator import allocate_cluster
from mesh.event_store import EventEnvelope, EventStore, NullEventStore
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

# Heartbeats arrive ~every 5s per node and otherwise accumulate without
# bound. Once the log grows past this many events, a heartbeat append
# triggers compaction (drop superseded heartbeats — see
# MeshRegistry._compact_heartbeats). 10k ≈ a generous ceiling well above any
# realistic node count, so compaction is rare and snapshot/allocator reads
# stay O(retained), not O(all-heartbeats-ever).
DEFAULT_MAX_EVENTS = 10_000


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


# Registry-owned codec between the live event objects and the opaque durable
# envelope (mesh.event_store). Kept here (not in the seam) so event schemas can
# evolve without changing the EventStore contract.
_EVENT_TYPES: dict[str, type[_Event]] = {
    "heartbeat": HeartbeatEvent,
    "node_left": NodeLeftEvent,
    "allocation": AllocationEvent,
    "quality_observation": QualityObservationEvent,
}


def _encode(ev: Event) -> EventEnvelope:
    return EventEnvelope(
        event_id=uuid4().hex,
        kind=ev.kind,
        ts=ev.ts.isoformat(),
        payload=ev.model_dump_json(),
    )


def _decode(env: EventEnvelope) -> Event:
    cls = _EVENT_TYPES.get(env.kind)
    if cls is None:
        raise ValueError(f"unknown event kind in durable store: {env.kind!r}")
    return cls.model_validate_json(env.payload)


# ---------------------------------------------------------------------------
# Request/response schemas (FastAPI handlers on slancha-api consume these)
# ---------------------------------------------------------------------------


class NodeIdentityCert(BaseModel):
    """Self-signed Ed25519 cert binding node_id ↔ public key (#102)."""

    model_config = {"frozen": True, "extra": "forbid"}
    node_id: str
    public_key_b64: str
    signature_b64: str


class HeartbeatPostRequest(BaseModel):
    """Body for POST /mesh/v1/heartbeat."""

    model_config = {"frozen": True, "extra": "forbid"}
    heartbeat: NodeHeartbeat
    node_url: str | None = Field(
        default=None,
        description="The node's OpenAI-compatible base URL. Required on first heartbeat.",
    )
    identity_cert: NodeIdentityCert | None = Field(
        default=None,
        description="Optional self-signed node identity cert (#102); when present, "
                    "the registry verifies it and pins node_id↔key.",
    )

    @field_validator("node_url")
    @classmethod
    def _safe_node_url(cls, v: str | None) -> str | None:
        return validate_node_url(v) if v else v  # SSRF guard (#98)


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
    """Event log with snapshot replay; superseded heartbeats are compacted.

    Thread-safety: every public mutation (`record_heartbeat`,
    `record_node_left`, `record_allocation`, `record_quality_observation`,
    `register_catalog`) and every reader that walks `self._events`
    (`snapshot`, `run_allocator`, `events`) is serialized by `self._lock`.
    The lock is a `threading.RLock` because compaction is called from
    inside `record_heartbeat` (re-entrant on the same thread).

    Why a thread lock (not `asyncio.Lock`): the FastAPI handlers in
    `mesh.registry_app` are *synchronous* `def`, so Starlette dispatches
    them on its anyio threadpool. Even with a single uvicorn worker, two
    `POST /heartbeat` calls hit `record_heartbeat` from different threads
    at once, so an `asyncio.Lock` would be the wrong primitive (no event
    loop is being shared between them) and a single-worker assumption is
    insufficient to serialize writers.

    The log is append-only except that, past `max_events`, a heartbeat
    append drops heartbeats that have been superseded by a newer one for the
    same node (`_compact_heartbeats`) — provably snapshot/allocator-preserving,
    since both only read the latest heartbeat per node.
    """

    def __init__(
        self,
        catalog: list[SpecialistCard] | None = None,
        *,
        max_events: int = DEFAULT_MAX_EVENTS,
        store: EventStore | None = None,
        clock: "Callable[[], datetime] | None" = None,
        require_node_identity: bool = False,
    ) -> None:
        self._events: list[Event] = []
        self._max_events = max_events
        # #102: node_id → pinned Ed25519 public key (TOFU). Once a node presents a
        # valid identity cert, a later heartbeat claiming the same node_id with a
        # different key (or no cert) is rejected. `require_node_identity` makes a
        # valid cert MANDATORY on every heartbeat (regulated profile).
        self._node_pubkeys: dict[NodeId, str] = {}
        self._require_node_identity = require_node_identity
        # Server clock for stamping heartbeat receive-time (#102). Injectable so
        # tests control "now"; defaults to real UTC.
        self._clock: "Callable[[], datetime]" = clock or (lambda: datetime.now(timezone.utc))
        # Durability seam (issue: on-prem persistence). Default = no durability
        # (in-memory only, pre-seam behavior). A durable store survives restarts.
        self._store: EventStore = store or NullEventStore()
        # specialist_id -> SpecialistCard, used by the snapshot to enrich
        # NodeBindings with card metadata.
        self._catalog: dict[SpecialistId, SpecialistCard] = {
            c.specialist_id: c for c in (catalog or [])
        }
        # node_id -> known node_url (last reported)
        self._node_urls: dict[NodeId, str] = {}
        # Re-entrant: _compact_heartbeats is called from inside
        # record_heartbeat, which already holds the lock.
        self._lock = threading.RLock()
        # Boot replay: rebuild the in-memory read model from the durable log
        # (no-op for NullEventStore). Done before serving any write.
        for env in self._store.replay():
            self._events.append(_decode(env))

    def _record(self, ev: Event) -> None:
        """Durably persist `ev`, then add it to the in-memory read model.

        Caller must hold `self._lock`. Durable-FIRST: if the store raises, the
        read model is left untouched, so the durable log and the read model never
        silently diverge (neither has the event; the caller sees the error).
        """
        self._store.append(_encode(ev))
        self._events.append(ev)

    # --- ingestion ---

    def _check_node_identity(self, req: HeartbeatPostRequest) -> None:
        """Verify + pin the node's identity cert (#102). Caller holds the lock.
        Raises NodeIdentityError on an invalid cert, a pin violation
        (impersonation), a missing cert when required, or a cert-less heartbeat
        from a node that previously authenticated (downgrade)."""
        node_id = req.heartbeat.node_id
        cert = req.identity_cert
        if cert is not None:
            if not verify_node_cert(cert.model_dump(), node_id):
                raise NodeIdentityError(f"invalid identity cert for node {node_id!r}")
            pinned = self._node_pubkeys.get(node_id)
            if pinned is not None and pinned != cert.public_key_b64:
                raise NodeIdentityError(
                    f"node {node_id!r} is pinned to a different key — impersonation refused")
            self._node_pubkeys[node_id] = cert.public_key_b64
            return
        if self._require_node_identity:
            raise NodeIdentityError(f"node {node_id!r} must present an identity cert (required)")
        if node_id in self._node_pubkeys:
            raise NodeIdentityError(
                f"node {node_id!r} previously presented an identity cert; a cert-less "
                "heartbeat is refused (downgrade)")

    def record_heartbeat(self, req: HeartbeatPostRequest) -> HeartbeatPostResponse:
        with self._lock:
            self._check_node_identity(req)
            if req.node_url:
                self._node_urls[req.heartbeat.node_id] = req.node_url
            self._record(
                # SERVER-STAMP the event time (#102): a node controls
                # `heartbeat.ts`, and using it for the unreachable/age calc let a
                # node send a future ts to look alive forever, or (via an
                # impersonated heartbeat) a far-past ts to evict a live peer. The
                # event ts is now the server's receive time; the node's reported
                # ts is preserved in `heartbeat.heartbeat.ts` as telemetry.
                HeartbeatEvent(
                    ts=self._clock(), node_url=req.node_url, heartbeat=req.heartbeat
                )
            )
            if len(self._events) > self._max_events:
                self._compact_heartbeats()
            return HeartbeatPostResponse(ack=True, next_due_seconds=5)

    def record_node_left(self, node_id: NodeId, reason: str = "graceful") -> None:
        with self._lock:
            self._record(
                NodeLeftEvent(ts=datetime.now(timezone.utc), node_id=node_id, reason=reason)
            )

    def record_allocation(
        self, strategy: str, suggestions: dict[NodeId, SpecialistId | None]
    ) -> None:
        with self._lock:
            self._record(
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
        with self._lock:
            self._record(ev)
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
        """Read-only view of the event log (for tests + debugging).

        Superseded heartbeats may have been dropped by compaction; the latest
        heartbeat per node and all non-heartbeat events are always present.
        """
        with self._lock:
            return list(self._events)

    def _compact_heartbeats(self) -> None:
        """Drop superseded heartbeats, keeping only the latest per node.

        Heartbeats arrive ~every 5s per node and otherwise accumulate without
        bound — and both readers (snapshot replay, run_allocator) grow O(n)
        with the log. This is safe to do without changing any result: both
        only ever use the *latest* heartbeat per node, and the snapshot's
        node_left handling compares a node_left against that node's latest
        heartbeat, never an older one. So removing any non-latest heartbeat
        leaves both the snapshot and the allocator output unchanged. Rare
        non-heartbeat events (node_left / allocation / quality_observation)
        are retained in place.

        Caller must hold `self._lock` (the two passes — read indices, then
        rebind — are not atomic on their own; a concurrent append between
        them would otherwise be filtered out and silently lost).
        """
        latest_hb_idx: dict[NodeId, int] = {}
        for i, ev in enumerate(self._events):
            if isinstance(ev, HeartbeatEvent):
                latest_hb_idx[ev.heartbeat.node_id] = i
        keep = set(latest_hb_idx.values())
        self._events = [
            ev
            for i, ev in enumerate(self._events)
            if not isinstance(ev, HeartbeatEvent) or i in keep
        ]
        # Bound the DURABLE log too: replace it with the compacted set so an
        # append-only store doesn't grow forever (slow boot replay). Optional —
        # a store without `compact` is simply left to grow (safe). Re-encoding is
        # fine: compact replaces the log wholesale.
        compact = getattr(self._store, "compact", None)
        if compact is not None:
            compact([_encode(ev) for ev in self._events])

    def register_catalog(self, cards: list[SpecialistCard]) -> None:
        with self._lock:
            for c in cards:
                self._catalog[c.specialist_id] = c

    # --- snapshot construction ---

    def snapshot(self, now: datetime | None = None) -> RegistrySnapshot:
        """Replay the event log into a RegistrySnapshot.

        Strategy: walk events in order, keeping only the latest heartbeat
        per node_id. Nodes whose latest heartbeat is older than
        NODE_UNREACHABLE_AFTER are marked unreachable; nodes with a
        subsequent NodeLeftEvent are dropped entirely.

        Held under `self._lock` so a concurrent compaction can't rebind
        `self._events` mid-walk to a list with the just-acked beat
        filtered out.
        """
        now = now or datetime.now(timezone.utc)
        latest: OrderedDict[NodeId, HeartbeatEvent] = OrderedDict()
        left: set[NodeId] = set()
        with self._lock:
            events_snapshot = list(self._events)
            catalog_snapshot = dict(self._catalog)
            node_urls_snapshot = dict(self._node_urls)
        for ev in events_snapshot:
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
            age = now - ev.ts  # server-stamped receive time, not the node's claimed ts (#102)
            health = hb.health
            if age > NODE_UNREACHABLE_AFTER:
                health = "unreachable"

            # Node-level fallback URL (last-reported); per-specialist URLs
            # on each LoadedModel take precedence so a node serving several
            # specialists on distinct ports binds each to the right one.
            node_url = ev.node_url or node_urls_snapshot.get(node_id)
            nodes[node_id] = NodeSummary(
                node_id=node_id,
                friendly_name=hb.hardware.friendly_name,
                health=health,
                last_seen=ev.ts,
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
                    node_url=lm.node_url or node_url,
                    last_seen=ev.ts,
                )
                specialists.setdefault(lm.specialist_id, []).append(binding)
                card = catalog_snapshot.get(lm.specialist_id)
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
            catalog=catalog_snapshot,
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

        Reads `self._events` + `self._catalog` under `self._lock` to get
        a consistent input view; the allocator itself (pure CPU on the
        snapshot) runs without the lock, and `record_allocation` re-takes
        the lock to append the resulting event.
        """
        with self._lock:
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
