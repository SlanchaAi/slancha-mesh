"""Pluggable durability seam for the event-sourced registry (opt-in).

The registry is in-memory + non-durable by default (restart → state rebuilt from
fresh heartbeats in ~5s). This seam lets a downstream supply a DURABLE event
store so control-plane state survives a restart — WITHOUT the registry walking
the durable log on every read. The split (from an architect/SRE review):

  • the registry keeps its in-memory read model for ALL reads (no per-request
    store I/O);
  • the store is pure DURABILITY: ``append`` each event, and ``replay`` once at
    boot to rebuild the read model.

Contract (this is a committed API — a downstream pins an exact mesh revision):

  • ``append(env)`` — persist durably BEFORE returning. Called on the heartbeat
    path, under the registry lock. It MUST NOT block on a transient store outage
    (a blocked append stalls the whole control plane) — a durable impl buffers /
    degrades internally and raises only on an error the caller must see. The
    registry is durable-FIRST: if ``append`` raises, the in-memory read model is
    NOT mutated, so the log and the read model never silently diverge.
  • ``replay()`` — called ONCE at boot, before the registry serves writes; yields
    every persisted envelope in append order. Read from the write-primary (no
    replica lag at startup).
  • Retention is the STORE's policy — the registry never deletes/trims the durable
    log (the in-memory read model bounds itself separately via heartbeat
    compaction). An audit-grade store keeps history.
  • Single-writer topology: the durable log + the in-memory read model are
    single-writer. Active/active needs external sequencing — out of scope here.

The envelope is OPAQUE: the registry owns the codec (``Event`` ↔ ``EventEnvelope``),
so mesh can evolve event schemas without breaking the store contract — the store
is a typed byte bucket keyed by ``(event_id, kind, ts)``.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class EventEnvelope:
    """Opaque, durable representation of one registry event.

    `payload` is a registry-owned encoding (JSON) the store treats as bytes — it
    never inspects it, so new event fields/kinds don't change this contract.
    """

    event_id: str  # registry-assigned unique id (idempotency key for the store)
    kind: str      # "heartbeat" | "node_left" | "allocation" | "quality_observation" | …
    ts: str        # ISO-8601 UTC, from the event
    payload: str   # opaque registry-owned encoding


class EventStore(Protocol):
    """Durability seam. See module docstring for the full contract."""

    def append(self, env: EventEnvelope) -> None:
        """Durably persist `env` before returning (see contract: must not block
        on transient outage; raise only on an unrecoverable error)."""
        ...

    def replay(self) -> Iterable[EventEnvelope]:
        """Yield every persisted envelope in append order. Called once at boot."""
        ...


class NullEventStore:
    """Default store: NO durability. `append` is a no-op; `replay` is empty — so
    the registry is in-memory only, identical to the pre-seam behavior."""

    def append(self, env: EventEnvelope) -> None:  # noqa: D102 - no-op by design
        return None

    def replay(self) -> Iterable[EventEnvelope]:  # noqa: D102 - nothing persisted
        return ()


__all__ = ["EventEnvelope", "EventStore", "NullEventStore"]
