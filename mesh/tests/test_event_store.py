"""Tests for the registry durability seam (mesh.event_store + registry wiring).

Exercises the shared `_record` mechanism via the lightweight `record_node_left`
writer (the heartbeat/allocation/quality writers share the same path; the
existing e2e suite is the regression net for those).
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from mesh.event_store import EventEnvelope, NullEventStore
from mesh.registry import (
    AllocationEvent,
    MeshRegistry,
    NodeLeftEvent,
    QualityObservationEvent,
    _decode,
    _encode,
)


class FakeStore:
    """In-memory durable store stand-in: append accumulates, replay yields all."""

    def __init__(self) -> None:
        self.events: list[EventEnvelope] = []

    def append(self, env: EventEnvelope) -> None:
        self.events.append(env)

    def replay(self):
        return list(self.events)


class RaisingStore:
    """A store whose durable append fails (e.g. unrecoverable error)."""

    def append(self, env: EventEnvelope) -> None:
        raise RuntimeError("store down")

    def replay(self):
        return ()


_NOW = datetime(2026, 6, 3, tzinfo=timezone.utc)


# ───────────────────────────── default behavior ─────────────────────────────


def test_default_store_is_in_memory_only():
    """No store → NullEventStore → identical pre-seam behavior."""
    reg = MeshRegistry()
    assert isinstance(reg._store, NullEventStore)
    reg.record_node_left("node-1", "bye")
    assert [e.kind for e in reg._events] == ["node_left"]


def test_null_store_replay_is_empty():
    assert list(NullEventStore().replay()) == []


# ───────────────────────────── codec round-trip ─────────────────────────────


def test_codec_round_trips_each_event_kind():
    events = [
        NodeLeftEvent(ts=_NOW, node_id="n1", reason="x"),
        AllocationEvent(ts=_NOW, strategy="diversify", suggestions={"n1": "spec-a"}),
        QualityObservationEvent(
            ts=_NOW, specialist_id="spec-a", score=3.5, sample_count=10,
            observation_source="synthetic",
        ),
    ]
    for ev in events:
        env = _encode(ev)
        assert env.kind == ev.kind and env.event_id  # registry-assigned id present
        assert _decode(env) == ev  # opaque payload round-trips losslessly


def test_decode_rejects_unknown_kind():
    with pytest.raises(ValueError, match="unknown event kind"):
        _decode(EventEnvelope(event_id="x", kind="bogus", ts="2026", payload="{}"))


# ─────────────────── durable append + boot replay (the point) ────────────────


def test_durable_append_then_replay_rebuilds_state_across_restart():
    store = FakeStore()
    reg = MeshRegistry(store=store)
    reg.record_node_left("node-1", "a")
    reg.record_node_left("node-2", "b")
    assert len(store.events) == 2  # durably persisted

    # "Restart": a fresh registry over the same durable store replays at boot.
    reg2 = MeshRegistry(store=store)
    assert [e.kind for e in reg2._events] == ["node_left", "node_left"]
    assert [e.node_id for e in reg2._events] == ["node-1", "node-2"]  # order preserved


def test_replay_preserves_append_order():
    store = FakeStore()
    a = MeshRegistry(store=store)
    for i in range(5):
        a.record_node_left(f"n{i}", "x")
    b = MeshRegistry(store=store)
    assert [e.node_id for e in b._events] == [f"n{i}" for i in range(5)]


# ─────────────────── durable-first: no silent divergence ─────────────────────


def test_failed_durable_append_does_not_mutate_read_model():
    """Durable-FIRST: if the store raises, the in-memory read model is untouched
    (neither side has the event — no silent divergence; the caller sees it)."""
    reg = MeshRegistry(store=RaisingStore())
    with pytest.raises(RuntimeError, match="store down"):
        reg.record_node_left("node-1", "a")
    assert reg._events == []
