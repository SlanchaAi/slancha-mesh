"""Tests for Phase 6 — probe runner, drift detection, /quality_observation."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from mesh.models import SpecialistCard
from mesh.quality_probe import (
    DEFAULT_DRIFT_THRESHOLD,
    DEFAULT_PROBE_SET,
    DriftEvent,
    ProbeRunner,
    QualityObservation,
    StubScorer,
    detect_drift,
)
from mesh.registry import MeshRegistry, QualityObservationEvent
from mesh.service import create_mesh_app


# ── Drift detection ─────────────────────────────────────────────────────────


def test_drift_no_prior_returns_none():
    """First observation cannot drift — drift is longitudinal."""
    assert detect_drift(prior=None, current=4.0) is None


def test_drift_within_threshold_returns_none():
    """|delta| <= threshold → no drift event."""
    out = detect_drift(prior=4.0, current=4.3, threshold=0.5)
    assert out is None


def test_drift_above_threshold_returns_event_down():
    out = detect_drift(prior=4.5, current=3.0, threshold=0.5, specialist_id="x")
    assert isinstance(out, DriftEvent)
    assert out.direction == "down"
    assert out.delta == pytest.approx(-1.5)
    assert out.specialist_id == "x"


def test_drift_above_threshold_returns_event_up():
    out = detect_drift(prior=2.0, current=3.5, threshold=0.5)
    assert isinstance(out, DriftEvent)
    assert out.direction == "up"
    assert out.delta == pytest.approx(1.5)


def test_drift_default_threshold_is_half_point():
    assert DEFAULT_DRIFT_THRESHOLD == 0.5


# ── StubScorer ──────────────────────────────────────────────────────────────


def test_stub_scorer_empty_response_zero():
    assert StubScorer().score(DEFAULT_PROBE_SET[0], "") == 0.0


def test_stub_scorer_clamps_to_five():
    """log(N+1) for very long responses still clamps at 5.0."""
    huge = "x" * 100_000
    assert StubScorer().score(DEFAULT_PROBE_SET[0], huge) == 5.0


def test_stub_scorer_monotone_on_length():
    short = StubScorer().score(DEFAULT_PROBE_SET[0], "hello world")
    long = StubScorer().score(DEFAULT_PROBE_SET[0], "hello world" * 20)
    assert long >= short


# ── ProbeRunner — uses an HTTP stub ─────────────────────────────────────────


def _patched_send_chat(monkeypatch, *, response_text: str):
    """Patch ProbeRunner._send_chat to return a canned response."""
    monkeypatch.setattr(ProbeRunner, "_send_chat", lambda self, **kwargs: response_text)


def test_probe_runner_returns_observation_with_expected_count(monkeypatch):
    """Number of scores collected matches the number of applicable probes."""
    _patched_send_chat(monkeypatch, response_text="ok ok")

    runner = ProbeRunner()
    obs = runner.probe_one(
        specialist_id="paul-voice-v8",
        model_id="paul-voice-v8",
        node_url="http://node:8000/v1",
        domain="writing",
    )

    # The default probe set has a writing entry + a general entry that
    # also matches (writing or general); 2 applicable.
    assert obs.specialist_id == "paul-voice-v8"
    assert obs.sample_count == 2
    assert obs.score > 0.0
    assert obs.observation_source == "synthetic"


def test_probe_runner_empty_response_yields_zero_score(monkeypatch):
    _patched_send_chat(monkeypatch, response_text="")
    runner = ProbeRunner()
    obs = runner.probe_one(
        specialist_id="paul-voice-v8",
        model_id="paul-voice-v8",
        node_url="http://node:8000/v1",
        domain="writing",
    )
    assert obs.score == 0.0


def test_probe_runner_unknown_domain_uses_general_fallback(monkeypatch):
    """For a domain with no matching probes, the 'general' bucket
    always applies (acts as universal-fallback). DEFAULT_PROBE_SET has
    one general probe.
    """
    _patched_send_chat(monkeypatch, response_text="some words")
    runner = ProbeRunner()
    obs = runner.probe_one(
        specialist_id="x",
        model_id="x",
        node_url="http://node:8000/v1",
        domain="extremely-niche-domain-not-in-probe-set",
    )
    assert obs.sample_count == 1


def test_probe_runner_truly_no_applicable_probes_falls_back_to_all(monkeypatch):
    """If we strip 'general' entirely AND the domain doesn't match
    anything, last-resort probes every entry (so a misconfigured
    probe-set never blanks the round).
    """
    _patched_send_chat(monkeypatch, response_text="some words")
    no_general = [p for p in DEFAULT_PROBE_SET if p.domain != "general"]
    runner = ProbeRunner(probe_set=no_general)
    obs = runner.probe_one(
        specialist_id="x",
        model_id="x",
        node_url="http://node:8000/v1",
        domain="extremely-niche-domain-not-in-probe-set",
    )
    assert obs.sample_count == len(no_general)


# ── MeshRegistry.record_quality_observation ─────────────────────────────────


def _card(specialist_id: str, **overrides) -> SpecialistCard:
    base = dict(
        model_id=specialist_id,
        specialist_id=specialist_id,
        domain="general",
        difficulty_tiers=["medium"],
        languages=["en"],
        required_backend="vllm",
        storage_gb=10.0,
        runtime_gb=12.0,
        min_vram_gb=8.0,
        context_window=8192,
        n_layers=32,
    )
    base.update(overrides)
    return SpecialistCard(**base)


def test_record_quality_observation_appends_event_and_updates_card():
    reg = MeshRegistry(catalog=[_card("spec-a")])

    prior, ev = reg.record_quality_observation(
        specialist_id="spec-a",
        score=4.0,
        sample_count=10,
        observation_source="synthetic",
    )
    assert prior is None  # first observation
    assert isinstance(ev, QualityObservationEvent)
    assert ev.score == 4.0
    assert ev.sample_count == 10
    snap = reg.snapshot()
    assert snap.catalog["spec-a"].quality_router_observed == 4.0
    assert snap.catalog["spec-a"].quality_sample_count == 10
    assert snap.catalog["spec-a"].quality_observation_source == "synthetic"


def test_second_observation_returns_prior_score():
    reg = MeshRegistry(catalog=[_card("spec-a")])
    reg.record_quality_observation(
        specialist_id="spec-a",
        score=4.0,
        sample_count=5,
        observation_source="synthetic",
    )
    prior, _ = reg.record_quality_observation(
        specialist_id="spec-a",
        score=3.5,
        sample_count=5,
        observation_source="synthetic",
    )
    assert prior == 4.0


def test_sample_count_accumulates_across_observations():
    reg = MeshRegistry(catalog=[_card("spec-a")])
    reg.record_quality_observation(
        specialist_id="spec-a", score=4.0, sample_count=10, observation_source="synthetic"
    )
    reg.record_quality_observation(
        specialist_id="spec-a", score=4.2, sample_count=15, observation_source="synthetic"
    )
    snap = reg.snapshot()
    assert snap.catalog["spec-a"].quality_sample_count == 25  # 10 + 15


# ── /quality_observation endpoint ───────────────────────────────────────────


def _make_client(monkeypatch, registry: MeshRegistry):
    monkeypatch.setenv("SLANCHA_NODE_TOKEN", "test-token")
    app = create_mesh_app(registry=registry)
    return TestClient(app)


def test_endpoint_writes_observation_and_returns_no_drift_on_first(monkeypatch):
    reg = MeshRegistry(catalog=[_card("spec-a")])
    client = _make_client(monkeypatch, reg)

    resp = client.post(
        "/quality_observation",
        json={
            "specialist_id": "spec-a",
            "score": 4.0,
            "sample_count": 10,
            "observation_source": "synthetic",
        },
        headers={"Authorization": "Bearer test-token"},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["ack"] is True
    assert body["prior_score"] is None
    assert body["drift"] is None


def test_endpoint_returns_drift_event_when_above_threshold(monkeypatch):
    reg = MeshRegistry(catalog=[_card("spec-a")])
    client = _make_client(monkeypatch, reg)

    headers = {"Authorization": "Bearer test-token"}
    client.post(
        "/quality_observation",
        json={
            "specialist_id": "spec-a",
            "score": 4.5,
            "sample_count": 10,
            "observation_source": "synthetic",
        },
        headers=headers,
    )
    resp = client.post(
        "/quality_observation",
        json={
            "specialist_id": "spec-a",
            "score": 2.5,  # 2.0 drop, > 0.5 threshold
            "sample_count": 10,
            "observation_source": "synthetic",
        },
        headers=headers,
    )
    body = resp.json()
    assert body["prior_score"] == 4.5
    assert body["drift"] is not None
    assert body["drift"]["direction"] == "down"
    assert body["drift"]["delta"] == pytest.approx(-2.0)


def test_endpoint_rejects_no_auth(monkeypatch):
    reg = MeshRegistry(catalog=[_card("spec-a")])
    client = _make_client(monkeypatch, reg)
    resp = client.post(
        "/quality_observation",
        json={
            "specialist_id": "spec-a",
            "score": 4.0,
            "sample_count": 1,
            "observation_source": "synthetic",
        },
    )
    assert resp.status_code in (401, 403)


def test_endpoint_rejects_invalid_score(monkeypatch):
    reg = MeshRegistry(catalog=[_card("spec-a")])
    client = _make_client(monkeypatch, reg)
    resp = client.post(
        "/quality_observation",
        json={
            "specialist_id": "spec-a",
            "score": 9.9,  # > 5.0 cap
            "sample_count": 1,
            "observation_source": "synthetic",
        },
        headers={"Authorization": "Bearer test-token"},
    )
    assert resp.status_code == 422


def test_endpoint_rejects_invalid_observation_source(monkeypatch):
    reg = MeshRegistry(catalog=[_card("spec-a")])
    client = _make_client(monkeypatch, reg)
    resp = client.post(
        "/quality_observation",
        json={
            "specialist_id": "spec-a",
            "score": 4.0,
            "sample_count": 1,
            "observation_source": "human_rated",  # not in enum
        },
        headers={"Authorization": "Bearer test-token"},
    )
    assert resp.status_code == 422
