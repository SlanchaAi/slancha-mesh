"""Tests for select_mesh_route_with_pref — Phase 5 substrate.

Hard ceilings filter routes (max_latency_ms_p95, require_capabilities);
soft preferences (quality_weight) shape the rank; decision_reason_structured
gets emitted so the routing-transparency UI has a stable shape.

Admin ceiling clamps client-supplied pref values BEFORE filtering so an
adversarial agent can't escalate beyond what the operator pre-approved.
"""

from __future__ import annotations

from datetime import datetime, timezone

from mesh.models import (
    HealthState,
    MeshSelectionResult,
    NodeBinding,
    NodeSummary,
    RegistrySnapshot,
    Route,
    SpecialistCard,
)
from mesh.select import (
    ClassifierSignals,
    PrefVector,
    select_mesh_route_with_pref,
)


def _make_snapshot(
    routes_by_key: dict[str, list[Route]],
    catalog_by_id: dict[str, SpecialistCard],
    nodes_by_id: dict[str, NodeSummary] | None = None,
) -> RegistrySnapshot:
    ts = datetime(2026, 5, 22, 12, 0, 0, tzinfo=timezone.utc)
    nodes = nodes_by_id or {
        r.node_id: NodeSummary(
            node_id=r.node_id,
            friendly_name=r.node_id,
            health="healthy",
            last_seen=ts,
            loaded_specialist_ids=[r.specialist_id for r in routes_by_key.get(k, [])],
            queue_depth=0,
            p95_latency_ms_60s=r.p95_latency_ms,
            node_url=r.node_url,
        )
        for k in routes_by_key
        for r in routes_by_key[k]
    }
    return RegistrySnapshot(
        snapshot_ts=ts,
        nodes=nodes,
        specialists={},
        coverage={},
        ranked_routes=routes_by_key,
        catalog=catalog_by_id,
    )


def _card(
    specialist_id: str,
    capabilities: list[str] | None = None,
    quality_router_observed: float | None = None,
    quality_node_self_reported: float | None = None,
) -> SpecialistCard:
    return SpecialistCard(
        model_id=specialist_id,
        specialist_id=specialist_id,
        domain="general",
        difficulty_tiers=["easy", "medium", "hard"],
        languages=["en"],
        required_backend="vllm",
        storage_gb=10.0,
        runtime_gb=12.0,
        min_vram_gb=8.0,
        context_window=8192,
        n_layers=32,
        capabilities=capabilities or [],
        quality_router_observed=quality_router_observed,
        quality_node_self_reported=quality_node_self_reported,
    )


def _route(
    specialist_id: str,
    node_id: str = "spark-1",
    p95_ms: float = 600.0,
    queue_ms: int = 0,
    cost_cents: float = 0.0,
) -> Route:
    return Route(
        specialist_id=specialist_id,
        node_id=node_id,
        node_url=f"http://{node_id}:8000/v1",
        estimated_queue_ms=queue_ms,
        p95_latency_ms=p95_ms,
        cost_estimate_cents=cost_cents,
    )


def _signals(domain: str = "general", difficulty: str = "medium") -> ClassifierSignals:
    return ClassifierSignals(domain=domain, difficulty=difficulty)


# ── Hard ceilings ───────────────────────────────────────────────────────────


def test_max_latency_filter_drops_slow_routes():
    """Routes with p95 > max_latency_ms_p95 are filtered out."""
    snap = _make_snapshot(
        {"general|medium": [_route("fast", p95_ms=300.0), _route("slow", node_id="spark-2", p95_ms=3000.0)]},
        {"fast": _card("fast"), "slow": _card("slow")},
    )
    pref = PrefVector(max_latency_ms_p95=1000)
    result = select_mesh_route_with_pref(_signals(), snap, pref=pref)
    assert result.specialist_id == "fast"


def test_require_capabilities_filter_drops_lacking_routes():
    """Routes whose specialist lacks required capability are dropped."""
    snap = _make_snapshot(
        {"general|medium": [_route("no-tools"), _route("with-tools", node_id="spark-2")]},
        {
            "no-tools": _card("no-tools", capabilities=["streaming"]),
            "with-tools": _card("with-tools", capabilities=["tools", "streaming"]),
        },
    )
    pref = PrefVector(require_capabilities=["tools"])
    result = select_mesh_route_with_pref(_signals(), snap, pref=pref)
    assert result.specialist_id == "with-tools"


def test_no_routes_after_filter_falls_through_to_cloud():
    snap = _make_snapshot(
        {"general|medium": [_route("slow", p95_ms=3000.0)]},
        {"slow": _card("slow")},
    )
    pref = PrefVector(max_latency_ms_p95=500)
    result = select_mesh_route_with_pref(_signals(), snap, pref=pref)
    assert result.specialist_id is None  # cloud fallback
    assert result.cluster_coverage_used is False


# ── Quality-weight scoring ──────────────────────────────────────────────────


def test_high_quality_weight_picks_higher_quality_specialist():
    """quality_weight=0.9 → higher router_observed quality wins over faster."""
    snap = _make_snapshot(
        {"general|medium": [
            _route("fast-low-q", p95_ms=200.0),
            _route("slower-hi-q", node_id="spark-2", p95_ms=800.0),
        ]},
        {
            "fast-low-q": _card("fast-low-q", quality_router_observed=2.0),
            "slower-hi-q": _card("slower-hi-q", quality_router_observed=4.5),
        },
    )
    pref = PrefVector(quality_weight=0.9)
    result = select_mesh_route_with_pref(_signals(), snap, pref=pref)
    assert result.specialist_id == "slower-hi-q"


def test_low_quality_weight_picks_faster_specialist():
    """quality_weight=0.1 → latency dominates."""
    snap = _make_snapshot(
        {"general|medium": [
            _route("fast-low-q", p95_ms=200.0),
            _route("slower-hi-q", node_id="spark-2", p95_ms=800.0),
        ]},
        {
            "fast-low-q": _card("fast-low-q", quality_router_observed=2.0),
            "slower-hi-q": _card("slower-hi-q", quality_router_observed=4.5),
        },
    )
    pref = PrefVector(quality_weight=0.1)
    result = select_mesh_route_with_pref(_signals(), snap, pref=pref)
    assert result.specialist_id == "fast-low-q"


def test_cold_start_uses_self_reported_with_discount():
    """quality_router_observed=None → bootstrap_discount × node_self_reported.

    Per NC4 tunable, default 0.5. Allows new specialists to earn traffic
    instead of starving at quality=null forever.
    """
    snap = _make_snapshot(
        {"general|medium": [
            _route("bootstrap"),
            _route("established", node_id="spark-2"),
        ]},
        {
            # bootstrap: self-reports 4.0, observed=null → effective = 4.0 × 0.5 = 2.0
            "bootstrap": _card("bootstrap", quality_node_self_reported=4.0, quality_router_observed=None),
            # established: observed 3.0 (real)
            "established": _card("established", quality_router_observed=3.0),
        },
    )
    pref = PrefVector(quality_weight=0.95)  # quality dominates
    result = select_mesh_route_with_pref(_signals(), snap, pref=pref)
    # established at 3.0 > bootstrap effective 2.0
    assert result.specialist_id == "established"


def test_cold_start_with_high_bootstrap_discount():
    """Tunable bootstrap_discount=0.8 → bootstrap effective = 4.0 × 0.8 = 3.2,
    > established 3.0."""
    snap = _make_snapshot(
        {"general|medium": [
            _route("bootstrap"),
            _route("established", node_id="spark-2"),
        ]},
        {
            "bootstrap": _card("bootstrap", quality_node_self_reported=4.0),
            "established": _card("established", quality_router_observed=3.0),
        },
    )
    pref = PrefVector(quality_weight=0.95)
    result = select_mesh_route_with_pref(
        _signals(), snap, pref=pref, bootstrap_discount=0.8
    )
    assert result.specialist_id == "bootstrap"


# ── decision_reason_structured shape ────────────────────────────────────────


def test_decision_reason_structured_has_required_keys():
    """Conformance corpus shape per NC5."""
    snap = _make_snapshot(
        {"general|medium": [
            _route("winner"),
            _route("runner-up", node_id="spark-2"),
        ]},
        {
            "winner": _card("winner", quality_router_observed=4.5),
            "runner-up": _card("runner-up", quality_router_observed=3.0),
        },
    )
    pref = PrefVector(quality_weight=0.7)
    result = select_mesh_route_with_pref(_signals(), snap, pref=pref)

    drs = result.decision_reason_structured
    assert drs is not None
    assert "winner" in drs
    assert "alternatives_considered" in drs
    assert "deciding_axes" in drs
    assert "preset_applied" in drs
    assert drs["winner"] == "winner"
    assert len(drs["alternatives_considered"]) == 1
    assert drs["alternatives_considered"][0]["id"] == "runner-up"
    assert "delta" in drs["alternatives_considered"][0]
    assert "losing_axes" in drs["alternatives_considered"][0]


def test_losing_axes_is_the_axis_delta_not_the_full_set():
    """`losing_axes` must report the axes that distinguish the winner from
    this alternative — not every axis considered.

    Regression-lock for a set-precedence bug: `deciding_axes - ax | ax`
    parses as `(deciding_axes - ax) | ax`, which always equals
    `deciding_axes | ax` (the full union), so every alternative reported
    *all* axes regardless of where it actually differed.

    Here the winner carries a `cost` axis (cost_estimate_cents > 0) that the
    cheaper mesh-local runner-up lacks; quality dominates the rank so the
    pricier route still wins. The only axis distinguishing them is `cost`,
    so `losing_axes` must be exactly `["cost"]` — both `deciding - ax` and
    the symmetric difference agree on that. The pre-fix bug emitted all
    three axes (`["cost", "latency", "quality"]`).
    """
    snap = _make_snapshot(
        {"general|medium": [
            _route("winner", cost_cents=5.0),
            _route("runner-up", node_id="spark-2", cost_cents=0.0),
        ]},
        {
            "winner": _card("winner", quality_router_observed=4.8),
            "runner-up": _card("runner-up", quality_router_observed=3.0),
        },
    )
    pref = PrefVector(quality_weight=0.9)
    result = select_mesh_route_with_pref(_signals(), snap, pref=pref)

    drs = result.decision_reason_structured
    assert drs is not None
    assert drs["winner"] == "winner"
    alt = drs["alternatives_considered"][0]
    assert alt["id"] == "runner-up"
    assert alt["losing_axes"] == ["cost"]


def test_decision_reason_emits_preset_when_service_tier_set():
    snap = _make_snapshot(
        {"general|medium": [_route("only")]},
        {"only": _card("only")},
    )
    pref = PrefVector(quality_weight=0.5)
    result = select_mesh_route_with_pref(
        _signals(), snap, pref=pref, preset_applied="balanced"
    )
    assert result.decision_reason_structured["preset_applied"] == "balanced"


def test_no_pref_yields_backwards_compatible_behavior():
    """Calling without a pref returns same result as plain select_mesh_route
    (decision_reason_structured may be None or minimal)."""
    snap = _make_snapshot(
        {"general|medium": [_route("only")]},
        {"only": _card("only")},
    )
    result = select_mesh_route_with_pref(_signals(), snap, pref=None)
    assert result.specialist_id == "only"


# ── Admin ceiling (defense against H15) ─────────────────────────────────────


def test_admin_ceiling_clamps_max_latency():
    """Client asks for max_latency_ms_p95=5000 but admin caps at 800.

    Adversarial agent that tried to use mesh for a slow expensive query
    gets clamped to the operator's allowed cap. Routes with p95 in
    (800, 5000] no longer pass.
    """
    snap = _make_snapshot(
        {"general|medium": [_route("ok", p95_ms=600.0), _route("over-cap", node_id="spark-2", p95_ms=1200.0)]},
        {"ok": _card("ok"), "over-cap": _card("over-cap")},
    )
    pref = PrefVector(max_latency_ms_p95=5000)  # client tries to relax
    result = select_mesh_route_with_pref(
        _signals(), snap, pref=pref, admin_ceiling={"max_latency_ms_p95": 800}
    )
    assert result.specialist_id == "ok"


def test_admin_ceiling_quality_floor():
    """Admin enforces minimum quality regardless of client request."""
    snap = _make_snapshot(
        {"general|medium": [_route("low-q"), _route("high-q", node_id="spark-2")]},
        {
            "low-q": _card("low-q", quality_router_observed=2.0),
            "high-q": _card("high-q", quality_router_observed=4.0),
        },
    )
    # Client wants cheap (low quality_weight)
    pref = PrefVector(quality_weight=0.1)
    result = select_mesh_route_with_pref(
        _signals(),
        snap,
        pref=pref,
        admin_ceiling={"quality_weight_min": 0.9},  # force high quality
    )
    assert result.specialist_id == "high-q"
