"""Mesh router — spec §6.

Extends slancha-api `select_model_lmarena()` with `(specialist, node_url)`
selection. v0.0.1 implementation is standalone (doesn't import from
slancha-api directly so this package can be used in isolation), but
mirrors the pareto-mode logic: rank by composite score, prefer
mesh-local before cloud fallback.

Inputs:  classifier signals (domain, difficulty, language, needs_tools)
Output:  MeshSelectionResult with chosen node_url + fallback chain.

If no mesh route matches: returns a MeshSelectionResult with
`node_id=None, cluster_coverage_used=False, reason="cloud fallback ..."`
so the caller knows to defer to slancha-cloud.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from mesh.models import (
    MeshSelectionResult,
    ModelId,
    NodeId,
    RegistrySnapshot,
    Route,
    SpecialistCard,
    SpecialistId,
)
from mesh.registry import build_ranked_routes

# Queue budget per spec §6 — drop routes with > this much queue time
# unless the caller explicitly relaxes it (e.g., batch class).
DEFAULT_MAX_QUEUE_MS = 2000

# p95 budget for hot-interactive routes (per spec §6 latency budget table).
HOT_INTERACTIVE_P95_MS = 1500


@dataclass(frozen=True)
class ClassifierSignals:
    """The signals the slancha-api classifier produces. Mirrors the
    fields `select_model_lmarena` consumes; we keep it self-contained
    so the mesh package doesn't depend on slancha-api importable."""

    domain: str
    difficulty: str  # "easy" | "medium" | "hard"
    language: str = "en"
    needs_tools: bool = False
    route_class: str = "standard"  # "hot_interactive" | "standard" | "batch"


def _route_score(route: Route, route_class: str) -> float:
    """Composite quality score. Lower queue + lower p95 = better.

    For hot-interactive, p95 dominates; for batch, we mostly care about
    queue depth. Tunable; current weights are a starting point and will
    be re-tuned once we have live mesh traffic in Langfuse.
    """
    queue_pen = route.estimated_queue_ms / 1000.0  # seconds
    p95 = route.p95_latency_ms or 500.0
    if route_class == "hot_interactive":
        return -(queue_pen * 2.0 + p95 / 100.0)
    if route_class == "batch":
        return -(queue_pen * 0.5 + p95 / 500.0)
    # standard
    return -(queue_pen + p95 / 250.0)


def _domain_for_signals(signals: ClassifierSignals) -> str:
    """Map classifier domain → catalog domain.

    The classifier emits LMArena-aligned categories ("math", "code",
    "computer science", ...); the catalog uses canonical short forms.
    We normalize a few common synonyms here so the mesh side doesn't
    need a perfect mirror of the classifier's taxonomy.
    """
    d = signals.domain.lower().strip()
    if d in {"code", "computer science", "engineering", "programming", "coding"}:
        return "code"
    if d in {"math", "physics", "chemistry", "mathematics"}:
        return "math"
    if d in {"multilingual"} or signals.language not in ("en", ""):
        return "multilingual"
    if d in {"reasoning", "analysis"}:
        return "reasoning"
    return "general"


def _filter_routes(
    routes: list[Route],
    route_class: str,
    max_queue_ms: int,
) -> list[Route]:
    """Apply per-spec §6 health + queue + p95 budget filters."""
    out: list[Route] = []
    for r in routes:
        if r.estimated_queue_ms > max_queue_ms:
            continue
        if route_class == "hot_interactive":
            if r.p95_latency_ms is not None and r.p95_latency_ms > HOT_INTERACTIVE_P95_MS:
                continue
        out.append(r)
    return out


def select_mesh_route(
    signals: ClassifierSignals,
    registry_snapshot: RegistrySnapshot,
    max_queue_ms: int = DEFAULT_MAX_QUEUE_MS,
    cloud_fallback_model: str = "claude-sonnet-4-7",
) -> MeshSelectionResult:
    """Pick (specialist, node) for this request, or fall through to cloud.

    Flow (spec §6):
      1. Map classifier signals → catalog domain.
      2. Lookup `(domain, difficulty)` candidates from snapshot.
      3. Filter unhealthy / over-queue / over-p95.
      4. Rank by `_route_score` (route-class aware).
      5. Top = primary; rest = fallback chain.
      6. None survive → cloud fallback.

    Snapshot's `ranked_routes` may already be populated; if empty we
    build on the fly so callers that pass raw snapshots still work.
    """
    domain = _domain_for_signals(signals)
    difficulty = signals.difficulty.lower().strip()
    key = f"{domain}|{difficulty}"

    ranked = registry_snapshot.ranked_routes
    if not ranked:
        ranked = build_ranked_routes(registry_snapshot)

    candidates = list(ranked.get(key, []))

    # Difficulty fall-through: if no routes at this tier, try harder tiers
    # (a specialist that handles 'hard' can usually handle 'medium').
    if not candidates and difficulty == "easy":
        candidates = list(ranked.get(f"{domain}|medium", []))
    if not candidates and difficulty in ("easy", "medium"):
        candidates = list(ranked.get(f"{domain}|hard", []))

    # Domain fall-through: try `general` if domain-specific had nothing.
    if not candidates and domain != "general":
        candidates = list(ranked.get(f"general|{difficulty}", []))

    filtered = _filter_routes(candidates, signals.route_class, max_queue_ms)
    if not filtered:
        return MeshSelectionResult(
            model=cloud_fallback_model,
            specialist_id=None,
            node_id=None,
            node_url=None,
            reason=(
                f"no mesh route for {domain}/{difficulty}/{signals.language} "
                f"(candidates={len(candidates)} filtered_out={len(candidates) - len(filtered)}); "
                f"falling through to cloud"
            ),
            queue_ms_estimated=0,
            cluster_coverage_used=False,
            fallback_chain=[(cloud_fallback_model, None)],
        )

    ranked_filtered = sorted(filtered, key=lambda r: -_route_score(r, signals.route_class))
    primary = ranked_filtered[0]
    fallback_chain: list[tuple[ModelId, NodeId | None]] = [
        (_model_id_for(r, registry_snapshot), r.node_id) for r in ranked_filtered[1:]
    ]
    # Always end with cloud
    fallback_chain.append((cloud_fallback_model, None))

    return MeshSelectionResult(
        model=_model_id_for(primary, registry_snapshot),
        specialist_id=primary.specialist_id,
        node_id=primary.node_id,
        node_url=primary.node_url,
        reason=(
            f"mesh: {primary.specialist_id} @ {primary.node_id} "
            f"queue={primary.estimated_queue_ms}ms "
            f"p95={primary.p95_latency_ms}ms route_class={signals.route_class}"
        ),
        queue_ms_estimated=primary.estimated_queue_ms,
        cluster_coverage_used=True,
        fallback_chain=fallback_chain,
    )


def _model_id_for(route: Route, snap: RegistrySnapshot) -> ModelId:
    """Resolve specialist_id → upstream model_id via catalog."""
    card = snap.catalog.get(route.specialist_id)
    return card.model_id if card else route.specialist_id


# ---------------------------------------------------------------------------
# Phase 5 — pref-aware selection (X-Slancha-Pref substrate)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PrefVector:
    """Per-request agent preferences. Mirrors slancha-api's SlanchaPref
    (kept inline here so the mesh package doesn't depend on the SaaS).

    All fields optional. Any field set acts as a hard ceiling
    (max_*, require_*) or soft weight (quality_weight).
    """

    max_cost_cents: int | None = None
    max_latency_ms_p95: int | None = None
    max_ttft_ms: int | None = None
    min_throughput_tps: int | None = None
    quality_weight: float | None = None  # 0.0 (fastest) .. 1.0 (highest quality)
    require_capabilities: list[str] | None = None
    require_streaming: bool | None = None
    allow_fallbacks: bool | None = None
    min_context_window: int | None = None


# NC4 default bootstrap discount — when quality.router_observed is null,
# trust node_self_reported × this fraction so new specialists earn traffic
# without being trusted at face value. Tunable; range 0.3-0.7.
DEFAULT_BOOTSTRAP_DISCOUNT = 0.5


def _effective_quality(card: SpecialistCard | None, bootstrap_discount: float) -> float:
    """Quality score the router actually uses to rank.

    Priority: router-observed > discounted node-self-reported > middle default.
    """
    if card is None:
        return 2.5  # mid-of-5 default if card missing
    if card.quality_router_observed is not None:
        return card.quality_router_observed
    if card.quality_node_self_reported is not None:
        return card.quality_node_self_reported * bootstrap_discount
    return 2.5  # default neutral score


def _clamp_pref_with_ceiling(
    pref: PrefVector,
    ceiling: dict[str, float | int],
) -> PrefVector:
    """Apply admin ceiling clamps to client-supplied pref values (H15)."""
    updates: dict[str, float | int] = {}
    # Cap shapes: max_X capped to ceiling[max_X].
    for key in ("max_cost_cents", "max_latency_ms_p95", "max_ttft_ms"):
        cap = ceiling.get(key)
        if cap is None:
            continue
        cur = getattr(pref, key)
        if cur is None or cur > cap:
            updates[key] = cap
    # Floor shapes: X floored to ceiling["X_min"].
    quality_floor = ceiling.get("quality_weight_min")
    if quality_floor is not None:
        cur_q = pref.quality_weight
        if cur_q is None or cur_q < quality_floor:
            updates["quality_weight"] = quality_floor
    if not updates:
        return pref
    new_kwargs = {**pref.__dict__, **updates}
    return PrefVector(**new_kwargs)


def _passes_capability_gate(card: SpecialistCard | None, required: list[str]) -> bool:
    if card is None:
        # Without a card we can't verify — fail safely (drop the route).
        return False
    have = set(card.capabilities or [])
    return all(c in have for c in required)


def _pref_filter(
    routes: list[Route],
    catalog: dict[SpecialistId, SpecialistCard],
    pref: PrefVector,
) -> tuple[list[Route], list[tuple[Route, str]]]:
    """Apply hard ceilings + capability gates. Returns (survivors, drops).

    drops: list of (route, reason) so we can populate decision_reason
    with WHY a route fell out — useful for the transparency UI.
    """
    survivors: list[Route] = []
    drops: list[tuple[Route, str]] = []
    for r in routes:
        if pref.max_latency_ms_p95 is not None and r.p95_latency_ms is not None:
            if r.p95_latency_ms > pref.max_latency_ms_p95:
                drops.append((r, f"p95 {r.p95_latency_ms:.0f}ms > cap {pref.max_latency_ms_p95}ms"))
                continue
        if pref.max_cost_cents is not None and r.cost_estimate_cents > pref.max_cost_cents:
            drops.append((r, f"cost {r.cost_estimate_cents}c > cap {pref.max_cost_cents}c"))
            continue
        if pref.require_capabilities:
            card = catalog.get(r.specialist_id)
            if not _passes_capability_gate(card, pref.require_capabilities):
                drops.append((r, f"missing capability {pref.require_capabilities}"))
                continue
        if pref.min_context_window is not None:
            card = catalog.get(r.specialist_id)
            if card is None or card.context_window < pref.min_context_window:
                drops.append((r, f"context_window < {pref.min_context_window}"))
                continue
        survivors.append(r)
    return survivors, drops


def _pref_score(
    route: Route,
    card: SpecialistCard | None,
    pref: PrefVector,
    bootstrap_discount: float,
) -> tuple[float, set[str]]:
    """Pref-aware composite score. Returns (score, axes_used).

    Axes: quality, latency, cost. quality_weight in [0,1] mixes
    quality-dominant vs latency-dominant. When all routes are mesh-local
    (cost=0), cost axis collapses but the function still names it in
    axes_used for transparency consistency.
    """
    quality = _effective_quality(card, bootstrap_discount)
    qw = pref.quality_weight if pref.quality_weight is not None else 0.5

    # Normalize latency to a [0,1]-ish penalty: 200ms → 0, 2000ms → 1.
    p95 = route.p95_latency_ms if route.p95_latency_ms is not None else 800.0
    latency_pen = min(max((p95 - 200.0) / 1800.0, 0.0), 1.5)

    cost_pen = route.cost_estimate_cents / 10.0  # 10 cents = 1.0 penalty

    # Composite: high quality + low penalties. quality is on a ~5-point
    # scale (LMArena-aligned); scale to [0,1] by /5 before mixing.
    quality_norm = min(quality / 5.0, 1.0)

    score = (
        qw * quality_norm
        + (1.0 - qw) * (1.0 - 0.6 * latency_pen - 0.4 * cost_pen)
    )

    axes = {"quality", "latency"}
    if route.cost_estimate_cents > 0:
        axes.add("cost")
    return score, axes


def select_mesh_route_with_pref(
    signals: ClassifierSignals,
    registry_snapshot: RegistrySnapshot,
    *,
    pref: PrefVector | None = None,
    max_queue_ms: int = DEFAULT_MAX_QUEUE_MS,
    cloud_fallback_model: str = "claude-sonnet-4-7",
    admin_ceiling: dict[str, float | int] | None = None,
    bootstrap_discount: float = DEFAULT_BOOTSTRAP_DISCOUNT,
    preset_applied: str | None = None,
) -> MeshSelectionResult:
    """Pareto-aware mesh selector with structured decision reason.

    Behavior without pref: identical to select_mesh_route (backwards
    compatible).

    Behavior with pref:
      1. Clamp client pref against admin ceiling (H15).
      2. Catalog domain + difficulty fall-through (same as select_mesh_route).
      3. Apply base health/queue/route_class filter.
      4. Apply pref hard ceilings + capability gates.
      5. Score by composite (quality_weight × quality + (1-qw) × latency/cost).
      6. Emit decision_reason_structured per NC5.
    """
    if pref is None:
        # Backwards-compatible — same call shape as select_mesh_route.
        return select_mesh_route(
            signals, registry_snapshot, max_queue_ms, cloud_fallback_model
        )

    if admin_ceiling:
        pref = _clamp_pref_with_ceiling(pref, admin_ceiling)

    domain = _domain_for_signals(signals)
    difficulty = signals.difficulty.lower().strip()
    key = f"{domain}|{difficulty}"

    ranked = registry_snapshot.ranked_routes
    if not ranked:
        ranked = build_ranked_routes(registry_snapshot)

    candidates = list(ranked.get(key, []))
    # Same difficulty + domain fall-through as the base selector.
    if not candidates and difficulty == "easy":
        candidates = list(ranked.get(f"{domain}|medium", []))
    if not candidates and difficulty in ("easy", "medium"):
        candidates = list(ranked.get(f"{domain}|hard", []))
    if not candidates and domain != "general":
        candidates = list(ranked.get(f"general|{difficulty}", []))

    # Base route-class + queue filter.
    base_filtered = _filter_routes(candidates, signals.route_class, max_queue_ms)

    # Pref filter (capabilities + ceilings).
    catalog = registry_snapshot.catalog
    pref_filtered, drops = _pref_filter(base_filtered, catalog, pref)

    if not pref_filtered:
        # No route survived. Falling through to cloud.
        return MeshSelectionResult(
            model=cloud_fallback_model,
            specialist_id=None,
            node_id=None,
            node_url=None,
            reason=(
                f"no mesh route satisfies pref for {domain}/{difficulty}; "
                f"base_candidates={len(candidates)} dropped_by_pref={len(drops)}; "
                f"falling through to cloud"
            ),
            queue_ms_estimated=0,
            cluster_coverage_used=False,
            fallback_chain=[(cloud_fallback_model, None)],
            decision_reason_structured={
                "winner": cloud_fallback_model,
                "alternatives_considered": [
                    {
                        "id": r.specialist_id,
                        "delta": 0.0,
                        "losing_axes": [reason],
                    }
                    for r, reason in drops
                ],
                "deciding_axes": ["pref_gate"],
                "preset_applied": preset_applied,
            },
        )

    # Score remaining routes.
    scored: list[tuple[Route, float, set[str]]] = []
    for r in pref_filtered:
        score, axes = _pref_score(r, catalog.get(r.specialist_id), pref, bootstrap_discount)
        scored.append((r, score, axes))
    scored.sort(key=lambda t: -t[1])

    winner_route, winner_score, deciding_axes = scored[0]
    losers = scored[1:]

    fallback_chain: list[tuple[ModelId, NodeId | None]] = [
        (_model_id_for(r, registry_snapshot), r.node_id) for r, _, _ in losers
    ]
    fallback_chain.append((cloud_fallback_model, None))

    structured = {
        "winner": winner_route.specialist_id,
        "alternatives_considered": [
            {
                "id": r.specialist_id,
                "delta": round(winner_score - s, 4),
                "losing_axes": sorted(deciding_axes - ax),
            }
            for r, s, ax in losers
        ]
        + [
            {
                "id": r.specialist_id,
                "delta": 0.0,
                "losing_axes": [reason],
            }
            for r, reason in drops
        ],
        "deciding_axes": sorted(deciding_axes),
        "preset_applied": preset_applied,
    }

    return MeshSelectionResult(
        model=_model_id_for(winner_route, registry_snapshot),
        specialist_id=winner_route.specialist_id,
        node_id=winner_route.node_id,
        node_url=winner_route.node_url,
        reason=(
            f"mesh: {winner_route.specialist_id} @ {winner_route.node_id} "
            f"score={winner_score:.3f} qw={pref.quality_weight} "
            f"(filtered {len(drops)} drops, considered {len(scored)})"
        ),
        queue_ms_estimated=winner_route.estimated_queue_ms,
        cluster_coverage_used=True,
        fallback_chain=fallback_chain,
        decision_reason_structured=structured,
    )


__all__ = [
    "ClassifierSignals",
    "DEFAULT_BOOTSTRAP_DISCOUNT",
    "DEFAULT_MAX_QUEUE_MS",
    "HOT_INTERACTIVE_P95_MS",
    "PrefVector",
    "select_mesh_route",
    "select_mesh_route_with_pref",
]


# Silence unused-import lint of math (kept for parity with slancha-api
# _pareto_score which uses log2; if we add pareto-mode cost weighting
# in v0.0.2 we'll need it).
_ = math
