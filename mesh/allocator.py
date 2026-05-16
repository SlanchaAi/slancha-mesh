"""Specialist allocator — spec §3.2 + §4.

Two pure functions:

- `model_fit_score(spec, node, coverage)` — scalar fitness of (spec, node).
- `allocate_cluster(nodes, catalog, traffic_mix, strategy)` — picks a
  primary specialist per node + ranked alternates.

Pure: no I/O. The registry calls these with snapshot data.
"""

from __future__ import annotations

import math
from typing import Literal

from mesh.models import (
    DomainId,
    NodeId,
    NodeProbe,
    NodeSuggestion,
    SpecialistCard,
)

Strategy = Literal["best_per_machine", "full_set", "tiered"]

# Coverage tier groupings per spec §4 Strategy C. Domains are matched
# case-insensitively against `SpecialistCard.domain`.
TIER_1_DOMAINS: frozenset[str] = frozenset({"math", "code", "general", "reasoning"})
TIER_2_DOMAINS: frozenset[str] = frozenset({"multilingual", "tool_use", "summarization"})
TIER_3_DOMAINS: frozenset[str] = frozenset({"vision", "embeddings", "image_gen", "whisper"})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _effective_vram_gb(node: NodeProbe) -> float:
    """Effective memory budget for model fit.

    Unified-memory nodes (Apple Silicon, GB10): use available RAM minus a
    fixed 8GB OS reserve. Discrete-VRAM nodes: use vram_available_gb.
    Falls back to RAM if VRAM probe failed.
    """
    if node.unified_memory:
        return max(0.0, node.ram_available_gb - 8.0)
    if node.vram_available_gb is not None:
        return node.vram_available_gb
    return max(0.0, node.ram_available_gb - 4.0)


def _estimated_tps(spec: SpecialistCard, node: NodeProbe) -> float:
    """Decode tok/s estimate.

    Primary: `node.memory_bandwidth_gbs / spec.runtime_gb` (per spec §3.1).
    Fallback: `spec.estimated_tps_at[<chip_family>]` for chips that don't
    expose memory bandwidth (GB10 today).
    """
    if node.memory_bandwidth_gbs:
        return max(1.0, node.memory_bandwidth_gbs / max(spec.runtime_gb, 0.1))
    # Map chip → catalog key
    chip = node.chip.lower()
    if "gb10" in chip:
        key = "gb10"
    elif "m4 pro" in chip or "m4pro" in chip:
        key = "m4_pro"
    elif "m3 ultra" in chip or "m3ultra" in chip:
        key = "m3_ultra"
    elif "l40" in chip:
        key = "l40"
    else:
        key = ""
    return float(spec.estimated_tps_at.get(key, 20.0))


# ---------------------------------------------------------------------------
# model_fit_score — spec §3.2
# ---------------------------------------------------------------------------


def model_fit_score(
    spec: SpecialistCard,
    node: NodeProbe,
    cluster_coverage: dict[DomainId, set[NodeId]],
) -> float:
    """Per spec §3.2.

    Hard filters return -inf:
      - required_backend not on node
      - storage_gb > disk_free_gb
      - min_vram_gb > effective memory
      - requires_fp4 but node has no Blackwell-class CUDA capability

    Soft scores combine coverage need (heaviest weight), throughput,
    headroom, and network position.
    """
    # --- Hard filters ---
    if spec.required_backend not in node.available_backends:
        return -math.inf
    if spec.storage_gb > node.disk_free_gb:
        return -math.inf

    effective_mem = _effective_vram_gb(node)
    if spec.min_vram_gb > effective_mem:
        return -math.inf

    if spec.requires_fp4 and node.cuda_capability not in {"10.0", "12.0", "12.1"}:
        # Blackwell consumer reports 12.0; GB10 reports 12.1 in driver 580+.
        return -math.inf

    # --- Soft scores ---
    headroom_gb = effective_mem - spec.runtime_gb
    headroom_score = min(headroom_gb / max(spec.runtime_gb, 0.1), 2.0)

    est_tok_per_s = _estimated_tps(spec, node)
    throughput_score = math.log(max(est_tok_per_s, 1.0)) / math.log(50.0)

    nodes_already_hosting = len(cluster_coverage.get(spec.domain, set()))
    coverage_score = 3.0 if nodes_already_hosting == 0 else 1.0 / (1 + nodes_already_hosting)

    rtt = node.rtt_to_master_ms if node.rtt_to_master_ms is not None else 0.0
    network_score = max(0.5, 1.0 - rtt / 100.0)

    return (
        2.0 * coverage_score
        + 1.5 * throughput_score
        + 0.5 * headroom_score
        + 0.3 * network_score
    )


# ---------------------------------------------------------------------------
# allocate_cluster — spec §4
# ---------------------------------------------------------------------------


def _domain_tier(domain: DomainId) -> int:
    """Map a specialist's domain to its coverage tier (1..3)."""
    if domain in TIER_1_DOMAINS:
        return 1
    if domain in TIER_2_DOMAINS:
        return 2
    if domain in TIER_3_DOMAINS:
        return 3
    return 2  # unknown domains default to tier 2 (don't crowd out essentials)


def _rank_specs_for_node(
    node: NodeProbe,
    catalog: list[SpecialistCard],
    coverage: dict[DomainId, set[NodeId]],
) -> list[tuple[float, SpecialistCard]]:
    """Score every catalog entry against a node; sort high-to-low.

    Entries returning -inf (hard-filter fail) are kept and sorted to the
    end so callers can show 'fits but no GPU' diagnostics.
    """
    scored: list[tuple[float, SpecialistCard]] = []
    for spec in catalog:
        scored.append((model_fit_score(spec, node, coverage), spec))
    scored.sort(key=lambda t: t[0], reverse=True)
    return scored


def allocate_cluster(
    nodes: list[NodeProbe],
    catalog: list[SpecialistCard],
    traffic_mix: dict[DomainId, float] | None = None,
    strategy: Strategy = "tiered",
) -> dict[NodeId, NodeSuggestion]:
    """Run the allocator. Returns one NodeSuggestion per node.

    Strategies:
      - "best_per_machine": each node picks its individually-highest-scoring
        specialist, ignoring duplication.
      - "full_set":         allocator ensures every domain has ≤1 host before
        any domain gets a second host.
      - "tiered":           fills tier-1 domains first, then tier-2, then
        tier-3 only if `traffic_mix` shows ≥5% in that domain. After all
        tiers covered, promotes replicas for the heaviest-traffic domain.

    Empty `traffic_mix` → tier-3 specialists are skipped (per spec §4 C).

    Edge: if a node has no specialist with finite score (all filtered),
    `primary=None`, rationale notes which hard filter failed.
    """
    traffic_mix = traffic_mix or {}
    coverage: dict[DomainId, set[NodeId]] = {}
    suggestions: dict[NodeId, NodeSuggestion] = {}

    if strategy == "best_per_machine":
        for node in nodes:
            ranked = _rank_specs_for_node(node, catalog, coverage)
            primary_score, primary = (ranked[0][0], ranked[0][1]) if ranked else (-math.inf, None)
            if primary is None or primary_score == -math.inf:
                suggestions[node.node_id] = NodeSuggestion(
                    node_id=node.node_id,
                    primary=None,
                    alternates=[],
                    rationale="no specialist passes hard filters on this node",
                    fit_score=-math.inf,
                )
                continue
            coverage.setdefault(primary.domain, set()).add(node.node_id)
            alternates = [s for sc, s in ranked[1:] if sc > -math.inf][:3]
            suggestions[node.node_id] = NodeSuggestion(
                node_id=node.node_id,
                primary=primary,
                alternates=alternates,
                rationale=f"best_per_machine: {primary.specialist_id} score={primary_score:.2f}",
                fit_score=primary_score,
            )
        return suggestions

    if strategy == "full_set":
        # Greedy: at each step, pick the (node, spec) pair with highest score
        # such that spec.domain is not yet covered. After all domains covered,
        # fall back to best_per_machine on remaining nodes.
        remaining_nodes = list(nodes)
        uncovered_domains = {s.domain for s in catalog}
        while remaining_nodes and uncovered_domains:
            best: tuple[float, NodeProbe, SpecialistCard] | None = None
            for node in remaining_nodes:
                for sc, spec in _rank_specs_for_node(node, catalog, coverage):
                    if spec.domain in uncovered_domains and sc > -math.inf:
                        if best is None or sc > best[0]:
                            best = (sc, node, spec)
                        break  # only consider top-eligible per node
            if best is None:
                break
            sc, node, spec = best
            coverage.setdefault(spec.domain, set()).add(node.node_id)
            uncovered_domains.discard(spec.domain)
            ranked_alts = _rank_specs_for_node(node, catalog, coverage)
            alternates = [s for s_, s in ranked_alts if s_ > -math.inf and s.specialist_id != spec.specialist_id][:3]
            suggestions[node.node_id] = NodeSuggestion(
                node_id=node.node_id,
                primary=spec,
                alternates=alternates,
                rationale=f"full_set: {spec.specialist_id} covers domain={spec.domain} score={sc:.2f}",
                fit_score=sc,
            )
            remaining_nodes = [n for n in remaining_nodes if n.node_id != node.node_id]
        # Any nodes left → best_per_machine fill
        for node in remaining_nodes:
            ranked = _rank_specs_for_node(node, catalog, coverage)
            if not ranked or ranked[0][0] == -math.inf:
                suggestions[node.node_id] = NodeSuggestion(
                    node_id=node.node_id, primary=None,
                    rationale="full_set fill: no eligible specialist", fit_score=-math.inf,
                )
                continue
            sc, spec = ranked[0]
            coverage.setdefault(spec.domain, set()).add(node.node_id)
            alts = [s for s_, s in ranked[1:] if s_ > -math.inf][:3]
            suggestions[node.node_id] = NodeSuggestion(
                node_id=node.node_id, primary=spec, alternates=alts,
                rationale=f"full_set fill: {spec.specialist_id} score={sc:.2f}", fit_score=sc,
            )
        return suggestions

    # ---- "tiered" (default) ----
    return _allocate_tiered(nodes, catalog, traffic_mix)


def _allocate_tiered(
    nodes: list[NodeProbe],
    catalog: list[SpecialistCard],
    traffic_mix: dict[DomainId, float],
) -> dict[NodeId, NodeSuggestion]:
    """Per spec §4 Strategy C.

    1. Fill tier-1 essentials (math/code/general) first.
    2. Fill tier-2 important when tier-1 covered.
    3. Fill tier-3 specialized only if traffic_mix shows ≥5% in domain.
    4. After all covered, promote replicas based on heaviest traffic share.
    """
    coverage: dict[DomainId, set[NodeId]] = {}
    suggestions: dict[NodeId, NodeSuggestion] = {}
    remaining = list(nodes)

    # Filter catalog by tier
    tier_specs: dict[int, list[SpecialistCard]] = {1: [], 2: [], 3: []}
    for s in catalog:
        tier_specs[_domain_tier(s.domain)].append(s)

    def _fill_tier(tier: int, domains_filter: frozenset[str] | None = None) -> None:
        candidates = tier_specs[tier]
        if domains_filter is not None:
            candidates = [c for c in candidates if c.domain in domains_filter]
        uncovered = {c.domain for c in candidates}
        # Don't re-cover something already done
        uncovered -= set(coverage.keys())
        while remaining and uncovered:
            best: tuple[float, NodeProbe, SpecialistCard] | None = None
            for node in remaining:
                for sc, spec in _rank_specs_for_node(node, candidates, coverage):
                    if spec.domain in uncovered and sc > -math.inf:
                        if best is None or sc > best[0]:
                            best = (sc, node, spec)
                        break
            if best is None:
                break
            sc, node, spec = best
            coverage.setdefault(spec.domain, set()).add(node.node_id)
            uncovered.discard(spec.domain)
            alt_ranked = _rank_specs_for_node(node, catalog, coverage)
            alternates = [s for s_, s in alt_ranked if s_ > -math.inf and s.specialist_id != spec.specialist_id][:3]
            suggestions[node.node_id] = NodeSuggestion(
                node_id=node.node_id,
                primary=spec,
                alternates=alternates,
                rationale=(
                    f"tiered tier-{tier}: {spec.specialist_id} covers "
                    f"domain={spec.domain} score={sc:.2f}"
                ),
                fit_score=sc,
            )
            remaining[:] = [n for n in remaining if n.node_id != node.node_id]

    _fill_tier(1)
    _fill_tier(2)
    # Tier 3: only fill domains with ≥5% traffic
    hot_t3 = frozenset(d for d, share in traffic_mix.items() if share >= 0.05)
    if hot_t3:
        _fill_tier(3, hot_t3 & TIER_3_DOMAINS)

    # Multi-specialist coexistence (spec §3.3 secondaries):
    # On any already-assigned node whose effective memory has 2× headroom
    # past primary.runtime_gb, opportunistically add up to one more
    # specialist from an uncovered tier-1/tier-2 domain. This is how a
    # 128GB-unified-mem Spark hosts code AND general on one box, instead
    # of forcing a 1-domain-per-node fan-out that wastes RAM.
    _fill_secondaries(nodes, suggestions, catalog, coverage)

    # Promote replicas for nodes still unassigned: pick the most-trafficked
    # tier-1 domain (default "general") and let the node host a replica.
    if remaining:
        heavy_domain = max(
            traffic_mix.items(), key=lambda kv: kv[1], default=("general", 0.0)
        )[0]
        for node in remaining:
            ranked = _rank_specs_for_node(node, catalog, coverage)
            chosen: SpecialistCard | None = None
            chosen_score = -math.inf
            # Prefer something in the hottest domain
            for sc, spec in ranked:
                if sc > -math.inf and spec.domain == heavy_domain:
                    chosen, chosen_score = spec, sc
                    break
            if chosen is None and ranked and ranked[0][0] > -math.inf:
                chosen_score, chosen = ranked[0]
            if chosen is None:
                suggestions[node.node_id] = NodeSuggestion(
                    node_id=node.node_id, primary=None,
                    rationale="tiered: no eligible specialist after hard filters",
                    fit_score=-math.inf,
                )
                continue
            coverage.setdefault(chosen.domain, set()).add(node.node_id)
            alts = [s for s_, s in ranked if s_ > -math.inf and s.specialist_id != chosen.specialist_id][:3]
            suggestions[node.node_id] = NodeSuggestion(
                node_id=node.node_id,
                primary=chosen,
                alternates=alts,
                rationale=(
                    f"tiered replica: {chosen.specialist_id} (domain={chosen.domain}) "
                    f"score={chosen_score:.2f}"
                ),
                fit_score=chosen_score,
            )

    # Any nodes with NO assignment yet (e.g., the inner loops above broke
    # before reaching them): record an explicit no-fit suggestion so the
    # registry sees them and can ship them registry-only mode.
    for node in nodes:
        if node.node_id not in suggestions:
            suggestions[node.node_id] = NodeSuggestion(
                node_id=node.node_id,
                primary=None,
                rationale="tiered: no eligible specialist (hard-filter exhaustion)",
                fit_score=-math.inf,
            )

    return suggestions


def _fill_secondaries(
    nodes: list[NodeProbe],
    suggestions: dict[NodeId, NodeSuggestion],
    catalog: list[SpecialistCard],
    coverage: dict[DomainId, set[NodeId]],
) -> None:
    """Promote a second specialist onto high-memory nodes (spec §3.3).

    Rule: on each node where the assigned primary uses ≤ half the node's
    effective memory, search the catalog for the highest-scoring uncovered
    tier-1/tier-2 specialist that ALSO fits in remaining headroom. If
    found, attach as secondary and mark domain covered. One secondary
    per node max in v0.0.3 (n-way packing is v0.0.4+).

    Skipped if primary is None (registry-only node) or if no uncovered
    tier-1/tier-2 specialist fits the remaining memory budget.

    This mutates `suggestions` in place by replacing the NodeSuggestion
    record with one carrying `secondaries=[chosen]`. Coverage map is also
    updated so subsequent suggestion lookups see the new domain.
    """
    node_by_id = {n.node_id: n for n in nodes}
    for node_id, sugg in list(suggestions.items()):
        if sugg.primary is None:
            continue
        node = node_by_id.get(node_id)
        if node is None:
            continue
        eff_mem = _effective_vram_gb(node)
        if eff_mem <= 0:
            continue
        primary_share = sugg.primary.runtime_gb / eff_mem
        if primary_share > 0.5:
            # Primary already takes >50% of memory; coexistence risks OOM.
            continue
        remaining_mem = eff_mem - sugg.primary.runtime_gb
        # Candidate domains: tier-1 + tier-2 not yet covered (anywhere).
        covered_domains = set(coverage.keys())
        eligible_tiers = TIER_1_DOMAINS | TIER_2_DOMAINS
        candidates = [
            s for s in catalog
            if s.domain in eligible_tiers
            and s.domain not in covered_domains
            and s.specialist_id != sugg.primary.specialist_id
            and s.runtime_gb <= remaining_mem
        ]
        if not candidates:
            continue
        # Rank by score under current coverage; pick highest finite.
        scored = _rank_specs_for_node(node, candidates, coverage)
        chosen: SpecialistCard | None = None
        chosen_score = -math.inf
        for sc, spec in scored:
            if sc > -math.inf:
                chosen, chosen_score = spec, sc
                break
        if chosen is None:
            continue
        coverage.setdefault(chosen.domain, set()).add(node.node_id)
        suggestions[node_id] = NodeSuggestion(
            node_id=node_id,
            primary=sugg.primary,
            secondaries=[chosen],
            alternates=sugg.alternates,
            rationale=(
                f"{sugg.rationale} + secondary {chosen.specialist_id} "
                f"(domain={chosen.domain} score={chosen_score:.2f}, "
                f"fits {chosen.runtime_gb:.1f}GB of {remaining_mem:.1f}GB headroom)"
            ),
            sticky=sugg.sticky,
            fit_score=sugg.fit_score,
        )


__all__ = [
    "Strategy",
    "TIER_1_DOMAINS",
    "TIER_2_DOMAINS",
    "TIER_3_DOMAINS",
    "allocate_cluster",
    "model_fit_score",
]
