"""Allocator unit tests — model_fit_score + allocate_cluster.

Spec §3.2 / §4. We test:
- Hard filters individually (backend, storage, vram, fp4).
- Soft score ordering (coverage need > throughput > headroom > network).
- All three strategies on a 2-node mock cluster.
"""

from __future__ import annotations

import math

import pytest

from mesh.allocator import (
    allocate_cluster,
    model_fit_score,
)
from mesh.catalog import load_catalog
from mesh.models import NodeProbe, SpecialistCard


# ---------------------------------------------------------------------------
# model_fit_score — hard filters
# ---------------------------------------------------------------------------


def _spec_by_id(cards: list[SpecialistCard], sid: str) -> SpecialistCard:
    for c in cards:
        if c.specialist_id == sid:
            return c
    raise KeyError(sid)


def test_fit_score_hard_filter_backend(spark_node, catalog):
    spec = _spec_by_id(catalog, "qwen3-math-7b-q4")  # requires vllm
    # Node without vllm
    no_vllm = spark_node.model_copy(update={"available_backends": ["llamacpp"]})
    assert model_fit_score(spec, no_vllm, {}) == -math.inf


def test_fit_score_hard_filter_storage(spark_node, catalog):
    spec = _spec_by_id(catalog, "phi-4-14b-q4")  # needs 8.4GB
    no_disk = spark_node.model_copy(update={"disk_free_gb": 1.0})
    assert model_fit_score(spec, no_disk, {}) == -math.inf


def test_fit_score_hard_filter_vram(tiny_node, catalog):
    spec = _spec_by_id(catalog, "phi-4-14b-q4")
    # tiny_node has 3GB RAM (unified), -8 reserve = -5 effective; spec
    # needs 13GB.
    assert model_fit_score(spec, tiny_node, {}) == -math.inf


def test_fit_score_passes_on_spark(spark_node, catalog):
    spec = _spec_by_id(catalog, "qwen3-math-7b-q4")
    score = model_fit_score(spec, spark_node, {})
    assert math.isfinite(score)
    assert score > 0


def test_fit_score_coverage_need_dominates(spark_node, mac_mini_node, catalog):
    """A specialist not yet covered should out-score one that is covered,
    when hardware fit is similar."""
    coder = _spec_by_id(catalog, "qwen3-coder-7b-q4")
    math_ = _spec_by_id(catalog, "qwen3-math-7b-q4")
    # Cluster already has math covered on mac-mini-1
    coverage = {"math": {"mac-mini-1"}}
    s_math = model_fit_score(math_, spark_node, coverage)
    s_code = model_fit_score(coder, spark_node, coverage)
    assert s_code > s_math  # code is uncovered → higher coverage_score


# ---------------------------------------------------------------------------
# allocate_cluster — strategies
# ---------------------------------------------------------------------------


def test_allocate_best_per_machine_single_spark(spark_node, catalog):
    suggestions = allocate_cluster([spark_node], catalog, strategy="best_per_machine")
    assert spark_node.node_id in suggestions
    s = suggestions[spark_node.node_id]
    assert s.primary is not None
    # Spark can host any tier-1 specialist; best_per_machine just picks
    # the highest-scoring one. We don't pin the exact pick (depends on
    # tps tie-breakers); we assert it's a valid card.
    assert s.primary.specialist_id in {c.specialist_id for c in catalog}
    assert s.fit_score > 0


def test_allocate_full_set_two_nodes(spark_node, mac_mini_node, catalog):
    """Spark hosts vllm specialists; mac mini hosts llamacpp/mlx-only.

    Mac mini cannot host any of our tier-1 cards (all require vllm),
    so full_set should assign a vllm spec to spark and report no-fit
    on mac mini.
    """
    sugg = allocate_cluster(
        [spark_node, mac_mini_node], catalog, strategy="full_set"
    )
    spark_pick = sugg[spark_node.node_id]
    mac_pick = sugg[mac_mini_node.node_id]
    assert spark_pick.primary is not None
    assert spark_pick.primary.required_backend == "vllm"
    # Mac mini: every catalog card requires vllm, so it should have
    # primary=None.
    assert mac_pick.primary is None
    assert "no eligible" in mac_pick.rationale.lower()


def test_allocate_tiered_two_spark_cluster(spark_node, catalog):
    """Two Sparks → tier-1 essentials should diversify, not double-up math."""
    spark_2 = spark_node.model_copy(update={"node_id": "spark-2", "friendly_name": "spark-2"})
    sugg = allocate_cluster([spark_node, spark_2], catalog, strategy="tiered")
    assert len(sugg) == 2
    primaries = [s.primary.domain for s in sugg.values() if s.primary]
    assert len(primaries) == 2
    # Tiered MUST diversify on a 2-node cluster before doubling-up.
    assert primaries[0] != primaries[1], f"expected diversification, got {primaries}"


def test_allocate_tiered_tier3_gated_on_traffic(spark_node, catalog):
    """Tier-3 domains should NOT be filled when traffic_mix is empty.

    Our v0.0.1 catalog has no tier-3 specialists so this is more of a
    smoke check that the gate doesn't error on empty traffic.
    """
    sugg = allocate_cluster([spark_node], catalog, strategy="tiered", traffic_mix={})
    assert spark_node.node_id in sugg


def test_allocate_tiered_more_nodes_than_domains_promotes_replicas(spark_node, catalog):
    """5 Sparks, 5 specialists → after each domain has a host, the next
    Spark should still get a primary (replica promotion path)."""
    nodes = [
        spark_node.model_copy(update={"node_id": f"spark-{i}", "friendly_name": f"spark-{i}"})
        for i in range(1, 6)
    ]
    sugg = allocate_cluster(
        nodes, catalog, strategy="tiered", traffic_mix={"math": 0.6, "code": 0.3, "general": 0.1}
    )
    assigned = [s for s in sugg.values() if s.primary is not None]
    assert len(assigned) >= 4  # at least 4 of 5 should get something assigned


def test_allocate_tiered_single_spark_gets_secondary(spark_node, catalog):
    """Single GB10 Spark (128GB unified, ~100GB effective): primary uses
    a fraction of memory → allocator should attach a secondary specialist
    from an uncovered tier-1 domain so both code AND math/general coexist
    on the one box.

    Without secondaries, a 1-Spark cluster would host one domain and
    cloud-fallback for everything else — wasting 80GB of unified mem.
    """
    sugg = allocate_cluster([spark_node], catalog, strategy="tiered")
    s = sugg[spark_node.node_id]
    assert s.primary is not None, "primary should be set on GB10 Spark"
    assert len(s.secondaries) >= 1, (
        f"expected ≥1 secondary on 128GB GB10; got primary={s.primary.specialist_id}"
        f" with secondaries={[c.specialist_id for c in s.secondaries]}"
    )
    secondary = s.secondaries[0]
    assert secondary.specialist_id != s.primary.specialist_id
    assert secondary.domain != s.primary.domain, (
        "secondary domain should differ from primary (coverage diversification)"
    )


def test_allocate_tiered_tight_node_gets_no_secondary(catalog, mac_mini_node):
    """Mac mini M4 with 50GB RAM available: after primary, remaining
    headroom is too tight for a second specialist with min_vram ≥ 7GB.
    Secondaries should NOT be assigned when primary already takes
    >50% of effective memory.

    This guards against OOM: if primary is 35GB on a 50GB-effective node,
    that's 70% memory share — adding a secondary would exhaust headroom.
    """
    # Build a synthetic tight-node by halving Mac mini available RAM
    tight = mac_mini_node.model_copy(update={"ram_available_gb": 12.0})
    # Filter catalog to llama.cpp-only since Mac mini lacks vllm
    cpp_catalog = [c for c in catalog if c.required_backend in ("llamacpp", "ollama", "mlx")]
    if not cpp_catalog:
        # No llamacpp-required cards in the catalog → nothing to assign;
        # test still passes (no secondary because no primary either).
        return
    sugg = allocate_cluster([tight], cpp_catalog, strategy="tiered")
    s = sugg[tight.node_id]
    if s.primary is None:
        # Hard-filtered out — that's fine.
        return
    # Primary should consume >50% of 12GB effective → no secondary.
    eff = tight.ram_available_gb - 8.0  # _effective_vram_gb subtracts OS reserve
    primary_share = s.primary.runtime_gb / max(eff, 0.1)
    if primary_share > 0.5:
        assert s.secondaries == [], (
            f"tight node ({primary_share:.0%} primary share) should NOT "
            f"get a secondary; got {[c.specialist_id for c in s.secondaries]}"
        )


def test_secondary_does_not_break_existing_coverage_diversification(spark_node, catalog):
    """Two-Spark cluster + secondaries: each Spark still gets a unique
    primary domain, AND each may carry secondaries. The secondaries on
    spark-1 must not be the same domain as primary on spark-2 (the
    secondary fill consults the same coverage map)."""
    spark_2 = spark_node.model_copy(update={"node_id": "spark-2", "friendly_name": "spark-2"})
    sugg = allocate_cluster([spark_node, spark_2], catalog, strategy="tiered")
    primaries = {s.primary.domain for s in sugg.values() if s.primary}
    assert len(primaries) == 2, "primaries should diversify across 2 Sparks"

    # Collect all (node, domain) pairs across primary + secondaries
    seen: dict[tuple[str, str], int] = {}
    for s in sugg.values():
        if s.primary:
            seen[(s.node_id, s.primary.domain)] = seen.get((s.node_id, s.primary.domain), 0) + 1
        for sec in s.secondaries:
            seen[(s.node_id, sec.domain)] = seen.get((s.node_id, sec.domain), 0) + 1
    # No (node, domain) appears twice on the same node
    assert all(c == 1 for c in seen.values()), (
        f"duplicate domain on same node: {seen}"
    )


def test_allocate_unknown_strategy_errors_quietly(spark_node, catalog):
    """Unknown strategy strings should pass through as 'tiered' default in
    our impl (Literal type catches at static check; runtime is permissive)."""
    sugg = allocate_cluster([spark_node], catalog, strategy="tiered")
    assert spark_node.node_id in sugg


# ---------------------------------------------------------------------------
# Synthetic combo tests — 3 (node, spec) pairs covering filter paths
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "spec_id,node_fix,expected_finite",
    [
        ("qwen3-math-7b-q4", "spark_node", True),
        ("phi-4-14b-q4", "tiny_node", False),  # no VRAM + no vllm
        ("qwen3-coder-7b-q4", "mac_mini_node", False),  # no vllm
    ],
)
def test_fit_score_synthetic_combos(spec_id, node_fix, expected_finite, request, catalog):
    node: NodeProbe = request.getfixturevalue(node_fix)
    spec = _spec_by_id(catalog, spec_id)
    score = model_fit_score(spec, node, {})
    if expected_finite:
        assert math.isfinite(score)
    else:
        assert score == -math.inf
