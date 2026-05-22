"""Catalog tests — TOML cards load correctly and conform to schema."""

from __future__ import annotations

from mesh.catalog import load_catalog
from mesh.models import SpecialistCard


def test_catalog_loads_expected_set():
    """Catalog covers the v0.0.2 set: 5 tier-1/2 + 1 real-weights card.

    `qwen3-coder-30b-a3b-fp8` was added in v0.0.2 — it's the first card
    backed by actually-cached weights served by vLLM on the local Spark.
    """
    cards = load_catalog()
    ids = {c.specialist_id for c in cards}
    required = {
        "aya-expanse-8b-q4",
        "llama-3.1-8b-instruct-q4",
        "phi-4-14b-q4",
        "qwen3-coder-7b-q4",
        "qwen3-math-7b-q4",
        "qwen3-coder-30b-a3b-fp8",
    }
    assert required.issubset(ids), f"missing: {required - ids}"


def test_each_card_is_specialist_card():
    for c in load_catalog():
        assert isinstance(c, SpecialistCard)
        assert c.runtime_gb >= c.storage_gb  # weights fit in runtime budget
        assert c.min_vram_gb >= 1.0
        assert c.context_window >= 2048
        assert c.required_backend in ("vllm", "llamacpp", "ollama", "mlx", "hf_transformers")
        assert "gb10" in c.estimated_tps_at  # we benchmark on Spark first
        assert c.coverage_tier in (1, 2, 3)


def test_tier_1_specialists_cover_math_code_general():
    """Spec §4 Strategy C invariant: tier-1 covers essentials."""
    cards = load_catalog()
    tier_1 = [c for c in cards if c.coverage_tier == 1]
    domains = {c.domain for c in tier_1}
    # Math + code + general/reasoning must be in tier 1.
    assert "math" in domains
    assert "code" in domains
    # reasoning OR general fills the third essential slot.
    assert "reasoning" in domains or "general" in domains
