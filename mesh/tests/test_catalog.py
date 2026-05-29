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


# Ollama / LocalLLaMA-class cards added so the catalog isn't FP8-30B-only.
# These IDs are checked-in DRAFT cards (see docs/CATALOG_STATUS.md); the
# tests here only assert schema invariants — bring-up validation lives in
# the status doc.
_OLLAMA_CARD_IDS = {
    "llama-3.1-8b-instruct-q5-ollama",
    "qwen2.5-coder-7b-q4-ollama",
    "deepseek-coder-v2-16b-lite-q4-ollama",
    "phi-3.5-mini-q5-ollama",
    "gemma-2-9b-q4-ollama",
    "mistral-nemo-12b-q4-ollama",
}


def test_ollama_cards_load_with_required_fields():
    """The new GGUF-on-Ollama cards parse + carry their `ollama_tag`."""
    cards = {c.specialist_id: c for c in load_catalog()}
    missing = _OLLAMA_CARD_IDS - set(cards)
    assert not missing, f"Ollama cards not loaded: {missing}"
    for sid in _OLLAMA_CARD_IDS:
        card = cards[sid]
        assert card.required_backend == "ollama", f"{sid}: required_backend != ollama"
        assert card.ollama_tag, f"{sid}: ollama_tag empty — OllamaBackend would refuse"
        # The catalog ID and the Ollama tag must agree on the rough family
        # name so a future operator doesn't ship a llama-3-named card
        # pointing at qwen weights. Cheap structural check, not semantic.
        family = sid.split("-q")[0].split("-ollama")[0]
        # gemma-2 / phi-3.5 etc. have dots in their Ollama tag; normalize.
        tag_head = card.ollama_tag.split(":")[0].replace(".", "-")
        # We allow a one-segment vs many-segment mismatch (qwen2.5-coder
        # ↔ qwen2.5-coder-7b) — the heuristic is "tag prefix appears
        # somewhere in the family".
        assert (
            tag_head.split("-")[0] in family
            or family.split("-")[0] in tag_head
        ), f"{sid}: tag '{card.ollama_tag}' family doesn't match specialist_id"


def test_catalog_spans_engines_after_ollama_cards():
    """Catalog must expose at least two engine choices so engine_select's
    Ollama recommendation has a card to bind to. Pre-#45 the catalog was
    vllm-only and a Mac / sub-24-GB-NVIDIA recommendation dead-ended at
    `NullBackend`.
    """
    backends = {c.required_backend for c in load_catalog()}
    assert {"vllm", "ollama"}.issubset(backends), backends


def test_ollama_cards_have_no_duplicate_specialist_ids():
    """Catalog IDs are the routing keys; uniqueness is a hard invariant."""
    cards = load_catalog()
    ids = [c.specialist_id for c in cards]
    assert len(ids) == len(set(ids)), "duplicate specialist_id in catalog"
