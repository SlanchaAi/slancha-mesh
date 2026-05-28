"""Tests for mesh.dashboard.cost — pure-function counterfactual cost math."""

from __future__ import annotations

import pytest

from mesh.dashboard.cost import (
    DEFAULT_INPUT_TOKENS,
    MODEL_PRICING,
    Pricing,
    _resolve_pricing,
    compute_actual_cost,
    cost_summary,
    counterfactual_cost,
    estimate_tokens,
)


# ---------------------------------------------------------------------------
# Pricing resolution
# ---------------------------------------------------------------------------


def test_resolve_exact_match():
    p = _resolve_pricing("claude-sonnet-4-7")
    assert p.family == "anthropic"
    assert p.input_per_1k == 0.003


def test_resolve_substring_match_qwen_variant():
    """Spark may emit 'qwen3-coder-30b-a3b-fp8' — should match local pricing."""
    p = _resolve_pricing("qwen3-coder-30b-a3b-fp8")
    assert p.family == "local"


def test_resolve_unknown_falls_to_local_zero_cost():
    p = _resolve_pricing("some-random-model-not-in-table")
    assert p.family == "local"
    assert p.cost(100, 100) == 0.0


def test_resolve_none_is_zero():
    p = _resolve_pricing(None)
    assert p.cost(1000, 1000) == 0.0


# ---------------------------------------------------------------------------
# Pricing.cost
# ---------------------------------------------------------------------------


def test_pricing_cost_basic():
    p = Pricing(input_per_1k=0.003, output_per_1k=0.015)
    # 1000 input + 1000 output → exactly 0.003 + 0.015
    assert p.cost(1000, 1000) == pytest.approx(0.018)


def test_pricing_cost_scales_linearly():
    p = MODEL_PRICING["claude-sonnet-4-7"]
    assert p.cost(2000, 2000) == pytest.approx(2 * p.cost(1000, 1000))


def test_local_models_are_free():
    for k in ("qwen3-coder-30b", "qwen3-8b", "phi4:14b", "codestral:22b"):
        assert MODEL_PRICING[k].cost(99999, 99999) == 0.0


# ---------------------------------------------------------------------------
# Token estimation
# ---------------------------------------------------------------------------


def test_estimate_tokens_prefers_explicit():
    rec = {
        "picked": "qwen3-coder-30b",
        "first_latency_ms": 5000,
        "tokens": {"input": 234, "output": 567},
    }
    in_tok, out_tok = estimate_tokens(rec)
    assert in_tok == 234
    assert out_tok == 567


def test_estimate_tokens_falls_back_to_latency():
    """1000ms × 38 TPS = 38 tokens output for qwen3-coder."""
    rec = {"picked": "qwen3-coder-30b", "first_latency_ms": 1000}
    in_tok, out_tok = estimate_tokens(rec)
    assert in_tok == DEFAULT_INPUT_TOKENS
    assert out_tok == 38


def test_estimate_tokens_substring_fallback_to_coder():
    """A variant name should bucket to qwen3-coder-30b TPS."""
    rec = {"picked": "qwen3-coder-30b-a3b-fp8", "first_latency_ms": 1000}
    _, out_tok = estimate_tokens(rec)
    assert out_tok == 38


def test_estimate_tokens_substring_fallback_to_8b():
    rec = {"picked": "qwen3-8b", "first_latency_ms": 1000}
    _, out_tok = estimate_tokens(rec)
    assert out_tok == 72


def test_estimate_tokens_cloud_bucket():
    rec = {"picked": "openrouter/llama-3.1-70b", "first_latency_ms": 1000}
    _, out_tok = estimate_tokens(rec)
    assert out_tok == 45


def test_estimate_tokens_zero_latency_emits_zero_output():
    rec = {"picked": "qwen3-coder-30b", "first_latency_ms": 0}
    in_tok, out_tok = estimate_tokens(rec)
    assert out_tok == 0


def test_estimate_tokens_picked_to_field_works_for_overrides():
    """Override rows use 'picked_to' not 'picked'."""
    rec = {"picked_to": "qwen3-coder-30b", "first_latency_ms": 1000}
    _, out_tok = estimate_tokens(rec)
    assert out_tok == 38


# ---------------------------------------------------------------------------
# compute_actual_cost
# ---------------------------------------------------------------------------


def test_compute_actual_cost_all_local_is_zero():
    decisions = [
        {"picked": "qwen3-coder-30b", "first_latency_ms": 1000},
        {"picked": "qwen3-8b",        "first_latency_ms": 500},
        {"picked": "qwen3-coder-30b", "first_latency_ms": 2000},
    ]
    cost = compute_actual_cost(decisions)
    assert cost.total_usd == 0.0
    assert cost.decision_count == 3
    assert cost.total_output_tokens > 0  # tokens still counted


def test_compute_actual_cost_cloud_paid_charges():
    decisions = [
        {"picked": "deepseek/deepseek-v3", "first_latency_ms": 1000,
         "tokens": {"input": 1000, "output": 1000}},
    ]
    cost = compute_actual_cost(decisions)
    # deepseek = 0.0014 input + 0.0028 output per 1k
    assert cost.total_usd == pytest.approx(0.0042)
    assert cost.by_family["cloud-paid"] == pytest.approx(0.0042)


def test_compute_actual_cost_by_backend_breakdown():
    decisions = [
        {"picked": "qwen3-coder-30b",         "first_latency_ms": 1000},
        {"picked": "qwen3-coder-30b",         "first_latency_ms": 1000},
        {"picked": "meta-llama/llama-3.1-70b","first_latency_ms": 500,
         "tokens": {"input": 100, "output": 100}},
    ]
    cost = compute_actual_cost(decisions)
    assert cost.by_backend["qwen3-coder-30b"] == 0.0
    assert cost.by_backend["meta-llama/llama-3.1-70b"] > 0


# ---------------------------------------------------------------------------
# counterfactual_cost
# ---------------------------------------------------------------------------


def test_counterfactual_claude_is_higher_than_local():
    """All-Claude must cost > all-local on the same tokens."""
    decisions = [
        {"picked": "qwen3-coder-30b", "first_latency_ms": 2000,
         "tokens": {"input": 500, "output": 500}},
    ]
    actual = compute_actual_cost(decisions)
    cf = counterfactual_cost(decisions, "claude-sonnet-4-7")
    assert cf > actual.total_usd


def test_counterfactual_uses_baseline_pricing_uniformly():
    """Every record gets re-costed at baseline, regardless of original pick."""
    decisions = [
        {"picked": "qwen3-coder-30b", "first_latency_ms": 1000,
         "tokens": {"input": 1000, "output": 1000}},
        {"picked": "qwen3-coder-30b", "first_latency_ms": 1000,
         "tokens": {"input": 1000, "output": 1000}},
    ]
    cf = counterfactual_cost(decisions, "claude-sonnet-4-7")
    # 2 × 1000 input + 2 × 1000 output @ Claude-Sonnet rates
    expected = 2 * (0.003 + 0.015)
    assert cf == pytest.approx(expected)


def test_counterfactual_opus_higher_than_sonnet():
    decisions = [
        {"picked": "qwen3-coder-30b", "first_latency_ms": 1000,
         "tokens": {"input": 1000, "output": 1000}},
    ]
    sonnet = counterfactual_cost(decisions, "claude-sonnet-4-7")
    opus   = counterfactual_cost(decisions, "claude-opus-4-7")
    assert opus > sonnet


# ---------------------------------------------------------------------------
# cost_summary (top-card output)
# ---------------------------------------------------------------------------


def test_cost_summary_empty():
    s = cost_summary([])
    assert s.actual_usd == 0.0
    assert s.savings_vs_claude_pct == 0.0
    assert s.decision_count == 0


def test_cost_summary_local_only_savings_is_100_pct():
    """All-local actual = $0 → savings vs Claude is 100%."""
    decisions = [
        {"picked": "qwen3-coder-30b", "first_latency_ms": 1000,
         "tokens": {"input": 100, "output": 100}}
        for _ in range(10)
    ]
    s = cost_summary(decisions)
    assert s.actual_usd == 0.0
    assert s.counterfactual_claude_usd > 0
    assert s.savings_vs_claude_pct == 100.0


def test_cost_summary_mixed_local_and_cloud_partial_savings():
    decisions = [
        # 9 local, 1 cloud-paid
        *[{"picked": "qwen3-coder-30b", "first_latency_ms": 1000,
           "tokens": {"input": 100, "output": 100}} for _ in range(9)],
        {"picked": "deepseek/deepseek-v3", "first_latency_ms": 1000,
         "tokens": {"input": 100, "output": 100}},
    ]
    s = cost_summary(decisions)
    assert s.actual_usd > 0
    assert s.counterfactual_claude_usd > s.actual_usd
    assert 0 < s.savings_vs_claude_pct < 100


def test_cost_summary_by_family_buckets():
    decisions = [
        {"picked": "qwen3-coder-30b", "first_latency_ms": 1000,
         "tokens": {"input": 100, "output": 100}},
        {"picked": "deepseek/deepseek-v3", "first_latency_ms": 1000,
         "tokens": {"input": 100, "output": 100}},
    ]
    s = cost_summary(decisions)
    assert "local" in s.by_family
    assert "cloud-paid" in s.by_family
    assert s.by_family["local"] == 0.0
    assert s.by_family["cloud-paid"] > 0


def test_cost_summary_marks_estimated_when_no_tokens():
    decisions = [{"picked": "qwen3-coder-30b", "first_latency_ms": 1000}]
    s = cost_summary(decisions)
    assert s.estimated is True


def test_cost_summary_marks_unestimated_when_all_tokens_present():
    decisions = [
        {"picked": "qwen3-coder-30b", "first_latency_ms": 1000,
         "tokens": {"input": 100, "output": 100}}
    ]
    s = cost_summary(decisions)
    assert s.estimated is False
