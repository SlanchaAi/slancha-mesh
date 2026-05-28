"""Counterfactual cost computation over the pipeline's decisions log.

The headline question this answers: *"Routing through slancha vs always
calling Claude — what did we save?"*

Inputs: a list of decision records (from `dashboard/decisions.jsonl` via
`live_run.py` or `oracle.py` loaders). Each record names the model that
served the prompt and gives a `first_latency_ms`. We don't have token
counts in decisions.jsonl yet — they live in the floodgate ledger and
will eventually be propagated. Until then we estimate tokens from
latency using per-backend throughput constants (`BACKEND_TPS_OUT`) and a
naive input-token assumption. Estimation is clearly labeled in the
output so nobody mistakes it for a billed figure.

Pricing table:
  - Local vLLM:        $0 marginal (hardware amortized; not a per-request cost)
  - OpenRouter free tier: $0 (`olmo-3-32b-think` etc. on free routes)
  - OpenRouter paid:   conservative per-1k token rates
  - Claude Sonnet 4.7: counterfactual baseline ("what if always-Claude")
  - Claude Opus 4.7:   alternative baseline ("what if always-Opus")

All prices in USD. Update `MODEL_PRICING` when OpenRouter rates change.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, TypeAlias

DecisionRecord: TypeAlias = dict[str, Any]


# ---------------------------------------------------------------------------
# Pricing — conservative per-1k token rates (USD)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Pricing:
    """Per-1k-token prices for a model."""

    input_per_1k:  float
    output_per_1k: float
    family: str = "?"        # "local" | "cloud-free" | "cloud-paid" | "anthropic"

    def cost(self, input_tokens: int, output_tokens: int) -> float:
        """Cost in USD for one request at these rates."""
        return (
            (input_tokens  / 1000.0) * self.input_per_1k +
            (output_tokens / 1000.0) * self.output_per_1k
        )


# As of 2026-05-17; sources: OpenRouter dashboard, Anthropic pricing page.
# Keep this table as the single source of truth — `compute_actual_cost()`
# resolves model keys here. When a model isn't in the table, falls back to
# UNKNOWN_LOCAL (treated as $0 marginal — assumed local hardware).
MODEL_PRICING: dict[str, Pricing] = {
    # ---- Local vLLM (hardware amortized, $0 per-request) ----
    "qwen3-coder-30b":             Pricing(0.0,   0.0,   "local"),
    "qwen3-coder-30b-a3b-fp8":     Pricing(0.0,   0.0,   "local"),
    "qwen3-8b":                    Pricing(0.0,   0.0,   "local"),
    # Legacy ollama tags retained for old decisions data
    "phi4:14b":                    Pricing(0.0,   0.0,   "local"),
    "codestral:22b":               Pricing(0.0,   0.0,   "local"),
    "gemma2:9b":                   Pricing(0.0,   0.0,   "local"),
    "qwen3:8b":                    Pricing(0.0,   0.0,   "local"),
    "qwen3:4b":                    Pricing(0.0,   0.0,   "local"),

    # ---- OpenRouter cloud OSS (paid tiers, conservative) ----
    "meta-llama/llama-3.1-70b":    Pricing(0.0006, 0.0008, "cloud-paid"),
    "mistralai/mixtral-8x7b":      Pricing(0.0007, 0.0007, "cloud-paid"),
    "deepseek/deepseek-v3":        Pricing(0.0014, 0.0028, "cloud-paid"),

    # ---- OpenRouter cloud OSS (free tier — used when available) ----
    "olmo-3-32b-think":            Pricing(0.0,   0.0,   "cloud-free"),
    "allenai/olmo-3-32b-think":    Pricing(0.0,   0.0,   "cloud-free"),

    # ---- Anthropic Claude (counterfactual baseline) ----
    "claude-sonnet-4-7":           Pricing(0.003, 0.015, "anthropic"),
    "claude-opus-4-7":             Pricing(0.015, 0.075, "anthropic"),
    "claude-haiku-4-5":            Pricing(0.0008, 0.004, "anthropic"),
}

# Default per-backend output tokens-per-second for token estimation when
# decisions.jsonl lacks explicit token counts. Conservative; honest about
# being estimates.
BACKEND_TPS_OUT: dict[str, float] = {
    "qwen3-coder-30b": 38.0,    # vLLM batch decode @ FP8, observed 19s/req ÷ ~700 tok
    "qwen3-8b":        72.0,    # vLLM FP16, observed faster on shorter outputs
    "cloud":           45.0,    # OpenRouter average, varies wildly by provider
    # Legacy ollama for completeness on historical data
    "phi4:14b":        14.0,
    "codestral:22b":   11.0,
    "gemma2:9b":       28.0,
    "qwen3:8b":        18.0,
    "qwen3:4b":        42.0,
}
# Naive input-token assumption when not present. Most prompts in v3.1 are
# under 200 tokens; this is a deliberate underestimate so cost-savings
# numbers stay on the conservative side.
DEFAULT_INPUT_TOKENS = 80
UNKNOWN_LOCAL = Pricing(0.0, 0.0, "local")


# ---------------------------------------------------------------------------
# Token estimation
# ---------------------------------------------------------------------------


def _resolve_pricing(model_key: str | None) -> Pricing:
    """Substring-tolerant lookup mirroring pipeline.html's resolveModelPaths.
    Falls back to UNKNOWN_LOCAL (i.e. $0 marginal) for anything not in the
    table so we never inflate cost on an unknown local model. Cloud/Anthropic
    matches need explicit substrings."""
    if not model_key:
        return UNKNOWN_LOCAL
    if model_key in MODEL_PRICING:
        return MODEL_PRICING[model_key]
    key = model_key.lower()
    for table_key, pricing in MODEL_PRICING.items():
        if table_key.lower() in key or key in table_key.lower():
            return pricing
    return UNKNOWN_LOCAL


def estimate_tokens(record: DecisionRecord) -> tuple[int, int]:
    """Return (input_tokens, output_tokens). Prefers explicit fields when
    spark has propagated them; falls back to latency-derived estimate."""
    tokens = record.get("tokens") or {}
    in_tok = tokens.get("input")
    out_tok = tokens.get("output")
    if isinstance(in_tok, (int, float)) and isinstance(out_tok, (int, float)):
        return int(in_tok), int(out_tok)

    # Estimate from latency + backend TPS
    picked = record.get("picked") or record.get("picked_to") or "?"
    latency_ms = record.get("first_latency_ms") or 0
    if latency_ms <= 0:
        return DEFAULT_INPUT_TOKENS, 0

    # Resolve a TPS — try exact, then bucket
    tps = BACKEND_TPS_OUT.get(picked)
    if tps is None:
        key = picked.lower()
        if "coder" in key:
            tps = BACKEND_TPS_OUT["qwen3-coder-30b"]
        elif "8b" in key:
            tps = BACKEND_TPS_OUT["qwen3-8b"]
        elif "cloud" in key or "olmo" in key or "openrouter" in key:
            tps = BACKEND_TPS_OUT["cloud"]
        else:
            tps = 30.0  # conservative middle ground
    est_out = max(1, int((latency_ms / 1000.0) * tps))
    return DEFAULT_INPUT_TOKENS, est_out


def estimated_field_explanation() -> str:
    """One-line caption telling the operator the cost numbers are estimates
    until spark propagates real token counts into decisions.jsonl."""
    return (
        "Estimated via latency × backend-TPS (decisions log lacks per-request "
        "tokens). Real billing numbers will replace these when spark adds "
        "tokens to decisions.jsonl."
    )


# ---------------------------------------------------------------------------
# Cost computation
# ---------------------------------------------------------------------------


@dataclass
class CostBreakdown:
    """Detailed per-backend cost breakdown over a decision set."""

    total_usd: float = 0.0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    by_backend: dict[str, float] = field(default_factory=dict)
    by_family: dict[str, float] = field(default_factory=dict)
    decision_count: int = 0


def compute_actual_cost(decisions: list[DecisionRecord]) -> CostBreakdown:
    """Sum cost across all decisions using each row's chosen model."""
    out = CostBreakdown()
    for d in decisions:
        picked = (d.get("picked") or d.get("picked_to") or "?")
        pricing = _resolve_pricing(picked)
        in_tok, out_tok = estimate_tokens(d)
        cost = pricing.cost(in_tok, out_tok)
        out.total_usd            += cost
        out.total_input_tokens   += in_tok
        out.total_output_tokens  += out_tok
        out.decision_count       += 1
        out.by_backend[picked]    = out.by_backend.get(picked, 0.0) + cost
        out.by_family[pricing.family] = out.by_family.get(pricing.family, 0.0) + cost
    return out


def counterfactual_cost(
    decisions: list[DecisionRecord],
    baseline_model: str,
) -> float:
    """Cost if EVERY decision had gone to `baseline_model` instead.

    Token estimation still uses the original decision's latency-derived
    output count, which is a slight inaccuracy: a different model would
    likely emit a different number of tokens. But for headline-level
    Stripe-pitch numbers ("Claude would've cost X") this approximation
    is conservative and honest enough.
    """
    baseline = _resolve_pricing(baseline_model)
    total = 0.0
    for d in decisions:
        in_tok, out_tok = estimate_tokens(d)
        total += baseline.cost(in_tok, out_tok)
    return total


@dataclass
class CostSummary:
    """Top-card summary for dashboard rendering."""

    actual_usd:                float
    counterfactual_claude_usd: float
    counterfactual_opus_usd:   float
    savings_vs_claude_usd:     float
    savings_vs_claude_pct:     float
    savings_vs_opus_usd:       float
    savings_vs_opus_pct:       float
    decision_count:            int
    input_tokens:              int
    output_tokens:             int
    by_family:                 dict[str, float]
    by_backend:                dict[str, float]
    estimated:                 bool


def cost_summary(decisions: list[DecisionRecord]) -> CostSummary:
    """Compute the headline cost numbers: actual + 2 counterfactuals."""
    if not decisions:
        return CostSummary(
            actual_usd=0.0,
            counterfactual_claude_usd=0.0,
            counterfactual_opus_usd=0.0,
            savings_vs_claude_usd=0.0,
            savings_vs_claude_pct=0.0,
            savings_vs_opus_usd=0.0,
            savings_vs_opus_pct=0.0,
            decision_count=0,
            input_tokens=0,
            output_tokens=0,
            by_family={},
            by_backend={},
            estimated=True,
        )
    actual = compute_actual_cost(decisions)
    cf_claude = counterfactual_cost(decisions, "claude-sonnet-4-7")
    cf_opus   = counterfactual_cost(decisions, "claude-opus-4-7")
    # All estimated until any decision has real tokens — we mark estimated=True
    # whenever ANY row was estimated. For now: always True.
    estimated = any(
        not isinstance((d.get("tokens") or {}).get("output"), (int, float))
        for d in decisions
    )
    sav_claude = max(0.0, cf_claude - actual.total_usd)
    sav_opus   = max(0.0, cf_opus   - actual.total_usd)
    return CostSummary(
        actual_usd                = round(actual.total_usd, 6),
        counterfactual_claude_usd = round(cf_claude, 6),
        counterfactual_opus_usd   = round(cf_opus, 6),
        savings_vs_claude_usd     = round(sav_claude, 6),
        savings_vs_claude_pct     = round(100 * sav_claude / cf_claude, 2) if cf_claude > 0 else 0.0,
        savings_vs_opus_usd       = round(sav_opus, 6),
        savings_vs_opus_pct       = round(100 * sav_opus / cf_opus, 2) if cf_opus > 0 else 0.0,
        decision_count            = actual.decision_count,
        input_tokens              = actual.total_input_tokens,
        output_tokens             = actual.total_output_tokens,
        by_family                 = {k: round(v, 6) for k, v in actual.by_family.items()},
        by_backend                = {k: round(v, 6) for k, v in actual.by_backend.items()},
        estimated                 = estimated,
    )
