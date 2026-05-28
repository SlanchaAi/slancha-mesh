"""Tests for mesh.eval.gate (the promotion gate)."""

from __future__ import annotations

import json
from pathlib import Path

from mesh.eval.gate import (
    DEFAULT_MEAN_SCORE_DELTA,
    GateThresholds,
    PromotionVerdict,
    append_verdict,
    decide,
)


def _row(
    router_version: str,
    mean_score: float,
    per_domain: dict[str, float] | None = None,
    n_eval: int = 500,
    judge_model: str = "qwen3-coder-30b",
    ts: str = "2026-05-20T00:00:00Z",
) -> dict:
    return {
        "ts":              ts,
        "router_version":  router_version,
        "n_eval":          n_eval,
        "judge_model":     judge_model,
        "mean_score":      mean_score,
        "per_domain_mean": per_domain or {},
    }


def test_accepts_clean_improvement():
    champ = _row("v1", 3.50, {"code": 3.5, "general": 3.5})
    chall = _row("v2", 3.80, {"code": 3.8, "general": 3.8})
    v = decide(champ, chall)
    assert v.accept is True
    assert v.reject_reasons == ()
    assert v.mean_delta == pytest.approx(0.30)
    assert v.per_domain_deltas == pytest.approx({"code": 0.3, "general": 0.3})


def test_rejects_insufficient_mean_lift():
    champ = _row("v1", 3.50)
    chall = _row("v2", 3.50 + DEFAULT_MEAN_SCORE_DELTA / 2)  # below required
    v = decide(champ, chall)
    assert v.accept is False
    assert any("mean_delta" in r for r in v.reject_reasons)


def test_rejects_per_domain_regression_even_with_mean_lift():
    """Mean rises but one domain collapses → must reject."""
    champ = _row("v1", 3.50, {"code": 3.8, "general": 3.2})
    # +0.40 on general, -0.40 on code; overall mean rises
    chall = _row("v2", 3.85, {"code": 3.4, "general": 3.6})
    v = decide(champ, chall)
    assert v.accept is False
    assert any("regression on 'code'" in r for r in v.reject_reasons)


def test_rejects_small_per_domain_dip_within_tolerance_is_ok():
    """A dip smaller than per_domain_max_regression is allowed."""
    champ = _row("v1", 3.50, {"code": 3.8, "general": 3.2})
    chall = _row("v2", 3.80, {"code": 3.75, "general": 3.6})  # code dips 0.05
    v = decide(champ, chall)
    assert v.accept is True


def test_rejects_insufficient_sample_size():
    champ = _row("v1", 3.50, n_eval=10)
    chall = _row("v2", 3.90, n_eval=600)
    v = decide(champ, chall)
    assert v.accept is False
    assert any("champion n_eval 10" in r for r in v.reject_reasons)


def test_rejects_judge_mismatch_by_default():
    champ = _row("v1", 3.50, judge_model="judge-a")
    chall = _row("v2", 3.90, judge_model="judge-b")
    v = decide(champ, chall)
    assert v.accept is False
    assert any("judge_model mismatch" in r for r in v.reject_reasons)


def test_allows_judge_mismatch_when_opted_in():
    champ = _row("v1", 3.50, judge_model="judge-a")
    chall = _row("v2", 3.90, judge_model="judge-b")
    v = decide(champ, chall, GateThresholds(require_judge_match=False))
    assert v.accept is True


def test_skips_domains_not_in_both_rows():
    """A new domain in challenger doesn't count as a regression."""
    champ = _row("v1", 3.50, {"code": 3.5})
    chall = _row("v2", 3.80, {"code": 3.8, "new-domain": 0.0})
    v = decide(champ, chall)
    assert v.accept is True
    assert "new-domain" not in v.per_domain_deltas


def test_verdict_carries_audit_metadata():
    champ = _row("v1", 3.50, {"code": 3.5}, judge_model="j")
    chall = _row("v2", 3.80, {"code": 3.8}, judge_model="j")
    v = decide(champ, chall)
    assert v.champion_version == "v1"
    assert v.challenger_version == "v2"
    assert v.thresholds["mean_score_delta"] == DEFAULT_MEAN_SCORE_DELTA
    assert v.decided_at.endswith("Z")


def test_append_verdict_writes_jsonl(tmp_path: Path):
    out = tmp_path / "promotions.jsonl"
    v = PromotionVerdict(
        accept=True, reject_reasons=(), mean_delta=0.3,
        per_domain_deltas={"code": 0.3},
        champion_version="v1", challenger_version="v2",
    )
    append_verdict(out, v)
    append_verdict(out, v)  # idempotent at the row level (append-only)
    lines = out.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    row = json.loads(lines[0])
    assert row["accept"] is True
    assert row["reject_reasons"] == []  # tuple → list


# pytest import is at the bottom so the helpers above stay near the top
import pytest  # noqa: E402
