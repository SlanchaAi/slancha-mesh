"""Tests for mesh.scripts.build_corpus_v3_1 — pure-function helpers.

HF-streaming code paths (supplement_undercount) are network-bound and
covered via smoke runs in CI. Tests below cover the local logic:
classified ingest + merge, target-distribution rebalance, dist-string parsing.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mesh.scripts.build_corpus_v3_1 import (
    DEFAULT_TARGET_DISTRIBUTION,
    _parse_dist,
    iter_jsonl,
    load_classified_index,
    merge_with_classified,
    rebalance_to_target,
)


# ---------------------------------------------------------------------------
# JSONL I/O
# ---------------------------------------------------------------------------


def test_iter_jsonl_skips_blanks(tmp_path: Path):
    p = tmp_path / "x.jsonl"
    p.write_text("\n" + json.dumps({"a": 1}) + "\n\n" + json.dumps({"b": 2}) + "\n", encoding="utf-8")
    rows = list(iter_jsonl(p))
    assert rows == [{"a": 1}, {"b": 2}]


def test_iter_jsonl_reports_line_on_bad_json(tmp_path: Path):
    p = tmp_path / "x.jsonl"
    p.write_text('{"good":1}\n{bad}\n', encoding="utf-8")
    it = iter_jsonl(p)
    next(it)  # first row OK
    with pytest.raises(ValueError, match="line 2"):
        next(it)


# ---------------------------------------------------------------------------
# Classified ingest
# ---------------------------------------------------------------------------


def test_load_classified_accepts_signals_key(tmp_path: Path):
    p = tmp_path / "c.jsonl"
    p.write_text(
        json.dumps({"prompt_id": "a", "signals": {"domain": "code"}}) + "\n", encoding="utf-8",
    )
    idx = load_classified_index(p)
    assert idx == {"a": {"domain": "code"}}


def test_load_classified_accepts_signals_mmbert_key(tmp_path: Path):
    p = tmp_path / "c.jsonl"
    p.write_text(
        json.dumps({"prompt_id": "a", "signals_mmbert": {"domain": "math"}}) + "\n",
        encoding="utf-8",
    )
    idx = load_classified_index(p)
    assert idx == {"a": {"domain": "math"}}


def test_load_classified_skips_rows_missing_id_or_signals(tmp_path: Path):
    p = tmp_path / "c.jsonl"
    p.write_text(
        json.dumps({"prompt_id": "a"}) + "\n" +
        json.dumps({"signals": {"domain": "code"}}) + "\n" +
        json.dumps({"prompt_id": "b", "signals": {"domain": "math"}}) + "\n",
        encoding="utf-8",
    )
    idx = load_classified_index(p)
    assert list(idx.keys()) == ["b"]


# ---------------------------------------------------------------------------
# Merge — overlay + provenance tagging
# ---------------------------------------------------------------------------


def test_merge_translates_mmbert_signals():
    """mmbert academic-taxonomy signals (with domain_confidence) get
    translated to the routing taxonomy. computer science → code."""
    v3 = [
        {"prompt_id": "a", "signals": {"domain": "general", "language": "en"}},
        {"prompt_id": "b", "signals": {"domain": "general"}},
    ]
    cls = {"a": {"domain": "computer science", "domain_confidence": 0.95,
                 "language": "en", "difficulty": "medium"}}
    merged, n_overlaid = merge_with_classified(v3, cls)
    assert n_overlaid == 1
    # a got mmbert overlay → translated domain "code" + translated provenance
    assert merged[0]["signals"]["domain"] == "code"
    assert merged[0]["signals"]["classifier"] == "mmbert-6h-translated"
    # Original mmbert academic-domain preserved for posterity
    assert merged[0]["signals"]["mmbert_domain"] == "computer science"
    # b keeps heuristic + heuristic-v3 tag
    assert merged[1]["signals"]["domain"] == "general"
    assert merged[1]["signals"]["classifier"] == "heuristic-v3"


def test_merge_translates_non_english_to_multilingual():
    """mmbert.language != 'en' overrides domain to multilingual."""
    v3 = [{"prompt_id": "a", "signals": {"domain": "general"}}]
    cls = {"a": {"domain": "history", "domain_confidence": 0.8,
                 "language": "de", "language_confidence": 0.9}}
    merged, _ = merge_with_classified(v3, cls)
    assert merged[0]["signals"]["domain"] == "multilingual"
    # mmbert provenance survives
    assert merged[0]["signals"]["mmbert_language"] == "de"


def test_merge_preserves_non_signal_fields():
    v3 = [
        {"prompt_id": "a", "prompt_text": "hi", "source": "x",
         "signals": {"domain": "general"}},
    ]
    cls = {"a": {"domain": "code"}}
    merged, _ = merge_with_classified(v3, cls)
    assert merged[0]["prompt_text"] == "hi"
    assert merged[0]["source"] == "x"


def test_merge_with_no_classified_tags_heuristic():
    v3 = [{"prompt_id": "a", "signals": {"domain": "general"}}]
    merged, n_overlaid = merge_with_classified(v3, {})
    assert n_overlaid == 0
    assert merged[0]["signals"]["classifier"] == "heuristic-v3"


# ---------------------------------------------------------------------------
# Rebalance — target distribution + undercount detection
# ---------------------------------------------------------------------------


def _make_records(domain: str, n: int) -> list[dict]:
    return [
        {"prompt_id": f"{domain}-{i}", "signals": {"domain": domain}} for i in range(n)
    ]


def test_rebalance_exact_quotas_when_full():
    records = (
        _make_records("code", 100) +
        _make_records("general", 100) +
        _make_records("math", 100)
    )
    target = {"code": 0.5, "general": 0.3, "math": 0.2}
    selected, undercount = rebalance_to_target(records, target, total_target=100, seed=42)
    counts = {}
    for r in selected:
        d = r["signals"]["domain"]
        counts[d] = counts.get(d, 0) + 1
    assert counts == {"code": 50, "general": 30, "math": 20}
    assert undercount == {}


def test_rebalance_detects_undercount():
    records = _make_records("code", 10) + _make_records("general", 100)
    target = {"code": 0.5, "general": 0.5}
    selected, undercount = rebalance_to_target(records, target, total_target=100, seed=42)
    # code wanted 50, only 10 available
    assert undercount == {"code": 40}
    counts = {}
    for r in selected:
        d = r["signals"]["domain"]
        counts[d] = counts.get(d, 0) + 1
    assert counts["code"] == 10
    assert counts["general"] == 50


def test_rebalance_drops_domains_not_in_target():
    records = _make_records("code", 100) + _make_records("creative", 100)
    target = {"code": 1.0}
    selected, undercount = rebalance_to_target(records, target, total_target=100, seed=42)
    assert all(r["signals"]["domain"] == "code" for r in selected)
    assert len(selected) == 100
    assert undercount == {}


def test_rebalance_seed_is_deterministic():
    records = _make_records("code", 1000)
    target = {"code": 1.0}
    a, _ = rebalance_to_target(records, target, total_target=100, seed=42)
    b, _ = rebalance_to_target(records, target, total_target=100, seed=42)
    # Same seed → same selection
    assert [r["prompt_id"] for r in a] == [r["prompt_id"] for r in b]


def test_rebalance_different_seeds_differ():
    records = _make_records("code", 1000)
    target = {"code": 1.0}
    a, _ = rebalance_to_target(records, target, total_target=100, seed=42)
    b, _ = rebalance_to_target(records, target, total_target=100, seed=43)
    assert [r["prompt_id"] for r in a] != [r["prompt_id"] for r in b]


# ---------------------------------------------------------------------------
# Distribution string parsing
# ---------------------------------------------------------------------------


def test_parse_dist_accepts_0_to_1():
    d = _parse_dist("code:0.5,general:0.5")
    assert d["code"] == pytest.approx(0.5)
    assert d["general"] == pytest.approx(0.5)


def test_parse_dist_accepts_0_to_100():
    d = _parse_dist("code:50,general:50")
    assert d["code"] == pytest.approx(0.5)
    assert d["general"] == pytest.approx(0.5)


def test_parse_dist_renormalizes():
    """Non-unit-sum input gets normalized to sum to 1.0."""
    d = _parse_dist("code:60,general:60")  # sum 120 → normalized
    assert sum(d.values()) == pytest.approx(1.0)
    assert d["code"] == pytest.approx(0.5)
    assert d["general"] == pytest.approx(0.5)


def test_parse_dist_skips_malformed_chunks():
    d = _parse_dist("code:50,malformed,general:50")
    assert "malformed" not in d
    assert set(d.keys()) == {"code", "general"}


# ---------------------------------------------------------------------------
# Default distribution sanity
# ---------------------------------------------------------------------------


def test_default_target_distribution_sums_to_1():
    assert sum(DEFAULT_TARGET_DISTRIBUTION.values()) == pytest.approx(1.0)


def test_default_distribution_covers_required_domains():
    """Operator's preset must cover all named buckets."""
    required = {"code", "general", "reasoning", "math", "multilingual", "creative", "tool-use"}
    assert set(DEFAULT_TARGET_DISTRIBUTION.keys()) == required
