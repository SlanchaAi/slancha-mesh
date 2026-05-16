"""Tests for mesh.scripts.mesh_replay — corpus parsing + replay + output shape."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mesh.registry import HeartbeatPostRequest, MeshRegistry
from mesh.scripts.mesh_replay import (
    CorpusLine,
    _prompt_hash,
    iter_corpus,
    replay_corpus,
    replay_one,
)
from mesh.select import ClassifierSignals
from mesh.tests.conftest import make_heartbeat


# ---------------------------------------------------------------------------
# Corpus parsing
# ---------------------------------------------------------------------------


def _write_corpus(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def test_iter_corpus_basic(tmp_path):
    corpus_path = tmp_path / "c.jsonl"
    _write_corpus(corpus_path, [
        {"prompt_id": "p1", "prompt_text": "hi", "signals": {"domain": "code", "difficulty": "easy"}},
        {"prompt_id": "p2", "signals": {"domain": "math", "difficulty": "hard", "route_class": "batch"}},
    ])
    lines = list(iter_corpus(corpus_path))
    assert len(lines) == 2
    assert lines[0].prompt_id == "p1"
    assert lines[0].signals.domain == "code"
    assert lines[0].signals.route_class == "standard"  # default
    assert lines[1].signals.route_class == "batch"


def test_iter_corpus_missing_required_fields(tmp_path):
    corpus_path = tmp_path / "c.jsonl"
    _write_corpus(corpus_path, [{"prompt_id": "p1", "signals": {"domain": "code"}}])  # no difficulty
    with pytest.raises(ValueError, match="signals must include domain"):
        list(iter_corpus(corpus_path))


def test_iter_corpus_skips_blank_lines(tmp_path):
    corpus_path = tmp_path / "c.jsonl"
    corpus_path.write_text(
        '\n'
        '{"prompt_id": "p1", "signals": {"domain": "code", "difficulty": "easy"}}\n'
        '\n\n'
        '{"prompt_id": "p2", "signals": {"domain": "math", "difficulty": "hard"}}\n'
    )
    lines = list(iter_corpus(corpus_path))
    assert [ln.prompt_id for ln in lines] == ["p1", "p2"]


def test_iter_corpus_invalid_json_raises(tmp_path):
    corpus_path = tmp_path / "c.jsonl"
    corpus_path.write_text("{not json}\n")
    with pytest.raises(ValueError, match="line 1: invalid JSON"):
        list(iter_corpus(corpus_path))


# ---------------------------------------------------------------------------
# Prompt-hash
# ---------------------------------------------------------------------------


def test_prompt_hash_none():
    assert _prompt_hash(None) is None


def test_prompt_hash_deterministic():
    h1 = _prompt_hash("hello world")
    h2 = _prompt_hash("hello world")
    assert h1 == h2
    assert h1.startswith("sha256:")
    assert _prompt_hash("hello world!") != h1


# ---------------------------------------------------------------------------
# replay_one — single-prompt routing
# ---------------------------------------------------------------------------


def test_replay_one_mesh_hit(spark_node, catalog, fresh_now):
    reg = MeshRegistry(catalog=catalog)
    hb = make_heartbeat(spark_node, fresh_now, ["qwen3-coder-30b-a3b-fp8"], catalog)
    reg.record_heartbeat(HeartbeatPostRequest(heartbeat=hb, node_url="http://spark:8001/v1"))
    snap = reg.snapshot(now=fresh_now)

    line = CorpusLine(
        prompt_id="p1",
        prompt_text="write a python sorter",
        signals=ClassifierSignals(domain="code", difficulty="medium"),
    )
    rec = replay_one(line, snap, cloud_fallback="claude-sonnet-4-7")
    assert rec["prompt_id"] == "p1"
    assert rec["prompt_hash"].startswith("sha256:")
    assert rec["signals"]["domain"] == "code"
    assert rec["decision"]["mesh_hit"] is True
    assert rec["decision"]["chosen_node"] == spark_node.node_id
    assert rec["decision"]["chosen_specialist"] == "qwen3-coder-30b-a3b-fp8"
    assert rec["decision"]["vs_cloud_baseline_cost"] == 0.0


def test_replay_one_cloud_fallback(catalog, fresh_now):
    """Empty registry → no mesh route → cloud fallback."""
    reg = MeshRegistry(catalog=catalog)
    snap = reg.snapshot(now=fresh_now)

    line = CorpusLine(
        prompt_id="p1",
        prompt_text=None,
        signals=ClassifierSignals(domain="code", difficulty="hard"),
    )
    rec = replay_one(line, snap, cloud_fallback="claude-sonnet-4-7")
    assert rec["decision"]["mesh_hit"] is False
    assert rec["decision"]["chosen_node"] is None
    assert rec["decision"]["chosen_specialist"] is None
    assert rec["decision"]["model"] == "claude-sonnet-4-7"
    assert rec["prompt_hash"] is None  # no prompt_text


# ---------------------------------------------------------------------------
# replay_corpus — end-to-end with monkeypatched fetch
# ---------------------------------------------------------------------------


def test_replay_corpus_end_to_end(tmp_path, monkeypatch, spark_node, catalog, fresh_now):
    # Build a real snapshot from a populated registry
    reg = MeshRegistry(catalog=catalog)
    hb = make_heartbeat(spark_node, fresh_now, ["qwen3-coder-30b-a3b-fp8"], catalog)
    reg.record_heartbeat(HeartbeatPostRequest(heartbeat=hb, node_url="http://spark:8001/v1"))
    snap = reg.snapshot(now=fresh_now)

    # Bypass httpx with a stub fetcher
    def stub_fetch(base_url, token=None, timeout=10.0):
        return snap

    monkeypatch.setattr("mesh.scripts.mesh_replay.fetch_snapshot", stub_fetch)

    corpus_path = tmp_path / "corpus.jsonl"
    _write_corpus(corpus_path, [
        {"prompt_id": "p1", "prompt_text": "hi", "signals": {"domain": "code", "difficulty": "medium"}},
        {"prompt_id": "p2", "signals": {"domain": "knitting", "difficulty": "hard"}},  # no mesh match → cloud
        {"prompt_id": "p3", "signals": {"domain": "code", "difficulty": "easy"}},
    ])
    output_path = tmp_path / "out.jsonl"
    counters = replay_corpus(
        corpus_path=corpus_path,
        output_path=output_path,
        base_url="http://stub",
        token=None,
    )
    assert counters["processed"] == 3
    assert counters["mesh_hits"] == 2  # p1 + p3 (code), p2 falls through
    assert counters["cloud_fallbacks"] == 1

    lines = [json.loads(line) for line in output_path.read_text().splitlines() if line.strip()]
    assert len(lines) == 3
    assert {ln["prompt_id"] for ln in lines} == {"p1", "p2", "p3"}
    by_id = {ln["prompt_id"]: ln for ln in lines}
    assert by_id["p1"]["decision"]["mesh_hit"] is True
    assert by_id["p2"]["decision"]["mesh_hit"] is False
    assert by_id["p3"]["decision"]["mesh_hit"] is True


def test_replay_corpus_snapshot_refresh(tmp_path, monkeypatch, spark_node, catalog, fresh_now):
    """snapshot_refresh_every triggers fetch on the right cadence."""
    reg = MeshRegistry(catalog=catalog)
    hb = make_heartbeat(spark_node, fresh_now, ["qwen3-coder-30b-a3b-fp8"], catalog)
    reg.record_heartbeat(HeartbeatPostRequest(heartbeat=hb, node_url="http://spark:8001/v1"))
    snap = reg.snapshot(now=fresh_now)

    fetch_count = {"n": 0}

    def counting_fetch(base_url, token=None, timeout=10.0):
        fetch_count["n"] += 1
        return snap

    monkeypatch.setattr("mesh.scripts.mesh_replay.fetch_snapshot", counting_fetch)

    corpus_path = tmp_path / "corpus.jsonl"
    _write_corpus(corpus_path, [
        {"prompt_id": f"p{i}", "signals": {"domain": "code", "difficulty": "medium"}}
        for i in range(5)
    ])
    output_path = tmp_path / "out.jsonl"
    replay_corpus(
        corpus_path=corpus_path,
        output_path=output_path,
        base_url="http://stub",
        snapshot_refresh_every=3,
    )
    # With refresh_every=3, processed=0,1,2,3,4 → fetches at 0, 3 (processed % 3 == 0)
    # Initial None → fetch (processed=0). Then processed=3 triggers fetch again.
    assert fetch_count["n"] == 2


# ---------------------------------------------------------------------------
# CI smoke — 5-prompt run representative of the live corpus shapes
# ---------------------------------------------------------------------------


def test_smoke_five_prompts_in_under_one_second(
    tmp_path, monkeypatch, spark_node, mac_mini_node, catalog, fresh_now
):
    """5-prompt CI smoke: simulates the spark-side workflow end-to-end.

    Corpus mirrors the shape spark's preclassify_corpus.py emits — id
    prefixes (math500, mbpp, gsm8k, hellaswag, mmlu) with corresponding
    signals. Registry has both spark (code) and mac-mini (math) loaded.
    Asserts decision shape + total wall under 1s (CI cost-cap).
    """
    import time

    reg = MeshRegistry(catalog=catalog)
    reg.record_heartbeat(
        HeartbeatPostRequest(
            heartbeat=make_heartbeat(spark_node, fresh_now, ["qwen3-coder-30b-a3b-fp8"], catalog),
            node_url="http://spark:8001/v1",
        )
    )
    reg.record_heartbeat(
        HeartbeatPostRequest(
            heartbeat=make_heartbeat(mac_mini_node, fresh_now, ["qwen3-math-7b-q4"], catalog),
            node_url="http://mac-mini:8001/v1",
        )
    )
    snap = reg.snapshot(now=fresh_now)

    monkeypatch.setattr(
        "mesh.scripts.mesh_replay.fetch_snapshot",
        lambda *a, **k: snap,
    )

    corpus_path = tmp_path / "smoke.jsonl"
    _write_corpus(corpus_path, [
        {"prompt_id": "math500-001", "signals": {"domain": "math", "difficulty": "hard"}},
        {"prompt_id": "mbpp-042",    "signals": {"domain": "code", "difficulty": "medium"}},
        {"prompt_id": "gsm8k-100",   "signals": {"domain": "math", "difficulty": "easy"}},
        {"prompt_id": "hellaswag-7", "signals": {"domain": "general", "difficulty": "easy"}},
        {"prompt_id": "mmlu-cs-3",   "signals": {"domain": "code", "difficulty": "hard"}},
    ])
    output_path = tmp_path / "smoke_out.jsonl"

    t0 = time.time()
    counters = replay_corpus(
        corpus_path=corpus_path,
        output_path=output_path,
        base_url="http://stub",
    )
    elapsed = time.time() - t0

    assert counters["processed"] == 5
    assert elapsed < 1.0, f"smoke run too slow: {elapsed:.3f}s (CI threshold = 1.0s)"

    decisions = [json.loads(line) for line in output_path.read_text().splitlines() if line.strip()]
    assert len(decisions) == 5

    # Every decision has the contract fields
    for d in decisions:
        assert "prompt_id" in d
        assert "decision" in d
        dec = d["decision"]
        for field in ("chosen_specialist", "chosen_node", "node_url", "model",
                      "reason", "queue_ms", "fallback_chain", "mesh_hit",
                      "vs_cloud_baseline_cost"):
            assert field in dec, f"decision missing field {field!r}: {dec}"
        # mesh_hit is a bool, not None
        assert isinstance(dec["mesh_hit"], bool)
        # fallback_chain is a list of [model_id, node_id|None] pairs
        assert isinstance(dec["fallback_chain"], list)

    # Sanity: math-domain prompts should route to mac-mini (math specialist),
    # code-domain to spark (code specialist). Both nodes were heartbeated.
    by_id = {d["prompt_id"]: d for d in decisions}
    assert by_id["math500-001"]["decision"]["chosen_node"] == mac_mini_node.node_id
    assert by_id["mbpp-042"]["decision"]["chosen_node"] == spark_node.node_id
    assert by_id["mmlu-cs-3"]["decision"]["chosen_node"] == spark_node.node_id
