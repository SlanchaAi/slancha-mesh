"""Tests for mesh.curation — difficulty-ranked holdout + synthetic gap-fill.

Pure Python, no heavy deps: scorers and synthetic generators are injected
fakes, embeddings are tiny float lists — same discipline as test_generator.
"""

from __future__ import annotations

import base64
import json
import struct

import pytest

from mesh.curation import (
    CurationResult,
    coverage_bands,
    curate_cluster,
    default_difficulty_scorer,
    fill_gaps,
    holdout_content_ref,
    is_synthetic,
    rank_by_difficulty,
    split_hard_holdout,
    trace_embedding,
    write_curation,
)
from mesh.generator import _Cluster, build_spec, centroid_ref, generate
from mesh.loop_runner import read_queue


# ──────────────────────────── fixtures / helpers ─────────────────────────────


def _trace(i: int, *, score: float | None = None, emb: list[float] | None = None,
           prompt: str | None = None, source: str | None = None) -> dict:
    t: dict = {"id": f"t{i}", "prompt": prompt or f"prompt number {i}"}
    if score is not None:
        t["judge_score"] = score
    if emb is not None:
        t["embedding"] = emb
    if source is not None:
        t["source"] = source
    return t


def _cluster_traces(n: int = 100) -> list[dict]:
    """n traces: ids t0..t(n-1); t0..t9 are HARD (low judge score, far from
    centroid), the rest easy (high score, on-centroid)."""
    out = []
    for i in range(n):
        hard = i < 10
        out.append(
            _trace(
                i,
                score=2.0 if hard else 9.0,
                emb=[0.0, 1.0] if hard else [1.0, 0.0],
            )
        )
    return out


CENTROID = [1.0, 0.0]


# ─────────────────────────── trace field access ──────────────────────────────


def test_trace_embedding_from_float_list():
    assert trace_embedding({"embedding": [1, 2.5]}) == [1.0, 2.5]


def test_trace_embedding_from_b64_float32_le():
    raw = struct.pack("<3f", 0.5, -1.0, 2.0)
    t = {"embedding_b64": base64.b64encode(raw).decode()}
    assert trace_embedding(t) == [0.5, -1.0, 2.0]


def test_trace_embedding_garbage_is_none():
    assert trace_embedding({"embedding_b64": "!!!not-base64!!!"}) is None
    assert trace_embedding({}) is None


# ───────────────────────────── difficulty ranking ────────────────────────────


def test_default_scorer_far_and_low_scored_is_hardest():
    s = default_difficulty_scorer()
    hard = _trace(0, score=2.0, emb=[0.0, 1.0])   # orthogonal + low grade
    easy = _trace(1, score=9.0, emb=[1.0, 0.0])   # on-centroid + high grade
    assert s(hard, CENTROID) > s(easy, CENTROID)


def test_default_scorer_missing_signals_is_neutral():
    s = default_difficulty_scorer()
    assert s({"prompt": "x"}, None) == pytest.approx(0.5)


def test_rank_is_deterministic_across_input_order():
    traces = _cluster_traces(40)
    r1 = rank_by_difficulty(traces, CENTROID)
    # same traces, reversed input order → same fingerprint sequence
    rev = list(reversed(traces))
    r2 = rank_by_difficulty(rev, CENTROID)
    assert [r.fingerprint for r in r1] == [r.fingerprint for r in r2]


def test_rank_hardest_first():
    traces = _cluster_traces(40)
    ranked = rank_by_difficulty(traces, CENTROID)
    top10 = {traces[r.index]["id"] for r in ranked[:10]}
    assert top10 == {f"t{i}" for i in range(10)}


# ─────────────────────────── hard-holdout selection ──────────────────────────


def test_split_selects_hardest_as_holdout():
    traces = _cluster_traces(100)
    ranked = rank_by_difficulty(traces, CENTROID)
    split = split_hard_holdout(traces, ranked)
    held = {traces[i]["id"] for i in split.holdout_indices}
    # 20% of 100 = 20 → the 10 hard ones must all be in the exam
    assert {f"t{i}" for i in range(10)} <= held
    assert len(split.holdout_indices) == 20
    assert len(split.train_indices) == 80


def test_split_refuses_tiny_cluster():
    traces = _cluster_traces(30)  # < min_holdout + min_train
    ranked = rank_by_difficulty(traces, CENTROID)
    with pytest.raises(ValueError, match="too small"):
        split_hard_holdout(traces, ranked)


def test_synthetic_rows_never_enter_holdout():
    traces = _cluster_traces(100)
    # make the 30 HARDEST-looking rows synthetic — they must all be skipped
    for i in range(30):
        traces[i]["source"] = "sdg"
        traces[i]["judge_score"] = 1.0
        traces[i]["embedding"] = [0.0, 1.0]
    ranked = rank_by_difficulty(traces, CENTROID)
    split = split_hard_holdout(traces, ranked)
    assert all(not is_synthetic(traces[i]) for i in split.holdout_indices)
    assert split.skipped_synthetic_for_holdout >= 20


def test_duplicate_train_prompts_dropped():
    traces = _cluster_traces(100)
    # t50 exact-duplicates hard t0's prompt → t0 lands in holdout, t50 must drop
    traces[50]["prompt"] = traces[0]["prompt"]
    ranked = rank_by_difficulty(traces, CENTROID)
    split = split_hard_holdout(traces, ranked)
    held_fps = {traces[i]["prompt"] for i in split.holdout_indices}
    train_prompts = [traces[i]["prompt"] for i in split.train_indices]
    assert not (held_fps & set(train_prompts))
    assert split.dropped_duplicate_train >= 1


def test_holdout_content_ref_is_order_independent_and_content_keyed():
    rows = [{"id": "a", "prompt": "x"}, {"id": "b", "prompt": "y"}]
    assert holdout_content_ref(rows) == holdout_content_ref(list(reversed(rows)))
    assert holdout_content_ref(rows).startswith("frozen://sha256:")
    changed = [{"id": "a", "prompt": "x"}, {"id": "b", "prompt": "z"}]
    assert holdout_content_ref(rows) != holdout_content_ref(changed)


# ───────────────────────── coverage bands + gap fill ─────────────────────────


def test_coverage_bands_partition_train_pool():
    traces = _cluster_traces(100)
    idx = list(range(100))
    bands = coverage_bands(traces, idx, CENTROID, n_bands=4)
    assert sum(len(b.indices) for b in bands) == 100
    # hard rows (distance 1.0) land in the top band, easy (0.0) in the bottom
    assert len(bands[-1].indices) == 10
    assert len(bands[0].indices) == 90


def test_fill_gaps_stamps_provenance_and_respects_holdout():
    traces = _cluster_traces(100)
    idx = list(range(100))
    bands = coverage_bands(traces, idx, CENTROID, n_bands=4)
    held_fp = {"deadbeef"}  # no real collision; checked separately below

    def fake_sdg(exemplars: list[dict], n: int) -> list[dict]:
        return [{"prompt": f"synthetic {exemplars[0]['id']} {k}"} for k in range(n)]

    rows, stats = fill_gaps(traces, bands, fake_sdg, holdout_fingerprints=held_fp,
                            sdg_model="kimi-k2")
    assert rows, "sparse top band should have been filled"
    for r in rows:
        assert r["source"] == "sdg"
        assert r["sdg_model"] == "kimi-k2"
        assert "sdg_exemplar_ids" in r and r["sdg_exemplar_ids"]
    assert stats["produced"] == len(rows)


def test_fill_gaps_drops_holdout_duplicates():
    traces = _cluster_traces(100)
    bands = coverage_bands(traces, list(range(100)), CENTROID, n_bands=4)

    leak = "the exam question"
    import hashlib
    fp = hashlib.sha256(" ".join(leak.split()).lower().encode()).hexdigest()

    def leaky_sdg(exemplars: list[dict], n: int) -> list[dict]:
        return [{"prompt": leak} for _ in range(n)]

    rows, stats = fill_gaps(traces, bands, leaky_sdg, holdout_fingerprints={fp})
    assert rows == []
    assert stats["dropped_holdout_dup"] > 0


# ───────────────────────────── curate_cluster (e2e) ──────────────────────────


def test_curate_cluster_end_to_end():
    traces = _cluster_traces(100)
    # give the train side a sparse mid-distance region (a coverage gap):
    # most easy rows sit on the centroid, 15 sit at ~45° — the exam takes the
    # hardest 20 (10 hard + 10 of these outliers), the surviving 5 outliers
    # leave a sparse band in the train pool
    for i in range(80, 95):
        traces[i]["embedding"] = [0.7, 0.7]

    def fake_sdg(exemplars: list[dict], n: int) -> list[dict]:
        return [{"prompt": f"syn {exemplars[0]['id']} {k}"} for k in range(n)]

    res = curate_cluster(traces, CENTROID, synthetic_generator=fake_sdg,
                         sdg_model="kimi-k2")
    assert len(res.holdout) == 20
    assert len(res.train) == 80
    assert res.synthetic  # the sparse hard band got filled
    assert res.holdout_ref.startswith("frozen://sha256:")
    m = res.manifest
    assert m["n_holdout"] == 20 and m["n_train_real"] == 80
    assert m["sdg"]["model"] == "kimi-k2"
    # exam is harder than the train pool on average
    assert m["difficulty"]["holdout_mean"] > m["difficulty"]["train_mean"]
    # no synthetic row in the exam — the hard guarantee
    assert all(not is_synthetic(r) for r in res.holdout)


def test_curate_cluster_is_deterministic():
    traces = _cluster_traces(100)
    r1 = curate_cluster(traces, CENTROID)
    r2 = curate_cluster(list(reversed(traces)), CENTROID)
    assert r1.holdout_ref == r2.holdout_ref


def test_write_curation_roundtrip(tmp_path):
    traces = _cluster_traces(100)
    res = curate_cluster(traces, CENTROID)
    manifest = write_curation(tmp_path / "c0", res)
    holdout = [json.loads(line) for line in
               (tmp_path / "c0" / "holdout.jsonl").read_text().splitlines()]
    train = [json.loads(line) for line in
             (tmp_path / "c0" / "train.jsonl").read_text().splitlines()]
    assert len(holdout) == manifest["n_holdout"]
    assert len(train) == manifest["n_train_real"] + manifest["n_train_synthetic"]
    assert json.loads((tmp_path / "c0" / "manifest.json").read_text())[
        "holdout_ref"] == res.holdout_ref


# ─────────────────────── generator integration (seam) ────────────────────────


def test_build_spec_pins_curated_holdout_ref():
    curated_ref = "frozen://sha256:" + "ab" * 32
    spec = build_spec(
        route="code", cluster_id=0, n_traces=600, drift=0.05,
        exemplar_trace_ids=["t1"], centroid=[1.0, 0.0],
        base_model_id="Qwen/Qwen3-8B",
        holdout_ref=curated_ref, curation={"n_holdout": 20},
    )
    assert spec["gate"]["holdout_ref"] == curated_ref
    # judge keyed on the FROZEN EXAM bytes, not the centroid
    assert spec["gate"]["judge_model"].endswith("@" + ("ab" * 32)[:12])
    assert spec["generator"]["curation"] == {"n_holdout": 20}
    # centroid ref unchanged — cluster identity still keys the spec id
    assert spec["generator"]["centroid_ref"] == centroid_ref([1.0, 0.0])
    assert spec["id"].endswith(centroid_ref([1.0, 0.0]).rsplit(":", 1)[-1][:12])


def test_build_spec_without_curation_unchanged():
    spec = build_spec(
        route="code", cluster_id=0, n_traces=600, drift=0.05,
        exemplar_trace_ids=["t1"], centroid=[1.0, 0.0],
        base_model_id="Qwen/Qwen3-8B",
    )
    cref = centroid_ref([1.0, 0.0])
    assert spec["gate"]["holdout_ref"] == cref
    assert "curation" not in spec["generator"]


def test_generate_with_curate_fn_seam(tmp_path):
    """End-to-end: ignited cluster → curation seam runs → spec carries the
    curated holdout_ref + manifest."""
    traces = _cluster_traces(600)
    cluster = _Cluster(route="code", cluster_id=0,
                       trace_indices=list(range(600)), centroid=CENTROID)
    queue = tmp_path / "queue.jsonl"
    drift_state = tmp_path / "drift.json"

    def curate(cluster_traces: list[dict], centroid) -> CurationResult:
        return curate_cluster(cluster_traces, centroid)

    # 3 windows to settle the drift gate, then ignite
    for _ in range(3):
        result = generate(
            traces, queue_path=queue, drift_state_path=drift_state,
            cluster_fn=lambda _t: [cluster], curate_fn=curate,
        )
    assert result.specs, "cluster should have ignited on the settled window"
    spec = result.specs[0]
    assert spec["gate"]["holdout_ref"].startswith("frozen://sha256:")
    assert spec["gate"]["holdout_ref"] != spec["generator"]["centroid_ref"]
    assert spec["generator"]["curation"]["n_holdout"] > 0
    # and it landed in the queue
    rows = read_queue(queue)
    assert any(r["id"] == spec["id"] for r in rows)
