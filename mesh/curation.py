"""Difficulty-ranked holdout curation + synthetic gap-fill — the curation stage.

The ignition stage (``mesh.generator``) finds a settled, high-volume traffic
cluster and emits a GATE-CONTRACT experiment spec. This module is the stage
between ignition and training: it decides **which rows become the frozen
holdout the gate scores against, and which rows the specialist trains on**.

The pattern (demand defines the eval, GATE-CONTRACT binding #7):

  1. **Rank every real trace in the cluster by difficulty.** The open default
     scorer blends two signals that need no extra infrastructure: distance
     from the cluster centroid (an outlier is harder than the cluster's bread
     and butter) and the graded judge-score shortfall (a prompt the frontier
     model itself scored poorly on is hard evidence of difficulty). The
     scorer is an injectable seam — deployments with richer difficulty
     models plug them in here.
  2. **The hardest rows become the holdout.** The exam is the hard tail of
     *actual usage*, not a random sample — a specialist that matches the
     teacher on the hardest real prompts has earned promotion. Selection is
     deterministic (content-hash tie-breaks), and the resulting set is
     content-hashed into a ``frozen://sha256:`` ref the spec's gate binding
     pins (so a swapped holdout = a new ref = judge-match fires).
  3. **The remaining rows train the specialist, with synthetic gap-fill.**
     Coverage gaps in the train pool (sparse distance bands of the cluster)
     are filled through the ``SyntheticGenerator`` seam — an open
     near-frontier model prompted from real exemplars. Mesh ships the seam,
     the provenance stamping, and the guards; the generator itself is
     deployment-injected.

Two guarantees are enforced, not advised:

  * **Synthetic data NEVER enters the holdout.** Holdout eligibility requires
    a real trace (``source != "sdg"``), and gap-fill runs strictly on the
    train side. The exam is real usage by construction.
  * **Holdout and train sets are disjoint** — exact-duplicate prompts are
    removed from the train pool (and from synthetic output) so the gate
    can't be aced by memorization.

Pure Python, no heavy deps: embeddings arrive as ``embedding`` (float list)
or ``embedding_b64`` (base64 float32 LE, slancha-local's wire shape) and are
decoded with ``struct``. An optional embedding near-duplicate guard exists
behind ``near_dup_cosine`` but defaults OFF — pure-Python O(H×T·d) is fine
for test scale, deployments inject a vectorized check if they want it.
"""

from __future__ import annotations

import base64
import hashlib
import json
import math
import struct
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

# ─────────────────────────────── defaults ────────────────────────────────────

HOLDOUT_FRAC = 0.2        # hardest 20% of the cluster becomes the exam …
MIN_HOLDOUT = 20          # … but never fewer than this (a 5-row exam is noise)
MAX_HOLDOUT = 500         # … and never more (eval cost is per-promotion-attempt)
MIN_TRAIN = 50            # refuse to curate a cluster that can't also train
MAX_JUDGE_SCORE = 10.0    # grading scale ceiling (mesh.grading judge_score)


# ─────────────────────────── trace field access ──────────────────────────────


def prompt_text(trace: dict[str, Any]) -> str:
    """The prompt body of a trace, tolerant of the shapes in the wild
    (slancha-local traces, corpus rows, holdout rows)."""
    for key in ("prompt_text", "prompt", "input", "text"):
        v = trace.get(key)
        if isinstance(v, str) and v:
            return v
    return json.dumps(trace, sort_keys=True, ensure_ascii=False)


def _prompt_fingerprint(trace: dict[str, Any]) -> str:
    """Normalized content hash of the prompt — the exact-duplicate key used by
    the holdout/train disjointness guard."""
    norm = " ".join(prompt_text(trace).split()).lower()
    return hashlib.sha256(norm.encode("utf-8")).hexdigest()


def trace_embedding(trace: dict[str, Any]) -> list[float] | None:
    """Decode a trace embedding: ``embedding`` (float list) or
    ``embedding_b64`` (base64-encoded float32 little-endian, the
    slancha-local wire shape). None when absent/undecodable."""
    emb = trace.get("embedding")
    if isinstance(emb, (list, tuple)) and emb:
        try:
            return [float(x) for x in emb]
        except (TypeError, ValueError):
            return None
    b64 = trace.get("embedding_b64")
    if isinstance(b64, str) and b64:
        try:
            raw = base64.b64decode(b64)
        except (ValueError, TypeError):
            return None
        if len(raw) % 4:
            return None
        return list(struct.unpack(f"<{len(raw) // 4}f", raw))
    return None


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    if len(a) != len(b) or not a:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def is_synthetic(trace: dict[str, Any]) -> bool:
    """True when a row was produced by synthetic data generation. Synthetic
    rows are train-only — they are never eligible for the holdout."""
    return trace.get("source") == "sdg"


# ─────────────────────────── the difficulty seam ─────────────────────────────


class DifficultyScorer(Protocol):
    """The difficulty seam: ``(trace, centroid) -> score`` (higher = harder).
    The default blends centroid distance and judge-score shortfall; richer
    deployment-specific difficulty models inject here."""

    def __call__(
        self, trace: dict[str, Any], centroid: Sequence[float] | None
    ) -> float: ...


def default_difficulty_scorer(
    *,
    w_distance: float = 0.5,
    w_grade: float = 0.5,
    max_judge_score: float = MAX_JUDGE_SCORE,
) -> DifficultyScorer:
    """The open difficulty default: ``w_distance·(1 − cos(emb, centroid)) +
    w_grade·(1 − judge_score/max)``. Both terms land in [0, 1].

    * Distance term: a trace far from the cluster centroid is an outlier of
      the pattern — harder than the cluster's typical case. Missing
      embedding/centroid → neutral 0.5 (unknown, not easy).
    * Grade term: a prompt the (frontier) grader scored low on is observed
      difficulty, not inferred. Missing judge_score → neutral 0.5.
    """

    def _score(trace: dict[str, Any], centroid: Sequence[float] | None) -> float:
        emb = trace_embedding(trace)
        if emb is not None and centroid is not None and len(centroid) > 0:
            dist = 1.0 - _cosine(emb, [float(x) for x in centroid])
            dist = min(max(dist, 0.0), 1.0)
        else:
            dist = 0.5
        js = trace.get("judge_score")
        if isinstance(js, (int, float)) and max_judge_score > 0:
            shortfall = 1.0 - min(max(float(js) / max_judge_score, 0.0), 1.0)
        else:
            shortfall = 0.5
        return w_distance * dist + w_grade * shortfall

    return _score


@dataclass(frozen=True)
class ScoredTrace:
    """One trace's difficulty rank entry — index into the caller's trace list,
    the score, and the deterministic tie-break key."""

    index: int
    score: float
    fingerprint: str


def rank_by_difficulty(
    traces: list[dict[str, Any]],
    centroid: Sequence[float] | None,
    scorer: DifficultyScorer | None = None,
) -> list[ScoredTrace]:
    """Score every trace and sort hardest-first. Ties break on the prompt
    fingerprint so the ranking (and therefore the holdout) is deterministic
    across runs regardless of input order."""
    s = scorer or default_difficulty_scorer()
    ranked = [
        ScoredTrace(index=i, score=float(s(t, centroid)), fingerprint=_prompt_fingerprint(t))
        for i, t in enumerate(traces)
    ]
    ranked.sort(key=lambda r: (-r.score, r.fingerprint))
    return ranked


# ───────────────────────── hard-holdout selection ────────────────────────────


@dataclass(frozen=True)
class HoldoutSplit:
    """The curated split: holdout/train index lists into the caller's traces,
    plus what the leakage guard removed (for the manifest)."""

    holdout_indices: tuple[int, ...]
    train_indices: tuple[int, ...]
    dropped_duplicate_train: int
    skipped_synthetic_for_holdout: int


def split_hard_holdout(
    traces: list[dict[str, Any]],
    ranked: list[ScoredTrace],
    *,
    holdout_frac: float = HOLDOUT_FRAC,
    min_holdout: int = MIN_HOLDOUT,
    max_holdout: int = MAX_HOLDOUT,
    min_train: int = MIN_TRAIN,
    near_dup_cosine: float | None = None,
) -> HoldoutSplit:
    """Walk the ranking hardest-first; the top slice becomes the holdout, the
    rest the train pool. Enforces the two hard guarantees:

    * synthetic rows are skipped for holdout eligibility (train-only), and
    * any train row whose prompt exact-duplicates a holdout row is dropped
      (with ``near_dup_cosine`` set, embedding near-duplicates too).

    Raises ``ValueError`` when the cluster is too small to yield both a
    meaningful exam and a train pool — curating a tiny cluster would produce
    a noise gate, better to refuse loudly.
    """
    n_real = sum(1 for t in traces if not is_synthetic(t))
    target = min(max(int(round(holdout_frac * n_real)), min_holdout), max_holdout)
    if n_real < min_holdout + min_train:
        raise ValueError(
            f"cluster too small to curate: {n_real} real trace(s), need at least "
            f"{min_holdout} (holdout) + {min_train} (train)"
        )

    holdout: list[int] = []
    holdout_fps: set[str] = set()
    holdout_embs: list[list[float]] = []
    skipped_synthetic = 0
    dropped = 0
    rest: list[ScoredTrace] = []

    for r in ranked:
        if len(holdout) < target:
            if is_synthetic(traces[r.index]):
                skipped_synthetic += 1
                rest.append(r)
                continue
            if r.fingerprint in holdout_fps:
                # an exact-dup of a row already in the exam adds no signal and
                # can't go to train either (it IS the exam) — drop, counted.
                dropped += 1
                continue
            holdout.append(r.index)
            holdout_fps.add(r.fingerprint)
            if near_dup_cosine is not None:
                emb = trace_embedding(traces[r.index])
                if emb is not None:
                    holdout_embs.append(emb)
        else:
            rest.append(r)

    train: list[int] = []
    for r in rest:
        if r.fingerprint in holdout_fps:
            dropped += 1
            continue
        if near_dup_cosine is not None and holdout_embs:
            emb = trace_embedding(traces[r.index])
            if emb is not None and any(
                _cosine(emb, h) >= near_dup_cosine for h in holdout_embs
            ):
                dropped += 1
                continue
        train.append(r.index)

    if len(train) < min_train:
        raise ValueError(
            f"train pool too small after holdout split: {len(train)} < {min_train}"
        )
    return HoldoutSplit(
        holdout_indices=tuple(holdout),
        train_indices=tuple(train),
        dropped_duplicate_train=dropped,
        skipped_synthetic_for_holdout=skipped_synthetic,
    )


def holdout_content_ref(rows: list[dict[str, Any]]) -> str:
    """Content-hash of the curated holdout rows → ``frozen://sha256:<64hex>``
    (GATE-CONTRACT: frozen refs are content-hashes). Rows are canonicalized
    (sorted keys) and order-normalized by fingerprint so the same set of rows
    always hashes equal — this ref is what pins the exam bytes in the spec."""
    canon = sorted(
        json.dumps(r, sort_keys=True, ensure_ascii=False) for r in rows
    )
    h = hashlib.sha256()
    for line in canon:
        h.update(line.encode("utf-8"))
        h.update(b"\n")
    return "frozen://sha256:" + h.hexdigest()


# ───────────────────────── synthetic gap-fill seam ───────────────────────────


class SyntheticGenerator(Protocol):
    """The SDG seam: given real exemplar rows from an under-covered region of
    the cluster and a count, return up to ``n`` new synthetic rows (each at
    least carrying a prompt field). Deployments bind this to an open
    near-frontier model; mesh stamps provenance and enforces the guards."""

    def __call__(
        self, exemplars: list[dict[str, Any]], n: int
    ) -> list[dict[str, Any]]: ...


@dataclass(frozen=True)
class CoverageBand:
    """One centroid-distance band of the train pool: [lo, hi) distance range
    and the train-pool indices that fall in it."""

    lo: float
    hi: float
    indices: tuple[int, ...]


def coverage_bands(
    traces: list[dict[str, Any]],
    train_indices: Sequence[int],
    centroid: Sequence[float] | None,
    *,
    n_bands: int = 4,
) -> list[CoverageBand]:
    """Partition the train pool into equal-width centroid-distance bands.
    Sparse bands are the cluster's coverage gaps — the regions the specialist
    would otherwise under-learn. Rows without embeddings (or no centroid) all
    land in a single full-range band (no gap signal, nothing to fill)."""
    if centroid is None or len(centroid) == 0:
        return [CoverageBand(lo=0.0, hi=1.0, indices=tuple(train_indices))]
    c = [float(x) for x in centroid]
    dists: list[tuple[int, float]] = []
    no_emb: list[int] = []
    for i in train_indices:
        emb = trace_embedding(traces[i])
        if emb is None:
            no_emb.append(i)
        else:
            dists.append((i, min(max(1.0 - _cosine(emb, c), 0.0), 1.0)))
    if not dists:
        return [CoverageBand(lo=0.0, hi=1.0, indices=tuple(train_indices))]
    lo = min(d for _, d in dists)
    hi = max(d for _, d in dists)
    if hi <= lo:  # all rows equidistant — one band
        return [CoverageBand(lo=lo, hi=hi, indices=tuple(i for i, _ in dists) + tuple(no_emb))]
    width = (hi - lo) / n_bands
    bands: list[list[int]] = [[] for _ in range(n_bands)]
    for i, d in dists:
        k = min(int((d - lo) / width), n_bands - 1)
        bands[k].append(i)
    bands[0].extend(no_emb)  # un-embeddable rows count toward the densest-by-default band
    return [
        CoverageBand(lo=lo + k * width, hi=lo + (k + 1) * width, indices=tuple(b))
        for k, b in enumerate(bands)
    ]


def fill_gaps(
    traces: list[dict[str, Any]],
    bands: list[CoverageBand],
    generator: SyntheticGenerator,
    *,
    holdout_fingerprints: set[str],
    sdg_model: str = "unknown",
    max_exemplars: int = 8,
    max_synthetic_per_band: int = 200,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Fill under-covered bands up to the median band count via the SDG seam.

    Every synthetic row is stamped with provenance (``source="sdg"``, the
    generating model, the band, the exemplar trace ids) and dropped — counted,
    not silently — when it exact-duplicates a holdout row (the exam stays
    unaceable) or duplicates an existing/just-generated train prompt.
    """
    counts = sorted(len(b.indices) for b in bands)
    median = counts[len(counts) // 2] if counts else 0
    synthetic: list[dict[str, Any]] = []
    stats = {"requested": 0, "produced": 0, "dropped_holdout_dup": 0, "dropped_train_dup": 0}
    seen_train_fps = {
        _prompt_fingerprint(traces[i]) for b in bands for i in b.indices
    }

    for band_k, band in enumerate(bands):
        want = min(median - len(band.indices), max_synthetic_per_band)
        if want <= 0:
            continue  # band already at/above median coverage
        # Seed exemplars from the band itself; an EMPTY band (a true coverage
        # hole) borrows from the nearest populated band — the generator is
        # told to extrapolate from the closest real usage it has.
        exemplar_idx: Sequence[int] = band.indices
        if not exemplar_idx:
            for off in range(1, len(bands)):
                for k2 in (band_k - off, band_k + off):
                    if 0 <= k2 < len(bands) and bands[k2].indices:
                        exemplar_idx = bands[k2].indices
                        break
                if exemplar_idx:
                    break
        if not exemplar_idx:
            continue  # no real rows anywhere — nothing to seed from
        exemplars = [traces[i] for i in exemplar_idx[:max_exemplars]]
        stats["requested"] += want
        rows = generator(exemplars, want)[:want]
        for row in rows:
            fp = _prompt_fingerprint(row)
            if fp in holdout_fingerprints:
                stats["dropped_holdout_dup"] += 1
                continue
            if fp in seen_train_fps:
                stats["dropped_train_dup"] += 1
                continue
            seen_train_fps.add(fp)
            stamped = dict(row)
            stamped["source"] = "sdg"
            stamped["sdg_model"] = sdg_model
            stamped["sdg_band"] = band_k
            stamped["sdg_exemplar_ids"] = [
                traces[i].get("id", i) for i in exemplar_idx[:max_exemplars]
            ]
            synthetic.append(stamped)
            stats["produced"] += 1
    return synthetic, stats


# ───────────────────────────── orchestration ─────────────────────────────────


@dataclass
class CurationResult:
    """One cluster's curated artifacts: the frozen exam, the train pool
    (real + synthetic), and an auditable manifest."""

    holdout: list[dict[str, Any]] = field(default_factory=list)
    train: list[dict[str, Any]] = field(default_factory=list)
    synthetic: list[dict[str, Any]] = field(default_factory=list)
    holdout_ref: str = ""
    manifest: dict[str, Any] = field(default_factory=dict)


def curate_cluster(
    traces: list[dict[str, Any]],
    centroid: Sequence[float] | None,
    *,
    scorer: DifficultyScorer | None = None,
    synthetic_generator: SyntheticGenerator | None = None,
    sdg_model: str = "unknown",
    holdout_frac: float = HOLDOUT_FRAC,
    min_holdout: int = MIN_HOLDOUT,
    max_holdout: int = MAX_HOLDOUT,
    min_train: int = MIN_TRAIN,
    near_dup_cosine: float | None = None,
    n_bands: int = 4,
) -> CurationResult:
    """The full curation pass for one ignited cluster:

    rank by difficulty → hardest real rows become the frozen holdout →
    the rest train → coverage gaps filled through the SDG seam (when bound).

    Deterministic for fixed inputs and a deterministic scorer/generator —
    re-running curation on an unchanged cluster yields the identical
    ``holdout_ref``, which is what makes the spec emit idempotent.
    """
    ranked = rank_by_difficulty(traces, centroid, scorer)
    split = split_hard_holdout(
        traces,
        ranked,
        holdout_frac=holdout_frac,
        min_holdout=min_holdout,
        max_holdout=max_holdout,
        min_train=min_train,
        near_dup_cosine=near_dup_cosine,
    )
    score_of = {r.index: r.score for r in ranked}
    holdout_rows = [dict(traces[i]) for i in split.holdout_indices]
    train_rows = [dict(traces[i]) for i in split.train_indices]
    holdout_fps = {_prompt_fingerprint(r) for r in holdout_rows}

    synthetic_rows: list[dict[str, Any]] = []
    sdg_stats: dict[str, int] = {}
    if synthetic_generator is not None:
        bands = coverage_bands(
            traces, split.train_indices, centroid, n_bands=n_bands
        )
        synthetic_rows, sdg_stats = fill_gaps(
            traces,
            bands,
            synthetic_generator,
            holdout_fingerprints=holdout_fps,
            sdg_model=sdg_model,
        )

    # The two guarantees, enforced at the boundary (not just by construction):
    assert not any(is_synthetic(r) for r in holdout_rows), "synthetic row in holdout"
    assert not (holdout_fps & {_prompt_fingerprint(r) for r in train_rows + synthetic_rows}), (
        "holdout/train prompt overlap survived the leakage guard"
    )

    ref = holdout_content_ref(holdout_rows)
    h_scores = [score_of[i] for i in split.holdout_indices]
    t_scores = [score_of[i] for i in split.train_indices]
    result = CurationResult(
        holdout=holdout_rows,
        train=train_rows,
        synthetic=synthetic_rows,
        holdout_ref=ref,
        manifest={
            "holdout_ref": ref,
            "n_input_traces": len(traces),
            "n_holdout": len(holdout_rows),
            "n_train_real": len(train_rows),
            "n_train_synthetic": len(synthetic_rows),
            "dropped_duplicate_train": split.dropped_duplicate_train,
            "skipped_synthetic_for_holdout": split.skipped_synthetic_for_holdout,
            "difficulty": {
                "holdout_mean": round(sum(h_scores) / len(h_scores), 4) if h_scores else None,
                "holdout_min": round(min(h_scores), 4) if h_scores else None,
                "train_mean": round(sum(t_scores) / len(t_scores), 4) if t_scores else None,
            },
            "sdg": {"model": sdg_model, **sdg_stats} if synthetic_generator else None,
            "scorer": "default" if scorer is None else getattr(
                scorer, "__name__", scorer.__class__.__name__
            ),
        },
    )
    return result


def write_curation(out_dir: Path, result: CurationResult) -> dict[str, Any]:
    """Persist one cluster's curation: ``holdout.jsonl`` (the frozen exam),
    ``train.jsonl`` (real + synthetic train rows, provenance-stamped), and
    ``manifest.json``. Returns the manifest (with paths + built_at added)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    holdout_path = out_dir / "holdout.jsonl"
    train_path = out_dir / "train.jsonl"
    with holdout_path.open("w", encoding="utf-8") as f:
        for r in result.holdout:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with train_path.open("w", encoding="utf-8") as f:
        for r in result.train + result.synthetic:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    manifest = dict(result.manifest)
    manifest["holdout_path"] = str(holdout_path)
    manifest["train_path"] = str(train_path)
    manifest["built_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return manifest


__all__ = [
    "HOLDOUT_FRAC",
    "MAX_HOLDOUT",
    "MIN_HOLDOUT",
    "MIN_TRAIN",
    "CoverageBand",
    "CurationResult",
    "DifficultyScorer",
    "HoldoutSplit",
    "ScoredTrace",
    "SyntheticGenerator",
    "coverage_bands",
    "curate_cluster",
    "default_difficulty_scorer",
    "fill_gaps",
    "holdout_content_ref",
    "is_synthetic",
    "prompt_text",
    "rank_by_difficulty",
    "split_hard_holdout",
    "trace_embedding",
    "write_curation",
]
