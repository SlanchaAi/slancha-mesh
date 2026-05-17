"""v3.1 builder — production-representative corpus via mmbert-classified ingest,
target-distribution rebalance, and HF supplementation of undercount domains.

Composes with `build_corpus_v3.py` helpers (heuristic classifier as fallback)
and operates over the v3.0 output. Pipeline:

    1. Load v3.0 prompts.jsonl (heuristic signals).
    2. Load --ingest-classified JSONL (spark's mmbert output) and OVERLAY
       its `signals` onto v3.0 records by prompt_id. Each merged record
       gains `signals.classifier = "mmbert-6h"`.
    3. Rebalance to --target-distribution: bucket by domain, take random
       sample per domain matching target count. Track per-domain undercount.
    4. Supplement undercount domains by streaming from HF supplemental
       sources (CodeSearchNet, MMLU-Pro, HumanEval, GlaiveFC) — each
       newly-sourced row tagged `classifier = "heuristic-v3"`.
    5. Emit v3.1 JSONL + manifest with target_distribution,
       achieved_distribution, supplementation_log, plus the input
       v3.0 + classified sha256s for provenance.

Target distribution (operator-driven, defaults):

    code         27.5%
    general      22.5%
    reasoning    15%
    math         10%
    multilingual 10%
    creative     10%
    tool-use     5%

Usage:

    python -m mesh.scripts.build_corpus_v3_1 \\
        --v3-corpus       corpus/training/v3/prompts.jsonl \\
        --ingest-classified  corpus/training/v3/v3.0_mmbert.jsonl \\
        --output          corpus/training/v3.1/prompts.jsonl \\
        --manifest        corpus/training/v3.1/manifest.json \\
        --total           100000

`--ingest-classified` is optional: omit it and the rebalance runs over
the heuristic signals from v3.0 directly. This is the "no mmbert
available" fallback path; v3.1 quality will be heuristic-limited.

Supplemental sources can be disabled with `--no-supplement` if you want
just the rebalanced subset of v3.0 (likely smaller than target_total).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator

from mesh.scripts.build_corpus_v3 import (
    GatedDatasetError,
    signals_for,
)


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------


DEFAULT_TARGET_DISTRIBUTION: dict[str, float] = {
    "code":         0.275,
    "general":      0.225,
    "reasoning":    0.15,
    "math":         0.10,
    "multilingual": 0.10,
    "creative":     0.10,
    "tool-use":     0.05,
}


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------


def _sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for i, raw in enumerate(f, start=1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                yield json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}: invalid JSON at line {i} ({exc})") from exc


def load_classified_index(path: Path) -> dict[str, dict[str, Any]]:
    """Build a {prompt_id: signals} map from spark's mmbert JSONL.

    Accepts either {prompt_id, signals: {...}} or {prompt_id, signals_mmbert: {...}}
    (spark's wire said either field name works in their normalizer).
    """
    out: dict[str, dict[str, Any]] = {}
    for rec in iter_jsonl(path):
        pid = rec.get("prompt_id") or rec.get("id")
        if not pid:
            continue
        sig = rec.get("signals") or rec.get("signals_mmbert")
        if not sig:
            continue
        out[pid] = sig
    return out


# mmbert's 6-head domain output uses an MMLU-style academic taxonomy
# (psychology / history / philosophy / computer science / ...) which does
# not align 1:1 with the operator's routing taxonomy (code / general /
# reasoning / math / multilingual / creative / tool-use). The translator
# below collapses mmbert's vocabulary onto the routing taxonomy so the
# rebalancer can match domains correctly.
#
# Notes:
#   - mmbert has NO concept of "code" — it emits "computer science".
#   - mmbert has NO concept of "reasoning" — closest is general academic.
#   - mmbert has NO concept of "creative" or "tool-use" — these come only
#     from HF supplements (dolly creative, glaive function-calling).
#   - mmbert's language head IS reliable for multilingual detection, so we
#     promote non-English rows to domain="multilingual" regardless of
#     mmbert's domain output. This is the single highest-confidence axis
#     in the mmbert 6-head output.
MMBERT_TO_ROUTING_DOMAIN: dict[str, str] = {
    "computer science": "code",
    "math": "math",
    "physics": "math",
    "engineering": "math",
    "chemistry": "math",
    # All academic subjects below collapse to "general" — they're
    # conversational/knowledge-question prompts in practice.
    "psychology":  "general",
    "history":     "general",
    "philosophy":  "general",
    "law":         "general",
    "biology":     "general",
    "business":    "general",
    "economics":   "general",
    "health":      "general",
    "other":       "general",
}


def translate_mmbert_signals(mmbert: dict[str, Any]) -> dict[str, Any]:
    """Translate a mmbert signals dict into the routing taxonomy.

    Output keys: domain (routing-style), difficulty, language, needs_tools,
    plus all original mmbert fields kept under `mmbert_*` prefixes so
    nothing's lost. Multilingual override: language != "en" → routing
    domain = "multilingual" (mmbert language is the most reliable axis).
    """
    mm_domain = mmbert.get("domain", "other")
    routing_domain = MMBERT_TO_ROUTING_DOMAIN.get(mm_domain, "general")
    lang = mmbert.get("language", "en")
    # mmbert may emit ISO codes like "de", "fr", "zh", "en", or "english"
    is_english = lang in ("en", "english", "")
    if not is_english:
        routing_domain = "multilingual"
    return {
        "domain":      routing_domain,
        "difficulty":  mmbert.get("difficulty", "medium"),
        "language":    "other" if not is_english else "en",
        "needs_tools": bool(mmbert.get("needs_tools", False)),
        # Preserve full mmbert provenance under mmbert_* keys for posterity
        # — useful for the dashboard panels that want confidence-band views
        "mmbert_domain":               mm_domain,
        "mmbert_domain_confidence":    mmbert.get("domain_confidence"),
        "mmbert_difficulty_confidence": mmbert.get("difficulty_confidence"),
        "mmbert_language":             lang,
        "mmbert_language_confidence":  mmbert.get("language_confidence"),
        "mmbert_is_jailbreak":         mmbert.get("is_jailbreak"),
        "mmbert_jailbreak_confidence": mmbert.get("jailbreak_confidence"),
        "mmbert_has_pii":              mmbert.get("has_pii"),
        "mmbert_pii_confidence":       mmbert.get("pii_confidence"),
        "mmbert_tool_confidence":      mmbert.get("tool_confidence"),
    }


# ---------------------------------------------------------------------------
# Merge step — overlay mmbert signals onto v3.0 records
# ---------------------------------------------------------------------------


def merge_with_classified(
    v3_records: list[dict[str, Any]],
    classified: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], int]:
    """Overlay classified[pid] signals onto v3 record; tag classifier provenance.

    When the classified signals come from mmbert (academic-taxonomy 6-head),
    they are TRANSLATED to the routing taxonomy via translate_mmbert_signals
    before overlay. mmbert provenance (raw mmbert_* fields + confidences)
    is preserved on the record so dashboards can still surface confidence
    bands.

    Returns (merged_records, n_overlaid). Records whose prompt_id isn't in
    the classified index keep their heuristic signals (and classifier =
    "heuristic-v3"). Records whose signals were overlaid get
    classifier = "mmbert-6h-translated".
    """
    out: list[dict[str, Any]] = []
    n_overlaid = 0
    for rec in v3_records:
        pid = rec["prompt_id"]
        new = dict(rec)
        signals = dict(new.get("signals", {}))
        if pid in classified:
            # Detect mmbert academic taxonomy: presence of domain_confidence
            # is the cheap discriminator. (Heuristic signals don't have it.)
            raw = classified[pid]
            if "domain_confidence" in raw or raw.get("domain") in MMBERT_TO_ROUTING_DOMAIN:
                translated = translate_mmbert_signals(raw)
                # Keep heuristic-only fields that translation doesn't cover
                # (e.g., route_class), but let mmbert truth win on overlaps.
                for k, v in translated.items():
                    signals[k] = v
                signals["classifier"] = "mmbert-6h-translated"
            else:
                # Non-mmbert classified source (e.g., a second mac-side pass).
                # Copy through verbatim.
                for k, v in raw.items():
                    signals[k] = v
                signals["classifier"] = "classified"
            # Re-derive route_class from (possibly new) difficulty so it
            # remains internally consistent with the overlaid signals.
            signals["route_class"] = route_class_from_difficulty(
                signals.get("difficulty", "medium")
            )
            n_overlaid += 1
        else:
            signals.setdefault("classifier", "heuristic-v3")
        new["signals"] = signals
        out.append(new)
    return out, n_overlaid


def route_class_from_difficulty(difficulty: str) -> str:
    """Mirror of build_corpus_v3's route_class derivation, re-exposed
    here so translated-mmbert records keep route_class self-consistent
    with the (possibly new) difficulty value."""
    if difficulty == "hard":
        return "batch"
    if difficulty == "medium":
        return "standard"
    return "hot_interactive"


# ---------------------------------------------------------------------------
# Target-distribution rebalance
# ---------------------------------------------------------------------------


def rebalance_to_target(
    records: list[dict[str, Any]],
    target_distribution: dict[str, float],
    total_target: int,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Sample records per-domain to match target distribution.

    Returns (selected_records, undercount_per_domain). Domains not in the
    target distribution are dropped from the rebalanced output (caller
    can re-include via a residual bucket if desired). When a domain has
    fewer records than its target count, the full bucket is taken and
    the shortfall is reported as undercount for supplementation.
    """
    rng = random.Random(seed)
    needed: dict[str, int] = {
        d: int(round(pct * total_target))
        for d, pct in target_distribution.items()
    }
    by_domain: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in records:
        d = r.get("signals", {}).get("domain", "general")
        by_domain[d].append(r)

    selected: list[dict[str, Any]] = []
    undercount: dict[str, int] = {}
    for domain, target_count in needed.items():
        bucket = by_domain.get(domain, [])
        if len(bucket) >= target_count:
            selected.extend(rng.sample(bucket, target_count))
        else:
            selected.extend(bucket)
            undercount[domain] = target_count - len(bucket)
    return selected, undercount


# ---------------------------------------------------------------------------
# HF supplementation
# ---------------------------------------------------------------------------


@dataclass
class SupplementalSource:
    name: str
    hf_id: str
    config: str | None
    split: str
    extract: Callable[[dict], str | None]
    expected_domain: str  # what we expect rows from this source to classify as


def _humaneval_extract(rec: dict) -> str | None:
    p = (rec.get("prompt") or "").strip()
    return p or None


def _mmlupro_extract(rec: dict) -> str | None:
    q = (rec.get("question") or "").strip()
    return q or None


def _glaive_extract(rec: dict) -> str | None:
    chat = (rec.get("chat") or "").strip()
    if not chat:
        return None
    # Pull just the USER turn out of the alternating-roles chat string.
    for line in chat.split("USER: "):
        line = line.strip()
        if line and not line.startswith("ASSISTANT"):
            return line.split("ASSISTANT:", 1)[0].strip()[:1500] or None
    return None


def _codesearchnet_extract(rec: dict) -> str | None:
    # Older CodeSearchNet schemas vary; try common fields in order
    for field in ("func_documentation_string", "docstring", "func_code_string"):
        v = rec.get(field)
        if v and isinstance(v, str) and v.strip():
            return v.strip()[:1500]
    return None


def _writingprompts_extract(rec: dict) -> str | None:
    # r/WritingPrompts dump on HF — "prompt" field
    p = (rec.get("prompt") or "").strip()
    return p[:1500] if p else None


def _no_robots_extract(rec: dict) -> str | None:
    # HuggingFaceH4/no_robots — messages list with role=user first
    msgs = rec.get("messages") or rec.get("prompt") or []
    if isinstance(msgs, str):
        return msgs.strip()[:1500] or None
    for m in msgs:
        if isinstance(m, dict) and m.get("role") == "user":
            content = (m.get("content") or "").strip()
            return content[:1500] if content else None
    return None


def _dolly_creative_extract(rec: dict) -> str | None:
    if rec.get("category") not in ("creative_writing", "brainstorming", "open_qa"):
        return None
    instr = (rec.get("instruction") or "").strip()
    return instr[:1500] if instr else None


def _arc_extract(rec: dict) -> str | None:
    q = (rec.get("question") or "").strip()
    return q[:1500] if q else None


def _openbookqa_extract(rec: dict) -> str | None:
    """OpenBookQA — the question_stem is the prompt."""
    q = (rec.get("question_stem") or "").strip()
    if not q and "question" in rec:
        # Some dump variants wrap as {question: {stem: ...}}
        inner = rec.get("question")
        if isinstance(inner, dict):
            q = (inner.get("stem") or "").strip()
        elif isinstance(inner, str):
            q = inner.strip()
    return q[:1500] if q else None


def _boolq_extract(rec: dict) -> str | None:
    q = (rec.get("question") or "").strip()
    passage = (rec.get("passage") or "").strip()
    if not q:
        return None
    # Combine question + supporting passage for richer reasoning prompt
    if passage:
        text = f"{q}\n\nContext: {passage}"
    else:
        text = q
    return text[:1500]


def _aya_non_english_extract(rec: dict) -> str | None:
    """CohereForAI/aya_dataset — accept only rows where the prompt is in a
    non-English language (language_code != 'eng' / 'en'). Multilingual
    source for the multilingual domain bucket."""
    lang = (rec.get("language_code") or rec.get("language") or "").lower()
    if lang in ("", "eng", "en", "english"):
        return None
    text = (rec.get("inputs") or "").strip()
    return text[:1500] if text else None


def _opus100_non_english_extract(rec: dict) -> str | None:
    """Helsinki-NLP/opus-100 — translation pairs. Take the non-English
    side as a multilingual seed prompt."""
    t = rec.get("translation") or {}
    for k, v in t.items():
        if k != "en" and isinstance(v, str) and v.strip():
            return v.strip()[:1500]
    return None


SUPPLEMENTAL_SOURCES: dict[str, list[SupplementalSource]] = {
    "code": [
        SupplementalSource(
            name="humaneval",
            hf_id="openai_humaneval",
            config=None,
            split="test",
            extract=_humaneval_extract,
            expected_domain="code",
        ),
        SupplementalSource(
            name="codesearchnet",
            hf_id="code_search_net",
            config="python",
            split="train",
            extract=_codesearchnet_extract,
            expected_domain="code",
        ),
    ],
    "reasoning": [
        SupplementalSource(
            name="mmlu-pro",
            hf_id="TIGER-Lab/MMLU-Pro",
            config=None,
            split="test",
            extract=_mmlupro_extract,
            expected_domain="reasoning",
        ),
        SupplementalSource(
            name="arc-challenge",
            hf_id="allenai/ai2_arc",
            config="ARC-Challenge",
            split="train",
            extract=_arc_extract,
            expected_domain="reasoning",
        ),
        SupplementalSource(
            name="openbookqa",
            hf_id="allenai/openbookqa",
            config="main",
            split="train",
            extract=_openbookqa_extract,
            expected_domain="reasoning",
        ),
        SupplementalSource(
            name="boolq",
            hf_id="google/boolq",
            config=None,
            split="train",
            extract=_boolq_extract,
            expected_domain="reasoning",
        ),
    ],
    "math": [
        SupplementalSource(
            name="mmlu-pro-math",
            hf_id="TIGER-Lab/MMLU-Pro",
            config=None,
            split="test",
            extract=_mmlupro_extract,
            expected_domain="math",
        ),
    ],
    "tool-use": [
        SupplementalSource(
            name="glaive-fc",
            hf_id="glaiveai/glaive-function-calling-v2",
            config=None,
            split="train",
            extract=_glaive_extract,
            expected_domain="tool-use",
        ),
    ],
    "multilingual": [
        SupplementalSource(
            name="aya-nonen",
            hf_id="CohereForAI/aya_dataset",
            config=None,
            split="train",
            extract=_aya_non_english_extract,
            expected_domain="multilingual",
        ),
        SupplementalSource(
            name="opus100-zh",
            hf_id="Helsinki-NLP/opus-100",
            config="en-zh",
            split="train",
            extract=_opus100_non_english_extract,
            expected_domain="multilingual",
        ),
    ],
    "creative": [
        SupplementalSource(
            name="dolly-creative",
            hf_id="databricks/databricks-dolly-15k",
            config=None,
            split="train",
            extract=_dolly_creative_extract,
            expected_domain="creative",
        ),
        SupplementalSource(
            name="no-robots",
            hf_id="HuggingFaceH4/no_robots",
            config=None,
            split="train",
            extract=_no_robots_extract,
            expected_domain="creative",
        ),
        SupplementalSource(
            name="writingprompts",
            hf_id="euclaise/writingprompts",
            config=None,
            split="train",
            extract=_writingprompts_extract,
            expected_domain="creative",
        ),
    ],
    # multilingual + general: rely on v3.0 baseline for now
}


def supplement_undercount(
    undercount: dict[str, int],
    seed: int,
    starting_idx: int = 0,
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    """Stream supplemental HF sources, classify each row, accept rows that
    match the target domain until that domain's quota is met.

    Returns (extra_records, log). `log[domain]` is a one-line audit of
    what got supplemented (or why a domain couldn't be supplemented).
    """
    from datasets import load_dataset
    from datasets.exceptions import DatasetNotFoundError

    extra: list[dict[str, Any]] = []
    log: dict[str, str] = {}
    rng = random.Random(seed)
    idx = starting_idx

    for domain, count_needed in undercount.items():
        sources = SUPPLEMENTAL_SOURCES.get(domain, [])
        if not sources:
            log[domain] = f"no supplemental source registered (short {count_needed})"
            continue

        accepted = 0
        attempts: list[str] = []
        OVERSAMPLE = 4

        for src in sources:
            if accepted >= count_needed:
                break
            scanned = 0
            src_accepted_before = accepted
            try:
                ds = load_dataset(src.hf_id, src.config, split=src.split, streaming=True)
                try:
                    ds = ds.shuffle(seed=seed, buffer_size=2_000)
                except Exception:
                    pass
                for rec in ds:
                    scanned += 1
                    if scanned > count_needed * OVERSAMPLE:
                        break
                    try:
                        text = src.extract(rec)
                    except Exception:
                        text = None
                    if not text:
                        continue
                    # Source-provenance domain assignment: trust the
                    # supplementary source's expected_domain rather than
                    # the heuristic classifier (which is too strict to
                    # recognize tool-use / creative / and miscategorizes
                    # short MMLU-Pro reasoning prompts as "general").
                    sig = signals_for(text)
                    sig["domain"] = src.expected_domain
                    # Recompute route_class to stay consistent with the
                    # difficulty bucket we ended up in (signals_for sets
                    # route_class based on difficulty, which is still valid).
                    sig["classifier"] = "heuristic-v3+source-domain"
                    extra.append({
                        "prompt_id": f"sup-{src.name}-{idx:07d}",
                        "prompt_text": text,
                        "source": f"supplement:{src.hf_id}",
                        "signals": sig,
                    })
                    idx += 1
                    accepted += 1
                    if accepted >= count_needed:
                        break
                attempts.append(f"{src.name}={accepted - src_accepted_before}")
            except DatasetNotFoundError as e:
                if "gated" in str(e).lower():
                    attempts.append(f"{src.name}=gated")
                else:
                    attempts.append(f"{src.name}=err:{e}")
                continue
            except Exception as e:
                attempts.append(f"{src.name}=err:{type(e).__name__}")
                continue

        if accepted >= count_needed:
            log[domain] = f"supplemented {accepted}/{count_needed} via [{', '.join(attempts)}]"
        else:
            log[domain] = (
                f"PARTIAL: {accepted}/{count_needed} via [{', '.join(attempts)}] "
                f"(short {count_needed - accepted})"
            )

    return extra, log


# ---------------------------------------------------------------------------
# Top-level pipeline
# ---------------------------------------------------------------------------


def build_v3_1(
    v3_corpus_path: Path,
    output_path: Path,
    manifest_path: Path,
    total_target: int,
    target_distribution: dict[str, float],
    seed: int,
    ingest_classified_path: Path | None = None,
    skip_supplement: bool = False,
) -> dict[str, Any]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()

    # 1. Load v3.0 corpus
    print(f"[v3.1] loading v3.0 corpus from {v3_corpus_path} ...", file=sys.stderr, flush=True)
    v3_records = list(iter_jsonl(v3_corpus_path))
    v3_sha = _sha256_file(v3_corpus_path)
    print(f"[v3.1] loaded {len(v3_records)} rows; sha256={v3_sha[:16]}...", file=sys.stderr)

    # 2. Optional overlay with classified signals
    n_overlaid = 0
    classified_sha: str | None = None
    if ingest_classified_path is not None:
        print(f"[v3.1] overlaying classified from {ingest_classified_path} ...", file=sys.stderr, flush=True)
        classified = load_classified_index(ingest_classified_path)
        classified_sha = _sha256_file(ingest_classified_path)
        merged, n_overlaid = merge_with_classified(v3_records, classified)
        print(
            f"[v3.1] overlaid {n_overlaid}/{len(v3_records)} rows with mmbert signals",
            file=sys.stderr,
        )
    else:
        # Tag all as heuristic-v3 since we have no mmbert
        merged = []
        for rec in v3_records:
            new = dict(rec)
            s = dict(new.get("signals", {}))
            s.setdefault("classifier", "heuristic-v3")
            new["signals"] = s
            merged.append(new)

    # 3. Target-distribution rebalance
    print(f"[v3.1] rebalancing to target distribution {target_distribution} ...", file=sys.stderr)
    selected, undercount = rebalance_to_target(
        merged, target_distribution, total_target, seed=seed,
    )
    print(
        f"[v3.1] rebalanced to {len(selected)} rows; undercount={undercount}",
        file=sys.stderr,
    )

    # 4. Supplement undercount
    supplementation_log: dict[str, str] = {}
    extra_records: list[dict[str, Any]] = []
    if undercount and not skip_supplement:
        print(f"[v3.1] supplementing {sum(undercount.values())} rows across {len(undercount)} domains ...", file=sys.stderr)
        extra_records, supplementation_log = supplement_undercount(
            undercount, seed=seed, starting_idx=len(selected),
        )
        print(f"[v3.1] supplemented {len(extra_records)} rows; log={supplementation_log}", file=sys.stderr)

    final_records = selected + extra_records

    # 5. Write output + manifest
    print(f"[v3.1] writing {len(final_records)} rows to {output_path} ...", file=sys.stderr)
    h = hashlib.sha256()
    with output_path.open("w", encoding="utf-8") as fout:
        for rec in final_records:
            line = json.dumps(rec, ensure_ascii=False)
            fout.write(line + "\n")
            h.update(line.encode("utf-8"))

    achieved: dict[str, int] = defaultdict(int)
    for r in final_records:
        achieved[r.get("signals", {}).get("domain", "general")] += 1

    classifier_provenance: dict[str, int] = defaultdict(int)
    for r in final_records:
        classifier_provenance[r.get("signals", {}).get("classifier", "unknown")] += 1

    manifest = {
        "version": "3.1",
        "seed": seed,
        "total_target": total_target,
        "total": len(final_records),
        "target_distribution_pct": target_distribution,
        "achieved_distribution": dict(achieved),
        "achieved_distribution_pct": {
            d: round(n / max(len(final_records), 1), 4) for d, n in achieved.items()
        },
        "classifier_provenance": dict(classifier_provenance),
        "undercount_per_domain": undercount,
        "supplementation_log": supplementation_log,
        "v3_source_sha256": v3_sha,
        "classified_source_sha256": classified_sha,
        "output_sha256": h.hexdigest(),
        "output_path": str(output_path),
        "elapsed_s": round(time.time() - started, 2),
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(
        f"[v3.1] DONE total={len(final_records)} "
        f"elapsed={time.time() - started:.1f}s "
        f"sha256={h.hexdigest()[:16]}... manifest={manifest_path}",
        file=sys.stderr,
    )
    return manifest


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_dist(s: str) -> dict[str, float]:
    """Parse a `domain:pct,domain:pct` string into a {domain: float} dict.

    Percentages can be 0-100 or 0-1; both are normalized to sum to ~1.0.
    """
    out: dict[str, float] = {}
    for chunk in s.split(","):
        if ":" not in chunk:
            continue
        k, v = chunk.split(":", 1)
        out[k.strip()] = float(v.strip())
    total = sum(out.values())
    if total > 2.0:  # caller used 0-100 instead of 0-1
        out = {k: v / 100.0 for k, v in out.items()}
        total = sum(out.values())
    # Re-normalize (silently) so manifest math is consistent
    if total > 0:
        out = {k: v / total for k, v in out.items()}
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--v3-corpus",
        type=Path,
        default=Path("corpus/training/v3/prompts.jsonl"),
        help="Input v3.0 corpus (default corpus/training/v3/prompts.jsonl)",
    )
    ap.add_argument(
        "--ingest-classified",
        type=Path,
        default=None,
        help="Optional JSONL of pre-classified signals (mmbert-output) keyed by prompt_id",
    )
    ap.add_argument(
        "--output",
        type=Path,
        default=Path("corpus/training/v3.1/prompts.jsonl"),
    )
    ap.add_argument(
        "--manifest",
        type=Path,
        default=Path("corpus/training/v3.1/manifest.json"),
    )
    ap.add_argument("--total", type=int, default=100_000)
    ap.add_argument(
        "--target-distribution",
        type=_parse_dist,
        default=DEFAULT_TARGET_DISTRIBUTION,
        help="Target dist as comma-separated 'domain:pct' (default operator preset)",
    )
    ap.add_argument(
        "--seed",
        type=lambda s: int(s, 0),
        default=0xC0FFEE,
    )
    ap.add_argument(
        "--no-supplement",
        action="store_true",
        help="Skip HF supplementation of undercount domains. Output will be smaller than --total.",
    )
    args = ap.parse_args(argv)

    if not args.v3_corpus.exists():
        print(f"v3 corpus not found: {args.v3_corpus}", file=sys.stderr)
        return 2

    build_v3_1(
        v3_corpus_path=args.v3_corpus,
        output_path=args.output,
        manifest_path=args.manifest,
        total_target=args.total,
        target_distribution=args.target_distribution,
        seed=args.seed,
        ingest_classified_path=args.ingest_classified,
        skip_supplement=args.no_supplement,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
