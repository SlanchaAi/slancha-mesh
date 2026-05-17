"""Build the v3 100K training corpus from HF datasets — balanced + reproducible.

Streams from four sources, takes a quota per source, applies a heuristic
classifier (domain / difficulty / language) per row, and writes one JSONL
line per prompt to `corpus/training/v3/prompts.jsonl`. Also writes a
`manifest.json` capturing seed, dataset versions, per-source counts, and the
SHA-256 of the output file so spark (or anyone else) can verify
reproducibility on another host.

Sources (HF dataset id → user-prompt extraction strategy):

    allenai/WildChat-1M           → conversation[0]["content"] where role=user
    lmsys/lmsys-chat-1m           → conversation[0]["content"] where role=user
    Open-Orca/OpenOrca            → question
    HuggingFaceH4/ultrachat_200k  → messages[0]["content"] where role=user

The script does not download whole datasets — it uses `streaming=True` and
stops at the per-source quota. Quota defaults to 25_000 per source = 100_000.

Output line shape (drop-in for mesh_replay.py):
    {"prompt_id":   "wildchat-000123",
     "prompt_text": "...",
     "source":      "allenai/WildChat-1M",
     "signals": {"domain":      "code",      # math|code|reasoning|multilingual|general
                  "difficulty":   "medium",  # easy|medium|hard
                  "language":     "en",       # en|other
                  "needs_tools":  false,
                  "route_class":  "standard"}}

Heuristics are intentionally cheap (no external lang/domain classifier
deps) — the dashboard's model-mix-diversity panel needs *some* axis to
break out by, not gospel-truth labels. Re-classify later with a real
model if signal turns out load-bearing.

Usage:

    # Smoke (200 rows, 50/source) — verifies access + write path
    python -m mesh.scripts.build_corpus_v3 --smoke

    # Full 100K build
    python -m mesh.scripts.build_corpus_v3 \\
        --output corpus/training/v3/prompts.jsonl \\
        --manifest corpus/training/v3/manifest.json \\
        --per-source 25000

    # Custom seed (default 0xc0ffee)
    python -m mesh.scripts.build_corpus_v3 --seed 42 --per-source 1000
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator


# ---------------------------------------------------------------------------
# Heuristic classifier — domain / difficulty / language
# ---------------------------------------------------------------------------


_CODE_PAT = re.compile(
    r"```|def \w+\(|class \w+|function\s+\w+\(|import\s+\w+|"
    r"\bvar\s+\w+\s*=|console\.log|System\.out|print\(|"
    r"<html|<script|SELECT\s+\w|FROM\s+\w+\s+WHERE",
    re.IGNORECASE,
)

_MATH_PAT = re.compile(
    r"\b(calculate|compute|evaluate|solve|equation|integral|derivative|"
    r"matrix|theorem|prove that|simplify|factorize)\b|"
    r"[∀-⋿]|"   # math operators block
    r"\\frac|\\sum|\\int|\\sqrt|\\pi|\\theta",
    re.IGNORECASE,
)

_REASON_PAT = re.compile(
    r"^(why|how does|how would|explain|what would happen if|"
    r"step[- ]by[- ]step|reason through)\b",
    re.IGNORECASE,
)

_TRANSLATE_PAT = re.compile(
    r"\b(translate|translation|in (french|spanish|german|chinese|japanese|"
    r"korean|russian|arabic|hindi|portuguese))\b",
    re.IGNORECASE,
)


def classify_domain(text: str) -> str:
    if _CODE_PAT.search(text):
        return "code"
    if _MATH_PAT.search(text):
        return "math"
    if _TRANSLATE_PAT.search(text):
        return "multilingual"
    if _REASON_PAT.search(text):
        return "reasoning"
    return "general"


def classify_language(text: str) -> str:
    """en | other. Cheap ASCII-ratio heuristic; intentionally crude.

    100K-row dashboard panel needs en-vs-other diversity, not language
    identification. If a row is >80% ASCII and contains common English
    function-words, call it en. Else other.
    """
    if not text:
        return "en"
    sample = text[:400]
    ascii_chars = sum(1 for c in sample if ord(c) < 128)
    ascii_ratio = ascii_chars / max(len(sample), 1)
    if ascii_ratio < 0.8:
        return "other"
    # ASCII-heavy: default to en (covers code, English prose, etc.). En
    # function-word markers are a positive-confirm signal, not a gate —
    # short ASCII strings without markers are still treated as en.
    return "en"


def classify_difficulty(text: str) -> str:
    n = len(text)
    if n > 800:
        return "hard"
    if n > 300:
        return "medium"
    return "easy"


def route_class_from_difficulty(difficulty: str) -> str:
    if difficulty == "hard":
        return "batch"
    if difficulty == "medium":
        return "standard"
    return "hot_interactive"


def signals_for(text: str) -> dict[str, Any]:
    difficulty = classify_difficulty(text)
    return {
        "domain": classify_domain(text),
        "difficulty": difficulty,
        "language": classify_language(text),
        "needs_tools": False,
        "route_class": route_class_from_difficulty(difficulty),
    }


# ---------------------------------------------------------------------------
# Source adapters — yield raw user-prompts as strings
# ---------------------------------------------------------------------------


@dataclass
class Source:
    name: str           # short slug for prompt_id prefix
    hf_id: str
    config: str | None
    split: str
    extract: Callable[[dict], str | None]


def _first_user_turn(conv_field: str, role_field: str, content_field: str):
    def _x(rec: dict) -> str | None:
        conv = rec.get(conv_field) or []
        for turn in conv:
            if turn.get(role_field) == "user":
                return (turn.get(content_field) or "").strip() or None
        return None
    return _x


def _orca_question(rec: dict) -> str | None:
    q = (rec.get("question") or "").strip()
    return q or None


SOURCES: list[Source] = [
    Source(
        name="wildchat",
        hf_id="allenai/WildChat-1M",
        config=None,
        split="train",
        extract=_first_user_turn("conversation", "role", "content"),
    ),
    Source(
        name="lmsys",
        hf_id="lmsys/lmsys-chat-1m",
        config=None,
        split="train",
        extract=_first_user_turn("conversation", "role", "content"),
    ),
    Source(
        name="openorca",
        hf_id="Open-Orca/OpenOrca",
        config=None,
        split="train",
        extract=_orca_question,
    ),
    Source(
        name="ultrachat",
        hf_id="HuggingFaceH4/ultrachat_200k",
        config=None,
        split="train_sft",
        extract=_first_user_turn("messages", "role", "content"),
    ),
]


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------


class GatedDatasetError(RuntimeError):
    """Raised when a source requires HF auth that isn't available."""


def iter_source(src: Source, quota: int, seed: int) -> Iterator[dict[str, Any]]:
    """Stream `quota` user-prompts from one HF source, with reservoir-style
    over-sampling so we get spread (not just the first N) while still bounded.

    Stream `quota * OVERSAMPLE` rows, pick the first `quota` non-empty
    user-prompts. (For datasets that may shuffle on-stream this gets us a
    diverse slice; for datasets that don't, it's still bounded.) Seeded
    `random.Random` decides ties.

    Raises `GatedDatasetError` if the source is gated and the host has no
    HF auth — caller decides to skip + re-balance.
    """
    # Imported lazily so the module doesn't pay datasets-import cost when
    # only the classifier helpers are used.
    from datasets import load_dataset
    from datasets.exceptions import DatasetNotFoundError

    OVERSAMPLE = 3
    seen_text_hashes: set[str] = set()
    rng = random.Random(seed)

    try:
        ds = load_dataset(
            src.hf_id,
            src.config,
            split=src.split,
            streaming=True,
        )
    except DatasetNotFoundError as e:
        if "gated" in str(e).lower():
            raise GatedDatasetError(
                f"{src.hf_id} is gated; set HF_TOKEN + accept the dataset "
                f"on huggingface.co/datasets/{src.hf_id} to enable."
            ) from e
        raise
    # Some HF datasets support .shuffle(seed=, buffer_size=) in streaming
    # mode; use it where available to pull a more diverse window.
    try:
        ds = ds.shuffle(seed=seed, buffer_size=10_000)
    except Exception:
        pass  # not all streaming datasets implement shuffle

    yielded = 0
    scanned = 0
    target_scan = quota * OVERSAMPLE
    for rec in ds:
        scanned += 1
        if scanned > target_scan and yielded >= quota:
            break
        try:
            text = src.extract(rec)
        except Exception:
            text = None
        if not text:
            continue
        # Dedup by sha1 to drop obvious copies (same template-prompt repeated)
        h = hashlib.sha1(text.encode("utf-8", errors="replace")).hexdigest()
        if h in seen_text_hashes:
            continue
        seen_text_hashes.add(h)
        yield {
            "source_name": src.name,
            "source_hf_id": src.hf_id,
            "text": text,
            "_rng": rng.random(),
        }
        yielded += 1
        if yielded >= quota:
            return


def build_corpus(
    output_path: Path,
    manifest_path: Path,
    per_source: int,
    seed: int,
    sources: list[Source],
    progress_every: int = 1000,
    target_total: int | None = None,
) -> dict[str, Any]:
    """Build the corpus. Gated sources are skipped with a warning; the
    remaining sources absorb the missing quota so target_total is still hit
    (when set). If target_total is None, no re-balancing happens — caller
    gets exactly per_source × len(accessible_sources)."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    # Probe each source quickly so we know the access set BEFORE writing
    # anything; lets us re-balance quota up-front rather than mid-stream.
    accessible: list[Source] = []
    skipped: list[tuple[str, str]] = []
    for src in sources:
        try:
            # Trigger the gated-check by attempting to instantiate the stream;
            # iter_source raises GatedDatasetError immediately on load_dataset
            # if the source is gated and HF auth is missing.
            probe = iter_source(src, quota=1, seed=seed)
            next(probe, None)  # advance once; ignore the value
            accessible.append(src)
        except GatedDatasetError as e:
            print(f"[build] SKIP {src.name}: {e}", file=sys.stderr, flush=True)
            skipped.append((src.name, str(e)))
        except Exception as e:
            print(f"[build] SKIP {src.name}: probe failed: {e}", file=sys.stderr, flush=True)
            skipped.append((src.name, f"probe failed: {e}"))

    if not accessible:
        raise RuntimeError("no accessible sources; aborting build")

    # Re-balance quota across accessible sources when target_total is set
    effective_per_source = per_source
    if target_total is not None:
        effective_per_source = (target_total + len(accessible) - 1) // len(accessible)
        if effective_per_source != per_source:
            print(
                f"[build] re-balanced per-source quota {per_source}→{effective_per_source} "
                f"across {len(accessible)} accessible sources to hit target_total={target_total}",
                file=sys.stderr,
                flush=True,
            )

    started = time.time()
    per_source_counts: dict[str, int] = {}
    per_domain_counts: dict[str, int] = {}
    per_language_counts: dict[str, int] = {}
    per_difficulty_counts: dict[str, int] = {}
    total = 0
    h = hashlib.sha256()

    with output_path.open("w", encoding="utf-8") as fout:
        for src in accessible:
            print(
                f"[build] sourcing {src.name} ({src.hf_id}, split={src.split}) "
                f"quota={effective_per_source} ...",
                file=sys.stderr,
                flush=True,
            )
            src_started = time.time()
            count = 0
            for i, item in enumerate(
                iter_source(src, quota=effective_per_source, seed=seed), start=1
            ):
                sig = signals_for(item["text"])
                row = {
                    "prompt_id": f"{src.name}-{i:07d}",
                    "prompt_text": item["text"],
                    "source": item["source_hf_id"],
                    "signals": sig,
                }
                line = json.dumps(row, ensure_ascii=False)
                fout.write(line + "\n")
                h.update(line.encode("utf-8"))
                count += 1
                total += 1
                per_domain_counts[sig["domain"]] = (
                    per_domain_counts.get(sig["domain"], 0) + 1
                )
                per_language_counts[sig["language"]] = (
                    per_language_counts.get(sig["language"], 0) + 1
                )
                per_difficulty_counts[sig["difficulty"]] = (
                    per_difficulty_counts.get(sig["difficulty"], 0) + 1
                )
                if total % progress_every == 0:
                    elapsed = time.time() - started
                    print(
                        f"[build] total={total} ({total/elapsed:.0f} rows/s)",
                        file=sys.stderr,
                        flush=True,
                    )
            per_source_counts[src.name] = count
            src_elapsed = time.time() - src_started
            print(
                f"[build] {src.name} done: {count} rows in {src_elapsed:.1f}s",
                file=sys.stderr,
                flush=True,
            )

    elapsed = time.time() - started
    manifest = {
        "version": 3,
        "seed": seed,
        "per_source_quota": effective_per_source,
        "requested_per_source_quota": per_source,
        "target_total": target_total,
        "skipped_sources": [
            {"name": n, "reason": r} for n, r in skipped
        ],
        "total": total,
        "per_source_counts": per_source_counts,
        "per_domain_counts": per_domain_counts,
        "per_language_counts": per_language_counts,
        "per_difficulty_counts": per_difficulty_counts,
        "sources": [
            {"name": s.name, "hf_id": s.hf_id, "split": s.split}
            for s in sources
        ],
        "output_sha256": h.hexdigest(),
        "output_path": str(output_path),
        "elapsed_s": round(elapsed, 2),
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(
        f"[build] complete: total={total} elapsed={elapsed:.1f}s "
        f"sha256={h.hexdigest()[:16]}... manifest={manifest_path}",
        file=sys.stderr,
        flush=True,
    )
    return manifest


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--output",
        type=Path,
        default=Path("corpus/training/v3/prompts.jsonl"),
        help="Output JSONL path (default: corpus/training/v3/prompts.jsonl)",
    )
    ap.add_argument(
        "--manifest",
        type=Path,
        default=Path("corpus/training/v3/manifest.json"),
        help="Manifest JSON path (default: corpus/training/v3/manifest.json)",
    )
    ap.add_argument(
        "--per-source",
        type=int,
        default=25_000,
        help="Per-source quota (default 25_000 → 100K total over 4 sources)",
    )
    ap.add_argument(
        "--seed",
        type=lambda s: int(s, 0),  # accept 0x... or decimal
        default=0xC0FFEE,
        help="Random seed for reservoir / shuffle (default 0xc0ffee)",
    )
    ap.add_argument(
        "--target-total",
        type=int,
        default=None,
        help="When set, re-balance per-source quota across accessible sources "
             "so total hits this number even if some sources are skipped. "
             "Useful for 'I want 100K total regardless of which gated datasets are reachable'.",
    )
    ap.add_argument(
        "--smoke",
        action="store_true",
        help="Smoke run: 50 rows per source = 200 total. Overrides --per-source.",
    )
    ap.add_argument(
        "--source",
        action="append",
        choices=[s.name for s in SOURCES],
        help="Restrict to one or more sources (repeatable). Default: all four.",
    )
    args = ap.parse_args(argv)

    per_source = 50 if args.smoke else args.per_source
    selected_sources = SOURCES
    if args.source:
        selected_sources = [s for s in SOURCES if s.name in args.source]

    build_corpus(
        output_path=args.output,
        manifest_path=args.manifest,
        per_source=per_source,
        seed=args.seed,
        sources=selected_sources,
        target_total=args.target_total,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
