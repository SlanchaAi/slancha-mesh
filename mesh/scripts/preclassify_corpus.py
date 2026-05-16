"""Pre-classify a slancha-test corpus for mesh_replay.py consumption.

slancha-test/corpus/v1/combined70.jsonl ships {id, prompt} only. Mac's
mesh_replay.py expects {prompt_id, prompt_text, signals: {domain,
difficulty, language, route_class, needs_tools}}.

This script bridges the two via prompt-id-prefix heuristics until the
real slancha-api classifier is wired into the mesh path. Not a classifier;
just enough of one to drive the replay against the mesh router.

Prefix → domain map:
    math500-*, gsm8k-*, math-*       → math
    mbpp-*, humaneval-*, code-*      → code
    mmlu-*                            → general
    flores-*, xnli-*                  → multilingual
    gpqa-*                            → reasoning
    *                                 → general (fallback)

Difficulty is derived from prompt length (proxy): >800 chars → hard,
>300 → medium, otherwise easy.

Usage:
    python -m mesh.scripts.preclassify_corpus \\
        --in  ~/Source/slancha-test/corpus/v1/combined70.jsonl \\
        --out ./prompts_classified.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

DOMAIN_PREFIXES: list[tuple[str, str]] = [
    ("math500", "math"),
    ("gsm8k", "math"),
    ("math-", "math"),
    ("mbpp", "code"),
    ("humaneval", "code"),
    ("code-", "code"),
    ("livecodebench", "code"),
    ("mmlu", "general"),
    ("flores", "multilingual"),
    ("xnli", "multilingual"),
    ("gpqa", "reasoning"),
]


def classify_id(prompt_id: str) -> str:
    pid = prompt_id.lower()
    for prefix, domain in DOMAIN_PREFIXES:
        if pid.startswith(prefix):
            return domain
    return "general"


def difficulty_from_length(text: str) -> str:
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


def transform_line(rec: dict) -> dict:
    pid = rec.get("id") or rec.get("prompt_id")
    prompt = rec.get("prompt") or rec.get("prompt_text") or ""
    if pid is None:
        raise KeyError("corpus line missing 'id' or 'prompt_id'")
    domain = classify_id(pid)
    difficulty = difficulty_from_length(prompt)
    return {
        "prompt_id": pid,
        "prompt_text": prompt,
        "signals": {
            "domain": domain,
            "difficulty": difficulty,
            "language": "en",
            "needs_tools": False,
            "route_class": route_class_from_difficulty(difficulty),
        },
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--in", dest="input", required=True, type=Path)
    ap.add_argument("--out", dest="output", required=True, type=Path)
    args = ap.parse_args(argv)

    n_in = n_out = 0
    counts: dict[str, int] = {}
    with args.input.open("r", encoding="utf-8") as f_in, args.output.open(
        "w", encoding="utf-8"
    ) as f_out:
        for line in f_in:
            line = line.strip()
            if not line:
                continue
            n_in += 1
            rec = json.loads(line)
            out = transform_line(rec)
            f_out.write(json.dumps(out) + "\n")
            n_out += 1
            d = out["signals"]["domain"]
            counts[d] = counts.get(d, 0) + 1
    print(f"in={n_in} out={n_out} domains={counts}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
