"""Mesh-replay harness — drive a pre-classified prompt corpus through the mesh.

Reads a JSONL corpus of `{prompt_id, signals: {domain, difficulty, language,
needs_tools, route_class}, ...}` lines, fetches a `RegistrySnapshot` from
a running mesh service, and routes each prompt through `select_mesh_route()`.
Emits one JSONL decision line per prompt.

Purpose: feed into the streamlit dashboard on Spark (post-sprint) to render
mesh-hit-rate, fallback-chain shapes, queue distributions per route class.

Usage:
    python -m mesh.scripts.mesh_replay \\
        --corpus  ./prompts.jsonl \\
        --registry-url http://localhost:8088 \\
        --output  ./replay.jsonl \\
        [--token TOKEN] \\
        [--snapshot-refresh-every 25]

Corpus line shape (extra keys ignored):
    {"prompt_id": "p-001",
     "prompt_text": "...",          # optional; hashed if present
     "signals": {"domain": "code", "difficulty": "medium",
                 "language": "en",   "route_class": "standard",
                 "needs_tools": false}}

Output line shape:
    {"prompt_id": "p-001",
     "prompt_hash": "sha256:...",   # null if no prompt_text
     "signals": {...},               # echoed verbatim
     "decision": {
        "chosen_specialist": "qwen3-coder-30b-a3b-fp8" | null,
        "chosen_node":       "spark-1" | null,
        "node_url":          "http://..." | null,
        "reason":            "mesh: ..."
        "queue_ms":          120,
        "fallback_chain":    [["model_id", "node_id" | null], ...],
        "mesh_hit":          true | false,
        "vs_cloud_baseline_cost": 0.0   # placeholder; spark fills post-sprint
     },
     "snapshot_ts": "2026-05-16T15:00:00Z"}
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator

import httpx

from mesh.models import RegistrySnapshot
from mesh.select import ClassifierSignals, select_mesh_route


# ---------------------------------------------------------------------------
# Snapshot fetch
# ---------------------------------------------------------------------------


def fetch_snapshot(
    base_url: str,
    token: str | None = None,
    timeout: float = 10.0,
) -> RegistrySnapshot:
    """GET `{base_url}/registry` and parse the response into a RegistrySnapshot."""
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    resp = httpx.get(f"{base_url.rstrip('/')}/registry", headers=headers, timeout=timeout)
    resp.raise_for_status()
    snap_dict = resp.json()["snapshot"]
    return RegistrySnapshot.model_validate(snap_dict)


# ---------------------------------------------------------------------------
# Corpus I/O
# ---------------------------------------------------------------------------


@dataclass
class CorpusLine:
    prompt_id: str
    prompt_text: str | None
    signals: ClassifierSignals


def iter_corpus(path: Path) -> Iterator[CorpusLine]:
    with path.open("r", encoding="utf-8") as f:
        for i, raw in enumerate(f, start=1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                rec = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"corpus line {i}: invalid JSON ({exc})") from exc
            sig_dict = rec.get("signals", {})
            if "domain" not in sig_dict or "difficulty" not in sig_dict:
                raise ValueError(
                    f"corpus line {i}: signals must include domain + difficulty"
                )
            signals = ClassifierSignals(
                domain=sig_dict["domain"],
                difficulty=sig_dict["difficulty"],
                language=sig_dict.get("language", "en"),
                needs_tools=sig_dict.get("needs_tools", False),
                route_class=sig_dict.get("route_class", "standard"),
            )
            yield CorpusLine(
                prompt_id=str(rec.get("prompt_id", f"line-{i}")),
                prompt_text=rec.get("prompt_text"),
                signals=signals,
            )


def _prompt_hash(text: str | None) -> str | None:
    if text is None:
        return None
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Replay engine
# ---------------------------------------------------------------------------


def replay_one(
    line: CorpusLine,
    snapshot: RegistrySnapshot,
    cloud_fallback: str,
) -> dict[str, Any]:
    """Route one prompt and produce the JSONL output dict."""
    result = select_mesh_route(
        signals=line.signals,
        registry_snapshot=snapshot,
        cloud_fallback_model=cloud_fallback,
    )
    return {
        "prompt_id": line.prompt_id,
        "prompt_hash": _prompt_hash(line.prompt_text),
        "signals": asdict(line.signals),
        "decision": {
            "chosen_specialist": result.specialist_id,
            "chosen_node": result.node_id,
            "node_url": result.node_url,
            "model": result.model,
            "reason": result.reason,
            "queue_ms": result.queue_ms_estimated,
            "fallback_chain": [list(p) for p in result.fallback_chain],
            "mesh_hit": result.node_id is not None,
            "vs_cloud_baseline_cost": 0.0,
        },
        "snapshot_ts": snapshot.snapshot_ts.isoformat(),
    }


def replay_corpus(
    corpus_path: Path,
    output_path: Path,
    base_url: str,
    token: str | None = None,
    cloud_fallback: str = "claude-sonnet-4-7",
    snapshot_refresh_every: int = 25,
) -> dict[str, int]:
    """Drive the full corpus through the mesh; emit JSONL to `output_path`.

    Refreshes the registry snapshot every `snapshot_refresh_every` prompts
    so a long replay reflects live registry changes (heartbeats expire,
    new nodes appear) without round-tripping per prompt.

    Returns a counters dict: {processed, mesh_hits, cloud_fallbacks}.
    """
    processed = 0
    mesh_hits = 0
    snapshot: RegistrySnapshot | None = None

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as out:
        for line in iter_corpus(corpus_path):
            if snapshot is None or processed % snapshot_refresh_every == 0:
                snapshot = fetch_snapshot(base_url, token=token)
            rec = replay_one(line, snapshot, cloud_fallback)
            out.write(json.dumps(rec) + "\n")
            processed += 1
            if rec["decision"]["mesh_hit"]:
                mesh_hits += 1

    return {
        "processed": processed,
        "mesh_hits": mesh_hits,
        "cloud_fallbacks": processed - mesh_hits,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Replay a prompt corpus through the mesh router.")
    ap.add_argument("--corpus", type=Path, required=True, help="Input JSONL corpus path.")
    ap.add_argument("--output", type=Path, required=True, help="Output JSONL decisions path.")
    ap.add_argument(
        "--registry-url",
        default="http://localhost:8088",
        help="Base URL of the mesh service (e.g., http://mesh-mac.laulpogan.com).",
    )
    ap.add_argument("--token", default=None, help="Bearer token for the mesh service (optional).")
    ap.add_argument(
        "--cloud-fallback",
        default="claude-sonnet-4-7",
        help="Model id returned when no mesh route matches.",
    )
    ap.add_argument(
        "--snapshot-refresh-every",
        type=int,
        default=25,
        help="Re-fetch /registry every N prompts (default 25).",
    )
    args = ap.parse_args(argv)

    if not args.corpus.exists():
        print(f"corpus not found: {args.corpus}", file=sys.stderr)
        return 2

    started = time.time()
    counters = replay_corpus(
        corpus_path=args.corpus,
        output_path=args.output,
        base_url=args.registry_url,
        token=args.token,
        cloud_fallback=args.cloud_fallback,
        snapshot_refresh_every=args.snapshot_refresh_every,
    )
    elapsed = time.time() - started
    print(
        f"replay complete: {counters['processed']} prompts, "
        f"{counters['mesh_hits']} mesh-hit, "
        f"{counters['cloud_fallbacks']} cloud-fallback, "
        f"{elapsed:.2f}s wall, "
        f"output={args.output}",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
