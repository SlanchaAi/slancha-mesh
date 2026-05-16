"""Traffic replay store — caches recent (prompt, oracle-response) pairs
for use as a LoRA fine-tune corpus.

Spec §7 paragraph 2: "Pulls recent traffic from slancha-api (oracled with
cloud responses or user feedback) for this node's primary specialist's
domain."

This module is the in-memory store side. It does not pull from
slancha-api itself; the ServeDaemon (v0.0.5) feeds it as inferences
complete + cloud-grader feedback lands.

Design:
- Bounded ring buffer by `max_size`; LRU eviction.
- Dedup by `prompt_hash` (SHA-256 of normalized prompt text). A second
  observation of the same prompt updates the response but preserves
  the original insertion slot.
- `recent(n)` returns most-recent N entries newest-first.
- Persistence to JSONL is optional via `dump(path)` / `load(path)` so
  daemon restarts don't drop the corpus.

Privacy posture (spec §11): the store lives on the node and never
leaves. SPARTA gossip in v0.0.6+ will share parameter deltas, not
raw prompts. Bookkeeping here is in-memory only by default.
"""

from __future__ import annotations

import hashlib
import json
import threading
from collections import OrderedDict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator


def _hash_prompt(text: str) -> str:
    """SHA-256 hex of normalized (stripped, lowercased) prompt text."""
    norm = text.strip().lower().encode("utf-8")
    return hashlib.sha256(norm).hexdigest()


@dataclass
class ReplayEntry:
    """One captured (prompt, oracle-response) pair with provenance."""

    prompt_hash: str
    prompt_text: str
    oracle_response: str
    domain: str
    difficulty: str
    captured_at: datetime
    # Optional: which specialist served this in production, so the
    # store can later be sliced by serving-spec for per-domain LoRA.
    served_by_specialist: str | None = None
    # Optional: per-token cost of the oracle response (helps prioritize
    # high-cost prompts for distillation).
    oracle_cost_usd: float | None = None

    def to_jsonl(self) -> str:
        d = asdict(self)
        d["captured_at"] = self.captured_at.isoformat()
        return json.dumps(d)

    @classmethod
    def from_jsonl(cls, line: str) -> "ReplayEntry":
        d = json.loads(line)
        d["captured_at"] = datetime.fromisoformat(d["captured_at"])
        return cls(**d)


class TrafficReplayStore:
    """In-memory bounded LRU store of (prompt, oracle-response) pairs.

    Thread-safe: backed by an internal lock; reads + writes from the
    serving path are serialized. Lock granularity is coarse (whole
    store); for the v0.0.4 traffic volume (≤1k entries) this is fine.
    Profile + shard if it ever bites.
    """

    def __init__(self, max_size: int = 1024) -> None:
        if max_size < 1:
            raise ValueError(f"max_size must be ≥1; got {max_size}")
        self._max_size = max_size
        self._entries: OrderedDict[str, ReplayEntry] = OrderedDict()
        self._lock = threading.Lock()

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)

    def add(
        self,
        prompt_text: str,
        oracle_response: str,
        domain: str,
        difficulty: str,
        served_by_specialist: str | None = None,
        oracle_cost_usd: float | None = None,
        captured_at: datetime | None = None,
    ) -> ReplayEntry:
        """Insert or update a (prompt, oracle) pair.

        Dedup by prompt_hash: a second `add` with the same normalized
        prompt updates fields but keeps the original captured_at + slot.
        New unique entries land at the MRU end; LRU is evicted when
        max_size is exceeded.
        """
        captured_at = captured_at or datetime.now(timezone.utc)
        h = _hash_prompt(prompt_text)
        with self._lock:
            if h in self._entries:
                existing = self._entries[h]
                # Update fields but keep original insertion slot
                existing.oracle_response = oracle_response
                existing.domain = domain
                existing.difficulty = difficulty
                if served_by_specialist is not None:
                    existing.served_by_specialist = served_by_specialist
                if oracle_cost_usd is not None:
                    existing.oracle_cost_usd = oracle_cost_usd
                self._entries.move_to_end(h)  # bump MRU
                return existing
            entry = ReplayEntry(
                prompt_hash=h,
                prompt_text=prompt_text,
                oracle_response=oracle_response,
                domain=domain,
                difficulty=difficulty,
                captured_at=captured_at,
                served_by_specialist=served_by_specialist,
                oracle_cost_usd=oracle_cost_usd,
            )
            self._entries[h] = entry
            # Evict LRU if over cap
            while len(self._entries) > self._max_size:
                self._entries.popitem(last=False)
            return entry

    def recent(self, n: int = 100, domain: str | None = None) -> list[ReplayEntry]:
        """Return up to N most-recent entries newest-first.

        Optionally filter by domain — used by per-domain LoRA training
        passes that only want corpus matching the loaded specialist.
        """
        if n < 0:
            raise ValueError("n must be ≥ 0")
        with self._lock:
            items = list(self._entries.values())
        # OrderedDict is insertion-order; MRU is last → reverse for newest-first
        items.reverse()
        if domain is not None:
            items = [e for e in items if e.domain == domain]
        return items[:n]

    def __iter__(self) -> Iterator[ReplayEntry]:
        """Snapshot iteration; safe for concurrent add."""
        with self._lock:
            snap = list(self._entries.values())
        return iter(snap)

    def dump(self, path: Path) -> int:
        """Write all entries as JSONL; returns count. Atomic via .tmp swap."""
        with self._lock:
            snap = list(self._entries.values())
        tmp = path.with_suffix(path.suffix + ".tmp")
        with tmp.open("w", encoding="utf-8") as f:
            for e in snap:
                f.write(e.to_jsonl() + "\n")
        tmp.replace(path)
        return len(snap)

    @classmethod
    def load(cls, path: Path, max_size: int = 1024) -> "TrafficReplayStore":
        """Read JSONL into a fresh store. Missing file → empty store."""
        store = cls(max_size=max_size)
        if not path.exists():
            return store
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                entry = ReplayEntry.from_jsonl(line)
                with store._lock:
                    store._entries[entry.prompt_hash] = entry
        # Trim if file had more than max_size
        with store._lock:
            while len(store._entries) > store._max_size:
                store._entries.popitem(last=False)
        return store


__all__ = ["ReplayEntry", "TrafficReplayStore"]
