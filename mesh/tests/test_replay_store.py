"""TrafficReplayStore tests — dedup, LRU, threading, JSONL round-trip."""

from __future__ import annotations

import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from mesh.replay_store import ReplayEntry, TrafficReplayStore, _hash_prompt


# ---------------------------------------------------------------------------
# Hash + entry
# ---------------------------------------------------------------------------


def test_hash_is_normalized():
    """Whitespace + case shouldn't produce different hashes."""
    a = _hash_prompt("  Hello, World!  ")
    b = _hash_prompt("hello, world!")
    assert a == b


def test_hash_differs_on_content():
    assert _hash_prompt("foo") != _hash_prompt("bar")


def test_entry_jsonl_roundtrip():
    now = datetime(2026, 5, 16, 12, 0, 0, tzinfo=timezone.utc)
    e = ReplayEntry(
        prompt_hash="abc",
        prompt_text="What is 2+2?",
        oracle_response="4",
        domain="math",
        difficulty="easy",
        captured_at=now,
        served_by_specialist="qwen3-math-7b-q4",
        oracle_cost_usd=0.0001,
    )
    line = e.to_jsonl()
    back = ReplayEntry.from_jsonl(line)
    assert back == e


# ---------------------------------------------------------------------------
# Store basics
# ---------------------------------------------------------------------------


def test_store_init_rejects_zero_size():
    with pytest.raises(ValueError):
        TrafficReplayStore(max_size=0)


def test_store_add_returns_entry():
    s = TrafficReplayStore(max_size=10)
    e = s.add("hello", "world", "general", "easy")
    assert isinstance(e, ReplayEntry)
    assert e.prompt_text == "hello"
    assert e.oracle_response == "world"
    assert len(s) == 1


def test_store_dedup_updates_in_place():
    s = TrafficReplayStore(max_size=10)
    e1 = s.add("same", "response-v1", "code", "medium")
    e2 = s.add("Same", "response-v2", "code", "medium")  # case-insensitive dedup
    assert len(s) == 1
    assert e1.prompt_hash == e2.prompt_hash
    # The response should have been updated
    assert e2.oracle_response == "response-v2"
    # captured_at preserved from first insert
    assert e2.captured_at == e1.captured_at


def test_store_lru_eviction_on_overflow():
    s = TrafficReplayStore(max_size=3)
    s.add("a", "a-r", "code", "easy")
    s.add("b", "b-r", "code", "easy")
    s.add("c", "c-r", "code", "easy")
    s.add("d", "d-r", "code", "easy")  # evicts "a"
    hashes = {e.prompt_hash for e in s}
    assert _hash_prompt("a") not in hashes
    assert _hash_prompt("d") in hashes
    assert len(s) == 3


def test_store_dedup_bumps_mru_no_eviction():
    s = TrafficReplayStore(max_size=3)
    s.add("a", "a-r", "code", "easy")
    s.add("b", "b-r", "code", "easy")
    s.add("c", "c-r", "code", "easy")
    s.add("a", "a-r-v2", "code", "easy")  # bump A to MRU
    s.add("d", "d-r", "code", "easy")  # should evict B (LRU after the bump)
    hashes = {e.prompt_hash for e in s}
    assert _hash_prompt("a") in hashes
    assert _hash_prompt("b") not in hashes


# ---------------------------------------------------------------------------
# recent() ordering + filtering
# ---------------------------------------------------------------------------


def test_recent_returns_newest_first():
    s = TrafficReplayStore(max_size=10)
    s.add("p1", "r1", "code", "easy")
    s.add("p2", "r2", "code", "easy")
    s.add("p3", "r3", "code", "easy")
    out = s.recent(n=10)
    assert [e.prompt_text for e in out] == ["p3", "p2", "p1"]


def test_recent_caps_n():
    s = TrafficReplayStore(max_size=10)
    for i in range(5):
        s.add(f"p{i}", f"r{i}", "code", "easy")
    out = s.recent(n=2)
    assert len(out) == 2
    assert [e.prompt_text for e in out] == ["p4", "p3"]


def test_recent_domain_filter():
    s = TrafficReplayStore(max_size=10)
    s.add("p1", "r1", "math", "easy")
    s.add("p2", "r2", "code", "easy")
    s.add("p3", "r3", "math", "easy")
    out = s.recent(n=10, domain="math")
    assert [e.prompt_text for e in out] == ["p3", "p1"]
    out_code = s.recent(n=10, domain="code")
    assert [e.prompt_text for e in out_code] == ["p2"]


def test_recent_n_zero_returns_empty():
    s = TrafficReplayStore(max_size=10)
    s.add("p1", "r1", "code", "easy")
    assert s.recent(n=0) == []


def test_recent_n_negative_raises():
    s = TrafficReplayStore(max_size=10)
    with pytest.raises(ValueError):
        s.recent(n=-1)


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def test_dump_and_load_roundtrip(tmp_path: Path):
    s = TrafficReplayStore(max_size=10)
    s.add("p1", "r1", "code", "easy", served_by_specialist="qwen3-coder-7b-q4")
    s.add("p2", "r2", "math", "medium", oracle_cost_usd=0.005)
    path = tmp_path / "replay.jsonl"
    n_dumped = s.dump(path)
    assert n_dumped == 2

    loaded = TrafficReplayStore.load(path, max_size=10)
    assert len(loaded) == 2
    entries = list(loaded)
    by_text = {e.prompt_text: e for e in entries}
    assert by_text["p1"].served_by_specialist == "qwen3-coder-7b-q4"
    assert by_text["p2"].oracle_cost_usd == 0.005


def test_load_missing_file_returns_empty(tmp_path: Path):
    s = TrafficReplayStore.load(tmp_path / "nonexistent.jsonl")
    assert len(s) == 0


def test_load_trims_to_max_size(tmp_path: Path):
    """A persisted file with N > max_size entries should load trimmed."""
    s_big = TrafficReplayStore(max_size=100)
    for i in range(10):
        s_big.add(f"p{i}", f"r{i}", "code", "easy")
    path = tmp_path / "big.jsonl"
    s_big.dump(path)
    s_small = TrafficReplayStore.load(path, max_size=3)
    assert len(s_small) == 3


# ---------------------------------------------------------------------------
# Concurrency smoke
# ---------------------------------------------------------------------------


def test_concurrent_adds_dont_corrupt_store():
    """100 threads each add 100 unique entries → no exceptions,
    final state is consistent."""
    s = TrafficReplayStore(max_size=20000)
    errors: list[Exception] = []

    def worker(tid: int):
        try:
            for i in range(100):
                s.add(f"thread-{tid}-{i}", "r", "code", "easy")
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=worker, args=(t,)) for t in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=5)
    assert errors == []
    assert len(s) == 1000  # 10 threads * 100 entries, all unique
