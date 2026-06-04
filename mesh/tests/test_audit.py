"""Stress tests for mesh.audit — each encodes a security/SRE/architect finding.

The persona review of the design is operationalized here: tamper-evidence, chain
integrity, concurrency, injection/size safety, actor validation, and the
sink-must-never-break-promotion ordering.
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from pathlib import Path

import pytest

from mesh.audit import (
    MAX_ROW_BYTES,
    AuditError,
    AuditRecorder,
    verify_chain,
)


@dataclass
class _Verdict:
    """Minimal stand-in with the .to_row() the recorder consumes."""

    accept: bool = True
    mean_delta: float = 0.3

    def to_row(self) -> dict:
        return {"accept": self.accept, "mean_delta": self.mean_delta}


def _log(tmp_path: Path) -> Path:
    return tmp_path / "audit.jsonl"


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


# ───────────────────────── chain integrity / tamper-evidence ─────────────────


def test_records_and_verifies_clean_chain(tmp_path: Path):
    rec = AuditRecorder(_log(tmp_path))
    for _ in range(5):
        rec.record(_Verdict())
    ok, err = verify_chain(_log(tmp_path))
    assert ok, err
    rows = _rows(_log(tmp_path))
    assert [r["seq"] for r in rows] == [0, 1, 2, 3, 4]
    assert all(r["schema_version"] == "audit-v1" for r in rows)
    # Server-stamped, not caller-supplied.
    assert all("ts" in r and r["ts"].endswith("Z") for r in rows)


def test_editing_a_row_breaks_verification(tmp_path: Path):
    """SECURITY: a row edited in place is detected (row_hash mismatch)."""
    rec = AuditRecorder(_log(tmp_path))
    rec.record(_Verdict(mean_delta=0.3))
    rec.record(_Verdict(mean_delta=0.4))
    rows = _rows(_log(tmp_path))
    rows[0]["mean_delta"] = 9.9  # tamper, but keep the (now-stale) row_hash
    _log(tmp_path).write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    ok, err = verify_chain(_log(tmp_path))
    assert not ok and "row_hash mismatch" in err


def test_deleting_a_row_breaks_the_chain(tmp_path: Path):
    """SECURITY: a removed row is detected (broken prev-hash link / seq gap)."""
    rec = AuditRecorder(_log(tmp_path))
    for _ in range(3):
        rec.record(_Verdict())
    rows = _rows(_log(tmp_path))
    del rows[1]  # remove the middle row
    _log(tmp_path).write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    ok, err = verify_chain(_log(tmp_path))
    assert not ok and ("broken chain" in err or "seq gap" in err)


def test_chain_resumes_across_recorder_restart(tmp_path: Path):
    """A new recorder over the same file continues seq + chain (durable restart)."""
    AuditRecorder(_log(tmp_path)).record(_Verdict())
    AuditRecorder(_log(tmp_path)).record(_Verdict())  # fresh instance, same file
    rec3 = AuditRecorder(_log(tmp_path))
    rec3.record(_Verdict())
    ok, err = verify_chain(_log(tmp_path))
    assert ok, err
    assert [r["seq"] for r in _rows(_log(tmp_path))] == [0, 1, 2]


# ───────────────────────────── concurrency ──────────────────────────────────


def test_concurrent_writers_keep_chain_intact(tmp_path: Path):
    """SRE: many threads on one recorder → no interleaving/corruption, chain ok."""
    rec = AuditRecorder(_log(tmp_path))
    n_threads, per = 8, 25

    def worker():
        for _ in range(per):
            rec.record(_Verdict())

    threads = [threading.Thread(target=worker) for _ in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    ok, err = verify_chain(_log(tmp_path))
    assert ok, err
    rows = _rows(_log(tmp_path))
    assert len(rows) == n_threads * per
    assert sorted(r["seq"] for r in rows) == list(range(n_threads * per))


# ─────────────────────── injection / size safety ────────────────────────────


def test_newline_in_actor_or_context_cannot_inject_a_row(tmp_path: Path):
    """SECURITY: a newline in actor/context stays inside one JSON line."""
    rec = AuditRecorder(_log(tmp_path))
    rec.record(
        _Verdict(),
        actor="evil\n{\"seq\": 999, \"injected\": true}",
        context={"note": "line1\nline2"},
    )
    # Exactly one physical line, and the chain still verifies.
    assert len([ln for ln in _log(tmp_path).read_text().splitlines() if ln.strip()]) == 1
    ok, err = verify_chain(_log(tmp_path))
    assert ok, err


def test_oversized_context_is_rejected_and_nothing_written(tmp_path: Path):
    """SECURITY: an oversized row is rejected before the file write (no partial)."""
    rec = AuditRecorder(_log(tmp_path))
    big = {"blob": "x" * (MAX_ROW_BYTES + 1)}
    with pytest.raises(AuditError, match="MAX_ROW_BYTES"):
        rec.record(_Verdict(), context=big)
    assert not _log(tmp_path).exists() or _rows(_log(tmp_path)) == []


# ───────────────────────────── actor validation ─────────────────────────────


@pytest.mark.parametrize("bad", ["", "   ", "None", "null", "NULL"])
def test_bad_actor_rejected(tmp_path: Path, bad: str):
    rec = AuditRecorder(_log(tmp_path))
    with pytest.raises(AuditError, match="invalid actor"):
        rec.record(_Verdict(), actor=bad)


def test_actor_is_normalized_and_marked_asserted(tmp_path: Path):
    rec = AuditRecorder(_log(tmp_path))
    # Full-width chars NFKC-normalize to ASCII.
    row = rec.record(_Verdict(), actor="ａlice")  # "ａlice" -> "alice"
    assert row["actor"] == "alice"
    assert row["actor_asserted"] is True


# ───────────────── sink: tee, never blocks/fails the promotion ───────────────


def test_sink_receives_each_row(tmp_path: Path):
    seen = []

    class _Sink:
        def emit(self, event):
            seen.append(event)

    rec = AuditRecorder(_log(tmp_path), sink=_Sink())
    rec.record(_Verdict())
    rec.record(_Verdict())
    assert [e["seq"] for e in seen] == [0, 1]


def test_sink_failure_never_breaks_the_promotion_record(tmp_path: Path):
    """SRE: local write is source of truth; a raising sink is caught, the row is
    still durable, and record() returns normally."""
    errors = []

    class _BoomSink:
        def emit(self, event):
            raise RuntimeError("audit store unreachable")

    rec = AuditRecorder(
        _log(tmp_path),
        sink=_BoomSink(),
        on_sink_error=lambda exc, row: errors.append((type(exc).__name__, row["seq"])),
    )
    row = rec.record(_Verdict())          # must NOT raise
    assert row["seq"] == 0
    ok, err = verify_chain(_log(tmp_path))  # local record is durable + valid
    assert ok, err
    assert errors == [("RuntimeError", 0)]  # failure surfaced, not swallowed silently


# ───────────────────────── keyed chain + external anchor (#100) ──────────────
def test_keyed_chain_verifies_with_key_and_rejects_without(tmp_path: Path):
    key = b"audit-hmac-key-held-outside-the-db"
    rec = AuditRecorder(_log(tmp_path), hmac_key=key)
    for _ in range(3):
        rec.record(_Verdict())
    assert verify_chain(_log(tmp_path), hmac_key=key) == (True, None)
    # Verifying a KEYED log without the key (or wrong key) must fail.
    ok, err = verify_chain(_log(tmp_path))
    assert not ok
    ok, err = verify_chain(_log(tmp_path), hmac_key=b"wrong-key")
    assert not ok


def test_keyed_chain_defeats_rechain_attack(tmp_path: Path):
    """A file-level attacker edits a row and naively re-hashes with plain SHA-256;
    the keyed verify recomputes HMAC and catches it."""
    import hashlib
    import json as _json

    key = b"k"
    rec = AuditRecorder(_log(tmp_path), hmac_key=key)
    for _ in range(3):
        rec.record(_Verdict())
    rows = _rows(_log(tmp_path))
    rows[1]["mean_delta"] = 9.9  # tamper
    body = {k: v for k, v in rows[1].items() if k != "row_hash"}
    canon = _json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    rows[1]["row_hash"] = "sha256:" + hashlib.sha256(canon.encode()).hexdigest()  # attacker w/o key
    _log(tmp_path).write_text("\n".join(_json.dumps(r) for r in rows) + "\n")
    ok, err = verify_chain(_log(tmp_path), hmac_key=key)
    assert not ok and "row_hash mismatch" in err


def test_external_anchor_detects_truncation(tmp_path: Path):
    rec = AuditRecorder(_log(tmp_path), hmac_key=b"k")
    for _ in range(4):
        rec.record(_Verdict())
    last_seq, tip = rec.tip
    assert last_seq == 3
    # Truncate the tail (a valid prefix still verifies WITHOUT the anchor).
    rows = _rows(_log(tmp_path))[:2]
    import json as _json
    _log(tmp_path).write_text("\n".join(_json.dumps(r) for r in rows) + "\n")
    assert verify_chain(_log(tmp_path), hmac_key=b"k")[0] is True            # blind to truncation
    # With the external anchor it is caught.
    ok, err = verify_chain(_log(tmp_path), hmac_key=b"k", expected_tip=tip)
    assert not ok and "tip" in err
    ok, err = verify_chain(_log(tmp_path), hmac_key=b"k", expected_count=4)
    assert not ok and "truncated" in err
