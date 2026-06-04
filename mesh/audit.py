"""Tamper-evident, append-only audit recorder — the audit spine (opt-in).

`mesh.eval.gate.append_verdict` records promotion verdicts as plain append-only
JSONL — fine for a dashboard, but not a defensible audit trail for a regulated
review (no ordering proof, no tamper-evidence, no actor). This module is the
hardened, OPT-IN path that turns the same event stream into one:

  • monotonic `seq` + server-side UTC `ts` (caller can't forge ordering/time);
  • a row-level **hash chain** (`prev_row_hash` → `row_hash`) so a deleted or
    edited row is detectable — distinct from the model-lineage SHAs the verdict
    already carries (those prove WHAT was promoted; the chain proves the LOG
    itself wasn't rewritten);
  • canonical serialization (sorted keys) so hashes are stable cross-platform;
  • a size cap + newline-safety so a row can't inject phantom log lines or wedge
    the file mid-write;
  • `actor` recorded as **asserted, not verified** (identity ENFORCEMENT — SSO /
    RBAC — is a downstream layer's job; this only records what it's told and says
    so), NFKC-normalized, empty/`none` rejected;
  • an `AuditSink` seam: each row is teed to an injected sink (a downstream
    immutable/central audit store) AFTER the local durable write — sink failures
    are caught + logged and NEVER block or fail the promotion record.

Design notes (from a security/SRE/architect review):
  • The local JSONL is the SOURCE OF TRUTH; the sink is a tee. Ordering is always
    local-write-then-sink. A local IOError propagates (a dropped audit record is
    not acceptable); a sink error never propagates (an audit-store outage must not
    halt promotions — reconcile from the local log instead).
  • Single-writer per file: the hash chain + seq are inherently single-writer.
    One process owns an audit log; a per-instance lock makes it thread-safe.
    Multi-process / multi-tenant fan-in is the sink's job (a central store).
  • `append_verdict` is intentionally NOT changed — this is additive and opt-in,
    so existing readers and the default gate path are untouched.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import threading
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

AUDIT_SCHEMA_VERSION = "audit-v1"
# Hard cap on a serialized row. A row over this is rejected (not truncated —
# truncation would corrupt meaning); keeps a runaway/oversized `context` from
# filling disk or wedging the file mid-write.
MAX_ROW_BYTES = 64 * 1024
# Genesis link for the first row in a fresh log.
_GENESIS = "sha256:" + "0" * 64


class AuditError(ValueError):
    """A row was rejected before it could corrupt the log (oversized, bad actor,
    or — defensively — a newline that survived serialization)."""


class AuditSink(Protocol):
    """Downstream tee for audit events (e.g. an immutable/central audit store).

    One method. Receives the fully-formed, hash-chained row as a mapping. The
    row carries `schema_version`, so a sink stays forward-compatible as the row
    gains fields — bind to the keys you need, ignore the rest. The sink MUST be
    fast / non-blocking (run its own async/queue if it talks to a network store):
    the recorder calls it synchronously on the promotion path and will catch —
    but cannot time out — a blocking sink.
    """

    def emit(self, event: dict[str, Any]) -> None: ...


def _canonical(row: dict[str, Any]) -> str:
    """Deterministic serialization for hashing + writing: sorted keys, compact,
    UTF-8 preserved. Stable across Python versions/platforms."""
    return json.dumps(row, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _row_hash(row_without_hash: dict[str, Any], hmac_key: bytes | None = None) -> str:
    """Chain link for a row. Plain SHA-256 by default (back-compat / OSS); a
    KEYED HMAC-SHA256 when an ``hmac_key`` is configured (#100).

    Why keyed matters: a plain hash is recomputable by anyone, so a DB/file-level
    attacker can edit a row and re-chain the whole tail undetected. With a key the
    attacker can't forge a valid ``row_hash`` for an edited row, so ``verify_chain``
    (run with the same key, held OUTSIDE the DB admin's reach) catches the edit."""
    data = _canonical(row_without_hash).encode("utf-8")
    if hmac_key is not None:
        return "hmac-sha256:" + hmac.new(hmac_key, data, hashlib.sha256).hexdigest()
    return "sha256:" + hashlib.sha256(data).hexdigest()


def _clean_actor(actor: str) -> str:
    """NFKC-normalize + validate an asserted actor identity. Rejects empty and
    the literal `none`/`null` (case-insensitive) so a missing identity can't
    masquerade as a real one."""
    a = unicodedata.normalize("NFKC", actor).strip()
    if not a or a.lower() in {"none", "null"}:
        raise AuditError(f"invalid actor identity: {actor!r}")
    return a


class AuditRecorder:
    """Append-only, hash-chained audit recorder for one log file.

    Construct once per log (single writer); call :meth:`record` per event. On
    construction it resumes the chain from the existing file (so restarts keep a
    continuous `seq` + hash chain). Thread-safe within the owning process.
    """

    def __init__(
        self,
        output: Path,
        *,
        hmac_key: bytes | None = None,
        sink: AuditSink | None = None,
        on_sink_error: Any = None,
    ) -> None:
        self.output = Path(output)
        self.output.parent.mkdir(parents=True, exist_ok=True)
        # When set, the chain is keyed (HMAC) — tamper-resistant against an
        # attacker who can write the file/DB but does NOT hold the key. Keep the
        # key outside the audit store's admin domain (HSM / KMS / separate secret).
        self._hmac_key = hmac_key
        self._sink = sink
        # Called as on_sink_error(exc, row) when a sink raises; defaults to a
        # stderr warning. Never re-raises into the promotion path.
        self._on_sink_error = on_sink_error or _default_sink_error
        self._lock = threading.Lock()
        self._seq, self._prev = self._resume()

    @property
    def tip(self) -> tuple[int, str]:
        """``(last_seq, last_row_hash)`` — the chain head. Persist this to an
        EXTERNAL witness (a store the DB admin can't rewrite) and pass it to
        ``verify_chain(expected_tip=..., expected_count=...)`` to detect a
        truncation of the tail (which an internal-only chain cannot catch)."""
        return self._seq - 1, self._prev

    def _resume(self) -> tuple[int, str]:
        """Read the existing log's tail to continue seq + chain; (0, genesis) for
        a fresh/empty/header-less file."""
        if not self.output.exists():
            return 0, _GENESIS
        last = None
        with self.output.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    last = line
        if last is None:
            return 0, _GENESIS
        row = json.loads(last)
        return int(row["seq"]) + 1, str(row["row_hash"])

    def record(
        self,
        verdict: Any,
        *,
        actor: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Record one verdict as a tamper-evident, chained audit row.

        Returns the written row. Raises AuditError if the row is rejected
        (oversized / bad actor) and propagates an IOError if the local write
        fails (a dropped audit record is never acceptable). A sink failure is
        caught + reported via `on_sink_error`, never raised.
        """
        # Build the row (verdict payload + audit envelope). Server stamps seq +
        # ts; the caller cannot influence ordering or time.
        with self._lock:
            row: dict[str, Any] = dict(verdict.to_row())
            row["schema_version"] = AUDIT_SCHEMA_VERSION
            row["seq"] = self._seq
            row["ts"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
            if actor is not None:
                row["actor"] = _clean_actor(actor)
                row["actor_asserted"] = True  # recorded, NOT cryptographically verified
            if context is not None:
                row["audit_context"] = context
            row["prev_row_hash"] = self._prev
            row["row_hash"] = _row_hash(row, self._hmac_key)

            line = _canonical(row)
            if "\n" in line or "\r" in line:  # defense; json.dumps escapes these
                raise AuditError("serialized row contains a newline; refusing to write")
            if len(line.encode("utf-8")) > MAX_ROW_BYTES:
                raise AuditError(
                    f"audit row {len(line)} bytes exceeds MAX_ROW_BYTES "
                    f"{MAX_ROW_BYTES} (oversized context?)"
                )

            # Local durable write FIRST (source of truth). O_APPEND + flush; a
            # single short line write is atomic at the OS level under the lock.
            with self.output.open("a", encoding="utf-8") as f:
                f.write(line + "\n")
                f.flush()

            # Commit in-memory chain state only after the durable write.
            self._seq += 1
            self._prev = row["row_hash"]

        # Tee to the downstream sink OUTSIDE the lock and AFTER the local write.
        # A sink failure must never block or fail the (already-durable) record.
        if self._sink is not None:
            try:
                self._sink.emit(dict(row))
            except Exception as exc:  # noqa: BLE001 - audit tee must not break promotion
                self._on_sink_error(exc, row)
        return row


def _default_sink_error(exc: Exception, row: dict[str, Any]) -> None:
    import sys

    print(
        f"[audit] sink.emit failed for seq={row.get('seq')} "
        f"({type(exc).__name__}: {exc}); row is durable locally, reconcile from "
        f"the log: {row.get('row_hash')}",
        file=sys.stderr,
    )


def verify_chain(
    path: Path,
    *,
    hmac_key: bytes | None = None,
    expected_tip: str | None = None,
    expected_count: int | None = None,
) -> tuple[bool, str | None]:
    """Verify an audit log's integrity end-to-end.

    Returns (True, None) if every row's `row_hash` recomputes, the chain links
    (`prev_row_hash` == prior `row_hash`), and `seq` increments by one with no
    gaps. Otherwise (False, "<reason at seq N>"). This is what an auditor (or a
    test) runs to prove the log was not edited or had rows removed.

    Pass ``hmac_key`` to verify a KEYED chain (must match the recorder's key).

    TRUNCATION (#100): an internal chain can't detect that the *tail* was deleted
    — a valid prefix still verifies. Pass ``expected_tip`` (the last `row_hash`)
    and/or ``expected_count`` (the row count) from an EXTERNAL witness to catch a
    truncated or rewound log.
    """
    if not path.exists():
        return False, "log does not exist"
    prev = _GENESIS
    expected_seq = 0
    saw_any = False
    last_hash = _GENESIS
    with path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            saw_any = True
            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                return False, f"line {lineno}: malformed JSON ({e})"
            stored = row.get("row_hash")
            recomputed = _row_hash({k: v for k, v in row.items() if k != "row_hash"}, hmac_key)
            if stored != recomputed:
                return False, f"seq {row.get('seq')}: row_hash mismatch (row edited or wrong key)"
            if row.get("prev_row_hash") != prev:
                return False, f"seq {row.get('seq')}: broken chain (row inserted/removed)"
            if row.get("seq") != expected_seq:
                return False, f"line {lineno}: seq gap (expected {expected_seq}, got {row.get('seq')})"
            prev = stored
            last_hash = stored
            expected_seq += 1
    if not saw_any:
        return False, "log is empty"
    if expected_count is not None and expected_seq != expected_count:
        return False, f"row count {expected_seq} != expected {expected_count} (log truncated/extended)"
    if expected_tip is not None and last_hash != expected_tip:
        return False, "chain tip != expected (log truncated or rewound)"
    return True, None


__all__ = [
    "AUDIT_SCHEMA_VERSION",
    "MAX_ROW_BYTES",
    "AuditError",
    "AuditRecorder",
    "AuditSink",
    "verify_chain",
]
