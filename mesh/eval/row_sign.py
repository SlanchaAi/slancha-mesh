"""HMAC row-signing for eval_results.jsonl (#104).

The promotion gate reads fields (notably `meta_stub`) straight off an eval row in
`eval_results.jsonl` — a plain append-only file with no integrity protection. A
local writer could append a crafted row (`meta_stub: false`, inflated scores) to
push a challenger through the gate. When a key is configured, each row carries a
keyed HMAC over its content and the gate refuses a row that is missing or whose
HMAC doesn't verify (`--require-signed-rows`).

The key is opaque bytes the operator supplies (`SLANCHA_EVAL_HMAC_KEY`); in the
Kanpai distribution it is resolved from Vault (`kanpai.secrets`) so it lives
outside the filesystem an attacker would write to. Mirrors the keyed audit chain
(`mesh.audit`); same custody rule (key outside the writable store)."""

from __future__ import annotations

import hashlib
import hmac
import json
from typing import Any

_HMAC_FIELD = "row_hmac"


def _canonical(row: dict[str, Any]) -> bytes:
    """Deterministic encoding of the row WITHOUT its hmac field (sorted keys)."""
    body = {k: v for k, v in row.items() if k != _HMAC_FIELD}
    return json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def eval_row_hmac(row: dict[str, Any], key: bytes) -> str:
    return "hmac-sha256:" + hmac.new(key, _canonical(row), hashlib.sha256).hexdigest()


def verify_eval_row(row: dict[str, Any], key: bytes) -> bool:
    """True iff the row carries a valid keyed HMAC over its content."""
    stored = row.get(_HMAC_FIELD)
    if not isinstance(stored, str):
        return False
    return hmac.compare_digest(stored, eval_row_hmac(row, key))


__all__ = ["eval_row_hmac", "verify_eval_row"]
