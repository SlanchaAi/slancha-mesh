"""Verify a held-out seed before any eval pass reads it.

Why this exists: the held-out set is the **single promotion guard**
(see SELF_ORGANIZING_LOOP_SCOPE invariant #5). The security persona's
finding is explicit: "the holdout seed must be curated/trusted (never
auto-derived from possibly-poisoned traffic)". Today the seed is
sha256-pinned in its manifest, but nothing actually refuses to load a
tampered jsonl. This module closes that gap.

Two layers:

  1. **sha256 check (always on).** Re-hash the JSONL the same way
     `write_holdout` did and refuse-load on mismatch. Cheap, no key
     material, catches accidental edits + most casual tampering.

  2. **detached ed25519 signature (opt-in).** When `require_signature` is
     True the manifest must carry a `signature` block
     {signer_did, sig_b64} over the canonical manifest minus the
     signature field, and the caller-supplied `trusted_signers` set must
     contain `signer_did`. Composes with the wire identity the operator
     already has — no new key material.

The runner consults this; nothing else in the repo needs to change.
"""

from __future__ import annotations

import base64
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class SeedVerificationError(Exception):
    """Raised when a seed manifest or its JSONL fails verification."""


@dataclass(frozen=True)
class VerifiedSeed:
    """A held-out seed whose integrity has been confirmed."""

    records: list[dict[str, Any]]
    manifest: dict[str, Any]
    jsonl_path: Path
    manifest_path: Path


def _canonical_manifest_bytes(manifest: dict[str, Any]) -> bytes:
    """Canonicalize a manifest for signing/verifying.

    Strips the `signature` field, then JSON-encodes with sorted keys and
    no whitespace, UTF-8. Matches the convention used elsewhere in the
    wire/mesh repos for detached signatures over JSON.
    """
    canon = {k: v for k, v in manifest.items() if k != "signature"}
    return json.dumps(canon, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def _verify_ed25519(
    signer_did: str,
    sig_b64: str,
    payload: bytes,
    trusted_signers: dict[str, bytes],
) -> None:
    """Verify a detached ed25519 signature.

    `trusted_signers` maps signer_did -> 32-byte raw ed25519 public key.
    Raises SeedVerificationError on any failure. nacl is imported
    lazily so this module stays import-clean for callers who don't use
    the signature path.
    """
    if signer_did not in trusted_signers:
        raise SeedVerificationError(
            f"signer not in trusted set: {signer_did!r}"
        )
    try:
        from nacl.exceptions import BadSignatureError
        from nacl.signing import VerifyKey
    except ImportError as exc:  # pragma: no cover - optional dep
        raise SeedVerificationError(
            "ed25519 verification requires PyNaCl; install or pass "
            "require_signature=False"
        ) from exc
    try:
        sig = base64.b64decode(sig_b64.encode("ascii"), validate=True)
    except Exception as exc:
        raise SeedVerificationError(f"signature is not valid base64: {exc}") from exc
    try:
        VerifyKey(trusted_signers[signer_did]).verify(payload, sig)
    except BadSignatureError as exc:
        raise SeedVerificationError("ed25519 signature does not verify") from exc


def load_verified_holdout(
    jsonl_path: Path,
    manifest_path: Path | None = None,
    *,
    require_signature: bool = False,
    trusted_signers: dict[str, bytes] | None = None,
) -> VerifiedSeed:
    """Load a held-out JSONL and verify it against its manifest.

    Steps:
      1. Read manifest JSON. Default manifest path is
         `<jsonl_path>.manifest.json` (matches write_holdout).
      2. Re-hash the JSONL; refuse-load on sha256 mismatch.
      3. If `require_signature`, demand a manifest.signature block,
         verify it over the canonical manifest, and require the signer
         to appear in `trusted_signers`.
      4. Return parsed records + manifest + paths.

    Designed to be the single entry point every eval-time loader uses.
    """
    if manifest_path is None:
        manifest_path = jsonl_path.with_suffix(jsonl_path.suffix + ".manifest.json")
        if not manifest_path.exists():
            # write_holdout also accepts <stem>.manifest.json
            manifest_path = jsonl_path.with_suffix(".manifest.json")
    if not jsonl_path.exists():
        raise SeedVerificationError(f"holdout jsonl not found: {jsonl_path}")
    if not manifest_path.exists():
        raise SeedVerificationError(f"holdout manifest not found: {manifest_path}")

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SeedVerificationError(f"manifest is not valid JSON: {exc}") from exc

    expected_sha = manifest.get("output_sha256")
    if not isinstance(expected_sha, str):
        raise SeedVerificationError(
            "manifest is missing output_sha256 — refusing to load"
        )
    got_sha = _sha256_file(jsonl_path)
    if got_sha != expected_sha:
        raise SeedVerificationError(
            f"holdout sha256 mismatch: manifest says {expected_sha[:16]}…, "
            f"file is {got_sha[:16]}…"
        )

    if require_signature:
        sig_block = manifest.get("signature")
        if not isinstance(sig_block, dict):
            raise SeedVerificationError(
                "require_signature=True but manifest has no signature block"
            )
        signer_did = sig_block.get("signer_did")
        sig_b64 = sig_block.get("sig_b64")
        if not isinstance(signer_did, str) or not isinstance(sig_b64, str):
            raise SeedVerificationError(
                "manifest.signature must have string {signer_did, sig_b64}"
            )
        if not trusted_signers:
            raise SeedVerificationError(
                "require_signature=True but no trusted_signers passed"
            )
        _verify_ed25519(
            signer_did=signer_did,
            sig_b64=sig_b64,
            payload=_canonical_manifest_bytes(manifest),
            trusted_signers=trusted_signers,
        )

    # Re-use the existing loader for record parsing
    from mesh.eval.holdout import iter_jsonl

    records = iter_jsonl(jsonl_path)
    return VerifiedSeed(
        records=records,
        manifest=manifest,
        jsonl_path=jsonl_path,
        manifest_path=manifest_path,
    )
