"""Tests for mesh.eval.seed_verify."""

from __future__ import annotations

import base64
import json
from pathlib import Path

import pytest

from mesh.eval.holdout import write_holdout
from mesh.eval.seed_verify import (
    SeedVerificationError,
    _canonical_manifest_bytes,
    load_verified_holdout,
)


def _write_seed(tmp_path: Path, n: int = 4) -> tuple[Path, Path, dict]:
    records = [
        {"prompt_id": f"p-{i}", "prompt_text": f"text-{i}",
         "signals": {"domain": "code"}}
        for i in range(n)
    ]
    out = tmp_path / "seed.jsonl"
    manifest = write_holdout(
        output=out, samples=records, holdout_version=1, seed=42,
    )
    manifest_path = out.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return out, manifest_path, manifest


def test_load_verified_holdout_happy(tmp_path: Path):
    jsonl, manifest_path, manifest = _write_seed(tmp_path)
    seed = load_verified_holdout(jsonl, manifest_path)
    assert seed.manifest == manifest
    assert len(seed.records) == 4
    assert seed.jsonl_path == jsonl


def test_load_verified_holdout_default_manifest_path(tmp_path: Path):
    """Manifest path is discoverable from the jsonl path."""
    jsonl, _explicit_manifest, _ = _write_seed(tmp_path)
    seed = load_verified_holdout(jsonl)  # no manifest_path passed
    assert len(seed.records) == 4


def test_load_verified_holdout_refuses_tampered_jsonl(tmp_path: Path):
    jsonl, manifest_path, _ = _write_seed(tmp_path)
    # Tamper: append a row the manifest doesn't know about
    with jsonl.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"prompt_id": "evil", "signals": {"domain": "code"}}) + "\n")
    with pytest.raises(SeedVerificationError, match="sha256 mismatch"):
        load_verified_holdout(jsonl, manifest_path)


def test_load_verified_holdout_refuses_missing_manifest(tmp_path: Path):
    jsonl, manifest_path, _ = _write_seed(tmp_path)
    manifest_path.unlink()
    with pytest.raises(SeedVerificationError, match="manifest not found"):
        load_verified_holdout(jsonl, manifest_path)


def test_load_verified_holdout_refuses_manifest_missing_sha(tmp_path: Path):
    jsonl, manifest_path, manifest = _write_seed(tmp_path)
    del manifest["output_sha256"]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(SeedVerificationError, match="output_sha256"):
        load_verified_holdout(jsonl, manifest_path)


def test_require_signature_with_no_signature_block_refuses(tmp_path: Path):
    jsonl, manifest_path, _ = _write_seed(tmp_path)
    with pytest.raises(SeedVerificationError, match="no signature block"):
        load_verified_holdout(
            jsonl, manifest_path,
            require_signature=True, trusted_signers={"did:example:x": b"\x00" * 32},
        )


def test_require_signature_with_no_trusted_signers_refuses(tmp_path: Path):
    jsonl, manifest_path, manifest = _write_seed(tmp_path)
    manifest["signature"] = {"signer_did": "did:example:x", "sig_b64": "AAAA"}
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(SeedVerificationError, match="no trusted_signers"):
        load_verified_holdout(
            jsonl, manifest_path,
            require_signature=True, trusted_signers=None,
        )


def test_require_signature_with_unknown_signer_refuses(tmp_path: Path):
    nacl = pytest.importorskip("nacl.signing")
    jsonl, manifest_path, manifest = _write_seed(tmp_path)
    signing = nacl.SigningKey.generate()
    # Sign canonically but with a signer_did the caller doesn't trust
    payload = _canonical_manifest_bytes({**manifest, "signature": {
        "signer_did": "did:example:rogue", "sig_b64": "",
    }})
    sig = signing.sign(payload).signature
    manifest["signature"] = {
        "signer_did": "did:example:rogue",
        "sig_b64": base64.b64encode(sig).decode("ascii"),
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(SeedVerificationError, match="not in trusted set"):
        load_verified_holdout(
            jsonl, manifest_path,
            require_signature=True,
            trusted_signers={"did:example:other": bytes(signing.verify_key)},
        )


def test_require_signature_with_valid_signature_accepts(tmp_path: Path):
    nacl = pytest.importorskip("nacl.signing")
    jsonl, manifest_path, manifest = _write_seed(tmp_path)
    signing = nacl.SigningKey.generate()
    signer_did = "did:example:trusted"
    # The signature covers manifest *minus* signature field — sign that
    payload = _canonical_manifest_bytes(manifest)
    sig = signing.sign(payload).signature
    manifest["signature"] = {
        "signer_did": signer_did,
        "sig_b64": base64.b64encode(sig).decode("ascii"),
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    seed = load_verified_holdout(
        jsonl, manifest_path,
        require_signature=True,
        trusted_signers={signer_did: bytes(signing.verify_key)},
    )
    assert len(seed.records) == 4


def test_require_signature_with_tampered_signature_refuses(tmp_path: Path):
    nacl = pytest.importorskip("nacl.signing")
    jsonl, manifest_path, manifest = _write_seed(tmp_path)
    signing = nacl.SigningKey.generate()
    signer_did = "did:example:trusted"
    # Tamper: invalid signature bytes
    manifest["signature"] = {
        "signer_did": signer_did,
        "sig_b64": base64.b64encode(b"\x00" * 64).decode("ascii"),
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(SeedVerificationError, match="signature does not verify"):
        load_verified_holdout(
            jsonl, manifest_path,
            require_signature=True,
            trusted_signers={signer_did: bytes(signing.verify_key)},
        )
