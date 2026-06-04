"""Per-node cryptographic identity for the registry (#102).

Today the registry trusts a node's self-declared `node_id` (auth is a single
SHARED bearer token, so any token holder can heartbeat AS any node — impersonate
a peer, poison its bindings, evict it). This adds an Ed25519 node identity: a node
holds a keypair and presents a self-signed cert binding `node_id ↔ public_key`;
the registry verifies the self-signature and PINS `node_id → public_key` on first
sight, refusing any later heartbeat that claims the same node_id with a different
key. (Trust-on-first-use; an operator can pre-seed the pin for stronger-than-TOFU.)

Shape follows SlanchaAi/wire's `did:wire` convention (Ed25519, base64 sigs, an
8-hex pubkey fingerprint) so the two systems interoperate, but this is a small
pure-Python primitive — no Rust, no wire daemon. Ed25519 via PyNaCl, imported
lazily (same optional-dep pattern as `mesh.eval.seed_verify`): the default
path (no cert) never imports it; install the `signing` extra to use identity.

The cert is SELF-signed (the key vouches for the node_id binding). That alone is
TOFU — what stops impersonation is the registry PIN: once node X is bound to key
K, no one without K's private key can heartbeat as X.
"""

from __future__ import annotations

import base64
import hashlib
from typing import Any

_CERT_CONTEXT = "slancha-node-cert/v1"


class NodeIdentityError(Exception):
    """A node identity cert was invalid, or a node_id↔key pin was violated."""


def _b64e(b: bytes) -> str:
    return base64.b64encode(b).decode("ascii")


def _b64d(s: str) -> bytes:
    return base64.b64decode(s.encode("ascii"), validate=True)


def _cert_message(node_id: str, public_key_b64: str) -> bytes:
    """The exact bytes the node signs — context-prefixed binding of id↔key."""
    return f"{_CERT_CONTEXT}|{node_id}|{public_key_b64}".encode("utf-8")


def _require_nacl():  # pragma: no cover - trivial import shim
    try:
        from nacl import signing  # noqa: PLC0415
        from nacl.exceptions import BadSignatureError  # noqa: PLC0415
    except ModuleNotFoundError as e:
        raise NodeIdentityError(
            "node identity requires PyNaCl — install the 'signing' extra "
            "(pip install 'slancha-mesh[signing]')"
        ) from e
    return signing, BadSignatureError


def generate_node_keypair() -> tuple[str, str]:
    """Return (secret_key_b64, public_key_b64) — a fresh Ed25519 node keypair.
    Persist the secret to a 0600 file (e.g. /etc/slancha/node.key); publish only
    the public key (it rides in the cert)."""
    signing, _ = _require_nacl()
    sk = signing.SigningKey.generate()
    return _b64e(bytes(sk)), _b64e(bytes(sk.verify_key))


def did_for(node_id: str, public_key_b64: str) -> str:
    """`did:wire`-shaped id: did:wire:<node_id>-<8hex pubkey fingerprint>."""
    fp = hashlib.sha256(_b64d(public_key_b64)).hexdigest()[:8]
    return f"did:wire:{node_id}-{fp}"


def build_node_cert(node_id: str, secret_key_b64: str) -> dict[str, str]:
    """Self-sign a `{node_id, public_key_b64, signature_b64}` cert. The node sends
    this in its heartbeat; the registry verifies + pins it."""
    signing, _ = _require_nacl()
    sk = signing.SigningKey(_b64d(secret_key_b64))
    pub_b64 = _b64e(bytes(sk.verify_key))
    sig = sk.sign(_cert_message(node_id, pub_b64)).signature
    return {"node_id": node_id, "public_key_b64": pub_b64, "signature_b64": _b64e(sig)}


def verify_node_cert(cert: dict[str, Any], expected_node_id: str) -> bool:
    """True iff `cert` is a valid self-signed binding for `expected_node_id`
    (signature verifies under the cert's own public key, and the node_id matches).
    Never raises on a bad/garbage cert — returns False."""
    if not isinstance(cert, dict):
        return False
    node_id = cert.get("node_id")
    pub_b64 = cert.get("public_key_b64")
    sig_b64 = cert.get("signature_b64")
    if not (isinstance(node_id, str) and isinstance(pub_b64, str) and isinstance(sig_b64, str)):
        return False
    if node_id != expected_node_id:
        return False
    signing, BadSignatureError = _require_nacl()
    try:
        signing.VerifyKey(_b64d(pub_b64)).verify(_cert_message(node_id, pub_b64), _b64d(sig_b64))
    except (BadSignatureError, ValueError, Exception):  # noqa: BLE001 - any decode/verify failure = invalid
        return False
    return True


__all__ = [
    "NodeIdentityError",
    "generate_node_keypair",
    "did_for",
    "build_node_cert",
    "verify_node_cert",
]
