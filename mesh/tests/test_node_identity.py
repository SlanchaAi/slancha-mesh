"""Per-node identity cert + registry pinning (#102)."""

from __future__ import annotations

import pytest

pytest.importorskip("nacl.signing")  # identity needs PyNaCl (the 'signing' extra)

from mesh.identity import (  # noqa: E402
    NodeIdentityError,
    build_node_cert,
    did_for,
    generate_node_keypair,
    verify_node_cert,
)
from mesh.registry import HeartbeatPostRequest, MeshRegistry  # noqa: E402
from mesh.tests.conftest import make_heartbeat  # noqa: E402


# ── identity module ──────────────────────────────────────────────────────────
def test_keypair_cert_roundtrip():
    sk, pk = generate_node_keypair()
    cert = build_node_cert("node-a", sk)
    assert cert["node_id"] == "node-a" and cert["public_key_b64"] == pk
    assert verify_node_cert(cert, "node-a") is True
    assert did_for("node-a", pk).startswith("did:wire:node-a-")


def test_verify_rejects_tamper_and_mismatch():
    sk, _ = generate_node_keypair()
    cert = build_node_cert("node-a", sk)
    assert verify_node_cert(cert, "node-b") is False          # wrong expected id
    assert verify_node_cert(dict(cert, node_id="node-b"), "node-b") is False  # tampered id
    sk2, pk2 = generate_node_keypair()                         # another key can't vouch for node-a
    forged = {"node_id": "node-a", "public_key_b64": pk2, "signature_b64": cert["signature_b64"]}
    assert verify_node_cert(forged, "node-a") is False
    assert verify_node_cert({}, "node-a") is False
    assert verify_node_cert("nope", "node-a") is False


# ── registry pinning ─────────────────────────────────────────────────────────
def _req(node, fresh_now, catalog, cert=None, url="http://n:8000/v1"):
    hb = make_heartbeat(node, fresh_now, [], catalog)
    return HeartbeatPostRequest(heartbeat=hb, node_url=url, identity_cert=cert)


def test_pins_node_id_and_refuses_impersonation(spark_node, catalog, fresh_now):
    nid = spark_node.node_id
    sk, pk = generate_node_keypair()
    reg = MeshRegistry(catalog=catalog)
    reg.record_heartbeat(_req(spark_node, fresh_now, catalog, build_node_cert(nid, sk)))
    assert reg._node_pubkeys[nid] == pk
    # a DIFFERENT key claiming the same node_id is refused (the impersonation fix)
    sk2, _ = generate_node_keypair()
    with pytest.raises(NodeIdentityError, match="pinned"):
        reg.record_heartbeat(_req(spark_node, fresh_now, catalog, build_node_cert(nid, sk2)))


def test_certless_after_pinned_is_downgrade(spark_node, catalog, fresh_now):
    nid = spark_node.node_id
    sk, _ = generate_node_keypair()
    reg = MeshRegistry(catalog=catalog)
    reg.record_heartbeat(_req(spark_node, fresh_now, catalog, build_node_cert(nid, sk)))
    with pytest.raises(NodeIdentityError, match="downgrade"):
        reg.record_heartbeat(_req(spark_node, fresh_now, catalog, cert=None))


def test_require_identity_rejects_certless(spark_node, catalog, fresh_now):
    reg = MeshRegistry(catalog=catalog, require_node_identity=True)
    with pytest.raises(NodeIdentityError, match="required"):
        reg.record_heartbeat(_req(spark_node, fresh_now, catalog, cert=None))


def test_invalid_cert_rejected(spark_node, catalog, fresh_now):
    nid = spark_node.node_id
    sk, _ = generate_node_keypair()
    cert = build_node_cert(nid, sk)
    cert["signature_b64"] = "AAAA"  # corrupt the signature
    reg = MeshRegistry(catalog=catalog)
    with pytest.raises(NodeIdentityError, match="invalid"):
        reg.record_heartbeat(_req(spark_node, fresh_now, catalog, cert))


def test_certless_default_path_unaffected(spark_node, catalog, fresh_now):
    """Back-compat: no cert + not required + never pinned → accepted."""
    reg = MeshRegistry(catalog=catalog)
    resp = reg.record_heartbeat(_req(spark_node, fresh_now, catalog, cert=None))
    assert resp.ack is True
