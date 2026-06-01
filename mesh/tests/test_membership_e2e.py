"""End-to-end acceptance test for issue #9 — pull (discover) is THE default
membership model for the local / LAN / tailnet path.

This is swift-harbor's 6-step acceptance test, run hermetically: no real
network, no real models. It composes the *actual* shipped seams, faked the
same way the existing tests fake them:

* a node's self-description is a ``/models?include=routing_meta`` payload —
  shape copied from ``test_discovery._models_payload``;
* discovery is the real ``discover_specialists(status, fetch=...)`` with a
  **sync** injected ``fetch`` (mirrors ``test_discovery``) — LAN shape via
  ``synthesize_lan_status``, tailnet shape via a real ``tailscale status``
  dict (mirrors ``test_discovery._STATUS``);
* the router is the real ``create_router_app(snapshot_source=..., http_client=
  ...)`` fed by ``discovery_to_snapshot`` — exactly how ``slancha-mesh router``
  bridges pull discovery to the OpenAI surface — with an ``httpx.MockTransport``
  upstream (mirrors ``test_router_app._client``) so no socket is opened.

The thing this test defends: a node that is *up* must be discovered and must
appear in the routing table **exactly once**. "Registered but invisible" (a
push-only node nobody discovers) and "appears twice" (a node double-counted by
both push and pull) are both made hard failures below.
"""

from __future__ import annotations

import httpx
from fastapi.testclient import TestClient

from mesh.discovery import discover_specialists, synthesize_lan_status
from mesh.router_app import create_router_app, discovery_to_snapshot

# A real catalog id (so model flow-through / ollama_tag rewrite is exercised
# the same as in production); the matching card lives in mesh/catalog/.
SPECIALIST_ID = "qwen2.5-coder-7b-q4-ollama"

# The node serves on its node-info port; the advertised node_url uses a model
# port. The advertised host is deliberately bogus ("evil.example") to prove
# discovery host-pins to the dialed peer, not to what the node claims.
MODEL_PORT = 11434

LAN_HOST = "127.0.0.1"
LAN_NODE_URL = f"http://{LAN_HOST}:{MODEL_PORT}"

TAILNET_DNS = "phi-box.taila.ts.net."  # trailing dot, as Tailscale reports it
TAILNET_HOST = "phi-box.taila.ts.net"
TAILNET_NODE_URL = f"http://{TAILNET_HOST}:{MODEL_PORT}"

COMPLETION = {
    "id": "chatcmpl-e2e",
    "object": "chat.completion",
    "model": SPECIALIST_ID,
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "pong from the discovered node"},
            "finish_reason": "stop",
        }
    ],
}


def _models_payload() -> dict:
    """One node's GET /models?include=routing_meta — copied from
    test_discovery._models_payload's shape (advertised host is bogus on
    purpose; host-pinning rewrites it to the dialed peer)."""
    return {
        "object": "list",
        "data": [
            {
                "id": SPECIALIST_ID,
                "object": "model",
                "owned_by": "slancha-mesh",
                "routing_meta": {
                    "model_id": f"vendor/{SPECIALIST_ID}",
                    "domain": "code",
                    "capabilities": ["streaming", "system_prompt"],
                    "quality": {"router_observed": 4.2},
                    "node_urls": [f"http://evil.example:{MODEL_PORT}"],
                },
            }
        ],
    }


def _fetch(host, port):
    """Sync fetch(host, port) — every probed peer answers its self-description.
    Matches the FetchFn contract and the test_discovery fake exactly."""
    return _models_payload()


def _discover(shape):
    """Run the documented default consumer path (pull/discover) for the given
    discovery shape; return (DiscoveryResult, expected_node_url)."""
    if shape == "lan":
        # raw-LAN --peer / 127.0.0.1 shape
        status = synthesize_lan_status([LAN_HOST])
        result = discover_specialists(status, fetch=_fetch)
        return result, LAN_NODE_URL
    # tailnet-walk shape — a real `tailscale status --json` with one online,
    # tag:specialist peer (mirrors test_discovery._STATUS).
    status = {
        "Self": {
            "HostName": "router",
            "DNSName": "router.taila.ts.net.",
            "Online": True,
            "Tags": ["tag:gateway"],  # the consumer itself is not a specialist
        },
        "Peer": {
            "nodekey:phi": {
                "HostName": "phi-box",
                "DNSName": TAILNET_DNS,
                "Online": True,
                "Tags": ["tag:specialist"],
            }
        },
        "MagicDNSSuffix": "taila.ts.net",
    }
    result = discover_specialists(status, fetch=_fetch)
    return result, TAILNET_NODE_URL


def _router_client(snapshot):
    """A TestClient over the real router app, with an httpx.MockTransport
    upstream that records which node_url it was POSTed to (mirrors
    test_router_app._client). Returns (TestClient, posted_urls_list)."""
    posted: list[str] = []

    def transport_handler(request: httpx.Request) -> httpx.Response:
        posted.append(str(request.url))
        return httpx.Response(200, json=COMPLETION)

    upstream = httpx.AsyncClient(transport=httpx.MockTransport(transport_handler))
    app = create_router_app(snapshot_source=lambda: snapshot, http_client=upstream)
    return TestClient(app), posted


# --------------------------------------------------------------------------- #
# Steps 1-3 + 5: node up (pull mode) -> default path discovers it -> it lands
# in the routing table EXACTLY ONCE, for BOTH the LAN and tailnet shapes.
# --------------------------------------------------------------------------- #


def _assert_visible_exactly_once(shape):
    result, expected_url = _discover(shape)

    # Step 2: the default consumer path discovered the served node. "Registered
    # but invisible" failure guard — a served node MUST be discoverable.
    assert SPECIALIST_ID in result.specialists, (
        f"[{shape}] served node is invisible to discovery: {result!r}"
    )
    spec = result.specialists[SPECIALIST_ID]

    # Step 3: it appears EXACTLY ONCE. No duplicate from a mixed push+pull
    # membership model. This is the "appears twice" failure guard.
    assert spec.node_urls.count(expected_url) == 1, (
        f"[{shape}] node must appear exactly once, got node_urls={spec.node_urls!r}"
    )
    assert len(spec.node_urls) == 1, (
        f"[{shape}] exactly one binding expected, got {spec.node_urls!r}"
    )

    # The routing table the router actually consumes must agree: one binding.
    snap = discovery_to_snapshot(result)
    bindings = snap.specialists[SPECIALIST_ID]
    binding_urls = [b.node_url for b in bindings]
    assert binding_urls.count(expected_url) == 1, (
        f"[{shape}] routing table must list the node once, got {binding_urls!r}"
    )
    assert len(bindings) == 1, f"[{shape}] one binding expected, got {bindings!r}"
    return result, expected_url


def test_lan_pull_default_node_visible_exactly_once():
    _assert_visible_exactly_once("lan")


def test_tailnet_pull_default_node_visible_exactly_once():
    _assert_visible_exactly_once("tailnet")


# --------------------------------------------------------------------------- #
# Step 4: a /v1/chat/completions request through the router reaches THAT node —
# wired exactly as the `slancha-mesh router` CLI does (discover_specialists ->
# discovery_to_snapshot -> create_router_app). Run for both discovery shapes so
# the whole pull-default happy path is covered end to end.
# --------------------------------------------------------------------------- #


def _assert_request_reaches_node(shape):
    result, expected_url = _assert_visible_exactly_once(shape)

    snap = discovery_to_snapshot(result)
    client, posted = _router_client(snap)

    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": SPECIALIST_ID,
            "messages": [{"role": "user", "content": "ping"}],
        },
    )

    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["choices"][0]["message"]["content"] == "pong from the discovered node"

    # The router must have forwarded to the DISCOVERED node_url, exactly once.
    assert posted == [f"{expected_url}/v1/chat/completions"], (
        f"[{shape}] router must forward to the discovered node exactly once, "
        f"got posted={posted!r}"
    )
    # And the routing-audit header must name the node id derived from that url.
    assert resp.headers["X-Slancha-Specialist"] == SPECIALIST_ID


def test_lan_pull_default_request_reaches_discovered_node():
    _assert_request_reaches_node("lan")


def test_tailnet_pull_default_request_reaches_discovered_node():
    _assert_request_reaches_node("tailnet")
