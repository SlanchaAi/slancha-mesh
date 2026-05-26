"""Pull-based discovery — walk the tailnet, pull each specialist node's
self-description, aggregate into routes.

Discovery is the OSS-clean alternative to heartbeat-push: the consumer
(gateway, or any operator) enumerates `tag:specialist` peers from
`tailscale status --json` and GETs each one's `/models?include=routing_meta`
over the tailnet. The security property under test: the routed `node_url`
is **host-pinned to the peer we actually dialed**, so a node cannot
advertise another node's address (no claim-hijack).
"""

from __future__ import annotations

import json

from mesh.discovery import (
    DiscoveryResult,
    discover_specialists,
    parse_specialist_peers,
    pin_host,
)

# A captured `tailscale status --json`: Self (a specialist) + two peers, one
# tagged specialist + online, one tagged specialist but offline, one a
# gateway (wrong tag → must be ignored).
_STATUS = {
    "Self": {
        "HostName": "gb10-self",
        "DNSName": "gb10-self.taila.ts.net.",
        "Online": True,
        "Tags": ["tag:specialist"],
    },
    "Peer": {
        "nodekey:aaa": {
            "HostName": "mac-mini",
            "DNSName": "mac-mini.taila.ts.net.",
            "Online": True,
            "Tags": ["tag:specialist"],
        },
        "nodekey:bbb": {
            "HostName": "rtx-box",
            "DNSName": "rtx-box.taila.ts.net.",
            "Online": False,
            "Tags": ["tag:specialist"],
        },
        "nodekey:ccc": {
            "HostName": "cloud-gw",
            "DNSName": "cloud-gw.taila.ts.net.",
            "Online": True,
            "Tags": ["tag:gateway"],
        },
    },
    "MagicDNSSuffix": "taila.ts.net",
}


def _models_payload(specialist_id: str, port: int, domain: str = "code", advertised_host: str = "evil.example") -> dict:
    """A node's GET /models?include=routing_meta response.

    `advertised_host` deliberately differs from the peer's real MagicDNS
    name to prove the aggregator host-pins to the dialed peer, not to what
    the node claims.
    """
    return {
        "object": "list",
        "data": [
            {
                "id": specialist_id,
                "object": "model",
                "owned_by": "slancha-mesh",
                "routing_meta": {
                    "model_id": f"vendor/{specialist_id}",
                    "domain": domain,
                    "capabilities": ["streaming", "system_prompt"],
                    "quality": {"router_observed": 4.2, "node_self_reported": 4.0},
                    "node_urls": [f"http://{advertised_host}:{port}"],
                },
            }
        ],
    }


# ---------------------------------------------------------------------------
# parse_specialist_peers
# ---------------------------------------------------------------------------


def test_parse_specialist_peers_includes_self_and_online_peers():
    peers = parse_specialist_peers(_STATUS)
    hosts = {p.host for p in peers}
    assert "gb10-self.taila.ts.net" in hosts  # Self, tagged specialist
    assert "mac-mini.taila.ts.net" in hosts  # online specialist peer
    # Offline specialist + wrong-tag gateway are excluded by default.
    assert "rtx-box.taila.ts.net" not in hosts
    assert "cloud-gw.taila.ts.net" not in hosts


def test_parse_specialist_peers_accepts_json_string():
    peers = parse_specialist_peers(json.dumps(_STATUS))
    assert any(p.host == "mac-mini.taila.ts.net" for p in peers)


def test_parse_specialist_peers_can_exclude_self():
    peers = parse_specialist_peers(_STATUS, include_self=False)
    hosts = {p.host for p in peers}
    assert "gb10-self.taila.ts.net" not in hosts
    assert "mac-mini.taila.ts.net" in hosts


def test_parse_specialist_peers_custom_tag():
    peers = parse_specialist_peers(_STATUS, specialist_tag="tag:gateway", include_self=False)
    assert {p.host for p in peers} == {"cloud-gw.taila.ts.net"}


def test_parse_specialist_peers_empty_on_garbage():
    assert parse_specialist_peers("not json") == []
    assert parse_specialist_peers({}) == []


# ---------------------------------------------------------------------------
# pin_host — force the node_url host to the dialed peer
# ---------------------------------------------------------------------------


def test_pin_host_overrides_host_keeps_port_scheme_path():
    assert pin_host("http://evil.example:8003/v1", "mac-mini.ts.net") == "http://mac-mini.ts.net:8003/v1"


def test_pin_host_no_port():
    assert pin_host("http://evil.example", "mac-mini.ts.net") == "http://mac-mini.ts.net"


# ---------------------------------------------------------------------------
# discover_specialists — end to end with injected fetch
# ---------------------------------------------------------------------------


def _fetch_factory():
    """Return a fetch(host, port) that serves per-host model payloads."""
    served = {
        "gb10-self.taila.ts.net": _models_payload("qwen3-coder", 8003),
        "mac-mini.taila.ts.net": _models_payload("paul-voice", 8004, domain="writing"),
    }

    def fetch(host: str, port: int):
        return served.get(host)

    return fetch


def test_discover_aggregates_specialists_across_peers():
    result = discover_specialists(_STATUS, fetch=_fetch_factory())
    assert isinstance(result, DiscoveryResult)
    assert set(result.specialists) == {"qwen3-coder", "paul-voice"}
    assert result.specialists["paul-voice"].domain == "writing"


def test_discover_host_pins_node_url_to_dialed_peer():
    """The advertised host in the payload (evil.example) must be discarded;
    the routed URL is the peer we actually pulled from."""
    result = discover_specialists(_STATUS, fetch=_fetch_factory())
    coder_urls = result.specialists["qwen3-coder"].node_urls
    assert coder_urls == ("http://gb10-self.taila.ts.net:8003",)
    assert all("evil.example" not in u for u in coder_urls)


def test_discover_marks_unreachable_peers():
    def fetch(host, port):
        if host == "mac-mini.taila.ts.net":
            return None  # simulate timeout / refused
        return _models_payload("qwen3-coder", 8003)

    result = discover_specialists(_STATUS, fetch=fetch)
    assert "gb10-self.taila.ts.net" in result.reachable
    assert "mac-mini.taila.ts.net" in result.unreachable
    assert "qwen3-coder" in result.specialists
    assert "paul-voice" not in result.specialists


def test_discover_merges_same_specialist_on_multiple_nodes():
    def fetch(host, port):
        # Both nodes serve the same specialist on :8003.
        return _models_payload("qwen3-coder", 8003)

    result = discover_specialists(_STATUS, fetch=fetch)
    urls = set(result.specialists["qwen3-coder"].node_urls)
    assert urls == {
        "http://gb10-self.taila.ts.net:8003",
        "http://mac-mini.taila.ts.net:8003",
    }
