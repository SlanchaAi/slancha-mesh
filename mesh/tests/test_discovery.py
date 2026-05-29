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

import pytest

from mesh.discovery import (
    DEFAULT_SPECIALIST_TAG,
    DiscoveryResult,
    discover_specialists,
    parse_specialist_peers,
    pin_host,
    synthesize_lan_status,
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
        "mac-mini.taila.ts.net": _models_payload("demo-model", 8004, domain="writing"),
    }

    def fetch(host: str, port: int):
        return served.get(host)

    return fetch


def test_discover_aggregates_specialists_across_peers():
    result = discover_specialists(_STATUS, fetch=_fetch_factory())
    assert isinstance(result, DiscoveryResult)
    assert set(result.specialists) == {"qwen3-coder", "demo-model"}
    assert result.specialists["demo-model"].domain == "writing"


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
    assert "demo-model" not in result.specialists


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


# ---------------------------------------------------------------------------
# Malformed advertised port — must not abort the discovery pass (a node
# can't DoS discovery through the very seam meant to defend against it).
# ---------------------------------------------------------------------------


def test_pin_host_raises_on_malformed_port():
    # pin_host is a strict primitive; the aggregator is what swallows this.
    with pytest.raises(ValueError):
        pin_host("http://evil.example:notaport/v1", "mac-mini.ts.net")


def test_discover_does_not_crash_on_malformed_advertised_port():
    # One node advertises a malformed port; the pass must still complete and
    # the other node must still aggregate (regression guard for the crash).
    def fetch(host, port):
        if host == "mac-mini.taila.ts.net":
            return _models_payload("code-bad", port="notaport")  # type: ignore[arg-type]
        return _models_payload("code-good", port=8003)

    result = discover_specialists(_STATUS, fetch=fetch)
    assert "code-good" in result.specialists  # healthy node unaffected
    assert "code-bad" not in result.specialists  # bad-port entry skipped (unroutable)
    assert "mac-mini.taila.ts.net" in result.reachable  # peer still reached


def test_discover_keeps_valid_url_when_another_is_malformed():
    payload = {
        "object": "list",
        "data": [
            {
                "id": "code-mix",
                "object": "model",
                "routing_meta": {
                    "domain": "code",
                    "node_urls": ["http://evil.example:notaport", "http://evil.example:8003"],
                },
            }
        ],
    }

    result = discover_specialists(_STATUS, fetch=lambda host, port: payload)
    spec = result.specialists["code-mix"]
    assert spec.node_urls  # the valid url survived
    assert all("notaport" not in u for u in spec.node_urls)  # malformed dropped
    assert all(u.endswith(":8003") for u in spec.node_urls)  # only the good port
    assert all("evil.example" not in u for u in spec.node_urls)  # still host-pinned


# ---------------------------------------------------------------------------
# LAN-only mode — `--peer` synthesizes a tailscale-status shape
# ---------------------------------------------------------------------------


def test_synthesize_lan_status_empty_returns_empty_peer_map():
    """No peers = no walk; downstream `discover_specialists` returns zeroes."""
    status = synthesize_lan_status([])
    assert status == {"Peer": {}}


def test_synthesize_lan_status_tags_each_host_as_online_specialist():
    """Each `--peer` host must come out as Online + tagged so it's walked."""
    status = synthesize_lan_status(["192.168.1.10", "mac-mini.local"])
    peer_map = status["Peer"]
    assert len(peer_map) == 2
    hosts = {entry["DNSName"] for entry in peer_map.values()}
    assert hosts == {"192.168.1.10", "mac-mini.local"}
    for entry in peer_map.values():
        assert entry["Online"] is True
        assert DEFAULT_SPECIALIST_TAG in entry["Tags"]


def test_synthesize_lan_status_omits_self_entry():
    """LAN mode is explicit: `--peer localhost` adds the local node; the
    synthesizer never invents a Self entry behind the operator's back."""
    status = synthesize_lan_status(["10.0.0.5"])
    assert "Self" not in status
    # parse_specialist_peers must therefore not return a self-peer when
    # include_self=True is passed to it (cli.py turns include_self off in
    # `--peer` mode, but the parser-level test pins the no-Self contract).
    peers = parse_specialist_peers(status, include_self=True)
    assert not any(p.is_self for p in peers)


def test_discover_via_synthesized_lan_status_walks_each_peer():
    """End-to-end: synthesize_lan_status → discover_specialists with a stub
    fetch returns the expected aggregate. This is the LAN happy path the
    README's 5-minute quickstart depends on."""
    fetches: list[str] = []

    def fetch(host, port):
        fetches.append(host)
        return _models_payload(f"code-{host.replace('.', '-')}", port=8003)

    status = synthesize_lan_status(["192.168.1.10", "192.168.1.20"])
    result = discover_specialists(status, fetch=fetch, node_info_port=8088)
    assert sorted(fetches) == ["192.168.1.10", "192.168.1.20"]
    assert sorted(result.reachable) == ["192.168.1.10", "192.168.1.20"]
    # Each peer's payload yields a distinct specialist (different ids); both
    # must end up in the aggregate.
    assert len(result.specialists) == 2
    for sid, spec in result.specialists.items():
        assert spec.node_urls, f"{sid} has no node_urls"


def test_discover_via_synthesized_lan_status_honors_custom_tag():
    """`--tag` must propagate through synthesize → discover so a
    non-default tag (`tag:lab-host`) doesn't silently drop every peer."""
    status = synthesize_lan_status(["10.0.0.5"], specialist_tag="tag:lab-host")
    # Default tag check should EXCLUDE this peer (wrong tag).
    peers_default = parse_specialist_peers(status)
    assert peers_default == []
    # Custom-tag check should INCLUDE it.
    peers_lab = parse_specialist_peers(status, specialist_tag="tag:lab-host")
    assert len(peers_lab) == 1
    assert peers_lab[0].host == "10.0.0.5"
