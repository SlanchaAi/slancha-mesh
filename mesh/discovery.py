"""Pull-based specialist discovery — the OSS-clean alternative to push.

Instead of every node POSTing heartbeats to a central registry (which opens
a claim-hijack class: any writer can advertise any `node_url`), the consumer
**pulls**: enumerate `tag:specialist` peers from `tailscale status --json`,
GET each one's `/models?include=routing_meta` over the tailnet, and aggregate
into a routing table.

Why pull wins (see docs/SELF_ORGANIZING_LOOP_SCOPE.md):

- **Identity == address.** A node's self-description is fetched *from its own
  tailnet address*, so the routed `node_url` is host-pinned to the peer we
  actually dialed — a node cannot impersonate another (`pin_host`).
- **Tailnet membership is liveness.** A node that leaves the tailnet drops
  from the peer list; the next discovery pass deregisters it. No TTL sweep,
  no graceful-leave protocol.
- **Zero per-node config + zero write token.** The node only serves models
  and exposes its self-description; the tailnet ACL is the access control.

This module is pure + control-plane-agnostic (Tailscale and Headscale emit
identical `tailscale status --json`). `fetch` is injected so the aggregation
logic is unit-testable without a live tailnet.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Callable
from urllib.parse import urlsplit, urlunsplit

from mesh.tailnet import DEFAULT_SPECIALIST_TAG

# The node-info port the serve daemon exposes `/models` on (in-process
# registry app). Convention; overridable per call. Must be reachable by the
# consumer in the tailnet ACL (`tag:gateway -> tag:specialist:8088`).
DEFAULT_NODE_INFO_PORT = 8088

# A fetch returns the node's parsed `/models?include=routing_meta` JSON, or
# None on any failure (timeout, refused, non-200, unparseable) — the
# never-raise contract mirrors mesh/probe.py so one dead peer can't abort a
# whole discovery pass.
FetchFn = Callable[[str, int], "dict | None"]


@dataclass(frozen=True)
class SpecialistPeer:
    """A tailnet peer eligible to serve specialists."""

    host: str  # MagicDNS name, trailing dot stripped
    online: bool
    is_self: bool = False


@dataclass(frozen=True)
class DiscoveredSpecialist:
    """One specialist merged across every node that serves it."""

    specialist_id: str
    model_id: str | None = None
    domain: str | None = None
    capabilities: tuple[str, ...] = ()
    quality_router_observed: float | None = None
    node_urls: tuple[str, ...] = ()


@dataclass(frozen=True)
class DiscoveryResult:
    """Aggregated routing view from one discovery pass."""

    specialists: dict[str, DiscoveredSpecialist] = field(default_factory=dict)
    reachable: list[str] = field(default_factory=list)
    unreachable: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Tailnet peer enumeration
# ---------------------------------------------------------------------------


def _coerce(status: dict | str) -> dict | None:
    if isinstance(status, str):
        try:
            status = json.loads(status)
        except (json.JSONDecodeError, ValueError):
            return None
    return status if isinstance(status, dict) else None


def _host_of(node: dict) -> str | None:
    name = node.get("DNSName")
    if not isinstance(name, str) or not name.strip():
        return None
    return name.rstrip(".") or None


def parse_specialist_peers(
    status: dict | str,
    specialist_tag: str = DEFAULT_SPECIALIST_TAG,
    include_self: bool = True,
) -> list[SpecialistPeer]:
    """Pull tagged, online specialist peers from a `tailscale status --json`.

    Returns Self (when tagged + `include_self`) plus every online `Peer`
    carrying `specialist_tag`. Offline peers and wrong-tag peers are
    excluded — discovery only routes to nodes that are actually up. Returns
    `[]` on unparseable input (never raises).
    """
    data = _coerce(status)
    if data is None:
        return []

    peers: list[SpecialistPeer] = []

    if include_self:
        self_obj = data.get("Self")
        if isinstance(self_obj, dict) and specialist_tag in (self_obj.get("Tags") or []):
            host = _host_of(self_obj)
            # Self is "this node"; treat as online (we're running on it).
            if host:
                peers.append(SpecialistPeer(host=host, online=True, is_self=True))

    peer_map = data.get("Peer")
    if isinstance(peer_map, dict):
        for node in peer_map.values():
            if not isinstance(node, dict):
                continue
            if specialist_tag not in (node.get("Tags") or []):
                continue
            if not node.get("Online", False):
                continue
            host = _host_of(node)
            if host:
                peers.append(SpecialistPeer(host=host, online=True, is_self=False))

    return peers


# ---------------------------------------------------------------------------
# Host pinning — the security seam
# ---------------------------------------------------------------------------


def pin_host(node_url: str, peer_host: str) -> str:
    """Force `node_url`'s host to `peer_host`, keeping scheme/port/path.

    The node tells us *which port* serves a specialist; it does NOT get to
    tell us *which host* to route to — that's the address we pulled from.
    Pinning here is what makes claim-hijack structurally impossible: even a
    node lying in its `/models` response can only redirect traffic to itself.

    Raises ValueError if `node_url` carries a malformed port; callers handling
    node-advertised URLs skip such entries (see `_specialists_from_models`)
    rather than letting one bad node abort a discovery pass.
    """
    parts = urlsplit(node_url)
    port = parts.port
    netloc = f"{peer_host}:{port}" if port is not None else peer_host
    return urlunsplit((parts.scheme or "http", netloc, parts.path, parts.query, parts.fragment))


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def _merge(into: dict[str, DiscoveredSpecialist], spec: DiscoveredSpecialist) -> None:
    existing = into.get(spec.specialist_id)
    if existing is None:
        into[spec.specialist_id] = spec
        return
    # Same specialist on another node — union node_urls, keep first card meta.
    merged_urls = tuple(dict.fromkeys((*existing.node_urls, *spec.node_urls)))
    into[spec.specialist_id] = DiscoveredSpecialist(
        specialist_id=existing.specialist_id,
        model_id=existing.model_id or spec.model_id,
        domain=existing.domain or spec.domain,
        capabilities=existing.capabilities or spec.capabilities,
        quality_router_observed=(
            existing.quality_router_observed
            if existing.quality_router_observed is not None
            else spec.quality_router_observed
        ),
        node_urls=merged_urls,
    )


def _specialists_from_models(payload: dict, peer_host: str) -> list[DiscoveredSpecialist]:
    """Parse one node's `/models?include=routing_meta` into specialists.

    `node_url`s are host-pinned to `peer_host`. Entries without a usable
    advertised port are skipped (a card with no live binding isn't routable).
    """
    out: list[DiscoveredSpecialist] = []
    for entry in payload.get("data", []) or []:
        if not isinstance(entry, dict):
            continue
        spec_id = entry.get("id")
        if not isinstance(spec_id, str) or not spec_id:
            continue
        meta = entry.get("routing_meta")
        if not isinstance(meta, dict):
            continue
        raw_urls = meta.get("node_urls") or []
        pinned_urls: list[str] = []
        for u in raw_urls:
            if not isinstance(u, str) or not u:
                continue
            try:
                pinned_urls.append(pin_host(u, peer_host))
            except ValueError:
                # Malformed port/host in a node-advertised URL: unroutable.
                # Skip it rather than letting one bad node abort the whole
                # pass (honors the never-raise discovery contract).
                continue
        pinned = tuple(dict.fromkeys(pinned_urls))
        if not pinned:
            continue  # advertised but no reachable binding
        quality = meta.get("quality") or {}
        out.append(
            DiscoveredSpecialist(
                specialist_id=spec_id,
                model_id=meta.get("model_id"),
                domain=meta.get("domain"),
                capabilities=tuple(meta.get("capabilities") or []),
                quality_router_observed=quality.get("router_observed"),
                node_urls=pinned,
            )
        )
    return out


def discover_specialists(
    status: dict | str,
    fetch: FetchFn,
    *,
    node_info_port: int = DEFAULT_NODE_INFO_PORT,
    specialist_tag: str = DEFAULT_SPECIALIST_TAG,
    include_self: bool = True,
) -> DiscoveryResult:
    """Walk the tailnet, pull each specialist node, aggregate into routes.

    For every tagged + online peer, call `fetch(host, node_info_port)` to get
    that node's `/models?include=routing_meta`; parse + host-pin its
    specialists; merge across nodes. Peers whose fetch returns None are
    recorded as `unreachable` and contribute nothing.
    """
    peers = parse_specialist_peers(status, specialist_tag=specialist_tag, include_self=include_self)

    specialists: dict[str, DiscoveredSpecialist] = {}
    reachable: list[str] = []
    unreachable: list[str] = []

    for peer in peers:
        payload = fetch(peer.host, node_info_port)
        if not isinstance(payload, dict):
            unreachable.append(peer.host)
            continue
        reachable.append(peer.host)
        for spec in _specialists_from_models(payload, peer.host):
            _merge(specialists, spec)

    return DiscoveryResult(
        specialists=specialists,
        reachable=reachable,
        unreachable=unreachable,
    )


def make_http_fetch(
    *,
    scheme: str = "http",
    path: str = "/models",
    token: str | None = None,
    timeout: float = 4.0,
) -> FetchFn:
    """A live `fetch(host, port)` that GETs `/models?include=routing_meta`.

    Honors the same `SLANCHA_NODE_TOKEN` bearer the registry uses (None when
    the tailnet ACL is the only gate — the high-trust default). Never raises:
    timeout/refused/non-200/unparseable all return None so one dead peer
    can't abort a discovery pass (mirrors the FetchFn contract).
    """
    import httpx

    headers = {"Authorization": f"Bearer {token}"} if token else {}

    def fetch(host: str, port: int) -> dict | None:
        url = f"{scheme}://{host}:{port}{path}"
        try:
            resp = httpx.get(
                url, params={"include": "routing_meta"}, headers=headers, timeout=timeout
            )
        except Exception:  # noqa: BLE001 — never-raise discovery contract
            return None
        if resp.status_code != 200:
            return None
        try:
            data = resp.json()
        except Exception:  # noqa: BLE001
            return None
        return data if isinstance(data, dict) else None

    return fetch


# ---------------------------------------------------------------------------
# LAN-only mode — synthesize a tailscale-status shape from explicit hosts
# ---------------------------------------------------------------------------


def synthesize_lan_status(
    peer_hosts: list[str],
    *,
    specialist_tag: str = DEFAULT_SPECIALIST_TAG,
) -> dict:
    """Build a fake `tailscale status --json` dict from an explicit peer list.

    This is the seam that lets `slancha-mesh discover --peer 10.0.0.5 --peer
    10.0.0.6` work without Tailscale on the box — the LocalLLaMA / homelab
    happy path where every node is on the same LAN and the network itself
    is the trust boundary. Every host is emitted as an Online peer carrying
    `specialist_tag`, so the rest of the discovery pipeline
    (`parse_specialist_peers` → `discover_specialists`) walks them
    identically to a real tailnet.

    The synthesized status has NO `Self` entry — the local node should
    advertise itself by adding `localhost` (or its LAN IP) to `peer_hosts`
    explicitly. That matches the realistic shape: the CLI doesn't know if
    "this box" is also one of the serving peers and shouldn't guess.

    Empty list → an empty Peer map; `discover_specialists` will return
    zero reachable / zero unreachable nodes, which is the honest answer.
    """
    return {
        "Peer": {
            f"lan-{idx}": {
                "DNSName": host,
                "Online": True,
                "Tags": [specialist_tag],
            }
            for idx, host in enumerate(peer_hosts)
        }
    }


__all__ = [
    "DEFAULT_NODE_INFO_PORT",
    "DEFAULT_SPECIALIST_TAG",
    "DiscoveredSpecialist",
    "DiscoveryResult",
    "FetchFn",
    "SpecialistPeer",
    "discover_specialists",
    "make_http_fetch",
    "parse_specialist_peers",
    "pin_host",
    "synthesize_lan_status",
]
