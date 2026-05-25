"""Tailnet transport — MagicDNS advertise-host resolution + onboarding.

The mesh moved off per-host Cloudflare tunnels onto a Tailscale tailnet:
a cloud gateway (`tag:gateway`) reaches home model-serving nodes
(`tag:specialist`) over WireGuard by MagicDNS, on the model ports. The
tailnet ACL is the access control (deny-by-default,
`tag:gateway -> tag:specialist:<model ports>`).

This module is **control-plane-agnostic**: `tailscale status --json` →
`Self.DNSName` is populated identically by Tailscale SaaS and self-hosted
Headscale (both implement the `tailscale` CLI + LocalAPI). Node-side code
is identical; the ONLY divergence is the `--login-server` flag on
`tailscale up` at onboarding. No SaaS-only feature (Funnel/Serve/
app-connectors) is used, so the OSS Headscale path is first-class.

Everything here is **config-gated**: with `TailnetConfig.enabled=False`
(the default) nothing runs and the daemon keeps its loopback behavior, so
non-tailnet dev is unchanged.

Subprocess calls follow `mesh/probe.py`'s never-raise contract: any
failure returns None rather than crashing the heartbeat loop.
"""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass, field
from typing import Callable, Literal
from urllib.parse import urlsplit, urlunsplit

ControlPlane = Literal["tailscale", "headscale"]

DEFAULT_SPECIALIST_TAG = "tag:specialist"
# Matches the live ACL grant `tag:gateway -> tag:specialist:8003,8004`
# (vLLM :8003, HF :8004). Advisory here — the node serves on whatever
# ports build_daemon assigns; this is the documented convention.
DEFAULT_MODEL_PORTS: dict[str, int] = {"vllm": 8003, "hf": 8004}


@dataclass(frozen=True)
class TailnetConfig:
    """How (and whether) this node advertises itself over the tailnet.

    `enabled=False` (default) keeps the daemon on loopback — no tailscale
    calls, existing behavior unchanged. `bind_host` is where the model
    server LISTENS (0.0.0.0 so the tailnet interface is reachable);
    `advertise_host` is what the gateway DIALS (a MagicDNS name) — the two
    are intentionally distinct (you bind broadly, advertise a routable
    name). Leave `advertise_host` None to auto-discover via MagicDNS.
    """

    enabled: bool = False
    advertise_host: str | None = None
    bind_host: str = "0.0.0.0"
    control_plane: ControlPlane = "tailscale"
    login_server: str | None = None  # Headscale only; Tailscale ignores it
    tags: list[str] = field(default_factory=lambda: [DEFAULT_SPECIALIST_TAG])
    tailscale_bin: str = "tailscale"

    @classmethod
    def from_env(cls, environ: dict[str, str] | None = None) -> "TailnetConfig":
        """Build from SLANCHA_TAILNET_* env vars. All optional."""
        env = environ if environ is not None else os.environ
        enabled = env.get("SLANCHA_TAILNET_ENABLED", "").strip().lower() in (
            "1", "true", "yes", "on",
        )
        tags_raw = env.get("SLANCHA_TAILNET_TAGS", "").strip()
        tags = [t.strip() for t in tags_raw.split(",") if t.strip()] or [DEFAULT_SPECIALIST_TAG]
        cp = env.get("SLANCHA_TAILNET_CONTROL_PLANE", "tailscale").strip().lower()
        control_plane: ControlPlane = "headscale" if cp == "headscale" else "tailscale"
        return cls(
            enabled=enabled,
            advertise_host=env.get("SLANCHA_TAILNET_ADVERTISE_HOST", "").strip() or None,
            bind_host=env.get("SLANCHA_TAILNET_BIND_HOST", "").strip() or "0.0.0.0",
            control_plane=control_plane,
            login_server=env.get("SLANCHA_TAILNET_LOGIN_SERVER", "").strip() or None,
            tags=tags,
            tailscale_bin=env.get("SLANCHA_TAILNET_BIN", "").strip() or "tailscale",
        )


# ---------------------------------------------------------------------------
# MagicDNS resolution
# ---------------------------------------------------------------------------


def parse_magicdns_name(status: dict | str) -> str | None:
    """Pull `Self.DNSName` from a `tailscale status --json` payload.

    Returns the FQDN with the trailing dot stripped, or None if the
    payload is missing/empty/unparseable. Identical shape on Tailscale
    and Headscale.
    """
    if isinstance(status, str):
        try:
            status = json.loads(status)
        except (json.JSONDecodeError, ValueError):
            return None
    if not isinstance(status, dict):
        return None
    self_obj = status.get("Self")
    if not isinstance(self_obj, dict):
        return None
    name = self_obj.get("DNSName")
    if not isinstance(name, str) or not name.strip():
        return None
    return name.rstrip(".") or None


def resolve_magicdns_name(config: TailnetConfig) -> str | None:
    """Run `tailscale status --json` and return this node's MagicDNS name.

    Never raises (probe.py contract): missing binary, non-zero exit, or
    unparseable output all yield None so the heartbeat loop survives.
    """
    try:
        out = subprocess.run(
            [config.tailscale_bin, "status", "--json"],
            capture_output=True,
            text=True,
            timeout=4.0,
            check=False,
        )
    except (subprocess.SubprocessError, OSError, FileNotFoundError):
        return None
    if out.returncode != 0 or not out.stdout:
        return None
    return parse_magicdns_name(out.stdout)


def resolve_advertise_host(
    config: TailnetConfig,
    _magicdns_resolver: Callable[[TailnetConfig], str | None] = resolve_magicdns_name,
) -> str | None:
    """The host the registry should advertise for this node.

    Priority: explicit `config.advertise_host` > MagicDNS discovery > None.
    Returns None when tailnet is disabled or no name can be found — the
    caller then keeps the loopback URL (dev mode). `_magicdns_resolver` is
    injectable for tests.
    """
    if not config.enabled:
        return None
    if config.advertise_host:
        return config.advertise_host
    return _magicdns_resolver(config)


# ---------------------------------------------------------------------------
# Advertised-URL construction
# ---------------------------------------------------------------------------


def advertise_url(base_url: str, advertise_host: str | None) -> str:
    """Rewrite a backend's bind URL into a tailnet-dialable URL.

    Swaps the host (e.g. 0.0.0.0 / 127.0.0.1 → MagicDNS name), preserving
    scheme + port + path. `advertise_host=None` returns `base_url`
    unchanged — that's the back-compat loopback path for non-tailnet dev.
    """
    if not advertise_host:
        return base_url
    parts = urlsplit(base_url)
    port = parts.port
    netloc = f"{advertise_host}:{port}" if port is not None else advertise_host
    return urlunsplit((parts.scheme, netloc, parts.path, parts.query, parts.fragment))


# ---------------------------------------------------------------------------
# Onboarding command (docs + CLI helper)
# ---------------------------------------------------------------------------


def onboarding_command(config: TailnetConfig, auth_key: str = "<AUTH_KEY>") -> str:
    """The `tailscale up` command a new specialist node runs to join.

    Tailscale and Headscale produce IDENTICAL commands except Headscale
    adds `--login-server`. The auth key is minted out-of-band (Tailscale
    admin console, Headscale `headscale preauthkeys create`, or the
    slancha-api `POST /api/v1/mesh/hosts` endpoint).
    """
    parts = ["sudo", "tailscale", "up", f"--auth-key={auth_key}"]
    parts.append(f"--advertise-tags={','.join(config.tags)}")
    if config.control_plane == "headscale" and config.login_server:
        parts.append(f"--login-server={config.login_server}")
    return " ".join(parts)


__all__ = [
    "ControlPlane",
    "DEFAULT_MODEL_PORTS",
    "DEFAULT_SPECIALIST_TAG",
    "TailnetConfig",
    "advertise_url",
    "onboarding_command",
    "parse_magicdns_name",
    "resolve_advertise_host",
    "resolve_magicdns_name",
]
