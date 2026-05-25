"""Tailnet transport helpers — MagicDNS resolution + onboarding.

Control-plane-agnostic: `tailscale status --json` → `Self.DNSName` is
identical on Tailscale SaaS and self-hosted Headscale. The ONLY divergence
is the `tailscale up` login-server flag at onboarding. These tests lock
both: identical node-side resolution, control-plane-specific onboarding.
"""

from __future__ import annotations

import json

from mesh.tailnet import (
    TailnetConfig,
    advertise_url,
    onboarding_command,
    parse_magicdns_name,
    resolve_advertise_host,
)

# A captured `tailscale status --json` Self block. Both Tailscale and
# Headscale populate Self.DNSName with a trailing-dot FQDN.
_STATUS_JSON = {
    "Self": {
        "HostName": "promaxgb10-d325",
        "DNSName": "promaxgb10-d325.taila93596.ts.net.",
        "TailscaleIPs": ["100.101.102.103", "fd7a:115c:a1e0::1234"],
        "Online": True,
        "Tags": ["tag:specialist"],
    },
    "MagicDNSSuffix": "taila93596.ts.net",
}


# ---------------------------------------------------------------------------
# parse_magicdns_name — pure parse of captured status JSON
# ---------------------------------------------------------------------------


def test_parse_magicdns_name_strips_trailing_dot():
    assert parse_magicdns_name(_STATUS_JSON) == "promaxgb10-d325.taila93596.ts.net"


def test_parse_magicdns_name_accepts_json_string():
    assert parse_magicdns_name(json.dumps(_STATUS_JSON)) == "promaxgb10-d325.taila93596.ts.net"


def test_parse_magicdns_name_none_on_missing_self():
    assert parse_magicdns_name({}) is None
    assert parse_magicdns_name({"Self": {}}) is None
    assert parse_magicdns_name("not json at all") is None
    assert parse_magicdns_name({"Self": {"DNSName": ""}}) is None


# ---------------------------------------------------------------------------
# advertise_url — swap bind host for the advertised (dialable) host
# ---------------------------------------------------------------------------


def test_advertise_url_swaps_host_keeps_port_and_scheme():
    out = advertise_url("http://0.0.0.0:8003", "promaxgb10-d325.taila93596.ts.net")
    assert out == "http://promaxgb10-d325.taila93596.ts.net:8003"


def test_advertise_url_swaps_loopback_too():
    out = advertise_url("http://127.0.0.1:8004", "gb10.ts.net")
    assert out == "http://gb10.ts.net:8004"


def test_advertise_url_none_host_returns_unchanged():
    # Back-compat: no advertise host → loopback URL stays as-is (dev mode).
    assert advertise_url("http://127.0.0.1:8001", None) == "http://127.0.0.1:8001"


# ---------------------------------------------------------------------------
# resolve_advertise_host — priority: explicit > magicdns > None
# ---------------------------------------------------------------------------


def test_resolve_advertise_host_prefers_explicit_override():
    cfg = TailnetConfig(enabled=True, advertise_host="myhost.example")
    # Explicit override wins even if a (stub) resolver would return something.
    assert resolve_advertise_host(cfg, _magicdns_resolver=lambda c: "other.ts.net") == "myhost.example"


def test_resolve_advertise_host_falls_back_to_magicdns():
    cfg = TailnetConfig(enabled=True)
    got = resolve_advertise_host(cfg, _magicdns_resolver=lambda c: "gb10.taila.ts.net")
    assert got == "gb10.taila.ts.net"


def test_resolve_advertise_host_none_when_disabled():
    cfg = TailnetConfig(enabled=False)
    assert resolve_advertise_host(cfg, _magicdns_resolver=lambda c: "gb10.ts.net") is None


def test_resolve_advertise_host_none_when_magicdns_unavailable():
    cfg = TailnetConfig(enabled=True)
    assert resolve_advertise_host(cfg, _magicdns_resolver=lambda c: None) is None


# ---------------------------------------------------------------------------
# Headscale vs Tailscale — node-side identical, onboarding differs
# ---------------------------------------------------------------------------


def test_resolution_identical_across_control_planes():
    """Same status JSON → same MagicDNS name regardless of control plane."""
    ts = parse_magicdns_name(_STATUS_JSON)
    hs = parse_magicdns_name(_STATUS_JSON)
    assert ts == hs == "promaxgb10-d325.taila93596.ts.net"


def test_onboarding_command_tailscale_has_no_login_server():
    cfg = TailnetConfig(enabled=True, control_plane="tailscale")
    cmd = onboarding_command(cfg, auth_key="tskey-abc")
    assert "tailscale up" in cmd
    assert "--auth-key=tskey-abc" in cmd
    assert "--advertise-tags=tag:specialist" in cmd
    assert "--login-server" not in cmd


def test_onboarding_command_headscale_adds_login_server():
    cfg = TailnetConfig(
        enabled=True,
        control_plane="headscale",
        login_server="https://headscale.example.org",
    )
    cmd = onboarding_command(cfg, auth_key="hskey-xyz")
    assert "--login-server=https://headscale.example.org" in cmd
    assert "--advertise-tags=tag:specialist" in cmd
    assert "--auth-key=hskey-xyz" in cmd


def test_onboarding_command_respects_custom_tags():
    cfg = TailnetConfig(enabled=True, tags=["tag:specialist", "tag:gpu"])
    cmd = onboarding_command(cfg, auth_key="k")
    assert "--advertise-tags=tag:specialist,tag:gpu" in cmd


# ---------------------------------------------------------------------------
# TailnetConfig.from_env
# ---------------------------------------------------------------------------


def test_from_env_disabled_by_default(monkeypatch):
    for k in list(__import__("os").environ):
        if k.startswith("SLANCHA_TAILNET_"):
            monkeypatch.delenv(k, raising=False)
    cfg = TailnetConfig.from_env()
    assert cfg.enabled is False


def test_from_env_reads_overrides(monkeypatch):
    monkeypatch.setenv("SLANCHA_TAILNET_ENABLED", "1")
    monkeypatch.setenv("SLANCHA_TAILNET_ADVERTISE_HOST", "gb10.ts.net")
    monkeypatch.setenv("SLANCHA_TAILNET_BIND_HOST", "0.0.0.0")
    monkeypatch.setenv("SLANCHA_TAILNET_CONTROL_PLANE", "headscale")
    monkeypatch.setenv("SLANCHA_TAILNET_LOGIN_SERVER", "https://hs.example")
    monkeypatch.setenv("SLANCHA_TAILNET_TAGS", "tag:specialist,tag:gpu")
    cfg = TailnetConfig.from_env()
    assert cfg.enabled is True
    assert cfg.advertise_host == "gb10.ts.net"
    assert cfg.bind_host == "0.0.0.0"
    assert cfg.control_plane == "headscale"
    assert cfg.login_server == "https://hs.example"
    assert cfg.tags == ["tag:specialist", "tag:gpu"]
