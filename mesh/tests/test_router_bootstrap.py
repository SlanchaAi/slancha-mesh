"""Router bootstrap — establish a local router (slancha-local) on first node.

A pure specialist node serves models but cannot route; a home mesh needs a
router (`slancha-local serve`) somewhere to be the OpenAI endpoint that fans
out. `ensure_router` detects whether a router is already reachable (a
`tag:gateway` peer) or installed locally, and — only when asked — installs +
launches one. Install/launch are injected so tests never spawn a process.
"""

from __future__ import annotations

from mesh.router_bootstrap import detect_router, ensure_router

_GATEWAY_STATUS = {
    "Self": {"DNSName": "self.ts.net.", "Online": True, "Tags": ["tag:specialist"]},
    "Peer": {"k": {"DNSName": "gw.ts.net.", "Online": True, "Tags": ["tag:gateway"]}},
}
_NO_GATEWAY_STATUS = {
    "Self": {"DNSName": "self.ts.net.", "Online": True, "Tags": ["tag:specialist"]},
    "Peer": {},
}


def test_detect_router_finds_command_on_path():
    st = detect_router(_NO_GATEWAY_STATUS, which=lambda c: "/usr/bin/slancha-local" if c == "slancha-local" else None)
    assert st.on_path is True
    assert st.present is True


def test_detect_router_finds_gateway_peer():
    st = detect_router(_GATEWAY_STATUS, which=lambda c: None)
    assert st.gateway_peer is True
    assert st.present is True


def test_detect_router_none():
    st = detect_router(_NO_GATEWAY_STATUS, which=lambda c: None)
    assert st.present is False


def test_ensure_router_noop_when_gateway_peer_exists():
    calls = []
    res = ensure_router(
        install=True, status_json=_GATEWAY_STATUS,
        which=lambda c: None,
        installer=lambda spec: calls.append(("install", spec)) or (True, ""),
        launcher=lambda: calls.append("launch") or 4321,
    )
    assert res["action"] == "none"
    assert calls == []  # a router is already reachable; do nothing


def test_ensure_router_instructs_when_not_installed_and_install_false():
    res = ensure_router(
        install=False, status_json=_NO_GATEWAY_STATUS, which=lambda c: None,
        installer=lambda spec: (True, ""), launcher=lambda: 1,
    )
    assert res["action"] == "instruct"
    cmds = " ".join(s.get("command", "") for s in res["next_steps"])
    assert "slancha-local" in cmds  # tells the agent exactly what to run


def test_ensure_router_installs_then_launches():
    calls = []
    res = ensure_router(
        install=True, status_json=_NO_GATEWAY_STATUS, which=lambda c: None,
        installer=lambda spec: calls.append(("install", spec)) or (True, "ok"),
        launcher=lambda: calls.append("launch") or 9999,
    )
    assert res["action"] == "launched"
    assert res["pid"] == 9999
    assert ("install", "slancha-local") in calls and "launch" in calls


def test_ensure_router_launches_without_install_when_already_on_path():
    calls = []
    res = ensure_router(
        install=True, status_json=_NO_GATEWAY_STATUS,
        which=lambda c: "/usr/bin/slancha-local" if c == "slancha-local" else None,
        installer=lambda spec: calls.append("install") or (True, ""),
        launcher=lambda: calls.append("launch") or 7,
    )
    assert res["action"] == "launched"
    assert "install" not in calls  # already installed → skip install, just launch


def test_ensure_router_reports_install_failure():
    res = ensure_router(
        install=True, status_json=_NO_GATEWAY_STATUS, which=lambda c: None,
        installer=lambda spec: (False, "no matching distribution"),
        launcher=lambda: 1,
    )
    assert res["action"] == "install_failed"
    assert "no matching distribution" in res["error"]


def test_ensure_router_honors_install_spec_override():
    captured = {}
    ensure_router(
        install=True, install_spec="git+https://example/slancha-local",
        status_json=_NO_GATEWAY_STATUS, which=lambda c: None,
        installer=lambda spec: captured.update(spec=spec) or (True, ""),
        launcher=lambda: 1,
    )
    assert captured["spec"] == "git+https://example/slancha-local"
