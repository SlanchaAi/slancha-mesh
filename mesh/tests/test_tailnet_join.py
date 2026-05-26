"""Idempotent tailnet join — `ensure_joined`.

The one human-facing trust action is joining the tailnet tagged
`tag:specialist`. `ensure_joined` makes that safe to call every boot: it's a
no-op when already up with the tag, joins with an auth key otherwise, and
fails *loudly with the exact command* when no key is available — never a
silent half-state. Subprocess + status are injected so the logic is tested
without a live tailnet.
"""

from __future__ import annotations

from mesh.tailnet import TailnetConfig, ensure_joined


def _self(online=True, tags=("tag:specialist",), dns="gb10.taila.ts.net."):
    return {"Self": {"Online": online, "Tags": list(tags), "DNSName": dns}}


class _FakeProc:
    def __init__(self, returncode=0, stderr=""):
        self.returncode = returncode
        self.stderr = stderr
        self.stdout = ""


def test_noop_when_already_joined_with_tag():
    calls = []

    def runner(cmd, **kw):
        calls.append(cmd)
        return _FakeProc()

    res = ensure_joined(
        TailnetConfig(enabled=True),
        auth_key="tskey-should-not-be-used",
        _status_fn=lambda c: _self(),
        _runner=runner,
    )
    assert res.joined and res.already
    assert res.host == "gb10.taila.ts.net"
    assert calls == []  # already up → tailscale never invoked


def test_joins_with_key_when_not_up():
    calls = []

    def runner(cmd, **kw):
        calls.append(cmd)
        return _FakeProc(returncode=0)

    # First status: not joined; after join: joined.
    states = iter([None, _self()])

    res = ensure_joined(
        TailnetConfig(enabled=True),
        auth_key="tskey-abc",
        _status_fn=lambda c: next(states),
        _runner=runner,
    )
    assert res.joined and not res.already
    assert res.host == "gb10.taila.ts.net"
    assert len(calls) == 1
    joined_cmd = " ".join(calls[0])
    assert "tailscale up" in joined_cmd
    assert "--auth-key=tskey-abc" in joined_cmd
    assert "--advertise-tags=tag:specialist" in joined_cmd


def test_headscale_join_adds_login_server():
    calls = []
    res = ensure_joined(
        TailnetConfig(enabled=True, control_plane="headscale", login_server="https://hs.example"),
        auth_key="hskey-1",
        _status_fn=lambda c: None if not calls else _self(),
        _runner=lambda cmd, **kw: (calls.append(cmd) or _FakeProc()),
    )
    assert res.joined
    assert "--login-server=https://hs.example" in " ".join(calls[0])


def test_no_key_and_not_up_fails_with_command_not_silent():
    res = ensure_joined(
        TailnetConfig(enabled=True),
        auth_key=None,
        _status_fn=lambda c: None,
        _runner=lambda cmd, **kw: _FakeProc(),
    )
    assert not res.joined
    assert "tailscale up" in res.message  # tells the human exactly what to run


def test_join_failure_surfaces_stderr():
    res = ensure_joined(
        TailnetConfig(enabled=True),
        auth_key="tskey-bad",
        _status_fn=lambda c: None,
        _runner=lambda cmd, **kw: _FakeProc(returncode=1, stderr="key expired"),
    )
    assert not res.joined
    assert "key expired" in res.message


def test_disabled_config_is_noop():
    res = ensure_joined(TailnetConfig(enabled=False), _status_fn=lambda c: _self(), _runner=lambda *a, **k: _FakeProc())
    assert not res.joined
    assert not res.already


def test_already_up_but_missing_tag_triggers_join():
    """On the tailnet but WITHOUT tag:specialist → not a valid specialist
    node; must (re)join to advertise the tag, not silently pass."""
    calls = []
    states = iter([_self(tags=("tag:laptop",)), _self()])
    res = ensure_joined(
        TailnetConfig(enabled=True),
        auth_key="tskey-x",
        _status_fn=lambda c: next(states),
        _runner=lambda cmd, **kw: (calls.append(cmd) or _FakeProc()),
    )
    assert res.joined and not res.already
    assert len(calls) == 1
