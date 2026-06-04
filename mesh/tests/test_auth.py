"""Tests for the mesh.auth seam (human/operator authn + RBAC).

Unit tests for BearerAuthenticator + FastAPI integration for the
authenticate/require_role dependencies and override-based injection.
"""

from __future__ import annotations

import pytest
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient

from mesh.auth import (
    AuthError,
    BearerAuthenticator,
    Principal,
    authenticate,
    get_authenticator,
    require_role,
)


@pytest.fixture(autouse=True)
def _clear_token(monkeypatch):
    monkeypatch.delenv("SLANCHA_NODE_TOKEN", raising=False)


# ───────────────────────── BearerAuthenticator (unit) ───────────────────────


def test_disabled_mode_returns_dev_admin_principal():
    p = BearerAuthenticator().authenticate(None)
    assert p.actor == "anonymous" and p.auth_method == "disabled"
    assert "admin" in p.roles  # dev mode = full access, preserving today's behavior


def test_configured_bearer_accepts_correct_token():
    p = BearerAuthenticator(expected_token="s3cret").authenticate("Bearer s3cret")
    assert p.auth_method == "bearer" and "admin" in p.roles


def test_configured_bearer_rejects_wrong_token():
    with pytest.raises(AuthError) as e:
        BearerAuthenticator(expected_token="s3cret").authenticate("Bearer nope")
    assert e.value.status_hint == 403


@pytest.mark.parametrize("hdr", [None, "", "Token x", "s3cret"])
def test_configured_bearer_rejects_missing_or_malformed_header(hdr):
    with pytest.raises(AuthError) as e:
        BearerAuthenticator(expected_token="s3cret").authenticate(hdr)
    assert e.value.status_hint == 401


def test_principal_extra_is_a_forward_compat_bag():
    p = Principal("alice", frozenset({"viewer"}), "oidc", extra={"tenant": "acme"})
    assert p.extra["tenant"] == "acme"


# ───────────────────────── FastAPI integration ──────────────────────────────


def _app() -> FastAPI:
    app = FastAPI()

    @app.get("/who")
    def who(p: Principal = Depends(authenticate)):
        return {"actor": p.actor, "method": p.auth_method}

    @app.get("/admin")
    def admin(_p: Principal = Depends(require_role("admin"))):
        return {"ok": True}

    return app


def test_disabled_mode_refuses_role_gated_route():
    """#97: a role-gated route must REFUSE a disabled-auth principal (RBAC theater
    otherwise) — even though authenticate itself still returns the dev principal."""
    client = TestClient(_app())
    assert client.get("/who").json()["actor"] == "anonymous"   # authenticate is permissive
    assert client.get("/admin").status_code == 403             # require_role refuses disabled


def test_require_role_permit_disabled_opt_in():
    """A route can explicitly allow disabled-auth (e.g. a localhost dev tool)."""
    app = FastAPI()

    @app.get("/dev")
    def dev(_p: Principal = Depends(require_role("admin", permit_disabled=True))):
        return {"ok": True}

    assert TestClient(app).get("/dev").status_code == 200


def test_assert_bind_safe(monkeypatch):
    import pytest

    from mesh.auth import assert_bind_safe

    monkeypatch.delenv("SLANCHA_AUTH_REQUIRED", raising=False)
    # loopback is always fine
    assert_bind_safe("127.0.0.1", token_present=False)
    assert_bind_safe("localhost", token_present=False)
    # public bind with a token is fine
    assert_bind_safe("0.0.0.0", token_present=True)
    # public bind, no token, no override → refuse
    with pytest.raises(SystemExit):
        assert_bind_safe("0.0.0.0", token_present=False)


def test_override_authenticator_enforces_rbac():
    """The enterprise pattern: inject an authenticator via dependency_overrides;
    a principal without the role gets 403, with it gets 200 — no global mutation."""
    app = _app()

    class _Oidc:
        def __init__(self, roles):
            self._roles = frozenset(roles)

        def authenticate(self, authorization):
            return Principal("alice@acme", self._roles, "oidc", extra={"tenant": "acme"})

    app.dependency_overrides[get_authenticator] = lambda: _Oidc({"viewer"})
    client = TestClient(app)
    assert client.get("/who").json()["actor"] == "alice@acme"
    assert client.get("/admin").status_code == 403  # lacks 'admin'

    app.dependency_overrides[get_authenticator] = lambda: _Oidc({"viewer", "admin"})
    assert client.get("/admin").status_code == 200  # now has it


def test_autherror_maps_to_http_status():
    app = _app()

    class _Deny:
        def authenticate(self, authorization):
            raise AuthError("nope", 401)

    app.dependency_overrides[get_authenticator] = lambda: _Deny()
    r = TestClient(app).get("/who")
    assert r.status_code == 401
    assert "WWW-Authenticate" in r.headers
