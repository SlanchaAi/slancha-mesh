"""Pluggable authn/authz seam for the control plane (opt-in, default-preserving).

Today the registry authenticates with a single shared bearer
(``SLANCHA_NODE_TOKEN``) — machine identity, fine for nodes, but it can't answer
"which PERSON did this?" for a regulated review. This seam adds a HUMAN / operator
auth plane a downstream can fill with OIDC/SAML + RBAC, WITHOUT touching the node
(machine) bearer path.

**The two planes stay separate** (a security-review requirement): swapping node
auth to OIDC would lock nodes out of their own heartbeat endpoint. So this module
adds NEW dependencies for human/operator routes; ``registry_app.verify_node_token``
(machine plane) is untouched.

- ``Authenticator`` — Protocol: ``authenticate(authorization) -> Principal``,
  raising the transport-neutral ``AuthError`` (so the authenticator is usable
  outside HTTP too; the web layer maps it to an HTTP status).
- ``Principal`` — ``actor`` (the audit identity), ``roles`` (RBAC), ``auth_method``,
  and ``extra`` (a forward-compat bag for tenant_id / claims / expiry, so the
  pinned cross-repo contract can evolve without a breaking change).
- ``BearerAuthenticator`` — the default; preserves today's bearer behavior. With
  no token configured it returns a dev-mode admin principal (``auth_method=
  "disabled"``). An enterprise profile OVERRIDES ``get_authenticator`` with a
  fail-CLOSED OIDC authenticator and asserts auth is configured at startup — it
  must never ship prod on the disabled default.

FastAPI wiring: ``get_authenticator`` is an override-able dependency
(``app.dependency_overrides[get_authenticator] = lambda: OIDCAuth(...)`` — scoped,
test-safe, no mutable global to race). ``authenticate`` returns the Principal
(mapping ``AuthError`` → ``HTTPException``); ``require_role(r)`` composes
``authenticate`` for authz so a route declares exactly one, never both.

Open-core note: in OSS, an authenticated principal is coarse (full access — the
shared bearer is all-or-nothing). Fine-grained, tenant-scoped RBAC is what the
enterprise authenticator supplies via granular ``roles`` + ``extra['tenant']``.
"""

from __future__ import annotations

import hmac
import os
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Annotated, Any, Protocol

from fastapi import Depends, Header, HTTPException, status

NODE_TOKEN_ENV = "SLANCHA_NODE_TOKEN"


class AuthError(Exception):
    """Transport-neutral auth failure. The web layer maps ``status_hint`` to an
    HTTP status; a non-HTTP caller can handle it directly."""

    def __init__(self, detail: str, status_hint: int = 401) -> None:
        super().__init__(detail)
        self.detail = detail
        self.status_hint = status_hint


@dataclass(frozen=True)
class Principal:
    """An authenticated caller. ``actor`` is the stable audit identity; ``roles``
    drive RBAC; ``extra`` is a forward-compat bag (tenant_id, raw claims, token
    expiry, …) so the contract grows without breaking the downstream pin."""

    actor: str
    roles: frozenset[str]
    auth_method: str
    extra: dict[str, Any] = field(default_factory=dict)


class Authenticator(Protocol):
    """authorization header value → Principal. Raises ``AuthError`` on failure."""

    def authenticate(self, authorization: str | None) -> Principal: ...


# OSS coarse role set: an authenticated bearer caller has full access (the shared
# token is all-or-nothing today). The enterprise authenticator returns granular
# roles instead. require_role checks membership against whatever the authenticator
# put here.
_FULL_ACCESS = frozenset({"viewer", "operator", "approver", "admin", "node"})


class BearerAuthenticator:
    """Default authenticator — preserves the existing shared-bearer behavior.

    No token configured → dev mode: an ``anonymous`` principal with full access
    (``auth_method="disabled"``). Configured → constant-time bearer check.
    """

    def __init__(self, expected_token: str | None = None) -> None:
        # Resolve lazily at authenticate() time by default so env changes / test
        # monkeypatching work; an explicit token pins it.
        self._explicit = expected_token

    def _expected(self) -> str | None:
        if self._explicit is not None:
            return self._explicit or None
        return os.environ.get(NODE_TOKEN_ENV, "").strip() or None

    def authenticate(self, authorization: str | None) -> Principal:
        expected = self._expected()
        if expected is None:
            return Principal("anonymous", _FULL_ACCESS, "disabled")
        if not authorization or not authorization.startswith("Bearer "):
            raise AuthError("missing or malformed Authorization header", 401)
        received = authorization[len("Bearer ") :].strip()
        if not hmac.compare_digest(received, expected):
            raise AuthError("invalid bearer token", 403)
        return Principal("bearer", _FULL_ACCESS, "bearer")


_default_authenticator: Authenticator = BearerAuthenticator()


def get_authenticator() -> Authenticator:
    """FastAPI dependency returning the active authenticator. Override per-app:
    ``app.dependency_overrides[get_authenticator] = lambda: OIDCAuth(...)`` —
    scoped + test-safe, no process-global to race or reset."""
    return _default_authenticator


def authenticate(
    authorization: Annotated[str | None, Header()] = None,
    auth: Authenticator = Depends(get_authenticator),
) -> Principal:
    """Authn dependency: returns the Principal, mapping AuthError → HTTPException."""
    try:
        return auth.authenticate(authorization)
    except AuthError as e:
        headers = {"WWW-Authenticate": 'Bearer realm="slancha-mesh"'} if e.status_hint == 401 else None
        raise HTTPException(status_code=e.status_hint, detail=e.detail, headers=headers) from e


def require_role(role: str) -> Callable[..., Principal]:
    """Authz dependency factory. Composes ``authenticate`` (so a route declares
    exactly one of authenticate / require_role) and 403s if the role is absent."""

    def _dep(principal: Annotated[Principal, Depends(authenticate)]) -> Principal:
        if role not in principal.roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"requires role {role!r}",
            )
        return principal

    return _dep


__all__ = [
    "AuthError",
    "Authenticator",
    "BearerAuthenticator",
    "Principal",
    "authenticate",
    "get_authenticator",
    "require_role",
]
