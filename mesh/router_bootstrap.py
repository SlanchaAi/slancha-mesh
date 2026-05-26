"""Establish a local router (slancha-local) on a first-node home mesh.

A `slancha-mesh` node serves a *specialist* but cannot *route* — a home mesh
needs a router (`slancha-local serve`, the OpenAI-compat endpoint that
pareto-ranks + fans out) somewhere. When you're the first node and no router
is reachable, `ensure_router` can install + launch one.

Detection is real + cheap; install/launch are **injected** (defaults shell out
to `pip` / `slancha-local serve`) so the decision logic is unit-testable
without spawning anything. install/launch are gated behind an explicit
`install=True` (the CLI's `--with-router`) — never silent, matching the
"ask before heavy installs" judgment boundary in AGENTS.md.

Cross-repo note: slancha-local currently *pushes* heartbeats
(`SLANCHA_MESH_REGISTRY_URL`). Its consumption of pull discovery
(`mesh.discovery`) is the follow-up wire; this module only stands the router
process up.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from typing import Callable

from mesh.discovery import parse_specialist_peers

# slancha-local exposes both console scripts (pyproject [project.scripts]).
ROUTER_COMMANDS = ("slancha-local", "slancha")
# Not on PyPI yet → override with a path/git spec via env or --router-spec.
INSTALL_SPEC_ENV = "SLANCHA_LOCAL_INSTALL_SPEC"
DEFAULT_INSTALL_SPEC = "slancha-local"
GATEWAY_TAG = "tag:gateway"


@dataclass(frozen=True)
class RouterStatus:
    """Is a router available to this node?"""

    on_path: bool  # a slancha-local/slancha command is installed locally
    gateway_peer: bool  # a tag:gateway peer (a router) is reachable on the tailnet

    @property
    def present(self) -> bool:
        return self.on_path or self.gateway_peer


def detect_router(
    status_json: dict | str | None = None,
    *,
    which: Callable[[str], str | None] = shutil.which,
) -> RouterStatus:
    """Is a router installed locally or already reachable on the tailnet?"""
    on_path = any(which(cmd) for cmd in ROUTER_COMMANDS)
    gateway_peer = False
    if status_json is not None:
        gateway_peer = bool(
            parse_specialist_peers(status_json, specialist_tag=GATEWAY_TAG, include_self=False)
        )
    return RouterStatus(on_path=on_path, gateway_peer=gateway_peer)


def _pip_install(spec: str) -> tuple[bool, str]:
    """Default installer: `pip install <spec>`. Returns (ok, message)."""
    proc = subprocess.run(
        [sys.executable, "-m", "pip", "install", spec],
        capture_output=True, text=True, check=False,
    )
    if proc.returncode == 0:
        return True, "installed"
    return False, (proc.stderr or proc.stdout or "pip failed").strip()[:300]


def _launch_serve() -> int:
    """Default launcher: spawn `slancha-local serve` detached. Returns pid."""
    cmd = shutil.which("slancha-local") or shutil.which("slancha") or "slancha-local"
    proc = subprocess.Popen(  # noqa: S603 — fixed command, detached router
        [cmd, "serve"], start_new_session=True,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    return proc.pid


def _install_step(spec: str) -> dict:
    return {"action": "run", "command": f"pip install {spec}",
            "why": "slancha-local (the router) is not installed"}


def _launch_step() -> dict:
    return {"action": "run", "command": "slancha-local serve",
            "why": "start the local router so this mesh can serve traffic"}


def ensure_router(
    *,
    install: bool = False,
    install_spec: str | None = None,
    status_json: dict | str | None = None,
    which: Callable[[str], str | None] = shutil.which,
    detector: Callable[..., RouterStatus] | None = None,
    installer: Callable[[str], tuple[bool, str]] = _pip_install,
    launcher: Callable[[], int] = _launch_serve,
) -> dict:
    """Ensure a router is available; install + launch one when asked.

    - A `tag:gateway` peer already reachable → no-op.
    - `install=False` → return `instruct` with the exact commands (don't act).
    - `install=True` → install slancha-local if absent, then launch
      `slancha-local serve` detached. Install failure is reported, not raised.

    Returns a dict with `action` ∈ {none, instruct, launched, install_failed}
    plus `next_steps` / `pid` / `error` as relevant.
    """
    det = detector or detect_router
    status = det(status_json, which=which)

    if status.gateway_peer:
        return {"action": "none",
                "reason": "a router (tag:gateway) is already reachable on the tailnet"}

    spec = install_spec or os.environ.get(INSTALL_SPEC_ENV) or DEFAULT_INSTALL_SPEC

    if not install:
        steps = []
        if not status.on_path:
            steps.append(_install_step(spec))
        steps.append(_launch_step())
        return {"action": "instruct",
                "reason": "no router present; first-node home mesh needs one",
                "next_steps": steps}

    if not status.on_path:
        ok, msg = installer(spec)
        if not ok:
            return {"action": "install_failed", "error": msg,
                    "next_steps": [_install_step(spec), _launch_step()]}

    pid = launcher()
    return {"action": "launched", "pid": pid,
            "message": "launched `slancha-local serve` (detached)"}


__all__ = ["RouterStatus", "detect_router", "ensure_router"]
