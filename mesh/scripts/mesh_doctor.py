"""mesh-doctor — diagnose common slancha-mesh deployment issues.

Walks the local mesh-related state + the configured registry endpoint
and emits a pass/warn/fail table per check. Modeled on `wire doctor`
(see ~/Source/wire); the goal is to make "is the mesh OK on this host?"
a 5-second answer instead of a 5-minute log dive.

Checks (alphabetical groups):

  ENV     SLANCHA_MESH_REGISTRY_URL set? SLANCHA_NODE_TOKEN set?
  HARDWARE  nvidia-smi available? (mesh.gpu.probe requires it for non-CPU nodes)
  REGISTRY  GET /health → 200? GET /registry → has this node?
  AUTH    if token set, does it actually authenticate?
  SYSTEMD (Linux only)  mesh-registry / mesh-nightly-smoke.timer /
                       mesh-dashboard.service active state.
  PORTS   port collision check on 8088 (registry default)

Usage:
    python -m mesh.scripts.mesh_doctor
    python -m mesh.scripts.mesh_doctor --json
    python -m mesh.scripts.mesh_doctor --registry-url http://...:8088

Exit codes:
    0 — all checks pass (or warn)
    1 — at least one fail
    2 — invocation error
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import socket
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from typing import Callable, Literal


CheckStatus = Literal["pass", "warn", "fail", "skip"]

DEFAULT_REGISTRY_URL = "http://127.0.0.1:8088"
REGISTRY_URL_ENV = "SLANCHA_MESH_REGISTRY_URL"
NODE_TOKEN_ENV = "SLANCHA_NODE_TOKEN"
NODE_ID_ENV = "SLANCHA_NODE_ID"


@dataclass
class CheckResult:
    """One row in the doctor's output table."""

    id: str
    status: CheckStatus
    detail: str
    fix: str = ""


@dataclass
class DoctorReport:
    """Full diagnostic — list of CheckResults + an aggregate verdict."""

    checks: list[CheckResult] = field(default_factory=list)
    verdict: Literal["all-green", "warnings", "failures"] = "all-green"

    def append(self, result: CheckResult) -> None:
        self.checks.append(result)

    def finalize(self) -> None:
        statuses = {c.status for c in self.checks}
        if "fail" in statuses:
            self.verdict = "failures"
        elif "warn" in statuses:
            self.verdict = "warnings"
        else:
            self.verdict = "all-green"


# ---------------------------------------------------------------------------
# Individual checks — each returns a CheckResult
# ---------------------------------------------------------------------------


def check_registry_url_env() -> CheckResult:
    """SLANCHA_MESH_REGISTRY_URL set in env?"""
    val = os.environ.get(REGISTRY_URL_ENV, "").strip()
    if val:
        return CheckResult(
            id="env.registry_url",
            status="pass",
            detail=f"{REGISTRY_URL_ENV}={val}",
        )
    return CheckResult(
        id="env.registry_url",
        status="warn",
        detail=f"{REGISTRY_URL_ENV} unset",
        fix=f"export {REGISTRY_URL_ENV}={DEFAULT_REGISTRY_URL}",
    )


def check_node_token_env() -> CheckResult:
    """SLANCHA_NODE_TOKEN — informational; unset = dev mode (no auth)."""
    val = os.environ.get(NODE_TOKEN_ENV, "").strip()
    if val:
        return CheckResult(
            id="env.node_token",
            status="pass",
            detail="SLANCHA_NODE_TOKEN set (auth enforced)",
        )
    return CheckResult(
        id="env.node_token",
        status="warn",
        detail="SLANCHA_NODE_TOKEN unset — dev mode (no auth)",
        fix="set on production deploys; unset is fine for local",
    )


def check_nvidia_smi() -> CheckResult:
    """nvidia-smi available? Skipped on Darwin (Macs don't ship it)."""
    if platform.system() == "Darwin":
        return CheckResult(
            id="hardware.nvidia_smi",
            status="skip",
            detail="Darwin host — nvidia-smi not applicable",
        )
    if shutil.which("nvidia-smi"):
        return CheckResult(
            id="hardware.nvidia_smi",
            status="pass",
            detail="nvidia-smi on PATH",
        )
    return CheckResult(
        id="hardware.nvidia_smi",
        status="warn",
        detail="nvidia-smi not found — mesh.gpu.probe will report empty",
        fix="install NVIDIA driver; or skip GPU coordination for this node",
    )


def check_registry_health(registry_url: str, timeout: float = 3.0) -> CheckResult:
    """GET {registry_url}/health → 200?"""
    try:
        import httpx
    except ImportError:
        return CheckResult(
            id="registry.health",
            status="skip",
            detail="httpx not installed; cannot probe registry",
        )
    try:
        resp = httpx.get(f"{registry_url.rstrip('/')}/health", timeout=timeout)
        if resp.status_code == 200:
            body = resp.json()
            auth_req = body.get("auth_required", False)
            return CheckResult(
                id="registry.health",
                status="pass",
                detail=f"/health → 200, auth_required={auth_req}",
            )
        return CheckResult(
            id="registry.health",
            status="fail",
            detail=f"/health → {resp.status_code} (expected 200)",
            fix="check `journalctl --user -u mesh-registry.service`",
        )
    except (httpx.HTTPError, ConnectionError) as exc:
        return CheckResult(
            id="registry.health",
            status="fail",
            detail=f"unreachable: {type(exc).__name__}",
            fix=(
                "1) start the registry: systemctl --user start mesh-registry.service "
                "(or `uvicorn mesh.registry_app:app --port 8088`)  "
                "2) check the URL is correct"
            ),
        )


def check_registry_lists_this_node(
    registry_url: str, expected_node_id: str | None = None, token: str | None = None,
) -> CheckResult:
    """GET /registry — is this host's node_id in the snapshot?

    Helpful for confirming THIS deployment is actually heartbeating to
    the registry. Cross-checks node_id env / hostname-derived id against
    the registry's known nodes.
    """
    if expected_node_id is None:
        expected_node_id = os.environ.get(NODE_ID_ENV) or socket.gethostname()
    try:
        import httpx
    except ImportError:
        return CheckResult(
            id="registry.has_this_node",
            status="skip",
            detail="httpx not installed",
        )
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    try:
        resp = httpx.get(
            f"{registry_url.rstrip('/')}/registry", headers=headers, timeout=3.0,
        )
    except (httpx.HTTPError, ConnectionError) as exc:
        return CheckResult(
            id="registry.has_this_node",
            status="fail",
            detail=f"registry unreachable for snapshot read: {type(exc).__name__}",
        )
    if resp.status_code != 200:
        return CheckResult(
            id="registry.has_this_node",
            status="fail",
            detail=f"/registry → {resp.status_code} (expected 200)",
            fix="check token; check registry logs",
        )
    nodes = resp.json().get("snapshot", {}).get("nodes", {})
    if not nodes:
        return CheckResult(
            id="registry.has_this_node",
            status="warn",
            detail="registry has zero nodes (no heartbeats received yet)",
            fix="start slancha-local or a serve daemon with SLANCHA_MESH_REGISTRY_URL set",
        )
    if expected_node_id in nodes:
        return CheckResult(
            id="registry.has_this_node",
            status="pass",
            detail=f"node_id={expected_node_id} present in snapshot",
        )
    return CheckResult(
        id="registry.has_this_node",
        status="warn",
        detail=(
            f"node_id={expected_node_id} NOT in snapshot.nodes "
            f"(known: {sorted(nodes)[:3]}{'...' if len(nodes) > 3 else ''})"
        ),
        fix=(
            "1) heartbeat hasn't arrived yet (wait ~5s + retry)  "
            "2) SLANCHA_NODE_ID env disagrees with what the daemon posts  "
            "3) the daemon isn't actually running"
        ),
    )


def check_auth_works(registry_url: str, token: str) -> CheckResult:
    """If token set, does it actually authenticate?"""
    try:
        import httpx
    except ImportError:
        return CheckResult(
            id="auth.token_valid",
            status="skip",
            detail="httpx not installed",
        )
    headers = {"Authorization": f"Bearer {token}"}
    try:
        resp = httpx.get(
            f"{registry_url.rstrip('/')}/registry", headers=headers, timeout=3.0,
        )
    except (httpx.HTTPError, ConnectionError) as exc:
        return CheckResult(
            id="auth.token_valid",
            status="fail",
            detail=f"registry unreachable: {type(exc).__name__}",
        )
    if resp.status_code == 200:
        return CheckResult(
            id="auth.token_valid",
            status="pass",
            detail="bearer token accepted",
        )
    if resp.status_code in (401, 403):
        return CheckResult(
            id="auth.token_valid",
            status="fail",
            detail=f"token rejected ({resp.status_code})",
            fix=(
                "1) SLANCHA_NODE_TOKEN here != registry's expected  "
                "2) registry may be in dev mode (no auth) — try unset token"
            ),
        )
    return CheckResult(
        id="auth.token_valid",
        status="fail",
        detail=f"unexpected status {resp.status_code}",
    )


def check_systemd_unit(unit: str) -> CheckResult:
    """systemctl --user is-active {unit}? Skipped on non-Linux."""
    if platform.system() != "Linux":
        return CheckResult(
            id=f"systemd.{unit}",
            status="skip",
            detail="non-Linux host; systemctl not applicable",
        )
    if not shutil.which("systemctl"):
        return CheckResult(
            id=f"systemd.{unit}",
            status="skip",
            detail="systemctl not on PATH",
        )
    result = subprocess.run(
        ["systemctl", "--user", "is-active", unit],
        capture_output=True, text=True, check=False,
    )
    state = result.stdout.strip()
    if state == "active":
        return CheckResult(
            id=f"systemd.{unit}",
            status="pass",
            detail=f"unit {unit} is active",
        )
    if state == "inactive":
        return CheckResult(
            id=f"systemd.{unit}",
            status="warn",
            detail=f"unit {unit} is inactive (not enabled or stopped)",
            fix=f"systemctl --user enable --now {unit}",
        )
    return CheckResult(
        id=f"systemd.{unit}",
        status="warn",
        detail=f"unit {unit}: {state or 'unknown'}",
        fix=f"systemctl --user status {unit} for details",
    )


def _port_has_listener(port: int) -> bool:
    """True if something is accepting connections on 127.0.0.1:{port}."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        return sock.connect_ex(("127.0.0.1", port)) == 0
    finally:
        sock.close()


def check_port_listener(port: int) -> CheckResult:
    """Is port bound locally? Useful for collision detection on 8088."""
    if _port_has_listener(port):
        return CheckResult(
            id=f"ports.{port}",
            status="pass",
            detail=f"port {port} has a listener (likely the registry)",
        )
    return CheckResult(
        id=f"ports.{port}",
        status="warn",
        detail=f"port {port} has no listener",
        fix="if expecting the registry here, start it",
    )


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def run_doctor(
    registry_url: str | None = None,
    token: str | None = None,
    skip_systemd: bool = False,
) -> DoctorReport:
    """Run every check + return a finalized DoctorReport."""
    if registry_url is None:
        registry_url = os.environ.get(REGISTRY_URL_ENV) or DEFAULT_REGISTRY_URL
    if token is None:
        token = os.environ.get(NODE_TOKEN_ENV) or None

    report = DoctorReport()
    report.append(check_registry_url_env())
    report.append(check_node_token_env())
    report.append(check_nvidia_smi())
    report.append(check_registry_health(registry_url))
    report.append(check_registry_lists_this_node(registry_url, token=token))
    if token:
        report.append(check_auth_works(registry_url, token))
    if not skip_systemd:
        for unit in (
            "mesh-registry.service",
            "mesh-nightly-smoke.timer",
            "mesh-dashboard.service",
        ):
            report.append(check_systemd_unit(unit))
    # Port collision against the registry's default
    expected_port = 8088
    if registry_url and ":" in registry_url:
        try:
            expected_port = int(registry_url.rsplit(":", 1)[-1].rstrip("/"))
        except ValueError:
            pass
    report.append(check_port_listener(expected_port))
    report.finalize()
    return report


# ---------------------------------------------------------------------------
# Pull / tailnet model checks (v0.0.7) — the `slancha-mesh doctor` surface.
# The checks above target the push/central-registry model; these target a
# node that joins a tailnet and is pull-discovered.
# ---------------------------------------------------------------------------

NODE_INFO_PORT = 8088


def check_tailnet_specialist_ready() -> CheckResult:
    """On the tailnet AND advertising tag:specialist? (else gateway can't reach)."""
    from mesh.tailnet import DEFAULT_SPECIALIST_TAG, TailnetConfig, tailnet_status

    status = tailnet_status(TailnetConfig(enabled=True))
    if status is None:
        return CheckResult(
            id="tailnet.specialist_ready", status="skip",
            detail="not on a tailnet (no `tailscale status`) — loopback/dev mode",
        )
    self_obj = status.get("Self") or {}
    online = bool(self_obj.get("Online"))
    tags = self_obj.get("Tags") or []
    host = (self_obj.get("DNSName") or "").rstrip(".")
    if online and DEFAULT_SPECIALIST_TAG in tags:
        return CheckResult(
            id="tailnet.specialist_ready", status="pass",
            detail=f"online as {DEFAULT_SPECIALIST_TAG} ({host})",
        )
    if not online:
        return CheckResult(
            id="tailnet.specialist_ready", status="fail",
            detail="node offline on the tailnet",
            fix="check tailscaled; re-run `slancha-mesh up --key <tagged-key>`",
        )
    return CheckResult(
        id="tailnet.specialist_ready", status="fail",
        detail=f"missing {DEFAULT_SPECIALIST_TAG} (tags={tags}) — gateway can't discover this node",
        fix="re-join tagged: `slancha-mesh up --key <tagged-key>`",
    )


def check_recommended_engine_installed() -> CheckResult:
    """Is the hardware-recommended serving engine installed?"""
    from mesh.engine_select import recommend_engine
    from mesh.probe import probe_node

    probe = probe_node()
    rec = recommend_engine(probe)
    if rec.installed:
        return CheckResult(
            id="engine.installed", status="pass",
            detail=f"recommended engine {rec.backend} ({rec.quant}) present",
        )
    return CheckResult(
        id="engine.installed", status="warn",
        detail=f"recommended engine {rec.backend} not in available_backends={list(probe.available_backends)}",
        fix=f"install {rec.backend} — {rec.rationale}",
    )


def check_router_reachable() -> CheckResult:
    """Is a router (tag:gateway peer, or local slancha-local) available?"""
    from mesh.router_bootstrap import detect_router
    from mesh.tailnet import TailnetConfig, tailnet_status

    status = tailnet_status(TailnetConfig(enabled=True))
    st = detect_router(status)
    if st.gateway_peer:
        return CheckResult(
            id="router.reachable", status="pass",
            detail="a tag:gateway router is reachable on the tailnet",
        )
    if st.on_path:
        return CheckResult(
            id="router.reachable", status="warn",
            detail="slancha-local installed locally but no gateway peer seen",
            fix="start it: `slancha-local serve`",
        )
    return CheckResult(
        id="router.reachable", status="warn",
        detail="no router found (no tag:gateway peer, no local slancha-local)",
        fix="`slancha-mesh up --with-router` (first-node home mesh), or run a router elsewhere",
    )


# Common model-server ports OUTSIDE the gateway ACL accept-set. A node that
# serves here (slancha-local's :8000 default, a vLLM dev :8001) heartbeats +
# registers fine but is un-routable — the gateway is not permitted to dial it.
OFF_ACL_MODEL_PORTS = (8000, 8001)


def check_model_port_acl_reachable(
    is_listening: Callable[[int], bool] | None = None,
) -> CheckResult:
    """Is a served model port inside the gateway ACL accept-set?

    The tailnet ACL only opens the convention model ports
    (`tag:gateway -> tag:specialist:8003,8004` — `mesh.tailnet.DEFAULT_MODEL_PORTS`).
    A node serving OUTSIDE that set (e.g. slancha-local's :8000 default)
    registers + heartbeats fine but is **un-routable**: the gateway is not
    permitted to dial it (slancha-mesh#8, Windows dogfood). This turns that
    silent failure into a loud, actionable warning.

    Detection mirrors `check_port_listener` (loopback connect) and is advisory:
    it only warns on a *positive* off-ACL listener, so a node serving on the
    tailnet interface alone degrades to `skip`, never a false alarm.
    """
    from mesh.tailnet import DEFAULT_MODEL_PORTS

    if is_listening is None:
        is_listening = _port_has_listener
    acl_ports = sorted(set(DEFAULT_MODEL_PORTS.values()))
    on_acl = [p for p in acl_ports if is_listening(p)]
    if on_acl:
        return CheckResult(
            id="ports.model_acl", status="pass",
            detail=f"serving on ACL model port(s) {on_acl} — gateway-routable",
        )
    on_off = [p for p in OFF_ACL_MODEL_PORTS if is_listening(p)]
    if on_off:
        acl_str = ",".join(str(p) for p in acl_ports)
        return CheckResult(
            id="ports.model_acl", status="warn",
            detail=(
                f"model server on {on_off} but nothing on the ACL accept-set "
                f"{acl_ports} — the gateway ACL (tag:gateway -> "
                f"tag:specialist:{acl_str}) cannot reach this node. Invariant: "
                f"a discoverable node's advertised model URL MUST be "
                f"ACL-reachable, else it is 'up but unroutable'"
            ),
            fix=f"re-serve on a convention model port: `slancha-mesh up --base-port {acl_ports[0]}`",
        )
    return CheckResult(
        id="ports.model_acl", status="skip",
        detail=(
            f"no model server detected on the ACL set {acl_ports} or common "
            f"off-ACL ports {list(OFF_ACL_MODEL_PORTS)}"
        ),
    )


def check_node_info_discoverable(
    node_info_port: int = NODE_INFO_PORT,
    is_listening: Callable[[int], bool] | None = None,
) -> CheckResult:
    """Is this node *pull-discoverable* — does it expose a node-info endpoint?

    Pull discovery (`slancha-mesh discover`) fetches each peer's `/models`
    from its node-info port (default :8088 — what `build_node` / `slancha-mesh
    up` serves). A node can be on the tailnet, tagged `tag:specialist`, and
    serving models yet be **invisible** to discovery if nothing answers on the
    node-info port: the "tagged but undiscoverable" trap (a raw `vllm serve`
    with a tailscale tag, no node-server). This surfaces that as a loud,
    actionable warning rather than a silent zero-result discovery pass.

    Detection mirrors `check_model_port_acl_reachable` (loopback connect,
    advisory); `is_listening` is injectable for tests.
    """
    from mesh.tailnet import DEFAULT_MODEL_PORTS

    if is_listening is None:
        is_listening = _port_has_listener
    if is_listening(node_info_port):
        return CheckResult(
            id="ports.node_info", status="pass",
            detail=f"node-info on :{node_info_port} — pull-discoverable via `slancha-mesh discover`",
        )
    serving = [p for p in sorted(set(DEFAULT_MODEL_PORTS.values())) if is_listening(p)]
    if serving:
        return CheckResult(
            id="ports.node_info", status="warn",
            detail=(
                f"serving on model port(s) {serving} but NO node-info responder on "
                f":{node_info_port} — pull consumers can't fetch /models, so this node "
                f"is tagged but undiscoverable"
            ),
            fix="`slancha-mesh up` serves node-info; a bare `vllm serve` + tailscale tag is not enough",
        )
    return CheckResult(
        id="ports.node_info", status="skip",
        detail=f"no node-info on :{node_info_port} and no model server detected (not a mesh node here)",
    )


def run_node_doctor(node_info_port: int = NODE_INFO_PORT) -> DoctorReport:
    """Pull/tailnet-model diagnostic for a `slancha-mesh` specialist node."""
    report = DoctorReport()
    report.append(check_tailnet_specialist_ready())
    report.append(check_recommended_engine_installed())
    report.append(check_router_reachable())
    report.append(check_model_port_acl_reachable())
    report.append(check_nvidia_smi())
    report.append(check_node_token_env())
    report.append(check_node_info_discoverable(node_info_port))
    report.finalize()
    return report


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


_STATUS_GLYPH = {
    "pass": "✓",
    "warn": "!",
    "fail": "✗",
    "skip": "-",
}


def render_text(report: DoctorReport) -> str:
    """Human-readable status table."""
    lines = ["mesh-doctor — diagnostic report", ""]
    max_id_len = max((len(c.id) for c in report.checks), default=20)
    for c in report.checks:
        glyph = _STATUS_GLYPH.get(c.status, "?")
        lines.append(f"  {glyph} [{c.status:4}] {c.id:<{max_id_len}}  {c.detail}")
        if c.status in ("warn", "fail") and c.fix:
            lines.append(f"           fix: {c.fix}")
    lines.append("")
    lines.append(f"verdict: {report.verdict}")
    return "\n".join(lines)


def render_json(report: DoctorReport) -> str:
    return json.dumps(
        {
            "verdict": report.verdict,
            "checks": [asdict(c) for c in report.checks],
        },
        indent=2,
    )


def render_fixes(report: DoctorReport) -> str:
    """Emit only the fix-hints, formatted as a shell-runnable script.

    Useful for: run `mesh-doctor` once → see what's broken → run with
    `--print-fixes` to dump the recommended commands → copy-paste-run.

    Each fix is prefixed with `# <check_id>: <detail>` so the operator
    can see what they're about to do before they run it. Commands the
    fix-hint already shows (multi-step joined by "  ") become separate
    bash lines.

    Returns empty string if no fixes apply (all checks pass / skip).
    """
    lines: list[str] = []
    actionable = [c for c in report.checks if c.status in ("warn", "fail") and c.fix]
    if not actionable:
        return ""

    lines.append("#!/usr/bin/env bash")
    lines.append("# mesh-doctor --print-fixes")
    lines.append("# Apply each section to address the corresponding diagnostic.")
    lines.append("# Review each command before running; some are best-effort.")
    lines.append("set -e")
    lines.append("")
    for c in actionable:
        lines.append(f"# [{c.status}] {c.id}: {c.detail}")
        # Multi-step fixes joined by "  " (two spaces) in the original;
        # split into ordered shell lines so a copy-paste runs them
        # sequentially.
        parts = [p.strip() for p in c.fix.split("  ") if p.strip()]
        for p in parts:
            lines.append(p)
        lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Diagnose slancha-mesh deployment.")
    ap.add_argument("--registry-url", default=None, help="Override SLANCHA_MESH_REGISTRY_URL.")
    ap.add_argument("--token", default=None, help="Override SLANCHA_NODE_TOKEN.")
    ap.add_argument("--json", action="store_true", help="Emit JSON instead of text.")
    ap.add_argument(
        "--print-fixes",
        action="store_true",
        help="Emit a shell-runnable script of the recommended fixes only.",
    )
    ap.add_argument("--skip-systemd", action="store_true", help="Skip systemd checks.")
    args = ap.parse_args(argv)

    if args.json and args.print_fixes:
        print("error: --json and --print-fixes are mutually exclusive", file=sys.stderr)
        return 2

    report = run_doctor(
        registry_url=args.registry_url,
        token=args.token,
        skip_systemd=args.skip_systemd,
    )

    if args.json:
        print(render_json(report))
    elif args.print_fixes:
        fixes = render_fixes(report)
        if fixes:
            print(fixes)
        else:
            print("# mesh-doctor: nothing to fix.", file=sys.stderr)
    else:
        print(render_text(report))

    return 0 if report.verdict != "failures" else 1


if __name__ == "__main__":
    sys.exit(main())
