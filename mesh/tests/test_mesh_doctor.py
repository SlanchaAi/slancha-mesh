"""Tests for mesh.scripts.mesh_doctor — diagnostic checks + rendering."""

from __future__ import annotations

import json
import types

import httpx

from mesh.scripts.mesh_doctor import (
    CheckResult,
    DoctorReport,
    NODE_ID_ENV,
    NODE_INFO_PORT,
    NODE_TOKEN_ENV,
    REGISTRY_URL_ENV,
    check_model_port_acl_reachable,
    check_node_token_env,
    check_nvidia_smi,
    check_port_listener,
    check_recommended_engine_installed,
    check_registry_health,
    check_registry_lists_this_node,
    check_registry_url_env,
    check_router_reachable,
    check_systemd_unit,
    check_tailnet_specialist_ready,
    render_json,
    render_text,
    run_doctor,
    run_node_doctor,
)


class _StubResponse:
    def __init__(self, status_code: int, body: dict | None = None) -> None:
        self.status_code = status_code
        self._body = body or {}

    def json(self) -> dict:
        return self._body


def _stub_httpx_get(monkeypatch, status_code: int = 200, body: dict | None = None):
    def _get(url, headers=None, timeout=None):
        return _StubResponse(status_code, body)
    monkeypatch.setattr(httpx, "get", _get)


def _stub_httpx_raises(monkeypatch, exc: Exception):
    def _get(url, headers=None, timeout=None):
        raise exc
    monkeypatch.setattr(httpx, "get", _get)


def test_check_registry_url_env_set(monkeypatch):
    monkeypatch.setenv(REGISTRY_URL_ENV, "http://x:8088")
    r = check_registry_url_env()
    assert r.status == "pass"
    assert "http://x:8088" in r.detail


def test_check_registry_url_env_unset(monkeypatch):
    monkeypatch.delenv(REGISTRY_URL_ENV, raising=False)
    r = check_registry_url_env()
    assert r.status == "warn"
    assert r.fix


def test_check_node_token_env_set(monkeypatch):
    monkeypatch.setenv(NODE_TOKEN_ENV, "secret")
    r = check_node_token_env()
    assert r.status == "pass"


def test_check_node_token_env_unset(monkeypatch):
    monkeypatch.delenv(NODE_TOKEN_ENV, raising=False)
    r = check_node_token_env()
    assert r.status == "warn"
    assert "dev mode" in r.detail


def test_check_nvidia_smi_skipped_on_darwin(monkeypatch):
    import platform
    monkeypatch.setattr(platform, "system", lambda: "Darwin")
    r = check_nvidia_smi()
    assert r.status == "skip"


def test_check_nvidia_smi_pass_on_linux_with_binary(monkeypatch):
    import platform
    import shutil
    monkeypatch.setattr(platform, "system", lambda: "Linux")
    monkeypatch.setattr(shutil, "which", lambda b: "/usr/bin/nvidia-smi" if b == "nvidia-smi" else None)
    r = check_nvidia_smi()
    assert r.status == "pass"


def test_check_nvidia_smi_warn_on_linux_without_binary(monkeypatch):
    import platform
    import shutil
    monkeypatch.setattr(platform, "system", lambda: "Linux")
    monkeypatch.setattr(shutil, "which", lambda b: None)
    r = check_nvidia_smi()
    assert r.status == "warn"


def test_check_registry_health_pass(monkeypatch):
    _stub_httpx_get(monkeypatch, 200, {"status": "ok", "auth_required": False})
    r = check_registry_health("http://x:8088")
    assert r.status == "pass"


def test_check_registry_health_fail_non_200(monkeypatch):
    _stub_httpx_get(monkeypatch, 503, {"detail": "down"})
    r = check_registry_health("http://x:8088")
    assert r.status == "fail"
    assert "503" in r.detail


def test_check_registry_health_unreachable(monkeypatch):
    _stub_httpx_raises(monkeypatch, httpx.ConnectError("refused"))
    r = check_registry_health("http://x:8088")
    assert r.status == "fail"


def test_check_registry_lists_this_node_pass(monkeypatch):
    monkeypatch.setenv(NODE_ID_ENV, "node-a")
    body = {"snapshot": {"nodes": {"node-a": {"node_id": "node-a"}}}}
    _stub_httpx_get(monkeypatch, 200, body)
    r = check_registry_lists_this_node("http://x:8088")
    assert r.status == "pass"


def test_check_registry_lists_this_node_warn_missing(monkeypatch):
    monkeypatch.setenv(NODE_ID_ENV, "node-a")
    body = {"snapshot": {"nodes": {"node-b": {"node_id": "node-b"}}}}
    _stub_httpx_get(monkeypatch, 200, body)
    r = check_registry_lists_this_node("http://x:8088")
    assert r.status == "warn"


def test_check_registry_lists_this_node_warn_no_nodes(monkeypatch):
    body = {"snapshot": {"nodes": {}}}
    _stub_httpx_get(monkeypatch, 200, body)
    r = check_registry_lists_this_node("http://x:8088")
    assert r.status == "warn"
    assert "zero nodes" in r.detail


def test_check_systemd_unit_skipped_on_darwin(monkeypatch):
    import platform
    monkeypatch.setattr(platform, "system", lambda: "Darwin")
    r = check_systemd_unit("mesh-registry.service")
    assert r.status == "skip"


def test_check_systemd_unit_active(monkeypatch):
    import platform
    import shutil
    import subprocess
    monkeypatch.setattr(platform, "system", lambda: "Linux")
    monkeypatch.setattr(shutil, "which", lambda b: "/usr/bin/systemctl")
    class _R:
        stdout = "active\n"
        returncode = 0
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: _R())
    r = check_systemd_unit("mesh-registry.service")
    assert r.status == "pass"


def test_check_systemd_unit_inactive(monkeypatch):
    import platform
    import shutil
    import subprocess
    monkeypatch.setattr(platform, "system", lambda: "Linux")
    monkeypatch.setattr(shutil, "which", lambda b: "/usr/bin/systemctl")
    class _R:
        stdout = "inactive\n"
        returncode = 3
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: _R())
    r = check_systemd_unit("mesh-registry.service")
    assert r.status == "warn"
    assert r.fix


def test_check_port_listener_returns_pass_or_warn():
    r = check_port_listener(8088)
    assert r.status in ("pass", "warn")


def test_check_model_port_acl_pass_on_acl_port():
    # serving on the vLLM convention port (8003) → gateway-routable
    r = check_model_port_acl_reachable(is_listening=lambda p: p == 8003)
    assert r.status == "pass"
    assert "8003" in r.detail


def test_check_model_port_acl_warns_on_off_acl_port():
    # slancha-local default :8000, nothing on the ACL set → the #8 silent failure
    r = check_model_port_acl_reachable(is_listening=lambda p: p == 8000)
    assert r.status == "warn"
    assert "8000" in r.detail
    assert "8003" in r.fix  # steers back to the convention port


def test_check_model_port_acl_prefers_acl_when_both_listening():
    # a dev vLLM on :8001 AND the real specialist on :8003 → still routable
    r = check_model_port_acl_reachable(is_listening=lambda p: p in (8001, 8003))
    assert r.status == "pass"


def test_check_model_port_acl_skip_when_nothing_listening():
    r = check_model_port_acl_reachable(is_listening=lambda p: False)
    assert r.status == "skip"


def test_run_doctor_returns_finalized_report(monkeypatch):
    import platform
    monkeypatch.delenv(REGISTRY_URL_ENV, raising=False)
    monkeypatch.delenv(NODE_TOKEN_ENV, raising=False)
    monkeypatch.setattr(platform, "system", lambda: "Darwin")
    _stub_httpx_raises(monkeypatch, httpx.ConnectError("refused"))

    report = run_doctor(registry_url="http://localhost:9999")
    assert isinstance(report, DoctorReport)
    ids = {c.id for c in report.checks}
    assert "env.registry_url" in ids
    assert "hardware.nvidia_smi" in ids
    assert "registry.health" in ids
    assert "ports.9999" in ids
    assert any(c.id.startswith("systemd.") for c in report.checks)
    assert report.verdict == "failures"


def test_run_doctor_skip_systemd_flag(monkeypatch):
    import platform
    monkeypatch.setattr(platform, "system", lambda: "Linux")
    _stub_httpx_raises(monkeypatch, httpx.ConnectError("refused"))
    report = run_doctor(registry_url="http://x:9999", skip_systemd=True)
    assert not any(c.id.startswith("systemd.") for c in report.checks)


def test_render_text_contains_verdict_and_glyphs():
    rep = DoctorReport()
    rep.append(CheckResult(id="x", status="pass", detail="ok"))
    rep.append(CheckResult(id="y", status="fail", detail="bad", fix="do thing"))
    rep.finalize()
    out = render_text(rep)
    assert "✓" in out
    assert "✗" in out
    assert "do thing" in out
    assert "failures" in out


def test_render_json_round_trips():
    rep = DoctorReport()
    rep.append(CheckResult(id="x", status="pass", detail="ok"))
    rep.finalize()
    out = render_json(rep)
    parsed = json.loads(out)
    assert parsed["verdict"] == "all-green"
    assert parsed["checks"][0]["id"] == "x"


def test_doctor_report_verdict_promotes_warn_to_warnings():
    rep = DoctorReport()
    rep.append(CheckResult(id="x", status="warn", detail="w"))
    rep.append(CheckResult(id="y", status="pass", detail="p"))
    rep.finalize()
    assert rep.verdict == "warnings"


def test_doctor_report_verdict_all_skip_is_all_green():
    rep = DoctorReport()
    rep.append(CheckResult(id="x", status="skip", detail="n/a"))
    rep.finalize()
    assert rep.verdict == "all-green"


# ---------------------------------------------------------------------------
# --print-fixes mode
# ---------------------------------------------------------------------------


from mesh.scripts.mesh_doctor import render_fixes  # noqa: E402 — grouped


def test_render_fixes_empty_when_all_passing():
    rep = DoctorReport()
    rep.append(CheckResult(id="x", status="pass", detail="ok"))
    rep.append(CheckResult(id="y", status="skip", detail="n/a"))
    rep.finalize()
    assert render_fixes(rep) == ""


def test_render_fixes_emits_shell_script_with_actionable_fixes():
    rep = DoctorReport()
    rep.append(CheckResult(id="x", status="pass", detail="ok"))
    rep.append(
        CheckResult(
            id="y", status="warn", detail="env unset",
            fix="export FOO=bar",
        ),
    )
    rep.append(
        CheckResult(
            id="z", status="fail", detail="thing down",
            fix="systemctl --user start thing.service",
        ),
    )
    rep.finalize()
    out = render_fixes(rep)
    assert out.startswith("#!/usr/bin/env bash")
    assert "set -e" in out
    assert "# [warn] y: env unset" in out
    assert "export FOO=bar" in out
    assert "# [fail] z: thing down" in out
    assert "systemctl --user start thing.service" in out


def test_render_fixes_handles_multi_step_fix_hints():
    """Fix hints with '  ' (two-space) separators split into ordered lines."""
    rep = DoctorReport()
    rep.append(
        CheckResult(
            id="a", status="fail", detail="multi",
            fix="step one  step two  step three",
        ),
    )
    rep.finalize()
    out = render_fixes(rep)
    lines = out.splitlines()
    # Each step appears as its own line in order
    one_idx = next(i for i, ln in enumerate(lines) if "step one" in ln)
    two_idx = next(i for i, ln in enumerate(lines) if "step two" in ln)
    three_idx = next(i for i, ln in enumerate(lines) if "step three" in ln)
    assert one_idx < two_idx < three_idx


def test_render_fixes_skips_checks_without_fix_field():
    rep = DoctorReport()
    rep.append(CheckResult(id="x", status="warn", detail="no fix here", fix=""))
    rep.finalize()
    assert render_fixes(rep) == ""


# ---------------------------------------------------------------------------
# Pull/tailnet node-doctor checks (the `slancha-mesh doctor` surface).
# These hit external probes via call-time local imports, so we monkeypatch
# the source modules (mesh.tailnet / mesh.probe / mesh.engine_select /
# mesh.router_bootstrap).
# ---------------------------------------------------------------------------


def test_check_tailnet_specialist_ready_skip_when_no_tailnet(monkeypatch):
    monkeypatch.setattr("mesh.tailnet.tailnet_status", lambda cfg: None)
    r = check_tailnet_specialist_ready()
    assert r.status == "skip"


def test_check_tailnet_specialist_ready_pass_when_online_and_tagged(monkeypatch):
    monkeypatch.setattr(
        "mesh.tailnet.tailnet_status",
        lambda cfg: {"Self": {"Online": True, "Tags": ["tag:specialist"], "DNSName": "n.ts.net."}},
    )
    r = check_tailnet_specialist_ready()
    assert r.status == "pass"
    assert "n.ts.net" in r.detail


def test_check_tailnet_specialist_ready_fail_when_offline(monkeypatch):
    monkeypatch.setattr(
        "mesh.tailnet.tailnet_status",
        lambda cfg: {"Self": {"Online": False, "Tags": ["tag:specialist"]}},
    )
    r = check_tailnet_specialist_ready()
    assert r.status == "fail"
    assert "offline" in r.detail


def test_check_tailnet_specialist_ready_fail_when_tag_missing(monkeypatch):
    monkeypatch.setattr(
        "mesh.tailnet.tailnet_status",
        lambda cfg: {"Self": {"Online": True, "Tags": ["tag:other"]}},
    )
    r = check_tailnet_specialist_ready()
    assert r.status == "fail"
    assert r.fix  # tells you to re-join tagged


def test_check_recommended_engine_pass_when_installed(monkeypatch):
    monkeypatch.setattr("mesh.probe.probe_node", lambda: types.SimpleNamespace(available_backends=["vllm"]))
    monkeypatch.setattr(
        "mesh.engine_select.recommend_engine",
        lambda probe: types.SimpleNamespace(installed=True, backend="vllm", quant="fp8", rationale="fits"),
    )
    r = check_recommended_engine_installed()
    assert r.status == "pass"
    assert "vllm" in r.detail


def test_check_recommended_engine_warn_when_missing(monkeypatch):
    monkeypatch.setattr("mesh.probe.probe_node", lambda: types.SimpleNamespace(available_backends=[]))
    monkeypatch.setattr(
        "mesh.engine_select.recommend_engine",
        lambda probe: types.SimpleNamespace(installed=False, backend="llamacpp", quant="gguf-q4", rationale="cpu node"),
    )
    r = check_recommended_engine_installed()
    assert r.status == "warn"
    assert "llamacpp" in r.fix


def test_check_router_reachable_pass_with_gateway_peer(monkeypatch):
    monkeypatch.setattr("mesh.tailnet.tailnet_status", lambda cfg: {})
    monkeypatch.setattr(
        "mesh.router_bootstrap.detect_router",
        lambda status: types.SimpleNamespace(gateway_peer=True, on_path=False),
    )
    r = check_router_reachable()
    assert r.status == "pass"


def test_check_router_reachable_warn_when_local_only(monkeypatch):
    monkeypatch.setattr("mesh.tailnet.tailnet_status", lambda cfg: {})
    monkeypatch.setattr(
        "mesh.router_bootstrap.detect_router",
        lambda status: types.SimpleNamespace(gateway_peer=False, on_path=True),
    )
    r = check_router_reachable()
    assert r.status == "warn"
    assert r.fix


def test_check_router_reachable_warn_when_no_router(monkeypatch):
    monkeypatch.setattr("mesh.tailnet.tailnet_status", lambda cfg: None)
    monkeypatch.setattr(
        "mesh.router_bootstrap.detect_router",
        lambda status: types.SimpleNamespace(gateway_peer=False, on_path=False),
    )
    r = check_router_reachable()
    assert r.status == "warn"


def test_run_node_doctor_assembles_expected_checks(monkeypatch):
    import platform

    monkeypatch.setattr("mesh.tailnet.tailnet_status", lambda cfg: None)
    monkeypatch.setattr(
        "mesh.router_bootstrap.detect_router",
        lambda status: types.SimpleNamespace(gateway_peer=False, on_path=False),
    )
    monkeypatch.setattr("mesh.probe.probe_node", lambda: types.SimpleNamespace(available_backends=[]))
    monkeypatch.setattr(
        "mesh.engine_select.recommend_engine",
        lambda probe: types.SimpleNamespace(installed=True, backend="vllm", quant="fp8", rationale="x"),
    )
    monkeypatch.setattr("mesh.scripts.mesh_doctor._port_has_listener", lambda p: False)
    monkeypatch.setattr(platform, "system", lambda: "Darwin")
    monkeypatch.delenv(NODE_TOKEN_ENV, raising=False)

    report = run_node_doctor()
    assert isinstance(report, DoctorReport)
    ids = {c.id for c in report.checks}
    assert {
        "tailnet.specialist_ready",
        "engine.installed",
        "router.reachable",
        "ports.model_acl",
        f"ports.{NODE_INFO_PORT}",
    } <= ids
    # tailnet skip + engine pass + router warn + model_acl skip + nvidia skip +
    # token warn + port warn → warnings, no failures.
    assert report.verdict == "warnings"
