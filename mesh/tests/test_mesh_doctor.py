"""Tests for mesh.scripts.mesh_doctor — diagnostic checks + rendering."""

from __future__ import annotations

import json

import httpx
import pytest

from mesh.scripts.mesh_doctor import (
    CheckResult,
    DoctorReport,
    NODE_ID_ENV,
    NODE_TOKEN_ENV,
    REGISTRY_URL_ENV,
    check_node_token_env,
    check_nvidia_smi,
    check_port_listener,
    check_registry_health,
    check_registry_lists_this_node,
    check_registry_url_env,
    check_systemd_unit,
    render_json,
    render_text,
    run_doctor,
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
