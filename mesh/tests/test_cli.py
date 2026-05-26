"""`slancha-mesh` CLI — argument wiring + the non-blocking paths.

`up`'s server loop and `discover`'s live tailscale call aren't exercised
here (covered by the discovery/node-server unit tests); these lock the
parsing, the tailnet-config inference, and `up --dry-run` building a node
without starting anything.
"""

from __future__ import annotations

import pytest

from mesh.cli import _tailnet_from_args, auto_select, build_parser, main
from mesh.models import NodeProbe, SpecialistCard


def _parse(argv):
    return build_parser().parse_args(argv)


# ---------------------------------------------------------------------------
# parser
# ---------------------------------------------------------------------------


def test_up_parses_repeatable_specialist_and_ports():
    args = _parse(["up", "--specialist", "a", "--specialist", "b", "--base-port", "9000"])
    assert args.specialist == ["a", "b"]
    assert args.base_port == 9000


def test_missing_command_errors():
    with pytest.raises(SystemExit):
        _parse([])


# ---------------------------------------------------------------------------
# tailnet inference
# ---------------------------------------------------------------------------


def test_tailnet_enabled_when_key_present(monkeypatch):
    monkeypatch.delenv("SLANCHA_TAILNET_ENABLED", raising=False)
    args = _parse(["up", "--key", "tskey-1"])
    cfg = _tailnet_from_args(args)
    assert cfg.enabled is True


def test_tailnet_disabled_without_flags_or_env(monkeypatch):
    for k in list(__import__("os").environ):
        if k.startswith("SLANCHA_TAILNET_"):
            monkeypatch.delenv(k, raising=False)
    args = _parse(["up", "--specialist", "x"])
    cfg = _tailnet_from_args(args)
    assert cfg.enabled is False


def test_headscale_flags_flow_through():
    args = _parse(["up", "--key", "k", "--control-plane", "headscale", "--login-server", "https://hs"])
    cfg = _tailnet_from_args(args)
    assert cfg.control_plane == "headscale"
    assert cfg.login_server == "https://hs"


# ---------------------------------------------------------------------------
# up --dry-run (no tailnet → loopback, no server start)
# ---------------------------------------------------------------------------


def test_up_dry_run_builds_node_without_starting(monkeypatch, capsys, tmp_path):
    # A synthetic catalog with one non-vllm (NullBackend) specialist.
    card = tmp_path / "demo.toml"
    card.write_text(
        'model_id = "vendor/demo"\n'
        'specialist_id = "demo"\n'
        'domain = "code"\n'
        'difficulty_tiers = ["easy"]\n'
        'required_backend = "ollama"\n'
        'storage_gb = 4.0\n'
        'runtime_gb = 5.0\n'
        'min_vram_gb = 4.0\n'
        'context_window = 8192\n'
        'n_layers = 32\n'
    )
    for k in list(__import__("os").environ):
        if k.startswith("SLANCHA_TAILNET_"):
            monkeypatch.delenv(k, raising=False)

    rc = main(["up", "--specialist", "demo", "--catalog-dir", str(tmp_path), "--dry-run"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "--dry-run" in out
    assert "demo" in out


def _probe(vram=24.0, backends=("vllm",)) -> NodeProbe:
    return NodeProbe(
        node_id="auto-node", friendly_name="auto-box", chip="gb10", arch="aarch64",
        ram_total_gb=128.0, ram_available_gb=120.0, vram_available_gb=vram,
        available_backends=list(backends), disk_free_gb=1000.0,
    )


def _fit_card(sid: str, min_vram: float, backend: str = "vllm") -> SpecialistCard:
    return SpecialistCard(
        model_id=f"vendor/{sid}", specialist_id=sid, domain="code",
        difficulty_tiers=["easy"], required_backend=backend,
        storage_gb=min_vram, runtime_gb=min_vram, min_vram_gb=min_vram,
        context_window=8192, n_layers=32,
    )


def test_auto_select_picks_a_fitting_specialist():
    catalog = [_fit_card("small", 8.0), _fit_card("huge", 999.0)]
    chosen = auto_select(catalog, _probe(vram=24.0))
    assert chosen == ["small"]  # "huge" hard-filtered by VRAM


def test_auto_select_empty_when_nothing_fits():
    catalog = [_fit_card("huge", 999.0)]
    assert auto_select(catalog, _probe(vram=8.0)) == []


def test_plan_json_emits_engine_and_mesh_state(monkeypatch, capsys, tmp_path):
    card = tmp_path / "demo.toml"
    card.write_text(
        'model_id = "vendor/demo"\nspecialist_id = "demo"\ndomain = "code"\n'
        'difficulty_tiers = ["easy"]\nrequired_backend = "ollama"\nstorage_gb = 4.0\n'
        'runtime_gb = 5.0\nmin_vram_gb = 4.0\ncontext_window = 8192\nn_layers = 32\n'
    )
    # No tailnet → first_node; deterministic probe via monkeypatch.
    monkeypatch.setattr("mesh.cli.tailnet_status", lambda cfg: None)
    monkeypatch.setattr("mesh.probe.probe_node", lambda *a, **k: _probe(backends=("ollama",)))

    rc = main(["plan", "--catalog-dir", str(tmp_path), "--json"])
    assert rc == 0
    import json as _json
    out = _json.loads(capsys.readouterr().out)
    assert out["recommended_engine"]["backend"] in ("vllm", "ollama", "llamacpp", "mlx")
    assert out["mesh"]["state"] == "first_node"
    assert any(s.get("command", "").startswith("slancha-mesh up") for s in out["next_steps"])


def test_discover_json_no_tailnet_returns_error(monkeypatch, capsys):
    # No tailscale on PATH in CI → tailnet_status None → graceful exit 1.
    monkeypatch.setattr("mesh.cli.tailnet_status", lambda cfg: None)
    rc = main(["discover", "--json"])
    assert rc == 1
    assert "tailscale status" in capsys.readouterr().out


def test_doctor_json_emits_checks_and_verdict(monkeypatch, capsys):
    # No tailnet → specialist_ready skips; engine check runs off a probe.
    monkeypatch.setattr("mesh.tailnet.tailnet_status", lambda cfg: None)
    rc = main(["doctor", "--json"])
    import json as _json
    out = _json.loads(capsys.readouterr().out)
    assert "verdict" in out and isinstance(out["checks"], list)
    ids = {c["id"] for c in out["checks"]}
    assert {"tailnet.specialist_ready", "engine.installed", "router.reachable"} <= ids
    assert rc in (0, 1)


def test_up_with_router_dry_run_does_not_bootstrap(monkeypatch, capsys, tmp_path):
    # --dry-run must NOT install/launch a router (guarded in cmd_up).
    card = tmp_path / "demo.toml"
    card.write_text(
        'model_id="vendor/demo"\nspecialist_id="demo"\ndomain="code"\n'
        'difficulty_tiers=["easy"]\nrequired_backend="ollama"\nstorage_gb=4.0\n'
        'runtime_gb=5.0\nmin_vram_gb=4.0\ncontext_window=8192\nn_layers=32\n'
    )
    for k in list(__import__("os").environ):
        if k.startswith("SLANCHA_TAILNET_"):
            monkeypatch.delenv(k, raising=False)
    rc = main(["up", "--specialist", "demo", "--catalog-dir", str(tmp_path),
               "--with-router", "--dry-run"])
    assert rc == 0
    assert "[router]" not in capsys.readouterr().out  # router bootstrap skipped under --dry-run


def test_discover_renders_table(monkeypatch, capsys):
    # Inject a fake tailnet status + fetch via the cli seams.
    status = {
        "Self": {"DNSName": "self.ts.net.", "Online": True, "Tags": ["tag:specialist"]},
        "Peer": {},
    }
    monkeypatch.setattr("mesh.cli.tailnet_status", lambda cfg: status)

    def fake_make_fetch(**kw):
        def fetch(host, port):
            return {
                "object": "list",
                "data": [{
                    "id": "demo",
                    "object": "model",
                    "routing_meta": {
                        "model_id": "vendor/demo",
                        "domain": "code",
                        "capabilities": [],
                        "quality": {"router_observed": None},
                        "node_urls": ["http://ignored:8003"],
                    },
                }],
            }
        return fetch

    monkeypatch.setattr("mesh.cli.make_http_fetch", fake_make_fetch)
    rc = main(["discover"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "demo" in out
    assert "self.ts.net:8003" in out  # host-pinned to the dialed peer
