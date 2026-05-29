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


def test_discover_peer_flag_skips_tailscale_and_walks_explicit_hosts(monkeypatch, capsys):
    """`--peer` flag enables raw-LAN federation: no Tailscale required.

    The flag is what makes the README's 5-minute LAN quickstart work
    without forcing every LocalLLaMA-class user onto Tailscale.
    """

    def boom(_cfg):
        raise AssertionError("tailnet_status must NOT be called when --peer is set")

    monkeypatch.setattr("mesh.cli.tailnet_status", boom)

    dialed: list[str] = []

    def fake_make_fetch(**kw):
        def fetch(host, port):
            dialed.append(host)
            return {
                "object": "list",
                "data": [{
                    "id": f"demo-from-{host.replace('.', '-')}",
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
    rc = main(["discover", "--peer", "192.168.1.10", "--peer", "10.0.0.5"])
    assert rc == 0
    assert sorted(dialed) == ["10.0.0.5", "192.168.1.10"]
    out = capsys.readouterr().out
    # Both peers' specialists must surface in the table; host-pinned to the
    # peer we actually dialed (claim-hijack defense still applies).
    assert "demo-from-192-168-1-10" in out
    assert "demo-from-10-0-0-5" in out
    assert "192.168.1.10:8003" in out
    assert "10.0.0.5:8003" in out


def test_discover_no_peer_no_tailnet_still_errors(monkeypatch, capsys):
    """Without `--peer` AND without Tailscale, the operator gets the original
    error with a hint about `--peer` for raw-LAN discovery."""
    monkeypatch.setattr("mesh.cli.tailnet_status", lambda cfg: None)
    rc = main(["discover"])
    assert rc == 1
    out = capsys.readouterr().out
    assert "tailscale status" in out
    assert "--peer" in out  # hint at the LAN-mode escape hatch


# ---------------------------------------------------------------------------
# router subcommand
# ---------------------------------------------------------------------------


def test_router_parser_carries_expected_defaults():
    """Lock the public flag surface so we don't silently rename them."""
    args = _parse(["router"])
    assert args.bind == "127.0.0.1"
    assert args.port == 8080  # != registry's 8088 by convention
    assert args.peer == []
    assert args.refresh_s == 5.0  # matches heartbeat cadence
    assert args.tag  # has a default specialist tag
    assert args.log_level == "info"


def test_router_parser_accepts_lan_and_tailnet_modes_via_flags():
    args_lan = _parse(["router", "--peer", "192.168.1.10", "--peer", "192.168.1.20"])
    assert args_lan.peer == ["192.168.1.10", "192.168.1.20"]
    args_tn = _parse(["router", "--port", "9090", "--refresh-s", "10"])
    assert args_tn.port == 9090 and args_tn.refresh_s == 10.0


def test_cmd_router_starts_uvicorn_with_router_app_and_stops_refresher(monkeypatch):
    """Wire test: cmd_router builds a refresher, hands snapshot_source to
    the router app, runs uvicorn, and stops the refresher on exit."""
    monkeypatch.setattr(
        "mesh.cli.tailnet_status",
        lambda cfg: {
            "Self": {
                "DNSName": "self.ts.net.",
                "Online": True,
                "Tags": ["tag:specialist"],
            },
            "Peer": {},
        },
    )

    # Stub fetch so discover_specialists returns a single specialist.
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
                        "node_urls": [f"http://{host}:8003"],
                    },
                }],
            }
        return fetch

    monkeypatch.setattr("mesh.cli.make_http_fetch", fake_make_fetch)

    # Capture what cmd_router passes to uvicorn + assert the refresher stops.
    captured: dict = {}

    def fake_uvicorn_run(app, host, port, log_level):
        # Snapshot source must yield the demo specialist we stubbed in fetch.
        snap = app.state.snapshot_source()
        captured["specialists"] = list(snap.specialists)
        captured["host"] = host
        captured["port"] = port

    monkeypatch.setattr("uvicorn.run", fake_uvicorn_run)

    rc = main(["router", "--port", "9091", "--bind", "127.0.0.1", "--refresh-s", "30"])
    assert rc == 0
    assert captured["host"] == "127.0.0.1" and captured["port"] == 9091
    # The discovered specialist must show up in the snapshot the router app uses.
    assert "demo" in captured["specialists"]


def test_cmd_router_uses_explicit_peers_when_set(monkeypatch):
    """`--peer` flips the refresher into LAN-only mode; tailnet_status must
    not be called."""

    def boom(_cfg):
        raise AssertionError("tailnet_status must NOT be called when --peer is set")

    monkeypatch.setattr("mesh.cli.tailnet_status", boom)

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
                        "node_urls": [f"http://{host}:8003"],
                    },
                }],
            }
        return fetch

    monkeypatch.setattr("mesh.cli.make_http_fetch", fake_make_fetch)
    monkeypatch.setattr("uvicorn.run", lambda app, host, port, log_level: None)

    rc = main([
        "router",
        "--peer", "10.0.0.5",
        "--peer", "10.0.0.6",
        "--refresh-s", "30",
    ])
    assert rc == 0
