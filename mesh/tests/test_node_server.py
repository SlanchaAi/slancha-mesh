"""In-process node server — daemon + app share ONE registry.

The property under test is the wire that pull-discovery depends on: a
heartbeat pushed by the daemon becomes visible in the *same* registry the
FastAPI app serves, with the per-specialist `node_url` advertised under the
node's tailnet (MagicDNS) host — not loopback.
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from mesh.backends import NullBackend
from mesh.models import NodeProbe, SpecialistCard
from mesh.node_server import build_node
from mesh.serve import ServeDaemon
from mesh.service import create_mesh_app


def _card(specialist_id: str = "test-spec", backend: str = "ollama") -> SpecialistCard:
    return SpecialistCard(
        model_id=f"vendor/{specialist_id}",
        specialist_id=specialist_id,
        domain="code",
        difficulty_tiers=["easy", "medium"],
        required_backend=backend,
        storage_gb=8.0,
        runtime_gb=10.0,
        min_vram_gb=8.0,
        context_window=8192,
        n_layers=32,
        capabilities=["streaming", "system_prompt"],
    )


def _probe() -> NodeProbe:
    return NodeProbe(
        node_id="node-test",
        friendly_name="test-box",
        chip="gb10",
        arch="aarch64",
        cuda_capability="12.0",
        ram_total_gb=128.0,
        ram_available_gb=100.0,
        available_backends=["vllm"],
    )


def test_build_node_shares_one_registry():
    """daemon, app, and the returned handle must all point at the same store."""
    node = build_node(specialist_ids=[], catalog=[_card()], base_port=8003)
    assert node.daemon.registry is node.registry
    assert node.app.state.registry is node.registry


def test_heartbeat_is_visible_in_models_with_advertised_host():
    """A daemon heartbeat shows up in the shared app's /models, host-pinned
    to the advertised MagicDNS name."""
    from mesh.registry import MeshRegistry

    card = _card("demo-model")
    registry = MeshRegistry(catalog=[card])
    backend = NullBackend(card=card, base_url="http://0.0.0.0:8004")
    daemon = ServeDaemon(
        backends=[backend],
        probe=_probe(),
        registry=registry,
        advertise_host="test-box.taila.ts.net",
    )
    daemon.start(wait_ready=True)
    daemon.post_heartbeat()  # one heartbeat → into the shared registry

    app = create_mesh_app(registry=registry)
    client = TestClient(app)
    resp = client.get("/models", params={"include": "routing_meta"})
    assert resp.status_code == 200
    data = {m["id"]: m for m in resp.json()["data"]}
    assert "demo-model" in data
    node_urls = data["demo-model"]["routing_meta"]["node_urls"]
    assert node_urls == ["http://test-box.taila.ts.net:8004"]
    assert all("0.0.0.0" not in u and "127.0.0.1" not in u for u in node_urls)
