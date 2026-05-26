"""In-process node server — the missing wire for pull discovery.

Until now the two node-side processes were disconnected: `ServeDaemon`
(mesh/serve.py) heartbeats into an in-process `MeshRegistry` *or just logs*,
while `create_mesh_app` (mesh/service.py) built its *own* registry that no
heartbeat ever reached. So `/models` listed catalog cards with empty
`node_urls` — nothing was actually routable.

`build_node` wires both halves to **one** `MeshRegistry` in a single
process: the daemon's heartbeat loop populates exactly the registry the
FastAPI app serves. The node then exposes a live, pull-able self-description
on its tailnet interface (`GET /models?include=routing_meta`), which is what
`mesh.discovery` consumes. No heartbeat-push, no central write surface.

Threading note: the daemon heartbeat loop is the **only writer**; uvicorn
request handlers read. This matches the existing single-writer assumption in
`MeshRegistry` (docstring: "rely on uvicorn's single-worker mode"). Run the
node server single-worker.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from fastapi import FastAPI

from mesh.catalog import load_catalog
from mesh.models import SpecialistCard
from mesh.registry import MeshRegistry
from mesh.serve import ServeDaemon, build_daemon
from mesh.service import create_mesh_app
from mesh.tailnet import TailnetConfig

# Convention port for the node-info / discovery surface. Must be reachable by
# the consumer in the tailnet ACL (`tag:gateway -> tag:specialist:8088`).
DEFAULT_NODE_INFO_PORT = 8088


@dataclass
class NodeServer:
    """A daemon + its self-description app, sharing one registry."""

    daemon: ServeDaemon
    app: FastAPI
    registry: MeshRegistry

    def run(
        self,
        *,
        node_info_host: str = "0.0.0.0",
        node_info_port: int = DEFAULT_NODE_INFO_PORT,
        ready_timeout: float = 600.0,
    ) -> None:
        """Start backends + heartbeat loop, then serve the node-info app.

        Blocks until the app is shut down (SIGINT/SIGTERM, handled by
        uvicorn), then stops the daemon. Backends that fail to come up don't
        abort the server — the node runs degraded and heartbeats reflect it.
        """
        import uvicorn

        ok = self.daemon.start(wait_ready=True, ready_timeout=ready_timeout)
        if not ok:
            self.daemon._log("[node] one or more backends failed; serving degraded")
        self.daemon.run_in_thread()
        try:
            uvicorn.run(self.app, host=node_info_host, port=node_info_port, log_level="info")
        finally:
            self.daemon.stop()


def build_node(
    specialist_ids: list[str] | None = None,
    *,
    tailnet: TailnetConfig | None = None,
    catalog: list[SpecialistCard] | None = None,
    base_port: int = 8003,
    log_dir: Path | None = None,
) -> NodeServer:
    """Construct a node server with the daemon + app sharing one registry.

    The catalog seeds the registry (so `/models` lists cards from boot) and
    drives backend selection. `base_port` defaults to the gateway-ACL vLLM
    convention (8003). `tailnet` switches backends to bind 0.0.0.0 and
    advertise the node's MagicDNS host on each specialist's `node_url`.
    """
    cards = catalog if catalog is not None else load_catalog()
    registry = MeshRegistry(catalog=cards)
    daemon = build_daemon(
        specialist_ids=specialist_ids,
        catalog=cards,
        registry=registry,
        base_port=base_port,
        log_dir=log_dir,
        tailnet=tailnet,
    )
    app = create_mesh_app(registry=registry)
    return NodeServer(daemon=daemon, app=app, registry=registry)


__all__ = ["DEFAULT_NODE_INFO_PORT", "NodeServer", "build_node"]
