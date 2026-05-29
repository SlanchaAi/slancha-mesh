"""OpenAI-compatible router app — drop-in `/v1` endpoint over a mesh of nodes.

The point of this module: a LocalLLaMA user runs `slancha-mesh` on a few
boxes, then points Open WebUI / LiteLLM / their own `curl` at ONE URL.
The router picks the right node behind the scenes — same OpenAI shape
clients already speak, no per-node fan-out, no manual `discover` →
`curl` two-step.

What it does:

  - `GET  /v1/models`            → list every specialist the snapshot
                                   currently knows about, in OpenAI's
                                   list shape.
  - `POST /v1/chat/completions`  → look up `body.model` (a specialist_id),
                                   pick the first reachable binding, proxy
                                   the request to that node's `/v1/chat/
                                   completions`, return the response.

What it does NOT do (yet — incremental scope):

  - Streaming SSE. Phase-1 ships non-streaming only; the response body
    is awaited end-to-end before returning. Followup PR adds
    `text/event-stream` passthrough.
  - Fallback-on-upstream-5xx. Phase-1 returns the upstream error
    verbatim. Followup PR walks the snapshot's secondary bindings.
  - Classifier-driven domain inference. Phase-1 requires the client to
    send `model = <specialist_id>` directly. The classifier-on-prompt
    path (route by domain/difficulty instead of by id) is a followup
    that depends on slancha-api's classifier being importable.

Mounting:

    from mesh.router_app import create_router_app
    app = create_router_app(registry=shared_registry)
    # serve standalone via uvicorn, or mount into slancha-api

Auth: same `SLANCHA_NODE_TOKEN` bearer the registry app uses — set the
env var to enforce, unset for dev. When set, the router also forwards
the bearer to the upstream node's `/v1/chat/completions` (the node may
be behind its own token gate).

Observability: every routed response carries three response headers so a
client can audit what the router picked without parsing logs:

  - `X-Slancha-Specialist`: the catalog id served
  - `X-Slancha-Node`:       the node id picked
  - `X-Slancha-Reason`:     short human string ("primary, queue=120ms")
"""

from __future__ import annotations

import hmac
import logging
import os
import threading
from datetime import datetime, timezone
from typing import Annotated, Callable
from urllib.parse import urlsplit

import httpx
from fastapi import Depends, FastAPI, Header, HTTPException, Request, Response, status
from fastapi.responses import JSONResponse

from mesh.discovery import DiscoveryResult
from mesh.models import NodeBinding, NodeSummary, RegistrySnapshot, SpecialistCard
from mesh.registry import MeshRegistry

NODE_TOKEN_ENV = "SLANCHA_NODE_TOKEN"
UPSTREAM_TIMEOUT_S = 120.0  # generous; covers a cold Ollama load on the upstream

_log = logging.getLogger(__name__)

SnapshotSource = Callable[[], RegistrySnapshot]


# ---------------------------------------------------------------------------
# Auth (mirror of mesh.registry_app.verify_node_token so the router can be
# mounted standalone without dragging the registry_app dependency)
# ---------------------------------------------------------------------------


def _expected_token() -> str | None:
    tok = os.environ.get(NODE_TOKEN_ENV, "").strip()
    return tok or None


def verify_router_token(
    authorization: Annotated[str | None, Header()] = None,
) -> None:
    """Validate Bearer token against `SLANCHA_NODE_TOKEN`.

    Returns silently if auth is disabled OR the token matches.
    Raises 401 on missing / malformed header, 403 on wrong token.
    """
    expected = _expected_token()
    if expected is None:
        return
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="missing or malformed Authorization header",
            headers={"WWW-Authenticate": 'Bearer realm="slancha-mesh-router"'},
        )
    received = authorization[len("Bearer ") :].strip()
    if not hmac.compare_digest(received, expected):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="invalid bearer token",
        )


# ---------------------------------------------------------------------------
# Routing helpers — pure, testable in isolation from FastAPI
# ---------------------------------------------------------------------------


def _pick_binding(
    specialist_id: str,
    snapshot: RegistrySnapshot,
) -> NodeBinding | None:
    """Return the first reachable `NodeBinding` for the specialist, or None.

    "First reachable" today = first binding whose health is not
    `unreachable` AND whose `node_url` is set. The snapshot's binding
    order reflects insertion order (heartbeat arrival), which is
    deterministic per replay. A future PR can wire a quality-aware
    pick using `MeshSelectionResult` semantics directly.
    """
    bindings = snapshot.specialists.get(specialist_id) or []
    for b in bindings:
        if b.health == "unreachable":
            continue
        if not b.node_url:
            continue
        return b
    return None


def _rewrite_model_for_upstream(
    body: dict,
    specialist_id: str,
    snapshot: RegistrySnapshot,
) -> dict:
    """If the chosen specialist's backend is Ollama, swap `model` → `ollama_tag`.

    The mesh-facing model id is the catalog `specialist_id` (`qwen2.5-coder
    -7b-q4-ollama`); Ollama's `/v1/chat/completions` needs the engine tag
    (`qwen2.5-coder:7b-instruct-q4_K_M`). For vLLM specialists the
    upstream's `--served-model-name` is already set to `specialist_id`
    (see `mesh.backends.VLLMBackend.start`), so no rewrite needed.

    Returns a shallow copy of `body` so the caller's dict isn't mutated.
    """
    card = snapshot.catalog.get(specialist_id)
    if card is None:
        return dict(body)
    if card.required_backend == "ollama" and card.ollama_tag:
        return {**body, "model": card.ollama_tag}
    return dict(body)


def _upstream_headers(authorization: str | None) -> dict[str, str]:
    """Headers to forward to the upstream node.

    Forwards the caller's bearer (so a node behind its own token gate
    accepts the proxied request) and sets a clean Content-Type.
    """
    h = {"Content-Type": "application/json"}
    if authorization:
        h["Authorization"] = authorization
    return h


# ---------------------------------------------------------------------------
# Discovery → snapshot translation — lets the router run without a registry
# ---------------------------------------------------------------------------


def _node_id_for_url(url: str) -> str:
    """Stable node id derived from a `node_url` (host[:port]).

    The pull discovery has no real node ids — peers are just hosts that
    answered `/models`. We synthesize a deterministic id so `NodeBinding`s
    have something to put in `node_id` (and the routing-audit header
    surfaces something human-meaningful).
    """
    parts = urlsplit(url)
    host = parts.hostname or "unknown"
    if parts.port:
        return f"{host}:{parts.port}"
    return host


def discovery_to_snapshot(
    discovery: DiscoveryResult,
    *,
    catalog: list[SpecialistCard] | None = None,
) -> RegistrySnapshot:
    """Translate a pull-discovery result into a `RegistrySnapshot`.

    The router needs a snapshot for `_pick_binding` + `_rewrite_model_for_
    upstream`. In push mode the snapshot comes from `MeshRegistry.snapshot()`;
    in pull mode we synthesize one from `discover_specialists`. Bindings
    are minimal — we know the URL is currently reachable (we just GET'd
    `/models`) but not the queue depth or p95 latency, which only the
    heartbeat-push topology carries.

    `catalog` is the LOCAL catalog (`load_catalog()`); it provides
    `ollama_tag` / `required_backend` per specialist so the router's
    upstream-model-rewrite path works. Specialists in the discovery
    result that aren't in the local catalog still route (no rewrite —
    `model = specialist_id` flows through unchanged).
    """
    now = datetime.now(timezone.utc)
    cards_by_id = {c.specialist_id: c for c in (catalog or [])}

    bindings_by_specialist: dict[str, list[NodeBinding]] = {}
    nodes: dict[str, NodeSummary] = {}

    for sid, spec in discovery.specialists.items():
        for url in spec.node_urls:
            node_id = _node_id_for_url(url)
            binding = NodeBinding(
                node_id=node_id,
                specialist_id=sid,
                health="healthy",
                queue_depth=0,
                p95_latency_ms_60s=None,
                node_url=url,
                last_seen=now,
            )
            bindings_by_specialist.setdefault(sid, []).append(binding)
            # Multiple specialists may live behind one node URL; keep the
            # first NodeSummary we synthesize (they all look the same here).
            nodes.setdefault(
                node_id,
                NodeSummary(
                    node_id=node_id,
                    friendly_name=node_id,
                    health="healthy",
                    last_seen=now,
                    loaded_specialist_ids=[sid],
                    queue_depth=0,
                    p95_latency_ms_60s=None,
                    node_url=url,
                ),
            )

    return RegistrySnapshot(
        snapshot_ts=now,
        nodes=nodes,
        specialists=bindings_by_specialist,
        coverage={},
        ranked_routes={},
        catalog=cards_by_id,
    )


class _RefreshingSnapshot:
    """Thread-safe snapshot holder + background refresher for pull mode.

    The router's `snapshot_source` is called per request. Calling
    `discover_specialists` per request would hammer every peer's
    `/models` endpoint with the same frequency as inbound traffic
    (potentially many qps); instead, we keep a cached snapshot and
    refresh it on a fixed cadence (default 5 s, matching the
    heartbeat interval).

    Start the refresher with `.start()`; stop with `.stop()`. The
    background thread is daemonized so a process exit doesn't deadlock
    on it.
    """

    def __init__(
        self,
        refresher: Callable[[], DiscoveryResult],
        *,
        catalog: list[SpecialistCard] | None = None,
        refresh_s: float = 5.0,
    ) -> None:
        self._refresher = refresher
        self._catalog = list(catalog or [])
        self._refresh_s = refresh_s
        self._lock = threading.Lock()
        self._snapshot: RegistrySnapshot | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def get(self) -> RegistrySnapshot:
        """Return the latest cached snapshot; refresh inline if none yet."""
        with self._lock:
            if self._snapshot is not None:
                return self._snapshot
        # Cold start — do a synchronous refresh under no lock so a
        # concurrent .get() can still serve a slightly older snapshot
        # later if the background loop has started by then.
        snap = discovery_to_snapshot(self._refresher(), catalog=self._catalog)
        with self._lock:
            self._snapshot = snap
        return snap

    def refresh_once(self) -> None:
        """Run one discovery pass + swap the snapshot under the lock."""
        snap = discovery_to_snapshot(self._refresher(), catalog=self._catalog)
        with self._lock:
            self._snapshot = snap

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                self.refresh_once()
            except Exception as exc:  # noqa: BLE001 — refresh must not crash the router
                _log.warning("[router] discovery refresh failed: %s", exc)
            # Wait either for the refresh interval OR an early stop.
            self._stop.wait(timeout=self._refresh_s)

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------


def create_router_app(
    registry: MeshRegistry | None = None,
    *,
    snapshot_source: SnapshotSource | None = None,
    http_client: httpx.Client | None = None,
) -> FastAPI:
    """Build the OpenAI-compatible router app.

    Snapshot source resolution:
      - `snapshot_source` (callable returning `RegistrySnapshot`) wins
        when provided — the seam used by tests + the discovery-driven
        deployment shape.
      - Else, `registry.snapshot()` is called per request.
      - Both None → `ValueError`. The router needs *some* way to know
        what's reachable.

    `http_client` is injected for tests so we don't open real sockets;
    production passes None and the factory creates a long-lived
    `httpx.Client` with the upstream timeout pre-set.
    """
    if snapshot_source is None and registry is None:
        raise ValueError(
            "create_router_app needs either `registry=` (push mode) or "
            "`snapshot_source=` (discovery-driven pull mode); both were None."
        )

    def _snapshot() -> RegistrySnapshot:
        if snapshot_source is not None:
            return snapshot_source()
        # registry is guaranteed non-None here by the ValueError above.
        return registry.snapshot()  # type: ignore[union-attr]

    client = http_client or httpx.Client(timeout=UPSTREAM_TIMEOUT_S)

    app = FastAPI(
        title="Slancha-Mesh OpenAI-compatible router",
        version="0.0.8",
        description=(
            "Drop-in OpenAI `/v1` endpoint over a mesh of self-hosted nodes. "
            "Pick `model = <specialist_id>`; the router proxies to the right "
            "node and returns the upstream response."
        ),
    )
    app.state.registry = registry
    app.state.snapshot_source = _snapshot
    app.state.http_client = client

    @app.get("/v1/models", summary="OpenAI-compatible list of mesh specialists")
    def list_models(
        _: Annotated[None, Depends(verify_router_token)],
    ) -> dict:
        snap = _snapshot()
        return {
            "object": "list",
            "data": [
                {
                    "id": sid,
                    "object": "model",
                    "owned_by": "slancha-mesh",
                }
                for sid in sorted(snap.specialists)
            ],
        }

    @app.post(
        "/v1/chat/completions",
        summary="OpenAI-compatible chat completions, proxied to the picked node",
    )
    async def chat_completions(
        request: Request,
        _: Annotated[None, Depends(verify_router_token)],
        authorization: Annotated[str | None, Header()] = None,
    ) -> Response:
        try:
            body = await request.json()
        except Exception as exc:  # noqa: BLE001 — anything not-JSON is a 400
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"request body is not valid JSON: {exc}",
            ) from exc
        if not isinstance(body, dict):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="request body must be a JSON object",
            )
        specialist_id = body.get("model")
        if not isinstance(specialist_id, str) or not specialist_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="`model` must be a non-empty specialist_id",
            )
        # Phase-1 deliberately rejects streaming so a client doesn't get a
        # silently-buffered response that violates the OpenAI contract.
        # Followup PR proxies the SSE stream.
        if body.get("stream") is True:
            raise HTTPException(
                status_code=status.HTTP_501_NOT_IMPLEMENTED,
                detail="streaming responses are not yet supported by the mesh router",
            )

        snap = _snapshot()
        binding = _pick_binding(specialist_id, snap)
        if binding is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=(
                    f"no reachable node for specialist {specialist_id!r}; "
                    f"check `GET /v1/models` for what's currently routable."
                ),
            )

        upstream_url = f"{binding.node_url.rstrip('/')}/v1/chat/completions"
        upstream_body = _rewrite_model_for_upstream(body, specialist_id, snap)

        try:
            upstream = client.post(
                upstream_url,
                json=upstream_body,
                headers=_upstream_headers(authorization),
            )
        except (httpx.HTTPError, OSError) as exc:
            _log.warning(
                "[router] upstream %s for %s unreachable: %s",
                upstream_url,
                specialist_id,
                exc,
            )
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=(
                    f"upstream node {binding.node_id} unreachable at {upstream_url}: "
                    f"{exc.__class__.__name__}"
                ),
            ) from exc

        # Forward the upstream body byte-for-byte; we touch only the headers
        # the OpenAI spec lets us own + add our routing-audit headers.
        media_type = upstream.headers.get("content-type", "application/json")
        slancha_headers = {
            "X-Slancha-Specialist": specialist_id,
            "X-Slancha-Node": binding.node_id,
            "X-Slancha-Reason": (
                f"primary; queue_depth={binding.queue_depth} "
                f"p95={binding.p95_latency_ms_60s}"
            ),
        }
        return Response(
            content=upstream.content,
            status_code=upstream.status_code,
            media_type=media_type,
            headers=slancha_headers,
        )

    @app.get("/health")
    def health() -> JSONResponse:
        """Liveness — the one open endpoint, mirrors registry_app's posture."""
        snap = _snapshot()
        return JSONResponse(
            {
                "status": "ok",
                "auth_required": _expected_token() is not None,
                "specialists_reachable": sum(
                    1
                    for bindings in snap.specialists.values()
                    if any(b.health != "unreachable" and b.node_url for b in bindings)
                ),
            }
        )

    return app


__all__ = [
    "NODE_TOKEN_ENV",
    "UPSTREAM_TIMEOUT_S",
    "create_router_app",
    "discovery_to_snapshot",
    "verify_router_token",
]
