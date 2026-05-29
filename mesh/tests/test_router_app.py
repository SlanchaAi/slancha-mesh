"""Router-app tests — the OpenAI-compatible `/v1` proxy.

The router takes a snapshot source + an httpx client (injected here so we
never open a real socket), looks up `body.model` as a specialist_id,
picks the first reachable binding, and proxies. These tests pin:

  - the auth surface mirrors registry_app (bearer or open),
  - phase-1 rejects `stream: true` cleanly (501, not silent buffer),
  - Ollama specialists get `model` rewritten to `ollama_tag`,
  - vLLM specialists' `model` passes through untouched,
  - response carries the routing-audit headers,
  - upstream errors degrade to 502, not raise,
  - `/health` is unauthenticated and reports `auth_required`.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient

from mesh.models import (
    HealthState,
    NodeBinding,
    NodeSummary,
    RegistrySnapshot,
    SpecialistCard,
)
from mesh.router_app import NODE_TOKEN_ENV, create_router_app


# ---------------------------------------------------------------------------
# Fixtures — synthetic snapshot + card catalog + a mock httpx upstream
# ---------------------------------------------------------------------------


def _card(
    *,
    specialist_id: str,
    required_backend: str = "ollama",
    ollama_tag: str | None = "qwen2.5-coder:7b-instruct-q4_K_M",
) -> SpecialistCard:
    return SpecialistCard(
        model_id="Qwen/Qwen2.5-Coder-7B-Instruct",
        specialist_id=specialist_id,
        domain="code",
        difficulty_tiers=["medium"],
        required_backend=required_backend,  # type: ignore[arg-type]
        ollama_tag=ollama_tag if required_backend == "ollama" else None,
        storage_gb=5.0,
        runtime_gb=6.0,
        min_vram_gb=8.0,
        context_window=32768,
        n_layers=28,
        estimated_tps_at={"gb10": 40.0},
    )


def _binding(
    *,
    node_id: str = "node-a",
    specialist_id: str,
    health: HealthState = "healthy",
    node_url: str | None = "http://10.0.0.5:11434",
    queue_depth: int = 0,
) -> NodeBinding:
    return NodeBinding(
        node_id=node_id,
        specialist_id=specialist_id,
        health=health,
        queue_depth=queue_depth,
        p95_latency_ms_60s=420.0,
        node_url=node_url,
        last_seen=datetime.now(timezone.utc),
    )


def _snapshot(
    *,
    cards: list[SpecialistCard],
    bindings: dict[str, list[NodeBinding]],
) -> RegistrySnapshot:
    now = datetime.now(timezone.utc)
    return RegistrySnapshot(
        snapshot_ts=now,
        nodes={
            b.node_id: NodeSummary(
                node_id=b.node_id,
                friendly_name=b.node_id,
                health=b.health,
                last_seen=now,
                loaded_specialist_ids=[b.specialist_id],
                queue_depth=b.queue_depth,
                p95_latency_ms_60s=b.p95_latency_ms_60s,
                node_url=b.node_url,
            )
            for bs in bindings.values()
            for b in bs
        },
        specialists=bindings,
        coverage={},
        ranked_routes={},
        catalog={c.specialist_id: c for c in cards},
    )


def _client(snapshot: RegistrySnapshot, handler) -> TestClient:
    """Build a TestClient over a router app whose http_client is a MockTransport.

    `handler` is called per upstream request and returns a tuple
    `(status_code, json_dict | bytes, headers | None)`.
    """

    def transport_handler(request: httpx.Request) -> httpx.Response:
        result = handler(request)
        if isinstance(result, httpx.Response):
            return result
        status_code, payload, headers = result
        if isinstance(payload, dict):
            return httpx.Response(status_code, json=payload, headers=headers or {})
        return httpx.Response(status_code, content=payload, headers=headers or {})

    upstream = httpx.Client(transport=httpx.MockTransport(transport_handler))
    app = create_router_app(snapshot_source=lambda: snapshot, http_client=upstream)
    return TestClient(app)


# ---------------------------------------------------------------------------
# Factory contracts
# ---------------------------------------------------------------------------


def test_create_router_app_refuses_with_neither_source():
    with pytest.raises(ValueError, match="registry"):
        create_router_app()


# ---------------------------------------------------------------------------
# /v1/models
# ---------------------------------------------------------------------------


def test_list_models_returns_sorted_specialist_ids():
    snap = _snapshot(
        cards=[
            _card(specialist_id="qwen2.5-coder-7b-q4-ollama"),
            _card(specialist_id="phi-3.5-mini-q5-ollama"),
        ],
        bindings={
            "qwen2.5-coder-7b-q4-ollama": [
                _binding(specialist_id="qwen2.5-coder-7b-q4-ollama")
            ],
            "phi-3.5-mini-q5-ollama": [
                _binding(node_id="node-b", specialist_id="phi-3.5-mini-q5-ollama")
            ],
        },
    )
    client = _client(snap, lambda req: (200, {}, None))
    r = client.get("/v1/models")
    assert r.status_code == 200
    payload = r.json()
    assert payload["object"] == "list"
    assert [m["id"] for m in payload["data"]] == [
        "phi-3.5-mini-q5-ollama",
        "qwen2.5-coder-7b-q4-ollama",
    ]
    for entry in payload["data"]:
        assert entry["object"] == "model"
        assert entry["owned_by"] == "slancha-mesh"


# ---------------------------------------------------------------------------
# /v1/chat/completions — happy path + rewrites + headers
# ---------------------------------------------------------------------------


def test_chat_completions_ollama_rewrites_model_to_ollama_tag():
    """Phase-1 contract: Ollama specialists get `model` swapped to `ollama_tag`."""
    snap = _snapshot(
        cards=[
            _card(
                specialist_id="qwen2.5-coder-7b-q4-ollama",
                required_backend="ollama",
                ollama_tag="qwen2.5-coder:7b-instruct-q4_K_M",
            )
        ],
        bindings={
            "qwen2.5-coder-7b-q4-ollama": [
                _binding(specialist_id="qwen2.5-coder-7b-q4-ollama")
            ]
        },
    )

    captured: dict[str, Any] = {}

    def handler(request: httpx.Request):
        captured["url"] = str(request.url)
        captured["body"] = request.read()
        return (
            200,
            {"id": "chatcmpl-1", "choices": [{"message": {"content": "ok"}}]},
            None,
        )

    client = _client(snap, handler)
    r = client.post(
        "/v1/chat/completions",
        json={
            "model": "qwen2.5-coder-7b-q4-ollama",
            "messages": [{"role": "user", "content": "hi"}],
        },
    )
    assert r.status_code == 200, r.text
    assert r.json()["choices"][0]["message"]["content"] == "ok"
    # Upstream URL must hit /v1/chat/completions on the bound node_url.
    assert captured["url"] == "http://10.0.0.5:11434/v1/chat/completions"
    # The mesh-facing specialist_id MUST be rewritten to the Ollama engine tag.
    import json as _json
    body = _json.loads(captured["body"])
    assert body["model"] == "qwen2.5-coder:7b-instruct-q4_K_M"


def test_chat_completions_vllm_passes_model_through():
    """vLLM specialists already serve `model=specialist_id` via
    `--served-model-name`; the router must NOT rewrite, or vLLM 404s."""
    snap = _snapshot(
        cards=[
            _card(
                specialist_id="qwen3-coder-30b-a3b-fp8",
                required_backend="vllm",
                ollama_tag=None,
            )
        ],
        bindings={
            "qwen3-coder-30b-a3b-fp8": [
                _binding(
                    node_id="spark-1",
                    specialist_id="qwen3-coder-30b-a3b-fp8",
                    node_url="http://spark-1.taila.ts.net:8013",
                )
            ]
        },
    )

    captured: dict[str, Any] = {}

    def handler(request: httpx.Request):
        captured["body"] = request.read()
        return (200, {"id": "ok"}, None)

    client = _client(snap, handler)
    r = client.post(
        "/v1/chat/completions",
        json={
            "model": "qwen3-coder-30b-a3b-fp8",
            "messages": [{"role": "user", "content": "hi"}],
        },
    )
    assert r.status_code == 200
    import json as _json
    assert _json.loads(captured["body"])["model"] == "qwen3-coder-30b-a3b-fp8"


def test_chat_completions_sets_routing_audit_headers():
    """Client can audit the routing decision without parsing the body."""
    snap = _snapshot(
        cards=[_card(specialist_id="qwen2.5-coder-7b-q4-ollama")],
        bindings={
            "qwen2.5-coder-7b-q4-ollama": [
                _binding(
                    node_id="rtx-3090-1",
                    specialist_id="qwen2.5-coder-7b-q4-ollama",
                    queue_depth=3,
                )
            ]
        },
    )
    client = _client(snap, lambda req: (200, {"id": "ok"}, None))
    r = client.post(
        "/v1/chat/completions",
        json={"model": "qwen2.5-coder-7b-q4-ollama", "messages": []},
    )
    assert r.status_code == 200
    assert r.headers["X-Slancha-Specialist"] == "qwen2.5-coder-7b-q4-ollama"
    assert r.headers["X-Slancha-Node"] == "rtx-3090-1"
    assert "queue_depth=3" in r.headers["X-Slancha-Reason"]


def test_chat_completions_forwards_authorization_to_upstream():
    """SLANCHA_NODE_TOKEN on the node side must still accept proxied calls."""
    snap = _snapshot(
        cards=[_card(specialist_id="qwen2.5-coder-7b-q4-ollama")],
        bindings={
            "qwen2.5-coder-7b-q4-ollama": [
                _binding(specialist_id="qwen2.5-coder-7b-q4-ollama")
            ]
        },
    )

    captured: dict[str, Any] = {}

    def handler(request: httpx.Request):
        captured["authorization"] = request.headers.get("authorization")
        return (200, {"id": "ok"}, None)

    client = _client(snap, handler)
    r = client.post(
        "/v1/chat/completions",
        json={"model": "qwen2.5-coder-7b-q4-ollama", "messages": []},
        headers={"Authorization": "Bearer node-secret"},
    )
    assert r.status_code == 200
    assert captured["authorization"] == "Bearer node-secret"


# ---------------------------------------------------------------------------
# Error paths — never raise, always return a sane status
# ---------------------------------------------------------------------------


def test_chat_completions_404_when_no_specialist_known():
    snap = _snapshot(cards=[], bindings={})
    client = _client(snap, lambda req: (200, {}, None))
    r = client.post(
        "/v1/chat/completions",
        json={"model": "nope-7b", "messages": []},
    )
    assert r.status_code == 404
    assert "no reachable node" in r.json()["detail"]


def test_chat_completions_404_when_only_unreachable_bindings():
    """All bindings marked `unreachable` = same as no specialist."""
    snap = _snapshot(
        cards=[_card(specialist_id="qwen2.5-coder-7b-q4-ollama")],
        bindings={
            "qwen2.5-coder-7b-q4-ollama": [
                _binding(
                    specialist_id="qwen2.5-coder-7b-q4-ollama",
                    health="unreachable",
                )
            ]
        },
    )
    client = _client(snap, lambda req: (200, {}, None))
    r = client.post(
        "/v1/chat/completions",
        json={"model": "qwen2.5-coder-7b-q4-ollama", "messages": []},
    )
    assert r.status_code == 404


def test_chat_completions_skips_unreachable_picks_healthy_next():
    """Mixed-health bindings: the first reachable one wins."""
    snap = _snapshot(
        cards=[_card(specialist_id="qwen2.5-coder-7b-q4-ollama")],
        bindings={
            "qwen2.5-coder-7b-q4-ollama": [
                _binding(
                    node_id="dead-node",
                    specialist_id="qwen2.5-coder-7b-q4-ollama",
                    health="unreachable",
                ),
                _binding(
                    node_id="live-node",
                    specialist_id="qwen2.5-coder-7b-q4-ollama",
                    node_url="http://10.0.0.6:11434",
                ),
            ]
        },
    )
    client = _client(snap, lambda req: (200, {"id": "ok"}, None))
    r = client.post(
        "/v1/chat/completions",
        json={"model": "qwen2.5-coder-7b-q4-ollama", "messages": []},
    )
    assert r.status_code == 200
    assert r.headers["X-Slancha-Node"] == "live-node"


def test_chat_completions_502_on_upstream_connect_failure():
    """Upstream death must turn into 502, not a 5xx FastAPI traceback."""
    snap = _snapshot(
        cards=[_card(specialist_id="qwen2.5-coder-7b-q4-ollama")],
        bindings={
            "qwen2.5-coder-7b-q4-ollama": [
                _binding(specialist_id="qwen2.5-coder-7b-q4-ollama")
            ]
        },
    )

    def handler(request: httpx.Request):
        raise httpx.ConnectError("connection refused", request=request)

    client = _client(snap, handler)
    r = client.post(
        "/v1/chat/completions",
        json={"model": "qwen2.5-coder-7b-q4-ollama", "messages": []},
    )
    assert r.status_code == 502
    assert "unreachable" in r.json()["detail"]


def test_chat_completions_forwards_upstream_non_200_verbatim():
    """If the node returns 400/500, the client sees that — not 502."""
    snap = _snapshot(
        cards=[_card(specialist_id="qwen2.5-coder-7b-q4-ollama")],
        bindings={
            "qwen2.5-coder-7b-q4-ollama": [
                _binding(specialist_id="qwen2.5-coder-7b-q4-ollama")
            ]
        },
    )

    def handler(request: httpx.Request):
        return (400, {"error": {"message": "bad request"}}, None)

    client = _client(snap, handler)
    r = client.post(
        "/v1/chat/completions",
        json={"model": "qwen2.5-coder-7b-q4-ollama", "messages": []},
    )
    assert r.status_code == 400
    assert r.json()["error"]["message"] == "bad request"


def test_chat_completions_400_when_model_field_missing():
    snap = _snapshot(cards=[], bindings={})
    client = _client(snap, lambda req: (200, {}, None))
    r = client.post("/v1/chat/completions", json={"messages": []})
    assert r.status_code == 400


def test_chat_completions_400_when_body_not_json():
    snap = _snapshot(cards=[], bindings={})
    client = _client(snap, lambda req: (200, {}, None))
    r = client.post(
        "/v1/chat/completions",
        content=b"not json",
        headers={"Content-Type": "application/json"},
    )
    assert r.status_code == 400


def test_chat_completions_501_on_stream_true_until_streaming_lands():
    """Phase-1 explicitly refuses `stream: true` — silent buffering would
    violate the OpenAI contract and break clients that expect SSE."""
    snap = _snapshot(
        cards=[_card(specialist_id="qwen2.5-coder-7b-q4-ollama")],
        bindings={
            "qwen2.5-coder-7b-q4-ollama": [
                _binding(specialist_id="qwen2.5-coder-7b-q4-ollama")
            ]
        },
    )
    client = _client(snap, lambda req: (200, {}, None))
    r = client.post(
        "/v1/chat/completions",
        json={
            "model": "qwen2.5-coder-7b-q4-ollama",
            "messages": [],
            "stream": True,
        },
    )
    assert r.status_code == 501


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------


def test_auth_disabled_by_default(monkeypatch):
    monkeypatch.delenv(NODE_TOKEN_ENV, raising=False)
    snap = _snapshot(cards=[], bindings={})
    client = _client(snap, lambda req: (200, {}, None))
    r = client.get("/v1/models")
    assert r.status_code == 200


def test_auth_required_when_env_set(monkeypatch):
    monkeypatch.setenv(NODE_TOKEN_ENV, "mesh-secret")
    snap = _snapshot(cards=[], bindings={})
    client = _client(snap, lambda req: (200, {}, None))
    # No header → 401.
    r = client.get("/v1/models")
    assert r.status_code == 401
    # Wrong header → 403.
    r = client.get("/v1/models", headers={"Authorization": "Bearer wrong"})
    assert r.status_code == 403
    # Right header → 200.
    r = client.get("/v1/models", headers={"Authorization": "Bearer mesh-secret"})
    assert r.status_code == 200


def test_health_is_unauthenticated_and_reports_auth_required(monkeypatch):
    """`/health` must never require the bearer (mirrors registry_app)."""
    monkeypatch.setenv(NODE_TOKEN_ENV, "mesh-secret")
    snap = _snapshot(
        cards=[_card(specialist_id="qwen2.5-coder-7b-q4-ollama")],
        bindings={
            "qwen2.5-coder-7b-q4-ollama": [
                _binding(specialist_id="qwen2.5-coder-7b-q4-ollama")
            ]
        },
    )
    client = _client(snap, lambda req: (200, {}, None))
    r = client.get("/health")
    assert r.status_code == 200
    payload = r.json()
    assert payload["status"] == "ok"
    assert payload["auth_required"] is True
    assert payload["specialists_reachable"] == 1
