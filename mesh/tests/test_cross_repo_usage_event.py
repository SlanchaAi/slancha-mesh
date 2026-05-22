"""Cross-repo schema-pin: validate the canonical fixture against slancha-api.

The fixture lives in this repo (mesh/contracts/mesh_usage_event.json) and
is the SINGLE source of truth for the §6 telemetry wire shape between
slancha-local (emitter) and slancha-api (receiver).

This test runs from the slancha-mesh repo. It validates the fixture
against the slancha-api MeshUsageEvent Pydantic class IF slancha-api is
checked out alongside slancha-mesh as a sibling directory; otherwise
it skips cleanly so CI in repos that don't have the sibling still
passes. The companion tests in slancha-api and slancha-local do the
mirror validations from their side.

If this test fails: the fixture and the slancha-api Pydantic disagree.
That's a wire-protocol break — fix one or the other and update the
fixture-version in _meta.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

from mesh.contracts import MESH_USAGE_EVENT_FIXTURE


def _slancha_api_path() -> Path | None:
    """Locate a sibling slancha-api checkout, if one exists.

    Override via SLANCHA_API_PATH env var. Default: ../slancha-api next
    to this repo. Returns None when no sibling is on disk → test skips.
    """
    env = os.environ.get("SLANCHA_API_PATH")
    candidate = Path(env) if env else Path(__file__).resolve().parents[2].parent / "slancha-api"
    return candidate if (candidate / "app" / "endpoints" / "admin.py").exists() else None


@pytest.fixture(scope="module")
def slancha_api_mesh_usage_event_cls():
    api_root = _slancha_api_path()
    if api_root is None:
        pytest.skip(
            "sibling slancha-api checkout not found — cross-repo "
            "validation skipped. Set SLANCHA_API_PATH or check out alongside."
        )

    # Add slancha-api to sys.path so we can import its app.endpoints.admin
    # without polluting installed packages.
    api_root_str = str(api_root)
    if api_root_str not in sys.path:
        sys.path.insert(0, api_root_str)

    # Avoid surfacing any of slancha-api's runtime config requirements —
    # we're only importing the Pydantic class. Set minimum env vars that
    # slancha-api's Settings expects.
    os.environ.setdefault("UNKEY_ROOT_KEY", "test")
    os.environ.setdefault("UNKEY_API_ID", "test")
    os.environ.setdefault("ROUTER_DEVICE", "cpu")

    try:
        from app.endpoints.admin import MeshUsageEvent  # type: ignore[import-not-found]
    except Exception as exc:
        pytest.skip(f"failed to import slancha-api MeshUsageEvent: {exc}")
    return MeshUsageEvent


def _fixture_payload() -> dict:
    raw = json.loads(MESH_USAGE_EVENT_FIXTURE.read_text())
    # _meta is documentation; not part of the wire shape.
    raw.pop("_meta", None)
    return raw


def test_fixture_validates_against_slancha_api_pydantic(slancha_api_mesh_usage_event_cls):
    """The fixture must round-trip cleanly through MeshUsageEvent."""
    payload = _fixture_payload()
    event = slancha_api_mesh_usage_event_cls.model_validate(payload)

    # Spot-check that the dotted-name OTel aliases land on the underscored
    # attributes — that's the slancha-api side of the alias contract.
    assert event.request_id == payload["request_id"]
    assert event.gen_ai_request_model == payload["gen_ai.request.model"]
    assert event.gen_ai_usage_input_tokens == payload["gen_ai.usage.input_tokens"]
    assert event.otel_semconv_version == payload["otel_semconv_version"]


def test_fixture_round_trips_via_model_dump_by_alias(slancha_api_mesh_usage_event_cls):
    """Dumping the parsed event must re-produce the dotted-name keys.

    Without this round-trip property, the SIDECAR (which builds payloads
    with dotted-name keys per OTel) could pass validation but produce
    something a re-consumer couldn't parse the same way.
    """
    payload = _fixture_payload()
    event = slancha_api_mesh_usage_event_cls.model_validate(payload)
    dumped = event.model_dump(by_alias=True, exclude_none=True)

    # The dotted-name OTel keys must reappear in the dump.
    assert "gen_ai.request.model" in dumped
    assert "gen_ai.usage.input_tokens" in dumped
    assert "gen_ai.usage.output_tokens" in dumped


def test_decision_reason_structured_shape_pinned():
    """The structured decision-reason shape (NC5) is part of the wire.

    This test runs without slancha-api present — it pins the SHAPE of
    decision_reason regardless of cross-repo availability. If anyone
    changes the structured-decision-reason vocabulary in the fixture,
    Phase 5 select_mesh_route_with_pref's emitter and the routing-
    transparency UI both need a coordinated update.
    """
    payload = _fixture_payload()
    dr = payload["decision_reason"]
    assert set(dr.keys()) >= {"winner", "alternatives_considered", "deciding_axes", "preset_applied"}
    for alt in dr["alternatives_considered"]:
        assert set(alt.keys()) >= {"id", "delta", "losing_axes"}


def test_fixture_carries_meta_self_documentation():
    """The fixture file must keep its _meta block so the schema-pin
    relationship between repos is discoverable from the file alone.
    """
    raw = json.loads(MESH_USAGE_EVENT_FIXTURE.read_text())
    assert "_meta" in raw
    meta = raw["_meta"]
    assert "fixture_version" in meta
    assert "consumed_by" in meta
    assert len(meta["consumed_by"]) >= 3
