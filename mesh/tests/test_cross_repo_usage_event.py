"""Cross-repo schema-pin: validate the canonical fixture against the published
MeshUsageEvent JSON Schema in slancha-shared (PUBLIC).

The fixture (mesh/contracts/mesh_usage_event.json) is the golden trace for the
§6 telemetry wire shape between slancha-local (emitter) and slancha-api
(receiver). The canonical schema is published at
`slancha-shared/schemas/mesh-usage-event.schema.json`, generated from
slancha-api's `MeshUsageEvent` Pydantic (by_alias).

This validates the fixture against that **public** schema — no private
slancha-api checkout (that path 404'd in CI: private repo + default token).
The companion check in slancha-api regenerates + diffs the schema from its
side, so drift in either direction is caught.

Locates a sibling slancha-shared checkout (`SLANCHA_SHARED_PATH` env, else
`../slancha-shared`); skips cleanly when absent so dev without the sibling
still passes. The api-independent structural pins below always run.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from mesh.contracts import MESH_USAGE_EVENT_FIXTURE


def _shared_schema_path() -> Path | None:
    """Locate slancha-shared's published schema, if the sibling is on disk.

    Override via SLANCHA_SHARED_PATH (points at the slancha-shared repo root).
    Default: ../slancha-shared next to this repo. Returns None → test skips.
    """
    env = os.environ.get("SLANCHA_SHARED_PATH")
    root = Path(env) if env else Path(__file__).resolve().parents[2].parent / "slancha-shared"
    schema = root / "schemas" / "mesh-usage-event.schema.json"
    return schema if schema.exists() else None


def _fixture_payload() -> dict:
    raw = json.loads(MESH_USAGE_EVENT_FIXTURE.read_text())
    raw.pop("_meta", None)  # _meta is documentation, not part of the wire shape
    return raw


@pytest.fixture(scope="module")
def shared_schema() -> dict:
    path = _shared_schema_path()
    if path is None:
        pytest.skip(
            "sibling slancha-shared checkout not found — set SLANCHA_SHARED_PATH "
            "or check out ../slancha-shared. Cross-repo validation skipped."
        )
    return json.loads(path.read_text())


def test_fixture_validates_against_shared_schema(shared_schema):
    """The canonical fixture must validate against the published JSON Schema.

    If this fails: the fixture and the published MeshUsageEvent schema disagree
    — a wire-protocol break. Fix one or the other and bump _meta.fixture_version
    (and, if the schema changed, regenerate it in slancha-api).
    """
    jsonschema = pytest.importorskip("jsonschema")
    jsonschema.validate(_fixture_payload(), shared_schema)


def test_shared_schema_declares_otel_dotted_aliases(shared_schema):
    """The OTel dotted-name aliases are part of the wire contract (H19)."""
    props = shared_schema.get("properties", {})
    for key in ("gen_ai.request.model", "gen_ai.usage.input_tokens", "gen_ai.usage.output_tokens"):
        assert key in props, f"shared schema missing OTel alias {key!r}"


def test_fixture_carries_otel_dotted_keys():
    """Fixture side of the alias contract — runs without the sibling."""
    payload = _fixture_payload()
    assert payload["gen_ai.request.model"] == payload["model"]
    assert payload["gen_ai.usage.input_tokens"] == payload["tokens_in"]
    assert payload["gen_ai.usage.output_tokens"] == payload["tokens_out"]


def test_decision_reason_structured_shape_pinned():
    """The structured decision-reason shape (NC5) is part of the wire.

    Runs without slancha-shared present — it pins the SHAPE of decision_reason
    (a dict[str, Any] in the Pydantic, so JSON Schema can't enforce it).
    """
    payload = _fixture_payload()
    dr = payload["decision_reason"]
    assert set(dr.keys()) >= {"winner", "alternatives_considered", "deciding_axes", "preset_applied"}
    for alt in dr["alternatives_considered"]:
        assert set(alt.keys()) >= {"id", "delta", "losing_axes"}


def test_fixture_carries_meta_self_documentation():
    """The fixture must keep its _meta block so the schema-pin relationship
    between repos is discoverable from the file alone."""
    raw = json.loads(MESH_USAGE_EVENT_FIXTURE.read_text())
    assert "_meta" in raw
    meta = raw["_meta"]
    assert "fixture_version" in meta
    assert "consumed_by" in meta
    assert len(meta["consumed_by"]) >= 3
