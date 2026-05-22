"""Cross-repo schema fixtures.

The `mesh_usage_event.json` fixture in this directory is the single
canonical golden trace for the §6 telemetry contract. It is consumed by:

  - slancha-api/tests/test_cross_repo_mesh_usage.py  (receiver side —
    validates via MeshUsageEvent Pydantic)
  - slancha-local/tests/integration/test_usage_sidecar_contract.py
    (emitter side — asserts the sidecar produces a payload that matches
    this fixture's keys + types)
  - slancha-mesh/mesh/tests/test_cross_repo_usage_event.py (this repo
    — validates via sibling slancha-api's Pydantic when checked out
    alongside, skips cleanly otherwise)

Any change to this fixture is a wire-protocol change. Failing tests in
multiple repos is the intended behavior — it surfaces the drift to
review before merging.
"""

from pathlib import Path

FIXTURES_DIR = Path(__file__).resolve().parent
MESH_USAGE_EVENT_FIXTURE = FIXTURES_DIR / "mesh_usage_event.json"

__all__ = ["FIXTURES_DIR", "MESH_USAGE_EVENT_FIXTURE"]
