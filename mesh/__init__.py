"""Slancha-Mesh v0 — personal mesh of specialist small models.

See `docs/SLANCHA_MESH_V0_SPEC.md` for the full design. This package
implements the v0.0.1 minimal viable mesh:

- `probe`     — hardware probe → NodeProbe JSON
- `catalog`   — specialist TOML cards
- `allocator` — model_fit_score + allocate_cluster
- `registry`  — event-sourced heartbeat ingestion + RegistrySnapshot
- `select`    — mesh routing (extends slancha-api lmarena_selector pareto mode)

Out of scope for v0.0.1 (see `mesh/README.md`): vLLM provisioning,
libp2p discovery, idle fine-tune daemon, MCP wrapper integration.
"""

from mesh.models import (
    DifficultyTier,
    DomainId,
    LoadedModel,
    MeshSelectionResult,
    NetworkLink,
    NodeBinding,
    NodeHeartbeat,
    NodeId,
    NodeProbe,
    NodeSuggestion,
    NodeSummary,
    NodeUtilization,
    RegistrySnapshot,
    Route,
    SpecialistCard,
    SpecialistId,
)

__all__ = [
    "DifficultyTier",
    "DomainId",
    "LoadedModel",
    "MeshSelectionResult",
    "NetworkLink",
    "NodeBinding",
    "NodeHeartbeat",
    "NodeId",
    "NodeProbe",
    "NodeSuggestion",
    "NodeSummary",
    "NodeUtilization",
    "RegistrySnapshot",
    "Route",
    "SpecialistCard",
    "SpecialistId",
]

__version__ = "0.0.6"
