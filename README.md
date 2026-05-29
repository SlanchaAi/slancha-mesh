# Slancha-Mesh

Mesh of specialist small models on user hardware, fronted by Slancha's
classifier-driven router.

> **Transport:** nodes are reached privately over a Tailscale/Headscale
> tailnet (a cloud gateway dials home `tag:specialist` nodes by MagicDNS on
> the model ports), not per-host public tunnels. See `ONBOARDING.md`.

> **Status**: bring-up validated on DGX Spark (GB10) serving
> Qwen3-Coder-30B-A3B-FP8 via Marlin weight-only FP8 fallback (~46 tok/s,
> ~150ms TTFT).

## Install

```bash
git clone https://github.com/SlanchaAi/slancha-mesh.git
cd slancha-mesh
uv venv && source .venv/bin/activate
uv pip install -e ".[dev]"           # core + tests
uv pip install -e ".[dev,serve]"     # adds vLLM for live serving
```

## Quickstart — join the mesh (one command)

```bash
pip install -e .                            # from source (not yet on PyPI); installs the `slancha-mesh` command
slancha-mesh up --auto --key tskey-...      # join tailnet (tagged), fit+serve a specialist, expose discovery
slancha-mesh up --auto                       # thereafter (already on the tailnet)
slancha-mesh status                          # this box's tailnet identity + specialist-readiness
slancha-mesh discover                        # any consumer: walk the tailnet → routing table
```

Discovery is **pull-based**: a consumer walks `tailscale status` for
`tag:specialist` peers and pulls each node's `/models` over the tailnet — no
central registry to run, no heartbeat-push between nodes, no shared write
token. Tailnet membership + the ACL is the credential.

Each node *does* keep a registry — but a local one, filled by its own
heartbeat loop and served on its `/models` endpoint. A central, cross-node
registry is **optional** (the *push* model): run it standalone with the
[`docker/`](docker/docker-compose.yml) image, or mount `mesh.service` into
slancha-api ([Wire to slancha-api](#wire-to-slancha-api)). One gotcha — the
per-node `/models` endpoint and the central registry **both default to
`:8088`** but mean opposite things (a node's own self-description vs. one
shared store that nodes push to), so don't point a pull consumer at a push
registry; pick one model per deployment. Contributor walkthrough:
`NODE_SETUP.md`.

## What ships (current state)

| Module | Status | Notes |
|---|---|---|
| `mesh/probe.py` | v0.0.1 | NodeProbe with GB10 unified-mem detection |
| **`mesh/cli.py`** | **v0.0.7** | `slancha-mesh` CLI: `up` / `discover` / `status` / `serve` |
| **`mesh/discovery.py`** | **v0.0.7** | Pull discovery: walk tailnet → routes, host-pinned `node_url` |
| **`mesh/node_server.py`** | **v0.0.7** | `build_node()` — daemon + self-description app share one registry |
| `mesh/catalog/*.toml` | v0.0.2 | 6 specialist cards; only `qwen3-coder-30b-a3b-fp8` is bring-up-validated on real hardware — the others are config-complete but unproven-servable |
| `mesh/allocator.py` | v0.0.1 | `model_fit_score` + 3 cluster strategies |
| `mesh/registry.py` | v0.0.1 | Event-sourced; in-memory; deterministic replay |
| `mesh/select.py` | v0.0.1 | `select_mesh_route` with route_class + cloud fallback |
| **`mesh/backends.py`** | **v0.0.2** | `BaseBackend` Protocol + `VLLMBackend` + `NullBackend` (spec §9) |
| **`mesh/serve.py`** | **v0.0.2** | `ServeDaemon` boots backends, runs heartbeat loop |
| **`mesh/scripts/bring-up-spark.sh`** | **v0.0.2** | One-command boot on a Spark |
| `mesh/tests/` | v0.0.2 | 600+ unit tests + live-vLLM integration suite (gated) |

## How to run

```bash
# Unit tests (hermetic, ~0.5s)
uv run pytest mesh/tests/ -v

# Live vLLM integration tests (require a running vLLM)
VLLM_LIVE_URL=http://127.0.0.1:8001 \
  uv run pytest mesh/tests/test_integration_vllm.py -v

# Probe the local machine
uv run python -m mesh.probe --pretty

# Bring up a Spark node end-to-end (probe → vLLM serve → smoke test)
bash mesh/scripts/bring-up-spark.sh qwen3-coder-30b-a3b-fp8 8001

# Run the serving daemon (heartbeats every 5s)
uv run python -m mesh.serve --specialist qwen3-coder-30b-a3b-fp8
```

The v0.0.2 bring-up surfaced the FP8 kernel coverage gap on GB10
(Blackwell sm_121) and the Marlin weight-only fall-back path; see the
commit history for details. Practical consequence: the FP8 flagship card
isn't always a clean `vllm serve` on a GB10. To exercise the mesh/serving
plumbing without fighting the kernel gap (or OOMing a shared box), serve a
small cached model under the catalog's `--served-model-name` — routing,
discovery, and the backend lifecycle are model-agnostic.

## Backend support

| Backend | Status | Notes |
|---|---|---|
| `vllm` | wired, kernel-gated on Blackwell | Works on Hopper/Ada; sm_121 FP8 ops missing in vLLM 0.17 |
| `llamacpp` | sketched, not implemented | Falls back to `NullBackend` until v0.0.3 |
| `ollama` | not implemented | Falls back to `NullBackend` |
| `mlx` | not implemented (Mac-only) | Falls back to `NullBackend` |

The `BaseBackend` Protocol is the seam: adding a new backend means
adding a `start/wait_ready/stop/utilization` implementation in
`mesh/backends.py` and a branch in `build_backend()`.

## Design decisions worth remembering

- **Unified-mem nodes get RAM - 8GB OS reserve** as their effective
  model-fit budget. GB10 reports `[N/A]` for VRAM via nvidia-smi; the
  probe detects this and falls back to RAM with a warning.
- **Tiered allocator diversifies before duplicating**. 2-Spark cluster
  → one math, one code. 5-Spark cluster → 3 tier-1 + 2 replicas of
  highest-traffic domain.
- **Routes are pre-ranked at snapshot time**, not per-request.
- **Snapshot replay is pure** from the event log.
- **Backend abstraction lets us swap engines without router changes.**
  `ServeDaemon` doesn't know vLLM exists — only `BaseBackend` does.
- **Process adoption on port-busy** — if vLLM is already serving on
  the requested port (e.g. a manual warm-up), `VLLMBackend.start()`
  adopts it via PID lookup so cold-load doesn't repeat unnecessarily.
- **Heartbeat reports degraded, never crashes the daemon.** Backend
  death → next heartbeat says `health="degraded"` and
  `loaded_models=[]`; the router naturally falls through to the next
  route in the fallback chain (spec §6.6).

## How to extend

### Add a specialist

Drop a TOML in `mesh/catalog/` matching the `SpecialistCard` schema.
For the card to *actually serve*, the model weights need to be in HF
cache or downloadable by `huggingface-cli download`. See
`qwen3-coder-30b-a3b-fp8.toml` for the v0.0.2 example.

### Add a backend

1. Append to the `Backend` Literal in `mesh/models.py`.
2. Add detection to `mesh/probe.py:_detect_backends`.
3. Implement the `BaseBackend` Protocol in `mesh/backends.py`.
4. Add a branch in `mesh/serve.py:build_backend()`.

### Wire to slancha-api

This is one of the optional **central**-registry (push) modes — the standalone
mesh above is pull-only and needs none of it. `mesh/registry.py` exposes the FastAPI
request/response shapes (`HeartbeatPostRequest`, `RegistryGetResponse`). Import
on slancha-api and wrap a `MeshRegistry` instance behind
`POST /mesh/v1/heartbeat` + `GET /mesh/v1/registry`.

### Plug into existing selector

`mesh/select.py:select_mesh_route` returns a `MeshSelectionResult` that
extends slancha-api's `SelectionResult` shape. Call it before falling
through to `select_model_lmarena`; on `cluster_coverage_used=False`,
defer to the existing cloud selector.
