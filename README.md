# Slancha-Mesh

**Federate your local LLM nodes — Macs, GPU boxes, small homelab rigs —
into one OpenAI-compatible endpoint with hardware-aware routing across
specialists.**

You probably already run Ollama or vLLM on one box. Slancha-Mesh is the
layer on top: discover every node on your LAN (or tailnet), figure out
who's good at what (code / reasoning / multilingual / small-and-fast),
and route a prompt to the right one. No central server required (one is
optional). No data leaves your hardware. Apache-2.0.

> **Status, honestly:** the routing + discovery + heartbeat substrate is
> hardened (650+ unit tests, ruff-clean, 3 live demos on real GB10
> hardware). The catalog has 1 bring-up-validated specialist
> (`qwen3-coder-30b-a3b-fp8`) and 11 DRAFT cards spanning Ollama
> (Llama-3.1-8B, Qwen2.5-Coder-7B, DeepSeek-Coder-V2-Lite-16B,
> Phi-3.5-mini, Gemma-2-9B, Mistral-Nemo-12B) and vLLM (Qwen3-Coder/Math,
> Llama-3.1-8B, Aya-Expanse-8B, Phi-4-14B). See
> [`docs/CATALOG_STATUS.md`](docs/CATALOG_STATUS.md) for the per-card truth.

## 60-second quickstart (one box, Ollama already installed)

Copy/paste this whole block. By the end you'll have a model answering a
real prompt *through the mesh router* — and you'll know exactly which line
proves it worked.

```bash
# 1. Install the project. uv is fastest; the plain pip path works identically.
#    uv:  curl -LsSf https://astral.sh/uv/install.sh | sh
#    (or skip uv entirely and use the python -m venv path on the next line)
git clone https://github.com/SlanchaAi/slancha-mesh.git
cd slancha-mesh
uv venv && source .venv/bin/activate && uv pip install -e ".[dev]"
# --- pip-only alternative (no uv): ------------------------------------
# python -m venv .venv && source .venv/bin/activate && pip install -e ".[dev]"
# ---------------------------------------------------------------------

# 2. Pull a model your hardware can serve, through your existing Ollama:
ollama pull qwen2.5-coder:7b-instruct-q4_K_M

# 3. Start the node — adopts your running Ollama daemon, advertises on :8088.
slancha-mesh up --specialist qwen2.5-coder-7b-q4-ollama

# 4. In another terminal: a drop-in OpenAI /v1 endpoint over your mesh.
slancha-mesh router --peer 127.0.0.1 --port 8080

# 5. In a third terminal: ask a question — same shape as api.openai.com /v1.
curl -s http://localhost:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen2.5-coder-7b-q4-ollama",
    "messages": [{"role":"user","content":"reverse a string in python"}]
  }' | jq -r '.choices[0].message.content'
```

Expected output from step 5 — a short code answer coming back:

```text
You can reverse a string in Python with a slice:

    s = "hello"
    print(s[::-1])   # 'olleh'
```

**That text is the proof.** It came back from `localhost:8080` — your
local mesh router — which means the router discovered your node and
routed the prompt to the model running on *your* Ollama daemon. No cloud,
no API key, one box.

Want to see *what's reachable* instead of just getting an answer? The
`discover` command is the route-table / reachability inspector (secondary
to the answer above):

```bash
# Reads local node-info, skips Tailscale, shows the node as reachable=1
# with the specialist bound to your Ollama daemon's URL.
slancha-mesh discover --peer 127.0.0.1
```

### which backend actually runs on your hardware?

The planner recommends an engine per OS; the table below is what *actually
serves today* vs. what's recommended. Pick the model size your VRAM fits.

| Hardware | Real backend today | Notes |
|---|---|---|
| Apple Silicon Mac (e.g. 16GB) | **Ollama** | planner prefers MLX, but the MLX backend isn't wired yet → falls back to Ollama. Good for 7B Q4. |
| Windows + NVIDIA (e.g. 8GB) | **Ollama** | native CUDA. vLLM is Linux/WSL-only. Good for 7B Q4. |
| Linux + NVIDIA discrete ≥24GB (3090/4090) | **vLLM** | throughput; FP8 on Ada/Hopper+, else AWQ. Ollama also works. |
| Linux + NVIDIA <24GB | **Ollama** | GGUF fits. |
| GB10 / DGX Spark (aarch64 unified) | **Ollama** | no official vLLM sm_121 wheels yet. |
| CPU-only | **Ollama** | planner says llama.cpp, but that backend isn't wired yet → use Ollama. |

> Ollama is the universal real backend today; vLLM adds throughput on
> Linux. MLX/llama.cpp native paths are planned (recommended by the
> planner, not yet wired). See
> [`docs/CATALOG_STATUS.md`](docs/CATALOG_STATUS.md) for the per-card truth.

## 5-minute quickstart (two boxes, LAN, still no Tailscale)

On **box A** (say a Mac mini, IP `192.168.1.10`):

```bash
OLLAMA_HOST=0.0.0.0:11434 ollama serve              # one-time: bind Ollama on the LAN
ollama pull phi3.5:3.8b-mini-instruct-q5_K_M
slancha-mesh up --specialist phi-3.5-mini-q5-ollama --node-info-host 0.0.0.0
```

On **box B** (say a 3090 box, IP `192.168.1.20`):

```bash
ollama pull qwen2.5-coder:7b-instruct-q4_K_M
slancha-mesh up --specialist qwen2.5-coder-7b-q4-ollama --node-info-host 0.0.0.0
```

From either box (or your laptop), federate:

```bash
slancha-mesh discover --peer 192.168.1.10 --peer 192.168.1.20
```

You get a routing table that knows box A is good at small/easy and box B
is good at code. The router (`mesh.select.select_mesh_route`) takes a
classifier verdict and picks the right one based on `domain` +
`difficulty_tiers` + live queue depth + measured p95 latency.

See [`docs/HOMELAB.md`](docs/HOMELAB.md) for the longer walkthrough
(2-GPU rigs, mixed Mac+Linux, fault-tolerant routing).

## Why not just …

| Existing tool | What it does well | What Slancha-Mesh adds |
|---|---|---|
| **Ollama / LM Studio** | Easy single-box model serving; great UX. | Federating *N* such boxes into one routed endpoint with hardware-aware specialist allocation. Ollama is a great backend for Slancha-Mesh — that's the default Ollama path here. |
| **exo / petals** | Splits *one* model's layers/tensors across nodes for memory-bound inference. | Opposite topology: route *different models* (specialists) to *different nodes*. Complementary, not competing — if you want to run a single 70B model split across 4 Macs, use exo; if you want each box to be the right size for a specialist and a router to pick which specialist answers, use Slancha-Mesh. |
| **vLLM / llama.cpp directly** | Best-in-class single-engine throughput. | The mesh treats them as backends. The router still sees `/v1/chat/completions`; the engine choice happens behind that seam. |
| **Litellm / OpenRouter** | Unified API across N hosted providers. | Same OpenAI-compat surface, but every node is *yours* on *your* hardware — no third-party inference billing, no data egress. |

## What ships (today)

| Module | Status | Notes |
|---|---|---|
| `mesh/cli.py` | v0.0.7 | `slancha-mesh` CLI: `up` / `discover` / `status` / `serve` / `doctor` / `plan` |
| `mesh/discovery.py` | v0.0.7 | Pull discovery: walk tailnet OR explicit `--peer` list → routes, host-pinned `node_url`s |
| `mesh/node_server.py` | v0.0.7 | `build_node()` — daemon + `/models` self-description share one registry |
| `mesh/registry.py` | v0.0.7 | Event-sourced; thread-safe (compaction race fixed in #44); deterministic replay |
| `mesh/backends.py` | v0.0.7 | `VLLMBackend`, `OllamaBackend` (#45), `NullBackend`. `BaseBackend` Protocol is the seam — adding `llamacpp` / `mlx` is one class. |
| `mesh/serve.py` | v0.0.7 | `ServeDaemon` boots backends, runs heartbeat loop |
| `mesh/select.py` | v0.0.7 | `select_mesh_route` — classifier verdict + snapshot → ranked routes + cloud fallback |
| `mesh/allocator.py` | v0.0.1 | `model_fit_score` + 3 cluster strategies |
| `mesh/probe.py` | v0.0.1 | NodeProbe with GB10 unified-mem detection |
| `mesh/catalog/*.toml` | 12 cards | 1 bring-up-validated + 11 DRAFT — [`docs/CATALOG_STATUS.md`](docs/CATALOG_STATUS.md) |
| `mesh/tests/` | 650+ tests | hermetic unit suite + live-vLLM integration tests (gated by `VLLM_LIVE_URL`) |

## Backend support

| Backend | Status | How to use |
|---|---|---|
| `vllm` | wired, kernel-gated on Blackwell | Linux/WSL + CUDA. Native FP8 on Hopper/Ada; Marlin weight-only fallback on Blackwell consumer (sm_120/sm_121, vLLM 0.17 ships no `cutlass_scaled_mm` FP8 GEMM yet). |
| `ollama` | **wired (#45)** | Mac (any), AMD, Windows + NVIDIA, GB10, small NVIDIA. Adopts your running Ollama daemon at `127.0.0.1:11434` (or `OLLAMA_HOST=0.0.0.0:11434` for LAN exposure). `OLLAMA_PORT` env honored. |
| `llamacpp` | sketched, not wired | Set `required_backend = "ollama"` + add `ollama_tag` to serve GGUF through Ollama meanwhile. |
| `mlx` | sketched, not wired | Same workaround: route Apple Silicon through Ollama, which ships native Metal acceleration. |

## Multi-machine over Tailscale (production posture)

Once you outgrow LAN — boxes on different networks, no port forwarding,
encrypted transport — Slancha-Mesh's pull discovery walks a Tailscale /
Headscale tailnet for `tag:specialist` peers and pulls each node's
`/models` over WireGuard. The tailnet ACL is the credential; nothing is
exposed to the open internet.

```bash
slancha-mesh up --tailnet --auto --key tskey-...
slancha-mesh discover --tailnet
```

See [`ONBOARDING.md`](ONBOARDING.md) for the full tailnet bring-up
(tagging, ACL shape, MagicDNS resolution, the `tag:specialist` two-way
membrane).

## How to run

```bash
# Hermetic unit tests (~3 s, 650+ tests)
uv run pytest mesh/tests/ -v

# Live vLLM integration tests (require a running vLLM)
VLLM_LIVE_URL=http://127.0.0.1:8001 \
  uv run pytest mesh/tests/test_integration_vllm.py -v

# Probe the local machine
uv run python -m mesh.probe --pretty

# Plan: what would mesh allocate to this box?
slancha-mesh plan

# Doctor: diagnose tagged-but-undiscoverable, ACL gaps, etc.
slancha-mesh doctor

# Bring up a Spark node end-to-end (probe → vLLM serve → smoke test)
bash mesh/scripts/bring-up-spark.sh qwen3-coder-30b-a3b-fp8 8001
```

The v0.0.2 bring-up surfaced the FP8 kernel coverage gap on GB10
(Blackwell sm_121) and the Marlin weight-only fall-back path; see the
commit history for details. To exercise the mesh/serving plumbing
without fighting the kernel gap (or OOMing a shared box), serve a
small cached model under the catalog's `--served-model-name` — routing,
discovery, and the backend lifecycle are model-agnostic.

## Two control planes share `:8088` — pick one

Discovery and the optional central registry both default to `:8088` but
mean *opposite* things:

- **Pull / per-node `/models`** (default; what `slancha-mesh up` runs):
  each node serves its own self-description; a consumer walks the
  tailnet (or your explicit `--peer` list) and pulls. No central server
  to run, no shared write token.
- **Push / central registry** (optional, for ops dashboards or
  slancha-api integration): one shared `MeshRegistry` instance behind
  `POST /heartbeat` + `GET /registry`. Run it standalone with the
  [`docker/`](docker/docker-compose.yml) image, or mount
  `mesh.registry_app` into slancha-api ([Wire to slancha-api](#wire-to-slancha-api)).

Don't point a pull consumer at a push registry; pick one model per
deployment.

## Design decisions worth remembering

- **Unified-mem nodes get `RAM - 8GB OS reserve`** as their effective
  model-fit budget. GB10 reports `[N/A]` for VRAM via nvidia-smi; the
  probe detects this and falls back to RAM with a warning.
- **Tiered allocator diversifies before duplicating**. 2-Spark cluster
  → one math, one code. 5-Spark cluster → 3 tier-1 + 2 replicas of
  highest-traffic domain.
- **Routes are pre-ranked at snapshot time**, not per-request.
- **Snapshot replay is pure** from the event log; the registry itself
  is thread-safe under concurrent `POST /heartbeat` (the compaction
  pass-1/pass-2 lost-update race was closed in #44).
- **Backend abstraction lets us swap engines without router changes.**
  `ServeDaemon` doesn't know vLLM or Ollama exists — only
  `BaseBackend` does.
- **Adopt-don't-own the local daemon.** `VLLMBackend` adopts a
  port-busy `vllm serve`; `OllamaBackend` adopts the user's running
  Ollama daemon. Mesh never SIGTERMs a process it didn't spawn.
- **Heartbeat reports degraded, never crashes the daemon.** Backend
  death → next heartbeat says `health="degraded"` and
  `loaded_models=[]`; the router naturally falls through to the next
  route in the fallback chain (spec §6.6).

## How to extend

### Add a specialist

Drop a TOML in `mesh/catalog/` matching the `SpecialistCard` schema. For
the card to *actually serve*, set `required_backend` to `vllm` or
`ollama` (and provide the corresponding `ollama_tag` / live `model_id`).
For an end-to-end working example see
`mesh/catalog/qwen2.5-coder-7b-q4-ollama.toml` (Ollama) and
`mesh/catalog/qwen3-coder-30b-a3b-fp8.toml` (vLLM).

### Add a backend

1. Append to the `Backend` Literal in `mesh/models.py`.
2. Add detection to `mesh/probe.py:_detect_backends`.
3. Implement the `BaseBackend` Protocol in `mesh/backends.py`
   (mirror `OllamaBackend` for an adopt-the-daemon shape, or
   `VLLMBackend` for an own-the-subprocess shape).
4. Add a branch in `mesh/serve.py:build_backend()`.

### Wire to slancha-api

This is one of the optional **central**-registry (push) modes — the
standalone mesh above is pull-only and needs none of it.
`mesh/registry.py` exposes the FastAPI request/response shapes
(`HeartbeatPostRequest`, `RegistryGetResponse`). Mount
`mesh.registry_app.create_mesh_app(registry=shared_registry)` on
slancha-api at `/mesh/v1`.

### Plug into existing selector

`mesh/select.py:select_mesh_route` returns a `MeshSelectionResult` that
extends slancha-api's `SelectionResult` shape. Call it before falling
through to `select_model_lmarena`; on `cluster_coverage_used=False`,
defer to the existing cloud selector.

## License

Apache-2.0. See [`LICENSE`](LICENSE).
