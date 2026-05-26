# Session log — 2026-05-25 — one-line specialist onboarding

## Goal
Get adding a machine/specialist to slancha-mesh as close to one command as
possible. Low human touch, high-trust (tailnet = credential, no hand-carried
secrets), works as an OSS library independent of slancha cloud (cloud link is
P1; simple setup is P0). Easy to spin specialists up **and down** on a
heterogeneous compute network.

## Key finding that reframed the work
The node-runs-a-registry onboarding (ONBOARDING Case B) was **half-wired**:
`mesh/serve.py` `ServeDaemon` heartbeats in-process *or just logs* — it never
POSTs to the `/heartbeat` endpoint `mesh/service.py` exposes. The two
documented processes (`uvicorn mesh.service` + `python -m mesh.serve`) didn't
share state, so `/models` listed cards with empty `node_urls`. The gateway
used static `SLANCHA_VLLM_BASE_URL` env, not the registry. So "registry-driven
discovery" was aspirational — and **how we finish the loop (push vs pull) was
the real decision.**

## Decision: pull-based discovery (P2), not push
Persona review (systems/programmer/infra/security/SRE/GPU/OSS/DX) →
`docs/MESH_ONELINE_SETUP_PROPOSAL_2026_05_25.md`. Pull won decisively:

- **Security (decisive):** pulling a node's self-description *from its own
  tailnet address* binds identity to address — a node can't advertise
  another's `node_url`. Push lets any `tag:specialist` claim any `node_url`
  → routing hijack. We host-pin defensively too (`pin_host`).
- **Spin up/down:** node leaving the tailnet = gone next pull. No TTL sweep,
  no graceful-leave protocol. Matches "easy up and down."
- **OSS story:** ships two primitives — `slancha-mesh up` (node) +
  `discover_specialists()` (walk tailnet → routes). Zero central infra, zero
  tokens; cloud is just one consumer of the same discovery lib.
- **Cost:** O(N) GETs per pull (fine for a handful of boxes; cache/shard at
  hundreds — out of P0 scope). Push fallback (`/heartbeat`) stays for nodes
  the gateway can't dial.

Trust model: tailnet membership + `tag:specialist` ACL **is** the credential;
no `SLANCHA_NODE_TOKEN` exchange for intra-tailnet discovery (token still
available for a standalone/public registry).

## Shipped (all TDD, full suite 521 passed / 13 skipped, ruff clean)
1. `mesh/discovery.py` — `discover_specialists()` + `pin_host()` +
   `make_http_fetch()`. Pure aggregation; host-pinning enforced.
   (`test_discovery.py`, 11)
2. `mesh/node_server.py` — `build_node()`: daemon heartbeat loop +
   `create_mesh_app` share ONE `MeshRegistry`. **The missing wire** — now
   `/models` reflects live loaded specialists + MagicDNS `node_url`.
   (`test_node_server.py`, 2)
3. `mesh/tailnet.py` — `ensure_joined()` (idempotent; no-op when up with tag,
   joins with key otherwise, fails loudly with the exact command when no key)
   + `tailnet_status()`. (`test_tailnet_join.py`, 7)
4. `mesh/cli.py` + `[project.scripts] slancha-mesh` — `up` / `discover` /
   `status` / `serve`. `up --auto` fits the best specialist to probed
   hardware via the allocator; `up --dry-run` prints the plan. (`test_cli.py`, 10)
5. Docs: `JAMES_NODE_SETUP.md` rewritten to lead with the one-liner;
   ONBOARDING Case B now recommends the CLI + adds `:8088` to the ACL;
   README quickstart; proposal build-order marked shipped.

### Node setup, before → after
```
# before: 7 manual steps, 2 disconnected processes, hand-carried node token
# after:
pip install slancha-mesh && slancha-mesh up --auto --key tskey-...
# thereafter: slancha-mesh up --auto
```

## Threading note (intentional)
The in-process node server runs the daemon heartbeat loop (the only writer)
+ uvicorn (readers). Matches `MeshRegistry`'s documented single-writer
assumption. List-append + dict-get under the GIL is safe; no lock added.
Run single-worker.

## Not done (follow-ups, flagged in the proposal)
- `slancha-mesh add <model>` — auto-card-gen from a HF `config.json`
  (`n_layers`/`context_window` inferable; `capabilities`/`quality`
  conservative defaults, never invented).
- Cross-repo: slancha-api gateway imports `discover_specialists()` on an
  interval, retiring static `SLANCHA_VLLM_BASE_URL` — the seamless
  cloud↔mesh link (P1).
- `:8088` node-info port must be in the live tailnet ACL for `tag:gateway`.

## Iteration 2 — agent-driven onboarding chain + cloud topology

Founder asked: walk the agent-driven setup chain step by step, find gaps,
persona-critique, resolve the cloud↔mesh routing topology, decide on AGENTS.md.
Ran 3 parallel web-research agents (engine selection, Tailscale multi-tenant,
AGENTS.md conventions) — note: slancha-delegate web is NOT configured (no
EXA/TAVILY key), fell back to built-in research agents.

**Findings that settled open questions:**
- Tailscale `tagOwners` is **control-plane-enforced** — a rogue node cannot
  self-assign `tag:specialist`. Softens the earlier "rests entirely on ACL"
  critique to "rests on tagOwners being set (server-enforced)."
- Peer `DNSName` is **control-plane-attested** → "identity == address" holds;
  `HostName` is self-reported (untrusted). `pin_host` uses DNSName → sound.
- Engine choice is hardware-dependent; catalog is vLLM-only → every non-CUDA
  box got "nothing fits." Root cause of the Mac `--auto` failure.

**Shipped (TDD, 529 passed / 13 skipped, ruff clean):**
- `mesh/engine_select.py recommend_engine(probe)` — hardware→engine+quant
  decision tree (Apple Silicon→MLX, GB10/Spark→Ollama not vLLM, discrete
  NVIDIA≥24GB→vLLM, CPU→llama.cpp). (`test_engine_select.py`, 7)
- `slancha-mesh plan [--json]` — agent-facing decision surface: probe +
  recommended engine + recommended specialist + `mesh.state`
  (first_node|joining_existing) + `router_present` + `next_steps[]`.
  Live-validated on the M1 Max (→ MLX, joining_existing, no router). (+1 test)
- `AGENTS.md` — prepended a node-setup section above the GitNexus auto-block
  (preserved the `<!-- gitnexus:start/end -->` markers).
- `docs/AGENT_ONBOARDING_CHAIN_2026_05_25.md` — full chain walk + per-step
  built/partial/gap + persona critique + topology resolution + gap table.

**Cloud↔mesh topology DECISION:** cloud does NOT route through a home central
node. Same `slancha-local-proxy` runs (1) locally for home traffic and (2) on
a **per-account cloud gateway node on the customer tailnet** (`tag:gateway`)
that pull-discovers specialists and dials them directly. Home node = specialist
only. Multi-tenant: per-account gateway node (GA) now, API-generated tailnets
(alpha) later. PUSH/PULL reconciled: pull for membership/address
(security-critical), optional push for load signals.

**Still open (founder calls):** (1) mesh↔slancha-local install handshake for
first-node home mesh; (2) commit to per-account gateway nodes; (3) keep push
for load or go pure-pull; (4) auto-card-gen; (5) `doctor --json` reusing
existing `mesh_doctor`.

## Iteration 3 — founder decisions + router bootstrap + doctor (agent loop closed)

Founder resolved the 3 open decisions: (1) YES install slancha-local on first
node; (2) multi-tenant cloud = **paid-only, opt-in, per-account gateway node**;
(3) **push + pull hybrid** (my call) — pull authoritative for
membership/address, push retained for hot load signals.

**Shipped (TDD, 540 passed / 13 skipped, new code ruff-clean):**
- `mesh/router_bootstrap.py` — `detect_router` (slancha-local on PATH /
  `tag:gateway` peer) + `ensure_router` (injectable install+launch of
  `slancha-local serve`; gated behind `install=True`). (`test_router_bootstrap.py`, 9)
- `slancha-mesh up --with-router [--router-spec]` — bootstraps a router when
  none reachable; skipped under `--dry-run`. slancha-local entrypoint
  confirmed: `slancha-local serve` (src layout, console scripts
  slancha-local/slancha).
- `slancha-mesh doctor [--json]` — added pull/tailnet checks to
  `mesh_doctor.py` (`check_tailnet_specialist_ready`,
  `check_recommended_engine_installed`, `check_router_reachable`) +
  `run_node_doctor`. Completes the **plan → up → doctor** agent loop.
  Live-validated on the M1 Max: correctly flags missing tag:specialist, MLX
  not installed, no router, :8088 down — each with a fix hint.
- Docs: AGENTS.md (loop + router + doctor), chain doc (decisions resolved +
  shipped table), this log.

Pre-existing lint nit (F541 in `check_systemd_unit`, line 297) left untouched
per surgical-changes discipline — not mine.

## Artifacts
- `docs/MESH_ONELINE_SETUP_PROPOSAL_2026_05_25.md` — proposals + persona review + chosen design
- `mesh/discovery.py`, `mesh/node_server.py`, `mesh/cli.py` — new modules
- `mesh/tailnet.py` — `ensure_joined` + `tailnet_status` added
- `mesh/tests/test_{discovery,node_server,tailnet_join,cli}.py` — 30 new tests
- `JAMES_NODE_SETUP.md`, `ONBOARDING.md`, `README.md` — onboarding docs updated
