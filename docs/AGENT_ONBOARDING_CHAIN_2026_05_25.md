# Agent-driven onboarding chain — walkthrough, gap analysis, topology

> 2026-05-25. Walks the "pull slancha-mesh → one command → live specialist
> (+ optional cloud)" chain the founder described, step by step, against the
> actual code + 2026 research (Tailscale/Headscale, inference engines,
> AGENTS.md / agent-CLI conventions). Marks each step **built / partial /
> gap**, persona-critiques, and resolves the cloud↔mesh routing topology.
> Companion: `MESH_ONELINE_SETUP_PROPOSAL_2026_05_25.md` (the pull-discovery
> decision this builds on).

## TL;DR

- The node-side one-liner exists (`slancha-mesh up`). The **agent-facing**
  surface (machine-readable plan/decision, AGENTS.md) does not — that's the
  biggest gap for "my agent sets it up."
- "Figure out a right-sized model **and engine**" is **half-built**: the
  allocator fit-scores *catalog* specialists, but nothing chooses an
  **engine** for the hardware, and the catalog is vLLM-centric — so a Mac /
  CPU / GB10 box gets "nothing fits." Engine selection is the missing primitive.
- "First install vs existing mesh" is **answerable today** (count online
  `tag:specialist` peers) but **not wired**, and "establish the mesh +
  install the local router (slancha-local)" is **not done** — mesh and
  slancha-local are separate repos with no install handshake.
- **Topology resolved:** cloud traffic must NOT fan out through a home
  "central node." The same `slancha-local-proxy` router runs on a **per-account
  cloud gateway node that is itself on the tailnet** (`tag:gateway`), which
  **pull-discovers** specialists and dials them directly. The home node is
  never in the cloud critical path. The founder's "t3g worker on the tailnet"
  instinct is correct; the "fan out through slancha-local at the central mesh
  node" instinct is the gap.

---

## The chain, step by step

### Step 1 — "I have a tailnet, I pull slancha-mesh, I run a command"
**Status: built (human), gap (agent).**
`slancha-mesh up` exists and is idempotent. But:
- `pip install slancha-mesh` is **not on PyPI** (404) — install is from source today. [verified]
- The CLI is **human-text only**. For an agent to drive it we need
  machine-readable output (`--json`), deterministic exit codes, and
  `next_steps` hints — the agent-CLI baseline that `gh --json`, `terraform
  -json`, `tailscale status --json` all meet. [P, cli.github.com; developer.hashicorp.com; 2026]
- No `AGENTS.md` telling an agent *how* to drive the setup. agents.md is a
  real cross-tool convention (60k+ repos; OpenAI Codex origin; per-dir
  nesting; "commands + judgment boundaries the agent can't infer"). [P, agents.md; S, github.blog 2026]

→ **Action:** add `AGENTS.md` (done this session) + `--json`/`next_steps` on
`plan`/`status`/`discover` (down-payment this session: `plan --json`).

### Step 2 — "mesh + my agent figure out a model AND serving engine right-sized for my hardware"
**Status: half-built — this is the weakest link.**
What exists: `mesh/allocator.py model_fit_score` hard-filters catalog cards by
`required_backend ∈ probe.available_backends`, `min_vram_gb ≤ effective_mem`,
`storage_gb ≤ disk_free_gb`, then soft-scores throughput/coverage. `up --auto`
picks the top fit. What's missing:
1. **No engine selection.** The card *declares* `required_backend`; nothing
   recommends an engine for the box. Engine choice is strongly
   hardware-dependent [P, research 2026]:
   - Apple Silicon → **MLX** (or Ollama) — vLLM not supported.
   - GB10 / DGX Spark (sm_121, aarch64) → **Ollama / llama.cpp** easiest;
     vLLM is **community-build-only** (no official sm_121 aarch64 wheels — the
     vLLM team says "use Ollama"); TensorRT-LLM (NVFP4) is beta. [P, vLLM issue #36821; nvidia TRT-LLM rel-notes; 2026]
   - Consumer RTX 8–16 GB → Ollama/llama.cpp GGUF; 24 GB → vLLM+AWQ.
   - CPU-only → llama.cpp/Ollama GGUF.
   This is exactly why `--auto` on the founder's Mac returned **"nothing
   fits"** — every catalog card needs `vllm`, which a Mac doesn't have. The
   allocator did the right thing on the wrong inputs.
2. **Catalog is the *operator's* models, not "what this box should serve."**
   An external contributor's catalog should be derived from their hardware +
   the mesh's coverage gaps, not paul's 8 vLLM cards.
3. **"Mesh intelligence decides what to put there" is genuinely
   agent-territory.** The allocator gives fit scores + cluster coverage; the
   *judgment* (given my hardware, the mesh's gaps, and what models exist, what
   should I pull and serve?) is what the agent adds. The right shape:
   `slancha-mesh plan --json` emits {probe, recommended_engine,
   eligible_specialists, cluster_coverage_gaps}; the agent decides; `up
   --specialist <chosen>` executes.

→ **Action:** add `recommend_engine(probe)` (hardware→engine+quant decision
tree from research) + `slancha-mesh plan --json` (done this session). Auto-card
generation from a chosen HF model stays a follow-up.

### Step 3 — "first install → establish mesh + install local router; existing mesh → hook up + broadcast capabilities. How do we confirm existing?"
**Status: detection answerable (not wired); router-install gap.**
- **"Is there an existing mesh?"** No Tailscale "first node" API exists; the
  correct check is **count online `tag:specialist` peers** in `tailscale
  status --json` [P, research]. `parse_specialist_peers()` already does this —
  `len(peers) == (1 if self tagged else 0)` ⇒ first node. Also detect a
  `tag:gateway`/router peer to know if a router already exists. **Not yet
  surfaced as a decision the onboarding flow acts on.**
- **"Establish the mesh + install slancha-local (the local router)."** Today
  slancha-mesh ≠ slancha-local; `up` serves a *specialist* but does **not**
  stand up a router. In a pure home mesh you need a router (the OpenAI
  endpoint that fans out). Gap: first-node onboarding should detect "no router
  present" and offer to install/run `slancha-local` (or designate this node
  as router). Cross-repo install handshake doesn't exist.
- **"Broadcast capabilities card."** In the pull model this is just *exposing*
  `/models` (built). "Start with capabilities and let the routing node decide
  what to place" = run `allocate_cluster` over the live cluster view — exists
  as a function, not wired into onboarding.

→ **Action (proposed, needs founder call):** (a) `plan` reports
`mesh_state: first_node | joining_existing` + `router_present: bool`; (b)
decide whether `slancha-mesh` should be able to bootstrap `slancha-local`
(install handshake) or stay decoupled and just *instruct* the agent.

### Step 4 — "hit the local router; requests routed to me as a specialist; returned to caller"
**Status: built across repos, discovery-wire is the follow-up.**
slancha-local-proxy is the OpenAI-compat router (pareto-rank + dispatch); it
already has a `MeshHeartbeatLoop` and reads the registry. The remaining wire:
slancha-local should build its routing table from **pull discovery**
(`discover_specialists`) — or from the registry it already reads. Specialist
return path is plain OpenAI SSE over the tailnet (transport-agnostic). No new
mesh-side work; the consumer-side wire is the cross-repo follow-up.

### Step 5 — "cloud account → connect via tailnet → model into the cloud registry too"
**Status: topology decided below; auto-registration is free under pull.**
Under pull discovery, a per-account **cloud gateway node** that is on the
customer's tailnet (`tag:gateway`) **discovers the new specialist
automatically** — there is no separate "push into the cloud registry" step.
The cloud's routing table *is* the discovered set. Adding a model = it appears
on the gateway's next discovery pass.

---

## Topology resolution (the founder's explicit question)

> "routing through slancha-local at the central mesh node because of fan out?
> Or route through whatever that t3g worker connected to the tailnet for
> scale out?"

**Answer: the latter.** The canonical design (`SLANCHA_PROTOCOL_v0.1_DRAFT`)
already puts `slancha-local-proxy` **on the cloud gateway**, not on a home
node:

```
                         api.slancha.ai (CloudFront + L@E routing decision)
                                   │  primary origin = the account's gateway
                                   ▼
   ┌──────────────────────────────────────────────────────────┐
   │  PER-ACCOUNT CLOUD GATEWAY NODE  (tag:gateway, on tailnet) │
   │  runs slancha-local-proxy (router) + pull-discovers        │
   └───────────┬───────────────────────┬──────────────────────┘
               │ WireGuard (MagicDNS, model ports)             │
               ▼                                               ▼
   home specialist A (tag:specialist)            home specialist B (tag:specialist)
   serves /v1 + /models (:8088)                  serves /v1 + /models (:8088)
```

Why NOT "fan out through slancha-local at a central home node":
- **SPOF + bottleneck.** A consumer box on home internet in the cloud critical
  path = the whole account's cloud throughput is gated by one home uplink, and
  dies when that box sleeps.
- **Extra hop + latency.** cloud → home-router → home-specialist is two
  WAN-ish hops; cloud-gateway → specialist is one.
- **Wrong trust seam.** The home router would need to accept cloud traffic and
  re-dispatch — re-introducing an inbound surface the tailnet ACL was designed
  to remove.

So: **one router codebase (`slancha-local-proxy`), two deployments** — (1) a
*local* instance for home-origin traffic, (2) a *cloud gateway* instance for
api.slancha.ai traffic — both consuming the **same** pull-discovered routing
table, both dialing the same specialists. The home node is a specialist, never
the cloud's router.

**Multi-tenant scale-out** (how one SaaS reaches many private tailnets) [P, research]:
- **Now (GA, production-safe):** a **per-tenant gateway node** joined to each
  customer's tailnet (the "t3g worker per account"). Stateless/autoscalable;
  strong isolation; each consumes one tagged seat in the customer tailnet.
- **Later (alpha, purpose-built):** Tailscale **API-generated tailnets** — one
  tailnet per customer, created via API, only tagged devices, no human users.
  Tailscale positions this exactly for SaaS embedding; gated behind their
  alpha. [P, tailscale.com/blog/multiple-tailnets-alpha — alpha as of 2026]
- **Rejected:** node-sharing (quarantined, wrong direction) and subnet
  routers/Services (within-tailnet only) — neither solves cross-tailnet fan-out.

---

## Push vs pull, reconciled (resolves the cross-repo tension)

v0.0.5 shipped **push** (slancha-local `MeshHeartbeatLoop` → central
`/heartbeat`); my v0.0.7 added **pull** discovery. They are not rivals —
split by data volatility:
- **PULL = membership + capability + address** (who exists, what they serve,
  where). Slow-changing, **security-critical** (identity must == address) →
  pull is mandatory here.
- **PUSH = liveness + load** (queue depth, p95, health). Fast-changing,
  low-stakes → a node may push these to the gateway's registry between pulls
  to keep routing decisions fresh.

Recommendation: **pull for the routing table, optional push for load
signals.** This keeps the security property (no claim-hijack) while preserving
the existing heartbeat infra for hot load data.

---

## Persona critique (of the streamlined target)

- **Security:** pull + `tagOwners`-enforced tags + control-plane-attested
  `DNSName` = a node can neither impersonate another nor self-join the
  specialist pool. Residual: `tagOwners` MUST be configured (server-enforced,
  but a misconfig opens the pool); `:8088` must be ACL-scoped to `tag:gateway`.
  Both are deployment prerequisites — belong in AGENTS.md + a `doctor` check.
- **GPU/heterogeneous:** engine recommender is the unlock — without it, every
  non-CUDA box is excluded. With it, Mac→MLX, GB10→Ollama, CPU→llama.cpp all
  become first-class. This is the single highest-leverage fix for "works on a
  heterogeneous network."
- **SRE:** per-account stateless cloud gateways autoscale; pull self-heals;
  the home node leaving the tailnet auto-deregisters. No central-registry HA
  dependency for *discovery*.
- **DX/agent:** the agent needs `plan --json` (decide) → `up --specialist`
  (act) → `doctor --json` (verify). Today only the middle exists in
  human-readable form. Close that and an agent can run the whole chain.
- **OSS:** the home-mesh story (no cloud) must stand alone: `up` on N boxes +
  a local `slancha-local` router + `discover`. The "install the router"
  handshake (Step 3) is the missing OSS-standalone piece.

## Prioritized gap list

| # | Gap | Leverage | This session |
|---|-----|----------|--------------|
| 1 | No hardware→engine selection; catalog vLLM-only → non-CUDA boxes excluded | **Highest** (unblocks heterogeneity) | ✅ `recommend_engine` + `plan` |
| 2 | No agent-facing surface (AGENTS.md, `--json`, `next_steps`) | High (the founder's ask) | ✅ AGENTS.md + `plan --json` |
| 3 | First-node vs existing-mesh not surfaced as a decision | Medium | ✅ `plan` reports `mesh_state` |
| 4 | No "install/establish local router" handshake (mesh↔slancha-local) | High (OSS standalone) | ⬜ needs founder call |
| 5 | slancha-local consume pull discovery (cloud + local router) | High (the cloud link) | ⬜ cross-repo follow-up |
| 6 | Auto-card-gen from a chosen HF model | Medium | ⬜ follow-up |
| 7 | `doctor --json` (tagOwners, :8088 ACL, backend present, reachability) | Medium (safety) | ⬜ reuse existing `mesh_doctor` |

## Founder decisions — RESOLVED 2026-05-25

1. **Bootstrap slancha-local: YES.** `slancha-mesh up --with-router` detects
   no router (no `tag:gateway` peer, no local `slancha-local`) and installs +
   launches `slancha-local serve` (gated, not silent — heavy install asks
   first). Shipped: `mesh/router_bootstrap.py` + `--with-router`.
2. **Multi-tenant: per-account gateway, paid-only, opt-in.** Cloud
   connectivity is a **paid feature, provisioned only on request** — which
   removes the always-on multi-tailnet burden. When a paying account opts in,
   Slancha provisions a **per-account gateway node** joined to *that* account's
   tailnet (`tag:gateway`), running `slancha-local-proxy`, pull-discovering
   that account's specialists. Stateless/autoscalable; one tagged seat per
   account; strong isolation. Free/OSS tier = local mesh only (no cloud
   gateway). Migration to Tailscale **API-generated tailnets** (alpha) stays
   open but isn't required for paid-opt-in.
3. **Push + pull HYBRID (committed).** **Pull** is authoritative for
   membership/capability/address (security-critical: identity == address).
   **Push** (`MeshHeartbeatLoop`) is retained for *hot load signals only*
   (queue depth, p95, health) so routing stays fresh between pulls. Different
   data by volatility; neither dropped.

## What shipped against this analysis (2026-05-25)
- Gap #1 (engine selection): `mesh/engine_select.py` + `plan`.
- Gap #2 (agent surface): `AGENTS.md` + `plan --json` + `doctor --json`.
- Gap #3 (mesh-state detection): `plan.mesh.state` + `doctor`.
- Gap #4 (router install): `mesh/router_bootstrap.py` + `up --with-router`.
- Gap #7 (doctor): `slancha-mesh doctor` (tailnet/pull checks; completes
  plan → up → doctor).
- Cross-repo follow-ups remain: #5 (slancha-local consumes pull discovery,
  the live cloud link) and #6 (auto-card-gen from a HF model).
