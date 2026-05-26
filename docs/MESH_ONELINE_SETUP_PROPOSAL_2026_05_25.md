# One-line specialist onboarding — proposals + persona review

> 2026-05-25. Goal: get adding a new machine/specialist to slancha-mesh as
> close to a single command as possible. Low-touch from the human side,
> high-trust (tailnet membership ≈ the credential — no hand-carried secrets),
> works as an OSS library independent of slancha cloud (cloud link is P1;
> simple setup is P0). Easy to spin specialists up **and down** on a
> heterogeneous compute network.

## Where we are (grounded in the code, not the docs)

The onboarding docs (ONBOARDING.md Case B, `JAMES_NODE_SETUP.md`) describe a
7-step flow. Reading the code, the flow is **half-wired**:

- `mesh/serve.py` `ServeDaemon` heartbeats **in-process** to a `MeshRegistry`
  object, or — with no registry injected — just **logs to disk**
  (`serve.py:15`, `:243`). It never POSTs to a remote registry.
- `mesh/service.py` **does** expose `POST /heartbeat`, `GET /registry`,
  `GET /models?include=routing_meta` — but `create_mesh_app()` builds its own
  `MeshRegistry` with the catalog auto-loaded and **no heartbeats reaching
  it** (the serve daemon is a different process that doesn't push).
- So Case B's two processes (`uvicorn mesh.service` + `python -m mesh.serve`)
  **don't share state**. `/models` shows cards but `node_urls: []`.
- The gateway today doesn't use the registry at all — it reads static env
  (`SLANCHA_VLLM_BASE_URL=http://<magicdns>:8003`), per the transport survey.

So "registry-driven discovery" is aspirational, and the half-built loop is
the thing to finish. **How we finish it (push vs pull) is the real decision**
— it determines the trust model, the per-node config, and the spin-up/down
ergonomics.

There is also no `slancha-mesh` console command (`pyproject.toml` has no
`[project.scripts]`); everything is `python -m mesh.serve` / `uvicorn ...`.

## The four proposals

### P1 — Push CLI (finish the loop the obvious way)
Add an HTTP heartbeat client to `ServeDaemon`; `slancha-mesh join --key K
--registry URL` joins the tailnet, serves, and POSTs heartbeats to a central
registry. Closes the loop with the existing `POST /heartbeat`.

### P2 — Pull discovery (tailnet-native) ★
The node only **serves models + exposes its own self-description** on the
tailnet. The gateway (or any consumer) **walks `tailscale status --json`**,
filters peers tagged `tag:specialist`, and **pulls each node's
`/models?include=routing_meta`** over the tailnet, aggregating into a routing
table. No heartbeat push, no central write surface, no per-node registry URL,
no shared write token. A node deregisters simply by **leaving the tailnet**.

### P3 — Single ephemeral key (secret minimization) — orthogonal
The only secret is one **ephemeral, single-use tailscale auth key**. Minted
by Tailscale OAuth client / `headscale preauthkeys create` (OSS) or
`POST /api/v1/mesh/hosts` (cloud). Pairs with P1 or P2.

### P4 — Install path — orthogonal
`pipx install slancha-mesh` (console script) or `curl https://get.slancha.ai
| sh` + a systemd unit. Wraps whichever core model.

## Persona review (each tries to break it; P1 push vs P2 pull)

| Persona | P1 push | P2 pull |
|---|---|---|
| **Systems designer** | Central writable surface; needs its own liveness/TTL protocol (already `NODE_UNREACHABLE_AFTER=5m`). More invariants. | Discovery state is **derived from tailnet truth** — peer up/down *is* liveness. Fewer invariants. Cost: O(N) GETs per interval + pull latency. |
| **Programmer** | Build HTTP heartbeat client + retry/backoff/auth (~150 LOC + tests). | Node side ≈ **free**: run existing `create_mesh_app` in-process sharing the daemon's registry. Net-new is the walker/aggregator (~120 LOC). Reuse > new. |
| **Infra** | Every node depends on the registry being up to be discoverable; registry becomes HA-critical. | If the gateway is down, routing's down anyway — no *extra* HA surface. One ACL addition: node-info port reachable by `tag:gateway`. |
| **Security** ⚠️ | Any `tag:specialist` that can POST /heartbeat can **claim to host any specialist at any `node_url`** → routing hijack / forward user traffic to attacker URL. Needs server-side identity binding (node_url host == caller's tailnet identity), which we don't have. | **Structurally eliminates claim-hijack**: you pull a node's self-description *from its own address*, so identity == the address you'll route to. A node can't impersonate another. Residual risk = a node lying about *its own* quality/caps — the already-documented "self-reports untrusted, verify via probe" model. **Decisive win.** |
| **SRE** | Dead node lingers to TTL; heartbeat storms if interval low. | Dead node drops from the peer list in ~seconds; next pull drops it. Self-healing, faster failure detection. |
| **GPU / heterogeneous (the actual use case)** | Spin specialist up → appears next heartbeat; down → needs graceful `node_left` or waits for TTL. | Spin up → appears next pull; kill → gone next pull. **Matches "easy up and down" exactly.** |
| **OSS maintainer** | "Works standalone" still means run a registry + distribute write tokens — higher adoption barrier; couples to a central service. | Ships two clean primitives: `slancha-mesh up` (node) + `discover_specialists()` (walk tailnet → routes). Adopter with a tailnet gets a mesh with **zero central infra, zero tokens**. Cloud is just one consumer of the OSS lib. **Best OSS story.** |
| **DX** | `join --key K --registry URL` — two required args. | `up --key K` first time, then `up` — registry URL never needed (the gateway finds you). Closest to one line. |

### Verdict
**P2 (pull) wins on nearly every axis** — and the security argument is
decisive: pull-discovery binds *identity to address*, killing the
claim-hijack class that push opens. Adopt **P2 + P3 + P4**.

The one real cost the personas surface is **pull scale**: O(N) GETs per
interval is trivial for a handful of heterogeneous boxes (the actual case)
but wants caching/sharding at hundreds of nodes — explicitly a later-scale
concern, out of P0 scope. A **push fallback** (the half-built `/heartbeat`)
stays available for nodes the gateway can't dial (asymmetric ACL); keep v1
pure-pull for simplicity and note the hybrid as future.

## Chosen design

**Node — one process, one command:**

```bash
pip install slancha-mesh           # or pipx / curl|sh  (P4)
slancha-mesh up --key tskey-...    # first time: joins tailnet (tagged) + serves
slancha-mesh up                    # thereafter: already on the tailnet, just serve
```

`up` does, idempotently:
1. **Ensure tailnet membership** — `tailscale up --advertise-tags=tag:specialist
   [--login-server …]`; skip if already up with the tag. No `--key` and not up
   → fail with the exact join command (don't guess).
2. **Resolve the MagicDNS advertise host** (existing `resolve_advertise_host`).
3. **Pick specialists** — from the catalog, or auto-fit to the local probe via
   the existing allocator (`--auto`).
4. **Serve backends** bound `0.0.0.0` on the convention ports (vLLM :8003, HF
   :8004).
5. **Expose self-description** — run `create_mesh_app` **in-process**, sharing
   the daemon's `MeshRegistry`, on the node-info port (:8088) bound to the
   tailnet. This is the missing wire: the daemon's heartbeats feed the same
   registry the app serves, so `/models?include=routing_meta` reflects live
   loaded specialists + per-specialist MagicDNS `node_url`.
6. **Self-check** — confirm each backend is reachable on its advertised
   MagicDNS:port and print green/red before declaring live.

**Gateway / any consumer — OSS lib + CLI:**

```bash
slancha-mesh discover    # walk tailnet tag:specialist peers, pull each, print routing table
```

`discover_specialists(status_json, fetch=…) -> RoutingTable` — pure and
unit-testable (injected tailscale-status fixture + injected fetcher). The
slancha-api gateway imports it and refreshes routes on an interval, retiring
the static `SLANCHA_VLLM_BASE_URL` env. Cloud thus links to the mesh through
the *same* OSS code path an external adopter uses — no special-casing.

**Trust model:** tailnet membership + the `tag:specialist` ACL grant **is**
the credential. No `SLANCHA_NODE_TOKEN` exchange for intra-tailnet discovery
(the token stays available for a standalone/public registry deployment).
Secret surface = one ephemeral tailscale auth key (P3).

## Build order (each slice tested) — SHIPPED 2026-05-25

1. ✅ `[project.scripts] slancha-mesh = "mesh.cli:main"` + `mesh/cli.py` with
   `up` / `discover` / `status` / `serve` subcommands. `up` gained `--auto`
   (fit best specialist to probed hardware) + `--dry-run`. (`test_cli.py`)
2. ✅ `mesh/discovery.py` — `discover_specialists()` + `pin_host()` +
   `make_http_fetch()`; pure aggregation TDD'd, host-pinning enforced.
   (`test_discovery.py`)
3. ✅ In-process node server — `mesh/node_server.py` `build_node()`: daemon
   heartbeat loop + `create_mesh_app` share ONE `MeshRegistry`, so `/models`
   reflects live specialists + MagicDNS `node_url`. The missing wire.
   (`test_node_server.py`)
4. ✅ `ensure_joined()` idempotent join + `tailnet_status()` in `tailnet.py`.
   (`test_tailnet_join.py`)
5. ⬜ (Follow-up) auto-card-gen from a HF model id; `slancha-mesh add <model>`.
6. ⬜ (Follow-up, cross-repo) slancha-api gateway imports
   `discover_specialists()` on an interval, retiring static
   `SLANCHA_VLLM_BASE_URL` — the seamless cloud↔mesh link.

Full suite after the slice: **519 passed / 13 skipped** (was 491); ruff clean;
zero regressions.

### Node setup, before → after
```
# before: 7 manual steps, 2 disconnected processes, hand-carried node token
install ; tailscale up --advertise-tags=tag:specialist ; confirm magicdns ;
uvicorn mesh.service:create_mesh_app --host 0.0.0.0 --port 8088 ;
python -m mesh.serve --tailnet --specialist X --base-port 8003 ; write card ; validate
# after:
pip install slancha-mesh && slancha-mesh up --auto --key tskey-...   # then just: slancha-mesh up --auto
```

## Open scope flags (decide as we go)
- **Node-info port in the ACL.** Pull needs `tag:gateway -> tag:specialist:8088`
  (or fold node-info onto a model-adjacent port). One-line ACL addition.
- **`node_id` ≠ tailnet identity.** `node_id` is `/etc/machine-id`/hostname;
  pull doesn't depend on it for trust (address is identity), but the cluster
  GPU view keys on it — keep it as the stable local id.
- **Auto-card-gen fidelity.** Inferring `n_layers`/`context_window` from a HF
  `config.json` is reliable; `quality_*`/`capabilities` need a probe — ship a
  conservative default + validate, never invent.
