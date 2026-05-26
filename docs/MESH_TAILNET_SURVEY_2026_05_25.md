# Slancha-Mesh transport survey — Cloudflare → Tailscale/Headscale

> 2026-05-25. Step 1 of the tailnet-transport migration. Read-only survey:
> current transport model + every place that assumes the old one. No code
> changed in this pass.

## TL;DR — the one bug that matters

`node_url` is hardwired to **loopback** (`http://127.0.0.1:<port>`).

- **Old design:** `slancha-local-proxy` ran **on the home box**, co-located
  with vLLM. The per-host Cloudflare tunnel exposed *the proxy*; the proxy
  dialed `127.0.0.1` to reach the model. Loopback worked.
- **New design:** the proxy moved to the **cloud gateway** (`tag:gateway`),
  reaching home nodes **over the tailnet by MagicDNS** on the model ports.
  A loopback `node_url` is now unreachable. The registry hands the gateway a
  URL it cannot dial.

The fix is to make nodes advertise a **tailnet-reachable** base URL
(`http://<magicdns-name>:<port>`) and bind the model server off loopback.
Everything else (event-sourcing, replay, selection) is transport-agnostic
and stays intact.

## Lifecycle of `node_url`

1. **Set (node side)** — `mesh/serve.py:post_heartbeat()` (L220-230):
   ```python
   first_backend = self.backends[0] if self.backends else None
   node_url = first_backend.base_url if first_backend else None
   req = HeartbeatPostRequest(heartbeat=hb, node_url=node_url)
   ```
   `base_url` comes from the backend: `VLLMBackend.base_url` =
   `http://{host}:{port}` with `host = "127.0.0.1"` (dataclass default,
   `mesh/backends.py:113`). `NullBackend.base_url` = `http://127.0.0.1:0`.
   → **node_url is loopback, always.**

2. **Reported (heartbeat)** — `node_url` rides *alongside* the heartbeat in
   `HeartbeatPostRequest` (`mesh/registry.py:90-98`), not inside
   `NodeHeartbeat`. Docstring says "Required on first heartbeat" but the
   field is `Optional[str] = None` (not enforced).

3. **Stored (registry)** — `MeshRegistry.record_heartbeat()` writes it two
   ways: into `self._node_urls[node_id]` (last-reported cache) and into the
   appended `HeartbeatEvent.node_url`. Event-sourced; replay in `snapshot()`
   resolves `ev.node_url or self._node_urls.get(node_id)` (`registry.py:246`).

4. **Selection → router** — `snapshot()` stamps `node_url` onto every
   `NodeSummary` and `NodeBinding`; `build_ranked_routes()` copies it into
   `Route.node_url`; `select_mesh_route[_with_pref]()` returns
   `MeshSelectionResult.node_url = primary.node_url`. The gateway forwards
   the request to that URL.

## Every place that assumes the old transport

| File | Assumption | Needs |
|---|---|---|
| `mesh/backends.py:113` | `VLLMBackend.host = "127.0.0.1"` (binds loopback) | bind off-loopback (tailnet iface / `0.0.0.0`) when tailnet on |
| `mesh/backends.py:129` | `base_url` = bind URL, doubles as advertised URL | separate **bind URL** from **advertised URL** |
| `mesh/serve.py:220-230` | `node_url = backends[0].base_url` (loopback) | build from an **advertise host** + port |
| `mesh/serve.py:68` | `node_url_template` field, unused | wire it or drop it |
| `mesh/probe.py:245` | `public_ipv4` via `hostname -I`; no tailnet awareness | optional: surface MagicDNS / tailnet IP |
| `mesh/scripts/bring-up-spark.sh:27,151` | `HOST=127.0.0.1`; prints "node_url to advertise: http://127.0.0.1:port" | `--host 0.0.0.0`; advertise MagicDNS |
| `mesh/scripts/bring-up-multi-spark.sh:60,157` | `--host 127.0.0.1`; per-port loopback URLs | same |
| `ONBOARDING.md` Case B §3-6 | CF Tunnel + CF Access token + L@E allowlist + KVS seed | replace with `tailscale up --advertise-tags` flow |
| `docs/SLANCHA_PROTOCOL_v0.1_DRAFT.md` | mesh leg = per-user CF tunnel (`mesh.<user>.laulpogan.com`), CF Access service token, L@E origin groups (#51, #53, #80) | gateway→tailnet; CF tunnel per-host is stale |
| `docs/SLANCHA_MESH_V0_SPEC.archived.md` §8/§11 | "v1 adds Tailscale tunnel"; "mTLS over Tailscale (already deployed)" | superseded doc; historical only |

## NOT the data plane (don't confuse)

`mesh/deploy/mesh-dashboard-tunnel*` + `install_dashboard.sh` = a Cloudflare
tunnel for the **Streamlit dashboard** (`evals.laulpogan.com` → `:8501`).
That's observability, not the model transport. It can stay on CF or move to
the tailnet separately — out of scope for the node-reachability fix.

## Pre-existing issues found (flag, decide scope)

1. **`node_url` is per-node, not per-specialist.** `post_heartbeat` only
   advertises `backends[0].base_url`, but a node can load multiple
   specialists on **distinct ports** (`bring-up-multi-spark.sh`: vLLM :8001
   + :8002; new arch: vLLM :8003, HF :8004). The registry binds the *single*
   node_url to *every* `loaded_model`, so a request for the :8004 specialist
   gets routed to :8003. Loopback masked this (same host); the tailnet (one
   MagicDNS name, many ports) does **not**. Migration is the natural time to
   carry a per-`LoadedModel` port/url. **Scope decision needed.**

2. **Broken doc link.** `README.md:4`, `mesh/__init__.py:3`, and four
   `mesh/deploy/*.service` `Documentation=` lines point at
   `docs/SLANCHA_MESH_V0_SPEC.md`, which does not exist (the file is
   `SLANCHA_MESH_V0_SPEC.archived.md`, superseded by
   `SLANCHA_PROTOCOL_v0.1_DRAFT.md`). The task says "update
   `docs/SLANCHA_MESH_V0_SPEC.md`" — **which doc is canonical?**

3. **Out of repo:** `serve_v8_hf.py` (the `paul-voice` vs `paul-voice-v8`
   cosmetic name bug mentioned in the task) is **not in slancha-mesh** — it
   lives in slancha-local. Nothing to fix here.

## What's transport-agnostic (leave alone)

Event log, `snapshot()` replay determinism, allocator, idle/training,
quality-probe, GPU coordination, dashboard panels, selection scoring. None
of these read or assume a URL scheme — they just pass `node_url` through.

## Control-plane-agnostic discovery (Headscale == Tailscale)

`tailscale status --json` → `Self.DNSName` yields the MagicDNS name on
**both** Tailscale SaaS and self-hosted Headscale (both implement the
`tailscale` CLI + LocalAPI; MagicDNS populates `Self.DNSName` identically).
The only divergence is **onboarding**: Headscale adds `--login-server=<url>`
to `tailscale up`. Node-side Python is identical. No SaaS-only feature
(Funnel/Serve/app-connectors) is used.

## Decisions + what shipped (2026-05-25)

Resolved against this survey:

1. **Per-specialist node_url — fixed.** `LoadedModel` gained an optional
   `node_url`; the registry binds each specialist to its own port.
2. **Canonical doc — v0.1 draft.** Tailnet transport update went into
   `SLANCHA_PROTOCOL_v0.1_DRAFT.md`; broken `SLANCHA_MESH_V0_SPEC.md` links
   repointed there; ONBOARDING Case B rewritten.
3. **Onboarding API.** `POST /api/v1/mesh/hosts` (slancha-api) is SaaS-side
   **key minting** (admin-gated; returns the `tailscale up` join command +
   `model_ports`). The node does **not** call it — it stays heartbeat-push
   and just advertises its tailnet `node_url`. ONBOARDING references it.
4. **Config — generic + env-gated.** `mesh/tailnet.py:TailnetConfig`
   (`SLANCHA_TAILNET_*` env / `--tailnet` CLI), default off.

Implemented: `mesh/tailnet.py` (new), `LoadedModel.node_url`,
`ServeDaemon.advertise_host` + bind/advertise split, registry per-binding
URL. 22 new tests; full hermetic suite 491 passed / 2 skipped.

Cross-repo facts (slancha-api): ACL `tag:gateway -> tag:specialist:8003,8004`
(`infra/tailscale/acl.hujson`); gateway currently uses static env base URLs
(`SLANCHA_VLLM_BASE_URL=http://<magicdns>:8003`) — this change lets it move
to registry-driven discovery. Out of repo: `serve_v8_hf.py` name bug lives
in slancha-local, not here.
