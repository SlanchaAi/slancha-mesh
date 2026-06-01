# Onboarding a new specialist (or a new self-hoster)

The v0.1 substrate is "drop a card → register → restart." This walk-through
covers both cases — an existing operator adding a new specialist to their
own mesh node, and a new self-hoster standing up their first mesh node.

## Case A: new specialist on an existing mesh node

You already have a running `slancha-local` + `vLLM` (or `llama.cpp` /
`mlx-lm` / `ollama`) serving an OpenAI-compatible endpoint. You trained
or downloaded a new LoRA / merged model. Goal: get it visible in the
mesh registry so the router can pick it.

1. **Write the card TOML.** Drop a file at `mesh/catalog/<your-id>.toml`:

   ```toml
   [specialist]
   model_id            = "vendor/base-model-name"
   specialist_id       = "your-id"               # local handle, e.g. "voice-essay-v9"
   domain              = "writing"               # writing | code | math | reasoning | general | multilingual
   difficulty_tiers    = ["easy", "medium", "hard"]
   languages           = ["en"]
   required_backend    = "vllm"
   storage_gb          = 24.0                    # weights on disk
   runtime_gb          = 30.0                    # weights + KV cache at runtime budget
   min_vram_gb         = 16.0
   context_window      = 32768
   n_layers            = 32

   # Phase 5 — what agents passing X-Slancha-Pref will gate on
   capabilities        = ["streaming", "system_prompt", "tools"]

   # Phase 6 — node-self-reported quality (DISPLAY-ONLY for routers
   # by default — they trust router_observed once probes run)
   quality_node_self_reported = 4.0
   ```

2. **Restart the mesh registry** so it auto-loads the new TOML:
   ```bash
   systemctl --user restart mesh-registry.service
   ```

3. **Confirm registration:**
   ```bash
   curl -s -H "Authorization: Bearer $SLANCHA_NODE_TOKEN" \
     http://localhost:8088/registry | jq '.snapshot.catalog | keys'
   ```
   You should see `"your-id"` in the catalog list.

4. **Load the model** in your serving backend (`vllm serve ...` with
   `--lora-modules your-id=/path/to/lora`). The heartbeat client picks
   up `loaded_models` on the next 5-second tick.

5. **Confirm the binding** is visible:
   ```bash
   curl -s -H "Authorization: Bearer $SLANCHA_NODE_TOKEN" \
     http://localhost:8088/models?include=routing_meta | jq
   ```
   The new specialist appears in `data[]` with `node_urls` populated.

6. **Start the quality probe** (one-shot or via cron):
   ```bash
   python -m mesh.quality_probe \
     --base-url http://localhost:8088 \
     --token $SLANCHA_NODE_TOKEN
   ```
   This sends a probe set, writes `quality_router_observed` back into
   the card. Drift events emit to the `mesh.quality` logger.

Total time: ~5 minutes once the model is loaded.

## Case B: standing up a new mesh node from scratch

Say this is your first node. You have a beefy local machine, you want to
join the mesh, and contribute a specialist. The substrate steps:

> **Transport (2026-05-25):** nodes are reached **privately over a
> Tailscale/Headscale tailnet**, not per-host Cloudflare tunnels. A cloud
> gateway (`tag:gateway`) is the single CloudFront origin and dials home
> nodes by **MagicDNS** on the model ports. The tailnet ACL
> (`tag:gateway -> tag:specialist:<ports>`, deny-by-default) is the access
> control. No `cloudflared` / per-host public tunnel anymore. This works
> identically on Tailscale SaaS and self-hosted **Headscale** — node-side
> steps are the same; only the auth-key source + `--login-server` differ.

### Recommended: one command (`slancha-mesh up`)

Since 2026-05-25 the steps below are wrapped by a single CLI. Install the
package and run:

```bash
pip install -e .                         # from source (not yet on PyPI) → `slancha-mesh` command
slancha-mesh up --auto --key tskey-...   # first join (tags, serves, exposes discovery)
slancha-mesh up --auto                   # thereafter (already on the tailnet)
```

`up` is idempotent: it joins the tailnet tagged `tag:specialist` (only when
needed), fits the best specialist to the box's hardware (`--auto`, or use
`--specialist <id>`), serves it bound to the tailnet, and exposes the
pull-able self-description on `:8088`. Discovery is **pull-based** — the
gateway finds the node by walking the tailnet (`slancha-mesh discover`), so
there's no registry URL, node token, or per-node gateway config. Verify with
`slancha-mesh status` (tailnet identity + specialist-readiness). Full
design: `docs/SELF_ORGANIZING_LOOP_SCOPE.md`.

The manual steps below are what `up` automates — keep them for debugging or a
bespoke setup.

### One-time mesh-node setup (the long way — what `up` does for you)

1. **Install slancha-local + slancha-mesh** on your box. Pull both
   repos, install requirements.

2. **Join the tailnet as a specialist.** Get a `tag:specialist`
   ephemeral auth key, then run the join command:
   ```bash
   # Tailscale SaaS:
   sudo tailscale up --auth-key=<KEY> --advertise-tags=tag:specialist
   # Self-hosted Headscale — add your control server:
   sudo tailscale up --auth-key=<KEY> --advertise-tags=tag:specialist \
     --login-server=https://<your-headscale-host>
   ```
   Where the key comes from:
   - **Tailscale:** admin console → *Settings → Keys → Generate auth key*
     (tagged `tag:specialist`, ephemeral).
   - **Headscale:** `headscale preauthkeys create --user <user> --ephemeral`
     (with `tag:specialist` in the node's ACL tag owners).
   - **Via the dashboard:** an onboarding admin calls
     `POST /api/v1/mesh/hosts` on slancha-api, which mints a tagged key and
     returns the exact `join_command` + `model_ports` — no shell needed.

3. **Confirm your MagicDNS name** (this is what the gateway dials):
   ```bash
   tailscale status --json | jq -r .Self.DNSName   # e.g. gb10-1.<tailnet>.ts.net.
   ```
   `mesh.serve --tailnet` auto-discovers this; override with
   `--advertise-host <name>` or `SLANCHA_TAILNET_ADVERTISE_HOST`.

4. **Boot the mesh registry**, bound so the tailnet can reach it:
   ```bash
   SLANCHA_NODE_TOKEN=$(openssl rand -hex 32) \
   uvicorn mesh.registry_app:create_mesh_app --factory --host 0.0.0.0 --port 8088
   ```

5. **Serve specialists on the tailnet interface** at the model ports
   (convention matching the gateway ACL: vLLM `:8003`, HF `:8004`). The
   serve daemon binds `0.0.0.0` and advertises your MagicDNS host:
   ```bash
   python -m mesh.serve --tailnet \
     --specialist demo-model --base-port 8003
   ```
   Each loaded specialist now heartbeats a `node_url` of
   `http://<your-magicdns>:<port>` — reachable by the gateway over WireGuard.

### SaaS-side setup (gateway operator does this)

6. **Grant the gateway reach** in the tailnet ACL (deny-by-default
   otherwise): `{"src": ["tag:gateway"], "dst": ["tag:specialist"],
   "ip": ["tcp:8003", "tcp:8004", "tcp:8088"]}`. The model ports (8003/8004)
   carry inference; `:8088` is the node-info port the gateway **pulls** for
   discovery (`slancha-mesh discover` → each node's `/models?include=routing_meta`).
   SSH (`:22`) stays denied to the gateway. This is a one-time ACL entry, not
   per node — no per-user CloudFront tunnel or L@E origin entry is created.

   > **The routability invariant:** a node that registers/discovers must
   > advertise its model URL on an ACL-opened port (`:8003`/`:8004`), or it
   > is "up but unroutable." Serving on an off-ACL port (`:8000`/`:8001`)
   > registers fine but the gateway cannot reach it — `slancha-mesh doctor`
   > warns on this. See the
   > [port convention](README.md#port-convention--the-routability-invariant).

### Per-specialist (you again, repeated for each model)

7. **Follow Case A above** to register specialists on your node.

### Validation

8. **Reachability from the gateway** (run on the gateway, or any
   `tag:gateway`/admin device):
   ```bash
   curl -s http://<your-magicdns>:8003/v1/models | jq .   # backend up over tailnet
   ```
   Then end-to-end from a SaaS-shape client:
   ```bash
   curl https://api.slancha.ai/v1/chat/completions \
     -H "Authorization: Bearer slancha_<your-bearer>" \
     -H "Content-Type: application/json" \
     -d '{"model": "your-id", "messages": [{"role":"user","content":"hi"}]}'
   ```
   Expect SSE chunks streaming back via the gateway.

## Anti-patterns this onboarding prevents

- **DON'T bind the model server to `127.0.0.1`** when joining the mesh.
  The gateway is off-box now (cloud), reaching you over the tailnet —
  loopback is unreachable. Bind `0.0.0.0` (or the tailnet IP) and let the
  daemon advertise your MagicDNS name.
- **DON'T expose the model port publicly.** The tailnet ACL is the access
  control; there is no public tunnel. Don't add Funnel/port-forward —
  `tag:gateway -> tag:specialist:<ports>` is the only path in.
- **DON'T copy/paste an existing TOML** without auditing `domain` and
  `capabilities`. The router gates on these — wrong tags break routing.
- **DON'T set `quality_router_observed` directly.** That field is
  written by the central probe service. Self-reported is `quality_node_self_reported`.
- **DON'T claim capabilities your backend doesn't support.** A request
  with `require_capabilities=["tools"]` will be sent to you; if you
  don't actually do tools, it's a routing failure.

## Where to look when things break

- Node not on the tailnet: `tailscale status` shows `tag:specialist` and
  `Online`? On Headscale, did you pass `--login-server`?
- Heartbeat not registering: `GET /registry` shows your node? If no,
  check `SLANCHA_NODE_TOKEN` env on both ends.
- Registered but gateway can't reach it: from a `tag:gateway` host,
  `curl http://<magicdns>:<port>/v1/models`. Connection refused →
  backend bound to loopback (bind `0.0.0.0`). Timeout → ACL grant missing
  for that port.
- `node_url` is loopback in `/registry`: the daemon has no advertise host
  — run with `--tailnet` (or set `SLANCHA_TAILNET_ENABLED=1`), confirm
  `tailscale status --json` returns `Self.DNSName`.
- Specialist registered but no traffic: `GET /models?include=routing_meta`
  — capabilities listed? router_observed populated?
