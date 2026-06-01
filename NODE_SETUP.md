# Joining slancha-mesh as a specialist node

A how-to for the agent setting up a contributor's machine. Goal: get the contributor's box
serving a model the mesh can route to, with the least possible human touch.

## TL;DR — the one-liner

```bash
pip install -e .                  # from source (not yet on PyPI) → installs `slancha-mesh`
slancha-mesh up --auto --key tskey-...     # first time (joins the tailnet)
slancha-mesh up --auto                      # every time after (already joined)
```

That's it. `up` joins the tailnet tagged `tag:specialist`, picks the
best-fit specialist for the box's hardware (`--auto`), serves it bound to the
tailnet, and exposes a pull-able self-description the gateway discovers on its
own. **No registry URL, no node token, no per-node config on the gateway, no
secret exchange between machines.** The only secret is one ephemeral tailscale
auth key, used once.

Replace `--auto` with `--specialist <id>` (repeatable) to serve specific
models. Run `slancha-mesh up --auto --dry-run` first to see the plan without
starting anything.

## The trust model (why it's this simple)

Tailnet membership **is** the credential. A node tagged `tag:specialist` on
the tailnet is, by the ACL (`tag:gateway -> tag:specialist:<ports>`,
deny-by-default), reachable by the gateway and nothing else. Discovery is
**pull-based**: the gateway walks its own `tailscale status` peer list and
fetches each specialist node's `/models` over the tailnet. Because it pulls a
node's description *from that node's own address*, a node can never advertise
another node's address — identity is the address. So there's no heartbeat
token to hand out and no way to hijack routing by lying.

A node leaves the mesh simply by leaving the tailnet (or stopping `up`) — it
drops off the next discovery pass. Spinning specialists up and down is just
starting and stopping the process.

## What you need from the gateway operator (once)

1. **An ephemeral, tagged auth key.** Cleanest: the operator runs
   `POST /api/v1/mesh/hosts` (slancha-api) → mints a `tag:specialist` key and
   returns the exact join command. OSS / self-hosted: Tailscale admin console
   (auth key tagged `tag:specialist`) or Headscale
   `headscale preauthkeys create --user <u> --ephemeral`.
2. **The ACL grant exists for the ports you serve** —
   `tag:gateway -> tag:specialist:8003,8004` (model ports) **and `:8088`**
   (the node-info / discovery port the gateway pulls). This is a one-time ACL
   entry, not per node. The routability invariant follows: advertise your
   model URL on an ACL-opened port (`:8003`/`:8004`) or the node is "up but
   unroutable" — see the
   [port convention](README.md#port-convention--the-routability-invariant).
3. **Headscale only:** the `--login-server=https://<host>` URL → pass
   `--control-plane headscale --login-server <url>` to `up`.

## What `slancha-mesh up` does, step by step

1. **Ensure tailnet membership** (idempotent). Already up with the tag →
   no-op. Not up + `--key` → `tailscale up --advertise-tags=tag:specialist
   [--login-server …]`. Not up + no key → fails loudly with the exact command
   to run (never a silent half-state).
2. **Resolve the MagicDNS advertise host** the gateway will dial.
3. **Pick specialists** — `--auto` fits the catalog to the probed hardware
   (VRAM/backend/throughput); or explicit `--specialist`.
4. **Serve backends** bound `0.0.0.0` on the model ports (vLLM :8003).
5. **Expose self-description** — runs the node-info app on :8088 (bound to the
   tailnet), serving live `/models?include=routing_meta` with each
   specialist's MagicDNS `node_url`. This is what the gateway pulls.

Check it from the box: `slancha-mesh status` (shows tailnet identity +
whether the box is `specialist-ready`).

## Verify it's live

```bash
# On the box:
slancha-mesh status                    # online + tag:specialist present?

# From any tag:gateway / admin device — discover what the mesh sees:
slancha-mesh discover                  # table of specialists → host-pinned node_urls
# the box should appear with node_urls like http://<its-magicdns>:8003

# End-to-end through the gateway:
curl https://api.slancha.ai/v1/chat/completions \
  -H "Authorization: Bearer slancha_<bearer>" -H "Content-Type: application/json" \
  -d '{"model":"<specialist-id>","messages":[{"role":"user","content":"hi"}]}'
```

## Running it as a service (survives reboots)

```ini
# ~/.config/systemd/user/slancha-mesh.service
[Unit]
Description=slancha-mesh specialist node
After=network-online.target tailscaled.service
Wants=network-online.target

[Service]
Type=simple
# --key only needed for the very first join; safe to drop afterward.
ExecStart=%h/.local/bin/slancha-mesh up --auto
Restart=on-failure
RestartSec=15

[Install]
WantedBy=default.target
```

```bash
systemctl --user enable --now slancha-mesh.service
```

## Don'ts (silent-failure traps `up` already guards against)

- **Don't hand-run `tailscale up` without `--advertise-tags=tag:specialist`.**
  Wrong/missing tag = on the tailnet but invisible to the gateway. `up` always
  sets the tag; `status` flags a missing one.
- **Don't bind models to `127.0.0.1`.** The gateway is off-box; loopback is
  unreachable. `up --tailnet`/`--key` binds `0.0.0.0` + advertises MagicDNS.
- **Don't expose ports publicly** (no Funnel/port-forward). The tailnet ACL is
  the only path in.
- **Don't claim capabilities the backend lacks** in a card — the router gates
  on them. `--auto` only serves cards already in the catalog.

## When it breaks

| Symptom | Cause / fix |
|---|---|
| `up` exits "not on the tailnet… run: tailscale up …" | No key and not joined. Get a tagged key from the operator, pass `--key`. |
| `status` shows `specialist-ready: False` | Missing `tag:specialist`. Re-run `up --key …` (re-joins with the tag). |
| `discover` doesn't list the box | Not online, or node-info :8088 not in the ACL for `tag:gateway`. |
| `discover` lists it but routing fails | Model port (8003/8004) not in the ACL, or backend didn't come up — check `up` logs. |
| `up --auto` says "nothing fits this box" | No catalog card passes the hardware filter (VRAM/backend). Add a fitting card or use `--specialist`. |

## Under the hood / OSS

`slancha-mesh discover` is the same library call (`mesh.discovery.
discover_specialists`) the slancha-api gateway uses to build its routing
table — so an external OSS adopter with their own tailnet gets a working mesh
with zero central infrastructure, and the cloud is just one consumer of that
discovery path. Full design + the push-vs-pull decision:
`docs/SELF_ORGANIZING_LOOP_SCOPE.md`.
