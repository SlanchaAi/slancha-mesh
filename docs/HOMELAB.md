# Homelab walkthrough — 2 boxes, LAN-only, no Tailscale

The shortest path to a real federated mesh: two boxes on the same LAN,
Ollama on each, slancha-mesh tying them together. No Tailscale, no
docker, no central server.

The example uses an M2 Mac mini and a Linux box with a single RTX 3090,
but the shape is identical for any two boxes that can `curl` each other
on a known port (two Linux laptops, a Mac + Pi5 + eGPU, two Windows
gaming rigs).

## Topology

```
+---------------------+        LAN        +---------------------+
|  Mac mini M2        |  <----------->    |  Linux + RTX 3090   |
|  Ollama :11434      |                   |  Ollama :11434      |
|  phi-3.5-mini-q5    |                   |  qwen2.5-coder-7b   |
|  slancha-mesh up    |                   |  slancha-mesh up    |
|  node-info :8088    |                   |  node-info :8088    |
+---------------------+                   +---------------------+
                          \\               //
                           \\             //
                       +--------------------+
                       |  Any box on LAN    |
                       |  (your laptop)     |
                       |  slancha-mesh      |
                       |  discover --peer   |
                       +--------------------+
```

Both boxes run Ollama, both run slancha-mesh's serving daemon. The third
box is anyone with `slancha-mesh` installed who wants a routing table —
your laptop, a router service, or a gateway machine.

## One-time setup (on each serving box)

```bash
# Install Ollama (skip if already present).
# macOS:
brew install ollama
# Linux:
curl -fsSL https://ollama.com/install.sh | sh

# Make Ollama listen on the LAN, not just localhost.
# macOS — edit ~/Library/LaunchAgents/com.ollama.plist (OLLAMA_HOST=0.0.0.0:11434)
# Linux — systemd drop-in:
sudo mkdir -p /etc/systemd/system/ollama.service.d
echo -e "[Service]\nEnvironment=OLLAMA_HOST=0.0.0.0:11434" | \
  sudo tee /etc/systemd/system/ollama.service.d/lan.conf
sudo systemctl daemon-reload && sudo systemctl restart ollama

# Install slancha-mesh.
git clone https://github.com/SlanchaAi/slancha-mesh.git
cd slancha-mesh
uv venv && source .venv/bin/activate
uv pip install -e ".[dev]"
```

> **Trust note.** Binding Ollama to `0.0.0.0` exposes it to every host on
> the LAN with no authentication. That's the correct posture for a
> homelab where your router is the trust boundary; it's the wrong
> posture on a shared / coworking / café network. If unsure, keep
> Ollama on `127.0.0.1` and use Tailscale (see the main README).

## Box A: Mac mini, runs phi-3.5-mini (low-latency easy path)

```bash
ollama pull phi3.5:3.8b-mini-instruct-q5_K_M

slancha-mesh up \
  --specialist phi-3.5-mini-q5-ollama \
  --node-info-host 0.0.0.0 \
  --base-port 8013
```

`--node-info-host 0.0.0.0` is what lets the third (querying) box reach
this node's `/models` self-description. `--base-port` is informational
for Ollama specialists (Ollama multiplexes on 11434), but you'd set it
for any vLLM card on the same box.

## Box B: Linux + 3090, runs qwen2.5-coder-7b (code workhorse)

```bash
ollama pull qwen2.5-coder:7b-instruct-q4_K_M

slancha-mesh up \
  --specialist qwen2.5-coder-7b-q4-ollama \
  --node-info-host 0.0.0.0 \
  --base-port 8013
```

## Box C: federate + route

From your laptop (or either box above):

```bash
slancha-mesh discover \
  --peer 192.168.1.10 \
  --peer 192.168.1.20
```

You should see:

```
reachable specialist nodes: 2  unreachable: 0

specialist                   domain       nodes  node_urls
--------------------------------------------------------------------------------
phi-3.5-mini-q5-ollama       general      1      http://192.168.1.10:11434
qwen2.5-coder-7b-q4-ollama   code         1      http://192.168.1.20:11434
```

Each `node_url` is **host-pinned to the peer you actually dialed** —
even if a node lied in its `/models` response, the router only sends
traffic to its real LAN address.

## Routing an actual request

The `node_url`s above are standard OpenAI-compatible endpoints. Point
any client at one:

```bash
curl -s http://192.168.1.20:11434/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen2.5-coder:7b-instruct-q4_K_M",
    "messages": [{"role":"user","content":"reverse a string in python"}]
  }' | jq -r '.choices[0].message.content'
```

(The `model` field is Ollama's tag, not slancha-mesh's `specialist_id`.
Ollama doesn't know about specialist IDs; the mesh handles the mapping
when you route through `mesh.select`.)

For programmatic routing — classifier verdict → specialist → node_url —
the seam is `mesh.select.select_mesh_route`. See the `select_mesh_route`
docstring + `mesh/tests/test_select.py` for the call shape.

## Adding more nodes

The CLI's `--peer` is repeatable. A 4-box homelab:

```bash
slancha-mesh discover \
  --peer 192.168.1.10 \
  --peer 192.168.1.20 \
  --peer 192.168.1.30 \
  --peer 192.168.1.40
```

The discovery pass is parallel-tolerant — one slow or dead peer doesn't
block the rest. Unreachable peers surface in `unreachable: [...]`
rather than failing the whole pass.

## When to graduate to Tailscale

LAN mode (`--peer`) is right when:

- All boxes share one physical / WiFi network.
- You trust every device on that network.
- You don't need access from outside the LAN.

Tailscale mode (`--tailnet`) is right when:

- Boxes are on different networks (home + colo + cloud).
- You want encrypted transport (WireGuard) end-to-end.
- You want the ACL to be the credential instead of relying on the LAN
  being trustworthy.
- You're running this in mixed-trust environments (coworking, café,
  guest WiFi).

See [`ONBOARDING.md`](../ONBOARDING.md) for the Tailscale bring-up.

## Troubleshooting

- **`discover` reports `unreachable: [host]`** → `slancha-mesh doctor`
  on that box. The most common cause is `--node-info-host 127.0.0.1`
  (the default before `up` is wired with `0.0.0.0`); the second is the
  host's firewall blocking inbound `:8088`.
- **`/v1/chat/completions` returns 404** → you pointed at the Ollama
  daemon URL but used a wrong tag. Run `curl http://host:11434/api/tags`
  to see what's actually loaded.
- **`up` reports "Ollama daemon not reachable"** → Ollama isn't running,
  or `OLLAMA_HOST` doesn't match what slancha-mesh is dialing. Override
  with `OLLAMA_PORT` env var if you've put Ollama on a non-default port.
- **One box doesn't appear in `discover`** even with `--peer` set →
  check `slancha-mesh status` on that box; `tag:specialist` should be
  in the local tag list, and `/models` should respond at `:8088`.
