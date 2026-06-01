# Running a slancha-mesh node on Windows (NVIDIA)

> Status: **living doc** — written during the first real Windows bring-up
> (2026-05-26, a Windows + NVIDIA box driven by a Claude Code agent over
> `wire`). Sections marked _[verifying]_ are being confirmed on hardware now;
> _[confirmed]_ has been observed working.

## TL;DR

On **Windows**, `slancha-mesh up` works natively: it adopts your running
**Ollama** daemon via the `OllamaBackend` (wired in #45) with CUDA
acceleration. No `slancha-local` detour required for serving.

```
Ollama (GPU, native Windows)  ←  slancha-mesh up (OllamaBackend, adopts the daemon)  ←  discoverable mesh node
```

vLLM remains Linux/WSL-only. The llama.cpp backend is now wired (a card
with `gguf_path` spawns/adopts `llama-server`), so a CPU-only Windows box
with a GGUF can serve natively; the MLX backend is wired too but
Apple-Silicon-only and refuses on Windows with a clear error. On Windows +
NVIDIA the easiest real path is still Ollama, and
`slancha-mesh up --specialist <ollama-card>` serves it directly.
`plan` / `doctor` / `discover` also run on Windows and tell you what to do.

## `slancha-mesh up` on Windows — what runs

- **vLLM is Linux/WSL-only.** The OS-aware engine recommender (`recommend_engine`)
  detects Windows and recommends **Ollama** (which runs natively with CUDA
  acceleration), not vLLM. _[confirmed in code; verifying on hardware]_
- **The Ollama backend is wired (#45).** An `ollama` card resolves to
  `OllamaBackend` in `build_backend` (`mesh/serve.py`), which adopts an
  already-running Ollama daemon, ensures the tag is pulled, and serves its
  OpenAI-compat `/v1` endpoints. So `slancha-mesh up --specialist <ollama-card>`
  does real inference on Windows.
- **`llamacpp` cards are wired (#61).** A card with `gguf_path` resolves to
  `LlamaCppBackend`, which spawns (or adopts) a `llama-server` process — the
  CPU-only path that runs on Windows too. `mlx` cards resolve to `MLXBackend`
  but refuse on Windows (Apple-Silicon-only); a card missing the required
  model field (`gguf_path` / `mlx_repo`) still falls back to `NullBackend`
  with a log hint.

`slancha-local` is still a fine way to run a node (richer routing), but it's
no longer *required* for Windows serving.

## Prerequisites

- Windows 10/11, an NVIDIA GPU.
- **Python 3.11+** — slancha-mesh requires it (`tomllib` + modern syntax);
  slancha-local needs **3.12**. Stock Windows boxes often ship 3.10, which
  `pip install -e .` will refuse. Install 3.12 and use it explicitly:
  ```powershell
  winget install -e --id Python.Python.3.12
  py -3.12 -m pip install -e .          # then drive everything with py -3.12
  ```
- **Tailnet (only for Step 4 / mesh registration)**, tagged `tag:specialist`:
  `tailscale up --advertise-tags=tag:specialist`. Steps 1–3 (plan/doctor,
  Ollama, local serve) work **without** the tailnet — `plan` just reports
  `mesh.on_tailnet=false` and the doctor tailnet check skips. (`tailscale` is
  `tailscale.exe`; `shutil.which` resolves it.)
- The console script `slancha-mesh` may not land on PATH on Windows; the
  module form is equivalent: `py -3.12 -m mesh.cli plan --json`.

## Bring-up

### 1. Onboarding intelligence (slancha-mesh)
```powershell
git clone https://github.com/SlanchaAi/slancha-mesh && cd slancha-mesh
pip install -e .
slancha-mesh plan --json     # → recommends "ollama" (installed:false until Ollama is on PATH)
slancha-mesh doctor --json   # → readiness punch list with fix hints
slancha-mesh status
```
The core deps (httpx/pydantic/fastapi/uvicorn/psutil) are pure-Python and
install cleanly on Windows. `plan`/`doctor`/`status` run on Windows because the
POSIX-only bits (process group signals) are only reached at *real serve/stop*
time, not during planning.

### 2. The engine (Ollama, native + GPU)
Install Ollama for Windows: <https://ollama.com/download>. Then:
```powershell
ollama pull qwen2.5-coder:7b      # or qwen2.5:3b on low-VRAM cards
ollama list
curl http://127.0.0.1:11434/v1/models
```
Once `ollama.exe` is on PATH, `slancha-mesh plan` flips `recommended_engine.installed`
to `true` (probe uses `shutil.which("ollama")`, which is cross-OS). _[verifying]_

### 3. The router that serves (slancha-local)
```powershell
git clone https://github.com/SlanchaAi/slancha-local && cd slancha-local
pip install -e .
slancha serve     # ollama_enabled defaults true → auto-uses 127.0.0.1:11434
```
The local ML classifier (treelite/libomp) is usually absent on Windows →
slancha-local falls back to **rules-based** routing, which is fine. Verify real
inference:
```powershell
curl http://127.0.0.1:8000/v1/chat/completions -H "Content-Type: application/json" ^
  -d "{\"model\":\"qwen2.5-coder:7b\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]}"
```

### 4. Join the mesh (discoverable node)
Set so slancha-local advertises onto the tailnet:
```powershell
$env:SLANCHA_MESH_REGISTRY_URL = "<registry-or-gateway-url>"
$env:SLANCHA_MESH_ADVERTISE_HOST = "<your-magicdns-name>"
slancha serve
```
From any `tag:gateway`/admin box: `slancha-mesh discover` → this node appears.

## Windows gotchas (do not fight these)

| Gotcha | Why | Do instead |
|---|---|---|
| `slancha-mesh up --specialist X` (llamacpp/mlx card) doesn't really serve | the card is missing its model field (`gguf_path` for llamacpp, `mlx_repo` for mlx) so it falls back to `NullBackend`; or it's an `mlx` card on Windows (Apple-Silicon-only) | set `gguf_path` and install `llama-server`, or use an `ollama` card (wired, #45) |
| `slancha-mesh up --with-router` fails | its process-launch is POSIX-only (`start_new_session`) | run `slancha serve` directly |
| `plan` recommends Ollama, not vLLM | vLLM is Linux/WSL-only | correct — or run vLLM under WSL2 for its throughput |
| local classifier won't load | treelite/libomp not on Windows | rules-fallback routing is automatic + fine |
| `plan` shows `engine.installed=false` after installing Ollama | winget installs to `%LOCALAPPDATA%\Programs\Ollama` and doesn't add it to the *current* shell's PATH; `shutil.which("ollama")` misses it | open a **new** terminal (PATH refreshes), or add that dir to PATH, then re-run |

## Findings log (live)

_Recorded as the first Windows bring-up proceeds; each becomes a fix or a
doc note._

- 2026-05-26: OS-aware `recommend_engine` shipped (Windows+NVIDIA → Ollama, not
  vLLM). Backend detection confirmed cross-OS (`shutil.which("ollama")`).
- 2026-05-26 first real Windows box (driven by an agent over wire):
  - **Python 3.10.11** on the box → `pip install -e .` refused (needs ≥3.11).
    Resolution: install 3.12 via winget, drive with `py -3.12`. (Documented as
    a hard prereq above — the #1 stumble for Windows users.)
  - **GTX 1070, 8 GB, Pascal sm_61** — older card, no FP8/tensor cores; Ollama's
    CUDA build supports it. 8 GB → start with `qwen2.5:3b` (q4 ~2 GB);
    `qwen2.5-coder:7b` q4 (~4.5 GB) fits but is tight with context.
  - **tailscale not installed** — only blocks Step 4 (mesh registration).
    Steps 1–3 proceed locally; joining the tailnet tagged `tag:specialist`
    needs a human-minted auth key.
  - **RAM probe fix VERIFIED on Win10**: after the psutil change,
    `ram_available_gb` read `20.15` (was `0.0`) on the GTX-1070 box. cc 6.1 +
    vram 7.14 also correct. → merged to main.
  - **Ollama PATH**: winget installs `ollama.exe` to
    `%LOCALAPPDATA%\Programs\Ollama`, not added to the current shell's PATH →
    `shutil.which` (and `plan`'s `installed` check) miss it until a new shell.
    Open a fresh terminal or add the dir to PATH. (Documented in gotchas.)
  - **wire gotcha (meta):** both agents' wire daemons were initially down →
    messages queued but never delivered ("duplicate" on re-push = already in
    the relay slot, just un-pulled). Fix: `wire up` / `wire daemon` on both
    ends. (Not a slancha issue, but worth noting for agent-driven setups.)
  - **`plan` worked first try**: `recommended_engine.backend="ollama"`, cc 6.1
    read from nvidia-smi.exe, vLLM correctly ruled out (the OS-aware fix).
  - **doctor crashed twice, two distinct Windows bugs, both fixed:** (a) uncaught
    `httpx.ConnectTimeout` (Windows raises it on a closed port where Linux raises
    ConnectError) → catch `httpx.HTTPError`; (b) once fixed, it unmasked a
    `UnicodeEncodeError` — Rich box-drawing glyphs on the cp1252 console → force
    UTF-8 stdio at CLI import. Both verified on-box (slancha-local).
  - **End-to-end serve ✅**: `qwen2.5:3b` returned real text via
    curl → slancha-local :8000 (rules-fallback classifier) → Ollama :11434 →
    GTX 1070 CUDA.
  - **Mesh registration ✅ (push)**: with `SLANCHA_MESH_REGISTRY_URL` set,
    slancha-local's heartbeat registered the node in a registry over the tailnet
    — visible with `node_url=…:8000` + `loaded=[qwen2.5:3b]`.
  - **Tag gap**: the box joined the tailnet as a personal device (untagged).
    Tagged `tag:specialist` via the Tailscale admin API
    (`POST /api/v2/device/{id}/tags`) — pull-discovery + the `tag:gateway →
    tag:specialist` ACL require it. (Or join with a tagged auth key.)

## Known limitations / follow-ups (surfaced by the live run)

- **Windows Firewall blocks inbound** by default — `serve --host 0.0.0.0` binds
  the tailnet interface, but nothing reaches `:8000` from another host until an
  (elevated) rule is added:
  `New-NetFirewallRule -DisplayName "slancha 8000" -Direction Inbound -LocalPort 8000 -Protocol TCP -Action Allow`.
  Outbound (the heartbeat) is unaffected, so push-registration works regardless.
- **The `tag:specialist` tag is a TWO-WAY data-plane membrane — the registry
  must live on a `tag:gateway` host.** [headline finding] Once a node is tagged
  `tag:specialist`, the deny-by-default ACL (`tag:gateway → tag:specialist:8003,8004`)
  governs it in *both* directions: the push heartbeat, the registry's active
  health-probe, AND inference all collapse to `tag:gateway`-only. Smoking gun from
  the live run: the node heartbeated fine into a dev registry on a `tag:dev-laptop`
  laptop **while untagged**, then `last_seen` froze the instant `tag:specialist`
  propagated (heartbeat `20:14:58` vs tag-landing `20:17:30`) — the outbound
  `specialist → dev-laptop:8088` POST is now ACL-denied too. **Implication:** a dev
  registry on an untagged laptop only works pre-tag; in production the registry
  (the `SLANCHA_MESH_REGISTRY_URL` target) MUST run on a `tag:gateway` host (or be
  explicitly ACL-permitted), else every tagged specialist shows `unreachable` and
  never heartbeats. Onboarding sequencing rule: **don't tag a node until its
  registry target is ACL-reachable from that tag.** Relatedly, "my laptop can't
  curl the node" is the ACL *working*, not a bug — Tailscale even filters the node
  out of a non-gateway peer's netmap (MagicDNS won't resolve; `tailscale ping` →
  "no matching peer"). The real cross-machine proof is `gateway → specialist:8003`,
  run from the `tag:gateway` box.
- **Silent heartbeat death (fixed in slancha-local).** When the heartbeat died
  post-tag, slancha-local gave zero local signal — it kept serving and the
  heartbeat-failure log sat at `INFO` (suppressed under default `WARNING`), so the
  node believed it was healthy while it had fallen off the mesh. Fixed: slancha-local
  now logs the healthy→failing transition + recovery at `WARNING` and exposes
  `mesh` registration status (`registered` / `heartbeats_sent` / `last_success`)
  under `GET /health/detailed`. Compounds the "heartbeat INFO invisible on Windows"
  gotcha above.
- **Push vs pull discovery aren't unified for slancha-local nodes.**
  slancha-local registers via the **push** heartbeat (`SLANCHA_MESH_REGISTRY_URL`),
  but `slancha-mesh discover` (pull) fetches node-info on **:8088** — which
  slancha-local doesn't serve. A pure-pull consumer (the gateway) won't see a
  slancha-local node's models; the push registry does. Unifying these
  (slancha-local exposing the `/models` node-info surface, or the gateway reading
  the push registry) is the open architecture item.
- **vLLM-on-Windows** is not supported native (Linux/WSL). slancha-mesh's serving
  daemon is vLLM-only, so Windows nodes serve through slancha-local + Ollama.
