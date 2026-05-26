# Running a slancha-mesh node on Windows (NVIDIA)

> Status: **living doc** — written during the first real Windows bring-up
> (2026-05-26, a Windows + NVIDIA box driven by a Claude Code agent over
> `wire`). Sections marked _[verifying]_ are being confirmed on hardware now;
> _[confirmed]_ has been observed working.

## TL;DR

slancha-mesh's serving daemon is vLLM-centric (Linux/WSL). On **Windows**, a
node serves through the cross-OS path instead:

```
Ollama (GPU, native Windows)  ←  slancha-local (OpenAI endpoint + tailnet heartbeat)  ←  discoverable mesh node
```

`slancha-mesh` itself provides the **agent-facing intelligence** —
`plan` / `doctor` / `discover` — which work on Windows and tell you what to do.

## Why not just `slancha-mesh up` on Windows?

Two reasons, both intentional:
1. **vLLM is Linux/WSL-only.** The OS-aware engine recommender (`recommend_engine`)
   detects Windows and recommends **Ollama** (which runs natively with CUDA
   acceleration), not vLLM. _[confirmed in code; verifying on hardware]_
2. **slancha-mesh only implements the vLLM backend.** `ollama`/`llamacpp` cards
   fall back to `NullBackend` (a stub that "pretends to serve"). So real
   inference on Windows comes from **slancha-local + Ollama**, not
   `slancha-mesh up`.

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
| `slancha-mesh up --specialist X` doesn't really serve | only vLLM backend is wired; ollama → `NullBackend` stub | serve via `slancha serve` (slancha-local + Ollama) |
| `slancha-mesh up --with-router` fails | its process-launch is POSIX-only (`start_new_session`) | run `slancha serve` directly |
| `plan` recommends Ollama, not vLLM | vLLM is Linux/WSL-only | correct — or run vLLM under WSL2 for its throughput |
| local classifier won't load | treelite/libomp not on Windows | rules-fallback routing is automatic + fine |

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
  - **wire gotcha (meta):** both agents' wire daemons were initially down →
    messages queued but never delivered ("duplicate" on re-push = already in
    the relay slot, just un-pulled). Fix: `wire up` / `wire daemon` on both
    ends. (Not a slancha issue, but worth noting for agent-driven setups.)
- _(pending: real `plan --json` / `doctor --json` output once 3.12 is installed)_
