# Catalog status — what's been bring-up-validated and what hasn't

The catalog (`mesh/catalog/*.toml`) is the single source of truth for which
specialists slancha-mesh knows about. **The fact that a card is in the
catalog says only that it parses cleanly and has plausible VRAM math.** It
does NOT mean a node has been brought up and seen to serve + heartbeat +
route end-to-end for that specialist. This file records that distinction
honestly, so contributors and users know which cards are safe to point a
node at and which are still DRAFT.

## How a card moves to "validated"

A specialist is "validated" when, on at least one mesh node:

1. `slancha-mesh up --specialist <id> --base-port <P>` brings the backend
   up healthy (vLLM `/health` 200, or Ollama `/api/ps` shows the tag
   loaded).
2. The node's `/heartbeat` POST reports it loaded with sane `runtime_gb`
   / VRAM numbers (no OOM, no truncation).
3. A `slancha-mesh discover` from another box on the tailnet sees the
   specialist as `reachable=1` with the expected `node_url`.
4. A real OpenAI-compat `POST /v1/chat/completions` to the discovered
   `node_url` returns a coherent completion (≥1 token, no `error`).

If you've done that, edit this file (move the card from DRAFT → VALIDATED),
note the hardware you tested on, and add a short note. PRs are welcome.

## Validated

| specialist_id | engine | hardware | date | notes |
|---|---|---|---|---|
| `qwen3-coder-30b-a3b-fp8` | vllm (FP8) | DGX Spark GB10, ~49 GB GPU free, `gpu-mem-util 0.12` stand-in | 2026-05-29 | discover + e2e route through MagicDNS host worked; see `memory/mesh-sharpening-loop.md` `SERVED-SPECIALIST CAPSTONE DEMO`. The 30B-FP8 itself OOMs on consumer GB10 sm_121 due to the vLLM 0.17 cutlass FP8 gap; validated via a Qwen2.5-7B stand-in under the same served-model-name. |

## DRAFT — unvalidated

These cards' VRAM / runtime numbers are derived from upstream Ollama /
vLLM / llama.cpp specs and community benchmarks, but no slancha-mesh
bring-up has been run against them. Treat as "best-effort, please report
back" — the persistent `_validated: false` flag would require a card
schema change; today's marker is this file plus a leading DRAFT comment in
each TOML.

| specialist_id | engine | upstream | first-load box you'd expect to work |
|---|---|---|---|
| `qwen3-coder-7b-q4` | vllm | `Qwen/Qwen3-Coder-7B-Instruct` | any ≥ 8 GB VRAM CUDA + vLLM (Linux/WSL) |
| `qwen3-math-7b-q4` | vllm | `Qwen/Qwen3-Math-7B-Instruct` | any ≥ 8 GB VRAM CUDA + vLLM (Linux/WSL) |
| `llama-3.1-8b-instruct-q4` | vllm | `meta-llama/Llama-3.1-8B-Instruct` | any ≥ 8 GB VRAM CUDA + vLLM (Linux/WSL) |
| `aya-expanse-8b-q4` | vllm | `CohereLabs/aya-expanse-8b` | any ≥ 8 GB VRAM CUDA + vLLM (Linux/WSL) |
| `phi-4-14b-q4` | vllm | `microsoft/phi-4` | any ≥ 14 GB VRAM CUDA + vLLM (Linux/WSL) |
| `llama-3.1-8b-instruct-q5-ollama` | ollama | `meta-llama/Llama-3.1-8B-Instruct` | any 8 GB box w/ Ollama (Mac M-series 16+ GB, RTX 3060+, Windows + Ollama) |
| `qwen2.5-coder-7b-q4-ollama` | ollama | `Qwen/Qwen2.5-Coder-7B-Instruct` | any 6 GB box w/ Ollama (Mac M-series 16+ GB, RTX 3060, GB10) |
| `deepseek-coder-v2-16b-lite-q4-ollama` | ollama | `deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct` | 12 GB+ Ollama box |
| `phi-3.5-mini-q5-ollama` | ollama | `microsoft/Phi-3.5-mini-instruct` | tiny — Mac mini 8 GB, RTX 3060, Pi 5 + eGPU |
| `gemma-2-9b-q4-ollama` | ollama | `google/gemma-2-9b-it` | 7 GB+ Ollama box (multilingual fallback) |
| `mistral-nemo-12b-q4-ollama` | ollama | `mistralai/Mistral-Nemo-Instruct-2407` | 10 GB+ Ollama box (tools + reasoning) |

## Why the brutal honesty

A catalog that silently ships cards "as if validated" is the failure
mode the LocalLLaMA crowd will spot immediately — top comment becomes
"I tried X, it OOM'd, this is fake." The opposite mistake — refusing
to ship any card without validation — leaves the catalog too thin to
demo the actual product (heterogeneous mesh routing). The compromise:
ship the cards, mark them DRAFT, document the bring-up criterion, accept
PRs from anyone who runs the bring-up and reports back.
