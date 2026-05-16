#!/usr/bin/env bash
# Bring up TWO specialists on a single Spark GB10 (multi-specialist coexistence).
#
# Demonstrates spec §3.3 secondaries: 128GB unified memory hosts both a
# primary (default: Qwen3-Coder-30B-A3B-FP8 :8001) and a secondary
# (default: Qwen3-8B :8002) simultaneously. Each vLLM gets ~50% memory.
#
# What it does, in order:
#   1. Probe local hardware (must be GB10 or other ≥80GB-effective unified-mem node)
#   2. Cold-boot primary vLLM on :8001 in background (Marlin FP8 fallback for sm_121)
#   3. Cold-boot secondary vLLM on :8002 in background (BF16 native path)
#   4. Wait for both /health endpoints
#   5. Hit each with a real chat completion; print tok/s + response excerpt
#   6. Print aggregate memory + per-port URLs for mesh registry binding
#
# Usage:
#   bash mesh/scripts/bring-up-multi-spark.sh \
#       [PRIMARY_MODEL] [PRIMARY_PORT] [SECONDARY_MODEL] [SECONDARY_PORT]
#
# Defaults:
#   PRIMARY_MODEL   = Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8
#   PRIMARY_PORT    = 8001
#   SECONDARY_MODEL = Qwen/Qwen3-8B
#   SECONDARY_PORT  = 8002
#
# Each vLLM is launched with gpu-memory-utilization=0.4 so two can coexist
# under unified-memory pressure (combined ≤ 80% of effective RAM). Adjust
# upward if you're on a 256GB+ node or downward if other workloads share
# the GPU.

set -euo pipefail

PRIMARY_MODEL="${1:-Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8}"
PRIMARY_PORT="${2:-8001}"
SECONDARY_MODEL="${3:-Qwen/Qwen3-8B}"
SECONDARY_PORT="${4:-8002}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RUNTIME_DIR="${REPO_ROOT}/mesh/.runtime"
mkdir -p "${RUNTIME_DIR}"

step() {
  echo
  echo "=== $* ==="
}

# ----- helpers -----
launch_vllm() {
  local model="$1"
  local port="$2"
  local log="${RUNTIME_DIR}/vllm-${port}.log"
  echo "  model:  ${model}"
  echo "  port:   ${port}"
  echo "  log:    ${log}"
  # GB10 sm_121 needs Marlin fallback for FP8 GEMM; BF16 dense paths
  # don't need it but the env-var is no-op for non-FP8 weights.
  VLLM_TEST_FORCE_FP8_MARLIN=1 \
  TORCH_CUDA_ARCH_LIST=12.0 \
    nohup vllm serve "${model}" \
      --host 127.0.0.1 --port "${port}" \
      --max-model-len 8192 \
      --gpu-memory-utilization 0.4 \
      --dtype auto \
      --enforce-eager \
      --trust-remote-code \
      > "${log}" 2>&1 &
  echo "  pid:    $!"
}

wait_for_health() {
  local port="$1"
  local timeout="${2:-600}"
  local elapsed=0
  while (( elapsed < timeout )); do
    if curl -sf "http://127.0.0.1:${port}/health" >/dev/null 2>&1; then
      echo "  ✓ :${port} healthy after ${elapsed}s"
      return 0
    fi
    sleep 5
    elapsed=$((elapsed + 5))
    if (( elapsed % 30 == 0 )); then
      echo "  …waiting on :${port} (${elapsed}s)"
    fi
  done
  echo "  ✗ :${port} did NOT become healthy in ${timeout}s — check log" >&2
  return 1
}

chat_smoke() {
  local port="$1"
  local model="$2"
  local prompt="$3"
  echo "  POST /v1/chat/completions @ :${port}"
  local t0=$(date +%s.%N)
  local resp
  resp=$(curl -sf "http://127.0.0.1:${port}/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d "$(printf '{"model":"%s","messages":[{"role":"user","content":"%s"}],"max_tokens":64}' \
        "${model}" "${prompt}")" || echo "")
  local t1=$(date +%s.%N)
  local elapsed=$(awk "BEGIN {print ${t1}-${t0}}")
  if [[ -z "${resp}" ]]; then
    echo "  ✗ chat failed at :${port}" >&2
    return 1
  fi
  local tokens=$(echo "${resp}" | python3 -c 'import json,sys; d=json.load(sys.stdin); print(d["usage"]["completion_tokens"])' 2>/dev/null || echo 0)
  local content=$(echo "${resp}" | python3 -c 'import json,sys; d=json.load(sys.stdin); print(d["choices"][0]["message"]["content"][:120])' 2>/dev/null || echo "(parse-fail)")
  printf "  wall: %.2fs  tokens: %s  tok/s: %.1f\n" \
    "${elapsed}" "${tokens}" "$(awk "BEGIN {print ${tokens}/${elapsed}}")"
  echo "  excerpt: ${content}"
}

mem_snapshot() {
  echo "  RAM:"; free -h | head -2
  if command -v nvidia-smi >/dev/null 2>&1; then
    echo "  GPU:"; nvidia-smi --query-gpu=memory.used,memory.free,utilization.gpu --format=csv,noheader | head -1
  fi
}

# ----- main -----
step "[1/6] Pre-flight probe"
cd "${REPO_ROOT}"
python3 -m mesh.probe --pretty > "${RUNTIME_DIR}/probe-multi.json" 2>/dev/null || \
  uv run python -m mesh.probe --pretty > "${RUNTIME_DIR}/probe-multi.json"
chip=$(python3 -c 'import json; print(json.load(open("'"${RUNTIME_DIR}"'/probe-multi.json"))["chip"])')
ram_avail=$(python3 -c 'import json; print(json.load(open("'"${RUNTIME_DIR}"'/probe-multi.json"))["ram_available_gb"])')
echo "  chip: ${chip}"
echo "  available RAM: ${ram_avail} GB"

step "[2/6] Pre-launch memory baseline"
mem_snapshot

step "[3/6] Launch primary vLLM (background)"
launch_vllm "${PRIMARY_MODEL}" "${PRIMARY_PORT}"

step "[4/6] Launch secondary vLLM (background)"
launch_vllm "${SECONDARY_MODEL}" "${SECONDARY_PORT}"

step "[5/6] Waiting for both /health (≤10 min cold-boot each)"
wait_for_health "${PRIMARY_PORT}" 600 &
wait_for_health "${SECONDARY_PORT}" 600 &
wait

step "[6/6] Coexistence smoke + memory snapshot"
mem_snapshot
echo
echo "  --- primary chat ---"
chat_smoke "${PRIMARY_PORT}" "${PRIMARY_MODEL}" \
  "Write a Python one-liner that reverses a string."
echo
echo "  --- secondary chat ---"
chat_smoke "${SECONDARY_PORT}" "${SECONDARY_MODEL}" \
  "What is 17 squared?"

step "READY"
echo "  mesh-registry node binding:"
echo "    primary   → http://127.0.0.1:${PRIMARY_PORT}"
echo "    secondary → http://127.0.0.1:${SECONDARY_PORT}"
echo
echo "  Logs in ${RUNTIME_DIR}/vllm-${PRIMARY_PORT}.log + vllm-${SECONDARY_PORT}.log"
echo
echo "  To shut down both:"
echo "    pkill -f 'vllm serve.*:${PRIMARY_PORT}'"
echo "    pkill -f 'vllm serve.*:${SECONDARY_PORT}'"
