#!/usr/bin/env bash
# Bring up the Slancha-Mesh on a single Spark GB10.
#
# What it does, in order:
#   1. Probe the local hardware → mesh/.runtime/probe.json (sanity check).
#   2. Confirm vLLM + huggingface-cli are on PATH.
#   3. Confirm the catalog has at least one specialist whose weights are
#      already cached under ~/.cache/huggingface/. If none → instruct.
#   4. Spawn `vllm serve <model>` in the background and tail until /health.
#   5. Run a real OpenAI chat completion against it; print the response.
#   6. Print the URL the mesh registry would advertise.
#
# Usage:
#   bash mesh/scripts/bring-up-spark.sh [SPECIALIST_ID] [PORT]
# Defaults:
#   SPECIALIST_ID = qwen3-coder-30b-a3b-fp8
#   PORT          = 8001
#
# This is the v0.0.2 manual replacement for the `bring-up-mesh-node.sh`
# in spec §12 day-5. v0.0.3 will replace it with `python -m mesh.serve`
# directly, once we trust the daemon's wait-ready logic on cold-boot.

set -euo pipefail

SPECIALIST="${1:-qwen3-coder-30b-a3b-fp8}"
PORT="${2:-8001}"
# BIND_HOST is where vLLM listens. Default loopback for solo dev; set to
# 0.0.0.0 so a cloud gateway can reach this node over the tailnet.
#   BIND_HOST=0.0.0.0 bash bring-up-spark.sh ...
HOST="${BIND_HOST:-127.0.0.1}"
# ADVERTISE_HOST is what the registry hands the gateway. Auto-discovered
# from MagicDNS when on a tailnet; override to force a name.
ADVERTISE_HOST="${ADVERTISE_HOST:-$(tailscale status --json 2>/dev/null | python3 -c 'import json,sys; d=json.load(sys.stdin); print((d.get("Self",{}).get("DNSName") or "").rstrip("."))' 2>/dev/null || true)}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RUNTIME_DIR="${REPO_ROOT}/mesh/.runtime"
mkdir -p "${RUNTIME_DIR}"

step() {
  echo
  echo "=== $* ==="
}

step "[1/6] Probing local hardware"
cd "${REPO_ROOT}"
uv run python -m mesh.probe --pretty > "${RUNTIME_DIR}/probe.json"
python3 -c "
import json, sys
d = json.load(open('${RUNTIME_DIR}/probe.json'))
print(f\"  chip:    {d['chip']}\")
print(f\"  arch:    {d['arch']}\")
print(f\"  CUDA cap:{d['cuda_capability']}\")
print(f\"  RAM:     {d['ram_total_gb']:.0f} GB total, {d['ram_available_gb']:.0f} GB free\")
print(f\"  unified: {d['unified_memory']}\")
print(f\"  backends:{d['available_backends']}\")
for w in d['probe_warnings']:
    print(f\"  WARN:    {w}\")
"

step "[2/6] Checking tools"
command -v vllm >/dev/null || { echo "vllm not on PATH; install via 'pip install vllm'"; exit 1; }
command -v huggingface-cli >/dev/null || command -v hf >/dev/null || \
  { echo "hf cli missing; install via 'pip install huggingface_hub[cli]'"; exit 1; }
echo "  vllm:  $(vllm --version)"
echo "  cuda:  $(nvidia-smi --query-gpu=driver_version,compute_cap --format=csv,noheader,nounits | head -1)"

step "[3/6] Resolving specialist card"
MODEL_ID=$(python3 -c "
from mesh.catalog import load_catalog
by_id = {c.specialist_id: c for c in load_catalog()}
card = by_id.get('${SPECIALIST}')
if card is None:
    raise SystemExit(f\"unknown specialist '${SPECIALIST}'\")
print(card.model_id)
")
echo "  model_id:    ${MODEL_ID}"

# Sniff the HF cache. Don't trigger a download here — that's the user's call.
SAFE_DIR_NAME="models--$(echo "${MODEL_ID}" | sed 's|/|--|g')"
HF_DIR="${HOME}/.cache/huggingface/hub/${SAFE_DIR_NAME}"
if ls "${HF_DIR}"/snapshots/*/*.safetensors >/dev/null 2>&1; then
  echo "  weights:    cached locally (${HF_DIR})"
else
  echo "  weights:    NOT cached. Run: huggingface-cli download ${MODEL_ID}"
  echo "  (vllm will trigger the download on first launch, expect 5-30 min)"
fi

step "[4/6] Starting vLLM serve"
LOG="${RUNTIME_DIR}/vllm-${SPECIALIST}.log"
if ss -ltn 2>/dev/null | awk '{print $4}' | grep -q ":${PORT}\$"; then
  echo "  port ${PORT} already bound; will adopt the existing process"
else
  echo "  launching vllm serve ${MODEL_ID} on ${HOST}:${PORT}"
  echo "  log: ${LOG}"
  TORCH_CUDA_ARCH_LIST=12.0 \
  nohup vllm serve "${MODEL_ID}" \
    --host "${HOST}" \
    --port "${PORT}" \
    --served-model-name "${SPECIALIST}" \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.55 \
    --dtype auto \
    --trust-remote-code \
    --enforce-eager \
    > "${LOG}" 2>&1 &
  echo "  pid: $!"
fi

step "[5/6] Waiting for /health (up to 10 min)"
for i in $(seq 1 300); do
  if curl -sf "http://${HOST}:${PORT}/health" >/dev/null 2>&1; then
    echo "  ready after ${i} x 2s polls"
    break
  fi
  if [ "$i" -eq 300 ]; then
    echo "  TIMEOUT — tail of log:"
    tail -30 "${LOG}"
    exit 1
  fi
  sleep 2
done

step "[6/6] Smoke chat completion + tok/s measurement"
python3 - "${HOST}" "${PORT}" "${SPECIALIST}" <<'PYEOF'
import json, sys, time, urllib.request

host, port, spec = sys.argv[1], sys.argv[2], sys.argv[3]
url = f"http://{host}:{port}/v1/chat/completions"
payload = {
    "model": spec,
    "messages": [
        {"role": "user", "content": "Write a Python one-liner that reverses a string."},
    ],
    "max_tokens": 80,
    "temperature": 0.2,
}

t0 = time.time()
req = urllib.request.Request(
    url,
    data=json.dumps(payload).encode("utf-8"),
    headers={"Content-Type": "application/json"},
)
with urllib.request.urlopen(req, timeout=120) as resp:
    body = json.loads(resp.read())
elapsed = time.time() - t0

content = body["choices"][0]["message"]["content"]
usage = body.get("usage", {})
out_tokens = usage.get("completion_tokens", len(content.split()))
tps = out_tokens / elapsed if elapsed > 0 else 0

print(f"  ✓ response in {elapsed:.2f}s, {out_tokens} completion tokens, {tps:.1f} tok/s")
print(f"  ---")
print(f"  {content[:200]}")
print(f"  ---")
PYEOF

echo
if [ -n "${ADVERTISE_HOST}" ]; then
  echo "  node_url to advertise (tailnet): http://${ADVERTISE_HOST}:${PORT}"
else
  echo "  node_url to advertise (local):   http://${HOST}:${PORT}"
  echo "  (no MagicDNS name found — on a tailnet, join as tag:specialist and"
  echo "   re-run, or pass ADVERTISE_HOST=<magicdns-name>)"
fi

echo
echo "DONE. The mesh can now register this node:"
if [ -n "${ADVERTISE_HOST}" ]; then
  echo "  uv run python -m mesh.serve --tailnet --specialist ${SPECIALIST} --base-port ${PORT}"
else
  echo "  uv run python -m mesh.serve --specialist ${SPECIALIST} --base-port ${PORT}"
fi
