#!/usr/bin/env bash
# install_mesh_registry.sh — install the mesh registry HTTP service.
#
# Usage (from a Linux host with systemd + slancha-mesh installed):
#   ./mesh/deploy/install_mesh_registry.sh [--port 8088] [--bind 127.0.0.1]
#
# What it does:
#   1. Sanity-checks uvicorn is importable from %h/Source/slancha-mesh.
#   2. Copies mesh-registry.service into ~/.config/systemd/user/.
#   3. Reload + enable --now's the unit; the registry binds to
#      MESH_REGISTRY_HOST:MESH_REGISTRY_PORT (defaults 127.0.0.1:8088).
#   4. Prints `systemctl --user status` + curl /health for verification.
#
# Idempotent: re-running re-copies + re-enables (no-op if already on).
#
# Uninstall:
#   systemctl --user disable --now mesh-registry.service
#   rm ~/.config/systemd/user/mesh-registry.service
#   systemctl --user daemon-reload

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
USER_UNIT_DIR="${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user"
PORT="${MESH_REGISTRY_PORT:-8088}"
HOST="${MESH_REGISTRY_HOST:-127.0.0.1}"

# Parse args
while [[ $# -gt 0 ]]; do
    case "$1" in
        --port) PORT="$2"; shift 2 ;;
        --port=*) PORT="${1#*=}"; shift ;;
        --bind) HOST="$2"; shift 2 ;;
        --bind=*) HOST="${1#*=}"; shift ;;
        -h|--help)
            sed -n '/^# Usage:/,/^[^#]/p' "$0" | sed 's/^# \{0,1\}//' | head -n -1
            exit 0
            ;;
        *) echo "unknown flag: $1" >&2; exit 2 ;;
    esac
done

# ───── Sanity: systemd available ─────
if ! command -v systemctl >/dev/null 2>&1; then
    echo "error: systemctl not found. This installer is Linux/systemd-only." >&2
    exit 1
fi
if ! systemctl --user list-units --type=service >/dev/null 2>&1; then
    echo "error: 'systemctl --user' isn't available in this session." >&2
    echo "       'loginctl enable-linger \$USER' may help if running headless." >&2
    exit 1
fi

# ───── Sanity: uvicorn importable from the source checkout ─────
# We test-launch a `python3 -c 'import uvicorn; from mesh.service import app'`
# against the same PYTHONPATH the unit will use, so a missing dep surfaces
# during install rather than at first boot.
SOURCE_DIR="$HOME/Source/slancha-mesh"
if [[ ! -d "$SOURCE_DIR" ]]; then
    echo "error: source checkout not found at $SOURCE_DIR" >&2
    echo "       The unit assumes the standard layout; adjust paths if elsewhere." >&2
    exit 1
fi
if ! PYTHONPATH="$SOURCE_DIR" python3 -c \
    'import uvicorn; from mesh.service import app' >/dev/null 2>&1; then
    cat >&2 <<EOF
error: uvicorn + mesh.service can't be imported from $SOURCE_DIR.
       Common fixes:
         pip install -e '$SOURCE_DIR'
         # or for non-editable:
         pip install fastapi uvicorn pydantic httpx
       Then re-run this script.
EOF
    exit 1
fi

# ───── Install unit ─────
src="$SCRIPT_DIR/mesh-registry.service"
dst="$USER_UNIT_DIR/mesh-registry.service"
mkdir -p "$USER_UNIT_DIR"
if [[ ! -f "$src" ]]; then
    echo "error: missing unit file $src" >&2
    exit 1
fi
cp "$src" "$dst"
echo "installed: $dst"

# Apply env overrides via a drop-in if --port/--bind changed defaults.
if [[ "$PORT" != "8088" || "$HOST" != "127.0.0.1" ]]; then
    DROPIN_DIR="$USER_UNIT_DIR/mesh-registry.service.d"
    mkdir -p "$DROPIN_DIR"
    cat > "$DROPIN_DIR/override.conf" <<EOF
[Service]
Environment="MESH_REGISTRY_HOST=$HOST"
Environment="MESH_REGISTRY_PORT=$PORT"
EOF
    echo "installed override: $DROPIN_DIR/override.conf"
fi

systemctl --user daemon-reload
systemctl --user enable --now mesh-registry.service

# Give uvicorn a beat to bind before probing /health.
sleep 1

echo
echo "─── unit status ────────────────────────────────────────────────────"
systemctl --user --no-pager status mesh-registry.service || true
echo
echo "─── /health probe ──────────────────────────────────────────────────"
if curl -fsS "http://${HOST}:${PORT}/health" 2>&1; then
    echo
    echo "  registry is responding."
else
    echo
    echo "  /health did not respond on http://${HOST}:${PORT}. Check:"
    echo "    journalctl --user -u mesh-registry.service -f"
fi
echo
echo "Done. The mesh registry is reachable at http://${HOST}:${PORT}."
echo "Configure consumers:"
echo "  export SLANCHA_MESH_REGISTRY_URL=http://${HOST}:${PORT}"
echo "  # slancha-local (proxy) + nightly-smoke + dashboard all read this env"
