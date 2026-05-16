#!/usr/bin/env bash
# install_dashboard.sh — install the mesh dashboard + cloudflared uplink.
#
# Usage (from a Linux host with systemd + slancha-mesh installed):
#   ./mesh/deploy/install_dashboard.sh [--port 8501]
#
# What it does:
#   1. Sanity-checks streamlit + cloudflared are on PATH; surfaces install
#      hints if not (does NOT install them automatically — explicit acts).
#   2. Verifies the cloudflared `mesh-dashboard` named tunnel exists. If
#      not, prints the one-time provisioning commands and exits 0 (the
#      operator runs them, re-runs this script).
#   3. Copies mesh-dashboard.service + mesh-dashboard-tunnel.service to
#      ~/.config/systemd/user/.
#   4. Reloads + enable --now's BOTH units (tunnel requires dashboard).
#   5. Prints `systemctl --user status` for verification.
#
# Idempotent: re-running re-copies units and re-enables (no-op if already on).
#
# Uninstall:
#   systemctl --user disable --now mesh-dashboard-tunnel.service mesh-dashboard.service
#   rm ~/.config/systemd/user/mesh-dashboard{,-tunnel}.service
#   systemctl --user daemon-reload

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
USER_UNIT_DIR="${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user"
TUNNEL_NAME="mesh-dashboard"
PORT="${MESH_DASHBOARD_PORT:-8501}"

# Parse args
while [[ $# -gt 0 ]]; do
    case "$1" in
        --port) PORT="$2"; shift 2 ;;
        --port=*) PORT="${1#*=}"; shift ;;
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

# ───── Sanity: streamlit ─────
if ! command -v streamlit >/dev/null 2>&1; then
    cat >&2 <<EOF
error: streamlit not found on PATH.
       The dashboard service requires the [dashboard] extra:
         cd ~/Source/slancha-mesh
         pip install -e '.[dashboard]'
       Or system-install: pipx install streamlit
       Then re-run this script.
EOF
    exit 1
fi

# ───── Sanity: cloudflared ─────
CLOUDFLARED="${CLOUDFLARED:-$HOME/.local/bin/cloudflared}"
if [[ ! -x "$CLOUDFLARED" ]] && ! command -v cloudflared >/dev/null 2>&1; then
    cat >&2 <<EOF
error: cloudflared not found at ~/.local/bin/cloudflared or on PATH.
       Install per global CLAUDE.md uplink protocol:
         curl -L -o ~/.local/bin/cloudflared \\
           https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64
         chmod +x ~/.local/bin/cloudflared
         ~/.local/bin/cloudflared tunnel login
       Then re-run this script.
EOF
    exit 1
fi
# Use the PATH version if ~/.local/bin variant absent
[[ ! -x "$CLOUDFLARED" ]] && CLOUDFLARED="$(command -v cloudflared)"

# ───── Sanity: named tunnel exists ─────
if ! "$CLOUDFLARED" tunnel list 2>/dev/null | grep -q "$TUNNEL_NAME"; then
    cat >&2 <<EOF
note: named tunnel "$TUNNEL_NAME" not found.
      One-time provisioning (operator runs these):
        $CLOUDFLARED tunnel create $TUNNEL_NAME
        $CLOUDFLARED tunnel route dns $TUNNEL_NAME evals.example.com
        cp $SCRIPT_DIR/mesh-dashboard-tunnel-config.yml.example \\
           ~/.cloudflared/mesh-dashboard-config.yml
        # edit the UUID + credentials-file paths in that yml
      Then re-run this installer.
EOF
    exit 0
fi

# ───── Install units ─────
mkdir -p "$USER_UNIT_DIR"
for unit in mesh-dashboard.service mesh-dashboard-tunnel.service; do
    src="$SCRIPT_DIR/$unit"
    dst="$USER_UNIT_DIR/$unit"
    if [[ ! -f "$src" ]]; then
        echo "error: missing unit file $src" >&2
        exit 1
    fi
    cp "$src" "$dst"
    echo "installed: $dst"
done

systemctl --user daemon-reload
# Enable + start in dependency order. tunnel Requires= dashboard so a
# single enable --now on the tunnel pulls dashboard automatically, but
# enabling both explicitly is clearer in `systemctl --user list-unit-files`.
systemctl --user enable --now mesh-dashboard.service
systemctl --user enable --now mesh-dashboard-tunnel.service

echo
echo "─── unit status ────────────────────────────────────────────────────"
systemctl --user --no-pager status mesh-dashboard.service mesh-dashboard-tunnel.service || true
echo
echo "─── port check (should show streamlit on 127.0.0.1:$PORT) ──────────"
ss -tlnp 2>/dev/null | grep ":$PORT" || echo "(port $PORT not yet bound; streamlit may still be starting)"
echo
echo "Done. Dashboard should appear at https://evals.example.com within"
echo "30s. First-boot Cloudflare Universal SSL provisioning can take up"
echo "to 15min — poll evals.example.com/_stcore/health until 200."
echo
echo "Logs:"
echo "  journalctl --user -u mesh-dashboard.service -f"
echo "  journalctl --user -u mesh-dashboard-tunnel.service -f"
