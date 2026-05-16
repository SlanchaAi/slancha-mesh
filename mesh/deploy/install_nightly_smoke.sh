#!/usr/bin/env bash
# install_nightly_smoke.sh — install the nightly smoke timer under user systemd.
#
# Usage (from a host with systemd + slancha-mesh installed):
#   ./mesh/deploy/install_nightly_smoke.sh
#
# What it does:
#   1. Verifies systemctl --user works on this host.
#   2. Copies mesh-nightly-smoke.{service,timer} to ~/.config/systemd/user/.
#   3. Reloads the user unit registry.
#   4. enable --now's the .timer so it fires at the next 03:00 UTC.
#   5. Prints `systemctl --user status` + `list-timers` for verification.
#
# Idempotent: re-running re-copies the units and re-runs daemon-reload,
# which is what you want if you've edited the unit files.
#
# Uninstall:
#   systemctl --user disable --now mesh-nightly-smoke.timer
#   rm ~/.config/systemd/user/mesh-nightly-smoke.{service,timer}
#   systemctl --user daemon-reload

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
USER_UNIT_DIR="${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user"

if ! command -v systemctl >/dev/null 2>&1; then
    echo "error: systemctl not found. This host doesn't have systemd; the nightly" >&2
    echo "       smoke is only supported on Linux hosts with user systemd." >&2
    exit 1
fi

# Quick health check: systemctl --user fails clearly when run as root w/o
# linger configured, or in some headless setups. Surface the error early.
if ! systemctl --user list-units --type=service >/dev/null 2>&1; then
    echo "error: 'systemctl --user' isn't available in this session." >&2
    echo "       If running under sudo / a service account, ensure the user has" >&2
    echo "       a logind session: 'loginctl enable-linger \$USER' may help." >&2
    exit 1
fi

mkdir -p "$USER_UNIT_DIR"

for unit in mesh-nightly-smoke.service mesh-nightly-smoke.timer; do
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
systemctl --user enable --now mesh-nightly-smoke.timer

echo
echo "─── timer status ───────────────────────────────────────────────────"
systemctl --user list-timers mesh-nightly-smoke.timer --no-pager || true
echo
echo "─── service status (next fire) ─────────────────────────────────────"
systemctl --user status mesh-nightly-smoke.timer --no-pager || true
echo
echo "Done. The smoke will fire at the next scheduled 03:00 UTC."
echo "Manual trigger (for verification):"
echo "  systemctl --user start mesh-nightly-smoke.service"
echo "  journalctl --user -u mesh-nightly-smoke.service -f"
