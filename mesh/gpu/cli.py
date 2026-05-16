"""`mesh-gpu` CLI — local + cluster GPU coordination.

Commands:
  mesh-gpu status         — local nvidia-smi + active reservations
  mesh-gpu reserve --gb N --duration H [--purpose TXT] [--pid PID]
  mesh-gpu release ID
  mesh-gpu wait --gb N [--timeout 30m]
  mesh-gpu cluster-status [--registry URL]
  mesh-gpu cluster-reserve --gb N --duration H [--on auto|NODE_ID]

`cluster-*` subcommands talk to a slancha-mesh registry. URL via
--registry or SLANCHA_MESH_REGISTRY_URL env. Token via
SLANCHA_NODE_TOKEN env (sent as Bearer).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import httpx

from mesh.gpu.cluster import build_cluster_view_from_heartbeats, pick_best_node
from mesh.gpu.probe import probe_gpu
from mesh.gpu.reservations import ReservationStore


def _fmt_gb(v: Optional[float]) -> str:
    return f"{v:.1f}" if v is not None else "n/a"


def _fmt_pct(v: Optional[float]) -> str:
    return f"{v:.0f}%" if v is not None else "n/a"


def _parse_duration(s: str) -> float:
    """Accept '30m', '1h', '2h30m', '120s', or plain seconds."""
    s = s.strip().lower()
    if s.isdigit():
        return float(s)
    total = 0.0
    cur = ""
    for ch in s:
        if ch.isdigit() or ch == ".":
            cur += ch
        elif ch in ("h", "m", "s"):
            if not cur:
                raise ValueError(f"bad duration: {s}")
            val = float(cur)
            cur = ""
            total += val * {"h": 3600, "m": 60, "s": 1}[ch]
        else:
            raise ValueError(f"bad duration char {ch!r} in {s}")
    if cur:  # trailing digits without unit → seconds
        total += float(cur)
    return total


# ---------------------------------------------------------------------------
# Local commands
# ---------------------------------------------------------------------------


def cmd_status(args: argparse.Namespace) -> int:
    snap = probe_gpu()
    store = ReservationStore()
    print(f"# GPU status @ {snap.probed_at.isoformat()}")
    if not snap.nvidia_smi_available:
        print("  nvidia-smi: NOT AVAILABLE (non-CUDA host?)")
        return 0
    print(f"  util:     {_fmt_pct(snap.util_pct)}")
    print(
        f"  mem:      used={_fmt_gb(snap.mem_used_gb)} GB / "
        f"free={_fmt_gb(snap.mem_free_gb)} GB / "
        f"total={_fmt_gb((snap.mem_total_mib or 0)/1024 if snap.mem_total_mib else None)} GB"
    )
    print(f"  procs (by GPU mem):")
    if not snap.processes:
        print("    (none)")
    for p in sorted(snap.processes, key=lambda x: x.used_memory_mib, reverse=True):
        user = p.user or "?"
        rt = f"{p.runtime_s}s" if p.runtime_s is not None else "?"
        cmd = (p.cmdline or p.process_name)[:80]
        print(
            f"    pid {p.pid:>7d}  {user:<8s}  {p.used_memory_mib/1024:5.1f} GB  "
            f"{rt:>10s}  {cmd}"
        )
    print(f"  reservations (cooperative, file-based):")
    active = store.list_active()
    if not active:
        print("    (none)")
    for r in active:
        print(
            f"    {r.reservation_id}  {r.user:<8s}  {r.gb_requested:5.1f} GB  "
            f"remaining {r.remaining_s/60:.1f} min  pid={r.pid or 'n/a'}  "
            f"{r.purpose}"
        )
    if active:
        print(f"  total reserved: {store.total_reserved_gb():.1f} GB")
    return 0


def cmd_reserve(args: argparse.Namespace) -> int:
    store = ReservationStore()
    duration_s = _parse_duration(args.duration)
    rid = store.reserve(
        gb_requested=args.gb,
        duration_s=duration_s,
        purpose=args.purpose or "",
        pid=args.pid,
    )
    print(f"reserved {args.gb:.1f} GB for {duration_s/60:.1f} min (id={rid})")
    return 0


def cmd_release(args: argparse.Namespace) -> int:
    store = ReservationStore()
    ok = store.release(args.id)
    if ok:
        print(f"released {args.id}")
        return 0
    print(f"no such reservation {args.id}", file=sys.stderr)
    return 1


def cmd_wait(args: argparse.Namespace) -> int:
    """Block until `gb` is free locally, then optionally auto-reserve."""
    store = ReservationStore()
    timeout_s = _parse_duration(args.timeout)
    deadline = time.time() + timeout_s
    poll_s = max(2.0, min(10.0, timeout_s / 30))
    while time.time() < deadline:
        snap = probe_gpu()
        if not snap.nvidia_smi_available:
            print("nvidia-smi not available; cannot wait", file=sys.stderr)
            return 2
        total = (snap.mem_total_mib or 0) / 1024
        used = snap.total_proc_memory_gb
        reserved = store.total_reserved_gb()
        free = max(0.0, (total or 0.0) - max(used, reserved))
        if total and free >= args.gb:
            print(
                f"OK: {free:.1f} GB free (total={total:.1f}, "
                f"used={used:.1f}, reserved={reserved:.1f})"
            )
            if args.auto_reserve:
                rid = store.reserve(
                    gb_requested=args.gb,
                    duration_s=_parse_duration(args.auto_reserve),
                    purpose=args.purpose or "wait --auto-reserve",
                )
                print(f"auto-reserved id={rid}")
            return 0
        time.sleep(poll_s)
    print(f"TIMEOUT: never saw {args.gb} GB free", file=sys.stderr)
    return 1


# ---------------------------------------------------------------------------
# Cluster commands — talk to mesh registry
# ---------------------------------------------------------------------------


def _registry_url(args: argparse.Namespace) -> Optional[str]:
    return args.registry or os.environ.get("SLANCHA_MESH_REGISTRY_URL")


def _registry_get(url: str, path: str) -> Optional[dict]:
    headers = {}
    token = os.environ.get("SLANCHA_NODE_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    try:
        r = httpx.get(f"{url.rstrip('/')}{path}", headers=headers, timeout=5.0)
        if r.status_code != 200:
            print(f"registry {path} returned {r.status_code}", file=sys.stderr)
            return None
        return r.json()
    except httpx.HTTPError as exc:
        print(f"registry unreachable: {exc}", file=sys.stderr)
        return None


def cmd_cluster_status(args: argparse.Namespace) -> int:
    url = _registry_url(args)
    if not url:
        print("no registry URL; set SLANCHA_MESH_REGISTRY_URL or --registry", file=sys.stderr)
        return 2
    # Try /gpu/cluster first (planned endpoint), then fall back to
    # building the view from /registry's raw heartbeats.
    body = _registry_get(url, "/gpu/cluster")
    if body is None:
        body = _registry_get(url, "/registry")
        if body is None:
            return 2
        snap = body.get("snapshot", {})
        # The /registry shape doesn't currently embed gpu payloads
        # (registry-side aggregation is v0.0.7); we just report node
        # names + suggest installing the registry GPU endpoint.
        nodes = snap.get("nodes", {})
        if not nodes:
            print("registry has no nodes")
            return 0
        print(f"# cluster status (no /gpu/cluster endpoint; falling back to /registry)")
        for nid, n in nodes.items():
            print(f"  {nid}  health={n.get('health')}  "
                  f"loaded={n.get('loaded_specialist_ids')}  "
                  f"node_url={n.get('node_url')}")
        print("(install registry GPU aggregation for richer view)")
        return 0
    # Real /gpu/cluster response
    print(f"# cluster GPU view @ {body.get('snapshot_ts', '?')}")
    nodes = body.get("nodes", {})
    if not nodes:
        print("  no GPU-reporting nodes in cluster")
        return 0
    total_free = sum(n.get("free_gb_after_reservations") or 0 for n in nodes.values())
    total_used = sum(n.get("used_gb") or 0 for n in nodes.values())
    total_reserved = sum(n.get("reserved_gb") or 0 for n in nodes.values())
    print(f"  CLUSTER: free={total_free:.1f} GB  used={total_used:.1f} GB  reserved={total_reserved:.1f} GB")
    for nid, n in nodes.items():
        print(
            f"  {n.get('friendly_name', nid):<20s}  "
            f"free={_fmt_gb(n.get('free_gb_after_reservations'))} GB  "
            f"used={_fmt_gb(n.get('used_gb'))} GB  "
            f"reserved={_fmt_gb(n.get('reserved_gb'))} GB  "
            f"tags={','.join(n.get('hardware_tags', []))}"
        )
    return 0


def cmd_cluster_reserve(args: argparse.Namespace) -> int:
    url = _registry_url(args)
    if not url:
        print("no registry URL; set SLANCHA_MESH_REGISTRY_URL or --registry", file=sys.stderr)
        return 2
    # Pull cluster view → pick best node locally → POST reservation to that
    # node's local store (when registry exposes /gpu/reserve). For v0.0.6
    # this is informational; the actual reservation is written to the
    # local node via mesh service /gpu/reserve in v0.0.7.
    body = _registry_get(url, "/gpu/cluster")
    if body is None:
        print("cluster view unavailable; cannot place", file=sys.stderr)
        return 2
    heartbeats = list(body.get("nodes", {}).values())
    view = build_cluster_view_from_heartbeats(
        [{"node_id": nid, "gpu": n, "friendly_name": n.get("friendly_name")}
         for nid, n in body.get("nodes", {}).items()]
    )
    require_tags = args.require_tags.split(",") if args.require_tags else None
    if args.on == "auto":
        result = pick_best_node(
            view, gb_requested=args.gb, require_hardware_tags=require_tags,
        )
        if not result.ok:
            print(f"NO PLACEMENT: {result.reason}", file=sys.stderr)
            for nid, why in result.rejected.items():
                print(f"  reject {nid}: {why}", file=sys.stderr)
            return 1
        print(f"chose: {result.reason}")
        chosen = result.chosen_node_id
    else:
        chosen = args.on
        if chosen not in view.nodes:
            print(f"node {chosen} not in cluster view", file=sys.stderr)
            return 1
        print(f"forced placement on {chosen}")
    # Try POSTing to mesh service /gpu/reserve (v0.0.7+); fall back to
    # printing the directive when endpoint absent.
    payload = {
        "node_id": chosen,
        "gb_requested": args.gb,
        "duration_s": _parse_duration(args.duration),
        "purpose": args.purpose or "",
    }
    headers = {}
    token = os.environ.get("SLANCHA_NODE_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    try:
        r = httpx.post(
            f"{url.rstrip('/')}/gpu/reserve",
            json=payload, headers=headers, timeout=5.0,
        )
        if r.status_code == 200:
            print(f"reservation created: {r.json()}")
            return 0
        print(f"registry /gpu/reserve returned {r.status_code}; "
              f"this is fine if you're on registry pre-v0.0.7. "
              f"Manual fallback: SSH to {chosen} + run `mesh-gpu reserve "
              f"--gb {args.gb} --duration {args.duration}"
              f"{' --purpose '+args.purpose if args.purpose else ''}`")
        return 0
    except httpx.HTTPError as exc:
        print(f"reserve POST failed: {exc}", file=sys.stderr)
        return 2


# ---------------------------------------------------------------------------
# Entry
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="mesh-gpu", description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_status = sub.add_parser("status", help="local nvidia-smi + reservations")
    p_status.set_defaults(func=cmd_status)

    p_reserve = sub.add_parser("reserve", help="claim local GPU memory")
    p_reserve.add_argument("--gb", type=float, required=True)
    p_reserve.add_argument("--duration", type=str, required=True, help="e.g. 30m, 1h, 2h30m")
    p_reserve.add_argument("--purpose", type=str, default="")
    p_reserve.add_argument("--pid", type=int, help="auto-prune when this pid dies")
    p_reserve.set_defaults(func=cmd_reserve)

    p_release = sub.add_parser("release", help="release a reservation by id")
    p_release.add_argument("id")
    p_release.set_defaults(func=cmd_release)

    p_wait = sub.add_parser("wait", help="block until N GB is free")
    p_wait.add_argument("--gb", type=float, required=True)
    p_wait.add_argument("--timeout", type=str, default="30m")
    p_wait.add_argument("--auto-reserve", type=str,
                        help="if set (duration), reserve as soon as GB is free")
    p_wait.add_argument("--purpose", type=str, default="")
    p_wait.set_defaults(func=cmd_wait)

    p_cstatus = sub.add_parser("cluster-status", help="cluster-wide GPU view via registry")
    p_cstatus.add_argument("--registry", type=str)
    p_cstatus.set_defaults(func=cmd_cluster_status)

    p_creserve = sub.add_parser("cluster-reserve", help="place workload on best-fit node")
    p_creserve.add_argument("--gb", type=float, required=True)
    p_creserve.add_argument("--duration", type=str, required=True)
    p_creserve.add_argument("--on", type=str, default="auto", help="auto | NODE_ID")
    p_creserve.add_argument("--purpose", type=str, default="")
    p_creserve.add_argument("--require-tags", type=str, default="",
                            help="comma-separated hardware tags required")
    p_creserve.add_argument("--registry", type=str)
    p_creserve.set_defaults(func=cmd_cluster_reserve)

    return ap


def main(argv: Optional[list[str]] = None) -> int:
    ap = build_parser()
    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
