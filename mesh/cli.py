"""`slancha-mesh` — one command to bring a specialist node up, and to
discover the mesh from any consumer.

The headline is `slancha-mesh up`: idempotently join the tailnet (tagged
`tag:specialist`), serve the chosen specialists bound to the tailnet, and
expose a live, pull-able self-description on the node-info port. The gateway
(or anyone) runs `slancha-mesh discover` to walk the tailnet and build a
routing table — no heartbeat-push, no central registry, no shared write
token. Tailnet membership + the ACL is the credential.

Design + rationale: docs/MESH_ONELINE_SETUP_PROPOSAL_2026_05_25.md
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import replace
from pathlib import Path

from mesh.discovery import (
    DEFAULT_NODE_INFO_PORT,
    discover_specialists,
    make_http_fetch,
    parse_specialist_peers,
)
from mesh.tailnet import (
    DEFAULT_SPECIALIST_TAG,
    TailnetConfig,
    ensure_joined,
    tailnet_status,
)

NODE_TOKEN_ENV = "SLANCHA_NODE_TOKEN"


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _tailnet_from_args(args: argparse.Namespace) -> TailnetConfig:
    """Build a TailnetConfig from env defaults + CLI overrides.

    Tailnet is enabled when any tailnet-shaped flag is present (a key, an
    explicit advertise host, or --tailnet), else falls back to the
    SLANCHA_TAILNET_* env defaults.
    """
    cfg = TailnetConfig.from_env()
    want = bool(getattr(args, "key", None) or getattr(args, "advertise_host", None) or getattr(args, "tailnet", False))
    if want:
        cfg = replace(
            cfg,
            enabled=True,
            advertise_host=getattr(args, "advertise_host", None) or cfg.advertise_host,
            control_plane=getattr(args, "control_plane", None) or cfg.control_plane,
            login_server=getattr(args, "login_server", None) or cfg.login_server,
        )
    return cfg


def _print(msg: str) -> None:
    print(msg, flush=True)


def auto_select(catalog, probe, strategy: str = "best_per_machine") -> list[str]:
    """Pick the best-fit specialist for this box via the cluster allocator.

    Single-node allocation: `best_per_machine` ranks the catalog against the
    probed hardware (memory/backend hard-filters + tps score) and returns the
    top fit. Empty when nothing in the catalog fits (e.g. every model needs
    more VRAM than the box has) — the node then runs hardware-only.
    """
    from mesh.allocator import allocate_cluster

    suggestions = allocate_cluster([probe], catalog, strategy=strategy)  # type: ignore[arg-type]
    sugg = suggestions.get(probe.node_id)
    return [sugg.primary.specialist_id] if sugg and sugg.primary else []


# ---------------------------------------------------------------------------
# up — join + serve + expose self-description
# ---------------------------------------------------------------------------


def cmd_up(args: argparse.Namespace) -> int:
    from mesh.node_server import build_node

    tailnet = _tailnet_from_args(args)

    # Pull discovery relies on the tailnet ACL as the gate; a stray node
    # token makes /models require Bearer, so `discover` silently sees the
    # node as unreachable unless consumers pass the same token. Warn loudly.
    if os.environ.get(NODE_TOKEN_ENV):
        _print(f"[warn] {NODE_TOKEN_ENV} is set: the node-info endpoint will require it, "
               "so consumers running `slancha-mesh discover` must pass --token (or the "
               "same env). In the tailnet-ACL trust model you can leave it unset.")

    # 1. Ensure tailnet membership (idempotent). Disabled config → skip
    #    (loopback dev mode), which is fine for a single-box smoke test.
    if tailnet.enabled:
        res = ensure_joined(tailnet, auth_key=args.key)
        _print(f"[tailnet] {res.message}")
        if not (res.joined or res.already):
            _print("[tailnet] cannot continue — node is not a reachable specialist.")
            return 2
        # Pin the resolved MagicDNS host so backends advertise it.
        if res.host and not tailnet.advertise_host:
            tailnet = replace(tailnet, advertise_host=res.host)

    # Establish a router (slancha-local) if asked. A specialist node can't
    # route; a home mesh needs a router somewhere. --with-router installs +
    # launches one when none is reachable (decision: founder, 2026-05-25).
    if args.with_router and not args.dry_run:
        from mesh.router_bootstrap import ensure_router

        status_json = tailnet_status(TailnetConfig(enabled=True)) if tailnet.enabled else None
        router = ensure_router(install=True, install_spec=args.router_spec, status_json=status_json)
        _print(f"[router] {router.get('action')}: "
               f"{router.get('reason') or router.get('message') or router.get('error') or ''}")
        for step in router.get("next_steps", []):
            _print(f"[router]   - {step.get('command') or step.get('message')}")

    # 2. Resolve which specialists to serve. Explicit --specialist wins;
    #    --auto fits the best specialist to THIS box's probed hardware so
    #    the operator doesn't need to know catalog ids.
    from mesh.catalog import load_catalog

    catalog = load_catalog(Path(args.catalog_dir)) if args.catalog_dir else load_catalog()
    specialist_ids = list(args.specialist)
    if args.auto and not specialist_ids:
        from mesh.probe import probe_node

        specialist_ids = auto_select(catalog, probe_node())
        _print(f"[auto] hardware fit → {specialist_ids or '(nothing in the catalog fits this box)'}")

    # 3. Build the node (daemon + self-description app share one registry).
    node = build_node(
        specialist_ids=specialist_ids,
        tailnet=tailnet,
        catalog=catalog,
        base_port=args.base_port,
    )

    advertise = node.daemon.advertise_host
    _print(f"[up] specialists={specialist_ids or '(hardware-only heartbeat)'} "
           f"base_port={args.base_port} node_info_port={args.node_info_port}")
    _print(f"[up] advertise_host={advertise or '(loopback — not on a tailnet)'}")
    for be in node.daemon.backends:
        _print(f"[up]   {be.card.specialist_id} -> bind {be.base_url}")

    if args.dry_run:
        _print("[up] --dry-run: not starting backends or the server.")
        return 0

    node.run(
        node_info_host=args.node_info_host,
        node_info_port=args.node_info_port,
        ready_timeout=args.ready_timeout,
    )
    return 0


# ---------------------------------------------------------------------------
# plan — agent-facing: what should THIS box run, and is there a mesh already?
# ---------------------------------------------------------------------------


def build_plan(catalog_dir: str | None = None, *, specialist_tag: str = DEFAULT_SPECIALIST_TAG) -> dict:
    """Assemble the onboarding decision an agent needs, as a plain dict.

    Hardware probe → recommended engine (hardware-fit) + recommended specialist
    (catalog-fit) + cluster state (first node vs joining) + next steps. Pure-ish
    (probes hardware + reads the local tailnet); no mutation. JSON-serializable.
    """
    from mesh.allocator import allocate_cluster
    from mesh.catalog import load_catalog
    from mesh.engine_select import recommend_engine
    from mesh.probe import probe_node

    probe = probe_node()
    engine = recommend_engine(probe)
    catalog = load_catalog(Path(catalog_dir)) if catalog_dir else load_catalog()

    sugg = allocate_cluster([probe], catalog, strategy="best_per_machine").get(probe.node_id)
    recommended = sugg.primary.specialist_id if sugg and sugg.primary else None
    alternates = [a.specialist_id for a in getattr(sugg, "alternates", []) or []] if sugg else []

    # Cluster state from the local tailnet view (no central call).
    status = tailnet_status(TailnetConfig(enabled=True))
    on_tailnet = status is not None
    specialist_peers = parse_specialist_peers(status, specialist_tag=specialist_tag, include_self=False) if status else []
    gateway_peers = parse_specialist_peers(status, specialist_tag="tag:gateway", include_self=False) if status else []
    mesh_state = "joining_existing" if specialist_peers else "first_node"

    next_steps: list[dict] = []
    if not engine.installed:
        next_steps.append({"action": "install_backend", "backend": engine.backend,
                           "why": "recommended engine for this hardware is not installed"})
    if recommended:
        next_steps.append({"action": "run", "command": f"slancha-mesh up --specialist {recommended}"})
    else:
        next_steps.append({"action": "note",
                           "message": "no catalog specialist fits this hardware; add a card matching the recommended engine"})
    if mesh_state == "first_node":
        next_steps.append({"action": "note",
                           "message": "first node on this tailnet — a router (slancha-local) must run somewhere to serve traffic"})

    return {
        "hardware": {
            "node_id": probe.node_id, "friendly_name": probe.friendly_name,
            "chip": probe.chip, "arch": probe.arch,
            "cuda_capability": probe.cuda_capability,
            "vram_available_gb": probe.vram_available_gb,
            "ram_available_gb": probe.ram_available_gb,
            "unified_memory": probe.unified_memory,
            "available_backends": list(probe.available_backends),
        },
        "recommended_engine": engine.as_dict(),
        "recommended_specialist": recommended,
        "alternate_specialists": alternates,
        "allocator_rationale": getattr(sugg, "rationale", None) if sugg else None,
        "mesh": {
            "on_tailnet": on_tailnet,
            "state": mesh_state,
            "online_specialist_peers": [p.host for p in specialist_peers],
            "router_present": bool(gateway_peers),
        },
        "next_steps": next_steps,
    }


def cmd_plan(args: argparse.Namespace) -> int:
    plan = build_plan(args.catalog_dir)
    if args.json:
        _print(json.dumps(plan, indent=2))
        return 0
    hw = plan["hardware"]
    eng = plan["recommended_engine"]
    _print(f"hardware: {hw['chip']} / {hw['arch']} / cc={hw['cuda_capability']} "
           f"vram={hw['vram_available_gb']} ram={hw['ram_available_gb']} unified={hw['unified_memory']}")
    _print(f"engine:   {eng['backend']} ({eng['quant']})  installed={eng['installed']}")
    _print(f"          {eng['rationale']}")
    _print(f"specialist: {plan['recommended_specialist'] or '(none fits — see next_steps)'}"
           + (f"   alternates: {', '.join(plan['alternate_specialists'])}" if plan['alternate_specialists'] else ""))
    m = plan["mesh"]
    _print(f"mesh:     on_tailnet={m['on_tailnet']} state={m['state']} "
           f"router_present={m['router_present']} peers={len(m['online_specialist_peers'])}")
    _print("next steps:")
    for step in plan["next_steps"]:
        _print(f"  - {step}")
    return 0


# ---------------------------------------------------------------------------
# discover — walk the tailnet, build a routing table
# ---------------------------------------------------------------------------


def cmd_discover(args: argparse.Namespace) -> int:
    status = tailnet_status(TailnetConfig(enabled=True))
    if status is None:
        _print("[discover] could not read `tailscale status --json` "
               "(is tailscale up? is the binary on PATH?)")
        return 1

    token = args.token or os.environ.get(NODE_TOKEN_ENV) or None
    fetch = make_http_fetch(token=token, timeout=args.timeout)
    result = discover_specialists(
        status,
        fetch=fetch,
        node_info_port=args.node_info_port,
        specialist_tag=args.tag,
        include_self=not args.exclude_self,
    )

    if args.json:
        _print(json.dumps(
            {
                "specialists": {
                    sid: {
                        "model_id": s.model_id,
                        "domain": s.domain,
                        "capabilities": list(s.capabilities),
                        "quality_router_observed": s.quality_router_observed,
                        "node_urls": list(s.node_urls),
                    }
                    for sid, s in result.specialists.items()
                },
                "reachable": result.reachable,
                "unreachable": result.unreachable,
            },
            indent=2,
        ))
        return 0

    _print(f"reachable specialist nodes: {len(result.reachable)}  "
           f"unreachable: {len(result.unreachable)}")
    if result.unreachable:
        _print(f"  unreachable: {', '.join(result.unreachable)}")
    if not result.specialists:
        _print("no specialists discovered.")
        return 0
    _print("")
    _print(f"{'specialist':<28} {'domain':<12} {'nodes':<6} node_urls")
    _print("-" * 80)
    for sid, s in sorted(result.specialists.items()):
        _print(f"{sid:<28} {(s.domain or '-'):<12} {len(s.node_urls):<6} {', '.join(s.node_urls)}")
    return 0


# ---------------------------------------------------------------------------
# status — what does this box look like to the mesh?
# ---------------------------------------------------------------------------


def cmd_status(args: argparse.Namespace) -> int:
    status = tailnet_status(TailnetConfig(enabled=True))
    if status is None:
        _print("[status] not on a tailnet (no `tailscale status`). Loopback dev mode.")
        return 0
    self_obj = status.get("Self") or {}
    host = (self_obj.get("DNSName") or "").rstrip(".") or "(none)"
    _print(f"host:   {host}")
    _print(f"online: {self_obj.get('Online', False)}")
    _print(f"tags:   {', '.join(self_obj.get('Tags') or []) or '(none)'}")
    is_specialist = DEFAULT_SPECIALIST_TAG in (self_obj.get("Tags") or [])
    _print(f"specialist-ready: {is_specialist}"
           + ("" if is_specialist else f"  (missing {DEFAULT_SPECIALIST_TAG})"))
    return 0


# ---------------------------------------------------------------------------
# doctor — verify leg of the agent loop (plan → up → doctor)
# ---------------------------------------------------------------------------


def cmd_doctor(args: argparse.Namespace) -> int:
    from mesh.scripts.mesh_doctor import render_json, render_text, run_node_doctor

    report = run_node_doctor(node_info_port=args.node_info_port)
    _print(render_json(report) if args.json else render_text(report))
    return 1 if report.verdict == "failures" else 0


# ---------------------------------------------------------------------------
# serve — thin alias to the daemon-only path (dev / non-tailnet)
# ---------------------------------------------------------------------------


def cmd_serve(args: argparse.Namespace) -> int:
    from mesh.serve import main as serve_main

    return serve_main(args.rest)


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="slancha-mesh", description=__doc__)
    sub = ap.add_subparsers(dest="command", required=True)

    # up
    up = sub.add_parser("up", help="Join the tailnet + serve specialists + expose self-description.")
    up.add_argument("--specialist", action="append", default=[],
                    help="Specialist id to serve. Repeat for several. Empty = hardware-only heartbeat.")
    up.add_argument("--auto", action="store_true",
                    help="Auto-fit the best specialist to this box's hardware (when no --specialist given).")
    up.add_argument("--base-port", type=int, default=8003, help="First model port (vLLM convention 8003).")
    up.add_argument("--node-info-port", type=int, default=DEFAULT_NODE_INFO_PORT,
                    help="Port the pull-able /models self-description listens on.")
    up.add_argument("--node-info-host", default="0.0.0.0", help="Bind host for the node-info app.")
    up.add_argument("--key", default=None, help="Tailscale/Headscale ephemeral auth key (first join only).")
    up.add_argument("--tailnet", action="store_true", help="Force tailnet mode (else inferred from --key/env).")
    up.add_argument("--advertise-host", default=None, help="Override MagicDNS advertise host.")
    up.add_argument("--control-plane", choices=["tailscale", "headscale"], default=None)
    up.add_argument("--login-server", default=None, help="Headscale control server URL.")
    up.add_argument("--catalog-dir", default=None, help="Catalog dir (default: bundled mesh/catalog).")
    up.add_argument("--with-router", action="store_true",
                    help="Install + launch a local router (slancha-local serve) if none is reachable. "
                         "For a first-node home mesh that needs an OpenAI endpoint.")
    up.add_argument("--router-spec", default=None,
                    help="pip spec for slancha-local (default: env SLANCHA_LOCAL_INSTALL_SPEC or 'slancha-local').")
    up.add_argument("--ready-timeout", type=float, default=600.0)
    up.add_argument("--dry-run", action="store_true", help="Resolve + build, print the plan, don't start.")
    up.set_defaults(func=cmd_up)

    # plan (agent-facing)
    pl = sub.add_parser("plan", help="Recommend engine + specialist for this box; report mesh state. Agent-facing.")
    pl.add_argument("--catalog-dir", default=None, help="Catalog dir (default: bundled mesh/catalog).")
    pl.add_argument("--json", action="store_true", help="Emit the plan as JSON (for an agent to act on).")
    pl.set_defaults(func=cmd_plan)

    # discover
    disc = sub.add_parser("discover", help="Walk the tailnet → routing table of specialists.")
    disc.add_argument("--node-info-port", type=int, default=DEFAULT_NODE_INFO_PORT)
    disc.add_argument("--tag", default=DEFAULT_SPECIALIST_TAG, help="Tailnet tag to enumerate.")
    disc.add_argument("--token", default=None, help=f"Bearer token (else ${NODE_TOKEN_ENV}).")
    disc.add_argument("--timeout", type=float, default=4.0)
    disc.add_argument("--exclude-self", action="store_true", help="Skip the local node.")
    disc.add_argument("--json", action="store_true", help="Emit JSON instead of a table.")
    disc.set_defaults(func=cmd_discover)

    # status
    st = sub.add_parser("status", help="Show this box's tailnet identity + specialist-readiness.")
    st.set_defaults(func=cmd_status)

    # doctor (verify leg of plan → up → doctor)
    dr = sub.add_parser("doctor", help="Diagnose node readiness: tailnet tag, engine, router, ports. Agent-facing.")
    dr.add_argument("--node-info-port", type=int, default=DEFAULT_NODE_INFO_PORT)
    dr.add_argument("--json", action="store_true", help="Emit the report as JSON.")
    dr.set_defaults(func=cmd_doctor)

    # serve (alias to daemon-only mesh.serve for dev / non-tailnet)
    sv = sub.add_parser("serve", help="Daemon-only serve (dev/non-tailnet); passthrough to `python -m mesh.serve`.")
    sv.add_argument("rest", nargs=argparse.REMAINDER, help="Args forwarded to mesh.serve.")
    sv.set_defaults(func=cmd_serve)

    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
