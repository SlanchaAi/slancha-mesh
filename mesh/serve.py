"""Node-side serving daemon — spec §5 + §9.

Reads the catalog, spawns one backend per enabled specialist, and posts
heartbeats to the registry so the router can place traffic. Designed to
run on a Spark / Mac mini / RTX box; the same module is used by the
in-process integration tests with `NullBackend`.

CLI:
    python -m mesh.serve --catalog mesh/catalog --port 8001 \
        --specialist qwen3-coder-30b-a3b-fp8

Programmatic:
    daemon = ServeDaemon(...); daemon.start(); daemon.run_forever()

Heartbeats are kept in-process for v0.0.2 (no live FastAPI registry yet).
The integration test injects a MeshRegistry; the CLI logs them to disk.
"""

from __future__ import annotations

import argparse
import json
import signal
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from mesh.backends import BaseBackend, NullBackend, VLLMBackend
from mesh.catalog import load_catalog
from mesh.idle import IdleDetector
from mesh.models import (
    LoadedModel,
    NodeHeartbeat,
    NodeProbe,
    NodeUtilization,
    SpecialistCard,
)
from mesh.probe import probe_node
from mesh.registry import HeartbeatPostRequest, MeshRegistry

HEARTBEAT_INTERVAL_S = 5.0
RUNTIME_DIR = Path(__file__).parent / ".runtime"


# ---------------------------------------------------------------------------
# Daemon
# ---------------------------------------------------------------------------


@dataclass
class ServeDaemon:
    """Owns the backends + heartbeat loop for one mesh node.

    The daemon is intentionally small: it does NOT serve HTTP itself —
    each backend exposes its own OpenAI-compatible endpoint. The daemon
    is the glue between (a) the local backends and (b) the registry the
    router queries.
    """

    backends: list[BaseBackend]
    probe: NodeProbe
    registry: MeshRegistry | None = None
    node_url_template: str = "http://{host}:{port}"
    heartbeat_interval_s: float = HEARTBEAT_INTERVAL_S
    log_path: Path | None = None
    # Idle fine-tune detector — observes util signals each heartbeat;
    # caller spawns training thread on its READY_TO_TRAIN edge. Default
    # None (disabled) so v0.0.3-shape callers stay unchanged.
    idle_detector: IdleDetector | None = None

    _stop: threading.Event = field(default_factory=threading.Event, init=False)
    _thread: threading.Thread | None = field(default=None, init=False)
    _heartbeats_sent: int = field(default=0, init=False)

    # --- lifecycle ---

    def start(self, wait_ready: bool = True, ready_timeout: float = 600.0) -> bool:
        """Spawn all backends. Optionally block until each is serving.

        Returns True if every backend came up healthy within timeout.
        On any backend failure we DO NOT short-circuit — the daemon
        runs with whatever survives, and heartbeats reflect reduced
        capacity (spec §6.6: caller falls through to next route).
        """
        ok = True
        for be in self.backends:
            try:
                be.start()
            except Exception as exc:  # noqa: BLE001 — record any backend-launch error
                self._log(f"[start] backend {be.card.specialist_id} failed: {exc}")
                ok = False
                continue
            if wait_ready:
                if not be.wait_ready(timeout=ready_timeout):
                    self._log(f"[start] backend {be.card.specialist_id} did not become ready")
                    ok = False
        return ok

    def stop(self, timeout: float = 30.0) -> None:
        self._stop.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5.0)
        for be in self.backends:
            try:
                be.stop(timeout=timeout)
            except Exception as exc:  # noqa: BLE001
                self._log(f"[stop] backend {be.card.specialist_id} stop failed: {exc}")

    def run_forever(self) -> None:
        """Block on the heartbeat loop. Handles SIGINT/SIGTERM cleanly."""
        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)
        self._loop()

    def run_in_thread(self) -> None:
        """Background heartbeat loop, used by tests."""
        if self._thread and self._thread.is_alive():
            return
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    # --- heartbeat ---

    def heartbeat(self) -> NodeHeartbeat:
        """Build one NodeHeartbeat from the live backends + probe."""
        now = datetime.now(timezone.utc)
        loaded: list[LoadedModel] = []
        max_queue = 0
        for be in self.backends:
            if not be.is_alive():
                continue
            tps = be.card.estimated_tps_at.get("gb10")  # mirrors probe's chip-keyed lookup
            loaded.append(
                LoadedModel(
                    specialist_id=be.card.specialist_id,
                    model_id=be.card.model_id,
                    loaded_at=now,
                    estimated_tps=tps,
                )
            )
            util = be.utilization() or {}
            max_queue = max(max_queue, int(util.get("queue_depth", 0)))

        util_obj = NodeUtilization(queue_depth=max_queue)
        # Detector observation + health override:
        # If the detector is in TRAINING state, the heartbeat reports
        # "training" instead of "healthy" so the router drops
        # hot-interactive traffic per spec §6.4 + §7.
        # When no backends are loaded, "degraded" wins (capacity > training).
        if loaded and self.idle_detector is not None:
            self.idle_detector.observe(util_obj, now)
            base_health = self.idle_detector.health()
            # Preempt: if a hot request arrived while we were TRAINING,
            # the queue_depth > 0 signal we just observed should also
            # tell the training thread to yield.
            if (
                self.idle_detector.state.value == "training"
                and not self.idle_detector._is_idle(util_obj)
            ):
                self.idle_detector.signal_preempt()
        else:
            base_health = "healthy" if loaded else "degraded"

        return NodeHeartbeat(
            node_id=self.probe.node_id,
            ts=now,
            hardware=self.probe,
            loaded_models=loaded,
            util=util_obj,
            health=base_health,
        )

    def post_heartbeat(self) -> None:
        """Push one heartbeat into the registry (or log if no registry)."""
        hb = self.heartbeat()
        first_backend = self.backends[0] if self.backends else None
        node_url = first_backend.base_url if first_backend else None
        req = HeartbeatPostRequest(heartbeat=hb, node_url=node_url)
        if self.registry is not None:
            self.registry.record_heartbeat(req)
        else:
            self._log(f"[heartbeat] {hb.node_id} loaded={[lm.specialist_id for lm in hb.loaded_models]} q={hb.util.queue_depth}")
        self._heartbeats_sent += 1

    @property
    def heartbeats_sent(self) -> int:
        return self._heartbeats_sent

    # --- internals ---

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                self.post_heartbeat()
            except Exception as exc:  # noqa: BLE001
                self._log(f"[heartbeat] failed: {exc}")
            self._stop.wait(self.heartbeat_interval_s)

    def _signal_handler(self, signum, _frame) -> None:
        self._log(f"[serve] received signal {signum}, shutting down")
        self.stop()
        sys.exit(0)

    def _log(self, msg: str) -> None:
        if self.log_path:
            with self.log_path.open("a", encoding="utf-8") as f:
                f.write(f"{datetime.now().isoformat()} {msg}\n")
        else:
            print(msg, flush=True)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def build_backend(
    card: SpecialistCard,
    port: int,
    log_dir: Path | None = None,
    cuda_capability: str | None = None,
) -> BaseBackend:
    """Pick a backend implementation from the card's `required_backend`.

    `cuda_capability` is threaded in from NodeProbe so vLLM can pick the
    right FP8 kernel path per-chip. Blackwell consumer (sm_120/sm_121)
    needs the Marlin weight-only FP8 fallback; Hopper/Ada don't. None →
    conservative (no Marlin force, use native path).
    """
    log_path = None
    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"{card.specialist_id}.log"

    if card.required_backend == "vllm":
        return VLLMBackend(
            card=card,
            port=port,
            log_path=log_path,
            cuda_capability=cuda_capability,
        )
    if card.required_backend in ("llamacpp", "ollama", "mlx"):
        # v0.0.2 ships vLLM only; non-vllm cards return NullBackend so the
        # daemon doesn't crash on a mixed-backend catalog. Real llamacpp
        # support is v0.0.3 (see docs/MESH_V002_BUILD_2026_05_16.md).
        return NullBackend(card=card)
    raise ValueError(f"unknown backend {card.required_backend!r}")


def build_daemon(
    specialist_ids: list[str] | None = None,
    catalog: list[SpecialistCard] | None = None,
    probe: NodeProbe | None = None,
    registry: MeshRegistry | None = None,
    base_port: int = 8001,
    log_dir: Path | None = None,
) -> ServeDaemon:
    """Construct a ServeDaemon from the catalog + an optional filter.

    `specialist_ids` defaults to the empty list, which means "don't load
    anything; just heartbeat the hardware." Pass a list to start specific
    specialists.
    """
    if catalog is None:
        catalog = load_catalog()
    by_id = {c.specialist_id: c for c in catalog}
    if specialist_ids is None:
        specialist_ids = []
    selected = [by_id[sid] for sid in specialist_ids if sid in by_id]
    missing = [sid for sid in specialist_ids if sid not in by_id]
    if missing:
        raise KeyError(f"unknown specialist_id(s): {missing}")

    if probe is None:
        probe = probe_node()

    backends: list[BaseBackend] = []
    port = base_port
    for card in selected:
        backends.append(
            build_backend(
                card,
                port=port,
                log_dir=log_dir,
                cuda_capability=probe.cuda_capability,
            )
        )
        port += 1

    return ServeDaemon(backends=backends, probe=probe, registry=registry, log_path=log_dir / "serve.log" if log_dir else None)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Slancha-Mesh node serving daemon")
    ap.add_argument(
        "--specialist",
        action="append",
        default=[],
        help="Specialist id(s) to load. Repeat for multiple. If empty, registry-only mode.",
    )
    ap.add_argument("--base-port", type=int, default=8001)
    ap.add_argument("--ready-timeout", type=float, default=600.0)
    ap.add_argument("--once", action="store_true", help="Send one heartbeat then exit (smoke test).")
    ap.add_argument("--log-dir", type=Path, default=RUNTIME_DIR)
    ap.add_argument("--print-heartbeat", action="store_true", help="Print first heartbeat as JSON to stdout.")
    args = ap.parse_args(argv)

    daemon = build_daemon(
        specialist_ids=args.specialist,
        base_port=args.base_port,
        log_dir=args.log_dir,
    )

    ok = daemon.start(wait_ready=True, ready_timeout=args.ready_timeout)
    if not ok:
        daemon._log("[main] one or more backends failed to come up; continuing in degraded mode")

    if args.print_heartbeat:
        print(json.dumps(daemon.heartbeat().model_dump(mode="json"), default=str, indent=2))

    if args.once:
        daemon.post_heartbeat()
        daemon.stop()
        return 0

    try:
        daemon.run_forever()
    finally:
        daemon.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
