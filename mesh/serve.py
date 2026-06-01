"""Node-side serving daemon — spec §5 + §9.

Reads the catalog, spawns one backend per enabled specialist, and posts
heartbeats to the registry so the router can place traffic. Designed to
run on a Spark / Mac mini / RTX box; the same module is used by the
in-process integration tests with `NullBackend`.

CLI:
    python -m mesh.serve --base-port 8001 --specialist qwen3-coder-30b-a3b-fp8

Programmatic:
    daemon = ServeDaemon(...); daemon.start(); daemon.run_forever()

Heartbeats are kept in-process for v0.0.2 (no live FastAPI registry yet).
The integration test injects a MeshRegistry; the CLI logs them to disk.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import threading
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path

from mesh.backends import (
    DEFAULT_OLLAMA_PORT,
    BaseBackend,
    NullBackend,
    OllamaBackend,
    VLLMBackend,
)
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
from mesh.replay_store import TrafficReplayStore
from mesh.tailnet import TailnetConfig, advertise_url, resolve_advertise_host
from mesh.training import TrainingPass

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
    # Tailnet-dialable host (e.g. a MagicDNS name) the registry should
    # advertise for this node. None → advertise the backend's own
    # (loopback) base_url unchanged — back-compat for non-tailnet dev.
    # When set, each backend's bind URL has its host swapped to this so
    # the cloud gateway can reach the node over WireGuard.
    advertise_host: str | None = None
    heartbeat_interval_s: float = HEARTBEAT_INTERVAL_S
    log_path: Path | None = None
    # Idle fine-tune detector — observes util signals each heartbeat;
    # daemon spawns training thread on its READY_TO_TRAIN edge when
    # training_replay_store + training_checkpoint_dir are both set.
    # Default None (disabled) so v0.0.3-shape callers stay unchanged.
    idle_detector: IdleDetector | None = None
    # Training integration (v0.0.5 #39): both must be set to enable
    # idle-fine-tune. Detector → fires READY_TO_TRAIN → daemon spawns
    # TrainingPass thread that respects detector.preempt_event. On
    # return (natural or preempt): daemon calls detector.finish_training.
    # If either is None, training is disabled even when detector is set.
    training_replay_store: TrafficReplayStore | None = None
    training_checkpoint_dir: Path | None = None
    # Per-pass kwargs (n_examples, n_steps_planned, per_step_sleep_s, seed).
    # Defaults are TrainingPass defaults (20 stub steps × 1ms).
    training_kwargs: dict | None = None

    _stop: threading.Event = field(default_factory=threading.Event, init=False)
    _thread: threading.Thread | None = field(default=None, init=False)
    _heartbeats_sent: int = field(default=0, init=False)
    _training_thread: threading.Thread | None = field(default=None, init=False)
    _last_checkpoint_path: Path | None = field(default=None, init=False)

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
        # Preempt in-flight training pass so we don't orphan it on shutdown.
        if self._training_thread is not None and self._training_thread.is_alive():
            if self.idle_detector is not None:
                self.idle_detector.signal_preempt()
            self._training_thread.join(timeout=5.0)
            self._training_thread = None
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
                    # Advertise THIS backend's port under the tailnet host
                    # (no-op swap when advertise_host is None).
                    node_url=advertise_url(be.base_url, self.advertise_host),
                )
            )
            util = be.utilization() or {}
            max_queue = max(max_queue, int(util.get("queue_depth", 0)))

        util_obj = NodeUtilization(queue_depth=max_queue)
        # Detector observation + training spawn + health override.
        # If detector is set: observe util, transition state, fold
        # detector.health() into the heartbeat.
        # If training is configured + detector says READY_TO_TRAIN:
        # spawn a TrainingPass thread + transition detector to TRAINING.
        # If state is TRAINING + traffic returned: signal preempt; the
        # training thread polls preempt_event and yields cleanly.
        # On natural training completion: thread join triggers
        # finish_training() to enter COOLDOWN.
        # When no backends are loaded, "degraded" wins (capacity > training).
        if loaded and self.idle_detector is not None:
            self.idle_detector.observe(util_obj, now)

            # Reap completed training thread → COOLDOWN transition.
            if (
                self._training_thread is not None
                and not self._training_thread.is_alive()
                and self.idle_detector.state.value == "training"
            ):
                self._training_thread = None
                try:
                    self.idle_detector.finish_training(now)
                except RuntimeError as exc:
                    # A benign race (state flipped between the alive-check
                    # above and here) is expected. But a persistent
                    # RuntimeError would silently strand the detector in
                    # TRAINING and starve every future pass — log it so a
                    # stuck daemon is diagnosable instead of mute.
                    self._log(
                        f"[training] finish_training skipped "
                        f"(state={self.idle_detector.state.value}): {exc}"
                    )

            # Spawn training on READY_TO_TRAIN edge.
            if self.idle_detector.should_start_training() and self._training_enabled() and loaded:
                self._spawn_training_thread(primary=loaded[0])

            base_health = self.idle_detector.health()

            # Preempt: traffic returned mid-training → tell the training
            # thread to checkpoint + yield. Thread reap happens on the
            # next heartbeat.
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
        # Node-level fallback URL (representative); per-specialist URLs ride
        # on hb.loaded_models. Swapped to the tailnet host when advertising.
        node_url = (
            advertise_url(first_backend.base_url, self.advertise_host)
            if first_backend
            else None
        )
        req = HeartbeatPostRequest(heartbeat=hb, node_url=node_url)
        if self.registry is not None:
            self.registry.record_heartbeat(req)
        else:
            self._log(f"[heartbeat] {hb.node_id} loaded={[lm.specialist_id for lm in hb.loaded_models]} q={hb.util.queue_depth}")
        self._heartbeats_sent += 1

    @property
    def heartbeats_sent(self) -> int:
        return self._heartbeats_sent

    # --- training integration (v0.0.5 #39) ---

    def _training_enabled(self) -> bool:
        """True iff all training-config fields are set."""
        return (
            self.training_replay_store is not None
            and self.training_checkpoint_dir is not None
            and self._training_thread is None
        )

    def _spawn_training_thread(self, primary: LoadedModel) -> None:
        """Build a TrainingPass for the primary specialist + spawn it.

        Detector transitions READY_TO_TRAIN → TRAINING via
        mark_training_started; thread runs in background, polls
        preempt_event each step, writes checkpoint on return.
        """
        assert self.idle_detector is not None
        assert self.training_replay_store is not None
        assert self.training_checkpoint_dir is not None

        kwargs = dict(self.training_kwargs or {})
        # Find the matching SpecialistCard to extract domain / base model.
        primary_card = None
        for be in self.backends:
            if be.card.specialist_id == primary.specialist_id:
                primary_card = be.card
                break
        if primary_card is None:
            self._log(f"[training] no card for {primary.specialist_id}; skipping spawn")
            return

        pass_ = TrainingPass(
            specialist_id=primary_card.specialist_id,
            base_model_id=primary_card.model_id,
            domain=primary_card.domain,
            replay_store=self.training_replay_store,
            checkpoint_dir=self.training_checkpoint_dir,
            # v0.0.4 intentionally runs the contract-only stub (issue #55):
            # this leg exists to exercise the daemon thread-spawn + preempt
            # wiring, not to produce a real adapter. Opt in explicitly so the
            # stub does not raise StubTrainingError. Real PEFT lands in #65.
            # Placed before **kwargs so a caller can still override it.
            allow_stub=True,
            **kwargs,
        )
        try:
            self.idle_detector.mark_training_started()
        except RuntimeError as exc:
            # Lost the race; detector moved out of READY_TO_TRAIN.
            self._log(f"[training] mark_training_started lost race: {exc}")
            return

        preempt_event = self.idle_detector.preempt_event

        def _runner() -> None:
            try:
                self._last_checkpoint_path = pass_.run(preempt_event=preempt_event)
                self._log(
                    f"[training] checkpoint @ {self._last_checkpoint_path} "
                    f"steps={pass_.meta.n_steps_completed if pass_.meta else '?'} "
                    f"preempted={pass_.meta.preempted if pass_.meta else '?'}"
                )
            except Exception as exc:  # noqa: BLE001
                self._log(f"[training] pass failed: {exc}")

        self._training_thread = threading.Thread(
            target=_runner, daemon=True, name=f"training-{primary_card.specialist_id}"
        )
        self._training_thread.start()

    @property
    def last_checkpoint_path(self) -> Path | None:
        return self._last_checkpoint_path

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
                f.write(f"{datetime.now(timezone.utc).isoformat()} {msg}\n")
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
    bind_host: str = "127.0.0.1",
) -> BaseBackend:
    """Pick a backend implementation from the card's `required_backend`.

    `cuda_capability` is threaded in from NodeProbe so vLLM can pick the
    right FP8 kernel path per-chip. Blackwell consumer (sm_120/sm_121)
    needs the Marlin weight-only FP8 fallback; Hopper/Ada don't. None →
    conservative (no Marlin force, use native path).

    `bind_host` is where the serving process LISTENS. Defaults to loopback
    (`127.0.0.1`); set to `0.0.0.0` (or the tailnet IP) so the cloud
    gateway can reach the backend over the tailnet. The advertised URL
    (a MagicDNS name) is set separately on the daemon, not here.
    """
    log_path = None
    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"{card.specialist_id}.log"

    if card.required_backend == "vllm":
        return VLLMBackend(
            card=card,
            host=bind_host,
            port=port,
            log_path=log_path,
            cuda_capability=cuda_capability,
        )
    if card.required_backend == "ollama":
        # Ollama multiplexes every loaded model on one daemon port (default
        # 11434), so the per-specialist `port` from the serve loop is
        # informational here — we advertise the daemon URL. The card needs
        # an `ollama_tag` (validated inside the backend) and `OLLAMA_PORT`
        # in the env wins if a non-default port is in use.
        ollama_port = int(os.environ.get("OLLAMA_PORT", DEFAULT_OLLAMA_PORT))
        # `card.ollama_tag` missing → NullBackend with a clear log line so
        # mixed-catalog serve still boots and the operator gets a hint.
        if card.ollama_tag is None:
            return NullBackend(card=card)
        return OllamaBackend(
            card=card,
            host=bind_host,
            port=ollama_port,
            log_path=log_path,
        )
    if card.required_backend in ("llamacpp", "mlx"):
        # Not yet wired — set the card's `required_backend` to "ollama" +
        # add `ollama_tag` to serve GGUF/MLX models through the Ollama path
        # in the meantime.
        return NullBackend(card=card)
    raise ValueError(f"unknown backend {card.required_backend!r}")


def build_daemon(
    specialist_ids: list[str] | None = None,
    catalog: list[SpecialistCard] | None = None,
    probe: NodeProbe | None = None,
    registry: MeshRegistry | None = None,
    base_port: int = 8001,
    log_dir: Path | None = None,
    tailnet: TailnetConfig | None = None,
) -> ServeDaemon:
    """Construct a ServeDaemon from the catalog + an optional filter.

    `specialist_ids` defaults to the empty list, which means "don't load
    anything; just heartbeat the hardware." Pass a list to start specific
    specialists.

    `tailnet` (optional) switches the node onto a Tailscale/Headscale
    tailnet: backends bind to `tailnet.bind_host` (0.0.0.0) and the daemon
    advertises a MagicDNS host (explicit `advertise_host` or auto-resolved
    via `tailscale status --json`). When None or `enabled=False`, the
    daemon binds loopback and advertises loopback — unchanged dev behavior.
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

    bind_host = "127.0.0.1"
    advertise_host: str | None = None
    if tailnet is not None and tailnet.enabled:
        bind_host = tailnet.bind_host
        advertise_host = resolve_advertise_host(tailnet)

    backends: list[BaseBackend] = []
    port = base_port
    for card in selected:
        backends.append(
            build_backend(
                card,
                port=port,
                log_dir=log_dir,
                cuda_capability=probe.cuda_capability,
                bind_host=bind_host,
            )
        )
        port += 1

    return ServeDaemon(
        backends=backends,
        probe=probe,
        registry=registry,
        advertise_host=advertise_host,
        log_path=log_dir / "serve.log" if log_dir else None,
    )


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
    ap.add_argument(
        "--tailnet",
        action="store_true",
        help="Advertise over a Tailscale/Headscale tailnet (bind 0.0.0.0, "
        "advertise a MagicDNS host). Also enabled via SLANCHA_TAILNET_ENABLED=1.",
    )
    ap.add_argument(
        "--advertise-host",
        default=None,
        help="MagicDNS host the registry advertises (overrides auto-discovery "
        "via `tailscale status --json`). Implies --tailnet.",
    )
    ap.add_argument(
        "--bind-host",
        default=None,
        help="Host the model backends bind to (default 0.0.0.0 under --tailnet, "
        "else 127.0.0.1).",
    )
    args = ap.parse_args(argv)

    # CLI flags layer on top of SLANCHA_TAILNET_* env defaults.
    tailnet = TailnetConfig.from_env()
    if args.tailnet or args.advertise_host or args.bind_host:
        tailnet = replace(
            tailnet,
            enabled=True,
            advertise_host=args.advertise_host or tailnet.advertise_host,
            bind_host=args.bind_host or tailnet.bind_host,
        )

    daemon = build_daemon(
        specialist_ids=args.specialist,
        base_port=args.base_port,
        log_dir=args.log_dir,
        tailnet=tailnet,
    )
    if tailnet.enabled:
        daemon._log(
            f"[main] tailnet on: bind={tailnet.bind_host} "
            f"advertise={daemon.advertise_host or '(unresolved — check tailscale status)'}"
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
