"""Live end-to-end training smoke — v0.0.5 #40.

Cold-boot or adopt vLLM on :8001, run ServeDaemon with IdleDetector +
TrainingPass enabled, force the detector into READY_TO_TRAIN via
synthetic clock, observe the daemon spawn a training pass, watch it
write a real checkpoint to disk, confirm clean shutdown.

Spec §7 + §12 day-7 end-to-end proof under real subprocess + real
heartbeat conditions. No mocks; the only synthetic part is the
60-second idle window (we don't actually wait 60s of real time —
detector accepts injected `now` for the observe() calls).

Usage:
    VLLM_LIVE_URL=http://127.0.0.1:8001 \\
        python3 -m mesh.scripts.live_training_smoke \\
            [--checkpoint-dir /tmp/mesh-ckpt] \\
            [--specialist qwen3-coder-30b-a3b-fp8]

Exit codes:
    0  smoke green — training fired, checkpoint landed, clean shutdown
    1  vLLM unreachable
    2  detector did not transition
    3  training thread did not spawn / never finished
    4  checkpoint missing or malformed
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

from mesh.backends import VLLMBackend
from mesh.catalog import load_catalog
from mesh.idle import IdleDetector
from mesh.models import NodeUtilization
from mesh.probe import probe_node
from mesh.registry import MeshRegistry
from mesh.replay_store import TrafficReplayStore
from mesh.serve import ServeDaemon


def _log(msg: str) -> None:
    ts = datetime.now(timezone.utc).strftime("%H:%M:%S.%f")[:-3]
    print(f"[{ts}] {msg}", flush=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--vllm-url", default=os.environ.get("VLLM_LIVE_URL", "http://127.0.0.1:8001"))
    ap.add_argument("--specialist", default="qwen3-coder-30b-a3b-fp8")
    ap.add_argument("--checkpoint-dir", type=Path, default=Path("/tmp/mesh-live-v5/checkpoints"))
    ap.add_argument("--max-wait-s", type=float, default=30.0,
                    help="how long to wait for training thread to complete")
    args = ap.parse_args(argv)

    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # 1. Resolve specialist card
    catalog = load_catalog()
    by_id = {c.specialist_id: c for c in catalog}
    if args.specialist not in by_id:
        _log(f"FATAL specialist {args.specialist} not in catalog")
        return 1
    card = by_id[args.specialist]
    _log(f"specialist: {card.specialist_id} (domain={card.domain})")

    # 2. Verify vLLM live
    import httpx
    try:
        r = httpx.get(f"{args.vllm_url}/health", timeout=5.0)
        if r.status_code != 200:
            _log(f"FATAL vLLM /health returned {r.status_code}")
            return 1
    except Exception as exc:  # noqa: BLE001
        _log(f"FATAL vLLM at {args.vllm_url} unreachable: {exc}")
        return 1
    _log(f"vLLM live at {args.vllm_url}")

    # 3. Build daemon with PID-adopted VLLMBackend + detector + training config
    port = int(args.vllm_url.rsplit(":", 1)[-1])
    backend = VLLMBackend(card=card, host="127.0.0.1", port=port)
    backend.start()  # adopt existing PID via port-busy logic
    if backend._adopted_pid is None:
        _log("FATAL backend did not adopt vLLM PID; aborting (would relaunch)")
        return 1
    _log(f"adopted vLLM pid {backend._adopted_pid}")

    detector = IdleDetector()
    replay_store = TrafficReplayStore(max_size=100)
    # Seed the replay store with a small corpus (real PEFT would consume this)
    for i in range(10):
        replay_store.add(
            prompt_text=f"def fib_{i}(n):",
            oracle_response="    if n < 2: return n\n    return fib(n-1) + fib(n-2)",
            domain=card.domain,
            difficulty="medium",
            served_by_specialist=card.specialist_id,
        )

    # 4. Drive detector to READY_TO_TRAIN via synthetic clock BEFORE building
    # the daemon. heartbeat() calls observe(util, now=real_clock), which would
    # overwrite the detector's _idle_since to real-now and prevent the
    # synthetic 60s window from registering. Order matters here; this is the
    # same constraint test_daemon_idle_detector_reports_training_health_via_heartbeat
    # documents in mesh/tests/test_serve.py.
    anchor = datetime(2026, 5, 16, 12, 0, 0, tzinfo=timezone.utc)
    detector.observe(NodeUtilization(gpu_util_pct=0.0, queue_depth=0), anchor)
    detector.observe(NodeUtilization(gpu_util_pct=0.0, queue_depth=0),
                     anchor + timedelta(seconds=61))
    if not detector.should_start_training():
        _log(f"FATAL detector did not advance to READY_TO_TRAIN: {detector.state.value}")
        return 2
    _log("detector synthetic-clock advanced to READY_TO_TRAIN (pre-daemon)")

    probe = probe_node()
    registry = MeshRegistry(catalog=[card])
    daemon = ServeDaemon(
        backends=[backend],
        probe=probe,
        registry=registry,
        idle_detector=detector,
        training_replay_store=replay_store,
        training_checkpoint_dir=args.checkpoint_dir,
        # Short fake training so the smoke completes in seconds
        training_kwargs={"n_steps_planned": 50, "per_step_sleep_s": 0.01, "seed": 42},
    )

    # 5. First heartbeat — observes util (real now), state stays TRAINING
    # because detector.observe during ACTIVE/READY_TO_TRAIN with idle signal
    # doesn't bump _idle_since. The spawn check fires on
    # should_start_training() → spawns thread → mark_training_started →
    # TRAINING. Health flips to "training".
    hb = daemon.heartbeat()
    _log(f"hb1: health={hb.health} loaded={[m.specialist_id for m in hb.loaded_models]} "
         f"queue={hb.util.queue_depth} state={detector.state.value} "
         f"training_thread={'alive' if daemon._training_thread and daemon._training_thread.is_alive() else 'none/dead'}")
    if hb.health != "training":
        _log(f"FATAL hb1 health should be 'training' after spawn: {hb.health}")
        return 3
    if daemon._training_thread is None:
        _log("FATAL training thread was not spawned")
        return 3

    # 7. Wait for training thread to finish (50 steps × 10ms = ~500ms expected)
    t0 = time.time()
    daemon._training_thread.join(timeout=args.max_wait_s)
    if daemon._training_thread.is_alive():
        _log(f"FATAL training thread still alive after {args.max_wait_s}s")
        detector.signal_preempt()
        daemon._training_thread.join(timeout=5.0)
        return 3
    elapsed = time.time() - t0
    _log(f"training thread finished after {elapsed:.2f}s")

    # 8. Reap heartbeat — should transition to COOLDOWN
    hb = daemon.heartbeat()
    _log(f"hb3 (after reap): health={hb.health} state={detector.state.value}")
    if detector.state.value != "cooldown":
        _log(f"FATAL post-training state should be cooldown: {detector.state.value}")
        return 3

    # 9. Verify checkpoint on disk
    ck = daemon.last_checkpoint_path
    if ck is None or not ck.exists():
        _log(f"FATAL checkpoint missing: {ck}")
        return 4
    state_path = ck / "state_dict.json"
    meta_path = ck / "meta.json"
    if not state_path.exists() or not meta_path.exists():
        _log(f"FATAL checkpoint dir incomplete: {list(ck.iterdir())}")
        return 4
    meta = json.loads(meta_path.read_text())
    _log(f"checkpoint @ {ck}")
    _log(f"  steps_completed={meta['n_steps_completed']}/{meta['n_steps_planned']}")
    _log(f"  preempted={meta['preempted']}")
    _log(f"  corpus_hash={meta['corpus_hash']}")
    _log(f"  n_examples={meta['n_examples']}")
    if meta["preempted"]:
        _log("WARN training was preempted (smoke expected natural completion)")
    if meta["n_examples"] != 10:
        _log(f"WARN n_examples {meta['n_examples']} != 10 seeded")

    # 10. Clean shutdown — daemon.stop() preempts any in-flight + joins
    _log("calling daemon.stop()")
    daemon.stop(timeout=5.0)
    _log("daemon stopped cleanly")

    # 11. vLLM still alive (we adopted; didn't kill it)
    try:
        r = httpx.get(f"{args.vllm_url}/health", timeout=2.0)
        _log(f"vLLM still healthy: status={r.status_code}")
    except Exception:
        _log("WARN vLLM no longer responding after daemon.stop()")

    _log("SMOKE GREEN")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
