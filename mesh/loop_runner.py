"""Supervised autonomous loop-runner around the champion gate — issue #82.

This is the **executor stage** of the self-improving loop; the gate
(`mesh/eval/gate.py`) is the **terminator stage**. The runner WIRES the
pieces that already exist — it does not reinvent them:

  * `mesh/idle.py`     — `IdleDetector` decides *when* it is safe to train
    (sustained-idle → `READY_TO_TRAIN`; preempts on traffic return). The
    runner consumes that edge; it never decides idleness itself.
  * `mesh/training.py` — `TrainingPass(...).run(preempt_event)` produces a
    checkpoint dir; `ChampionRegistry` holds the best-so-far + rollback.
    The runner is injected an `execute_fn` so tests pass a GPU-free fake.
  * `mesh/eval/gate.py`— `decide(champion, challenger, thresholds) ->
    PromotionVerdict`. The runner calls it after a scored experiment and
    acts on `verdict.accept` (promote vs archive). The gate logic lives
    there; the runner only routes inputs in and the verdict out.

Loop stages (`run_once` is one tick; `run_forever` drives ticks):

    queue → pick → idle-gate → execute → gate → promote/archive
          ↘ empty queue → idle-WAIT (sleep, don't spin)
          ↘ N consecutive fast-fails → circuit-break (pause + record)

Design rule (so the whole loop is unit-testable with NO GPU / NO torch):
every side-effecting seam — filesystem dir, clock, sleep, idle predicate,
execute function, gate decision — is injected. The defaults wire the real
modules; tests inject fakes. Importing this module pulls in nothing heavy.

The contract honoured (docs/GATE-CONTRACT.md): never co-host train with
routed serve traffic (enforced here by gating `train` experiments behind
the idle detector / a drain predicate, NOT by killing processes — the
mesh binding routes around a `health=draining` node); promote-or-archive
is the gate's call, never the runner's; every decision is event-sourced.
"""

from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from mesh.eval.gate import GateThresholds, PromotionVerdict, decide

# ── tunables (issue #82) ─────────────────────────────────────────────────────
# Empty queue → sleep this long before re-checking, instead of busy-looping.
IDLE_WAIT_S: float = 30.0
# A run shorter than this is a "fast-fail" (crash-loop signature, not real work).
FAST_FAIL_S: float = 10.0
# After this many *consecutive* fast-fails, trip the breaker: pause + record a
# decision item rather than burning cycles on a wedged experiment.
CIRCUIT_BREAKER_THRESHOLD: int = 5


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


# ── experiment outcome (what an execute leg returns) ─────────────────────────


@dataclass
class ExperimentResult:
    """Outcome of executing one experiment spec.

    `challenger_row` / `champion_row` are EvalRecord-shaped dicts (as written
    by `mesh/eval/runner.py`: mean_score, per_domain_mean, n_eval, judge_model,
    meta_stub, router_version, …) — exactly the inputs `gate.decide` consumes.
    `checkpoint` is the artifact dir a `train` experiment produced (for
    promotion via ChampionRegistry); None for an `eval`-only experiment or a
    failure. A scored experiment supplies both rows so the gate can run.
    """

    ok: bool
    checkpoint: Path | None = None
    champion_row: dict[str, Any] | None = None
    challenger_row: dict[str, Any] | None = None
    error: str | None = None

    @property
    def scored(self) -> bool:
        """True iff this result carries both rows the gate needs."""
        return self.champion_row is not None and self.challenger_row is not None


# ── queue I/O (append-only JSONL of experiment specs) ────────────────────────


def read_queue(path: Path) -> list[dict[str, Any]]:
    """Read `queue.jsonl` → list of spec dicts (order preserved).

    Missing file → empty queue (a fresh runner has nothing queued yet).
    Blank lines tolerated; a malformed line raises (corruption is loud, not
    silently skipped — a half-written spec must not be silently dropped).
    """
    path = Path(path)
    if not path.exists():
        return []
    specs: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        specs.append(json.loads(line))
    return specs


def write_queue(path: Path, specs: list[dict[str, Any]]) -> None:
    """Atomically rewrite `queue.jsonl` (tmp file + os.replace).

    Atomic so a crash mid-write can never leave a truncated queue — readers
    see either the old file or the new one, never a partial. JSONL, one spec
    per line, in list order.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for spec in specs:
            f.write(json.dumps(spec, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def enqueue(path: Path, spec: dict[str, Any]) -> bool:
    """Append `spec` to the queue, deduped by `id`. Returns True if added.

    Dedup is by `id`: re-enqueuing a spec whose id already exists is a no-op
    (the queue is the source of truth; an operator re-running the same
    generator must not double-train the same candidate). Specs without an
    `id` are always appended (caller owns uniqueness then).
    """
    spec = dict(spec)
    spec.setdefault("status", "pending")
    specs = read_queue(path)
    sid = spec.get("id")
    if sid is not None and any(s.get("id") == sid for s in specs):
        return False
    specs.append(spec)
    write_queue(path, specs)
    return True


def _pick_pending(specs: list[dict[str, Any]]) -> int | None:
    """Index of the highest-priority pending spec, or None if none pending.

    Highest priority = lowest `priority` number (default 100 when absent);
    ties broken by queue order (stable — first-enqueued wins). Only specs
    with status `pending` (or missing status) are eligible.
    """
    eligible = [
        i for i, s in enumerate(specs) if s.get("status", "pending") == "pending"
    ]
    if not eligible:
        return None
    return min(eligible, key=lambda i: (specs[i].get("priority", 100), i))


# ── the runner ───────────────────────────────────────────────────────────────


@dataclass
class LoopRunner:
    """Supervised, queue-driven loop around the champion gate.

    Injected seams (defaults wire the real modules; tests pass fakes):
      * `run_dir`        — where queue.jsonl / status.json / decisions.jsonl /
                           runner.log live.
      * `execute_fn`     — `(spec, preempt_event) -> ExperimentResult`. The
                           real one calls `TrainingPass(...).run(...)` + an
                           eval pass; tests pass a GPU-free fake. REQUIRED
                           (no GPU default — the runner must never import
                           torch just to construct).
      * `is_safe_to_train` — `(spec) -> bool` predicate. Default wraps an
                           injected `IdleDetector.should_start_training()`.
                           A `train`-type experiment only starts when this
                           is True (the "never co-host train+serve" rule).
      * `gate_decide`    — defaults to `mesh.eval.gate.decide`.
      * `thresholds`     — `GateThresholds` passed to the gate.
      * `promote_fn` / `archive_fn` — act on the verdict. Defaults write the
                           champion pointer / archive note under run_dir;
                           the real promote wires `ChampionRegistry.promote`.
      * `clock` / `sleep` — injectable time so tests are deterministic.

    Not injected but configurable: `idle_wait_s`, `fast_fail_s`,
    `circuit_breaker_threshold`.
    """

    run_dir: Path
    execute_fn: Callable[[dict[str, Any], threading.Event], "ExperimentResult"]
    idle_detector: Any | None = None
    is_safe_to_train: Callable[[dict[str, Any]], bool] | None = None
    gate_decide: Callable[..., PromotionVerdict] = decide
    thresholds: GateThresholds = field(default_factory=GateThresholds)
    promote_fn: Callable[[Path | None, dict[str, Any], PromotionVerdict], None] | None = None
    archive_fn: Callable[[Path | None, dict[str, Any], PromotionVerdict], None] | None = None
    clock: Callable[[], datetime] = _utcnow
    sleep: Callable[[float], None] = time.sleep
    idle_wait_s: float = IDLE_WAIT_S
    fast_fail_s: float = FAST_FAIL_S
    circuit_breaker_threshold: int = CIRCUIT_BREAKER_THRESHOLD

    # ── runtime state (not init args) ──
    consecutive_fast_fails: int = field(default=0, init=False)
    paused: bool = field(default=False, init=False)
    preempt_event: threading.Event = field(
        default_factory=threading.Event, init=False
    )

    def __post_init__(self) -> None:
        self.run_dir = Path(self.run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        if self.is_safe_to_train is None:
            self.is_safe_to_train = self._default_is_safe_to_train
        if self.promote_fn is None:
            self.promote_fn = self._default_promote
        if self.archive_fn is None:
            self.archive_fn = self._default_archive

    # ── derived paths ──
    @property
    def queue_path(self) -> Path:
        return self.run_dir / "queue.jsonl"

    @property
    def status_path(self) -> Path:
        return self.run_dir / "status.json"

    @property
    def decisions_path(self) -> Path:
        return self.run_dir / "decisions.jsonl"

    @property
    def log_path(self) -> Path:
        return self.run_dir / "runner.log"

    # ── defaults that wire the real modules (lazy / no GPU) ──

    def _default_is_safe_to_train(self, spec: dict[str, Any]) -> bool:
        """Only `train` experiments are gated on idleness; `eval` is always ok.

        Defers to the injected IdleDetector's `should_start_training()` edge
        — the contract's "never co-host train+serve" rule, enforced by the
        detector (sustained-idle / route-around), NOT by killing processes.
        With no detector wired, a `train` experiment is refused (fail-safe:
        we never co-host by default).
        """
        if spec.get("type") != "train":
            return True
        if self.idle_detector is None:
            return False
        return bool(self.idle_detector.should_start_training())

    def _default_promote(
        self, checkpoint: Path | None, spec: dict[str, Any], verdict: PromotionVerdict
    ) -> None:
        """Record a promotion pointer under run_dir (no GPU/registry import).

        The real binding swaps `ChampionRegistry.promote(checkpoint)`; the
        default just persists the winning checkpoint pointer so a CPU test
        (and an operator) can see what was promoted without a model on disk.
        """
        (self.run_dir / "champion.json").write_text(
            json.dumps(
                {
                    "checkpoint": str(checkpoint) if checkpoint else None,
                    "experiment_id": spec.get("id"),
                    "mean_delta": verdict.mean_delta,
                    "promoted_at": self.clock().isoformat(),
                },
                indent=2,
            )
        )

    def _default_archive(
        self, checkpoint: Path | None, spec: dict[str, Any], verdict: PromotionVerdict
    ) -> None:
        """Archive a rejected challenger — champion stays (no rollback needed).

        Append-only archive note; the champion pointer is untouched, which is
        the GATE-CONTRACT "champion-stays" rollback binding.
        """
        with (self.run_dir / "archive.jsonl").open("a", encoding="utf-8") as f:
            f.write(
                json.dumps(
                    {
                        "experiment_id": spec.get("id"),
                        "checkpoint": str(checkpoint) if checkpoint else None,
                        "reject_reasons": list(verdict.reject_reasons),
                        "archived_at": self.clock().isoformat(),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    # ── observability ──

    def _log(self, msg: str) -> None:
        line = f"{self.clock().isoformat()} {msg}"
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")

    def _record_decision(self, kind: str, detail: dict[str, Any]) -> None:
        """Append a NON-BLOCKING operator decision item to decisions.jsonl.

        The loop never blocks on an operator: a candidate needing a human
        call (circuit-break, branch-on-decision) is recorded here and the
        loop moves on. Append-only so the history is auditable.
        """
        with self.decisions_path.open("a", encoding="utf-8") as f:
            f.write(
                json.dumps(
                    {"kind": kind, "at": self.clock().isoformat(), **detail},
                    ensure_ascii=False,
                )
                + "\n"
            )

    def write_status(self, state: str, **extra: Any) -> None:
        """Heartbeat: overwrite status.json with the current runner state."""
        self.status_path.write_text(
            json.dumps(
                {
                    "state": state,
                    "paused": self.paused,
                    "consecutive_fast_fails": self.consecutive_fast_fails,
                    "updated_at": self.clock().isoformat(),
                    **extra,
                },
                indent=2,
            )
        )

    # ── one tick ──

    def run_once(self) -> str:
        """Run a single loop tick. Returns a short state token for the caller.

        Tokens: "paused" (breaker tripped), "idle_wait" (empty queue),
        "blocked_idle" (train held back — node not idle), "promoted",
        "archived", "ran" (scored but no gate inputs), "failed".
        """
        if self.paused:
            self.write_status("paused")
            return "paused"

        specs = read_queue(self.queue_path)
        idx = _pick_pending(specs)
        if idx is None:
            # Idle-WAIT, not spin: nothing to do → sleep, don't busy-enqueue.
            self.write_status("idle_wait")
            self.sleep(self.idle_wait_s)
            return "idle_wait"

        spec = specs[idx]
        sid = spec.get("id")

        # Idle gate: a train experiment only starts when it is safe (the
        # detector says READY_TO_TRAIN / drain predicate true). Held-back
        # experiments stay `pending` — picked up on a later tick once idle.
        if not self.is_safe_to_train(spec):  # type: ignore[misc]
            self._log(f"hold train experiment {sid}: not safe to train (not idle)")
            self.write_status("blocked_idle", experiment_id=sid)
            self.sleep(self.idle_wait_s)
            return "blocked_idle"

        # Mark running + persist (atomic) so a crash leaves a clear trail.
        spec["status"] = "running"
        write_queue(self.queue_path, specs)
        self.write_status("running", experiment_id=sid)
        self._log(f"start experiment {sid} type={spec.get('type')}")

        self.preempt_event.clear()
        started = self.clock()
        try:
            result = self.execute_fn(spec, self.preempt_event)
        except Exception as exc:  # noqa: BLE001 — a crashing experiment must not kill the loop
            result = ExperimentResult(ok=False, error=f"{type(exc).__name__}: {exc}")
        elapsed_s = (self.clock() - started).total_seconds()

        if not result.ok:
            return self._handle_failure(specs, idx, spec, result, elapsed_s)

        # Success resets the fast-fail streak.
        self.consecutive_fast_fails = 0

        if not result.scored:
            # An experiment that ran but produced no gate inputs (e.g. an
            # eval-only probe with nothing to compare) — done, no verdict.
            spec["status"] = "done"
            write_queue(self.queue_path, specs)
            self._log(f"experiment {sid} ran (no gate inputs) in {elapsed_s:.1f}s")
            self.write_status("ran", experiment_id=sid)
            return "ran"

        return self._gate_and_act(specs, idx, spec, result, elapsed_s)

    def _handle_failure(
        self,
        specs: list[dict[str, Any]],
        idx: int,
        spec: dict[str, Any],
        result: ExperimentResult,
        elapsed_s: float,
    ) -> str:
        sid = spec.get("id")
        spec["status"] = "failed"
        write_queue(self.queue_path, specs)
        fast = elapsed_s < self.fast_fail_s
        if fast:
            self.consecutive_fast_fails += 1
        else:
            # A slow failure is a real attempt, not a crash-loop — reset.
            self.consecutive_fast_fails = 0
        self._log(
            f"experiment {sid} FAILED in {elapsed_s:.1f}s "
            f"(fast={fast}, streak={self.consecutive_fast_fails}): {result.error}"
        )
        # Circuit-breaker: N consecutive fast-fails → pause + surface, don't spin.
        if self.consecutive_fast_fails >= self.circuit_breaker_threshold:
            self.paused = True
            self._record_decision(
                "circuit_breaker",
                {
                    "experiment_id": sid,
                    "consecutive_fast_fails": self.consecutive_fast_fails,
                    "last_error": result.error,
                    "note": "paused after consecutive fast-fails; operator review needed",
                },
            )
            self._log(
                f"CIRCUIT BREAKER tripped after {self.consecutive_fast_fails} "
                "fast-fails — pausing for operator review"
            )
            self.write_status("paused", experiment_id=sid)
            return "paused"
        self.write_status("failed", experiment_id=sid)
        return "failed"

    def _gate_and_act(
        self,
        specs: list[dict[str, Any]],
        idx: int,
        spec: dict[str, Any],
        result: ExperimentResult,
        elapsed_s: float,
    ) -> str:
        """Run the champion gate on the scored result and act on the verdict."""
        sid = spec.get("id")
        verdict = self.gate_decide(
            result.champion_row,  # type: ignore[arg-type]
            result.challenger_row,  # type: ignore[arg-type]
            self.thresholds,
        )
        self._record_decision(
            "gate_verdict",
            {
                "experiment_id": sid,
                "accept": verdict.accept,
                "mean_delta": verdict.mean_delta,
                "reject_reasons": list(verdict.reject_reasons),
            },
        )
        if verdict.accept:
            self.promote_fn(result.checkpoint, spec, verdict)  # type: ignore[misc]
            spec["status"] = "promoted"
            write_queue(self.queue_path, specs)
            self._log(
                f"experiment {sid} PROMOTED (mean_delta={verdict.mean_delta:+.3f}) "
                f"in {elapsed_s:.1f}s"
            )
            self.write_status("promoted", experiment_id=sid)
            return "promoted"

        # Rejected → champion stays; archive the challenger.
        self.archive_fn(result.checkpoint, spec, verdict)  # type: ignore[misc]
        spec["status"] = "archived"
        write_queue(self.queue_path, specs)
        self._log(
            f"experiment {sid} ARCHIVED ({'; '.join(verdict.reject_reasons)}) "
            f"in {elapsed_s:.1f}s"
        )
        self.write_status("archived", experiment_id=sid)
        return "archived"

    # ── the long-lived loop ──

    # ── opt-in: GATE-CONTRACT #8 cloud spot-check drift governor ──

    def enable_cloud_spotcheck(
        self,
        cloud_judge: Any,
        tracker: Any | None = None,
    ) -> None:
        """OPT-IN: wrap `gate_decide` with the #8 cloud-spot-check governor.

        Off by default — it costs cloud tokens. When enabled, an accepted
        verdict that falls in the spot-check sample (~10% of accepts + 100% of
        marginal ones) is re-graded by the INJECTED `cloud_judge`; if the local
        judge has drifted (rolling Spearman < 0.7) the promotion is FROZEN.

        Minimal/additive: this only re-points `self.gate_decide` at a wrapper
        over the existing decide fn — the gate's core logic is untouched, and
        the runner's promote/archive path is unchanged. `tracker` defaults to a
        `DriftTracker` persisted under `run_dir` so the window survives restarts.
        `decisive_gain` is sourced from the runner's own `thresholds`.
        """
        from mesh.eval.spotcheck import DriftTracker, cloud_spotcheck_gate

        if tracker is None:
            tracker = DriftTracker(self.run_dir / "spotcheck.jsonl")
        self.gate_decide = cloud_spotcheck_gate(
            self.gate_decide,
            cloud_judge,
            tracker,
            self.thresholds.mean_score_delta,
        )

    def run_forever(self, max_ticks: int | None = None) -> int:
        """Drive ticks until paused or `max_ticks` is reached.

        `max_ticks` bounds the loop for tests / a one-shot drain; None = run
        until paused (circuit-breaker) or externally stopped. Returns the
        number of ticks executed. A SIGINT-style stop is the operator's job
        (systemd `Restart=always` brings it back) — this just loops.
        """
        ticks = 0
        while max_ticks is None or ticks < max_ticks:
            token = self.run_once()
            ticks += 1
            if token == "paused":
                break
        return ticks


__all__ = [
    "CIRCUIT_BREAKER_THRESHOLD",
    "ExperimentResult",
    "FAST_FAIL_S",
    "IDLE_WAIT_S",
    "LoopRunner",
    "enqueue",
    "read_queue",
    "write_queue",
]
