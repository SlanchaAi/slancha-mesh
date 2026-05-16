"""Idle fine-tune detector + state machine — spec §7.

Watches the heartbeat utilization stream and decides when a node has
been quiet long enough to start a LoRA training pass. Preempts itself
when traffic returns.

Pure state machine: callers feed observations via `observe(util)`,
read `state` + `health()` after each call. No threads, no I/O. The
training pipeline (mesh/training.py — v0.0.5) consumes the
`TRAINING` state edge to actually fire a pass; the detector only
decides WHEN.

State diagram (spec §7 paragraph 1):

    ACTIVE ←──────────────── (gpu≥10% OR queue>0) ────────────────┐
       │                                                           │
       └──── 60s sustained idle ──→ READY_TO_TRAIN ───→ TRAINING ──┤
                                                          │        │
                                                          │ preempt│ traffic returns
                                                          ↓        ↑
                                                       COOLDOWN ───┘
                                                          │
                                                          │ 5min elapsed
                                                          ↓
                                                       ACTIVE

The COOLDOWN tier avoids thrashing between TRAINING and ACTIVE when
traffic is near the threshold. Without it, a burst-quiet cluster
would start + preempt training every few seconds, blocking forward
progress entirely.

Health mapping for the heartbeat:
    ACTIVE          → "healthy"
    READY_TO_TRAIN  → "healthy"   (we're not training yet, just decided)
    TRAINING        → "training"  (router drops hot-interactive traffic)
    COOLDOWN        → "healthy"
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Callable

from mesh.models import HealthState, NodeUtilization


class IdleState(str, Enum):
    """Internal states for the idle detector. Distinct from HealthState."""

    ACTIVE = "active"  # under load OR just started observing
    READY_TO_TRAIN = "ready_to_train"  # quiet long enough; training pass can begin
    TRAINING = "training"  # actively training; preempt window open
    COOLDOWN = "cooldown"  # post-training quiet period before re-eligibility


# Spec §7 thresholds. Conservatively tuned; tunable per-deployment.
GPU_IDLE_THRESHOLD_PCT: float = 10.0
QUEUE_IDLE_THRESHOLD: int = 0
SUSTAINED_IDLE_S: float = 60.0
COOLDOWN_S: float = 300.0  # 5 min — avoid train/preempt thrash


# Note: idle-classification lives on IdleDetector as `_is_idle`, since
# thresholds are per-instance (custom-tuned per deployment). The
# module-level constants are defaults only.


@dataclass
class IdleDetector:
    """Spec §7 state machine, no I/O.

    Feed it `NodeUtilization` observations via `observe(util, now)`.
    Read `state`, `health()`, `should_start_training()` after each call.

    The detector does NOT spawn threads or call training itself — it
    just exposes signals. The serving daemon (mesh.serve.ServeDaemon)
    is the integration point that:
      1. Calls `observe(...)` each heartbeat tick.
      2. Reads `state` to fold into the heartbeat health field.
      3. On the rising edge of `should_start_training()`, spawns a
         training thread that respects `preempt_event`.
      4. On heartbeat with new traffic, calls `signal_preempt()` if
         state is TRAINING — the training thread checks `preempt_event`
         each step and yields.

    Reason this is decoupled from training: training pipeline (PEFT
    + transformers) is heavyweight and platform-specific; detector is
    pure Python + testable in 0.01s. Future training backends
    (DiLoCo, SPARTA) bolt on without touching detection logic.
    """

    gpu_idle_threshold_pct: float = GPU_IDLE_THRESHOLD_PCT
    queue_idle_threshold: int = QUEUE_IDLE_THRESHOLD
    sustained_idle_s: float = SUSTAINED_IDLE_S
    cooldown_s: float = COOLDOWN_S

    state: IdleState = field(default=IdleState.ACTIVE, init=False)
    _idle_since: datetime | None = field(default=None, init=False)
    _training_started_at: datetime | None = field(default=None, init=False)
    _cooldown_until: datetime | None = field(default=None, init=False)
    _preempt_event: threading.Event = field(
        default_factory=threading.Event, init=False
    )
    _on_training_start: Callable[[], None] | None = field(default=None, init=False)

    # --- public API ---

    def _is_idle(self, util: NodeUtilization) -> bool:
        return (
            util.gpu_util_pct < self.gpu_idle_threshold_pct
            and util.queue_depth <= self.queue_idle_threshold
        )

    def observe(self, util: NodeUtilization, now: datetime | None = None) -> IdleState:
        """Feed one utilization sample. Returns the new state.

        `now` is injectable for tests; defaults to `datetime.now(utc)`.
        """
        now = now or datetime.now(timezone.utc)
        idle = self._is_idle(util)

        if self.state == IdleState.COOLDOWN:
            if self._cooldown_until is not None and now < self._cooldown_until:
                return self.state
            # Cooldown elapsed — drop back to ACTIVE; next idle window
            # accumulates from now.
            self.state = IdleState.ACTIVE
            self._cooldown_until = None
            self._idle_since = now if idle else None
            return self.state

        if self.state == IdleState.TRAINING:
            # In training; only exit via preempt() or finish_training().
            # observe() during training doesn't transition state by
            # itself — that's the training thread's job.
            return self.state

        # ACTIVE or READY_TO_TRAIN: idle vs busy drives transitions.
        if not idle:
            self._idle_since = None
            self.state = IdleState.ACTIVE
            return self.state

        # idle == True
        if self._idle_since is None:
            self._idle_since = now
            return self.state

        elapsed_s = (now - self._idle_since).total_seconds()
        if elapsed_s >= self.sustained_idle_s and self.state == IdleState.ACTIVE:
            self.state = IdleState.READY_TO_TRAIN
        return self.state

    def should_start_training(self) -> bool:
        """Edge signal: True iff caller should fire a training pass NOW.

        Idempotent only on consecutive calls in the same state — once
        the caller starts training and calls `mark_training_started()`,
        further calls return False until the next idle window.
        """
        return self.state == IdleState.READY_TO_TRAIN

    def mark_training_started(self, now: datetime | None = None) -> None:
        """Transition READY_TO_TRAIN → TRAINING.

        Caller fires after spawning the training thread. Resets the
        preempt event so the new training pass can listen on it.
        """
        if self.state != IdleState.READY_TO_TRAIN:
            raise RuntimeError(
                f"mark_training_started requires READY_TO_TRAIN; was {self.state.value}"
            )
        self.state = IdleState.TRAINING
        self._training_started_at = now or datetime.now(timezone.utc)
        self._preempt_event.clear()

    def signal_preempt(self) -> None:
        """Tell the training thread to yield ASAP.

        Sets the preempt event; the training loop checks it each step.
        Idempotent. Should be called by ServeDaemon when traffic returns
        mid-training. The training thread is expected to checkpoint +
        return; on its return, caller invokes `finish_training()`.
        """
        self._preempt_event.set()

    @property
    def preempt_event(self) -> threading.Event:
        """The event a training thread polls each step."""
        return self._preempt_event

    def finish_training(self, now: datetime | None = None) -> None:
        """Transition TRAINING → COOLDOWN.

        Caller invokes this AFTER the training thread has returned
        (whether via natural completion or preempt). Starts the cooldown
        timer; until it elapses, no new training pass eligible.
        """
        if self.state != IdleState.TRAINING:
            raise RuntimeError(
                f"finish_training requires TRAINING; was {self.state.value}"
            )
        now = now or datetime.now(timezone.utc)
        self.state = IdleState.COOLDOWN
        self._cooldown_until = now + timedelta(seconds=self.cooldown_s)
        self._training_started_at = None
        self._idle_since = None

    def health(self) -> HealthState:
        """Map current state to the HealthState reported in heartbeats.

        Spec §7 paragraph 1: only TRAINING externalizes as "training".
        Everything else looks healthy to the router.
        """
        if self.state == IdleState.TRAINING:
            return "training"
        return "healthy"

    # --- diagnostics ---

    def __repr__(self) -> str:
        return (
            f"IdleDetector(state={self.state.value}, "
            f"idle_since={self._idle_since}, "
            f"cooldown_until={self._cooldown_until}, "
            f"preempted={self._preempt_event.is_set()})"
        )


__all__ = [
    "COOLDOWN_S",
    "GPU_IDLE_THRESHOLD_PCT",
    "IdleDetector",
    "IdleState",
    "QUEUE_IDLE_THRESHOLD",
    "SUSTAINED_IDLE_S",
]
