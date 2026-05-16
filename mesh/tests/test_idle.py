"""Idle detector + state machine tests — spec §7.

Pure state-machine tests using injected `now`. No clocks, no threads
(except the preempt-event smoke). All <0.05s.
"""

from __future__ import annotations

import threading
import time
from datetime import datetime, timedelta, timezone

import pytest

from mesh.idle import (
    COOLDOWN_S,
    GPU_IDLE_THRESHOLD_PCT,
    IdleDetector,
    IdleState,
    QUEUE_IDLE_THRESHOLD,
    SUSTAINED_IDLE_S,
)
from mesh.models import NodeUtilization


def _busy() -> NodeUtilization:
    return NodeUtilization(gpu_util_pct=50.0, queue_depth=3)


def _idle() -> NodeUtilization:
    return NodeUtilization(gpu_util_pct=2.0, queue_depth=0)


def _t(offset_s: float, anchor: datetime | None = None) -> datetime:
    anchor = anchor or datetime(2026, 5, 16, 12, 0, 0, tzinfo=timezone.utc)
    return anchor + timedelta(seconds=offset_s)


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------


def test_initial_state_is_active():
    d = IdleDetector()
    assert d.state == IdleState.ACTIVE
    assert d.health() == "healthy"
    assert not d.should_start_training()


def test_thresholds_match_spec_defaults():
    """Spec §7: gpu < 10%, queue == 0, sustained 60s."""
    assert GPU_IDLE_THRESHOLD_PCT == 10.0
    assert QUEUE_IDLE_THRESHOLD == 0
    assert SUSTAINED_IDLE_S == 60.0
    # Cooldown is our addition, not in spec — defaults to 5 min.
    assert COOLDOWN_S == 300.0


# ---------------------------------------------------------------------------
# ACTIVE → READY_TO_TRAIN edge
# ---------------------------------------------------------------------------


def test_busy_keeps_state_active():
    d = IdleDetector()
    d.observe(_busy(), _t(0))
    d.observe(_busy(), _t(5))
    d.observe(_busy(), _t(70))  # any duration, still busy
    assert d.state == IdleState.ACTIVE


def test_short_idle_does_not_transition():
    d = IdleDetector()
    d.observe(_idle(), _t(0))
    d.observe(_idle(), _t(59))  # below threshold
    assert d.state == IdleState.ACTIVE


def test_sustained_idle_transitions_to_ready_to_train():
    d = IdleDetector()
    d.observe(_idle(), _t(0))  # start idle window
    d.observe(_idle(), _t(60))  # exactly 60s elapsed — transition
    assert d.state == IdleState.READY_TO_TRAIN
    assert d.should_start_training()
    # Health is still "healthy" — we haven't actually started training.
    assert d.health() == "healthy"


def test_busy_in_middle_of_idle_window_resets():
    d = IdleDetector()
    d.observe(_idle(), _t(0))
    d.observe(_idle(), _t(40))
    d.observe(_busy(), _t(45))  # traffic returned
    d.observe(_idle(), _t(50))  # fresh window
    d.observe(_idle(), _t(105))  # only 55s since fresh start
    assert d.state == IdleState.ACTIVE


def test_gpu_just_under_threshold_idle():
    d = IdleDetector()
    near_idle = NodeUtilization(gpu_util_pct=9.99, queue_depth=0)
    d.observe(near_idle, _t(0))
    d.observe(near_idle, _t(60))
    assert d.state == IdleState.READY_TO_TRAIN


def test_gpu_at_threshold_busy():
    """Threshold is `<`, not `<=`. 10.0 should NOT count as idle."""
    d = IdleDetector()
    at_threshold = NodeUtilization(gpu_util_pct=10.0, queue_depth=0)
    d.observe(at_threshold, _t(0))
    d.observe(at_threshold, _t(60))
    assert d.state == IdleState.ACTIVE


def test_queue_depth_breaks_idle():
    """Even with 0% GPU, queue_depth > 0 means we have queued work."""
    d = IdleDetector()
    queued = NodeUtilization(gpu_util_pct=0.0, queue_depth=1)
    d.observe(queued, _t(0))
    d.observe(queued, _t(60))
    assert d.state == IdleState.ACTIVE


# ---------------------------------------------------------------------------
# TRAINING transitions + preempt
# ---------------------------------------------------------------------------


def test_mark_training_started_requires_ready_state():
    d = IdleDetector()
    with pytest.raises(RuntimeError):
        d.mark_training_started(_t(0))


def test_training_health_flips_to_training():
    d = IdleDetector()
    d.observe(_idle(), _t(0))
    d.observe(_idle(), _t(60))
    assert d.state == IdleState.READY_TO_TRAIN
    d.mark_training_started(_t(60))
    assert d.state == IdleState.TRAINING
    assert d.health() == "training"


def test_observe_during_training_does_not_transition():
    """Once TRAINING, only signal_preempt + finish_training move state."""
    d = IdleDetector()
    d.observe(_idle(), _t(0))
    d.observe(_idle(), _t(60))
    d.mark_training_started(_t(60))
    d.observe(_busy(), _t(70))
    d.observe(_idle(), _t(80))
    assert d.state == IdleState.TRAINING


def test_preempt_event_set_by_signal_preempt():
    d = IdleDetector()
    d.observe(_idle(), _t(0))
    d.observe(_idle(), _t(60))
    d.mark_training_started(_t(60))
    assert not d.preempt_event.is_set()
    d.signal_preempt()
    assert d.preempt_event.is_set()
    # State stays TRAINING until finish_training is called by the training thread
    assert d.state == IdleState.TRAINING


def test_finish_training_transitions_to_cooldown():
    d = IdleDetector()
    d.observe(_idle(), _t(0))
    d.observe(_idle(), _t(60))
    d.mark_training_started(_t(60))
    d.finish_training(_t(120))
    assert d.state == IdleState.COOLDOWN
    assert d.health() == "healthy"


def test_finish_training_requires_training_state():
    d = IdleDetector()
    with pytest.raises(RuntimeError):
        d.finish_training(_t(0))


# ---------------------------------------------------------------------------
# COOLDOWN
# ---------------------------------------------------------------------------


def test_cooldown_blocks_immediate_re_training():
    d = IdleDetector()
    d.observe(_idle(), _t(0))
    d.observe(_idle(), _t(60))
    d.mark_training_started(_t(60))
    d.finish_training(_t(120))
    # Even a long idle window during cooldown should not retrigger
    d.observe(_idle(), _t(200))
    d.observe(_idle(), _t(300))
    assert d.state == IdleState.COOLDOWN


def test_cooldown_expires_then_idle_window_reaccumulates():
    """After 5min cooldown elapses, observe drops state to ACTIVE;
    a fresh 60s idle window then transitions back to READY_TO_TRAIN."""
    d = IdleDetector()
    d.observe(_idle(), _t(0))
    d.observe(_idle(), _t(60))
    d.mark_training_started(_t(60))
    d.finish_training(_t(120))
    # cooldown ends at 120 + 300 = 420
    d.observe(_idle(), _t(420))  # exactly at cooldown end
    assert d.state == IdleState.ACTIVE
    d.observe(_idle(), _t(481))  # 61s of fresh idle
    assert d.state == IdleState.READY_TO_TRAIN


def test_cooldown_resets_idle_window():
    """The pre-cooldown idle accumulator shouldn't carry over."""
    d = IdleDetector()
    d.observe(_idle(), _t(0))
    d.observe(_idle(), _t(60))
    d.mark_training_started(_t(60))
    d.finish_training(_t(120))
    d.observe(_idle(), _t(420))  # cooldown end, ACTIVE
    # We've only been "idle" for 1 sample since cooldown end.
    # Spec §7 sustained-60s rule must apply to the post-cooldown window
    # alone, not to the pre-cooldown carry.
    assert d.state == IdleState.ACTIVE


# ---------------------------------------------------------------------------
# Preempt event semantics under concurrency (smoke)
# ---------------------------------------------------------------------------


def test_preempt_event_unblocks_thread_within_grace():
    """A training thread polling preempt_event each step should see set
    within the same OS scheduling tick after signal_preempt."""
    d = IdleDetector()
    d.observe(_idle(), _t(0))
    d.observe(_idle(), _t(60))
    d.mark_training_started(_t(60))

    saw_preempt = threading.Event()

    def fake_training():
        # Poll every 1ms; should pick up within ~10ms.
        deadline = time.time() + 1.0
        while time.time() < deadline:
            if d.preempt_event.is_set():
                saw_preempt.set()
                return
            time.sleep(0.001)

    t = threading.Thread(target=fake_training, daemon=True)
    t.start()
    time.sleep(0.01)  # let the thread enter its poll loop
    d.signal_preempt()
    t.join(timeout=0.5)
    assert saw_preempt.is_set(), "training thread did not observe preempt within 500ms"


# ---------------------------------------------------------------------------
# Custom thresholds
# ---------------------------------------------------------------------------


def test_custom_thresholds_honored():
    """Tighter thresholds: 30s sustained, queue tolerance 2."""
    d = IdleDetector(
        gpu_idle_threshold_pct=20.0,
        queue_idle_threshold=2,
        sustained_idle_s=30.0,
        cooldown_s=10.0,
    )
    util = NodeUtilization(gpu_util_pct=15.0, queue_depth=2)
    d.observe(util, _t(0))
    d.observe(util, _t(30))
    assert d.state == IdleState.READY_TO_TRAIN
    d.mark_training_started(_t(30))
    d.finish_training(_t(35))
    d.observe(_idle(), _t(45))  # cooldown ends at 45 (10s after 35)
    assert d.state == IdleState.ACTIVE
