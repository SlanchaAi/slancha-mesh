"""Supervised loop-runner tests — issue #82. Hermetic, NO GPU, NO torch.

Everything that touches the world is injected: the execute leg (a fake
that returns champion/challenger rows + a checkpoint without a GPU), the
clock + sleep (so fast-fail timing is deterministic), the idle predicate
(a fake IdleDetector), and the filesystem dir (`tmp_path`). The real
`mesh.eval.gate.decide` runs unmocked — the integration the runner exists
to wire — so a promote-vs-archive decision is the genuine gate verdict.
"""

from __future__ import annotations

import json
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path

from mesh.eval.gate import GateThresholds
from mesh.loop_runner import (
    CIRCUIT_BREAKER_THRESHOLD,
    ExperimentResult,
    LoopRunner,
    enqueue,
    read_queue,
    write_queue,
)


# ── fakes ────────────────────────────────────────────────────────────────────


class FakeClock:
    """Deterministic clock; `advance()` moves it, so elapsed-time is exact."""

    def __init__(self) -> None:
        self.t = datetime(2026, 6, 1, 12, 0, 0, tzinfo=timezone.utc)

    def __call__(self) -> datetime:
        return self.t

    def advance(self, seconds: float) -> None:
        self.t = self.t + timedelta(seconds=seconds)


class FakeIdle:
    """Stand-in for IdleDetector — just the edge the runner consumes."""

    def __init__(self, ready: bool) -> None:
        self.ready = ready

    def should_start_training(self) -> bool:
        return self.ready


def _eval_row(mean: float, *, n: int = 200, judge: str = "judge-v1",
              per_domain: dict | None = None, stub: bool = False) -> dict:
    """An EvalRecord-shaped row, exactly what gate.decide consumes."""
    return {
        "router_version": f"v-{mean}",
        "n_eval": n,
        "judge_model": judge,
        "mean_score": mean,
        "per_domain_mean": per_domain or {"general": mean, "code": mean},
        "meta_stub": stub,
    }


def _train_spec(sid: str, *, priority: int = 100) -> dict:
    return {"id": sid, "type": "train", "priority": priority, "status": "pending"}


def _runner(tmp_path: Path, execute_fn, **kw) -> LoopRunner:
    """LoopRunner with a fake clock + no-op sleep by default; idle by default."""
    clock = kw.pop("clock", FakeClock())
    sleeps: list[float] = []
    runner = LoopRunner(
        run_dir=tmp_path,
        execute_fn=execute_fn,
        idle_detector=kw.pop("idle_detector", FakeIdle(ready=True)),
        clock=clock,
        sleep=kw.pop("sleep", lambda s: sleeps.append(s)),
        **kw,
    )
    runner._test_clock = clock  # type: ignore[attr-defined]
    runner._test_sleeps = sleeps  # type: ignore[attr-defined]
    return runner


# ── queue I/O: read / write / dedup / atomicity ──────────────────────────────


def test_read_missing_queue_is_empty(tmp_path):
    assert read_queue(tmp_path / "queue.jsonl") == []


def test_write_then_read_roundtrip_preserves_order(tmp_path):
    q = tmp_path / "queue.jsonl"
    specs = [{"id": "a", "priority": 1}, {"id": "b", "priority": 2}]
    write_queue(q, specs)
    assert read_queue(q) == specs


def test_write_is_atomic_no_tmp_left_behind(tmp_path):
    q = tmp_path / "queue.jsonl"
    write_queue(q, [{"id": "a"}])
    assert q.exists()
    assert not (tmp_path / "queue.jsonl.tmp").exists()


def test_enqueue_dedups_by_id(tmp_path):
    q = tmp_path / "queue.jsonl"
    assert enqueue(q, {"id": "x"}) is True
    assert enqueue(q, {"id": "x"}) is False  # dup → skipped
    assert enqueue(q, {"id": "y"}) is True
    rows = read_queue(q)
    assert [r["id"] for r in rows] == ["x", "y"]


def test_enqueue_sets_default_status_pending(tmp_path):
    q = tmp_path / "queue.jsonl"
    enqueue(q, {"id": "x"})
    assert read_queue(q)[0]["status"] == "pending"


def test_read_blank_lines_tolerated(tmp_path):
    q = tmp_path / "queue.jsonl"
    q.write_text('{"id": "a"}\n\n{"id": "b"}\n')
    assert [r["id"] for r in read_queue(q)] == ["a", "b"]


# ── priority pick ────────────────────────────────────────────────────────────


def test_pick_lowest_priority_number_first(tmp_path):
    picked: list[str] = []

    def execute(spec, preempt):
        picked.append(spec["id"])
        return ExperimentResult(ok=True)  # ran, no gate inputs

    runner = _runner(tmp_path, execute)
    write_queue(runner.queue_path, [
        {"id": "low", "type": "eval", "priority": 100, "status": "pending"},
        {"id": "high", "type": "eval", "priority": 1, "status": "pending"},
        {"id": "mid", "type": "eval", "priority": 50, "status": "pending"},
    ])
    runner.run_once()
    runner.run_once()
    runner.run_once()
    assert picked == ["high", "mid", "low"]


def test_pick_ties_broken_by_queue_order(tmp_path):
    picked: list[str] = []

    def execute(spec, preempt):
        picked.append(spec["id"])
        return ExperimentResult(ok=True)

    runner = _runner(tmp_path, execute)
    write_queue(runner.queue_path, [
        {"id": "first", "type": "eval", "priority": 5, "status": "pending"},
        {"id": "second", "type": "eval", "priority": 5, "status": "pending"},
    ])
    runner.run_once()
    runner.run_once()
    assert picked == ["first", "second"]


# ── idle-WAIT on empty queue (no spin) ───────────────────────────────────────


def test_empty_queue_sleeps_does_not_spin(tmp_path):
    calls = {"n": 0}

    def execute(spec, preempt):
        calls["n"] += 1
        return ExperimentResult(ok=True)

    runner = _runner(tmp_path, execute, idle_wait_s=42.0)
    token = runner.run_once()
    assert token == "idle_wait"
    assert calls["n"] == 0  # execute never called on empty queue
    assert runner._test_sleeps == [42.0]  # slept exactly once, the idle wait


# ── idle-gate blocks train when detector not READY ───────────────────────────


def test_train_blocked_when_not_idle(tmp_path):
    ran: list[str] = []

    def execute(spec, preempt):
        ran.append(spec["id"])
        return ExperimentResult(ok=True)

    runner = _runner(tmp_path, execute, idle_detector=FakeIdle(ready=False))
    write_queue(runner.queue_path, [_train_spec("t1")])
    token = runner.run_once()
    assert token == "blocked_idle"
    assert ran == []  # execute NOT called — train held back
    # spec stays pending so a later (idle) tick can pick it up
    assert read_queue(runner.queue_path)[0]["status"] == "pending"


def test_train_runs_when_idle(tmp_path):
    ran: list[str] = []

    def execute(spec, preempt):
        ran.append(spec["id"])
        return ExperimentResult(ok=True)

    runner = _runner(tmp_path, execute, idle_detector=FakeIdle(ready=True))
    write_queue(runner.queue_path, [_train_spec("t1")])
    runner.run_once()
    assert ran == ["t1"]


def test_eval_not_gated_on_idle(tmp_path):
    """An eval-type experiment runs even when the node is NOT idle."""
    ran: list[str] = []

    def execute(spec, preempt):
        ran.append(spec["id"])
        return ExperimentResult(ok=True)

    runner = _runner(tmp_path, execute, idle_detector=FakeIdle(ready=False))
    write_queue(runner.queue_path, [
        {"id": "e1", "type": "eval", "priority": 1, "status": "pending"},
    ])
    runner.run_once()
    assert ran == ["e1"]


def test_default_is_safe_refuses_train_with_no_detector(tmp_path):
    """Fail-safe: no detector wired → never co-host (train refused)."""
    runner = _runner(tmp_path, lambda s, p: ExperimentResult(ok=True),
                     idle_detector=None)
    assert runner.is_safe_to_train(_train_spec("t1")) is False
    assert runner.is_safe_to_train({"id": "e", "type": "eval"}) is True


# ── gate integration: promote vs archive (real gate.decide) ──────────────────


def test_promote_on_real_gain(tmp_path):
    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()

    def execute(spec, preempt):
        return ExperimentResult(
            ok=True,
            checkpoint=ckpt,
            champion_row=_eval_row(3.0),
            challenger_row=_eval_row(3.5),  # +0.5 mean, no domain regression
        )

    runner = _runner(tmp_path, execute)
    write_queue(runner.queue_path, [_train_spec("t1")])
    token = runner.run_once()
    assert token == "promoted"
    champ = json.loads((tmp_path / "champion.json").read_text())
    assert champ["experiment_id"] == "t1"
    assert champ["checkpoint"] == str(ckpt)
    assert read_queue(runner.queue_path)[0]["status"] == "promoted"


def test_archive_on_no_gain(tmp_path):
    def execute(spec, preempt):
        return ExperimentResult(
            ok=True,
            champion_row=_eval_row(3.0),
            challenger_row=_eval_row(3.0),  # wash → REJECT_NO_GAIN
        )

    runner = _runner(tmp_path, execute)
    write_queue(runner.queue_path, [_train_spec("t1")])
    token = runner.run_once()
    assert token == "archived"
    arch = (tmp_path / "archive.jsonl").read_text().strip()
    assert "t1" in arch
    assert read_queue(runner.queue_path)[0]["status"] == "archived"
    # champion pointer never written on an archive
    assert not (tmp_path / "champion.json").exists()


def test_archive_on_per_domain_regression(tmp_path):
    """Mean lifts but one domain collapses → gate rejects (no silent promote)."""
    def execute(spec, preempt):
        return ExperimentResult(
            ok=True,
            champion_row=_eval_row(3.0, per_domain={"general": 3.0, "code": 3.0}),
            challenger_row=_eval_row(
                3.3, per_domain={"general": 4.0, "code": 2.0}  # code −1.0 cliff
            ),
        )

    runner = _runner(tmp_path, execute)
    write_queue(runner.queue_path, [_train_spec("t1")])
    assert runner.run_once() == "archived"


def test_stub_challenger_archived(tmp_path):
    """A stub artifact can never promote, even with a clean gain."""
    def execute(spec, preempt):
        return ExperimentResult(
            ok=True,
            champion_row=_eval_row(3.0),
            challenger_row=_eval_row(3.9, stub=True),
        )

    runner = _runner(tmp_path, execute)
    write_queue(runner.queue_path, [_train_spec("t1")])
    assert runner.run_once() == "archived"


def test_gate_verdict_recorded_as_decision(tmp_path):
    def execute(spec, preempt):
        return ExperimentResult(
            ok=True, checkpoint=tmp_path / "c",
            champion_row=_eval_row(3.0), challenger_row=_eval_row(3.5),
        )

    runner = _runner(tmp_path, execute)
    write_queue(runner.queue_path, [_train_spec("t1")])
    runner.run_once()
    decisions = [json.loads(line) for line in
                 (tmp_path / "decisions.jsonl").read_text().splitlines()]
    verdicts = [d for d in decisions if d["kind"] == "gate_verdict"]
    assert len(verdicts) == 1
    assert verdicts[0]["accept"] is True


def test_thresholds_are_passed_to_gate(tmp_path):
    """A custom min_n_eval flows into the gate (challenger n below floor)."""
    def execute(spec, preempt):
        return ExperimentResult(
            ok=True,
            champion_row=_eval_row(3.0, n=5),
            challenger_row=_eval_row(3.9, n=5),  # big gain but n < min_n
        )

    runner = _runner(tmp_path, execute,
                     thresholds=GateThresholds(min_n_eval=100))
    write_queue(runner.queue_path, [_train_spec("t1")])
    assert runner.run_once() == "archived"  # rejected on min_n, not promoted


def test_scored_but_eval_only_marks_done(tmp_path):
    """ok=True with no rows → ran, no verdict (an eval-only probe)."""
    def execute(spec, preempt):
        return ExperimentResult(ok=True)

    runner = _runner(tmp_path, execute)
    write_queue(runner.queue_path, [
        {"id": "e1", "type": "eval", "priority": 1, "status": "pending"},
    ])
    assert runner.run_once() == "ran"
    assert read_queue(runner.queue_path)[0]["status"] == "done"


# ── circuit-breaker: trips after N consecutive fast-fails ────────────────────


def test_circuit_breaker_trips_after_five_fast_fails(tmp_path):
    clock = FakeClock()

    def execute(spec, preempt):
        # fast-fail: return immediately (no clock advance → elapsed 0s < 10s)
        return ExperimentResult(ok=False, error="boom")

    runner = _runner(tmp_path, execute, clock=clock)
    specs = [
        {"id": f"f{i}", "type": "eval", "priority": 1, "status": "pending"}
        for i in range(CIRCUIT_BREAKER_THRESHOLD + 2)
    ]
    write_queue(runner.queue_path, specs)

    tokens = [runner.run_once() for _ in range(CIRCUIT_BREAKER_THRESHOLD)]
    assert tokens[:-1] == ["failed"] * (CIRCUIT_BREAKER_THRESHOLD - 1)
    assert tokens[-1] == "paused"  # 5th fast-fail trips the breaker
    assert runner.paused is True

    # paused → further ticks are no-ops, execute never called again
    assert runner.run_once() == "paused"

    # breaker recorded a non-blocking decision item
    decisions = [json.loads(line) for line in
                 (tmp_path / "decisions.jsonl").read_text().splitlines()]
    assert any(d["kind"] == "circuit_breaker" for d in decisions)


def test_slow_failure_does_not_trip_breaker(tmp_path):
    """A slow failure is a real attempt, not a crash-loop — streak resets."""
    clock = FakeClock()

    def execute(spec, preempt):
        clock.advance(30.0)  # 30s > fast_fail_s → NOT a fast-fail
        return ExperimentResult(ok=False, error="slow boom")

    runner = _runner(tmp_path, execute, clock=clock)
    specs = [
        {"id": f"s{i}", "type": "eval", "priority": 1, "status": "pending"}
        for i in range(CIRCUIT_BREAKER_THRESHOLD + 1)
    ]
    write_queue(runner.queue_path, specs)
    for _ in range(CIRCUIT_BREAKER_THRESHOLD + 1):
        runner.run_once()
    assert runner.paused is False
    assert runner.consecutive_fast_fails == 0


def test_success_resets_fast_fail_streak(tmp_path):
    clock = FakeClock()
    outcomes = iter([
        ExperimentResult(ok=False, error="boom"),
        ExperimentResult(ok=False, error="boom"),
        ExperimentResult(ok=True),  # success resets
        ExperimentResult(ok=False, error="boom"),
    ])

    def execute(spec, preempt):
        return next(outcomes)

    runner = _runner(tmp_path, execute, clock=clock)
    write_queue(runner.queue_path, [
        {"id": f"x{i}", "type": "eval", "priority": 1, "status": "pending"}
        for i in range(4)
    ])
    for _ in range(4):
        runner.run_once()
    # 2 fails, reset by success, then 1 fail → streak 1, never paused
    assert runner.consecutive_fast_fails == 1
    assert runner.paused is False


# ── preempt event honored ────────────────────────────────────────────────────


def test_preempt_event_passed_to_execute(tmp_path):
    seen: dict = {}

    def execute(spec, preempt):
        seen["is_event"] = isinstance(preempt, threading.Event)
        seen["clear_at_start"] = not preempt.is_set()
        return ExperimentResult(ok=True)

    runner = _runner(tmp_path, execute)
    runner.preempt_event.set()  # dirty it; run_once must clear before execute
    write_queue(runner.queue_path, [
        {"id": "e1", "type": "eval", "priority": 1, "status": "pending"},
    ])
    runner.run_once()
    assert seen["is_event"] is True
    assert seen["clear_at_start"] is True  # cleared fresh each tick


def test_execute_can_observe_preempt_signal(tmp_path):
    """An execute leg that polls the event sees a set signal honored."""
    observed: dict = {}

    def execute(spec, preempt):
        # simulate: signal arrives mid-run, leg checks and yields
        preempt.set()
        observed["preempted"] = preempt.is_set()
        return ExperimentResult(ok=True)

    runner = _runner(tmp_path, execute)
    write_queue(runner.queue_path, [
        {"id": "e1", "type": "eval", "priority": 1, "status": "pending"},
    ])
    runner.run_once()
    assert observed["preempted"] is True


# ── a crashing execute leg does not kill the loop ────────────────────────────


def test_execute_exception_becomes_failure_not_crash(tmp_path):
    def execute(spec, preempt):
        raise RuntimeError("kaboom")

    runner = _runner(tmp_path, execute)
    write_queue(runner.queue_path, [
        {"id": "e1", "type": "eval", "priority": 1, "status": "pending"},
    ])
    token = runner.run_once()  # must not raise
    assert token == "failed"
    assert read_queue(runner.queue_path)[0]["status"] == "failed"


# ── status heartbeat ─────────────────────────────────────────────────────────


def test_status_json_written_each_tick(tmp_path):
    def execute(spec, preempt):
        return ExperimentResult(
            ok=True, checkpoint=tmp_path / "c",
            champion_row=_eval_row(3.0), challenger_row=_eval_row(3.5),
        )

    runner = _runner(tmp_path, execute)
    write_queue(runner.queue_path, [_train_spec("t1")])
    runner.run_once()
    status = json.loads(runner.status_path.read_text())
    assert status["state"] == "promoted"
    assert status["experiment_id"] == "t1"
    assert status["paused"] is False


# ── run_forever drains then idle-waits / stops on pause ──────────────────────


def test_run_forever_bounded_by_max_ticks(tmp_path):
    n = {"c": 0}

    def execute(spec, preempt):
        n["c"] += 1
        return ExperimentResult(ok=True)

    runner = _runner(tmp_path, execute)
    write_queue(runner.queue_path, [
        {"id": "e1", "type": "eval", "priority": 1, "status": "pending"},
        {"id": "e2", "type": "eval", "priority": 2, "status": "pending"},
    ])
    ticks = runner.run_forever(max_ticks=2)
    assert ticks == 2
    assert n["c"] == 2


def test_run_forever_stops_when_paused(tmp_path):
    def execute(spec, preempt):
        return ExperimentResult(ok=False, error="boom")  # all fast-fail

    runner = _runner(tmp_path, execute)
    write_queue(runner.queue_path, [
        {"id": f"f{i}", "type": "eval", "priority": 1, "status": "pending"}
        for i in range(20)
    ])
    ticks = runner.run_forever(max_ticks=100)
    # stops as soon as the breaker trips (5 fast-fails), not all 20
    assert ticks == CIRCUIT_BREAKER_THRESHOLD
    assert runner.paused is True
