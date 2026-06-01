"""Tests for mesh.eval.spotcheck (GATE-CONTRACT #8 cloud-spot-check governor).

Hermetic: NO real cloud calls — the cloud judge is a canned fake everywhere.
"""

from __future__ import annotations

from mesh.eval.gate import PromotionVerdict
from mesh.eval.spotcheck import (
    DEFAULT_CORR_THRESHOLD,
    DriftTracker,
    cloud_spotcheck_gate,
    is_marginal,
    should_spotcheck,
    spearman,
)


# ── fakes / helpers ──────────────────────────────────────────────────────────


class FakeCloudJudge:
    """Returns canned scores in call order; records every call. No network."""

    def __init__(self, scores: list[float]) -> None:
        self._scores = list(scores)
        self._i = 0
        self.calls: list[tuple[str, str]] = []

    def score(self, prompt_text: str, response_text: str) -> float:
        self.calls.append((prompt_text, response_text))
        s = self._scores[self._i % len(self._scores)]
        self._i += 1
        return s


def _verdict(
    *,
    accept: bool = True,
    mean_delta: float = 0.30,
    challenger_version: str = "v-cand",
) -> PromotionVerdict:
    return PromotionVerdict(
        accept=accept,
        reject_reasons=() if accept else ("synthetic reject",),
        mean_delta=mean_delta,
        challenger_version=challenger_version,
    )


def _samples(n: int, local_scores: list[float]) -> list[tuple[str, str, float]]:
    return [(f"prompt-{i}", f"resp-{i}", local_scores[i]) for i in range(n)]


# ── Spearman correctness on a tiny known vector ──────────────────────────────


def test_spearman_perfect_monotonic():
    # Strictly increasing, non-linear → Spearman == 1.0 (Pearson would be <1).
    assert spearman([1, 2, 3, 4], [1, 4, 9, 16]) == 1.0


def test_spearman_perfect_inverse():
    assert spearman([1, 2, 3, 4], [4, 3, 2, 1]) == -1.0


def test_spearman_known_value():
    # Classic worked example: rho = 1 - 6*Σd²/(n(n²-1)) with no ties.
    # x ranks 1..5, y = [2,1,4,3,5] → d² sum = 1+1+1+1+0 = 4
    # rho = 1 - 6*4/(5*24) = 1 - 24/120 = 0.8
    rho = spearman([1, 2, 3, 4, 5], [2, 1, 4, 3, 5])
    assert abs(rho - 0.8) < 1e-9


def test_spearman_handles_ties():
    # All-equal one side → undefined variance → 0.0, not a crash / fabricated 1.
    assert spearman([1, 2, 3], [5, 5, 5]) == 0.0


# ── should_spotcheck ─────────────────────────────────────────────────────────


def test_marginal_always_spotchecked():
    decisive = 0.10
    # mean_delta 0.15 < 2*0.10=0.20 → marginal → always True.
    v = _verdict(mean_delta=0.15)
    assert is_marginal(v, decisive) is True
    assert should_spotcheck(v, decisive) is True


def test_nonmarginal_sampled_by_deterministic_hash():
    decisive = 0.10
    # Well above 2*decisive so NOT marginal — falls to the 10% hash bucket.
    # Same id → same outcome every call (reproducible, not random).
    v = _verdict(mean_delta=1.0, challenger_version="alpha-123")
    first = should_spotcheck(v, decisive)
    assert first == should_spotcheck(v, decisive)  # deterministic
    assert is_marginal(v, decisive) is False
    # A rate=1.0 forces every non-marginal accept in; rate=0.0 forces all out.
    assert should_spotcheck(v, decisive, rate=1.0) is True
    assert should_spotcheck(v, decisive, rate=0.0) is False


def test_known_id_in_and_out_of_sample():
    decisive = 0.10
    # Scan ids to pin a concrete in-sample and out-of-sample id at the 10% rate,
    # then assert those exact ids stay stable (guards the hash from drifting).
    in_id = out_id = None
    for i in range(200):
        cid = f"cand-{i}"
        v = _verdict(mean_delta=1.0, challenger_version=cid)
        if should_spotcheck(v, decisive):
            in_id = in_id or cid
        else:
            out_id = out_id or cid
        if in_id and out_id:
            break
    assert in_id is not None and out_id is not None
    assert should_spotcheck(_verdict(mean_delta=1.0, challenger_version=in_id), decisive) is True
    assert should_spotcheck(_verdict(mean_delta=1.0, challenger_version=out_id), decisive) is False


def test_reject_never_spotchecked():
    assert should_spotcheck(_verdict(accept=False, mean_delta=0.01), 0.10) is False


# ── rolling tracker ──────────────────────────────────────────────────────────


def test_tracker_correlated_does_not_freeze():
    t = DriftTracker(window=50, min_pairs=8)
    for i in range(10):
        t.record(float(i), float(i) + 0.1)  # near-perfect agreement
    assert t.correlation() > 0.9
    assert t.should_freeze() is False


def test_tracker_decorrelated_freezes():
    t = DriftTracker(window=50, min_pairs=8)
    for i in range(10):
        t.record(float(i), float(10 - i))  # inverted → strong negative corr
    assert t.correlation() < DEFAULT_CORR_THRESHOLD
    assert t.should_freeze() is True


def test_tracker_below_min_pairs_never_freezes():
    t = DriftTracker(window=50, min_pairs=8)
    for i in range(3):  # only 3 pairs, even if inverted
        t.record(float(i), float(3 - i))
    assert t.n_pairs == 3
    assert t.should_freeze() is False  # not enough evidence yet


def test_tracker_window_eviction():
    t = DriftTracker(window=4, min_pairs=2)
    for i in range(10):
        t.record(float(i), float(i))
    assert t.n_pairs == 4  # capped at window size — oldest evicted


def test_tracker_persists_across_instances(tmp_path):
    path = tmp_path / "spotcheck.jsonl"
    t1 = DriftTracker(path, window=50, min_pairs=2)
    t1.record(1.0, 2.0, challenger_version="v1")
    t1.record(2.0, 3.0, challenger_version="v2")
    # Fresh instance over the same file reloads the window from disk.
    t2 = DriftTracker(path, window=50, min_pairs=2)
    assert t2.n_pairs == 2


# ── governor wrapper ─────────────────────────────────────────────────────────


def test_governor_freezes_on_drifted_tracker():
    decisive = 0.10
    # Pre-load the tracker with inverted (drifted) pairs so it should freeze.
    tracker = DriftTracker(window=50, min_pairs=8)
    for i in range(10):
        tracker.record(float(i), float(10 - i))
    assert tracker.should_freeze() is True

    judge = FakeCloudJudge([3.0])
    base = _verdict(mean_delta=0.15)  # marginal → guaranteed spot-check
    challenger = {"_spotcheck_samples": _samples(3, [4.0, 4.0, 4.0])}

    def decide_fn(champion, challenger, thresholds=None):
        return base

    gated = cloud_spotcheck_gate(decide_fn, judge, tracker, decisive)
    out = gated({}, challenger)
    assert out.accept is False
    assert any("FROZEN" in r and "Spearman" in r for r in out.reject_reasons)
    assert len(judge.calls) == 3  # re-graded all 3 spot-check samples


def test_governor_passes_through_on_healthy_tracker():
    decisive = 0.10
    # Healthy: lots of well-correlated history.
    tracker = DriftTracker(window=50, min_pairs=8)
    for i in range(10):
        tracker.record(float(i), float(i))
    assert tracker.should_freeze() is False

    # Cloud agrees with the local scores → stays healthy after recording.
    judge = FakeCloudJudge([4.0, 4.0, 4.0])
    base = _verdict(mean_delta=0.15)  # marginal → spot-checked
    challenger = {"_spotcheck_samples": _samples(3, [4.0, 4.0, 4.0])}

    def decide_fn(champion, challenger, thresholds=None):
        return base

    gated = cloud_spotcheck_gate(decide_fn, judge, tracker, decisive)
    out = gated({}, challenger)
    assert out.accept is True
    assert out is not None
    assert len(judge.calls) == 3


def test_governor_reject_never_spotchecks():
    judge = FakeCloudJudge([0.0])
    tracker = DriftTracker(window=50, min_pairs=8)
    base = _verdict(accept=False, mean_delta=0.01)

    def decide_fn(champion, challenger, thresholds=None):
        return base

    gated = cloud_spotcheck_gate(decide_fn, judge, tracker, 0.10)
    out = gated({}, {"_spotcheck_samples": _samples(3, [1.0, 1.0, 1.0])})
    assert out.accept is False
    assert out is base  # passed straight through, untouched
    assert judge.calls == []  # cloud judge never called on a reject


def test_governor_nonmarginal_out_of_sample_passes_through():
    decisive = 0.10
    tracker = DriftTracker(window=50, min_pairs=8)
    # Find a non-marginal id that the hash leaves OUT of the 10% sample.
    out_id = None
    for i in range(200):
        cid = f"x-{i}"
        if not should_spotcheck(_verdict(mean_delta=1.0, challenger_version=cid), decisive):
            out_id = cid
            break
    assert out_id is not None

    judge = FakeCloudJudge([0.0])
    base = _verdict(mean_delta=1.0, challenger_version=out_id)

    def decide_fn(champion, challenger, thresholds=None):
        return base

    gated = cloud_spotcheck_gate(decide_fn, judge, tracker, decisive)
    out = gated({}, {"_spotcheck_samples": _samples(3, [1.0, 1.0, 1.0])})
    assert out is base  # not in sample → no cloud call, no change
    assert judge.calls == []
