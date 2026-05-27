"""Tests for mesh.grading.grade_replay_corpus — GRADING → both sinks.

Mirrors the repo's test doubles: a real (in-memory) TrafficReplayStore, a
fake Scorer that records calls (and can raise ScorerError), and a fake
registry that captures record_quality_observation calls.
"""

from __future__ import annotations

import pytest

from mesh.grading import grade_replay_corpus
from mesh.quality_probe import ProbePrompt, ScorerError
from mesh.replay_store import TrafficReplayStore, _hash_prompt


# ── Test doubles ─────────────────────────────────────────────────────────────


class _RecordingScorer:
    """Scorer double: returns a fixed score (or per-response override),
    raises ScorerError for any response in `raise_on`, and records every
    (probe, response_text) it was asked to grade."""

    def __init__(self, score=4.0, score_map=None, raise_on=()):
        self._score = score
        self._score_map = score_map or {}
        self.raise_on = set(raise_on)
        self.calls: list[tuple[ProbePrompt, str]] = []

    def score(self, probe: ProbePrompt, response_text: str) -> float:
        self.calls.append((probe, response_text))
        if response_text in self.raise_on:
            raise ScorerError(f"judge down on {response_text!r}")
        return self._score_map.get(response_text, self._score)


class _FakeRegistry:
    """Captures record_quality_observation calls; return value is ignored
    by grade_replay_corpus so (None, None) is fine."""

    def __init__(self):
        self.calls: list[dict] = []

    def record_quality_observation(
        self, *, specialist_id, score, sample_count, observation_source, observed_at=None
    ):
        self.calls.append(
            {
                "specialist_id": specialist_id,
                "score": score,
                "sample_count": sample_count,
                "observation_source": observation_source,
            }
        )
        return None, None


def _entry(store: TrafficReplayStore, prompt_text: str):
    """Fetch the stored entry by prompt_text (store yields a snapshot)."""
    return next(e for e in store if e.prompt_text == prompt_text)


# ── Sink (b): per-entry judge label ──────────────────────────────────────────


def test_stamps_judge_fields_on_graded_entries():
    s = TrafficReplayStore(max_size=10)
    s.add("p1", "r1", "code", "easy", served_by_specialist="a")
    out = grade_replay_corpus(s, _RecordingScorer(score=4.0), _FakeRegistry())

    e = _entry(s, "p1")
    assert e.judge_score == 4.0
    assert e.judge_source == "_RecordingScorer"
    assert out["graded"] == 1


def test_judge_source_override():
    s = TrafficReplayStore(max_size=10)
    s.add("p1", "r1", "code", "easy", served_by_specialist="a")
    grade_replay_corpus(
        s, _RecordingScorer(), _FakeRegistry(), judge_source="local-judge:qwen3-7b"
    )
    assert _entry(s, "p1").judge_source == "local-judge:qwen3-7b"


def test_probe_built_from_entry_and_oracle_response_is_graded():
    s = TrafficReplayStore(max_size=10)
    s.add("the prompt", "the oracle answer", "code", "easy", served_by_specialist="a")
    scorer = _RecordingScorer()
    grade_replay_corpus(s, scorer, _FakeRegistry())

    probe, response = scorer.calls[0]
    assert probe.text == "the prompt"
    assert probe.domain == "code"
    assert probe.reference is None  # captured traffic has no gold reference
    assert response == "the oracle answer"  # the oracle_response is graded


def test_already_graded_entries_are_skipped():
    s = TrafficReplayStore(max_size=10)
    pre = s.add("p1", "r1", "code", "easy", served_by_specialist="a")
    pre.judge_score = 3.0  # pre-graded
    s.add("p2", "r2", "code", "easy", served_by_specialist="a")

    scorer = _RecordingScorer()
    out = grade_replay_corpus(s, scorer, _FakeRegistry())

    assert out["graded"] == 1
    assert [resp for (_p, resp) in scorer.calls] == ["r2"]  # r1 not re-scored
    assert _entry(s, "p1").judge_score == 3.0  # untouched


# ── Sink (a): registry observation ───────────────────────────────────────────


def test_one_observation_per_specialist_with_mean_and_count():
    s = TrafficReplayStore(max_size=10)
    s.add("p1", "r1", "code", "easy", served_by_specialist="a")
    s.add("p2", "r2", "code", "easy", served_by_specialist="a")
    s.add("p3", "r3", "writing", "easy", served_by_specialist="b")
    scorer = _RecordingScorer(score_map={"r1": 4.0, "r2": 2.0, "r3": 5.0})
    reg = _FakeRegistry()

    out = grade_replay_corpus(s, scorer, reg)

    by_sid = {c["specialist_id"]: c for c in reg.calls}
    assert by_sid["a"]["score"] == pytest.approx(3.0)  # (4+2)/2
    assert by_sid["a"]["sample_count"] == 2
    assert by_sid["b"]["score"] == pytest.approx(5.0)
    assert by_sid["b"]["sample_count"] == 1
    assert out["specialists"]["a"] == {"score": pytest.approx(3.0), "sample_count": 2}


def test_observation_source_defaults_to_real_traffic():
    s = TrafficReplayStore(max_size=10)
    s.add("p1", "r1", "code", "easy", served_by_specialist="a")
    reg = _FakeRegistry()
    grade_replay_corpus(s, _RecordingScorer(), reg)
    assert reg.calls[0]["observation_source"] == "real_traffic"


def test_observation_source_override():
    s = TrafficReplayStore(max_size=10)
    s.add("p1", "r1", "code", "easy", served_by_specialist="a")
    reg = _FakeRegistry()
    grade_replay_corpus(s, _RecordingScorer(), reg, observation_source="shadow_traffic")
    assert reg.calls[0]["observation_source"] == "shadow_traffic"


def test_entry_without_specialist_is_stamped_but_no_observation():
    s = TrafficReplayStore(max_size=10)
    s.add("p1", "r1", "code", "easy")  # served_by_specialist=None
    reg = _FakeRegistry()

    out = grade_replay_corpus(s, _RecordingScorer(score=4.0), reg)

    assert out["graded"] == 1
    assert out["skipped_no_specialist"] == 1
    assert reg.calls == []  # nobody to credit
    assert _entry(s, "p1").judge_score == 4.0  # sink (b) still applied


# ── ScorerError handling ─────────────────────────────────────────────────────


def test_scorer_error_skips_entry_and_does_not_abort_batch():
    s = TrafficReplayStore(max_size=10)
    s.add("p1", "bad", "code", "easy", served_by_specialist="a")
    s.add("p2", "good", "code", "easy", served_by_specialist="a")
    scorer = _RecordingScorer(score=4.0, raise_on={"bad"})
    reg = _FakeRegistry()

    out = grade_replay_corpus(s, scorer, reg)

    assert out["graded"] == 1  # only the good entry
    assert len(out["errors"]) == 1
    assert out["errors"][0]["prompt_hash"] == _hash_prompt("p1")
    assert _entry(s, "p1").judge_score is None  # broken judge != a 0
    assert _entry(s, "p2").judge_score == 4.0
    # observation reflects only the successfully-graded entry
    assert reg.calls[0]["sample_count"] == 1
    assert reg.calls[0]["score"] == pytest.approx(4.0)


def test_all_entries_error_writes_no_observation():
    s = TrafficReplayStore(max_size=10)
    s.add("p1", "x", "code", "easy", served_by_specialist="a")
    reg = _FakeRegistry()
    out = grade_replay_corpus(s, _RecordingScorer(raise_on={"x"}), reg)
    assert out["graded"] == 0
    assert reg.calls == []
    assert len(out["errors"]) == 1


# ── domain filter + limit + empty ────────────────────────────────────────────


def test_domain_filter_only_grades_matching_entries():
    s = TrafficReplayStore(max_size=10)
    s.add("p1", "r1", "code", "easy", served_by_specialist="a")
    s.add("p2", "r2", "writing", "easy", served_by_specialist="b")
    out = grade_replay_corpus(s, _RecordingScorer(), _FakeRegistry(), domain="code")
    assert out["graded"] == 1
    assert set(out["specialists"]) == {"a"}
    assert _entry(s, "p2").judge_score is None  # filtered out


def test_limit_grades_only_most_recent():
    s = TrafficReplayStore(max_size=10)
    s.add("p1", "r1", "code", "easy", served_by_specialist="a")
    s.add("p2", "r2", "code", "easy", served_by_specialist="a")
    s.add("p3", "r3", "code", "easy", served_by_specialist="a")
    out = grade_replay_corpus(s, _RecordingScorer(), _FakeRegistry(), limit=2)
    assert out["graded"] == 2  # newest two (p3, p2)
    assert _entry(s, "p1").judge_score is None  # oldest, beyond limit


def test_empty_store_returns_empty_summary():
    out = grade_replay_corpus(TrafficReplayStore(max_size=10), _RecordingScorer(), _FakeRegistry())
    assert out == {"specialists": {}, "graded": 0, "skipped_no_specialist": 0, "errors": []}
