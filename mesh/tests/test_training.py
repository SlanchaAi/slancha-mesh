"""TrainingPass tests (stub mode) — checkpoint round-trip + preempt cooperation.

Real PEFT lands in v0.0.5; this file locks the contract v0.0.4 ships:
- checkpoint schema (state_dict.json + meta.json under specialist/<ts>/)
- preempt cooperation via threading.Event
- corpus_hash reproducibility property
- load_checkpoint round-trip
"""

from __future__ import annotations

import json
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import pytest

from mesh.replay_store import TrafficReplayStore
from mesh.training import (
    FAST_FAKE_STEPS,
    CheckpointMeta,
    TrainingPass,
    _corpus_hash,
    _stub_state_dict,
    load_checkpoint,
)


def _store_with(n: int) -> TrafficReplayStore:
    s = TrafficReplayStore(max_size=1000)
    for i in range(n):
        s.add(
            prompt_text=f"prompt-{i}",
            oracle_response=f"response-{i}",
            domain="math",
            difficulty="medium",
        )
    return s


# ---------------------------------------------------------------------------
# Corpus hash
# ---------------------------------------------------------------------------


def test_corpus_hash_empty():
    assert _corpus_hash([]) == "sha256:empty"


def test_corpus_hash_deterministic_same_inputs():
    s1 = _store_with(5)
    s2 = _store_with(5)
    h1 = _corpus_hash(s1.recent(n=5))
    h2 = _corpus_hash(s2.recent(n=5))
    assert h1 == h2


def test_corpus_hash_order_invariant():
    """Same set of entries in different order → same hash (sorted by prompt_hash)."""
    s = _store_with(3)
    forward = s.recent(n=3)
    backward = list(reversed(forward))
    assert _corpus_hash(forward) == _corpus_hash(backward)


# ---------------------------------------------------------------------------
# Stub state dict
# ---------------------------------------------------------------------------


def test_stub_state_dict_deterministic():
    a = _stub_state_dict(seed=42, n_steps_completed=10, corpus_hash="sha256:x")
    b = _stub_state_dict(seed=42, n_steps_completed=10, corpus_hash="sha256:x")
    assert a == b


def test_stub_state_dict_varies_on_seed():
    a = _stub_state_dict(seed=1, n_steps_completed=10, corpus_hash="sha256:x")
    b = _stub_state_dict(seed=2, n_steps_completed=10, corpus_hash="sha256:x")
    assert a != b


def test_stub_state_dict_shape():
    """Locks the schema v0.0.5 PEFT must emit."""
    sd = _stub_state_dict(seed=0, n_steps_completed=0, corpus_hash="sha256:empty")
    assert set(sd.keys()) == {"lora_A", "lora_B", "rank", "alpha"}
    assert isinstance(sd["lora_A"], list) and len(sd["lora_A"]) == 2
    assert isinstance(sd["lora_B"], list) and len(sd["lora_B"]) == 2


# ---------------------------------------------------------------------------
# TrainingPass full run
# ---------------------------------------------------------------------------


def test_training_pass_runs_to_completion(tmp_path: Path):
    store = _store_with(20)
    tp = TrainingPass(
        specialist_id="qwen3-math-7b-q4",
        base_model_id="Qwen/Qwen3-Math-7B",
        domain="math",
        replay_store=store,
        checkpoint_dir=tmp_path,
        n_steps_planned=10,
        per_step_sleep_s=0,
    )
    out = tp.run()
    assert out.is_dir()
    assert tp.meta is not None
    assert tp.meta.preempted is False
    assert tp.meta.n_steps_completed == 10
    assert tp.meta.n_examples == 20


def test_training_pass_writes_state_and_meta(tmp_path: Path):
    store = _store_with(5)
    tp = TrainingPass(
        specialist_id="qwen3-coder-7b-q4",
        base_model_id="Qwen/Qwen3-Coder-7B",
        domain="code",
        replay_store=store,
        checkpoint_dir=tmp_path,
        n_steps_planned=5,
        per_step_sleep_s=0,
    )
    out = tp.run()
    state_path = out / "state_dict.json"
    meta_path = out / "meta.json"
    assert state_path.exists() and meta_path.exists()
    state = json.loads(state_path.read_text())
    meta_dict = json.loads(meta_path.read_text())
    assert "lora_A" in state
    assert meta_dict["specialist_id"] == "qwen3-coder-7b-q4"
    assert meta_dict["domain"] == "code"
    assert meta_dict["stub"] is True


def test_training_pass_load_checkpoint_roundtrip(tmp_path: Path):
    store = _store_with(3)
    tp = TrainingPass(
        specialist_id="phi-4-14b-q4",
        base_model_id="microsoft/phi-4-14b",
        domain="general",
        replay_store=store,
        checkpoint_dir=tmp_path,
        n_steps_planned=3,
        per_step_sleep_s=0,
        seed=99,
    )
    out = tp.run()
    state, meta = load_checkpoint(out)
    assert isinstance(meta, CheckpointMeta)
    assert meta.specialist_id == "phi-4-14b-q4"
    assert meta.seed == 99
    assert meta.n_steps_completed == 3
    assert state == _stub_state_dict(
        seed=99, n_steps_completed=3, corpus_hash=meta.corpus_hash
    )


def test_training_pass_default_steps_match_fast_fake():
    tp = TrainingPass(
        specialist_id="x",
        base_model_id="y",
        domain="general",
        replay_store=TrafficReplayStore(),
        checkpoint_dir=Path("/tmp"),
    )
    assert tp.n_steps_planned == FAST_FAKE_STEPS


# ---------------------------------------------------------------------------
# Preempt cooperation
# ---------------------------------------------------------------------------


def test_training_pass_preempt_yields_clean(tmp_path: Path):
    """Set preempt_event before run starts → training writes a checkpoint
    with preempted=True and n_steps_completed=0."""
    store = _store_with(5)
    tp = TrainingPass(
        specialist_id="qwen3-math-7b-q4",
        base_model_id="Qwen/Qwen3-Math-7B",
        domain="math",
        replay_store=store,
        checkpoint_dir=tmp_path,
        n_steps_planned=100,
        per_step_sleep_s=0,
    )
    ev = threading.Event()
    ev.set()  # already preempted before first step
    out = tp.run(preempt_event=ev)
    assert tp.meta is not None
    assert tp.meta.preempted is True
    assert tp.meta.n_steps_completed == 0
    # Checkpoint still written even on preempt-before-start
    assert (out / "meta.json").exists()


def test_training_pass_preempt_mid_loop(tmp_path: Path):
    """Race: start training, signal preempt from another thread, verify
    it stops early with preempted=True."""
    store = _store_with(10)
    tp = TrainingPass(
        specialist_id="qwen3-math-7b-q4",
        base_model_id="Qwen/Qwen3-Math-7B",
        domain="math",
        replay_store=store,
        checkpoint_dir=tmp_path,
        n_steps_planned=1000,  # would take ~1s at default sleep
        per_step_sleep_s=0.001,
    )
    ev = threading.Event()
    result: list[Path] = []

    def runner():
        result.append(tp.run(preempt_event=ev))

    t = threading.Thread(target=runner, daemon=True)
    t.start()
    time.sleep(0.05)  # let some steps run
    ev.set()
    t.join(timeout=2.0)

    assert tp.meta is not None
    assert tp.meta.preempted is True
    assert 0 < tp.meta.n_steps_completed < 1000
    assert len(result) == 1


def test_training_pass_no_preempt_event_runs_to_completion(tmp_path: Path):
    """Passing None for preempt_event should behave like a never-set event."""
    store = _store_with(2)
    tp = TrainingPass(
        specialist_id="x",
        base_model_id="y",
        domain="general",
        replay_store=store,
        checkpoint_dir=tmp_path,
        n_steps_planned=5,
        per_step_sleep_s=0,
    )
    out = tp.run(preempt_event=None)
    assert tp.meta is not None
    assert tp.meta.preempted is False
    assert tp.meta.n_steps_completed == 5
