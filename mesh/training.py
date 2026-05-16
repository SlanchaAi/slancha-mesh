"""LoRA training pass — STUB for v0.0.4 — spec §7 paragraph 3.

This module ships the *contract* for idle fine-tune passes without
running real PEFT. v0.0.4 validates:

  * checkpoint round-trip (state-dict serialization, deterministic
    seed-based content for byte-identity reproducibility)
  * preempt cooperation (training loop polls threading.Event every step)
  * traffic-replay-store integration point (training reads from store)
  * checkpoint metadata schema (so v0.0.5's real PEFT path can be
    consumed by the same downstream merger)

v0.0.5 will replace `_apply_lora_step` with a real PEFT-driven update
against the loaded base model. Everything else (preempt, checkpoint
schema, store interface) is locked here.

Why a stub at all: idle fine-tune is the spec §7 moat, but a real PEFT
pass needs GPU memory not currently shared with vLLM serving + a
training dataloader that captures cloud-grader feedback we don't
collect yet. Stubbing the wrapper now lets us:

  1. Lock the contract between detector → training → checkpoint merger
  2. Test the preempt / yield path under threading conditions
  3. Validate checkpoint schema + round-trip without GPU
  4. Let the daemon-side integration (spawning training threads on
     detector edges) land in v0.0.4 against a deterministic stub
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from mesh.replay_store import ReplayEntry, TrafficReplayStore


# Spec §7 paragraph 3 mentions ~30min real passes with checkpoint every
# 100 steps. The stub fakes this with FAST_FAKE_STEPS so tests run in
# milliseconds. Real PEFT lands at v0.0.5 with the real constants.
FAST_FAKE_STEPS: int = 20
PER_STEP_SLEEP_S: float = 0.001  # 1ms per step → ~20ms total stub training


@dataclass
class CheckpointMeta:
    """Per-checkpoint metadata; persisted alongside the (stub) state-dict.

    The schema is locked in v0.0.4 so v0.0.5's real PEFT path produces
    the same JSON shape and the downstream merger (v0.0.6) reads both
    stub + real checkpoints interchangeably.
    """

    specialist_id: str
    base_model_id: str
    domain: str
    seed: int
    n_examples: int
    n_steps_completed: int
    n_steps_planned: int
    started_at: datetime
    finished_at: datetime
    preempted: bool
    # Hash of (sorted prompt_hashes) used for this pass — gives a
    # reproducibility key for the training corpus.
    corpus_hash: str
    # v0.0.4 stub marker; real PEFT bumps to a version number.
    stub: bool = True

    def to_json(self) -> dict:
        d = asdict(self)
        d["started_at"] = self.started_at.isoformat()
        d["finished_at"] = self.finished_at.isoformat()
        return d

    @classmethod
    def from_json(cls, d: dict) -> "CheckpointMeta":
        d = dict(d)
        d["started_at"] = datetime.fromisoformat(d["started_at"])
        d["finished_at"] = datetime.fromisoformat(d["finished_at"])
        return cls(**d)


def _corpus_hash(examples: list[ReplayEntry]) -> str:
    """Deterministic SHA-256 of sorted prompt_hashes — corpus identity."""
    if not examples:
        return "sha256:empty"
    joined = "\n".join(sorted(e.prompt_hash for e in examples))
    return "sha256:" + hashlib.sha256(joined.encode("utf-8")).hexdigest()


def _stub_state_dict(seed: int, n_steps_completed: int, corpus_hash: str) -> dict:
    """Deterministic fake state-dict.

    Real PEFT would return torch tensors; the stub returns a JSON-safe
    dict shaped like a small LoRA delta. Byte-identity for fixed seed
    + step count + corpus → reproducibility property for tests.
    """
    # Tiny deterministic "weight delta" — 4 floats derived from inputs.
    # No randomness; same inputs → same output.
    key = f"{seed}|{n_steps_completed}|{corpus_hash}".encode("utf-8")
    h = hashlib.sha256(key).digest()
    floats = [int.from_bytes(h[i : i + 4], "big") / 2**32 for i in (0, 4, 8, 12)]
    return {
        "lora_A": floats[:2],
        "lora_B": floats[2:],
        "rank": 8,
        "alpha": 16,
    }


@dataclass
class TrainingPass:
    """One end-to-end fine-tune pass (stub-only in v0.0.4).

    Args at construct time:
      specialist_id, base_model_id, domain   — what we're training
      replay_store                            — corpus source
      seed                                    — reproducibility
      n_examples                              — how many recent entries to pull
      n_steps_planned                         — how many "fake" steps to take
      checkpoint_dir                          — where to write state + meta

    Call `run(preempt_event)` from a background thread. It polls the
    event each step; on set, it checkpoints + returns with
    `preempted=True` in the metadata.
    """

    specialist_id: str
    base_model_id: str
    domain: str
    replay_store: TrafficReplayStore
    checkpoint_dir: Path
    seed: int = 0
    n_examples: int = 64
    n_steps_planned: int = FAST_FAKE_STEPS
    per_step_sleep_s: float = PER_STEP_SLEEP_S

    _meta: CheckpointMeta | None = field(default=None, init=False)

    def run(self, preempt_event: threading.Event | None = None) -> Path:
        """Execute the (stub) training pass; return path to checkpoint dir.

        `preempt_event`: if set mid-loop, training yields cleanly with
        a checkpoint reflecting steps-completed-so-far + preempted=True.
        Pass `None` to disable preemption (tests that don't need it).
        """
        if preempt_event is None:
            preempt_event = threading.Event()  # never set; loops to completion

        examples = self.replay_store.recent(n=self.n_examples, domain=self.domain)
        corpus_h = _corpus_hash(examples)

        started_at = datetime.now(timezone.utc)
        steps_completed = 0
        preempted = False
        for step in range(self.n_steps_planned):
            if preempt_event.is_set():
                preempted = True
                break
            self._apply_lora_step(step, examples)
            steps_completed = step + 1

        finished_at = datetime.now(timezone.utc)
        self._meta = CheckpointMeta(
            specialist_id=self.specialist_id,
            base_model_id=self.base_model_id,
            domain=self.domain,
            seed=self.seed,
            n_examples=len(examples),
            n_steps_completed=steps_completed,
            n_steps_planned=self.n_steps_planned,
            started_at=started_at,
            finished_at=finished_at,
            preempted=preempted,
            corpus_hash=corpus_h,
        )
        return self._write_checkpoint()

    def _apply_lora_step(self, step: int, examples: list[ReplayEntry]) -> None:
        """STUB: real PEFT call lands in v0.0.5.

        Sleep is the only side-effect — simulates "this step took time"
        so preempt-mid-training tests can race signal_preempt against
        the loop. Real PEFT would do forward + backward + optimizer
        step against `examples` here.
        """
        time.sleep(self.per_step_sleep_s)

    def _write_checkpoint(self) -> Path:
        """Persist state-dict + metadata under checkpoint_dir/<corpus-hash>/.

        Path structure:
          checkpoint_dir/
            <specialist_id>/<started_at_isoformat>/
              state_dict.json
              meta.json
        """
        assert self._meta is not None
        sub = (
            self.checkpoint_dir
            / self.specialist_id
            / self._meta.started_at.isoformat().replace(":", "-")
        )
        sub.mkdir(parents=True, exist_ok=True)
        state = _stub_state_dict(
            self.seed, self._meta.n_steps_completed, self._meta.corpus_hash
        )
        (sub / "state_dict.json").write_text(json.dumps(state, indent=2))
        (sub / "meta.json").write_text(json.dumps(self._meta.to_json(), indent=2))
        return sub

    @property
    def meta(self) -> CheckpointMeta | None:
        return self._meta


def load_checkpoint(checkpoint_path: Path) -> tuple[dict, CheckpointMeta]:
    """Read a checkpoint directory written by `TrainingPass._write_checkpoint`.

    Returns `(state_dict, meta)`. v0.0.6 merger will consume this same
    pair; real PEFT (v0.0.5) writes the same shape so the merger
    doesn't need to know whether a checkpoint is stub or real (the
    `meta.stub` flag is informational).
    """
    state = json.loads((checkpoint_path / "state_dict.json").read_text())
    meta = CheckpointMeta.from_json(
        json.loads((checkpoint_path / "meta.json").read_text())
    )
    return state, meta


__all__ = [
    "CheckpointMeta",
    "FAST_FAKE_STEPS",
    "PER_STEP_SLEEP_S",
    "TrainingPass",
    "load_checkpoint",
]
