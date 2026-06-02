"""LoRA training pass — stub contract + real PEFT leg — spec §7 paragraph 3.

This module ships two paths behind one `TrainingPass`:

  * the **stub contract** (v0.0.4, issue #55): `allow_stub=True` runs a
    deterministic, GPU-free pass that locks checkpoint schema, preempt
    cooperation, corpus-hash reproducibility, and the round-trip the
    downstream merger consumes. It writes placeholder weights
    (`meta.stub=True`) and the eval gate refuses to promote it.
  * the **real PEFT leg** (issue #65): `allow_stub=False` + a real base
    model id → `run()` loads the base model + tokenizer via
    `transformers`, wraps it with a `peft` `LoraConfig`, trains a few
    steps on the ReplayEntry corpus, and saves a real adapter via
    `peft`'s `save_pretrained`. It emits `meta.stub=False` with a real
    `base_model_fingerprint`, so the eval gate (`mesh/eval/gate.py`) can
    promote it once it passes the held-out non-regression check.

The heavy deps (`torch`/`transformers`/`peft`/`datasets`) are an
**optional extra** — `pip install -e ".[train]"`. They are imported
**lazily inside the real path** so the package still imports and the
test suite still runs without them installed. If the real path is taken
without them present, it raises `MissingTrainingDepsError` (a clear,
actionable error) rather than silently falling back to the stub — a
silent stub fallback is exactly the failure issue #55 fixed.

What is locked across both paths (so the merger doesn't care which ran):
the checkpoint directory layout, the `CheckpointMeta` JSON schema, the
preempt/yield path (the loop polls a `threading.Event` each step), and
the corpus-hash reproducibility key.

Why the stub still exists: idle fine-tune is the spec §7 moat, but the
real pass needs a GPU; the stub lets the daemon-side integration (preempt,
checkpoint, merge) be tested deterministically without one. Full
GB10/DGX-Spark validation of the real leg under `gpu-launch --kind
training` is the remaining acceptance step for #65.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
import warnings
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from mesh.replay_store import ReplayEntry, TrafficReplayStore


# Spec §7 paragraph 3 mentions ~30min real passes with checkpoint every
# 100 steps. The stub fakes this with FAST_FAKE_STEPS so tests run in
# milliseconds. Real PEFT lands at v0.0.5 with the real constants.
FAST_FAKE_STEPS: int = 20
PER_STEP_SLEEP_S: float = 0.001  # 1ms per step → ~20ms total stub training


class StubTrainingError(RuntimeError):
    """Raised when `TrainingPass.run()` is called without opting into the stub.

    The stub performs no real PEFT (issue #55); real PEFT is tracked in
    issue #65. Refusing by default keeps a demo from treating the stub's
    seed-derived checkpoint as a real improved model and promoting it.
    """


class MissingTrainingDepsError(RuntimeError):
    """Raised when the real PEFT path is taken but `[train]` deps are absent.

    The real leg imports torch/transformers/peft lazily. If they are not
    installed, we refuse loudly with an actionable message rather than
    falling back to the stub — a silent-stub fallback is exactly the
    failure issue #55 fixed (a stub checkpoint masquerading as a real
    adapter). Install with: pip install -e ".[train]" (issue #65).
    """


class RealTrainingError(RuntimeError):
    """Raised when the real PEFT path is misconfigured (e.g. empty corpus).

    Distinct from MissingTrainingDepsError (deps present, but the pass
    cannot run meaningfully) and StubTrainingError (stub not opted in).
    """


def _base_model_fingerprint(base_model_id: str, config: dict | None = None) -> str:
    """Deterministic fingerprint of (base_model_id + its config).

    The eval/promotion gate (issue #57 provenance) refuses to load an
    adapter whose base differs from the champion's — a base/adapter
    mismatch is silent garbage (SELF_ORGANIZING_LOOP_SCOPE "Failure
    modes"). This stamps a stable identity into CheckpointMeta so the
    gate can compare bases. `config` is the resolved model config dict
    when available (architecture + hidden size etc.); when absent we
    fingerprint the id alone (still stable per base).
    """
    h = hashlib.sha256()
    h.update(base_model_id.encode("utf-8"))
    if config:
        # Sort keys so the fingerprint is order-independent + JSON-stable.
        h.update(json.dumps(config, sort_keys=True, default=str).encode("utf-8"))
    return f"{base_model_id}@sha256:{h.hexdigest()[:16]}"


def _import_train_deps():
    """Lazily import the `[train]` extra; raise MissingTrainingDepsError if absent.

    Imported inside the real path so the package + test suite work without
    torch/transformers/peft installed. Returns `(torch, transformers, peft)`.
    """
    try:
        import peft  # type: ignore
        import torch  # type: ignore
        import transformers  # type: ignore
    except ImportError as e:  # pragma: no cover - exercised via monkeypatch in tests
        raise MissingTrainingDepsError(
            "the real PEFT/LoRA training leg (issue #65) requires the optional "
            "'[train]' dependencies (torch, transformers, peft, datasets). They "
            "are not installed. Install them with:\n"
            '    pip install -e ".[train]"\n'
            "Refusing to fall back to the stub (that would re-introduce the "
            f"silent-stub problem #55 fixed). Original import error: {e}"
        ) from e
    return torch, transformers, peft


@dataclass(frozen=True)
class ImprovementRationale:
    """Human-readable WHY a challenger adapter was built (issue #80).

    Prior art: hexo-ai/sia writes an `improvement.md` per generation. The
    #57 provenance HASHES say *what was compared* (artifact / corpus /
    base-model / holdout identities); this says *why the thing exists and
    what it set out to fix* — the plain-language complement, auditable
    without reading logs (mirrors the GATE-CONTRACT "every verdict explains
    itself" goal). Structured (not freeform) so the producer is forced to
    state a falsifiable expectation, not just narrate.
    """

    hypothesis: str        # the cluster/traffic signal that motivated the build
    change_summary: str     # what the challenger changed vs the champion
    expected_effect: str    # the measurable lift it was built to produce


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
    # v0.0.4 stub marker; real PEFT (issue #65) sets stub=False.
    stub: bool = True
    # Fingerprint of (base_model_id + resolved config) — issue #57/#65
    # provenance. None on legacy stub checkpoints written before #65; the
    # real PEFT path always stamps it so the gate can compare bases and
    # refuse base/adapter mismatch. Optional so from_json on old rows works.
    base_model_fingerprint: str | None = None
    # Human-readable improvement rationale (issue #80). None on stub /
    # legacy checkpoints; the real PEFT path sets it from the cluster signal
    # that motivated the build. Additive + optional so from_json on old
    # checkpoints (which lack the key) still works.
    rationale: ImprovementRationale | None = None

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
        # asdict() flattened a nested ImprovementRationale to a plain dict on
        # write; rebuild it. Absent (old checkpoints) → stays None.
        r = d.get("rationale")
        if isinstance(r, dict):
            d["rationale"] = ImprovementRationale(**r)
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
    # Explicit opt-in required to run the contract-only stub (issue #55).
    # Defaults False so a caller can never accidentally execute a pass that
    # does no real PEFT and writes a checkpoint mistakable for a real adapter.
    allow_stub: bool = False
    # Explicit opt-in to the REAL PEFT/LoRA leg (issue #65). Kept separate
    # from `allow_stub` so the #55 default-refuse contract is preserved: with
    # neither flag set, run() still raises StubTrainingError (a caller must
    # consciously pick the stub *or* the real path — never get one implicitly).
    allow_real: bool = False

    # ── Real-path (issue #65) knobs; ignored by the stub path ──────────────
    # LoRA hyperparameters for the real `peft` LoraConfig.
    lora_rank: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.0
    learning_rate: float = 1e-4
    max_seq_len: int = 128

    # Human-readable rationale for THIS build (issue #80). Optional; when
    # given, the real PEFT path stamps it onto CheckpointMeta so the eval
    # gate verdict can explain why the challenger exists in plain language.
    # The stub path ignores it (a stub is never a real improvement).
    rationale: ImprovementRationale | None = None

    _meta: CheckpointMeta | None = field(default=None, init=False)

    def run(self, preempt_event: threading.Event | None = None) -> Path:
        """Execute the training pass; return path to the checkpoint dir.

        Dispatches on `allow_stub`:
          * `allow_stub=True`  → the contract-only stub (issue #55): no real
            PEFT, deterministic placeholder weights, meta.stub=True.
          * `allow_stub=False` → the real PEFT/LoRA leg (issue #65): loads
            the base model, trains a few LoRA steps on the replay corpus,
            saves a real adapter, meta.stub=False. Requires the `[train]`
            extra; absent it, raises MissingTrainingDepsError (never a
            silent stub fallback).

        `preempt_event`: if set mid-loop, training yields cleanly with a
        checkpoint reflecting steps-completed-so-far + preempted=True. Pass
        `None` to disable preemption (tests that don't need it).
        """
        if preempt_event is None:
            preempt_event = threading.Event()  # never set; loops to completion
        if self.allow_stub:
            return self._run_stub(preempt_event)
        if self.allow_real:
            return self._run_real(preempt_event)
        # Refuse to run silently: this is a contract-only stub by default
        # (issue #55). It performs no real PEFT — real training is issue #65.
        # A caller must explicitly construct with allow_stub=True (knowing the
        # checkpoint holds placeholder weights, not a trained adapter) OR with
        # allow_real=True (the real PEFT leg, which needs the [train] extra).
        raise StubTrainingError(
            "TrainingPass is a contract-only STUB: it performs no real "
            "PEFT and the checkpoint it would write contains placeholder "
            "weights (meta.stub=True), not a trained adapter. Refusing to "
            "run. Construct TrainingPass(..., allow_stub=True) to run it "
            "knowingly as a stub, or TrainingPass(..., allow_real=True) for "
            "the real PEFT leg. Real PEFT is tracked in issue #65."
        )

    def _run_stub(self, preempt_event: threading.Event) -> Path:
        """The v0.0.4 contract-only stub path (issue #55). No real PEFT."""
        # Loud tripwire: this is the v0.0.4 stub. _apply_lora_step does no
        # real PEFT, so the checkpoint written below holds seed-derived
        # placeholder weights (meta.stub=True), not a trained adapter. The
        # docstrings say so, but a caller wiring this into a real loop needs
        # a runtime signal too — don't let a stub pass masquerade as training.
        warnings.warn(
            "TrainingPass is a v0.0.4 STUB: it performs no real PEFT and the "
            "checkpoint it writes contains placeholder weights (meta.stub=True), "
            "not a trained adapter. Do not treat the result as a real quality "
            "improvement.",
            stacklevel=3,
        )

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
            stub=True,
        )
        return self._write_checkpoint()

    def _run_real(self, preempt_event: threading.Event) -> Path:
        """The real PEFT/LoRA leg (issue #65).

        Lazily imports the `[train]` extra; loads the base model + tokenizer,
        wraps with a `peft` LoraConfig, trains `n_steps_planned` steps on the
        replay corpus (polling `preempt_event` each step so the mesh-drain /
        idle-yield path still works), and saves a real adapter via
        `save_pretrained`. Emits meta.stub=False + a base_model_fingerprint
        so the eval gate can promote it.
        """
        torch, transformers, peft = _import_train_deps()

        examples = self.replay_store.recent(n=self.n_examples, domain=self.domain)
        if not examples:
            raise RealTrainingError(
                f"real training pass for domain={self.domain!r} has an empty "
                "corpus; nothing to fine-tune. Capture replay traffic first."
            )
        corpus_h = _corpus_hash(examples)

        torch.manual_seed(self.seed)

        tokenizer = transformers.AutoTokenizer.from_pretrained(self.base_model_id)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = transformers.AutoModelForCausalLM.from_pretrained(self.base_model_id)
        model.train()

        fingerprint = _base_model_fingerprint(
            self.base_model_id, dict(getattr(model.config, "to_dict", dict)())
        )

        lora_cfg = peft.LoraConfig(
            r=self.lora_rank,
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout,
            task_type=peft.TaskType.CAUSAL_LM,
        )
        model = peft.get_peft_model(model, lora_cfg)

        # Issue #65: train on the GPU when one is present (the GB10/Spark
        # target) — `from_pretrained` lands on CPU by default, so without this
        # the "real" pass silently runs on CPU even on a GB10. Falls back to
        # CPU so the contract tests + CPU dev machines behave unchanged.
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)

        optimizer = torch.optim.AdamW(
            (p for p in model.parameters() if p.requires_grad),
            lr=self.learning_rate,
        )

        # Build a tiny supervised corpus: prompt + oracle response, masked so
        # loss is over the full sequence (good enough for a contract-real
        # tiny-model pass; the FT-bundle projection refines this later).
        texts = [f"{e.prompt_text}\n{e.oracle_response}" for e in examples]

        started_at = datetime.now(timezone.utc)
        steps_completed = 0
        preempted = False
        for step in range(self.n_steps_planned):
            if preempt_event.is_set():
                preempted = True
                break
            text = texts[step % len(texts)]
            batch = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_seq_len,
                padding=False,
            )
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()
            out = model(input_ids=batch["input_ids"], labels=batch["input_ids"])
            out.loss.backward()
            optimizer.step()
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
            stub=False,
            base_model_fingerprint=fingerprint,
            rationale=self.rationale,
        )
        return self._write_real_checkpoint(model)

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

    def _write_real_checkpoint(self, peft_model) -> Path:
        """Persist a REAL peft adapter + metadata (issue #65).

        Mirrors the stub layout (checkpoint_dir/<specialist>/<ts>/) but the
        artifact is a real peft adapter directory written via
        `save_pretrained` (adapter_config.json + adapter_model.safetensors)
        instead of the stub's state_dict.json. meta.json carries
        stub=False + base_model_fingerprint so the eval gate can promote it.
        """
        assert self._meta is not None
        sub = (
            self.checkpoint_dir
            / self.specialist_id
            / self._meta.started_at.isoformat().replace(":", "-")
        )
        sub.mkdir(parents=True, exist_ok=True)
        # peft writes the real adapter (config + safetensors) under sub/.
        peft_model.save_pretrained(str(sub))
        (sub / "meta.json").write_text(json.dumps(self._meta.to_json(), indent=2))
        return sub

    @property
    def meta(self) -> CheckpointMeta | None:
        return self._meta


def load_checkpoint(checkpoint_path: Path) -> tuple[dict, CheckpointMeta]:
    """Read a checkpoint directory written by `TrainingPass._write_checkpoint`.

    Returns `(state_dict, meta)`. For a STUB checkpoint `state_dict` is the
    JSON placeholder dict. For a REAL adapter (issue #65) there is no
    `state_dict.json` — the artifact is a peft adapter on disk — so `state`
    comes back as a small descriptor `{"adapter_dir": <path>}` and the
    caller loads the adapter via peft. The `meta.stub` flag distinguishes
    the two; the downstream merger reads both via this one entrypoint.
    """
    meta = CheckpointMeta.from_json(
        json.loads((checkpoint_path / "meta.json").read_text())
    )
    state_json = checkpoint_path / "state_dict.json"
    if state_json.exists():
        state = json.loads(state_json.read_text())
    else:
        # Real peft adapter: no JSON state dict; point at the adapter dir.
        state = {"adapter_dir": str(checkpoint_path)}
    return state, meta


def load_meta(checkpoint_path: Path) -> CheckpointMeta:
    """Read just the CheckpointMeta from a checkpoint dir (stub or real)."""
    return CheckpointMeta.from_json(
        json.loads((checkpoint_path / "meta.json").read_text())
    )


class ChampionRegistry:
    """Tracks the current champion adapter + supports rollback (issue #65).

    The self-organizing loop promotes a challenger adapter only if it passes
    the eval gate; if promotion fails (gate reject, hot-swap error, or a
    post-promote regression) we must restore the prior champion. This is the
    "adapters as pointers → instant rollback" invariant in
    SELF_ORGANIZING_LOOP_SCOPE.

    Filesystem-backed, no GPU/model needed:
      registry_dir/
        champion.json        → {"checkpoint": <path>, "promoted_at": ...}
        champion.prev.json    → snapshot of the prior champion (for rollback)

    A checkpoint is just a directory path (the meta.json lives inside it).
    The registry stores *pointers*, never copies the adapter weights — drop
    the pointer to roll back, exactly the cheap-rollback property the design
    relies on.
    """

    def __init__(self, registry_dir: Path) -> None:
        self.registry_dir = Path(registry_dir)
        self.registry_dir.mkdir(parents=True, exist_ok=True)
        self._current = self.registry_dir / "champion.json"
        self._prev = self.registry_dir / "champion.prev.json"

    def current(self) -> Path | None:
        """Return the current champion checkpoint path, or None if unset."""
        if not self._current.exists():
            return None
        return Path(json.loads(self._current.read_text())["checkpoint"])

    def previous(self) -> Path | None:
        """Return the prior champion checkpoint path, or None if none kept."""
        if not self._prev.exists():
            return None
        return Path(json.loads(self._prev.read_text())["checkpoint"])

    def _write_pointer(self, path: Path, checkpoint: Path) -> None:
        path.write_text(
            json.dumps(
                {
                    "checkpoint": str(checkpoint),
                    "promoted_at": datetime.now(timezone.utc).isoformat(),
                },
                indent=2,
            )
        )

    def promote(self, challenger_checkpoint: Path, *, verify_real: bool = True) -> Path:
        """Make `challenger_checkpoint` the champion; keep the prior for rollback.

        `verify_real` (default True): refuse to promote a stub checkpoint —
        mirrors the eval gate's stub-rejection (issue #55) so the registry
        can't be made to point at placeholder weights. Set False only for
        tests that deliberately exercise the pointer mechanics with fakes.
        """
        challenger_checkpoint = Path(challenger_checkpoint)
        if verify_real:
            meta = load_meta(challenger_checkpoint)
            if meta.stub:
                raise StubTrainingError(
                    f"refusing to promote stub checkpoint {challenger_checkpoint} "
                    "to champion (meta.stub=True). Stub artifacts hold "
                    "placeholder weights, not a trained adapter (issue #55)."
                )
        # Snapshot the current champion so rollback can restore it.
        if self._current.exists():
            self._prev.write_text(self._current.read_text())
        elif self._prev.exists():
            # No current champion to back up → clear any stale prev so a
            # rollback after a first-ever promote is a clean no-op.
            self._prev.unlink()
        self._write_pointer(self._current, challenger_checkpoint)
        return challenger_checkpoint

    def rollback(self) -> Path | None:
        """Restore the prior champion. Returns the restored path, or None.

        Idempotent-ish: with no prior champion (first-ever promote just
        happened) this clears the current pointer and returns None — the
        mesh falls back to the base model (adapters-as-pointers: drop the
        pointer → base), which is the safe default.
        """
        if self._prev.exists():
            self._current.write_text(self._prev.read_text())
            self._prev.unlink()
            return self.current()
        # No prior champion: drop the current pointer → fall back to base.
        if self._current.exists():
            self._current.unlink()
        return None


__all__ = [
    "ChampionRegistry",
    "CheckpointMeta",
    "FAST_FAKE_STEPS",
    "MissingTrainingDepsError",
    "PER_STEP_SLEEP_S",
    "RealTrainingError",
    "StubTrainingError",
    "TrainingPass",
    "load_checkpoint",
    "load_meta",
]
