"""Real PEFT/LoRA leg tests (issue #65) + rollback + missing-deps guard.

Three groups:

  * The REAL training path — gated behind `pytest.importorskip("torch")` /
    `("peft")` / `("transformers")` so it RUNS only where the `[train]`
    extra is installed and SKIPS cleanly in the hermetic CI (where it is
    not). It trains a tiny LoRA pass on a few-MB CPU model over a 3-example
    corpus and asserts a real adapter artifact (stub=False +
    base_model_fingerprint + corpus_hash) that the eval gate would promote.
  * The ChampionRegistry rollback helper — deterministic, no GPU/model:
    fake checkpoint dirs (just a meta.json) exercise promote/rollback.
  * The missing-deps guard — monkeypatch the lazy import to fail and assert
    MissingTrainingDepsError (NOT a silent stub fallback, issue #55).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mesh.eval.gate import decide
from mesh.replay_store import TrafficReplayStore
from mesh.training import (
    ChampionRegistry,
    CheckpointMeta,
    MissingTrainingDepsError,
    RealTrainingError,
    StubTrainingError,
    TrainingPass,
    _base_model_fingerprint,
    load_meta,
)

# A few-MB causal-LM that trains on CPU in seconds. Used only when torch+peft
# are present; otherwise the real-path tests skip before touching it.
TINY_MODEL = "sshleifer/tiny-gpt2"


def _store_with(n: int, domain: str = "math") -> TrafficReplayStore:
    s = TrafficReplayStore(max_size=1000)
    for i in range(n):
        s.add(
            prompt_text=f"prompt-{i}",
            oracle_response=f"the answer to {i} is {i * 2}",
            domain=domain,
            difficulty="medium",
        )
    return s


# ---------------------------------------------------------------------------
# base_model_fingerprint (no deps needed)
# ---------------------------------------------------------------------------


def test_fingerprint_deterministic_and_id_sensitive():
    a = _base_model_fingerprint("Qwen/Qwen3-8B", {"hidden": 4096})
    b = _base_model_fingerprint("Qwen/Qwen3-8B", {"hidden": 4096})
    c = _base_model_fingerprint("Qwen/Qwen3-14B", {"hidden": 4096})
    assert a == b
    assert a != c
    assert a.startswith("Qwen/Qwen3-8B@sha256:")


def test_fingerprint_config_order_invariant():
    a = _base_model_fingerprint("m", {"x": 1, "y": 2})
    b = _base_model_fingerprint("m", {"y": 2, "x": 1})
    assert a == b


# ---------------------------------------------------------------------------
# Dispatch guard: neither flag → StubTrainingError (issue #55 contract)
# ---------------------------------------------------------------------------


def test_neither_flag_still_raises_stub_error(tmp_path: Path):
    """Default (no allow_stub, no allow_real) preserves the #55 refuse."""
    tp = TrainingPass(
        specialist_id="s",
        base_model_id="b",
        domain="math",
        replay_store=_store_with(3),
        checkpoint_dir=tmp_path,
    )
    with pytest.raises(StubTrainingError) as exc:
        tp.run()
    msg = str(exc.value)
    assert "allow_stub=True" in msg
    assert "allow_real=True" in msg
    assert "#65" in msg
    assert tp.meta is None
    assert not any(tmp_path.iterdir())


# ---------------------------------------------------------------------------
# Missing-deps guard (issue #55: no silent stub fallback)
# ---------------------------------------------------------------------------


def test_real_path_missing_deps_raises(tmp_path: Path, monkeypatch):
    """allow_real=True with the [train] extra absent → MissingTrainingDepsError,
    never a silent stub fallback. Simulated by forcing the lazy import to fail."""
    import builtins

    real_import = builtins.__import__

    def _no_train_deps(name, *args, **kwargs):
        if name in {"torch", "peft", "transformers", "datasets"}:
            raise ImportError(f"simulated-missing: {name}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _no_train_deps)

    tp = TrainingPass(
        specialist_id="s",
        base_model_id=TINY_MODEL,
        domain="math",
        replay_store=_store_with(3),
        checkpoint_dir=tmp_path,
        allow_real=True,
        n_steps_planned=2,
    )
    with pytest.raises(MissingTrainingDepsError) as exc:
        tp.run()
    msg = str(exc.value)
    assert '".[train]"' in msg
    assert "#55" in msg
    # No checkpoint written on the failed path.
    assert not any(tmp_path.iterdir())


# ---------------------------------------------------------------------------
# ChampionRegistry rollback (deterministic, no GPU/model)
# ---------------------------------------------------------------------------


def _fake_checkpoint(tmp_path: Path, name: str, *, stub: bool) -> Path:
    """Write a minimal checkpoint dir (just meta.json) for registry tests."""
    sub = tmp_path / name
    sub.mkdir(parents=True, exist_ok=True)
    from datetime import datetime, timezone

    meta = CheckpointMeta(
        specialist_id=name,
        base_model_id="b",
        domain="math",
        seed=0,
        n_examples=3,
        n_steps_completed=2,
        n_steps_planned=2,
        started_at=datetime.now(timezone.utc),
        finished_at=datetime.now(timezone.utc),
        preempted=False,
        corpus_hash="sha256:x",
        stub=stub,
        base_model_fingerprint=None if stub else "b@sha256:deadbeef",
    )
    (sub / "meta.json").write_text(json.dumps(meta.to_json(), indent=2))
    return sub


def test_registry_first_promote_then_rollback_to_base(tmp_path: Path):
    """First-ever promote has no prior champion → rollback drops the pointer
    (fall back to base), returns None."""
    reg = ChampionRegistry(tmp_path / "registry")
    assert reg.current() is None
    ckpt_a = _fake_checkpoint(tmp_path, "adapter-a", stub=False)
    reg.promote(ckpt_a)
    assert reg.current() == ckpt_a
    restored = reg.rollback()
    assert restored is None
    assert reg.current() is None


def test_registry_promote_keeps_prior_and_rollback_restores(tmp_path: Path):
    """Promote B over champion A; rollback restores A."""
    reg = ChampionRegistry(tmp_path / "registry")
    ckpt_a = _fake_checkpoint(tmp_path, "adapter-a", stub=False)
    ckpt_b = _fake_checkpoint(tmp_path, "adapter-b", stub=False)
    reg.promote(ckpt_a)
    reg.promote(ckpt_b)
    assert reg.current() == ckpt_b
    assert reg.previous() == ckpt_a
    restored = reg.rollback()
    assert restored == ckpt_a
    assert reg.current() == ckpt_a
    # Prev consumed; a second rollback now falls back to base.
    assert reg.rollback() is None


def test_registry_refuses_stub_promotion(tmp_path: Path):
    """The registry mirrors the gate: a stub checkpoint can't become champion."""
    reg = ChampionRegistry(tmp_path / "registry")
    stub_ckpt = _fake_checkpoint(tmp_path, "stub-adapter", stub=True)
    with pytest.raises(StubTrainingError):
        reg.promote(stub_ckpt)
    assert reg.current() is None


def test_registry_promote_stub_allowed_with_verify_off(tmp_path: Path):
    """verify_real=False exercises pure pointer mechanics with a stub fake."""
    reg = ChampionRegistry(tmp_path / "registry")
    stub_ckpt = _fake_checkpoint(tmp_path, "stub-adapter", stub=True)
    reg.promote(stub_ckpt, verify_real=False)
    assert reg.current() == stub_ckpt


# ---------------------------------------------------------------------------
# REAL PEFT pass — RUNS only with the [train] extra; SKIPS otherwise.
# ---------------------------------------------------------------------------


def test_real_lora_pass_produces_promotable_adapter(tmp_path: Path):
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    pytest.importorskip("peft")

    store = _store_with(3, domain="math")
    tp = TrainingPass(
        specialist_id="tiny-gpt2-math",
        base_model_id=TINY_MODEL,
        domain="math",
        replay_store=store,
        checkpoint_dir=tmp_path,
        allow_real=True,
        n_steps_planned=3,
        lora_rank=4,
        lora_alpha=8,
        max_seq_len=32,
        seed=7,
    )
    out = tp.run()

    # A real adapter artifact exists (peft save_pretrained).
    assert out.is_dir()
    assert (out / "adapter_config.json").exists()
    assert (out / "meta.json").exists()

    meta = load_meta(out)
    assert meta.stub is False
    assert meta.base_model_fingerprint is not None
    assert meta.base_model_fingerprint.startswith(TINY_MODEL + "@sha256:")
    assert meta.corpus_hash.startswith("sha256:")
    assert meta.n_steps_completed == 3
    assert meta.n_examples == 3

    # Eval-gate linkage: a non-stub challenger with a clean lift is NOT
    # rejected as a stub (issue #65 → gate promotes real adapters).
    champ = {
        "router_version": "v1",
        "n_eval": 500,
        "judge_model": "j",
        "mean_score": 3.50,
        "per_domain_mean": {"math": 3.5},
        "meta_stub": False,
    }
    chall = {
        "router_version": "v2",
        "n_eval": 500,
        "judge_model": "j",
        "mean_score": 3.80,
        "per_domain_mean": {"math": 3.8},
        "meta_stub": meta.stub,  # False → not rejected
        "base_model_fingerprint": meta.base_model_fingerprint,
        "training_corpus_hash": meta.corpus_hash,
    }
    verdict = decide(champ, chall)
    assert verdict.accept is True
    assert not any("stub" in r for r in verdict.reject_reasons)
    assert verdict.base_model_fingerprint_challenger == meta.base_model_fingerprint


def test_real_lora_pass_empty_corpus_raises(tmp_path: Path):
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    pytest.importorskip("peft")

    tp = TrainingPass(
        specialist_id="tiny-gpt2-empty",
        base_model_id=TINY_MODEL,
        domain="no-such-domain",  # store has none → empty corpus
        replay_store=_store_with(3, domain="math"),
        checkpoint_dir=tmp_path,
        allow_real=True,
        n_steps_planned=2,
    )
    with pytest.raises(RealTrainingError):
        tp.run()
