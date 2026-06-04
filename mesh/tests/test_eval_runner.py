"""Tests for mesh.eval.runner (held-out eval-pass writer)."""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import httpx
import pytest

from mesh.eval.holdout import write_holdout
from mesh.eval.runner import (
    EndpointError,
    HttpxEndpointDispatcher,
    append_pass,
    run_eval_pass,
)
from mesh.eval.seed_verify import load_verified_holdout
from mesh.quality_probe import ProbePrompt, ScorerError


# ─── fakes ──────────────────────────────────────────────────────────────────


class FakeDispatcher:
    """In-memory dispatcher: serves a configured response per prompt."""

    def __init__(
        self,
        responses: dict[str, str] | None = None,
        served_model: str = "fake-model",
        fail_for: set[str] | None = None,
    ) -> None:
        self.responses = responses or {}
        self.served_model = served_model
        self.fail_for = fail_for or set()
        self.calls: list[tuple[str, str | None]] = []

    def dispatch(self, prompt_text: str, *, domain: str | None = None):
        self.calls.append((prompt_text, domain))
        if prompt_text in self.fail_for:
            raise EndpointError("forced failure")
        return self.responses.get(prompt_text, f"answer to {prompt_text}"), self.served_model


class FakeScorer:
    """Returns scores from a fixed sequence; can raise ScorerError on demand."""

    def __init__(
        self,
        scores: list[float],
        raise_for_response: set[str] | None = None,
        model: str = "fake-judge",
    ) -> None:
        self._scores = iter(itertools.chain(scores, itertools.repeat(scores[-1] if scores else 0.0)))
        self.raise_for_response = raise_for_response or set()
        self.model = model
        self.calls: list[ProbePrompt] = []

    def score(self, probe: ProbePrompt, response_text: str) -> float:
        self.calls.append(probe)
        if response_text in self.raise_for_response:
            raise ScorerError("forced scorer failure")
        return next(self._scores)


# ─── seed builder ──────────────────────────────────────────────────────────


def _build_verified_seed(tmp_path: Path, mix: dict[str, int]):
    records = []
    for domain, n in mix.items():
        for i in range(n):
            records.append({
                "prompt_id": f"{domain}-{i}",
                "prompt_text": f"{domain}-prompt-{i}",
                "signals": {"domain": domain},
            })
    out = tmp_path / "seed.jsonl"
    manifest = write_holdout(out, records, holdout_version=1, seed=0)
    manifest_path = out.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return load_verified_holdout(out, manifest_path)


# ─── tests ─────────────────────────────────────────────────────────────────


def test_run_eval_pass_aggregates_per_domain_and_overall(tmp_path: Path):
    seed = _build_verified_seed(tmp_path, {"code": 4, "general": 2})
    # code prompts score 4.0, general prompts score 2.0 → mean = 3.333…
    scores = [4.0] * 4 + [2.0] * 2
    dispatcher = FakeDispatcher()
    scorer = FakeScorer(scores=scores)
    ts_counter = itertools.count(start=1000.0, step=0.5)
    ep = run_eval_pass(
        seed=seed, dispatcher=dispatcher, scorer=scorer,
        router_version="v-test", fast_head_version=3, overrides_version=17,
        now=lambda: next(ts_counter),
    )
    assert ep.n_eval == 6
    assert ep.mean_score == pytest.approx((4.0 * 4 + 2.0 * 2) / 6)
    assert ep.per_domain_mean["code"] == pytest.approx(4.0)
    assert ep.per_domain_mean["general"] == pytest.approx(2.0)
    assert ep.per_model_mean == {"fake-model": pytest.approx((4 * 4 + 2 * 2) / 6)}
    assert ep.judge_model == "fake-judge"
    assert ep.router_version == "v-test"
    assert ep.fast_head_version == 3
    assert ep.holdout_version == 1
    assert ep.pct_acceptable == pytest.approx(4 / 6)  # only the 4.0 scores are ≥3
    assert ep.pct_failure == 0.0


def test_run_eval_pass_handles_dispatch_failures(tmp_path: Path):
    seed = _build_verified_seed(tmp_path, {"code": 3})
    # First prompt fails to dispatch
    fail_prompt = "code-prompt-0"
    dispatcher = FakeDispatcher(fail_for={fail_prompt})
    scorer = FakeScorer(scores=[5.0, 5.0])  # only the 2 successful ones get scored
    ep = run_eval_pass(
        seed=seed, dispatcher=dispatcher, scorer=scorer, router_version="v",
    )
    assert ep.n_dispatch_failures == 1
    assert ep.n_eval == 3
    # 0.0 (failed) + 5.0 + 5.0 → 10/3
    assert ep.mean_score == pytest.approx(10 / 3)
    assert ep.pct_failure == pytest.approx(1 / 3)
    # Failed dispatch tagged separately so per_model_mean doesn't pollute
    assert "<failed>" in ep.per_model_mean
    assert ep.per_model_mean["<failed>"] == 0.0


def test_run_eval_pass_handles_scorer_failures(tmp_path: Path):
    seed = _build_verified_seed(tmp_path, {"code": 2})
    dispatcher = FakeDispatcher(responses={
        "code-prompt-0": "BAD",  # scorer will refuse this one
        "code-prompt-1": "GOOD",
    })
    scorer = FakeScorer(scores=[4.0], raise_for_response={"BAD"})
    ep = run_eval_pass(
        seed=seed, dispatcher=dispatcher, scorer=scorer, router_version="v",
    )
    assert ep.n_scorer_failures == 1
    assert ep.n_dispatch_failures == 0
    # The scorer-failed prompt gets 0; the other gets 4
    assert ep.mean_score == pytest.approx(2.0)


def test_run_eval_pass_with_empty_seed_returns_zeros(tmp_path: Path):
    seed = _build_verified_seed(tmp_path, {"code": 0})
    dispatcher = FakeDispatcher()
    scorer = FakeScorer(scores=[])
    ep = run_eval_pass(
        seed=seed, dispatcher=dispatcher, scorer=scorer, router_version="v",
    )
    assert ep.n_eval == 0
    assert ep.mean_score == 0.0
    assert ep.median_score == 0.0
    assert ep.pct_acceptable == 0.0
    assert ep.pct_failure == 0.0


def test_eval_pass_row_round_trips_through_dashboard_reader(tmp_path: Path):
    """The runner's row must be readable by the dashboard's loader."""
    from mesh.dashboard.eval import (
        eval_summary,
        load_eval_results,
        mean_score_over_time,
        per_domain_score_over_time,
        per_version_summary,
    )

    seed = _build_verified_seed(tmp_path, {"code": 3, "general": 2})
    dispatcher = FakeDispatcher()
    scorer = FakeScorer(scores=[5.0, 4.0, 3.0, 2.0, 1.0])
    ep = run_eval_pass(
        seed=seed, dispatcher=dispatcher, scorer=scorer,
        router_version="fast_head_v3+overrides_v17",
        fast_head_version=3, overrides_version=17,
    )
    out = tmp_path / "eval_results.jsonl"
    append_pass(out, ep)
    append_pass(out, ep)  # two passes; reader must order them
    rows = load_eval_results(out)
    assert len(rows) == 2
    assert mean_score_over_time(rows)  # non-empty
    assert per_domain_score_over_time(rows)
    versions = per_version_summary(rows)
    assert versions and versions[0]["router_version"] == "fast_head_v3+overrides_v17"
    summary = eval_summary(rows)
    assert summary["n_passes"] == 2
    assert summary["latest_router_version"] == "fast_head_v3+overrides_v17"


def test_eval_pass_row_omits_no_required_fields(tmp_path: Path):
    seed = _build_verified_seed(tmp_path, {"code": 1})
    ep = run_eval_pass(
        seed=seed, dispatcher=FakeDispatcher(), scorer=FakeScorer(scores=[5.0]),
        router_version="v",
    )
    row = ep.to_row()
    # Schema documented at the top of mesh/dashboard/eval.py
    for required in (
        "ts", "router_version", "fast_head_version", "overrides_version",
        "holdout_version", "n_eval", "judge_model", "mean_score", "median_score",
        "pct_acceptable", "pct_failure", "per_domain_mean", "per_model_mean",
        "elapsed_seconds",
    ):
        assert required in row, f"missing {required!r}"


# ─── provenance (issue #57) ──────────────────────────────────────────────────


def test_eval_pass_computes_local_provenance(tmp_path: Path):
    """holdout_manifest_sha256 + code_sha are computed by the runner."""
    seed = _build_verified_seed(tmp_path, {"code": 1})
    ep = run_eval_pass(
        seed=seed, dispatcher=FakeDispatcher(), scorer=FakeScorer(scores=[5.0]),
        router_version="v",
    )
    # Manifest hash = sha256 of the manifest file the seed was loaded from.
    import hashlib
    expected = hashlib.sha256(seed.manifest_path.read_bytes()).hexdigest()
    assert ep.holdout_manifest_sha256 == expected
    # Running inside this git checkout → a 40-char hex SHA.
    assert ep.code_sha is not None
    assert len(ep.code_sha) == 40


def test_eval_pass_threads_caller_provenance(tmp_path: Path):
    """Caller-supplied provenance (artifact/corpus/base-model/router/code)
    lands on both the dataclass and the row."""
    seed = _build_verified_seed(tmp_path, {"code": 1})
    ep = run_eval_pass(
        seed=seed, dispatcher=FakeDispatcher(), scorer=FakeScorer(scores=[5.0]),
        router_version="v",
        artifact_sha256="sha256:artifact",
        training_corpus_hash="sha256:corpus",
        base_model_fingerprint="qwen3-8b@abc",
        router_config_hash="sha256:routercfg",
        code_sha="deadbeef",
    )
    assert ep.artifact_sha256 == "sha256:artifact"
    assert ep.training_corpus_hash == "sha256:corpus"
    assert ep.base_model_fingerprint == "qwen3-8b@abc"
    assert ep.router_config_hash == "sha256:routercfg"
    assert ep.code_sha == "deadbeef"  # explicit override, not git
    row = ep.to_row()
    assert row["artifact_sha256"] == "sha256:artifact"
    assert row["training_corpus_hash"] == "sha256:corpus"
    assert row["base_model_fingerprint"] == "qwen3-8b@abc"
    assert row["router_config_hash"] == "sha256:routercfg"
    assert row["code_sha"] == "deadbeef"
    assert row["holdout_manifest_sha256"] is not None


def test_eval_pass_provenance_defaults_none_when_not_supplied(tmp_path: Path):
    """Caller-sourced provenance defaults to None — old-shape callers stay
    intact; only locally-computable fields auto-populate."""
    seed = _build_verified_seed(tmp_path, {"code": 1})
    ep = run_eval_pass(
        seed=seed, dispatcher=FakeDispatcher(), scorer=FakeScorer(scores=[5.0]),
        router_version="v",
    )
    row = ep.to_row()
    assert row["artifact_sha256"] is None
    assert row["training_corpus_hash"] is None
    assert row["base_model_fingerprint"] is None
    assert row["router_config_hash"] is None


def test_eval_pass_threads_caller_rationale(tmp_path: Path):
    """The human-readable rationale (issue #80) threads onto the dataclass
    and the row as an opaque dict; defaults None when not supplied."""
    seed = _build_verified_seed(tmp_path, {"code": 1})
    rationale = {
        "hypothesis": "code cluster c_0427 underperforms base",
        "change_summary": "LoRA r=8 on 1840 c_0427 traces",
        "expected_effect": "+0.25 on c_0427 holdout",
    }
    ep = run_eval_pass(
        seed=seed, dispatcher=FakeDispatcher(), scorer=FakeScorer(scores=[5.0]),
        router_version="v", rationale=rationale,
    )
    assert ep.rationale == rationale
    assert ep.to_row()["rationale"] == rationale

    # Not supplied → None, row stays old-shape-compatible.
    ep2 = run_eval_pass(
        seed=seed, dispatcher=FakeDispatcher(), scorer=FakeScorer(scores=[5.0]),
        router_version="v",
    )
    assert ep2.rationale is None
    assert ep2.to_row()["rationale"] is None


def test_provenance_row_parses_through_dashboard_reader(tmp_path: Path):
    """A row carrying provenance must still be parseable by the existing
    dashboard consumer; an old-shape row (no provenance keys) must too."""
    from mesh.dashboard.eval import load_eval_results, mean_score_over_time

    seed = _build_verified_seed(tmp_path, {"code": 2})
    ep = run_eval_pass(
        seed=seed, dispatcher=FakeDispatcher(), scorer=FakeScorer(scores=[5.0, 4.0]),
        router_version="v", artifact_sha256="sha256:art",
    )
    out = tmp_path / "eval_results.jsonl"
    append_pass(out, ep)
    # An old-shape row with none of the new keys present at all.
    old_row = {
        "ts": "2026-05-01T00:00:00Z", "router_version": "old",
        "n_eval": 2, "judge_model": "j", "mean_score": 3.0,
        "median_score": 3.0, "pct_acceptable": 0.5, "pct_failure": 0.0,
        "per_domain_mean": {"code": 3.0}, "per_model_mean": {"fake-model": 3.0},
    }
    with out.open("a", encoding="utf-8") as f:
        f.write(json.dumps(old_row) + "\n")
    rows = load_eval_results(out)
    assert len(rows) == 2
    assert mean_score_over_time(rows)  # consumer unaffected by extra/missing keys


# ─── HttpxEndpointDispatcher — real HTTP dispatch error isolation ────────────
# The runner's failure-isolation model (a raised EndpointError → one failed
# prompt, not an aborted pass) depends on dispatch() raising EndpointError on
# every fault. FakeDispatcher exercises the runner; these lock the real
# dispatcher's four error branches + the success path, which had no coverage.


class _FakeHttpResp:
    def __init__(self, status_code: int, body: dict | None = None) -> None:
        self.status_code = status_code
        self._body = body if body is not None else {}

    def json(self) -> dict:
        return self._body


def _dispatcher() -> HttpxEndpointDispatcher:
    return HttpxEndpointDispatcher(base_url="http://ep.test", model="m")


def test_httpx_dispatcher_happy_path(monkeypatch):
    body = {"choices": [{"message": {"content": "hi"}}], "model": "served-x"}
    monkeypatch.setattr(httpx, "post", lambda *a, **k: _FakeHttpResp(200, body))
    text, served = _dispatcher().dispatch("q", domain="general")
    assert text == "hi"
    assert served == "served-x"


def test_httpx_dispatcher_transport_error_raises(monkeypatch):
    def _boom(*a, **k):
        raise httpx.ConnectError("refused")

    monkeypatch.setattr(httpx, "post", _boom)
    with pytest.raises(EndpointError, match="request failed"):
        _dispatcher().dispatch("q")


def test_httpx_dispatcher_non_2xx_raises(monkeypatch):
    monkeypatch.setattr(httpx, "post", lambda *a, **k: _FakeHttpResp(500))
    with pytest.raises(EndpointError, match="HTTP 500"):
        _dispatcher().dispatch("q")


def test_httpx_dispatcher_missing_content_raises(monkeypatch):
    monkeypatch.setattr(httpx, "post", lambda *a, **k: _FakeHttpResp(200, {"choices": []}))
    with pytest.raises(EndpointError, match="missing assistant content"):
        _dispatcher().dispatch("q")


def test_httpx_dispatcher_content_not_str_raises(monkeypatch):
    body = {"choices": [{"message": {"content": 123}}]}
    monkeypatch.setattr(httpx, "post", lambda *a, **k: _FakeHttpResp(200, body))
    with pytest.raises(EndpointError, match="not text"):
        _dispatcher().dispatch("q")


def test_main_env_requires_signed_seed(tmp_path, monkeypatch):
    """#104: SLANCHA_REQUIRE_SIGNED_SEED=1 enforces signature even without the
    --require-signed-seed flag — an unsigned seed is rejected before any network."""
    from mesh.eval.holdout import write_holdout
    from mesh.eval.runner import main

    out = tmp_path / "seed.jsonl"
    records = [{"prompt_id": "general-0", "prompt_text": "q", "signals": {"domain": "general"}}]
    manifest = write_holdout(out, records, holdout_version=1, seed=0)
    mpath = out.with_suffix(".manifest.json")
    mpath.write_text(json.dumps(manifest), encoding="utf-8")

    monkeypatch.setenv("SLANCHA_REQUIRE_SIGNED_SEED", "1")
    rc = main(["--corpus", str(out), "--manifest", str(mpath),
                "--endpoint", "http://x", "--model", "m", "--judge-model", "j",
                "--router-version", "v1"])
    assert rc == 2  # signature required + unsigned/no-trusted-signers → verification fails
