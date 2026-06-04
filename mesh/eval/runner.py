"""Held-out eval-pass runner — the writer that fills `eval_results.jsonl`.

Why this exists: `mesh/dashboard/eval.py` reads from
`dashboard/eval_results.jsonl` to render the headline "is the router
actually getting better?" panel. Before this module, nothing wrote that
file — the panel rendered empty. This is the writer.

One eval pass:

  1. Load the held-out seed via `load_verified_holdout` (sha256-checked,
     optionally signature-checked — see seed_verify.py).
  2. Route each prompt through an OpenAI-compatible chat-completions
     endpoint (the router itself, or a candidate specialist).
  3. Score every response with an injected `Scorer` (LocalJudgeScorer
     fits structurally — see mesh/quality_probe.py).
  4. Aggregate into the EvalRecord schema dashboard/eval.py expects.
  5. Append one line to eval_results.jsonl.

Design constraints honoured:

  * No hot-path mutation. Reads the seed JSONL, calls an endpoint over
    HTTP, writes a row file. No registry mutation, no replay_store
    write — those are P0 lane 1's sinks. The promotion gate (gate.py)
    is the consumer of the row file.

  * Pure orchestration over injected collaborators. The endpoint
    dispatcher and the scorer are passed in by the caller (so the test
    can fake the HTTP layer entirely without monkeypatching).

  * Failure isolation. A single prompt's endpoint/scorer failure must
    not abort the pass — score = 0.0 + a counter increment. A pass-wide
    network outage surfaces via `pct_failure` near 1.0.

  * Idempotency at the row level. The caller can either append (the
    default) or replace-by-key on (router_version, holdout_version, ts).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Protocol

from mesh.eval.seed_verify import (
    SeedVerificationError,
    VerifiedSeed,
    load_verified_holdout,
)
from mesh.quality_probe import ProbePrompt, Scorer, ScorerError

# Anything at or above this score is "acceptable" in the dashboard's
# headline percentages. Matches the LocalJudgeScorer 0..5 ladder; 3 = a
# usable answer. Tuned with onyx-ridge's P0 grading conventions in mind.
ACCEPTABLE_SCORE_THRESHOLD = 3.0
FAILURE_SCORE_THRESHOLD = 1.0


class EndpointDispatcher(Protocol):
    """Send one prompt to an OpenAI-compatible endpoint, return text.

    Implementations MUST return the assistant response text on success
    and raise `EndpointError` on any failure (transport, non-2xx,
    malformed body). The runner converts a raised EndpointError into a
    failed prompt (score 0, failure-count++), not an aborted pass.
    """

    def dispatch(
        self,
        prompt_text: str,
        *,
        domain: str | None = None,
    ) -> tuple[str, str]:
        """Return (response_text, model_or_specialist_id)."""
        ...


class EndpointError(Exception):
    """Raised by an EndpointDispatcher when a single dispatch fails."""


class HttpxEndpointDispatcher:
    """Default dispatcher against an OpenAI-compatible /v1/chat/completions.

    httpx is already a hard dep (see pyproject.toml line 15), matching
    the LocalJudgeScorer call pattern in mesh/quality_probe.py:172 —
    keeping the HTTP layer uniform across eval and judge.
    """

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        api_key: str | None = None,
        timeout_s: float = 60.0,
        temperature: float = 0.0,
    ) -> None:
        self.base_url = base_url
        self.model = model
        self.api_key = api_key
        self.timeout_s = timeout_s
        self.temperature = temperature

    def dispatch(
        self,
        prompt_text: str,
        *,
        domain: str | None = None,
    ) -> tuple[str, str]:
        import httpx

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        try:
            resp = httpx.post(
                f"{self.base_url.rstrip('/')}/v1/chat/completions",
                json={
                    "model": self.model,
                    "messages": [{"role": "user", "content": prompt_text}],
                    "temperature": self.temperature,
                },
                headers=headers,
                timeout=self.timeout_s,
            )
        except (httpx.HTTPError, ConnectionError) as exc:
            raise EndpointError(f"endpoint request failed: {type(exc).__name__}") from exc
        if not 200 <= resp.status_code < 300:
            raise EndpointError(f"endpoint returned HTTP {resp.status_code}")
        try:
            body = resp.json()
            content = body["choices"][0]["message"]["content"]
            served_model = body.get("model") or self.model
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise EndpointError("endpoint response missing assistant content") from exc
        if not isinstance(content, str):
            raise EndpointError("endpoint content is not text")
        return content, str(served_model)


@dataclass(frozen=True)
class EvalPass:
    """Result of one eval pass — exactly the EvalRecord schema
    `mesh/dashboard/eval.py` documents at the top of the file."""

    ts: str
    router_version: str
    fast_head_version: int | None
    overrides_version: int | None
    holdout_version: int
    n_eval: int
    judge_model: str
    mean_score: float
    median_score: float
    pct_acceptable: float
    pct_failure: float
    per_domain_mean: dict[str, float]
    per_model_mean: dict[str, float]
    elapsed_seconds: float
    n_dispatch_failures: int
    n_scorer_failures: int

    # ── provenance (issue #57) ───────────────────────────────────────────
    # All additive + optional (default None) so dashboard/eval.py and any
    # existing row reader keep parsing old-shape rows. These let a verdict
    # reconstruct the exact evaluated artifacts + holdout/corpus identities
    # without reading logs. Computed-locally fields (holdout_manifest_sha256,
    # code_sha) are filled by run_eval_pass; caller-supplied fields
    # (artifact_sha256, base_model_fingerprint, training_corpus_hash,
    # router_config_hash) are threaded through as parameters.
    artifact_sha256: str | None = None
    holdout_manifest_sha256: str | None = None
    training_corpus_hash: str | None = None
    base_model_fingerprint: str | None = None
    router_config_hash: str | None = None
    code_sha: str | None = None
    # Human-readable improvement rationale (issue #80), carried as an opaque
    # dict ({hypothesis, change_summary, expected_effect}) so this module and
    # the gate stay decoupled from training.ImprovementRationale. The caller
    # populates it from CheckpointMeta.rationale (asdict). None on stub /
    # legacy passes — old-shape row preserved.
    rationale: dict[str, Any] | None = None

    def to_row(self) -> dict[str, Any]:
        """JSON-serializable row for eval_results.jsonl."""
        return {
            "ts":                 self.ts,
            "router_version":     self.router_version,
            "fast_head_version":  self.fast_head_version,
            "overrides_version":  self.overrides_version,
            "holdout_version":    self.holdout_version,
            "n_eval":             self.n_eval,
            "judge_model":        self.judge_model,
            "mean_score":         round(self.mean_score, 4),
            "median_score":       round(self.median_score, 4),
            "pct_acceptable":     round(self.pct_acceptable, 4),
            "pct_failure":        round(self.pct_failure, 4),
            "per_domain_mean":    {d: round(s, 4) for d, s in self.per_domain_mean.items()},
            "per_model_mean":     {m: round(s, 4) for m, s in self.per_model_mean.items()},
            "elapsed_seconds":    round(self.elapsed_seconds, 3),
            "n_dispatch_failures": self.n_dispatch_failures,
            "n_scorer_failures":  self.n_scorer_failures,
            "artifact_sha256":         self.artifact_sha256,
            "holdout_manifest_sha256": self.holdout_manifest_sha256,
            "training_corpus_hash":    self.training_corpus_hash,
            "base_model_fingerprint":  self.base_model_fingerprint,
            "router_config_hash":      self.router_config_hash,
            "code_sha":                self.code_sha,
            "rationale":               self.rationale,
        }


def _domain_of(record: dict[str, Any]) -> str:
    sig = record.get("signals")
    if isinstance(sig, dict):
        d = sig.get("domain")
        if isinstance(d, str):
            return d
    return "unknown"


def _sha256_file(path: Path) -> str | None:
    """sha256 of a file's bytes; None if it can't be read (missing/IO)."""
    try:
        h = hashlib.sha256()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(1 << 16), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return None


def _git_code_sha() -> str | None:
    """`git rev-parse HEAD` for the working tree; None outside a git repo."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    sha = out.stdout.strip()
    return sha or None


def _record_text(record: dict[str, Any]) -> str:
    """Best-effort prompt text extraction matching corpus conventions."""
    t = record.get("prompt_text")
    if isinstance(t, str):
        return t
    t = record.get("text")
    if isinstance(t, str):
        return t
    return json.dumps(record, ensure_ascii=False)


def run_eval_pass(
    seed: VerifiedSeed,
    dispatcher: EndpointDispatcher,
    scorer: Scorer,
    *,
    router_version: str,
    fast_head_version: int | None = None,
    overrides_version: int | None = None,
    artifact_sha256: str | None = None,
    training_corpus_hash: str | None = None,
    base_model_fingerprint: str | None = None,
    router_config_hash: str | None = None,
    code_sha: str | None = None,
    rationale: dict[str, Any] | None = None,
    now: Callable[[], float] = time.time,
) -> EvalPass:
    """Route every seed record through `dispatcher`, score each with
    `scorer`, return an aggregated EvalPass.

    Provenance (issue #57): the holdout manifest sha256 is computed here
    from `seed.manifest_path`; the code SHA defaults to `git rev-parse
    HEAD` (None outside a git checkout) unless `code_sha` is passed. The
    candidate artifact hash, training-corpus hash, base-model fingerprint,
    and router-config hash must come from the caller (populate from
    `CheckpointMeta`: corpus_hash → training_corpus_hash, base_model_id →
    base_model_fingerprint) — they cannot be derived from the eval pass
    itself. All are optional; a row written without them stays old-shape.

    `rationale` (issue #80) is the human-readable WHY for the candidate,
    carried as an opaque dict; populate from `CheckpointMeta.rationale`
    (`asdict(meta.rationale)` if present, else None). None for the champion /
    baseline passes — only a built challenger has a rationale.

    Failure model:
      - dispatcher raises EndpointError → record gets score 0, counted in
        n_dispatch_failures, and tagged with model="<failed>".
      - scorer raises ScorerError → record gets score 0, counted in
        n_scorer_failures, model is the served_model returned by the
        dispatcher.
      - Either failure type increments pct_failure if the resulting
        score is below FAILURE_SCORE_THRESHOLD.

    `judge_model` on the row is whatever attribute the scorer exposes
    (LocalJudgeScorer has `.model`); falls back to "unknown" — keeps
    this loose so non-LocalJudgeScorer Scorers don't have to change.
    """
    started = now()
    scores: list[float] = []
    per_domain_scores: dict[str, list[float]] = defaultdict(list)
    per_model_scores: dict[str, list[float]] = defaultdict(list)
    n_dispatch_failures = 0
    n_scorer_failures = 0

    for rec in seed.records:
        prompt_id = str(rec.get("prompt_id") or "?")
        domain = _domain_of(rec)
        text = _record_text(rec)
        reference = rec.get("reference") if isinstance(rec.get("reference"), str) else None

        try:
            response_text, served_model = dispatcher.dispatch(text, domain=domain)
        except EndpointError:
            score = 0.0
            served_model = "<failed>"
            n_dispatch_failures += 1
        else:
            probe = ProbePrompt(
                prompt_id=prompt_id, domain=domain, text=text, reference=reference,
            )
            try:
                score = float(scorer.score(probe, response_text))
            except ScorerError:
                score = 0.0
                n_scorer_failures += 1

        scores.append(score)
        per_domain_scores[domain].append(score)
        per_model_scores[served_model].append(score)

    n_eval = len(scores)
    mean = statistics.fmean(scores) if scores else 0.0
    median = statistics.median(scores) if scores else 0.0
    pct_acc = (
        sum(1 for s in scores if s >= ACCEPTABLE_SCORE_THRESHOLD) / n_eval
        if n_eval else 0.0
    )
    pct_fail = (
        sum(1 for s in scores if s < FAILURE_SCORE_THRESHOLD) / n_eval
        if n_eval else 0.0
    )

    return EvalPass(
        ts=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(started)),
        router_version=router_version,
        fast_head_version=fast_head_version,
        overrides_version=overrides_version,
        holdout_version=int(seed.manifest.get("holdout_version") or 0),
        n_eval=n_eval,
        judge_model=str(getattr(scorer, "model", None) or "unknown"),
        mean_score=mean,
        median_score=median,
        pct_acceptable=pct_acc,
        pct_failure=pct_fail,
        per_domain_mean={d: statistics.fmean(s) for d, s in per_domain_scores.items() if s},
        per_model_mean={m: statistics.fmean(s) for m, s in per_model_scores.items() if s},
        elapsed_seconds=now() - started,
        n_dispatch_failures=n_dispatch_failures,
        n_scorer_failures=n_scorer_failures,
        artifact_sha256=artifact_sha256,
        holdout_manifest_sha256=_sha256_file(seed.manifest_path),
        training_corpus_hash=training_corpus_hash,
        base_model_fingerprint=base_model_fingerprint,
        router_config_hash=router_config_hash,
        code_sha=code_sha if code_sha is not None else _git_code_sha(),
        rationale=rationale,
    )


def append_pass(output: Path, ep: EvalPass) -> None:
    """Append one EvalPass to `output` as a JSONL row."""
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("a", encoding="utf-8") as f:
        f.write(json.dumps(ep.to_row(), ensure_ascii=False) + "\n")


# ───────────────────────────── CLI ──────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Route the held-out seed through an OpenAI-compatible endpoint "
                    "and append one scored row to eval_results.jsonl.",
    )
    ap.add_argument("--corpus", type=Path, required=True,
                    help="Path to held-out JSONL (e.g., corpus/eval/holdout_v1.jsonl)")
    ap.add_argument("--manifest", type=Path, default=None,
                    help="Manifest path (defaults to <corpus>.manifest.json or "
                         "<corpus_stem>.manifest.json)")
    ap.add_argument("--endpoint", required=True,
                    help="Base URL of the OpenAI-compatible endpoint to route through "
                         "(e.g., http://localhost:11434)")
    ap.add_argument("--model", required=True,
                    help="Model identifier the endpoint should serve")
    ap.add_argument("--api-key", default=None,
                    help="Optional Authorization bearer token for the endpoint")
    ap.add_argument("--judge-endpoint", default=None,
                    help="Base URL for the LLM-judge scorer. If omitted, uses --endpoint.")
    ap.add_argument("--judge-model", required=True,
                    help="Model identifier the judge endpoint should use")
    ap.add_argument("--judge-api-key", default=None,
                    help="Optional Authorization bearer token for the judge endpoint")
    ap.add_argument("--router-version", required=True,
                    help="Human-readable router version tag stamped on the row "
                         "(e.g., 'fast_head_v3+overrides_v17')")
    ap.add_argument("--fast-head-version", type=int, default=None)
    ap.add_argument("--overrides-version", type=int, default=None)
    ap.add_argument("--artifact-sha256", default=None,
                    help="Hash of the candidate artifact being evaluated "
                         "(provenance, issue #57)")
    ap.add_argument("--training-corpus-hash", default=None,
                    help="Training corpus hash of the artifact "
                         "(CheckpointMeta.corpus_hash)")
    ap.add_argument("--base-model-fingerprint", default=None,
                    help="Base-model fingerprint the artifact fine-tuned from "
                         "(CheckpointMeta.base_model_id)")
    ap.add_argument("--router-config-hash", default=None,
                    help="Hash of the router config used for this pass")
    ap.add_argument("--output", type=Path,
                    default=Path("dashboard/eval_results.jsonl"),
                    help="Append target (default: dashboard/eval_results.jsonl)")
    ap.add_argument("--require-signed-seed", action="store_true",
                    help="Refuse to load if manifest has no verified ed25519 signature")
    ap.add_argument("--trusted-signers", type=Path, default=None,
                    help="JSON file mapping signer_did -> base64(32-byte ed25519 pubkey)")
    args = ap.parse_args(argv)

    trusted: dict[str, bytes] | None = None
    if args.trusted_signers is not None:
        try:
            raw = json.loads(args.trusted_signers.read_text(encoding="utf-8"))
            import base64 as _b64
            trusted = {k: _b64.b64decode(v) for k, v in raw.items()}
        except Exception as exc:
            print(f"failed to load trusted_signers: {exc}", file=sys.stderr)
            return 3

    # #104: require a signed seed via the flag OR a regulated-profile env var, so
    # an operator pins enforcement once instead of remembering --require-signed-seed
    # on every invocation.
    require_sig = args.require_signed_seed or (
        os.environ.get("SLANCHA_REQUIRE_SIGNED_SEED", "").strip().lower()
        in ("1", "true", "yes", "on")
    )
    try:
        seed = load_verified_holdout(
            args.corpus,
            args.manifest,
            require_signature=require_sig,
            trusted_signers=trusted,
        )
    except SeedVerificationError as exc:
        print(f"[runner] seed verification failed: {exc}", file=sys.stderr)
        return 2
    if not require_sig:
        # sha256 alone doesn't stop a local writer who swaps the seed AND
        # regenerates the manifest hash — surface it (the finding: signature off
        # by default was silent). Loud, not fatal.
        print(
            "[runner] WARNING: holdout seed loaded with sha256 integrity ONLY (no ed25519 "
            "signature). A local writer could swap the seed and regenerate the manifest hash. "
            "Set SLANCHA_REQUIRE_SIGNED_SEED=1 + --trusted-signers in a regulated deployment.",
            file=sys.stderr,
        )

    from mesh.quality_probe import LocalJudgeScorer

    dispatcher = HttpxEndpointDispatcher(
        base_url=args.endpoint, model=args.model, api_key=args.api_key,
    )
    scorer = LocalJudgeScorer(
        base_url=args.judge_endpoint or args.endpoint,
        model=args.judge_model,
        api_key=args.judge_api_key,
    )

    ep = run_eval_pass(
        seed=seed,
        dispatcher=dispatcher,
        scorer=scorer,
        router_version=args.router_version,
        fast_head_version=args.fast_head_version,
        overrides_version=args.overrides_version,
        artifact_sha256=args.artifact_sha256,
        training_corpus_hash=args.training_corpus_hash,
        base_model_fingerprint=args.base_model_fingerprint,
        router_config_hash=args.router_config_hash,
    )
    append_pass(args.output, ep)
    print(
        f"[runner] {ep.n_eval} prompts → mean={ep.mean_score:.3f} "
        f"median={ep.median_score:.3f} acc={ep.pct_acceptable:.2%} "
        f"fail={ep.pct_failure:.2%} dispatch_fail={ep.n_dispatch_failures} "
        f"scorer_fail={ep.n_scorer_failures} elapsed={ep.elapsed_seconds:.1f}s → "
        f"{args.output}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
