"""Promotion gate — pure function over two eval rows.

Why this exists: persona-review (security + systems) says the held-out
mean is necessary but not sufficient. A challenger can lift the mean
while regressing badly in one domain (e.g., +0.4 on `general`, −0.8 on
`code` — net positive, but a quality cliff for code users). Without a
per-domain non-regression check we'd silently promote that adapter and
have to roll back via SRE escalation.

So the gate's decision is:

  ACCEPT iff
    (1) challenger.mean - champion.mean >= mean_score_delta            [signal]
    (2) ∀ shared domain d:                                              [no cliff]
          challenger.per_domain_mean[d] - champion.per_domain_mean[d]
            >= -per_domain_max_regression
    (3) challenger.n_eval >= min_n_eval                                 [sample size]
    (4) judge_model identifiers match (or judge-mismatch is opted in)   [comparable]

Every verdict is event-sourced — the SRE finding requires
"every promotion is an event". append_verdict() writes a JSONL row to
`dashboard/promotions.jsonl` so a future operator can audit why an
adapter was let in (or kept out).

The module is a pure function over the two rows + thresholds. No HTTP,
no registry mutation. The runner produces rows; this consumes them.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

DEFAULT_MEAN_SCORE_DELTA = 0.05
DEFAULT_PER_DOMAIN_MAX_REGRESSION = 0.15
DEFAULT_MIN_N_EVAL = 100


@dataclass(frozen=True)
class GateThresholds:
    """Knobs the SRE persona surfaced as needing to be operator-tunable.

    `mean_score_delta` — the headline lift required to even consider a
    promotion. Smaller values trade noise tolerance for sensitivity.

    `per_domain_max_regression` — how much any single domain may slip in
    absolute judge-score points before the gate refuses. Hysteresis: set
    larger than the typical inter-pass noise on the held-out mean.

    `min_n_eval` — refuse if either side ran fewer prompts than this;
    avoids being fooled by a tiny pass.

    `require_judge_match` — when True, refuse cross-judge comparisons.
    When False, the gate still records the mismatch in the verdict.
    """

    mean_score_delta: float = DEFAULT_MEAN_SCORE_DELTA
    per_domain_max_regression: float = DEFAULT_PER_DOMAIN_MAX_REGRESSION
    min_n_eval: int = DEFAULT_MIN_N_EVAL
    require_judge_match: bool = True


@dataclass(frozen=True)
class PromotionVerdict:
    """The decision plus enough audit detail to explain itself later."""

    accept: bool
    reject_reasons: tuple[str, ...]
    mean_delta: float
    per_domain_deltas: dict[str, float] = field(default_factory=dict)
    champion_version: str = ""
    challenger_version: str = ""
    n_eval_champion: int = 0
    n_eval_challenger: int = 0
    judge_model_champion: str = ""
    judge_model_challenger: str = ""
    decided_at: str = ""
    thresholds: dict[str, Any] = field(default_factory=dict)

    # ── provenance (issue #57) ───────────────────────────────────────────
    # Additive + optional so existing callers/tests keep working. Sourced
    # from the eval rows decide() consumes (the runner now stamps these),
    # so a verdict can reconstruct the exact artifacts + holdout/corpus
    # identities on both sides without re-reading the eval log. Per-side
    # because champion and challenger are different artifacts.
    artifact_sha256_champion: str | None = None
    artifact_sha256_challenger: str | None = None
    holdout_manifest_sha256: str | None = None
    training_corpus_hash_champion: str | None = None
    training_corpus_hash_challenger: str | None = None
    base_model_fingerprint_champion: str | None = None
    base_model_fingerprint_challenger: str | None = None
    router_config_hash_champion: str | None = None
    router_config_hash_challenger: str | None = None
    code_sha_champion: str | None = None
    code_sha_challenger: str | None = None

    def to_row(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "reject_reasons": list(self.reject_reasons),
        }


def _row_field(row: dict[str, Any], key: str, default: Any) -> Any:
    v = row.get(key)
    return default if v is None else v


def decide(
    champion: dict[str, Any],
    challenger: dict[str, Any],
    thresholds: GateThresholds = GateThresholds(),
    champion_is_stub: bool | None = None,
    challenger_is_stub: bool | None = None,
) -> PromotionVerdict:
    """Return a PromotionVerdict for (champion, challenger).

    Inputs are EvalRecord rows as written by mesh.eval.runner. Domains
    not present in *both* rows are skipped from the per-domain check —
    we cannot say a domain regressed if the champion never saw it.

    Stub artifacts can never be promoted (issue #55): if either side was
    produced by the contract-only TrainingPass stub (no real PEFT, only
    placeholder weights), the verdict is reject regardless of scores. A
    side is treated as a stub when its eval row carries `meta_stub == True`
    (stamped from the checkpoint's CheckpointMeta.stub), or when the
    explicit `champion_is_stub` / `challenger_is_stub` overrides say so.
    """
    champion_mean = float(_row_field(champion, "mean_score", 0.0))
    challenger_mean = float(_row_field(challenger, "mean_score", 0.0))
    mean_delta = challenger_mean - champion_mean

    champion_per_dom = _row_field(champion, "per_domain_mean", {}) or {}
    challenger_per_dom = _row_field(challenger, "per_domain_mean", {}) or {}
    shared_domains = sorted(set(champion_per_dom) & set(challenger_per_dom))
    per_dom_deltas: dict[str, float] = {
        d: float(challenger_per_dom[d]) - float(champion_per_dom[d])
        for d in shared_domains
    }

    judge_a = str(_row_field(champion, "judge_model", "unknown"))
    judge_b = str(_row_field(challenger, "judge_model", "unknown"))
    n_a = int(_row_field(champion, "n_eval", 0))
    n_b = int(_row_field(challenger, "n_eval", 0))

    champ_stub = (
        bool(_row_field(champion, "meta_stub", False))
        if champion_is_stub is None
        else champion_is_stub
    )
    chall_stub = (
        bool(_row_field(challenger, "meta_stub", False))
        if challenger_is_stub is None
        else challenger_is_stub
    )

    reasons: list[str] = []

    if champ_stub:
        reasons.append("champion stub artifact cannot be promoted")
    if chall_stub:
        reasons.append("challenger stub artifact cannot be promoted")
    if thresholds.require_judge_match and judge_a != judge_b:
        reasons.append(
            f"judge_model mismatch: champion={judge_a!r} challenger={judge_b!r}"
        )
    if n_a < thresholds.min_n_eval:
        reasons.append(
            f"champion n_eval {n_a} below min_n_eval {thresholds.min_n_eval}"
        )
    if n_b < thresholds.min_n_eval:
        reasons.append(
            f"challenger n_eval {n_b} below min_n_eval {thresholds.min_n_eval}"
        )
    if mean_delta < thresholds.mean_score_delta:
        reasons.append(
            f"mean_delta {mean_delta:+.3f} below required "
            f"{thresholds.mean_score_delta:+.3f}"
        )
    for d in shared_domains:
        delta = per_dom_deltas[d]
        if delta < -thresholds.per_domain_max_regression:
            reasons.append(
                f"per-domain regression on {d!r}: {delta:+.3f} exceeds "
                f"-{thresholds.per_domain_max_regression:.3f}"
            )

    return PromotionVerdict(
        accept=not reasons,
        reject_reasons=tuple(reasons),
        mean_delta=mean_delta,
        per_domain_deltas=per_dom_deltas,
        champion_version=str(_row_field(champion, "router_version", "")),
        challenger_version=str(_row_field(challenger, "router_version", "")),
        n_eval_champion=n_a,
        n_eval_challenger=n_b,
        judge_model_champion=judge_a,
        judge_model_challenger=judge_b,
        decided_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        thresholds=asdict(thresholds),
        # Provenance carried straight off the eval rows (issue #57). The
        # runner stamps these; older rows simply have them absent → None.
        artifact_sha256_champion=champion.get("artifact_sha256"),
        artifact_sha256_challenger=challenger.get("artifact_sha256"),
        holdout_manifest_sha256=(
            challenger.get("holdout_manifest_sha256")
            or champion.get("holdout_manifest_sha256")
        ),
        training_corpus_hash_champion=champion.get("training_corpus_hash"),
        training_corpus_hash_challenger=challenger.get("training_corpus_hash"),
        base_model_fingerprint_champion=champion.get("base_model_fingerprint"),
        base_model_fingerprint_challenger=challenger.get("base_model_fingerprint"),
        router_config_hash_champion=champion.get("router_config_hash"),
        router_config_hash_challenger=challenger.get("router_config_hash"),
        code_sha_champion=champion.get("code_sha"),
        code_sha_challenger=challenger.get("code_sha"),
    )


def append_verdict(output: Path, verdict: PromotionVerdict) -> None:
    """Event-source the verdict to `output` (default: dashboard/promotions.jsonl).

    Append-only — never rewrite a row. The SRE finding requires every
    promotion be an event so an operator can ask "why did we promote
    this one?" months later.
    """
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("a", encoding="utf-8") as f:
        f.write(json.dumps(verdict.to_row(), ensure_ascii=False) + "\n")


# ───────────────────────────── CLI ──────────────────────────────────────────


def _latest_row_for_version(rows: list[dict[str, Any]], version: str) -> dict[str, Any]:
    candidates = [r for r in rows if r.get("router_version") == version]
    if not candidates:
        raise SystemExit(f"no eval row found for router_version={version!r}")
    return max(candidates, key=lambda r: r.get("ts") or "")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Decide ACCEPT/REJECT for a challenger router_version vs a "
                    "champion router_version, given eval_results.jsonl.",
    )
    ap.add_argument("--eval-jsonl", type=Path,
                    default=Path("dashboard/eval_results.jsonl"),
                    help="Path to eval_results.jsonl (default: dashboard/eval_results.jsonl)")
    ap.add_argument("--champion", required=True,
                    help="router_version string for the incumbent")
    ap.add_argument("--challenger", required=True,
                    help="router_version string for the candidate")
    ap.add_argument("--mean-score-delta", type=float, default=DEFAULT_MEAN_SCORE_DELTA)
    ap.add_argument("--per-domain-max-regression", type=float,
                    default=DEFAULT_PER_DOMAIN_MAX_REGRESSION)
    ap.add_argument("--min-n-eval", type=int, default=DEFAULT_MIN_N_EVAL)
    ap.add_argument("--allow-judge-mismatch", action="store_true",
                    help="Compare across judge_model values (default refuses)")
    ap.add_argument("--promotions-log", type=Path,
                    default=Path("dashboard/promotions.jsonl"),
                    help="Append target for the verdict event log "
                         "(default: dashboard/promotions.jsonl)")
    args = ap.parse_args(argv)

    from mesh.dashboard.eval import load_eval_results

    rows = load_eval_results(args.eval_jsonl)
    if not rows:
        print(f"[gate] no eval rows in {args.eval_jsonl}", file=sys.stderr)
        return 2

    champion = _latest_row_for_version(rows, args.champion)
    challenger = _latest_row_for_version(rows, args.challenger)

    thresholds = GateThresholds(
        mean_score_delta=args.mean_score_delta,
        per_domain_max_regression=args.per_domain_max_regression,
        min_n_eval=args.min_n_eval,
        require_judge_match=not args.allow_judge_mismatch,
    )
    verdict = decide(champion, challenger, thresholds)
    append_verdict(args.promotions_log, verdict)

    headline = "ACCEPT" if verdict.accept else "REJECT"
    print(
        f"[gate] {headline} {verdict.challenger_version!r} vs "
        f"{verdict.champion_version!r}: mean_delta={verdict.mean_delta:+.3f}",
        file=sys.stderr,
    )
    for reason in verdict.reject_reasons:
        print(f"[gate]   - {reason}", file=sys.stderr)
    return 0 if verdict.accept else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
