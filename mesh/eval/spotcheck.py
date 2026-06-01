"""Cloud spot-check grader-drift governor — GATE-CONTRACT invariant #8.

Why this exists: the local gate (`mesh/eval/gate.py`) trusts a *local* judge
model to score champion vs challenger. But that judge is itself a model, and
over time it can DRIFT from ground truth. A drifting judge silently poisons
promotions — it keeps saying "challenger is better" when it no longer can tell.
Invariant #7 (goodhart_guard, the frozen-holdout tripwire) is the cheap
every-cycle hard check; invariant #8 is the *sampled slow-drift early-warning*.
They compose — #8 does NOT replace #7.

The mechanism (exactly per docs/GATE-CONTRACT.md #8):

  * SAMPLING — at promotion time, spot-check ~10% of PROMOTE verdicts plus
    100% of MARGINAL promotions (mean_delta < 2× the decisive gain). A marginal
    promotion is where a drifting judge does the most damage: the local mean is
    barely over the bar, so a small judge bias flips the call.
  * INDEPENDENT JUDGE — re-grade the spot-check prompts with an independent
    frontier/cloud judge (INJECTED — this module never hard-depends on a
    provider SDK; the caller wires a real cloud client, tests fake it).
  * THRESHOLD — track the Spearman rank correlation between local and cloud
    scores over a ROLLING WINDOW. `corr < 0.7` ⇒ FREEZE promotions and signal a
    re-fit of the local grader.

Composition with the runner: `cloud_spotcheck_gate(...)` returns a
`gate_decide`-compatible callable that WRAPS the real `decide`. It never
modifies the gate's core logic — on an `accept=True` verdict that falls in the
spot-check sample it runs the cloud check, records the (local, cloud) pair, and
can flip `accept` → frozen. Otherwise it passes the verdict through unchanged.
It is wired into `LoopRunner.gate_decide` as a DOCUMENTED OPT-IN (it costs cloud
tokens — off by default).

Spearman is computed in pure Python (rank → Pearson on ranks) — no scipy/numpy
dependency added.
"""

from __future__ import annotations

import json
import time
from collections import deque
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Deque, Protocol

from mesh.eval.gate import PromotionVerdict

# ── tunables (GATE-CONTRACT #8) ──────────────────────────────────────────────
# Spot-check ~this fraction of *non-marginal* accepts (deterministic, by hash).
DEFAULT_SPOTCHECK_RATE: float = 0.10
# Marginal = local mean_delta below this multiple of the decisive gain. A
# drifting judge does the most damage on barely-over-the-bar promotions, so
# those are spot-checked 100%.
DEFAULT_MARGINAL_FACTOR: float = 2.0
# Spearman below this over the rolling window ⇒ FREEZE + re-fit the local judge.
DEFAULT_CORR_THRESHOLD: float = 0.70
# How many recent (local, cloud) pairs the correlation is computed over.
DEFAULT_WINDOW: int = 50
# Need at least this many pairs before a correlation is trustworthy enough to
# freeze on — too few points and Spearman is pure noise (don't freeze on n=2).
DEFAULT_MIN_PAIRS: int = 8


# ── Spearman (pure Python — no scipy/numpy) ──────────────────────────────────


def _rank(values: list[float]) -> list[float]:
    """Fractional ranks (1-based), averaging ties — standard Spearman ranking."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    n = len(values)
    while i < n:
        j = i
        while j + 1 < n and values[order[j + 1]] == values[order[i]]:
            j += 1
        # positions i..j are tied → assign the average rank (1-based)
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def _pearson(xs: list[float], ys: list[float]) -> float:
    """Pearson correlation of two equal-length vectors; 0.0 if undefined."""
    n = len(xs)
    if n == 0:
        return 0.0
    mx = sum(xs) / n
    my = sum(ys) / n
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    denom = (vx * vy) ** 0.5
    if denom == 0.0:
        # No variance on at least one side → correlation is undefined. Treat as
        # 0.0 (no evidence of agreement) rather than fabricating 1.0.
        return 0.0
    return cov / denom


def spearman(local_scores: list[float], cloud_scores: list[float]) -> float:
    """Spearman rank correlation = Pearson on the ranks. 0.0 if undefined."""
    if len(local_scores) != len(cloud_scores):
        raise ValueError("local/cloud score vectors must be equal length")
    if len(local_scores) < 2:
        return 0.0
    return _pearson(_rank(local_scores), _rank(cloud_scores))


# ── rolling-window correlation tracker ───────────────────────────────────────


@dataclass(frozen=True)
class SpotcheckPair:
    """One spot-checked (local, cloud) score pair, with provenance."""

    local_score: float
    cloud_score: float
    challenger_version: str = ""
    at: str = ""


class DriftTracker:
    """Rolling window of (local, cloud) score pairs → Spearman freeze signal.

    Persists pairs to a JSONL under the champion-registry / run dir so the
    window survives across runner restarts (drift is a longitudinal signal —
    an in-memory-only window would reset every restart and never trip). Pure
    and testable: construct with `path=None` for an in-memory window (tests
    that don't care about persistence).

    `should_freeze()` is True iff there are at least `min_pairs` in the window
    AND the Spearman correlation over the window is below `corr_threshold`.
    """

    def __init__(
        self,
        path: Path | None = None,
        *,
        window: int = DEFAULT_WINDOW,
        corr_threshold: float = DEFAULT_CORR_THRESHOLD,
        min_pairs: int = DEFAULT_MIN_PAIRS,
    ) -> None:
        self.path = Path(path) if path is not None else None
        self.window = window
        self.corr_threshold = corr_threshold
        self.min_pairs = min_pairs
        self._pairs: Deque[SpotcheckPair] = deque(maxlen=window)
        if self.path is not None and self.path.exists():
            self._load()

    def _load(self) -> None:
        assert self.path is not None
        for line in self.path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            self._pairs.append(
                SpotcheckPair(
                    local_score=float(d["local_score"]),
                    cloud_score=float(d["cloud_score"]),
                    challenger_version=str(d.get("challenger_version", "")),
                    at=str(d.get("at", "")),
                )
            )

    def record(
        self,
        local_score: float,
        cloud_score: float,
        *,
        challenger_version: str = "",
    ) -> SpotcheckPair:
        """Append one (local, cloud) pair to the window (and persist if backed)."""
        pair = SpotcheckPair(
            local_score=float(local_score),
            cloud_score=float(cloud_score),
            challenger_version=challenger_version,
            at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        )
        self._pairs.append(pair)
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as f:
                f.write(
                    json.dumps(
                        {
                            "local_score": pair.local_score,
                            "cloud_score": pair.cloud_score,
                            "challenger_version": pair.challenger_version,
                            "at": pair.at,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
        return pair

    @property
    def n_pairs(self) -> int:
        return len(self._pairs)

    def correlation(self) -> float:
        """Spearman over the current window (0.0 if fewer than 2 pairs)."""
        local = [p.local_score for p in self._pairs]
        cloud = [p.cloud_score for p in self._pairs]
        return spearman(local, cloud)

    def should_freeze(self) -> bool:
        """True ⇒ FREEZE promotions: enough pairs AND Spearman below threshold."""
        if self.n_pairs < self.min_pairs:
            return False
        return self.correlation() < self.corr_threshold


# ── cloud judge protocol (injected — no provider hard-dep) ───────────────────


class CloudJudge(Protocol):
    """Independent frontier/cloud judge for the spot-check.

    Structurally identical to the existing `Scorer` / `EndpointDispatcher`
    shape (mesh/quality_probe.py, mesh/eval/runner.py): one method, returns a
    0..5 score for a prompt+response. The governor INJECTS this so it never
    hard-depends on a specific cloud provider — a real impl wraps an
    OpenAI-compatible frontier endpoint; tests pass a canned fake.
    """

    def score(self, prompt_text: str, response_text: str) -> float:
        """Return the cloud judge's 0..5 score for (prompt, response)."""
        ...


# ── sampling decision ────────────────────────────────────────────────────────


def _sample_hash(challenger_version: str) -> float:
    """Deterministic [0, 1) bucket for a challenger id (reproducible in tests).

    Hash the challenger_version (NOT random) so the same candidate always lands
    in or out of the sample — a test can assert a known id's outcome, and a
    re-run never re-rolls the dice. FNV-1a over the bytes, scaled to [0, 1).
    """
    h = 0x811C9DC5
    for b in challenger_version.encode("utf-8"):
        h ^= b
        h = (h * 0x01000193) & 0xFFFFFFFF
    return h / 0xFFFFFFFF


def is_marginal(verdict: PromotionVerdict, decisive_gain: float) -> bool:
    """True iff the promotion is marginal: local mean_delta < 2× decisive gain.

    `decisive_gain` is the gate's decisive mean-score threshold (typically
    `GateThresholds.mean_score_delta`). A promotion whose local mean_delta sits
    below `marginal_factor × decisive_gain` is barely over the bar — exactly
    where a drifting judge flips the call — so it is spot-checked 100%.
    """
    return verdict.mean_delta < DEFAULT_MARGINAL_FACTOR * decisive_gain


def should_spotcheck(
    verdict: PromotionVerdict,
    decisive_gain: float,
    *,
    rate: float = DEFAULT_SPOTCHECK_RATE,
    marginal_factor: float = DEFAULT_MARGINAL_FACTOR,
) -> bool:
    """Decide whether to cloud-spot-check this accepted verdict.

    Per GATE-CONTRACT #8:
      * 100% of MARGINAL promotions (mean_delta < marginal_factor × decisive
        gain) — a drifting judge does the most damage there;
      * ~`rate` (default 10%) of all other accepts, sampled DETERMINISTICALLY by
        hashing the challenger_version so it's reproducible.

    A reject is never spot-checked by the caller, but this returns False on a
    non-accept too (defensive — nothing to validate if we're not promoting).
    """
    if not verdict.accept:
        return False
    if verdict.mean_delta < marginal_factor * decisive_gain:
        return True
    return _sample_hash(verdict.challenger_version) < rate


# ── the governor wrapper ──────────────────────────────────────────────────────

# A spot-check prompt + the response the local judge graded, so the cloud judge
# re-grades the SAME (prompt, response) pair. The caller threads these on the
# spec / experiment result; the governor stays agnostic to where they came from.
SpotcheckSample = tuple[str, str, float]  # (prompt_text, response_text, local_score)


def _freeze_verdict(verdict: PromotionVerdict, corr: float, threshold: float) -> PromotionVerdict:
    """Flip an accepted verdict to a FROZEN reject, preserving all audit fields."""
    reason = (
        f"FROZEN: local judge drift, Spearman {corr:.2f} < {threshold:.2f} "
        "— re-fit grader (GATE-CONTRACT #8)"
    )
    return replace(
        verdict,
        accept=False,
        reject_reasons=verdict.reject_reasons + (reason,),
    )


def cloud_spotcheck_gate(
    decide_fn: Callable[..., PromotionVerdict],
    cloud_judge: CloudJudge,
    tracker: DriftTracker,
    decisive_gain: float,
    *,
    samples_fn: Callable[..., list[SpotcheckSample]] | None = None,
    rate: float = DEFAULT_SPOTCHECK_RATE,
    marginal_factor: float = DEFAULT_MARGINAL_FACTOR,
) -> Callable[..., PromotionVerdict]:
    """Wrap a `gate_decide`-compatible `decide_fn` with the #8 drift governor.

    Returns a callable with the same signature as `mesh.eval.gate.decide`
    (`(champion, challenger, thresholds=..., ...) -> PromotionVerdict`). It:

      1. calls `decide_fn(*args, **kwargs)` — the real gate, UNTOUCHED;
      2. if the verdict is NOT accept, returns it unchanged (no spot-check);
      3. if `should_spotcheck(verdict, decisive_gain)` is False, returns it
         unchanged;
      4. otherwise re-grades the spot-check samples with `cloud_judge`, records
         each (local, cloud) pair in `tracker`, and — if `tracker.should_freeze()`
         — returns a FROZEN verdict (`accept=False` + a re-fit reason). Else the
         original accept passes through.

    The spot-check samples come from `samples_fn(champion, challenger)` — the
    (prompt, response, local_score) triples the local judge graded. Default:
    pull a `_spotcheck_samples` list off the challenger row if present, else
    skip the cloud call gracefully (record nothing — can't validate without the
    graded pairs, so we don't fabricate a correlation point).
    """

    def _default_samples(champion: dict[str, Any], challenger: dict[str, Any]) -> list[SpotcheckSample]:
        raw = (challenger or {}).get("_spotcheck_samples") or []
        out: list[SpotcheckSample] = []
        for item in raw:
            try:
                out.append((str(item[0]), str(item[1]), float(item[2])))
            except (TypeError, ValueError, IndexError, KeyError):
                continue
        return out

    get_samples = samples_fn or _default_samples

    def _governed(*args: Any, **kwargs: Any) -> PromotionVerdict:
        verdict = decide_fn(*args, **kwargs)
        if not verdict.accept:
            return verdict
        if not should_spotcheck(
            verdict, decisive_gain, rate=rate, marginal_factor=marginal_factor
        ):
            return verdict

        # Resolve champion/challenger from the call (positional or kw) so the
        # samples_fn can locate the graded prompts the local judge scored.
        champion = kwargs.get("champion", args[0] if len(args) > 0 else {})
        challenger = kwargs.get("challenger", args[1] if len(args) > 1 else {})
        for prompt_text, response_text, local_score in get_samples(champion, challenger):
            cloud_score = cloud_judge.score(prompt_text, response_text)
            tracker.record(
                local_score,
                cloud_score,
                challenger_version=verdict.challenger_version,
            )

        if tracker.should_freeze():
            return _freeze_verdict(verdict, tracker.correlation(), tracker.corr_threshold)
        return verdict

    return _governed


__all__ = [
    "CloudJudge",
    "DEFAULT_CORR_THRESHOLD",
    "DEFAULT_MARGINAL_FACTOR",
    "DEFAULT_MIN_PAIRS",
    "DEFAULT_SPOTCHECK_RATE",
    "DEFAULT_WINDOW",
    "DriftTracker",
    "SpotcheckPair",
    "SpotcheckSample",
    "cloud_spotcheck_gate",
    "is_marginal",
    "should_spotcheck",
    "spearman",
]
