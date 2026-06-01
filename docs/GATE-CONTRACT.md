# GATE-CONTRACT.md — the promotion gate every self-improving loop must obey

> **Status:** v0, co-authored across two independently-built self-improving loops that
> converged on the same gate — slancha-mesh (multi-node, taxonomy emergent from live
> traffic) and forge (single-box GB10, operator/queue-seeded experiments). The loops
> differ in *ignition* and *topology*; the gate is identical. This file is the canonical
> contract; each loop cites it and binds it to its own runner.

## Why a contract

A self-improving loop that applies every candidate it produces is an **unguarded
hill-climb** — it silently degrades the moment a candidate looks better on the mean
but is worse where it matters, or the moment the grader drifts. (See
[hexo-ai/sia](https://github.com/hexo-ai/sia): generations kept sequentially, eval
never used to accept/reject, no rollback — it continues through failure.)

The gate is the single thing that makes the loop **monotonic-or-flat, never silently
worse.** It is the differentiator, not the loop. Everything below is the minimum
contract a candidate must pass before it reaches production traffic.

## The invariants

A candidate is **PROMOTED** only if it clears *every* check. Any failure → the champion
stays and the candidate is archived (never served).

1. **Best-so-far champion.** A persisted `ChampionRegistry` holds the current best per
   task/cluster. Promotion is always *challenger vs reigning champion*, never
   challenger vs nothing.

2. **Per-axis non-regression.** The candidate must not be worse than the champion on
   **any** tracked axis — not just the headline mean. A candidate that lifts the mean
   while collapsing one domain/axis is a **ROLLBACK**, not a promotion. (This is the
   single highest-value check; mean-only gates ship regressions.)

3. **Minimum gain.** The candidate must beat the champion's primary metric by
   `>= min_gain`. A wash is not a promotion (avoids churn on noise).

4. **Stub / degenerate rejection.** Reject outright, before any comparison: missing or
   `NaN` primary metric, output below an absolute floor, or `n == 0`. A degenerate
   artifact can *never* promote. (slancha-mesh: a `meta.stub=True` training artifact is
   refused even if its scores would otherwise pass.)

5. **Minimum sample size (`min_n`).** Both champion and candidate need `n >= min_n`
   scored outputs, or the delta is noise, not signal — `REJECT_MIN_N`.

6. **Judge-model match.** A grade delta computed across *different* grader models (or
   grader versions) is meaningless. Stamp `judge_model` (id + version) into every
   metrics record at grade time; refuse champion-vs-challenger comparison when they
   disagree (`REJECT_JUDGE_MISMATCH`) unless `allow_judge_mismatch` is explicitly set.
   To compare again, re-grade the champion with the new judge.

7. **Frozen-holdout governor.** The set the gate scores against is a **version-pinned,
   hand-curated seed set — never auto-derived from live traffic.** This is what stops
   the loop from optimizing toward a drifting or poisoned signal (Goodhart). Both loops
   independently landed here: slancha-mesh's curated holdout; forge's frozen Paul-email
   embedding centroid (precomputed offline from a held-out corpus).
   - **Goodhart regression test (mandatory):** the gate's test suite must assert that
     adversarially-degraded output — e.g. corporate-slop with the known "AI-tell"
     phrases stripped — scores **lowest** on the frozen metric. If a phrase-stripped
     producer can climb the score, the gate is optimizing a detector, not quality.

8. **Cloud spot-check (when the grader is itself a model).** At promotion time, sample
   `K` held-out prompts, grade with an independent frontier/cloud judge, and require the
   local↔cloud grades to correlate. Catches local-judge drift before it poisons a
   promotion. Cheap: small `K`, promotion-time only.

9. **Oscillation guard (hysteresis).** A promotion gain must be **sustained across the
   holdout**, not won on a single noisy eval, and a fresh champion gets a
   **minimum lifetime** (K cycles) before it can be dethroned. Prevents A→B→A flapping
   on grader noise.

10. **Rollback.** A regression is recoverable, atomically. Two valid bindings:
    - *Champion-stays* (forge today): on regression the champion is simply never
      replaced; the candidate is archived.
    - *Adapters-as-pointers* (slancha-mesh, O(1)): promotion swaps an adapter
      *reference* over a shared base; rollback drops the ref → instant base fallback.
      In-flight requests keep the adapter they started with; only new requests pick up
      the swapped ref. A full base retrain uses heartbeat `health=draining` so the
      router routes around the node mid-train, then it rejoins.

## Two bindings of the same contract

| | **slancha-mesh** (multi-node) | **forge** (single-box GB10) |
|---|---|---|
| Ignition | clusters **emergent from live traffic** (embed → cluster → FT a specialist per stable, high-volume cluster) | operator/queue-seeded experiments (`queue.jsonl`) |
| Never co-host train+serve | enforced by routing: `health=draining` + mesh route-around | enforced by process: `stop_serving()` + kill `VLLM::EngineCore` before train |
| Rollback | adapters-as-pointers (O(1), instant base fallback) | champion-stays + checkpoint swap |
| Runner | (in progress, #82) | systemd `Restart=always` + circuit-breaker + idle-WAIT |

The experiment *source* and the *topology* differ. **The gate is the same.** A runner
on either side calls the same `evaluate_candidate(...)` after any scored experiment and
acts on the verdict.

## Reference verdicts

`PROMOTE` · `ACCEPT_FIRST` · `ROLLBACK_REGRESSION` · `REJECT_NO_GAIN` ·
`REJECT_STUB` · `REJECT_MIN_N` · `REJECT_JUDGE_MISMATCH`

Every verdict carries: the champion id, the gain, and the list of regressing axes — so a
promotion (or refusal) is auditable without reading logs. (Pairs with the provenance
fields on the eval row + verdict: artifact / holdout-manifest / corpus / base-model /
router-config hashes + code SHA.)

---

*Prior art / provenance: slancha-mesh `docs/SELF_ORGANIZING_LOOP_SCOPE.md`, issues
#55 (stub-gate) / #57 (provenance) / #65 (real PEFT) / #82 (loop-runner); forge's
`gate.py` + `runner.py`. Drafted from a 2026-06-01 cross-loop comparison over wire.*
