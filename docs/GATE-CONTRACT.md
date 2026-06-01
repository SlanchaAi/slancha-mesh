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

8. **Cloud spot-check (when the grader is itself a model).** At promotion time, grade a
   sample with an independent frontier/cloud judge and require the local↔cloud grades to
   correlate. Catches slow local-judge drift before it poisons a promotion.
   - **Sampling:** spot-check ~10% of `PROMOTE` verdicts + **100% of marginal** ones
     (gain `< 2× decisive_gain`) — that's where a drifting local judge does the most
     damage.
   - **Threshold:** track Spearman correlation between local and cloud scores on the
     spot-check sample over a rolling window; `corr < ~0.7` ⇒ **freeze promotions** +
     trigger a re-fit of the local grader (re-embed the frozen holdout / re-anchor the
     centroid).
   - **Composition with #7:** #7 (goodhart_guard) is the cheap, every-cycle hard tripwire
     for gameable-by-construction; #8 is the sampled, cloud-token-costing early-warning
     for slow drift. Belt **and** suspenders — run both.

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
| Runner | `mesh/loop_runner.py` (#82, merged) — injected seams, CPU-testable | systemd `Restart=always` + circuit-breaker + idle-WAIT |

The experiment *source* and the *topology* differ. **The gate is the same.** A runner
on either side calls the same gate (`gate.decide(...)` / `evaluate_candidate(...)`) after
any scored experiment and acts on the verdict.

## Experiment-spec interface (the generator ↔ runner ↔ gate seam)

The contract is wired through **one queue line** — an append-only `queue.jsonl` of
experiment specs. This decouples three swappable stages: **ignition (generator) ⊥
execution+safety (runner) ⊥ promotion (gate).** A spec emitted by one loop's generator
runs on the other loop's runner unmodified, because both speak this line.

```jsonc
{
  "id": "ft_cluster_c0427_<ts>",   // unique; runner dedups by id
  "type": "train",                  // train | eval
  "priority": 5,                    // lower = higher
  "source": "traffic_cluster",      // provenance of ignition: "traffic_cluster" | "operator"
  "generator": {                    // ignition payload; runner ignores fields it doesn't need
    "cluster_id": "c_0427",
    "centroid_ref": "frozen://sha256:5ab9456e…", // content-hash of the cluster centroid — see "frozen refs are content-hashes" + "centroid is the judge"
    "n_traces": 1840,               // rolling-window volume (drives priority + the ignition gate)
    "drift": 0.11,                  // centroid cosine drift across recent windows
    "exemplar_trace_ids": [ ... ]   // for corpus assembly
  },
  "cmd": "<train invocation>",      // runner wraps this with the GB10 safety gates
  "gate": {                         // the gate binding — REQUIRED; runner passes ALL of it to the gate
    "task": "cluster:c_0427",       // champion-registry key
    "primary": "mean_holdout_score",
    "axes": ["per_domain_score", "coherence"],   // per-axis non-regression set
    "floor": 1.0,                   // stub-reject below this
    "min_gain": 0.0,                // separate floor a gain must clear at all
    "min_n": 20,                    // a.k.a. min_n_eval in slancha-mesh GateThresholds
    "judge_model": "qwen3-8b@5ab9456e",  // "<grader-id>@<holdout content-hash short>" — keys judge-match on the FROZEN BYTES, not just a model name (closes the silent-swap gap)
    "min_champion_lifetime_s": 3600,// hysteresis: fresh champion can't be dethroned on a marginal gain
    "decisive_gain": 2.0,           // ABSOLUTE primary-metric delta; gain ≥ this bypasses hysteresis (not a multiple of min_gain)
    "holdout_ref": "frozen://sha256:<centroid-hash>+<stats-hash>"  // centroid bytes AND its {mean,std} normalization stats
  },
  "env": { "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True" },  // single-token form only (torch 2.11)
  "status": "pending"
}
```

**Field-name note (portability):** `min_n` (forge) ≡ `min_n_eval` (slancha-mesh
`GateThresholds`). A runner accepts either; emit `min_n` for cross-loop portability.

**Frozen refs are content-hashes, not paths.** `centroid_ref` / `holdout_ref` use
`frozen://sha256:<64hex>`; the runner resolves the hash against a content-addressed store
and **verifies the bytes on load**. This is not cosmetic — it makes "frozen" *provable*.
A bare path can be silently swapped: the judge drifts but `judge_model`'s id is unchanged,
so judge-match (#6) can't catch it. With a content-hash ref, any change to the frozen
judge = a new ref = a new `judge_model` short-hash = judge-match **fires by construction**.
So the spec stamps `judge_model = "<grader-id>@<holdout-content-hash-short>"`, keying #6 on
the actual frozen bytes rather than a mutable name. A runner MAY accept a bare path in dev,
but MUST emit `frozen://sha256` for portability + audit. `decisive_gain` is an **absolute**
primary-metric delta (`gain = candidate[primary] − champion[primary]`), distinct from
`min_gain` (the floor a gain must clear at all). *(Both pinned with forge, 2026-06-01.)*

### Contract rules for the seam

1. **Generator never trains or serves.** It is a pure function of (traffic window →
   candidate specs): it emits the spec + the frozen `centroid_ref`/`holdout_ref`, nothing
   more. slancha-mesh's generator = an mmBERT cluster pass; forge's = the operator queue.
   Both write the same line.
2. **Runner never decides quality.** It executes under the safety gates and calls the gate
   per this contract. Generator-blind; gate-deferring.
3. **`source` + `generator` are advisory to the runner, authoritative for audit** — a
   verdict traces back to *which traffic cluster* (or operator) ignited it. Pairs with the
   provenance hashes.

### Ignition gate (when a cluster earns a train slot)

A `traffic_cluster`-sourced `train` spec should only be emitted when **all** hold
(field-tested defaults from forge throughput; tune per deployment):

- **Volume:** `n_traces >= 500` in the rolling window. Below that, don't mint a
  specialist — route to base.
- **Stability:** centroid cosine `drift < 0.15` across `>= 3` consecutive windows. Ignite
  on a *settled* cluster, never a transient spike.
- **Need:** ignite only if **(no champion for this cluster)** OR **(the existing champion
  is regressing on a fresh cluster eval)**. Never retrain a winning specialist.

### The cluster centroid *is* the frozen judge

For a traffic-emergent specialist, the cluster's own centroid (frozen at ignition) is the
eval anchor for that experiment — **demand defines the eval.** This makes
traffic-emergent ignition **Goodhart-resistant by construction**: the thing being
optimized for (serving this cluster's real traffic well) is the thing being measured, so
there is no separable detector to game. It is the strongest binding of invariant #7
(frozen-holdout governor): the holdout isn't curated *near* the traffic, it *is* the
traffic's own shape, version-pinned.

## Reference verdicts

`PROMOTE` · `ACCEPT_FIRST` · `ROLLBACK_REGRESSION` · `REJECT_NO_GAIN` ·
`REJECT_STUB` · `REJECT_MIN_N` · `REJECT_JUDGE_MISMATCH` · `REJECT_HYSTERESIS`

Every verdict carries: the champion id, the gain, and the list of regressing axes — so a
promotion (or refusal) is auditable without reading logs. (Pairs with the provenance
fields on the eval row + verdict: artifact / holdout-manifest / corpus / base-model /
router-config hashes + code SHA.)

---

*Prior art / provenance: slancha-mesh `docs/SELF_ORGANIZING_LOOP_SCOPE.md`, issues
#55 (stub-gate) / #57 (provenance) / #65 (real PEFT) / #82 (loop-runner); forge's
`gate.py` + `runner.py`. Drafted from a 2026-06-01 cross-loop comparison over wire.*
