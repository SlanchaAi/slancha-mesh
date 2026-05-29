---
name: lane-3-judge-holdout
description: Probe-set curation and held-out eval seed (`mesh/eval/holdout.py`) that gates every promotion per loop-scope §"The loop" step 5. Take this lane when a peer (lane-1 live-wiring, lane-2 clustering, future lane-4 head-retrain, lane-5 specialist-FT) needs an authoritative non-regression signal before their work promotes to production routing.
owner: did:wire:swift-harbor-4092b577
status: proposal
inputs:
  - mesh/eval/holdout_seed.jsonl                # curated, trusted; per persona-review (Security)
  - corpus/probe_set.jsonl                       # current probe set
  - <candidate-checkpoint>/CheckpointMeta.json   # base_fingerprint; per loop-scope failure modes
outputs:
  - mesh/eval/holdout_report.json                # consumed by gated-redeploy
  - docs/lanes/.verdicts/lane-3-judge-holdout/<UTC>_<event_id>.json
                                                 # 6-state verdict per lane-handoff-contract.md
exit_criteria:
  - holdout_report.json present, schema-valid
  - verdict in {PASS, WARN}
  - judge_score distribution non-regressive vs current champion (delta within tolerance)
  - audited_input_hashes verified (no STALE inputs)
allowed-tools: [Bash, Read, Grep, Glob, Write]
gitnexus-impact-required: true  # per AGENTS.md "Always Do" — run before editing mesh/eval/holdout.py
---

# Lane 3 — Judge + held-out evaluation

> **Status**: proposal (2026-05-28). Active once this PR merges.
> **Owner**: swift-harbor (collectively-authored across Copilot CLI sessions).

## Mission

Be the gate. Every adapter promotion (P2 head retrain, P3 specialist FT) and
every routing-table change candidate flows through this lane's holdout
evaluation. The output — a typed verdict per
[lane-handoff-contract.md](./lane-handoff-contract.md) — is the single signal
`gated-redeploy` consults.

Concretely: P0 live-wiring (lane-1) populates `judge_score` on real traffic;
P1 clustering (lane-2) proposes a new cluster taxonomy; a future P2 lane
retrains heads to those clusters. None of those changes reach production
routing without this lane's `PASS` (or `WARN` with operator override).

## Boundaries (what this lane does NOT do)

- Does **not** modify the holdout seed (`mesh/eval/holdout_seed.jsonl`). The
  seed is curated and trusted (loop-scope persona-review §Security: "the
  holdout seed must be curated/trusted, never auto-derived from possibly-
  poisoned traffic"). Adding/removing seed examples is an operator
  decision, not this lane's call. If the seed appears insufficient,
  emit `verdict: BLOCKED` with `reason_code: insufficient_seed_coverage`.
- Does **not** train or fine-tune anything. Pure evaluation.
- Does **not** route traffic. Pure evaluation.
- Does **not** mutate the registry. Writes `holdout_report.json` and a
  verdict artifact; `gated-redeploy` reads both.

## Inputs (must exist; `BLOCKED` if missing)

| Input | Source | Notes |
|---|---|---|
| `mesh/eval/holdout_seed.jsonl` | Operator-curated | Trusted; never auto-derived |
| `corpus/probe_set.jsonl` | lane-2 (current; future a probe-curator lane) | Current k=N probe set |
| `<candidate-checkpoint>/CheckpointMeta.json` | Lane requesting promotion | Must include `base_fingerprint`; refuse-load on mismatch per loop-scope failure-modes |
| Current champion `CheckpointMeta.json` | Registry | For delta computation |

If any input is missing → `verdict: BLOCKED`, `reason_code: <missing-input>`,
write artifact with the missing-path enumerated in `details`.

## Outputs

### `mesh/eval/holdout_report.json`

The structured eval output. Consumed by `gated-redeploy` and by this lane's
own verdict-emit step. Schema TBD by lane-3's first implementation PR (out
of scope for this proposal). Suggested fields: `champion`, `candidate`,
`mean_judge_score_delta`, `per_cluster_deltas`, `regression_flags`.

### Verdict artifact

Written to `docs/lanes/.verdicts/lane-3-judge-holdout/<UTC-iso>_<event_id>.json`
per [lane-handoff-contract.md](./lane-handoff-contract.md).

Exit-criteria-to-verdict mapping:

| Condition | Verdict | `reason_code` |
|---|---|---|
| All checks pass; delta within tolerance | `PASS` | `holdout_non_regressive` |
| Mean delta within tolerance but one or more cluster shows drift | `WARN` | `per_cluster_drift_warn` |
| Mean delta exceeds regression tolerance | `FAIL` | `holdout_regressive` |
| Any input missing | `BLOCKED` | `<missing-input>` |
| `base_fingerprint` mismatch on candidate | `BLOCKED` | `base_fingerprint_mismatch` |
| Judge invocation failed (network / OOM / parse) | `ERROR` | `judge_invocation_failed` |
| No applicable holdout entries for this candidate's domain | `NOT_APPLICABLE` | `domain_not_in_seed` |

## Workflow

```
1. Resolve inputs; bail BLOCKED on missing.
2. Verify base_fingerprint matches champion's base; bail BLOCKED on mismatch.
3. For each holdout entry: route to candidate, then to champion, then judge.
4. Aggregate per-cluster and overall judge_score deltas.
5. Compare against tolerance (champion mean ± epsilon).
6. Write holdout_report.json.
7. Emit verdict artifact per the mapping above.
8. Surface a one-line summary to the requesting lane via wire / PR comment.
```

## Reviewer-independence note (future)

If/when ARIS-style reviewer-independence becomes mandatory (open question Q2
in [`README.md`](./README.md)), step 3 above should route the **judge** call
to a different-family peer than the **executor** that produced the candidate.
Concretely: a candidate produced by a Claude-backed peer should be judged via
a Codex- or Gemini-backed peer's `LocalJudgeScorer`. This makes the noisy
sensor's noise less correlated with the candidate's noise — directly
addressing loop-scope persona-review §Systems-designer's oscillation risk.

This is a soft recommendation until the reviewer-independence shared-ref
doc lands. Until then, lane-3 should record the executor and reviewer
families in `reviewer.model_family` so the gap is visible in audit.

## See also

- [`./README.md`](./README.md) — directory overview + open coordination questions
- [`./lane-handoff-contract.md`](./lane-handoff-contract.md) — verdict schema
- `../SELF_ORGANIZING_LOOP_SCOPE.md` §"The loop" step 5 — the gate this lane embodies
- `../SELF_ORGANIZING_LOOP_SCOPE.md` §"Persona-review findings" — the systems
  + security reasoning that makes the holdout the single promotion guard
