# Lane Handoff Contract

> Lifted with permission of the Apache-2.0 license from ARIS
> [`shared-references/assurance-contract.md`](https://github.com/wanshuiyin/Auto-claude-code-research-in-sleep/blob/main/skills/shared-references/assurance-contract.md),
> adapted to slancha-mesh's promotion-gate-on-holdout discipline.

Every lane sign-off — whether it gates a promotion (`P0` live-wiring →
`P1` clustering → `P2` head retrain → `P3` specialist FT) or just hands a
corpus to the next lane — emits a machine-readable verdict. This contract
specifies the schema, the six allowed verdicts, and what each verdict means
for downstream gating.

## Why a typed enum, not prose

The loop's promotion path has one canonical gate: `mesh/eval/holdout.py`
non-regression. The persona-review (loop scope §"Persona-review findings")
flags two related risks:

- **Closed loop + long delay + noisy sensor → oscillation.** A typed
  verdict makes "the gate failed" indistinguishable across peers and
  retrieval contexts (e.g., after a session compaction or peer handoff).
- **Adapter/base fingerprint mismatch = silent garbage.** Same shape of
  problem: a checked thing must produce a verifiable artifact, not a vibe.

Prose verdicts ("looks good", "needs work") fail both: they're peer-
specific, context-specific, and unparseable by `gated-redeploy` automation.

## The six verdicts

| Verdict | Meaning | Audit ran? | Gate-blocking? |
|---|---|---|---|
| `PASS` | All checks passed | Yes | No |
| `WARN` | Issues found, none disqualifying | Yes | No |
| `FAIL` | Disqualifying issues found | Yes | **Yes** |
| `NOT_APPLICABLE` | Detector negative; nothing to audit (e.g., a lane whose inputs aren't yet produced upstream) | Audit phase ran; verdict artifact still written | No |
| `BLOCKED` | Audit should apply but prerequisites missing (e.g., holdout claims a regression but `results/` is empty) | Could not complete | **Yes** |
| `ERROR` | Audit invocation failed (network, timeout, malformed reviewer output) | Attempted but errored | **Yes** |

### Why `BLOCKED` is more dangerous than `NOT_APPLICABLE`

`NOT_APPLICABLE` means **we checked, there's nothing to verify** (the
detector returned negative). The artifact file exists and documents the
absence — verifiable from outside the agent.

`BLOCKED` means **the audit should have run but cannot** — e.g., lane-3
tries to compute holdout non-regression but the new candidate adapter's
inference outputs are missing. Treating this as `NOT_APPLICABLE` masks the
danger; `BLOCKED` surfaces it and blocks the gate.

### Why `NOT_APPLICABLE` is not the same as silent skip

A silent skip leaves no record — there's no way to distinguish "we checked
and there was nothing" from "we forgot." This contract makes that
distinction mandatory: the verdict artifact MUST be written even for
`NOT_APPLICABLE`.

## Required artifact schema

Every lane sign-off writes a JSON artifact (optionally with a sibling
human-readable Markdown). The JSON must contain at minimum:

```json
{
  "lane": "lane-3-judge-holdout",
  "verdict": "PASS",
  "reason_code": "holdout_non_regressive",
  "summary": "Holdout mean judge-score 4.12 (vs champion 4.09); 50/50 probe set; no schema violations.",
  "audited_input_hashes": {
    "mesh/eval/holdout_seed.jsonl": "sha256:a3f8...",
    "mesh/eval/probe_set.jsonl":    "sha256:b2d1...",
    "<candidate-adapter-fingerprint>": "sha256:c9e4..."
  },
  "trace_path": "docs/lanes/.traces/lane-3-judge-holdout/2026-05-28_run01/",
  "reviewer": {
    "did": "did:wire:onyx-ridge-8255653f",
    "model_family": "claude"
  },
  "generated_at": "2026-05-28T18:00:00Z",
  "details": {}
}
```

### Field semantics

- **`lane`** — kebab-case lane name; matches the SKILL.md file stem.
- **`verdict`** — one of the six above. Anything else is a contract violation.
- **`reason_code`** — short skill-specific string (e.g.,
  `holdout_non_regressive`, `cluster_unstable_across_epochs`,
  `judge_score_distribution_shift`). Enables grep-based filtering across
  many verdict artifacts.
- **`summary`** — one-paragraph human-readable. The reason for the verdict
  in operator-facing prose.
- **`audited_input_hashes`** — SHA256 of every file the audit consumed.
  Paths are relative to repo root for in-repo files, absolute otherwise.
  A future `tools/verify_lane_verdicts.sh` (analogous to ARIS's
  `verify_paper_audits.sh`) can recompute these and flag `STALE` if a file
  changed after the verdict was emitted. Directly mirrors loop invariant #4
  + the `base_fingerprint`/`CheckpointMeta` discipline.
- **`trace_path`** — directory containing the full reviewer prompt + response
  pair, if the lane was reviewed by another peer. Required for verdicts
  emitted by lanes that gate a promotion (lane-3, future P2/P3 lanes);
  optional otherwise.
- **`reviewer.did`** + **`reviewer.model_family`** — proves cross-family
  review invariant was honored (if it becomes mandatory; see ARIS
  `reviewer-independence.md`).
- **`generated_at`** — UTC ISO-8601.
- **`details`** — skill-specific structured data. e.g., for lane-3, the
  per-cluster judge-score deltas.

## Subskill contract: "Always Emit, Never Block"

Child lanes (auditors) follow this contract:

- **Always emit a verdict artifact**, even on detector-negative or error paths.
- **Never block the parent's flow themselves** — they only emit verdicts.
- **The parent** (`gated-redeploy`, the promotion automation, the next lane
  in the chain) decides whether a given verdict blocks. This decision lives
  in one place, not duplicated across each lane.

## Verifier contract (future tool)

`tools/verify_lane_verdicts.sh <run-dir>` (canonical name; resolution via a
future `integration-contract.md`) is the single source of truth for "are
mandatory lane verdicts complete and current?" It must:

1. Locate the lane manifest for `<run-dir>` (which lanes apply).
2. For each, check verdict JSON exists at expected path.
3. Validate JSON against the required-fields schema above.
4. Verify `verdict` is one of the six allowed values.
5. Recompute SHA256 of every file in `audited_input_hashes`; flag `STALE` on mismatch.
6. Verify `trace_path` exists and is non-empty for promotion-gating lanes.
7. Output structured JSON and exit 0 (all green) or 1 (any `FAIL` /
   `BLOCKED` / `ERROR` / `STALE` / missing artifact).

`gated-redeploy` invokes the verifier; non-zero exit blocks the promotion.

This tool is out of scope for the initial PR; the contract is documented now
so lanes can emit verdict artifacts in the expected shape from day 1.

## See also

- [`./README.md`](./README.md) — directory overview
- [`./lane-3-judge-holdout.SKILL.md`](./lane-3-judge-holdout.SKILL.md) — worked example
- `../SELF_ORGANIZING_LOOP_SCOPE.md` §"Persona-review findings" — the systems
  motivation for typed verdicts (oscillation risk + fingerprint discipline)
- ARIS upstream contract: <https://github.com/wanshuiyin/Auto-claude-code-research-in-sleep/blob/main/skills/shared-references/assurance-contract.md>
