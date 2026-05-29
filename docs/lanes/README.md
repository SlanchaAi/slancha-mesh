# Lanes — coordination contracts for parallel loop work

> Status: **proposal** (2026-05-28; tracks issue #N). Files in this directory
> are normative iff merged. Until then, treat as RFC. Audience: peer agents
> (and humans) carving the [self-organizing loop](../SELF_ORGANIZING_LOOP_SCOPE.md)
> into parallel work-streams.

## Why this directory exists

The self-organizing loop (`docs/SELF_ORGANIZING_LOOP_SCOPE.md`) lays out the
data plane → control plane → improvement plane split and the phased plan
(Track 0, P0 → P4). When multiple peers collaborate on those phases, they
naturally carve the work into **lanes** — e.g.,

- *lane-1 P0 live-wiring*: serve daemon stamps `Scorer` → `registry.record_quality_observation` + writes `judge_score` into the replay store.
- *lane-2 P1 clustering*: stabilize `slancha_local/train/cluster.py` (centroid matching, auto-k).
- *lane-3 judge + held-out eval*: probe-set curation and `mesh/eval/holdout.py` — the gate every promotion must pass.

Until now the lane split has lived only in ad-hoc messages between peers
(wire / GH comments). Anyone joining mid-flight had to reconstruct the lane
definition from inbox archeology, and "is this lane done?" was a prose
judgment, not a typed enum.

This directory makes lanes **first-class files** with two guarantees:

1. Each lane has a canonical `<lane-name>.SKILL.md` describing inputs,
   outputs, exit criteria, and allowed tools.
2. Each lane handoff emits a `LANE_VERDICT.json` per
   [lane-handoff-contract.md](./lane-handoff-contract.md) — verdict is the
   gate output, machine-readable, indistinguishable across peers.

## Prior art

This pattern is lifted from the [ARIS skill library](https://github.com/wanshuiyin/Auto-claude-code-research-in-sleep)
(Apache-2.0; 81 skills + 9-document `shared-references/` system-contracts
directory). ARIS coordinates a similar multi-agent research workflow with
cross-model adversarial review and gated promotion on a curated holdout —
the same shape as the loop. Specifically:

- `SKILL.md` with YAML frontmatter (`name`, `description`, `argument-hint`,
  `allowed-tools`) → `<lane>.SKILL.md` here.
- `shared-references/assurance-contract.md` (the 6-state verdict schema) →
  `lane-handoff-contract.md` here.
- `AGENT_GUIDE.md`'s artifact-contracts table → §"Artifact contracts" below.

## Artifact contracts

> Stub — to be filled in by each lane's SKILL.md as it lands.

Lanes communicate through plain-text files in known locations, not via direct
calls. This is the same discipline as loop invariant #4 ("everything is a
projection of `GradedTrace`"); this table just makes the producer/consumer
edges explicit so the next peer in the chain doesn't have to guess.

| Artifact | Produced by | Consumed by | Schema |
|---|---|---|---|
| `<repo>/replay-store/*.jsonl` (with `judge_score`) | lane-1 P0 live-wiring | lane-2 P1 clustering, lane-3 judge corpus build | existing `ReplayEntry` |
| `<repo>/cluster-snapshots/*.json` (centroids + stable cluster ids) | lane-2 P1 clustering | lane-3 (cluster-conditional holdout), P2 head retrain | TBD by lane-2 |
| `mesh/eval/holdout_report.json` | lane-3 judge + held-out eval | gated-redeploy (P2/P3 promotion gate) | TBD by lane-3 |
| `LANE_VERDICT.json` (per handoff) | every lane | promotion gate / next lane | [lane-handoff-contract.md](./lane-handoff-contract.md) |

## Conventions

- **Lane file name**: `lane-<id>-<short-role>.SKILL.md` — kebab-case, role
  matches the loop-scope phase it serves.
- **Owner**: each lane's frontmatter `owner` field is a DID
  (`did:wire:<peer>-<keyhash>`) of the peer holding the lane. Multiple peers
  may co-own (comma-separated DIDs).
- **Status**: `proposal` → `active` → `done` → `superseded`. Mirrored in
  the file's first paragraph for human readers.
- **Verdict artifacts**: written to `docs/lanes/.verdicts/<lane>/<UTC-iso>_<event_id>.json`
  (gitignored if noisy; otherwise committed for forensic record).

## Open coordination questions

These are tracked in the proposal issue and will collapse to decisions as
lanes adopt the contract:

- Q1. **Directory location.** `docs/lanes/` (this PR) vs `.claude/skills/lanes/`
  (matches AGENTS.md's existing reference shape) vs `mesh/lanes/` (co-locates
  with the code). This PR picks `docs/lanes/` because lanes are coordination
  artifacts, not runtime code — but it's the easiest thing to move.
- Q2. **Reviewer independence.** ARIS hard rule: executor ≠ reviewer model
  family. Worth making mandatory for verdict-bearing audits given the
  persona-review finding "noisy sensor (LLM judge) → oscillation risk." This
  PR ships only the verdict schema; the independence rule is a future
  shared-references doc if accepted.
- Q3. **Trace dirs.** ARIS writes `.aris/traces/<skill>/<date>_run<NN>/` for
  every invocation. Worth lifting here as `docs/lanes/.traces/...` for
  forensic replay. Deferred to a follow-up PR if Q1 lands.

## See also

- [`./lane-handoff-contract.md`](./lane-handoff-contract.md) — 6-state verdict schema
- [`./lane-3-judge-holdout.SKILL.md`](./lane-3-judge-holdout.SKILL.md) — worked example
- `../SELF_ORGANIZING_LOOP_SCOPE.md` — the scope these lanes carve up
- `../../AGENTS.md` — repo-wide agent guidance (the routing index)
- ARIS upstream: <https://github.com/wanshuiyin/Auto-claude-code-research-in-sleep>
