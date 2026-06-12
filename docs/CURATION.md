# Curation — difficulty-ranked holdout + synthetic gap-fill

The stage between **ignition** (`mesh.generator`: a settled, high-volume
traffic cluster earns a train slot) and **training**. It answers the question
the gate depends on: *which rows are the exam, and which rows teach?*

```
graded traffic window
        │ cluster seam (slancha-local / injected)
        ▼
   subtask cluster ──► ignition gate (volume / drift / no-champion)
        │ ignited
        ▼
┌──────────────────── mesh.curation ────────────────────┐
│ 1. rank every REAL trace by difficulty   (scorer seam) │
│ 2. hardest slice  → frozen holdout       (the exam)    │
│    rest           → train pool                         │
│ 3. sparse distance bands of the train pool             │
│    → synthetic rows via SDG seam, provenance-stamped   │
└────────────────────────────────────────────────────────┘
        │
        ▼
 spec: gate.holdout_ref = content-hash of the EXAM bytes
 judge_model = "<grader>@<holdout-hash-short>"  (GATE-CONTRACT #6/#7)
```

## Why the hardest rows are the exam

A random holdout asks "is the specialist OK on average?" — the average is
where small models were already fine. The promotion claim that matters is
**"matches the teacher on the hard tail of real usage"**. So the exam *is*
the hard tail: rank the cluster's real traces by difficulty, freeze the top
slice, gate on it. A specialist that clears that bar has earned the route.

Demand defines the eval (GATE-CONTRACT binding #7): the exam is selected
from actual usage, content-hashed into `frozen://sha256:…`, and the judge is
keyed on those frozen bytes — a swapped or re-curated holdout is a new ref,
and judge-match fires.

## The difficulty scorer (seam)

`mesh.curation.DifficultyScorer` — `(trace, centroid) → score`, higher =
harder. The open default (`default_difficulty_scorer`) blends two signals
that need no extra infrastructure:

| term | signal | rationale |
|---|---|---|
| centroid distance | `1 − cos(embedding, centroid)` | an outlier of the pattern is harder than its bread-and-butter |
| grade shortfall | `1 − judge_score / max` | a prompt the frontier grader scored low on is *observed* difficulty |

Missing signal → neutral 0.5 (unknown ≠ easy). Deployments with richer
difficulty models (sub-cluster density, learned scorers, …) inject their own
scorer; ranking, selection, hashing and the guards are shared machinery.

## Synthetic gap-fill (seam)

After the exam is carved out, the train pool can have coverage holes —
distance bands of the cluster with few or no rows. `SyntheticGenerator` —
`(exemplars, n) → rows` — is the seam a deployment binds to an open
near-frontier model. Mesh's side of the contract:

- under-median bands are filled toward the median; empty bands borrow
  exemplars from the nearest populated band;
- every synthetic row is stamped `source="sdg"`, `sdg_model`, `sdg_band`,
  `sdg_exemplar_ids` — provenance is queryable forever;
- synthetic rows that duplicate a holdout prompt are dropped and counted.

## Hard guarantees (enforced, not advised)

1. **Synthetic never enters the holdout.** Holdout eligibility requires a
   real trace (`source != "sdg"`); gap-fill runs strictly on the train side;
   the boundary asserts it. The exam is real usage by construction.
2. **Holdout ∩ train = ∅.** Exact-duplicate prompts (normalized content
   hash) are dropped from the train pool and from synthetic output — the
   gate can't be aced by memorization. An optional embedding near-dup guard
   (`near_dup_cosine`) extends this to paraphrase-level leakage.
3. **Deterministic.** Content-hash tie-breaks make the ranking — and
   therefore the exam and its `holdout_ref` — independent of input order.
   Re-curating an unchanged cluster is a no-op.

## Wiring

```python
from mesh.curation import curate_cluster, write_curation
from mesh.generator import generate

def curate(cluster_traces, centroid):
    return curate_cluster(
        cluster_traces, centroid,
        synthetic_generator=my_sdg,      # bind your near-frontier model
        sdg_model="kimi-k2",
    )

generate(traces, queue_path=..., drift_state_path=..., curate_fn=curate)
```

With `curate_fn` bound, each ignited cluster's spec carries
`gate.holdout_ref` = the curated exam's content-hash (instead of the raw
centroid ref) and `generator.curation` = the audit manifest (sizes,
difficulty stats, leakage drops, SDG provenance). Without it, specs emit
exactly as before — the stage is opt-in.

Persistence: `write_curation(dir, result)` → `holdout.jsonl` (the frozen
exam), `train.jsonl` (real + synthetic, stamped), `manifest.json`.
