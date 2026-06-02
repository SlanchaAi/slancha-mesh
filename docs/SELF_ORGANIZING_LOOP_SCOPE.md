# Self-Organizing Specialization Loop — Design Scope

> Status: **design scope** (2026-05-27). Captures the architecture, the
> closed self-improvement loop, what already exists vs. what must be built,
> the phased plan, and the persona-review findings. Audience: future-me who
> forgot everything.

## The vision (one breath)

A boring, fast hot path — one OpenAI-compatible endpoint — wrapped by an
**optional, off-hot-path improvement plane** that clusters your traffic,
trains adapters for the clusters that pass a trusted holdout, and hot-swaps
them in — all shipped as `docker compose up`, with the mesh re-routing around
any node that's busy training. The specialist taxonomy is **emergent**, not
predefined: specialists are *discovered* from the operator's own traffic.

This is the differentiator. The LLM-router field (see RouterArena: Sqwish,
R2-Router, vLLM-SR, RouteLLM, …) routes across *existing* models. **None turn
idle compute into new specialists, and none route across machines.** That is
slancha's uncontested wedge.

**Prior art on the self-improving loop itself — and where we diverge.**
Autonomous self-improvement frameworks exist (e.g. [hexo-ai/sia](https://github.com/hexo-ai/sia):
meta-agent → target-agent → feedback-agent over N generations). They're useful
validation that the *shape* is sound, but SIA's `orchestrator.py` is an
**unguarded hill-climb**: each generation runs, is "evaluated," and is fed to
the next regardless of score — there is **no acceptance gate, no best-so-far
retention, no rollback** (a failed generation still spawns the next). That is
exactly the regression footgun our design forbids. Our loop's load-bearing
difference is invariant #5 below: **every promotion passes a curated holdout
gate (with per-domain non-regression), stub/low-quality artifacts are refused,
and `ChampionRegistry` keeps the prior champion for instant rollback.** Gated
promotion + rollback — not the generational loop — is the defensible part.

## Architecture — five invariants that keep it elegant

1. **Everything is an OpenAI-compatible endpoint.** Node, mesh-gateway,
   cloud, the fixed-taxonomy bootstrap, an emergent-cluster specialist, and a
   bring-your-own router (R2 / vLLM-SR) are all the same interface. The
   routing table is just `(cluster|domain) → endpoint`.
2. **Three planes; the hot path is trivial.**
   - **Data plane** — the node: `embed → route → serve → log trace`. Nothing else.
   - **Control plane** — registry / discovery / allocator: membership, routing table, quality. Event-sourced.
   - **Improvement plane** — grade → cluster → retrain heads → FT → gated redeploy. **100% offline, idle-gated, optional.**
   The magic is never on the critical path.
3. **Adapters as pointers.** Redeploy = swap an adapter *reference*, never
   replace a model. Reversible (drop adapter → base), composable, instant rollback.
4. **One canonical record: `GradedTrace`** `{prompt, embedding, route,
   response, grade, cluster}`. `ReplayEntry`, the `oracle` record,
   `MeshUsageEvent`, and the axolotl FT-bundle row are all *projections* of it.
5. **The holdout gate is the single promotion guard** — for retrained heads
   *and* new specialists. Nothing reaches traffic without a held-out
   non-regression against a **curated, trusted** seed set. The full set of
   invariants the gate must obey is the [`GATE-CONTRACT.md`](GATE-CONTRACT.md)
   — best-so-far · per-axis non-regression · stub-reject · min-n · judge-match ·
   frozen-holdout governor · sustained-gain hysteresis · cloud spot-check ·
   rollback. (Co-authored with a second, independently-built self-improving loop
   that converged on the same gate — strong validation it's the right invariant.)

## The loop

```
live traffic → embed (mmBERT ✓) → GradedTrace store  [persist embedding + grade]
  ── idle / offline ──
  1. CLUSTER embeddings → emergent clusters {C₁..Cₖ}      train/cluster.py ✓ (KMeans)
  2. RELABEL corpus by cluster → RETRAIN heads to clusters    heads ✓, corpus builders ✓
  3. per stable/high-volume cluster → FT a specialist         bundle.py ✓ / PEFT = REAL (behind `[train]` extra, #65; stub kept for contract tests) — GB10/Spark validation pending
  4. REDEPLOY: hot-load adapter → catalog/registry/router cutover
  5. GATE every promotion on eval/holdout mean-score          holdout ✓
  └──────────────────────────── repeat ────────────────────────────┘
```

## Exists vs. gaps (grounded in the repos)

**Exists:** mmBERT embedder + per-request embedding; `slancha_local/train/`
(`cluster_by_route`, `build_train_bundle` → axolotl JSONL); treelite heads;
corpus builders + `preclassify_corpus`; serve daemon + `TrainingPass` (stub
contract **+ real PEFT leg behind the `[train]` extra**, #65) +
checkpoint/merge contract; `idle.py` detector; `replay_store`; `eval/holdout`;
`registry.record_quality_observation`; `quality_probe` + `Scorer` protocol;
`quality_*` schema with three observation sources; node Docker image with the
classifier baked in.

**Gaps (the build):**
1. **Substrate** — persist `embedding + grade` in one `GradedTrace` record.
2. **The grader** — replace `StubScorer` with a real local-judge `Scorer`; wire grade → registry quality **and** → labeled replay (today the judge writes JSONL for a dashboard only).
3. **Cluster stability** — KMeans is run-unstable; need stable cluster identity (centroid matching / incremental) + auto-k bounded by fleet capacity. *(Deferred past v1 — see plan.)*
4. **Head retrain target** — heads predict fixed domains today; retrain to predict **cluster id**.
5. **Real per-cluster FT** — *done in code (#65)*: `TrainingPass(allow_real=True)` loads the base model + tokenizer (transformers), wraps with a `peft` LoRA config, trains a few steps on the replay corpus, and saves a real adapter via `save_pretrained` — `meta.stub=False` + a `base_model_fingerprint`, so the eval gate promotes it. Heavy deps are the optional **`[train]` extra** (`pip install -e ".[train]"`), imported lazily; absent them the real path raises `MissingTrainingDepsError` (never a silent stub fallback, #55). The stub path (`allow_stub=True`) is kept for the contract tests. **Remaining:** full GB10/DGX-Spark validation of a real pass under `gpu-launch --kind training` (acceptance step for #65).
6. **Gated redeploy** — adapter hot-swap + base-fingerprint guard + registry/router atomic cutover + champion/challenger promotion. *Partial (#65):* `ChampionRegistry` (in `mesh/training.py`) tracks the current champion adapter as a pointer and keeps the prior one so a failed promotion rolls back instantly (adapters-as-pointers). Hot-swap + router cutover still to wire.
7. **Container** — `docker/Dockerfile` + `docker/docker-compose.yml` ship the registry control plane (`mesh.registry_app`). The loop (generator + runner + gate) is bundled into one deployable image via the `[loop]` extra (see item 9) — "everything in the container", no second slancha-local runtime.
8. **Non-weight improvement lever** — *done in code (#79)*: weights aren't the only lever. `mesh/feedback.py` proposes **bounded** edits to the non-weight routing surface (`PrefVector` defaults, a per-cluster system preamble, tunable `SpecialistCard` fields — whitelisted in `ALLOWED_FIELDS`, **no free-form code/prompt-exec**) as **challengers** routed through the *same* `mesh.eval.gate.decide` and a config-pointer `ConfigChampionRegistry` (promote/rollback) as a retrained adapter. This is the deliberate divergence from SIA: SIA rewrites the harness *ungated*; here a config change must clear the curated holdout (per-domain non-regression, invariant #5) or it is rejected and **never applied**. **Human-gated:** `record_proposal` only proposes; `gate_config_proposal` → `promote_if_accepted` is the explicit, gate-guarded apply. Auto-promotion behind the gate is a later step. The reference proposer is rule-based + pluggable (an LLM proposer drops in behind the `ConfigProposer` Protocol — the gate, not the proposer, decides what ships).
9. **Ignition stage (generator)** — *done in code (#87)*: `mesh/generator.py` turns a window of graded traffic into queued, gated experiment specs — completing the GATE-CONTRACT fusion interface (ignition ⊥ runner #82 ⊥ gate). It reuses the **public slancha-local** clustering substrate (`cluster_by_route` — mmBERT embeddings, KMeans-per-route, stable cluster identity) rather than vendoring it: the **bundle decision** (operator: "everything in the container"). slancha-local is installed into the same image via the **`[loop]` extra**, imported lazily, and the clustering function is injectable, so the adapter + its tests run with no numpy/sklearn/slancha-local present. It applies the ignition gate (`n_traces≥500`, centroid `drift<0.15` ×3 consecutive windows, **no healthy champion**), emits a GATE-CONTRACT spec with the cluster centroid as the frozen judge anchor (binding #7), and enqueues via `loop_runner.enqueue` (idempotent — spec id embeds the centroid hash). **Remaining:** wire a serve-time `GradedTrace` producer (the embedder runs at serve time) + a registry-backed `has_healthy_champion` predicate; package the loop image (`mesh[loop]`) as a compose service.

## The grader = 3 tiers = the free/paid line

| Tier | Source | Where | Tier |
|---|---|---|---|
| 0 — synthetic / local judge | `synthetic` | probe set scored by an on-mesh model or held-out exact-match | **free / OSS** |
| 1 — shadow / real, local judge | `shadow_traffic` / `real_traffic` | sample X% of live traffic, re-judge locally | free / OSS |
| 2 — cloud-oracle | `real_traffic` (cloud judge) | frontier judge → premium labels for routing **and** FT corpus | **paid, opt-in** |

The schema's three `observation_source` values already map onto this ladder.
Local grading is free and good enough to prove routing + bootstrap FT;
cloud-oracle labels are the paid upgrade that makes *better* specialists.

## Phased plan (the convergence cut)

**Decoupled headline stories:** (A) self-organizing **single node** and
(B) **mesh federation** ship independently. The wow-demo needs zero
tailnet/registry/allocator.

- **Track 0 — infra (now, independent of ML):** compose **profiles**
  (`cpu`/`nvidia`) on the node; a **slancha-mesh registry container + mesh
  compose** (the missing L2 on-ramp); `slancha demo` replay (visible loop in
  ~2 min); publish images so `docker compose up` needs no clone. Layered
  on-ramp: **L1 node → L2 mesh → L3 self-improving**, each opt-in.
- **P0 — substrate:** `GradedTrace` (one record, +embedding +grade);
  local-judge `Scorer` → registry quality + labeled replay. Flips
  `quality_router_observed` null→real with no ML. *(Also closes the audit's
  routing-quality gap.)*
- **P1 — taxonomy, read-only, single node:** seeded fixed-k cluster the graded
  corpus → candidate taxonomy; observe vs. the fixed taxonomy on holdout.
- **P2 — gated head retrain, single node:** retrain heads to clusters,
  champion/challenger on the curated holdout. **← first self-organizing release.**
- **P3 — specialist FT + gated redeploy:** real PEFT *(landed in code, #65 —
  `TrainingPass(allow_real=True)` + `[train]` extra; GB10/Spark validation
  pending)*; adapters-as-pointers + `base_fingerprint` *(`ChampionRegistry`
  rollback landed, #65)*; mesh-drain-on-train (frees the node's GPU); atomic
  hot-swap. **← the "compute → specialists" release.**
- **P4 — federation + cloud-oracle (paid):** mesh-wide clusters, cloud labels.

**Cut from v1:** HDBSCAN auto-k + stability alignment (seeded fixed-k first),
the cloud (Tier-0 only), cross-node federation, multi-trust.

## The two cadences (shapes the scheduler)

- **Heads retrain cheap** (CPU minutes) → follow cluster drift often, reversible.
- **Specialists FT expensive** (GPU hours) → only for **stable, high-volume**
  clusters, idle-gated, conservative promotion (hysteresis + min-lifetime).

## Persona-review findings (load-bearing)

- **Systems designer:** closed loop with long delay (FT hours) + noisy sensor
  (LLM judge) → oscillation risk. The holdout gate is the governor;
  adapters-as-pointers give instant rollback; the slow loop must be conservative.
- **GPU researcher (mesh-native fix):** idle-FT and serving fight for the same
  VRAM. **On entering `TRAINING`, the mesh re-routes the node's traffic to
  peers, freeing its GPU.** No single-box router can do this. Also: small
  specialists win on *narrow distribution* (style/jargon/format), not raw
  capability — target those clusters; the holdout auto-rejects the rest.
- **Security:** poisoned traffic → poisoned corpus → poisoned specialist; the
  judge can be gamed. → the **holdout seed must be curated/trusted** (never
  auto-derived from possibly-poisoned traffic); gate + instant adapter rollback
  contain it. Federation shares centroids/deltas, never prompts.
- **SRE:** a self-modifying system needs `slancha freeze` (disable
  auto-promote), `slancha rollback <specialist>` (drop adapter), budget caps
  (max FT-passes/day, GPU-hr ceiling), and loud signals (cluster-churn,
  promote accept/reject, holdout trend). Every promotion is an event
  (registry is event-sourced).
- **Failure modes:** **adapter/base fingerprint mismatch** = silent garbage →
  fingerprint base into `CheckpointMeta`, refuse-load on mismatch. Cold-start
  invisibility → ship pre-trained fixed heads + `slancha demo`. Atomic
  hot-swap (drain in-flight). Seed clustering for determinism/auditability.

## Economics / moat

- **The moat is not the code** (OSS) — it's the **accumulated per-operator
  personalized adapters + cloud-oracle label quality**, which compound with use
  (switching cost + data flywheel).
- **Free/paid follows the plane split, as a rule:** the **data plane never
  carries a paid dependency** (preserves the no-cloud OSS promise); hosted
  control plane + cloud-oracle labels + managed FT compute = paid.
- **Cost-trust:** train only when idle (optionally in a cheap-power window);
  surface "this pass ≈ N GPU-hours."
- **Trust-domain boundary (v1):** the mesh is **your own boxes** (single trust
  domain). Sharing *other people's* random compute (untrusted code execution,
  sandboxing, credits) is explicitly **out of v1**.

## Containerization (shareability linchpin)

Containerizing the node fixes three things at once: cross-OS install pain (the
Windows saga), the classifier silent-degrade (deps baked in), and FT-dep
isolation. Plan:
- **`slancha-node`** (slim, GHCR): onnx embedder + treelite heads baked in +
  router + Ollama backend. `docker compose up` = L1. Compose profiles `cpu` /
  `nvidia`. *(Image already bakes the classifier — verified.)*
- **`slancha-trainer`** (fat, separate: torch + peft + axolotl + CUDA) — only
  training nodes pull it. Matches the optional-extra split in `pyproject`.
- **`slancha-registry`** — the control plane. **Missing today** — add it + a
  mesh compose.
- **tailscale sidecar** (userspace) for cross-machine.
- **Not mandatory:** keep pip/bare-metal; v0.2 Rust single-binary is the slim
  path. Containers = default reproducible on-ramp.

## KPIs / the demo that proves it

- `quality_router_observed` populated (`sample_count > 0`) for every served specialist.
- `eval/holdout` mean judge-score measurable across router/specialist versions.
- **The headline demo:** *"mean held-out judge score climbs over a week of your
  own traffic — with no human picking domains or models."* Self-organizing,
  measurable, honest. `slancha demo` shows a compressed version in ~2 minutes.

## Open assumptions to verify

- **A1** local judge good enough to grade (med confidence; the 472e oracle pass used one). Mitigate: curated holdout + cloud spot-check.
- **A2** idle GPU room for a LoRA pass alongside serving — *unsolved*; the mesh-drain answer (P3) is the plan.
- **A3** on-node graded volume enough to move a specialist (bounded replay ring → multi-day accumulation + persistence).
