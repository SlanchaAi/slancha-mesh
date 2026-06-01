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
   non-regression against a **curated, trusted** seed set.

## The loop

```
live traffic → embed (mmBERT ✓) → GradedTrace store  [persist embedding + grade]
  ── idle / offline ──
  1. CLUSTER embeddings → emergent clusters {C₁..Cₖ}      train/cluster.py ✓ (KMeans)
  2. RELABEL corpus by cluster → RETRAIN heads to clusters    heads ✓, corpus builders ✓
  3. per stable/high-volume cluster → FT a specialist         bundle.py ✓ / PEFT = STUB (contract-only stub today; real PEFT tracked in #65)
  4. REDEPLOY: hot-load adapter → catalog/registry/router cutover
  5. GATE every promotion on eval/holdout mean-score          holdout ✓
  └──────────────────────────── repeat ────────────────────────────┘
```

## Exists vs. gaps (grounded in the repos)

**Exists:** mmBERT embedder + per-request embedding; `slancha_local/train/`
(`cluster_by_route`, `build_train_bundle` → axolotl JSONL); treelite heads;
corpus builders + `preclassify_corpus`; serve daemon + `TrainingPass` (stub) +
checkpoint/merge contract; `idle.py` detector; `replay_store`; `eval/holdout`;
`registry.record_quality_observation`; `quality_probe` + `Scorer` protocol;
`quality_*` schema with three observation sources; node Docker image with the
classifier baked in.

**Gaps (the build):**
1. **Substrate** — persist `embedding + grade` in one `GradedTrace` record.
2. **The grader** — replace `StubScorer` with a real local-judge `Scorer`; wire grade → registry quality **and** → labeled replay (today the judge writes JSONL for a dashboard only).
3. **Cluster stability** — KMeans is run-unstable; need stable cluster identity (centroid matching / incremental) + auto-k bounded by fleet capacity. *(Deferred past v1 — see plan.)*
4. **Head retrain target** — heads predict fixed domains today; retrain to predict **cluster id**.
5. **Real per-cluster FT** — `_apply_lora_step` (`time.sleep`) → real PEFT/axolotl.
6. **Gated redeploy** — adapter hot-swap + base-fingerprint guard + registry/router atomic cutover + champion/challenger promotion.
7. **No mesh container** — slancha-mesh has no Docker/compose (only systemd units).

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
- **P3 — specialist FT + gated redeploy:** real PEFT; adapters-as-pointers +
  `base_fingerprint`; mesh-drain-on-train (frees the node's GPU); atomic
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
