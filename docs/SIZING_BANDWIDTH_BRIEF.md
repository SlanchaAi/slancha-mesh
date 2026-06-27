# Brief — Memory-Bandwidth-Sensitive Sizing for Slancha-Mesh

**Status:** design brief, hardened after verification + adversarial review (2026-06-27).
Internal. Hardware figures are externally-sourced and trust-tagged; everything else is
code-grounded against `/Users/laul_pogan/Source/slancha-mesh` (the live target — three
sibling repos carry a `mesh/` dir, but only this one has `idle.py`, `backends.py`, and
the full selector).
**Scope:** the allocator/sizing path (`mesh/allocator.py`, `mesh/probe.py`,
`mesh/models.py`, `SpecialistCard`). Routing-time selection (`select.py`/`router_app.py`)
is **in-scope as a prerequisite**, not out of scope — see §0: the SLO gate has *no live
caller* until the pref-aware selector is wired into the running router, so the rollout
now opens with that wiring step rather than assuming it.

---

## 0. The wiring reality that reorders everything (read first)

There are **two selection layers**, and only one is live:

1. **Library selector** — `mesh/select.py`. Rich: `route_class`
   (`hot_interactive|standard|batch`, `select.py:50`), `PrefVector`
   (`select.py:211–228`, already carries `min_throughput_tps`), ceilings, structured
   decision reasons. **Invoked only by tests and offline scripts**
   (`mesh/scripts/mesh_replay.py`, `build_corpus`).
2. **Live router** — `mesh/router_app.py`. Routes purely by
   `body['model'] == specialist_id` + `_reachable_bindings` (`router_app.py:591–611`).
   It does **not** call `select_mesh_route_with_pref`, carries no `route_class`/`PrefVector`,
   and never counts prompt tokens. `router_app.py:26–29` states classifier/domain routing
   is "a followup that depends on slancha-api's classifier."

**Consequence:** the SLO gate (§3.3) lives in `select_mesh_route_with_pref`, which has
**no caller on the live `/v1` path**. Per the no-shelf-ware rule, flipping the gate is a
no-op for production until layer (1) is wired into layer (2). That wiring is itself gated
on slancha-api's classifier being importable. So the rollout opens with **Step 0: wire the
pref-aware selector into `router_app.py` (or slancha-api's selector)**; steps that change
allocation/selection behaviour are dead code before it lands.

Also load-bearing and already true in code: `PrefVector.min_throughput_tps` **is the
`min_decode_tps` floor in semantic form, and it is deliberately disabled** —
`_reject_unsupported_pref` raises `ValueError` (`select.py:285–306`) because no per-route
throughput is measured. So the floor is **not a new field**; the work is to *un-disable* an
existing one once §4 supplies the measurement. Do not add a second competing throughput
field.

---

## 1. Thesis

Decode latency is **memory-bandwidth-bound at batch-1**: each generated token must
re-read the resident weights (plus the per-context KV cache) from memory, and arithmetic
intensity is ~O(1), so

```
decode_tok_per_s  ≈  MBU · node_memory_bandwidth  /  bytes_moved_per_token        (UPPER BOUND, batch-1, dense)
bytes_moved_per_token = weight_bytes + kv_bytes_per_token · context_tokens
```

This is an **upper bound under stated conditions**, not a general law. It holds for our
interactive/agent/voice traffic (low concurrency) and breaks in four regimes the sizing
model must respect (§3.6): real MBU is ~60–85% not 100%; **batch>1** amortizes the
weight read across the batch and crosses over to compute-bound; **MoE** moves only
*active* params per token, not total; **long context** lets the KV term dominate the
weight term.

Capacity ("does the model **fit**") and bandwidth ("does it **decode fast enough**") are
**orthogonal** constraints. The allocator today hard-gates the first (`min_vram_gb`,
`allocator.py:111`) and only *softly* scores the second
(`throughput_score`, weight 1.5, `allocator.py:131–136`). A model can therefore be placed
where it fits in VRAM and then decodes at single-digit tok/s — the GB10 failure mode
(273 GB/s unified LPDDR5X) versus a 600W RTX PRO 6000 Blackwell (96GB GDDR7,
1.792 TB/s — ~6.6×).

**Goal:** make predicted decode tok/s a **first-class, SLO-gating sizing dimension** —
*measured* per node, *modelled* per (model, context, batch regime), enforced as a hard
filter for interactive traffic **with a wired degraded-coverage fallback** — while keeping
the capacity gate intact.

---

## 2. What already exists vs. the gap

**Already in the code (don't rebuild):**

- `NodeProbe.memory_bandwidth_gbs: float | None = None` (`models.py:72`; docstring
  `models.py:47–49` notes GB10 doesn't expose it via `nvidia-smi`). **It is not `None` on
  GB10 in practice** — `mesh/probe.py` fills it from a **hardcoded table**
  `_MEMORY_BANDWIDTH_GBS` (`probe.py:51–58`), with `'NVIDIA GB10' = 273.0` (`probe.py:52`).
  So today GB10 is served a **datasheet constant**, not a measurement and not the
  estimated-tps fallback. (Correcting the prior draft's "None → falls back to
  `estimated_tps_at`" framing — that path only fires for chips *absent* from the table,
  `probe.py:383–386`.)
- `_estimated_tps(spec, node)` (`allocator.py:59–80`): primary path
  `node.memory_bandwidth_gbs / max(spec.runtime_gb, 0.1)` (`allocator.py:67`) — the correct
  physics; fallback to `spec.estimated_tps_at[<chip>]` (`allocator.py:68–80`, default
  **20.0** tok/s at `allocator.py:80`) **only when `memory_bandwidth_gbs` is falsy**. Guard
  is truthiness (`if node.memory_bandwidth_gbs:`), so a probed **0.0** also falls through to
  the table — equivalent to `None` in practice.
- `throughput_score` folded into `model_fit_score` at weight **1.5**, against
  `coverage_score` **2.0**, `headroom_score` **0.5**, `network_score` **0.3**
  (`allocator.py:131–136`).
- `SpecialistCard` carries `runtime_gb`, `min_vram_gb` (`models.py:114`), `context_window`
  (`models.py:115`), `n_layers` (`models.py:116`), `hidden_size` (`models.py:117`),
  `estimated_tps_at` (`models.py:119`). **`hidden_size` is `int | None`** — the only
  nullable one of the five; don't assume it's populated.
- `PrefVector.min_throughput_tps` (`select.py:224`) — the SLO floor in semantic form,
  **disabled** via `_reject_unsupported_pref` (`select.py:285–306`).
- `SECONDARY_PRIMARY_SHARE_CEILING = 0.5` (`allocator.py:37`) and the secondaries
  machinery (`_fill_secondaries`, `allocator.py:426–440`; `NodeSuggestion.secondaries`,
  `models.py:178`) — the co-residence call site already exists.
- Idle-time LoRA training state machine (`idle.py:50–56`,
  `ACTIVE→READY_TO_TRAIN→TRAINING→COOLDOWN`) — GB10's real value lever.

**The gap (what this brief adds):**

| # | Gap | Current behaviour | Consequence |
|---|-----|-------------------|-------------|
| G1 | **No latency SLO hard filter.** | bandwidth is a weight-1.5 soft score; `coverage_score` (2.0) outweighs it. | A node that decodes at 3 tok/s can *win* an interactive specialist when coverage is unmet. The floor field exists (`min_throughput_tps`) but raises `ValueError`. |
| G2 | **Bandwidth is a datasheet constant, not measured.** | GB10 served `273.0` from the hardcoded table (`probe.py:52`); chips absent from the table fall to `estimated_tps_at`. | **273 is the unified *peak*** shared CPU+GPU; achieved batch-1 BW is materially lower (MBU ~0.6–0.85 + CPU contention), so `predicted_tps` is **over**-estimated. The gate would then *pass* GB10 for models that actually decode below floor — worse than useless until §4 lands. The danger is **datasheet-overestimate, not table-staleness.** |
| G3 | **tps model ignores KV-cache traffic.** | `bytes/token == runtime_gb` (weights only). | ~2× optimistic at long context: a GQA 27B at 128K reads KV that rivals the 27GB of weights → real roofline ~4–6 tok/s, not ~10. The card has `n_layers`/`hidden_size` to start the fix. |
| G4 | **Co-residence gate is VRAM-share only.** | secondaries attach under `SECONDARY_PRIMARY_SHARE_CEILING = 0.5` (a *memory* share; `allocator.py:427`), no bandwidth term. | Co-resident models **split the node's bandwidth**; a secondary silently halves the primary's decode tok/s — acute on unified-memory nodes (GB10/Mac), and the practical co-residence target is the **Ollama** multi-model daemon (§4 note). |
| G5 | **Quant not surfaced as the bandwidth lever.** | quant is implicit in `runtime_gb`. | When no node meets the SLO, the allocator gives up instead of down-quanting (fewer bytes/token → faster decode) the same model. |

---

## 3. The sizing model

### 3.1 Per-token bytes (replaces weights-only)

```
B(spec, ctx) = weight_bytes(spec)                               # see 3.5 — active params for MoE, not total
             + kv_bytes_per_token(spec) · ctx_tokens            # KV reads, grow with context

kv_bytes_per_token = 2 (K+V) · n_layers · n_kv_heads · head_dim · kv_dtype_bytes      # MHA/GQA/MQA
```

- `head_dim = hidden_size / n_heads`; with **GQA**, `n_kv_heads << n_heads` (Llama-3-70B:
  8 KV vs 64 query = 8×) — this dominates the KV term and **must** be modelled. Pure MHA
  is the `n_kv_heads = n_heads` case (no reduction); MQA is the `n_kv_heads = 1` limit.
  [P, GQA paper Ainslie 2023 + Llama-3/Mistral configs, 90]
- **The formula is not universal** — make `kv_bytes_per_token` a **per-card-overridable
  function**, not a hardcoded expression (R3), because:
  - **MLA** (DeepSeek V2/V3) caches one compressed latent per token (~per-layer latent
    dim, e.g. 512), an order of magnitude smaller — the GQA formula is simply wrong.
  - **Sliding-window** (Mistral 7B, Gemma-2/3 interleaved) caps cached KV at window `W`,
    so bytes/token plateaus at `min(ctx, W)` rather than growing with full context.
  - **Per-layer KV quant** (FP8/INT8 KV) means `kv_dtype_bytes` is not always 2; some
    stacks store K and V in different dtypes; cross-layer KV sharing divides by the share
    factor.
- **Missing fields:** `SpecialistCard` has `n_layers`/`hidden_size` but not
  `n_kv_heads`/`head_dim`/`kv_dtype`/`kv_arch`. Add them (sourced from model config;
  `kv_arch ∈ {mha, gqa, mqa, mla, sliding_window}` selects the override). `model_config`
  is `extra='forbid'` + frozen (`models.py:35`), so **optional fields with defaults are
  additive-safe.** Until the fields are present, fall back to `B = weight_bytes` (today's
  behaviour) so the change is non-regressive.

### 3.2 Predicted decode tok/s

```
predicted_tps(spec, node, ctx) = MBU · node.effective_bw_gbs / B(spec, ctx)
```

- `effective_bw_gbs` is **measured** (§4), not the datasheet figure and not the static
  table.
- `MBU` (memory-bandwidth utilization, ~0.6–0.85) is folded in **once** so the model
  stops quoting unachievable peak numbers. If `effective_bw_gbs` is itself an *achieved*
  decode measurement (§4), `MBU` is already baked in and set to 1.0 — do not double-apply.
- **Below some model size the bound stops being bandwidth** and hits kernel-launch /
  latency floors; don't extrapolate the roofline to tiny models.

### 3.3 The SLO gate (the core change, G1) — fail **soft**, not closed

Request classes already exist (`route_class`, `select.py:50`). Map a decode floor onto the
**existing** `PrefVector.min_throughput_tps` (do **not** add a parallel field):

| Class | floor |
|-------|-------|
| interactive (`hot_interactive`: chat/agent/voice) | e.g. **30** tok/s |
| standard / batch / async | **0** (no floor) |

The gate lives in the pref filter (`_pref_filter`, `select.py:309–347`), and shipping it
requires **removing the `_reject_unsupported_pref` rejection** (`select.py:285–306`) — not
adding a new branch. Behaviour:

```python
# pseudo — runs only after Step 0 wires the pref selector into the live router
predicted = predicted_tps(spec, node, ctx)         # ctx: representative until per-request ctx exists (§0/§4)
if floor and predicted < floor:
    if has_alternative_that_passes(domain) or has_downquant_variant(spec):   # §3.5
        score = -math.inf            # hard-refuse THIS node; a better option exists
    else:
        place(node, degraded=True)   # DEGRADED-COVERAGE DEFAULT — never drop the domain
        emit("slo.degraded_placement", domain, node, predicted, floor)
```

**Why not a bare `-math.inf`:** the only "interactive decode tier" node is the RTX PRO
6000 (§5). A bare fail-closed gate means losing that one box (reboot, driver crash, OOM)
drops **every** interactive domain at 3am. R1's degraded fallback must be **wired into
`model_fit_score`/`allocate_cluster` as code**, with a test that kills the only high-BW
node and asserts interactive domains stay *covered (degraded)*, not dropped. Reserve the
hard `-inf` for when a passing or down-quanted alternative actually exists. Capacity
(`min_vram_gb`) stays as-is; bandwidth becomes the **second** gate. The soft
`throughput_score` stays as a tie-breaker among nodes that pass the floor.

### 3.4 Bandwidth-aware co-residence (G4) — with hysteresis

Add a **bandwidth budget** parallel to the existing memory-share check in
`_fill_secondaries` (`allocator.py:426–440`):

```
sum(active_model_bandwidth_demand) <= steady_state_bw_gbs · BW_UTIL_CEILING
model_demand ≈ B(spec, ctx) · target_tps
```

A secondary that would push the primary below its decode floor is **refused even if VRAM
fits**. **Critical stability requirement** — the budget must be fed by a *slow-moving*
bandwidth estimate, never a live single-sample reading, or it flaps cluster-wide:
attaching a secondary lowers measured BW → node fails its own floor → evict → node idles →
re-measure reads high → re-admit → repeat. Therefore:

1. Act on an **EWMA / p50 over N samples**, not the last measurement.
2. **Separate admit vs evict thresholds** (hysteresis band): clearly below floor to evict,
   clearly above to re-admit.
3. **Cooldown** after any eviction during which placement is frozen.
4. Compute the budget against a **steady-state** BW estimate, not a live-under-load read.

Practical note: this governs **node-level** co-residence in the allocator, but the real
multi-model co-residence target is the **Ollama** daemon (single process, models loaded on
demand — the only backend with live load/unload; vLLM/llama.cpp/MLX are one-process-per-
model, cold load 2–4 min on Spark, `backends.py:215–216`). So the budget mostly governs
Ollama nodes.

### 3.5 Quant ladder (G5)

When no node passes the floor at the card's default quant, substitute a smaller-
`runtime_gb` variant of the **same** model (a `quant_ladder` on the card) before declaring
the domain uncovered — an explicit, logged bandwidth-for-quality trade. **FP8→NVFP4 roughly
halves weight bytes → ~2× the decode roofline** on the same box (caveat: NVFP4+marlin's
runtime FP4→BF16 decompress claws back part of the win — needs one measured data point).
Catalog cards (`mesh/catalog/*.toml`) carry `estimated_tps_at` but **no `quant_ladder`
today**; without it R1's down-quant mitigation is dead code. Add a **`validate_card.py`
check that flags interactive-eligible cards lacking a populated `quant_ladder`**, so the
fallback is real where it's relied on.

### 3.6 Regime boundaries (the model's validity domain — state on every quoted number)

- **MBU:** apply ~0.8 once (§3.2); peak figures are unachievable.
- **Batch>1:** weights are read once and amortized across the batch → arithmetic intensity
  rises, decode crosses over to **compute-bound** past a model-dependent batch size; KV
  stays per-sequence so KV bytes still scale with `batch · ctx`. The "smaller model is the
  only lever" and "Mac does nothing" conclusions are **batch-1-only**.
- **MoE:** `weight_bytes` = **active** expert params + shared/attention params + router,
  **not total** params (Mixtral 8×7B ~13B active of 47B; DeepSeek-V3 ~37B of 671B). Using
  total badly under-estimates tok/s and **mis-ranks A3B/A4B MoEs as slow when they are
  among the fastest decoders.** Capacity/fit uses **total** quantized bytes; decode
  roofline uses **active** bytes. Carry a worked MoE example.
- **Long context:** the KV term grows linearly and eventually dominates weights → tok/s
  decays with context. Publish **two rooflines** per model: weight-dominated (empty/short
  ctx) and KV-dominated (full ctx).
- **Prefill ≠ decode:** TTFT (time-to-first-token) is prefill, which is **compute/FLOP-
  bound**, plus cold-start. The bandwidth roofline predicts **inter-token latency**, not
  TTFT. The "route voice to the small model" lever improves ITL; say so explicitly so
  nobody expects it to predict TTFT.
- **Speculative decode (future lever, not in this brief):** batch-1 bandwidth-bound is
  exactly the regime where spec-decode pays *most* (a draft amortizes one weight-read over
  K accepted tokens), so flag depth-≥3 draft trees / EAGLE-3 as a **still-open** decode
  lever — do not treat any negative single-token-MTP result as closing it.

---

## 4. Probe change — measure, don't guess (G2)

`mesh/probe.py` is static today: every `_detect_*` is an `nvidia-smi`/`sysctl`/`psutil`
query, and `memory_bandwidth_gbs` comes from the hardcoded `_MEMORY_BANDWIDTH_GBS` table
(`probe.py:51–58`), not a measurement. (`quality_probe.py` measures response *quality* via
LLM-judge, not tps — not a timing hook.) The change attaches an **effective-bandwidth
micro-benchmark inside `probe_node()`** (`probe.py:338`) that **replaces** the hardcoded
constant (`probe.py:52`) with an achieved figure:

```
effective_bw_gbs = bytes_moved / elapsed_s
```

This kills dependence on the static table over time, captures *real* achieved bandwidth
(kernel efficiency, unified-memory CPU contention) rather than the datasheet peak, and
self-updates for new silicon with zero catalog edits.

**The bench must be specified, or it's self-defeating on exactly its target nodes:**

- `probe_node()` runs at bring-up, **before allocation — there is no resident model to
  benchmark.** And the co-residence nodes this brief targets (GB10/Mac, many small models)
  are **never idle**, so "measure at idle on the resident model" is unachievable where the
  measurement matters most.
- **Use a synthetic memory-bandwidth kernel** (a fixed memcpy/GEMV-style streaming kernel,
  no model loaded) so the bench is model-agnostic, works pre-placement, and yields a clean
  node attribute. (A real-model decode bench is acceptable only as a later refinement, run
  async to the heartbeat in a guarded low-utilization window.)
- **Time-box by wall-clock**, not token count: a fixed K tokens on a slow node (~3–5 tok/s
  for a big model) is 30–40s and can overrun the 60s heartbeat (`probe.py:6`) → node
  flagged dead → re-allocation storm. Measure for ≤N seconds, derive the rate.
- **Decouple from the heartbeat loop.** Run async; never re-measure during active eviction
  or a traffic spike (a mid-spike read demotes a healthy node — an outage caused by the SLO
  machinery itself).
- **Cold join → "guessed" mode** (R4 tag). The `min_decode_tps` gate **must not emit
  refusals from a datasheet/table number** — only from a landed measurement. Until one
  lands, the node is tagged `bw_source=guessed` and routing discounts it.

**`[VERIFY] before any code:** run `probe_node()` on a live GB10 and record the actual
`memory_bandwidth_gbs`. Confirm whether `_detect_chip` (`probe.py:83`,
`nvidia-smi --query-gpu=name`) string-matches `'NVIDIA GB10'` (→ served 273 datasheet) or
not (→ falls to `estimated_tps_at`). The two cases have opposite gate failure modes; the
brief assumes the former (273 from `probe.py:52`) but it is unverified on the live box.

---

## 5. Node-tier sizing guidance (the placement payoff)

Bandwidth tiers the fleet for *where the interactive hot path goes*. Figures are
**datasheet-derived estimates** — the probe (§4) replaces them with measured values, and
the gate must run on measured, not these:

| Node | Mem BW (datasheet) | Source | Sizing role |
|------|--------------------|--------|-------------|
| RTX PRO 6000 Blackwell (WS/Max-Q/Server) | **1.792 TB/s** (96GB GDDR7, 512-bit, 28 Gbps/pin) | [P, nvidia.com + Blackwell PRO whitepaper v1.0, 95] | **interactive decode tier** — hot path, big models at usable tok/s, co-resident hot set. Max-Q/Server share the same memory subsystem (compute clocks throttle, not memory). **Do not conflate with RTX 6000 Ada (48GB GDDR6, ~960 GB/s).** |
| Mac M-Ultra (M3 819 / M2 800 GB/s) | ~0.8 TB/s | [P, apple.com Mac Studio + M3/M4 specs, 90] | interactive (mid) |
| Mac full M4 Max (40-core) | ~0.55 TB/s | same | small/mid-interactive |
| Mac M4 Pro / M3 Max / binned M4 Max | ~0.27–0.41 TB/s | same | small-interactive / overflow |
| **GB10 / DGX Spark** | **273 GB/s** (128GB LPDDR5X-8533, 256-bit, **unified CPU+GPU**) | [P, docs.nvidia.com/dgx/dgx-spark + Tom's Hardware GB10 review, 93] | **batch · coverage · idle-train tier — keep big interactive models OFF it** |

Notes: the GB10 273 GB/s is **shared with CPU traffic on the unified bus**, so effective
inference BW under contention is **below** 273 — reinforcing G2. The base M4 mini is only
~120 GB/s (not 273 — that's the M4 *Pro* bin). M5-generation silicon exists as of 2026 if
current-gen coverage is wanted later.

Rule the allocator should encode: **route interactive to high-BW nodes; park batch /
coverage-only / idle-train specialists on low-BW nodes.** GB10's value is its 128GB unified
capacity + idle-time training (`idle.py`), not its decode latency. On Mac the correct claim
is "**no bandwidth headroom over GB10**" (not "buying one does nothing") — the model-size
and quant levers still apply, and MLX often hits a higher % of peak than vLLM-on-sm_121.

### Hardware verification (2026-06-27, measured — not datasheet)

Ran on the two live nodes. Confirms the model end-to-end; numbers below are **measured**
and should seed the probe table / gate, replacing the datasheet estimates above.

| Node | `nvidia-smi` name | Datasheet | **Measured BW** | **MBU** | In-situ decode |
|------|-------------------|-----------|-----------------|---------|----------------|
| GB10 (`promaxgb10-d325`) | `NVIDIA GB10` (matches probe table key → 273 served) | 273 GB/s | synthetic bench **blocked** — box saturated (117/121 GB used, 88 GB vLLM resident, ~3 GB free; a 2 GiB alloc OOM'd) | — | `dot-voice` (small) **46 tok/s**, `dot-backbone` (~88 GB) **8 tok/s** |
| RTX PRO 6000 (`dellpromax`) | `NVIDIA RTX PRO 6000 Blackwell Workstation Edition` | 1792 GB/s | **1467 GB/s** (zero-install ctypes `cuMemcpyDtoD_v2`, 2 GiB, best/50) | **0.82** | idle (no resident model) |

What it proves:

- **MBU ≈ 0.8 is real** — measured 0.82 on the RTX PRO 6000; datasheet peak is unachievable,
  fold MBU once (§3.2) holds. **~6.5× achieved BW gap** GB10→RTX PRO 6000 (1467 vs ~224 =
  273·0.82), matching this brief's "~6.6×."
- **The 30 tok/s interactive floor discriminates correctly, measured:** on the *same* GB10,
  the big backbone decodes **8 tok/s** (reject) and the small voice model **46 tok/s** (pass)
  — single-digit decode on a big model on GB10 is the predicted failure mode, now observed.
- **GB10 is never idle** (3 GB free) — §4's reason for a synthetic kernel + maintenance-window
  measurement is empirically forced, not theoretical.

Three concrete code bugs surfaced (file the fixes against the steps noted):

1. **Probe name-match brittleness (Step 2).** `nvidia-smi` returns the *full* name
   `"NVIDIA RTX PRO 6000 Blackwell Workstation Edition"`, which misses **both**
   `_MEMORY_BANDWIDTH_GBS` (absent) **and** `_FP4_TOPS_BY_CHIP` (key is the shorter
   `"NVIDIA RTX PRO 6000"`; probe does an exact `.get()`, no substring) → the high-BW
   interactive node today gets `bw=None` + `fp4_tops=None` and falls to `estimated_tps_at`.
   Fix needs the full name as the table key (or substring matching) — exactly what the §4
   measured bench obviates by self-updating regardless of name.
2. **Stale docstrings.** `mesh/models.py:46-49,72,98-100` claim GB10 records
   `memory_bandwidth_gbs=None`; the probe actually serves `273.0` from the table. And
   `mesh/probe.py:52` comment says "LPDDR5X-9600" — 273 GB/s ÷ 256-bit = 8533, so the §5
   figure (8533) is right and the comment is wrong.
3. **MoE card data bug (Step 3).** `mesh/catalog/qwen3-coder-30b-a3b-fp8.toml` has
   `runtime_gb=70 > min_vram_gb=30`; `_estimated_tps` would compute `bw/70` and mis-rank a
   ~3B-active MoE as the slowest decoder — the §3.6 MoE gap, live in the catalog. Needs the
   active-param weight term (and a sanity look at the 70 value).

---

## 6. Success criteria (falsifiable) — including runtime observability

Offline accuracy:

- **SLO compliance:** 0 interactive placements below the decode floor after the gate ships
  (baseline: count today's soft-score violations, N>0).
- **Probe accuracy:** measured `effective_bw_gbs` yields `predicted_tps` within **±15%** of
  observed batch-1 decode tok/s across ≥3 node classes, **under a stated, matched load
  condition** — both predicted and observed measured at idle, *or* both under a defined
  co-resident load. (The ±15% is unfalsifiable if predicted is idle-measured and observed
  is production-co-resident — they diverge by far more than 15% by construction, G4.)
- **KV realism:** at 8k **and** 128k context, `predicted_tps` error stays within ±15%
  (weights-only model fails this — the regression the KV term fixes).
- **Co-residence safety:** attaching a secondary degrades the primary's decode tok/s by
  **≤10%** vs. the unbounded VRAM-only gate.
- **No capacity regression:** every placement that passed `min_vram_gb` before still
  passes; the bandwidth gate only *removes* placements (or marks them degraded), never adds
  an OOM.

Runtime telemetry (a gate whose job is silently refusing placements **must** be observable
— ship these in Step 2, before the gate flips):

- Counter of placements **refused by the bandwidth gate**, by domain/node.
- Per-node **`bw_source ∈ {measured, guessed}`** flag, **surfaced into routing
  (`select.py`)** so guessed nodes are discounted — R4 currently only *tags* it in the
  heartbeat; confirm `select.py`/`router_app.py` actually *consume* the tag or the discount
  is built-but-not-wired.
- **Eviction events** from the co-residence budget.
- **Page-able alerts:** "domain transitioned to uncovered/degraded"; "node `effective_bw`
  dropped >X% between measurements."

---

## 7. Load-bearing assumptions (verify before locking)

1. **A1 — decode is batch-1 bandwidth-bound for our traffic.** Holds for interactive/agent/
   voice (low concurrency). At high batch the bound shifts to compute (§3.6) — evaluate the
   floor **per request class**, not globally. *Confidence: high.*
2. **A2 — `runtime_gb` already reflects served quant.** **Hard build-blocker until
   verified against one real Ollama Q4 card.** If `runtime_gb` is the FP16 figure
   regardless of served quant, then `B(spec,ctx)`, `predicted_tps`, the gate threshold,
   **and** the quant ladder are all wrong-valued simultaneously. *Verify first, build
   second.*
3. **A3 — model config exposes `n_kv_heads`/GQA (and `kv_arch` for MLA/sliding-window).**
   Required for §3.1. If absent and unobtainable, KV term degrades to weights-only
   (non-regressive, loses the long-context win).
4. **A4 — a synthetic micro-bench reflects steady-state achieved bandwidth.** Mitigated by
   the synthetic kernel + EWMA + idle-window measurement (§4); a single live read under
   contention does not (§3.4 thrash).
5. **A5 — the pref selector can be wired into the live router.** Gated on slancha-api's
   classifier being importable (`router_app.py:26–29`). If it can't be, the entire SLO gate
   (§3.3) stays shelf-ware regardless of the allocator work. *Verify the wiring path before
   committing to Step 0.* **[VERIFY]**

---

## 8. Risks & mitigations

- **R1 — SLO gate leaves a domain uncovered.** Mitigation is now **in the gate's code
  path** (§3.3), not prose: degraded-coverage placement is the **default** when no
  alternative passes, with `degraded=true` + structured log + page-able metric; hard `-inf`
  only when a passing/down-quant alternative exists. Wire into `model_fit_score`/
  `allocate_cluster` with a kill-the-high-BW-node test.
- **R2 — micro-bench picks a bad number / adds bring-up latency.** Synthetic kernel,
  wall-clock-boxed, async to heartbeat, EWMA, no re-measure during eviction/spike (§4).
- **R3 — KV model wrong for MLA / sliding-window / non-GQA.** `kv_bytes_per_token` is a
  per-card-overridable function keyed on `kv_arch`, not a hardcoded formula (§3.1).
- **R4 — static-table/datasheet drift during transition.** Mixed measured/guessed values
  coexist; `bw_source` tag is **consumed by routing** (not just emitted) so guessed nodes
  are discounted and never emit gate refusals (§4, §6).
- **R5 — thrash from reactive bandwidth gating.** EWMA + hysteresis band + eviction
  cooldown + steady-state estimate (§3.4).
- **R6 — gate is dead code on the live path.** The running router bypasses the pref
  selector (§0). Step 0 wires it in; nothing downstream is "done" until a live in-situ
  round-trip exercises it — the 650+ unit suite covers the *library* function, not the
  `/v1` path.

---

## 9. Rollout (additive; one PR per step; matches "additive schema, no drops")

> Reordered after the plumbing review. **Step 0 is the true prerequisite** — without it,
> steps 1/3/5 change only code that production traffic never reaches.

0. **Wire the pref-aware selector into the live path.** Make `router_app.py` call
   `select_mesh_route_with_pref` (or have slancha-api's selector do so), gated on the
   classifier being importable (`router_app.py:26–29`, A5). Until this lands, the SLO gate
   is shelf-ware. **This is the no-shelf-ware gate for the whole effort.**
   *Also in this step or before Step 3:* **surface prompt length (`ctx_tokens`) in
   `router_app.py`** (today it extracts only `body['model']`, `router_app.py:591`) — the
   KV-aware model has **no live ctx at either selection or allocation time** (Q4 refuted).
   Without it, §3.1's KV term can only run on a representative/worst-case ctx in the
   offline allocator, with no per-request accuracy.

1. **Schema (backward-compatible).** Add `n_kv_heads`/`head_dim`/`kv_dtype`/`kv_arch` +
   optional `quant_ladder` to `SpecialistCard` (additive: `extra='forbid'` frozen,
   `models.py:35`). **Do not add a `min_decode_tps` field** — **reuse
   `PrefVector.min_throughput_tps`** (`select.py:224`) and **remove the
   `_reject_unsupported_pref` rejection** (`select.py:285–306`) instead; otherwise two
   competing throughput floors exist, one of which raises `ValueError`. `None` everywhere =
   today's behaviour.

2. **Probe + observability.** Ship the synthetic effective-bandwidth micro-bench in
   `probe_node()` (`probe.py:338`) → **replace** the hardcoded `273.0` (`probe.py:52`) with
   the measured value; tag `bw_source`. **Ship the §6 runtime telemetry here**, before any
   gate flips. (`_estimated_tps`'s primary path already consumes `memory_bandwidth_gbs`;
   this is a pure improvement, no gate yet.)

3. **Bytes model.** Extend `_estimated_tps(spec, node)` (`allocator.py:59`) →
   `predicted_tps(spec, node, ctx)` with the per-card KV override and active-param (MoE)
   weight term; weights-only fallback when fields absent. Sole live call site today is
   `model_fit_score` (`allocator.py:122`, **offline** placement) — so the KV term runs on a
   representative ctx unless Step 0 surfaced `ctx_tokens`. Sequence **after** Step 0.

4. **Co-residence budget.** Add the bandwidth-share check beside the memory-share check in
   `_fill_secondaries` (`allocator.py:426–440`) with EWMA + hysteresis (§3.4). Localized
   additive change; the call site and memory gate already exist.

5. **Flip the gate (fail-soft).** Un-disable the floor for `hot_interactive` in
   `_pref_filter` (`select.py:309–347`), with the **degraded-coverage default wired**
   (§3.3, R1) — not a bare `-inf`. Only behaviour-changing step; ship last, behind measured
   bandwidth + KV model + Step 0 wiring, with the §6 baseline + telemetry captured first.
   **"Done" = exercised on a live `/v1` round-trip, not unit-green.**

Each step is independently revertable (`git revert <sha>`) and independently testable
against the existing 650+ unit suite — **plus** Steps 0 and 5 require a live in-situ
round-trip through `router_app.py`, since the unit suite covers the library selector, not
the running router.