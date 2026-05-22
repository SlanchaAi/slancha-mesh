# SLANCHA MESH PROTOCOL v0.1 — Consolidated Plan (DRAFT)

**Status**: Draft for paul-mac sign-off. **Revision 3** (2026-05-22) — incorporates Rev 2 verification + 7 new concerns + H22 partial fix. **SAFE TO TAG conditional on Rev 3 lock** per mac PR #1.
**Author**: claude (slancha-spark session, 2026-05-22)
**Inputs**: 5 parallel research streams (LLM gateway architectures, AWS routing patterns, personal AI cloud-bridge patterns, router benchmarks, federation protocol design) + mac's triple-check pass.

This document supersedes `SLANCHA_MESH_V0_SPEC.md` once mac signs off. The prior spec framed mesh as a router with its own classifier; this draft separates concerns — mesh = placement substrate + signal-rich protocol, cloud classifier = decision maker, edge = transparent dispatcher.

**Revision 3 changelog** (mac PR #1 verification + new concerns):
- B6 carryover: slug drift fixed — all bare `paul-v8-essay` replaced with canonical `paul-v8-essay-dpo-iter1-beta-0.1`; `voice-paul-v8` residue removed from §11
- H22 fix: #65d rotation tooling task added (HMAC KID seed/swap + CF Access rotator + mesh-ingest-token rotator + dry-run CLI) — "build tooling first" now backed by actual task
- NC1: §11 surfaces central-probe disclosure inline (was buried in §9 Q1)
- NC2: KVS sizing recompute — Rev 2 record shape pushed past 200B/record; shard `pref_max` to separate KVS key `<user>:pref` (rotates slow) vs `<user>:route` (rotates fast). ~25K users still achievable post-shard
- NC3+M18: §4.1 auth direction matrix — three tokens × (direction, rotation, grace, revocation, replay defense)
- NC4: bootstrap × 0.5 marked tunable per-router config (documented range 0.3–0.7); calibrate empirically
- NC5: `decision_reason_structured` added to conformance corpus (#63a) — schema + golden trace
- NC6: Phase 0 owner = **slancha-spark** explicitly (v8 artifacts live in `~/finetuning/output/`)
- Q5 flip: endpoints[] — Anthropic REQUIRED-IF-ADVERTISED (not OpenAI-only with slot); spec MAY treat OpenAI as reference shape, impl ships both day 1
- M9: JCS pins (NFC normalization mandate, ban numbers >2^53, lib version pin)
- M11: model_hash semantics in §3 schema decisions (already partially landed in Rev 2; explicit block added)
- M14: card-level `node_capabilities[]` for protocol extensions (already landed in Rev 2; documented MCP precedent)
- Phase 4+5 interleave clarifications: conformance suite runs against each lever as lands; spec-freeze marker = first lever produces structured decision_reason on real traffic

**Revision 2 changelog** (mac triple-check responses):
- B1: HMAC drops `body_hash`; mesh-side computes body MAC after passthrough → L@E never reads body → unbounded request size (Claude 200k contexts work)
- B2: Secret store = KMS Decrypt at L@E cold start, in-process cached for replica lifetime
- B3: Health gate moved upstream to CF Function (viewer-request); L@E reads injected header
- B4: CF Origin Groups primary=mesh-tunnel, secondary=lambda-default (DNS-fail failover)
- B5: usage_logs partial unique index on request_id + `ON CONFLICT DO NOTHING`
- B6: voice-paul-v8 slug was placeholder; canonical artifact = paul-v8-essay-dpo-iter1-beta-0.1 family
- H17: License bifurcate — spec=Apache 2.0, ref impl=AGPL-3.0, conformance=Apache 2.0
- H19: Card schema gains `endpoints[]` array allowing OpenAI + Anthropic shapes
- H6/H7/H8: RFC 8941 header format; semantic service_tier enum; quantizations[] allowlist
- Phase 0 added; Phase 4+5 interleaved; Phase 8 split off for AGPL+governance

---

## 1. The architecture

```
Client (Cursor / Aider / OpenAI SDK / curl / agent)
   base_url=https://api.slancha.ai/v1
   Authorization: Bearer sk-slancha-<user>
   X-Slancha-Pref: max-cost-cents=5, max-latency-ms-p95=2000, quality-weight=0.6   (RFC 8941 dict)
   X-Slancha-Capability: streaming, tools                                          (hard gate)
        │
        ▼
   CloudFront distribution (already exists per cf_origin_secret_middleware)
   CF Origin Groups:
       primary   = user's mesh tunnel (cf-tunnel: mesh.<user>.laulpogan.com)
       secondary = Lambda Function URL (existing slancha-api router)
   ── Origin Groups handle DNS-fail / 5xx failover natively. L@E selects WHICH
   ── group; CF handles failover within the group.
        │
        ├── viewer-request CloudFront Function (<1ms, no I/O outbound):
        │     KVS sharded into 2 stores (NC2):
        │       slancha-route   : bearer_sha256 → {user_id, route_target,
        │                                          mesh_origin_id, mesh_healthy}
        │                          ~90B/record → ~50K users/KVS
        │                          (rotates fast — mesh_healthy flips on prober tick)
        │       slancha-ceiling : user_id      → {pref_max, ceiling_pref}
        │                          ~260B/record → ~19K users/KVS
        │                          (rotates slow — admin-set)
        │     - bearer SHA-256 → KVS read 1 (slancha-route)
        │     - user_id → KVS read 2 (slancha-ceiling)
        │     - inject X-Slancha-User-Id, X-Slancha-Route-Target,
        │              X-Slancha-Mesh-Healthy (1|0), X-Slancha-Origin-Id headers
        │     - enforce pref ceiling: parse X-Slancha-Pref, apply min(client, ceiling)
        │     - drop any client-set X-Slancha-Forward-Sig (prevent spoof; only L@E sets it)
        │
        ├── origin-request Lambda@Edge (~20-40ms warm, ~250ms cold + KMS):
        │     Cold-start: KMS Decrypt baked-in ciphertext (HMAC_KEY + CF_ACCESS_TOKEN);
        │                 cache in module scope for replica lifetime.
        │
        │     IF X-Slancha-Route-Target == "mesh" AND X-Slancha-Mesh-Healthy == "1":
        │       . validate X-Slancha-Origin-Id ∈ registered allowlist (NOT raw URL)
        │       . strip client Authorization (don't leak SaaS bearer downstream)
        │       . inject X-Slancha-Forward-Sig (HMAC over
        │                user_id + timestamp + nonce + route_target + origin_id;
        │                NO body_hash → L@E never reads body → unbounded request size)
        │       . inject CF-Access-Client-Id/Secret (KMS-decrypted at cold start)
        │       . select Origin Group: primary (mesh tunnel)
        │       . async PUT to Firehose: routing decision (request_id, user_id,
        │                                                  decision, origin_id)
        │       . return mutated request — Lambda@Edge NOT in response path
        │     ELSE:
        │       . select Origin Group: secondary (Lambda default)
        │       . OPTIONAL emit X-Mesh-Fallback: <reason> for telemetry
        │
        ▼  CloudFront connects to chosen origin directly
        ▼  CF Origin Group fails over primary→secondary on DNS resolution failure
        ▼  or 5xx; failover emits X-Mesh-Fallback in response headers for visibility.
        ▼  SSE chunks stream through unmodified (no L@E origin-response trigger ever).
        │
   ┌────┴────────────────────────────────┐
   ▼                                      ▼
Lambda Function URL                    Cloudflare Tunnel → user's mesh
(existing slancha-api router:           (slancha-local: OpenAI-compat surface)
classifier + provider registry +         │
log_usage telemetry)                     ├── Verify CF Access service token (CF policy)
   │                                     ├── Verify HMAC X-Slancha-Forward-Sig
   │                                     │   (±300s timestamp tolerance, nonce LRU
   │                                     │    dedup over 600s window)
   │                                     ├── Read X-Slancha-Pref + X-Slancha-Capability
   │                                     ├── Compute body MAC server-side; persist
   │                                     │   (audit trail; not in HMAC scope)
   │                                     ├── Local registry: pareto-rank specialists
   │                                     │     (filter on capability; score on prefs)
   │                                     ├── Dispatch to vLLM/mlx-lm specialist
   │                                     ├── Stream SSE response back via tunnel
   │                                     ├── Write durable local jsonl
   │                                     │     (~/.slancha-local/usage-buffer.jsonl)
   │                                     │     BEFORE async telemetry POST
   │                                     └── After response close:
   │                                           POST sidecar telemetry to
   │                                           api.slancha.ai/v1/admin/usage
   │                                           (counts only, no prompt body)
   │                                           Retry: 1s/5s/30s exponential.
   │                                           DLQ → usage-dlq.jsonl after N=3.
   │                                           Nightly reconcile job.
   ▼                                      ▼
SSE response to client                SSE response to client
```

### Why this shape (cross-referenced to research + mac triple-check)

| Concern | Resolution | Source |
|---------|-----------|--------|
| Existing clients unchanged | Same URL, same bearer, same response shape | hard requirement |
| Bypass Lambda on mesh path | L@E origin-request selects Origin Group; CloudFront connects to tunnel directly; main Lambda never invoked on mesh requests | AWS L@E origin-request docs + CF chunked-encoding pass-through |
| SSE pass-through | CF natively streams chunked encoding; L@E in NEITHER request body read NOR response path; origin-response trigger forbidden, CI-enforced | CF `RequestAndResponseBehaviorCustomOrigin` |
| Long-context support (B1 fix) | L@E never reads body → no 1MB cap → Claude 200k contexts work | L@E body-inspection cost per restrictions doc |
| Per-user routing | CloudFront KVS (≤5MB read-only edge KV; ~25K users/KVS at 200B/record); millisecond reads from CF Function only | CloudFront KVS spec |
| L@E secret material (B2 fix) | KMS Decrypt at cold start of ciphertext baked in bundle; cached in module scope for replica lifetime; +50-100ms p99 cold tax | L@E forbids env vars + can't read KVS |
| Health gate (B3 fix) | CF Function (viewer-request) reads `mesh_healthy` from KVS; injects header; L@E reads injected header only. Regional liveness-prober Lambda writes KVS | L@E can't read KVS |
| DNS-failure failover (B4 fix) | CF Origin Groups primary=tunnel, secondary=lambda-default; CF handles DNS-fail / 5xx natively. L@E selects GROUP, not raw `domainName` | AWS L@E restrictions: DNS-resolve precedes L@E |
| Telemetry shape preserved | Mesh POSTs sidecar in `log_usage` shape; idempotent via partial unique index on `request_id` (B5 fix) | OpenRouter Private Models pattern, Helicone async mode precedent |
| Sidecar mid-stream resilience (H16 fix) | Mesh writes durable local jsonl BEFORE async POST; retry 1s/5s/30s; DLQ + nightly reconcile | Stripe webhook pattern in slancha-api |
| Privacy: prompts don't traverse SaaS Lambda on mesh path | L@E sees only headers; main Lambda not invoked at all | LM Link "device list only" posture as design north star |
| SSRF defense (H2 hardened) | `mesh_origin_id` validated against registered allowlist; CF tunnel name allowlist pinned to CF tunnel IP ranges; reject DNS rebinds to internal/IMDS targets | Portkey RFC1918 block; DNS-rebind attack surface |
| Tunnel-direct-attack defense | CF Access service-token policy + HMAC `X-Slancha-Forward-Sig` (over user_id + timestamp + nonce + route_target + origin_id) | CF Access mandate; "tunnel-as-auth" is anti-pattern; Ollama 175k exposed hosts |
| Pref ceiling enforcement (H15 fix) | CF Function reads `pref_max` from KVS per-user; applies `min(client_pref, pref_max)`; emit `slancha.pref.ceiling_exceeded` event on excess | OWASP injection class; mirrors slancha-api a1f582e enabled_models fix |
| Bearer leak prevention | SaaS bearer stripped before forwarding; mesh-side auth is service-to-service | Portkey `forwardHeaders` precedent, two-header pattern from CF AI Gateway |
| Fallback when mesh dead | CF Origin Group failover (primary→secondary); L@E sees mesh_healthy=0 in header and routes to default group; X-Mesh-Fallback emitted | Vercel silent-fallback critique → must surface fallback events |
| Cost @ 10K rpd (H3 honest) | $1-3/mo at 10K rpd assuming ≤50KB avg completion; egress dominates beyond. KMS cold-start +100ms p99 | AWS pricing + CF egress |
| Operator gate scope | ZERO changes to slancha-api hot path. New surface area: CF config (Origin Groups + KVS + Function), L@E function w/ KMS, small `/v1/admin/usage` endpoint + idempotency migration | derives from architecture |
| Future N=many self-hosters | KVS sharded (NC2): slancha-route ~50K users/KVS, slancha-ceiling ~19K users/KVS. Cap = min(50K, 19K) = 19K w/ pref ceilings; 50K without. Sharding plan deferred to v0.2 if >19K self-hosters need ceilings; v0.1 tested single-tenant | CF KVS hard cap |

### What this replaces

The prior `claude/mesh-wire-in` branch on slancha-api (mesh_node_url field on RoutingResult, MeshProvider class proposed, chat.py wire-in proposed) is **architecturally wrong** under this plan. The routing decision belongs at the edge, not inside the Lambda's classifier output. The mesh-first wire was a useful prototype but should be deprecated when this lands.

---

## 2. Lever set — what agents can manipulate

**Tier 1 — MUST-HAVE (lock in v0.1):**

| Lever | Type | Notes | Source |
|-------|------|-------|--------|
| `max-cost-cents` | int ≥0 | Hard cost ceiling per request | RouterArena |
| `max-latency-ms-p95` | int | Hard latency ceiling — **p95**, not avg (averages hide jitter; RouterBench finding) | RouterArena |
| `min-throughput-tps` | int | Min sustained tokens/sec under load | OpenRouter `sort:throughput` |
| `quality-weight` | float [0,1] | RouterBench-style λ; smooth pareto knob | RouterBench |
| `require-capabilities` | string[] | Hard gate: `tools`, `json_mode`, `vision`, `streaming`, `seed`, `system_prompt`, `parallel_tool_calls` | OpenRouter `require_parameters` |
| `allow-fallbacks` | bool | Fall through to cloud on mesh failure | LiteLLM |
| `fallback-strategy` | enum | `price` \| `quality` \| `latency` \| `none` | Portkey |
| `request-id` | string | Required for idempotent telemetry (B5 fix) | telemetry contract |
| `reasoning-effort` | enum | `low` \| `medium` \| `high` (OpenAI native; affects cost+latency) | H9 |
| `cache-control` | enum | `ephemeral` \| Anthropic prompt-caching trigger (90% discount on hit; ignoring picks wrong specialist) | H9 |
| `cache-ttl-s` | int | Anthropic cache TTL hint | H9 |

**Tier 2 — SHOULD-HAVE (privacy/compliance, cheap to add):**

| Lever | Type | Notes |
|-------|------|-------|
| `zdr` | bool (`?1`/`?0`) | Zero-data-retention only |
| `region` | string | Data residency (`us`, `eu`, `lan`) |
| `service-tier` | enum | Semantic enum bridging providers: `cheap_slow` \| `balanced` \| `fast_premium` \| `async_batch`. Translated at gateway to provider-specific tier (OpenAI `flex/default/priority`; Anthropic request `standard_only/auto`). Do NOT expose provider-specific names directly (they don't bridge). |
| `quantizations` | string[] | Allow-list mirroring OpenRouter: `int4` \| `int8` \| `fp4` \| `fp6` \| `fp8` \| `fp16` \| `bf16` \| `fp32` \| `unknown`. Capability gate, not score (per "Give Me BF16 or Give Me Death" arxiv 2411.02355: precision tier ≠ measured eval loss). |

**Tier 3 — NICE-TO-HAVE:**

| Lever | Type | Notes |
|-------|------|-------|
| `min-context-window` | int | Hard gate when known |
| `require-determinism` | bool (`?1`/`?0`) | Seed + temp-stable models only |
| `language` | string | Hint for multilingual routing |
| `streaming-jitter-max-ms` | int | For voice/UX-critical streams |

**SKIP (intentional rejection):**
- Safety/refusal calibration as request lever — drifts adversarially, handle via guardrails
- Hallucination rate as request lever — too noisy, domain-dependent, fold into quality
- Cold-start as request param — provider concern, not agent concern
- Robustness-to-perturbation — research metric only, surface as model-card metadata

**Moat framing (H5 corrected)**: OpenRouter already exposes ~70% of Tier 1 per-request (`max_price`, `sort: throughput|latency|price`, `require_parameters`, `data_collection: deny`, `zdr`, `quantizations[]`, `allow_fallbacks`). "First multi-axis" is refutable. The defensible moat is **multi-axis preferences AS PEERS WITH** (a) router-observed quality, (b) signed-card provenance via did:web, (c) **per-USER-data quality scoring** (`paul-v8-essay-dpo-iter1-beta-0.1` measured on Paul's writing corpus, not public benchmarks), (d) open protocol + reference impl. Each cloud router is structurally blocked from (b) and (c) — they don't own the user's data and they don't sign per-deployment.

### API surfaces

```
# Shape 1: Headers — RFC 8941 Structured Fields Dictionary (typed, parseable, escapable)
X-Slancha-Pref: max-cost-cents=5, max-latency-ms-p95=2000, quality-weight=0.6, allow-fallbacks=?1
X-Slancha-Capability: tools, json_mode
X-Slancha-Service-Tier: balanced

# Booleans: ?1 / ?0 per RFC 8941
# Strings:  quoted, escape with \"
# Tokens:   bare (e.g. service-tier=balanced)
# Decimals: explicit decimal point
# Integers: bare

# Shape 2: Body — same names, kebab→snake conversion at API boundary
{
  "model": "auto",
  "pref": {
    "max_cost_cents": 5,
    "max_latency_ms_p95": 2000,
    "quality_weight": 0.6,
    "require_capabilities": ["tools"],
    "allow_fallbacks": true,
    "fallback_strategy": "quality"
  }
}

# Shape 3: Service-tier semantic presets (named bundles for agents)
# Semantic enum — translated at gateway to provider-specific tier.
# Do NOT use provider-specific names directly — OpenAI{auto,default,flex,priority}
# and Anthropic{auto,standard_only} don't bridge.
{"model": "auto", "service_tier": "cheap_slow"}      # OpenAI flex; Anthropic standard
{"model": "auto", "service_tier": "balanced"}        # default, OpenAI default
{"model": "auto", "service_tier": "fast_premium"}    # OpenAI priority; Anthropic auto (priority-eligible)
{"model": "auto", "service_tier": "async_batch"}     # routed to batch endpoints, 50% off, ≤24h

# Shape 4: Explicit (override everything)
{"model": "paul-v8-essay-dpo-iter1-beta-0.1"}                           # hard pin, skip pareto
```

### Discovery API (for agents)

```
GET /v1/models?include=routing_meta&domain=writing&pref.quality_weight=0.7
→ ranked list of specialists w/ explainability metadata
```

### Architectural insight from Portkey/Bifrost

Flat preference dict is **wrong shape**. Need nestable strategy tree — `fallback ∘ loadbalance ∘ conditional` — because "prefer cheap, fall back to expensive on 5xx" can't be expressed as flat weighted-sum. v0.1 supports 1-level nesting (single fallback strategy); CEL/nested deferred to 1.0.

---

## 3. Specialist card schema v0.1

Per protocol research: **JSON + JSON Schema 2020-12, no JSON-LD, no Protobuf.**

HTTP media type: `application/vnd.slancha.specialist-card.v1+json` on `.well-known/slancha-card.json`. Dispatch by media-type at HTTP layer (M12).

```json
{
  "schema_version": "0.1",
  "node_id": "did:web:mesh.paul.laulpogan.com",
  "node_capabilities": ["com.slancha.lora_registry@1"],

  "heartbeat": {
    "interval_s": 30,
    "last_seen": "2026-05-22T18:14:00Z"
  },

  "specialists": [
    {
      "id": "paul-v8-essay-dpo-iter1-beta-0.1",
      "family": "writing",
      "base_model": "hermes-3-3b",
      "quantization": "bf16",
      "model_hash": "sha256:<sha256 of base weights as published by base author, HF model-card aligned>",
      "lora_adapter_hash": "sha256:<sha256 of safetensors LoRA file>",
      "merged_identity": "sha256:<base + adapter + quant triple>",

      "endpoints": [
        {
          "shape": "openai_chat",
          "path": "/v1/chat/completions",
          "version": "v1"
        },
        {
          "shape": "anthropic_messages",
          "path": "/v1/messages",
          "version": "v1"
        }
      ],

      "capabilities": ["streaming", "system_prompt"],
      "modalities": {"in": ["text"], "out": ["text"]},
      "context_length": 32768,
      "supported_parameters": ["temperature", "top_p", "seed", "max_tokens",
                                "reasoning_effort", "cache_control",
                                "parallel_tool_calls"],

      "cost": {
        "unit": "1M_tokens",
        "currency": "USD",
        "input": 0.0,
        "output": 0.0,
        "amortized_hourly": 0.05
      },

      "latency": {
        "p50_ttfb_ms": 120,
        "p95_ttfb_ms": 280,
        "p99_ttfb_ms": 410,
        "tokens_per_second_p50": 50,
        "tokens_per_second_p95": 45,
        "cold_start_ms": null,
        "queue_depth": 0,
        "concurrency_max": 4,
        "window_minutes": 5
      },

      "quality": {
        "router_observed": null,
        "node_self_reported": null,
        "observation_source": "synthetic",
        "ttl_s": 86400,
        "eval_set": "holdout_v1.jsonl",
        "eval_set_sha": "54bc05af",
        "last_evaluated_at": "2026-05-21T03:00:00Z",
        "sample_count": 500,
        "overall_score": 3.95,
        "per_domain": {"writing": 4.20, "math": 2.10, "code": 3.10}
      },

      "privacy": {
        "egress": "none",
        "logs_retention_s": 0,
        "data_processor": "self",
        "node_location": "home_lan_us_east",
        "attestation": null
      }
    }
  ],

  "signature": {
    "alg": "EdDSA",
    "kid": "did:web:mesh.paul.laulpogan.com#key-1",
    "protected": "eyJhbGc...",
    "signature": "..."
  },

  "extensions": {}
}
```

### Critical schema decisions (LOCKED for v0.1)

1. **`quality.router_observed`** is the trusted field for routing. `quality.node_self_reported` is published but routers IGNORE by default. Routers MUST NOT use cross-mesh node self-reports as routing input.
2. **Cold-start bootstrap (H12 + NC4)**: new specialist with `router_observed=null` falls back to `node_self_reported × bootstrap_discount` for first N=100 observed requests; switch to `router_observed` once `sample_count ≥ 100`. Avoids death-spiral where new specialists get zero traffic. `observation_source` ∈ {`synthetic`, `shadow`, `real_traffic`} tells consumers what they're trusting. **`bootstrap_discount` is router-tunable** (recommended range 0.3–0.7; default 0.5; calibrate empirically against shadow traffic). Lower = harder for trust-injection attacks; higher = faster bootstrap of legitimate new specialists.
3. **Cost router-computed, not node-attested (H13)**: cards publish `base_model` + native `cost.*` for amortized hardware cost. Routers compute `cloud_equivalent_*` from canonical price index (slancha-api `app/router/model_catalog.py`). Telemetry logs BOTH node claim and router-computed equivalent → `mesh.card.cost_drift` event when divergence >25%.
4. **JCS canonicalization (RFC 8785)** before signing — deterministic representation. **JCS pins (M9 explicit)**:
   - String values MUST be NFC-normalized BEFORE JCS (RFC 8785 §3.4 does NOT mandate NFC; spec adds it for cross-impl determinism)
   - Numeric fields MUST be representable in IEEE-754 double; ban integers > 2^53; ban non-double-precision sub-ULP floats
   - Pin Python lib: `rfc8785>=0.1.4` OR `jcs>=0.2.1` (conformance suite cross-validates both)
   - Pin JS lib: `@truestamp/canonify` (active maintenance) — alternative `canonicalize` (unmaintained)
   - Test vectors in conformance corpus (#63a) include: empty object, deep nesting, unicode-NFC edge cases, integer-boundary cases, all locale-sensitive number formats
5. **JWS-over-JCS procedure (H11 exact)**:
    1. Build card object WITHOUT `signature` field
    2. JCS-canonicalize per RFC 8785
    3. Compute detached JWS per RFC 7515 Appendix F with `b64=false, crit=["b64"]`
    4. Embed flattened JWS `{protected, signature}` in `card.signature`
    5. Conformance suite ships test vectors — single byte drift = unverifiable
6. **`endpoints[]` array per specialist (H19 + Q5 flip)**: each specialist MAY advertise multiple endpoint shapes. **Anthropic shape REQUIRED-IF-ADVERTISED** — if `endpoints[].shape == "anthropic_messages"`, the specialist MUST handle native Anthropic Messages API streaming + tool_use shape correctly. v0.1 normative MAY treat OpenAI shape as the reference, but reference impl ships both. Claude Code (primary distribution) uses `/v1/messages` natively — SaaS→OpenAI-translator→mesh-native-Anthropic round-trip is real implementation tax (cf. slancha-api ffd3901).
7. **`extensions` block** uses reverse-DNS keys (`com.slancha.lora_registry`) for experimental fields. Promotion to top-level via SEP. (OCI annotations pattern.)
8. **Per-endpoint versioning mechanism (M13)**: URL path prefix (`endpoints[].version: "v1"` maps to `/v1/...`). Card MAY advertise multiple endpoint versions concurrently. Sunset header per RFC 8594 announces deprecation. Routes deprecate independently of card schema. (Matrix lesson.)
9. **`node_capabilities[]`** lists protocol-level extensions (`com.slancha.lora_registry@1`). Specialist-level `capabilities` stays per-model (tools/streaming/seed/etc). Node-level for protocol extensions. (MCP `tools/list` precedent.)
10. **Version-skew rule (M15)**: Routers MUST accept future minor versions, ignore unknown fields (Postel). Routers MUST reject major-version mismatch. Specialists support N-1 minor for ≥6 months. Mirrors MCP 2025→2026 policy.
11. **`model_hash` semantics (M11)**: SHA-256 of base weights file as published by base author (matches HF model card). `lora_adapter_hash` = SHA-256 of safetensors LoRA file. `merged_identity` = SHA-256 of `(base + adapter + quant)` triple. `quantization` ∈ standard enum (see §2).
12. **OpenTelemetry GenAI attribute names** for telemetry payloads. Pin `otel_semconv_version: "1.36.0+dev"` in telemetry envelope so receivers dispatch on schema (M10).

### did:web key rotation + revocation (H10 — fills gap)

Spec adds explicit lifecycle:

- did.json MUST list ≥1 active `verificationMethod` entry per key
- Rotation: ≥7d overlap window where old + new keys both present
- `did.json` Cache-Control: `max-age` ≤ 300s; verifier MUST refresh on signature-verification failure before rejecting
- Registry pins `node_id_fingerprint: sha256:<did.json bytes at first registration>`. Fingerprint flip = drain routes from this node + alert operator (domain transfer / impersonation defense)
- Revocation: remove `verificationMethod` entry; verifier sees absence → reject. Optional `revoked: true` field on the entry for explicit signaling

### Privacy attestation honesty (H14)

GB10 (DGX Spark) has **no hardware-attested confidential compute** — NVIDIA staff explicit per dev forum (2025-10-16). No NRAS, no SEV-SNP (AMD only), no TDX (Intel only), no ARM CCA on Spark SKU.

Therefore:
- `privacy.attestation` in core schema = `null` by default
- Hardware attestation (TPM/SEV/TDX) lives in `extensions.com.slancha.privacy_attestation` ONLY
- Routers MUST surface `privacy.egress="none"` with `[unverified]` badge in UI until evidence present
- Honest framing: home-tunnel scenario is operator promise, not cryptographic fact

W3C Verifiable Credentials remain explicitly **skipped for v0.1**. Promote at 1.0 only if ecosystem proves need for delegation / selective disclosure / third-party endorsements.

---

## 4. Trust model

| Layer | Mechanism | Why |
|-------|-----------|-----|
| Node identity | `did:web:<domain>` with key rotation/revocation (§3 rotation block) | Rides on existing DNS+TLS, no CA bureaucracy |
| Card signing | JWS over JCS-canonicalized JSON per RFC 7515 Appendix F detached pattern (§3 procedure) | Prevents registry injection + impersonation |
| Quality signals | Router-observed (canary probes); cold-start uses `node_self_reported × bootstrap_discount` (tunable 0.3–0.7, default 0.5) | Self-attested scores are gameable; observation is verifiable; cold-start bootstrap avoids death spiral |
| Cost signals | Router-computed `cloud_equivalent_*` from canonical price index. Node `cost.*` = native amortized only | Self-attested cross-mesh cost is gameable in BOTH directions (inflate to win privacy moat, deflate to win pareto) |
| Privacy attestations | Self-attested in v0.1 with `[unverified]` UI badge; hardware attestation (TPM/SEV/TDX) ONLY in `extensions.com.slancha.privacy_attestation` | TEE formats churn; GB10 has no HW attestation per NVIDIA |
| Last-mile auth (SaaS→mesh) | CF Access service token + HMAC `X-Slancha-Forward-Sig` (HMAC over user_id+timestamp+nonce+route_target+origin_id; ±300s replay window; nonce LRU dedup 600s) | Tunnel URL alone is not auth (Ollama 175k exposed hosts) |
| Mesh→SaaS auth | Bearer `mesh-ingest-token` per-node, issued at registration. **Different key, different direction — DO NOT reuse HMAC key cross-direction** | Defense-in-depth + clean failure isolation |
| Mesh-originated SaaS callbacks | **Banned in v0.1**. SaaS gateway rejects requests where `X-Slancha-Forward-Sig` claims node origin. Defense-in-depth against agent-loop attacks (replaces H23 hop-count header which is trivially strippable). | Signed forwarding chain deferred to v1.0 |
| Model provenance | `model_hash` semantics defined (§3 M11). Sigstore Rekor entry in `extensions.dev.sigstore.provenance` | Optional for high-stakes deployments |
| Pref ceiling (H15) | KVS per-user `pref_max` admin ceiling. CF Function enforces `min(client_pref, pref_max)`. Emit `slancha.pref.ceiling_exceeded` on excess | Prevents client-injected ceiling bypass; mirrors slancha-api a1f582e fix |
| HMAC key rotation | 90d cycle with 14d grace window. KID-aware (multiple active versions). Rotation tooling built BEFORE auto-rotation enabled (H22) | Stripe convention; manual until N successful unattended runs |

### 4.1 Auth direction matrix (NC3 + M18)

Three tokens × four lifecycle dimensions. **No token reused across directions.** Symmetric documentation of asymmetric auth:

| Token | Direction | Rotation | Grace | Revocation | Replay defense |
|-------|-----------|----------|-------|------------|----------------|
| `X-Slancha-Forward-Sig` (HMAC) | SaaS → Mesh | 90d | 14d (KID-aware multi-version) | Drop KID from L@E bundle redeploy | ±300s timestamp tolerance + nonce LRU dedup 600s window |
| `CF-Access-Client-Id` / `Client-Secret` | SaaS → Mesh (CF edge gate) | 90d | 7d (CF Access default) | CF Access dashboard "revoke" | CF Access JWT signature + audience check |
| `mesh-ingest-token` (Bearer) | Mesh → SaaS (telemetry sidecar) | 90d | 14d (rolling per-node issue) | Revoke via slancha-api admin endpoint; node re-registers | None (idempotent endpoint via `request_id` unique index makes replay safe) |

**Defense-in-depth invariants:**
- HMAC key rotation does NOT require mesh-ingest-token rotation (different bus, different blast radius)
- CF Access token revocation does NOT invalidate HMAC (one is edge gate, other is body auth)
- Rotation tooling MUST handle all three independently (#65d)
- All three follow 90d cadence convention but desync-friendly (don't rotate same day)

---

## 5. Discovery

Three channels, in order of importance:

1. **Centralized registry (`registry.slancha.dev`)** — primary. Failed federations had no canonical hub; Matrix/Mastodon/OCI succeeded because of de-facto centers. Routers pull aggregated card index every N seconds.
2. **`.well-known/slancha-card.json`** (RFC 8615) — each node serves its own card here. Lets discovery work without registry. Mirrors A2A's `agent-card.json` pattern.
3. **DNS-SD `_slancha._tcp`** — LAN-local discovery for "my laptop finds my Spark across the room." RFC 6763. PTR + SRV + TXT records.

**Explicitly skipped for v0.1**: libp2p Kademlia DHT. Future work when >1000 nodes exist.

---

## 6. Telemetry contract

Mesh router POSTs to `https://api.slancha.ai/v1/admin/usage` after each request close.

### Auth direction matrix (M18)

- **Mesh → SaaS** (this endpoint): `Authorization: Bearer <mesh-ingest-token>` per-node, issued at registration. **No HMAC** — bearer alone.
- **SaaS → Mesh** (request forwarding): `X-Slancha-Forward-Sig` HMAC + CF Access service token. **No bearer** — service-to-service auth.
- **Keys are NOT shared across directions.**

### Payload

```json
POST /v1/admin/usage
Authorization: Bearer <mesh-ingest-token>
Content-Type: application/json

{
  "request_id": "uuid-...",
  "user_id": "user-paul",
  "specialist_id": "paul-v8-essay-dpo-iter1-beta-0.1",
  "endpoint": "/v1/chat/completions",
  "tokens_in": 230,
  "tokens_out": 1247,
  "latency_ms": 4830,
  "ttft_ms": 124,
  "tokens_per_second": 51.6,
  "cost_cents": 0,
  "cloud_equivalent_cost_cents_router_computed": 7,
  "cloud_equivalent_cost_cents_node_claimed": 7,
  "status_code": 200,
  "route_target": "mesh",
  "fallback_fired": false,
  "pref_applied": {"quality_weight": 0.6, "max_cost_cents": 5},
  "decision_reason_structured": {
    "winner": "paul-v8-essay-dpo-iter1-beta-0.1",
    "alternatives_considered": [
      {"id": "sonnet-4-6", "delta": 0.13, "losing_axes": ["cost"]}
    ],
    "deciding_axes": ["cost", "quality"],
    "preset_applied": "balanced"
  },
  "decision_reason": "paul-v8-essay-dpo-iter1-beta-0.1: pareto-winner over sonnet-4-6 by 0.13 (quality match + cost)",
  "otel_semconv_version": "1.36.0+dev",
  "gen_ai.request.model": "paul-v8-essay-dpo-iter1-beta-0.1",
  "gen_ai.usage.input_tokens": 230,
  "gen_ai.usage.output_tokens": 1247
}
```

### Critical contract guarantees

- **No prompt body, no completion body, ever.** Counts and metadata only.
- **OpenTelemetry GenAI semconv compliance** for `gen_ai.*` fields. `otel_semconv_version` pinned in envelope so receivers dispatch on schema.
- **Existing `log_usage` shape preserved** — dashboard works unchanged.
- **Idempotency (B5)**: `request_id` is unique key. `/v1/admin/usage` MUST `ON CONFLICT (request_id) WHERE route='mesh' DO NOTHING`. Partial unique index shipped via migration before endpoint goes live.
- **Decision provenance**: `decision_reason_structured` is the machine-readable explainability field. `decision_reason` is human-readable mirror. Dashboard renders both.
- **Cost double-recording (H13)**: BOTH `cloud_equivalent_cost_cents_router_computed` (trustworthy) AND `cloud_equivalent_cost_cents_node_claimed` (display + drift audit). Emit `mesh.card.cost_drift` event when divergence >25%.

### Retry semantics (M19, H16)

1. Mesh writes durable local jsonl `~/.slancha-local/usage-buffer.jsonl` BEFORE async POST
2. POST with bearer + idempotent request_id
3. On failure: exponential backoff 1s / 5s / 30s
4. After N=3 failures: append to DLQ `~/.slancha-local/usage-dlq.jsonl`
5. Nightly reconcile script backfills DLQ → /v1/admin/usage
6. Telemetry loss is acceptable; chat response is not. Mirrors Stripe webhook pattern from slancha-api CLAUDE.md.

---

## 7. Anti-patterns we explicitly avoid

| Failure mode | Lesson source | How we avoid |
|--------------|---------------|--------------|
| Lambda in mesh hot path billing for SSE duration | AWS billing model | L@E origin-request selects Origin Group only |
| L@E body inspection caps request size at 1MB (B1) | AWS L@E restrictions doc | HMAC over identity claims only; mesh-side body MAC after passthrough |
| L@E env-var/VPC/layers/arm64 reach for secrets (B2) | AWS L@E restrictions | KMS Decrypt at cold start, in-process cache for replica lifetime |
| L@E reads KVS (B3) | KVS scoped to CF Functions | Health gate moves upstream to CF Function viewer-request |
| Mesh DNS outage → CF 502 before L@E runs (B4, M2) | AWS L@E restrictions | CF Origin Groups primary=tunnel + secondary=lambda-default; native failover |
| `MeshProvider` as just-another-OpenAI-provider hiding mesh state | prior `claude/mesh-wire-in` design critique | Routing decision at edge, not at provider layer; deprecate that branch |
| L@E origin-response or viewer-response trigger buffers SSE (H4) | AWS L@E response-streaming gap | **Forbidden by spec; enforced via CloudFront distribution diff in CI** |
| Self-attested quality scores trusted by routers | RouterBench + protocol research | Router-observed canary probes; node claims = display-only; cold-start uses node × 0.5 discount until N≥100 |
| Self-attested cost across meshes (H13) | gaming incentive in both directions | Router-computed from canonical price index; node claim = display + drift audit only |
| Tunnel-as-auth (Ollama 175k exposed hosts) | SentinelOne/Censys early 2026 | CF Access service token + HMAC mandatory |
| Flat preference dict can't express composition | Portkey/Bifrost precedent | Recursive `pref` schema with MAX_DEPTH=3 validator; v0.1 rejects depth>1 but ships recursive shape |
| JSON-LD complexity tax | ActivityPub adoption pain | Plain JSON + JSON Schema 2020-12 |
| Breaking changes within 0.x | MCP 2025 churn | Public no-break commitment; experiments in `extensions` |
| Decentralized from day one with no gravity well | OStatus/Diaspora failure | Centralized registry primary |
| Registry SPOF (M16, npm 2022 outage) | distributed-systems history | Routers MUST cache last card index, serve stale up to N hours, expose staleness in `decision_reason` |
| Spec without reference impl | many IETF drafts | AGPL ref impl + conformance suite ship with v0.1 |
| AGPL on spec text kills adoption (H17) | Matrix/MCP/OCI all Apache-style | License bifurcation: spec = Apache 2.0 or CC-BY-4.0; ref impl = AGPL-3.0; conformance = Apache 2.0 |
| Governance undefined → SEP burns quarters (H18) | MCP 2025 governance pain | GOVERNANCE.md before tag: SEP template, ≥14d comment, BDFL+rough-consensus, contributor IP grant |
| Cost-attribution drift on silent fallback | Vercel BYOK silent fallback | `fallback_fired` flag in telemetry, dashboard alert |
| SSRF via attacker-supplied mesh_url; DNS rebind (H2) | Portkey RFC1918 block + DNS-rebind attack | Allowlist pinned to CF tunnel IP ranges (CF publishes); reject otherwise |
| Prompt-injection bypass on client-set pref (H15) | OWASP injection class; slancha-api a1f582e | KVS `pref_max` ceiling; CF Function enforces `min(client, ceiling)`; emit `pref.ceiling_exceeded` |
| Sidecar mid-stream crash = free compute / billing dispute (H16) | Stripe `gateway.usage.stripe_skipped` | Durable local jsonl before async POST; exp retry 1s/5s/30s; DLQ; nightly reconcile |
| Mesh→SaaS callback loop attacks (H23) | TCP-TTL-spoof analog | **Ban mesh-originated SaaS callbacks entirely**. SaaS gateway rejects requests claiming node origin in forward-sig. Signed forwarding chain deferred to v1.0 |
| Quality death spiral on new specialists (H12) | cold-start chicken-and-egg | Bootstrap via `node_self_reported × 0.5` until `sample_count ≥ 100`; `observation_source` field flags trust level |
| GB10 has no HW attestation despite "Confidential Compute" branding (H14) | NVIDIA dev forum 2025-10-16 | Spec MUST NOT promise HW attestation in v0.1; live in extensions only; `[unverified]` UI badge |
| AGPL re-licensing later requires every contributor sign-off | OSS license history | Lock licenses BEFORE v0.1.0-spec tag — bifurcation is one-way after publish |
| User changes preferences mid-conversation; session pin overrides | classifier/router.py session manager | Explicit `pref` overrides session pin |
| Multi-specialist-shared-VRAM lying about concurrency_current | resource accounting | Reserve `extensions.com.slancha.resource_group`; routers use `concurrency_max` static (H21) |
| HMAC replay over time | crypto hygiene | ±300s timestamp tolerance, nonce LRU dedup 600s window |
| CF Access service token expiry storm (M3) | secret-rotation cascade | Runbook + alert on TTL; 90d cycle, 14d grace |
| KVS multi-region staleness (~60s) | CF KVS eventual consistency | Health-cache TTL respects window; document propagation gap |
| KMS Decrypt cold-start tax | AWS L@E + KMS round-trip | Document +50-100ms p99 explicitly; pre-warm via CloudWatch synthetic |
| JSON Schema 2020-12 validator gap (Go qri-io = Draft 7) | ecosystem reality | Document fallback to Draft 7 with compatibility shim |
| Specialist version-skew (M15) | MCP 2025→2026 migration | Postel: accept future minor, ignore unknown fields; reject major mismatch; specialists support N-1 minor ≥6 months |
| model_hash semantics undefined (M11) | reproducibility | SHA-256 of base weights file as published by base author; LoRA = SHA-256 of safetensors; `merged_identity` = triple hash |
| Spec/conformance license gap blocking embedding | OSS adoption | Conformance suite = Apache 2.0 explicitly for max embedding |

---

## 8. Phased rollout (mac's recommended reorder)

| Phase | Tasks | Outcome | Estimate |
|-------|-------|---------|----------|
| **0 — Artifact verify (NEW)** | #76 | **Owner: slancha-spark** (artifacts in `~/finetuning/output/`). NOT willard-spark (renamed spark-472e — different SLA, gpu-scheduler shared box). Pick canonical v8 artifact; smoke vLLM hot-load; rename slug in spec | 1-2 hrs |
| 1 — LAN-direct voice | #50, #51, #53 | Chosen v8 specialist servable via tunnel | 2-4 hrs |
| 2 — Telemetry sidecar | #54-#57, #81 | Dashboard captures mesh-direct usage w/ idempotency migration | 4-8 hrs |
| 3 — Edge routing | #49, #58-#60, #78-#80 | api.slancha.ai → mesh transparently; CF Origin Groups; KMS at L@E cold start; CF Function health gate | 2-4 days |
| **4+5 INTERLEAVED — Spec ⊕ Levers** | #61-#64, #66-#70, #84, #85 | Spec drafts in parallel with lever impl; spec freeze AT END of Phase 5 only after one lever round-trips end-to-end | 5-8 days |
| 6 — Quality observability | #71-#73, **probe service promoted from this phase to Phase 5 for cold-start policy on Day 1** | Router-observed quality + drift alerts | 2-3 days |
| 7 — Discovery v0.1 complete | #74-#75 | DNS-SD + public registry MVP | 1-2 days |
| **8 — Governance + AGPL (NEW, split from Phase 4)** | #65, #82, #83 | License bifurcation (spec=Apache, ref impl=AGPL, conformance=Apache); GOVERNANCE.md; SEP scaffold; lock BEFORE v0.1.0-spec tag | 2-3 days |

**Total: ~3-4 weeks if sequential, parallelizable across mac/spark/api workstreams.**
**Phase 0 + 1 + 2 alone (~10-12 hrs) ship chosen v8 specialist end-to-end with telemetry.**

**Critical**: Phase 4+5 are INTERLEAVED, not sequential. Spec freeze happens at END of Phase 5 once a lever round-trips. Spec-first risks locking schema fields informed by implementation-naive thinking.

**Probe service promotion (H12)**: Phase 6's router-observed quality probe is promoted up to Phase 5 so cold-start specialists have a policy on Day 1 (otherwise `quality.router_observed=null` triggers death spiral).

---

## 9. Open questions — DECIDED (mac triple-check)

| # | Question | Decision | Why |
|---|----------|----------|-----|
| Q1 | Quality verification: central or federated probe? | **Central probe at v0.1, honestly framed**. `quality.observer` field on card; v0.1 observer=Slancha; opt-out path documented. Federate at 1.0. | Avoids claiming federation as moat then centralizing without saying so (mac caught this conflict with §11) |
| Q2 | Cost-claim verification across meshes | **Router computes from canonical price index**. Card publishes `base_model` + native cost only. Self-attested `cloud_equivalent_*` removed from card; node-published version = display + drift audit only. | Self-attest is gameable in both directions (H13) |
| Q3 | Strategy tree expressivity | **Recursive schema from day 1; validator gates depth>1 in v0.1**. `$ref: "#"` on `targets[]` mirrors Portkey shape (M6). | Avoids breaking-change at v1.0 |
| Q4 | Anthropic `/v1/messages` surface | **Anthropic shape REQUIRED-IF-ADVERTISED in `endpoints[]`**. Spec MAY treat OpenAI shape as reference, but impl ships both day 1. Claude Code is primary distribution; SaaS→OpenAI-translator→mesh-native-Anthropic tax is real (slancha-api ffd3901 active suffering on streaming tool-call shape). | Translation tax > spec scope tax |
| Q5 | Quality scoring under pref divergence | **Per-domain (already) + `quality-weight` low/med/high only**. NOT full pref-vector bucketing (2187 buckets ÷ 500 samples = 0.23/bucket = noise) (H20). | Statistical validity floor |
| Q6 | Multi-mesh load-balancing | **`concurrency_max` static (registry-side filter), 503/Retry-After at capacity**. `concurrency_current` = display only. Push-LRS deferred to v1.0 (H21). | Pull cadence too coarse for sub-second concurrency state |
| Q7 | Token rotation cadence | **90d rotation, 14d grace**. Build rotation tooling FIRST. Tighten only after N successful unattended runs (H22). | 30d × 3 secret types = 36 events/yr → cascade risk |
| Q8 | Agent-mediated callback loops | **Ban mesh-originated SaaS callbacks entirely**. SaaS gateway rejects requests claiming node origin in `X-Slancha-Forward-Sig`. Signed forwarding chain deferred to v1.0 (H23). | Header-based hop-count trivially strippable |

---

## 10. Task list (rev 2: 23 new + 10 from mac's triple-check + 3 reframed + Phase 0)

| # | Phase | Owner | Task |
|---|-------|-------|------|
| 76 | 0 | SPARK | Verify v8 LoRA artifact + rename slug in spec |
| 50 | 1 | SPARK | Register chosen v8 specialist (paul-v8-essay-dpo-iter1-beta-0.1) as mesh specialist |
| 51 | 1 | SPARK | CF tunnel mesh.paul.laulpogan.com → slancha-local |
| 53 | 1 | SPARK | CF Access service-token policy on mesh tunnel |
| 54 | 2 | SPARK | slancha-local: HMAC verify middleware + service token gate |
| 55 | 2 | SPARK | slancha-local: after-response telemetry sidecar POST (durable jsonl + retry + DLQ) |
| 56 | 2 | API | POST /v1/admin/usage ingestion endpoint |
| 81 | 2 | API | usage_logs idempotency partial unique index + ON CONFLICT DO NOTHING (B5) |
| 57 | 2 | MESH | Cross-repo round-trip test for usage payload |
| 49 | 3 | INFRA | CloudFront KVS: user→route config + paul seeded |
| 58 | 3 | INFRA | CloudFront Function (viewer-request): KVS lookup + header injection + pref ceiling + health gate |
| 59 | 3 | INFRA | Lambda@Edge (origin-request): SSRF allowlist + KMS Decrypt cold start + HMAC + Origin Group select |
| 80 | 3 | INFRA | CF Origin Groups primary=tunnel/secondary=lambda (B4 DNS failover) |
| 78 | 3 | INFRA | KMS Decrypt at L@E cold start for HMAC + CF-Access secrets (B2) |
| 79 | 3 | INFRA | Health-prober Lambda (regional, NOT @edge) writes mesh_healthy to KVS (B3) |
| 60 | 3 | INFRA | Firehose + S3 + Athena table for L@E decision telemetry |
| 60a | 3 | API | **Deprecate** `claude/mesh-wire-in` branch + revert MeshProvider/mesh_node_url shells if any merged (M25) |
| 61 | 4+5 | MESH | SpecialistCard JSON Schema 2020-12 + reference doc |
| 62a | 4+5 | MESH | JCS canonicalization (RFC 8785) lib pin + cross-validation tests |
| 62b | 4+5 | MESH | JWS sign/verify per RFC 7515 Appendix F (detached, b64=false, crit=["b64"]) |
| 62c | 4+5 | MESH | did:web resolution + key rotation/revocation lifecycle (H10) |
| 63 | 4+5 | MESH | slancha-conformance CLI + test suite |
| 63a | 4+5 | MESH | Conformance test corpus — JSON fixtures + golden-trace generator (M26) + **decision_reason_structured JSON Schema + golden trace fixture (NC5)** + **JCS test vectors (M9)** + **JWS-over-JCS detached test vectors (Q2)** |
| 64 | 4+5 | SPARK | .well-known/slancha-card.json endpoint on slancha-local |
| 66 | 4+5 | API | X-Slancha-Pref RFC 8941 Structured Fields header + body pref parsing |
| 67 | 4+5 | API | Service tier semantic enum (cheap_slow/balanced/fast_premium/async_batch) → provider translation |
| 68 | 4+5 | MESH | Pareto-frontier scoring in POST /place w/ cold-start + queue penalties |
| 69 | 4+5 | MESH | GET /v1/models?include=routing_meta discovery API |
| 70 | 4+5 | MESH | decision_reason_structured + human-readable mirror in telemetry |
| 70b | 4+5 | API | Dashboard panel: per-request routing decision breakdown (quality match + cost delta + capability gate audit) (M30) |
| 84 | 4+5 | MESH | Card schema endpoints[] array allowing OpenAI + Anthropic shapes (H19) |
| 71 | 5/6 | SLOW-LOOP | Continuous eval pipeline: nightly score → card.quality.router_observed |
| 72 | **5** (promoted) | INFRA | Router-observed quality probe service (canary prompts; cold-start policy on Day 1) |
| 73 | 6 | MESH | Quality drift alerts + dashboard panel |
| 74 | 7 | SPARK | DNS-SD _slancha._tcp service registration on slancha-local |
| 75 | 7 | INFRA | registry.slancha.dev MVP (read-only) |
| 82 | 8 | MESH | License bifurcation: spec=Apache 2.0, ref impl=AGPL-3.0, conformance=Apache 2.0 (H17) |
| 83 | 8 | MESH | GOVERNANCE.md before tag — SEP template, 14d comment, IP grant (H18) |
| 65 | 8 | MESH | Public spec repo + SEP process scaffold (renamed from "AGPL + SEP" — AGPL split to ref impl only) |
| 65a | 8 | INFRA | spec.slancha.dev rendered (M27) |
| 65b | 8 | INFRA | Operator runbook for token rotation (M27) |
| 65c | 8 | MESH | lever-recipes.md for agents (M27) |
| 65d | 8 | INFRA | **Rotation tooling (H22 fix)**: HMAC KID seed/swap script + CF Access service-token rotator + mesh-ingest-token rotator + dry-run CLI. Required BEFORE claiming "auto-rotation enabled." Three independent rotators per §4.1 matrix. |

Plus pending non-protocol work: **#45** (v0.0.7 live cluster smoke).

---

## 11. The moat (H5 tightened)

OpenRouter exposes ~70% of Tier 1 already (`max_price`, `sort: throughput|latency|price`, `require_parameters`, `data_collection: deny`, `zdr`, `quantizations[]`, `allow_fallbacks`). Claiming "first multi-axis" is refutable.

**Defensible moat**: multi-axis preferences AS PEERS WITH —

1. **Router-observed quality** (canary probes, not self-attested)
2. **Signed-card provenance** via `did:web` + JCS+JWS (verifiable identity for each deployment)
3. **Per-USER-data quality scoring** — `paul-v8-essay-dpo-iter1-beta-0.1` measured on Paul's writing corpus, not public benchmarks. **This is the structural moat closed routers can't replicate** — they don't own the user's data, can't train on it, can't measure against it.
4. **Open protocol** (Apache-licensed spec, AGPL ref impl, Apache conformance suite) — ecosystem implementers welcome
5. **Routing transparency** — `decision_reason_structured` shows alternatives + losing axes. Override at any lever. Closed routers structurally can't expose this.

Cloud routers can clone (1) and (5) eventually. Cloud routers structurally cannot clone (2), (3), or (4) without becoming a different product.

**Honest disclosure (NC1)**: in v0.1 the quality probe is **central** (Slancha-operated); self-hosters CAN opt out and publish their own signed observations under `quality.observation_source: real_traffic`. Federation of the probe service deferred to v1.0. Don't read "every node publishes signed signals" as "every node IS the observer" — observation is a function of the registry, signing is a function of the node.

Self-hosters run pure mesh + their own preference engine. Slancha-SaaS sells the polished dashboard + multi-user management + central quality probe on top. **Same protocol**.

UX commitment: dashboard panel showing per-request routing decision breakdown — quality match + cost delta + capability gate audit (M30, task #70b). Without this the "transparency as product feature" claim is overclaim.

---

## 12. Request to mac (rev 3 — SAFE TO TAG conditional verified)

Rev 3 incorporates Rev 2 verification + 7 new concerns (NC1-NC7) + H22 partial fix + slug-drift reconciliation. Per mac PR #1: **all blockers resolved**, 22/23 highs resolved, H22 now fully resolved, 7 NCs all addressed. **No remaining BLOCKERs, no HIGH gaps, no irreversible exposures.**

### Convergence checklist for v0.1.0-spec tag

Rev 2 (verified by mac):
- [x] B1 HMAC drops body_hash → §1, §4
- [x] B2 KMS Decrypt at L@E cold start → §1, §4, #78
- [x] B3 Health gate moved to CF Function viewer-request → §1, #58, #79
- [x] B4 CF Origin Groups for DNS failover → §1, #80
- [x] B5 usage_logs idempotency migration → §6, #81
- [x] B6 v8 artifact verified (canonical slug = paul-v8-essay-dpo-iter1-beta-0.1) → #76 done
- [x] H17 License bifurcation locked → #82
- [x] H18 GOVERNANCE.md before tag → #83
- [x] H19 endpoints[] array in card schema → §3, #84
- [x] H6/H7/H8 RFC 8941 + semantic service_tier + quantizations[] → §2, #85

Rev 3 fixes (mac PR #1 review):
- [x] B6 carryover: slug drift reconciled — all `paul-v8-essay` → `paul-v8-essay-dpo-iter1-beta-0.1`; `voice-paul-v8` residue removed from §11
- [x] H22 FULL: rotation tooling task #65d added — HMAC KID seed/swap + CF Access rotator + mesh-ingest-token rotator + dry-run CLI
- [x] NC1: §11 surfaces central-probe disclosure inline ("v0.1 probe is central, self-hosters opt-out via observation_source")
- [x] NC2: KVS sharded — slancha-route (~50K users) + slancha-ceiling (~19K users); architecture diagram updated; table claim corrected
- [x] NC3+M18: §4.1 auth direction matrix — three tokens × four lifecycle dimensions
- [x] NC4: bootstrap × 0.5 marked tunable (range 0.3–0.7; default 0.5)
- [x] NC5: decision_reason_structured added to conformance corpus #63a (schema + golden trace)
- [x] NC6: Phase 0 owner = **slancha-spark** explicit in §8
- [x] NC7: task #50 slug reconciled to full canonical
- [x] M9: JCS pins explicit (NFC mandate, integer 2^53 ban, Python/JS lib version pins, test vectors)
- [x] M11: model_hash semantics block (§3 schema decision 11)
- [x] M14: node_capabilities[] for protocol extensions (§3 schema decision 9)
- [x] Q5 flip: endpoints[] Anthropic REQUIRED-IF-ADVERTISED (not "spec slot only"); reference impl ships both day 1
- [x] Q6 interleave clarifications: conformance against each lever; spec-freeze marker = first lever produces structured decision_reason on real traffic

### Sign-off ask

Approve PR #1 + merge `claude/protocol-v0.1-draft` → `main`. Then cut clean `v0.1.0-spec` tag. Supersede `SLANCHA_MESH_V0_SPEC.md`. Execute Phase 0 → Phase 8 (~3-4 weeks total, parallelizable).

### Phase 0 next step (post-tag)

Bring up vLLM on slancha-spark with `paul-v8-essay-dpo-iter1-beta-0.1` LoRA hot-loaded against base `hermes-3-3b`. Smoke a sample completion. Phase 1+2 ~10hrs after that to specialist serving end-to-end with telemetry sidecar.
