# SLANCHA MESH PROTOCOL v0.1 — Consolidated Plan (DRAFT)

**Status**: Draft for paul-mac triple-check.
**Author**: claude (slancha-spark session, 2026-05-22)
**Inputs**: 5 parallel research streams (LLM gateway architectures, AWS routing patterns, personal AI cloud-bridge patterns, router benchmarks, federation protocol design).

This document supersedes `SLANCHA_MESH_V0_SPEC.md` once mac signs off. The prior spec framed mesh as a router with its own classifier; this draft separates concerns — mesh = placement substrate + signal-rich protocol, cloud classifier = decision maker, edge = transparent dispatcher.

---

## 1. The architecture

```
Client (Cursor / Aider / OpenAI SDK / curl / agent)
   base_url=https://api.slancha.ai/v1
   Authorization: Bearer sk-slancha-<user>
   X-Slancha-Pref: max_cost=5,max_latency=2000,quality_weight=0.6      (optional, agent-set)
   X-Slancha-Capability: streaming,tools                                (optional, hard gate)
        │
        ▼
   CloudFront (already exists per cf_origin_secret_middleware)
        │
        ├── viewer-request CloudFront Function (<1ms, no I/O):
        │     - bearer hash → KVS lookup → user_id, route_target, mesh_slug
        │     - inject X-Slancha-User-Id, X-Slancha-Route-Target headers
        │
        ├── origin-request Lambda@Edge (~20-40ms warm):
        │     IF route_target == "mesh":
        │       . validate mesh_url ∈ registered allowlist (SSRF defense)
        │       . check edge health cache; unhealthy → fall through, emit X-Mesh-Fallback
        │       . strip client Authorization
        │       . inject X-Slancha-Forward-Sig (HMAC over user_id+timestamp+body_hash)
        │       . inject CF-Access-Client-Id/Secret (service token, scoped to user's tunnel)
        │       . rewrite request.origin.customOrigin.domainName = mesh.<user>.laulpogan.com
        │       . async PUT to Firehose: routing decision telemetry
        │       . return mutated request — Lambda@Edge NOT in response path
        │     ELSE:
        │       . default origin = Lambda Function URL (existing slancha-api router)
        │
        ▼  CloudFront connects to chosen origin directly
        ▼  SSE chunks stream through unmodified
        │
   ┌────┴────────────────────────────────┐
   ▼                                      ▼
Lambda Function URL                    Cloudflare Tunnel → user's mesh
(existing slancha-api router:           (slancha-local: OpenAI-compat surface)
classifier + provider registry +         │
log_usage telemetry)                     ├── Verify CF Access service token (CF policy)
   │                                     ├── Verify HMAC X-Slancha-Forward-Sig
   │                                     ├── Read X-Slancha-Pref + X-Slancha-Capability
   │                                     ├── Local registry: pareto-rank specialists
   │                                     │     (filter on capability; score on prefs)
   │                                     ├── Dispatch to vLLM/mlx-lm specialist
   │                                     ├── Stream SSE response back via tunnel
   │                                     └── After response close:
   │                                           POST sidecar telemetry to
   │                                           api.slancha.ai/v1/admin/usage
   │                                           (counts only, no prompt body)
   ▼                                      ▼
SSE response to client                SSE response to client
```

### Why this shape (cross-referenced to research)

| Concern | Resolution | Source |
|---------|-----------|--------|
| Existing clients unchanged | Same URL, same bearer, same response shape | hard requirement |
| Bypass Lambda on mesh path | L@E origin-request rewrites origin; CloudFront connects to tunnel directly; main Lambda never invoked on mesh requests | AWS L@E origin-request docs + CF chunked-encoding pass-through |
| SSE pass-through | CloudFront natively streams chunked encoding; L@E is NOT in response path on origin-request | CF `RequestAndResponseBehaviorCustomOrigin` |
| Per-user routing | CloudFront KVS (≤5MB read-only edge KV); millisecond reads | CloudFront KVS spec |
| Telemetry shape preserved | Mesh router POSTs sidecar in `log_usage` shape after each request close | OpenRouter Private Models pattern, Helicone async mode precedent |
| Privacy: prompts don't traverse SaaS Lambda on mesh path | L@E sees only headers; main Lambda not invoked at all | LM Link "device list only" posture as design north star |
| SSRF defense | mesh_url validated against registered allowlist; never accept user-supplied raw URL at request time | Portkey blocks RFC1918 by default; Ollama 175k exposed hosts cautionary tale |
| Tunnel-direct-attack defense | CF Access service-token policy + HMAC `X-Slancha-Forward-Sig` | CF Access mandate; "tunnel-as-auth" is anti-pattern |
| Bearer leak prevention | SaaS bearer stripped before forwarding; mesh-side auth is service-to-service | Portkey `forwardHeaders` precedent, two-header pattern from CF AI Gateway |
| Fallback when mesh dead | L@E checks health cache; falls through to default origin with X-Mesh-Fallback header | Vercel silent-fallback critique → must surface fallback events |
| Cost @ 10K rpd | ~$1-2/mo all-in (CF requests + L@E + Firehose) | AWS pricing |
| Operator gate scope | ZERO changes to slancha-api hot path. New surface area: CloudFront config + L@E function + small `/v1/admin/usage` endpoint | derives from architecture |
| Future N=many self-hosters | KVS scales to ~50K user records; new self-hoster = KVS row + tunnel + Access policy | CF KVS limits |

### What this replaces

The prior `claude/mesh-wire-in` branch on slancha-api (mesh_node_url field on RoutingResult, MeshProvider class proposed, chat.py wire-in proposed) is **architecturally wrong** under this plan. The routing decision belongs at the edge, not inside the Lambda's classifier output. The mesh-first wire was a useful prototype but should be deprecated when this lands.

---

## 2. Lever set — what agents can manipulate

**Tier 1 — MUST-HAVE (lock in v0.1):**

| Lever | Type | Notes | Source |
|-------|------|-------|--------|
| `max_cost_cents` | int ≥0 | Hard cost ceiling per request | RouterArena |
| `max_latency_ms_p95` | int | Hard latency ceiling — **p95**, not avg (averages hide jitter; RouterBench finding) | RouterArena |
| `min_throughput_tps` | int | Min sustained tokens/sec under load | OpenRouter `sort:throughput` |
| `quality_weight` | float [0,1] | RouterBench-style λ; smooth pareto knob | RouterBench |
| `require_capabilities` | string[] | Hard gate: `tools`, `json_mode`, `vision`, `streaming`, `seed`, `system_prompt` | OpenRouter `require_parameters` |
| `allow_fallbacks` | bool | Fall through to cloud on mesh failure | LiteLLM |
| `fallback_strategy` | enum | `price` \| `quality` \| `latency` \| `none` | Portkey |

**Tier 2 — SHOULD-HAVE (privacy/compliance, cheap to add):**

| Lever | Type | Notes |
|-------|------|-------|
| `zdr` | bool | Zero-data-retention only |
| `region` | string | Data residency (`us`, `eu`, `lan`) |
| `service_tier` | enum | `auto` \| `flex` \| `standard` \| `priority` \| `batch` (bridges OpenAI/Anthropic) |
| `max_quantization_loss` | enum | `bf16` \| `fp8` \| `fp4` (capability gate, not score) |

**Tier 3 — NICE-TO-HAVE:**

| Lever | Type | Notes |
|-------|------|-------|
| `min_context_window` | int | Hard gate when known |
| `require_determinism` | bool | Seed + temp-stable models only |
| `language` | string | Hint for multilingual routing |
| `streaming_jitter_max_ms` | int | For voice/UX-critical streams |

**SKIP (intentional rejection):**
- Safety/refusal calibration as request lever — drifts adversarially, handle via guardrails
- Hallucination rate as request lever — too noisy, domain-dependent, fold into quality
- Cold-start as request param — provider concern, not agent concern
- Robustness-to-perturbation — research metric only, surface as model-card metadata

**Key insight (RouterArena synthesis)**: **no production router exposes multi-axis per-request preferences**. RouterBench has scalar λ, RouteLLM has scalar α — both single-knob cost-vs-quality. Multi-axis (speed × cost × quality × privacy × capability) at agent-controllable granularity is **novel territory**. That's the moat.

### API surfaces

```
# Shape 1: Headers (per-request, transport-neutral)
X-Slancha-Pref: max_cost_cents=5,quality_weight=0.6
X-Slancha-Capability: tools,json_mode

# Shape 2: Body (per-request, structured)
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

# Shape 3: Service-tier presets (named bundles for agents)
{"model": "auto", "service_tier": "flex"}       # cheap+slow
{"model": "auto", "service_tier": "priority"}   # fast+expensive
{"model": "auto", "service_tier": "quality"}    # quality_weight=0.8
{"model": "auto", "service_tier": "balanced"}   # default

# Shape 4: Explicit (override everything)
{"model": "voice-paul-v8"}                      # hard pin, skip pareto
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

```json
{
  "schema_version": "0.1",
  "node_id": "did:web:mesh.paul.laulpogan.com",
  "endpoint": "https://mesh.paul.laulpogan.com/v1",
  "heartbeat": {
    "interval_s": 30,
    "last_seen": "2026-05-22T18:14:00Z"
  },
  "specialists": [
    {
      "id": "voice-paul-v8",
      "family": "writing",
      "base_model": "hermes-3-3b",
      "model_hash": "sha256:abc...",
      "lora_adapter_hash": "sha256:def...",

      "capabilities": ["streaming", "system_prompt"],
      "modalities": {"in": ["text"], "out": ["text"]},
      "context_length": 32768,
      "supported_parameters": ["temperature", "top_p", "seed", "max_tokens"],

      "cost": {
        "unit": "1M_tokens",
        "currency": "USD",
        "input": 0.0,
        "output": 0.0,
        "amortized_hourly": 0.05,
        "cloud_equivalent_input": 3.00,
        "cloud_equivalent_output": 15.00
      },

      "latency": {
        "p50_ttfb_ms": 120,
        "p95_ttfb_ms": 280,
        "p99_ttfb_ms": 410,
        "tokens_per_second_p50": 50,
        "tokens_per_second_p95": 45,
        "cold_start_ms": null,
        "queue_depth": 0,
        "concurrency_current": 0,
        "concurrency_max": 4,
        "window_minutes": 5
      },

      "quality": {
        "router_observed": null,
        "node_self_reported": null,
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
        "node_location": "home_lan_us_east"
      }
    }
  ],

  "signature": {
    "alg": "EdDSA",
    "kid": "did:web:mesh.paul.laulpogan.com#key-1",
    "jws": "eyJhbGc..."
  },

  "extensions": {}
}
```

### Critical schema decisions (LOCKED for v0.1)

1. **`quality.router_observed`** is the trusted field. `quality.node_self_reported` is published but routers ignore by default. (Quality is router-observed, not node-claimed.)
2. **JCS canonicalization (RFC 8785)** before signing — deterministic representation for reproducible verification.
3. **JWS over JCS-canonicalized JSON** — signed via the `did:web` controller key.
4. **`extensions` block** uses reverse-DNS keys (`com.slancha.lora_registry`) for experimental fields. (OCI annotations pattern.)
5. **Per-endpoint versioning**: future-proofing route deprecation independent of card schema. (Matrix lesson.)
6. **OpenTelemetry GenAI attribute names** for any telemetry payloads (`gen_ai.request.model`, `gen_ai.usage.*`).

---

## 4. Trust model

| Layer | Mechanism | Why |
|-------|-----------|-----|
| Node identity | `did:web:<domain>` | Rides on existing DNS+TLS, no CA bureaucracy |
| Card signing | JWS over JCS-canonicalized JSON | Prevents registry injection + impersonation |
| Quality signals | Router-observed (canary probes) | Self-attested scores are gameable; observation is verifiable |
| Cost signals | Trust-by-default within user's own mesh | Cross-mesh = display cloud_equivalent only |
| Privacy attestations | Self-attested in v0.1; hardware attestation (TPM/SEV/TDX) in `extensions["privacy.attestation"]` | TEE formats churn too fast for core |
| Last-mile auth | CF Access service token + HMAC `X-Slancha-Forward-Sig` | Tunnel URL alone is not auth (Ollama 175k exposed hosts) |
| Model provenance | `model_hash` + Sigstore Rekor entry in `extensions["dev.sigstore.provenance"]` | Optional for high-stakes deployments |

**Explicitly skipped for v0.1**: W3C Verifiable Credentials. JSON-LD + DataIntegrityProofs add 2x impl cost. Promote to VC at 1.0 if ecosystem proves it needs delegation / selective disclosure / third-party endorsements.

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

```json
POST /v1/admin/usage
Authorization: Bearer <mesh-ingest-token>
X-Slancha-Forward-Sig: <HMAC>
Content-Type: application/json

{
  "request_id": "uuid-...",
  "user_id": "user-paul",
  "specialist_id": "voice-paul-v8",
  "endpoint": "/v1/chat/completions",
  "tokens_in": 230,
  "tokens_out": 1247,
  "latency_ms": 4830,
  "ttft_ms": 124,
  "tokens_per_second": 51.6,
  "cost_cents": 0,
  "cloud_equivalent_cost_cents": 7,
  "status_code": 200,
  "route_target": "mesh",
  "fallback_fired": false,
  "pref_applied": {"quality_weight": 0.6, "max_cost_cents": 5},
  "decision_reason": "voice-paul-v8: pareto-winner over sonnet-4-6 by 0.13 (quality match + cost)",
  "gen_ai.request.model": "voice-paul-v8",
  "gen_ai.usage.input_tokens": 230,
  "gen_ai.usage.output_tokens": 1247
}
```

**Critical contract guarantees:**
- **No prompt body, no completion body, ever.** Counts and metadata only.
- **OpenTelemetry GenAI semconv compliance** for the `gen_ai.*` fields.
- **Existing `log_usage` shape preserved** so dashboard works unchanged.
- **`decision_reason`** is the explainability field — visible to user in dashboard.

---

## 7. Anti-patterns we explicitly avoid

| Failure mode | Lesson source | How we avoid |
|--------------|---------------|--------------|
| Lambda in mesh hot path billing for SSE duration | AWS billing model | L@E origin-request mutates metadata only |
| `MeshProvider` as just-another-OpenAI-provider hiding mesh state | prior `claude/mesh-wire-in` design critique | Routing decision at edge, not at provider layer |
| Self-attested quality scores trusted by routers | RouterBench + protocol research | Router-observed canary probes; node claims = display-only |
| Tunnel-as-auth (Ollama 175k exposed hosts) | SentinelOne/Censys early 2026 | CF Access service token + HMAC mandatory |
| Flat preference dict can't express composition | Portkey/Bifrost precedent | Nestable `pref` + `fallback_strategy` |
| JSON-LD complexity tax | ActivityPub adoption pain | Plain JSON + JSON Schema 2020-12 |
| Breaking changes within 0.x | MCP 2025 churn | Public no-break commitment; experiments in `extensions` |
| L@E origin-response trigger buffers SSE | AWS L@E response-streaming gap | Forbidden by spec; documented as footgun |
| Decentralized from day one with no gravity well | OStatus/Diaspora failure | Centralized registry primary |
| Spec without reference impl | many IETF drafts | AGPL ref impl + conformance suite ship with v0.1 |
| Cost-attribution drift on silent fallback | Vercel BYOK silent fallback | `fallback_fired` flag in telemetry, dashboard alert |
| SSRF via attacker-supplied mesh_url | Portkey RFC1918 block | mesh_url validated against registered allowlist |
| User changes preferences mid-conversation; session pin overrides | classifier/router.py session manager | `pref` overrides session pin when explicit |

---

## 8. Phased rollout

| Phase | Tasks | Outcome | Estimate |
|-------|-------|---------|----------|
| 1 — LAN-direct voice v8 | #50, #51, #53 | voice-paul-v8 servable via tunnel | 2-4 hrs |
| 2 — Telemetry sidecar | #54-#57 | dashboard captures mesh-direct usage | 4-8 hrs |
| 3 — Edge routing | #49, #58-#60 | api.slancha.ai → mesh transparently | 1-3 days |
| 4 — Protocol v0.1 spec | #61-#65 | spec frozen + AGPL + conformance suite | 3-5 days |
| 5 — Multi-axis levers | #66-#70 | agent-controllable per-request prefs | 2-4 days |
| 6 — Quality observability | #71-#73 | router-observed quality + drift alerts | 2-3 days |
| 7 — Discovery v0.1 complete | #74-#75 | DNS-SD + public registry MVP | 1-2 days |

**Total: ~3 weeks if sequential, parallelizable across mac/spark/api workstreams.**
**Phase 1 + 2 alone (~10 hrs) ship voice-paul-v8 end-to-end with telemetry.**

---

## 9. Open questions for mac (decide before locking)

1. **Quality verification: who runs the canary probes?** Slancha-api hosting them = central trust; per-mesh-pair = O(N²) but decentralized. *Recommend*: central probe service in v0.1, federate at 1.0.
2. **Cost-claim verification across meshes**: trust YOUR mesh, distrust cross-mesh? *Recommend*: always display cloud_equivalent for cross-mesh, native cost for own-mesh.
3. **Strategy tree expressivity**: 1-level deep in v0.1 (single fallback strategy)? CEL/nested at 1.0? *Recommend*: yes.
4. **Anthropic-format `/v1/messages`**: support in mesh too, or OpenAI-compat only? *Recommend*: OpenAI-compat only in v0.1; translator at SaaS layer if needed.
5. **Quality scoring under pref divergence**: same model scores differently when used via different pref vectors (selection bias). *Recommend*: bucket quality observations by pref bucket; surface as separate scores.
6. **Multi-mesh load-balancing**: paul has Spark AND Mac, both serve voice-paul-v8. Who picks? *Recommend*: registry-side LB on `concurrency_current`; paul's mesh handles internally.
7. **Token rotation cadence**: CF Access service tokens, HMAC keys, did:web key rotation. *Recommend*: monthly auto-rotation with 7-day grace; document in runbook.
8. **Agent-mediated agent calls**: agent calls mesh, mesh internally calls back through SaaS for a sub-agent task. Loop detection? *Recommend*: `X-Slancha-Hop-Count` header, max 5.

---

## 10. Task list (final form, 23 new + 3 reframed)

| # | Phase | Owner | Task |
|---|-------|-------|------|
| 50 | 1 | SPARK | Register voice-paul-v8 as mesh specialist |
| 51 | 1 | SPARK | CF tunnel mesh.paul.laulpogan.com → slancha-local |
| 53 | 1 | SPARK | CF Access service-token policy on mesh tunnel |
| 54 | 2 | SPARK | slancha-local: HMAC verify middleware + service token gate |
| 55 | 2 | SPARK | slancha-local: after-response telemetry sidecar POST |
| 56 | 2 | API | POST /v1/admin/usage ingestion endpoint |
| 57 | 2 | MESH | Cross-repo round-trip test for usage payload |
| 49 | 3 | INFRA | CloudFront KVS: user→route config + paul seeded |
| 58 | 3 | INFRA | CloudFront Function (viewer-request): KVS lookup + header injection |
| 59 | 3 | INFRA | Lambda@Edge (origin-request): SSRF + health + HMAC + origin rewrite |
| 60 | 3 | INFRA | Firehose + S3 + Athena table for L@E decision telemetry |
| 61 | 4 | MESH | SpecialistCard JSON Schema 2020-12 + reference doc |
| 62 | 4 | MESH | did:web resolution + JCS canonicalization + JWS sign/verify |
| 63 | 4 | MESH | slancha-conformance CLI + test suite |
| 64 | 4 | SPARK | .well-known/slancha-card.json endpoint on slancha-local |
| 65 | 4 | MESH | Public spec repo + AGPL + SEP process scaffold |
| 66 | 5 | API | X-Slancha-Pref header + body pref field parsing |
| 67 | 5 | API | Service tier presets bridging OpenAI/Anthropic |
| 68 | 5 | MESH | Pareto-frontier scoring in POST /place w/ cold-start + queue penalties |
| 69 | 5 | MESH | GET /v1/models?include=routing_meta discovery API |
| 70 | 5 | MESH | decision_reason explainability in telemetry |
| 71 | 6 | SLOW-LOOP | Continuous eval pipeline: nightly score → card.quality.router_observed |
| 72 | 6 | INFRA | Router-observed quality probe service |
| 73 | 6 | MESH | Quality drift alerts + dashboard panel |
| 74 | 7 | SPARK | DNS-SD _slancha._tcp service registration on slancha-local |
| 75 | 7 | INFRA | registry.slancha.dev MVP (read-only) |

Plus pending non-protocol work: **#45** (v0.0.7 live cluster smoke).

---

## 11. The moat phrased cleanly

**Closed routers** (OpenRouter, Vercel AI Gateway, Cloudflare AI Gateway): hide upstream details, force trust in their selection, can't credibly emit hardware-specific cost / per-(model, domain) quality / privacy attestations from infrastructure they don't own.

**Open mesh protocol**: every node publishes signed signals across 5 axes (cost, latency, quality, capability, privacy); user owns the preference function; routing is parametric and explainable; SaaS layer surfaces the controls but doesn't gatekeep the math.

**Routing transparency as a product feature**, not a backend optimization. A user can see exactly why their request went to `voice-paul-v8` vs cloud, and override at any axis. That's a thing closed routers structurally can't offer.

AGPL'd protocol spec + reference implementation in `slancha-mesh` becomes the open standard. Slancha-the-SaaS sells the polished dashboard + multi-user management on top. Self-hosters run pure mesh + their own preference engine. Same protocol.

---

## 12. Request to mac

1. **Triple-check** the architecture diagram (§1) — is L@E origin-request the right edge primitive? Anything we're missing about CloudFront SSE behavior under load?
2. **Triple-check** the lever set (§2) — anything from your knowledge of RouterArena / Martian / Bifrost we should add or cut?
3. **Triple-check** the card schema (§3) — extension namespacing? Signature scheme? `quality.router_observed` vs `quality.node_self_reported` split right?
4. **Open questions (§9)** — recommend answers or push back.
5. **Phasing (§8)** — is the order right? Phase 4 (spec) before Phase 5 (levers), or interleave?
6. **Anti-patterns (§7)** — anything we're walking into that the table doesn't catch?

Annotate inline in this doc on the `claude/protocol-v0.1-draft` branch; comment in PR if you spin one up; or reply via wire with section-by-section.

Once we converge, this supersedes `SLANCHA_MESH_V0_SPEC.md` and we cut a clean `v0.1.0-spec` tag against the consolidated text. Then we execute the 25 tasks.
