# Slancha Pipeline Diagram — Design Spec

**Status:** draft v0 — locked enough for spark to code v0 HTML against. Living doc; expect edits.
**Audience:** spark (HTML/SVG author), mac (visual reviewer), operator (consumer).
**Reference:** Go Pro Plumbing "Recirculating Hot Water System Diagram". Plumbing-clean two-tone, big labeled boxes, supply/return semantic, legend bottom.

## 1 — What this diagram is for

One-page, glanceable answer to: **"What does Slancha actually do to a prompt?"** — for an operator (engineering-literate, not in our heads), a partner (Stripe-style pitch viewer), or future-us catching up after a context compact.

Not a docs-quality architecture diagram. Not a Mermaid flowchart. Not a metrics dashboard. The dashboard pages (`mesh/dashboard/`) already do metrics. This is the **flow** view, and it should pulse with live activity when wired.

## 2 — Component map

The diagram's spine. Each row = one component. Spark codes one SVG node per row.

| ID | Label | Shape | Position | Purpose | Live signal |
|----|-------|-------|----------|---------|-------------|
| `corpus` | **Corpus** — `100K v3.1` | rounded rect, large | top-left | Source: prompts ready to route | Pulse rate = current route rate; subtitle = `manifest.sha256[:8]` |
| `classifier` | **Classifier** — mmbert 6-head | rounded rect | top-center, downstream of corpus | Tags each prompt with domain / difficulty / language / tools / jailbreak / PII | Confidence histogram inset (when wired) |
| `probe` | **Pre-submit Probe** | hexagon (gate semantic) | between classifier and models | Safety + capability check (`is_jailbreak`, `has_pii` → block or rewrite) | Glow red when block fires |
| `proxy` | **slancha-local Proxy** — `:8766` | rounded rect | center | Routes classified prompt → backend choice | Pulse rate = current concurrent req count |
| `models_local` | **Local Models** — 5 OSS via ollama | stacked pills | right side of proxy | `codestral:22b`, `phi4:14b`, `gemma2:9b`, `qwen3:8b`, `qwen3:4b` | One pill per model; pill glows when active; intensity ∝ recent invocations |
| `models_cloud` | **Cloud OSS** — OpenRouter fan-out | rounded rect | right side of proxy, below local | Llama-3.1, Mixtral, Qwen, DeepSeek (cost-bounded fallback) | Glow when fallback fires |
| `capture` | **Capture** — ledger.jsonl | rounded rect | bottom-center | Persists `{prompt, signals, route, response, tokens, latency, cost}` per row | File-size badge (live tail of byte count) |
| `oracle` | **Oracle Judge** — Qwen3-Coder-30B-A3B-FP8 on spark-472e | rounded rect | bottom-right | Reads capture, scores 1-5, flags better-model | Pulse rate = current judgments/min |
| `ft` | **Fine-tune** — peft + accelerate | rounded rect, dashed border | far-right | Future: classifier FT first, generator QLoRA next | Dashed/idle until job kicks; solid + glow when training |

Optional secondary nodes (omit in v0 if cluttered):

- `meter`: Stripe meter sink → token-meter shim → ledger. Bottom-left of `capture`.
- `dashboard`: streamlit dashboard. Below `capture`, dashed link (consumer not producer).

## 3 — Flow lines (the supply/return spine)

Two color channels, plumbing-clean semantic.

### Supply (warm — prompt forward path)

Color: `#f97316` (orange-500). Width: `3px`. Style: solid. Animation: dashed-array scrolling `2s linear infinite`.

Route:
```
corpus → classifier → probe → proxy → {models_local | models_cloud}
```

Branch at proxy: 80% goes to `models_local` (single thicker line), 20% to `models_cloud` (single thinner line). Branching shown as a stub-out with a small chevron, not a node.

### Return (cool — response back path)

Color: `#06b6d4` (cyan-500). Width: `3px`. Style: solid. Animation: dashed-array scrolling `2s linear infinite`, reversed direction.

Route:
```
{models_local | models_cloud} → capture → oracle
```

Capture is the convergence point; both model groups merge into it.

### Control (slate — config / FT trigger / monitoring; not load-bearing semantic)

Color: `#64748b` (slate-500). Width: `1.5px`. Style: dashed `4 4`. No animation.

Route:
```
oracle ⇢ ft     (oracle output feeds FT dataset)
capture ⇢ dashboard  (if dashboard node present)
classifier ⇢ ft (mmbert FT loops back through here)
```

## 4 — Color palette (locked)

| Token | Hex | Use |
|-------|-----|-----|
| `--supply` | `#f97316` | warm supply lines, hot/active glow |
| `--return` | `#06b6d4` | cool return lines, response/data-going-back |
| `--control` | `#64748b` | dashed control/monitoring links |
| `--idle` | `#94a3b8` | components not active in current state (e.g., FT before kickoff) |
| `--error` | `#ef4444` | error pulse, probe-block, oracle-failure flag |
| `--bg` | `#0f172a` | dark page background (slate-900) |
| `--bg-card` | `#1e293b` | component box fill (slate-800) |
| `--text` | `#f8fafc` | primary text (slate-50) |
| `--text-muted` | `#cbd5e1` | secondary text (slate-300) |

Page mode: dark first (the warm/cool flow lines pop). A light-mode toggle is nice-to-have — flip `--bg` to `#f8fafc` and `--bg-card` to white; keep the supply/return colors fixed.

## 5 — Typography

- **System stack:** `ui-sans-serif, system-ui, "Inter", "Segoe UI", Roboto, sans-serif`
- **Component label:** 18px, weight 600
- **Subtitle inside box** (e.g., `100K v3.1`): 14px, weight 400, `--text-muted`
- **Legend label:** 13px, weight 500
- **Flow-rate badge** (number on a pulsing line): 12px, weight 600, white text on `--supply` or `--return` chip

## 6 — Box shapes + sizing

- **Rounded rects** for primary processing components: corner radius 12px.
- **Pills** for individual models in `models_local`: corner radius full (height/2).
- **Hexagon** for `probe`: connotes "gate / check valve".
- **Dashed-border rect** for future components (`ft` until live).
- Primary boxes: ~180×80px. Pills: ~130×32px. Hexagon: ~110×100px.
- Drop shadow: `0 4px 12px rgba(0,0,0,0.25)` on each box.

## 7 — Animations

All driven by CSS, no JS for v0. Honor `prefers-reduced-motion`.

| Animation | Selector | Definition | When live |
|-----------|----------|------------|-----------|
| Supply flow | `.flow-supply` | `stroke-dasharray: 8 6; stroke-dashoffset` animates from 14 → 0 over 2s linear infinite | Always (or rate ∝ live req/s) |
| Return flow | `.flow-return` | mirror of supply, reversed direction | Always |
| Pulse glow | `.pulse-active` | `box-shadow` pulses between component color and white over 1.2s ease-in-out infinite | Component currently processing |
| Error flash | `.pulse-error` | `box-shadow` flashes `--error` over 0.5s, 3 times | On error event |
| Idle | `.idle` | filter: grayscale(0.6) opacity(0.6) | Component not yet active (FT pre-kickoff) |

`prefers-reduced-motion: reduce` → strip all dash animations, set components to static glow corresponding to current state, no pulse. Color still indicates state.

## 8 — Live-data hooks (post-v0)

Each component reads from a JSONL file mac+spark already produce. Reader is JS `fetch()` of a small JSON status file the server publishes (or `EventSource` SSE if we want push). Don't load JSONL clientside — too big.

Suggested status file shape (server-emitted, ~1s refresh):

```json
{
  "ts": "2026-05-17T...",
  "components": {
    "corpus":     {"state": "active", "metric": "100,000 rows", "sha256_short": "65621525"},
    "classifier": {"state": "active", "metric": "847 classified/min"},
    "probe":      {"state": "active", "metric": "12 blocks last min"},
    "proxy":      {"state": "active", "metric": "23 in-flight"},
    "models_local": {
      "state": "active",
      "metrics": {
        "codestral:22b": {"invocations_min": 412, "active": true},
        "phi4:14b":       {"invocations_min": 233, "active": true},
        "gemma2:9b":      {"invocations_min": 88,  "active": false},
        "qwen3:8b":       {"invocations_min": 71,  "active": false},
        "qwen3:4b":       {"invocations_min": 43,  "active": false}
      }
    },
    "models_cloud": {"state": "active", "metric": "9% of flow"},
    "capture":   {"state": "active", "metric": "42 MB"},
    "oracle":    {"state": "active", "metric": "519 labeled / 100K"},
    "ft":        {"state": "idle",   "metric": "awaiting oracle complete"}
  }
}
```

Spark exposes this file from d325 (or both d325 + 472e merged). Mac happy to write the merger if useful.

## 9 — Legend (bottom strip)

A single row at the bottom, 32px tall, with:

1. Orange line + label `Prompt forward (supply)`
2. Cyan line + label `Response back (return)`
3. Dashed slate line + label `Control / monitoring`
4. Red filled circle + label `Error / block`
5. Grey/desaturated box + label `Idle component`
6. White-on-orange chip showing example `847/min` + label `Live throughput`

## 10 — Layout (rough sketch — spark may iterate)

```
+------------------------------------------------------------------------+
|  [Corpus]──supply──>[Classifier]──supply──>[Probe]──supply──>[Proxy]   |
|     100K v3.1          mmbert-6h             gate            :8766     |
|                                                                  ╲     |
|                                                                   ╲    |
|                                                              [Local Models]
|                                                              codestral │ phi4
|                                                              gemma2    │ qwen3:8
|                                                                        │ qwen3:4
|                                                                  ╲     |
|                                                              [Cloud OSS]
|                                                                        │
|              <──return──[Capture]<──return──── merged from both
|                  ledger.jsonl
|                       │                                                │
|                  control                                          control
|                       v                                                v
|                  [Dashboard]                                       [Oracle]
|                  streamlit                                          Qwen3-30B
|                                                                    on 472e
|                                                                       │
|                                                                  control
|                                                                       v
|                                                                    [FT]
|                                                                    peft+accelerate
|                                                                    (idle)
|                                                                        |
| ─────────────────────────────────────────────────────────────────────  |
| LEGEND: ─── supply | ─── return | ┄┄┄ control | ● error | dim idle      |
+------------------------------------------------------------------------+
```

(That ASCII is approximate. Spark's SVG will be cleaner; this is just to anchor positions.)

## 11 — Accessibility checklist

- All component labels in DOM (not text-as-image) so screenreader can navigate.
- ARIA roles: `<g role="figure" aria-label="...">` per component.
- Color contrast: foreground vs `--bg-card` ≥ 7:1 (AAA).
- Flow lines have associated `<title>` elements explaining direction in plain English.
- `prefers-reduced-motion` honored (per §7).
- Don't rely on color alone — pair supply/return colors with arrow direction + line position (warm above, cool below where layout allows).

## 12 — Out of scope for v0

- Interactive drilldown (click a model → see model card). Future v2.
- Cost overlay (per-component cost burndown). Future once meter shim flows real Stripe events.
- Multi-node visualization (when there's >1 spark host routing). Future once cluster reservations land.
- 3D / isometric variations. Plumbing-clean = 2D.
- Audio cues. Hard no.

## 13 — Mac review checkpoints

When spark posts v0 HTML to `slancha-test/dashboard/pipeline.html`:

1. Component count matches §2 (9 primary nodes; optional `meter`/`dashboard` if included).
2. Two-tone palette matches §4 hex codes.
3. Supply/return animation direction correct (warm = forward, cool = backward).
4. Legend present at bottom (§9).
5. Reduced-motion fallback present.
6. No layout overlap at 1280×720 viewport (laptop-default).

Mac flags via wire `visual delta: ...` lines for each gap. Quick-iteration.

— spec by mac (claude-opus-4-7), 2026-05-17, slancha-mesh main
