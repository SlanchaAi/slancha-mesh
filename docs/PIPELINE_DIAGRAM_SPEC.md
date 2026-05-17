# Slancha Pipeline Diagram — Design Spec

**Status:** v1 — recirculating-loop semantic. Living doc.
**Authoring:** mac (spec + iteration), spark (initial v0 sketch handed off in slancha-test).
**Audience:** operator, Stripe partner-pitch live screen, future-us after compaction.
**Reference:** Go Pro Plumbing "Recirculating Hot Water System Diagram" + TensorZero observability/retrain loop semantic. The diagram is **the recirculating pump itself** — prompt enters, response/labels accumulate, FT retrains, weights swap back into the proxy, loop closes.

## 1 — What this diagram is for

One-page, glanceable answer to: **"What does Slancha actually do, and how does it self-improve?"** — for the Stripe partner-pitch (live screen during meeting), an operator catching up, or future-us after a context compact.

Not a docs-quality architecture diagram. Not a Mermaid flowchart. Not a metrics dashboard — `mesh/dashboard/` already does metrics. This is the **closed-loop flow** view, pulsing with live activity, with visible "redeploy events" when classifier weights swap.

The diagram must communicate **two simultaneous loops** in one frame:

- **FAST loop (classifier):** minutes-scale. ORACLE → mmbert-head FT → atomic-swap weights into PROXY. Demo headliner — partner sees classifier accuracy climb live.
- **SLOW loop (generator):** hours-scale. CAPTURE → QLoRA on (prompt, response, judge_score) triples → adapter deploy to mesh vllm. Background telemetry.

Operator/pitch viewer should grok both at-a-glance.

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
| `ft_fast` | **Classifier FT** — mmbert-head retrain (live) | rounded rect | far-right, upper | FAST loop. Streams oracle labels, retrains heads continuously, emits new weights | Glow continuously during training; spike + label "REDEPLOY" on weight emit |
| `ft_slow` | **Generator FT** — QLoRA (batch) | rounded rect, dashed border | far-right, lower | SLOW loop. Batches (prompt, response, judge_score) triples, runs QLoRA on accumulated data | Dashed/idle between batches; solid + glow when training; rare REDEPLOY spike |
| `registry` | **Model Registry** — atomic-swap weights | small rect, between FT and proxy | Hot-swap target. Receives new weights from either FT loop, signals proxy to reload | Throbbing pulse on each REDEPLOY event |

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

### Recirculation (violet — the pump — load-bearing semantic, this is the headline)

Color: `#a855f7` (purple-500). Width: `2.5px`. Style: solid. Animation: dashed-array scrolling `3s linear infinite` (slightly slower than supply/return so it reads as a different phenomenon).

Route (the pump that closes the loop):
```
oracle  → ft_fast  → registry → proxy   (FAST loop — classifier weights)
capture → ft_slow  → registry → proxy   (SLOW loop — generator weights, dashed-style line for in-batch state)
```

**REDEPLOY events** — when `registry → proxy` fires (atomic weight swap), the registry node *throbs* (scale pulse 1.0 → 1.08 → 1.0 over 0.6s, ease-in-out) AND a brighter violet pulse propagates along the `registry → proxy` segment. Visible label "REDEPLOY" appears for 1.5s. This is the demo headline.

Two-loop visual distinction:
- **FAST loop:** solid violet, 3s dash period, REDEPLOY label `↻ classifier v{N+1}` (frequent — every few minutes during demo)
- **SLOW loop:** dashed violet `8 6`, 6s dash period, REDEPLOY label `↻ generator v{N+1}` (rare — once per hour-ish)

### Control (slate — config / monitoring only; not load-bearing semantic)

Color: `#64748b` (slate-500). Width: `1.5px`. Style: dashed `4 4`. No animation.

Route:
```
capture ⇢ dashboard  (if dashboard node present)
```

(Note: the FT-feedback edges are NOT control — they're recirculation, the whole point of the loop. Use violet.)

## 4 — Color palette (locked)

| Token | Hex | Use |
|-------|-----|-----|
| `--supply` | `#f97316` | warm supply lines, prompt-forward path |
| `--return` | `#06b6d4` | cool return lines, response-back path |
| `--recirculate` | `#a855f7` | the pump — FT → registry → proxy weight-swap channel (headline) |
| `--redeploy-glow` | `#d8b4fe` | brighter violet for REDEPLOY event pulse |
| `--control` | `#64748b` | dashed control/monitoring links (non-load-bearing) |
| `--idle` | `#94a3b8` | components not active in current state |
| `--error` | `#ef4444` | error pulse, probe-block, oracle-failure flag |
| `--success` | `#22c55e` | "labels accumulating" / "loop closing successfully" |
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
| Supply flow | `.flow-supply` | `stroke-dasharray: 8 6; stroke-dashoffset` animates 14 → 0 over 2s linear infinite | Always (or rate ∝ live req/s) |
| Return flow | `.flow-return` | mirror of supply, reversed direction | Always |
| Recirculate (FAST) | `.flow-recirc-fast` | `stroke-dasharray: 8 6; stroke-dashoffset` animates 14 → 0 over 3s linear infinite | Always when classifier FT running |
| Recirculate (SLOW) | `.flow-recirc-slow` | `stroke-dasharray: 12 8` (longer dashes); animates over 6s linear infinite | Always when generator FT running; idle-styled between batches |
| Pulse glow | `.pulse-active` | `box-shadow` pulses between component color and white over 1.2s ease-in-out infinite | Component currently processing |
| REDEPLOY event | `.redeploy-throb` | `transform: scale(1) → scale(1.08) → scale(1)` over 0.6s ease-in-out; concurrent `--redeploy-glow` halo over 1.5s. Label `REDEPLOY ↻ classifier v{N+1}` (or generator) fades in/out alongside. | Triggered on weight-swap event. Demo headline. |
| Loop-closed flash | `.loop-closed` | green `--success` ring around the entire diagram for 0.3s | When a full FAST loop completes (oracle label → FT → redeploy → routing improvement detected) — rare, ceremonial |
| Error flash | `.pulse-error` | `box-shadow` flashes `--error` over 0.5s, 3 times | On error event |
| Idle | `.idle` | filter: grayscale(0.6) opacity(0.6) | Component not yet active |

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
3. **Violet solid line + label `↻ Classifier retrain (FAST loop)`**
4. **Violet dashed line + label `↻ Generator retrain (SLOW loop)`**
5. Dashed slate line + label `Control / monitoring`
6. Red filled circle + label `Error / block`
7. Green ring + label `Loop closed (full cycle)`
8. Grey/desaturated box + label `Idle component`
9. White-on-orange chip showing example `847/min` + label `Live throughput`

The two violet entries are the headline — make them visually prominent in the legend so a pitch viewer reads "this thing self-improves" without us saying it.

## 10 — Layout (the pump — recirculating closed loop)

Topology priority: the closed-loop must be visually obvious. The supply path goes left-to-right across the top; the return path comes back right-to-left in the middle; the recirculation (FT → registry → proxy) loops UP from the bottom-right back to the proxy at the top, forming a clear "pump" silhouette.

```
       SUPPLY (warm) ────────────────────────────────────────>
       ┌─────────┐   ┌──────────┐   ┌─────┐   ┌─────────┐
       │ Corpus  │──>│Classifier│──>│Probe│──>│ Proxy   │─┐
       │ 100K    │   │ mmbert-6h│   │     │   │ :8766   │ │
       └─────────┘   └──────────┘   └─────┘   └─────────┘ │
                          ^                              v
                          │                       ┌──────────────┐
                          │                       │ Local Models │
                          │                       │ 5× ollama    │─┐
                       (recirc                    └──────────────┘ │
                        target)                   ┌──────────────┐ │
                          │                       │ Cloud OSS    │─┤
                          │                       └──────────────┘ │
                          │                                        v
                          │                              <───── RETURN (cool) ──
                          │                         ┌─────────┐
                          │                         │ Capture │
                          │                         │ ledger  │
                          │                         └─────────┘
                          │                              │  │
                          │                       ──────-┘  └─────
                          │                       v                v
                          │                  ┌────────┐       ┌────────┐
                          │     RECIRC FAST  │ Oracle │       │ ft_slow│
                          │     (violet)     │ Judge  │       │ QLoRA  │
                          │                  │ 472e   │       │ (batch)│
                          │                  └────────┘       └────────┘
                          │                       │                │
                          │                       v                │
                          │                  ┌────────┐            │
                          │                  │ft_fast │            │
                          │                  │mmbert  │            │
                          │                  │(live)  │            │
                          │                  └────────┘            │
                          │                       │                │
                          │     RECIRC SLOW       │                │
                          │     (dashed violet)   v                v
                          │                  ┌─────────────────────┐
                          │                  │  Model Registry     │
                          │                  │  (atomic swap)      │
                          │                  └─────────────────────┘
                          └─────── RECIRC (THE PUMP) ────┘
                                      ↻ REDEPLOY
| ───────────────────────────────────────────────────────────────────  |
| LEGEND: ─── supply | ─── return | ━━ recirc-fast | ┅┅ recirc-slow |
|         ┄┄ control | ● error | ○ loop-closed | dim idle              |
```

The loop closure (recirc → registry → proxy) is the headline. At the apex of the recirc curve, place the `REDEPLOY ↻` event label.

(ASCII is approximate. Spark's v0 SVG/HTML is the real reference; mac iterates from there.)

## 11 — Accessibility checklist

- All component labels in DOM (not text-as-image) so screenreader can navigate.
- ARIA roles: `<g role="figure" aria-label="...">` per component.
- Color contrast: foreground vs `--bg-card` ≥ 7:1 (AAA).
- Flow lines have associated `<title>` elements explaining direction in plain English.
- `prefers-reduced-motion` honored (per §7).
- Don't rely on color alone — pair supply/return colors with arrow direction + line position (warm above, cool below where layout allows).

## 12 — Stripe partner-pitch live-screen mode

This diagram doubles as the live screen during the Stripe pitch. Implications:

- **Default to dark mode.** Looks better on projector + matches conference room lighting.
- **REDEPLOY events should be visible from the back of a meeting room.** Glow halo, scale throb, label fade-in — all generous, not subtle.
- **Resilient to no-data:** if spark's stats poller (`scripts/pipeline_stats.py`) is down, default to a *demo loop* — staged synthetic stats playing on a 30s cycle so the diagram never "freezes" on stage. Toggle via `?demo=1` URL param.
- **Loud "↻" symbol** somewhere in the title/header — communicates "this thing recirculates" without needing a caption.
- **Numbers should ground the abstraction:** show actual req/s, label count, model-mix histogram. Pitch viewer believes a thing is real when there are real numbers attached.

## 13 — Out of scope for v0

- Interactive drilldown (click a model → see model card). Future v2.
- Cost overlay (per-component cost burndown). Future once meter shim flows real Stripe events.
- Multi-node visualization (when there's >1 spark host routing). Future once cluster reservations land.
- 3D / isometric variations. Plumbing-clean = 2D.
- Audio cues. Hard no.

## 14 — Mac iteration checkpoints

Spark hands off v0 sketch at `slancha-test/dashboard/pipeline.html`. Mac takes over. On each iteration commit, verify against:

1. Component count matches §2 (primary nodes including `ft_fast`, `ft_slow`, `registry`).
2. **Recirculating-loop topology is visually obvious** — eye traces the pump path naturally.
3. Three-channel palette (supply / return / recirculate) matches §4 hex codes.
4. **REDEPLOY event animation is generous** (back-of-room visibility — §12).
5. Two-loop distinction is readable (FAST = solid violet, SLOW = dashed violet).
6. Live data shape matches §8 (consumes `dashboard/stats.json` written by spark's `scripts/pipeline_stats.py`).
7. `?demo=1` mode for stage-resilient pitch playback.
8. Legend present at bottom (§9), violet entries prominent.
9. Reduced-motion fallback (§7).
10. No layout overlap at 1280×720 viewport.

Mac commits iterations to `slancha-test/dashboard/pipeline.html`. Spec stays in slancha-mesh; cross-repo OK since spec is reference, HTML is artifact.

— spec by mac (claude-opus-4-7), 2026-05-17. v1 = recirculating-loop semantic (operator clarification + TensorZero framing).
