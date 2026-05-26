# AGENTS.md — slancha-mesh

> Two audiences below. **(1) Setting up a node** (provisioning this box as a
> mesh specialist) — read this section. **(2) Editing this codebase** — read
> the GitNexus section further down. Humans: `README.md`, `JAMES_NODE_SETUP.md`.
> Deep design: `docs/AGENT_ONBOARDING_CHAIN_2026_05_25.md`.

## Setting up this machine as a mesh node

slancha-mesh turns a box into a **specialist inference node** on a private
Tailscale/Headscale tailnet. A router (the cloud gateway, or a local
`slancha-local` instance) **pull-discovers** nodes by walking the tailnet and
fetching each node's `/models` — there is **no central registry to push to and
no shared token to carry**. Tailnet membership + the `tag:specialist` ACL is
the credential.

### The setup chain (run in order)

```bash
pip install -e .                    # install (NOT yet on PyPI) → `slancha-mesh` command
slancha-mesh plan --json            # 1. ask the box what it should run (machine-readable)
slancha-mesh up --specialist <id> --key tskey-...   # 2. act: first join tags+serves+exposes discovery
slancha-mesh doctor --json          # 3. verify node readiness (tag, engine, router, ports)
slancha-mesh discover               #    from a gateway/admin box: confirm the node is routable
```

`up` is **idempotent** (safe to re-run); drop `--key` after the first join;
use `--auto` to let the box pick the best-fit catalog specialist; `--dry-run`
to preview. The agent loop is **plan → up → doctor**: `plan` decides, `up`
acts, `doctor` verifies (each is `--json`-capable with actionable `fix`/`next_steps`).

**First-node home mesh (no router yet):** a specialist node can't *route*. If
`plan` reports `mesh.router_present=false`, add `--with-router` to install +
launch a local router (`slancha-local serve`) — ASK THE HUMAN FIRST (heavy
install):
```bash
slancha-mesh up --auto --with-router --key tskey-...
```

### Reading `plan --json` (your decision inputs)

| Field | Use it to |
|---|---|
| `recommended_engine.{backend,quant,installed}` | Pick/INSTALL the right engine: MLX on Apple Silicon, Ollama/llama.cpp on GB10/CPU, vLLM on discrete NVIDIA ≥24 GB. |
| `recommended_specialist` / `alternate_specialists` | Catalog model(s) that fit. `null` ⇒ add a card matching the engine. |
| `mesh.state` | `first_node` (you are establishing the mesh — a router must run somewhere) vs `joining_existing`. |
| `mesh.router_present` | If false and `first_node`, a `slancha-local` router still needs to run to serve traffic. |
| `next_steps[]` | Machine-readable actions (`install_backend` / `run` / `note`) — act on these. |

### Judgment boundaries

**NEVER** hand-roll `tailscale up` without `--advertise-tags=tag:specialist`
(`up` sets it) · bind a model server to `127.0.0.1` when on the mesh (gateway
is off-box) · expose model/node-info ports publicly (the tailnet ACL is the
only path in) · set `quality_router_observed` or invent benchmark numbers ·
publish to PyPI / push / open PRs unless asked.

**ASK THE HUMAN FIRST** before installing a heavy engine (vLLM/CUDA) ·
downloading multi-GB weights · minting/handling a Tailscale auth key.

**ALWAYS** run `slancha-mesh plan --json` before acting · re-verify with
`status`/`discover` before declaring done · prefer `--dry-run`.

### Deployment prerequisites (verify; surface if missing)

- ACL grants `tag:gateway -> tag:specialist:8003,8004,8088` (8003/8004 model
  ports, **8088 node-info pull port** — without it the gateway can't discover).
- `tagOwners` restricts who may advertise `tag:specialist` (server-enforced; a
  misconfig opens the specialist pool to any device).

### Commands

| Command | Purpose | Machine-readable |
|---|---|---|
| `slancha-mesh plan [--json]` | Recommend engine + specialist; report mesh state | `--json` |
| `slancha-mesh up [--auto] [--specialist X] [--key K] [--with-router] [--dry-run]` | Join + serve + expose discovery; optionally bootstrap a router | exit code |
| `slancha-mesh doctor [--json]` | Diagnose readiness: tag, engine, router, ports (fix hints) | `--json`, exit 1 on fail |
| `slancha-mesh status` | Tailnet identity + specialist-readiness | — |
| `slancha-mesh discover [--json]` | Walk tailnet → routing table (host-pinned) | `--json` |

Tests: `uv run python -m pytest -q`.

---

<!-- gitnexus:start -->
# GitNexus — Code Intelligence

This project is indexed by GitNexus as **slancha-mesh** (2055 symbols, 5933 relationships, 171 execution flows). Use the GitNexus MCP tools to understand code, assess impact, and navigate safely.

> If any GitNexus tool warns the index is stale, run `npx gitnexus analyze` in terminal first.

## Always Do

- **MUST run impact analysis before editing any symbol.** Before modifying a function, class, or method, run `gitnexus_impact({target: "symbolName", direction: "upstream"})` and report the blast radius (direct callers, affected processes, risk level) to the user.
- **MUST run `gitnexus_detect_changes()` before committing** to verify your changes only affect expected symbols and execution flows.
- **MUST warn the user** if impact analysis returns HIGH or CRITICAL risk before proceeding with edits.
- When exploring unfamiliar code, use `gitnexus_query({query: "concept"})` to find execution flows instead of grepping. It returns process-grouped results ranked by relevance.
- When you need full context on a specific symbol — callers, callees, which execution flows it participates in — use `gitnexus_context({name: "symbolName"})`.

## When Debugging

1. `gitnexus_query({query: "<error or symptom>"})` — find execution flows related to the issue
2. `gitnexus_context({name: "<suspect function>"})` — see all callers, callees, and process participation
3. `READ gitnexus://repo/slancha-mesh/process/{processName}` — trace the full execution flow step by step
4. For regressions: `gitnexus_detect_changes({scope: "compare", base_ref: "main"})` — see what your branch changed

## When Refactoring

- **Renaming**: MUST use `gitnexus_rename({symbol_name: "old", new_name: "new", dry_run: true})` first. Review the preview — graph edits are safe, text_search edits need manual review. Then run with `dry_run: false`.
- **Extracting/Splitting**: MUST run `gitnexus_context({name: "target"})` to see all incoming/outgoing refs, then `gitnexus_impact({target: "target", direction: "upstream"})` to find all external callers before moving code.
- After any refactor: run `gitnexus_detect_changes({scope: "all"})` to verify only expected files changed.

## Never Do

- NEVER edit a function, class, or method without first running `gitnexus_impact` on it.
- NEVER ignore HIGH or CRITICAL risk warnings from impact analysis.
- NEVER rename symbols with find-and-replace — use `gitnexus_rename` which understands the call graph.
- NEVER commit changes without running `gitnexus_detect_changes()` to check affected scope.

## Tools Quick Reference

| Tool | When to use | Command |
|------|-------------|---------|
| `query` | Find code by concept | `gitnexus_query({query: "auth validation"})` |
| `context` | 360-degree view of one symbol | `gitnexus_context({name: "validateUser"})` |
| `impact` | Blast radius before editing | `gitnexus_impact({target: "X", direction: "upstream"})` |
| `detect_changes` | Pre-commit scope check | `gitnexus_detect_changes({scope: "staged"})` |
| `rename` | Safe multi-file rename | `gitnexus_rename({symbol_name: "old", new_name: "new", dry_run: true})` |
| `cypher` | Custom graph queries | `gitnexus_cypher({query: "MATCH ..."})` |

## Impact Risk Levels

| Depth | Meaning | Action |
|-------|---------|--------|
| d=1 | WILL BREAK — direct callers/importers | MUST update these |
| d=2 | LIKELY AFFECTED — indirect deps | Should test |
| d=3 | MAY NEED TESTING — transitive | Test if critical path |

## Resources

| Resource | Use for |
|----------|---------|
| `gitnexus://repo/slancha-mesh/context` | Codebase overview, check index freshness |
| `gitnexus://repo/slancha-mesh/clusters` | All functional areas |
| `gitnexus://repo/slancha-mesh/processes` | All execution flows |
| `gitnexus://repo/slancha-mesh/process/{name}` | Step-by-step execution trace |

## Self-Check Before Finishing

Before completing any code modification task, verify:
1. `gitnexus_impact` was run for all modified symbols
2. No HIGH/CRITICAL risk warnings were ignored
3. `gitnexus_detect_changes()` confirms changes match expected scope
4. All d=1 (WILL BREAK) dependents were updated

## Keeping the Index Fresh

After committing code changes, the GitNexus index becomes stale. Re-run analyze to update it:

```bash
npx gitnexus analyze
```

If the index previously included embeddings, preserve them by adding `--embeddings`:

```bash
npx gitnexus analyze --embeddings
```

To check whether embeddings exist, inspect `.gitnexus/meta.json` — the `stats.embeddings` field shows the count (0 means no embeddings). **Running analyze without `--embeddings` will delete any previously generated embeddings.**

> Claude Code users: A PostToolUse hook handles this automatically after `git commit` and `git merge`.

## CLI

| Task | Read this skill file |
|------|---------------------|
| Understand architecture / "How does X work?" | `.claude/skills/gitnexus/gitnexus-exploring/SKILL.md` |
| Blast radius / "What breaks if I change X?" | `.claude/skills/gitnexus/gitnexus-impact-analysis/SKILL.md` |
| Trace bugs / "Why is X failing?" | `.claude/skills/gitnexus/gitnexus-debugging/SKILL.md` |
| Rename / extract / split / refactor | `.claude/skills/gitnexus/gitnexus-refactoring/SKILL.md` |
| Tools, resources, schema reference | `.claude/skills/gitnexus/gitnexus-guide/SKILL.md` |
| Index, status, clean, wiki CLI commands | `.claude/skills/gitnexus/gitnexus-cli/SKILL.md` |

<!-- gitnexus:end -->
