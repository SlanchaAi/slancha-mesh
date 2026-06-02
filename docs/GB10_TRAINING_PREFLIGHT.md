# GB10 / DGX-Spark training preflight (issue #65)

The real PEFT/LoRA leg (`TrainingPass(allow_real=True)`, behind the `[train]`
extra) is **code-complete and CPU-tested**; the only remaining acceptance step
for #65 is a validated real pass on a GB10 / DGX-Spark box. This is a **hardware
session**, not a code one — run it on the Spark, not a laptop.

These gates are hard-won (each cost a reboot to discover, from the
forge / `sunlit-aurora` autonomous loop). **Do every one before the first real
pass.** They are ordered by how badly they bite.

---

## 1. Cap the GPU clocks — THE gate (do this first, verify it)

GB10 clock-boost under sustained training load trips the chassis VRM into
**overcurrent → a silent hard-lock of the whole box**. It is *not* an OOM and
*not* a CUDA error — it looks like random instability / a wedged node, so you'll
misdiagnose it as a memory bug for days.

```bash
sudo cp mesh/deploy/gb10-safe-clocks.service /etc/systemd/system/
sudo systemctl enable --now gb10-safe-clocks
nvidia-smi -q -d CLOCK          # confirm the locked range took effect
```

Verify the cap is **active** before every train. If `nvidia-smi -q -d CLOCK`
doesn't show the locked range, STOP — do not train.

## 2. `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` — single-token form ONLY

The multi-token form (anything comma-appended) **aborts torch 2.11 in c10**.
Set it to exactly `expandable_segments:True`, nothing else. (This is what the
generator stamps into each spec's `env`, GATE-CONTRACT — keep it single-token.)

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   # exactly this
```

## 3. Free the GPU before loading — kill `VLLM::EngineCore`

`systemctl stop` of a vLLM serve unit **misses the EngineCore subprocess**,
which leaves ~67 GB held. Kill it explicitly and confirm memory is freed:

```bash
sudo pkill -9 -f 'VLLM::EngineCore'
sudo pkill -9 -f 'api_server'
nvidia-smi                       # confirm VRAM actually freed before you load
```

## 4. Never co-host train + serve on the same box

Single-box: stop serving before training (gate #3). In the mesh (multi-node),
the node sets `health=draining` in its heartbeat so the router routes around it
while it trains, then it rejoins — same invariant, enforced by routing instead
of a process kill (GATE-CONTRACT "never co-host train+serve").

## 5. Launch through the GPU scheduler, via bash

On the shared GB10 (`spark-472e`), every training launch goes through
`gpu-launch --kind training` so it gets hard preempt-protection and shows up in
attribution:

```bash
export GPU_JOB_OWNER=slancha-spark
export GPU_JOB_TAG=slancha-spark/train-mesh65-validate
gpu-launch --kind training -- /bin/bash -c 'source venv_cu130/bin/activate && python ...'
```

Run commands via **`/bin/bash`, not `sh`** — `sh` has no `source` (→ `rc127`).
Use `executable="/bin/bash"` for any subprocess that sources an env.

## 6. Hybrid models only — flash-linear-attention

If the base is a Gated-DeltaNet / linear-attention hybrid (e.g. Qwen3.5-27B),
install `flash-linear-attention` (github `fla-org`, `--no-deps`) or the torch
GDN fallback OOMs ~3 GB/layer. **PyPI `fla` 0.5.0 is a broken stub — do not use
it.** (Skip this gate for standard transformer bases.)

## 7. Benchmark 5–10 steps before the full run

`PYTHONUNBUFFERED=1`, run 5–10 steps, confirm: clocks still capped (`nvidia-smi
-q -d CLOCK`), VRAM stable (no creep toward the ceiling), loss decreasing, no
VRM wedge. Only then launch the full pass.

---

## The #65 acceptance run

With gates 1–7 satisfied, on the Spark (`source cu130_env.sh`, `venv_cu130`,
`pip install -e ".[train]"`):

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export GPU_JOB_OWNER=slancha-spark GPU_JOB_TAG=slancha-spark/train-mesh65-validate
gpu-launch --kind training -- /bin/bash -c '
  source cu130_env.sh && source venv_cu130/bin/activate &&
  python -c "from mesh.training import TrainingPass; ..."   # allow_real=True, a small cluster corpus
'
```

**Pass criteria:** the real pass completes, writes a checkpoint with
`meta.stub=False` + a `base_model_fingerprint`, and the eval gate
(`mesh.eval.gate.decide`) promotes-or-rejects it against the champion on the
holdout — i.e. a real adapter flows through the full champion gate end-to-end,
on the actual hardware, with the box still alive afterward. That closes #65.

---

*Source: memory `gb10-training-safety-gates` (from the `sunlit-aurora` / forge
loop, 2026-06-01). Clock unit: `mesh/deploy/gb10-safe-clocks.service`.*
