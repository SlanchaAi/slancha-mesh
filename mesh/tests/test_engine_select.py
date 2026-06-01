"""Hardware → inference-engine recommendation.

Encodes the 2026 engine-selection decision tree (research-backed): the right
engine is a function of chip/arch/memory, not a per-card constant. This is the
primitive that lets a heterogeneous network self-configure — without it, a
vLLM-only catalog excludes every Mac / CPU / GB10 box (the observed "nothing
fits" on Apple Silicon).
"""

from __future__ import annotations

from mesh.engine_select import recommend_engine
from mesh.models import NodeProbe


def _probe(**kw) -> NodeProbe:
    base = dict(
        node_id="n", friendly_name="box", chip="generic", arch="x86_64",
        ram_total_gb=64.0, ram_available_gb=48.0,
    )
    base.update(kw)
    return NodeProbe(**base)


def test_apple_silicon_picks_mlx():
    rec = recommend_engine(_probe(arch="apple-silicon", chip="m3-max", unified_memory=True))
    assert rec.backend == "mlx"
    assert "vllm" not in rec.backend


def test_gb10_spark_picks_ollama_not_vllm():
    # GB10 / DGX Spark: aarch64 + CUDA + unified memory. vLLM has no official
    # sm_121 aarch64 wheels (research) → Ollama/llama.cpp is the supported path.
    rec = recommend_engine(_probe(
        arch="aarch64", chip="gb10", cuda_capability="12.1", unified_memory=True,
        ram_available_gb=110.0,
    ))
    assert rec.backend in ("ollama", "llamacpp")
    assert "vllm" in rec.rationale.lower()  # explains why NOT vllm


def test_discrete_nvidia_24gb_picks_vllm():
    rec = recommend_engine(_probe(
        arch="x86_64", chip="rtx-4090", cuda_capability="8.9", vram_available_gb=24.0,
    ), os_name="Linux")
    assert rec.backend == "vllm"
    assert rec.quant in ("fp8", "awq")


def test_windows_nvidia_picks_ollama_not_vllm():
    # vLLM is Linux/WSL-only; Windows + NVIDIA → Ollama (native CUDA).
    rec = recommend_engine(_probe(
        arch="x86_64", chip="rtx-4090", cuda_capability="8.9", vram_available_gb=24.0,
    ), os_name="Windows")
    assert rec.backend == "ollama"
    assert "gguf" in rec.quant
    assert "vllm" in rec.rationale.lower()


def test_windows_amd_gpu_picks_ollama_not_cpu():
    # Issue #63: Windows + AMD GPU (no CUDA) — Ollama uses the AMD GPU on
    # Windows, so it must beat the CPU/llama.cpp fallback.
    rec = recommend_engine(_probe(
        arch="x86_64", chip="AMD Radeon RX 7900 XTX",
        cuda_capability=None, gpu_vendor="amd",
    ), os_name="Windows")
    assert rec.backend == "ollama"
    assert rec.quant != "gguf-q4-cpu"  # not the CPU-only fallback
    assert "amd" in rec.rationale.lower()


def test_non_windows_amd_gpu_still_cpu_fallback():
    # gpu_vendor is set but the box is Linux — without a Linux ROCm path wired
    # in, this stays on the CPU fallback (the new branch is Windows-gated).
    rec = recommend_engine(_probe(
        arch="x86_64", chip="AMD Radeon RX 7900 XTX",
        cuda_capability=None, gpu_vendor="amd",
    ), os_name="Linux")
    assert rec.quant == "gguf-q4-cpu"


def test_windows_cpu_picks_gguf():
    rec = recommend_engine(_probe(arch="x86_64", chip="intel", cuda_capability=None), os_name="Windows")
    assert rec.backend in ("ollama", "llamacpp")
    assert "gguf" in rec.quant


def test_discrete_nvidia_small_vram_picks_gguf():
    rec = recommend_engine(_probe(
        arch="x86_64", chip="rtx-4060", cuda_capability="8.9", vram_available_gb=8.0,
    ), os_name="Linux")
    assert rec.backend in ("ollama", "llamacpp")
    assert "gguf" in rec.quant


def test_cpu_only_picks_llamacpp():
    rec = recommend_engine(_probe(arch="x86_64", chip="xeon", cuda_capability=None))
    assert rec.backend in ("llamacpp", "ollama")
    assert "gguf" in rec.quant


def test_installed_flag_reflects_available_backends():
    rec = recommend_engine(_probe(
        arch="apple-silicon", chip="m2", unified_memory=True, available_backends=["mlx", "ollama"],
    ))
    assert rec.backend == "mlx" and rec.installed is True

    rec2 = recommend_engine(_probe(
        arch="apple-silicon", chip="m2", unified_memory=True, available_backends=["ollama"],
    ))
    assert rec2.backend == "mlx" and rec2.installed is False  # recommended but not installed


def test_rec_is_serializable():
    rec = recommend_engine(_probe(arch="apple-silicon", chip="m1", unified_memory=True))
    d = rec.as_dict()
    assert set(d) >= {"backend", "quant", "installed", "rationale", "alternates"}
