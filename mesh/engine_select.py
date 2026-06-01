"""Hardware → inference-engine + quantization recommendation.

The right serving engine is a function of the node's silicon, not a constant
baked into each specialist card. A vLLM-only catalog silently excludes every
non-CUDA box; this module is what lets a heterogeneous network self-configure.

Decision tree (2026 research — see docs/SELF_ORGANIZING_LOOP_SCOPE.md):

  Apple Silicon            → MLX (4-bit)            — vLLM unsupported on Metal
  GB10 / DGX Spark         → Ollama / llama.cpp     — no official vLLM sm_121
  (aarch64 + CUDA + unified)  (GGUF Q4)               aarch64 wheels; TRT-LLM beta
  Discrete NVIDIA ≥24 GB   → vLLM (FP8 if Ada/      — throughput path
                              Hopper+, else AWQ)
  Discrete NVIDIA <24 GB   → Ollama / llama.cpp     — GGUF fits small VRAM
  CPU-only                 → llama.cpp / Ollama     — GGUF on RAM

Pure function over `NodeProbe`; the CLI `plan` command surfaces it and the
agent decides. `installed` reflects whether the recommended backend is already
in `probe.available_backends` (else the agent/operator installs it).
"""

from __future__ import annotations

import platform
from dataclasses import dataclass, field

from mesh.models import NodeProbe

# Compute capabilities with native FP8 tensor cores (Ada 8.9, Hopper 9.0,
# datacenter/consumer Blackwell 10.0/12.0). Ampere (8.x ≤ 8.6) + A100 (8.0)
# lack FP8 → fall back to AWQ. [P, docs.vllm.ai/quantization/fp8, 2026]
_FP8_CAPABLE = {"8.9", "9.0", "10.0", "12.0"}

# Below this (GB of GPU/effective memory) a discrete NVIDIA box is better
# served by GGUF on Ollama/llama.cpp than by vLLM.
_VLLM_VRAM_FLOOR_GB = 24.0


@dataclass(frozen=True)
class EngineRec:
    """A serving-engine recommendation for one node."""

    backend: str  # one of the Backend literals: vllm | llamacpp | ollama | mlx
    quant: str  # quantization hint: mlx-4bit | gguf-q4 | awq | fp8 | gguf-q4-cpu
    installed: bool  # is `backend` already in probe.available_backends?
    rationale: str
    alternates: tuple[str, ...] = field(default_factory=tuple)

    def as_dict(self) -> dict:
        return {
            "backend": self.backend,
            "quant": self.quant,
            "installed": self.installed,
            "rationale": self.rationale,
            "alternates": list(self.alternates),
        }


def _effective_gb(probe: NodeProbe) -> float:
    """Memory the model can actually use: discrete VRAM, else unified RAM."""
    if probe.vram_available_gb:
        return probe.vram_available_gb
    if probe.unified_memory:
        return probe.ram_available_gb
    return probe.ram_available_gb


def _is_spark_class(probe: NodeProbe) -> bool:
    """GB10 / DGX Spark / Thor-class: aarch64 + CUDA + unified memory."""
    if probe.cuda_capability is None:
        return False
    chip = (probe.chip or "").lower()
    if any(tag in chip for tag in ("gb10", "spark", "thor", "gh200")):
        return True
    return probe.arch == "aarch64" and probe.unified_memory


def recommend_engine(probe: NodeProbe, *, os_name: str | None = None) -> EngineRec:
    """Recommend a serving engine + quantization for this node's hardware.

    `os_name` is the host OS (`platform.system()` — "Windows" | "Linux" |
    "Darwin"); defaults to the live host. It matters because the probe's arch
    can't distinguish Windows from Linux on x86_64: vLLM is Linux/WSL-only,
    so a Windows + NVIDIA box must be steered to Ollama (which runs natively
    on Windows with CUDA acceleration), not vLLM.
    """
    os_name = os_name or platform.system()

    def rec(backend: str, quant: str, rationale: str, alternates: tuple[str, ...]) -> EngineRec:
        return EngineRec(
            backend=backend,
            quant=quant,
            installed=backend in (probe.available_backends or []),
            rationale=rationale,
            alternates=alternates,
        )

    # Apple Silicon → MLX (Metal). vLLM does not run on Apple Silicon.
    if probe.arch == "apple-silicon":
        return rec(
            "mlx", "mlx-4bit",
            "Apple Silicon: MLX is the native Metal path; vLLM is unsupported. "
            "Ollama (GGUF) is the zero-config alternative.",
            ("ollama", "llamacpp"),
        )

    # GB10 / DGX Spark (aarch64 + CUDA + unified) → Ollama/llama.cpp.
    if _is_spark_class(probe):
        return rec(
            "ollama", "gguf-q4",
            "GB10/DGX Spark-class (sm_121, aarch64, unified memory): vLLM has no "
            "official aarch64 sm_121 wheels (community build only); Ollama/llama.cpp "
            "are the supported path. TensorRT-LLM (NVFP4) is the beta throughput option.",
            ("llamacpp", "vllm"),
        )

    # Discrete NVIDIA (x86_64 + CUDA).
    if probe.cuda_capability is not None and probe.arch == "x86_64":
        gb = _effective_gb(probe)
        # vLLM is Linux/WSL-only. On Windows, Ollama runs natively with CUDA
        # acceleration — that's the right NVIDIA path there, not vLLM.
        if os_name == "Windows":
            return rec(
                "ollama", "gguf-q4",
                f"Windows + NVIDIA (cc {probe.cuda_capability}, ~{gb:.0f} GB): Ollama "
                "runs natively with CUDA acceleration; vLLM is Linux/WSL-only. Run "
                "vLLM under WSL2 if you need its throughput.",
                ("llamacpp",),
            )
        if gb >= _VLLM_VRAM_FLOOR_GB:
            quant = "fp8" if probe.cuda_capability in _FP8_CAPABLE else "awq"
            return rec(
                "vllm", quant,
                f"Discrete NVIDIA (cc {probe.cuda_capability}, ~{gb:.0f} GB): vLLM for "
                f"throughput; {quant} ({'native FP8 tensor cores' if quant == 'fp8' else 'no FP8 — AWQ via Marlin'}).",
                ("ollama", "llamacpp"),
            )
        return rec(
            "ollama", "gguf-q4",
            f"Discrete NVIDIA with limited memory (~{gb:.0f} GB): GGUF Q4 on "
            "Ollama/llama.cpp fits; vLLM+AWQ becomes viable above ~16 GB.",
            ("llamacpp", "vllm"),
        )

    # Windows + non-NVIDIA GPU (AMD/Radeon). There's no CUDA here, but Ollama
    # uses the AMD GPU on Windows (DirectML/ROCm path), so this beats CPU-only.
    # `gpu_vendor` is set by the probe when nvidia-smi found nothing but a WMI
    # adapter scan found AMD. NVIDIA still goes through the cuda_capability
    # branch above; this is strictly the non-CUDA-GPU case.
    if os_name == "Windows" and probe.gpu_vendor:
        return rec(
            "ollama", "gguf-q4",
            f"Windows + {probe.gpu_vendor.upper()} GPU ({probe.chip}): no CUDA, but "
            "Ollama uses the GPU on Windows (DirectML/ROCm) — better than CPU. "
            "vLLM is Linux/WSL + NVIDIA-only.",
            ("llamacpp",),
        )

    # No CUDA → CPU inference.
    return rec(
        "llamacpp", "gguf-q4-cpu",
        "No CUDA GPU detected: CPU inference via llama.cpp (GGUF Q4). Expect "
        "modest tok/s; prefer ≤14B models.",
        ("ollama",),
    )


__all__ = ["EngineRec", "recommend_engine"]
