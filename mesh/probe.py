"""Hardware probe — gathers a NodeProbe on the current machine.

Spec §3.1. Designed to never raise: fields the probe cannot determine
are `None`, with a human-readable note appended to `probe_warnings`. The
output is the JSON contract a node sends to the registry on boot and
periodically (every ~60s, per spec §5).

Run as a module: `python -m mesh.probe [--friendly-name NAME] [--json]`
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import shutil
import socket
import subprocess
import uuid
from pathlib import Path

from mesh.models import Backend, NodeProbe

# ---------------------------------------------------------------------------
# FP4/FP16 lookup — spec §3.1. Manually maintained per release. Numbers in
# TOPS at the indicated precision. Sources: vendor datasheets cross-checked
# with an exo-interop probe.
# ---------------------------------------------------------------------------

_FP4_TOPS_BY_CHIP: dict[str, float] = {
    # Blackwell consumer / Grace-Blackwell
    "NVIDIA GB10": 3800.0,  # dense; sparse ≈ 7600
    "NVIDIA RTX PRO 6000": 4000.0,
    # Hopper / Ada do not advertise FP4
}

_FP16_TOPS_BY_CHIP: dict[str, float] = {
    "NVIDIA GB10": 250.0,  # rough; vendor numbers vary by sparsity
    "NVIDIA H100": 989.0,
    "NVIDIA H200": 989.0,
    "NVIDIA L40": 362.0,
    "NVIDIA L4": 121.0,
    "Apple M4 Max": 36.0,
    "Apple M3 Ultra": 50.0,
}

# Memory bandwidth (GB/s). GB10 specifically does NOT expose this via
# nvidia-smi; we record the spec'd value here.
_MEMORY_BANDWIDTH_GBS: dict[str, float] = {
    "NVIDIA GB10": 273.0,  # LPDDR5X-8533, 256-bit bus (273 = 8533MT/s × 32B); per NVIDIA datasheet
    "NVIDIA H100": 3350.0,
    "NVIDIA H200": 4800.0,
    "NVIDIA L40": 864.0,
    # Measured achieved DtoD on dellpromax 2026-06-27 (1467 GB/s, MBU 0.82 of the
    # 1792 GB/s GDDR7 datasheet peak) — a better fallback than peak for the
    # bandwidth-bound decode estimate. The live bench (§4) overrides per-node.
    # Keyed by the family name; _lookup_chip_table prefix-matches the full
    # "NVIDIA RTX PRO 6000 Blackwell Workstation Edition" nvidia-smi reports.
    "NVIDIA RTX PRO 6000": 1467.0,
    "Apple M4 Max": 546.0,
    "Apple M3 Ultra": 819.0,
}


def _lookup_chip_table(table: dict[str, float], chip: str) -> float | None:
    """Per-chip constant tolerant of the marketing suffix nvidia-smi appends.

    nvidia-smi reports the full name (e.g. "NVIDIA RTX PRO 6000 Blackwell
    Workstation Edition") while the tables are keyed by the family name
    ("NVIDIA RTX PRO 6000"). Exact match first, then the LONGEST table key that
    is a prefix of the reported chip — so one family row covers the Max-Q /
    Server / Workstation variants without a brittle row per SKU. Returns None
    when nothing matches (caller then falls back further).
    """
    if chip in table:
        return table[chip]
    best_key: str | None = None
    for key in table:
        if chip.startswith(key) and (best_key is None or len(key) > len(best_key)):
            best_key = key
    return table[best_key] if best_key is not None else None


def _run(cmd: list[str], timeout: float = 4.0) -> str:
    """Run a command, return stdout or '' on any failure. Never raises."""
    try:
        out = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        return out.stdout or ""
    except (subprocess.SubprocessError, OSError, FileNotFoundError):
        return ""


# ---------------------------------------------------------------------------
# Detection routines
# ---------------------------------------------------------------------------


def _detect_chip(warnings: list[str]) -> str:
    """Best-effort chip name. NVIDIA first, then CPU brand."""
    out = _run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"])
    if out.strip():
        first = out.strip().splitlines()[0].strip()
        if first:
            return first
    # Apple
    if platform.system() == "Darwin":
        out = _run(["sysctl", "-n", "machdep.cpu.brand_string"])
        if out.strip():
            return out.strip()
    # Linux /proc/cpuinfo
    try:
        cpuinfo = Path("/proc/cpuinfo").read_text(encoding="utf-8", errors="ignore")
        for line in cpuinfo.splitlines():
            if line.lower().startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        warnings.append("could not read /proc/cpuinfo")
    return platform.processor() or "unknown"


def _detect_arch() -> str:
    m = platform.machine().lower()
    if m in ("arm64", "aarch64"):
        # macOS arm64 → apple-silicon, Linux aarch64 → aarch64
        if platform.system() == "Darwin":
            return "apple-silicon"
        return "aarch64"
    if m in ("x86_64", "amd64"):
        return "x86_64"
    return "x86_64"  # default fallback


def _detect_cuda_capability(warnings: list[str]) -> str | None:
    out = _run(
        ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"]
    )
    cap = out.strip().splitlines()[0].strip() if out.strip() else ""
    if cap and re.match(r"^\d+\.\d+$", cap):
        return cap
    if not out:
        warnings.append("no nvidia-smi or no CUDA GPU")
    return None


def _detect_amd_gpu(warnings: list[str]) -> str | None:
    """Best-effort AMD/Radeon GPU name on Windows. None if none found.

    NVIDIA exposes itself via nvidia-smi (handled in `_detect_chip` /
    `_detect_cuda_capability`); AMD does not, so a Windows + Radeon box would
    otherwise look CPU-only and get steered to CPU inference even though Ollama
    can use the AMD GPU on Windows. We enumerate display adapters via WMI
    (`wmic`, then PowerShell `Get-CimInstance` as a fallback for newer Windows
    where wmic is deprecated) and return the first AMD/Radeon adapter name.

    Windows-only and only meaningful when nvidia-smi found nothing; callers
    gate accordingly. Never raises (uses `_run`).
    """
    if platform.system() != "Windows":
        return None

    out = _run(["wmic", "path", "win32_VideoController", "get", "name"])
    if not out.strip():
        # wmic is removed on some Windows 11 builds — try PowerShell/CIM.
        out = _run(
            [
                "powershell",
                "-NoProfile",
                "-Command",
                "Get-CimInstance Win32_VideoController | "
                "Select-Object -ExpandProperty Name",
            ]
        )

    for line in out.splitlines():
        name = line.strip()
        if not name or name.lower() == "name":  # wmic header row
            continue
        low = name.lower()
        if "amd" in low or "radeon" in low:
            return name

    warnings.append("no AMD/Radeon display adapter found via wmic/CIM")
    return None


def _detect_memory_gb(warnings: list[str]) -> tuple[float, float]:
    """Return (ram_total_gb, ram_available_gb). Cross-OS via psutil.

    psutil (a hard dependency) reports total + available on Windows, Linux,
    and macOS — so this is the primary path. The /proc/meminfo + sysctl paths
    below are kept only as a defensive fallback if psutil is somehow
    unavailable. (Before this, Windows hit neither OS-specific path and
    reported 0.0 — found on a real GTX-1070 Win10 box, 2026-05-26.)
    """
    try:
        import psutil

        vm = psutil.virtual_memory()
        if vm.total > 0:
            return vm.total / (1024**3), vm.available / (1024**3)
    except Exception:  # noqa: BLE001 — fall back to OS-specific probes
        pass
    try:
        meminfo = Path("/proc/meminfo").read_text(encoding="utf-8", errors="ignore")
        total = avail = 0.0
        for line in meminfo.splitlines():
            if line.startswith("MemTotal:"):
                total = float(line.split()[1]) / (1024 * 1024)
            elif line.startswith("MemAvailable:"):
                avail = float(line.split()[1]) / (1024 * 1024)
        if total > 0:
            return total, avail or total * 0.5
    except OSError:
        pass
    # macOS / fallback via sysctl
    if platform.system() == "Darwin":
        out = _run(["sysctl", "-n", "hw.memsize"])
        if out.strip().isdigit():
            total = int(out.strip()) / (1024**3)
            return total, total * 0.5  # crude; vm_stat parsing skipped for v0.0.1
    warnings.append("memory probe failed; using 0.0")
    return 0.0, 0.0


def _detect_vram_gb(warnings: list[str]) -> tuple[float | None, float | None, bool]:
    """Return (vram_total_gb, vram_available_gb, unified_memory).

    GB10 reports vram via nvidia-smi but it's unified with the host RAM
    (same physical pool); we flag unified=True and report whichever value
    nvidia-smi gives. This matters for the allocator: unified-mem nodes
    can fit bigger models than the discrete-VRAM bucket suggests.
    """
    out = _run(
        [
            "nvidia-smi",
            "--query-gpu=memory.total,memory.free",
            "--format=csv,noheader,nounits",
        ]
    )
    line = out.strip().splitlines()[0].strip() if out.strip() else ""
    if line and "," in line:
        try:
            total_str, free_str = (s.strip() for s in line.split(","))
            # GB10 currently reports "[N/A]" for both fields — capture that.
            if total_str.upper() in ("[N/A]", "N/A", ""):
                warnings.append(
                    "nvidia-smi reports VRAM as [N/A] — likely GB10 unified-mem; "
                    "falling back to RAM-based fit"
                )
                return None, None, True
            total_mib = float(total_str)
            free_mib = float(free_str) if free_str.upper() not in ("[N/A]", "N/A") else total_mib
            return total_mib / 1024.0, free_mib / 1024.0, False
        except ValueError:
            warnings.append(f"could not parse nvidia-smi memory line: {line!r}")
    # No NVIDIA GPU. Apple Silicon = unified.
    if platform.system() == "Darwin":
        return None, None, True
    return None, None, False


def _detect_lan_interfaces() -> list[str]:
    """List up interfaces from `ip -o link` or `ifconfig -l`."""
    out = _run(["ip", "-o", "link", "show"])
    if out:
        names: list[str] = []
        for line in out.splitlines():
            m = re.match(r"^\d+:\s+([^:@]+)[:@]", line)
            if m:
                name = m.group(1).strip()
                if name != "lo":
                    names.append(name)
        return names
    # macOS
    out = _run(["ifconfig", "-l"])
    if out:
        return [n for n in out.strip().split() if n and n != "lo0"]
    return []


def _detect_backends() -> list[Backend]:
    """Best-effort detection of which inference backends are installed.

    For v0.0.1 we use shutil.which + Python import probes. False
    positives are fine (catalog hard-filters at fit-score time on
    `required_backend`); false negatives are the cost of being
    conservative.
    """
    backends: list[Backend] = []
    if shutil.which("vllm") or _try_import("vllm"):
        backends.append("vllm")
    if shutil.which("llama-cli") or shutil.which("llama") or _try_import("llama_cpp"):
        backends.append("llamacpp")
    if shutil.which("ollama"):
        backends.append("ollama")
    if platform.system() == "Darwin" and _try_import("mlx"):
        backends.append("mlx")
    return backends


def _try_import(modname: str) -> bool:
    import importlib.util

    try:
        return importlib.util.find_spec(modname) is not None
    except (ImportError, ValueError):
        return False


def _detect_disk_free_gb(path: str = ".") -> float:
    try:
        usage = shutil.disk_usage(path)
        return usage.free / (1024**3)
    except OSError:
        return 0.0


def _detect_public_ipv4() -> str | None:
    """Lightweight: look at `hostname -I` first; never call out to internet."""
    out = _run(["hostname", "-I"])
    for tok in out.split():
        if re.match(r"^\d+\.\d+\.\d+\.\d+$", tok) and not tok.startswith("127."):
            return tok
    return None


def _detect_thunderbolt5() -> bool:
    # Linux: look for tb5 / Thunderbolt 5 in lspci or /sys/bus/thunderbolt
    out = _run(["lspci"])
    if "Thunderbolt 5" in out or "TB5" in out:
        return True
    # Heuristic for our test fleet: GB10 ships with TB5
    return False


def _detect_node_id(warnings: list[str]) -> str:
    """Stable node id: prefer /etc/machine-id, fall back to hostname+mac."""
    try:
        mid = Path("/etc/machine-id").read_text(encoding="utf-8").strip()
        if mid:
            return mid
    except OSError:
        pass
    warnings.append("no /etc/machine-id; using hostname+uuid mix")
    host = socket.gethostname() or "unknown"
    seed = uuid.uuid5(uuid.NAMESPACE_DNS, host)
    return str(seed)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
# Effective-bandwidth micro-bench (§4) — measure, don't guess
# ---------------------------------------------------------------------------

# The bench drives libcuda directly (no torch dependency) and is GUARDED so it
# can never OOM a co-resident serve (the GB10 failure mode): it copies between
# two 2 GiB buffers and only runs if the GPU has comfortable free memory on top
# of them. Wall-clock boxed; never raises.
_BENCH_BUF_BYTES = 1 << 31          # 2 GiB working buffer (×2 for src + dst)
_BENCH_MIN_FREE_HEADROOM = 4 << 30  # require this much free BEYOND the buffers
_BENCH_MAX_SECONDS = 3.0


def _measure_memory_bandwidth_gbs(warnings: list[str]) -> float | None:
    """Measured device memory bandwidth (GB/s) via a libcuda DtoD copy, or None.

    Returns None — caller keeps the datasheet table value, tagged "guessed" —
    when this isn't an NVIDIA/CUDA box, libcuda is missing, the GPU lacks
    comfortable free memory (so we NEVER OOM a resident model), or any driver
    error occurs. Never raises; wall-clock boxed at _BENCH_MAX_SECONDS.
    """
    import ctypes
    import sys
    import time

    # Linux/Spark soname vs the Windows CUDA driver DLL. A non-CUDA box (Mac,
    # AMD-only, CPU) raises OSError on both → None (caller keeps the table value).
    cu = None
    for _libname in (["nvcuda"] if sys.platform == "win32" else ["libcuda.so.1"]):
        try:
            cu = ctypes.CDLL(_libname)
            break
        except OSError:
            continue
    if cu is None:
        return None
    vp = ctypes.c_void_p
    ull = ctypes.c_ulonglong
    ctx = vp()
    a = ull()
    b = ull()
    e0 = vp()
    e1 = vp()
    ctx_made = False
    try:
        # Full argtypes — a by-value 64-bit handle/size passed without argtypes
        # would default to c_int and truncate. All return codes are checked
        # below; an unchecked NULL event handle would SIGSEGV past this try.
        cu.cuInit.argtypes = [ctypes.c_uint]
        cu.cuDeviceGet.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int]
        cu.cuCtxCreate_v2.argtypes = [ctypes.POINTER(vp), ctypes.c_uint, ctypes.c_int]
        cu.cuCtxDestroy_v2.argtypes = [vp]
        cu.cuCtxSynchronize.argtypes = []
        cu.cuMemGetInfo_v2.argtypes = [ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_size_t)]
        cu.cuMemAlloc_v2.argtypes = [ctypes.POINTER(ull), ctypes.c_size_t]
        cu.cuMemFree_v2.argtypes = [ull]
        cu.cuMemsetD8_v2.argtypes = [ull, ctypes.c_ubyte, ctypes.c_size_t]
        cu.cuMemcpyDtoD_v2.argtypes = [ull, ull, ctypes.c_size_t]
        cu.cuEventCreate.argtypes = [ctypes.POINTER(vp), ctypes.c_uint]
        cu.cuEventDestroy_v2.argtypes = [vp]
        cu.cuEventRecord.argtypes = [vp, vp]
        cu.cuEventSynchronize.argtypes = [vp]
        cu.cuEventElapsedTime.argtypes = [ctypes.POINTER(ctypes.c_float), vp, vp]

        if cu.cuInit(0) != 0:
            return None
        dev = ctypes.c_int()
        if cu.cuDeviceGet(ctypes.byref(dev), 0) != 0:
            return None
        # A fresh floating context is safe HERE: serve bring-up runs this BEFORE
        # any model loads, and the vLLM/llama.cpp backends run in separate
        # processes (Popen) with their own contexts — so we never race a torch
        # primary-ctx init in this process (the cuCtxCreate-before-torch driver
        # bug). An in-process GPU user would instead need cuDevicePrimaryCtxRetain.
        if cu.cuCtxCreate_v2(ctypes.byref(ctx), 0, dev) != 0:
            return None
        ctx_made = True

        free = ctypes.c_size_t()
        total = ctypes.c_size_t()
        if cu.cuMemGetInfo_v2(ctypes.byref(free), ctypes.byref(total)) != 0:
            return None
        need = 2 * _BENCH_BUF_BYTES + _BENCH_MIN_FREE_HEADROOM  # buffers + headroom
        if free.value < need:
            warnings.append(
                f"bandwidth bench skipped: GPU free {free.value >> 30}GB "
                f"< {need >> 30}GB needed; keeping table value (guessed)"
            )
            return None

        n = _BENCH_BUF_BYTES
        if cu.cuMemAlloc_v2(ctypes.byref(a), n) != 0:
            return None
        if cu.cuMemAlloc_v2(ctypes.byref(b), n) != 0:
            return None
        if cu.cuEventCreate(ctypes.byref(e0), 0) != 0:  # NULL handle → SIGSEGV if unchecked
            return None
        if cu.cuEventCreate(ctypes.byref(e1), 0) != 0:
            return None

        cu.cuMemsetD8_v2(a, 1, n)
        deadline = time.monotonic() + _BENCH_MAX_SECONDS  # box INCLUDES warmup
        for _ in range(3):  # warmup
            if cu.cuMemcpyDtoD_v2(b, a, n) != 0:
                return None
        if cu.cuCtxSynchronize() != 0:
            return None

        best_ms: float | None = None
        ms = ctypes.c_float()
        iters = 0
        while time.monotonic() < deadline and iters < 50:
            cu.cuEventRecord(e0, None)
            if cu.cuMemcpyDtoD_v2(b, a, n) != 0:
                break  # a failed copy would record a bogus time → fall to None
            cu.cuEventRecord(e1, None)
            if cu.cuEventSynchronize(e1) != 0:
                break
            cu.cuEventElapsedTime(ctypes.byref(ms), e0, e1)
            if best_ms is None or ms.value < best_ms:
                best_ms = ms.value
            iters += 1
        if not best_ms or best_ms <= 0:
            return None
        return round(2.0 * n / (best_ms / 1e3) / 1e9, 1)  # DtoD = read + write
    except Exception as exc:  # noqa: BLE001 — a probe bench must never break bring-up
        warnings.append(f"bandwidth bench errored ({type(exc).__name__}); keeping table value")
        return None
    finally:
        # Release everything on every path (incl. early returns). Guard on the
        # handle value so an un-acquired resource is never double-freed.
        if a.value:
            cu.cuMemFree_v2(a)
        if b.value:
            cu.cuMemFree_v2(b)
        if e0.value:
            cu.cuEventDestroy_v2(e0)
        if e1.value:
            cu.cuEventDestroy_v2(e1)
        if ctx_made:
            cu.cuCtxDestroy_v2(ctx)


# ---------------------------------------------------------------------------


def probe_node(
    friendly_name: str | None = None,
    master_url: str | None = None,
    measure_bandwidth: bool = False,
) -> NodeProbe:
    """Probe the local machine and return a NodeProbe.

    `master_url` is reserved for bandwidth/RTT probing in v0.0.2 — v0.0.1
    just records None and lets the registry trigger active probes via
    `POST /mesh/v1/probe-network`.

    `measure_bandwidth=True` runs a guarded micro-bench to set
    `memory_bandwidth_gbs` from a real measurement (`bw_source="measured"`)
    instead of the datasheet table (`"guessed"`). Off by default so the ~60s
    heartbeat path never benches; bring-up (serve) opts in.
    """
    warnings: list[str] = []

    chip = _detect_chip(warnings)
    arch = _detect_arch()
    cuda_cap = _detect_cuda_capability(warnings)
    ram_total, ram_avail = _detect_memory_gb(warnings)
    vram_total, vram_avail, unified = _detect_vram_gb(warnings)

    # Windows + non-NVIDIA GPU: nvidia-smi found nothing, so `chip` is the CPU
    # brand and cuda_capability is None. Best-effort detect an AMD/Radeon
    # adapter so the engine selector can route to Ollama (it uses the AMD GPU
    # on Windows) instead of the CPU fallback. cuda_capability stays None —
    # this is NOT a CUDA GPU; we surface the vendor via `gpu_vendor` instead.
    gpu_vendor: str | None = None
    if cuda_cap is None:
        amd_name = _detect_amd_gpu(warnings)
        if amd_name:
            gpu_vendor = "amd"
            chip = amd_name  # CPU brand would otherwise be reported

    # GB10 is unified memory even though nvidia-smi reports VRAM normally
    # on some driver releases. Detect by chip name as a safety net.
    if "GB10" in chip:
        unified = True
        if vram_total is None:
            # Treat unified RAM as the model-fit budget on GB10. Subtract
            # 8GB for the OS — Spark Linux uses ~6-7GB at idle.
            warnings.append(
                "GB10 unified-mem: using ram_available_gb - 8 as effective VRAM"
            )

    chip_key = chip.strip()
    fp4 = _lookup_chip_table(_FP4_TOPS_BY_CHIP, chip_key)
    fp16 = _lookup_chip_table(_FP16_TOPS_BY_CHIP, chip_key)

    # Bandwidth: a live measurement when asked and possible, else the datasheet
    # table (tagged "guessed"), else None. The bench self-skips on non-CUDA
    # boxes and when GPU memory is tight, so it's safe to opt in at bring-up.
    bw = _lookup_chip_table(_MEMORY_BANDWIDTH_GBS, chip_key)
    bw_source: str | None = "guessed" if bw is not None else None
    if measure_bandwidth:
        measured = _measure_memory_bandwidth_gbs(warnings)
        if measured is not None:
            bw, bw_source = measured, "measured"
    if bw is None:
        warnings.append(
            f"no memory_bandwidth for {chip_key!r} (table miss, bench unavailable); "
            f"allocator falls back to estimated_tps_at"
        )

    return NodeProbe(
        node_id=_detect_node_id(warnings),
        friendly_name=friendly_name or socket.gethostname(),
        chip=chip,
        arch=arch,  # type: ignore[arg-type]
        cuda_capability=cuda_cap,
        gpu_vendor=gpu_vendor,
        fp4_tops=fp4,
        fp16_tops=fp16,
        ram_total_gb=round(ram_total, 2),
        ram_available_gb=round(ram_avail, 2),
        vram_total_gb=round(vram_total, 2) if vram_total else None,
        vram_available_gb=round(vram_avail, 2) if vram_avail else None,
        unified_memory=unified,
        memory_bandwidth_gbs=bw,
        bw_source=bw_source,  # type: ignore[arg-type]
        public_ipv4=_detect_public_ipv4(),
        lan_interfaces=_detect_lan_interfaces(),
        bandwidth_to_master_mbps=None,
        rtt_to_master_ms=None,
        thunderbolt5=_detect_thunderbolt5(),
        available_backends=_detect_backends(),
        disk_free_gb=round(_detect_disk_free_gb(os.getcwd()), 2),
        probe_warnings=warnings,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Probe the local node for the Slancha-Mesh registry.")
    ap.add_argument("--friendly-name", default=None)
    ap.add_argument("--json", action="store_true", help="Emit JSON (default).")
    ap.add_argument("--pretty", action="store_true", help="Pretty-print JSON.")
    ap.add_argument(
        "--measure-bandwidth",
        action="store_true",
        help="Run the guarded memory-bandwidth micro-bench (sets bw_source=measured). "
        "Skips safely if the GPU is busy or non-CUDA.",
    )
    args = ap.parse_args()

    probe = probe_node(
        friendly_name=args.friendly_name, measure_bandwidth=args.measure_bandwidth
    )
    data = probe.model_dump(mode="json")
    indent = 2 if args.pretty else None
    print(json.dumps(data, indent=indent, default=str))


if __name__ == "__main__":
    main()
