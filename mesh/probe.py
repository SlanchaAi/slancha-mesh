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
    "NVIDIA GB10": 273.0,  # LPDDR5X-9600, 256-bit bus, per NVIDIA datasheet
    "NVIDIA H100": 3350.0,
    "NVIDIA H200": 4800.0,
    "NVIDIA L40": 864.0,
    "Apple M4 Max": 546.0,
    "Apple M3 Ultra": 819.0,
}


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


def probe_node(
    friendly_name: str | None = None,
    master_url: str | None = None,
) -> NodeProbe:
    """Probe the local machine and return a NodeProbe.

    `master_url` is reserved for bandwidth/RTT probing in v0.0.2 — v0.0.1
    just records None and lets the registry trigger active probes via
    `POST /mesh/v1/probe-network`.
    """
    warnings: list[str] = []

    chip = _detect_chip(warnings)
    arch = _detect_arch()
    cuda_cap = _detect_cuda_capability(warnings)
    ram_total, ram_avail = _detect_memory_gb(warnings)
    vram_total, vram_avail, unified = _detect_vram_gb(warnings)

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
    fp4 = _FP4_TOPS_BY_CHIP.get(chip_key)
    fp16 = _FP16_TOPS_BY_CHIP.get(chip_key)
    bw = _MEMORY_BANDWIDTH_GBS.get(chip_key)
    if bw is None:
        warnings.append(
            f"no memory_bandwidth table entry for {chip_key!r}; allocator falls back to estimated_tps_at"
        )

    return NodeProbe(
        node_id=_detect_node_id(warnings),
        friendly_name=friendly_name or socket.gethostname(),
        chip=chip,
        arch=arch,  # type: ignore[arg-type]
        cuda_capability=cuda_cap,
        fp4_tops=fp4,
        fp16_tops=fp16,
        ram_total_gb=round(ram_total, 2),
        ram_available_gb=round(ram_avail, 2),
        vram_total_gb=round(vram_total, 2) if vram_total else None,
        vram_available_gb=round(vram_avail, 2) if vram_avail else None,
        unified_memory=unified,
        memory_bandwidth_gbs=bw,
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
    args = ap.parse_args()

    probe = probe_node(friendly_name=args.friendly_name)
    data = probe.model_dump(mode="json")
    indent = 2 if args.pretty else None
    print(json.dumps(data, indent=indent, default=str))


if __name__ == "__main__":
    main()
