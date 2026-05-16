"""nvidia-smi probe for GPU state — used by `mesh-gpu status` + the
mesh registry heartbeat (planned v0.0.7 — heartbeat carries this dict).

Returns process-level + GPU-level snapshots. Robust to GB10's [N/A]
report for memory.used/free/total (the chip hides those today; we
fall back to None and the CLI shows "n/a"). Process-level data
(pid, process_name, used_memory_mib) is reliable on GB10 — that's
the field that tells you WHO is consuming the GPU.

Pure function `probe_gpu(nvidia_smi_path: str = "nvidia-smi")` —
testable with a stubbed nvidia-smi via PATH manipulation or
subprocess monkey-patch.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)

_NA_TOKENS = {"[N/A]", "[Not Supported]", "N/A", ""}


@dataclass(frozen=True)
class GpuProcess:
    """One process holding the GPU. Fields from nvidia-smi
    --query-compute-apps=pid,process_name,used_memory."""

    pid: int
    process_name: str
    used_memory_mib: int
    # Filled by callers via ps(1) lookup; not in nvidia-smi output.
    user: str | None = None
    cmdline: str | None = None
    runtime_s: int | None = None


@dataclass(frozen=True)
class GpuSnapshot:
    """One point-in-time read of the GPU."""

    probed_at: datetime
    util_pct: Optional[float]  # gpu utilization %; None if N/A
    mem_used_mib: Optional[int]
    mem_free_mib: Optional[int]
    mem_total_mib: Optional[int]
    processes: list[GpuProcess] = field(default_factory=list)
    nvidia_smi_available: bool = True

    @property
    def mem_used_gb(self) -> Optional[float]:
        return self.mem_used_mib / 1024 if self.mem_used_mib is not None else None

    @property
    def mem_free_gb(self) -> Optional[float]:
        return self.mem_free_mib / 1024 if self.mem_free_mib is not None else None

    @property
    def mem_total_gb(self) -> Optional[float]:
        return self.mem_total_mib / 1024 if self.mem_total_mib is not None else None

    @property
    def total_proc_memory_gb(self) -> float:
        """Sum of process-level usage. Reliable on GB10 even when
        gpu-level memory.used is [N/A]."""
        return sum(p.used_memory_mib for p in self.processes) / 1024


def _parse_int_or_none(token: str) -> Optional[int]:
    token = token.strip()
    if token in _NA_TOKENS:
        return None
    try:
        # nvidia-smi sometimes appends " MiB" — strip suffix
        return int(token.split()[0])
    except (ValueError, IndexError):
        return None


def _parse_float_or_none(token: str) -> Optional[float]:
    token = token.strip()
    if token in _NA_TOKENS:
        return None
    try:
        # "12 %" → 12.0
        return float(token.rstrip("%").strip())
    except ValueError:
        return None


def probe_gpu(nvidia_smi: str = "nvidia-smi") -> GpuSnapshot:
    """Run nvidia-smi twice — once for GPU stats, once for compute-apps.

    Returns a GpuSnapshot. On any failure (no nvidia-smi on PATH, query
    errors, parse errors), returns a snapshot with nvidia_smi_available=
    False + empty processes — callers degrade gracefully.
    """
    now = datetime.now(timezone.utc)

    if shutil.which(nvidia_smi) is None:
        logger.info("nvidia-smi not on PATH; returning unavailable snapshot")
        return GpuSnapshot(
            probed_at=now, util_pct=None,
            mem_used_mib=None, mem_free_mib=None, mem_total_mib=None,
            processes=[], nvidia_smi_available=False,
        )

    util_pct = mem_used = mem_free = mem_total = None
    try:
        gpu_out = subprocess.check_output(
            [nvidia_smi,
             "--query-gpu=utilization.gpu,memory.used,memory.free,memory.total",
             "--format=csv,noheader"],
            text=True, timeout=5,
        )
        # Take the first GPU only — multi-GPU support is v0.0.7+
        for line in gpu_out.splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 4:
                util_pct = _parse_float_or_none(parts[0])
                mem_used = _parse_int_or_none(parts[1])
                mem_free = _parse_int_or_none(parts[2])
                mem_total = _parse_int_or_none(parts[3])
                break
    except (subprocess.SubprocessError, OSError) as exc:
        logger.info("nvidia-smi --query-gpu failed: %s", exc)

    processes: list[GpuProcess] = []
    try:
        proc_out = subprocess.check_output(
            [nvidia_smi,
             "--query-compute-apps=pid,process_name,used_memory",
             "--format=csv,noheader"],
            text=True, timeout=5,
        )
        for line in proc_out.splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 3:
                continue
            pid = _parse_int_or_none(parts[0])
            mem = _parse_int_or_none(parts[2])
            if pid is None or mem is None:
                continue
            processes.append(GpuProcess(
                pid=pid, process_name=parts[1], used_memory_mib=mem,
            ))
    except (subprocess.SubprocessError, OSError) as exc:
        logger.info("nvidia-smi --query-compute-apps failed: %s", exc)

    # Enrich processes with ps(1) for user + cmdline + etimes
    processes = [_enrich_process(p) for p in processes]

    return GpuSnapshot(
        probed_at=now,
        util_pct=util_pct,
        mem_used_mib=mem_used,
        mem_free_mib=mem_free,
        mem_total_mib=mem_total,
        processes=processes,
        nvidia_smi_available=True,
    )


def _enrich_process(p: GpuProcess) -> GpuProcess:
    """Add user / cmdline / runtime via ps(1). Returns the original
    process unchanged on lookup failure (the row already has pid +
    used_memory which are the load-bearing fields)."""
    try:
        out = subprocess.check_output(
            ["ps", "-p", str(p.pid),
             "-o", "user=,etimes=,args=",
             "--no-headers"],
            text=True, timeout=2,
        ).strip()
        if not out:
            return p
        # `admin 12345 /usr/bin/python ...`
        parts = out.split(None, 2)
        if len(parts) < 3:
            return p
        user, etimes_s, cmdline = parts
        try:
            runtime = int(etimes_s)
        except ValueError:
            runtime = None
        return GpuProcess(
            pid=p.pid, process_name=p.process_name,
            used_memory_mib=p.used_memory_mib,
            user=user, cmdline=cmdline, runtime_s=runtime,
        )
    except (subprocess.SubprocessError, FileNotFoundError, OSError):
        return p


__all__ = ["GpuProcess", "GpuSnapshot", "probe_gpu"]
