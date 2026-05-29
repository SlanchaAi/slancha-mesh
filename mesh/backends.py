"""Backend abstraction — spec §9.

A `BaseBackend` knows how to (a) spawn a process that serves a
specialist over an OpenAI-compatible HTTP endpoint, (b) report
liveness + basic utilization, and (c) terminate cleanly.

v0.0.2 ships `VLLMBackend` and a `NullBackend` for tests. `LlamaCppBackend`
is sketched but not wired — the catalog cards mark `required_backend =
"vllm"` for the only specialist with downloaded weights today.

Why this split? Two reasons:
  1. The mesh router only cares about `node_url` + heartbeat. A clean
     `Backend.start()` → URL contract lets us swap engines without
     touching the registry or the selector.
  2. vLLM 0.17 on GB10 sm_121 is bleeding-edge; if a future build
     breaks, the same TOML cards should slot into llama.cpp + GGUF
     by changing one `required_backend` field.
"""

from __future__ import annotations

import os
import shutil
import signal
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import httpx

from mesh.models import SpecialistCard


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


class BaseBackend(Protocol):
    """Minimal contract the mesh expects from a serving backend.

    Implementations are responsible for their own subprocess lifecycle
    and for emitting an OpenAI-compatible HTTP API at `base_url`.
    """

    card: SpecialistCard
    base_url: str
    name: str  # e.g. "vllm", "llamacpp"

    def start(self) -> None:
        """Spawn the serving process. Must not block beyond launch."""

    def wait_ready(self, timeout: float = 600.0) -> bool:
        """Block until the backend responds to a health probe or timeout.

        Returns True on success, False on timeout. Never raises.
        """

    def is_alive(self) -> bool:
        """Is the underlying process still up?"""

    def stop(self, timeout: float = 30.0) -> None:
        """Send SIGTERM, then SIGKILL after timeout. Idempotent."""

    def utilization(self) -> dict:
        """Best-effort util numbers for the heartbeat (queue depth, etc.)."""


# ---------------------------------------------------------------------------
# vLLM
# ---------------------------------------------------------------------------


# Blackwell consumer/Spark cuda_capability strings that need the Marlin
# FP8 fallback (no native `cutlass_scaled_mm` FP8 kernel in vLLM 0.17).
# Centralized so `_needs_fp8_marlin_fallback` stays testable in isolation
# from VLLMBackend.start().
_BLACKWELL_FP8_FALLBACK_CAPS = frozenset({"12.0", "12.1"})


def _needs_fp8_marlin_fallback(cuda_capability: str | None) -> bool:
    """Return True iff this chip needs `VLLM_TEST_FORCE_FP8_MARLIN=1`.

    Hopper (9.0) and Ada (8.9) have native cutlass FP8 kernels and run
    faster without the fallback. Blackwell consumer (sm_120/sm_121) ships
    no cutlass FP8 GEMM in vLLM 0.17 + torch 2.10, so the Marlin
    weight-only fallback is the only path that loads weights.

    Unknown / missing cuda_capability → do NOT force the flag. Lets a
    non-CUDA host (CPU vLLM build, or a future Ada/Hopper that doesn't
    report capability through our probe) use the native path. Setting
    the flag on a chip with native FP8 actively hurts perf (BF16 dequant
    each matmul), so the default-off posture is safer than default-on.
    """
    if cuda_capability is None:
        return False
    return cuda_capability in _BLACKWELL_FP8_FALLBACK_CAPS


@dataclass
class VLLMBackend:
    """vLLM serve subprocess. OpenAI-compatible on `base_url`/v1.

    `start()` is non-blocking — it forks the subprocess and returns. Use
    `wait_ready()` to block until the model has loaded weights and the
    health endpoint returns 200.
    """

    card: SpecialistCard
    host: str = "127.0.0.1"
    port: int = 8001
    gpu_memory_utilization: float = 0.55
    max_model_len: int = 8192
    enforce_eager: bool = True
    log_path: Path | None = None
    extra_args: list[str] = field(default_factory=list)
    # cuda_capability is threaded in from NodeProbe so backend chooses the
    # right FP8 kernel path per-chip. None → conservative: assume native
    # FP8 works and only flip to Marlin if explicitly known-Blackwell.
    cuda_capability: str | None = None

    name: str = "vllm"
    _proc: subprocess.Popen | None = field(default=None, init=False, repr=False)
    _adopted_pid: int | None = field(default=None, init=False, repr=False)

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    @property
    def health_url(self) -> str:
        return f"{self.base_url}/health"

    def start(self) -> None:
        """Spawn `vllm serve <model_id>`. Skips if already up.

        If the port is already bound by some other vLLM (e.g., started
        manually for warmup), we adopt it via PID lookup so `stop()`
        still works. This matters because Spark model loads take 2-4
        minutes and we don't want to relaunch on every mesh restart.
        """
        if self._proc is not None and self._proc.poll() is None:
            return
        if self._port_in_use():
            # Try to adopt an existing process listening on our port.
            self._adopted_pid = _find_pid_on_port(self.port)
            if self._adopted_pid:
                return
            raise RuntimeError(
                f"Port {self.port} already bound but no PID found; refusing to launch."
            )

        cmd = [
            shutil.which("vllm") or "vllm",
            "serve",
            self.card.model_id,
            "--host",
            self.host,
            "--port",
            str(self.port),
            "--served-model-name",
            self.card.specialist_id,
            "--max-model-len",
            str(self.max_model_len),
            "--gpu-memory-utilization",
            str(self.gpu_memory_utilization),
            "--dtype",
            "auto",
            "--trust-remote-code",
        ]
        if self.enforce_eager:
            cmd.append("--enforce-eager")
        cmd.extend(self.extra_args)

        env = os.environ.copy()
        env.setdefault("VLLM_LOGGING_LEVEL", "INFO")
        # GB10 sm_121: torch 2.10 only knows up to sm_120; nudge JIT only
        # when we're actually on Blackwell. Hopper/Ada don't need this.
        if _needs_fp8_marlin_fallback(self.cuda_capability):
            env.setdefault("TORCH_CUDA_ARCH_LIST", "12.0")
            # Blackwell consumer (sm_120/sm_121) has no `cutlass_scaled_mm`
            # FP8 kernel in vLLM 0.17 + torch 2.10. Marlin's weight-only
            # FP8 path bypasses it by dequantizing on the fly. The flag is
            # named `_TEST_` but is the documented workaround.
            # See the project history.
            env.setdefault("VLLM_TEST_FORCE_FP8_MARLIN", "1")

        log_file = open(self.log_path, "ab") if self.log_path else None
        try:
            self._proc = subprocess.Popen(
                cmd,
                env=env,
                stdout=log_file or subprocess.DEVNULL,
                stderr=subprocess.STDOUT,
                start_new_session=True,  # so SIGTERM hits the whole group
            )
        finally:
            # Popen dup'd the fd into the child; the parent's handle is no
            # longer needed. Close it so we don't leak a descriptor per
            # backend start — including the case where Popen raised (e.g.
            # vllm not on PATH), where the handle would otherwise dangle.
            if log_file is not None:
                log_file.close()

    def wait_ready(self, timeout: float = 600.0) -> bool:
        """Poll /health until 200 or timeout. vLLM cold load is 2-4 min on Spark."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            if not self.is_alive():
                return False
            try:
                r = httpx.get(self.health_url, timeout=2.0)
                if r.status_code == 200:
                    return True
            except (httpx.HTTPError, OSError):
                pass
            time.sleep(2.0)
        return False

    def is_alive(self) -> bool:
        if self._proc is not None:
            return self._proc.poll() is None
        if self._adopted_pid is not None:
            try:
                os.kill(self._adopted_pid, 0)
                return True
            except OSError:
                return False
        # Nothing started; consider alive iff port answers.
        return self._port_in_use()

    def stop(self, timeout: float = 30.0) -> None:
        """SIGTERM the process group then SIGKILL on timeout. Idempotent."""
        target_pid: int | None = None
        if self._proc is not None and self._proc.poll() is None:
            target_pid = self._proc.pid
        elif self._adopted_pid is not None:
            target_pid = self._adopted_pid
        if target_pid is None:
            return
        try:
            os.killpg(os.getpgid(target_pid), signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            return
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                os.kill(target_pid, 0)
            except OSError:
                self._proc = None
                self._adopted_pid = None
                return
            time.sleep(0.5)
        # Hard kill
        try:
            os.killpg(os.getpgid(target_pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
        self._proc = None
        self._adopted_pid = None

    def utilization(self) -> dict:
        """Pull /metrics → small dict used by heartbeat util fields.

        vLLM exposes Prometheus metrics; we cherry-pick the few the
        mesh registry actually consumes (queue depth, num running). On
        any HTTP failure, return zeros — the registry treats a missing
        signal as "unknown, healthy" rather than failing the node.
        """
        try:
            r = httpx.get(f"{self.base_url}/metrics", timeout=2.0)
            if r.status_code != 200:
                return {"queue_depth": 0, "running": 0}
            return _parse_vllm_metrics(r.text)
        except (httpx.HTTPError, OSError):
            return {"queue_depth": 0, "running": 0}

    def _port_in_use(self) -> bool:
        import socket as _s

        with _s.socket(_s.AF_INET, _s.SOCK_STREAM) as sock:
            sock.settimeout(0.3)
            try:
                sock.connect((self.host, self.port))
                return True
            except (ConnectionRefusedError, OSError):
                return False


def _trailing_float(line: str) -> float | None:
    """Last whitespace-delimited token of a Prometheus line, as a float.

    Returns None when it isn't numeric. vLLM metric line formats drift across
    versions (extra labels, exemplars, a truncated body); this parser must
    fall back to defaults, never raise into the heartbeat path.
    """
    try:
        return float(line.rsplit(" ", 1)[-1])
    except (ValueError, IndexError):
        return None


def _parse_vllm_metrics(body: str) -> dict:
    """Pull the handful of vLLM Prometheus gauges we care about.

    Robust to missing/renamed metrics across vLLM versions: returns 0 if
    the gauge isn't present (or its value is unparseable) rather than failing
    the heartbeat.
    """
    out = {"queue_depth": 0, "running": 0, "gpu_cache_pct": 0.0}
    for line in body.splitlines():
        if line.startswith("#") or not line.strip():
            continue
        # `vllm:num_requests_waiting{...} 3.0` — a malformed value yields the
        # default rather than raising (all three gauges handled uniformly).
        if "vllm:num_requests_waiting" in line:
            v = _trailing_float(line)
            if v is not None:
                out["queue_depth"] = int(v)
        elif "vllm:num_requests_running" in line:
            v = _trailing_float(line)
            if v is not None:
                out["running"] = int(v)
        elif "vllm:gpu_cache_usage_perc" in line:
            v = _trailing_float(line)
            if v is not None:
                out["gpu_cache_pct"] = v
    return out


def _find_pid_on_port(port: int) -> int | None:
    """Look up the PID owning `port` via ss(8). Returns None if not found."""
    try:
        out = subprocess.check_output(
            ["ss", "-ltnp", f"sport = :{port}"], text=True, timeout=2.0
        )
    except (subprocess.SubprocessError, FileNotFoundError):
        return None
    for line in out.splitlines():
        if f":{port} " in line and "pid=" in line:
            # users:(("python3",pid=2551460,fd=27))
            try:
                seg = line.split("pid=", 1)[1]
                pid_str = seg.split(",", 1)[0]
                return int(pid_str)
            except (ValueError, IndexError):
                continue
    return None


# ---------------------------------------------------------------------------
# Null backend (tests + registry-only nodes)
# ---------------------------------------------------------------------------


@dataclass
class NullBackend:
    """No-op backend that pretends to serve. Used by tests + registry-only nodes."""

    card: SpecialistCard
    base_url: str = "http://127.0.0.1:0"
    name: str = "null"
    _running: bool = field(default=False, init=False)

    def start(self) -> None:
        self._running = True

    def wait_ready(self, timeout: float = 1.0) -> bool:
        return self._running

    def is_alive(self) -> bool:
        return self._running

    def stop(self, timeout: float = 1.0) -> None:
        self._running = False

    def utilization(self) -> dict:
        return {"queue_depth": 0, "running": 0}


__all__ = [
    "BaseBackend",
    "NullBackend",
    "VLLMBackend",
]
