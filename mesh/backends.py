"""Backend abstraction — spec §9.

A `BaseBackend` knows how to (a) spawn a process that serves a
specialist over an OpenAI-compatible HTTP endpoint, (b) report
liveness + basic utilization, and (c) terminate cleanly.

Today ships `VLLMBackend`, `OllamaBackend`, `LlamaCppBackend`,
`MLXBackend`, and a `NullBackend` for tests.
The Ollama path is what unlocks every LocalLLaMA-style box (Mac, AMD,
Windows, small consumer NVIDIA) that the catalog's vLLM-only cards
silently exclude — `OllamaBackend` adopts the user's running Ollama
daemon and serves any specialist whose card sets `ollama_tag`.
`LlamaCppBackend` owns a `llama-server` subprocess for any box with a
GGUF (`gguf_path`) — the CPU-only / no-CUDA-no-Metal path. `MLXBackend`
owns an `mlx_lm.server` subprocess on Apple Silicon (`mlx_repo`) for
native Metal acceleration.

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
import sys
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
# Ollama
# ---------------------------------------------------------------------------


# Ollama's well-known port. Override per-backend via constructor; mesh's
# CLI per-specialist port (`--base-port` + i) is informational for Ollama
# because the daemon multiplexes all loaded models on one port.
DEFAULT_OLLAMA_PORT = 11434

# Models the user has running but the catalog hasn't yet pinned — when a
# pre-warm `/api/generate` returns inside this many seconds, we treat the
# model as already-loaded rather than as a cold pull.
_OLLAMA_WARM_TIMEOUT_S = 5.0


@dataclass
class OllamaBackend:
    """Adopt the local Ollama daemon and serve `card.ollama_tag` through it.

    Ollama is a single multi-model daemon (one port, models loaded on
    demand) — the opposite of vLLM's one-process-per-model. So this
    backend does NOT spawn or kill an Ollama server; it expects one
    already running on `host:port` (the typical homelab / LocalLLaMA
    setup — `systemctl --user start ollama` or the desktop install), and
    its job is to (1) make sure the daemon is reachable, (2) make sure
    `ollama_tag` is pulled, (3) keep it pre-warmed so the first heartbeat
    after `wait_ready()` sees it loaded, and (4) on `stop()` release the
    VRAM via `keep_alive: 0` *without* tearing down the daemon a human
    user (or another mesh node) might still need.

    Tailnet exposure: the daemon defaults to `127.0.0.1`. For other
    hosts on the tailnet to reach it, launch Ollama with
    `OLLAMA_HOST=0.0.0.0:11434 ollama serve` (or set the env var in the
    service unit). Mesh advertises the configured `host:port` and trusts
    the caller to have done the binding — same posture as `--tailnet
    --bind-host 0.0.0.0` on the vLLM path.

    The OpenAI-compat endpoints (`/v1/chat/completions`, `/v1/models`)
    are served at `base_url` so the rest of the mesh (selectors,
    routers, dashboards) treats Ollama nodes the same as vLLM ones.
    """

    card: SpecialistCard
    host: str = "127.0.0.1"
    port: int = DEFAULT_OLLAMA_PORT
    # How long to keep the model loaded after the last request — Ollama's
    # default is "5m"; "30m" matches the heartbeat cadence headroom so a
    # node with one Ollama specialist doesn't churn evict/load between
    # quiet windows.
    keep_alive: str = "30m"
    log_path: Path | None = None

    name: str = "ollama"
    _started: bool = field(default=False, init=False, repr=False)

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    @property
    def health_url(self) -> str:
        # `GET /` on Ollama returns 200 "Ollama is running" when up.
        return f"{self.base_url}/"

    def start(self) -> None:
        """Validate daemon + ensure model is pulled. Non-blocking on weights.

        We do NOT spawn `ollama serve` ourselves — the typical homelab
        already has it running, and starting/stopping it from mesh would
        race with the user's own sessions. Instead:

        1. Confirm the daemon is reachable on `host:port` (else raise a
           clear error pointing the operator at `ollama serve`).
        2. Confirm `ollama_tag` is set on the card (else raise — the
           catalog needs to map the specialist to an Ollama tag).
        3. Best-effort pull if the tag isn't in `/api/tags` yet. The
           pull is fire-and-forget here; `wait_ready()` is where we
           actually block on it appearing.
        """
        if self.card.ollama_tag is None:
            raise RuntimeError(
                f"Ollama backend needs `ollama_tag` on specialist card "
                f"{self.card.specialist_id!r} (e.g. 'qwen2.5-coder:7b'). "
                f"Set it in the card TOML and re-run."
            )
        if not self._daemon_alive():
            raise RuntimeError(
                f"Ollama daemon not reachable at {self.base_url}. Start it with "
                f"`OLLAMA_HOST={self.host}:{self.port} ollama serve` "
                f"(or the systemd / desktop service) and try again."
            )
        if not self._model_pulled():
            # Kick a pull in the background; wait_ready will poll for it.
            # We deliberately don't block here so mesh's parallel `start()`
            # loop (across N specialists) doesn't serialize on one pull.
            self._kick_pull()
        self._started = True

    def wait_ready(self, timeout: float = 600.0) -> bool:
        """Poll until the model is pulled and the daemon answers `/v1/models`.

        First-time pulls of a ~4 GB GGUF on a 100 Mbit link can run several
        minutes; the default timeout matches `VLLMBackend.wait_ready` so a
        mixed-engine mesh start has one knob.
        """
        if not self._started:
            return False
        if self.card.ollama_tag is None:
            return False
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self._daemon_alive() and self._model_pulled():
                # Pre-warm: a no-op `/api/generate` loads the model into VRAM
                # so the first routed request doesn't pay the cold-start cost.
                if self._prewarm():
                    return True
            time.sleep(2.0)
        return False

    def is_alive(self) -> bool:
        return self._daemon_alive() if self._started else False

    def stop(self, timeout: float = 30.0) -> None:
        """Release the model from VRAM, leave the daemon running. Idempotent.

        `keep_alive: 0` tells Ollama to unload immediately on the next
        completion of any in-flight request. We do NOT SIGTERM the
        daemon — there may be other Ollama clients on this box (the
        user, other mesh nodes, the desktop app).
        """
        if not self._started:
            return
        if self.card.ollama_tag is not None:
            try:
                httpx.post(
                    f"{self.base_url}/api/generate",
                    json={
                        "model": self.card.ollama_tag,
                        "prompt": "",
                        "keep_alive": 0,
                        "stream": False,
                    },
                    timeout=timeout,
                )
            except (httpx.HTTPError, OSError):
                # Daemon already down, or unloading raised — either way,
                # the VRAM will be released by the daemon's own eviction.
                pass
        self._started = False

    def utilization(self) -> dict:
        """Best-effort util via `/api/ps`. Ollama exposes no queue gauge.

        `/api/ps` returns the currently-loaded models with `size_vram`.
        We surface `running` = 1 iff `ollama_tag` is among them, and
        `gpu_cache_pct` as a coarse VRAM-share proxy when it's there;
        `queue_depth` stays 0 (Ollama doesn't publish one).
        """
        out = {"queue_depth": 0, "running": 0, "gpu_cache_pct": 0.0}
        if self.card.ollama_tag is None:
            return out
        try:
            r = httpx.get(f"{self.base_url}/api/ps", timeout=2.0)
            if r.status_code != 200:
                return out
            data = r.json()
        except (httpx.HTTPError, OSError, ValueError):
            return out
        for entry in data.get("models", []) or []:
            if entry.get("name") == self.card.ollama_tag or entry.get("model") == self.card.ollama_tag:
                out["running"] = 1
                size_vram = entry.get("size_vram")
                size_total = entry.get("size")
                if size_vram and size_total:
                    try:
                        out["gpu_cache_pct"] = float(size_vram) / float(size_total)
                    except (TypeError, ValueError, ZeroDivisionError):
                        pass
                break
        return out

    # --- internals ---

    def _daemon_alive(self) -> bool:
        try:
            r = httpx.get(self.health_url, timeout=2.0)
            return r.status_code == 200
        except (httpx.HTTPError, OSError):
            return False

    def _model_pulled(self) -> bool:
        """Is `ollama_tag` already in `/api/tags`? No pull required."""
        try:
            r = httpx.get(f"{self.base_url}/api/tags", timeout=2.0)
            if r.status_code != 200:
                return False
            data = r.json()
        except (httpx.HTTPError, OSError, ValueError):
            return False
        tags = {m.get("name") for m in (data.get("models") or [])}
        tags.update(m.get("model") for m in (data.get("models") or []) if m.get("model"))
        return self.card.ollama_tag in tags

    def _kick_pull(self) -> None:
        """Fire-and-forget `/api/pull`. wait_ready polls until it lands."""
        try:
            # `stream: false` lets us short-poll the result instead of holding
            # this connection open across a multi-minute download.
            httpx.post(
                f"{self.base_url}/api/pull",
                json={"name": self.card.ollama_tag, "stream": False},
                timeout=2.0,
            )
        except (httpx.HTTPError, OSError):
            # The pull is best-effort; wait_ready will retry via `_model_pulled`.
            pass

    def _prewarm(self) -> bool:
        """Touch `/api/generate` with an empty prompt so Ollama loads the model.

        Returns True if the daemon accepts the request (model is now hot),
        False on any transport error. We use `keep_alive` from the
        backend to set the live window.
        """
        try:
            r = httpx.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.card.ollama_tag,
                    "prompt": "",
                    "keep_alive": self.keep_alive,
                    "stream": False,
                },
                timeout=_OLLAMA_WARM_TIMEOUT_S,
            )
            return r.status_code == 200
        except (httpx.HTTPError, OSError):
            return False


# ---------------------------------------------------------------------------
# llama.cpp (llama-server)
# ---------------------------------------------------------------------------


@dataclass
class LlamaCppBackend:
    """llama.cpp's `llama-server` subprocess. OpenAI-compatible on `base_url`/v1.

    Same own-the-subprocess shape as `VLLMBackend`: `start()` forks
    `llama-server -m <gguf> --port <port> --host <host>` and returns; the
    server exposes `/v1/chat/completions` + `/health` (200 once weights are
    loaded). If the port is already bound (a llama-server launched manually
    for warmup), we adopt it via PID lookup so `stop()` still works.

    The model is a GGUF named by `card.gguf_path` — a local path or a
    `repo:file` HF identifier that llama-server can fetch. Missing → a clear
    error (mirrors OllamaBackend's `ollama_tag` requirement); `build_backend`
    catches the missing field up front and falls back to NullBackend.

    llama.cpp is the CPU-only / small-box path the planner recommends when
    there's no CUDA and no Apple Metal — it runs anywhere a GGUF + a CPU do.
    """

    card: SpecialistCard
    host: str = "127.0.0.1"
    port: int = 8001
    n_gpu_layers: int = 0  # 0 = CPU-only; bump for partial GPU offload
    log_path: Path | None = None
    extra_args: list[str] = field(default_factory=list)

    name: str = "llamacpp"
    _proc: subprocess.Popen | None = field(default=None, init=False, repr=False)
    _adopted_pid: int | None = field(default=None, init=False, repr=False)

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    @property
    def health_url(self) -> str:
        return f"{self.base_url}/health"

    def start(self) -> None:
        """Spawn `llama-server -m <gguf>`. Skips if already up; adopts a busy port.

        Like vLLM, this is non-blocking — use `wait_ready()` to block until
        `/health` returns 200. Requires `card.gguf_path`; raises otherwise.
        """
        if self.card.gguf_path is None:
            raise RuntimeError(
                f"llama.cpp backend needs `gguf_path` on specialist card "
                f"{self.card.specialist_id!r} (a local GGUF path or a "
                f"'repo:file' HF identifier). Set it in the card TOML and re-run."
            )
        if self._proc is not None and self._proc.poll() is None:
            return
        if self._port_in_use():
            self._adopted_pid = _find_pid_on_port(self.port)
            if self._adopted_pid:
                return
            raise RuntimeError(
                f"Port {self.port} already bound but no PID found; refusing to launch."
            )

        cmd = [
            shutil.which("llama-server") or "llama-server",
            "-m",
            self.card.gguf_path,
            "--host",
            self.host,
            "--port",
            str(self.port),
            "--n-gpu-layers",
            str(self.n_gpu_layers),
        ]
        cmd.extend(self.extra_args)

        env = os.environ.copy()
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
            # Close the parent's copy of the log fd even if Popen raised
            # (e.g. llama-server not on PATH), so we don't leak a descriptor.
            if log_file is not None:
                log_file.close()

    def wait_ready(self, timeout: float = 600.0) -> bool:
        """Poll /health until 200 or timeout. GGUF loads are usually under a minute."""
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
        try:
            os.killpg(os.getpgid(target_pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
        self._proc = None
        self._adopted_pid = None

    def utilization(self) -> dict:
        """Best-effort util via `/health`. llama-server publishes no queue gauge.

        Newer llama-server `/health` returns a JSON body with a `slots_idle`
        /`slots_processing` split when launched with `--metrics`; we surface
        `running` from it when present and otherwise fall back to zeros, never
        raising into the heartbeat path.
        """
        out = {"queue_depth": 0, "running": 0}
        try:
            r = httpx.get(self.health_url, timeout=2.0)
            if r.status_code != 200:
                return out
            data = r.json()
        except (httpx.HTTPError, OSError, ValueError):
            return out
        if isinstance(data, dict):
            processing = data.get("slots_processing")
            if isinstance(processing, (int, float)):
                out["running"] = int(processing)
        return out

    def _port_in_use(self) -> bool:
        import socket as _s

        with _s.socket(_s.AF_INET, _s.SOCK_STREAM) as sock:
            sock.settimeout(0.3)
            try:
                sock.connect((self.host, self.port))
                return True
            except (ConnectionRefusedError, OSError):
                return False


# ---------------------------------------------------------------------------
# MLX (mlx_lm.server) — Apple Silicon native
# ---------------------------------------------------------------------------


def _is_apple_silicon() -> bool:
    """True iff we're on macOS / arm64 (Apple Metal). MLX needs both.

    mlx_lm only runs on Apple Silicon; on any other platform the import
    itself fails. We gate at the backend so a mixed catalog on a Linux box
    degrades to a clear error rather than a confusing subprocess crash.
    """
    import platform

    return platform.system() == "Darwin" and platform.machine() == "arm64"


@dataclass
class MLXBackend:
    """`mlx_lm.server` subprocess on Apple Silicon. OpenAI-compatible on `base_url`/v1.

    Own-the-subprocess shape: mlx_lm.server is launched per-model
    (`python -m mlx_lm.server --model <repo> --port <port>`), so — like vLLM
    and unlike Ollama — this backend spawns and owns its process. The server
    serves `/v1/chat/completions` + `/v1/models`; we use `/v1/models` as the
    readiness probe (mlx_lm.server exposes no dedicated `/health`).

    The model is a HF repo named by `card.mlx_repo` (typically an
    `mlx-community/...` repo). Missing → a clear error (mirrors OllamaBackend's
    `ollama_tag`); `build_backend` catches it up front and falls back to
    NullBackend.

    Apple-only: `start()` refuses on non-Darwin/arm64 hosts so the failure is
    a legible "MLX needs Apple Silicon" rather than a Python ImportError deep
    in the child process.
    """

    card: SpecialistCard
    host: str = "127.0.0.1"
    port: int = 8001
    log_path: Path | None = None
    extra_args: list[str] = field(default_factory=list)

    name: str = "mlx"
    _proc: subprocess.Popen | None = field(default=None, init=False, repr=False)
    _adopted_pid: int | None = field(default=None, init=False, repr=False)

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    @property
    def health_url(self) -> str:
        # mlx_lm.server has no dedicated /health; /v1/models answers 200 once up.
        return f"{self.base_url}/v1/models"

    def start(self) -> None:
        """Spawn `python -m mlx_lm.server --model <repo>`. Apple Silicon only.

        Non-blocking — use `wait_ready()` to block on `/v1/models`. Requires
        `card.mlx_repo` and a Darwin/arm64 host; raises a clear error otherwise.
        Adopts an already-bound port like vLLM.
        """
        if not _is_apple_silicon():
            raise RuntimeError(
                f"MLX backend requires Apple Silicon (macOS / arm64); this host "
                f"is not. Use an 'ollama' or 'llamacpp' card for specialist "
                f"{self.card.specialist_id!r} on this platform."
            )
        if self.card.mlx_repo is None:
            raise RuntimeError(
                f"MLX backend needs `mlx_repo` on specialist card "
                f"{self.card.specialist_id!r} (an mlx-community HF repo, e.g. "
                f"'mlx-community/Qwen2.5-Coder-7B-Instruct-4bit'). Set it in the "
                f"card TOML and re-run."
            )
        if self._proc is not None and self._proc.poll() is None:
            return
        if self._port_in_use():
            self._adopted_pid = _find_pid_on_port(self.port)
            if self._adopted_pid:
                return
            raise RuntimeError(
                f"Port {self.port} already bound but no PID found; refusing to launch."
            )

        cmd = [
            sys.executable,
            "-m",
            "mlx_lm.server",
            "--model",
            self.card.mlx_repo,
            "--host",
            self.host,
            "--port",
            str(self.port),
        ]
        cmd.extend(self.extra_args)

        env = os.environ.copy()
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
            # Close the parent's copy of the log fd even if Popen raised
            # (e.g. python missing mlx_lm), so we don't leak a descriptor.
            if log_file is not None:
                log_file.close()

    def wait_ready(self, timeout: float = 600.0) -> bool:
        """Poll /v1/models until 200 or timeout. MLX weight loads are quick."""
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
        try:
            os.killpg(os.getpgid(target_pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
        self._proc = None
        self._adopted_pid = None

    def utilization(self) -> dict:
        """mlx_lm.server publishes no util gauges; report zeros (alive iff up).

        The registry treats a missing signal as "unknown, healthy", so a
        flat zeros dict is the right floor — never raises into the heartbeat.
        """
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


# ---------------------------------------------------------------------------
# External / static endpoint (mesh USES it, does not own its lifecycle)
# ---------------------------------------------------------------------------


@dataclass
class ExternalBackend:
    """An already-running, externally-managed OpenAI-compatible endpoint.

    Unlike every other backend, this one owns NO subprocess: the server (e.g. a
    24/7 systemd vLLM, or a model on another host) is managed outside the mesh.
    `start()`/`stop()` are deliberate no-ops — the mesh must never spawn or kill
    it. The router only needs `base_url` + a liveness probe, which is all this
    provides. Wired from a card with `required_backend = "external"` +
    `static_base_url`.
    """

    card: SpecialistCard
    base_url: str = "http://127.0.0.1:8011"
    name: str = "external"

    @property
    def health_url(self) -> str:
        return f"{self.base_url}/health"

    def start(self) -> None:
        # No-op: the endpoint is externally managed (systemd / another host).
        return None

    def is_alive(self) -> bool:
        # vLLM exposes /health at the root; /v1/models is the OpenAI-spec
        # fallback for servers that don't. Either 200 = alive. Never raises.
        for path in ("/health", "/v1/models"):
            try:
                r = httpx.get(f"{self.base_url}{path}", timeout=3.0)
                if r.status_code == 200:
                    return True
            except Exception:
                continue
        return False

    def wait_ready(self, timeout: float = 600.0) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.is_alive():
                return True
            time.sleep(2.0)
        return False

    def stop(self, timeout: float = 30.0) -> None:
        # No-op: NEVER terminate an externally-managed service.
        return None

    def utilization(self) -> dict:
        try:
            r = httpx.get(f"{self.base_url}/metrics", timeout=2.0)
            if r.status_code == 200:
                return {"running": 1}
        except Exception:
            pass
        return {"queue_depth": 0, "running": 0}


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
    "DEFAULT_OLLAMA_PORT",
    "ExternalBackend",
    "LlamaCppBackend",
    "MLXBackend",
    "NullBackend",
    "OllamaBackend",
    "VLLMBackend",
]
