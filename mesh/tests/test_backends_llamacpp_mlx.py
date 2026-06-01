"""LlamaCppBackend + MLXBackend tests — faked subprocess + HTTP, no real binaries.

These pin the contract the two own-the-subprocess backends have, mirroring
`test_backends.py` (vLLM) and `test_backends_ollama.py`:

  - construction exposes base_url / health_url / name (no process spawned);
  - start() refuses without the required model field (gguf_path / mlx_repo)
    with a clear error, like OllamaBackend refuses without ollama_tag;
  - MLXBackend.start() refuses off Apple Silicon with a legible error;
  - stop() before start() is an idempotent no-op (NullBackend-shape contract);
  - start() spawns the documented command (faked Popen) and adopts a busy port.

We never spawn a real `llama-server` / `mlx_lm.server` here — Popen and the
health HTTP probe are faked the same way the vLLM/Ollama suites do.
"""

from __future__ import annotations

from typing import Any

import pytest

from mesh.backends import LlamaCppBackend, MLXBackend
from mesh.models import SpecialistCard


def _card(
    *,
    required_backend: str = "llamacpp",
    gguf_path: str | None = "/models/qwen.gguf",
    mlx_repo: str | None = "mlx-community/Qwen2.5-Coder-7B-Instruct-4bit",
) -> SpecialistCard:
    return SpecialistCard(
        model_id="Qwen/Qwen2.5-Coder-7B-Instruct",
        specialist_id="qwen2.5-coder-7b",
        domain="code",
        difficulty_tiers=["medium"],
        required_backend=required_backend,  # type: ignore[arg-type]
        storage_gb=5.0,
        runtime_gb=6.0,
        min_vram_gb=8.0,
        context_window=32768,
        n_layers=28,
        estimated_tps_at={"gb10": 40.0},
        gguf_path=gguf_path,
        mlx_repo=mlx_repo,
    )


class _FakePopen:
    """Records the spawned argv; behaves like a live process until killed."""

    last_cmd: list[str] | None = None

    def __init__(self, cmd: list[str], **_kw: Any) -> None:
        type(self).last_cmd = cmd
        self.pid = 4321
        self._alive = True

    def poll(self) -> int | None:
        return None if self._alive else 0


# ---------------------------------------------------------------------------
# LlamaCppBackend
# ---------------------------------------------------------------------------


def test_llamacpp_construction_exposes_urls_and_name():
    be = LlamaCppBackend(card=_card(), host="127.0.0.1", port=8123)
    assert be.base_url == "http://127.0.0.1:8123"
    assert be.health_url == "http://127.0.0.1:8123/health"
    assert be.name == "llamacpp"


def test_llamacpp_start_refuses_without_gguf_path():
    """No GGUF → clear error, just like Ollama refuses without ollama_tag."""
    be = LlamaCppBackend(card=_card(gguf_path=None), port=9001)
    with pytest.raises(RuntimeError, match="gguf_path"):
        be.start()
    assert not be.is_alive()


def test_llamacpp_stop_when_never_started_is_noop():
    be = LlamaCppBackend(card=_card(), port=9999)
    be.stop()  # must not raise
    assert not be.is_alive()


def test_llamacpp_start_spawns_llama_server(monkeypatch):
    """start() forks `llama-server -m <gguf> --port ... --host ...` (faked Popen)."""
    monkeypatch.setattr("mesh.backends.subprocess.Popen", _FakePopen)
    # Port not in use → real launch path (not adoption).
    monkeypatch.setattr(LlamaCppBackend, "_port_in_use", lambda self: False)
    be = LlamaCppBackend(card=_card(), host="127.0.0.1", port=8137)
    be.start()
    cmd = _FakePopen.last_cmd
    assert cmd is not None
    assert "llama-server" in cmd[0]
    assert "-m" in cmd and cmd[cmd.index("-m") + 1] == "/models/qwen.gguf"
    assert cmd[cmd.index("--port") + 1] == "8137"
    assert cmd[cmd.index("--host") + 1] == "127.0.0.1"
    assert be.is_alive()


def test_llamacpp_adopts_busy_port(monkeypatch):
    """A llama-server already bound to our port is adopted, not relaunched."""
    monkeypatch.setattr(LlamaCppBackend, "_port_in_use", lambda self: True)
    monkeypatch.setattr("mesh.backends._find_pid_on_port", lambda port: 5555)

    def _no_spawn(*_a: Any, **_kw: Any):
        raise AssertionError("must not spawn when adopting a busy port")

    monkeypatch.setattr("mesh.backends.subprocess.Popen", _no_spawn)
    be = LlamaCppBackend(card=_card(), port=8137)
    be.start()
    assert be._adopted_pid == 5555


# ---------------------------------------------------------------------------
# MLXBackend
# ---------------------------------------------------------------------------


def test_mlx_construction_exposes_urls_and_name():
    be = MLXBackend(card=_card(required_backend="mlx"), host="127.0.0.1", port=8123)
    assert be.base_url == "http://127.0.0.1:8123"
    assert be.health_url == "http://127.0.0.1:8123/v1/models"
    assert be.name == "mlx"


def test_mlx_start_refuses_off_apple_silicon(monkeypatch):
    """Non-Darwin/arm64 host → legible 'needs Apple Silicon' error, not a child crash."""
    monkeypatch.setattr("mesh.backends._is_apple_silicon", lambda: False)
    be = MLXBackend(card=_card(required_backend="mlx"), port=9002)
    with pytest.raises(RuntimeError, match="Apple Silicon"):
        be.start()
    assert not be.is_alive()


def test_mlx_start_refuses_without_mlx_repo(monkeypatch):
    """On Apple Silicon but no repo → clear `mlx_repo` error."""
    monkeypatch.setattr("mesh.backends._is_apple_silicon", lambda: True)
    be = MLXBackend(card=_card(required_backend="mlx", mlx_repo=None), port=9002)
    with pytest.raises(RuntimeError, match="mlx_repo"):
        be.start()
    assert not be.is_alive()


def test_mlx_stop_when_never_started_is_noop():
    be = MLXBackend(card=_card(required_backend="mlx"), port=9999)
    be.stop()  # must not raise
    assert not be.is_alive()


def test_mlx_start_spawns_mlx_lm_server(monkeypatch):
    """start() forks `python -m mlx_lm.server --model <repo> ...` on Apple Silicon."""
    monkeypatch.setattr("mesh.backends._is_apple_silicon", lambda: True)
    monkeypatch.setattr("mesh.backends.subprocess.Popen", _FakePopen)
    monkeypatch.setattr(MLXBackend, "_port_in_use", lambda self: False)
    be = MLXBackend(card=_card(required_backend="mlx"), host="127.0.0.1", port=8139)
    be.start()
    cmd = _FakePopen.last_cmd
    assert cmd is not None
    assert cmd[1:3] == ["-m", "mlx_lm.server"]
    assert cmd[cmd.index("--model") + 1] == "mlx-community/Qwen2.5-Coder-7B-Instruct-4bit"
    assert cmd[cmd.index("--port") + 1] == "8139"
    assert be.is_alive()
