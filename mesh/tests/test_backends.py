"""Backend abstraction tests — VLLMBackend's metric parser + NullBackend lifecycle.

We don't spawn a real vLLM in pytest (4-min cold start, 70GB RAM); the
real-serving integration is in `test_integration_vllm.py` behind a
`VLLM_AVAILABLE` env guard.
"""

from __future__ import annotations

from mesh.backends import (
    NullBackend,
    VLLMBackend,
    _needs_fp8_marlin_fallback,
    _parse_vllm_metrics,
)
from mesh.models import SpecialistCard


def _card() -> SpecialistCard:
    """Minimal card; we don't actually serve it."""
    return SpecialistCard(
        model_id="test/model",
        specialist_id="test-spec",
        domain="code",
        difficulty_tiers=["easy"],
        required_backend="vllm",
        storage_gb=1.0,
        runtime_gb=2.0,
        min_vram_gb=4.0,
        context_window=2048,
        n_layers=2,
        estimated_tps_at={"gb10": 10.0},
    )


def test_parse_vllm_metrics_basic():
    body = """
# HELP vllm:num_requests_waiting Number of requests waiting in queue.
# TYPE vllm:num_requests_waiting gauge
vllm:num_requests_waiting{model_name="x"} 3.0
vllm:num_requests_running{model_name="x"} 1.0
vllm:gpu_cache_usage_perc{model_name="x"} 0.42
"""
    out = _parse_vllm_metrics(body)
    assert out["queue_depth"] == 3
    assert out["running"] == 1
    assert out["gpu_cache_pct"] == 0.42


def test_parse_vllm_metrics_missing_gauges_returns_zeros():
    """If vLLM renames metrics, heartbeats should still emit (zeros, not crash)."""
    out = _parse_vllm_metrics("# no useful metrics here\n")
    assert out == {"queue_depth": 0, "running": 0, "gpu_cache_pct": 0.0}


def test_parse_vllm_metrics_malformed_value_does_not_raise():
    """A non-numeric / truncated gauge value yields the default, never raises.

    The queue_depth / running branches previously did an unguarded
    int(float(...)), so a value drift or a truncated /metrics body raised
    ValueError up through utilization() into the heartbeat path.
    """
    body = (
        'vllm:num_requests_waiting{model_name="m"} notanumber\n'
        'vllm:num_requests_running{model_name="m"} \n'  # truncated: no value
    )
    out = _parse_vllm_metrics(body)  # must not raise
    assert out == {"queue_depth": 0, "running": 0, "gpu_cache_pct": 0.0}


def test_parse_vllm_metrics_keeps_valid_when_one_value_malformed():
    body = (
        'vllm:num_requests_waiting{model_name="m"} 5.0\n'
        'vllm:num_requests_running{model_name="m"} bogus\n'
    )
    out = _parse_vllm_metrics(body)
    assert out["queue_depth"] == 5  # valid gauge still parsed
    assert out["running"] == 0  # malformed one falls back to default


def test_vllm_backend_base_url_and_health_url():
    be = VLLMBackend(card=_card(), host="127.0.0.1", port=8123)
    assert be.base_url == "http://127.0.0.1:8123"
    assert be.health_url == "http://127.0.0.1:8123/health"
    assert be.name == "vllm"


def test_vllm_backend_stop_when_never_started_is_noop():
    """Idempotent stop is critical for graceful shutdown ordering."""
    be = VLLMBackend(card=_card(), port=9999)
    be.stop()  # should not raise
    assert not be.is_alive()


def test_fp8_marlin_fallback_blackwell_consumer():
    """sm_120 / sm_121 → flag set (no native cutlass FP8 GEMM in vLLM 0.17)."""
    assert _needs_fp8_marlin_fallback("12.0") is True
    assert _needs_fp8_marlin_fallback("12.1") is True


def test_fp8_marlin_fallback_hopper_ada_native():
    """Hopper / Ada have native FP8 kernels; forcing Marlin actively hurts perf."""
    assert _needs_fp8_marlin_fallback("9.0") is False  # Hopper H100/H200
    assert _needs_fp8_marlin_fallback("8.9") is False  # Ada L40/L4
    assert _needs_fp8_marlin_fallback("8.0") is False  # Ampere A100


def test_fp8_marlin_fallback_unknown_capability():
    """Unknown / CPU host → don't force Marlin (default-off is the safer floor)."""
    assert _needs_fp8_marlin_fallback(None) is False
    assert _needs_fp8_marlin_fallback("") is False
    assert _needs_fp8_marlin_fallback("13.0") is False  # future chip


def test_vllm_backend_carries_cuda_capability_field():
    """Field default is None; explicit value propagates to .start() env logic."""
    be = VLLMBackend(card=_card(), port=8123)
    assert be.cuda_capability is None
    be2 = VLLMBackend(card=_card(), port=8124, cuda_capability="12.1")
    assert be2.cuda_capability == "12.1"


def test_null_backend_lifecycle():
    """NullBackend is what unsupported backends fall back to in v0.0.2."""
    be = NullBackend(card=_card())
    assert not be.is_alive()
    be.start()
    assert be.is_alive()
    assert be.wait_ready(timeout=0.01)
    be.stop()
    assert not be.is_alive()
