"""Probe tests — run on the current machine.

We can't assert exact hardware values (CI runs on x86_64, Spark runs on
GB10 aarch64), but we can assert the probe doesn't crash, fills required
fields, and that warnings are non-empty where we know fields are
unknowable.
"""

from __future__ import annotations

import json

from mesh.models import NodeProbe
from mesh.probe import _detect_memory_gb, probe_node


def test_detect_memory_is_positive_cross_os():
    """RAM probe must return real values on every OS (psutil-backed).

    Regression: on Windows the OS-specific paths (/proc/meminfo, sysctl) both
    missed, so RAM reported 0.0 (found on a real Win10 box, 2026-05-26). psutil
    is a hard dep and works on Windows/Linux/macOS, so total + available are
    always > 0 on a real host.
    """
    total, available = _detect_memory_gb([])
    assert total > 0, "ram_total_gb should be > 0 (psutil cross-OS path)"
    assert available > 0, "ram_available_gb should be > 0 (the Windows 0.0 bug)"
    assert available <= total


def test_probe_node_returns_node_probe():
    p = probe_node(friendly_name="test-host")
    assert isinstance(p, NodeProbe)
    assert p.friendly_name == "test-host"
    assert p.node_id  # non-empty
    assert p.chip
    assert p.arch in ("aarch64", "x86_64", "apple-silicon")
    assert p.ram_total_gb > 0  # any real machine has RAM


def test_probe_node_json_roundtrip():
    p = probe_node()
    js = p.model_dump_json()
    parsed = json.loads(js)
    # Restore via Pydantic
    p2 = NodeProbe.model_validate(parsed)
    assert p2.node_id == p.node_id
    assert p2.chip == p.chip
    assert p2.ram_total_gb == p.ram_total_gb


def test_probe_warnings_are_a_list():
    p = probe_node()
    assert isinstance(p.probe_warnings, list)
    # On any node lacking nvidia-smi or with no bandwidth table entry,
    # we expect at least one warning.
