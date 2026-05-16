"""GPU coordination tests — probe parsing, local reservations,
cluster-wide placement (the v0.0.6 #46 deliverable)."""

from __future__ import annotations

import json
import os
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from mesh.gpu.cluster import (
    ClusterGpuView,
    NodeGpuView,
    build_cluster_view_from_heartbeats,
    filter_nodes_by_memory,
    pick_best_node,
)
from mesh.gpu.probe import (
    GpuProcess,
    GpuSnapshot,
    _parse_float_or_none,
    _parse_int_or_none,
    probe_gpu,
)
from mesh.gpu.reservations import (
    Reservation,
    ReservationStore,
)


# ---------------------------------------------------------------------------
# probe parser
# ---------------------------------------------------------------------------


def test_parse_int_handles_na_tokens():
    assert _parse_int_or_none("[N/A]") is None
    assert _parse_int_or_none("[Not Supported]") is None
    assert _parse_int_or_none("") is None
    assert _parse_int_or_none("123") == 123
    assert _parse_int_or_none("123 MiB") == 123
    assert _parse_int_or_none("not a number") is None


def test_parse_float_handles_pct_suffix():
    assert _parse_float_or_none("12 %") == 12.0
    assert _parse_float_or_none("0%") == 0.0
    assert _parse_float_or_none("[N/A]") is None


def test_probe_gpu_when_nvidia_smi_absent(monkeypatch):
    monkeypatch.setattr("mesh.gpu.probe.shutil.which", lambda _: None)
    snap = probe_gpu()
    assert snap.nvidia_smi_available is False
    assert snap.processes == []
    assert snap.util_pct is None


def test_probe_gpu_parses_real_output(monkeypatch, tmp_path):
    """Drive probe_gpu against a fake nvidia-smi that returns GB10-like
    output (memory.* are [N/A], compute-apps are real)."""
    fake = tmp_path / "fake-nvidia-smi"
    fake.write_text("""#!/bin/bash
if [[ "$*" == *"--query-gpu"* ]]; then
  echo "5 %, [N/A], [N/A], [N/A]"
elif [[ "$*" == *"--query-compute-apps"* ]]; then
  echo "12345, /path/to/python3, 20480 MiB"
  echo "67890, /path/to/vllm, 30000 MiB"
fi
""")
    fake.chmod(0o755)
    # Mock ps to avoid real lookups (test pids don't exist)
    monkeypatch.setattr("mesh.gpu.probe._enrich_process", lambda p: p)

    snap = probe_gpu(str(fake))
    assert snap.nvidia_smi_available is True
    assert snap.util_pct == 5.0
    assert snap.mem_used_mib is None  # [N/A] → None
    assert len(snap.processes) == 2
    assert snap.processes[0].pid == 12345
    assert snap.processes[0].used_memory_mib == 20480
    assert snap.processes[1].pid == 67890
    # Total proc memory in GB
    assert snap.total_proc_memory_gb == pytest.approx((20480 + 30000) / 1024)


# ---------------------------------------------------------------------------
# Reservations
# ---------------------------------------------------------------------------


def test_reservation_roundtrip():
    now = datetime(2026, 5, 16, 12, 0, 0, tzinfo=timezone.utc)
    r = Reservation(
        reservation_id="abc123",
        user="admin",
        hostname="spark-1",
        gb_requested=60.0,
        started_at=now,
        expires_at=now + timedelta(hours=1),
        purpose="fine-tune",
        pid=12345,
    )
    j = r.to_json()
    back = Reservation.from_json(j)
    assert back == r


def test_reservation_is_expired():
    now = datetime.now(timezone.utc)
    past = Reservation(
        reservation_id="x", user="u", hostname="h", gb_requested=1.0,
        started_at=now - timedelta(hours=2),
        expires_at=now - timedelta(hours=1),
    )
    future = Reservation(
        reservation_id="y", user="u", hostname="h", gb_requested=1.0,
        started_at=now,
        expires_at=now + timedelta(hours=1),
    )
    assert past.is_expired is True
    assert future.is_expired is False
    assert future.remaining_s > 3500


def test_store_reserve_release_roundtrip(tmp_path):
    store = ReservationStore(tmp_path)
    rid = store.reserve(gb_requested=10.0, duration_s=3600, purpose="t")
    active = store.list_active()
    assert len(active) == 1
    assert active[0].reservation_id == rid
    assert active[0].gb_requested == 10.0
    assert store.release(rid) is True
    assert store.list_active() == []


def test_store_release_nonexistent_returns_false(tmp_path):
    store = ReservationStore(tmp_path)
    assert store.release("nope") is False


def test_store_rejects_zero_gb(tmp_path):
    store = ReservationStore(tmp_path)
    with pytest.raises(ValueError):
        store.reserve(gb_requested=0, duration_s=10)


def test_store_rejects_zero_duration(tmp_path):
    store = ReservationStore(tmp_path)
    with pytest.raises(ValueError):
        store.reserve(gb_requested=1, duration_s=0)


def test_store_auto_prunes_expired(tmp_path):
    store = ReservationStore(tmp_path)
    # Manually write an expired reservation
    past_now = datetime.now(timezone.utc) - timedelta(hours=2)
    r = Reservation(
        reservation_id="dead", user="u", hostname="h", gb_requested=1.0,
        started_at=past_now,
        expires_at=past_now + timedelta(hours=1),
    )
    (tmp_path / "dead.json").write_text(json.dumps(r.to_json()))
    active = store.list_active()
    assert active == []
    assert not (tmp_path / "dead.json").exists()  # auto-pruned from disk


def test_store_auto_prunes_dead_pid(tmp_path, monkeypatch):
    """A reservation tied to a dead pid should be pruned."""
    monkeypatch.setattr("mesh.gpu.reservations._pid_alive", lambda _: False)
    store = ReservationStore(tmp_path)
    store.reserve(gb_requested=1, duration_s=3600, pid=99999)
    assert store.list_active() == []


def test_store_total_reserved_gb(tmp_path, monkeypatch):
    """Active reservations are summed correctly."""
    monkeypatch.setattr("mesh.gpu.reservations._pid_alive", lambda _: True)
    store = ReservationStore(tmp_path)
    store.reserve(gb_requested=10, duration_s=3600)
    store.reserve(gb_requested=25, duration_s=3600)
    assert store.total_reserved_gb() == 35.0


# ---------------------------------------------------------------------------
# Cluster scheduling
# ---------------------------------------------------------------------------


def _snap(util=10.0, total_mib=128*1024, procs_gb=None):
    procs = []
    procs_gb = procs_gb or []
    for i, g in enumerate(procs_gb):
        procs.append(GpuProcess(
            pid=1000+i, process_name="x", used_memory_mib=int(g*1024),
        ))
    return GpuSnapshot(
        probed_at=datetime.now(timezone.utc),
        util_pct=util,
        mem_used_mib=None, mem_free_mib=None, mem_total_mib=total_mib,
        processes=procs,
    )


def _view(*nodes_kwargs):
    return ClusterGpuView(
        snapshot_ts=datetime.now(timezone.utc),
        nodes={n["node_id"]: NodeGpuView(**n) for n in nodes_kwargs},
    )


def test_node_view_free_after_reservations():
    n = NodeGpuView(
        node_id="n1", friendly_name="spark-1",
        snapshot=_snap(total_mib=128*1024, procs_gb=[20]),  # 20 GB in use
        active_reservations=[Reservation(
            reservation_id="r", user="u", hostname="h", gb_requested=30,
            started_at=datetime.now(timezone.utc),
            expires_at=datetime.now(timezone.utc) + timedelta(hours=1),
        )],
        declared_total_gb=128.0,
    )
    # max(used=20, reserved=30) = 30 → free = 128 - 30 = 98
    assert n.free_gb_after_reservations == pytest.approx(98.0)


def test_filter_nodes_by_memory_excludes_unknown_total():
    """Node with no declared_total_gb AND no mem_total_mib → excluded."""
    n_unknown = NodeGpuView(
        node_id="n1", friendly_name="unknown",
        snapshot=_snap(total_mib=None),
    )
    n_known = NodeGpuView(
        node_id="n2", friendly_name="known",
        snapshot=_snap(total_mib=128*1024),
    )
    view = _view(
        {"node_id": "n1", "friendly_name": "u", "snapshot": n_unknown.snapshot},
        {"node_id": "n2", "friendly_name": "k", "snapshot": n_known.snapshot},
    )
    eligible = filter_nodes_by_memory(view, gb_required=10.0)
    assert {n.node_id for n in eligible} == {"n2"}


def test_pick_best_node_returns_lowest_load():
    """Two eligible nodes; pick the one with more headroom + fewer reservations."""
    n_busy = NodeGpuView(
        node_id="busy", friendly_name="busy",
        snapshot=_snap(total_mib=128*1024, procs_gb=[60]),  # 60 GB used
        declared_total_gb=128.0,
    )
    n_quiet = NodeGpuView(
        node_id="quiet", friendly_name="quiet",
        snapshot=_snap(total_mib=128*1024, procs_gb=[10]),  # 10 GB used
        declared_total_gb=128.0,
    )
    view = ClusterGpuView(
        snapshot_ts=datetime.now(timezone.utc),
        nodes={"busy": n_busy, "quiet": n_quiet},
    )
    result = pick_best_node(view, gb_requested=40.0)
    assert result.ok is True
    assert result.chosen_node_id == "quiet"


def test_pick_best_node_respects_hardware_tags():
    n_no_fp4 = NodeGpuView(
        node_id="hopper", friendly_name="h100",
        snapshot=_snap(total_mib=80*1024),
        declared_total_gb=80.0, hardware_tags=["hopper"],
    )
    n_fp4 = NodeGpuView(
        node_id="blackwell", friendly_name="gb10",
        snapshot=_snap(total_mib=128*1024),
        declared_total_gb=128.0, hardware_tags=["blackwell", "fp4"],
    )
    view = ClusterGpuView(
        snapshot_ts=datetime.now(timezone.utc),
        nodes={n_no_fp4.node_id: n_no_fp4, n_fp4.node_id: n_fp4},
    )
    result = pick_best_node(
        view, gb_requested=20.0, require_hardware_tags=["fp4"],
    )
    assert result.ok is True
    assert result.chosen_node_id == "blackwell"
    assert "hopper" in result.rejected


def test_pick_best_node_no_fit_returns_structured_reason():
    n_small = NodeGpuView(
        node_id="small", friendly_name="small",
        snapshot=_snap(total_mib=16*1024),
        declared_total_gb=16.0,
    )
    view = ClusterGpuView(
        snapshot_ts=datetime.now(timezone.utc),
        nodes={n_small.node_id: n_small},
    )
    result = pick_best_node(view, gb_requested=100.0)
    assert result.ok is False
    assert "no eligible node" in result.reason
    assert "small" in result.rejected


def test_pick_best_node_avoid_excludes_node():
    n1 = NodeGpuView(
        node_id="n1", friendly_name="n1",
        snapshot=_snap(total_mib=128*1024), declared_total_gb=128.0,
    )
    n2 = NodeGpuView(
        node_id="n2", friendly_name="n2",
        snapshot=_snap(total_mib=128*1024), declared_total_gb=128.0,
    )
    view = ClusterGpuView(
        snapshot_ts=datetime.now(timezone.utc),
        nodes={n1.node_id: n1, n2.node_id: n2},
    )
    result = pick_best_node(view, gb_requested=10.0, avoid_nodes={"n1"})
    assert result.chosen_node_id == "n2"


def test_build_cluster_view_from_heartbeats_skips_no_gpu_field():
    hb_no_gpu = {"node_id": "n1", "heartbeat": {"node_id": "n1"}}
    hb_with_gpu = {
        "node_id": "n2",
        "friendly_name": "spark",
        "gpu": {
            "snapshot": {
                "probed_at": "2026-05-16T12:00:00+00:00",
                "util_pct": 5.0,
                "mem_total_mib": 128*1024,
                "processes": [],
                "nvidia_smi_available": True,
            },
            "reservations": [],
            "declared_total_gb": 128.0,
            "hardware_tags": ["blackwell"],
        },
    }
    view = build_cluster_view_from_heartbeats([hb_no_gpu, hb_with_gpu])
    assert set(view.nodes.keys()) == {"n2"}
    assert view.nodes["n2"].hardware_tags == ["blackwell"]
    assert view.nodes["n2"].total_gb == 128.0
