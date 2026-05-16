"""Cooperative GPU reservations — file-based, no service required.

A reservation is a polite signal that this user intends to consume N GB
of GPU memory for D duration. Other mesh participants check the
reservation directory before launching new workloads. Nothing in the
kernel enforces it — this is a SOCIAL contract surfaced via
`mesh-gpu status`.

Storage: $XDG_RUNTIME_DIR/spark-gpu/ if present and writable, else
/tmp/spark-gpu/. One file per reservation: <reservation_id>.json.

Expired reservations are auto-pruned on read. A reservation whose
process died (pid no longer running) is also pruned — defends against
"reserved but workload crashed" leaks.

Thread-safe within a single process via an internal lock; cross-
process safety via atomic file rename (.tmp → final). No file locks
across processes — reservations are short-lived enough that a torn
read is harmless (next status call rebuilds the view).
"""

from __future__ import annotations

import getpass
import json
import os
import secrets
import socket
import threading
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional


def _default_reservations_dir() -> Path:
    """$XDG_RUNTIME_DIR/spark-gpu (typically /run/user/UID/spark-gpu) when
    available + writable; /tmp/spark-gpu fallback."""
    xdg = os.environ.get("XDG_RUNTIME_DIR")
    if xdg:
        candidate = Path(xdg) / "spark-gpu"
        try:
            candidate.mkdir(parents=True, exist_ok=True)
            # Confirm we can actually write
            probe = candidate / ".probe"
            probe.touch()
            probe.unlink()
            return candidate
        except (OSError, PermissionError):
            pass
    fallback = Path("/tmp/spark-gpu")
    fallback.mkdir(parents=True, exist_ok=True)
    return fallback


DEFAULT_RESERVATIONS_DIR: Path = _default_reservations_dir()


@dataclass
class Reservation:
    """One claim on the GPU.

    `pid` is optional but recommended — when set, the reservation is
    auto-pruned if the process disappears. Lets a crashed workload
    not block the GPU indefinitely until expires_at.
    """

    reservation_id: str
    user: str
    hostname: str
    gb_requested: float
    started_at: datetime
    expires_at: datetime
    purpose: str = ""
    pid: Optional[int] = None

    def to_json(self) -> dict:
        d = asdict(self)
        d["started_at"] = self.started_at.isoformat()
        d["expires_at"] = self.expires_at.isoformat()
        return d

    @classmethod
    def from_json(cls, d: dict) -> "Reservation":
        d = dict(d)
        d["started_at"] = datetime.fromisoformat(d["started_at"])
        d["expires_at"] = datetime.fromisoformat(d["expires_at"])
        return cls(**d)

    @property
    def is_expired(self) -> bool:
        return datetime.now(timezone.utc) >= self.expires_at

    @property
    def remaining_s(self) -> float:
        return max(0.0, (self.expires_at - datetime.now(timezone.utc)).total_seconds())


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, PermissionError):
        return False
    except OSError:
        # Other unexpected errors — assume alive rather than killing the reservation
        return True


class ReservationStore:
    """File-backed reservation store.

    Usage:
        store = ReservationStore()  # defaults to /run/user/UID/spark-gpu
        rid = store.reserve(gb_requested=60, duration_s=3600, purpose="Qwen3 LoRA")
        for r in store.list_active():
            print(r.user, r.gb_requested, r.remaining_s)
        store.release(rid)

    On every read (`list_active`, `total_reserved_gb`), expired + dead-pid
    reservations are auto-pruned + deleted from disk.
    """

    def __init__(self, dir_: Optional[Path] = None) -> None:
        self.dir = dir_ or DEFAULT_RESERVATIONS_DIR
        self.dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def reserve(
        self,
        gb_requested: float,
        duration_s: float,
        purpose: str = "",
        pid: Optional[int] = None,
        user: Optional[str] = None,
        hostname: Optional[str] = None,
        now: Optional[datetime] = None,
    ) -> str:
        """Create a reservation file. Returns reservation_id."""
        if gb_requested <= 0:
            raise ValueError("gb_requested must be > 0")
        if duration_s <= 0:
            raise ValueError("duration_s must be > 0")
        now = now or datetime.now(timezone.utc)
        rid = secrets.token_hex(6)
        res = Reservation(
            reservation_id=rid,
            user=user or _current_user(),
            hostname=hostname or socket.gethostname(),
            gb_requested=gb_requested,
            started_at=now,
            expires_at=now + timedelta(seconds=duration_s),
            purpose=purpose,
            pid=pid,
        )
        self._write(res)
        return rid

    def release(self, reservation_id: str) -> bool:
        """Delete a reservation file. Returns True if it existed."""
        path = self.dir / f"{reservation_id}.json"
        try:
            path.unlink()
            return True
        except FileNotFoundError:
            return False

    def list_active(self, now: Optional[datetime] = None) -> list[Reservation]:
        """Return non-expired reservations whose pid is still alive (if set).

        Auto-prunes expired/dead reservations from disk as a side effect.
        """
        now = now or datetime.now(timezone.utc)
        active: list[Reservation] = []
        with self._lock:
            for path in sorted(self.dir.glob("*.json")):
                try:
                    r = Reservation.from_json(json.loads(path.read_text()))
                except (OSError, json.JSONDecodeError, KeyError, TypeError):
                    continue
                if now >= r.expires_at:
                    path.unlink(missing_ok=True)
                    continue
                if r.pid is not None and not _pid_alive(r.pid):
                    path.unlink(missing_ok=True)
                    continue
                active.append(r)
        return active

    def total_reserved_gb(self, now: Optional[datetime] = None) -> float:
        return sum(r.gb_requested for r in self.list_active(now=now))

    def _write(self, res: Reservation) -> None:
        path = self.dir / f"{res.reservation_id}.json"
        tmp = path.with_suffix(".tmp")
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(res.to_json(), f, indent=2)
        tmp.replace(path)


def _current_user() -> str:
    """Best-effort current user; falls back to "unknown" on lookup failure."""
    try:
        return getpass.getuser()
    except Exception:  # noqa: BLE001 — getpass is finicky in some containers
        return os.environ.get("USER") or "unknown"


__all__ = [
    "DEFAULT_RESERVATIONS_DIR",
    "Reservation",
    "ReservationStore",
]
