"""GPU coordination — shared-host scheduling between multiple users.

Two users run workloads on the same Spark GB10 today.
nvidia-smi alone tells you who's USING the GPU; it doesn't tell you who
INTENDS to use it next or how much memory they need. This subpackage
adds:

  * `probe`       — parse nvidia-smi for current procs + util
  * `reservations` — file-based cooperative reservations under
                    /tmp/spark-gpu/ (or $XDG_RUNTIME_DIR/spark-gpu)
  * `cli`         — `mesh-gpu status / reserve / release / wait`

Cooperative: nothing here ENFORCES the reservation at the kernel level.
A reservation is a polite signal to other mesh participants that this
host has claimed N GB for D duration. Phase 2 (v0.0.7+) wires
reservations into the mesh service so they're visible cluster-wide.

Why file-based rather than mesh-service backed: zero infra dependency.
Two users can use this today against a shared filesystem
without the mesh service needing to be up. The contract is a directory
+ one JSON file per reservation; trivial to inspect with `ls + cat`.
"""

from mesh.gpu.probe import GpuProcess, GpuSnapshot, probe_gpu
from mesh.gpu.reservations import (
    DEFAULT_RESERVATIONS_DIR,
    Reservation,
    ReservationStore,
)

__all__ = [
    "DEFAULT_RESERVATIONS_DIR",
    "GpuProcess",
    "GpuSnapshot",
    "Reservation",
    "ReservationStore",
    "probe_gpu",
]
