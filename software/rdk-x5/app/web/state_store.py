"""Thread-safe JSON snapshot store for the RootScope dashboard."""

from __future__ import annotations

import copy
import threading
from datetime import datetime, timezone
from typing import Any, Mapping


def default_snapshot() -> dict[str, Any]:
    return {
        "schema_version": "rootscope.dashboard.v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "mode": "SIMULATED_ONLY",
        "state": "BOOT_LOCKED",
        "backend_actual": "fixture",
        "perception": {
            "source": "none",
            "class_id": "unknown",
            "confidence": None,
            "qualified": False,
        },
        "task": {
            "task_id": None,
            "channel": None,
            "profile": None,
            "completion_class": "SIMULATED_ONLY",
        },
        "safety": {
            "firmware_identity": False,
            "heartbeat_fresh": False,
            "estop_ok": False,
            "leak_ok": False,
            "cartridge_ok": False,
            "guard_ok": False,
            "mass_stable": False,
            "camera_fresh": False,
        },
        "mass": {"target_g": None, "measured_loss_g": 0.0, "samples": []},
        "wetting": {"passed": False, "target_changed_fraction": 0.0, "reasons": []},
        "evidence": {"head_hash": None, "record_count": 0},
        "alerts": ["LOCAL_FIXTURE_ONLY_NO_HARDWARE_TOUCHED"],
    }


class SnapshotStore:
    def __init__(self, initial: Mapping[str, Any] | None = None) -> None:
        self._lock = threading.RLock()
        self._value = copy.deepcopy(dict(initial) if initial is not None else default_snapshot())

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return copy.deepcopy(self._value)

    def replace(self, value: Mapping[str, Any]) -> None:
        with self._lock:
            self._value = copy.deepcopy(dict(value))

    def update(self, **fields: Any) -> None:
        with self._lock:
            self._value.update(copy.deepcopy(fields))
            self._value["generated_at"] = datetime.now(timezone.utc).isoformat()
