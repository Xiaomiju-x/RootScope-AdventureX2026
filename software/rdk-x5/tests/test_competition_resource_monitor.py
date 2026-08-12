from __future__ import annotations

import importlib.util
from pathlib import Path
from unittest import mock


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "tools"
    / "x5_competition_resource_monitor.py"
)
SPEC = importlib.util.spec_from_file_location(
    "x5_competition_resource_monitor", MODULE_PATH
)
assert SPEC is not None and SPEC.loader is not None
MONITOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MONITOR)


def test_read_key_values_parses_proc_style_text(tmp_path: Path) -> None:
    source = tmp_path / "meminfo"
    source.write_text(
        "MemAvailable:    123456 kB\n"
        "CmaFree:          65432 kB\n"
        "ignored text\n",
        encoding="ascii",
    )
    assert MONITOR.read_key_values(source) == {
        "MemAvailable": 123456,
        "CmaFree": 65432,
    }


def test_sample_is_observation_only() -> None:
    def fake_values(path: Path) -> dict[str, int]:
        if path.name == "meminfo":
            return {
                "MemAvailable": 900_000,
                "CmaFree": 200_000,
                "SwapFree": 0,
            }
        return {"oom_kill": 7}

    with (
        mock.patch.object(MONITOR, "read_key_values", side_effect=fake_values),
        mock.patch.object(MONITOR, "temperature_millicelsius", return_value=42_000),
        mock.patch.object(MONITOR, "camera_owner", return_value=""),
        mock.patch.object(
            MONITOR,
            "process_rss_kib",
            return_value={marker: 0 for marker in MONITOR.PROCESS_MARKERS},
        ),
    ):
        observed = MONITOR.sample()
    assert observed["mem_available_kib"] == 900_000
    assert observed["cma_free_kib"] == 200_000
    assert observed["oom_kill_count"] == 7
    assert observed["camera_owner_observed"] == ""
    assert observed["temperature_millicelsius"] == 42_000
    assert all(value is False for value in MONITOR.ZERO_AUTHORITY.values())
