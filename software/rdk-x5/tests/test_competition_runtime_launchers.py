from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / "tools/start_x5_competition_runtime_v2.sh"
LIVE = ROOT / "tools/start_x5_competition_live_vision_v2.sh"


def _text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_runtime_launcher_is_candidate_bound_foreground_and_zero_authority() -> None:
    text = _text(RUNTIME)
    assert "rootscope_competition_runtime_v2_candidate_20260723" in text
    assert 'APP_ROOT="${RELEASE_ROOT}/rootscope"' in text
    assert 'PYTHONPATH="${APP_ROOT}"' in text
    assert "rootscope-event-vision-overlay" not in text
    assert "systemctl" not in text
    assert "nohup" not in text
    assert "/dev/tty" not in text
    assert '"serial_opened": False' in text
    assert '"gpio_touched": False' in text
    assert '"pump_touched": False' in text
    assert "trap cleanup EXIT" in text
    assert "--live" in text and "--no-live" in text


def test_runtime_uses_one_loopback_qwen_and_one_three_role_call() -> None:
    text = _text(RUNTIME)
    assert text.count('"${LLAMA_SERVER}" \\') == 1
    assert "--host 127.0.0.1" in text
    assert "--port 9080" in text
    assert "--parallel 1" in text
    assert "--ctx-size 512" in text
    assert "--batch-size 32" in text
    assert "--ubatch-size 16" in text
    assert "--api-mode completion" in text
    assert "127.0.0.1:9080" in text
    assert "0.0.0.0" not in text
    assert text.count("-m app.competition_llm.competition_rag") == 1
    assert 'report["generation"]["inference_call_count"] != 1' in text
    assert '"resident_model_count": 1' in text
    assert '"logical_role_count": 3' in text


def test_runtime_starts_hash_bound_af_unix_bpu_worker_without_default_upgrade() -> None:
    text = _text(RUNTIME)
    assert text.count("-m app.competition_runtime.bpu_shadow_worker") == 1
    assert '--socket "${BPU_SOCKET}"' in text
    assert '--expected-model-sha256 "${EXPECTED_R7_SHA256}"' in text
    assert "pyeasy_dnn" not in text
    assert "--backend legacy_pyeasy" not in text
    assert '"bpu_transport": "AF_UNIX"' in text
    assert '"bpu_qualification": "SHADOW_CANDIDATE_NOT_DEFAULT"' in text
    assert '"selected_bin_changed": False' in text


def test_actual_llama_server_flags_are_preflighted_before_start() -> None:
    text = _text(RUNTIME)
    assert '"${LLAMA_SERVER}" --help' in text
    assert '"${LLAMA_SERVER}" --version' in text
    for flag in (
        "--model",
        "--host",
        "--port",
        "--ctx-size",
        "--threads",
        "--threads-batch",
        "--parallel",
        "--batch-size",
        "--ubatch-size",
        "--no-warmup",
        "--no-ui",
        "--cache-ram",
    ):
        assert flag in text


def test_live_launcher_uses_candidate_release_and_only_explicit_camera() -> None:
    text = _text(LIVE)
    assert 'APP_ROOT="${RELEASE_ROOT}/rootscope"' in text
    assert 'LIVE_SCRIPT="${TOOLS_ROOT}/x5_competition_live_vision_v2.py"' in text
    assert 'PYTHONPATH="${APP_ROOT}"' in text
    assert "rootscope-event-vision-overlay" not in text
    assert '--bpu-socket "${BPU_SOCKET}"' in text
    assert "usb-Web_Camera_Web_Camera_202604081837-video-index0" in text
    assert "discover" not in text.lower()
    assert "systemctl" not in text
    assert "/dev/tty" not in text
