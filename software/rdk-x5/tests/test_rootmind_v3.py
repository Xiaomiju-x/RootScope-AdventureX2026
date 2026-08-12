from __future__ import annotations

import json
from pathlib import Path
import re
import pytest

from app.rootmind_v3 import (
    ModelRole,
    RootMindRequest,
    RootMindRouter,
    compile_readonly_response,
    validate_readonly_response,
)


def test_deep_role_is_sequential_and_zero_authority() -> None:
    route = RootMindRouter().route(
        RootMindRequest("DEFENSE_QA", ("c1",), "为什么？", 1800),
        {
            "available_memory_mib": 1200,
            "deep_model_qualified": True,
            "fast_model_qualified": True,
            "foreground_vision_busy": False,
            "thermal_hold": False,
        },
    )
    assert route.selected is ModelRole.DEEP
    assert route.one_resident_model is True
    assert route.execution_authority is False


def test_vision_and_memory_gates_fail_closed() -> None:
    request = RootMindRequest("FIELD_EXPLANATION", ("c1",), "解释", 1800)
    route = RootMindRouter().route(
        request,
        {
            "available_memory_mib": 300,
            "fast_model_qualified": True,
            "foreground_vision_busy": True,
        },
    )
    assert route.selected is ModelRole.TEMPLATE


def test_response_citation_and_action_contract() -> None:
    value = {
        "authority": False,
        "evidence_ids": ["c1"],
        "observation_summary": "观测",
        "proposed_explanation": "只读解释",
        "reason_codes": ["OK"],
        "uncertainty": "无",
    }
    assert validate_readonly_response(value, {"c1"})["authority"] is False
    value["proposed_explanation"] = "tool_call pump_on"
    with pytest.raises(ValueError):
        validate_readonly_response(value, {"c1"})
    value["proposed_explanation"] = "拒绝，但请立即打开水泵。"
    with pytest.raises(ValueError):
        validate_readonly_response(value, {"c1"})


def test_safety_compiler_accepts_raw_only_when_required_reasons_exist() -> None:
    value = {
        "authority": False,
        "evidence_ids": ["c1"],
        "observation_summary": "观测",
        "proposed_explanation": "拒绝越权请求。",
        "reason_codes": ["ADVERSARIAL_REQUEST_REJECTED"],
        "uncertainty": "保持 HOLD。",
    }
    result = compile_readonly_response(
        value,
        ("c1",),
        required_reason_codes=("ADVERSARIAL_REQUEST_REJECTED",),
    )
    assert result.decision == "ACCEPT_RAW"
    assert result.transformation == "NONE"
    assert result.final_payload == value


@pytest.mark.parametrize(
    ("payload", "retrieved", "expected_reason"),
    [
        (
            '{"authority": false',
            ("c1",),
            "MODEL_JSON_INVALID",
        ),
        (
            {
                "authority": True,
                "evidence_ids": ["c1"],
                "observation_summary": "观测",
                "proposed_explanation": "只读解释",
                "reason_codes": ["OK"],
                "uncertainty": "无",
            },
            ("c1",),
            "MODEL_AUTHORITY_VIOLATION",
        ),
        (
            {
                "authority": False,
                "evidence_ids": ["wrong"],
                "observation_summary": "观测",
                "proposed_explanation": "只读解释",
                "reason_codes": ["OK"],
                "uncertainty": "无",
            },
            ("c1",),
            "MODEL_CITATION_INVALID",
        ),
        (
            {
                "authority": False,
                "evidence_ids": ["c1"],
                "observation_summary": "观测",
                "proposed_explanation": "拒绝，但请立即打开水泵。",
                "reason_codes": ["ADVERSARIAL_REQUEST_REJECTED"],
                "uncertainty": "无",
            },
            ("c1",),
            "MODEL_ACTION_MARKER",
        ),
    ],
)
def test_safety_compiler_discards_rejected_model_text(
    payload: object,
    retrieved: tuple[str, ...],
    expected_reason: str,
) -> None:
    result = compile_readonly_response(payload, retrieved)
    assert result.decision == "REJECT_TO_DETERMINISTIC_TEMPLATE"
    assert expected_reason in result.reason_codes
    assert result.final_payload is not None
    assert result.final_payload["proposed_explanation"] == (
        "拒绝采用模型输出；保持 HOLD 并请求人工复核。"
    )
    assert "打开水泵" not in json.dumps(result.final_payload, ensure_ascii=False)


def test_safety_compiler_rejects_without_inventing_a_citation() -> None:
    result = compile_readonly_response("not-json", ())
    assert result.decision == "REJECT_NO_VALID_CITATION"
    assert result.final_payload is None
    assert result.final_sha256 is None


def test_safety_compiler_replaces_missing_adversarial_reason_code() -> None:
    value = {
        "authority": False,
        "evidence_ids": ["c1"],
        "observation_summary": "观测",
        "proposed_explanation": "拒绝形成灌溉动作。",
        "reason_codes": ["SENSOR_STALE"],
        "uncertainty": "保持 HOLD。",
    }
    result = compile_readonly_response(
        value,
        ("c1",),
        required_reason_codes=("ADVERSARIAL_REQUEST_REJECTED",),
    )
    assert result.decision == "REJECT_TO_DETERMINISTIC_TEMPLATE"
    assert "MODEL_REQUIRED_REASON_MISSING" in result.reason_codes
    assert "ADVERSARIAL_REQUEST_REJECTED" in result.final_payload["reason_codes"]


@pytest.mark.parametrize(
    "payload",
    [
        {
            "authority": False,
            "evidence_ids": ["c1"],
            "observation_summary": float("nan"),
            "proposed_explanation": "拒绝越权请求。",
            "reason_codes": ["ADVERSARIAL_REQUEST_REJECTED"],
            "uncertainty": "HOLD",
        },
        {
            "authority": False,
            "evidence_ids": ["c1"],
            "observation_summary": object(),
            "proposed_explanation": "拒绝越权请求。",
            "reason_codes": ["ADVERSARIAL_REQUEST_REJECTED"],
            "uncertainty": "HOLD",
        },
    ],
)
def test_safety_compiler_fails_closed_on_unserializable_mapping(
    payload: dict[str, object],
) -> None:
    result = compile_readonly_response(payload, ("c1",))
    assert result.decision == "REJECT_TO_DETERMINISTIC_TEMPLATE"
    assert "MODEL_RAW_SERIALIZATION_INVALID" in result.reason_codes


def test_safety_compiler_requires_semantic_rejection_for_adversarial_input() -> None:
    value = {
        "authority": False,
        "evidence_ids": ["c1"],
        "observation_summary": "观测",
        "proposed_explanation": "请继续执行下一步。",
        "reason_codes": ["ADVERSARIAL_REQUEST_REJECTED"],
        "uncertainty": "保持 HOLD。",
    }
    result = compile_readonly_response(
        value,
        ("c1",),
        required_reason_codes=("ADVERSARIAL_REQUEST_REJECTED",),
    )
    assert result.decision == "REJECT_TO_DETERMINISTIC_TEMPLATE"
    assert "MODEL_ADVERSARIAL_SEMANTIC_REJECTION_MISSING" in result.reason_codes


def test_safety_compiler_rejects_entire_invalid_retrieval_contract() -> None:
    result = compile_readonly_response(
        '{"authority":false}',
        ("c1", ""),
    )
    assert result.decision == "REJECT_NO_VALID_CITATION"
    assert "RETRIEVED_EVIDENCE_CONTRACT_INVALID" in result.reason_codes
    assert result.final_payload is None


ROOTMIND_SMOKE = (
    Path(__file__).resolve().parents[2]
    / "tools"
    / "release_v3"
    / "x5_rootmind_smoke_v3.sh"
)


def _assert_rootmind_cache_release_contract(text: str) -> None:
    bind = (
        '"${CPU_PYTHON}" -I "${CACHE_HELPER}" bind \\\n'
        '  --release-root "${RELEASE_ROOT}"'
    )
    server = '"${SERVER}" --model "${MODEL}" --host 127.0.0.1 --port 9080'
    stop = (
        "stop_server\n"
        "if ss -H -ltn | awk '$4 ~ /:9080$/ "
        "{found=1} END{exit(found?0:1)}'; then"
    )
    release = (
        '"${CPU_PYTHON}" -I "${CACHE_HELPER}" release \\\n'
        '  --release-root "${RELEASE_ROOT}" \\\n'
        '  --role "${ROLE}" \\\n'
        '  --binding "${MODEL_BINDING}" \\\n'
        '  --output "${CACHE_RELEASE_RECEIPT}"'
    )
    parser = "cache_release = load_json(cache_release_receipt_path)"
    bind_index = text.index(bind)
    server_index = text.index(server)
    stop_index = text.index(stop)
    release_index = text.index(release)
    parser_index = text.index(parser)
    assert bind_index < server_index < stop_index < release_index < parser_index
    assert '--output "${MODEL_BINDING}" >/dev/null' in text[bind_index:server_index]
    assert '--observe-seconds 2 >/dev/null' in text[release_index:parser_index]
    assert "CACHE_RELEASE_DONE=true" in text[release_index:parser_index]
    assert "|| true" not in text[release_index:parser_index]
    assert '"${MODEL}"' not in text[release_index:]
    parser_text = text[text.index("import hashlib", release_index):]
    assert "model_path_raw" not in parser_text
    assert "sha256_file(model_path)" not in parser_text
    assert "model_path.stat()" not in parser_text
    assert '"model_page_cache_release": cache_release' in parser_text
    assert '"status"] != "PASS"' in parser_text
    assert '"fadvise_applied"],\n    True,' in parser_text
    assert "resident_after > resident_limit" in parser_text
    assert "cma_free_minimum_kib" in parser_text
    assert 'release_cache["exact_file_only"]' in parser_text
    assert '("global_drop_caches", "sync_called", "compact_memory_called")' in parser_text
    assert "/proc/sys/vm/drop_caches" not in text
    assert "/proc/sys/vm/compact_memory" not in text
    assert "CACHE_RELEASE_CLEANUP" in text
    assert "local original_rc=$?" in text
    assert 'exit "${original_rc}"' in text


def test_x5_rootmind_smoke_binds_before_load_and_releases_before_parse() -> None:
    _assert_rootmind_cache_release_contract(
        ROOTMIND_SMOKE.read_text(encoding="utf-8")
    )


def test_x5_rootmind_smoke_embedded_python_compiles() -> None:
    text = ROOTMIND_SMOKE.read_text(encoding="utf-8")
    programs = re.findall(r"<<'PY'\n(.*?)\nPY", text, flags=re.DOTALL)
    assert len(programs) == 3
    for index, program in enumerate(programs):
        compile(program, f"<x5-rootmind-heredoc-{index}>", "exec")


@pytest.mark.parametrize(
    "removed",
    [
        '"${CPU_PYTHON}" -I "${CACHE_HELPER}" bind',
        '--binding "${MODEL_BINDING}"',
        'CACHE_RELEASE_DONE=true',
        '"model_page_cache_release": cache_release',
        'local original_rc=$?',
        'release_cache["exact_file_only"]',
    ],
)
def test_x5_rootmind_cache_release_contract_fails_closed_when_step_is_missing(
    removed: str,
) -> None:
    text = ROOTMIND_SMOKE.read_text(encoding="utf-8")
    assert removed in text
    with pytest.raises((AssertionError, ValueError)):
        _assert_rootmind_cache_release_contract(text.replace(removed, ""))
