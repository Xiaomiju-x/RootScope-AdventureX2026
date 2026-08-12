#!/usr/bin/env python3
"""Independent audit of the staged RootScope read-only local-LLM release."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Mapping, Sequence


EXPECTED_MODEL_SHA = "6c8adc95de81b1d0103fcf900a5f369d5936d234e42ed64881dad56e0104e77b"
EXPECTED_MODEL_SIZE = 397805120
EXPECTED_MODEL_NAME = "qwen2_05b_distill.Q4_K_M.gguf"
EXPECTED_RELEASE_RELATIVE = "output/rootscope_llm_readonly_release_v1"
_SUM_RE = re.compile(r"^([0-9a-f]{64})  ([A-Za-z0-9_.-]+)$")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class Audit:
    def __init__(self) -> None:
        self.checks: list[dict[str, Any]] = []

    def check(self, name: str, condition: bool, detail: str = "") -> None:
        self.checks.append({"name": name, "pass": bool(condition), "detail": detail})

    @property
    def passed(self) -> bool:
        return bool(self.checks) and all(item["pass"] for item in self.checks)


def _mapping(value: Any) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("expected JSON object")
    return value


def _all_false(value: Any) -> bool:
    return isinstance(value, Mapping) and bool(value) and all(item is False for item in value.values())


def _parse_sums(path: Path) -> Mapping[str, str]:
    entries: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        match = _SUM_RE.fullmatch(line)
        if not match or match.group(2) in entries:
            raise ValueError("SHA256SUMS is malformed or contains duplicates")
        entries[match.group(2)] = match.group(1)
    return entries


def audit_release_core(
    audit: Audit,
    *,
    adventurex_root: Path,
    spec_path: Path,
    release_dir: Path,
    expected_sha: str = EXPECTED_MODEL_SHA,
    expected_size: int = EXPECTED_MODEL_SIZE,
    expected_name: str = EXPECTED_MODEL_NAME,
) -> None:
    root = adventurex_root.resolve(strict=True)
    spec = _mapping(json.loads(spec_path.read_text(encoding="utf-8")))
    selected = _mapping(spec.get("selected_artifact"))
    audit.check("spec_schema", spec.get("schema") == "rootscope.readonly_llm_release_spec.v1")
    audit.check("spec_status_unqualified", spec.get("status") == "STAGING_SPEC_READ_ONLY_LOCAL_LLM_NOT_X5_QUALIFIED")
    audit.check("spec_formal_flags_all_false", _all_false(spec.get("formal_flags")))
    audit.check("spec_model_name", selected.get("filename") == expected_name)
    audit.check("spec_model_size", selected.get("size_bytes") == expected_size)
    audit.check("spec_model_sha", selected.get("sha256") == expected_sha)
    runtime = _mapping(spec.get("runtime_contract"))
    audit.check(
        "spec_runtime_loopback_manual_readonly",
        runtime.get("host") == "127.0.0.1"
        and runtime.get("port") == 9080
        and runtime.get("default_enabled") is False
        and runtime.get("manual_start_only") is True
        and runtime.get("external_network_allowed") is False
        and runtime.get("read_only") is True
        and runtime.get("tool_execution") is False
        and runtime.get("actuator_access") is False,
    )
    dependency = _mapping(spec.get("llama_cpp_dependency"))
    audit.check(
        "spec_llama_cpp_external_not_bundled",
        dependency.get("bundled") is False
        and dependency.get("x5_binary_selected") is False
        and dependency.get("executable_sha256_required_before_start") is True,
    )

    source = (root / str(selected["source_relative_to_adventurex"])).resolve(strict=True)
    model = (release_dir / expected_name).resolve(strict=True)
    manifest_path = release_dir / "release_manifest.json"
    sums_path = release_dir / "SHA256SUMS"
    audit.check("source_size_exact", source.stat().st_size == expected_size)
    audit.check("source_sha_exact", sha256_file(source) == expected_sha)
    audit.check("staged_size_exact", model.stat().st_size == expected_size)
    audit.check("staged_sha_exact", sha256_file(model) == expected_sha)
    audit.check("source_and_staged_are_distinct_files", source != model)

    manifest = _mapping(json.loads(manifest_path.read_text(encoding="utf-8")))
    artifact = _mapping(manifest.get("artifact"))
    audit.check("manifest_schema", manifest.get("schema") == "rootscope.readonly_llm_release_manifest.v1")
    audit.check("manifest_status_unqualified", manifest.get("status") == "STAGED_READ_ONLY_LOCAL_LLM_NOT_X5_QUALIFIED")
    audit.check("manifest_artifact_staged_fact", manifest.get("artifact_staged") is True)
    audit.check("manifest_formal_flags_all_false", _all_false(manifest.get("formal_flags")))
    audit.check("manifest_runtime_matches_spec", manifest.get("runtime_contract") == spec.get("runtime_contract"))
    audit.check("manifest_dependency_matches_spec", manifest.get("llama_cpp_dependency") == spec.get("llama_cpp_dependency"))
    audit.check(
        "manifest_artifact_exact",
        artifact.get("filename") == expected_name
        and artifact.get("size_bytes") == expected_size
        and artifact.get("sha256") == expected_sha
        and artifact.get("destination_relative_to_adventurex")
        == f"{EXPECTED_RELEASE_RELATIVE}/{expected_name}",
    )
    spec_receipt = _mapping(manifest.get("staging_spec"))
    audit.check("manifest_binds_spec_hash", spec_receipt.get("sha256") == sha256_file(spec_path))

    sums = _parse_sums(sums_path)
    expected_coverage = {expected_name, "release_manifest.json"}
    audit.check("sha256sums_exact_coverage", set(sums) == expected_coverage)
    audit.check("sha256sums_model_exact", sums.get(expected_name) == expected_sha)
    audit.check("sha256sums_manifest_exact", sums.get("release_manifest.json") == sha256_file(manifest_path))
    audit.check(
        "release_directory_contains_only_covered_payload_and_sums",
        {item.name for item in release_dir.iterdir()} == expected_coverage | {"SHA256SUMS"},
    )


def audit_service_contract(audit: Audit, unit_text: str, start_text: str) -> None:
    audit.check("user_unit_has_no_hardcoded_user", "User=" not in unit_text and "Group=" not in unit_text)
    audit.check("user_unit_has_no_auto_enable", "[Install]" not in unit_text and "WantedBy=" not in unit_text)
    audit.check("user_unit_requires_gate", "ConditionPathExists=@GATE_FILE@" in unit_text)
    audit.check("user_unit_loopback_firewall", "IPAddressDeny=any" in unit_text and "IPAddressAllow=localhost" in unit_text)
    audit.check("user_unit_private_devices", "PrivateDevices=true" in unit_text)
    audit.check("user_unit_uses_fixed_launcher", "start_readonly_llm.sh" in unit_text and "bash -lc" not in unit_text)
    audit.check("launcher_fixed_numeric_loopback", '--host 127.0.0.1' in start_text and '--port 9080' in start_text)
    audit.check("launcher_requires_manual_ack", 'READ_ONLY_EXPLANATION_ONLY' in start_text and 'ROOTSCOPE_LLM_GATE_FILE' in start_text)
    audit.check(
        "launcher_false_authority_gates",
        'ROOTSCOPE_LLM_EXTERNAL_NETWORK" == "false"' in start_text
        and 'ROOTSCOPE_LLM_TOOL_EXECUTION" == "false"' in start_text
        and 'ROOTSCOPE_LLM_ACTUATOR_ACCESS" == "false"' in start_text,
    )
    forbidden_runtime = ("0.0.0.0", "https://", "curl ", "wget ", "ssh ", "socat ", "/dev/", "gpio", "relay", "serial")
    lowered = (unit_text + "\n" + start_text).lower()
    audit.check(
        "unit_and_launcher_have_no_external_or_device_path",
        not any(token in lowered for token in forbidden_runtime),
    )


def audit_python_source(audit: Audit, name: str, path: Path, *, health_client_allowed: bool) -> None:
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".", 1)[0])
    audit.check(
        f"{name}_no_process_or_hardware_library",
        not imported.intersection({"subprocess", "serial", "gpiod", "gpiozero", "cv2"}),
        detail=",".join(sorted(imported)),
    )
    lowered = source.lower()
    forbidden = ("/dev/", "0.0.0.0", "https://", "ssh ", "socat ", "serial.serial")
    audit.check(f"{name}_no_external_or_device_literal", not any(token in lowered for token in forbidden))
    if not health_client_allowed:
        audit.check(f"{name}_no_network_client", "http.client" not in source and "socket" not in imported)


def audit(adventurex_root: Path) -> Mapping[str, Any]:
    root = adventurex_root.resolve(strict=True)
    audit_state = Audit()
    spec_path = root / "rootscope/deploy/x5/readonly_llm_release_spec.json"
    release_dir = root / EXPECTED_RELEASE_RELATIVE
    audit_release_core(
        audit_state,
        adventurex_root=root,
        spec_path=spec_path,
        release_dir=release_dir,
    )

    deploy = root / "rootscope/deploy/x5"
    unit_path = deploy / "systemd/rootscope-llm-readonly.service.disabled-template"
    start_path = deploy / "scripts/start_readonly_llm.sh"
    audit_service_contract(
        audit_state,
        unit_path.read_text(encoding="utf-8"),
        start_path.read_text(encoding="utf-8"),
    )
    python_files = {
        "stager": deploy / "scripts/stage_readonly_llm.py",
        "installer": deploy / "scripts/install_readonly_llm.py",
        "preflight": deploy / "scripts/readonly_llm_preflight.py",
        "explicit_cli": deploy / "scripts/explain_readonly_snapshot.py",
    }
    for name, path in python_files.items():
        audit_python_source(audit_state, name, path, health_client_allowed=name in {"preflight"})

    explainer_path = root / "rootscope/app/llm/read_only_explainer.py"
    explainer = explainer_path.read_text(encoding="utf-8")
    audit_state.check(
        "explainer_direct_loopback_no_redirect",
        "http.client.HTTPConnection" in explainer
        and "_resolve_numeric_loopback" in explainer
        and "urlopen" not in explainer,
    )
    audit_state.check("explainer_has_no_output_file_option", '"--output-json"' not in explainer)
    audit_state.check(
        "explainer_authority_false",
        '"tool_execution": False' in explainer
        and '"actuator_access": False' in explainer
        and '"state_machine_write": False' in explainer,
    )
    # The exact AUTHORITY mapping is parsed independently instead of trusting a
    # textual false-positive search.
    module_tree = ast.parse(explainer)
    authority_literal: Mapping[str, Any] | None = None
    for node in module_tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == "AUTHORITY" for target in node.targets):
            authority_literal = ast.literal_eval(node.value)
            break
    audit_state.check("explainer_authority_mapping_all_false", _all_false(authority_literal))

    locked_service = (root / "rootscope/app/edge/service.py").read_text(encoding="utf-8").lower()
    audit_state.check(
        "locked_edge_service_does_not_query_llm",
        "app.llm" not in locked_service
        and "9080" not in locked_service
        and "chat/completions" not in locked_service
        and "http.client" not in locked_service,
    )
    for config_path in sorted(deploy.glob("capsule_config*.json")):
        config = _mapping(json.loads(config_path.read_text(encoding="utf-8")))
        llm = _mapping(config.get("llm"))
        audit_state.check(
            f"capsule_llm_disabled_{config_path.name}",
            llm.get("enabled") is False
            and llm.get("read_only") is True
            and llm.get("tool_execution") is False
            and llm.get("actuator_access") is False,
        )

    inventory_path = root / "rootscope/app/llm/xrd_llm_reuse_inventory.json"
    inventory = _mapping(json.loads(inventory_path.read_text(encoding="utf-8")))
    facts = _mapping(inventory.get("artifact_facts"))
    audit_state.check(
        "inventory_records_exact_staged_copy",
        facts.get("model_copied_to_adventurex_release") is True
        and facts.get("copy_size_and_sha_verified") is True,
    )
    audit_state.check("inventory_formal_flags_all_false", _all_false(inventory.get("formal_flags")))

    audited_files = [
        spec_path,
        release_dir / EXPECTED_MODEL_NAME,
        release_dir / "release_manifest.json",
        release_dir / "SHA256SUMS",
        unit_path,
        start_path,
        explainer_path,
        inventory_path,
        *python_files.values(),
    ]
    passed_count = sum(1 for item in audit_state.checks if item["pass"])
    return {
        "schema": "rootscope.readonly_llm_independent_audit.v1",
        "status": "PASS" if audit_state.passed else "FAIL",
        "scope": "LOCAL_FILES_ONLY_NO_SERVICE_DEVICE_SSH_OR_NETWORK_OPERATION",
        "checks_total": len(audit_state.checks),
        "checks_passed": passed_count,
        "checks_failed": len(audit_state.checks) - passed_count,
        "checks": audit_state.checks,
        "audited_file_sha256": {
            path.relative_to(root).as_posix(): sha256_file(path) for path in audited_files
        },
        "formal_flags": {
            "model_qualified": False,
            "x5_validated": False,
            "llama_cpp_bundled": False,
            "service_started": False,
            "hardware_touched": False,
            "network_touched": False,
            "external_network_allowed": False,
            "tool_execution": False,
            "actuator_access": False,
            "execution_authority": False,
            "physical_authority": False,
        },
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[2]
    parser.add_argument("--adventurex-root", type=Path, default=root)
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "evidence/rootscope_readonly_llm_release_audit_20260717.json",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    root = args.adventurex_root.resolve(strict=True)
    output = args.output.resolve()
    try:
        output.relative_to(root)
    except ValueError as exc:
        raise ValueError("audit output must remain inside AdventureX") from exc
    report = audit(root)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps({key: report[key] for key in ("status", "checks_total", "checks_passed", "checks_failed")}, indent=2))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
