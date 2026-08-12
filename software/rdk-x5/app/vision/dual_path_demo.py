"""Fail-closed semantic + registered-card evidence for the RootScope demo.

This module deliberately does not decide irrigation.  The seed-17 ONNX model
produces a raw semantic *hypothesis*.  A feature/homography matcher independently
checks exact registered card instances.  A narrowly named experimental consensus
is emitted only when exactly one registered template passes geometry, its class
agrees with the raw top-1 class, and explicit demo-only probability/margin gates
pass.  Even that consensus has zero execution authority.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Mapping, Sequence

import numpy as np
from PIL import Image

from app.edge.capsule import (
    CPU_PROVIDER,
    ROOTSCOPE_CLASS_ORDER,
    CapsuleConfig,
)
from app.edge.onnx_cpu import CpuOnnxRunner, OnnxCpuContractError, preprocess_rgb
from app.vision.card_geometric_matcher import (
    CLAIM_SCOPE as GEOMETRIC_CLAIM_SCOPE,
    SCHEMA_VERSION as GEOMETRIC_SCHEMA_VERSION,
    MatcherConfig,
    match_known_card,
)


SCHEMA_VERSION = "rootscope.dual-path-demo.v1"
REGISTRY_SCHEMA_VERSION = "rootscope.known-card-template-registry.v1"
REGISTRY_EMPTY_STATUS = "EMPTY_NO_REAL_TEMPLATES_REGISTERED"
REGISTRY_FROZEN_STATUS = "FROZEN_EXPERIMENTAL_DEMO_REFERENCES"
REGISTERED_ROLE = "DEMO_REFERENCE_NOT_HOLDOUT_ONCE_REGISTERED"
MODEL_STATUS = "MACHINE_CURATED_EXPERIMENTAL_NOT_QUALIFIED"
SEMANTIC_STATUS = "DEMO_HYPOTHESIS"
FORMAL_REJECTION_STATUS = "REJECT_ALL_INSUFFICIENT_PER_CLASS_EVIDENCE"
CONSENSUS_STATUS = "EXPERIMENTAL_KNOWN_CARD_CONSENSUS"
SEED17_MODEL_SHA256 = (
    "50ae8d2ec1cec0f3748efa127c6b9c00624684f8c112c8a7913c2a0e304a3bad"
)
ALLOWED_TEMPLATE_CLASSES = frozenset(ROOTSCOPE_CLASS_ORDER) - {"unknown"}
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")

AUTHORITY = {
    "irrigation_execution": False,
    "pump_command": False,
    "serial_write": False,
    "state_machine_write": False,
    "hardware_control": False,
    "physical_completion": False,
}


class DualPathContractError(RuntimeError):
    """The registry, semantic runner, or evidence violated the frozen contract."""


def _strict_keys(payload: Mapping[str, Any], expected: Sequence[str], context: str) -> None:
    expected_set = set(expected)
    actual = set(payload)
    missing = sorted(expected_set - actual)
    unknown = sorted(actual - expected_set)
    if missing or unknown:
        raise DualPathContractError(
            f"{context} keys mismatch: missing={missing} unknown={unknown}"
        )


def _json_object_no_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise DualPathContractError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _required_string(value: Any, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise DualPathContractError(f"{context} must be a non-empty string")
    return value


def _portable_relative_path(value: Any, context: str) -> PurePosixPath:
    text = _required_string(value, context)
    if "\\" in text:
        raise DualPathContractError(f"{context} must use portable '/' separators")
    path = PurePosixPath(text)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise DualPathContractError(f"{context} must be a normalized relative path")
    if path.as_posix() != text:
        raise DualPathContractError(f"{context} must be a normalized relative path")
    return path


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _raw_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class DemoThresholds:
    """Thresholds for the experimental display label, never a formal gate."""

    min_top1_probability: float = 0.70
    min_top1_margin: float = 0.20

    def __post_init__(self) -> None:
        for name in ("min_top1_probability", "min_top1_margin"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise DualPathContractError(f"{name} must be numeric")
            if not math.isfinite(float(value)) or not 0.0 <= float(value) <= 1.0:
                raise DualPathContractError(f"{name} must be finite and in [0,1]")
            object.__setattr__(self, name, float(value))

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "DemoThresholds":
        if not isinstance(payload, Mapping):
            raise DualPathContractError("demo thresholds must be an object")
        _strict_keys(
            payload,
            ("min_top1_probability", "min_top1_margin"),
            "demo thresholds",
        )
        return cls(
            min_top1_probability=payload["min_top1_probability"],
            min_top1_margin=payload["min_top1_margin"],
        )

    def to_dict(self) -> dict[str, float]:
        return dataclasses.asdict(self)


@dataclass(frozen=True)
class RegisteredTemplate:
    template_id: str
    class_name: str
    path: Path
    relative_path: str
    raw_sha256: str
    role: str
    dataset_record: Mapping[str, Any]

    def public_record(self) -> dict[str, Any]:
        return {
            "template_id": self.template_id,
            "class_name": self.class_name,
            "relative_path": self.relative_path,
            "raw_sha256": self.raw_sha256,
            "role": self.role,
            "dataset_record": dict(self.dataset_record),
        }


@dataclass(frozen=True)
class TemplateRegistry:
    path: Path
    status: str
    template_root: Path
    templates: tuple[RegisteredTemplate, ...]


def _validate_attribution(payload: Any, context: str) -> dict[str, str]:
    if not isinstance(payload, Mapping):
        raise DualPathContractError(f"{context} must be an object")
    _strict_keys(payload, ("creator", "license", "license_url"), context)
    return {
        key: _required_string(payload[key], f"{context}.{key}")
        for key in ("creator", "license", "license_url")
    }


def _validate_dataset_record(payload: Any, context: str) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise DualPathContractError(f"{context} must be an object")
    _strict_keys(
        payload,
        ("record_id", "source_manifest", "source_url", "attribution"),
        context,
    )
    record_id = _required_string(payload["record_id"], f"{context}.record_id")
    if not _IDENTIFIER_RE.fullmatch(record_id):
        raise DualPathContractError(f"{context}.record_id has an invalid format")
    source_manifest = str(
        _portable_relative_path(payload["source_manifest"], f"{context}.source_manifest")
    )
    return {
        "record_id": record_id,
        "source_manifest": source_manifest,
        "source_url": _required_string(payload["source_url"], f"{context}.source_url"),
        "attribution": _validate_attribution(
            payload["attribution"], f"{context}.attribution"
        ),
    }


def load_template_registry(path: str | Path) -> TemplateRegistry:
    """Load, strictly validate, and hash-check every registered template."""

    registry_path = Path(path).expanduser().resolve(strict=True)
    payload = json.loads(
        registry_path.read_text(encoding="utf-8"),
        object_pairs_hook=_json_object_no_duplicate_keys,
    )
    if not isinstance(payload, Mapping):
        raise DualPathContractError("template registry must be a JSON object")
    _strict_keys(
        payload,
        ("schema_version", "status", "template_root", "templates"),
        "template registry",
    )
    if payload["schema_version"] != REGISTRY_SCHEMA_VERSION:
        raise DualPathContractError("unsupported template registry schema_version")
    status = _required_string(payload["status"], "template registry.status")
    if status not in {REGISTRY_EMPTY_STATUS, REGISTRY_FROZEN_STATUS}:
        raise DualPathContractError("unsupported template registry status")

    relative_root = _portable_relative_path(
        payload["template_root"], "template registry.template_root"
    )
    registry_parent = registry_path.parent.resolve(strict=True)
    template_root = (registry_parent / Path(*relative_root.parts)).resolve(strict=False)
    if not _is_relative_to(template_root, registry_parent):
        raise DualPathContractError("template_root escapes the registry directory")

    items = payload["templates"]
    if not isinstance(items, list):
        raise DualPathContractError("template registry.templates must be an array")
    if status == REGISTRY_EMPTY_STATUS and items:
        raise DualPathContractError("EMPTY registry status requires templates=[]")
    if status == REGISTRY_FROZEN_STATUS and not items:
        raise DualPathContractError("FROZEN registry status requires at least one template")

    seen_ids: set[str] = set()
    seen_hashes: set[str] = set()
    templates: list[RegisteredTemplate] = []
    for index, item in enumerate(items):
        context = f"template registry.templates[{index}]"
        if not isinstance(item, Mapping):
            raise DualPathContractError(f"{context} must be an object")
        _strict_keys(
            item,
            (
                "template_id",
                "class_name",
                "relative_path",
                "raw_sha256",
                "role",
                "dataset_record",
            ),
            context,
        )
        template_id = _required_string(item["template_id"], f"{context}.template_id")
        if not _IDENTIFIER_RE.fullmatch(template_id):
            raise DualPathContractError(f"{context}.template_id has an invalid format")
        if template_id in seen_ids:
            raise DualPathContractError(f"duplicate template_id: {template_id}")
        seen_ids.add(template_id)

        class_name = _required_string(item["class_name"], f"{context}.class_name")
        if class_name not in ALLOWED_TEMPLATE_CLASSES:
            raise DualPathContractError(
                f"{context}.class_name must be one of {sorted(ALLOWED_TEMPLATE_CLASSES)}; "
                "unknown cannot be registered"
            )
        relative_path = _portable_relative_path(
            item["relative_path"], f"{context}.relative_path"
        )
        template_path = (template_root / Path(*relative_path.parts)).resolve(strict=True)
        if not _is_relative_to(template_path, template_root):
            raise DualPathContractError(f"{context}.relative_path escapes template_root")
        if not template_path.is_file():
            raise DualPathContractError(f"{context}.relative_path is not a file")

        raw_sha = _required_string(item["raw_sha256"], f"{context}.raw_sha256")
        if not _SHA256_RE.fullmatch(raw_sha):
            raise DualPathContractError(f"{context}.raw_sha256 is invalid")
        if raw_sha in seen_hashes:
            raise DualPathContractError(f"duplicate template raw_sha256: {raw_sha}")
        seen_hashes.add(raw_sha)
        actual_sha = _raw_sha256(template_path)
        if actual_sha != raw_sha:
            raise DualPathContractError(
                f"template hash mismatch for {template_id}: actual={actual_sha} expected={raw_sha}"
            )
        if item["role"] != REGISTERED_ROLE:
            raise DualPathContractError(f"{context}.role must be {REGISTERED_ROLE}")

        templates.append(
            RegisteredTemplate(
                template_id=template_id,
                class_name=class_name,
                path=template_path,
                relative_path=str(relative_path),
                raw_sha256=raw_sha,
                role=REGISTERED_ROLE,
                dataset_record=_validate_dataset_record(
                    item["dataset_record"], f"{context}.dataset_record"
                ),
            )
        )
    return TemplateRegistry(
        path=registry_path,
        status=status,
        template_root=template_root,
        templates=tuple(templates),
    )


def _load_rgb_file(path: str | Path, *, max_pixels: int = 20_000_000) -> tuple[np.ndarray, str, Path]:
    query_path = Path(path).expanduser().resolve(strict=True)
    if not query_path.is_file():
        raise DualPathContractError("query path must be a file")
    raw = query_path.read_bytes()
    raw_sha = hashlib.sha256(raw).hexdigest()
    try:
        with Image.open(query_path) as image:
            width, height = image.size
            if width <= 0 or height <= 0 or width * height > max_pixels:
                raise DualPathContractError("query image dimensions are invalid or too large")
            rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    except (OSError, ValueError) as exc:
        raise DualPathContractError(f"query is not a decodable RGB image: {exc}") from exc
    return np.ascontiguousarray(rgb), raw_sha, query_path


def run_seed17_semantic_hypothesis(
    query_path: str | Path,
    runner: CpuOnnxRunner,
    thresholds: DemoThresholds,
) -> dict[str, Any]:
    """Run the verified CPU-only seed-17 session and expose raw demo scores."""

    if getattr(runner, "model_sha256", None) != SEED17_MODEL_SHA256:
        raise DualPathContractError("semantic runner is not bound to the frozen seed17 ONNX hash")
    if tuple(getattr(runner, "class_order", ())) != tuple(ROOTSCOPE_CLASS_ORDER):
        raise DualPathContractError("semantic runner class order is not frozen RootScope order")
    if tuple(getattr(runner, "expected_output_shape", ())) != (1, 4):
        raise DualPathContractError("semantic runner output shape must be [1,4]")
    if list(getattr(runner, "providers", ())) != [CPU_PROVIDER]:
        raise DualPathContractError("semantic runner must use CPUExecutionProvider only")
    session = getattr(runner, "_session", None)
    if session is None:
        raise DualPathContractError("semantic runner has no verified ONNX session")

    image, query_sha, resolved_query = _load_rgb_file(query_path)
    tensor = preprocess_rgb(image, runner.preprocess)
    values = session.run(
        [runner.output_name],
        {runner.input_name: tensor},
    )
    if len(values) != 1:
        raise DualPathContractError("semantic ONNX session must return exactly one output")
    logits_array = np.asarray(values[0], dtype=np.float64)
    if logits_array.shape != (1, 4) or not np.isfinite(logits_array).all():
        raise DualPathContractError("semantic ONNX logits must be finite [1,4]")
    logits = logits_array[0]
    shifted = logits - float(np.max(logits))
    exponentials = np.exp(shifted)
    probabilities = exponentials / float(np.sum(exponentials))
    descending = np.argsort(-probabilities, kind="stable")
    top1_index = int(descending[0])
    top2_index = int(descending[1])
    top1_probability = float(probabilities[top1_index])
    margin = top1_probability - float(probabilities[top2_index])
    demo_gate = (
        top1_probability >= thresholds.min_top1_probability
        and margin >= thresholds.min_top1_margin
    )
    return {
        "schema": "rootscope.seed17-demo-hypothesis.v1",
        "status": SEMANTIC_STATUS,
        "model": {
            "selection": "seed17",
            "sha256": SEED17_MODEL_SHA256,
            "status": MODEL_STATUS,
            "provider": CPU_PROVIDER,
            "model_candidate": False,
            "model_qualified": False,
            "bpu_ready": False,
            "bpu_used": False,
        },
        "query": {
            "path": str(resolved_query),
            "raw_sha256": query_sha,
            "input_tensor_sha256": hashlib.sha256(
                tensor.tobytes(order="C")
            ).hexdigest(),
        },
        "class_order": list(ROOTSCOPE_CLASS_ORDER),
        "raw_logits": [float(value) for value in logits],
        "softmax": [float(value) for value in probabilities],
        "raw_top1_index": top1_index,
        "raw_top1_class": ROOTSCOPE_CLASS_ORDER[top1_index],
        "raw_top1_probability": top1_probability,
        "raw_top1_margin": margin,
        "experimental_demo_thresholds": thresholds.to_dict(),
        "experimental_demo_threshold_passed": demo_gate,
        "formal_rejection_gate": {
            "status": FORMAL_REJECTION_STATUS,
            "passed": False,
            "reason": "tiny validation support makes the frozen per-class Wilson gate reject all",
        },
        "claim_scope": "RAW_TOP1_DEMO_HYPOTHESIS_ONLY_NOT_PLANT_OR_IRRIGATION_AUTHORITY",
        "authority": dict(AUTHORITY),
    }


def _geometry_to_dict(result: Any) -> dict[str, Any]:
    if hasattr(result, "to_dict"):
        result = result.to_dict()
    if not isinstance(result, Mapping):
        raise DualPathContractError("geometric matcher result must be an object")
    return dict(result)


def _validated_geometry_pass(
    result: Mapping[str, Any], template: RegisteredTemplate, query_sha256: str
) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    expected_bindings = {
        "schema": GEOMETRIC_SCHEMA_VERSION,
        "claim_scope": GEOMETRIC_CLAIM_SCOPE,
        "status": "PASS",
        "passed": True,
        "template_id": template.template_id,
        "template_class": template.class_name,
        "template_sha256": template.raw_sha256,
        "query_sha256": query_sha256,
        "irrigation_execution_authority": False,
    }
    for key, expected in expected_bindings.items():
        if result.get(key) != expected:
            reasons.append(f"GEOMETRIC_{key.upper()}_BINDING_MISMATCH")
    nested_authority = result.get("authority")
    required_geometric_authority = {
        "irrigation_execution",
        "pump_command",
        "serial_write",
        "state_machine_write",
    }
    if (
        not isinstance(nested_authority, Mapping)
        or set(nested_authority) != required_geometric_authority
        or any(
            nested_authority[key] is not False
            for key in required_geometric_authority
        )
    ):
        reasons.append("GEOMETRIC_AUTHORITY_CONTRACT_VIOLATION")
    for key, value in result.items():
        if key.endswith("_authority") and value is not False:
            reasons.append("GEOMETRIC_TOP_LEVEL_AUTHORITY_VIOLATION")
            break
    provenance = result.get("provenance")
    if (
        not isinstance(provenance, Mapping)
        or provenance.get("semantic_recognition_performed") is not False
        or provenance.get("physical_hardware_touched") is not False
    ):
        reasons.append("GEOMETRIC_PROVENANCE_BOUNDARY_VIOLATION")
    return not reasons, reasons


Matcher = Callable[..., Any]


def evaluate_dual_path_demo(
    *,
    query_path: str | Path,
    runner: CpuOnnxRunner,
    registry_path: str | Path,
    thresholds: DemoThresholds | None = None,
    matcher_config: MatcherConfig | None = None,
    matcher: Matcher = match_known_card,
) -> dict[str, Any]:
    """Evaluate one file without opening a camera, serial port, pump, or network."""

    effective_thresholds = thresholds or DemoThresholds()
    registry = load_template_registry(registry_path)
    semantic = run_seed17_semantic_hypothesis(
        query_path, runner, effective_thresholds
    )
    query_resolved = Path(semantic["query"]["path"])
    geometry_items: list[dict[str, Any]] = []
    passed_templates: list[RegisteredTemplate] = []
    for template in registry.templates:
        try:
            raw_result = _geometry_to_dict(
                matcher(
                    template.path,
                    query_resolved,
                    template_id=template.template_id,
                    template_class=template.class_name,
                    config=matcher_config,
                )
            )
            contract_passed, contract_reasons = _validated_geometry_pass(
                raw_result, template, semantic["query"]["raw_sha256"]
            )
            if contract_passed:
                passed_templates.append(template)
            geometry_items.append(
                {
                    "template": template.public_record(),
                    "matcher_result": raw_result,
                    "contract_valid_pass": contract_passed,
                    "contract_reject_reasons": contract_reasons,
                }
            )
        except Exception as exc:  # matcher failure is evidence rejection, never authority
            geometry_items.append(
                {
                    "template": template.public_record(),
                    "matcher_result": None,
                    "contract_valid_pass": False,
                    "contract_reject_reasons": [
                        f"GEOMETRIC_MATCHER_ERROR:{type(exc).__name__}:{exc}"
                    ],
                }
            )

    reasons: list[str] = []
    if not semantic["experimental_demo_threshold_passed"]:
        reasons.append("EXPERIMENTAL_SEMANTIC_DEMO_THRESHOLD_NOT_MET")
    if len(passed_templates) == 0:
        reasons.append("NO_REGISTERED_TEMPLATE_GEOMETRIC_PASS")
    elif len(passed_templates) > 1:
        reasons.append("MULTIPLE_REGISTERED_TEMPLATES_GEOMETRIC_PASS")
    selected = passed_templates[0] if len(passed_templates) == 1 else None
    if selected is not None and semantic["raw_top1_class"] != selected.class_name:
        reasons.append("SEMANTIC_TEMPLATE_CLASS_DISAGREEMENT")

    consensus_passed = not reasons and selected is not None
    consensus = {
        "status": CONSENSUS_STATUS if consensus_passed else "REJECT",
        "passed": consensus_passed,
        "selected_template_id": selected.template_id if consensus_passed else None,
        "selected_template_class": selected.class_name if consensus_passed else None,
        "claim_scope": (
            "EXACT_REGISTERED_CARD_INSTANCE_PLUS_RAW_SEMANTIC_DEMO_CONSENSUS_ONLY"
            "_NOT_GENERAL_PLANT_RECOGNITION_NOT_IRRIGATION_AUTHORITY"
        ),
        "reject_reasons": reasons,
        "authority": dict(AUTHORITY),
    }
    return {
        "schema": SCHEMA_VERSION,
        "status": consensus["status"],
        "experimental_consensus_passed": consensus_passed,
        "registry": {
            "path": str(registry.path),
            "raw_sha256": _raw_sha256(registry.path),
            "status": registry.status,
            "registered_template_count": len(registry.templates),
            "registered_roles": sorted({item.role for item in registry.templates}),
        },
        "semantic": semantic,
        "geometry": {
            "claim_scope": GEOMETRIC_CLAIM_SCOPE,
            "registered_template_count": len(registry.templates),
            "contract_valid_pass_count": len(passed_templates),
            "items": geometry_items,
        },
        "consensus": consensus,
        "claims": {
            "general_plant_recognition": False,
            "model_candidate": False,
            "model_qualified": False,
            "camera_qualified": False,
            "x5_validated": False,
            "bpu_ready": False,
            "irrigation_decision": False,
            "physical_completion": False,
        },
        "authority": dict(AUTHORITY),
        "hardware_touched": False,
        "network_touched": False,
    }


def build_seed17_runner_from_capsule(
    config_path: str | Path, *, model_path: str | Path | None = None
) -> CpuOnnxRunner:
    """Build the same hash-bound CPU runner used by the clean-X5 capsule."""

    config = CapsuleConfig.from_json_file(config_path)
    if not config.model.enabled or config.model.sha256 != SEED17_MODEL_SHA256:
        raise DualPathContractError("capsule config is not the enabled frozen seed17 CPU config")
    effective_model_path = Path(model_path) if model_path is not None else Path(config.model.path)
    return CpuOnnxRunner(
        effective_model_path,
        config.model.sha256,
        config.model.preprocess,
        input_name=config.model.input_name,
        output_name=config.model.output_name,
        expected_output_shape=config.model.output_shape,
        class_order=config.model.class_order,
    )


def _load_optional_object(path: str | None) -> Mapping[str, Any] | None:
    if path is None:
        return None
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise DualPathContractError(f"{path} must contain a JSON object")
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run RootScope zero-authority semantic + registered-card demo evidence"
    )
    parser.add_argument("--query", required=True, help="single RGB image file")
    parser.add_argument("--registry", required=True, help="strict frozen template registry")
    parser.add_argument("--capsule-config", required=True, help="seed17 CPU capsule JSON")
    parser.add_argument(
        "--model-path",
        help="optional local location of the same hash-bound seed17 ONNX",
    )
    parser.add_argument("--thresholds-json", help="strict demo threshold JSON")
    parser.add_argument("--matcher-config-json", help="strict geometric matcher JSON")
    parser.add_argument("--output-json", help="optional evidence output path")
    args = parser.parse_args(argv)
    try:
        threshold_payload = _load_optional_object(args.thresholds_json)
        matcher_payload = _load_optional_object(args.matcher_config_json)
        result = evaluate_dual_path_demo(
            query_path=args.query,
            runner=build_seed17_runner_from_capsule(
                args.capsule_config, model_path=args.model_path
            ),
            registry_path=args.registry,
            thresholds=(
                DemoThresholds.from_mapping(threshold_payload)
                if threshold_payload is not None
                else DemoThresholds()
            ),
            matcher_config=(
                MatcherConfig.from_mapping(matcher_payload)
                if matcher_payload is not None
                else MatcherConfig()
            ),
        )
        rendered = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True)
        print(rendered)
        if args.output_json:
            Path(args.output_json).write_text(rendered + "\n", encoding="utf-8")
        return 0 if result["experimental_consensus_passed"] else 2
    except (
        DualPathContractError,
        OnnxCpuContractError,
        OSError,
        ValueError,
        json.JSONDecodeError,
    ) as exc:
        error = {
            "schema": SCHEMA_VERSION,
            "status": "ERROR",
            "experimental_consensus_passed": False,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "authority": dict(AUTHORITY),
            "hardware_touched": False,
            "network_touched": False,
        }
        print(json.dumps(error, ensure_ascii=False, indent=2, sort_keys=True))
        return 1


if __name__ == "__main__":
    sys.exit(main())
