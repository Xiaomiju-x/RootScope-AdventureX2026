#!/usr/bin/env python3
"""Independent post-build audit for the RootScope seed-17 Bayes-e search.

This auditor does not import the staging builder or replay program.  It
recomputes hashes, replay metrics and gates from primary artifacts, validates
the RGB/NCHW/DDR mapper contract, inspects compiled-graph metadata and
quantized ONNX structure, and fails closed if any r3-r7 candidate is presented
as eligible.  It is read-only except for its explicitly requested evidence
JSON and never invokes Docker, WSL, SSH, an X5, a camera, or irrigation I/O.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re
from typing import Any, Iterable

import onnx


WORKSPACE = Path(__file__).resolve().parents[2]
EVIDENCE_REL = Path("evidence/rootscope_seed17_bpu_compile_20260717")
STAGING_REL = Path("output/rootscope_bpu_seed17_staging_20260717_r3")
EXPECTED = {
    "source_model": "50ae8d2ec1cec0f3748efa127c6b9c00624684f8c112c8a7913c2a0e304a3bad",
    "dataset_manifest": "bf14c7423aad965b8af736c7d77cef1ba134d78dd1f905c03cc14cff1192f3fe",
    "calibration_manifest": "8f4d5370d3015062271eb86817e38f2cf6b4bc1ac5364fabe94dff5ae28246e6",
    "staging_receipt": "d61561291dfca23bcc658ca23255b132e910a041c8a9a887d46ccbc02ff05cac",
    "staging_sums": "a2d160cd26c48edf4e750af24e51231668661bc24770939566c41c352364de22",
    "generation1_plan": "a20afba3e6dff07cbfc9e6986670967aabc7211c2dcf9b2dcf956fbfedbda52b",
    "generation1_result": "0bac2d9c11646be2cb44b3d8c45c448381dcdcad30f633870d7a01b0151dc5d0",
    "generation2_plan": "976dbbe7ef39b3177b904a56695deef06e2f19f3cf23def1c0d11b90126d2e0d",
    "generation2_result": "7c981626fbbf1f29e723c642d431d4a6ff1eb006eeab0df76641b7e76d02c9dd",
}
GATES = {
    "top1_agreement_min_count": 23,
    "softmax_l1_mean_max": 0.03,
    "softmax_l1_p99_max": 0.10,
    "softmax_l1_max": 0.15,
    "centered_logit_abs_mean_max": 0.15,
    "centered_logit_abs_p99_max": 0.50,
    "centered_logit_abs_max": 1.00,
    "mapper_float_vs_frozen_max_abs_max": 0.0001,
}
EXPECTED_ROLE_COUNTS = {
    "EXPERIMENTAL_VAL_SUGGESTION": 9,
    "PRINT_DEMO_HOLDOUT_NOT_TRAIN": 6,
    "CREATOR_GROUP_HOLDOUT_NOT_TRAIN": 8,
}
TOOLCHAIN = {
    "image": "openexplorer/ai_toolchain_ubuntu_20_x5_cpu:v1.2.8-py310",
    "image_id": "sha256:9c536e7bbdcaf842ab68f7f335185aa61ef5c2536ee61e6f53fc7a45f1bf81c0",
    "repo_digest": "openexplorer/ai_toolchain_ubuntu_20_x5_cpu@sha256:9c536e7bbdcaf842ab68f7f335185aa61ef5c2536ee61e6f53fc7a45f1bf81c0",
    "hb_mapper": "1.24.3",
    "hbdk": "3.49.15",
    "hbdk_runtime": "3.15.55.0",
    "horizon_nn": "1.1.0",
}

CANDIDATES = (
    {
        "generation": "r3",
        "directory": "output/rootscope_bpu_seed17_staging_20260717_r3",
        "prefix": "rootscope_seed17_resnet18_224x224_rgb_ddr",
        "config": "config/rootscope_seed17_bayes_e.yaml",
        "replay": "r3_horizon_x86_replay.json",
        "calibration_fragments": ("calibration_type: 'default'",),
        "expected_bpu_dtype": "int8",
        "min_dtype_rows": 23,
    },
    {
        "generation": "r4",
        "directory": "output/rootscope_bpu_seed17_quant_variant_r4_max_pc_true_p100",
        "prefix": "rootscope_seed17_resnet18_224x224_rgb_ddr_r4_max_pc_true_p100",
        "config": "config.yaml",
        "replay": "r4_horizon_x86_replay.json",
        "calibration_fragments": (
            "calibration_type: 'max'", "per_channel: True", "max_percentile: 1.0"
        ),
        "expected_bpu_dtype": "int8",
        "min_dtype_rows": 23,
    },
    {
        "generation": "r5",
        "directory": "output/rootscope_bpu_seed17_quant_variant_r5_max_pc_true_p099995",
        "prefix": "rootscope_seed17_resnet18_224x224_rgb_ddr_r5_max_pc_true_p099995",
        "config": "config.yaml",
        "replay": "r5_horizon_x86_replay.json",
        "calibration_fragments": (
            "calibration_type: 'max'", "per_channel: True", "max_percentile: 0.99995"
        ),
        "expected_bpu_dtype": "int8",
        "min_dtype_rows": 23,
    },
    {
        "generation": "r6",
        "directory": "output/rootscope_bpu_seed17_quant_variant_r6_kl_pc_true",
        "prefix": "rootscope_seed17_resnet18_224x224_rgb_ddr_r6_kl_pc_true",
        "config": "config.yaml",
        "replay": "r6_horizon_x86_replay.json",
        "calibration_fragments": ("calibration_type: 'kl'", "per_channel: True"),
        "expected_bpu_dtype": "int8",
        "min_dtype_rows": 23,
    },
    {
        "generation": "r7",
        "directory": "output/rootscope_bpu_seed17_quant_variant_r7_default_int16_all_nodes",
        "prefix": "rootscope_seed17_resnet18_224x224_rgb_ddr_r7_default_int16_all_nodes",
        "config": "config.yaml",
        "replay": "r7_horizon_x86_replay.json",
        "calibration_fragments": (
            "calibration_type: 'default'", "optimization: 'set_all_nodes_int16'"
        ),
        "expected_bpu_dtype": "int16",
        "min_dtype_rows": 40,
    },
)


class AuditError(RuntimeError):
    """A post-build contract failed."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise AuditError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=reject_duplicates)
    if not isinstance(value, dict):
        raise AuditError(f"JSON root is not an object: {path}")
    return value


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            raise AuditError(f"blank JSONL row: {path}:{number}")
        row = json.loads(line, object_pairs_hook=reject_duplicates)
        if not isinstance(row, dict):
            raise AuditError(f"JSONL row is not an object: {path}:{number}")
        rows.append(row)
    return rows


def safe_relative(root: Path, relative: str) -> Path:
    if not isinstance(relative, str) or not relative or "\\" in relative:
        raise AuditError(f"non-canonical path: {relative!r}")
    pure = PurePosixPath(relative)
    if pure.is_absolute() or any(part in {"", ".", ".."} for part in pure.parts):
        raise AuditError(f"unsafe path: {relative!r}")
    target = (root / Path(*pure.parts)).resolve(strict=True)
    try:
        target.relative_to(root.resolve(strict=True))
    except ValueError as error:
        raise AuditError(f"path escapes root: {relative!r}") from error
    if target.is_symlink() or not target.is_file():
        raise AuditError(f"not a regular file: {relative!r}")
    return target


def softmax(values: Iterable[float]) -> list[float]:
    array = [float(value) for value in values]
    offset = max(array)
    exps = [math.exp(value - offset) for value in array]
    total = sum(exps)
    return [value / total for value in exps]


def higher_quantile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    index = math.ceil(q * (len(ordered) - 1))
    return ordered[index]


def summary(values: list[float]) -> dict[str, float | int]:
    if not values or not all(math.isfinite(value) for value in values):
        raise AuditError("metric vector is empty or non-finite")
    return {
        "count": len(values),
        "mean": sum(values) / len(values),
        "p99_higher": higher_quantile(values, 0.99),
        "max": max(values),
    }


def close_numeric(actual: Any, expected: Any, tolerance: float = 1e-12) -> bool:
    return abs(float(actual) - float(expected)) <= tolerance


def inspect_quantized_onnx(path: Path) -> dict[str, Any]:
    model = onnx.load(str(path), load_external_data=True)
    input_tensor = model.graph.input[0].type.tensor_type
    output_tensor = model.graph.output[0].type.tensor_type
    ops = Counter(node.op_type for node in model.graph.node)
    first_ops = [node.op_type for node in model.graph.node[:3]]
    if input_tensor.elem_type != onnx.TensorProto.INT8:
        raise AuditError(f"quantized ONNX input is not INT8 replay interface: {path}")
    if output_tensor.elem_type != onnx.TensorProto.FLOAT:
        raise AuditError(f"quantized ONNX output is not float logits: {path}")
    if first_ops[:2] != ["Transpose", "HzSQuantizedPreprocess"]:
        raise AuditError(f"quantized ONNX preprocess prefix differs: {first_ops}")
    return {
        "input_elem_type": int(input_tensor.elem_type),
        "output_elem_type": int(output_tensor.elem_type),
        "node_count": len(model.graph.node),
        "first_ops": first_ops,
        "operator_counts": dict(sorted(ops.items())),
    }


def recompute_replay(report: dict[str, Any]) -> dict[str, Any]:
    rows = report.get("rows")
    if not isinstance(rows, list) or len(rows) != 23:
        raise AuditError("replay must contain exactly 23 rows")
    role_counts = Counter(str(row.get("role")) for row in rows)
    if dict(role_counts) != EXPECTED_ROLE_COUNTS:
        raise AuditError(f"replay role population differs: {dict(role_counts)}")
    image_hashes = [str(row.get("image_sha256")) for row in rows]
    if len(set(image_hashes)) != 23 or any(not re.fullmatch(r"[0-9a-f]{64}", value) for value in image_hashes):
        raise AuditError("replay image identities are not 23 unique SHA-256 values")

    softmax_l1: list[float] = []
    centered: list[float] = []
    mapper_abs: list[float] = []
    top1 = 0
    for row in rows:
        source = [float(value) for value in row["source_logits"]]
        mapper = [float(value) for value in row["mapper_float_logits"]]
        quant = [float(value) for value in row["quantized_logits"]]
        if not (len(source) == len(mapper) == len(quant) == 4):
            raise AuditError("replay logit vector length differs")
        source_probability = softmax(source)
        quant_probability = softmax(quant)
        image_l1 = sum(abs(a - b) for a, b in zip(source_probability, quant_probability))
        softmax_l1.append(image_l1)
        source_mean = sum(source) / 4
        quant_mean = sum(quant) / 4
        centered.extend(abs((a - source_mean) - (b - quant_mean)) for a, b in zip(source, quant))
        mapper_abs.extend(abs(a - b) for a, b in zip(source, mapper))
        source_top1 = max(range(4), key=source.__getitem__)
        quant_top1 = max(range(4), key=quant.__getitem__)
        top1 += int(source_top1 == quant_top1)
        if not close_numeric(row["softmax_l1"], image_l1):
            raise AuditError(f"stored per-image softmax L1 differs: {row.get('filename')}")

    softmax_result = summary(softmax_l1)
    centered_result = summary(centered)
    mapper_result = summary(mapper_abs)
    gates = {
        "top1_agreement_23_of_23": top1 >= GATES["top1_agreement_min_count"],
        "softmax_l1_mean": softmax_result["mean"] <= GATES["softmax_l1_mean_max"],
        "softmax_l1_p99": softmax_result["p99_higher"] <= GATES["softmax_l1_p99_max"],
        "softmax_l1_max": softmax_result["max"] <= GATES["softmax_l1_max"],
        "centered_logit_abs_mean": centered_result["mean"] <= GATES["centered_logit_abs_mean_max"],
        "centered_logit_abs_p99": centered_result["p99_higher"] <= GATES["centered_logit_abs_p99_max"],
        "centered_logit_abs_max": centered_result["max"] <= GATES["centered_logit_abs_max"],
        "mapper_float_vs_frozen_max_abs": mapper_result["max"] <= GATES["mapper_float_vs_frozen_max_abs_max"],
    }
    stored = report.get("metrics", {})
    comparisons = (
        (stored.get("top1_agreement_count"), top1),
        (stored.get("softmax_l1_per_image", {}).get("mean"), softmax_result["mean"]),
        (stored.get("softmax_l1_per_image", {}).get("p99_higher"), softmax_result["p99_higher"]),
        (stored.get("softmax_l1_per_image", {}).get("max"), softmax_result["max"]),
        (stored.get("centered_logit_abs_per_class_value", {}).get("mean"), centered_result["mean"]),
        (stored.get("centered_logit_abs_per_class_value", {}).get("p99_higher"), centered_result["p99_higher"]),
        (stored.get("centered_logit_abs_per_class_value", {}).get("max"), centered_result["max"]),
        (stored.get("mapper_float_vs_frozen_abs_per_class_value", {}).get("max"), mapper_result["max"]),
    )
    if not all(close_numeric(actual, expected) for actual, expected in comparisons):
        raise AuditError("stored replay aggregate differs from independent recomputation")
    if report.get("gate_results") != gates:
        raise AuditError("stored gate vector differs from independent recomputation")
    return {
        "top1_agreement_count": top1,
        "softmax_l1": softmax_result,
        "centered_logit_abs": centered_result,
        "mapper_float_abs": mapper_result,
        "gate_results": gates,
        "all_gates_passed": all(gates.values()),
        "role_counts": dict(role_counts),
        "replay_image_hashes": image_hashes,
    }


def artifact(path: Path, workspace: Path) -> dict[str, Any]:
    return {
        "path": path.relative_to(workspace).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def audit(workspace: Path, xrd_root: Path) -> dict[str, Any]:
    workspace = workspace.resolve(strict=True)
    xrd_root = xrd_root.resolve(strict=True)
    checks: list[str] = []

    def require(condition: bool, name: str) -> None:
        if not condition:
            raise AuditError(name)
        checks.append(name)

    staging = workspace / STAGING_REL
    primary = {
        "source_model": staging / "model/rootscope_seed17_resnet18_static_b1x3x224x224_opset11.onnx",
        "dataset_manifest": workspace / "datasets/rootscope_machine_curated_provisional_v3/manifest.jsonl",
        "calibration_manifest": staging / "calibration_manifest.jsonl",
        "staging_receipt": staging / "staging_receipt.json",
        "staging_sums": staging / "STAGING_SHA256SUMS",
        "generation1_plan": workspace / EVIDENCE_REL / "quant_variant_search_plan.json",
        "generation1_result": workspace / EVIDENCE_REL / "quant_variant_search_generation1_result.json",
        "generation2_plan": workspace / EVIDENCE_REL / "quant_variant_search_generation2_plan.json",
        "generation2_result": workspace / EVIDENCE_REL / "quant_variant_search_generation2_result.json",
    }
    for key, path in primary.items():
        require(path.is_file() and not path.is_symlink(), f"primary artifact missing: {key}")
        require(sha256_file(path) == EXPECTED[key], f"primary artifact SHA differs: {key}")

    sums_rows: dict[str, str] = {}
    for number, line in enumerate(primary["staging_sums"].read_text(encoding="utf-8").splitlines(), 1):
        match = re.fullmatch(r"([0-9a-f]{64})  ([^\r\n]+)", line)
        require(match is not None, f"invalid immutable staging sums row {number}")
        digest, relative = match.groups()
        require(relative not in sums_rows, f"duplicate immutable staging sums path: {relative}")
        sums_rows[relative] = digest
    for relative, digest in sums_rows.items():
        require(sha256_file(safe_relative(staging, relative)) == digest, f"immutable staging input changed: {relative}")

    receipt = load_json(primary["staging_receipt"])
    for key in ("bpu_compiled", "x5_ready", "model_qualified"):
        require(receipt.get("formal_flags", {}).get(key) is False, f"staging receipt was rewritten: {key}")
    calibration_rows = load_jsonl(primary["calibration_manifest"])
    require(len(calibration_rows) == 256, "calibration population is not 256")
    calibration_hashes = {row.get("source_sha256") for row in calibration_rows}

    gen1 = load_json(primary["generation1_result"])
    gen2_plan = load_json(primary["generation2_plan"])
    gen2 = load_json(primary["generation2_result"])
    require(gen1.get("selection", {}).get("selected_variant") is None, "generation1 selected a failed variant")
    require(gen2_plan.get("generation1_result", {}).get("selected_variant") is None, "generation2 plan did not bind null generation1 selection")
    require(gen2.get("selection", {}).get("selected_variant") is None, "generation2 selected a failed variant")
    require(gen2.get("selection", {}).get("publishable_default_bpu_bin") is None, "generation2 exposed a default bin")

    official = (
        (
            "tools/bpu_transformer/rdk_model_zoo/samples/vision/classification/EfficientFormer/yaml/EfficientFormer_l1_config.yaml",
            "62c082781fb8b830ad319b27b7b233e83921a8619018d1cd13c12910fb03110c",
        ),
        (
            "tools/bpu_transformer/rdk_model_zoo/samples/vision/PaddleOCR/yaml/paddleocr_rec_config.yaml",
            "0955a52b670a259ee4b3d9330402e8b4787dafa7b5efb9a3acea0e432dc32482",
        ),
        (
            "tools/bpu_transformer/rdk_model_zoo/samples/vision/yolov5/ptq_yamls/yolov5_detect_bayese_640x640_nchw.yaml",
            "1bd2562c7368c3ee3c4f950e72e63a365dea40e25a31878c04b470e383319b9d",
        ),
        (
            "tools/bpu_transformer/rdk_model_zoo/samples/vision/yolov5/YOLOv5_Detect.py",
            "8e20ca8480b16beb1c3d5583171cd704bcc5f7f1c5ab0b52cdaeb04290a820b8",
        ),
    )
    official_artifacts: list[dict[str, Any]] = []
    for relative, digest in official:
        path = safe_relative(xrd_root, relative)
        require(sha256_file(path) == digest, f"local official syntax evidence changed: {relative}")
        text = path.read_text(encoding="utf-8", errors="replace")
        if "set_all_nodes_int16" in relative or "EfficientFormer" in relative or "paddleocr" in relative:
            require("set_all_nodes_int16" in text, f"int16 syntax absent: {relative}")
        official_artifacts.append({"path_from_xrd_root": relative, "sha256": digest})

    gen1_records = {row["generation"]: row for row in gen1.get("results", [])}
    gen2_record = gen2.get("candidate", {})
    results: list[dict[str, Any]] = []
    for spec in CANDIDATES:
        generation = spec["generation"]
        directory = (workspace / spec["directory"]).resolve(strict=True)
        config_path = directory / spec["config"]
        log_path = directory / "hb_mapper_makertbin.log"
        bin_path = directory / "model_output" / f"{spec['prefix']}.bin"
        quantized_path = directory / "model_output" / f"{spec['prefix']}_quantized_model.onnx"
        subgraph_path = directory / "model_output/main_graph_subgraph_0.json"
        replay_path = workspace / EVIDENCE_REL / spec["replay"]
        for path in (config_path, log_path, bin_path, quantized_path, subgraph_path, replay_path):
            require(path.is_file() and not path.is_symlink(), f"{generation} artifact missing: {path.name}")

        config_text = config_path.read_text(encoding="utf-8")
        for fragment in (
            "input_type_rt: 'rgb'",
            "input_layout_rt: 'NCHW'",
            "input_type_train: 'rgb'",
            "input_layout_train: 'NCHW'",
            "input_source: {'image': 'ddr'}",
            *spec["calibration_fragments"],
        ):
            require(fragment in config_text, f"{generation} config lacks {fragment}")

        log_text = log_path.read_text(encoding="utf-8", errors="replace")
        for fragment in (
            "hb_mapper version 1.24.3",
            "hbdk version 3.49.15",
            "input_type_rt       : rgb",
            "input_layout_rt     : NCHW",
            "input-source        : {'image': 'ddr'",
            "--input-source', 'ddr'",
            "Convert to runtime bin file successfully!",
        ):
            require(fragment in log_text, f"{generation} mapper log lacks {fragment}")
        for forbidden in ("'image': 'pyramid'", "'--input-source', 'pyramid'", "Default input-source pyramid"):
            require(forbidden not in log_text, f"{generation} unexpectedly uses pyramid input")
        if generation == "r7":
            require("All nodes in the model are set to datatype: int16" in log_text, "r7 did not enable all-int16")
            require("optimization        : set_all_nodes_int16;" in log_text, "r7 mapper did not record int16 optimization")

        dtype_rows = re.findall(r"(?m)^.*\sBPU\s+id\(0\).*\s(int8|int16)\s*$", log_text)
        require(dtype_rows.count(spec["expected_bpu_dtype"]) >= spec["min_dtype_rows"], f"{generation} BPU dtype placement differs")

        subgraph = load_json(subgraph_path).get("summary", {})
        require(subgraph.get("BPU march") == "B25E", f"{generation} subgraph march differs")
        require("--input-source ddr" in str(subgraph.get("compiling options")), f"{generation} subgraph is not DDR")
        require(subgraph.get("input features") == [["input name", "input size"], ["image", "1x3x224x224"]], f"{generation} subgraph input differs")

        replay = load_json(replay_path)
        replay_result = recompute_replay(replay)
        require(replay_result["all_gates_passed"] is False, f"{generation} unexpectedly became eligible")
        require(set(replay_result["replay_image_hashes"]).isdisjoint(calibration_hashes), f"{generation} replay overlaps calibration sources")
        quant_binding = replay.get("artifact_bindings", {}).get("horizon_quantized_onnx", {})
        require(quant_binding.get("sha256") == sha256_file(quantized_path), f"{generation} replay quantized ONNX binding differs")
        require(replay.get("formal_flags", {}).get("x5_replay_executed") is False, f"{generation} overstates X5 replay")
        require(replay.get("formal_flags", {}).get("x5_ready") is False, f"{generation} overstates X5 readiness")

        record = gen2_record if generation == "r7" else gen1_records[generation]
        require(record.get("all_replay_gates_passed", record.get("horizon_x86_replay", {}).get("all_gates_passed")) is False, f"{generation} result record overstates gates")
        for key, path in (("bin", bin_path), ("quantized_onnx", quantized_path), ("replay", replay_path)):
            binding = record.get(key)
            if generation == "r7":
                binding = record.get("compilation", {}).get(key) if key != "replay" else record.get("horizon_x86_replay")
            require(isinstance(binding, dict) and binding.get("sha256") == sha256_file(path), f"{generation} result hash binding differs: {key}")

        results.append({
            "generation": generation,
            "eligible": False,
            "config": artifact(config_path, workspace),
            "mapper_log": artifact(log_path, workspace),
            "bin": artifact(bin_path, workspace),
            "quantized_onnx": {**artifact(quantized_path, workspace), "contract": inspect_quantized_onnx(quantized_path)},
            "subgraph": {
                **artifact(subgraph_path, workspace),
                "estimated_fps": subgraph.get("FPS"),
                "estimated_latency_us": subgraph.get("latency (us)"),
                "estimated_ddr_bytes": subgraph.get("DDR bytes per frame"),
            },
            "replay": {**artifact(replay_path, workspace), **replay_result},
            "bpu_dtype_table_counts": dict(Counter(dtype_rows)),
        })

    require(not any(item["eligible"] for item in results), "an eligible variant exists despite null selection")
    checker = staging / "hb_mapper_checker.log"
    checker_text = checker.read_text(encoding="utf-8", errors="replace")
    require("End model checking...." in checker_text, "r3 checker did not complete")
    require("End to compile the model with march bayes-e." in checker_text, "r3 checker did not compile Bayes-e graph")

    return {
        "schema_version": "rootscope.seed17.bayes_e.postbuild_independent_audit.v1",
        "status": "PASS_NO_ELIGIBLE_DEFAULT_BPU",
        "created_date": "2026-07-17",
        "check_count": len(checks),
        "checks": checks,
        "toolchain": TOOLCHAIN,
        "immutable_staging_input_hash_count": len(sums_rows),
        "official_local_evidence": official_artifacts,
        "candidates": results,
        "selection": {
            "eligible_generations": [],
            "selected_variant": None,
            "publishable_default_bpu_bin": None,
        },
        "formal_flags": {
            "postbuild_audit_passed": True,
            "bpu_compilation_executed": True,
            "bpu_binaries_present": True,
            "eligible_bpu_variant_present": False,
            "publishable_default_bpu_bin_present": False,
            "bpu_bin_executed": False,
            "x5_replay_executed": False,
            "x5_ready": False,
            "camera_replay_executed": False,
            "model_qualified": False,
            "execution_authority": False,
            "irrigation_authority": False,
        },
        "read_only_audit": True,
        "docker_wsl_ssh_device_invocations_by_auditor": 0,
    }


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=WORKSPACE)
    parser.add_argument("--xrd-root", type=Path, default=WORKSPACE.parent)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    workspace = args.workspace.resolve(strict=True)
    output = args.out.resolve(strict=False)
    try:
        output.relative_to(workspace)
    except ValueError as error:
        raise AuditError("audit output must stay inside AdventureX") from error
    report = audit(workspace, args.xrd_root)
    write_json(output, report)
    print(json.dumps({
        "status": report["status"],
        "check_count": report["check_count"],
        "selected_variant": report["selection"]["selected_variant"],
        "output": str(output),
        "output_sha256": sha256_file(output),
    }, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
