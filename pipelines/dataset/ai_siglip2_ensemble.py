#!/usr/bin/env python3
"""Two-stage, fail-closed SigLIP2 ensemble for RootScope image curation.

This program is an AI-only suggestion generator.  It deliberately cannot
write the dataset manifest or the formal ``human_decisions`` journal.  Its
only production output is the immutable directory
``review/ai_ensemble_v1``.

The first stage contrasts whole-plant quality prompts with explicit reject
families.  The second stage classifies admissible images as grass clump, low
shrub, or young tree.  Conservative gates produce ``AUTO_TARGET``,
``AUTO_UNKNOWN``, or ``HOLD`` suggestions; none of those decisions grants
training, print, split, rights, or DATA_LOCKED authority.

Production inference loads a self-contained local SigLIP/SigLIP2 model with
``local_files_only=True``, ``trust_remote_code=False`` and safetensors-only
weights.  Tests may inject a fake scorer, but only behind the explicit
fixture boundary.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import stat
import sys
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, Protocol, Sequence

from PIL import Image, UnidentifiedImageError


SCRIPT_PATH = Path(__file__).resolve()
ADVENTUREX_ROOT = SCRIPT_PATH.parents[2]
DEFAULT_STAGING = ADVENTUREX_ROOT / "datasets" / "desert_plants_wikimedia_staging_e0"
DEFAULT_MANIFEST = DEFAULT_STAGING / "manifest.jsonl"
DEFAULT_QUEUE = DEFAULT_STAGING / "review" / "candidate_review_queue.jsonl"
DEFAULT_QUEUE_SUMMARY = DEFAULT_STAGING / "review" / "review_queue_summary.json"
DEFAULT_INTEGRITY_AUDIT = DEFAULT_STAGING / "integrity_audit.json"
DEFAULT_CLASS_CONTRACT = ADVENTUREX_ROOT / "rootscope" / "configs" / "class_contract.json"
DEFAULT_POLICY = SCRIPT_PATH.with_name("ai_siglip2_ensemble_policy_v1.json")
DEFAULT_OUTPUT_DIR = DEFAULT_STAGING / "review" / "ai_ensemble_v1"

HEX64 = re.compile(r"^[0-9a-f]{64}$")
MODEL_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/@:-]{1,127}$")
PROMPT_ID = re.compile(r"^[a-z][a-z0-9_.-]{2,95}$")
WINDOWS_RESERVED = {
    "CON",
    "PRN",
    "AUX",
    "NUL",
    *(f"COM{index}" for index in range(1, 10)),
    *(f"LPT{index}" for index in range(1, 10)),
}

TARGET_CLASSES = ["grass_clump", "low_shrub", "young_tree"]
ALL_HINTS = [*TARGET_CLASSES, "unknown"]
REJECT_FAMILIES = [
    "detail_crop",
    "non_photo_document",
    "no_target_object",
    "landscape_mixed",
    "out_of_contract",
    "poor_quality",
]

AUTHORITY = {
    "human_review": False,
    "dataset_manifest_write": False,
    "training_eligibility": False,
    "split_assignment": False,
    "print_eligibility": False,
    "data_locked": False,
}

POLICY_FIELDS = {
    "schema_version",
    "queue_schema_version",
    "manifest_schema_version",
    "result_schema_version",
    "hold_schema_version",
    "stats_schema_version",
    "receipt_schema_version",
    "production_input_roots",
    "expected_candidate_count",
    "expected_acquisition_hint_counts",
    "target_classes",
    "all_class_hints",
    "image_contract",
    "inference",
    "thresholds",
    "output_contract",
    "authority",
    "explicit_non_claims",
}

QUEUE_FIELDS = {
    "acquisition_mode",
    "acquisition_query",
    "asset",
    "class_hint",
    "class_hint_status",
    "creator",
    "creator_group",
    "dhash64",
    "download_height",
    "download_mime",
    "download_width",
    "license",
    "license_policy_sha256",
    "license_raw_name",
    "license_raw_url",
    "license_url",
    "license_url_basis",
    "local_path",
    "near_duplicate_family",
    "notes",
    "pageid",
    "print_eligible",
    "review_status",
    "reviewed_source_group",
    "reviewer",
    "rights_decision",
    "schema_version",
    "sha256",
    "source_group",
    "source_url",
    "species_hint",
    "species_hint_status",
    "split",
    "target_class",
    "title",
    "training_eligible",
    "visual_decision",
}

MANIFEST_REQUIRED_FIELDS = {
    "schema_version",
    "class_id",
    "domain",
    "split",
    "review_status",
    "training_eligible",
    "print_eligible",
    "source_provider",
    "source_group",
    "pageid",
    "source_page",
    "download_url",
    "artist",
    "license_canonical_name",
    "license_canonical_url",
    "filename",
    "download_sha256",
    "download_width",
    "download_height",
    "download_mime",
}


class EnsembleError(RuntimeError):
    """A fail-closed policy, binding, inference, or output error."""


class PromptScorer(Protocol):
    """Fixture-injectable raw image/text logit scorer."""

    def score(
        self,
        images: Sequence[Path],
        prompts: Sequence[Mapping[str, str]],
    ) -> Sequence[Sequence[float]]:
        """Return one finite raw logit per image and prompt, in input order."""


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def _parse_json(raw: bytes, context: str) -> Any:
    try:
        return json.loads(raw.decode("utf-8"), object_pairs_hook=_reject_duplicate_keys)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise EnsembleError(f"{context} is not strict UTF-8 JSON: {exc}") from exc


def _read_json(path: Path, context: str) -> tuple[Any, bytes]:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise EnsembleError(f"cannot read {context}: {exc}") from exc
    return _parse_json(raw, context), raw


def _read_jsonl(path: Path, context: str) -> tuple[list[dict[str, Any]], bytes]:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise EnsembleError(f"cannot read {context}: {exc}") from exc
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise EnsembleError(f"{context} is not UTF-8") from exc
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line:
            raise EnsembleError(f"{context} line {line_number} is blank")
        item = _parse_json(line.encode("utf-8"), f"{context} line {line_number}")
        if not isinstance(item, dict):
            raise EnsembleError(f"{context} line {line_number} is not an object")
        rows.append(item)
    if not rows:
        raise EnsembleError(f"{context} is empty")
    return rows, raw


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _jsonl_bytes(rows: Iterable[Mapping[str, Any]]) -> bytes:
    return b"".join(_canonical_bytes(row) + b"\n" for row in rows)


def _is_number(value: Any) -> bool:
    return type(value) in {int, float} and math.isfinite(float(value))


def _exact_keys(value: Any, expected: set[str], context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise EnsembleError(f"{context} must be an object")
    actual = set(value)
    if actual != expected:
        raise EnsembleError(
            f"{context} fields do not match strict schema; "
            f"missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
        )
    return value


def _required_keys(value: Any, required: set[str], context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise EnsembleError(f"{context} must be an object")
    missing = required - set(value)
    if missing:
        raise EnsembleError(f"{context} is missing required fields: {sorted(missing)}")
    return value


def _is_reparse(info: os.stat_result) -> bool:
    flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    return bool(getattr(info, "st_file_attributes", 0) & flag)


def _fingerprint(info: os.stat_result) -> tuple[int, int, int, int, int, int]:
    return (
        stat.S_IFMT(info.st_mode),
        info.st_size,
        getattr(info, "st_mtime_ns", int(info.st_mtime * 1_000_000_000)),
        getattr(info, "st_ctime_ns", int(info.st_ctime * 1_000_000_000)),
        info.st_dev,
        info.st_ino,
    )


def _open_identity(info: os.stat_result) -> tuple[int, int, int, int, int]:
    """Identity fields that are stable between path-stat and handle-stat on Windows."""

    return (
        stat.S_IFMT(info.st_mode),
        info.st_size,
        getattr(info, "st_mtime_ns", int(info.st_mtime * 1_000_000_000)),
        info.st_dev,
        info.st_ino,
    )


def _regular_file_lstat(path: Path, context: str) -> os.stat_result:
    try:
        info = os.lstat(path)
    except OSError as exc:
        raise EnsembleError(f"cannot inspect {context}: {path}: {exc}") from exc
    if stat.S_ISLNK(info.st_mode) or _is_reparse(info):
        raise EnsembleError(f"{context} must not be a link or reparse point: {path}")
    if not stat.S_ISREG(info.st_mode):
        raise EnsembleError(f"{context} is not a regular file: {path}")
    return info


def _sha256_file(path: Path) -> str:
    before = _regular_file_lstat(path, "hashed input")
    digest = hashlib.sha256()
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise EnsembleError(f"cannot open hashed input: {path}: {exc}") from exc
    try:
        with os.fdopen(descriptor, "rb") as handle:
            descriptor = -1
            opened = os.fstat(handle.fileno())
            # Windows may expose a different ctime representation through an
            # open handle than through lstat().  Size, mtime and file identity
            # still bind the object; the full fingerprint is used before/after
            # hashing to catch real mutation.
            if _open_identity(opened) != _open_identity(before):
                raise EnsembleError(f"hashed input changed while it was opened: {path}")
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
            if _fingerprint(os.fstat(handle.fileno())) != _fingerprint(opened):
                raise EnsembleError(f"hashed input changed while it was hashed: {path}")
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    if _fingerprint(_regular_file_lstat(path, "hashed input")) != _fingerprint(before):
        raise EnsembleError(f"hashed input changed while it was hashed: {path}")
    return digest.hexdigest()


def _safe_relative_path(root: Path, value: Any, context: str) -> tuple[Path, str]:
    if not isinstance(value, str) or not value or "\\" in value:
        raise EnsembleError(f"{context} must be a non-empty POSIX relative path")
    pure = PurePosixPath(value)
    if pure.is_absolute() or any(part in {"", ".", ".."} for part in pure.parts):
        raise EnsembleError(f"{context} is unsafe")
    for part in pure.parts:
        stem = part.split(".", 1)[0].upper()
        if ":" in part or part.endswith((" ", ".")) or stem in WINDOWS_RESERVED:
            raise EnsembleError(f"{context} is unsafe on Windows")
    unresolved = root.joinpath(*pure.parts)
    _regular_file_lstat(unresolved, context)
    try:
        resolved_root = root.resolve(strict=True)
        candidate = unresolved.resolve(strict=True)
        candidate.relative_to(resolved_root)
    except (OSError, ValueError) as exc:
        raise EnsembleError(f"{context} escapes the staging root or is missing") from exc
    return candidate, pure.as_posix()


def _artifact_root(path: Path) -> dict[str, Any]:
    try:
        source_info = os.lstat(path)
    except OSError as exc:
        raise EnsembleError(f"local model artifact does not exist: {path}") from exc
    if stat.S_ISLNK(source_info.st_mode) or _is_reparse(source_info):
        raise EnsembleError("local model artifact must not be a link or reparse point")
    try:
        root = path.resolve(strict=True)
    except OSError as exc:
        raise EnsembleError(f"local model artifact does not exist: {path}") from exc
    if root.is_file():
        size = _regular_file_lstat(root, "local model artifact").st_size
        if size <= 0:
            raise EnsembleError("local model artifact file is empty")
        sha = _sha256_file(root)
        entries = [{"path": root.name, "bytes": size, "sha256": sha}]
        return {
            "kind": "file",
            "sha256": _sha256_bytes(_canonical_bytes(entries)),
            "file_count": 1,
            "byte_count": size,
            "entries_sha256": _sha256_bytes(_canonical_bytes(entries)),
        }
    if not root.is_dir():
        raise EnsembleError("local model artifact must be a regular file or directory")
    entries: list[dict[str, Any]] = []
    for candidate in sorted(root.rglob("*"), key=lambda item: item.as_posix()):
        try:
            info = os.lstat(candidate)
        except OSError as exc:
            raise EnsembleError(f"cannot inspect model artifact entry: {candidate}: {exc}") from exc
        if stat.S_ISLNK(info.st_mode) or _is_reparse(info):
            raise EnsembleError("model artifact directory may not contain links or reparse points")
        if stat.S_ISDIR(info.st_mode):
            continue
        if not stat.S_ISREG(info.st_mode):
            raise EnsembleError("model artifact directory contains a non-regular entry")
        relative = candidate.relative_to(root).as_posix()
        entries.append(
            {"path": relative, "bytes": info.st_size, "sha256": _sha256_file(candidate)}
        )
    if not entries or sum(int(item["bytes"]) for item in entries) <= 0:
        raise EnsembleError("local model artifact directory has no non-empty file payload")
    payload = _canonical_bytes(entries)
    return {
        "kind": "directory",
        "sha256": _sha256_bytes(payload),
        "file_count": len(entries),
        "byte_count": sum(int(item["bytes"]) for item in entries),
        "entries_sha256": _sha256_bytes(payload),
    }


def _mean(values: Sequence[float]) -> float:
    if not values:
        raise EnsembleError("cannot aggregate an empty score group")
    return math.fsum(values) / len(values)


def _sigmoid(value: float) -> float:
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))
    exp_value = math.exp(value)
    return exp_value / (1.0 + exp_value)


def _softmax(values: Sequence[float]) -> list[float]:
    if not values:
        raise EnsembleError("cannot normalize an empty score vector")
    maximum = max(values)
    exponentials = [math.exp(value - maximum) for value in values]
    denominator = math.fsum(exponentials)
    if not math.isfinite(denominator) or denominator <= 0:
        raise EnsembleError("non-finite softmax denominator")
    return [value / denominator for value in exponentials]


def _round_float(value: float) -> float:
    if not math.isfinite(value):
        raise EnsembleError("attempted to serialize a non-finite score")
    return round(float(value), 12)


def _round_probabilities(keys: Sequence[str], values: Sequence[float]) -> dict[str, float]:
    rounded = [_round_float(value) for value in values]
    rounded[0] += 1.0 - math.fsum(rounded)
    return {key: rounded[index] for index, key in enumerate(keys)}


@dataclass(frozen=True)
class EnsembleConfig:
    queue_path: Path = DEFAULT_QUEUE
    manifest_path: Path = DEFAULT_MANIFEST
    queue_summary_path: Path = DEFAULT_QUEUE_SUMMARY
    integrity_audit_path: Path = DEFAULT_INTEGRITY_AUDIT
    class_contract_path: Path = DEFAULT_CLASS_CONTRACT
    policy_path: Path = DEFAULT_POLICY
    output_dir: Path = DEFAULT_OUTPUT_DIR
    model_path: Path | None = None
    tokenizer_path: Path | None = None
    model_id: str = ""
    backend: str = "local_siglip2"
    fixture_mode: bool = False
    device: str = "auto"
    batch_size: int = 8


class LocalSigLIP2Scorer:
    """Network-disabled Transformers scorer for a self-contained local model."""

    def __init__(self, model_path: Path, device: str, batch_size: int) -> None:
        self.model_path = model_path
        self.device = device
        self.batch_size = batch_size

    def score(
        self,
        images: Sequence[Path],
        prompts: Sequence[Mapping[str, str]],
    ) -> Sequence[Sequence[float]]:
        previous = {
            name: os.environ.get(name)
            for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE")
        }
        os.environ.update(
            {
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
                "HF_DATASETS_OFFLINE": "1",
            }
        )
        try:
            try:
                import torch
                from transformers import AutoModel, AutoProcessor
            except ImportError as exc:
                raise EnsembleError("local SigLIP2 inference requires torch and transformers") from exc

            if self.device == "auto":
                selected_device = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                selected_device = self.device
            if selected_device == "cuda" and not torch.cuda.is_available():
                raise EnsembleError("CUDA was requested but is unavailable")
            torch.manual_seed(0)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(0)
            try:
                processor = AutoProcessor.from_pretrained(
                    str(self.model_path),
                    local_files_only=True,
                    trust_remote_code=False,
                )
                model = AutoModel.from_pretrained(
                    str(self.model_path),
                    local_files_only=True,
                    trust_remote_code=False,
                    use_safetensors=True,
                )
                model.to(selected_device)
                model.eval()
            except Exception as exc:
                raise EnsembleError(
                    f"cannot load the local SigLIP2 model: {type(exc).__name__}: {exc}"
                ) from exc

            texts = [item["text"] for item in prompts]
            matrix: list[list[float]] = []
            for start in range(0, len(images), self.batch_size):
                batch_paths = images[start : start + self.batch_size]
                opened: list[Image.Image] = []
                try:
                    for path in batch_paths:
                        with Image.open(path) as source:
                            source.load()
                            opened.append(source.convert("RGB"))
                    inputs = processor(text=texts, images=opened, return_tensors="pt", padding=True)
                    inputs = {
                        key: value.to(selected_device) if hasattr(value, "to") else value
                        for key, value in inputs.items()
                    }
                    with torch.inference_mode():
                        output = model(**inputs)
                    logits = getattr(output, "logits_per_image", None)
                    expected_shape = (len(batch_paths), len(prompts))
                    if logits is None or tuple(logits.shape) != expected_shape:
                        raise EnsembleError(
                            f"SigLIP2 returned the wrong logit shape; expected={expected_shape}, "
                            f"actual={None if logits is None else tuple(logits.shape)}"
                        )
                    matrix.extend(logits.float().detach().cpu().tolist())
                except EnsembleError:
                    raise
                except Exception as exc:
                    raise EnsembleError(
                        f"local SigLIP2 inference failed: {type(exc).__name__}: {exc}"
                    ) from exc
                finally:
                    for image in opened:
                        image.close()
            return matrix
        finally:
            for name, value in previous.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value


class LocalOpenCLIPBigVisionScorer:
    """Offline OpenCLIP scorer for Google's official SigLIP2 Big Vision NPZ."""

    def __init__(
        self,
        model_path: Path,
        tokenizer_path: Path,
        model_name: str,
        context_length: int,
        device: str,
        batch_size: int,
    ) -> None:
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path
        self.model_name = model_name
        self.context_length = context_length
        self.device = device
        self.batch_size = batch_size

    def score(
        self,
        images: Sequence[Path],
        prompts: Sequence[Mapping[str, str]],
    ) -> Sequence[Sequence[float]]:
        previous = {
            name: os.environ.get(name)
            for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE")
        }
        os.environ.update(
            {
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
                "HF_DATASETS_OFFLINE": "1",
            }
        )
        try:
            try:
                import torch
                import open_clip
                from open_clip.tokenizer import HFTokenizer
            except ImportError as exc:
                raise EnsembleError(
                    "OpenCLIP Big Vision inference requires torch, transformers and open_clip_torch"
                ) from exc

            if self.device == "auto":
                selected_device = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                selected_device = self.device
            if selected_device == "cuda" and not torch.cuda.is_available():
                raise EnsembleError("CUDA was requested but is unavailable")
            torch.manual_seed(0)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(0)
            try:
                model, _, preprocess = open_clip.create_model_and_transforms(
                    self.model_name,
                    pretrained=str(self.model_path),
                    precision="fp32",
                    device=selected_device,
                    cache_dir=None,
                    output_dict=False,
                )
                tokenizer = HFTokenizer(
                    str(self.tokenizer_path),
                    context_length=self.context_length,
                    clean="canonicalize",
                    local_files_only=True,
                    trust_remote_code=False,
                )
                model.eval()
            except Exception as exc:
                raise EnsembleError(
                    f"cannot load the local OpenCLIP Big Vision SigLIP2 model: "
                    f"{type(exc).__name__}: {exc}"
                ) from exc

            texts = [item["text"] for item in prompts]
            try:
                text_tokens = tokenizer(texts).to(selected_device)
                with torch.inference_mode():
                    text_features = model.encode_text(text_tokens, normalize=True)
                    logit_scale = model.logit_scale.exp()
                    logit_bias = model.logit_bias if model.logit_bias is not None else 0.0
            except Exception as exc:
                raise EnsembleError(
                    f"OpenCLIP SigLIP2 text encoding failed: {type(exc).__name__}: {exc}"
                ) from exc

            matrix: list[list[float]] = []
            for start in range(0, len(images), self.batch_size):
                batch_paths = images[start : start + self.batch_size]
                try:
                    tensors = []
                    for path in batch_paths:
                        with Image.open(path) as source:
                            source.load()
                            tensors.append(preprocess(source.convert("RGB")))
                    pixel_values = torch.stack(tensors, dim=0).to(selected_device)
                    with torch.inference_mode():
                        image_features = model.encode_image(pixel_values, normalize=True)
                        logits = logit_scale * image_features @ text_features.T + logit_bias
                    expected_shape = (len(batch_paths), len(prompts))
                    if tuple(logits.shape) != expected_shape:
                        raise EnsembleError(
                            f"OpenCLIP SigLIP2 returned the wrong logit shape; "
                            f"expected={expected_shape}, actual={tuple(logits.shape)}"
                        )
                    matrix.extend(logits.float().detach().cpu().tolist())
                except EnsembleError:
                    raise
                except Exception as exc:
                    raise EnsembleError(
                        f"OpenCLIP SigLIP2 image inference failed: "
                        f"{type(exc).__name__}: {exc}"
                    ) from exc
            return matrix
        finally:
            for name, value in previous.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value


class AISigLIP2Ensemble:
    """Bind frozen inputs, score every prompt, and render immutable AI suggestions."""

    def __init__(self, config: EnsembleConfig, scorer: PromptScorer | None = None) -> None:
        self.config = config
        self.queue_path = Path(config.queue_path)
        self.manifest_path = Path(config.manifest_path)
        self.queue_summary_path = Path(config.queue_summary_path)
        self.integrity_audit_path = Path(config.integrity_audit_path)
        self.class_contract_path = Path(config.class_contract_path)
        self.policy_path = Path(config.policy_path)
        self.output_dir = Path(config.output_dir)
        self.model_path = Path(config.model_path) if config.model_path is not None else None
        self.tokenizer_path = (
            Path(config.tokenizer_path) if config.tokenizer_path is not None else None
        )
        self.scorer = scorer
        self._validate_scope()
        policy, policy_raw = _read_json(self.policy_path, "ensemble policy")
        self.policy = self._validate_policy(policy)
        self.policy_sha256 = _sha256_bytes(policy_raw)
        self.implementation_sha256 = _sha256_file(SCRIPT_PATH)
        if self.model_path is None:
            raise EnsembleError("--model-path is required and must already exist locally")
        if not MODEL_ID.fullmatch(config.model_id):
            raise EnsembleError("model_id does not match the strict stable identifier pattern")
        if config.device not in {"auto", "cpu", "cuda"}:
            raise EnsembleError("device must be auto, cpu, or cuda")
        if type(config.batch_size) is not int or not 1 <= config.batch_size <= 128:
            raise EnsembleError("batch_size must be an integer in [1,128]")
        if config.backend not in {
            "local_siglip2",
            "local_openclip_bigvision_npz",
            "fixture_fake",
        }:
            raise EnsembleError("unsupported scoring backend")
        if config.backend == "fixture_fake":
            if not config.fixture_mode or scorer is None:
                raise EnsembleError("fixture_fake requires explicit fixture mode and an injected scorer")
        elif scorer is not None:
            raise EnsembleError("an injected scorer is allowed only for fixture_fake")
        self.model_artifact = _artifact_root(self.model_path)
        self.tokenizer_artifact: dict[str, Any] | None = None
        if config.backend == "local_siglip2":
            self._validate_local_model()
            if self.tokenizer_path is not None:
                raise EnsembleError("local_siglip2 binds its tokenizer inside the model directory")
        elif config.backend == "local_openclip_bigvision_npz":
            self._validate_openclip_bigvision_inputs()
        self.prompt_spec = {
            "whole_quality_prompts": self.policy["inference"]["whole_quality_prompts"],
            "class_prompts": self.policy["inference"]["class_prompts"],
            "reject_family_prompts": self.policy["inference"]["reject_family_prompts"],
        }
        self.prompt_set_sha256 = _sha256_bytes(_canonical_bytes(self.prompt_spec))
        self.calibration = self.policy["inference"]["calibration"]
        self.calibration_sha256 = _sha256_bytes(_canonical_bytes(self.calibration))
        self.prompt_records = self._flatten_prompts()
        self.prompt_index = {
            item["id"]: index for index, item in enumerate(self.prompt_records)
        }
        self.image_records: list[dict[str, Any]] = []
        self.input_roots: dict[str, str] = {}
        self.image_payload_set_sha256 = ""

    def _validate_scope(self) -> None:
        values = [
            (self.queue_path, DEFAULT_QUEUE),
            (self.manifest_path, DEFAULT_MANIFEST),
            (self.queue_summary_path, DEFAULT_QUEUE_SUMMARY),
            (self.integrity_audit_path, DEFAULT_INTEGRITY_AUDIT),
            (self.class_contract_path, DEFAULT_CLASS_CONTRACT),
            (self.policy_path, DEFAULT_POLICY),
            (self.output_dir, DEFAULT_OUTPUT_DIR),
        ]
        self.production_mode = all(
            path.resolve(strict=False) == default.resolve(strict=False)
            for path, default in values
        )
        if self.production_mode and self.config.fixture_mode:
            raise EnsembleError("fixture mode may not be used with all production paths")
        if not self.production_mode and not self.config.fixture_mode:
            raise EnsembleError("custom inputs require explicit --fixture-mode")
        expected_output = self.queue_path.parent / "ai_ensemble_v1"
        if self.output_dir.resolve(strict=False) != expected_output.resolve(strict=False):
            raise EnsembleError("output must be exactly review/ai_ensemble_v1 beside the bound queue")
        if not self.production_mode and self.output_dir.resolve(strict=False) == DEFAULT_OUTPUT_DIR.resolve(
            strict=False
        ):
            raise EnsembleError("a fixture may not write the production AI ensemble directory")
        if "human_decisions" in {part.lower() for part in self.output_dir.parts}:
            raise EnsembleError("AI ensemble output may never be nested under human_decisions")
        if self.output_dir.resolve(strict=False) in {
            self.queue_path.resolve(strict=False),
            self.manifest_path.resolve(strict=False),
        }:
            raise EnsembleError("output path overlaps a frozen input")

    def _validate_policy(self, value: Any) -> dict[str, Any]:
        policy = _exact_keys(value, POLICY_FIELDS, "ensemble policy")
        if policy["schema_version"] != "rootscope.ai_siglip2_ensemble_policy.v1":
            raise EnsembleError("unexpected ensemble policy schema_version")
        if policy["queue_schema_version"] != "rootscope.wikimedia_human_review_queue.v1":
            raise EnsembleError("policy queue schema is not frozen")
        if policy["manifest_schema_version"] != "rootscope.wikimedia_candidate.v1":
            raise EnsembleError("policy manifest schema is not frozen")
        if policy["target_classes"] != TARGET_CLASSES or policy["all_class_hints"] != ALL_HINTS:
            raise EnsembleError("policy class order is not the frozen morphology order")
        roots = policy["production_input_roots"]
        expected_root_keys = {
            "candidate_review_queue_sha256",
            "staging_manifest_sha256",
            "review_queue_summary_sha256",
            "integrity_audit_sha256",
            "class_contract_sha256",
        }
        _exact_keys(roots, expected_root_keys, "policy input roots")
        if any(type(item) is not str or not HEX64.fullmatch(item) for item in roots.values()):
            raise EnsembleError("policy contains an invalid input SHA-256")
        if type(policy["expected_candidate_count"]) is not int or policy["expected_candidate_count"] <= 0:
            raise EnsembleError("policy expected_candidate_count is invalid")
        hint_counts = policy["expected_acquisition_hint_counts"]
        if not isinstance(hint_counts, dict) or set(hint_counts) != set(ALL_HINTS):
            raise EnsembleError("policy acquisition hint counts do not cover the frozen classes")
        if any(type(item) is not int or item < 0 for item in hint_counts.values()):
            raise EnsembleError("policy acquisition hint counts are invalid")
        image_contract = _exact_keys(
            policy["image_contract"],
            {"allowed_mime", "minimum_width", "minimum_height", "verify_decode"},
            "policy image contract",
        )
        if image_contract["allowed_mime"] != ["image/jpeg", "image/png"]:
            raise EnsembleError("policy image MIME allowlist is not frozen")
        if any(
            type(image_contract[key]) is not int or image_contract[key] <= 0
            for key in ("minimum_width", "minimum_height")
        ) or image_contract["verify_decode"] is not True:
            raise EnsembleError("policy image contract is invalid")
        inference = _exact_keys(
            policy["inference"],
            {
                "supported_backends",
                "model_type_allowlist",
                "openclip_model_name",
                "openclip_context_length",
                "normalization",
                "whole_quality_prompts",
                "class_prompts",
                "reject_family_prompts",
                "calibration",
            },
            "policy inference",
        )
        if inference["supported_backends"] != [
            "local_siglip2",
            "local_openclip_bigvision_npz",
        ]:
            raise EnsembleError("policy local inference backend allowlist is not frozen")
        if inference["model_type_allowlist"] != ["siglip", "siglip2"]:
            raise EnsembleError("policy model type allowlist is not frozen")
        if inference["openclip_model_name"] != "ViT-B-16-SigLIP2":
            raise EnsembleError("policy OpenCLIP architecture is not frozen")
        if inference["openclip_context_length"] != 64:
            raise EnsembleError("policy OpenCLIP context length is not frozen")
        if inference["normalization"] != "two_stage_calibrated_prompt_logits":
            raise EnsembleError("policy normalization contract is not frozen")
        if not isinstance(inference["class_prompts"], dict) or set(
            inference["class_prompts"]
        ) != set(TARGET_CLASSES):
            raise EnsembleError("policy class prompt groups do not cover the frozen classes")
        if not isinstance(inference["reject_family_prompts"], dict) or set(
            inference["reject_family_prompts"]
        ) != set(REJECT_FAMILIES):
            raise EnsembleError("policy reject prompt families do not cover the frozen families")
        prompt_groups = [inference["whole_quality_prompts"]]
        prompt_groups.extend(inference["class_prompts"].values())
        prompt_groups.extend(inference["reject_family_prompts"].values())
        prompt_ids: list[str] = []
        for group in prompt_groups:
            if not isinstance(group, list) or not group:
                raise EnsembleError("every prompt group must be a non-empty list")
            for prompt in group:
                _exact_keys(prompt, {"id", "text"}, "prompt record")
                if type(prompt["id"]) is not str or not PROMPT_ID.fullmatch(prompt["id"]):
                    raise EnsembleError("prompt id is invalid")
                if (
                    type(prompt["text"]) is not str
                    or not prompt["text"].strip()
                    or len(prompt["text"]) > 512
                ):
                    raise EnsembleError("prompt text is invalid")
                prompt_ids.append(prompt["id"])
        if len(prompt_ids) != len(set(prompt_ids)):
            raise EnsembleError("prompt ids must be globally unique")
        calibration = _exact_keys(
            inference["calibration"],
            {
                "status",
                "admissibility_bias",
                "admissibility_temperature",
                "morphology_temperature",
                "morphology_bias_by_class",
            },
            "policy calibration",
        )
        if type(calibration["status"]) is not str or not calibration["status"]:
            raise EnsembleError("calibration status is invalid")
        for key in ("admissibility_bias", "admissibility_temperature", "morphology_temperature"):
            if not _is_number(calibration[key]):
                raise EnsembleError("calibration contains a non-finite number")
        if calibration["admissibility_temperature"] <= 0 or calibration["morphology_temperature"] <= 0:
            raise EnsembleError("calibration temperatures must be positive")
        biases = calibration["morphology_bias_by_class"]
        if not isinstance(biases, dict) or set(biases) != set(TARGET_CLASSES):
            raise EnsembleError("morphology biases are not bound to all frozen classes")
        if any(not _is_number(item) for item in biases.values()):
            raise EnsembleError("morphology biases contain a non-finite number")
        thresholds = _exact_keys(
            policy["thresholds"],
            {
                "auto_target_min_admissible",
                "auto_target_min_probability_by_class",
                "auto_target_min_margin",
                "require_acquisition_hint_agreement",
                "auto_unknown_max_admissible",
                "auto_unknown_min_reject_probability",
                "auto_unknown_max_target_probability",
                "probability_sum_tolerance",
            },
            "policy thresholds",
        )
        per_class = thresholds["auto_target_min_probability_by_class"]
        if not isinstance(per_class, dict) or set(per_class) != set(TARGET_CLASSES):
            raise EnsembleError("target probability thresholds do not cover all target classes")
        numbers = [
            thresholds["auto_target_min_admissible"],
            *per_class.values(),
            thresholds["auto_target_min_margin"],
            thresholds["auto_unknown_max_admissible"],
            thresholds["auto_unknown_min_reject_probability"],
            thresholds["auto_unknown_max_target_probability"],
            thresholds["probability_sum_tolerance"],
        ]
        if any(not _is_number(item) or not 0 <= float(item) <= 1 for item in numbers):
            raise EnsembleError("policy thresholds must be finite numbers in [0,1]")
        if thresholds["require_acquisition_hint_agreement"] is not True:
            raise EnsembleError("conservative policy must require acquisition hint agreement")
        if policy["authority"] != AUTHORITY:
            raise EnsembleError("policy attempts to grant forbidden authority")
        nonclaims = policy["explicit_non_claims"]
        if not isinstance(nonclaims, list) or not nonclaims or any(type(item) is not str for item in nonclaims):
            raise EnsembleError("policy explicit_non_claims is invalid")
        output = _exact_keys(
            policy["output_contract"],
            {"results_filename", "hold_filename", "stats_filename", "receipt_filename", "status"},
            "policy output contract",
        )
        names = [output[key] for key in ("results_filename", "hold_filename", "stats_filename", "receipt_filename")]
        if any(type(name) is not str or Path(name).name != name for name in names):
            raise EnsembleError("every output filename must be a basename")
        if len(names) != len(set(names)):
            raise EnsembleError("output filenames must be unique")
        return policy

    def _validate_local_model(self) -> None:
        assert self.model_path is not None
        if not self.model_path.is_dir():
            raise EnsembleError("local_siglip2 requires a self-contained model directory")
        config_path = self.model_path / "config.json"
        config, _ = _read_json(config_path, "local model config.json")
        if not isinstance(config, dict) or config.get("model_type") not in self.policy["inference"][
            "model_type_allowlist"
        ]:
            raise EnsembleError("local model config is not SigLIP/SigLIP2")
        weights = [path for path in self.model_path.rglob("*.safetensors") if path.is_file()]
        if not weights or all(path.stat().st_size <= 0 for path in weights):
            raise EnsembleError("local SigLIP2 model has no non-empty safetensors weights")

    def _validate_openclip_bigvision_inputs(self) -> None:
        assert self.model_path is not None
        if not self.model_path.is_file() or self.model_path.suffix.lower() != ".npz":
            raise EnsembleError("OpenCLIP Big Vision backend requires one complete local .npz file")
        if self.model_path.name.endswith(".part") or self.model_path.stat().st_size <= 0:
            raise EnsembleError("OpenCLIP Big Vision NPZ is incomplete or empty")
        if self.tokenizer_path is None or not self.tokenizer_path.is_dir():
            raise EnsembleError("OpenCLIP Big Vision backend requires a local tokenizer directory")
        required = ["tokenizer_config.json", "special_tokens_map.json"]
        for name in required:
            path = self.tokenizer_path / name
            if not path.is_file() or path.stat().st_size <= 0:
                raise EnsembleError(f"local tokenizer directory is missing {name}")
        payloads = [self.tokenizer_path / "tokenizer.json", self.tokenizer_path / "tokenizer.model"]
        if not any(path.is_file() and path.stat().st_size > 0 for path in payloads):
            raise EnsembleError("local tokenizer directory has no tokenizer.json/tokenizer.model payload")
        tokenizer_config, _ = _read_json(
            self.tokenizer_path / "tokenizer_config.json", "local tokenizer config"
        )
        if not isinstance(tokenizer_config, dict) or tokenizer_config.get("tokenizer_class") not in {
            "GemmaTokenizer",
            "GemmaTokenizerFast",
        }:
            raise EnsembleError("local tokenizer is not the expected SigLIP2 Gemma tokenizer")
        self.tokenizer_artifact = _artifact_root(self.tokenizer_path)

    def _flatten_prompts(self) -> list[dict[str, str]]:
        inference = self.policy["inference"]
        flattened: list[dict[str, str]] = []
        for prompt in inference["whole_quality_prompts"]:
            flattened.append({"id": prompt["id"], "text": prompt["text"]})
        for class_id in TARGET_CLASSES:
            for prompt in inference["class_prompts"][class_id]:
                flattened.append({"id": prompt["id"], "text": prompt["text"]})
        for family in REJECT_FAMILIES:
            for prompt in inference["reject_family_prompts"][family]:
                flattened.append({"id": prompt["id"], "text": prompt["text"]})
        return flattened

    def _assert_model_unchanged(self) -> None:
        assert self.model_path is not None
        if _artifact_root(self.model_path) != self.model_artifact:
            raise EnsembleError("local model artifact changed after its initial SHA-256 binding")
        if self.tokenizer_artifact is not None:
            assert self.tokenizer_path is not None
            if _artifact_root(self.tokenizer_path) != self.tokenizer_artifact:
                raise EnsembleError("local tokenizer artifact changed after its initial SHA-256 binding")

    def _current_input_roots(self) -> dict[str, str]:
        return {
            "candidate_review_queue_sha256": _sha256_file(self.queue_path),
            "staging_manifest_sha256": _sha256_file(self.manifest_path),
            "review_queue_summary_sha256": _sha256_file(self.queue_summary_path),
            "integrity_audit_sha256": _sha256_file(self.integrity_audit_path),
            "class_contract_sha256": _sha256_file(self.class_contract_path),
        }

    def _load_inputs(self) -> None:
        self.input_roots = self._current_input_roots()
        if self.input_roots != self.policy["production_input_roots"]:
            raise EnsembleError("actual input roots do not match the policy-bound roots")
        summary, _ = _read_json(self.queue_summary_path, "review queue summary")
        integrity, _ = _read_json(self.integrity_audit_path, "staging integrity audit")
        contract, _ = _read_json(self.class_contract_path, "class contract")
        if not isinstance(summary, dict) or summary.get("candidate_count") != self.policy[
            "expected_candidate_count"
        ]:
            raise EnsembleError("review queue summary candidate_count is not policy-bound")
        if summary.get("inputs", {}).get("staging_manifest_sha256") != self.input_roots[
            "staging_manifest_sha256"
        ]:
            raise EnsembleError("review queue summary does not bind the staging manifest")
        if summary.get("outputs", {}).get("candidate_review_queue.jsonl") != self.input_roots[
            "candidate_review_queue_sha256"
        ]:
            raise EnsembleError("review queue summary does not bind the candidate queue")
        if (
            not isinstance(integrity, dict)
            or integrity.get("result") != "PASS_STAGING_INTEGRITY_NOT_TRAIN_READY"
            or integrity.get("failure_count") != 0
            or integrity.get("failures") != []
            or integrity.get("manifest_sha256") != self.input_roots["staging_manifest_sha256"]
        ):
            raise EnsembleError("staging integrity audit is not a bound zero-failure PASS")
        if not isinstance(contract, dict) or contract.get("class_order") != ALL_HINTS:
            raise EnsembleError("class contract does not bind the frozen class order")

        manifest_rows, _ = _read_jsonl(self.manifest_path, "staging manifest")
        queue_rows, _ = _read_jsonl(self.queue_path, "candidate review queue")
        expected_count = self.policy["expected_candidate_count"]
        if len(manifest_rows) != expected_count or len(queue_rows) != expected_count:
            raise EnsembleError("manifest/queue count does not match the policy")
        manifest_by_pageid: dict[int, dict[str, Any]] = {}
        for index, row in enumerate(manifest_rows, start=1):
            _required_keys(row, MANIFEST_REQUIRED_FIELDS, f"manifest line {index}")
            if row.get("schema_version") != self.policy["manifest_schema_version"]:
                raise EnsembleError(f"manifest line {index} has an unexpected schema")
            pageid = row.get("pageid")
            if type(pageid) is not int or pageid <= 0 or pageid in manifest_by_pageid:
                raise EnsembleError(f"manifest line {index} has an invalid/duplicate pageid")
            manifest_by_pageid[pageid] = row

        seen_assets: set[str] = set()
        seen_sha: set[str] = set()
        hint_counts: Counter[str] = Counter()
        images: list[dict[str, Any]] = []
        image_binding: list[dict[str, str]] = []
        for index, row in enumerate(queue_rows, start=1):
            _exact_keys(row, QUEUE_FIELDS, f"candidate queue line {index}")
            if row.get("schema_version") != self.policy["queue_schema_version"]:
                raise EnsembleError(f"candidate queue line {index} has an unexpected schema")
            pageid = row.get("pageid")
            sha = row.get("sha256")
            asset = row.get("asset")
            hint = row.get("class_hint")
            if type(pageid) is not int or pageid <= 0 or pageid not in manifest_by_pageid:
                raise EnsembleError(f"candidate queue line {index} has an invalid pageid")
            if type(sha) is not str or not HEX64.fullmatch(sha) or sha in seen_sha:
                raise EnsembleError(f"candidate queue line {index} has an invalid/duplicate SHA-256")
            if asset != f"wikimedia:{pageid}@sha256:{sha}" or asset in seen_assets:
                raise EnsembleError(f"candidate queue line {index} has an invalid/duplicate asset binding")
            if hint not in ALL_HINTS:
                raise EnsembleError(f"candidate queue line {index} has an invalid acquisition hint")
            if (
                row.get("review_status") != "UNREVIEWED"
                or row.get("training_eligible") is not False
                or row.get("print_eligible") is not False
                or row.get("split") != "UNASSIGNED_DO_NOT_TRAIN"
                or row.get("target_class") != ""
                or row.get("visual_decision") != ""
                or row.get("rights_decision") != ""
                or row.get("reviewer") != ""
            ):
                raise EnsembleError(f"candidate queue line {index} is no longer pristine/unreviewed")
            manifest = manifest_by_pageid[pageid]
            relationships = {
                "class_id": hint,
                "filename": row.get("local_path"),
                "download_sha256": sha,
                "download_width": row.get("download_width"),
                "download_height": row.get("download_height"),
                "download_mime": row.get("download_mime"),
                "source_group": row.get("source_group"),
                "source_page": row.get("source_url"),
            }
            for field, expected in relationships.items():
                if manifest.get(field) != expected:
                    raise EnsembleError(
                        f"candidate queue line {index} disagrees with manifest field {field}"
                    )
            absolute, relative = _safe_relative_path(
                self.queue_path.parent.parent,
                row.get("local_path"),
                f"candidate queue line {index} local_path",
            )
            if _sha256_file(absolute) != sha:
                raise EnsembleError(f"candidate image bytes changed: {relative}")
            mime = row.get("download_mime")
            width = row.get("download_width")
            height = row.get("download_height")
            image_contract = self.policy["image_contract"]
            if mime not in image_contract["allowed_mime"]:
                raise EnsembleError(f"candidate image MIME is not allowed: {relative}")
            if (
                type(width) is not int
                or type(height) is not int
                or width < image_contract["minimum_width"]
                or height < image_contract["minimum_height"]
            ):
                raise EnsembleError(f"candidate image dimensions are below policy minimum: {relative}")
            try:
                with Image.open(absolute) as image:
                    image.load()
                    if image.size != (width, height):
                        raise EnsembleError(f"decoded image dimensions changed: {relative}")
            except (OSError, UnidentifiedImageError) as exc:
                raise EnsembleError(f"candidate image cannot be decoded: {relative}: {exc}") from exc
            seen_assets.add(asset)
            seen_sha.add(sha)
            hint_counts[hint] += 1
            images.append(
                {
                    "asset": asset,
                    "pageid": pageid,
                    "candidate_sha256": sha,
                    "local_path": relative,
                    "absolute_path": absolute,
                    "class_hint": hint,
                }
            )
            image_binding.append({"asset": asset, "sha256": sha, "local_path": relative})
        actual_hint_counts = {class_id: hint_counts[class_id] for class_id in ALL_HINTS}
        if actual_hint_counts != self.policy["expected_acquisition_hint_counts"]:
            raise EnsembleError("candidate acquisition hint counts do not match policy")
        self.image_records = images
        self.image_payload_set_sha256 = _sha256_bytes(_canonical_bytes(image_binding))

    def _assert_images_unchanged(self) -> None:
        binding: list[dict[str, str]] = []
        for row in self.image_records:
            actual = _sha256_file(row["absolute_path"])
            if actual != row["candidate_sha256"]:
                raise EnsembleError(f"candidate image changed during inference: {row['local_path']}")
            binding.append(
                {
                    "asset": row["asset"],
                    "sha256": row["candidate_sha256"],
                    "local_path": row["local_path"],
                }
            )
        if _sha256_bytes(_canonical_bytes(binding)) != self.image_payload_set_sha256:
            raise EnsembleError("candidate image payload set changed during inference")

    def preflight(self) -> dict[str, Any]:
        self._assert_model_unchanged()
        self._load_inputs()
        if self.config.backend == "local_siglip2":
            try:
                import torch  # noqa: F401
                import transformers  # noqa: F401
            except ImportError as exc:
                raise EnsembleError("local SigLIP2 preflight requires torch and transformers") from exc
        elif self.config.backend == "local_openclip_bigvision_npz":
            try:
                import torch  # noqa: F401
                import transformers  # noqa: F401
                import open_clip  # noqa: F401
            except ImportError as exc:
                raise EnsembleError(
                    "OpenCLIP Big Vision preflight requires torch, transformers and open_clip_torch"
                ) from exc
        mode = "PRODUCTION" if self.production_mode else "FIXTURE"
        return {
            "schema_version": "rootscope.ai_siglip2_ensemble_preflight.v1",
            "mode": mode,
            "status": f"{mode}_AI_SIGLIP2_ENSEMBLE_PREFLIGHT_PASS_NOT_HUMAN_REVIEWED",
            "candidate_count": len(self.image_records),
            "input_roots": self.input_roots,
            "image_payload_set_sha256": self.image_payload_set_sha256,
            "model_id": self.config.model_id,
            "model_artifact_sha256": self.model_artifact["sha256"],
            "prompt_set_sha256": self.prompt_set_sha256,
            "calibration_sha256": self.calibration_sha256,
            "prompt_count": len(self.prompt_records),
            "output_dir": self.output_dir.name,
            "output_exists": self.output_dir.exists(),
            "writes_performed": False,
            "authority": AUTHORITY,
            "explicit_non_claims": self.policy["explicit_non_claims"],
        }

    def _score_matrix(self) -> tuple[list[list[float]], str]:
        if self.config.backend == "fixture_fake":
            assert self.scorer is not None
            scorer = self.scorer
            provenance = "FIXTURE_INJECTED_SCORER_NOT_MODEL_EXECUTION_PROOF"
        elif self.config.backend == "local_siglip2":
            assert self.model_path is not None
            scorer = LocalSigLIP2Scorer(
                self.model_path,
                self.config.device,
                self.config.batch_size,
            )
            provenance = "IN_PROCESS_LOCAL_FILES_ONLY_TRANSFORMERS_SIGLIP2"
        else:
            assert self.model_path is not None
            assert self.tokenizer_path is not None
            inference = self.policy["inference"]
            scorer = LocalOpenCLIPBigVisionScorer(
                self.model_path,
                self.tokenizer_path,
                inference["openclip_model_name"],
                int(inference["openclip_context_length"]),
                self.config.device,
                self.config.batch_size,
            )
            provenance = "IN_PROCESS_LOCAL_FILES_ONLY_OPENCLIP_GOOGLE_BIG_VISION_SIGLIP2"
        try:
            raw = scorer.score(
                [row["absolute_path"] for row in self.image_records],
                self.prompt_records,
            )
            matrix = [list(row) for row in raw]
        except EnsembleError:
            raise
        except Exception as exc:
            raise EnsembleError(f"prompt scorer failed: {type(exc).__name__}: {exc}") from exc
        if len(matrix) != len(self.image_records):
            raise EnsembleError("prompt scorer returned the wrong image count")
        expected_columns = len(self.prompt_records)
        normalized: list[list[float]] = []
        for index, row in enumerate(matrix, start=1):
            if len(row) != expected_columns:
                raise EnsembleError(f"prompt scorer row {index} has the wrong prompt count")
            values: list[float] = []
            for value in row:
                if not _is_number(value):
                    raise EnsembleError(f"prompt scorer row {index} contains a non-finite score")
                values.append(float(value))
            normalized.append(values)
        return normalized, provenance

    def _group_scores(self, scores: Mapping[str, float]) -> tuple[float, dict[str, float], dict[str, float]]:
        inference = self.policy["inference"]
        quality = _mean([scores[item["id"]] for item in inference["whole_quality_prompts"]])
        classes = {
            class_id: _mean(
                [scores[item["id"]] for item in inference["class_prompts"][class_id]]
            )
            for class_id in TARGET_CLASSES
        }
        rejects = {
            family: _mean(
                [scores[item["id"]] for item in inference["reject_family_prompts"][family]]
            )
            for family in REJECT_FAMILIES
        }
        return quality, classes, rejects

    def _render_result(self, image: Mapping[str, Any], values: Sequence[float]) -> dict[str, Any]:
        prompt_scores_raw = {
            prompt["id"]: float(values[index]) for index, prompt in enumerate(self.prompt_records)
        }
        quality_raw, class_raw, reject_raw = self._group_scores(prompt_scores_raw)
        dominant_reject = max(REJECT_FAMILIES, key=lambda family: reject_raw[family])
        dominant_reject_raw = reject_raw[dominant_reject]
        calibration = self.calibration
        admission_logit = (
            quality_raw - dominant_reject_raw - float(calibration["admissibility_bias"])
        ) / float(calibration["admissibility_temperature"])
        admissible_raw = _sigmoid(admission_logit)
        reject_probability_raw = 1.0 - admissible_raw
        reject_family_probabilities_raw = {
            family: _sigmoid(
                (
                    reject_raw[family]
                    - quality_raw
                    + float(calibration["admissibility_bias"])
                )
                / float(calibration["admissibility_temperature"])
            )
            for family in REJECT_FAMILIES
        }
        morphology_logits = [
            (
                class_raw[class_id]
                - float(calibration["morphology_bias_by_class"][class_id])
            )
            / float(calibration["morphology_temperature"])
            for class_id in TARGET_CLASSES
        ]
        morphology_probabilities = _softmax(morphology_logits)
        rank = sorted(
            range(len(TARGET_CLASSES)),
            key=lambda index: (-morphology_probabilities[index], index),
        )
        top1_index, top2_index = rank[:2]
        top1 = TARGET_CLASSES[top1_index]
        top2 = TARGET_CLASSES[top2_index]
        top1_probability_raw = morphology_probabilities[top1_index]
        top2_probability_raw = morphology_probabilities[top2_index]
        margin_raw = top1_probability_raw - top2_probability_raw
        hint_agrees = image["class_hint"] == top1
        thresholds = self.policy["thresholds"]

        unknown_gates = {
            "admissibility": admissible_raw
            <= float(thresholds["auto_unknown_max_admissible"]),
            "reject_probability": reject_probability_raw
            >= float(thresholds["auto_unknown_min_reject_probability"]),
            "target_ceiling": top1_probability_raw
            <= float(thresholds["auto_unknown_max_target_probability"]),
        }
        if all(unknown_gates.values()):
            decision = "AUTO_UNKNOWN"
            suggested_class = "unknown"
            reasons = [
                "LOW_ADMISSIBILITY_GATE_PASSED",
                "HIGH_REJECT_PROBABILITY_GATE_PASSED",
                "NO_DOMINANT_TARGET_GATE_PASSED",
            ]
        else:
            reasons = []
            if admissible_raw < float(thresholds["auto_target_min_admissible"]):
                reasons.append("ADMISSIBILITY_BELOW_AUTO_TARGET_THRESHOLD")
            if top1_probability_raw < float(
                thresholds["auto_target_min_probability_by_class"][top1]
            ):
                reasons.append("TOP1_PROBABILITY_BELOW_CLASS_THRESHOLD")
            if margin_raw < float(thresholds["auto_target_min_margin"]):
                reasons.append("TOP1_TOP2_MARGIN_BELOW_THRESHOLD")
            if thresholds["require_acquisition_hint_agreement"] and not hint_agrees:
                reasons.append("ACQUISITION_HINT_DISAGREES")
            if reasons:
                decision = "HOLD"
                suggested_class = top1
            else:
                decision = "AUTO_TARGET"
                suggested_class = top1

        result = {
            "schema_version": self.policy["result_schema_version"],
            "mode": "PRODUCTION" if self.production_mode else "FIXTURE",
            "asset": image["asset"],
            "pageid": image["pageid"],
            "candidate_sha256": image["candidate_sha256"],
            "image_path": image["local_path"],
            "acquisition_class_hint": image["class_hint"],
            "model_id": self.config.model_id,
            "model_artifact_sha256": self.model_artifact["sha256"],
            "prompt_set_sha256": self.prompt_set_sha256,
            "calibration_sha256": self.calibration_sha256,
            "calibration_status": self.calibration["status"],
            "prompt_scores": {
                prompt_id: _round_float(prompt_scores_raw[prompt_id])
                for prompt_id in self.prompt_index
            },
            "quality_score": _round_float(quality_raw),
            "class_scores": {
                class_id: _round_float(class_raw[class_id]) for class_id in TARGET_CLASSES
            },
            "reject_family_scores": {
                family: _round_float(reject_raw[family]) for family in REJECT_FAMILIES
            },
            "reject_family_probabilities": {
                family: _round_float(reject_family_probabilities_raw[family])
                for family in REJECT_FAMILIES
            },
            "dominant_reject_family": dominant_reject,
            "dominant_reject_score": _round_float(dominant_reject_raw),
            "admissibility_probability": _round_float(admissible_raw),
            "reject_probability": _round_float(reject_probability_raw),
            "class_probabilities": _round_probabilities(
                TARGET_CLASSES, morphology_probabilities
            ),
            "top1_class": top1,
            "top1_probability": _round_float(top1_probability_raw),
            "top2_class": top2,
            "top2_probability": _round_float(top2_probability_raw),
            "top1_top2_margin": _round_float(margin_raw),
            "acquisition_hint_agrees": hint_agrees,
            "decision": decision,
            "suggested_class": suggested_class,
            "decision_reasons": reasons,
            "authority": AUTHORITY,
            "explicit_non_claims": self.policy["explicit_non_claims"],
        }
        return result

    def run(self) -> dict[str, Any]:
        preflight = self.preflight()
        if self.output_dir.exists():
            raise EnsembleError("AI ensemble output already exists; immutable runs are never overwritten")
        score_matrix, provenance = self._score_matrix()
        self._assert_model_unchanged()
        self._assert_images_unchanged()
        if self._current_input_roots() != self.input_roots:
            raise EnsembleError("a frozen input changed during ensemble inference")
        results = [
            self._render_result(image, scores)
            for image, scores in zip(self.image_records, score_matrix, strict=True)
        ]
        holds = [
            {
                "schema_version": self.policy["hold_schema_version"],
                "mode": result["mode"],
                "asset": result["asset"],
                "pageid": result["pageid"],
                "candidate_sha256": result["candidate_sha256"],
                "suggested_class": result["suggested_class"],
                "admissibility_probability": result["admissibility_probability"],
                "top1_probability": result["top1_probability"],
                "top1_top2_margin": result["top1_top2_margin"],
                "decision_reasons": result["decision_reasons"],
                "result_sha256": _sha256_bytes(_canonical_bytes(result)),
                "authority": AUTHORITY,
                "explicit_non_claims": self.policy["explicit_non_claims"],
            }
            for result in results
            if result["decision"] == "HOLD"
        ]
        decision_counts = Counter(result["decision"] for result in results)
        target_counts = Counter(
            result["suggested_class"]
            for result in results
            if result["decision"] == "AUTO_TARGET"
        )
        reject_counts = Counter(result["dominant_reject_family"] for result in results)
        hint_agreement = sum(1 for result in results if result["acquisition_hint_agrees"])
        mode = "PRODUCTION" if self.production_mode else "FIXTURE"
        status = self.policy["output_contract"]["status"]
        if not self.production_mode:
            status = "FIXTURE_" + status
        stats = {
            "schema_version": self.policy["stats_schema_version"],
            "mode": mode,
            "status": status,
            "candidate_count": len(results),
            "decision_counts": {
                decision: decision_counts[decision]
                for decision in ("AUTO_TARGET", "AUTO_UNKNOWN", "HOLD")
            },
            "auto_target_counts_by_class": {
                class_id: target_counts[class_id] for class_id in TARGET_CLASSES
            },
            "dominant_reject_family_counts": {
                family: reject_counts[family] for family in REJECT_FAMILIES
            },
            "acquisition_hint_agreement_count": hint_agreement,
            "acquisition_hint_disagreement_count": len(results) - hint_agreement,
            "prompt_count": len(self.prompt_records),
            "thresholds": self.policy["thresholds"],
            "calibration": self.calibration,
            "authority": AUTHORITY,
            "explicit_non_claims": self.policy["explicit_non_claims"],
        }
        results_payload = _jsonl_bytes(results)
        hold_payload = _jsonl_bytes(holds)
        stats_payload = _json_bytes(stats)
        output = self.policy["output_contract"]
        output_hashes = {
            output["results_filename"]: _sha256_bytes(results_payload),
            output["hold_filename"]: _sha256_bytes(hold_payload),
            output["stats_filename"]: _sha256_bytes(stats_payload),
        }
        score_binding = [
            {
                "asset": result["asset"],
                "candidate_sha256": result["candidate_sha256"],
                "prompt_scores": result["prompt_scores"],
            }
            for result in results
        ]
        score_matrix_sha256 = _sha256_bytes(_canonical_bytes(score_binding))
        run_binding = {
            "policy_sha256": self.policy_sha256,
            "implementation_sha256": self.implementation_sha256,
            "input_roots": self.input_roots,
            "image_payload_set_sha256": self.image_payload_set_sha256,
            "model_id": self.config.model_id,
            "model_artifact_sha256": self.model_artifact["sha256"],
            "prompt_set_sha256": self.prompt_set_sha256,
            "calibration_sha256": self.calibration_sha256,
            "score_matrix_sha256": score_matrix_sha256,
            "results_sha256": output_hashes[output["results_filename"]],
        }
        receipt = {
            "schema_version": self.policy["receipt_schema_version"],
            "mode": mode,
            "status": status,
            "run_id": "sha256:" + _sha256_bytes(_canonical_bytes(run_binding)),
            "policy_sha256": self.policy_sha256,
            "implementation_sha256": self.implementation_sha256,
            "input_roots": self.input_roots,
            "image_payload_set_sha256": self.image_payload_set_sha256,
            "candidate_count": len(results),
            "model": {
                "model_id": self.config.model_id,
                "backend": self.config.backend,
                "artifact": self.model_artifact,
                "local_files_only": self.config.backend in {
                    "local_siglip2",
                    "local_openclip_bigvision_npz",
                },
                "trust_remote_code": False,
                "safetensors_only": self.config.backend == "local_siglip2",
                "scoring_provenance": provenance,
            },
            "prompt_set_sha256": self.prompt_set_sha256,
            "calibration_sha256": self.calibration_sha256,
            "score_matrix_sha256": score_matrix_sha256,
            "counts": {
                "auto_target": decision_counts["AUTO_TARGET"],
                "auto_unknown": decision_counts["AUTO_UNKNOWN"],
                "hold": decision_counts["HOLD"],
            },
            "outputs": output_hashes,
            "output_scope": "review/ai_ensemble_v1 only",
            "human_review_files_touched": False,
            "dataset_manifest_written": False,
            "authority": AUTHORITY,
            "explicit_non_claims": self.policy["explicit_non_claims"],
        }
        receipt_payload = _json_bytes(receipt)
        payloads = {
            output["results_filename"]: results_payload,
            output["hold_filename"]: hold_payload,
            output["stats_filename"]: stats_payload,
            output["receipt_filename"]: receipt_payload,
        }
        self._commit_output(payloads)
        return {"preflight": preflight, "receipt": receipt}

    def _commit_output(self, payloads: Mapping[str, bytes]) -> None:
        expected_parent = self.queue_path.parent.resolve(strict=True)
        parent = self.output_dir.parent.resolve(strict=True)
        if parent != expected_parent or self.output_dir.name != "ai_ensemble_v1":
            raise EnsembleError("output scope changed before commit")
        temporary = Path(tempfile.mkdtemp(prefix=".ai_ensemble_v1.tmp-", dir=str(parent)))
        try:
            if temporary.parent.resolve(strict=True) != parent:
                raise EnsembleError("temporary output escaped the review directory")
            for name, payload in payloads.items():
                if Path(name).name != name:
                    raise EnsembleError("output filename is not a basename")
                with (temporary / name).open("xb") as handle:
                    handle.write(payload)
                    handle.flush()
                    os.fsync(handle.fileno())
            if self.output_dir.exists():
                raise EnsembleError("AI ensemble output appeared during commit")
            os.replace(temporary, self.output_dir)
        except Exception:
            if temporary.exists() and temporary.parent.resolve(strict=True) == parent:
                shutil.rmtree(temporary)
            raise


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "RootScope two-stage local SigLIP2 ensemble; emits AI suggestions only, "
            "never human review or DATA_LOCKED authority."
        )
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight", action="store_true", help="read-only validation; writes nothing")
    mode.add_argument("--run", action="store_true", help="write one immutable AI ensemble run")
    parser.add_argument("--queue", type=Path, default=DEFAULT_QUEUE)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--queue-summary", type=Path, default=DEFAULT_QUEUE_SUMMARY)
    parser.add_argument("--integrity-audit", type=Path, default=DEFAULT_INTEGRITY_AUDIT)
    parser.add_argument("--class-contract", type=Path, default=DEFAULT_CLASS_CONTRACT)
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument(
        "--tokenizer-path",
        type=Path,
        help="required local tokenizer directory for the OpenCLIP Big Vision backend",
    )
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--fixture-mode", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "--backend",
        choices=("local_siglip2", "local_openclip_bigvision_npz", "fixture_fake"),
        default="local_siglip2",
        help=argparse.SUPPRESS,
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    config = EnsembleConfig(
        queue_path=args.queue,
        manifest_path=args.manifest,
        queue_summary_path=args.queue_summary,
        integrity_audit_path=args.integrity_audit,
        class_contract_path=args.class_contract,
        policy_path=args.policy,
        output_dir=args.output_dir,
        model_path=args.model_path,
        tokenizer_path=args.tokenizer_path,
        model_id=args.model_id,
        backend=args.backend,
        fixture_mode=args.fixture_mode,
        device=args.device,
        batch_size=args.batch_size,
    )
    try:
        ensemble = AISigLIP2Ensemble(config)
        if args.preflight:
            report = ensemble.preflight()
            print(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2))
        else:
            result = ensemble.run()
            print(json.dumps(result["receipt"], ensure_ascii=False, sort_keys=True, indent=2))
        return 0
    except EnsembleError as exc:
        print(f"AI SigLIP2 ensemble refused: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
