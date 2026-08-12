#!/usr/bin/env python3
"""Train the four fixed RootScope answer-card classifier for the event demo.

This is deliberately scoped to the four printed cards used in the live demo.
It produces a static ResNet18 ONNX model plus rectified AKAZE templates.  The
small same-session holdout is evidence for the fixed-card demo only and must not
be described as natural-scene generalization.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from torchvision.transforms import InterpolationMode


CLASS_ORDER = ("grass_clump", "low_shrub", "young_tree", "non_target")
DISPLAY_NAMES = {
    "grass_clump": "草丛 / GRASS",
    "low_shrub": "灌木 / SHRUB",
    "young_tree": "幼树 / TREE",
    "non_target": "纯沙 / HOLD",
}
ACTION_LEVELS = {
    "grass_clump": 1,
    "low_shrub": 2,
    "young_tree": 3,
    "non_target": 0,
}
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace(
        "+00:00", "Z"
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def order_quad(points: np.ndarray) -> np.ndarray:
    pts = points.astype(np.float32).reshape(4, 2)
    sums = pts.sum(axis=1)
    diffs = np.diff(pts, axis=1).reshape(-1)
    return np.array(
        [
            pts[np.argmin(sums)],
            pts[np.argmin(diffs)],
            pts[np.argmax(sums)],
            pts[np.argmax(diffs)],
        ],
        dtype=np.float32,
    )


def rectify_card(image_bgr: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    """Find and rectify the dominant printed sheet, with a safe center fallback."""

    height, width = image_bgr.shape[:2]
    scale = min(1.0, 1280.0 / max(width, height))
    small = cv2.resize(
        image_bgr,
        (max(1, int(width * scale)), max(1, int(height * scale))),
        interpolation=cv2.INTER_AREA,
    )
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 35, 120)
    edges = cv2.morphologyEx(
        edges, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8), iterations=2
    )
    contours, _ = cv2.findContours(
        edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
    )
    frame_area = float(small.shape[0] * small.shape[1])
    candidates: list[tuple[float, np.ndarray]] = []
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < 0.22 * frame_area or area > 0.93 * frame_area:
            continue
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.025 * perimeter, True)
        if len(approx) != 4 or not cv2.isContourConvex(approx):
            continue
        quad = order_quad(approx / scale)
        top = np.linalg.norm(quad[1] - quad[0])
        bottom = np.linalg.norm(quad[2] - quad[3])
        left = np.linalg.norm(quad[3] - quad[0])
        right = np.linalg.norm(quad[2] - quad[1])
        long_side = max((top + bottom) * 0.5, (left + right) * 0.5)
        short_side = min((top + bottom) * 0.5, (left + right) * 0.5)
        if short_side <= 1.0:
            continue
        aspect = long_side / short_side
        if not 1.05 <= aspect <= 2.2:
            continue
        candidates.append((area, quad))

    if candidates:
        _, quad = max(candidates, key=lambda item: item[0])
        top = np.linalg.norm(quad[1] - quad[0])
        bottom = np.linalg.norm(quad[2] - quad[3])
        left = np.linalg.norm(quad[3] - quad[0])
        right = np.linalg.norm(quad[2] - quad[1])
        target_w = int(max(top, bottom))
        target_h = int(max(left, right))
        if target_h > target_w:
            quad = np.roll(quad, -1, axis=0)
            target_w, target_h = target_h, target_w
        target_w = int(np.clip(target_w, 640, 1600))
        target_h = int(np.clip(target_h, 420, 1200))
        destination = np.array(
            [
                [0, 0],
                [target_w - 1, 0],
                [target_w - 1, target_h - 1],
                [0, target_h - 1],
            ],
            dtype=np.float32,
        )
        matrix = cv2.getPerspectiveTransform(quad, destination)
        warped = cv2.warpPerspective(image_bgr, matrix, (target_w, target_h))
        return warped, {
            "method": "quadrilateral",
            "quad_xy": np.round(quad, 1).tolist(),
            "output_wh": [target_w, target_h],
        }

    # The answer card is deliberately centered during the demo.  The tighter
    # fallback removes the shared carpet/table background instead of letting a
    # classifier memorize it when the paper outline is not a clean contour.
    x0, x1 = int(width * 0.17), int(width * 0.83)
    y0, y1 = int(height * 0.11), int(height * 0.93)
    return image_bgr[y0:y1, x0:x1].copy(), {
        "method": "center_fallback",
        "crop_xyxy": [x0, y0, x1, y1],
        "output_wh": [x1 - x0, y1 - y0],
    }


@dataclass(frozen=True)
class Sample:
    path: Path
    class_id: str
    sequence: int
    source_sha256: str
    roi_path: Path


class CardDataset(Dataset[tuple[torch.Tensor, int]]):
    def __init__(
        self,
        samples: list[Sample],
        transform: Any,
        *,
        repeats: int = 1,
    ) -> None:
        self.samples = samples
        self.transform = transform
        self.repeats = repeats

    def __len__(self) -> int:
        return len(self.samples) * self.repeats

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        sample = self.samples[index % len(self.samples)]
        with Image.open(sample.roi_path) as image:
            rgb = image.convert("RGB")
            tensor = self.transform(rgb)
        return tensor, CLASS_ORDER.index(sample.class_id)


def load_and_rectify(session: Path, output: Path) -> tuple[list[Sample], list[dict[str, Any]]]:
    records = [
        json.loads(line)
        for line in (session / "captures.jsonl").read_text("utf-8").splitlines()
        if line.strip()
    ]
    counts = {class_id: 0 for class_id in CLASS_ORDER}
    samples: list[Sample] = []
    roi_records: list[dict[str, Any]] = []
    roi_root = output / "rectified"
    for record in records:
        class_id = str(record["class_id"])
        if class_id not in counts:
            raise RuntimeError(f"unexpected class in capture manifest: {class_id}")
        source = session / str(record["relative_path"])
        observed_sha = sha256_file(source)
        if observed_sha != record["sha256"]:
            raise RuntimeError(f"source hash mismatch: {source}")
        image = cv2.imread(str(source), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"cannot decode: {source}")
        roi, geometry = rectify_card(image)
        roi_scale = min(1.0, 960.0 / max(roi.shape[:2]))
        if roi_scale < 1.0:
            roi = cv2.resize(
                roi,
                (
                    max(1, int(round(roi.shape[1] * roi_scale))),
                    max(1, int(round(roi.shape[0] * roi_scale))),
                ),
                interpolation=cv2.INTER_AREA,
            )
            geometry["stored_wh"] = [int(roi.shape[1]), int(roi.shape[0])]
        sequence = int(record["sequence_in_class"])
        roi_dir = roi_root / class_id
        roi_dir.mkdir(parents=True, exist_ok=True)
        roi_path = roi_dir / f"{sequence:02d}.jpg"
        if not cv2.imwrite(str(roi_path), roi, [cv2.IMWRITE_JPEG_QUALITY, 97]):
            raise RuntimeError(f"cannot save ROI: {roi_path}")
        counts[class_id] += 1
        samples.append(
            Sample(
                path=source,
                class_id=class_id,
                sequence=sequence,
                source_sha256=observed_sha,
                roi_path=roi_path,
            )
        )
        roi_records.append(
            {
                "class_id": class_id,
                "sequence": sequence,
                "source": source.relative_to(session).as_posix(),
                "source_sha256": observed_sha,
                "roi": roi_path.relative_to(output).as_posix(),
                "roi_sha256": sha256_file(roi_path),
                "geometry": geometry,
            }
        )
    if any(counts[class_id] != 5 for class_id in CLASS_ORDER):
        raise RuntimeError(f"expected exactly five captures per class: {counts}")
    return samples, roi_records


def build_model() -> nn.Module:
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.avgpool = nn.AvgPool2d(kernel_size=7, stride=1)
    model.fc = nn.Linear(model.fc.in_features, len(CLASS_ORDER))
    for parameter in model.parameters():
        parameter.requires_grad = False
    for parameter in model.layer4.parameters():
        parameter.requires_grad = True
    for parameter in model.fc.parameters():
        parameter.requires_grad = True
    return model


def evaluate(model: nn.Module, loader: DataLoader[Any], device: torch.device) -> dict[str, Any]:
    model.eval()
    correct = 0
    total = 0
    rows: list[dict[str, Any]] = []
    with torch.inference_mode():
        for tensors, labels in loader:
            logits = model(tensors.to(device))
            probabilities = torch.softmax(logits, dim=1).cpu()
            predictions = probabilities.argmax(dim=1)
            for label, prediction, probs in zip(labels, predictions, probabilities):
                sorted_probs = torch.sort(probs, descending=True).values
                rows.append(
                    {
                        "truth": CLASS_ORDER[int(label)],
                        "prediction": CLASS_ORDER[int(prediction)],
                        "confidence": float(sorted_probs[0]),
                        "margin": float(sorted_probs[0] - sorted_probs[1]),
                    }
                )
                correct += int(label == prediction)
                total += 1
    return {"accuracy": correct / max(1, total), "count": total, "rows": rows}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--session", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=24)
    parser.add_argument("--seed", type=int, default=20260725)
    args = parser.parse_args()

    session = args.session.resolve()
    output = args.output.resolve()
    if output.exists():
        raise SystemExit(f"REFUSED: output already exists: {output}")
    output.mkdir(parents=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    completion = json.loads((session / "completion.json").read_text("utf-8"))
    if completion.get("total_saved") != 20:
        raise RuntimeError(f"capture session is incomplete: {completion}")
    samples, roi_records = load_and_rectify(session, output)
    write_json(output / "rectification_manifest.json", roi_records)

    # Sequence 1/3/5 train; 2/4 are a strict same-session temporal holdout.
    train_samples = [sample for sample in samples if sample.sequence in {1, 3, 5}]
    val_samples = [sample for sample in samples if sample.sequence in {2, 4}]
    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(
                224,
                scale=(0.72, 1.0),
                ratio=(0.88, 1.18),
                interpolation=InterpolationMode.BILINEAR,
            ),
            transforms.RandomPerspective(0.16, p=0.65),
            transforms.RandomRotation(7, interpolation=InterpolationMode.BILINEAR),
            transforms.RandomHorizontalFlip(p=0.35),
            transforms.ColorJitter(
                brightness=0.32, contrast=0.28, saturation=0.22, hue=0.035
            ),
            transforms.RandomApply(
                [transforms.GaussianBlur(3, sigma=(0.1, 1.2))], p=0.18
            ),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.Resize(256, interpolation=InterpolationMode.BILINEAR),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    train_dataset = CardDataset(train_samples, train_transform, repeats=24)
    val_dataset = CardDataset(val_samples, eval_transform)
    train_loader = DataLoader(
        train_dataset,
        batch_size=24,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=0)

    if not torch.cuda.is_available():
        raise RuntimeError("RTX GPU is required for this event training run")
    device = torch.device("cuda")
    model = build_model().to(device)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=2.5e-4,
        weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, args.epochs)
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=0.04)
    best_state: dict[str, torch.Tensor] | None = None
    best_score = (-1.0, -1.0)
    history: list[dict[str, Any]] = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_total = 0.0
        sample_total = 0
        for tensors, labels in train_loader:
            tensors = tensors.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(tensors)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            loss_total += float(loss) * int(labels.shape[0])
            sample_total += int(labels.shape[0])
        scheduler.step()
        validation = evaluate(model, val_loader, device)
        min_margin = min(row["margin"] for row in validation["rows"])
        score = (float(validation["accuracy"]), float(min_margin))
        if score > best_score:
            best_score = score
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
        row = {
            "epoch": epoch,
            "train_loss": loss_total / max(1, sample_total),
            "val_accuracy": validation["accuracy"],
            "val_min_margin": min_margin,
            "lr": optimizer.param_groups[0]["lr"],
        }
        history.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)

    if best_state is None:
        raise RuntimeError("training produced no checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    validation = evaluate(model, val_loader, device)
    training_reference = evaluate(
        model,
        DataLoader(CardDataset(train_samples, eval_transform), batch_size=12),
        device,
    )
    if validation["accuracy"] < 1.0:
        raise RuntimeError(f"same-session holdout did not reach 100%: {validation}")

    checkpoint = output / "rootscope_answer_cards_resnet18.pt"
    torch.save(
        {
            "schema": "rootscope.answer_cards.checkpoint.v1",
            "class_order": CLASS_ORDER,
            "state_dict": best_state,
        },
        checkpoint,
    )

    model_cpu = model.to("cpu")
    onnx_path = output / "rootscope_answer_cards_resnet18_opset11.onnx"
    dummy = torch.zeros(1, 3, 224, 224)
    torch.onnx.export(
        model_cpu,
        dummy,
        str(onnx_path),
        input_names=["input"],
        output_names=["logits"],
        opset_version=11,
        do_constant_folding=True,
        dynamic_axes=None,
        dynamo=False,
    )
    import onnx
    import onnxruntime as ort

    graph = onnx.load(str(onnx_path))
    onnx.checker.check_model(graph)
    session_ort = ort.InferenceSession(
        str(onnx_path), providers=["CPUExecutionProvider"]
    )
    with torch.inference_mode():
        torch_logits = model_cpu(dummy).numpy()
    ort_logits = session_ort.run(["logits"], {"input": dummy.numpy()})[0]
    max_abs_error = float(np.max(np.abs(torch_logits - ort_logits)))
    if max_abs_error > 1e-4:
        raise RuntimeError(f"Torch/ONNX mismatch: {max_abs_error}")

    templates_root = output / "templates"
    templates_root.mkdir()
    template_manifest: list[dict[str, Any]] = []
    for class_id in CLASS_ORDER:
        candidates = [sample for sample in samples if sample.class_id == class_id]
        selected = max(
            candidates,
            key=lambda sample: float(
                cv2.Laplacian(
                    cv2.imread(str(sample.roi_path), cv2.IMREAD_GRAYSCALE),
                    cv2.CV_64F,
                ).var()
            ),
        )
        destination = templates_root / f"{class_id}.jpg"
        shutil.copy2(selected.roi_path, destination)
        template_manifest.append(
            {
                "class_id": class_id,
                "display_name": DISPLAY_NAMES[class_id],
                "action_level": ACTION_LEVELS[class_id],
                "source_sequence": selected.sequence,
                "path": destination.relative_to(output).as_posix(),
                "sha256": sha256_file(destination),
            }
        )
    write_json(output / "templates_manifest.json", template_manifest)

    correct_rows = [
        row for row in validation["rows"] if row["truth"] == row["prediction"]
    ]
    confidence_threshold = max(
        0.60, min(row["confidence"] for row in correct_rows) * 0.78
    )
    margin_threshold = max(0.12, min(row["margin"] for row in correct_rows) * 0.70)
    manifest = {
        "schema": "rootscope.answer_cards.model_manifest.v1",
        "created_at_utc": utc_now(),
        "scope": "FOUR_FIXED_PRINTED_ANSWER_CARDS_ONLY",
        "truth_boundary": (
            "Same-session temporal holdout and exact printed-card demo; "
            "not a natural-scene or field-generalization claim."
        ),
        "source_session": str(session),
        "source_completion_sha256": sha256_file(session / "completion.json"),
        "class_order": list(CLASS_ORDER),
        "display_names": DISPLAY_NAMES,
        "action_levels": ACTION_LEVELS,
        "input": {
            "shape": [1, 3, 224, 224],
            "layout": "NCHW",
            "color": "RGB",
            "mean": IMAGENET_MEAN,
            "std": IMAGENET_STD,
            "preprocess": "rectify_card_resize256_center_crop224",
        },
        "training": {
            "gpu": torch.cuda.get_device_name(0),
            "torch": torch.__version__,
            "epochs": args.epochs,
            "seed": args.seed,
            "train_images": len(train_samples),
            "train_augmented_samples_per_epoch": len(train_dataset),
            "validation_images": len(val_samples),
            "history": history,
        },
        "metrics": {
            "train_reference": training_reference,
            "same_session_temporal_holdout": validation,
            "confidence_threshold": confidence_threshold,
            "margin_threshold": margin_threshold,
        },
        "artifacts": {
            "checkpoint": {
                "path": checkpoint.name,
                "sha256": sha256_file(checkpoint),
            },
            "onnx": {"path": onnx_path.name, "sha256": sha256_file(onnx_path)},
            "templates_manifest": {
                "path": "templates_manifest.json",
                "sha256": sha256_file(output / "templates_manifest.json"),
            },
        },
        "onnx_consistency": {
            "opset": 11,
            "static_shape": [1, 3, 224, 224],
            "max_abs_logit_error_zero_probe": max_abs_error,
            "passed": True,
        },
        "selected_for_answer_demo": True,
        "plant2action_authority": False,
    }
    write_json(output / "model_manifest.json", manifest)
    print(
        json.dumps(
            {
                "status": "PASS",
                "output": str(output),
                "validation_accuracy": validation["accuracy"],
                "onnx": str(onnx_path),
                "onnx_sha256": sha256_file(onnx_path),
                "max_abs_error": max_abs_error,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
