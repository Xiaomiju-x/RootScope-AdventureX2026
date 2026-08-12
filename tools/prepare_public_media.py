#!/usr/bin/env python3
"""Create privacy-scrubbed, deterministic-ish public media derivatives.

This script is intentionally explicit: it accepts no arbitrary input directory, keeps
the private originals out of Git, removes image metadata, crops phone watermarks,
redacts sensitive screens, drops audio, and cuts the demo before people enter frame.
The demonstration collection bottle is intentionally preserved without redaction at
the project owner's direction; it contains no project secret or private identifier.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path

import cv2
from PIL import Image, ImageFilter, ImageOps


REPO = Path(__file__).resolve().parents[1]


def required_directory(variable: str) -> Path:
    value = os.environ.get(variable)
    if not value:
        raise RuntimeError(f"Set {variable} to the private source directory; it is never committed")
    path = Path(value).expanduser().resolve()
    if not path.is_dir():
        raise RuntimeError(f"{variable} is not a directory: {path}")
    return path


DOWNLOADS = required_directory("ROOTSCOPE_MEDIA_SOURCE")
PRIVATE_ARCHIVE = required_directory("ROOTSCOPE_PRIVATE_ARCHIVE")
OUT = REPO / "assets" / "media"
PUBLIC_RIGHTS_BASIS = (
    "RootScope team archive; public release directed by the project owner on "
    "2026-08-12; likeness and trademark rights do not imply endorsement"
)


@dataclass(frozen=True)
class Photo:
    source: Path
    public: str
    caption: str
    crop: tuple[float, float, float, float] = (0.0, 0.0, 1.0, 1.0)
    blur_boxes: tuple[tuple[float, float, float, float], ...] = ()


PHOTOS = (
    Photo(DOWNLOADS / "mmexport1786529980151.jpg", "hero/rootscope-hero.jpg", "RootScope fixed irrigation chamber; wheeled chassis used only as a carrier.", (0.00, 0.00, 0.88, 1.00)),
    Photo(DOWNLOADS / "mmexport1786529978786.jpg", "hardware/rootscope-rig-top-view.jpg", "Top view of the reservoir, wiring, controller and gantry.", (0.00, 0.02, 0.96, 0.96)),
    Photo(DOWNLOADS / "mmexport1786529977040.jpg", "demo/rootsight-live-inference.jpg", "RootSight live answer-card inference on the physical rig.", (0.02, 0.05, 0.98, 0.94)),
    Photo(DOWNLOADS / "mmexport1786529975498.jpg", "hardware/rdk-x5-edge-computer.jpg", "RDK X5 edge computer used for perception and explanation.", (0.03, 0.02, 0.97, 0.94)),
    Photo(DOWNLOADS / "mmexport1786529973431.jpg", "hardware/stm32-control-stack.jpg", "STM32F103 control and power stack.", (0.03, 0.02, 0.97, 0.94)),
    Photo(DOWNLOADS / "mmexport1786529971968.jpg", "demo/dual-evidence-grass-clump.jpg", "Controlled-card dual-evidence decision: CNN plus geometric matching.", (0.00, 0.10, 1.00, 1.00)),
    Photo(DOWNLOADS / "mmexport1786529981697.jpg", "hardware/rack-pinion-probe-gantry.jpg", "Rack-and-pinion probe and water-line gantry.", (0.03, 0.01, 0.97, 0.94)),
    Photo(DOWNLOADS / "1786530040465.jpg", "event/hackathon-workbench-night.jpg", "Night-time development at AdventureX.", (0.00, 0.00, 1.00, 0.75), ((0.74, 0.45, 1.00, 0.93),)),
    Photo(DOWNLOADS / "1786530040478.jpg", "event/adventurex-venue-entrance.jpg", "AdventureX venue entrance.", (0.00, 0.00, 1.00, 0.75)),
    Photo(DOWNLOADS / "1786530040496.jpg", "event/adventurex-campus-banners.jpg", "AdventureX 2026 event banners.", (0.00, 0.00, 1.00, 0.75)),
    Photo(DOWNLOADS / "1786530040522.jpg", "event/team-building-at-hackathon.jpg", "RootScope team building the system during the hackathon.", (0.00, 0.00, 1.00, 0.70), ((0.00, 0.48, 0.72, 0.92),)),
    Photo(DOWNLOADS / "1786530040395.jpg", "award/award-stage.jpg", "D-Robotics Give AI a Body track award ceremony; context image only.", (0.00, 0.00, 1.00, 0.72)),
    Photo(DOWNLOADS / "1786530040428.jpg", "event/adventurex-campus-sunset.jpg", "AdventureX campus at sunset.", (0.00, 0.00, 1.00, 0.75)),
    Photo(DOWNLOADS / "1786530040445.jpg", "event/adventurex-campus-lakeside.jpg", "AdventureX campus lakeside.", (0.00, 0.00, 1.00, 0.75)),
    Photo(PRIVATE_ARCHIVE / "evidence" / "award" / "AdventureX2026_DRobotics_银奖第二名_团队合影.jpg", "award/team-award.jpg", "RootScope team with the Silver Award display and two RDK X5 prizes."),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def px_box(box: tuple[float, float, float, float], width: int, height: int) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = box
    return round(x0 * width), round(y0 * height), round(x1 * width), round(y1 * height)


def process_photo(item: Photo) -> dict[str, str | int]:
    target = OUT / item.public
    target.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(item.source) as opened:
        image = ImageOps.exif_transpose(opened).convert("RGB")
        image = image.crop(px_box(item.crop, *image.size))
        for box in item.blur_boxes:
            region_box = px_box(box, *image.size)
            region = image.crop(region_box).filter(ImageFilter.GaussianBlur(radius=28))
            image.paste(region, region_box)
        image.thumbnail((2400, 1800), Image.Resampling.LANCZOS)
        image.save(target, "JPEG", quality=88, optimize=True, progressive=True, exif=b"")
    return {
        "source_name": item.source.name,
        "source_sha256": sha256(item.source),
        "public_path": target.relative_to(REPO).as_posix(),
        "public_sha256": sha256(target),
        "bytes": target.stat().st_size,
        "caption": item.caption,
        "rights_basis": PUBLIC_RIGHTS_BASIS,
        "transform": "EXIF removed; privacy crop; documented screen/label blur where applicable; max 2400x1800 JPEG",
    }


def write_video(source: Path, public: str, start: float, end: float) -> dict[str, str | int | float]:
    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {source}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    cap.set(cv2.CAP_PROP_POS_MSEC, start * 1000)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    target = OUT / public
    target.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(target), cv2.VideoWriter_fourcc(*"avc1"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError("H.264 writer unavailable; put Cisco OpenH264 1.8 DLL on PATH")
    frame_index = 0
    max_frames = round((end - start) * fps)
    while frame_index < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        writer.write(frame)
        frame_index += 1
    writer.release()
    cap.release()
    return {
        "source_name": source.name,
        "source_sha256": sha256(source),
        "public_path": target.relative_to(REPO).as_posix(),
        "public_sha256": sha256(target),
        "bytes": target.stat().st_size,
        "duration_seconds": round(frame_index / fps, 3),
        "caption": "Physical probe and water-delivery demonstration; controlled technical segment.",
        "rights_basis": PUBLIC_RIGHTS_BASIS,
        "transform": f"trim {start:.1f}-{end:.1f}s; audio removed; H.264/yuv420p; bottle and label preserved; container metadata discarded",
    }


def make_poster(source: Path) -> dict[str, str | int]:
    cap = cv2.VideoCapture(str(source))
    cap.set(cv2.CAP_PROP_POS_MSEC, 27000)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError("Could not extract video poster")
    target = OUT / "demo" / "demo-overview.jpg"
    cv2.imwrite(str(target), frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
    # OpenCV does not copy EXIF; re-save through Pillow to ensure a progressive, metadata-free file.
    with Image.open(target) as opened:
        opened.convert("RGB").save(target, "JPEG", quality=88, optimize=True, progressive=True, exif=b"")
    return {
        "source_name": source.name,
        "source_sha256": sha256(source),
        "public_path": target.relative_to(REPO).as_posix(),
        "public_sha256": sha256(target),
        "bytes": target.stat().st_size,
        "caption": "Water-delivery frame from the physical demonstration.",
        "rights_basis": PUBLIC_RIGHTS_BASIS,
        "transform": "frame at 27.0s; bottle and label preserved; metadata removed",
    }


def make_gif_preview(
    source: Path,
    public: str,
    start: float,
    end: float,
) -> dict[str, str | int | float]:
    """Build a small, README-native preview directly from the unredacted video."""

    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {source}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    cap.set(cv2.CAP_PROP_POS_MSEC, start * 1000)
    sample_step = 6
    max_frames = round((end - start) * fps)
    frames: list[Image.Image] = []
    for frame_index in range(max_frames):
        ok, frame = cap.read()
        if not ok:
            break
        if frame_index % sample_step:
            continue
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)
        image.thumbnail((280, 500), Image.Resampling.LANCZOS)
        frames.append(
            image.quantize(
                colors=48,
                method=Image.Quantize.MEDIANCUT,
                dither=Image.Dither.FLOYDSTEINBERG,
            )
        )
    cap.release()
    if not frames:
        raise RuntimeError(f"Could not extract GIF preview from {source}")
    target = OUT / public
    target.parent.mkdir(parents=True, exist_ok=True)
    frame_duration_ms = round(sample_step / fps * 1000)
    frames[0].save(
        target,
        save_all=True,
        append_images=frames[1:],
        duration=frame_duration_ms,
        loop=0,
        optimize=True,
        disposal=2,
    )
    return {
        "source_name": source.name,
        "source_sha256": sha256(source),
        "public_path": target.relative_to(REPO).as_posix(),
        "public_sha256": sha256(target),
        "bytes": target.stat().st_size,
        "duration_seconds": round(len(frames) * frame_duration_ms / 1000, 3),
        "caption": "README-native animated preview derived from the physical demonstration.",
        "rights_basis": PUBLIC_RIGHTS_BASIS,
        "transform": (
            f"preview from {start:.1f}-{end:.1f}s; sampled at {1000 / frame_duration_ms:.1f} fps; "
            "280px maximum width; 48-color GIF; bottle and label preserved; audio absent; metadata removed"
        ),
    }


def main() -> None:
    records: list[dict[str, str | int | float]] = [process_photo(item) for item in PHOTOS]
    video = DOWNLOADS / "mmexport1786529983379.mp4"
    records.extend(
        [
            make_poster(video),
            write_video(video, "demo/rootscope-probe-and-irrigation-demo.mp4", 0.0, 33.5),
            write_video(video, "demo/probe-descent.mp4", 6.0, 24.0),
            write_video(video, "demo/water-delivery.mp4", 24.0, 31.5),
            make_gif_preview(video, "demo/probe-descent-preview.gif", 10.0, 18.0),
            make_gif_preview(video, "demo/water-delivery-preview.gif", 24.0, 31.5),
        ]
    )
    manifest_json = OUT / "ASSET_PROVENANCE.json"
    manifest_json.write_text(
        json.dumps(
            {"schema": "rootscope.public-media.v1", "assets": records},
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    with (OUT / "ASSET_PROVENANCE.csv").open("w", encoding="utf-8", newline="") as handle:
        columns = ["source_name", "source_sha256", "public_path", "public_sha256", "bytes", "duration_seconds", "caption", "rights_basis", "transform"]
        writer = csv.DictWriter(
            handle,
            fieldnames=columns,
            extrasaction="ignore",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(records)
    print(f"Prepared {len(records)} public assets under {OUT}")


if __name__ == "__main__":
    main()
