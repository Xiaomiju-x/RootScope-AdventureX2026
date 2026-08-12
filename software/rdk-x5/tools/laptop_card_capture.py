#!/usr/bin/env python3
"""Interactive RootScope card capture for a Windows laptop USB camera."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
import tkinter as tk
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageTk

try:
    from cv2_enumerate_cameras import enumerate_cameras
except ImportError:  # pragma: no cover - exercised by the operator environment
    enumerate_cameras = None


LABELS = {
    ord("1"): ("grass_clump", "G", "GRASS CLUMP"),
    ord("2"): ("low_shrub", "S", "LOW SHRUB"),
    ord("3"): ("young_tree", "T", "YOUNG TREE"),
    ord("4"): ("non_target", "U", "UNKNOWN / NON-TARGET"),
}

DEFAULT_CAMERA_NAME = "Insta360 Link 2C"
DEFAULT_CAMERA_VID_PID = "2e1a:4c03"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_camera_identity(
    *,
    requested_index: int | None,
    expected_name: str,
    expected_vid_pid: str,
) -> dict[str, Any]:
    """Resolve exactly one Windows DirectShow camera by stable USB identity."""

    if os.name != "nt":
        if requested_index is None:
            raise RuntimeError("--camera is required outside Windows")
        return {
            "index": requested_index,
            "name": "UNVERIFIED_NON_WINDOWS_CAMERA",
            "path": None,
            "vid": None,
            "pid": None,
            "vid_pid": None,
            "backend": "CAP_ANY",
            "identity_verified": False,
        }
    if enumerate_cameras is None:
        raise RuntimeError(
            "cv2-enumerate-cameras is required for identity-bound Windows capture; "
            "install cv2-enumerate-cameras==1.3.3 in adventurex/.ai_curation_venv"
        )

    parts = expected_vid_pid.lower().split(":")
    if len(parts) != 2 or any(len(part) != 4 for part in parts):
        raise RuntimeError("--expected-vid-pid must look like 2e1a:4c03")
    try:
        expected_vid, expected_pid = (int(part, 16) for part in parts)
    except ValueError as exc:
        raise RuntimeError("--expected-vid-pid is not hexadecimal") from exc

    cameras = list(enumerate_cameras(cv2.CAP_DSHOW))
    matches = [
        camera
        for camera in cameras
        if camera.name == expected_name
        and camera.vid == expected_vid
        and camera.pid == expected_pid
        and (requested_index is None or camera.index == requested_index)
    ]
    if len(matches) != 1:
        observed = [
            {
                "index": camera.index,
                "name": camera.name,
                "vid_pid": (
                    f"{camera.vid:04x}:{camera.pid:04x}"
                    if camera.vid is not None and camera.pid is not None
                    else None
                ),
                "path": camera.path,
            }
            for camera in cameras
        ]
        raise RuntimeError(
            "Expected exactly one DirectShow camera matching "
            f"name={expected_name!r}, vid_pid={expected_vid_pid!r}, "
            f"requested_index={requested_index!r}; observed={observed!r}"
        )

    camera = matches[0]
    return {
        "index": camera.index,
        "name": camera.name,
        "path": camera.path,
        "vid": f"{camera.vid:04x}",
        "pid": f"{camera.pid:04x}",
        "vid_pid": f"{camera.vid:04x}:{camera.pid:04x}",
        "backend": "CAP_DSHOW",
        "identity_verified": True,
    }


def fourcc_text(value: int) -> str:
    unsigned = value & 0xFFFFFFFF
    return "".join(chr((unsigned >> (8 * offset)) & 0xFF) for offset in range(4))


def quality_metrics(frame: np.ndarray) -> dict[str, Any]:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    brightness = float(gray.mean())
    blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    underexposed_fraction = float(np.mean(gray <= 12))
    overexposed_fraction = float(np.mean(gray >= 243))
    warnings: list[str] = []
    if blur_score < 60.0:
        warnings.append("POSSIBLE_BLUR")
    if brightness < 35.0:
        warnings.append("POSSIBLE_UNDEREXPOSURE")
    if brightness > 225.0:
        warnings.append("POSSIBLE_OVEREXPOSURE")
    if overexposed_fraction > 0.18:
        warnings.append("HIGHLIGHT_CLIPPING")
    return {
        "brightness_mean": round(brightness, 3),
        "laplacian_variance": round(blur_score, 3),
        "underexposed_fraction": round(underexposed_fraction, 6),
        "overexposed_fraction": round(overexposed_fraction, 6),
        "warnings": warnings,
    }


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def draw_overlay(
    frame: np.ndarray,
    *,
    camera_label: str,
    saved_counts: dict[str, int],
    status: str,
) -> np.ndarray:
    shown = frame.copy()
    height, width = shown.shape[:2]
    top_h = 118
    bottom_h = 70
    cv2.rectangle(shown, (0, 0), (width, top_h), (10, 24, 36), -1)
    cv2.rectangle(shown, (0, height - bottom_h), (width, height), (10, 24, 36), -1)
    cv2.putText(
        shown,
        f"RootScope | {camera_label} | {width}x{height}",
        (24, 38),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (88, 230, 190),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        shown,
        "1 GRASS   2 SHRUB   3 TREE   4 UNKNOWN   Q EXIT",
        (24, 82),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.78,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    counts = "  ".join(
        [
            f"G:{saved_counts['grass_clump']}",
            f"S:{saved_counts['low_shrub']}",
            f"T:{saved_counts['young_tree']}",
            f"U:{saved_counts['non_target']}",
        ]
    )
    cv2.putText(
        shown,
        f"{status} | {counts}",
        (24, height - 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.68,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return shown


def open_camera(index: int, width: int, height: int, fps: int) -> cv2.VideoCapture:
    backend = cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY
    capture = cv2.VideoCapture(index, backend)
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open camera index {index}.")
    capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    capture.set(cv2.CAP_PROP_FPS, fps)
    capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return capture


def build_parser() -> argparse.ArgumentParser:
    adventurex_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="RootScope laptop USB camera card capture")
    parser.add_argument(
        "--camera",
        type=int,
        default=None,
        help="Optional expected DirectShow index; USB identity matching is authoritative",
    )
    parser.add_argument("--expected-camera-name", default=DEFAULT_CAMERA_NAME)
    parser.add_argument("--expected-vid-pid", default=DEFAULT_CAMERA_VID_PID)
    parser.add_argument("--width", type=int, default=3840)
    parser.add_argument("--height", type=int, default=2160)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=adventurex_root / "captures",
        help="Parent directory for a new timestamped capture session",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    camera_identity = resolve_camera_identity(
        requested_index=args.camera,
        expected_name=args.expected_camera_name,
        expected_vid_pid=args.expected_vid_pid,
    )
    camera_index = int(camera_identity["index"])
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir = args.output_root.resolve() / f"laptop_card_session_{timestamp}"
    image_root = session_dir / "images"
    for class_id, _, _ in LABELS.values():
        (image_root / class_id).mkdir(parents=True, exist_ok=True)
    manifest_path = session_dir / "captures.jsonl"

    session = {
        "schema_version": "rootscope.laptop-card-capture-session.v2",
        "created_at_utc": utc_now(),
        "camera_index": camera_index,
        "camera_identity_before_open": camera_identity,
        "requested_mode": {
            "width": args.width,
            "height": args.height,
            "fps": args.fps,
            "fourcc": "MJPG",
        },
        "host": {
            "hostname": platform.node(),
            "platform": platform.platform(),
            "python": sys.version.split()[0],
            "opencv": cv2.__version__,
        },
        "labels": {
            "1": "grass_clump",
            "2": "low_shrub",
            "3": "young_tree",
            "4": "non_target",
        },
        "truth_boundary": "OPERATOR_LABELLED_LAPTOP_CAPTURE_NOT_HOLDOUT",
    }
    (session_dir / "session.json").write_text(
        json.dumps(session, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print("=" * 72)
    print("RootScope 笔记本 USB 摄像头采集")
    print(
        "相机身份: "
        f"{camera_identity['name']} | {camera_identity['vid_pid']} | "
        f"DirectShow index {camera_index}"
    )
    print(f"保存目录: {session_dir}")
    print("请先点击实时预览窗口，再按键：")
    print("  1 = 草丛    2 = 灌木    3 = 幼树    4 = 非目标/沙地")
    print("  Q / ESC = 完成并退出")
    print("同一个数字可以多按几次，每次都会另存一张，不会覆盖。")
    print("=" * 72)

    capture: cv2.VideoCapture | None = None
    root: tk.Tk | None = None
    saved_counts = {class_id: 0 for class_id, _, _ in LABELS.values()}
    last_status = "READY - place one card in the center"
    exit_reason = "UNKNOWN"
    latest_frame: np.ndarray | None = None

    try:
        negotiation_attempts: list[dict[str, Any]] = []
        actual_mode: dict[str, Any] | None = None
        for attempt in range(1, 3):
            capture = open_camera(camera_index, args.width, args.height, args.fps)
            successful_reads = 0
            for _ in range(20):
                ok, _ = capture.read()
                successful_reads += int(ok)
                time.sleep(0.02)
            actual_mode = {
                "width": int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
                "height": int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                "fps": float(capture.get(cv2.CAP_PROP_FPS)),
                "fourcc": fourcc_text(int(capture.get(cv2.CAP_PROP_FOURCC))),
                "successful_warmup_reads": successful_reads,
            }
            negotiation_attempts.append({"attempt": attempt, **actual_mode})
            mode_matches = (
                successful_reads > 0
                and actual_mode["width"] == args.width
                and actual_mode["height"] == args.height
                and abs(actual_mode["fps"] - args.fps) <= 1.0
            )
            if mode_matches:
                break
            capture.release()
            capture = None
            time.sleep(0.5)
        else:
            raise RuntimeError(
                "Camera mode negotiation mismatch after two attempts: "
                f"requested={args.width}x{args.height}@{args.fps}, "
                f"attempts={negotiation_attempts!r}"
            )

        camera_identity_after_open = resolve_camera_identity(
            requested_index=camera_index,
            expected_name=args.expected_camera_name,
            expected_vid_pid=args.expected_vid_pid,
        )
        if camera_identity_after_open != camera_identity:
            raise RuntimeError("Camera identity changed across the open boundary")
        session["camera_identity_after_open"] = camera_identity_after_open
        session["negotiated_mode"] = actual_mode
        session["negotiation_attempts"] = negotiation_attempts
        (session_dir / "session.json").write_text(
            json.dumps(session, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(
            "相机已打开，实际画面: "
            f"{actual_mode['width']}x{actual_mode['height']}@{actual_mode['fps']:.2f} "
            f"{actual_mode['fourcc']}"
        )

        root = tk.Tk()
        root.title("RootScope Laptop Card Capture")
        root.geometry("1280x760")
        root.configure(bg="#0a1824")
        preview = tk.Label(root, bg="#0a1824")
        preview.pack(fill=tk.BOTH, expand=True)

        def close_window(reason: str) -> None:
            nonlocal exit_reason
            exit_reason = reason
            if root is not None:
                root.quit()

        def save_current_frame(key: int) -> None:
            nonlocal last_status
            if latest_frame is None:
                last_status = "NO FRAME YET - please wait"
                return
            frame = latest_frame.copy()
            class_id, short_code, display_name = LABELS[key]
            saved_counts[class_id] += 1
            sequence = saved_counts[class_id]
            captured_at = datetime.now()
            filename = (
                f"{short_code}_{class_id}_{captured_at.strftime('%Y%m%d_%H%M%S_%f')[:-3]}"
                f"_{sequence:02d}.jpg"
            )
            image_path = image_root / class_id / filename
            if not cv2.imwrite(
                str(image_path),
                frame,
                [cv2.IMWRITE_JPEG_QUALITY, 96],
            ):
                saved_counts[class_id] -= 1
                last_status = f"SAVE FAILED: {display_name}"
                print(f"[失败] 无法保存 {image_path}")
                return

            metrics = quality_metrics(frame)
            record = {
                "schema_version": "rootscope.laptop-card-capture.v2",
                "captured_at_utc": utc_now(),
                "operator_key": chr(key),
                "class_id": class_id,
                "short_code": short_code,
                "operator_label": display_name,
                "sequence_in_class": sequence,
                "camera_index": camera_index,
                "camera_identity": camera_identity_after_open,
                "width": int(frame.shape[1]),
                "height": int(frame.shape[0]),
                "relative_path": image_path.relative_to(session_dir).as_posix(),
                "sha256": sha256_file(image_path),
                "quality": metrics,
                "truth_boundary": "OPERATOR_LABELLED_LAPTOP_CAPTURE_NOT_HOLDOUT",
            }
            append_jsonl(manifest_path, record)
            warning_text = ",".join(metrics["warnings"]) if metrics["warnings"] else "quality OK"
            last_status = f"SAVED {short_code} #{sequence} - {warning_text}"
            print(f"[已保存] {display_name} #{sequence}: {image_path}")
            print(
                "         "
                f"blur={metrics['laplacian_variance']} "
                f"brightness={metrics['brightness_mean']} "
                f"warnings={metrics['warnings']}"
            )

        def on_key(event: tk.Event[Any]) -> None:
            char = (event.char or "").lower()
            if char == "q" or event.keysym == "Escape":
                close_window("OPERATOR_EXIT")
                return
            if char and ord(char) in LABELS:
                save_current_frame(ord(char))

        def update_preview() -> None:
            nonlocal latest_frame, last_status
            if root is None or capture is None:
                return
            ok, frame = capture.read()
            if not ok or frame is None:
                last_status = "FRAME READ FAILED - waiting"
                root.after(80, update_preview)
                return
            latest_frame = frame
            shown = draw_overlay(
                frame,
                camera_label=f"{camera_identity['name']} ({camera_identity['vid_pid']})",
                saved_counts=saved_counts,
                status=last_status,
            )
            target_width = max(preview.winfo_width(), 960)
            target_height = max(preview.winfo_height(), 540)
            scale = min(target_width / shown.shape[1], target_height / shown.shape[0])
            display_size = (
                max(1, int(shown.shape[1] * scale)),
                max(1, int(shown.shape[0] * scale)),
            )
            shown_small = cv2.resize(shown, display_size, interpolation=cv2.INTER_AREA)
            shown_rgb = cv2.cvtColor(shown_small, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(shown_rgb)
            tk_image = ImageTk.PhotoImage(image=pil_image)
            preview.configure(image=tk_image)
            preview.image = tk_image
            root.after(30, update_preview)

        root.bind_all("<Key>", on_key)
        root.protocol("WM_DELETE_WINDOW", lambda: close_window("WINDOW_CLOSED"))
        root.after(10, update_preview)
        root.after(150, root.focus_force)
        root.mainloop()
        if exit_reason == "UNKNOWN":
            exit_reason = "WINDOW_LOOP_ENDED"
        root.destroy()
    except KeyboardInterrupt:
        exit_reason = "KEYBOARD_INTERRUPT"
        return_code = 130
    except Exception as exc:
        exit_reason = f"ERROR:{type(exc).__name__}"
        print(f"[错误] {exc}")
        return_code = 1
    else:
        return_code = 0
    finally:
        if capture is not None:
            capture.release()
        if root is not None:
            try:
                if root.winfo_exists():
                    root.destroy()
            except tk.TclError:
                pass
        completion = {
            "schema_version": "rootscope.laptop-card-capture-completion.v1",
            "completed_at_utc": utc_now(),
            "exit_reason": exit_reason,
            "saved_counts": saved_counts,
            "total_saved": sum(saved_counts.values()),
            "manifest": manifest_path.name,
        }
        (session_dir / "completion.json").write_text(
            json.dumps(completion, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print("=" * 72)
        print(f"采集结束，共保存 {completion['total_saved']} 张：{saved_counts}")
        print(f"目录：{session_dir}")
        print("=" * 72)

    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
