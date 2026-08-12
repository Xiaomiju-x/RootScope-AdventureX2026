"""Camera source contracts with explicit provenance and lazy hardware access."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
from PIL import Image, ImageOps


@dataclass(frozen=True)
class FramePacket:
    image: np.ndarray
    frame_id: int
    captured_monotonic: float
    source: str
    provenance: str

    def age_seconds(self, now: float | None = None) -> float:
        return max(0.0, (now if now is not None else time.monotonic()) - self.captured_monotonic)


class FrameSource(Protocol):
    def capture(self) -> FramePacket: ...

    def close(self) -> None: ...


class ImageFileSource:
    """Deterministic fixture source for local tests and replay."""

    def __init__(self, path: str | Path, provenance: str = "fixture_file") -> None:
        self.path = Path(path).resolve()
        self.provenance = provenance
        self._frame_id = 0

    def capture(self) -> FramePacket:
        with Image.open(self.path) as raw:
            image = np.asarray(ImageOps.exif_transpose(raw).convert("RGB"))
        self._frame_id += 1
        return FramePacket(
            image=image,
            frame_id=self._frame_id,
            captured_monotonic=time.monotonic(),
            source=str(self.path),
            provenance=self.provenance,
        )

    def close(self) -> None:
        return None


class OpenCVCameraSource:
    """Optional UVC source; importing this module never opens a device."""

    def __init__(self, index: int = 0, width: int | None = None, height: int | None = None) -> None:
        try:
            import cv2  # type: ignore
        except ImportError as exc:
            raise RuntimeError("OpenCV is required for live UVC capture") from exc
        self._cv2 = cv2
        self._capture = cv2.VideoCapture(index)
        if not self._capture.isOpened():
            self._capture.release()
            raise RuntimeError(f"unable to open UVC camera index {index}")
        if width:
            self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        if height:
            self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self._frame_id = 0
        self._source = f"uvc:{index}"

    def capture(self) -> FramePacket:
        ok, bgr = self._capture.read()
        if not ok or bgr is None:
            raise RuntimeError("UVC capture failed")
        self._frame_id += 1
        rgb = self._cv2.cvtColor(bgr, self._cv2.COLOR_BGR2RGB)
        return FramePacket(
            image=rgb,
            frame_id=self._frame_id,
            captured_monotonic=time.monotonic(),
            source=self._source,
            provenance="live_uvc",
        )

    def close(self) -> None:
        self._capture.release()
