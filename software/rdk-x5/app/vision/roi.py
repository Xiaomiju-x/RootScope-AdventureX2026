"""Pixel-region contracts for the frozen RootScope camera view."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PixelROI:
    """A half-open pixel rectangle: ``[x:x+width, y:y+height]``."""

    name: str
    x: int
    y: int
    width: int
    height: int

    def validate_for(self, image_width: int, image_height: int) -> None:
        if not self.name:
            raise ValueError("ROI name must not be empty")
        if min(self.x, self.y) < 0 or min(self.width, self.height) <= 0:
            raise ValueError(f"invalid ROI geometry: {self}")
        if self.x + self.width > image_width or self.y + self.height > image_height:
            raise ValueError(
                f"ROI {self.name!r} exceeds frame {image_width}x{image_height}: {self}"
            )

    @property
    def area(self) -> int:
        return self.width * self.height

    def slices(self) -> tuple[slice, slice]:
        return (slice(self.y, self.y + self.height), slice(self.x, self.x + self.width))
