"""Deterministic print/booth-domain augmentation for candidate training only."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from io import BytesIO
from typing import Any

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter


@dataclass(frozen=True)
class DomainRecipe:
    seed: int
    yellow_strength: float
    exposure: float
    gamma: float
    blur_radius: float
    perspective_fraction: float
    moire_amplitude: float
    moire_period_px: float
    jpeg_quality: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def sample_recipe(seed: int) -> DomainRecipe:
    rng = np.random.default_rng(seed)
    return DomainRecipe(
        seed=int(seed),
        yellow_strength=float(rng.uniform(0.08, 0.34)),
        exposure=float(rng.uniform(0.78, 1.20)),
        gamma=float(rng.uniform(0.82, 1.24)),
        blur_radius=float(rng.uniform(0.0, 1.8)),
        perspective_fraction=float(rng.uniform(0.0, 0.055)),
        moire_amplitude=float(rng.uniform(0.0, 11.0)),
        moire_period_px=float(rng.uniform(3.5, 10.0)),
        jpeg_quality=int(rng.integers(58, 94)),
    )


def _perspective_coefficients(
    output_points: list[tuple[float, float]], input_points: list[tuple[float, float]]
) -> tuple[float, ...]:
    matrix, vector = [], []
    for (x, y), (u, v) in zip(output_points, input_points):
        matrix.extend([[x, y, 1, 0, 0, 0, -u * x, -u * y],
                       [0, 0, 0, x, y, 1, -v * x, -v * y]])
        vector.extend([u, v])
    return tuple(np.linalg.solve(np.asarray(matrix), np.asarray(vector)).tolist())


def augment_image(image: Image.Image, recipe: DomainRecipe) -> Image.Image:
    """Apply reproducible yellow light, print, blur, perspective and JPEG effects."""

    rgb = image.convert("RGB")
    width, height = rgb.size
    jitter = recipe.perspective_fraction * min(width, height)
    rng = np.random.default_rng(recipe.seed)
    corners = [(float(rng.uniform(0, jitter)), float(rng.uniform(0, jitter))),
               (width - 1 - float(rng.uniform(0, jitter)), float(rng.uniform(0, jitter))),
               (width - 1 - float(rng.uniform(0, jitter)), height - 1 - float(rng.uniform(0, jitter))),
               (float(rng.uniform(0, jitter)), height - 1 - float(rng.uniform(0, jitter)))]
    output = [(0.0, 0.0), (width - 1.0, 0.0), (width - 1.0, height - 1.0), (0.0, height - 1.0)]
    coeffs = _perspective_coefficients(output, corners)
    rgb = rgb.transform((width, height), Image.Transform.PERSPECTIVE, coeffs, Image.Resampling.BICUBIC)
    values = np.asarray(rgb, dtype=np.float32)
    yellow = recipe.yellow_strength
    values *= np.asarray([1.0 + 0.34 * yellow, 1.0 + 0.12 * yellow, 1.0 - 0.72 * yellow])
    values *= recipe.exposure
    values = 255.0 * np.power(np.clip(values / 255.0, 0, 1), recipe.gamma)
    yy, xx = np.indices(values.shape[:2])
    angle = float(rng.uniform(0, np.pi))
    wave = np.sin((xx * np.cos(angle) + yy * np.sin(angle)) * 2 * np.pi / recipe.moire_period_px)
    values += wave[..., None] * recipe.moire_amplitude
    values += rng.normal(0, 1.2, size=values.shape)
    rgb = Image.fromarray(np.clip(values, 0, 255).astype(np.uint8), "RGB")
    if recipe.blur_radius > 0.05:
        rgb = rgb.filter(ImageFilter.GaussianBlur(recipe.blur_radius))
    rgb = ImageEnhance.Contrast(rgb).enhance(float(rng.uniform(0.86, 1.16)))
    buffer = BytesIO()
    rgb.save(buffer, format="JPEG", quality=recipe.jpeg_quality, subsampling=2)
    buffer.seek(0)
    return Image.open(buffer).convert("RGB").copy()
