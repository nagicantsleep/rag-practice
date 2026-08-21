"""Tiny dependency-free P3 PPM reader for the controlled multimodal lab."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


RGB = tuple[int, int, int]

PALETTE: dict[str, RGB] = {
    "white": (255, 255, 255),
    "gray": (220, 220, 220),
    "red": (230, 30, 30),
    "green": (30, 200, 60),
    "blue": (30, 90, 230),
    "yellow": (240, 210, 30),
    "black": (20, 20, 20),
}


@dataclass(frozen=True)
class RasterImage:
    width: int
    height: int
    max_value: int
    pixels: tuple[RGB, ...]

    def color_count(self, color: str) -> int:
        target = PALETTE[color]
        return sum(pixel == target for pixel in self.pixels)

    def dominant_quadrant(self, color: str) -> str | None:
        target = PALETTE[color]
        half_width = self.width // 2
        half_height = self.height // 2
        counts = {"upper-left": 0, "upper-right": 0, "lower-left": 0, "lower-right": 0}
        for index, pixel in enumerate(self.pixels):
            if pixel != target:
                continue
            y, x = divmod(index, self.width)
            vertical = "upper" if y < half_height else "lower"
            horizontal = "left" if x < half_width else "right"
            counts[f"{vertical}-{horizontal}"] += 1
        best = max(counts, key=counts.get)
        return best if counts[best] > 0 else None


def read_p3_ppm(path: Path) -> RasterImage:
    tokens: list[str] = []
    for line in path.read_text().splitlines():
        tokens.extend(line.split("#", 1)[0].split())
    if not tokens or tokens[0] != "P3":
        raise ValueError(f"{path} is not a P3 PPM")
    if len(tokens) < 4:
        raise ValueError(f"{path} has an incomplete PPM header")
    width, height, max_value = map(int, tokens[1:4])
    values = [int(value) for value in tokens[4:]]
    expected = width * height * 3
    if len(values) != expected:
        raise ValueError(f"{path} expected {expected} channel values, got {len(values)}")
    if width <= 0 or height <= 0 or max_value <= 0:
        raise ValueError(f"{path} has invalid dimensions or max value")
    if any(value < 0 or value > max_value for value in values):
        raise ValueError(f"{path} contains out-of-range channel values")
    pixels = tuple((values[i], values[i + 1], values[i + 2]) for i in range(0, len(values), 3))
    return RasterImage(width=width, height=height, max_value=max_value, pixels=pixels)
