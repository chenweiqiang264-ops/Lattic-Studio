"""Gradient lattice design parameters and invariants."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np


GradientAxis = Literal["U", "V", "W"]
GradientMode = Literal["linear", "power", "sigmoid", "layered"]
WallThicknessMethod = Literal["gradient_normalized", "field_threshold"]
GradientResolutionStrategy = Literal["automatic", "matlab_def"]

GRADIENT_MODES: tuple[GradientMode, ...] = (
    "linear",
    "power",
    "sigmoid",
    "layered",
)
WALL_THICKNESS_METHODS: tuple[WallThicknessMethod, ...] = (
    "gradient_normalized",
    "field_threshold",
)
GRADIENT_RESOLUTION_STRATEGIES: tuple[GradientResolutionStrategy, ...] = (
    "automatic",
    "matlab_def",
)


@dataclass(frozen=True)
class GradientControls:
    """Physical controls for a single TPMS gradient field."""

    enabled: bool = False
    axis: GradientAxis = "W"
    mode: GradientMode = "linear"
    power: float = 5.0
    layers: int = 5
    sigmoid_sharpness: float = 10.0
    use_thickness_gradient: bool = True
    thickness_soft_mm: float | None = None
    thickness_stiff_mm: float | None = None
    use_offset_gradient: bool = True
    offset_soft: float | None = None
    offset_stiff: float | None = None
    wall_thickness_method: WallThicknessMethod = "gradient_normalized"

    def validate(self) -> None:
        if self.axis not in ("U", "V", "W"):
            raise ValueError("gradient axis must be U, V, or W")
        if self.mode not in GRADIENT_MODES:
            raise ValueError(f"unsupported gradient mode: {self.mode}")
        if self.wall_thickness_method not in WALL_THICKNESS_METHODS:
            raise ValueError(
                f"unsupported wall-thickness method: {self.wall_thickness_method}"
            )
        if not np.isfinite(self.power) or self.power <= 0.0:
            raise ValueError("gradient power must be finite and positive")
        if int(self.layers) != self.layers or self.layers < 2:
            raise ValueError("gradient layers must be an integer of at least 2")
        if not np.isfinite(self.sigmoid_sharpness) or self.sigmoid_sharpness <= 0.0:
            raise ValueError("gradient sigmoid sharpness must be finite and positive")
        for value, name in (
            (self.thickness_soft_mm, "thickness_soft_mm"),
            (self.thickness_stiff_mm, "thickness_stiff_mm"),
        ):
            if value is not None and (
                not np.isfinite(value) or float(value) <= 0.0
            ):
                raise ValueError(f"{name} must be finite and positive or None")
        for value, name in (
            (self.offset_soft, "offset_soft"),
            (self.offset_stiff, "offset_stiff"),
        ):
            if value is not None and not np.isfinite(value):
                raise ValueError(f"{name} must be finite or None")


__all__ = [
    "GRADIENT_MODES",
    "GRADIENT_RESOLUTION_STRATEGIES",
    "WALL_THICKNESS_METHODS",
    "GradientAxis",
    "GradientControls",
    "GradientMode",
    "GradientResolutionStrategy",
    "WallThicknessMethod",
]

