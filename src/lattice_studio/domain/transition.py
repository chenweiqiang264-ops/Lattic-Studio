"""Transition design parameters and invariants."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np


TransitionWeightKind = Literal[
    "automatic",
    "linear",
    "smoothstep",
    "smootherstep",
    "sigmoid",
    "cosine",
]


@dataclass(frozen=True)
class TransitionSpec:
    """Physical and advanced controls for a compact spatial Ramp."""

    width_mm: float
    center_offset_mm: float = 0.0
    weight_kind: TransitionWeightKind = "automatic"
    sigmoid_sharpness: float = 1.0
    first_on_negative_side: bool = True
    minimum_feature_mm: float | None = None
    automatic_registration: bool = True
    topology_correction: bool = True

    def validate(self) -> None:
        values = (self.width_mm, self.center_offset_mm, self.sigmoid_sharpness)
        if not all(np.isfinite(value) for value in values):
            raise ValueError("transition parameters must be finite")
        if self.width_mm <= 0.0:
            raise ValueError("width_mm must be positive")
        if self.sigmoid_sharpness <= 0.0:
            raise ValueError("sigmoid_sharpness must be positive")
        if self.weight_kind not in (
            "automatic",
            "linear",
            "smoothstep",
            "smootherstep",
            "sigmoid",
            "cosine",
        ):
            raise ValueError("unsupported transition weight kind")
        if self.minimum_feature_mm is not None and (
            not np.isfinite(self.minimum_feature_mm)
            or self.minimum_feature_mm <= 0.0
        ):
            raise ValueError("minimum_feature_mm must be positive or None")


__all__ = ["TransitionSpec", "TransitionWeightKind"]
