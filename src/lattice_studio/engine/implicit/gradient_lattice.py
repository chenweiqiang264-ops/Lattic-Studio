"""Gradient TPMS evaluation on a continuous Cell Map.

The MATLAB prototype samples a TPMS field in a regular coordinate system and
changes its wall threshold and level-set offset along one coordinate.  This
module keeps that model independent of a particular UI or mesh extractor:
world points are mapped to the Cell Map's UVW frame, the gradient profile is
evaluated there, and the resulting signed field is returned with negative
values denoting solid material.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from lattice_studio.domain.cell_map import CellMapFrame
from lattice_studio.domain.gradient import (
    GRADIENT_MODES,
    GRADIENT_RESOLUTION_STRATEGIES,
    WALL_THICKNESS_METHODS,
    GradientAxis,
    GradientControls,
    GradientMode,
    GradientResolutionStrategy,
    WallThicknessMethod,
)
from lattice_studio.engine.implicit.tpms_compute import (
    TPMSFieldSpec,
    TPMSGradientSpec,
    get_default_tpms_backend,
)


def matlab_gradient_voxel_size(
    cell_size_mm,
    def_per_half_cell: int = 28,
) -> float:
    """Return the MATLAB prototype's uniform sample spacing in millimetres.

    The prototype defines ``A = L / 2`` as the half-cell length and samples
    it with ``Def`` intervals, therefore ``D = L / (2 * Def)``.  The current
    evaluator accepts independent U/V/W periods but uses a scalar regular
    grid, so the smallest physical period is used conservatively.
    """

    periods = np.asarray(cell_size_mm, dtype=np.float64)
    if periods.ndim == 0:
        periods = np.repeat(periods, 3)
    if (
        periods.shape != (3,)
        or not np.isfinite(periods).all()
        or np.any(periods <= 0.0)
    ):
        raise ValueError("cell_size_mm must contain three finite positive values")
    if int(def_per_half_cell) != def_per_half_cell or int(def_per_half_cell) < 1:
        raise ValueError("def_per_half_cell must be a positive integer")
    return float(np.min(periods) / (2.0 * int(def_per_half_cell)))


def _range_pair(values, name: str) -> tuple[float, float]:
    result = np.asarray(values, dtype=np.float64)
    if result.shape != (2,) or not np.isfinite(result).all() or result[0] >= result[1]:
        raise ValueError(f"{name} must contain two increasing finite values")
    return float(result[0]), float(result[1])


def gradient_profile(
    coordinate: np.ndarray,
    mode: GradientMode,
    *,
    power: float = 5.0,
    layers: int = 5,
    sigmoid_sharpness: float = 10.0,
) -> np.ndarray:
    """Return a bounded profile in ``[0, 1]`` using the MATLAB definitions."""

    values = np.asarray(coordinate, dtype=np.float32)
    if not np.isfinite(values).all():
        raise ValueError("gradient coordinate must be finite")
    if mode not in GRADIENT_MODES:
        raise ValueError(f"unsupported gradient mode: {mode}")
    if power <= 0.0 or not np.isfinite(power):
        raise ValueError("power must be finite and positive")
    if int(layers) != layers or layers < 2:
        raise ValueError("layers must be an integer of at least 2")
    if sigmoid_sharpness <= 0.0 or not np.isfinite(sigmoid_sharpness):
        raise ValueError("sigmoid_sharpness must be finite and positive")

    u = np.clip(values, 0.0, 1.0)
    if mode == "linear":
        result = u
    elif mode == "power":
        result = np.power(u, np.float32(power))
    elif mode == "sigmoid":
        argument = np.clip(
            np.float32(sigmoid_sharpness) * (u - np.float32(0.5)),
            -60.0,
            60.0,
        )
        result = 1.0 / (1.0 + np.exp(-argument))
    else:
        result = np.floor(u * np.float32(layers)) / np.float32(layers - 1)
        result = np.where(u >= 1.0, 1.0, result)
    return np.clip(np.asarray(result, dtype=np.float32), 0.0, 1.0)


def _gradient_profile_derivative(
    coordinate: np.ndarray,
    controls: GradientControls,
) -> np.ndarray:
    """Return the profile derivative for the smooth gradient modes."""

    u = np.clip(np.asarray(coordinate, dtype=np.float32), 0.0, 1.0)
    if controls.mode == "linear":
        return np.ones_like(u)
    if controls.mode == "power":
        return np.float32(controls.power) * np.power(
            u,
            np.float32(max(controls.power - 1.0, 0.0)),
        )
    if controls.mode == "sigmoid":
        profile = gradient_profile(
            u,
            controls.mode,
            power=controls.power,
            layers=controls.layers,
            sigmoid_sharpness=controls.sigmoid_sharpness,
        )
        return np.float32(controls.sigmoid_sharpness) * profile * (1.0 - profile)
    # A layered profile is intentionally piecewise constant.  Treat its
    # derivative as zero for the optional conservative offset-gradient term.
    return np.zeros_like(u)


def evaluate_gradient_tpms_field(
    points: np.ndarray,
    field_spec: TPMSFieldSpec,
    frame: CellMapFrame,
    gradient: GradientControls,
    gradient_bounds_mm: tuple[float, float],
    *,
    max_tile_points: int | None = 250_000,
) -> np.ndarray:
    """Evaluate a gradient TPMS shell field at world-space points.

    ``gradient_bounds_mm`` is the design-domain projection range on the chosen
    local UVW axis.  It is deliberately independent from the Cell Map padding,
    so changing the number of complete cells does not change the physical
    gradient definition.
    """

    field_spec.validate()
    gradient.validate()
    bounds = _range_pair(gradient_bounds_mm, "gradient_bounds_mm")
    values = np.asarray(points)
    if values.ndim != 2 or values.shape[1] != 3 or not np.isfinite(values).all():
        raise ValueError("points must have shape (N, 3) and contain finite values")
    if max_tile_points is not None and max_tile_points < 1:
        raise ValueError("max_tile_points must be positive or None")

    backend = get_default_tpms_backend()
    if not gradient.enabled:
        return backend.evaluate_shell(values, field_spec, max_tile_points)
    return backend.evaluate_gradient_shell(
        values,
        field_spec,
        make_tpms_gradient_spec(gradient, bounds),
        max_tile_points,
    )


def make_tpms_gradient_spec(
    controls: GradientControls,
    gradient_bounds_mm: tuple[float, float],
) -> TPMSGradientSpec:
    """Translate user-facing gradient controls into the compute contract."""

    controls.validate()
    bounds = _range_pair(gradient_bounds_mm, "gradient_bounds_mm")
    return TPMSGradientSpec(
        axis_index="UVW".index(controls.axis),
        coordinate_min_mm=bounds[0],
        coordinate_max_mm=bounds[1],
        mode=controls.mode,
        power=controls.power,
        layers=controls.layers,
        sigmoid_sharpness=controls.sigmoid_sharpness,
        use_thickness_gradient=controls.use_thickness_gradient,
        thickness_soft_mm=controls.thickness_soft_mm,
        thickness_stiff_mm=controls.thickness_stiff_mm,
        use_offset_gradient=controls.use_offset_gradient,
        offset_soft=controls.offset_soft,
        offset_stiff=controls.offset_stiff,
        wall_thickness_method=controls.wall_thickness_method,
    )


def gradient_projection_range(
    points: np.ndarray,
    frame: CellMapFrame,
    axis: GradientAxis,
) -> tuple[float, float]:
    """Project design-domain points onto one Cell Map UVW axis."""

    values = np.asarray(points, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 3 or len(values) == 0:
        raise ValueError("points must have non-empty shape (N, 3)")
    if axis not in ("U", "V", "W"):
        raise ValueError("axis must be U, V, or W")
    local = frame.to_local(values)[:, "UVW".index(axis)]
    lower, upper = float(np.min(local)), float(np.max(local))
    if upper - lower <= 1.0e-9:
        raise ValueError("design-domain projection has zero gradient span")
    return lower, upper


def estimate_layer_density(
    values: np.ndarray,
    coordinates: np.ndarray,
    layers: int = 12,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate solid fraction per gradient layer for the optional beta view."""

    field = np.asarray(values)
    axis_values = np.asarray(coordinates)
    if field.shape != axis_values.shape or field.size == 0:
        raise ValueError("values and coordinates must have the same non-empty shape")
    if int(layers) != layers or layers < 1:
        raise ValueError("layers must be a positive integer")
    lower, upper = float(np.min(axis_values)), float(np.max(axis_values))
    if upper <= lower:
        raise ValueError("coordinates must span a positive range")
    edges = np.linspace(lower, upper, int(layers) + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    density = np.empty(int(layers), dtype=np.float32)
    for index in range(int(layers)):
        mask = (
            (axis_values >= edges[index])
            & (
                (axis_values <= edges[index + 1])
                if index == int(layers) - 1
                else (axis_values < edges[index + 1])
            )
        )
        density[index] = np.mean(field[mask] <= 0.0) if np.any(mask) else 0.0
    return centers, density


__all__ = [
    "GRADIENT_MODES",
    "GRADIENT_RESOLUTION_STRATEGIES",
    "WALL_THICKNESS_METHODS",
    "GradientAxis",
    "GradientControls",
    "GradientMode",
    "GradientResolutionStrategy",
    "WallThicknessMethod",
    "estimate_layer_density",
    "evaluate_gradient_tpms_field",
    "gradient_profile",
    "gradient_projection_range",
    "make_tpms_gradient_spec",
    "matlab_gradient_voxel_size",
]
