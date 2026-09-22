"""Planning for regular grids used by STL reconstruction."""

from __future__ import annotations

import numpy as np

from lattice_studio.engine.implicit.field import ImplicitBody
from lattice_studio.engine.implicit.surface_extraction import grid_definition as extraction_grid_definition
from lattice_studio.domain.cell_map import CellMap
from lattice_studio.domain.parameters import ExportSpacingMode
from lattice_studio.engine.contracts import ExportGridEstimate

def export_spacing_limit_labels(
    spacing_mm: tuple[float, float, float] | np.ndarray,
    cell_map: CellMap,
    tolerance_mm: float,
    minimum_feature_mm: float,
    spacing_mode: ExportSpacingMode = "recommended",
) -> tuple[str, str, str]:
    """Identify the constraint that produced each export-grid spacing."""

    if spacing_mode == "exact":
        return ("用户指定 STL 间距",) * 3
    if spacing_mode != "recommended":
        raise ValueError("spacing_mode must be 'recommended' or 'exact'")

    spacing = np.asarray(spacing_mm, dtype=np.float64)
    periods = np.asarray(cell_map.spacing_mm, dtype=np.float64)
    tolerance = float(tolerance_mm)
    feature_limit = float(minimum_feature_mm) / 3.0
    period_limits = periods / 16.0
    global_period_limit = float(np.min(period_limits))
    labels: list[str] = []
    for axis, actual in enumerate(spacing):
        axis_candidates = (
            (tolerance, "STL 容差"),
            (feature_limit, "最小特征厚度 / 3"),
            (period_limits[axis], f"{'XYZ'[axis]} 晶胞尺寸 / 16"),
        )
        matched = [
            label
            for value, label in axis_candidates
            if np.isclose(actual, value, rtol=1.0e-6, atol=1.0e-9)
        ]
        if (
            not np.isclose(
                actual,
                period_limits[axis],
                rtol=1.0e-6,
                atol=1.0e-9,
            )
            and np.isclose(
                actual,
                global_period_limit,
                rtol=1.0e-6,
                atol=1.0e-9,
            )
        ):
            matched.append("最小晶胞尺寸 / 16")
        labels.append(" + ".join(matched) if matched else "提取器统一间距")
    return labels[0], labels[1], labels[2]


def recommend_export_spacing(
    cell_map,
    tolerance_mm: float,
    wall_thickness_mm: float,
) -> np.ndarray:
    """Choose a conservative export grid spacing from physical dimensions."""

    tolerance = float(tolerance_mm)
    wall = float(wall_thickness_mm)
    if tolerance <= 0.0 or not np.isfinite(tolerance):
        raise ValueError("tolerance_mm must be finite and positive")
    if wall <= 0.0 or not np.isfinite(wall):
        raise ValueError("wall_thickness_mm must be finite and positive")
    cell_spacing = np.asarray(cell_map.spacing_mm, dtype=np.float64)
    return np.minimum(
        np.minimum(np.full(3, tolerance, dtype=np.float64), wall / 3.0),
        cell_spacing / 16.0,
    )


def resolve_export_spacing(
    cell_map: CellMap,
    tolerance_mm: float,
    wall_thickness_mm: float,
    spacing_mode: ExportSpacingMode = "recommended",
) -> np.ndarray:
    """Resolve either a conservative recommendation or an exact user spacing."""

    if spacing_mode == "recommended":
        return recommend_export_spacing(
            cell_map,
            tolerance_mm,
            wall_thickness_mm,
        )
    if spacing_mode != "exact":
        raise ValueError("spacing_mode must be 'recommended' or 'exact'")
    spacing = float(tolerance_mm)
    if spacing <= 0.0 or not np.isfinite(spacing):
        raise ValueError("tolerance_mm must be finite and positive")
    return np.full(3, spacing, dtype=np.float64)


def estimate_export_grid(
    body: ImplicitBody,
    cell_map: CellMap,
    tolerance_mm: float,
    wall_thickness_mm: float,
    spacing_mode: ExportSpacingMode = "recommended",
    enforce_feature_limits: bool = True,
) -> ExportGridEstimate:
    """Calculate the exact padded STL sampling grid without allocating it."""

    if enforce_feature_limits:
        spacing = resolve_export_spacing(
            cell_map,
            tolerance_mm,
            wall_thickness_mm,
            spacing_mode,
        )
        spacing_limits = export_spacing_limit_labels(
            spacing,
            cell_map,
            tolerance_mm,
            wall_thickness_mm,
            spacing_mode,
        )
    else:
        spacing = np.full(3, float(tolerance_mm), dtype=np.float64)
        spacing_limits = ("设计域 STL 容差",) * 3
    _origin, shape, _axes = extraction_grid_definition(
        body,
        spacing,
        np.asarray(cell_map.bounds[0], dtype=np.float64),
    )
    return ExportGridEstimate(
        spacing_mm=tuple(float(value) for value in spacing),
        grid_shape=shape,
        total_voxels=int(np.prod(shape, dtype=np.int64)),
        spacing_limits=spacing_limits,
    )


