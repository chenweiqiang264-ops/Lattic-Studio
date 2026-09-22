"""Shared regular-grid metadata for implicit surface reconstruction."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from lattice_studio.engine.implicit.field import ImplicitBody


@dataclass(frozen=True)
class ExtractionStatistics:
    """Counts reported by the selected regular-grid extraction strategy."""

    dense_grid_points: int
    evaluated_points: int
    total_bricks: int
    active_bricks: int
    skipped_bricks: int


def grid_definition(
    body: ImplicitBody,
    spacing: np.ndarray,
    grid_anchor_mm: tuple[float, float, float] | np.ndarray | None = None,
) -> tuple[np.ndarray, tuple[int, int, int], list[np.ndarray]]:
    """Create a padded regular grid aligned to an optional Cell Map origin."""

    if not isinstance(body, ImplicitBody):
        raise TypeError("body must be an ImplicitBody")
    steps = np.asarray(spacing, dtype=np.float64)
    if steps.ndim == 0:
        steps = np.repeat(steps, 3)
    if steps.shape != (3,) or not np.isfinite(steps).all() or np.any(steps <= 0.0):
        raise ValueError("spacing must contain three finite positive values")

    lower = np.asarray(body.bounds[0], dtype=np.float64) - steps
    upper = np.asarray(body.bounds[1], dtype=np.float64) + steps
    if grid_anchor_mm is None:
        origin = lower
    else:
        anchor = np.asarray(grid_anchor_mm, dtype=np.float64)
        if anchor.shape != (3,) or not np.isfinite(anchor).all():
            raise ValueError("grid_anchor_mm must contain three finite values")
        origin = anchor + np.floor((lower - anchor) / steps) * steps
        upper = anchor + np.ceil((upper - anchor) / steps) * steps

    shape = tuple(
        int(np.rint((upper[axis] - origin[axis]) / steps[axis])) + 1
        for axis in range(3)
    )
    axes = [
        origin[axis] + np.arange(shape[axis], dtype=np.float64) * steps[axis]
        for axis in range(3)
    ]
    return origin, shape, axes


__all__ = ["ExtractionStatistics", "grid_definition"]
