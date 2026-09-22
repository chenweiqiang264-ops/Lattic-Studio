"""Continuous regular Cell Map primitives shared by lattice families."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Literal

import numpy as np

__all__ = ["CellMap", "CellMapBoundaryMode", "CellMapFrame"]


CellMapBoundaryMode = Literal["fit_bounds", "complete_cells"]


def _positive_triplet(values, name: str) -> tuple[float, float, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 0:
        array = np.repeat(array, 3)
    if array.shape != (3,) or not np.isfinite(array).all() or np.any(array <= 0.0):
        raise ValueError(f"{name} must contain three finite positive values")
    return tuple(float(value) for value in array)


def _finite_triplet(values, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.shape != (3,) or not np.isfinite(array).all():
        raise ValueError(f"{name} must contain three finite values")
    return array


@dataclass(frozen=True)
class CellMapFrame:
    """A right-handed orthonormal UVW frame in world coordinates."""

    origin: np.ndarray
    axes: np.ndarray

    def __post_init__(self) -> None:
        origin = _finite_triplet(self.origin, "origin")
        axes = np.asarray(self.axes, dtype=np.float64)
        if axes.shape != (3, 3) or not np.isfinite(axes).all():
            raise ValueError("axes must be a finite 3 by 3 matrix")
        if not np.allclose(axes @ axes.T, np.eye(3), rtol=1e-10, atol=1e-10):
            raise ValueError("axes must form an orthonormal basis")
        if not np.isclose(np.linalg.det(axes), 1.0, rtol=1e-10, atol=1e-10):
            raise ValueError("axes must form a right-handed basis")
        object.__setattr__(self, "origin", origin)
        object.__setattr__(self, "axes", axes)

    @classmethod
    def world(cls, origin) -> "CellMapFrame":
        return cls(_finite_triplet(origin, "origin"), np.eye(3, dtype=np.float64))

    @classmethod
    def from_origin_axes(cls, origin, u_axis, v_axis) -> "CellMapFrame":
        origin_array = _finite_triplet(origin, "origin")
        u = _finite_triplet(u_axis, "u_axis")
        v = _finite_triplet(v_axis, "v_axis")
        u_norm = float(np.linalg.norm(u))
        if u_norm <= 1.0e-12:
            raise ValueError("u_axis must have non-zero length")
        u /= u_norm
        v -= np.dot(v, u) * u
        v_norm = float(np.linalg.norm(v))
        if v_norm <= 1.0e-12:
            raise ValueError("u_axis and v_axis must not be parallel")
        v /= v_norm
        w = np.cross(u, v)
        return cls(origin_array, np.vstack((u, v, w)))

    @classmethod
    def aligned_to_points_minimum(
        cls,
        points: np.ndarray,
        u_axis,
        v_axis,
    ) -> "CellMapFrame":
        """Place a UVW frame at the point set's frame-aligned minimum."""

        values = np.asarray(points, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != 3 or len(values) == 0:
            raise ValueError("points must have non-empty shape (N, 3)")
        if not np.isfinite(values).all():
            raise ValueError("points must contain only finite values")
        orientation = cls.from_origin_axes((0.0, 0.0, 0.0), u_axis, v_axis)
        minimum_local = np.min(values @ orientation.axes.T, axis=0)
        origin = minimum_local @ orientation.axes
        return cls(origin, orientation.axes)

    @property
    def u_axis(self) -> np.ndarray:
        return self.axes[0]

    @property
    def v_axis(self) -> np.ndarray:
        return self.axes[1]

    @property
    def w_axis(self) -> np.ndarray:
        return self.axes[2]

    @property
    def is_world_aligned(self) -> bool:
        return bool(np.allclose(self.axes, np.eye(3), rtol=0.0, atol=1.0e-12))

    def to_local(self, points: np.ndarray) -> np.ndarray:
        values = np.asarray(points, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != 3 or not np.isfinite(values).all():
            raise ValueError("points must have shape (N, 3) and finite values")
        return (values - self.origin) @ self.axes.T

    def to_world(self, local_points: np.ndarray) -> np.ndarray:
        values = np.asarray(local_points, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != 3 or not np.isfinite(values).all():
            raise ValueError("local_points must have shape (N, 3) and finite values")
        return self.origin + values @ self.axes


@dataclass(frozen=True)
class CellMap:
    """A regular complete-cell spatial parameterization."""

    bounds: np.ndarray
    spacing_mm: tuple[float, float, float]
    cell_counts: tuple[int, int, int]
    requested_spacing_mm: tuple[float, float, float] | None = None
    boundary_mode: CellMapBoundaryMode = "complete_cells"
    frame: CellMapFrame | None = None
    index_min: tuple[int, int, int] = (0, 0, 0)

    def __post_init__(self) -> None:
        bounds = np.asarray(self.bounds, dtype=np.float64)
        if bounds.shape != (2, 3) or not np.isfinite(bounds).all():
            raise ValueError("Cell Map bounds must have shape (2, 3) and finite values")
        if not np.all(bounds[0] < bounds[1]):
            raise ValueError("Cell Map bounds must have positive extents")
        spacing = _positive_triplet(self.spacing_mm, "spacing_mm")
        requested_spacing = _positive_triplet(
            self.requested_spacing_mm if self.requested_spacing_mm is not None else spacing,
            "requested_spacing_mm",
        )
        counts = tuple(int(value) for value in self.cell_counts)
        if len(counts) != 3 or any(value < 1 for value in counts):
            raise ValueError("cell_counts must contain three positive integers")
        if self.boundary_mode not in ("fit_bounds", "complete_cells"):
            raise ValueError("boundary_mode must be 'fit_bounds' or 'complete_cells'")
        index_min = tuple(int(value) for value in self.index_min)
        if len(index_min) != 3:
            raise ValueError("index_min must contain three integers")
        frame = self.frame or CellMapFrame.world(bounds[0])
        local_min = np.asarray(index_min, dtype=np.float64) * np.asarray(spacing)
        local_max = local_min + np.asarray(counts, dtype=np.float64) * np.asarray(spacing)
        corners = np.asarray(tuple(product(*zip(local_min, local_max))), dtype=np.float64)
        world_corners = frame.to_world(corners)
        expected_bounds = np.vstack((world_corners.min(axis=0), world_corners.max(axis=0)))
        if not np.allclose(bounds, expected_bounds, rtol=1e-9, atol=1e-9):
            raise ValueError("Cell Map bounds do not match its frame and cell-index range")
        object.__setattr__(self, "bounds", bounds)
        object.__setattr__(self, "spacing_mm", spacing)
        object.__setattr__(self, "cell_counts", counts)
        object.__setattr__(self, "requested_spacing_mm", requested_spacing)
        object.__setattr__(self, "frame", frame)
        object.__setattr__(self, "index_min", index_min)

    @classmethod
    def from_bounds(
        cls,
        bounds: np.ndarray,
        spacing_mm,
        boundary_mode: CellMapBoundaryMode = "complete_cells",
    ) -> "CellMap":
        """Build a uniform map that fits or contains the requested bounds."""

        requested = np.asarray(bounds, dtype=np.float64)
        if requested.shape != (2, 3) or not np.isfinite(requested).all():
            raise ValueError("requested bounds must have shape (2, 3) and finite values")
        if not np.all(requested[0] < requested[1]):
            raise ValueError("requested bounds must have positive extents")
        spacing = _positive_triplet(spacing_mm, "spacing_mm")
        if boundary_mode not in ("fit_bounds", "complete_cells"):
            raise ValueError("boundary_mode must be 'fit_bounds' or 'complete_cells'")
        extent = requested[1] - requested[0]
        if boundary_mode == "fit_bounds":
            counts = tuple(
                max(1, int(np.floor(extent[axis] / spacing[axis] + 0.5)))
                for axis in range(3)
            )
            actual_spacing = extent / np.asarray(counts, dtype=np.float64)
            map_bounds = requested.copy()
        else:
            counts = tuple(
                max(1, int(np.ceil(extent[axis] / spacing[axis] - 1.0e-12)))
                for axis in range(3)
            )
            actual_spacing = np.asarray(spacing, dtype=np.float64)
            padded_extent = actual_spacing * np.asarray(counts)
            center = requested.mean(axis=0)
            origin = center - 0.5 * padded_extent
            map_bounds = np.stack((origin, origin + padded_extent))
        return cls(
            bounds=map_bounds,
            spacing_mm=tuple(float(value) for value in actual_spacing),
            cell_counts=counts,
            requested_spacing_mm=spacing,
            boundary_mode=boundary_mode,
        )

    @classmethod
    def from_points(
        cls,
        points: np.ndarray,
        spacing_mm,
        frame: CellMapFrame,
    ) -> "CellMap":
        """Cover world-space points with complete cells aligned to ``frame``."""

        values = np.asarray(points, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != 3 or not np.isfinite(values).all():
            raise ValueError("points must have shape (N, 3) and finite values")
        if len(values) == 0:
            raise ValueError("points must not be empty")
        if not isinstance(frame, CellMapFrame):
            raise TypeError("frame must be a CellMapFrame")
        spacing = np.asarray(_positive_triplet(spacing_mm, "spacing_mm"))
        local = frame.to_local(values)
        lower = np.floor(local.min(axis=0) / spacing + 1.0e-12).astype(np.int64)
        upper = np.ceil(local.max(axis=0) / spacing - 1.0e-12).astype(np.int64)
        upper = np.maximum(upper, lower + 1)
        counts = upper - lower
        local_min = lower.astype(np.float64) * spacing
        local_max = upper.astype(np.float64) * spacing
        corners = np.asarray(tuple(product(*zip(local_min, local_max))), dtype=np.float64)
        world_corners = frame.to_world(corners)
        bounds = np.vstack((world_corners.min(axis=0), world_corners.max(axis=0)))
        return cls(
            bounds=bounds,
            spacing_mm=tuple(float(value) for value in spacing),
            cell_counts=tuple(int(value) for value in counts),
            requested_spacing_mm=tuple(float(value) for value in spacing),
            boundary_mode="complete_cells",
            frame=frame,
            index_min=tuple(int(value) for value in lower),
        )

    @classmethod
    def from_frame_counts(
        cls,
        frame: CellMapFrame,
        spacing_mm,
        cell_counts,
        *,
        index_min: tuple[int, int, int] = (0, 0, 0),
        boundary_mode: CellMapBoundaryMode = "complete_cells",
    ) -> "CellMap":
        """Create a map from an explicit UVW origin and total cell counts.

        This is the manual-count counterpart of :meth:`from_points`.  The
        count is the number of complete cells, while ``index_min`` determines
        which integer UVW index is placed at the supplied frame origin.
        """

        if not isinstance(frame, CellMapFrame):
            raise TypeError("frame must be a CellMapFrame")
        spacing = np.asarray(_positive_triplet(spacing_mm, "spacing_mm"))
        counts_array = np.asarray(cell_counts)
        if counts_array.shape != (3,) or not np.isfinite(counts_array).all():
            raise ValueError("cell_counts must contain three finite integers")
        counts = tuple(int(value) for value in counts_array)
        if any(value < 1 or float(value) != float(raw) for value, raw in zip(counts, counts_array)):
            raise ValueError("cell_counts must contain three positive integers")
        indices = tuple(int(value) for value in index_min)
        if len(indices) != 3 or any(float(value) != float(raw) for value, raw in zip(indices, index_min)):
            raise ValueError("index_min must contain three integers")
        local_min = np.asarray(indices, dtype=np.float64) * spacing
        local_max = local_min + np.asarray(counts, dtype=np.float64) * spacing
        corners = np.asarray(tuple(product(*zip(local_min, local_max))), dtype=np.float64)
        world_corners = frame.to_world(corners)
        bounds = np.vstack((world_corners.min(axis=0), world_corners.max(axis=0)))
        return cls(
            bounds=bounds,
            spacing_mm=tuple(float(value) for value in spacing),
            cell_counts=counts,
            requested_spacing_mm=tuple(float(value) for value in spacing),
            boundary_mode=boundary_mode,
            frame=frame,
            index_min=indices,
        )

    @property
    def extent_mm(self) -> np.ndarray:
        return np.asarray(self.spacing_mm) * np.asarray(self.cell_counts)

    @property
    def grid_shape(self) -> tuple[int, int, int]:
        """Number of cell-map vertices, excluding any SDF sampling halo."""

        return tuple(value + 1 for value in self.cell_counts)

    def axes(self) -> list[np.ndarray]:
        if self.frame is None or not self.frame.is_world_aligned:
            raise RuntimeError("rotated Cell Maps use wireframe_segments()")
        local_axes = self.local_grid_coordinates()
        return [
            self.frame.origin[axis] + local_axes[axis]
            for axis in range(3)
        ]

    def local_grid_coordinates(self) -> list[np.ndarray]:
        return [
            (
                self.index_min[axis]
                + np.arange(self.cell_counts[axis] + 1, dtype=np.float64)
            )
            * self.spacing_mm[axis]
            for axis in range(3)
        ]

    def wireframe_segments(self) -> np.ndarray:
        """Return all Cell Map grid lines as world-space endpoint pairs."""

        axes = self.local_grid_coordinates()
        segments: list[tuple[np.ndarray, np.ndarray]] = []
        for u in axes[0]:
            for v in axes[1]:
                segments.append(
                    (np.array((u, v, axes[2][0])), np.array((u, v, axes[2][-1])))
                )
        for u in axes[0]:
            for w in axes[2]:
                segments.append(
                    (np.array((u, axes[1][0], w)), np.array((u, axes[1][-1], w)))
                )
        for v in axes[1]:
            for w in axes[2]:
                segments.append(
                    (np.array((axes[0][0], v, w)), np.array((axes[0][-1], v, w)))
                )
        local_endpoints = np.asarray(segments, dtype=np.float64).reshape((-1, 3))
        return self.frame.to_world(local_endpoints).reshape((-1, 2, 3))

    def to_cell_coordinates(self, points: np.ndarray) -> np.ndarray:
        """Map world points to cell coordinates where one period is one unit."""

        values = np.asarray(points, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != 3 or not np.isfinite(values).all():
            raise ValueError("points must have shape (N, 3) and finite values")
        return self.frame.to_local(values) / np.asarray(self.spacing_mm)

    def contains(self, points: np.ndarray, tolerance: float = 0.0) -> np.ndarray:
        values = np.asarray(points, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != 3:
            raise ValueError("points must have shape (N, 3)")
        margin = float(tolerance)
        if margin < 0.0 or not np.isfinite(margin):
            raise ValueError("tolerance must be finite and non-negative")
        local = self.frame.to_local(values)
        local_min = np.asarray(self.index_min) * np.asarray(self.spacing_mm)
        local_max = local_min + self.extent_mm
        return np.all((local >= local_min - margin) & (local <= local_max + margin), axis=1)
