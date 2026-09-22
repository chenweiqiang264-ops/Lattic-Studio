"""Layer-wise contour extraction for implicit fields.

The module deliberately keeps contours as 3-D polylines.  It does not create
triangles and therefore is suitable for interactive inspection of a field
without coupling the inspection view to STL reconstruction.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Literal

import numpy as np
from skimage import measure

from .field import ImplicitBody, SampledImplicitField


ContourFieldSource = Literal["render", "stl_reconstruction", "authoritative"]
LayerAxis = Literal["X", "Y", "Z"]


@dataclass(frozen=True)
class LayerContour:
    """One contour polyline in world coordinates."""

    points: np.ndarray
    closed: bool

    def __post_init__(self) -> None:
        points = np.ascontiguousarray(self.points, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 3 or len(points) < 2:
            raise ValueError("contour points must have shape (N, 3), N >= 2")
        if not np.isfinite(points).all():
            raise ValueError("contour points must be finite")
        object.__setattr__(self, "points", points)


@dataclass(frozen=True)
class LayerContourResult:
    """Contours extracted from one layer of one field."""

    source: ContourFieldSource
    axis: LayerAxis
    layer_position_mm: float
    level: float
    contours: tuple[LayerContour, ...]
    plane_origin_mm: np.ndarray
    plane_spacing_mm: tuple[float, float]
    grid_shape: tuple[int, int]

    @property
    def point_count(self) -> int:
        return sum(len(contour.points) for contour in self.contours)


class LayerContourCache:
    """Small LRU cache for layer results and on-demand 2-D field samples."""

    def __init__(self, max_entries: int = 256) -> None:
        if int(max_entries) < 1:
            raise ValueError("max_entries must be positive")
        self.max_entries = int(max_entries)
        self._entries: OrderedDict[object, LayerContourResult] = OrderedDict()

    def get(self, key: object) -> LayerContourResult | None:
        result = self._entries.get(key)
        if result is not None:
            self._entries.move_to_end(key)
        return result

    def put(self, key: object, result: LayerContourResult) -> None:
        self._entries[key] = result
        self._entries.move_to_end(key)
        while len(self._entries) > self.max_entries:
            self._entries.popitem(last=False)

    def clear(self) -> None:
        self._entries.clear()

    def __len__(self) -> int:
        return len(self._entries)


def axis_index(axis: LayerAxis) -> int:
    value = str(axis).upper()
    if value not in {"X", "Y", "Z"}:
        raise ValueError("axis must be X, Y or Z")
    return "XYZ".index(value)


def available_layer_positions(
    bounds: np.ndarray,
    axis: LayerAxis,
    spacing_mm: float | tuple[float, float, float],
) -> np.ndarray:
    """Return evenly spaced layer coordinates including both bounds."""

    values = np.asarray(bounds, dtype=np.float64)
    if values.shape != (2, 3) or not np.isfinite(values).all() or not np.all(values[0] <= values[1]):
        raise ValueError("bounds must have shape (2, 3) and ordered finite values")
    index = axis_index(axis)
    spacing = np.asarray(spacing_mm, dtype=np.float64)
    if spacing.ndim == 0:
        spacing = np.repeat(spacing, 3)
    if spacing.shape != (3,) or not np.isfinite(spacing).all() or np.any(spacing <= 0):
        raise ValueError("spacing_mm must be positive")
    start, stop = values[:, index]
    positions = np.arange(start, stop + spacing[index] * 0.5, spacing[index])
    if len(positions) == 0 or positions[-1] < stop - spacing[index] * 1.0e-7:
        positions = np.append(positions, stop)
    else:
        positions[-1] = min(positions[-1], stop)
    return positions


def resolve_layer_spacing(
    spacing_mm: float | tuple[float, float, float] | np.ndarray,
    axis: LayerAxis,
    fixed_layer_height_mm: float | None = None,
) -> np.ndarray:
    """Resolve layer spacing while preserving the in-plane field spacing.

    The optional fixed height replaces only the component along ``axis``.
    This separates the distance between inspected layers from the sampling
    resolution used to extract each 2-D contour.
    """

    index = axis_index(axis)
    # This value commonly comes directly from SampledImplicitField.spacing.
    # Always own a copy: replacing the selected layer-axis interval must not
    # mutate that field's physical grid spacing and rescale it in the viewer.
    spacing = np.array(spacing_mm, dtype=np.float64, copy=True)
    if spacing.ndim == 0:
        spacing = np.repeat(spacing, 3)
    if (
        spacing.shape != (3,)
        or not np.isfinite(spacing).all()
        or np.any(spacing <= 0.0)
    ):
        raise ValueError("spacing_mm must be positive")
    if fixed_layer_height_mm is None:
        return spacing.copy()
    fixed = float(fixed_layer_height_mm)
    if not np.isfinite(fixed) or fixed <= 0.0:
        raise ValueError("fixed_layer_height_mm must be finite and positive")
    spacing[index] = fixed
    return spacing


def _plane_axes(axis: LayerAxis) -> tuple[int, int]:
    index = axis_index(axis)
    return tuple(item for item in range(3) if item != index)  # type: ignore[return-value]


def _contours_from_plane(
    values: np.ndarray,
    plane_origin: np.ndarray,
    plane_spacing: tuple[float, float],
    fixed_axis: int,
    fixed_position: float,
    level: float,
    source: ContourFieldSource,
    axis: LayerAxis,
) -> LayerContourResult:
    scalar = np.asarray(values, dtype=np.float32)
    if scalar.ndim != 2 or min(scalar.shape) < 2:
        raise ValueError("layer field must be a 2-D grid with at least 2 samples per axis")
    if not np.isfinite(scalar).all():
        raise ValueError("layer field contains non-finite values")
    plane_origin = np.asarray(plane_origin, dtype=np.float64)
    if plane_origin.shape != (2,):
        raise ValueError("plane_origin must contain two coordinates")
    spacing = tuple(float(value) for value in plane_spacing)
    if not np.isfinite(spacing).all() or any(value <= 0 for value in spacing):
        raise ValueError("plane_spacing must be positive")

    contours: list[LayerContour] = []
    for raw in measure.find_contours(scalar, float(level), positive_orientation="low"):
        # skimage returns (row, column); the first plane coordinate is row.
        local = np.column_stack(
            (
                plane_origin[0] + raw[:, 0] * spacing[0],
                plane_origin[1] + raw[:, 1] * spacing[1],
            )
        )
        points = np.zeros((len(local), 3), dtype=np.float64)
        plane_axes = _plane_axes(axis)
        points[:, plane_axes[0]] = local[:, 0]
        points[:, plane_axes[1]] = local[:, 1]
        points[:, fixed_axis] = float(fixed_position)
        closed = bool(np.linalg.norm(raw[0] - raw[-1]) <= 1.0e-6)
        contours.append(LayerContour(points, closed))

    return LayerContourResult(
        source=source,
        axis=axis,
        layer_position_mm=float(fixed_position),
        level=float(level),
        contours=tuple(contours),
        plane_origin_mm=plane_origin,
        plane_spacing_mm=spacing,
        grid_shape=tuple(int(value) for value in scalar.shape),
    )


def sample_sampled_field_layer(
    field: SampledImplicitField,
    axis: LayerAxis,
    layer_position_mm: float,
    level: float = 0.0,
    source: ContourFieldSource = "render",
) -> LayerContourResult:
    """Extract a layer from an existing regular sampled field.

    A layer between two stored planes is linearly interpolated.  This keeps
    the contour view responsive while preserving the exact display grid.
    """

    index = axis_index(axis)
    coordinate = float(layer_position_mm)
    bounds = field.bounds
    if not bounds[0, index] - 1.0e-9 <= coordinate <= bounds[1, index] + 1.0e-9:
        raise ValueError("layer position lies outside the field bounds")
    position = np.clip(coordinate, bounds[0, index], bounds[1, index])
    fractional = (position - field.origin[index]) / field.spacing[index]
    lower = int(np.floor(fractional))
    upper = min(lower + 1, field.values.shape[index] - 1)
    lower = max(lower, 0)
    alpha = float(fractional - lower)
    moved = np.moveaxis(field.values, index, 0)
    plane = moved[lower] * (1.0 - alpha) + moved[upper] * alpha
    plane_axes = _plane_axes(axis)
    plane_origin = np.asarray([field.origin[item] for item in plane_axes], dtype=np.float64)
    plane_spacing = tuple(float(field.spacing[item]) for item in plane_axes)
    return _contours_from_plane(
        plane,
        plane_origin,
        plane_spacing,
        index,
        float(position),
        level,
        source,
        axis,
    )


def sample_implicit_body_layer(
    body: ImplicitBody,
    axis: LayerAxis,
    layer_position_mm: float,
    spacing_mm: float | tuple[float, float, float],
    level: float = 0.0,
    source: ContourFieldSource = "authoritative",
    max_points_per_batch: int = 250_000,
) -> LayerContourResult:
    """Sample one 2-D layer directly from an evaluator-backed implicit body."""

    if int(max_points_per_batch) < 1:
        raise ValueError("max_points_per_batch must be positive")
    index = axis_index(axis)
    coordinate = float(layer_position_mm)
    bounds = body.bounds
    if not bounds[0, index] - 1.0e-9 <= coordinate <= bounds[1, index] + 1.0e-9:
        raise ValueError("layer position lies outside the body bounds")
    spacing = np.asarray(spacing_mm, dtype=np.float64)
    if spacing.ndim == 0:
        spacing = np.repeat(spacing, 3)
    if spacing.shape != (3,) or not np.isfinite(spacing).all() or np.any(spacing <= 0):
        raise ValueError("spacing_mm must be positive")
    plane_axes = _plane_axes(axis)
    coordinates = []
    for item in plane_axes:
        values = np.arange(
            bounds[0, item],
            bounds[1, item] + spacing[item] * 0.5,
            spacing[item],
            dtype=np.float64,
        )
        if len(values) == 0 or values[-1] < bounds[1, item] - spacing[item] * 1.0e-7:
            values = np.append(values, bounds[1, item])
        else:
            values[-1] = min(values[-1], bounds[1, item])
        coordinates.append(values)
    first, second = np.meshgrid(coordinates[0], coordinates[1], indexing="ij")
    points = np.zeros((first.size, 3), dtype=np.float64)
    points[:, plane_axes[0]] = first.ravel()
    points[:, plane_axes[1]] = second.ravel()
    points[:, index] = np.clip(coordinate, bounds[0, index], bounds[1, index])
    values = np.empty(len(points), dtype=np.float32)
    batch_size = int(max_points_per_batch)
    for start in range(0, len(points), batch_size):
        stop = min(start + batch_size, len(points))
        values[start:stop] = body.evaluate_points(points[start:stop])
    plane = values.reshape(first.shape)
    plane_origin = np.asarray([coordinates[0][0], coordinates[1][0]], dtype=np.float64)
    plane_spacing = (float(spacing[plane_axes[0]]), float(spacing[plane_axes[1]]))
    return _contours_from_plane(
        plane,
        plane_origin,
        plane_spacing,
        index,
        float(points[0, index]),
        level,
        source,
        axis,
    )
