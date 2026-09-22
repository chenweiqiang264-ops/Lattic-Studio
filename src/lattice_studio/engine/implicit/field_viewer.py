"""On-demand 2-D inspection planes for implicit fields.

The field viewer intentionally samples only the plane requested by the user.
It therefore remains independent from both the interactive display volume and
the grid used to reconstruct an STL surface.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Literal

import numpy as np
from skimage import measure

from .field import ImplicitBody, SampledImplicitField


FieldColormap = Literal["implicit", "turbo", "distance"]
FieldSourceData = ImplicitBody | SampledImplicitField


def _unit(vector: np.ndarray, name: str) -> np.ndarray:
    """Return one finite normalized three-dimensional vector."""

    value = np.asarray(vector, dtype=np.float64).reshape(3)
    length = float(np.linalg.norm(value))
    if not np.isfinite(length) or length <= np.finfo(np.float64).eps:
        raise ValueError(f"{name} must be a finite non-zero 3D vector")
    return value / length


def orthonormal_plane_axes(
    normal: np.ndarray,
    preferred_u_axis: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Build stable in-plane axes for a requested plane normal."""

    unit_normal = _unit(normal, "plane normal")
    candidate = (
        np.asarray(preferred_u_axis, dtype=np.float64).reshape(3)
        if preferred_u_axis is not None
        else np.array((1.0, 0.0, 0.0), dtype=np.float64)
    )
    candidate -= unit_normal * float(np.dot(candidate, unit_normal))
    if np.linalg.norm(candidate) <= 1.0e-8:
        candidate = np.array((0.0, 1.0, 0.0), dtype=np.float64)
        candidate -= unit_normal * float(np.dot(candidate, unit_normal))
    u_axis = _unit(candidate, "plane U axis")
    v_axis = _unit(np.cross(unit_normal, u_axis), "plane V axis")
    return u_axis, v_axis


@dataclass(frozen=True)
class FieldPlaneState:
    """Placement and local frame of one finite rectangular field plane."""

    center_mm: np.ndarray
    u_axis: np.ndarray
    v_axis: np.ndarray
    width_mm: float
    height_mm: float

    def __post_init__(self) -> None:
        center = np.asarray(self.center_mm, dtype=np.float64).reshape(3)
        u_axis = _unit(self.u_axis, "plane U axis")
        v_candidate = np.asarray(self.v_axis, dtype=np.float64).reshape(3)
        v_candidate -= u_axis * float(np.dot(v_candidate, u_axis))
        v_axis = _unit(v_candidate, "plane V axis")
        width = float(self.width_mm)
        height = float(self.height_mm)
        if not np.isfinite(center).all():
            raise ValueError("plane center must contain finite coordinates")
        if not np.isfinite(width) or width <= 0.0:
            raise ValueError("plane width must be finite and positive")
        if not np.isfinite(height) or height <= 0.0:
            raise ValueError("plane height must be finite and positive")
        object.__setattr__(self, "center_mm", center)
        object.__setattr__(self, "u_axis", u_axis)
        object.__setattr__(self, "v_axis", v_axis)
        object.__setattr__(self, "width_mm", width)
        object.__setattr__(self, "height_mm", height)

    @property
    def normal(self) -> np.ndarray:
        """Return the right-handed normal of the displayed field plane."""

        return _unit(np.cross(self.u_axis, self.v_axis), "plane normal")

    @property
    def corner_mm(self) -> np.ndarray:
        """Return the plane's local ``(0, 0)`` corner in world coordinates."""

        return self.center_mm - 0.5 * (
            self.width_mm * self.u_axis + self.height_mm * self.v_axis
        )

    def with_center(self, center_mm: np.ndarray) -> "FieldPlaneState":
        return FieldPlaneState(
            center_mm,
            self.u_axis,
            self.v_axis,
            self.width_mm,
            self.height_mm,
        )

    def with_axes(self, u_axis: np.ndarray, v_axis: np.ndarray) -> "FieldPlaneState":
        return FieldPlaneState(
            self.center_mm,
            u_axis,
            v_axis,
            self.width_mm,
            self.height_mm,
        )

    def with_size(self, width_mm: float, height_mm: float | None = None) -> "FieldPlaneState":
        return FieldPlaneState(
            self.center_mm,
            self.u_axis,
            self.v_axis,
            width_mm,
            width_mm if height_mm is None else height_mm,
        )


@dataclass(frozen=True)
class FieldPlaneIsoline:
    """All connected isoline polylines corresponding to one scalar value."""

    value: float
    polylines: tuple[np.ndarray, ...]

    def __post_init__(self) -> None:
        value = float(self.value)
        if not np.isfinite(value):
            raise ValueError("isoline value must be finite")
        polylines = tuple(
            np.ascontiguousarray(polyline, dtype=np.float64)
            for polyline in self.polylines
        )
        for polyline in polylines:
            if polyline.ndim != 2 or polyline.shape[1] != 3 or len(polyline) < 2:
                raise ValueError("isoline polylines must have shape (N, 3), N >= 2")
            if not np.isfinite(polyline).all():
                raise ValueError("isoline polyline points must be finite")
        object.__setattr__(self, "value", value)
        object.__setattr__(self, "polylines", polylines)


@dataclass(frozen=True)
class FieldPlaneSample:
    """A sampled scalar field and its world-space plane coordinates."""

    state: FieldPlaneState
    values: np.ndarray
    points: np.ndarray
    source_bounds_mm: np.ndarray

    def __post_init__(self) -> None:
        values = np.ascontiguousarray(self.values, dtype=np.float32)
        points = np.ascontiguousarray(self.points, dtype=np.float64)
        bounds = np.asarray(self.source_bounds_mm, dtype=np.float64)
        if values.ndim != 2 or min(values.shape) < 2:
            raise ValueError("field-plane values must be a 2-D grid with at least 2 samples per axis")
        if points.shape != values.shape + (3,):
            raise ValueError("field-plane points must match the field sample shape")
        if not np.isfinite(values).all() or not np.isfinite(points).all():
            raise ValueError("field-plane samples must be finite")
        if bounds.shape != (2, 3) or not np.isfinite(bounds).all():
            raise ValueError("source bounds must have shape (2, 3)")
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "points", points)
        object.__setattr__(self, "source_bounds_mm", bounds)

    @property
    def shape(self) -> tuple[int, int]:
        return tuple(int(value) for value in self.values.shape)

    @property
    def valid_mask(self) -> np.ndarray:
        """Return plane samples defined inside the finite source field."""

        return np.logical_and(
            self.points >= self.source_bounds_mm[0] - 1.0e-9,
            self.points <= self.source_bounds_mm[1] + 1.0e-9,
        ).all(axis=2)

    def value_at(self, point_mm: np.ndarray) -> float | None:
        """Bilinearly interpolate one picked world-space point on the plane."""

        point = np.asarray(point_mm, dtype=np.float64).reshape(3)
        if not np.logical_and(
            point >= self.source_bounds_mm[0] - 1.0e-9,
            point <= self.source_bounds_mm[1] + 1.0e-9,
        ).all():
            return None
        local = point - self.state.corner_mm
        u_fraction = float(np.dot(local, self.state.u_axis) / self.state.width_mm)
        v_fraction = float(np.dot(local, self.state.v_axis) / self.state.height_mm)
        tolerance = 1.0e-6
        if (
            u_fraction < -tolerance
            or u_fraction > 1.0 + tolerance
            or v_fraction < -tolerance
            or v_fraction > 1.0 + tolerance
        ):
            return None
        u = np.clip(u_fraction, 0.0, 1.0) * (self.values.shape[0] - 1)
        v = np.clip(v_fraction, 0.0, 1.0) * (self.values.shape[1] - 1)
        u0 = int(np.floor(u))
        v0 = int(np.floor(v))
        u1 = min(u0 + 1, self.values.shape[0] - 1)
        v1 = min(v0 + 1, self.values.shape[1] - 1)
        fu = float(u - u0)
        fv = float(v - v0)
        lower = self.values[u0, v0] * (1.0 - fu) + self.values[u1, v0] * fu
        upper = self.values[u0, v1] * (1.0 - fu) + self.values[u1, v1] * fu
        return float(lower * (1.0 - fv) + upper * fv)


@dataclass(frozen=True)
class FieldSurfaceHit:
    """A zero-level hit used to place a field plane on a visible surface."""

    point_mm: np.ndarray
    normal: np.ndarray
    value: float

    def __post_init__(self) -> None:
        point = np.asarray(self.point_mm, dtype=np.float64).reshape(3)
        normal = _unit(self.normal, "surface normal")
        value = float(self.value)
        if not np.isfinite(point).all() or not np.isfinite(value):
            raise ValueError("surface hit must contain finite values")
        object.__setattr__(self, "point_mm", point)
        object.__setattr__(self, "normal", normal)
        object.__setattr__(self, "value", value)


class FieldPlaneCache:
    """Small LRU cache for plane samples independent of renderer resources."""

    def __init__(self, max_entries: int = 24) -> None:
        if int(max_entries) < 1:
            raise ValueError("max_entries must be positive")
        self.max_entries = int(max_entries)
        self._entries: OrderedDict[object, FieldPlaneSample] = OrderedDict()

    def get(self, key: object) -> FieldPlaneSample | None:
        result = self._entries.get(key)
        if result is not None:
            self._entries.move_to_end(key)
        return result

    def put(self, key: object, sample: FieldPlaneSample) -> None:
        self._entries[key] = sample
        self._entries.move_to_end(key)
        while len(self._entries) > self.max_entries:
            self._entries.popitem(last=False)

    def clear(self) -> None:
        self._entries.clear()

    def __len__(self) -> int:
        return len(self._entries)


def field_plane_cache_key(
    source_identity: object,
    state: FieldPlaneState,
    resolution: int | tuple[int, int],
) -> tuple[object, ...]:
    """Return a deterministic cache key for geometry-dependent plane data."""

    if isinstance(resolution, int):
        shape = (int(resolution), int(resolution))
    else:
        shape = tuple(int(value) for value in resolution)
    return (
        source_identity,
        tuple(round(float(value), 8) for value in state.center_mm),
        tuple(round(float(value), 8) for value in state.u_axis),
        tuple(round(float(value), 8) for value in state.v_axis),
        round(float(state.width_mm), 8),
        round(float(state.height_mm), 8),
        shape,
    )


def source_bounds(field: FieldSourceData) -> np.ndarray:
    """Return the finite world-space bounds shared by both field sources."""

    return np.asarray(field.bounds, dtype=np.float64)


def expand_implicit_body_bounds(
    body: ImplicitBody,
    padding_mm: float,
) -> ImplicitBody:
    """Return a view-only body whose evaluator is valid in a padded domain.

    The caller is responsible for using this only with evaluators that can
    truthfully evaluate points outside their original bounds.  In particular,
    the design-domain STL evaluator can query its signed distance in arbitrary
    surrounding space.  This helper never changes the source body, its mesh,
    or any sampled display cache.
    """

    if not isinstance(body, ImplicitBody):
        raise TypeError("body must be an ImplicitBody")
    padding = float(padding_mm)
    if not np.isfinite(padding) or padding < 0.0:
        raise ValueError("padding_mm must be finite and non-negative")
    if padding == 0.0:
        return body
    return ImplicitBody(
        name=body.name,
        bounds=np.vstack((body.bounds[0] - padding, body.bounds[1] + padding)),
        evaluate=body.evaluate,
        is_preview_only=body.is_preview_only,
    )


def source_sampling_spacing(field: FieldSourceData) -> np.ndarray:
    """Return a practical inspection spacing for automatic plane sampling."""

    if isinstance(field, SampledImplicitField):
        return np.asarray(field.spacing, dtype=np.float64)
    extent = np.asarray(field.bounds[1] - field.bounds[0], dtype=np.float64)
    return np.maximum(extent / 256.0, 1.0e-6)


def recommended_plane_resolution(
    state: FieldPlaneState,
    field: FieldSourceData,
    *,
    minimum: int = 96,
    maximum: int = 512,
) -> tuple[int, int]:
    """Recommend a bounded 2-D sample shape from source field detail."""

    if minimum < 2 or maximum < minimum:
        raise ValueError("plane resolution limits must satisfy 2 <= minimum <= maximum")
    spacing = float(np.min(source_sampling_spacing(field)))
    target_u = int(np.ceil(state.width_mm / spacing)) + 1
    target_v = int(np.ceil(state.height_mm / spacing)) + 1
    return (
        int(np.clip(target_u, minimum, maximum)),
        int(np.clip(target_v, minimum, maximum)),
    )


def _outside_value(field: FieldSourceData) -> np.float32:
    if isinstance(field, SampledImplicitField):
        magnitude = float(np.max(np.abs(field.values), initial=0.0))
    else:
        magnitude = float(np.max(field.bounds[1] - field.bounds[0]))
    return np.float32(max(magnitude, 1.0))


def evaluate_sampled_field_points(
    field: SampledImplicitField,
    points_mm: np.ndarray,
) -> np.ndarray:
    """Tri-linearly evaluate a regular sampled field without resampling it."""

    points = np.asarray(points_mm, dtype=np.float64).reshape(-1, 3)
    output = np.full(len(points), _outside_value(field), dtype=np.float32)
    bounds = field.bounds
    inside = np.logical_and(points >= bounds[0], points <= bounds[1]).all(axis=1)
    if not np.any(inside):
        return output
    local = (points[inside] - field.origin) / field.spacing
    lower = np.floor(local).astype(np.int64)
    lower = np.clip(lower, 0, np.asarray(field.values.shape) - 1)
    upper = np.minimum(lower + 1, np.asarray(field.values.shape) - 1)
    fraction = local - lower
    values = field.values
    c000 = values[lower[:, 0], lower[:, 1], lower[:, 2]]
    c100 = values[upper[:, 0], lower[:, 1], lower[:, 2]]
    c010 = values[lower[:, 0], upper[:, 1], lower[:, 2]]
    c110 = values[upper[:, 0], upper[:, 1], lower[:, 2]]
    c001 = values[lower[:, 0], lower[:, 1], upper[:, 2]]
    c101 = values[upper[:, 0], lower[:, 1], upper[:, 2]]
    c011 = values[lower[:, 0], upper[:, 1], upper[:, 2]]
    c111 = values[upper[:, 0], upper[:, 1], upper[:, 2]]
    fx, fy, fz = fraction.T
    c00 = c000 * (1.0 - fx) + c100 * fx
    c10 = c010 * (1.0 - fx) + c110 * fx
    c01 = c001 * (1.0 - fx) + c101 * fx
    c11 = c011 * (1.0 - fx) + c111 * fx
    c0 = c00 * (1.0 - fy) + c10 * fy
    c1 = c01 * (1.0 - fy) + c11 * fy
    output[inside] = c0 * (1.0 - fz) + c1 * fz
    return output


def evaluate_field_points(field: FieldSourceData, points_mm: np.ndarray) -> np.ndarray:
    """Evaluate either source type, keeping points outside its bounds positive."""

    if isinstance(field, SampledImplicitField):
        return evaluate_sampled_field_points(field, points_mm)

    points = np.asarray(points_mm, dtype=np.float64).reshape(-1, 3)
    output = np.full(len(points), _outside_value(field), dtype=np.float32)
    inside = np.logical_and(points >= field.bounds[0], points <= field.bounds[1]).all(axis=1)
    if np.any(inside):
        output[inside] = field.evaluate_points(points[inside])
    return output


def sample_field_plane(
    field: FieldSourceData,
    state: FieldPlaneState,
    resolution: int | tuple[int, int],
    *,
    max_points_per_batch: int = 250_000,
) -> FieldPlaneSample:
    """Sample one finite plane from a field in bounded point batches."""

    if isinstance(resolution, int):
        shape = (int(resolution), int(resolution))
    else:
        shape = tuple(int(value) for value in resolution)
    if len(shape) != 2 or min(shape) < 2:
        raise ValueError("field-plane resolution must contain two integers >= 2")
    if int(max_points_per_batch) < 1:
        raise ValueError("max_points_per_batch must be positive")

    u_values = np.linspace(-0.5, 0.5, shape[0], dtype=np.float64) * state.width_mm
    v_values = np.linspace(-0.5, 0.5, shape[1], dtype=np.float64) * state.height_mm
    u_grid, v_grid = np.meshgrid(u_values, v_values, indexing="ij")
    points = (
        state.center_mm
        + u_grid[..., np.newaxis] * state.u_axis
        + v_grid[..., np.newaxis] * state.v_axis
    )
    flat = points.reshape(-1, 3)
    values = np.empty(len(flat), dtype=np.float32)
    for start in range(0, len(flat), int(max_points_per_batch)):
        stop = min(start + int(max_points_per_batch), len(flat))
        values[start:stop] = evaluate_field_points(field, flat[start:stop])
    return FieldPlaneSample(
        state=state,
        values=values.reshape(shape),
        points=points,
        source_bounds_mm=source_bounds(field),
    )


def resolve_field_value_range(
    sample: FieldPlaneSample,
    colormap: FieldColormap,
    *,
    custom_range: tuple[float, float] | None = None,
) -> tuple[float, float]:
    """Resolve a stable color range, optionally preserving implicit symmetry."""

    if custom_range is not None:
        lower, upper = (float(value) for value in custom_range)
        if not np.isfinite((lower, upper)).all() or not lower < upper:
            raise ValueError("custom field range must contain two finite ordered values")
        return lower, upper
    valid_values = sample.values[sample.valid_mask]
    if valid_values.size == 0:
        # A plane may deliberately be positioned outside a finite field.  The
        # renderer then shows its outline and no-data area without inventing a
        # field-dependent colour range.
        return -1.0, 1.0
    lower = float(np.min(valid_values))
    upper = float(np.max(valid_values))
    if colormap == "implicit":
        magnitude = max(abs(lower), abs(upper), np.finfo(np.float32).eps)
        return -magnitude, magnitude
    if upper - lower <= np.finfo(np.float32).eps:
        epsilon = max(abs(lower), 1.0) * 1.0e-6
        return lower - epsilon, upper + epsilon
    return lower, upper


def adaptive_isoline_interval(lower: float, upper: float, target_count: int = 12) -> float:
    """Return a human-readable interval that yields roughly ``target_count`` lines."""

    if not lower < upper:
        raise ValueError("isoline range must be ordered")
    if int(target_count) < 1:
        raise ValueError("target_count must be positive")
    raw = (upper - lower) / int(target_count)
    magnitude = 10.0 ** np.floor(np.log10(raw))
    normalized = raw / magnitude
    for candidate in (1.0, 2.0, 2.5, 5.0, 10.0):
        if normalized <= candidate:
            return float(candidate * magnitude)
    return float(10.0 * magnitude)


def extract_field_plane_isolines(
    sample: FieldPlaneSample,
    *,
    value_range: tuple[float, float],
    interval: float = 0.0,
    maximum_levels: int = 80,
) -> tuple[FieldPlaneIsoline, ...]:
    """Extract world-space isolines from an already sampled field plane."""

    lower, upper = (float(value) for value in value_range)
    requested_interval = float(interval)
    if not np.isfinite((lower, upper, requested_interval)).all() or not lower < upper:
        raise ValueError("isoline range must be finite and ordered")
    if requested_interval < 0.0:
        raise ValueError("isoline interval must be non-negative")
    if int(maximum_levels) < 1:
        raise ValueError("maximum_levels must be positive")
    step = requested_interval or adaptive_isoline_interval(lower, upper)
    if step <= 0.0 or not np.isfinite(step):
        return ()
    start = np.ceil(lower / step) * step
    levels = np.arange(start, upper + step * 0.25, step, dtype=np.float64)
    if lower <= 0.0 <= upper and not np.any(np.isclose(levels, 0.0, atol=step * 1.0e-8)):
        levels = np.sort(np.append(levels, 0.0))
    if len(levels) > int(maximum_levels):
        step = adaptive_isoline_interval(lower, upper, max(1, int(maximum_levels) - 1))
        start = np.ceil(lower / step) * step
        levels = np.arange(start, upper + step * 0.25, step, dtype=np.float64)

    corner = sample.state.corner_mm
    u_scale = sample.state.width_mm / max(sample.values.shape[0] - 1, 1)
    v_scale = sample.state.height_mm / max(sample.values.shape[1] - 1, 1)
    result: list[FieldPlaneIsoline] = []
    for level in levels:
        raw_lines = measure.find_contours(
            sample.values,
            float(level),
            positive_orientation="low",
            mask=sample.valid_mask,
        )
        polylines = tuple(
            corner
            + line[:, 0:1] * u_scale * sample.state.u_axis
            + line[:, 1:2] * v_scale * sample.state.v_axis
            for line in raw_lines
            if len(line) >= 2
        )
        if polylines:
            result.append(FieldPlaneIsoline(float(level), polylines))
    return tuple(result)


def nearest_isoline_value(value: float, isolines: tuple[FieldPlaneIsoline, ...]) -> float | None:
    """Return the rendered isoline value closest to a probed scalar value."""

    if not isolines:
        return None
    return float(min(isolines, key=lambda item: abs(item.value - value)).value)


def _ray_box_interval(
    origin: np.ndarray,
    direction: np.ndarray,
    bounds: np.ndarray,
) -> tuple[float, float] | None:
    lower = np.asarray(bounds[0], dtype=np.float64)
    upper = np.asarray(bounds[1], dtype=np.float64)
    ray_origin = np.asarray(origin, dtype=np.float64).reshape(3)
    ray_direction = _unit(direction, "ray direction")
    starts = np.empty(3, dtype=np.float64)
    stops = np.empty(3, dtype=np.float64)
    for axis, component in enumerate(ray_direction):
        if abs(float(component)) <= np.finfo(np.float64).eps:
            if ray_origin[axis] < lower[axis] or ray_origin[axis] > upper[axis]:
                return None
            starts[axis] = -np.inf
            stops[axis] = np.inf
            continue
        first = (lower[axis] - ray_origin[axis]) / component
        second = (upper[axis] - ray_origin[axis]) / component
        starts[axis] = min(first, second)
        stops[axis] = max(first, second)
    start = float(np.max(starts))
    stop = float(np.min(stops))
    if stop < max(start, 0.0):
        return None
    return max(start, 0.0), stop


def field_surface_hit_from_ray(
    field: FieldSourceData,
    ray_origin_mm: np.ndarray,
    ray_direction: np.ndarray,
    *,
    sample_spacing_mm: float | None = None,
    maximum_samples: int = 2048,
) -> FieldSurfaceHit | None:
    """Find the closest zero-level crossing of a field along one world ray."""

    origin = np.asarray(ray_origin_mm, dtype=np.float64).reshape(3)
    direction = _unit(ray_direction, "ray direction")
    interval = _ray_box_interval(origin, direction, source_bounds(field))
    if interval is None:
        return None
    start, stop = interval
    extent = max(stop - start, 0.0)
    if extent <= 0.0:
        return None
    spacing = (
        float(sample_spacing_mm)
        if sample_spacing_mm is not None
        else float(np.min(source_sampling_spacing(field)))
    )
    if not np.isfinite(spacing) or spacing <= 0.0:
        raise ValueError("surface-pick sampling spacing must be finite and positive")
    count = int(np.clip(np.ceil(extent / spacing) + 1, 48, int(maximum_samples)))
    distances = np.linspace(start, stop, count, dtype=np.float64)
    values = evaluate_field_points(field, origin + distances[:, np.newaxis] * direction)
    crossings = np.flatnonzero(values[:-1] * values[1:] <= 0.0)
    if len(crossings) == 0:
        return None
    index = int(crossings[0])
    lower_distance = float(distances[index])
    upper_distance = float(distances[index + 1])
    lower_value = float(values[index])
    for _ in range(18):
        midpoint = (lower_distance + upper_distance) * 0.5
        midpoint_value = float(
            evaluate_field_points(field, origin + midpoint * direction)[0]
        )
        if lower_value * midpoint_value <= 0.0:
            upper_distance = midpoint
        else:
            lower_distance = midpoint
            lower_value = midpoint_value
    distance = (lower_distance + upper_distance) * 0.5
    point = origin + distance * direction
    local_spacing = max(spacing * 0.5, np.finfo(np.float64).eps)
    offsets = np.eye(3, dtype=np.float64) * local_spacing
    gradient = np.empty(3, dtype=np.float64)
    for axis in range(3):
        samples = np.vstack((point + offsets[axis], point - offsets[axis]))
        values_axis = evaluate_field_points(field, samples)
        gradient[axis] = (float(values_axis[0]) - float(values_axis[1])) / (2.0 * local_spacing)
    if np.linalg.norm(gradient) <= 1.0e-10:
        gradient = -direction
    return FieldSurfaceHit(
        point_mm=point,
        normal=gradient,
        value=float(evaluate_field_points(field, point)[0]),
    )
