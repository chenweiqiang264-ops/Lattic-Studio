"""Static high-fidelity rendering directly from authoritative implicit bodies.

The interactive viewport renders a bounded three-dimensional sampled-field
cache.  This module is intentionally independent: it casts image-space rays
against an :class:`ImplicitBody`, refines the first zero crossing, and shades
the resulting field-gradient normal without creating an STL or a dense 3-D
display grid.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from time import perf_counter
from typing import Callable

import numpy as np

if __package__:
    from .field import ImplicitBody
else:  # Support IDEs and direct ``python precise_render.py`` execution.
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from lattice_studio.engine.implicit.field import ImplicitBody


ProgressCallback = Callable[[str, float], None]


def _finite_vector(name: str, value: tuple[float, float, float]) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float64)
    if vector.shape != (3,) or not np.isfinite(vector).all():
        raise ValueError(f"{name} must contain three finite values")
    return vector


def _normalize(vector: np.ndarray, name: str) -> np.ndarray:
    length = float(np.linalg.norm(vector))
    if length <= 1.0e-12:
        raise ValueError(f"{name} must have non-zero length")
    return vector / length


@dataclass(frozen=True)
class PreciseRenderCamera:
    """Camera state captured from the interactive viewport."""

    position: tuple[float, float, float]
    focal_point: tuple[float, float, float]
    view_up: tuple[float, float, float]
    parallel_projection: bool
    parallel_scale: float
    view_angle_deg: float

    def __post_init__(self) -> None:
        position = _finite_vector("position", self.position)
        focal = _finite_vector("focal_point", self.focal_point)
        up = _finite_vector("view_up", self.view_up)
        forward = _normalize(focal - position, "camera direction")
        if np.linalg.norm(np.cross(forward, up)) <= 1.0e-8:
            raise ValueError("view_up must not be parallel to the camera direction")
        if not np.isfinite(self.parallel_scale) or self.parallel_scale <= 0.0:
            raise ValueError("parallel_scale must be finite and positive")
        if not np.isfinite(self.view_angle_deg) or not 0.0 < self.view_angle_deg < 179.0:
            raise ValueError("view_angle_deg must be between 0 and 179 degrees")


@dataclass(frozen=True)
class PreciseRenderSettings:
    """Finite accuracy contract for one static precise render."""

    width_px: int = 1920
    height_px: int = 1080
    minimum_feature_mm: float = 1.0
    tile_size_px: int = 128
    maximum_iterations: int = 1024
    root_refinement_steps: int = 9

    def __post_init__(self) -> None:
        for name in ("width_px", "height_px", "tile_size_px", "maximum_iterations"):
            value = getattr(self, name)
            if isinstance(value, bool) or int(value) != value or int(value) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.width_px > 16384 or self.height_px > 16384:
            raise ValueError("precise render dimensions must not exceed 16384 pixels")
        feature = float(self.minimum_feature_mm)
        if not np.isfinite(feature) or feature <= 0.0:
            raise ValueError("minimum_feature_mm must be finite and positive")
        if not 1 <= int(self.root_refinement_steps) <= 24:
            raise ValueError("root_refinement_steps must be between 1 and 24")


@dataclass(frozen=True)
class PreciseRenderMaterial:
    """Opaque CAD material used by the static implicit renderer."""

    color: tuple[float, float, float] = (0.72, 0.74, 0.78)
    metallic: float = 0.0
    roughness: float = 0.55

    def __post_init__(self) -> None:
        color = np.asarray(self.color, dtype=np.float64)
        if color.shape != (3,) or not np.isfinite(color).all():
            raise ValueError("material color must contain three finite values")
        if np.any(color < 0.0) or np.any(color > 1.0):
            raise ValueError("material color values must be between 0 and 1")
        for name in ("metallic", "roughness"):
            value = float(getattr(self, name))
            if not np.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1")


@dataclass(frozen=True)
class PreciseRenderReport:
    """Measured accuracy and resource facts for a completed render."""

    output_size_px: tuple[int, int]
    pixel_footprint_mm: float
    surface_tolerance_mm: float
    maximum_ray_step_mm: float
    hit_pixels: int
    evaluated_points: int
    tile_count: int
    max_resident_rays: int
    elapsed_seconds: float
    backend: str = "authoritative-evaluator-ray-cast"


@dataclass(frozen=True)
class PreciseRenderResult:
    image_rgb: np.ndarray
    report: PreciseRenderReport


def _camera_basis(camera: PreciseRenderCamera) -> tuple[np.ndarray, ...]:
    position = np.asarray(camera.position, dtype=np.float64)
    focal = np.asarray(camera.focal_point, dtype=np.float64)
    forward = _normalize(focal - position, "camera direction")
    right = _normalize(
        np.cross(forward, np.asarray(camera.view_up, dtype=np.float64)),
        "camera right axis",
    )
    up = _normalize(np.cross(right, forward), "camera up axis")
    return position, focal, forward, right, up


def _ray_box_intersections(
    origins: np.ndarray,
    directions: np.ndarray,
    bounds: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    parallel = np.abs(directions) <= 1.0e-14
    outside_parallel = parallel & (
        (origins < bounds[0][None, :]) | (origins > bounds[1][None, :])
    )
    safe_directions = np.where(parallel, 1.0, directions)
    first = (bounds[0][None, :] - origins) / safe_directions
    second = (bounds[1][None, :] - origins) / safe_directions
    axis_near = np.where(parallel, -np.inf, np.minimum(first, second))
    axis_far = np.where(parallel, np.inf, np.maximum(first, second))
    near = np.maximum(np.max(axis_near, axis=1), 0.0)
    far = np.min(axis_far, axis=1)
    valid = (~np.any(outside_parallel, axis=1)) & (far >= near)
    return near, far, valid


def _background_image(
    width: int,
    height: int,
    lower: tuple[float, float, float],
    upper: tuple[float, float, float],
) -> np.ndarray:
    bottom = np.asarray(lower, dtype=np.float64)
    top = np.asarray(upper, dtype=np.float64)
    if bottom.shape != (3,) or top.shape != (3,):
        raise ValueError("background colors must contain three values")
    if not np.isfinite((bottom, top)).all():
        raise ValueError("background colors must be finite")
    blend = np.linspace(1.0, 0.0, height, dtype=np.float64)[:, None, None]
    rows = top[None, None, :] * blend + bottom[None, None, :] * (1.0 - blend)
    image = np.broadcast_to(rows, (height, width, 3)).copy()
    return np.clip(np.rint(image * 255.0), 0.0, 255.0).astype(np.uint8)


def _shade_hits(
    body: ImplicitBody,
    points: np.ndarray,
    view_directions: np.ndarray,
    camera_basis: tuple[np.ndarray, ...],
    material: PreciseRenderMaterial,
    normal_step: float,
    evaluate: Callable[[np.ndarray], np.ndarray],
) -> np.ndarray:
    gradients = np.empty_like(points)
    for axis in range(3):
        offset = np.zeros(3, dtype=np.float64)
        offset[axis] = normal_step
        gradients[:, axis] = (
            evaluate(points + offset) - evaluate(points - offset)
        ) / (2.0 * normal_step)
    lengths = np.linalg.norm(gradients, axis=1, keepdims=True)
    normals = gradients / np.maximum(lengths, 1.0e-12)

    _position, _focal, forward, right, up = camera_basis
    key = _normalize(-forward - 0.35 * right + 0.55 * up, "key light")
    fill = _normalize(-forward + 0.80 * right + 0.15 * up, "fill light")
    rim = _normalize(forward - 0.25 * right + 0.65 * up, "rim light")
    ndotl_key = np.clip(normals @ key, 0.0, 1.0)
    ndotl_fill = np.clip(normals @ fill, 0.0, 1.0)
    ndotl_rim = np.clip(normals @ rim, 0.0, 1.0)

    view = -view_directions
    half_vector = key[None, :] + view
    half_vector /= np.maximum(
        np.linalg.norm(half_vector, axis=1, keepdims=True), 1.0e-12
    )
    roughness = max(float(material.roughness), 0.05)
    specular_power = float(np.clip(8.0 / (roughness**2), 6.0, 128.0))
    specular = np.clip(
        np.sum(normals * half_vector, axis=1), 0.0, 1.0
    ) ** specular_power
    base = np.asarray(material.color, dtype=np.float64)
    diffuse_strength = (
        0.20 + 0.62 * ndotl_key + 0.20 * ndotl_fill + 0.10 * ndotl_rim
    )
    diffuse_strength *= 1.0 - 0.55 * float(material.metallic)
    specular_strength = 0.10 + 0.42 * float(material.metallic)
    linear = base[None, :] * diffuse_strength[:, None]
    linear += specular_strength * specular[:, None]
    # Convert the linear lighting result to display-referred sRGB.
    srgb = np.where(
        linear <= 0.0031308,
        12.92 * linear,
        1.055 * np.power(np.maximum(linear, 0.0), 1.0 / 2.4) - 0.055,
    )
    return np.clip(np.rint(srgb * 255.0), 0.0, 255.0).astype(np.uint8)


def render_precise_implicit(
    body: ImplicitBody,
    *,
    camera: PreciseRenderCamera,
    settings: PreciseRenderSettings,
    material: PreciseRenderMaterial = PreciseRenderMaterial(),
    background_lower: tuple[float, float, float] = (0.945, 0.953, 0.965),
    background_upper: tuple[float, float, float] = (1.0, 1.0, 1.0),
    progress: ProgressCallback | None = None,
) -> PreciseRenderResult:
    """Render the first visible zero crossing of an authoritative implicit body."""

    if not isinstance(body, ImplicitBody):
        raise TypeError("body must be an ImplicitBody")
    if not isinstance(camera, PreciseRenderCamera):
        raise TypeError("camera must be a PreciseRenderCamera")
    if not isinstance(settings, PreciseRenderSettings):
        raise TypeError("settings must be PreciseRenderSettings")
    if not isinstance(material, PreciseRenderMaterial):
        raise TypeError("material must be a PreciseRenderMaterial")

    started = perf_counter()
    width = int(settings.width_px)
    height = int(settings.height_px)
    aspect = width / height
    position, focal, forward, right, up = _camera_basis(camera)
    focal_distance = float(np.linalg.norm(focal - position))
    if camera.parallel_projection:
        # VTK defines parallel_scale as the full vertical view height.
        pixel_footprint = float(camera.parallel_scale) / height
    else:
        visible_height = 2.0 * focal_distance * np.tan(
            np.deg2rad(float(camera.view_angle_deg)) * 0.5
        )
        pixel_footprint = visible_height / height
    feature = float(settings.minimum_feature_mm)
    surface_tolerance = min(pixel_footprint * 0.25, feature / 24.0)
    surface_tolerance = max(surface_tolerance, np.finfo(np.float32).eps * 64.0)
    minimum_ray_step = max(surface_tolerance * 0.75, feature / 256.0)
    maximum_ray_step = max(feature / 3.0, minimum_ray_step)
    normal_step = max(surface_tolerance * 2.0, feature / 160.0)

    image = _background_image(
        width,
        height,
        background_lower,
        background_upper,
    )
    tile_size = int(settings.tile_size_px)
    tile_count_x = (width + tile_size - 1) // tile_size
    tile_count_y = (height + tile_size - 1) // tile_size
    tile_count = tile_count_x * tile_count_y
    evaluated_points = 0
    hit_pixels = 0
    max_resident_rays = 0

    def evaluate(points: np.ndarray) -> np.ndarray:
        nonlocal evaluated_points
        evaluated_points += len(points)
        return body.evaluate_points(points)

    completed_tiles = 0
    for y0 in range(0, height, tile_size):
        y1 = min(y0 + tile_size, height)
        for x0 in range(0, width, tile_size):
            x1 = min(x0 + tile_size, width)
            pixel_x, pixel_y = np.meshgrid(
                np.arange(x0, x1, dtype=np.float64) + 0.5,
                np.arange(y0, y1, dtype=np.float64) + 0.5,
            )
            screen_x = (2.0 * pixel_x.ravel() / width - 1.0) * aspect
            screen_y = 1.0 - 2.0 * pixel_y.ravel() / height
            if camera.parallel_projection:
                origins = (
                    position[None, :]
                    + 0.5 * screen_x[:, None] * float(camera.parallel_scale) * right[None, :]
                    + 0.5 * screen_y[:, None] * float(camera.parallel_scale) * up[None, :]
                )
                directions = np.broadcast_to(forward, origins.shape).copy()
            else:
                tangent = np.tan(np.deg2rad(float(camera.view_angle_deg)) * 0.5)
                directions = (
                    forward[None, :]
                    + screen_x[:, None] * tangent * right[None, :]
                    + screen_y[:, None] * tangent * up[None, :]
                )
                directions /= np.linalg.norm(directions, axis=1, keepdims=True)
                origins = np.broadcast_to(position, directions.shape).copy()

            near, far, valid = _ray_box_intersections(origins, directions, body.bounds)
            ray_indices = np.flatnonzero(valid)
            max_resident_rays = max(max_resident_rays, len(ray_indices))
            hit_indices: list[np.ndarray] = []
            hit_points: list[np.ndarray] = []
            hit_directions: list[np.ndarray] = []
            if len(ray_indices):
                ray_origins = origins[ray_indices]
                ray_directions = directions[ray_indices]
                ray_far = far[ray_indices]
                ray_t = near[ray_indices] + minimum_ray_step
                previous_t = ray_t.copy()
                previous_values = evaluate(
                    ray_origins + ray_t[:, None] * ray_directions
                )
                active = np.ones(len(ray_indices), dtype=bool)
                for _iteration in range(int(settings.maximum_iterations)):
                    active_indices = np.flatnonzero(active)
                    if not len(active_indices):
                        break
                    values = previous_values[active_indices]
                    direct_hit = np.abs(values) <= surface_tolerance
                    if np.any(direct_hit):
                        local = active_indices[direct_hit]
                        hit_indices.append(ray_indices[local])
                        hit_points.append(
                            ray_origins[local]
                            + previous_t[local, None] * ray_directions[local]
                        )
                        hit_directions.append(ray_directions[local])
                        active[local] = False

                    moving = active_indices[~direct_hit]
                    if not len(moving):
                        continue
                    step = np.clip(
                        np.abs(previous_values[moving]) * 0.65,
                        minimum_ray_step,
                        maximum_ray_step,
                    )
                    next_t = previous_t[moving] + step
                    inside = next_t <= ray_far[moving]
                    if np.any(~inside):
                        active[moving[~inside]] = False
                    moving = moving[inside]
                    if not len(moving):
                        continue
                    next_t = next_t[inside]
                    next_points = (
                        ray_origins[moving]
                        + next_t[:, None] * ray_directions[moving]
                    )
                    next_values = evaluate(next_points)
                    crossed = previous_values[moving] * next_values <= 0.0
                    if np.any(crossed):
                        crossing = moving[crossed]
                        low_t = previous_t[crossing].copy()
                        high_t = next_t[crossed].copy()
                        low_values = previous_values[crossing].copy()
                        for _ in range(int(settings.root_refinement_steps)):
                            middle_t = 0.5 * (low_t + high_t)
                            middle_values = evaluate(
                                ray_origins[crossing]
                                + middle_t[:, None] * ray_directions[crossing]
                            )
                            same_side = low_values * middle_values > 0.0
                            low_t[same_side] = middle_t[same_side]
                            low_values[same_side] = middle_values[same_side]
                            high_t[~same_side] = middle_t[~same_side]
                        final_t = 0.5 * (low_t + high_t)
                        hit_indices.append(ray_indices[crossing])
                        hit_points.append(
                            ray_origins[crossing]
                            + final_t[:, None] * ray_directions[crossing]
                        )
                        hit_directions.append(ray_directions[crossing])
                        active[crossing] = False

                    continuing = moving[~crossed]
                    previous_t[continuing] = next_t[~crossed]
                    previous_values[continuing] = next_values[~crossed]

            if hit_points:
                indices = np.concatenate(hit_indices)
                points = np.concatenate(hit_points, axis=0)
                directions_at_hit = np.concatenate(hit_directions, axis=0)
                colors = _shade_hits(
                    body,
                    points,
                    directions_at_hit,
                    (position, focal, forward, right, up),
                    material,
                    normal_step,
                    evaluate,
                )
                local_height = y1 - y0
                local_width = x1 - x0
                tile = image[y0:y1, x0:x1].reshape(-1, 3)
                tile[indices] = colors
                image[y0:y1, x0:x1] = tile.reshape(local_height, local_width, 3)
                hit_pixels += len(indices)

            completed_tiles += 1
            if progress:
                progress(
                    f"精确渲染：光线求交 {completed_tiles}/{tile_count}",
                    completed_tiles / tile_count,
                )

    report = PreciseRenderReport(
        output_size_px=(width, height),
        pixel_footprint_mm=float(pixel_footprint),
        surface_tolerance_mm=float(surface_tolerance),
        maximum_ray_step_mm=float(maximum_ray_step),
        hit_pixels=int(hit_pixels),
        evaluated_points=int(evaluated_points),
        tile_count=tile_count,
        max_resident_rays=int(max_resident_rays),
        elapsed_seconds=perf_counter() - started,
    )
    return PreciseRenderResult(image_rgb=image, report=report)


__all__ = [
    "PreciseRenderCamera",
    "PreciseRenderMaterial",
    "PreciseRenderReport",
    "PreciseRenderResult",
    "PreciseRenderSettings",
    "render_precise_implicit",
]
