"""Automatic numerical backend for TPMS implicit-field evaluation."""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass
from typing import Literal, Protocol

import numpy as np

from lattice_studio.engine.implicit.cuda_probe import probe_numba_cuda


TPMSKind = Literal["G", "D", "IWP", "Primitive", "Neovius"]
TPMS_KINDS: tuple[TPMSKind, ...] = ("G", "D", "IWP", "Primitive", "Neovius")
CellSize = float | tuple[float, float, float]
TPMSGradientMode = Literal["linear", "power", "sigmoid", "layered"]
TPMSWallThicknessMethod = Literal["gradient_normalized", "field_threshold"]


_TPMS_KIND_CODES: dict[TPMSKind, int] = {
    "G": 0,
    "D": 1,
    "IWP": 2,
    "Primitive": 3,
    "Neovius": 4,
}
_GRADIENT_MODE_CODES: dict[TPMSGradientMode, int] = {
    "linear": 0,
    "power": 1,
    "sigmoid": 2,
    "layered": 3,
}
_WALL_METHOD_CODES: dict[TPMSWallThicknessMethod, int] = {
    "gradient_normalized": 0,
    "field_threshold": 1,
}


@dataclass(frozen=True)
class TPMSFieldSpec:
    """Physical parameters required to evaluate one TPMS shell field."""

    kind: TPMSKind
    cell_size_mm: CellSize
    wall_thickness_mm: float
    level: float = 0.0
    origin_mm: tuple[float, float, float] = (0.0, 0.0, 0.0)
    axes_world: tuple[tuple[float, float, float], ...] = (
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
    )

    def validate(self) -> None:
        if self.kind not in TPMS_KINDS:
            raise ValueError("unsupported TPMS kind")
        cell_size = np.asarray(self.cell_size_mm, dtype=np.float64)
        if cell_size.ndim == 0:
            cell_size = np.repeat(cell_size, 3)
        if (
            cell_size.shape != (3,)
            or not np.isfinite(cell_size).all()
            or not np.isfinite(self.wall_thickness_mm)
            or not np.isfinite(self.level)
        ):
            raise ValueError("TPMS field parameters must be finite")
        origin = np.asarray(self.origin_mm, dtype=np.float64)
        if origin.shape != (3,) or not np.isfinite(origin).all():
            raise ValueError("origin_mm must contain three finite values")
        axes = np.asarray(self.axes_world, dtype=np.float64)
        if axes.shape != (3, 3) or not np.isfinite(axes).all():
            raise ValueError("axes_world must be a finite 3 by 3 matrix")
        if not np.allclose(axes @ axes.T, np.eye(3), rtol=1e-6, atol=1e-6):
            raise ValueError("axes_world must form an orthonormal basis")
        if not np.isclose(np.linalg.det(axes), 1.0, rtol=1e-6, atol=1e-6):
            raise ValueError("axes_world must form a right-handed basis")
        if np.any(cell_size <= 0) or self.wall_thickness_mm <= 0:
            raise ValueError("cell size and wall thickness must be positive")


@dataclass(frozen=True)
class TPMSGradientSpec:
    """Numerical controls for a TPMS field varying along one local axis."""

    axis_index: int
    coordinate_min_mm: float
    coordinate_max_mm: float
    mode: TPMSGradientMode = "linear"
    power: float = 5.0
    layers: int = 5
    sigmoid_sharpness: float = 10.0
    use_thickness_gradient: bool = True
    thickness_soft_mm: float | None = None
    thickness_stiff_mm: float | None = None
    use_offset_gradient: bool = True
    offset_soft: float | None = None
    offset_stiff: float | None = None
    wall_thickness_method: TPMSWallThicknessMethod = "gradient_normalized"

    def validate(self) -> None:
        if int(self.axis_index) != self.axis_index or not 0 <= int(self.axis_index) <= 2:
            raise ValueError("axis_index must be 0, 1, or 2")
        values = np.asarray(
            (
                self.coordinate_min_mm,
                self.coordinate_max_mm,
                self.power,
                self.sigmoid_sharpness,
            ),
            dtype=np.float64,
        )
        if not np.isfinite(values).all():
            raise ValueError("gradient parameters must be finite")
        if self.coordinate_min_mm >= self.coordinate_max_mm:
            raise ValueError("gradient coordinate bounds must be increasing")
        if self.mode not in _GRADIENT_MODE_CODES:
            raise ValueError("unsupported TPMS gradient mode")
        if self.power <= 0.0 or self.sigmoid_sharpness <= 0.0:
            raise ValueError("gradient power and sigmoid sharpness must be positive")
        if int(self.layers) != self.layers or int(self.layers) < 2:
            raise ValueError("gradient layers must be an integer of at least 2")
        if self.wall_thickness_method not in _WALL_METHOD_CODES:
            raise ValueError("unsupported wall-thickness method")
        for value in (self.thickness_soft_mm, self.thickness_stiff_mm):
            if value is not None and (not np.isfinite(value) or value <= 0.0):
                raise ValueError("gradient thickness values must be positive or None")
        for value in (self.offset_soft, self.offset_stiff):
            if value is not None and not np.isfinite(value):
                raise ValueError("gradient offset values must be finite or None")


@dataclass(frozen=True)
class TransitionBlendSpec:
    """Physical controls for a sigmoid blend across a transition plane."""

    transition_width_mm: float
    center_offset_mm: float = 0.0
    sigmoid_sharpness: float = 1.0
    g_on_negative_side: bool = True

    def validate(self) -> None:
        values = (
            self.transition_width_mm,
            self.center_offset_mm,
            self.sigmoid_sharpness,
        )
        if not all(np.isfinite(value) for value in values):
            raise ValueError("transition parameters must be finite")
        if self.transition_width_mm <= 0 or self.sigmoid_sharpness <= 0:
            raise ValueError("transition width and sharpness must be positive")


def sigmoid_transition_weights(
    signed_plane_distance: np.ndarray,
    transition: TransitionBlendSpec,
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate the physical-width sigmoid weights on the CPU reference path."""

    transition.validate()
    centered = (
        np.asarray(signed_plane_distance, dtype=np.float32)
        - np.float32(transition.center_offset_mm)
    )
    normalized = centered / np.float32(0.5 * transition.transition_width_mm)
    exponent = np.float32(1.0 / transition.sigmoid_sharpness)
    shaped = np.sign(normalized) * np.power(np.abs(normalized), exponent)
    argument = np.clip(np.float32(np.log(9.0)) * shaped, -60.0, 60.0)
    d_weight = 1.0 / (1.0 + np.exp(-argument))
    if not transition.g_on_negative_side:
        d_weight = 1.0 - d_weight
    d_weight = np.asarray(d_weight, dtype=np.float32)
    return np.float32(1.0) - d_weight, d_weight


@dataclass(frozen=True)
class NumericalBackendStatus:
    active_backend: str
    using_gpu: bool
    device_name: str
    fallback_reason: str | None = None


class TPMSComputeAdapter(Protocol):
    name: str
    available: bool
    device_name: str

    def evaluate_shell(
        self,
        points: np.ndarray,
        spec: TPMSFieldSpec,
        max_tile_points: int | None,
    ) -> np.ndarray: ...

    def evaluate_gradient_shell(
        self,
        points: np.ndarray,
        spec: TPMSFieldSpec,
        gradient: TPMSGradientSpec,
        max_tile_points: int | None,
    ) -> np.ndarray: ...

    def evaluate_transition(
        self,
        points: np.ndarray,
        g_spec: TPMSFieldSpec,
        d_spec: TPMSFieldSpec,
        plane_point: np.ndarray,
        plane_normal: np.ndarray,
        transition: TransitionBlendSpec,
        max_tile_points: int | None,
    ) -> np.ndarray: ...


def _tile_ranges(
    point_count: int,
    max_tile_points: int | None,
) -> list[tuple[int, int]]:
    if point_count == 0:
        return []
    if max_tile_points is None:
        return [(0, point_count)]
    if max_tile_points < 1:
        raise ValueError("max_tile_points must be positive or None")
    return [
        (start, min(start + max_tile_points, point_count))
        for start in range(0, point_count, max_tile_points)
    ]


def _validated_points(points: np.ndarray) -> np.ndarray:
    values = np.asarray(points)
    if values.ndim != 2 or values.shape[1] != 3:
        raise ValueError("points must have shape (N, 3)")
    if not np.isfinite(values).all():
        raise ValueError("points must contain only finite values")
    return values


def _validated_plane(
    plane_point: np.ndarray,
    plane_normal: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    point = np.asarray(plane_point, dtype=np.float32).reshape(3)
    normal = np.asarray(plane_normal, dtype=np.float32).reshape(3)
    if not np.isfinite(point).all() or not np.isfinite(normal).all():
        raise ValueError("transition plane must contain finite values")
    length = float(np.linalg.norm(normal))
    if length <= 1e-12:
        raise ValueError("transition plane normal must be non-zero")
    return point, np.ascontiguousarray(normal / length, dtype=np.float32)


def evaluate_tpms_field_and_gradient(
    phase: np.ndarray,
    kind: TPMSKind,
) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Evaluate a supported TPMS field and its derivatives in phase space.

    ``phase`` contains the dimensionless arguments of the trigonometric
    functions.  The returned derivatives are with respect to those phase
    arguments; callers must apply the physical frequencies when converting
    them to spatial gradients.
    """

    values = np.asarray(phase, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != 3:
        raise ValueError("phase must have shape (N, 3)")
    if kind not in TPMS_KINDS:
        raise ValueError("unsupported TPMS kind")
    x, y, z = values[:, 0], values[:, 1], values[:, 2]
    sx, sy, sz = np.sin(x), np.sin(y), np.sin(z)
    cx, cy, cz = np.cos(x), np.cos(y), np.cos(z)
    if kind == "G":
        field = sx * cy + sy * cz + sz * cx
        gx = cx * cy - sz * sx
        gy = -sx * sy + cy * cz
        gz = -sy * sz + cz * cx
    elif kind == "D":
        field = sx * sy * sz + sx * cy * cz + cx * sy * cz + cx * cy * sz
        gx = cx * sy * sz + cx * cy * cz - sx * sy * cz - sx * cy * sz
        gy = sx * cy * sz - sx * sy * cz + cx * cy * cz - cx * sy * sz
        gz = sx * sy * cz - sx * cy * sz - cx * sy * sz + cx * cy * cz
    elif kind == "IWP":
        field = cx + cy + cz - 2.0 * cx * cy * cz
        gx = sx * (2.0 * cy * cz - 1.0)
        gy = sy * (2.0 * cx * cz - 1.0)
        gz = sz * (2.0 * cx * cy - 1.0)
    elif kind == "Primitive":
        field = cx + cy + cz
        gx, gy, gz = -sx, -sy, -sz
    else:
        field = 3.0 * (cx + cy + cz) + 4.0 * cx * cy * cz
        gx = -sx * (3.0 + 4.0 * cy * cz)
        gy = -sy * (3.0 + 4.0 * cx * cz)
        gz = -sz * (3.0 + 4.0 * cx * cy)
    return (
        np.asarray(field, dtype=np.float32),
        (
            np.asarray(gx, dtype=np.float32),
            np.asarray(gy, dtype=np.float32),
            np.asarray(gz, dtype=np.float32),
        ),
    )


def _gradient_profile_numpy(
    coordinate: np.ndarray,
    gradient: TPMSGradientSpec,
) -> np.ndarray:
    u = np.clip(np.asarray(coordinate, dtype=np.float32), 0.0, 1.0)
    if gradient.mode == "linear":
        result = u
    elif gradient.mode == "power":
        result = np.power(u, np.float32(gradient.power))
    elif gradient.mode == "sigmoid":
        argument = np.clip(
            np.float32(gradient.sigmoid_sharpness) * (u - np.float32(0.5)),
            -60.0,
            60.0,
        )
        result = 1.0 / (1.0 + np.exp(-argument))
    else:
        result = np.floor(u * np.float32(gradient.layers)) / np.float32(
            gradient.layers - 1
        )
        result = np.where(u >= 1.0, 1.0, result)
    return np.clip(np.asarray(result, dtype=np.float32), 0.0, 1.0)


def _gradient_profile_derivative_numpy(
    coordinate: np.ndarray,
    gradient: TPMSGradientSpec,
) -> np.ndarray:
    u = np.clip(np.asarray(coordinate, dtype=np.float32), 0.0, 1.0)
    if gradient.mode == "linear":
        return np.ones_like(u)
    if gradient.mode == "power":
        return np.float32(gradient.power) * np.power(
            u,
            np.float32(max(gradient.power - 1.0, 0.0)),
        )
    if gradient.mode == "sigmoid":
        profile = _gradient_profile_numpy(u, gradient)
        return np.float32(gradient.sigmoid_sharpness) * profile * (1.0 - profile)
    return np.zeros_like(u)


class NumpyTPMSAdapter:
    """CPU reference implementation of TPMS shell-field evaluation."""

    name = "NumPy CPU"
    available = True
    device_name = "CPU"

    def evaluate_shell(
        self,
        points: np.ndarray,
        spec: TPMSFieldSpec,
        max_tile_points: int | None,
    ) -> np.ndarray:
        spec.validate()
        point_array = _validated_points(points)
        result = np.empty(len(point_array), dtype=np.float32)
        frequency = np.asarray(
            2.0 * np.pi / np.asarray(spec.cell_size_mm, dtype=np.float32),
            dtype=np.float32,
        )
        if frequency.ndim == 0:
            frequency = np.repeat(frequency, 3)
        level = np.float32(spec.level)
        half_wall = np.float32(0.5 * spec.wall_thickness_mm)
        origin = np.asarray(spec.origin_mm, dtype=np.float32)
        axes = np.asarray(spec.axes_world, dtype=np.float32)
        for start, end in _tile_ranges(len(point_array), max_tile_points):
            phase = (
                (np.asarray(point_array[start:end], dtype=np.float32) - origin) @ axes.T
            ) * frequency
            field, (gx, gy, gz) = evaluate_tpms_field_and_gradient(
                phase,
                spec.kind,
            )
            gradient_norm = np.sqrt(
                (frequency[0] * gx) ** 2
                + (frequency[1] * gy) ** 2
                + (frequency[2] * gz) ** 2
            )
            safe_gradient = np.maximum(
                gradient_norm,
                np.float32(1e-6),
            )
            result[start:end] = np.abs(field - level) / safe_gradient - half_wall
        return result

    def evaluate_gradient_shell(
        self,
        points: np.ndarray,
        spec: TPMSFieldSpec,
        gradient: TPMSGradientSpec,
        max_tile_points: int | None,
    ) -> np.ndarray:
        spec.validate()
        gradient.validate()
        point_array = _validated_points(points)
        result = np.empty(len(point_array), dtype=np.float32)
        cell_sizes = np.asarray(spec.cell_size_mm, dtype=np.float32)
        if cell_sizes.ndim == 0:
            cell_sizes = np.repeat(cell_sizes, 3)
        frequency = np.float32(2.0 * np.pi) / cell_sizes
        origin = np.asarray(spec.origin_mm, dtype=np.float32)
        axes = np.asarray(spec.axes_world, dtype=np.float32)
        lower = np.float32(gradient.coordinate_min_mm)
        span = np.float32(gradient.coordinate_max_mm - gradient.coordinate_min_mm)
        base_thickness = np.float32(spec.wall_thickness_mm)
        soft_thickness = np.float32(
            gradient.thickness_soft_mm
            if gradient.thickness_soft_mm is not None
            else spec.wall_thickness_mm
        )
        stiff_thickness = np.float32(
            gradient.thickness_stiff_mm
            if gradient.thickness_stiff_mm is not None
            else spec.wall_thickness_mm
        )
        base_offset = np.float32(spec.level)
        soft_offset = np.float32(
            gradient.offset_soft if gradient.offset_soft is not None else spec.level
        )
        stiff_offset = np.float32(
            gradient.offset_stiff if gradient.offset_stiff is not None else spec.level
        )
        for start, end in _tile_ranges(len(point_array), max_tile_points):
            local = (
                np.asarray(point_array[start:end], dtype=np.float32) - origin
            ) @ axes.T
            phase = local * frequency
            field, (gx, gy, gz) = evaluate_tpms_field_and_gradient(phase, spec.kind)
            coordinate = (local[:, gradient.axis_index] - lower) / span
            profile = _gradient_profile_numpy(coordinate, gradient)
            thickness = (
                base_thickness
                if not gradient.use_thickness_gradient
                else soft_thickness + (stiff_thickness - soft_thickness) * profile
            )
            offset = (
                base_offset
                if not gradient.use_offset_gradient
                else soft_offset + (stiff_offset - soft_offset) * profile
            )
            shifted = field - offset
            if gradient.wall_thickness_method == "field_threshold":
                result[start:end] = np.abs(shifted) - thickness * np.float32(
                    np.pi / np.min(cell_sizes)
                )
                continue
            gradient_norm = np.sqrt(
                (frequency[0] * gx) ** 2
                + (frequency[1] * gy) ** 2
                + (frequency[2] * gz) ** 2
            )
            if gradient.use_offset_gradient:
                derivative = _gradient_profile_derivative_numpy(coordinate, gradient)
                offset_slope = (stiff_offset - soft_offset) * derivative / span
                gradient_norm = np.sqrt(gradient_norm**2 + offset_slope**2)
            result[start:end] = (
                np.abs(shifted) / np.maximum(gradient_norm, np.float32(1.0e-6))
                - np.float32(0.5) * thickness
            )
        return result

    def evaluate_transition(
        self,
        points: np.ndarray,
        g_spec: TPMSFieldSpec,
        d_spec: TPMSFieldSpec,
        plane_point: np.ndarray,
        plane_normal: np.ndarray,
        transition: TransitionBlendSpec,
        max_tile_points: int | None,
    ) -> np.ndarray:
        g_spec.validate()
        d_spec.validate()
        transition.validate()
        point_array = _validated_points(points)
        point, normal = _validated_plane(plane_point, plane_normal)
        result = np.empty(len(point_array), dtype=np.float32)
        for start, end in _tile_ranges(len(point_array), max_tile_points):
            tile_points = np.asarray(point_array[start:end], dtype=np.float32)
            g_sdf = self.evaluate_shell(tile_points, g_spec, None)
            d_sdf = self.evaluate_shell(tile_points, d_spec, None)
            signed_distance = (tile_points - point) @ normal
            g_weight, d_weight = sigmoid_transition_weights(
                signed_distance,
                transition,
            )
            result[start:end] = g_weight * g_sdf + d_weight * d_sdf
        return result


try:
    from numba import cuda as _numba_cuda

    @_numba_cuda.jit(device=True)
    def _cuda_shell_value(
        px,
        py,
        pz,
        kind_code,
        frequency_x,
        frequency_y,
        frequency_z,
        level,
        half_wall,
    ):
        x = px * frequency_x
        y = py * frequency_y
        z = pz * frequency_z
        sx, sy, sz = math.sin(x), math.sin(y), math.sin(z)
        cx, cy, cz = math.cos(x), math.cos(y), math.cos(z)
        if kind_code == 0:
            field = sx * cy + sy * cz + sz * cx
            gx = cx * cy - sz * sx
            gy = -sx * sy + cy * cz
            gz = -sy * sz + cz * cx
        elif kind_code == 1:
            field = sx * sy * sz + sx * cy * cz + cx * sy * cz + cx * cy * sz
            gx = cx * sy * sz + cx * cy * cz - sx * sy * cz - sx * cy * sz
            gy = sx * cy * sz - sx * sy * cz + cx * cy * cz - cx * sy * sz
            gz = sx * sy * cz - sx * cy * sz - cx * sy * sz + cx * cy * cz
        elif kind_code == 2:
            field = cx + cy + cz - 2.0 * cx * cy * cz
            gx = sx * (2.0 * cy * cz - 1.0)
            gy = sy * (2.0 * cx * cz - 1.0)
            gz = sz * (2.0 * cx * cy - 1.0)
        elif kind_code == 3:
            field = cx + cy + cz
            gx = -sx
            gy = -sy
            gz = -sz
        else:
            field = 3.0 * (cx + cy + cz) + 4.0 * cx * cy * cz
            gx = -sx * (3.0 + 4.0 * cy * cz)
            gy = -sy * (3.0 + 4.0 * cx * cz)
            gz = -sz * (3.0 + 4.0 * cx * cy)
        gradient = math.sqrt(
            (frequency_x * gx) * (frequency_x * gx)
            + (frequency_y * gy) * (frequency_y * gy)
            + (frequency_z * gz) * (frequency_z * gz)
        )
        if gradient < 1e-6:
            gradient = 1e-6
        return abs(field - level) / gradient - half_wall

    @_numba_cuda.jit
    def _cuda_shell_kernel(
        points,
        result,
        kind_code,
        frequency_x,
        frequency_y,
        frequency_z,
        origin_x,
        origin_y,
        origin_z,
        u_x,
        u_y,
        u_z,
        v_x,
        v_y,
        v_z,
        w_x,
        w_y,
        w_z,
        level,
        half_wall,
    ):
        index = _numba_cuda.grid(1)
        if index >= points.shape[0]:
            return
        dx = points[index, 0] - origin_x
        dy = points[index, 1] - origin_y
        dz = points[index, 2] - origin_z
        result[index] = _cuda_shell_value(
            dx * u_x + dy * u_y + dz * u_z,
            dx * v_x + dy * v_y + dz * v_z,
            dx * w_x + dy * w_y + dz * w_z,
            kind_code,
            frequency_x,
            frequency_y,
            frequency_z,
            level,
            half_wall,
        )

    @_numba_cuda.jit(device=True)
    def _cuda_gradient_shell_value(
        px,
        py,
        pz,
        gradient_coordinate,
        kind_code,
        frequency_x,
        frequency_y,
        frequency_z,
        level,
        base_thickness,
        coordinate_min,
        inverse_coordinate_span,
        gradient_mode_code,
        power,
        layers,
        sigmoid_sharpness,
        use_thickness_gradient,
        thickness_soft,
        thickness_stiff,
        use_offset_gradient,
        offset_soft,
        offset_stiff,
        wall_method_code,
        minimum_cell_size,
    ):
        x = px * frequency_x
        y = py * frequency_y
        z = pz * frequency_z
        sx, sy, sz = math.sin(x), math.sin(y), math.sin(z)
        cx, cy, cz = math.cos(x), math.cos(y), math.cos(z)
        if kind_code == 0:
            field = sx * cy + sy * cz + sz * cx
            gx = cx * cy - sz * sx
            gy = -sx * sy + cy * cz
            gz = -sy * sz + cz * cx
        elif kind_code == 1:
            field = sx * sy * sz + sx * cy * cz + cx * sy * cz + cx * cy * sz
            gx = cx * sy * sz + cx * cy * cz - sx * sy * cz - sx * cy * sz
            gy = sx * cy * sz - sx * sy * cz + cx * cy * cz - cx * sy * sz
            gz = sx * sy * cz - sx * cy * sz - cx * sy * sz + cx * cy * cz
        elif kind_code == 2:
            field = cx + cy + cz - 2.0 * cx * cy * cz
            gx = sx * (2.0 * cy * cz - 1.0)
            gy = sy * (2.0 * cx * cz - 1.0)
            gz = sz * (2.0 * cx * cy - 1.0)
        elif kind_code == 3:
            field = cx + cy + cz
            gx = -sx
            gy = -sy
            gz = -sz
        else:
            field = 3.0 * (cx + cy + cz) + 4.0 * cx * cy * cz
            gx = -sx * (3.0 + 4.0 * cy * cz)
            gy = -sy * (3.0 + 4.0 * cx * cz)
            gz = -sz * (3.0 + 4.0 * cx * cy)

        coordinate = (gradient_coordinate - coordinate_min) * inverse_coordinate_span
        if coordinate < 0.0:
            coordinate = 0.0
        elif coordinate > 1.0:
            coordinate = 1.0
        if gradient_mode_code == 0:
            profile = coordinate
            profile_derivative = 1.0
        elif gradient_mode_code == 1:
            profile = math.pow(coordinate, power)
            derivative_power = power - 1.0
            if derivative_power < 0.0:
                derivative_power = 0.0
            profile_derivative = power * math.pow(coordinate, derivative_power)
        elif gradient_mode_code == 2:
            argument = sigmoid_sharpness * (coordinate - 0.5)
            if argument < -60.0:
                argument = -60.0
            elif argument > 60.0:
                argument = 60.0
            profile = 1.0 / (1.0 + math.exp(-argument))
            profile_derivative = sigmoid_sharpness * profile * (1.0 - profile)
        else:
            profile = math.floor(coordinate * layers) / (layers - 1.0)
            if coordinate >= 1.0:
                profile = 1.0
            profile_derivative = 0.0

        thickness = base_thickness
        if use_thickness_gradient:
            thickness = thickness_soft + (thickness_stiff - thickness_soft) * profile
        offset = level
        if use_offset_gradient:
            offset = offset_soft + (offset_stiff - offset_soft) * profile
        shifted = field - offset
        if wall_method_code == 1:
            return abs(shifted) - thickness * math.pi / minimum_cell_size

        gradient_norm = math.sqrt(
            (frequency_x * gx) * (frequency_x * gx)
            + (frequency_y * gy) * (frequency_y * gy)
            + (frequency_z * gz) * (frequency_z * gz)
        )
        if use_offset_gradient:
            offset_slope = (
                (offset_stiff - offset_soft)
                * profile_derivative
                * inverse_coordinate_span
            )
            gradient_norm = math.sqrt(
                gradient_norm * gradient_norm + offset_slope * offset_slope
            )
        if gradient_norm < 1e-6:
            gradient_norm = 1e-6
        return abs(shifted) / gradient_norm - 0.5 * thickness

    @_numba_cuda.jit
    def _cuda_gradient_shell_kernel(
        points,
        result,
        kind_code,
        frequency_x,
        frequency_y,
        frequency_z,
        origin_x,
        origin_y,
        origin_z,
        u_x,
        u_y,
        u_z,
        v_x,
        v_y,
        v_z,
        w_x,
        w_y,
        w_z,
        level,
        base_thickness,
        gradient_axis_index,
        coordinate_min,
        inverse_coordinate_span,
        gradient_mode_code,
        power,
        layers,
        sigmoid_sharpness,
        use_thickness_gradient,
        thickness_soft,
        thickness_stiff,
        use_offset_gradient,
        offset_soft,
        offset_stiff,
        wall_method_code,
        minimum_cell_size,
    ):
        index = _numba_cuda.grid(1)
        if index >= points.shape[0]:
            return
        dx = points[index, 0] - origin_x
        dy = points[index, 1] - origin_y
        dz = points[index, 2] - origin_z
        local_u = dx * u_x + dy * u_y + dz * u_z
        local_v = dx * v_x + dy * v_y + dz * v_z
        local_w = dx * w_x + dy * w_y + dz * w_z
        gradient_coordinate = local_u
        if gradient_axis_index == 1:
            gradient_coordinate = local_v
        elif gradient_axis_index == 2:
            gradient_coordinate = local_w
        result[index] = _cuda_gradient_shell_value(
            local_u,
            local_v,
            local_w,
            gradient_coordinate,
            kind_code,
            frequency_x,
            frequency_y,
            frequency_z,
            level,
            base_thickness,
            coordinate_min,
            inverse_coordinate_span,
            gradient_mode_code,
            power,
            layers,
            sigmoid_sharpness,
            use_thickness_gradient,
            thickness_soft,
            thickness_stiff,
            use_offset_gradient,
            offset_soft,
            offset_stiff,
            wall_method_code,
            minimum_cell_size,
        )

    @_numba_cuda.jit
    def _cuda_transition_kernel(
        points,
        result,
        g_kind_code,
        g_frequency_x,
        g_frequency_y,
        g_frequency_z,
        g_origin_x,
        g_origin_y,
        g_origin_z,
        g_u_x,
        g_u_y,
        g_u_z,
        g_v_x,
        g_v_y,
        g_v_z,
        g_w_x,
        g_w_y,
        g_w_z,
        g_level,
        g_half_wall,
        d_kind_code,
        d_frequency_x,
        d_frequency_y,
        d_frequency_z,
        d_origin_x,
        d_origin_y,
        d_origin_z,
        d_u_x,
        d_u_y,
        d_u_z,
        d_v_x,
        d_v_y,
        d_v_z,
        d_w_x,
        d_w_y,
        d_w_z,
        d_level,
        d_half_wall,
        point_x,
        point_y,
        point_z,
        normal_x,
        normal_y,
        normal_z,
        half_width,
        center_offset,
        exponent,
        g_on_negative_side,
    ):
        index = _numba_cuda.grid(1)
        if index >= points.shape[0]:
            return
        px = points[index, 0]
        py = points[index, 1]
        pz = points[index, 2]
        g_dx = px - g_origin_x
        g_dy = py - g_origin_y
        g_dz = pz - g_origin_z
        g_sdf = _cuda_shell_value(
            g_dx * g_u_x + g_dy * g_u_y + g_dz * g_u_z,
            g_dx * g_v_x + g_dy * g_v_y + g_dz * g_v_z,
            g_dx * g_w_x + g_dy * g_w_y + g_dz * g_w_z,
            g_kind_code, g_frequency_x, g_frequency_y, g_frequency_z,
            g_level, g_half_wall
        )
        d_dx = px - d_origin_x
        d_dy = py - d_origin_y
        d_dz = pz - d_origin_z
        d_sdf = _cuda_shell_value(
            d_dx * d_u_x + d_dy * d_u_y + d_dz * d_u_z,
            d_dx * d_v_x + d_dy * d_v_y + d_dz * d_v_z,
            d_dx * d_w_x + d_dy * d_w_y + d_dz * d_w_z,
            d_kind_code, d_frequency_x, d_frequency_y, d_frequency_z,
            d_level, d_half_wall
        )
        signed_distance = (
            (px - point_x) * normal_x
            + (py - point_y) * normal_y
            + (pz - point_z) * normal_z
        )
        normalized = (signed_distance - center_offset) / half_width
        if normalized < 0.0:
            shaped = -math.pow(-normalized, exponent)
        else:
            shaped = math.pow(normalized, exponent)
        argument = math.log(9.0) * shaped
        if argument < -60.0:
            argument = -60.0
        elif argument > 60.0:
            argument = 60.0
        d_weight = 1.0 / (1.0 + math.exp(-argument))
        if not g_on_negative_side:
            d_weight = 1.0 - d_weight
        result[index] = (1.0 - d_weight) * g_sdf + d_weight * d_sdf

except ImportError:
    _numba_cuda = None
    _cuda_shell_value = None
    _cuda_shell_kernel = None
    _cuda_gradient_shell_value = None
    _cuda_gradient_shell_kernel = None
    _cuda_transition_kernel = None


class NumbaCudaTPMSAdapter:
    """CUDA implementation with bounded host/device transfer tiles."""

    name = "Numba CUDA"

    def __init__(self) -> None:
        self.available = False
        self.device_name = "CUDA unavailable"
        self.unavailable_reason = "Numba CUDA is not installed"
        if _numba_cuda is None:
            return
        probe = probe_numba_cuda()
        if not probe.available:
            self.unavailable_reason = probe.reason
            return
        try:
            self.available = bool(_numba_cuda.is_available())
            if self.available:
                self.device_name = probe.device_name
                self.unavailable_reason = None
            else:
                self.unavailable_reason = "CUDA runtime is unavailable"
        except Exception as exc:
            self.available = False
            self.unavailable_reason = f"{type(exc).__name__}: {exc}"

    def _launch_shell_device(self, device_points, device_result, spec: TPMSFieldSpec) -> None:
        """Launch one shell evaluation without synchronizing or copying."""

        if not self.available or _numba_cuda is None or _cuda_shell_kernel is None:
            raise RuntimeError("CUDA backend is unavailable")
        spec.validate()
        frequency = np.asarray(
            2.0 * np.pi / np.asarray(spec.cell_size_mm, dtype=np.float32),
            dtype=np.float32,
        )
        if frequency.ndim == 0:
            frequency = np.repeat(frequency, 3)
        origin = np.asarray(spec.origin_mm, dtype=np.float32)
        axes = np.asarray(spec.axes_world, dtype=np.float32)
        threads = 256
        blocks = (device_points.shape[0] + threads - 1) // threads
        _cuda_shell_kernel[blocks, threads](
            device_points,
            device_result,
            np.int32(_TPMS_KIND_CODES[spec.kind]),
            frequency[0],
            frequency[1],
            frequency[2],
            origin[0],
            origin[1],
            origin[2],
            axes[0, 0],
            axes[0, 1],
            axes[0, 2],
            axes[1, 0],
            axes[1, 1],
            axes[1, 2],
            axes[2, 0],
            axes[2, 1],
            axes[2, 2],
            np.float32(spec.level),
            np.float32(0.5 * spec.wall_thickness_mm),
        )

    def _launch_gradient_shell_device(
        self,
        device_points,
        device_result,
        spec: TPMSFieldSpec,
        gradient: TPMSGradientSpec,
    ) -> None:
        """Launch one gradient shell evaluation without host round trips."""

        if (
            not self.available
            or _numba_cuda is None
            or _cuda_gradient_shell_kernel is None
        ):
            raise RuntimeError("CUDA gradient backend is unavailable")
        spec.validate()
        gradient.validate()
        cell_sizes = np.asarray(spec.cell_size_mm, dtype=np.float32)
        if cell_sizes.ndim == 0:
            cell_sizes = np.repeat(cell_sizes, 3)
        frequency = np.float32(2.0 * np.pi) / cell_sizes
        origin = np.asarray(spec.origin_mm, dtype=np.float32)
        axes = np.asarray(spec.axes_world, dtype=np.float32)
        soft_thickness = np.float32(
            gradient.thickness_soft_mm
            if gradient.thickness_soft_mm is not None
            else spec.wall_thickness_mm
        )
        stiff_thickness = np.float32(
            gradient.thickness_stiff_mm
            if gradient.thickness_stiff_mm is not None
            else spec.wall_thickness_mm
        )
        soft_offset = np.float32(
            gradient.offset_soft if gradient.offset_soft is not None else spec.level
        )
        stiff_offset = np.float32(
            gradient.offset_stiff if gradient.offset_stiff is not None else spec.level
        )
        threads = 256
        blocks = (device_points.shape[0] + threads - 1) // threads
        _cuda_gradient_shell_kernel[blocks, threads](
            device_points,
            device_result,
            np.int32(_TPMS_KIND_CODES[spec.kind]),
            frequency[0],
            frequency[1],
            frequency[2],
            origin[0],
            origin[1],
            origin[2],
            axes[0, 0],
            axes[0, 1],
            axes[0, 2],
            axes[1, 0],
            axes[1, 1],
            axes[1, 2],
            axes[2, 0],
            axes[2, 1],
            axes[2, 2],
            np.float32(spec.level),
            np.float32(spec.wall_thickness_mm),
            np.int32(gradient.axis_index),
            np.float32(gradient.coordinate_min_mm),
            np.float32(
                1.0 / (gradient.coordinate_max_mm - gradient.coordinate_min_mm)
            ),
            np.int32(_GRADIENT_MODE_CODES[gradient.mode]),
            np.float32(gradient.power),
            np.int32(gradient.layers),
            np.float32(gradient.sigmoid_sharpness),
            bool(gradient.use_thickness_gradient),
            soft_thickness,
            stiff_thickness,
            bool(gradient.use_offset_gradient),
            soft_offset,
            stiff_offset,
            np.int32(_WALL_METHOD_CODES[gradient.wall_thickness_method]),
            np.float32(np.min(cell_sizes)),
        )

    def evaluate_shell(
        self,
        points: np.ndarray,
        spec: TPMSFieldSpec,
        max_tile_points: int | None,
    ) -> np.ndarray:
        if not self.available or _numba_cuda is None or _cuda_shell_kernel is None:
            raise RuntimeError("CUDA backend is unavailable")
        spec.validate()
        point_array = _validated_points(points)
        result = np.empty(len(point_array), dtype=np.float32)
        for start, end in _tile_ranges(len(point_array), max_tile_points):
            host_points = np.ascontiguousarray(point_array[start:end], dtype=np.float32)
            device_points = _numba_cuda.to_device(host_points)
            device_result = _numba_cuda.device_array(len(host_points), dtype=np.float32)
            self._launch_shell_device(device_points, device_result, spec)
            _numba_cuda.synchronize()
            device_result.copy_to_host(result[start:end])
            del device_points, device_result
        return result

    def evaluate_gradient_shell(
        self,
        points: np.ndarray,
        spec: TPMSFieldSpec,
        gradient: TPMSGradientSpec,
        max_tile_points: int | None,
    ) -> np.ndarray:
        if (
            not self.available
            or _numba_cuda is None
            or _cuda_gradient_shell_kernel is None
        ):
            raise RuntimeError("CUDA gradient backend is unavailable")
        spec.validate()
        gradient.validate()
        point_array = _validated_points(points)
        result = np.empty(len(point_array), dtype=np.float32)
        for start, end in _tile_ranges(len(point_array), max_tile_points):
            host_points = np.ascontiguousarray(point_array[start:end], dtype=np.float32)
            device_points = _numba_cuda.to_device(host_points)
            device_result = _numba_cuda.device_array(len(host_points), dtype=np.float32)
            self._launch_gradient_shell_device(
                device_points,
                device_result,
                spec,
                gradient,
            )
            _numba_cuda.synchronize()
            device_result.copy_to_host(result[start:end])
            del device_points, device_result
        return result

    def evaluate_transition(
        self,
        points: np.ndarray,
        g_spec: TPMSFieldSpec,
        d_spec: TPMSFieldSpec,
        plane_point: np.ndarray,
        plane_normal: np.ndarray,
        transition: TransitionBlendSpec,
        max_tile_points: int | None,
    ) -> np.ndarray:
        if (
            not self.available
            or _numba_cuda is None
            or _cuda_transition_kernel is None
        ):
            raise RuntimeError("CUDA backend is unavailable")
        g_spec.validate()
        d_spec.validate()
        transition.validate()
        point_array = _validated_points(points)
        point, normal = _validated_plane(plane_point, plane_normal)
        result = np.empty(len(point_array), dtype=np.float32)
        g_frequency = np.asarray(
            2.0 * np.pi / np.asarray(g_spec.cell_size_mm, dtype=np.float32),
            dtype=np.float32,
        )
        d_frequency = np.asarray(
            2.0 * np.pi / np.asarray(d_spec.cell_size_mm, dtype=np.float32),
            dtype=np.float32,
        )
        if g_frequency.ndim == 0:
            g_frequency = np.repeat(g_frequency, 3)
        if d_frequency.ndim == 0:
            d_frequency = np.repeat(d_frequency, 3)
        g_origin = np.asarray(g_spec.origin_mm, dtype=np.float32)
        d_origin = np.asarray(d_spec.origin_mm, dtype=np.float32)
        g_axes = np.asarray(g_spec.axes_world, dtype=np.float32)
        d_axes = np.asarray(d_spec.axes_world, dtype=np.float32)
        threads_per_block = 256
        for start, end in _tile_ranges(len(point_array), max_tile_points):
            host_points = np.ascontiguousarray(point_array[start:end], dtype=np.float32)
            device_points = _numba_cuda.to_device(host_points)
            device_result = _numba_cuda.device_array(len(host_points), dtype=np.float32)
            block_count = (len(host_points) + threads_per_block - 1) // threads_per_block
            _cuda_transition_kernel[block_count, threads_per_block](
                device_points,
                device_result,
                np.int32(_TPMS_KIND_CODES[g_spec.kind]),
                g_frequency[0],
                g_frequency[1],
                g_frequency[2],
                g_origin[0],
                g_origin[1],
                g_origin[2],
                g_axes[0, 0],
                g_axes[0, 1],
                g_axes[0, 2],
                g_axes[1, 0],
                g_axes[1, 1],
                g_axes[1, 2],
                g_axes[2, 0],
                g_axes[2, 1],
                g_axes[2, 2],
                np.float32(g_spec.level),
                np.float32(0.5 * g_spec.wall_thickness_mm),
                np.int32(_TPMS_KIND_CODES[d_spec.kind]),
                d_frequency[0],
                d_frequency[1],
                d_frequency[2],
                d_origin[0],
                d_origin[1],
                d_origin[2],
                d_axes[0, 0],
                d_axes[0, 1],
                d_axes[0, 2],
                d_axes[1, 0],
                d_axes[1, 1],
                d_axes[1, 2],
                d_axes[2, 0],
                d_axes[2, 1],
                d_axes[2, 2],
                np.float32(d_spec.level),
                np.float32(0.5 * d_spec.wall_thickness_mm),
                point[0],
                point[1],
                point[2],
                normal[0],
                normal[1],
                normal[2],
                np.float32(0.5 * transition.transition_width_mm),
                np.float32(transition.center_offset_mm),
                np.float32(1.0 / transition.sigmoid_sharpness),
                bool(transition.g_on_negative_side),
            )
            _numba_cuda.synchronize()
            device_result.copy_to_host(result[start:end])
            del device_points, device_result
        return result


class AutomaticTPMSBackend:
    """Select a CUDA adapter when available, otherwise use the CPU adapter."""

    def __init__(
        self,
        cpu_adapter: TPMSComputeAdapter,
        gpu_adapter: TPMSComputeAdapter | None,
    ) -> None:
        self._cpu_adapter = cpu_adapter
        self._gpu_adapter = gpu_adapter
        self._active_adapter = (
            gpu_adapter
            if gpu_adapter is not None and gpu_adapter.available
            else cpu_adapter
        )
        self._fallback_reason: str | None = (
            None
            if gpu_adapter is None or gpu_adapter.available
            else getattr(gpu_adapter, "unavailable_reason", None)
        )

    @property
    def status(self) -> NumericalBackendStatus:
        adapter = self._active_adapter
        return NumericalBackendStatus(
            active_backend=adapter.name,
            using_gpu=adapter is self._gpu_adapter,
            device_name=adapter.device_name,
            fallback_reason=self._fallback_reason,
        )

    def _run_with_fallback(self, method_name: str, *args):
        try:
            return getattr(self._active_adapter, method_name)(*args)
        except (TypeError, ValueError):
            raise
        except Exception as exc:
            if self._active_adapter is not self._gpu_adapter:
                raise
            self._fallback_reason = f"{type(exc).__name__}: {exc}"
            self._active_adapter = self._cpu_adapter
            return getattr(self._cpu_adapter, method_name)(*args)

    def evaluate_shell(
        self,
        points: np.ndarray,
        spec: TPMSFieldSpec,
        max_tile_points: int | None = 250_000,
    ) -> np.ndarray:
        spec.validate()
        point_array = _validated_points(points)
        raw_values = self._run_with_fallback(
            "evaluate_shell",
            point_array,
            spec,
            max_tile_points,
        )
        values = np.asarray(raw_values, dtype=np.float32)
        if values.shape != (len(point_array),) or not np.isfinite(values).all():
            raise RuntimeError("numerical backend returned an invalid TPMS field")
        return values

    def evaluate_gradient_shell(
        self,
        points: np.ndarray,
        spec: TPMSFieldSpec,
        gradient: TPMSGradientSpec,
        max_tile_points: int | None = 250_000,
    ) -> np.ndarray:
        spec.validate()
        gradient.validate()
        point_array = _validated_points(points)
        raw_values = self._run_with_fallback(
            "evaluate_gradient_shell",
            point_array,
            spec,
            gradient,
            max_tile_points,
        )
        values = np.asarray(raw_values, dtype=np.float32)
        if values.shape != (len(point_array),) or not np.isfinite(values).all():
            raise RuntimeError("numerical backend returned an invalid gradient TPMS field")
        return values

    def evaluate_transition(
        self,
        points: np.ndarray,
        g_spec: TPMSFieldSpec,
        d_spec: TPMSFieldSpec,
        plane_point: np.ndarray,
        plane_normal: np.ndarray,
        transition: TransitionBlendSpec,
        max_tile_points: int | None = 250_000,
    ) -> np.ndarray:
        g_spec.validate()
        d_spec.validate()
        transition.validate()
        point_array = _validated_points(points)
        raw_values = self._run_with_fallback(
            "evaluate_transition",
            point_array,
            g_spec,
            d_spec,
            plane_point,
            plane_normal,
            transition,
            max_tile_points,
        )
        values = np.asarray(raw_values, dtype=np.float32)
        if values.shape != (len(point_array),) or not np.isfinite(values).all():
            raise RuntimeError("numerical backend returned an invalid transition field")
        return values


_DEFAULT_BACKEND: AutomaticTPMSBackend | None = None
_DEFAULT_BACKEND_LOCK = threading.Lock()


def get_default_tpms_backend() -> AutomaticTPMSBackend:
    """Return the process-wide backend, creating CUDA support on first use."""

    global _DEFAULT_BACKEND
    if _DEFAULT_BACKEND is None:
        with _DEFAULT_BACKEND_LOCK:
            if _DEFAULT_BACKEND is None:
                _DEFAULT_BACKEND = AutomaticTPMSBackend(
                    cpu_adapter=NumpyTPMSAdapter(),
                    gpu_adapter=NumbaCudaTPMSAdapter(),
                )
    return _DEFAULT_BACKEND
