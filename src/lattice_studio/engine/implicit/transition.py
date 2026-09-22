"""Provider-agnostic implicit lattice transitions with bounded topology correction."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from lattice_studio.engine.implicit.field import ImplicitBody, StageReporter
from lattice_studio.domain.transition import TransitionSpec, TransitionWeightKind


@dataclass(frozen=True)
class TransitionOperand:
    """One implicit lattice source consumed by the transition module."""

    name: str
    body: ImplicitBody
    characteristic_feature_mm: float
    cell_spacing_mm: tuple[float, float, float]
    frame_axes_world: tuple[tuple[float, float, float], ...] = (
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
    )
    backend_name: str = "CPU"
    supports_gpu: bool = False

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("transition operand name must not be empty")
        feature = float(self.characteristic_feature_mm)
        spacing = np.asarray(self.cell_spacing_mm, dtype=np.float64)
        axes = np.asarray(self.frame_axes_world, dtype=np.float64)
        if not np.isfinite(feature) or feature <= 0.0:
            raise ValueError("characteristic_feature_mm must be positive")
        if spacing.shape != (3,) or not np.isfinite(spacing).all() or np.any(spacing <= 0.0):
            raise ValueError("cell_spacing_mm must contain three positive values")
        if axes.shape != (3, 3) or not np.isfinite(axes).all():
            raise ValueError("frame_axes_world must be a finite 3 by 3 matrix")
        if not np.allclose(axes @ axes.T, np.eye(3), rtol=1e-6, atol=1e-6):
            raise ValueError("frame_axes_world must be orthonormal")
        if not np.isclose(np.linalg.det(axes), 1.0, rtol=1e-6, atol=1e-6):
            raise ValueError("frame_axes_world must be right-handed")


@dataclass(frozen=True)
class TransitionDiagnostics:
    """Deterministic decisions made while building a transition evaluator."""

    resolved_weight_kind: str
    minimum_feature_mm: float
    recommended_width_mm: float
    width_below_recommendation: bool
    registration_translation_mm: tuple[float, float, float]
    execution_backend: str
    driver_name: str = "transition plane"
    field_driven: bool = False
    quality_validation_supported: bool = True


@dataclass(frozen=True)
class TransitionBuildResult:
    body: ImplicitBody
    field_body: ImplicitBody
    diagnostics: TransitionDiagnostics


@dataclass(frozen=True)
class TransitionQualityReport:
    """Non-blocking authoritative transition validation result."""

    cross_band_connected: bool
    minimum_feature_satisfied: bool
    isolated_fragment_count: int
    component_count: int
    inspection_spacing_mm: float
    inspected_samples: int
    issue_locations_mm: tuple[tuple[float, float, float], ...] = ()
    inspection_supported: bool = True
    status_message: str = ""

    @property
    def passed(self) -> bool:
        return bool(
            self.inspection_supported
            and
            self.cross_band_connected
            and self.minimum_feature_satisfied
            and self.isolated_fragment_count == 0
        )


def _validated_plane(
    plane_point_mm: tuple[float, float, float] | np.ndarray,
    plane_normal: tuple[float, float, float] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    point = np.asarray(plane_point_mm, dtype=np.float64)
    normal = np.asarray(plane_normal, dtype=np.float64)
    if point.shape != (3,) or normal.shape != (3,):
        raise ValueError("transition plane point and normal must have shape (3,)")
    if not np.isfinite(point).all() or not np.isfinite(normal).all():
        raise ValueError("transition plane must contain finite values")
    length = float(np.linalg.norm(normal))
    if length <= 1.0e-12:
        raise ValueError("transition plane normal must be non-zero")
    return point, normal / length


def _resolved_weight_kind(spec: TransitionSpec) -> str:
    return "smootherstep" if spec.weight_kind == "automatic" else spec.weight_kind


def transition_weights(
    signed_plane_distance: np.ndarray,
    spec: TransitionSpec,
) -> tuple[np.ndarray, np.ndarray]:
    """Map signed distance to clamped operand weights using a spatial Ramp."""

    spec.validate()
    distance = np.asarray(signed_plane_distance, dtype=np.float32)
    half_width = np.float32(0.5 * spec.width_mm)
    centered = distance - np.float32(spec.center_offset_mm)
    parameter = np.clip(
        (centered + half_width) / np.float32(spec.width_mm),
        0.0,
        1.0,
    ).astype(np.float32, copy=False)
    kind = _resolved_weight_kind(spec)
    if kind == "linear":
        second = parameter
    elif kind == "smoothstep":
        second = parameter * parameter * (3.0 - 2.0 * parameter)
    elif kind == "smootherstep":
        second = parameter**3 * (
            parameter * (parameter * 6.0 - 15.0) + 10.0
        )
    elif kind == "cosine":
        second = 0.5 - 0.5 * np.cos(np.pi * parameter)
    else:
        slope = np.float32(np.log(9.0) * spec.sigmoid_sharpness)
        raw = 1.0 / (1.0 + np.exp(np.clip(-slope * (2.0 * parameter - 1.0), -60.0, 60.0)))
        lower = np.float32(1.0 / (1.0 + np.exp(slope)))
        upper = np.float32(1.0 / (1.0 + np.exp(-slope)))
        second = (raw - lower) / np.maximum(upper - lower, np.float32(1.0e-7))
    second = np.asarray(second, dtype=np.float32)
    second[centered <= -half_width] = 0.0
    second[centered >= half_width] = 1.0
    if not spec.first_on_negative_side:
        second = np.float32(1.0) - second
    return np.float32(1.0) - second, second


def recommended_transition_width(
    first: TransitionOperand,
    second: TransitionOperand,
) -> float:
    """Estimate a non-binding width from feature, period, and Frame mismatch."""

    first_spacing = np.asarray(first.cell_spacing_mm, dtype=np.float64)
    second_spacing = np.asarray(second.cell_spacing_mm, dtype=np.float64)
    base_period = float(max(np.min(first_spacing), np.min(second_spacing)))
    size_mismatch = float(
        np.max(np.abs(np.log(np.maximum(first_spacing, 1.0e-9) / second_spacing)))
    )
    first_axes = np.asarray(first.frame_axes_world, dtype=np.float64)
    second_axes = np.asarray(second.frame_axes_world, dtype=np.float64)
    alignment = np.clip((np.trace(first_axes @ second_axes.T) - 1.0) * 0.5, -1.0, 1.0)
    angle_ratio = float(np.arccos(alignment) / np.pi)
    minimum_feature = min(
        first.characteristic_feature_mm,
        second.characteristic_feature_mm,
    )
    return float(
        max(
            3.0 * minimum_feature,
            base_period * (1.0 + 0.35 * size_mismatch + 0.5 * angle_ratio),
        )
    )


def _registration_candidates(operand: TransitionOperand, normal: np.ndarray) -> list[np.ndarray]:
    candidates = [np.zeros(3, dtype=np.float64)]
    spacing = np.asarray(operand.cell_spacing_mm, dtype=np.float64)
    axes = np.asarray(operand.frame_axes_world, dtype=np.float64)
    for axis, period in zip(axes, spacing):
        tangent = axis - np.dot(axis, normal) * normal
        length = float(np.linalg.norm(tangent))
        if length <= 1.0e-8:
            continue
        quarter_period = tangent / length * (0.25 * period)
        candidates.extend((quarter_period, -quarter_period))
    return candidates


def _plane_probe_points(
    bounds: np.ndarray,
    point: np.ndarray,
    normal: np.ndarray,
    scale_mm: float,
) -> np.ndarray:
    helper = np.array((1.0, 0.0, 0.0), dtype=np.float64)
    if abs(float(np.dot(helper, normal))) > 0.85:
        helper = np.array((0.0, 1.0, 0.0), dtype=np.float64)
    tangent_u = np.cross(normal, helper)
    tangent_u /= np.linalg.norm(tangent_u)
    tangent_v = np.cross(normal, tangent_u)
    center = np.clip(point, bounds[0], bounds[1])
    extent = max(float(scale_mm), 1.0e-3)
    coordinates = np.linspace(-extent, extent, 9, dtype=np.float64)
    first, second = np.meshgrid(coordinates, coordinates, indexing="ij")
    probes = center + first.reshape(-1, 1) * tangent_u + second.reshape(-1, 1) * tangent_v
    return np.clip(probes, bounds[0], bounds[1]).astype(np.float32)


def _estimate_registration(
    first: TransitionOperand,
    second: TransitionOperand,
    point: np.ndarray,
    normal: np.ndarray,
    minimum_feature_mm: float,
) -> np.ndarray:
    bounds = np.vstack(
        (
            np.maximum(first.body.bounds[0], second.body.bounds[0]),
            np.minimum(first.body.bounds[1], second.body.bounds[1]),
        )
    )
    if np.any(bounds[0] >= bounds[1]):
        return np.zeros(3, dtype=np.float64)
    probe_scale = max(
        float(np.min(first.cell_spacing_mm)),
        float(np.min(second.cell_spacing_mm)),
    )
    probes = _plane_probe_points(bounds, point, normal, probe_scale)
    first_values = first.body.evaluate_points(probes)
    scale = max(float(minimum_feature_mm), 1.0e-6)
    best_score = -np.inf
    best = np.zeros(3, dtype=np.float64)
    for candidate in _registration_candidates(second, normal):
        second_values = second.body.evaluate_points(probes + candidate.astype(np.float32))
        overlap = np.mean((first_values <= 0.0) & (second_values <= 0.0))
        proximity = np.mean(np.exp(-(np.abs(first_values) + np.abs(second_values)) / scale))
        score = float(2.0 * overlap + proximity)
        if score > best_score + 1.0e-12:
            best_score = score
            best = candidate
    return best


def _smooth_min(first: np.ndarray, second: np.ndarray, radius: float) -> np.ndarray:
    radius_value = np.float32(max(float(radius), 1.0e-7))
    blend = np.clip(
        0.5 + 0.5 * (second - first) / radius_value,
        0.0,
        1.0,
    )
    return (
        second * (1.0 - blend)
        + first * blend
        - radius_value * blend * (1.0 - blend)
    ).astype(np.float32, copy=False)


def build_transition_body(
    first: TransitionOperand,
    second: TransitionOperand,
    design_domain: ImplicitBody,
    *,
    plane_point_mm: tuple[float, float, float] | np.ndarray | None = None,
    plane_normal: tuple[float, float, float] | np.ndarray | None = None,
    driver: ImplicitBody | None = None,
    spec: TransitionSpec,
) -> TransitionBuildResult:
    """Build one authoritative transition from a plane or independent field.

    ``driver`` evaluates the scalar quantity consumed by the Ramp.  It is
    intentionally independent of both lattice operands.  A plane remains the
    compatible default when no driver is supplied.  Plane registration is not
    generalized to curved fields: translating a periodic field along one
    global vector has no defensible meaning for an arbitrary driver gradient.
    """

    spec.validate()
    field_driven = driver is not None
    if driver is None:
        if plane_point_mm is None or plane_normal is None:
            raise ValueError("a transition plane or scalar driver is required")
        point, normal = _validated_plane(plane_point_mm, plane_normal)
        driver_name = "transition plane"
    else:
        if plane_point_mm is not None or plane_normal is not None:
            raise ValueError("provide either a transition plane or a scalar driver, not both")
        point = None
        normal = None
        driver_name = driver.name
    minimum_feature = float(
        spec.minimum_feature_mm
        if spec.minimum_feature_mm is not None
        else min(first.characteristic_feature_mm, second.characteristic_feature_mm)
    )
    recommendation = recommended_transition_width(first, second)
    registration = (
        _estimate_registration(first, second, point, normal, minimum_feature)
        if spec.automatic_registration and point is not None and normal is not None
        else np.zeros(3, dtype=np.float64)
    )
    bounds = np.vstack(
        (
            np.maximum(np.maximum(first.body.bounds[0], second.body.bounds[0]), design_domain.bounds[0]),
            np.minimum(np.minimum(first.body.bounds[1], second.body.bounds[1]), design_domain.bounds[1]),
        )
    )
    if np.any(bounds[0] >= bounds[1]):
        raise ValueError("transition operands and design domain do not share finite bounds")

    def evaluate_field(points: np.ndarray, reporter: StageReporter | None) -> np.ndarray:
        point_array = np.asarray(points, dtype=np.float32)
        if point_array.ndim != 2 or point_array.shape[1] != 3:
            raise ValueError("points must have shape (N, 3)")
        if driver is None:
            assert point is not None and normal is not None
            signed = (point_array - point.astype(np.float32)) @ normal.astype(np.float32)
        else:
            signed = driver.evaluate_points(point_array, reporter)
        centered = signed - np.float32(spec.center_offset_mm)
        half_width = np.float32(0.5 * spec.width_mm)
        parameter = np.clip(
            (centered + half_width) / np.float32(spec.width_mm), 0.0, 1.0
        )
        registration_strength = np.sin(np.pi * parameter) ** 2
        second_points = point_array + (
            registration_strength[:, None] * registration.astype(np.float32)
        )
        first_values = first.body.evaluate_points(point_array, reporter)
        second_values = second.body.evaluate_points(second_points, reporter)
        first_weight, second_weight = transition_weights(signed, spec)
        blended = first_weight * first_values + second_weight * second_values
        if spec.topology_correction:
            correction_strength = 16.0 * parameter**2 * (1.0 - parameter) ** 2
            union = _smooth_min(
                first_values,
                second_values,
                radius=0.5 * minimum_feature,
            )
            corrected = (
                (1.0 - correction_strength) * blended
                + correction_strength * (union - np.float32(0.125 * minimum_feature))
            )
        else:
            corrected = blended
        if spec.first_on_negative_side:
            corrected = np.where(centered <= -half_width, first_values, corrected)
            corrected = np.where(centered >= half_width, second_values, corrected)
        else:
            corrected = np.where(centered <= -half_width, second_values, corrected)
            corrected = np.where(centered >= half_width, first_values, corrected)
        return np.asarray(corrected, dtype=np.float32)

    def evaluate(points: np.ndarray, reporter: StageReporter | None) -> np.ndarray:
        point_array = np.asarray(points, dtype=np.float32)
        corrected = evaluate_field(point_array, reporter)
        domain_values = design_domain.evaluate_points(point_array, reporter)
        return np.maximum(domain_values, corrected).astype(np.float32, copy=False)

    field_body = ImplicitBody(
        name=f"Transition field ({first.name} -> {second.name}; {driver_name})",
        bounds=np.vstack(
            (
                np.maximum(first.body.bounds[0], second.body.bounds[0]),
                np.minimum(first.body.bounds[1], second.body.bounds[1]),
            )
        ),
        evaluate=evaluate_field,
    )
    body = ImplicitBody(
        name=f"Transition ({first.name} -> {second.name}; {driver_name})",
        bounds=bounds,
        evaluate=evaluate,
    )
    if first.supports_gpu and second.supports_gpu:
        execution_backend = "GPU 晶胞求值 + CPU Ramp/拓扑约束"
    elif first.supports_gpu or second.supports_gpu:
        execution_backend = "混合 GPU/CPU 晶胞求值 + CPU Ramp/拓扑约束"
    else:
        execution_backend = "CPU（分批 Ramp/拓扑约束）"
    return TransitionBuildResult(
        body=body,
        field_body=field_body,
        diagnostics=TransitionDiagnostics(
            resolved_weight_kind=_resolved_weight_kind(spec),
            minimum_feature_mm=minimum_feature,
            recommended_width_mm=recommendation,
            width_below_recommendation=spec.width_mm < recommendation,
            registration_translation_mm=tuple(float(value) for value in registration),
            execution_backend=execution_backend,
            driver_name=driver_name,
            field_driven=field_driven,
            quality_validation_supported=not field_driven,
        ),
    )


def _evaluate_in_batches(body: ImplicitBody, points: np.ndarray) -> np.ndarray:
    values = np.empty(len(points), dtype=np.float32)
    for start in range(0, len(points), 250_000):
        end = min(start + 250_000, len(points))
        values[start:end] = body.evaluate_points(points[start:end])
    return values


def validate_transition(
    body: ImplicitBody,
    *,
    plane_point_mm: tuple[float, float, float] | np.ndarray,
    plane_normal: tuple[float, float, float] | np.ndarray,
    spec: TransitionSpec,
    inspection_spacing_mm: float | None = None,
    max_samples: int = 2_000_000,
) -> TransitionQualityReport:
    """Inspect cross-band topology independently from display and STL meshes."""

    spec.validate()
    point, normal = _validated_plane(plane_point_mm, plane_normal)
    minimum_feature = float(spec.minimum_feature_mm or spec.width_mm / 8.0)
    requested_spacing = float(
        inspection_spacing_mm
        if inspection_spacing_mm is not None
        else min(minimum_feature / 3.0, spec.width_mm / 16.0)
    )
    if not np.isfinite(requested_spacing) or requested_spacing <= 0.0:
        raise ValueError("inspection_spacing_mm must be positive or None")
    if int(max_samples) < 1_000:
        raise ValueError("max_samples must be at least 1000")
    helper = np.array((1.0, 0.0, 0.0), dtype=np.float64)
    if abs(float(np.dot(helper, normal))) > 0.85:
        helper = np.array((0.0, 1.0, 0.0), dtype=np.float64)
    tangent_u = np.cross(normal, helper)
    tangent_u /= np.linalg.norm(tangent_u)
    tangent_v = np.cross(normal, tangent_u)
    center_point = point + normal * spec.center_offset_mm
    corners = np.array(
        [
            (x, y, z)
            for x in (body.bounds[0, 0], body.bounds[1, 0])
            for y in (body.bounds[0, 1], body.bounds[1, 1])
            for z in (body.bounds[0, 2], body.bounds[1, 2])
        ],
        dtype=np.float64,
    )
    relative_corners = corners - center_point
    u_projection = relative_corners @ tangent_u
    v_projection = relative_corners @ tangent_v
    local_bounds = np.array(
        (
            (u_projection.min(), v_projection.min(), -0.5 * spec.width_mm - requested_spacing),
            (u_projection.max(), v_projection.max(), 0.5 * spec.width_mm + requested_spacing),
        ),
        dtype=np.float64,
    )
    local_extents = local_bounds[1] - local_bounds[0]
    shape = np.ceil(local_extents / requested_spacing).astype(np.int64) + 1
    sample_count = int(np.prod(shape, dtype=np.int64))
    spacing = requested_spacing
    if sample_count > max_samples:
        spacing *= (sample_count / float(max_samples)) ** (1.0 / 3.0)
        shape = np.ceil(local_extents / spacing).astype(np.int64) + 1
    local_axes = [
        np.linspace(local_bounds[0, axis], local_bounds[1, axis], int(shape[axis]))
        for axis in range(3)
    ]
    first, second, third = np.meshgrid(*local_axes, indexing="ij")
    points = (
        center_point
        + first.reshape(-1, 1) * tangent_u
        + second.reshape(-1, 1) * tangent_v
        + third.reshape(-1, 1) * normal
    ).astype(np.float32)
    centered = third.astype(np.float32, copy=False)
    axis_spacing = tuple(
        float(axis[1] - axis[0]) if len(axis) > 1 else float(spacing)
        for axis in local_axes
    )
    del first, second, third
    values = _evaluate_in_batches(body, points).reshape(tuple(int(value) for value in shape))
    half_width = 0.5 * spec.width_mm
    slab = np.abs(centered) <= half_width + spacing
    occupancy = (values <= 0.0) & slab

    from scipy import ndimage

    structure = ndimage.generate_binary_structure(3, 3)
    labels, component_count = ndimage.label(occupancy, structure=structure)
    layer_half = max(1.5 * spacing, 0.05 * spec.width_mm)
    negative_layer = occupancy & (np.abs(centered + half_width) <= layer_half)
    positive_layer = occupancy & (np.abs(centered - half_width) <= layer_half)
    negative_labels = set(int(value) for value in np.unique(labels[negative_layer]) if value)
    positive_labels = set(int(value) for value in np.unique(labels[positive_layer]) if value)
    crossing_labels = negative_labels & positive_labels
    boundary_labels = negative_labels | positive_labels
    isolated_labels = tuple(
        label
        for label in range(1, int(component_count) + 1)
        if label not in boundary_labels
    )
    isolated_count = len(isolated_labels)

    radius = 0.5 * minimum_feature
    core = ndimage.distance_transform_edt(occupancy, sampling=axis_spacing) >= radius
    core_labels, _ = ndimage.label(core, structure=structure)
    core_layer_half = max(1.5 * spacing, 0.05 * spec.width_mm)
    negative_core_position = -half_width + radius
    positive_core_position = half_width - radius
    negative_core = core & (
        np.abs(centered - negative_core_position) <= core_layer_half
    )
    positive_core = core & (
        np.abs(centered - positive_core_position) <= core_layer_half
    )
    negative_core_labels = set(
        int(value) for value in np.unique(core_labels[negative_core]) if value
    )
    positive_core_labels = set(
        int(value) for value in np.unique(core_labels[positive_core]) if value
    )
    minimum_feature_satisfied = bool(
        spec.width_mm >= 2.0 * radius
        and negative_core_labels.intersection(positive_core_labels)
    )
    issue_locations: list[tuple[float, float, float]] = []
    if not crossing_labels or not minimum_feature_satisfied:
        issue_locations.append(tuple(float(value) for value in center_point))
    for label in isolated_labels[:8]:
        local_index = ndimage.center_of_mass(labels == label)
        local_coordinate = np.array(
            [
                local_axes[axis][
                    int(np.clip(round(local_index[axis]), 0, len(local_axes[axis]) - 1))
                ]
                for axis in range(3)
            ],
            dtype=np.float64,
        )
        world_coordinate = (
            center_point
            + local_coordinate[0] * tangent_u
            + local_coordinate[1] * tangent_v
            + local_coordinate[2] * normal
        )
        issue_locations.append(tuple(float(value) for value in world_coordinate))
    return TransitionQualityReport(
        cross_band_connected=bool(crossing_labels),
        minimum_feature_satisfied=minimum_feature_satisfied,
        isolated_fragment_count=int(isolated_count),
        component_count=int(component_count),
        inspection_spacing_mm=float(max(axis_spacing)),
        inspected_samples=len(points),
        issue_locations_mm=tuple(issue_locations),
    )
