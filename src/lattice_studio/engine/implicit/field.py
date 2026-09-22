"""Resolution-independent implicit bodies and bounded sampled display caches."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


StageReporter = Callable[[str], None]
PointEvaluator = Callable[[np.ndarray, StageReporter | None], np.ndarray]


@dataclass(frozen=True)
class ImplicitBody:
    """An implicit solid represented by a point evaluator and finite bounds."""

    name: str
    bounds: np.ndarray
    evaluate: PointEvaluator
    is_preview_only: bool = False

    def __post_init__(self) -> None:
        bounds = np.asarray(self.bounds, dtype=np.float64)
        if bounds.shape != (2, 3) or not np.isfinite(bounds).all():
            raise ValueError("implicit body bounds must have shape (2, 3)")
        if not np.all(bounds[0] < bounds[1]):
            raise ValueError("implicit body bounds must have positive extents")
        if not callable(self.evaluate):
            raise TypeError("implicit body evaluator must be callable")
        object.__setattr__(self, "bounds", bounds)

    def evaluate_points(
        self,
        points: np.ndarray,
        stage_reporter: StageReporter | None = None,
    ) -> np.ndarray:
        """Evaluate finite point batches without imposing a grid resolution."""

        values = np.asarray(self.evaluate(np.asarray(points), stage_reporter), dtype=np.float32)
        if values.shape != (len(points),):
            raise ValueError(
                f"implicit evaluator returned {values.shape}, expected {(len(points),)}"
            )
        if not np.isfinite(values).all():
            raise ValueError("implicit evaluator returned non-finite values")
        return values

@dataclass(frozen=True)
class SampledImplicitField:
    """A disposable regular-grid cache used by an implicit renderer."""

    name: str
    values: np.ndarray
    origin: np.ndarray
    spacing: np.ndarray
    color: tuple[float, float, float, float]
    is_preview_only: bool = False
    metallic: float = 0.0
    roughness: float = 0.5

    def __post_init__(self) -> None:
        values = np.ascontiguousarray(self.values, dtype=np.float32)
        origin = np.asarray(self.origin, dtype=np.float64)
        spacing = np.asarray(self.spacing, dtype=np.float64)
        if values.ndim != 3 or any(size < 2 for size in values.shape):
            raise ValueError("sampled implicit values must be a 3-D grid with at least 2 samples per axis")
        if not np.isfinite(values).all():
            raise ValueError("sampled implicit values must be finite")
        if origin.shape != (3,) or not np.isfinite(origin).all():
            raise ValueError("sampled implicit origin must have shape (3,)")
        if spacing.shape != (3,) or not np.isfinite(spacing).all() or np.any(spacing <= 0):
            raise ValueError("sampled implicit spacing must be positive")
        if len(self.color) != 4 or not all(np.isfinite(value) for value in self.color):
            raise ValueError("sampled implicit color must contain four finite values")
        if not np.isfinite(self.metallic) or not 0.0 <= self.metallic <= 1.0:
            raise ValueError("sampled implicit metallic must be between 0 and 1")
        if not np.isfinite(self.roughness) or not 0.0 <= self.roughness <= 1.0:
            raise ValueError("sampled implicit roughness must be between 0 and 1")
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "origin", origin)
        object.__setattr__(self, "spacing", spacing)

    @property
    def bounds(self) -> np.ndarray:
        return np.vstack((self.origin, self.origin + (np.asarray(self.values.shape) - 1) * self.spacing))

    @property
    def estimated_bytes(self) -> int:
        return int(self.values.nbytes)


@dataclass(frozen=True)
class SampledFieldCleanupReport:
    """Summary of display-only negative islands removed from a sampled field."""

    removed_components: int
    removed_voxels: int
    retained_components: int
    max_component_voxels: int
    removed_volume_mm3: float
    volume_limit_mm3: float
    relative_volume_limit: float


def remove_small_negative_islands(
    field: SampledImplicitField,
    max_component_voxels: int,
    relative_volume_limit: float = 1.0e-6,
) -> tuple[SampledImplicitField, SampledFieldCleanupReport]:
    """Remove bounded sampled-field islands without changing the implicit body."""

    if not isinstance(field, SampledImplicitField):
        raise TypeError("field must be a SampledImplicitField")
    limit = int(max_component_voxels)
    relative_limit = float(relative_volume_limit)
    if limit < 1 or limit != max_component_voxels:
        raise ValueError("max_component_voxels must be a positive integer")
    if not np.isfinite(relative_limit) or relative_limit <= 0.0:
        raise ValueError("relative_volume_limit must be finite and positive")

    voxel_volume = float(np.prod(field.spacing))

    from scipy import ndimage

    labels, component_count = ndimage.label(
        field.values < 0.0,
        structure=ndimage.generate_binary_structure(3, 3),
    )
    if component_count <= 1:
        return field, SampledFieldCleanupReport(
            0,
            0,
            component_count,
            limit,
            0.0,
            voxel_volume,
            relative_limit,
        )

    sizes = np.bincount(labels.ravel(), minlength=component_count + 1)
    largest_label = int(np.argmax(sizes[1:])) + 1
    candidates = (sizes <= limit) & (np.arange(len(sizes)) != largest_label)
    candidates[0] = False
    main_volume_proxy = float(sizes[largest_label]) * voxel_volume
    relative_volume_threshold = main_volume_proxy * relative_limit
    scale = max(float(np.max(np.abs(field.values), initial=0.0)), 1.0)
    positive_epsilon = np.float32(np.finfo(np.float32).eps * scale * 8.0)
    removable = np.zeros(len(sizes), dtype=bool)
    component_volumes: dict[int, float] = {}

    from skimage import measure

    for label in np.flatnonzero(candidates):
        indices = np.argwhere(labels == label)
        candidate_lower = indices.min(axis=0)
        candidate_upper = indices.max(axis=0)
        if np.any(candidate_lower == 0) or np.any(
            candidate_upper == np.asarray(field.values.shape) - 1
        ):
            continue
        lower = candidate_lower - 1
        upper = candidate_upper + 2
        slices = tuple(slice(lower[axis], upper[axis]) for axis in range(3))
        local_values = field.values[slices].copy()
        local_labels = labels[slices]
        other_interiors = (local_values < 0.0) & (local_labels != label)
        local_values[other_interiors] = np.maximum(
            np.abs(local_values[other_interiors]),
            positive_epsilon,
        )
        try:
            vertices, faces, _normals, _values = measure.marching_cubes(
                local_values,
                level=0.0,
                spacing=tuple(float(value) for value in field.spacing),
                method="lorensen",
            )
        except ValueError:
            continue
        signed_volume = np.einsum(
            "ij,ij->",
            vertices[faces[:, 0]],
            np.cross(vertices[faces[:, 1]], vertices[faces[:, 2]]),
            optimize=True,
        ) / 6.0
        component_volume = abs(float(signed_volume))
        component_volumes[int(label)] = component_volume
        if (
            component_volume <= voxel_volume
            and component_volume <= relative_volume_threshold
        ):
            removable[label] = True

    removable[0] = False
    removed_labels = np.flatnonzero(removable)
    if not len(removed_labels):
        return field, SampledFieldCleanupReport(
            0,
            0,
            component_count,
            limit,
            0.0,
            voxel_volume,
            relative_limit,
        )

    cleaned_values = field.values.copy()
    removed_voxels = 0
    for start in range(0, cleaned_values.shape[0], 32):
        stop = min(start + 32, cleaned_values.shape[0])
        selected = removable[labels[start:stop]]
        removed_voxels += int(np.count_nonzero(selected))
        slab = cleaned_values[start:stop]
        slab[selected] = np.maximum(np.abs(slab[selected]), positive_epsilon)

    cleaned = SampledImplicitField(
        name=field.name,
        values=cleaned_values,
        origin=field.origin,
        spacing=field.spacing,
        color=field.color,
        is_preview_only=field.is_preview_only,
        metallic=field.metallic,
        roughness=field.roughness,
    )
    return cleaned, SampledFieldCleanupReport(
        removed_components=len(removed_labels),
        removed_voxels=removed_voxels,
        retained_components=component_count - len(removed_labels),
        max_component_voxels=limit,
        removed_volume_mm3=float(
            sum(component_volumes[int(label)] for label in removed_labels)
        ),
        volume_limit_mm3=voxel_volume,
        relative_volume_limit=relative_limit,
    )
