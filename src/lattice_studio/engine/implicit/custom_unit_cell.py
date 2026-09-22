"""STL-backed non-parametric unit cells evaluated through a Cell Map."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, Literal

import numpy as np
import trimesh

from lattice_studio.domain.cell_map import CellMap
from lattice_studio.engine.implicit.field import ImplicitBody, StageReporter
from lattice_studio.engine.implicit.geometry_compute import (
    AutomaticGeometryBackend,
    CpuGeometryAdapter,
    NumbaCudaGeometryAdapter,
)


class CustomUnitCellError(ValueError):
    """Raised when an STL cannot define a reliable non-parametric unit cell."""


class MeshDistanceBackend(Protocol):
    """Internal seam used by prepared mesh-backed unit cells."""

    def signed_distance(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
        points: np.ndarray,
        max_tile_points: int | None = 250_000,
    ) -> np.ndarray: ...


@dataclass(frozen=True)
class UnitCellDomain:
    """Three-dimensional source bounds mapped to one Cell Map period."""

    bounds_mm: np.ndarray

    def __post_init__(self) -> None:
        bounds = np.asarray(self.bounds_mm, dtype=np.float64)
        if bounds.shape != (2, 3) or not np.isfinite(bounds).all():
            raise ValueError("unit-cell domain bounds must have shape (2, 3)")
        if not np.all(bounds[0] < bounds[1]):
            raise ValueError("unit-cell domain must have positive extents")
        object.__setattr__(self, "bounds_mm", bounds)

    @property
    def extent_mm(self) -> np.ndarray:
        return self.bounds_mm[1] - self.bounds_mm[0]


@dataclass(frozen=True)
class CustomUnitCellImportReport:
    """Observable deterministic cleanup performed while importing a cell."""

    source_path: Path
    original_vertices: int
    original_faces: int
    prepared_vertices: int
    prepared_faces: int
    removed_degenerate_faces: int
    removed_duplicate_faces: int
    watertight: bool
    winding_consistent: bool
    connected_components: int
    characteristic_thickness_mm: float


PeriodicDirection = Literal["U", "V", "W"]


@dataclass(frozen=True)
class PeriodicSeamAxisReport:
    """Compatibility measurements for one pair of periodic cell faces."""

    direction: PeriodicDirection
    lower_solid_samples: int
    upper_solid_samples: int
    matching_solid_samples: int
    union_solid_samples: int
    mismatch_ratio: float
    compatible: bool
    bridge_enabled: bool
    effective_compatible: bool
    cell_spacing_mm: float
    probe_depth_mm: float
    lower_solid_area_mm2: float
    upper_solid_area_mm2: float
    matching_solid_area_mm2: float
    union_solid_area_mm2: float


@dataclass(frozen=True)
class PeriodicSeamReport:
    """Deterministic, warning-only report for periodic face compatibility."""

    axes: tuple[PeriodicSeamAxisReport, ...]
    samples_per_axis: int

    @property
    def incompatible_directions(self) -> tuple[PeriodicDirection, ...]:
        return tuple(axis.direction for axis in self.axes if not axis.compatible)


def _normalize_periodic_directions(
    directions: tuple[str, ...] | list[str] | set[str] | None,
) -> tuple[PeriodicDirection, ...]:
    if directions is None:
        return ()
    normalized = tuple(str(direction).upper() for direction in directions)
    invalid = sorted(set(normalized) - {"U", "V", "W"})
    if invalid:
        raise ValueError(
            "bridge_directions must contain only U, V, and W; "
            f"invalid values: {invalid}"
        )
    return tuple(direction for direction in ("U", "V", "W") if direction in normalized)


@dataclass(frozen=True)
class PreparedCustomUnitCell:
    """Immutable prepared body plus its private resident distance backend."""

    name: str
    domain: UnitCellDomain
    report: CustomUnitCellImportReport
    _vertices: np.ndarray = field(repr=False)
    _faces: np.ndarray = field(repr=False)
    _geometry_backend: MeshDistanceBackend = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        vertices = np.ascontiguousarray(self._vertices, dtype=np.float32)
        faces = np.ascontiguousarray(self._faces, dtype=np.int32)
        if vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) == 0:
            raise ValueError("prepared unit-cell vertices must have shape (N, 3)")
        if faces.ndim != 2 or faces.shape[1] != 3 or len(faces) == 0:
            raise ValueError("prepared unit-cell faces must have shape (M, 3)")
        object.__setattr__(self, "_vertices", vertices)
        object.__setattr__(self, "_faces", faces)

    @property
    def source_size_mm(self) -> tuple[float, float, float]:
        return tuple(float(value) for value in self.domain.extent_mm)

    @property
    def estimated_mesh_bytes(self) -> int:
        return int(self._vertices.nbytes + self._faces.nbytes)

    @property
    def evaluation_backend(self) -> tuple[str, bool]:
        """Return the active evaluator label and whether it currently uses GPU."""

        status = getattr(self._geometry_backend, "status", None)
        if status is None:
            return type(self._geometry_backend).__name__, False
        return str(status.active_backend), bool(status.using_gpu)

    def characteristic_feature_mm(self, cell_map: CellMap) -> float:
        """Scale the source body's volume-to-area thickness into world units."""

        scale = float(
            np.min(np.asarray(cell_map.spacing_mm) / self.domain.extent_mm)
        )
        return float(self.report.characteristic_thickness_mm * scale)

    def effective_feature_mm(
        self,
        cell_map: CellMap,
        target_feature_mm: float | None = None,
    ) -> float:
        """Resolve a user target against the scaled source characteristic."""

        native_feature_mm = self.characteristic_feature_mm(cell_map)
        effective_feature_mm = (
            native_feature_mm
            if target_feature_mm is None
            else float(target_feature_mm)
        )
        if not np.isfinite(effective_feature_mm) or effective_feature_mm <= 0.0:
            raise ValueError("target_feature_mm must be finite and positive or None")
        return effective_feature_mm

    def seam_report(
        self,
        cell_map: CellMap,
        *,
        samples_per_axis: int = 32,
        face_inset_ratio: float = 0.02,
        target_feature_mm: float | None = None,
        bridge_directions: tuple[str, ...] | list[str] | set[str] | None = None,
        bridge_depth_mm: float | None = None,
        max_tile_points: int | None = 250_000,
    ) -> PeriodicSeamReport:
        """Measure opposite-face solid overlap in the current Cell Map scale.

        The report is deliberately warning-only. A face pair with no solid
        material is incompatible because it cannot form a periodic connection,
        even though two empty masks would otherwise have an IoU of one.
        """

        if not isinstance(cell_map, CellMap):
            raise TypeError("cell_map must be a CellMap")
        if int(samples_per_axis) < 4:
            raise ValueError("samples_per_axis must be at least 4")
        if not 0.0 < float(face_inset_ratio) < 0.5:
            raise ValueError("face_inset_ratio must be in (0, 0.5)")
        directions = _normalize_periodic_directions(bridge_directions)
        native_feature_mm = self.characteristic_feature_mm(cell_map)
        effective_feature_mm = self.effective_feature_mm(
            cell_map,
            target_feature_mm,
        )
        feature_offset_mm = 0.5 * (effective_feature_mm - native_feature_mm)
        if bridge_depth_mm is not None and (
            not np.isfinite(bridge_depth_mm) or bridge_depth_mm <= 0.0
        ):
            raise ValueError("bridge_depth_mm must be finite and positive or None")

        resolution = int(samples_per_axis)
        grid = (np.arange(resolution, dtype=np.float64) + 0.5) / resolution
        first, second = np.meshgrid(grid, grid, indexing="ij")
        fractions = np.zeros((resolution * resolution, 3), dtype=np.float64)
        reports: list[PeriodicSeamAxisReport] = []
        source_extent = self.domain.extent_mm
        spacing = np.asarray(cell_map.spacing_mm, dtype=np.float64)
        world_scale = float(np.min(spacing / source_extent))

        for axis, direction in enumerate(("U", "V", "W")):
            transverse = [index for index in range(3) if index != axis]
            fractions[:, transverse[0]] = first.ravel()
            fractions[:, transverse[1]] = second.ravel()
            axis_probe_depth = (
                self._automatic_bridge_depth_mm(cell_map, effective_feature_mm)
                if direction in directions and bridge_depth_mm is None
                else (
                    float(bridge_depth_mm)
                    if direction in directions and bridge_depth_mm is not None
                    else float(spacing[axis] * face_inset_ratio)
                )
            )
            if axis_probe_depth >= 0.5 * spacing[axis]:
                raise ValueError(
                    "bridge_depth_mm must be less than half the selected cell spacing"
                )
            inset = axis_probe_depth / spacing[axis]
            lower = fractions.copy()
            upper = fractions.copy()
            lower[:, axis] = inset
            upper[:, axis] = 1.0 - inset
            lower_points = self.domain.bounds_mm[0] + lower * source_extent
            upper_points = self.domain.bounds_mm[0] + upper * source_extent
            lower_values = self._geometry_backend.signed_distance(
                self._vertices, self._faces, lower_points, max_tile_points
            )
            upper_values = self._geometry_backend.signed_distance(
                self._vertices, self._faces, upper_points, max_tile_points
            )
            lower_world_values = (
                np.asarray(lower_values, dtype=np.float32) * np.float32(world_scale)
                - np.float32(feature_offset_mm)
            )
            upper_world_values = (
                np.asarray(upper_values, dtype=np.float32) * np.float32(world_scale)
                - np.float32(feature_offset_mm)
            )
            lower_solid = lower_world_values <= 0.0
            upper_solid = upper_world_values <= 0.0
            union_count = int(np.count_nonzero(lower_solid | upper_solid))
            matching_count = int(np.count_nonzero(lower_solid & upper_solid))
            mismatch_ratio = (
                float(np.count_nonzero(lower_solid ^ upper_solid)) / union_count
                if union_count
                else 1.0
            )
            compatible = bool(
                np.count_nonzero(lower_solid)
                and np.count_nonzero(upper_solid)
                and mismatch_ratio <= 0.05
            )
            bridge_enabled = direction in directions
            effective_compatible = bool(
                compatible or (bridge_enabled and union_count > 0)
            )
            sample_area_mm2 = (
                spacing[transverse[0]] * spacing[transverse[1]]
                / float(resolution * resolution)
            )
            reports.append(
                PeriodicSeamAxisReport(
                    direction=direction,
                    lower_solid_samples=int(np.count_nonzero(lower_solid)),
                    upper_solid_samples=int(np.count_nonzero(upper_solid)),
                    matching_solid_samples=matching_count,
                    union_solid_samples=union_count,
                    mismatch_ratio=mismatch_ratio,
                    compatible=compatible,
                    bridge_enabled=bridge_enabled,
                    effective_compatible=effective_compatible,
                    cell_spacing_mm=float(spacing[axis]),
                    probe_depth_mm=float(axis_probe_depth),
                    lower_solid_area_mm2=float(
                        np.count_nonzero(lower_solid) * sample_area_mm2
                    ),
                    upper_solid_area_mm2=float(
                        np.count_nonzero(upper_solid) * sample_area_mm2
                    ),
                    matching_solid_area_mm2=float(matching_count * sample_area_mm2),
                    union_solid_area_mm2=float(union_count * sample_area_mm2),
                )
            )
        return PeriodicSeamReport(tuple(reports), resolution)

    def _automatic_bridge_depth_mm(
        self,
        cell_map: CellMap,
        effective_feature_mm: float,
    ) -> float:
        """Choose a conservative local bridge depth when the user uses Auto."""

        return float(
            min(
                0.5 * effective_feature_mm,
                float(np.min(np.asarray(cell_map.spacing_mm))) / 8.0,
            )
        )

    def evaluate_periodic(
        self,
        points: np.ndarray,
        cell_map: CellMap,
        target_feature_mm: float | None = None,
        max_tile_points: int | None = 250_000,
        bridge_directions: tuple[str, ...] | list[str] | set[str] | None = None,
        bridge_depth_mm: float | None = None,
    ) -> np.ndarray:
        """Evaluate the repeated source body without a design-domain trim."""

        point_array = np.asarray(points, dtype=np.float64)
        if (
            point_array.ndim != 2
            or point_array.shape[1] != 3
            or not np.isfinite(point_array).all()
        ):
            raise ValueError("points must have shape (N, 3) and finite values")
        if not isinstance(cell_map, CellMap):
            raise TypeError("cell_map must be a CellMap")
        directions = _normalize_periodic_directions(bridge_directions)
        cell_coordinates = cell_map.to_cell_coordinates(point_array)
        fractions = cell_coordinates - np.floor(cell_coordinates)
        source_points = (
            self.domain.bounds_mm[0] + fractions * self.domain.extent_mm
        )
        source_distance = self._geometry_backend.signed_distance(
            self._vertices,
            self._faces,
            source_points,
            max_tile_points,
        )
        # Non-uniform scaling has no scalar exact-distance transform. The
        # smallest singular value preserves the zero set and is conservative.
        world_scale = float(
            np.min(np.asarray(cell_map.spacing_mm) / self.domain.extent_mm)
        )
        world_distance = np.asarray(source_distance, dtype=np.float32) * np.float32(
            world_scale
        )
        native_feature_mm = self.characteristic_feature_mm(cell_map)
        effective_feature_mm = self.effective_feature_mm(
            cell_map,
            target_feature_mm,
        )
        feature_offset_mm = 0.5 * (effective_feature_mm - native_feature_mm)
        if feature_offset_mm:
            world_distance = world_distance - np.float32(feature_offset_mm)
        if directions:
            depth_mm = (
                self._automatic_bridge_depth_mm(cell_map, effective_feature_mm)
                if bridge_depth_mm is None
                else float(bridge_depth_mm)
            )
            if not np.isfinite(depth_mm) or depth_mm <= 0.0:
                raise ValueError("bridge_depth_mm must be finite and positive or None")
            spacing = np.asarray(cell_map.spacing_mm, dtype=np.float64)
            if np.any(depth_mm >= 0.5 * spacing[
                ["UVW".index(direction) for direction in directions]
            ]):
                raise ValueError("bridge_depth_mm must be less than half the selected cell spacing")
            for direction in directions:
                axis = "UVW".index(direction)
                inset = min(depth_mm / spacing[axis], 0.49)
                lower_fractions = fractions.copy()
                upper_fractions = fractions.copy()
                lower_fractions[:, axis] = inset
                upper_fractions[:, axis] = 1.0 - inset
                lower_points = self.domain.bounds_mm[0] + lower_fractions * self.domain.extent_mm
                upper_points = self.domain.bounds_mm[0] + upper_fractions * self.domain.extent_mm
                lower_distance = self._geometry_backend.signed_distance(
                    self._vertices, self._faces, lower_points, max_tile_points
                )
                upper_distance = self._geometry_backend.signed_distance(
                    self._vertices, self._faces, upper_points, max_tile_points
                )
                face_distance = np.minimum(lower_distance, upper_distance)
                face_distance = np.asarray(face_distance, dtype=np.float32) * np.float32(world_scale)
                if feature_offset_mm:
                    face_distance = face_distance - np.float32(feature_offset_mm)
                seam_distance = np.minimum(
                    fractions[:, axis] * spacing[axis],
                    (1.0 - fractions[:, axis]) * spacing[axis],
                )
                bridge_distance = np.maximum(
                    face_distance,
                    seam_distance.astype(np.float32) - np.float32(depth_mm),
                )
                world_distance = np.minimum(world_distance, bridge_distance)
        return world_distance


def _new_geometry_backend() -> AutomaticGeometryBackend:
    """Own a BVH cache independent from the design-domain backend."""

    return AutomaticGeometryBackend(
        cpu_adapter=CpuGeometryAdapter(),
        gpu_adapter=NumbaCudaGeometryAdapter(),
    )


def _load_mesh(path: Path) -> trimesh.Trimesh:
    if not path.is_file():
        raise CustomUnitCellError(f"custom unit-cell STL does not exist: {path}")
    loaded = trimesh.load(path, force="mesh", process=True)
    if not isinstance(loaded, trimesh.Trimesh) or len(loaded.faces) == 0:
        raise CustomUnitCellError("custom unit-cell STL is empty")
    return loaded


def prepare_stl_unit_cell(
    path: str | Path,
    *,
    domain_bounds_mm: np.ndarray | None = None,
    units_to_mm: float = 1.0,
    geometry_backend: MeshDistanceBackend | None = None,
) -> PreparedCustomUnitCell:
    """Import and deterministically prepare an STL as a non-parametric cell.

    Cleanup is intentionally conservative: duplicate and degenerate faces are
    removed, unreferenced vertices are discarded, and normals are made
    consistent. Holes are not filled and separate valid solids are preserved.
    """

    source_path = Path(path).expanduser().resolve()
    scale = float(units_to_mm)
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("units_to_mm must be finite and positive")
    mesh = _load_mesh(source_path)
    original_vertices = len(mesh.vertices)
    original_faces = len(mesh.faces)
    if not np.isfinite(mesh.vertices).all():
        raise CustomUnitCellError("custom unit-cell STL contains non-finite vertices")
    if scale != 1.0:
        mesh.apply_scale(scale)

    unique = np.asarray(mesh.unique_faces(), dtype=bool)
    nondegenerate = np.asarray(mesh.nondegenerate_faces(), dtype=bool)
    removed_duplicate = int(np.count_nonzero(~unique))
    removed_degenerate = int(np.count_nonzero(~nondegenerate))
    mesh.update_faces(unique & nondegenerate)
    mesh.remove_unreferenced_vertices()
    mesh.fix_normals()

    if len(mesh.faces) == 0:
        raise CustomUnitCellError("custom unit-cell STL has no valid faces")
    if not mesh.is_watertight:
        raise CustomUnitCellError(
            "custom unit-cell STL is not watertight after deterministic cleanup"
        )
    if not mesh.is_winding_consistent:
        raise CustomUnitCellError(
            "custom unit-cell STL has inconsistent winding after cleanup"
        )

    domain = UnitCellDomain(
        np.asarray(domain_bounds_mm, dtype=np.float64)
        if domain_bounds_mm is not None
        else np.asarray(mesh.bounds, dtype=np.float64)
    )
    tolerance = max(float(np.max(domain.extent_mm)) * 1.0e-8, 1.0e-9)
    if np.any(mesh.bounds[0] < domain.bounds_mm[0] - tolerance) or np.any(
        mesh.bounds[1] > domain.bounds_mm[1] + tolerance
    ):
        raise CustomUnitCellError("custom unit-cell body lies outside its domain")

    components = int(mesh.body_count)
    surface_area = float(mesh.area)
    volume = abs(float(mesh.volume))
    characteristic_thickness = (
        2.0 * volume / surface_area
        if surface_area > 0.0 and volume > 0.0
        else 0.0
    )
    if characteristic_thickness <= 0.0 or not np.isfinite(characteristic_thickness):
        raise CustomUnitCellError(
            "custom unit-cell STL has no reliable positive volume-to-area thickness"
        )
    report = CustomUnitCellImportReport(
        source_path=source_path,
        original_vertices=original_vertices,
        original_faces=original_faces,
        prepared_vertices=len(mesh.vertices),
        prepared_faces=len(mesh.faces),
        removed_degenerate_faces=removed_degenerate,
        removed_duplicate_faces=removed_duplicate,
        watertight=bool(mesh.is_watertight),
        winding_consistent=bool(mesh.is_winding_consistent),
        connected_components=components,
        characteristic_thickness_mm=characteristic_thickness,
    )
    return PreparedCustomUnitCell(
        name=source_path.stem,
        domain=domain,
        report=report,
        _vertices=np.asarray(mesh.vertices),
        _faces=np.asarray(mesh.faces),
        _geometry_backend=geometry_backend or _new_geometry_backend(),
    )


def make_periodic_lattice(
    unit_cell: PreparedCustomUnitCell,
    cell_map: CellMap,
    design_domain: ImplicitBody,
    *,
    target_feature_mm: float | None = None,
    max_tile_points: int | None = 250_000,
    bridge_directions: tuple[str, ...] | list[str] | set[str] | None = None,
    bridge_depth_mm: float | None = None,
) -> ImplicitBody:
    """Repeat one prepared body through ``cell_map`` and trim it once.

    ``target_feature_mm`` changes thickness with a world-space signed-distance
    offset. It does not scale or otherwise move the Cell Map or source body.
    """

    if not isinstance(unit_cell, PreparedCustomUnitCell):
        raise TypeError("unit_cell must be a PreparedCustomUnitCell")
    if not isinstance(cell_map, CellMap):
        raise TypeError("cell_map must be a CellMap")
    if not isinstance(design_domain, ImplicitBody):
        raise TypeError("design_domain must be an ImplicitBody")
    if design_domain.is_preview_only:
        raise CustomUnitCellError(
            "a preview-only design domain cannot trim a custom unit cell"
        )
    if max_tile_points is not None and max_tile_points < 1:
        raise ValueError("max_tile_points must be positive or None")
    unit_cell.effective_feature_mm(cell_map, target_feature_mm)
    normalized_directions = _normalize_periodic_directions(bridge_directions)
    if bridge_depth_mm is not None and (
        not np.isfinite(bridge_depth_mm) or bridge_depth_mm <= 0.0
    ):
        raise ValueError("bridge_depth_mm must be finite and positive or None")
    if normalized_directions and bridge_depth_mm is not None:
        selected_spacing = np.asarray(cell_map.spacing_mm)[
            ["UVW".index(direction) for direction in normalized_directions]
        ]
        if np.any(bridge_depth_mm >= 0.5 * selected_spacing):
            raise ValueError("bridge_depth_mm must be less than half the selected cell spacing")

    def evaluate(points: np.ndarray, stage_reporter: StageReporter | None) -> np.ndarray:
        if stage_reporter:
            stage_reporter("计算设计域 SDF")
        domain_values = design_domain.evaluate_points(points)
        if stage_reporter:
            stage_reporter("计算自定义晶胞场")
        cell_values = unit_cell.evaluate_periodic(
            points,
            cell_map,
            target_feature_mm,
            max_tile_points,
            normalized_directions,
            bridge_depth_mm,
        )
        if stage_reporter:
            stage_reporter("计算设计域与自定义晶胞场交集")
        return np.maximum(domain_values, cell_values)

    return ImplicitBody(
        name=unit_cell.name,
        bounds=np.asarray(design_domain.bounds, dtype=np.float64),
        evaluate=evaluate,
    )
