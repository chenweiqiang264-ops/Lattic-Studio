"""Stable data contracts returned by numerical engine workflows."""

from __future__ import annotations

from dataclasses import dataclass

import trimesh

from lattice_studio.engine.implicit.field import ImplicitBody, SampledImplicitField
from lattice_studio.engine.implicit.surface_extraction import ExtractionStatistics
from lattice_studio.engine.implicit.transition import TransitionDiagnostics, TransitionQualityReport
from lattice_studio.domain.cell_map import CellMap
from lattice_studio.domain.parameters import ExportSpacingMode


@dataclass(frozen=True)
class SamplingRecommendation:
    voxel_size_mm: float
    grid_shape: tuple[int, int, int]
    estimated_voxels: int
    samples_per_cell: float
    samples_per_wall: float
    limiting_constraint: str = "unknown"


@dataclass(frozen=True)
class ImplicitGenerationResult:
    """Authoritative evaluator and its disposable interactive display cache."""

    body: ImplicitBody
    display_field: SampledImplicitField
    recommendation: SamplingRecommendation
    cell_map: CellMap | None = None
    minimum_feature_mm: float | None = None
    metadata: object | None = None
    display_fragment_cleanup: FieldFragmentCleanupReport | None = None
    unclipped_body: ImplicitBody | None = None


@dataclass(frozen=True)
class DisplayRefinementReport:
    requested_spacing_mm: float
    applied_spacing_mm: float
    estimated_voxels: int
    refined: bool
    budget_limited: bool


@dataclass(frozen=True)
class StlReconstructionTarget:
    body: ImplicitBody
    extraction_map: CellMap
    minimum_feature_mm: float
    enforce_feature_limits: bool = True


@dataclass(frozen=True)
class PreciseRenderTarget:
    body: ImplicitBody
    minimum_feature_mm: float
    material_key: str


@dataclass(frozen=True)
class TransitionGenerationMetadata:
    first_name: str
    second_name: str
    diagnostics: TransitionDiagnostics
    quality: TransitionQualityReport


@dataclass(frozen=True)
class MeshQuality:
    finite: bool
    non_empty: bool
    watertight: bool
    winding_consistent: bool
    connected_components: int
    message: str = ""


@dataclass(frozen=True)
class SimplificationReport:
    accepted: bool
    backend: str
    before: MeshQuality
    after: MeshQuality
    message: str
    max_deviation_mm: float | None = None
    repair_attempted: bool = False
    repaired: bool = False


@dataclass(frozen=True)
class MeshConditioningReport:
    attempted: bool
    applied: bool
    backend: str
    tolerance_mm: float
    removed_vertices: int
    removed_faces: int
    topology_preserved: bool
    bounds_deviation_mm: float
    message: str
    max_deviation_mm: float | None = None
    poor_triangle_fraction_before: float | None = None
    poor_triangle_fraction_after: float | None = None
    edge_p01_mm_before: float | None = None
    edge_p01_mm_after: float | None = None


@dataclass(frozen=True)
class NumericalFragmentCleanupReport:
    removed_components: int
    removed_faces: int
    removed_volume_mm3: float
    retained_components: int
    volume_limit_mm3: float
    relative_volume_limit: float
    message: str


@dataclass(frozen=True)
class FieldFragmentCleanupReport:
    removed_components: int
    removed_voxels: int
    removed_volume_mm3: float
    retained_components: int
    volume_limit_mm3: float
    relative_volume_limit: float
    component_extent_limit_mm: float | None
    message: str


@dataclass(frozen=True)
class ExportMeshResult:
    mesh: trimesh.Trimesh
    quality: MeshQuality
    spacing_mm: tuple[float, float, float]
    tolerance_mm: float
    spacing_limits: tuple[str, str, str] = ("unknown", "unknown", "unknown")
    field_repaired: bool = False
    field_fragment_cleanup: FieldFragmentCleanupReport | None = None
    fragment_cleanup: NumericalFragmentCleanupReport | None = None
    extraction_backend: str = "dense-grid"
    extraction_statistics: ExtractionStatistics | None = None
    extraction_fallback_reason: str | None = None
    spacing_mode: ExportSpacingMode = "recommended"
    conditioning: MeshConditioningReport | None = None


@dataclass(frozen=True)
class ExportGridEstimate:
    spacing_mm: tuple[float, float, float]
    grid_shape: tuple[int, int, int]
    total_voxels: int
    spacing_limits: tuple[str, str, str]


@dataclass(frozen=True)
class CellDisplaySampleEstimate:
    samples_per_axis: tuple[float, float, float]
    voxel_samples_per_cell: float


__all__ = [
    "CellDisplaySampleEstimate",
    "DisplayRefinementReport",
    "ExportGridEstimate",
    "ExportMeshResult",
    "FieldFragmentCleanupReport",
    "ImplicitGenerationResult",
    "MeshConditioningReport",
    "MeshQuality",
    "NumericalFragmentCleanupReport",
    "PreciseRenderTarget",
    "SamplingRecommendation",
    "SimplificationReport",
    "StlReconstructionTarget",
    "TransitionGenerationMetadata",
]
