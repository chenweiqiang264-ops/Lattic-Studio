"""Implicit lattice generation, sampling, reconstruction, and repair workflows.

This module is the compatibility-preserving first extraction from the Qt
workbench. Its public surface will be narrowed as cohesive engine modules
replace direct function imports.
"""

from __future__ import annotations
import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Literal, TypedDict
import numpy as np
import trimesh
from lattice_studio.domain.cell_map import CellMap, CellMapFrame
from lattice_studio.engine.implicit.custom_unit_cell import (
    CustomUnitCellImportReport,
    PreparedCustomUnitCell,
    make_periodic_lattice,
    prepare_stl_unit_cell,
)
from lattice_studio.engine.implicit.field import (
    ImplicitBody,
    SampledImplicitField,
)
from lattice_studio.engine.implicit.primitives import (
    ImplicitPrimitive,
    PrimitiveKind,
    euler_deg_from_rotation_matrix,
)
from lattice_studio.engine.implicit.precise_render import (
    PreciseRenderCamera,
    PreciseRenderMaterial,
    PreciseRenderResult,
    PreciseRenderSettings,
    render_precise_implicit,
)
from lattice_studio.engine.design_domains import (
    AnalyticDesignDomain,
    DesignDomainValue,
    MeshDesignDomain,
)
from lattice_studio.engine.implicit.field_viewer import (
    FieldPlaneCache,
    FieldPlaneState,
    FieldSourceData,
    expand_implicit_body_bounds,
    extract_field_plane_isolines,
    field_plane_cache_key,
    field_surface_hit_from_ray,
    orthonormal_plane_axes,
    resolve_field_value_range,
    sample_field_plane,
    source_sampling_spacing,
)
from lattice_studio.engine.implicit.gradient_lattice import (
    GRADIENT_MODES,
    GRADIENT_RESOLUTION_STRATEGIES,
    WALL_THICKNESS_METHODS,
    GradientAxis,
    GradientControls,
    GradientMode,
    GradientResolutionStrategy,
    WallThicknessMethod,
    evaluate_gradient_tpms_field,
    gradient_projection_range,
    make_tpms_gradient_spec,
    matlab_gradient_voxel_size,
)
from lattice_studio.engine.implicit.geometry_compute import get_default_geometry_backend
from lattice_studio.engine.implicit.lattice_pipeline import get_default_lattice_pipeline
from lattice_studio.engine.implicit.layer_contours import (
    ContourFieldSource,
    LayerContourCache,
    LayerContourResult,
    available_layer_positions,
    resolve_layer_spacing,
    sample_implicit_body_layer,
    sample_sampled_field_layer,
)
from lattice_studio.engine.implicit.shell import (
    fuse_shell_lattice_fields,
    make_coordinated_shell_lattice_union,
    make_interior_shell,
    make_shell_lattice_union,
)
from lattice_studio.engine.implicit.surface_extraction import (
    ExtractionStatistics,
    grid_definition as extraction_grid_definition,
)
from lattice_studio.engine.implicit.tpms_compute import (
    TPMSFieldSpec as ComputeTPMSFieldSpec,
    TransitionBlendSpec as ComputeTransitionBlendSpec,
    get_default_tpms_backend,
    sigmoid_transition_weights as compute_sigmoid_transition_weights,
)
from lattice_studio.engine.implicit.transition import (
    TransitionDiagnostics,
    TransitionOperand,
    TransitionQualityReport,
    TransitionSpec as ProviderTransitionSpec,
    TransitionWeightKind,
    build_transition_body as build_provider_transition_body,
    recommended_transition_width as recommend_provider_transition_width,
    validate_transition as validate_provider_transition,
)

from lattice_studio.domain.parameters import (
    TPMSKind,
    LatticeKind,
    TPMS_KINDS,
    CellSize,
    CellMapMode,
    PlaneAxis,
    TransitionDriverMode,
    ProcessingMode,
    ExportSpacingMode,
    TPMSParameterState,
    FRAME_INPUT_DECIMALS,
    FRAME_ORIGIN_DEFAULT_ATOL_MM,
    TPMSParameters,
    CustomUnitCellParameters,
    LatticeParameters,
    SamplingParameters,
    TransitionParameters,
)
from lattice_studio.engine.contracts import (
    CellDisplaySampleEstimate,
    DisplayRefinementReport,
    ExportGridEstimate,
    ExportMeshResult,
    FieldFragmentCleanupReport,
    ImplicitGenerationResult,
    MeshConditioningReport,
    MeshQuality,
    NumericalFragmentCleanupReport,
    PreciseRenderTarget,
    SamplingRecommendation,
    SimplificationReport,
    StlReconstructionTarget,
    TransitionGenerationMetadata,
)
from lattice_studio.engine.mesh_quality import (
    _clean_display_field_components,
    inspect_mesh,
    recommended_floating_component_extent_mm,
    remove_numerical_fragments,
    remove_small_negative_field_components,
    repair_mesh,
    safe_mesh_volume,
)
from lattice_studio.engine.reconstruction_grid import (
    estimate_export_grid,
    export_spacing_limit_labels,
    recommend_export_spacing,
    resolve_export_spacing,
)




DISPLAY_FIELD_REFINEMENT_FACTORS = {"high": 0.75, "ultra": 0.50}
DISPLAY_QUALITY_LABELS = {"high": "高级", "ultra": "极佳"}
SHELL_RESULT_KEY = "Shell"
SHELL_UNION_RESULT_KEY = "ShellUnion"




















ProgressCallback = Callable[[str, float], None]


EvaluationStageCallback = Callable[[str], None]








PROGRESS_DOMAIN_SDF = "计算设计域 SDF"


PROGRESS_TPMS_FIELD = "计算 TPMS 场"


PROGRESS_INTERSECTION = "计算设计域与 TPMS 场交集"


PROGRESS_MARCHING_CUBES = "执行 Marching Cubes"


def _tpms_backend_label() -> str:
    status = get_default_tpms_backend().status
    if status.using_gpu:
        return f"CUDA GPU · {status.device_name}"
    if status.fallback_reason:
        return f"NumPy CPU 回退 · {status.fallback_reason}"
    return "NumPy CPU"


def _geometry_backend_label() -> str:
    status = get_default_geometry_backend().status
    if status.using_gpu:
        return f"CUDA GPU · {status.device_name}"
    if status.fallback_reason:
        return f"CPU 回退 · {status.fallback_reason}"
    return status.active_backend


def _lattice_pipeline_backend_label() -> str:
    status = get_default_lattice_pipeline().status
    if status.using_gpu:
        return f"CUDA device-resident · {status.device_name}"
    if status.fallback_reason:
        return f"CPU fallback · {status.fallback_reason}"
    return status.active_backend










def _format_spacing_mm(spacing_mm: tuple[float, float, float] | np.ndarray) -> str:
    spacing = np.asarray(spacing_mm, dtype=np.float64)
    if spacing.shape != (3,) or not np.isfinite(spacing).all():
        raise ValueError("spacing_mm must contain three finite values")
    if np.allclose(spacing, spacing[0], rtol=0.0, atol=1.0e-9):
        return f"{spacing[0]:.4f} mm"
    return " × ".join(f"{value:.4f}" for value in spacing) + " mm"


def describe_display_sampling(
    name: str,
    field: SampledImplicitField,
    base_recommendation: SamplingRecommendation,
    requested_spacing_mm: float | None,
) -> str:
    """Describe the display grid that was actually stored and rendered."""

    actual = np.asarray(field.spacing, dtype=np.float64)
    base = float(base_recommendation.voxel_size_mm)
    if requested_spacing_mm is not None:
        reason = f"手动设置 {float(requested_spacing_mm):.4f} mm"
    elif float(np.max(actual)) > base + max(base * 1.0e-6, 1.0e-9):
        reason = (
            f"自动显示预算调整（基础建议 {base:.4f} mm，"
            f"基础约束：{base_recommendation.limiting_constraint}）"
        )
    elif float(np.min(actual)) < base - max(base * 1.0e-6, 1.0e-9):
        reason = f"复用更细的共享显示网格（本对象基础建议 {base:.4f} mm）"
    else:
        reason = (
            f"自动基础建议 {base:.4f} mm"
            f"（{base_recommendation.limiting_constraint}）"
        )
    shape = tuple(int(value) for value in field.values.shape)
    return (
        f"{name}：实际间距 {_format_spacing_mm(actual)}，"
        f"网格 {shape[0]}×{shape[1]}×{shape[2]} "
        f"({int(np.prod(shape)):,} 体素)，{reason}"
    )


def _close_small_field_gaps(
    field: np.ndarray,
    spacing_mm: np.ndarray,
    tolerance_mm: float,
) -> tuple[np.ndarray, bool]:
    """Close only gaps representable within the selected repair tolerance."""

    tolerance = float(tolerance_mm)
    if tolerance <= 0.0:
        return np.asarray(field, dtype=np.float32), False
    iterations = int(np.floor(tolerance / float(np.max(spacing_mm)) + 1.0e-9))
    if iterations < 1:
        return np.asarray(field, dtype=np.float32), False
    from scipy import ndimage

    source = np.asarray(field, dtype=np.float32)
    inside = source <= 0.0
    closed = ndimage.binary_closing(
        inside,
        structure=ndimage.generate_binary_structure(3, 1),
        iterations=iterations,
        border_value=0,
    )
    added = closed & ~inside
    if not np.any(added):
        return source, False
    result = source.copy()
    result[added] = -min(float(np.min(spacing_mm)) * 0.25, tolerance * 0.25)
    return result, True


def _sample_triangle_quality(
    mesh: trimesh.Trimesh,
    max_faces: int = 250_000,
) -> tuple[float, float, float]:
    """Return bounded-cost sliver fractions and the first edge percentile."""

    face_count = len(mesh.faces)
    if face_count == 0:
        return 0.0, 0.0, 0.0
    if face_count <= max_faces:
        face_indices = np.arange(face_count, dtype=np.int64)
    else:
        face_indices = np.linspace(0, face_count - 1, max_faces, dtype=np.int64)
    triangles = np.asarray(mesh.vertices, dtype=np.float64)[
        np.asarray(mesh.faces, dtype=np.int64)[face_indices]
    ]
    edges = np.stack(
        (
            triangles[:, 1] - triangles[:, 0],
            triangles[:, 2] - triangles[:, 1],
            triangles[:, 0] - triangles[:, 2],
        ),
        axis=1,
    )
    edge_squared = np.einsum("...i,...i->...", edges, edges)
    double_area = np.linalg.norm(
        np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]),
        axis=1,
    )
    denominator = edge_squared.sum(axis=1)
    quality = np.divide(
        2.0 * np.sqrt(3.0) * double_area,
        denominator,
        out=np.zeros_like(double_area),
        where=denominator > 0.0,
    )
    return (
        float(np.mean(quality < 0.05)),
        float(np.mean(quality < 0.005)),
        float(np.quantile(np.sqrt(edge_squared), 0.01)),
    )


def _condition_mesh_for_slicing(
    mesh: trimesh.Trimesh,
    export_tolerance_mm: float,
    spacing_mm: np.ndarray,
    source_quality: MeshQuality | None = None,
) -> tuple[trimesh.Trimesh, MeshConditioningReport]:
    """Apply bounded topology-preserving decimation to a valid STL candidate."""

    source = mesh
    spacing = np.asarray(spacing_mm, dtype=np.float64)
    deviation_limit = max(
        min(float(export_tolerance_mm) * 0.25, float(np.min(spacing)) * 0.5),
        1.0e-7,
    )
    before_faces = int(len(source.faces))
    before_vertices = int(len(source.vertices))
    before_quality = source_quality or inspect_mesh(source)
    source_is_valid = (
        before_quality.finite
        and before_quality.non_empty
        and before_quality.watertight
        and before_quality.winding_consistent
        and before_quality.connected_components == 1
    )
    if not source_is_valid:
        return source.copy(), MeshConditioningReport(
            attempted=False,
            applied=False,
            backend="not run",
            tolerance_mm=deviation_limit,
            removed_vertices=0,
            removed_faces=0,
            topology_preserved=False,
            bounds_deviation_mm=0.0,
            message="conditioning skipped: source topology was invalid",
        )
    poor_before, very_poor_before, edge_p01_before = _sample_triangle_quality(source)
    try:
        target_faces = max(4, int(round(before_faces * 0.80)))
        candidate, backend = _compiled_simplify_mesh(source, target_faces)
        after_quality = inspect_mesh(candidate)
        topology_preserved = (
            after_quality.finite
            and after_quality.non_empty
            and after_quality.watertight
            and after_quality.winding_consistent
            and after_quality.connected_components == 1
        )
        bounds_deviation = float(
            np.max(
                np.abs(
                    np.asarray(candidate.bounds, dtype=np.float64)
                    - np.asarray(source.bounds, dtype=np.float64)
                )
            )
        )
        removed_vertices = max(before_vertices - len(candidate.vertices), 0)
        removed_faces = max(before_faces - len(candidate.faces), 0)
        if not topology_preserved or removed_faces == 0:
            raise RuntimeError("topology-preserving decimation was not accepted")
        max_deviation = _mesh_deviation_mm(source, candidate)
        poor_after, very_poor_after, edge_p01_after = _sample_triangle_quality(candidate)
        quality_improved = (
            poor_after <= poor_before + 1.0e-6
            and very_poor_after <= very_poor_before + 1.0e-6
            and edge_p01_after >= edge_p01_before * 0.98
        )
        geometry_preserved = (
            bounds_deviation <= deviation_limit
            and max_deviation <= deviation_limit
        )
        if not quality_improved or not geometry_preserved:
            return source.copy(), MeshConditioningReport(
                attempted=True,
                applied=False,
                backend=backend,
                tolerance_mm=deviation_limit,
                removed_vertices=0,
                removed_faces=0,
                topology_preserved=topology_preserved,
                bounds_deviation_mm=bounds_deviation,
                message="conditioning rollback: quality or deviation guard failed",
                max_deviation_mm=max_deviation,
                poor_triangle_fraction_before=poor_before,
                poor_triangle_fraction_after=poor_after,
                edge_p01_mm_before=edge_p01_before,
                edge_p01_mm_after=edge_p01_after,
            )
        return candidate, MeshConditioningReport(
            attempted=True,
            applied=True,
            backend=backend,
            tolerance_mm=deviation_limit,
            removed_vertices=removed_vertices,
            removed_faces=removed_faces,
            topology_preserved=True,
            bounds_deviation_mm=bounds_deviation,
            message="topology-preserving slicer optimization accepted",
            max_deviation_mm=max_deviation,
            poor_triangle_fraction_before=poor_before,
            poor_triangle_fraction_after=poor_after,
            edge_p01_mm_before=edge_p01_before,
            edge_p01_mm_after=edge_p01_after,
        )
    except (
        ImportError,
        AttributeError,
        MemoryError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as exc:
        return source.copy(), MeshConditioningReport(
            attempted=True,
            applied=False,
            backend="unavailable",
            tolerance_mm=deviation_limit,
            removed_vertices=0,
            removed_faces=0,
            topology_preserved=False,
            bounds_deviation_mm=0.0,
            message=f"conditioning skipped: {type(exc).__name__}",
            poor_triangle_fraction_before=poor_before,
            edge_p01_mm_before=edge_p01_before,
        )


def _reconstruct_mesh_by_processing_mode(
    body: ImplicitBody,
    spacing: np.ndarray,
    grid_anchor_mm: np.ndarray,
    processing_mode: ProcessingMode,
    batch_count: int,
    progress: ProgressCallback | None,
    field_postprocessor: Callable[[np.ndarray], np.ndarray] | None = None,
) -> tuple[trimesh.Trimesh, ExtractionStatistics, str]:
    """Extract an STL mesh using the selected memory strategy."""

    steps = np.asarray(spacing, dtype=np.float64)
    origin, shape, axes = extraction_grid_definition(
        body,
        steps,
        grid_anchor_mm,
    )
    dense_points = int(np.prod(shape))
    evaluated_points = 0

    def evaluate_points(
        points: np.ndarray,
        _max_tile_points: int | None,
        stage_reporter: EvaluationStageCallback | None,
    ) -> np.ndarray:
        nonlocal evaluated_points
        evaluated_points += len(points)
        return body.evaluate_points(points, stage_reporter)

    sampling = SamplingParameters(
        target_voxels=max(dense_points, 100_000),
        batch_count=int(batch_count),
        processing_mode=processing_mode,
    )
    mesh = _generate_mesh_by_processing_mode(
        axes,
        shape,
        tuple(float(value) for value in steps),
        origin,
        sampling,
        evaluate_points,
        progress,
        "STL",
        field_postprocessor,
    )
    chunk_count = (
        len(_batch_ranges(shape[0] - 1, sampling.batch_count))
        if processing_mode == "chunked_marching_cubes"
        else 1
    )
    statistics = ExtractionStatistics(
        dense_grid_points=dense_points,
        evaluated_points=evaluated_points,
        total_bricks=chunk_count,
        active_bricks=chunk_count,
        skipped_bricks=0,
    )
    backend = {
        "single_pass": "dense-grid",
        "batched_field": "batched-field-dense-grid",
        "chunked_marching_cubes": "chunked-marching-cubes",
    }[processing_mode]
    return mesh, statistics, backend


def reconstruct_implicit_mesh(
    body: ImplicitBody,
    cell_map: CellMap,
    tolerance_mm: float,
    wall_thickness_mm: float,
    repair_tolerance_mm: float = 0.0,
    clean_numerical_fragments: bool = False,
    progress: ProgressCallback | None = None,
    processing_mode: ProcessingMode = "single_pass",
    batch_count: int = 1,
    spacing_mode: ExportSpacingMode = "recommended",
    optimize_for_slicing: bool = False,
    enforce_feature_limits: bool = True,
) -> ExportMeshResult:
    """Derive one STL-ready mesh from the authoritative implicit evaluator."""

    if enforce_feature_limits:
        spacing = resolve_export_spacing(
            cell_map,
            tolerance_mm,
            wall_thickness_mm,
            spacing_mode,
        )
        spacing_limits = export_spacing_limit_labels(
            spacing,
            cell_map,
            tolerance_mm,
            wall_thickness_mm,
            spacing_mode,
        )
    else:
        spacing = np.full(3, float(tolerance_mm), dtype=np.float64)
        spacing_limits = ("设计域 STL 容差",) * 3
    field_repaired = False
    field_fragment_cleanup: FieldFragmentCleanupReport | None = None
    extraction_fallback_reason = None
    effective_processing_mode = processing_mode
    if effective_processing_mode not in (
        "single_pass",
        "batched_field",
        "chunked_marching_cubes",
    ):
        raise ValueError("processing_mode is invalid")
    if batch_count < 1:
        raise ValueError("batch_count must be at least 1")
    if effective_processing_mode == "single_pass" and batch_count != 1:
        raise ValueError("single_pass requires batch_count=1")
    if effective_processing_mode != "single_pass" and batch_count < 2:
        raise ValueError("batched modes require batch_count at least 2")
    if repair_tolerance_mm > 0.0 and effective_processing_mode == "chunked_marching_cubes":
        extraction_fallback_reason = (
            "场修复需要完整规则场，分块 Marching Cubes 已安全切换为完整场修复流程"
        )

    if repair_tolerance_mm <= 0.0:
        field_cleanup_holder: list[FieldFragmentCleanupReport | None] = [None]

        def postprocess_field(values: np.ndarray) -> np.ndarray:
            cleaned_values, report = remove_small_negative_field_components(
                values,
                spacing,
                tolerance_mm,
                relative_volume_limit=0.02,
                component_extent_limit_mm=recommended_floating_component_extent_mm(
                    cell_map,
                    tolerance_mm,
                    wall_thickness_mm,
                ),
            )
            field_cleanup_holder[0] = report
            return cleaned_values

        if progress:
            progress("STL：按当前计算模式准备隐式场", 0.05)
        mesh, extraction_statistics, extraction_backend = (
            _reconstruct_mesh_by_processing_mode(
                body,
                spacing,
                np.asarray(cell_map.bounds[0], dtype=np.float64),
                effective_processing_mode,
                batch_count,
                progress,
                postprocess_field if clean_numerical_fragments else None,
            )
        )
        field_fragment_cleanup = field_cleanup_holder[0]
        extraction_fallback_reason = None
        if progress:
            progress("STL：完成当前计算模式的 Marching Cubes", 0.78)
    else:
        lo, shape, axes = extraction_grid_definition(
            body,
            spacing,
            np.asarray(cell_map.bounds[0], dtype=np.float64),
        )

        def evaluate_tile(points: np.ndarray, tile_index: int, tile_count: int) -> np.ndarray:
            if progress:
                progress(
                    f"导出重建：计算隐式场（微分片 {tile_index}/{tile_count}）",
                    0.05 + 0.65 * tile_index / max(tile_count, 1),
                )
            return body.evaluate_points(points)

        field = _sample_grid_points(
            axes,
            shape,
            0,
            shape[0],
            evaluate_tile,
            max_tile_points=250_000,
        )
        # Field repair is a global morphological operation and intentionally
        # remains on the complete regular grid.
        halo_value = max(float(np.max(np.abs(field))), float(np.max(spacing)), 1.0e-6)
        field[[0, -1], :, :] = np.maximum(field[[0, -1], :, :], halo_value)
        field[:, [0, -1], :] = np.maximum(field[:, [0, -1], :], halo_value)
        field[:, :, [0, -1]] = np.maximum(field[:, :, [0, -1]], halo_value)
        field, field_repaired = _close_small_field_gaps(
            field,
            spacing,
            repair_tolerance_mm,
        )
        if clean_numerical_fragments:
            field, field_fragment_cleanup = remove_small_negative_field_components(
                field,
                spacing,
                tolerance_mm,
                relative_volume_limit=0.02,
                component_extent_limit_mm=recommended_floating_component_extent_mm(
                    cell_map,
                    tolerance_mm,
                    wall_thickness_mm,
                ),
            )
        if progress:
            progress("导出重建：执行全局 Marching Cubes", 0.78)
        mesh = _marching_cubes_field(
            field,
            tuple(float(value) for value in spacing),
            lo,
        )
        dense_points = int(np.prod(shape))
        extraction_backend = "dense-grid-repair"
        extraction_statistics = ExtractionStatistics(
            dense_grid_points=dense_points,
            evaluated_points=dense_points,
            total_bricks=1,
            active_bricks=1,
            skipped_bricks=0,
        )
    conditioning = None
    quality = inspect_mesh(mesh)
    if (
        quality.finite
        and quality.non_empty
        and (not quality.watertight or not quality.winding_consistent)
    ):
        if progress:
            progress("导出重建：修复网格拓扑", 0.86)
        mesh = repair_mesh(mesh, fill_holes=False)
        quality = inspect_mesh(mesh)
    fragment_cleanup = None
    if clean_numerical_fragments:
        if progress:
            progress("导出重建：清理数值碎片", 0.92)
        floating_extent_limit = recommended_floating_component_extent_mm(
            cell_map,
            tolerance_mm,
            wall_thickness_mm,
        )
        mesh, fragment_cleanup = remove_numerical_fragments(
            mesh,
            tolerance_mm,
            relative_volume_limit=0.02,
            component_extent_limit_mm=floating_extent_limit,
        )
        if fragment_cleanup.removed_components:
            quality = inspect_mesh(mesh)
    if optimize_for_slicing:
        if progress:
            progress("STL：执行拓扑保持的切片优化", 0.96)
        mesh, conditioning = _condition_mesh_for_slicing(
            mesh,
            float(tolerance_mm),
            np.asarray(spacing, dtype=np.float64),
            quality,
        )
        quality = inspect_mesh(mesh)
    if progress:
        progress("导出重建：完成拓扑检查", 1.0)
    return ExportMeshResult(
        mesh=mesh,
        quality=quality,
        spacing_mm=tuple(float(value) for value in spacing),
        tolerance_mm=float(tolerance_mm),
        spacing_limits=spacing_limits,
        field_repaired=field_repaired,
        field_fragment_cleanup=field_fragment_cleanup,
        fragment_cleanup=fragment_cleanup,
        extraction_backend=extraction_backend,
        extraction_statistics=extraction_statistics,
        extraction_fallback_reason=extraction_fallback_reason,
        spacing_mode=spacing_mode,
        conditioning=conditioning,
    )


def _compiled_simplify_mesh(mesh: trimesh.Trimesh, target_faces: int) -> tuple[trimesh.Trimesh, str]:
    """Run a compiled topology-preserving simplifier and return its backend."""

    try:
        import pyvista as pv

        faces = np.column_stack(
            (np.full(len(mesh.faces), 3, dtype=np.int32), mesh.faces.astype(np.int32))
        ).ravel()
        source = pv.PolyData(np.asarray(mesh.vertices, dtype=np.float64), faces)
        reduction = 1.0 - float(target_faces) / max(len(mesh.faces), 1)
        output = source.decimate_pro(
            reduction=float(np.clip(reduction, 0.0, 0.99)),
            preserve_topology=True,
            boundary_vertex_deletion=False,
        )
        output_faces = np.asarray(output.faces, dtype=np.int64)
        if output_faces.size == 0 or output_faces.size % 4:
            raise RuntimeError("VTK simplifier returned a non-triangular mesh")
        output_faces = output_faces.reshape(-1, 4)
        if not np.all(output_faces[:, 0] == 3):
            raise RuntimeError("VTK simplifier returned non-triangular cells")
        return (
            trimesh.Trimesh(
                vertices=np.asarray(output.points, dtype=np.float64),
                faces=output_faces[:, 1:4],
                process=False,
            ),
            "VTK C++",
        )
    except Exception as vtk_error:
        try:
            import open3d as o3d  # type: ignore

            o3m = o3d.geometry.TriangleMesh(
                vertices=o3d.utility.Vector3dVector(mesh.vertices),
                triangles=o3d.utility.Vector3iVector(mesh.faces),
            )
            o3m.remove_duplicated_vertices()
            o3m.remove_duplicated_triangles()
            o3m.remove_degenerate_triangles()
            o3m.remove_unreferenced_vertices()
            simplified = o3m.simplify_quadric_decimation(int(target_faces))
            return (
                trimesh.Trimesh(
                    vertices=np.asarray(simplified.vertices, dtype=np.float64),
                    faces=np.asarray(simplified.triangles, dtype=np.int64),
                    process=False,
                ),
                f"Open3D C++（VTK: {type(vtk_error).__name__}）",
            )
        except Exception as open3d_error:
            raise RuntimeError(
                f"compiled simplifier unavailable: {type(vtk_error).__name__}; "
                f"{type(open3d_error).__name__}"
            ) from open3d_error


def _mesh_deviation_mm(
    source: trimesh.Trimesh,
    candidate: trimesh.Trimesh,
    max_samples: int = 100_000,
) -> float:
    """Estimate symmetric surface deviation with VTK's compiled locator."""

    import pyvista as pv

    def sampled_vertices(mesh: trimesh.Trimesh) -> np.ndarray:
        vertices = np.asarray(mesh.vertices, dtype=np.float64)
        if len(vertices) <= max_samples:
            return vertices
        indices = np.linspace(0, len(vertices) - 1, max_samples, dtype=np.int64)
        return vertices[indices]

    source_surface = pv.wrap(source)
    candidate_surface = pv.wrap(candidate)
    candidate_cloud = pv.PolyData(sampled_vertices(candidate))
    source_cloud = pv.PolyData(sampled_vertices(source))
    candidate_distances = np.asarray(
        candidate_cloud.compute_implicit_distance(source_surface, inplace=False)[
            "implicit_distance"
        ],
        dtype=np.float64,
    )
    source_distances = np.asarray(
        source_cloud.compute_implicit_distance(candidate_surface, inplace=False)[
            "implicit_distance"
        ],
        dtype=np.float64,
    )
    return max(
        float(np.max(np.abs(candidate_distances), initial=0.0)),
        float(np.max(np.abs(source_distances), initial=0.0)),
    )


def simplify_mesh_with_quality(
    mesh: trimesh.Trimesh,
    target_faces: int,
    tolerance_mm: float,
) -> tuple[trimesh.Trimesh, SimplificationReport]:
    """Simplify, repair topology regressions, and reject unsafe results."""

    if tolerance_mm <= 0.0 or not np.isfinite(tolerance_mm):
        raise ValueError("tolerance_mm must be finite and positive")
    before = inspect_mesh(mesh)
    if target_faces <= 0 or len(mesh.faces) <= target_faces:
        return mesh.copy(), SimplificationReport(
            True, "未执行", before, before, "面片数未超过目标，保留原始网格"
        )
    try:
        candidate, backend = _compiled_simplify_mesh(mesh, int(target_faces))
    except RuntimeError as exc:
        return mesh.copy(), SimplificationReport(
            False, "不可用", before, before, f"简化失败，已回退：{exc}"
        )
    after = inspect_mesh(candidate)
    repair_attempted = False
    repaired = False
    repair_details: list[str] = []

    def topology_accepted(quality: MeshQuality) -> bool:
        return bool(
            quality.finite
            and quality.non_empty
            and quality.watertight
            and quality.winding_consistent
            and quality.connected_components == 1
        )

    if after.finite and after.non_empty and not topology_accepted(after):
        repair_attempted = True
        try:
            candidate = repair_mesh(candidate, fill_holes=True)
            after = inspect_mesh(candidate)
            repair_details.append("保守网格修复")
            if after.connected_components > 1:
                candidate, cleanup = remove_numerical_fragments(
                    candidate,
                    tolerance_mm=float(tolerance_mm),
                )
                after = inspect_mesh(candidate)
                repair_details.append(
                    f"数值碎片清理 {cleanup.removed_components} 个"
                )
            repaired = topology_accepted(after)
        except (TypeError, ValueError, RuntimeError) as exc:
            repair_details.append(f"修复异常：{type(exc).__name__}: {exc}")

    if not topology_accepted(after):
        detail = "；".join(repair_details) if repair_details else "未执行修复"
        return mesh.copy(), SimplificationReport(
            False,
            backend,
            before,
            after,
            f"简化后拓扑不合格，自动修复未通过（{detail}），已回退到原始网格",
            repair_attempted=repair_attempted,
            repaired=False,
        )
    try:
        deviation = _mesh_deviation_mm(mesh, candidate)
    except (AttributeError, RuntimeError, ValueError) as exc:
        return mesh.copy(), SimplificationReport(
            False,
            backend,
            before,
            after,
            f"无法验证几何容差，已回退：{type(exc).__name__}: {exc}",
            repair_attempted=repair_attempted,
            repaired=repaired,
        )
    if deviation > float(tolerance_mm):
        return mesh.copy(), SimplificationReport(
            False,
            backend,
            before,
            after,
            f"最大偏差 {deviation:.4f} mm 超过容差，已回退",
            deviation,
            repair_attempted,
            repaired,
        )
    repair_message = ""
    if repaired:
        repair_message = f"，自动修复通过（{'；'.join(repair_details)}）"
    return candidate, SimplificationReport(
        True,
        backend,
        before,
        after,
        f"简化通过（{backend}）{repair_message}，最大偏差 {deviation:.4f} mm",
        deviation,
        repair_attempted,
        repaired,
    )


def cell_map_for_parameters(
    mesh: trimesh.Trimesh,
    parameters: LatticeParameters,
) -> CellMap:
    """Build the selected Cell Map associated with one lattice family."""

    parameters.validate()
    default_origin = np.asarray(mesh.bounds[0], dtype=np.float64)
    frame = parameters.cell_map_frame(default_origin)
    uses_default_frame = frame.is_world_aligned and np.allclose(
        frame.origin,
        default_origin,
        rtol=0.0,
        atol=1.0e-9,
    )
    manual_counts = getattr(parameters, "manual_cell_counts", None)
    if manual_counts is not None:
        return CellMap.from_frame_counts(
            frame,
            parameters.cell_size_xyz_mm,
            manual_counts,
        )
    if uses_default_frame and parameters.cell_map_mode == "fit_bounds":
        return CellMap.from_bounds(
            mesh.bounds,
            parameters.cell_size_xyz_mm,
            boundary_mode="fit_bounds",
        )
    return CellMap.from_points(
        mesh.vertices,
        parameters.cell_size_xyz_mm,
        frame,
    )


def _parameters_for_cell_map(
    parameters: TPMSParameters,
    cell_map: CellMap,
) -> TPMSParameters:
    """Return physical TPMS parameters using one map's actual cell spacing."""

    effective = replace(
        parameters,
        cell_size_mm=cell_map.spacing_mm,
        frame_origin_mm=tuple(float(value) for value in cell_map.frame.origin),
        frame_u_axis=tuple(float(value) for value in cell_map.frame.u_axis),
        frame_v_axis=tuple(float(value) for value in cell_map.frame.v_axis),
    )
    effective.validate()
    return effective


def transition_cell_maps(
    mesh: trimesh.Trimesh,
    g_parameters: TPMSParameters,
    d_parameters: TPMSParameters,
) -> tuple[CellMap, CellMap, CellMap]:
    """Build independent family maps and their common sampling map."""

    g_map = cell_map_for_parameters(mesh, g_parameters)
    d_map = cell_map_for_parameters(mesh, d_parameters)
    union_bounds = np.stack(
        (
            np.minimum(g_map.bounds[0], d_map.bounds[0]),
            np.maximum(g_map.bounds[1], d_map.bounds[1]),
        )
    )
    shared_spacing = np.minimum(g_map.spacing_mm, d_map.spacing_mm)
    shared_mode: CellMapMode = (
        "fit_bounds"
        if g_map.boundary_mode == d_map.boundary_mode == "fit_bounds"
        else "complete_cells"
    )
    sampling_map = CellMap.from_bounds(
        union_bounds,
        shared_spacing,
        boundary_mode=shared_mode,
    )
    return g_map, d_map, sampling_map


def transition_cell_map(
    mesh: trimesh.Trimesh,
    g_parameters: TPMSParameters,
    d_parameters: TPMSParameters,
) -> CellMap:
    """Build one envelope containing both independent family cell maps."""

    return transition_cell_maps(mesh, g_parameters, d_parameters)[2]


def lattice_transition_cell_maps(
    mesh: trimesh.Trimesh,
    first_parameters: LatticeParameters,
    second_parameters: LatticeParameters,
) -> tuple[CellMap, CellMap, CellMap]:
    """Build independent maps and one common sampling envelope for any pair."""

    first_map = cell_map_for_parameters(mesh, first_parameters)
    second_map = cell_map_for_parameters(mesh, second_parameters)
    union_bounds = np.stack(
        (
            np.minimum(first_map.bounds[0], second_map.bounds[0]),
            np.maximum(first_map.bounds[1], second_map.bounds[1]),
        )
    )
    shared_spacing = np.minimum(first_map.spacing_mm, second_map.spacing_mm)
    shared_mode: CellMapMode = (
        "fit_bounds"
        if first_map.boundary_mode == second_map.boundary_mode == "fit_bounds"
        else "complete_cells"
    )
    shared_map = CellMap.from_bounds(
        union_bounds,
        shared_spacing,
        boundary_mode=shared_mode,
    )
    return first_map, second_map, shared_map


def lattice_sampling_recommendation(
    mesh: trimesh.Trimesh,
    parameters: LatticeParameters,
    sampling: SamplingParameters,
    voxel_size_mm: float | None = None,
) -> SamplingRecommendation:
    """Return the sampling recommendation for one family's padded Cell Map."""

    cell_map = cell_map_for_parameters(mesh, parameters)
    if isinstance(parameters, CustomUnitCellParameters):
        return custom_cell_sampling_recommendation(
            mesh,
            parameters,
            sampling,
            voxel_size_mm,
            cell_map=cell_map,
        )
    effective_parameters = _parameters_for_cell_map(parameters, cell_map)
    if voxel_size_mm is None:
        return recommend_voxel_size(
            mesh,
            effective_parameters,
            sampling,
            cell_map=cell_map,
        )
    return recommendation_for_voxel_size(
        mesh,
        effective_parameters,
        sampling,
        voxel_size_mm,
        cell_map=cell_map,
    )


def custom_cell_sampling_recommendation(
    mesh: trimesh.Trimesh,
    parameters: CustomUnitCellParameters,
    sampling: SamplingParameters,
    voxel_size_mm: float | None = None,
    *,
    cell_map: CellMap | None = None,
    minimum_feature_mm: float | None = None,
) -> SamplingRecommendation:
    """Recommend a dense grid for a fixed-geometry custom cell."""

    parameters.validate()
    sampling.validate()
    cell_map = cell_map or cell_map_for_parameters(mesh, parameters)
    extents = np.asarray(cell_map.bounds[1] - cell_map.bounds[0], dtype=np.float64)
    cell_sizes = np.asarray(cell_map.spacing_mm, dtype=np.float64)
    if voxel_size_mm is None:
        period_step = float(np.min(cell_sizes)) / max(
            sampling.min_samples_per_cell,
            32.0,
        )
        feature_step = (
            float(minimum_feature_mm) / sampling.min_samples_per_wall
            if minimum_feature_mm is not None
            else period_step
        )
        budget_step = (
            float(np.prod(extents)) / sampling.target_voxels
        ) ** (1.0 / 3.0)
        step = min(period_step, feature_step, budget_step)
        limiting_constraint = min(
            (
                (period_step, "晶胞周期采样"),
                (feature_step, "最小特征厚度采样"),
                (budget_step, "目标体素数"),
            ),
            key=lambda item: item[0],
        )[1]
    else:
        step = float(voxel_size_mm)
        if step <= 0.0 or not np.isfinite(step):
            raise ValueError("voxel_size_mm must be a finite positive number")
        limiting_constraint = "用户指定基础体素"
    shape = tuple(int(np.ceil(axis / step)) + 3 for axis in extents)
    feature = float(minimum_feature_mm or np.min(cell_sizes) / 32.0)
    return SamplingRecommendation(
        voxel_size_mm=step,
        grid_shape=shape,
        estimated_voxels=int(np.prod(shape)),
        samples_per_cell=float(np.min(cell_sizes) / step),
        samples_per_wall=float(feature / step),
        limiting_constraint=limiting_constraint,
    )


def recommend_voxel_size(
    mesh: trimesh.Trimesh,
    parameters: TPMSParameters,
    sampling: SamplingParameters | None = None,
    cell_map: CellMap | None = None,
) -> SamplingRecommendation:
    """Recommend a grid that resolves both the cell and the wall.

    The initial step is constrained by the two physical feature sizes.  If
    the resulting AABB grid is too large, the target voxel budget is used as
    a recommendation only.  There is no hard maximum-voxel clamp, so the
    returned grid may exceed the target when wall/cell sampling requires it.
    """

    parameters.validate()
    sampling = sampling or SamplingParameters()
    sampling.validate()
    extents = np.asarray(
        (
            cell_map.bounds[1] - cell_map.bounds[0]
            if cell_map is not None
            else mesh.bounds[1] - mesh.bounds[0]
        ),
        dtype=float,
    )
    if np.any(extents <= 0):
        raise ValueError("design domain has a zero-size axis")

    cell_sizes = np.asarray(parameters.cell_size_xyz_mm, dtype=np.float64)
    cell_step = float(np.min(cell_sizes)) / sampling.min_samples_per_cell
    wall_thickness = parameters.minimum_wall_thickness_mm
    wall_step = wall_thickness / sampling.min_samples_per_wall
    feature_step = min(cell_step, wall_step)
    budget_step = (float(np.prod(extents)) / sampling.target_voxels) ** (1.0 / 3.0)

    def shape_for(step: float) -> tuple[int, int, int]:
        return tuple(int(np.ceil(axis / step)) + 3 for axis in extents)

    if (
        parameters.gradient_enabled
        and parameters.gradient_resolution_strategy == "matlab_def"
    ):
        voxel = matlab_gradient_voxel_size(
            cell_sizes,
            parameters.gradient_def_per_half_cell,
        )
        shape = shape_for(voxel)
        return SamplingRecommendation(
            voxel_size_mm=float(voxel),
            grid_shape=shape,
            estimated_voxels=int(np.prod(shape)),
            samples_per_cell=float(np.min(cell_sizes) / voxel),
            samples_per_wall=float(wall_thickness / voxel),
            limiting_constraint="MATLAB Def 分辨率",
        )

    # The physical feature size is a hard lower bound on the resolution.
    # ``target_voxels`` is only a recommendation: when it asks for a coarser
    # grid than the wall/cell sampling requires, keep the finer physical grid
    # instead of silently violating the sampling guarantees.
    voxel = min(feature_step, budget_step)
    limiting_constraint = min(
        (
            (cell_step, "晶胞周期采样"),
            (wall_step, "壁厚采样"),
            (budget_step, "目标体素数"),
        ),
        key=lambda item: item[0],
    )[1]

    shape = shape_for(voxel)
    voxels = int(np.prod(shape))
    return SamplingRecommendation(
        voxel_size_mm=float(voxel),
        grid_shape=shape,
        estimated_voxels=voxels,
        samples_per_cell=float(np.min(cell_sizes) / voxel),
        samples_per_wall=float(wall_thickness / voxel),
        limiting_constraint=limiting_constraint,
    )


def recommendation_for_voxel_size(
    mesh: trimesh.Trimesh,
    parameters: TPMSParameters,
    sampling: SamplingParameters,
    voxel_size_mm: float,
    cell_map: CellMap | None = None,
) -> SamplingRecommendation:
    """Return the exact grid recommendation for a user-selected voxel size."""

    if voxel_size_mm <= 0 or not np.isfinite(voxel_size_mm):
        raise ValueError("voxel_size_mm must be a finite positive number")
    base = recommend_voxel_size(mesh, parameters, sampling, cell_map=cell_map)
    extents = np.asarray(
        (
            cell_map.bounds[1] - cell_map.bounds[0]
            if cell_map is not None
            else mesh.bounds[1] - mesh.bounds[0]
        ),
        dtype=float,
    )
    step = float(voxel_size_mm)
    shape = tuple(int(np.ceil(axis / step)) + 3 for axis in extents)
    return replace(
        base,
        voxel_size_mm=step,
        grid_shape=shape,
        estimated_voxels=int(np.prod(shape)),
        samples_per_cell=float(np.min(parameters.cell_size_xyz_mm) / step),
        samples_per_wall=float(parameters.minimum_wall_thickness_mm / step),
        limiting_constraint="用户指定基础体素",
    )


def _grid_definition(
    mesh: trimesh.Trimesh,
    recommendation: SamplingRecommendation,
    bounds: np.ndarray | None = None,
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Return the lower grid origin and the three regularly spaced axes."""

    source_bounds = np.asarray(bounds if bounds is not None else mesh.bounds, dtype=float)
    lo = source_bounds[0] - recommendation.voxel_size_mm
    axes = [
        lo[i] + np.arange(recommendation.grid_shape[i], dtype=float) * recommendation.voxel_size_mm
        for i in range(3)
    ]
    return lo, axes


def recommend_display_voxel_size(
    mesh: trimesh.Trimesh,
    base_voxel_size_mm: float,
    display_memory_budget_mb: float = 512.0,
    visible_field_count: int = 1,
) -> float:
    """Choose a preview spacing that fits the independent GPU cache budget."""

    if base_voxel_size_mm <= 0 or not np.isfinite(base_voxel_size_mm):
        raise ValueError("base_voxel_size_mm must be finite and positive")
    if display_memory_budget_mb <= 0 or not np.isfinite(display_memory_budget_mb):
        raise ValueError("display_memory_budget_mb must be finite and positive")
    if visible_field_count < 1:
        raise ValueError("visible_field_count must be at least 1")

    extents = np.asarray(mesh.bounds[1] - mesh.bounds[0], dtype=np.float64)
    if np.any(extents <= 0):
        raise ValueError("design domain has a zero-size axis")
    budget_bytes = float(display_memory_budget_mb) * 1024.0 * 1024.0
    max_voxels = max(8.0, budget_bytes / (4.0 * visible_field_count))
    step = float(base_voxel_size_mm)
    for _ in range(8):
        shape = np.ceil(extents / step).astype(np.int64) + 3
        estimated = float(np.prod(shape, dtype=np.int64))
        if estimated <= max_voxels:
            break
        step *= (estimated / max_voxels) ** (1.0 / 3.0) * 1.02
    return float(step)


def _display_recommendation(
    mesh: trimesh.Trimesh,
    base: SamplingRecommendation,
    display_voxel_size_mm: float | None,
    display_memory_budget_mb: float,
    visible_field_count: int,
) -> SamplingRecommendation:
    step = display_voxel_size_mm
    if step is None:
        step = recommend_display_voxel_size(
            mesh,
            base.voxel_size_mm,
            display_memory_budget_mb,
            visible_field_count,
        )
    return replace(
        base,
        voxel_size_mm=float(step),
        grid_shape=tuple(
            int(np.ceil(axis / float(step))) + 3
            for axis in (mesh.bounds[1] - mesh.bounds[0])
        ),
        estimated_voxels=int(
            np.prod(
                tuple(
                    int(np.ceil(axis / float(step))) + 3
                    for axis in (mesh.bounds[1] - mesh.bounds[0])
                )
            )
        ),
        samples_per_cell=base.samples_per_cell * base.voxel_size_mm / float(step),
        samples_per_wall=base.samples_per_wall * base.voxel_size_mm / float(step),
    )


def estimate_cell_display_samples(
    cell_map: CellMap,
    display_voxel_size_mm: float,
) -> CellDisplaySampleEstimate:
    """Estimate linear and volumetric samples for one mapped unit cell."""

    step = float(display_voxel_size_mm)
    if step <= 0.0 or not np.isfinite(step):
        raise ValueError("display_voxel_size_mm must be finite and positive")
    periods = np.asarray(cell_map.spacing_mm, dtype=np.float64)
    if periods.shape != (3,) or not np.isfinite(periods).all() or np.any(periods <= 0.0):
        raise ValueError("cell_map spacing must contain three finite positive values")
    samples = periods / step
    return CellDisplaySampleEstimate(
        samples_per_axis=tuple(float(value) for value in samples),
        voxel_samples_per_cell=float(np.prod(samples)),
    )


def _sample_implicit_body_on_grid(
    body: ImplicitBody,
    axes: list[np.ndarray],
    origin: np.ndarray,
    spacing: np.ndarray,
    color: tuple[float, float, float, float],
    progress: ProgressCallback | None = None,
    label: str = "隐式体",
    display_batch_count: int | None = None,
) -> SampledImplicitField:
    """Sample an evaluator onto an explicit regular display grid."""

    shape = tuple(len(axis) for axis in axes)
    if len(shape) != 3 or any(size < 2 for size in shape):
        raise ValueError("implicit display grid must have at least two samples per axis")

    def evaluate_tile(
        points: np.ndarray,
        batch_index: int,
        batch_count: int,
        tile_index: int,
        tile_count: int,
    ) -> np.ndarray:
        completed = (
            (batch_index - 1) + tile_index / max(tile_count, 1)
        ) / max(batch_count, 1)
        progress_suffix = _display_batch_progress_text(
            batch_index,
            batch_count,
            tile_index,
            tile_count,
        )
        if progress:
            progress(
                f"{label}：计算隐式场{progress_suffix}",
                0.10 + 0.80 * completed,
            )

        def report_stage(stage: str) -> None:
            if progress:
                progress(
                    f"{label}：{stage}{progress_suffix}",
                    0.10 + 0.80 * completed,
                )

        return body.evaluate_points(points, report_stage)

    values = _sample_display_grid_points(
        axes,
        shape,
        evaluate_tile,
        display_batch_count,
    )
    if progress:
        progress(f"{label}：完成显示缓存", 0.95)
    return SampledImplicitField(
        name=body.name,
        values=values,
        origin=np.asarray(origin, dtype=np.float64),
        spacing=np.asarray(spacing, dtype=np.float64),
        color=color,
        is_preview_only=body.is_preview_only,
    )


def _sample_implicit_body(
    body: ImplicitBody,
    mesh: trimesh.Trimesh,
    recommendation: SamplingRecommendation,
    color: tuple[float, float, float, float],
    progress: ProgressCallback | None = None,
    label: str = "隐式体",
    bounds: np.ndarray | None = None,
    display_batch_count: int | None = None,
) -> SampledImplicitField:
    """Sample an evaluator into one bounded display cache."""

    origin, axes = _grid_definition(mesh, recommendation, bounds=bounds)
    return _sample_implicit_body_on_grid(
        body,
        axes,
        origin,
        np.full(3, recommendation.voxel_size_mm, dtype=np.float64),
        color,
        progress,
        label,
        display_batch_count,
    )


def refine_implicit_display_field(
    body: ImplicitBody,
    source: SampledImplicitField,
    *,
    quality: str,
    display_memory_budget_mb: float,
    display_batch_count: int | None = None,
    progress: ProgressCallback | None = None,
) -> tuple[SampledImplicitField, DisplayRefinementReport]:
    """Re-sample an authoritative body for display quality within a hard budget.

    Ray-march step size controls integration quality but cannot restore surface
    detail absent from a sampled scalar grid.  High and Ultra quality therefore
    evaluate the authoritative body on progressively finer grids.  The result is
    clamped to the caller's per-layer display budget and never replaces the
    authoritative body or the normal interactive cache.
    """

    if not isinstance(body, ImplicitBody):
        raise TypeError("body must be an ImplicitBody")
    if not isinstance(source, SampledImplicitField):
        raise TypeError("source must be a SampledImplicitField")
    budget_mb = float(display_memory_budget_mb)
    if not np.isfinite(budget_mb) or budget_mb <= 0.0:
        raise ValueError("display memory budget must be finite and positive")

    source_spacing = float(np.min(source.spacing))
    refinement_factor = DISPLAY_FIELD_REFINEMENT_FACTORS.get(quality)
    requested_spacing = source_spacing * (
        refinement_factor if refinement_factor is not None else 1.0
    )
    if refinement_factor is None:
        return source, DisplayRefinementReport(
            requested_spacing,
            source_spacing,
            source.values.size,
            False,
            False,
        )

    bounds = np.asarray(source.bounds, dtype=np.float64)
    extents = bounds[1] - bounds[0]
    budget_bytes = int(budget_mb * 1024 * 1024)
    max_voxels = budget_bytes // np.dtype(np.float32).itemsize

    def grid_shape(step: float) -> tuple[int, int, int]:
        return tuple(int(np.ceil(extent / step)) + 1 for extent in extents)

    applied_spacing = requested_spacing
    shape = grid_shape(applied_spacing)
    estimated_voxels = int(np.prod(shape, dtype=np.int64))
    budget_limited = estimated_voxels > max_voxels
    for _ in range(16):
        if estimated_voxels <= max_voxels:
            break
        applied_spacing *= (estimated_voxels / max_voxels) ** (1.0 / 3.0) * 1.002
        shape = grid_shape(applied_spacing)
        estimated_voxels = int(np.prod(shape, dtype=np.int64))

    if estimated_voxels > max_voxels or applied_spacing >= source_spacing:
        return source, DisplayRefinementReport(
            requested_spacing,
            source_spacing,
            source.values.size,
            False,
            True,
        )

    axes = [
        source.origin[axis]
        + np.arange(shape[axis], dtype=np.float64) * applied_spacing
        for axis in range(3)
    ]
    refined = _sample_implicit_body_on_grid(
        body,
        axes,
        source.origin,
        np.full(3, applied_spacing, dtype=np.float64),
        source.color,
        progress,
        f"{source.name} {DISPLAY_QUALITY_LABELS[quality]}显示",
        display_batch_count,
    )
    refined = replace(
        refined,
        metallic=source.metallic,
        roughness=source.roughness,
    )
    return refined, DisplayRefinementReport(
        requested_spacing,
        applied_spacing,
        refined.values.size,
        True,
        budget_limited,
    )


def _display_field_matches_grid(
    field: SampledImplicitField,
    mesh: trimesh.Trimesh,
    recommendation: SamplingRecommendation,
) -> bool:
    """Return whether a sampled domain cache matches the requested display grid."""

    if field.is_preview_only:
        return False
    expected_origin, _axes = _grid_definition(
        mesh,
        recommendation,
        bounds=mesh.bounds,
    )
    expected_spacing = np.full(3, recommendation.voxel_size_mm, dtype=np.float64)
    return bool(
        field.values.shape == recommendation.grid_shape
        and np.allclose(field.origin, expected_origin, rtol=0.0, atol=1.0e-9)
        and np.allclose(field.spacing, expected_spacing, rtol=0.0, atol=1.0e-9)
    )


def _sampled_fields_share_grid(
    first: SampledImplicitField,
    second: SampledImplicitField,
) -> bool:
    """Return whether two display caches can be combined point by point."""

    return bool(
        first.values.shape == second.values.shape
        and np.allclose(first.origin, second.origin, rtol=0.0, atol=1.0e-9)
        and np.allclose(first.spacing, second.spacing, rtol=0.0, atol=1.0e-9)
    )


def _display_field_meets_recommendation(
    field: SampledImplicitField,
    recommendation: SamplingRecommendation,
) -> bool:
    """Return whether an existing cache is at least as fine as a new request."""

    requested = float(recommendation.voxel_size_mm)
    tolerance = max(requested * 1.0e-6, 1.0e-9)
    return bool(
        not field.is_preview_only
        and np.all(np.asarray(field.spacing, dtype=np.float64) <= requested + tolerance)
    )


def _sample_intersection_with_domain_cache(
    name: str,
    domain_field: SampledImplicitField,
    feature_evaluator: Callable[[np.ndarray], np.ndarray],
    color: tuple[float, float, float, float],
    progress: ProgressCallback | None = None,
    progress_stage: str | None = None,
    display_batch_count: int | None = None,
) -> SampledImplicitField:
    """Sample a feature on the domain cache grid and reuse its SDF values."""

    if domain_field.is_preview_only:
        raise ValueError("a preview-only domain shell cannot constrain a TPMS result")
    shape = tuple(int(value) for value in domain_field.values.shape)
    axes = [
        domain_field.origin[axis]
        + np.arange(shape[axis], dtype=np.float64) * domain_field.spacing[axis]
        for axis in range(3)
    ]

    def evaluate_tile(
        points: np.ndarray,
        batch_index: int,
        batch_count: int,
        tile_index: int,
        tile_count: int,
    ) -> np.ndarray:
        if progress:
            stage = progress_stage or f"{name}：计算 TPMS 场 [{_tpms_backend_label()}]"
            completed = (
                (batch_index - 1) + tile_index / max(tile_count, 1)
            ) / max(batch_count, 1)
            progress(
                stage
                + _display_batch_progress_text(
                    batch_index,
                    batch_count,
                    tile_index,
                    tile_count,
                ),
                0.10 + 0.65 * completed,
            )
        return feature_evaluator(points)

    feature_values = _sample_display_grid_points(
        axes,
        shape,
        evaluate_tile,
        display_batch_count,
    )
    if progress:
        progress(f"{name}：复用设计域 SDF，执行交集", 0.88)
    feature_values = get_default_geometry_backend().intersect_fields(
        domain_field.values,
        feature_values,
        max_tile_points=250_000,
    )
    return SampledImplicitField(
        name=name,
        values=feature_values,
        origin=domain_field.origin,
        spacing=domain_field.spacing,
        color=color,
    )


def _make_design_domain_body(
    mesh: trimesh.Trimesh,
    sampling: SamplingParameters,
    shell_thickness_mm: float | None = None,
) -> ImplicitBody:
    """Create a signed design-domain evaluator or a non-watertight preview shell."""

    is_watertight = bool(mesh.is_watertight)
    shell_thickness = float(shell_thickness_mm or 0.5)
    if shell_thickness <= 0 or not np.isfinite(shell_thickness):
        raise ValueError("shell_thickness_mm must be finite and positive")

    def evaluate(points: np.ndarray, stage_reporter: EvaluationStageCallback | None) -> np.ndarray:
        if stage_reporter:
            stage_reporter(PROGRESS_DOMAIN_SDF)
        sdf = _compute_design_domain_sdf(mesh, points, None, sampling.use_cpp_sdf)
        if is_watertight:
            return sdf
        # An unsigned distance cannot define a unique solid interior.  Offset
        # it into a thin shell strictly for display-only preview.
        return np.abs(sdf) - shell_thickness

    return ImplicitBody(
        name="domain",
        bounds=np.asarray(mesh.bounds, dtype=np.float64),
        evaluate=evaluate,
        is_preview_only=not is_watertight,
    )


def vtk_bounds_from_implicit_body(body: ImplicitBody) -> np.ndarray:
    """Convert evaluator bounds to VTK's xmin/xmax/ymin/ymax/zmin/zmax order."""

    bounds = np.asarray(body.bounds, dtype=np.float64)
    if bounds.shape != (2, 3) or not np.isfinite(bounds).all():
        raise ValueError("implicit body bounds must be a finite 2x3 array")
    if not np.all(bounds[0] < bounds[1]):
        raise ValueError("implicit body bounds must have positive extents")
    return np.asarray(
        (
            bounds[0, 0], bounds[1, 0],
            bounds[0, 1], bounds[1, 1],
            bounds[0, 2], bounds[1, 2],
        ),
        dtype=np.float64,
    )


def _make_lattice_feature_body(
    mesh: trimesh.Trimesh,
    parameters: TPMSParameters,
    cell_map: CellMap,
) -> ImplicitBody:
    """Create a TPMS feature evaluator before design-domain clipping."""

    parameters.validate()
    gradient_bounds = (
        gradient_projection_range(
            mesh.vertices,
            cell_map.frame,
            parameters.gradient_axis,
        )
        if parameters.gradient_enabled
        else None
    )
    def evaluate(
        points: np.ndarray,
        stage_reporter: EvaluationStageCallback | None,
    ) -> np.ndarray:
        if stage_reporter:
            stage_reporter(f"{PROGRESS_TPMS_FIELD} [{_tpms_backend_label()}]")
        return _cell_map_tpms_shell_field(
            parameters.kind,
            points,
            parameters,
            cell_map=cell_map,
            max_tile_points=250_000,
            gradient_bounds_mm=gradient_bounds,
        )

    return ImplicitBody(
        name=f"{parameters.kind} feature",
        bounds=np.asarray(cell_map.bounds, dtype=np.float64),
        evaluate=evaluate,
    )


def _make_lattice_body(
    mesh: trimesh.Trimesh,
    parameters: TPMSParameters,
    sampling: SamplingParameters,
    cell_map: CellMap | None = None,
    feature_body: ImplicitBody | None = None,
) -> ImplicitBody:
    """Create one evaluator-backed TPMS field intersected with the domain."""

    parameters.validate()
    if len(mesh.faces) == 0 or not mesh.is_watertight:
        raise ValueError("TPMS lattice generation requires a non-empty watertight design domain")
    if cell_map is None:
        cell_map = cell_map_for_parameters(mesh, parameters)
    feature = feature_body or _make_lattice_feature_body(mesh, parameters, cell_map)
    effective = _parameters_for_cell_map(parameters, cell_map)
    field_spec = _tpms_field_spec(effective, cell_map)
    gradient_bounds = (
        gradient_projection_range(
            mesh.vertices,
            cell_map.frame,
            effective.gradient_axis,
        )
        if effective.gradient_enabled
        else None
    )
    gradient_spec = (
        make_tpms_gradient_spec(
            _gradient_controls(effective),
            gradient_bounds,
        )
        if gradient_bounds is not None
        else None
    )

    def evaluate(points: np.ndarray, stage_reporter: EvaluationStageCallback | None) -> np.ndarray:
        if sampling.use_cpp_sdf:
            if stage_reporter:
                stage_reporter(
                    "Device-resident SDF + TPMS + intersection "
                    f"[{_lattice_pipeline_backend_label()}]"
                )
            return get_default_lattice_pipeline().evaluate_mesh_clipped_tpms(
                mesh.vertices,
                mesh.faces,
                points,
                field_spec,
                gradient_spec,
                max_tile_points=250_000,
            )
        if stage_reporter:
            stage_reporter(PROGRESS_DOMAIN_SDF)
        domain_sdf = _compute_design_domain_sdf(mesh, points, None, sampling.use_cpp_sdf)
        tpms_sdf = feature.evaluate_points(points, stage_reporter)
        if stage_reporter:
            stage_reporter(PROGRESS_INTERSECTION)
        return implicit_intersection(domain_sdf, tpms_sdf)

    return ImplicitBody(
        name=parameters.kind,
        bounds=np.asarray(
            cell_map.bounds if cell_map is not None else mesh.bounds,
            dtype=np.float64,
        ),
        evaluate=evaluate,
    )


def _make_custom_lattice_feature_body(
    prepared: PreparedCustomUnitCell,
    cell_map: CellMap,
    parameters: CustomUnitCellParameters,
) -> ImplicitBody:
    """Create a periodic custom-cell evaluator before domain clipping."""

    def evaluate(
        points: np.ndarray,
        stage_reporter: EvaluationStageCallback | None,
    ) -> np.ndarray:
        if stage_reporter:
            stage_reporter("计算自定义晶胞场")
        return prepared.evaluate_periodic(
            points,
            cell_map,
            parameters.target_feature_mm,
            250_000,
            parameters.bridge_directions,
            parameters.bridge_depth_mm,
        )

    return ImplicitBody(
        name="Custom feature",
        bounds=np.asarray(cell_map.bounds, dtype=np.float64),
        evaluate=evaluate,
    )


def _make_transition_operand(
    mesh: trimesh.Trimesh,
    parameters: LatticeParameters,
    cell_map: CellMap,
    prepared_custom_cell: PreparedCustomUnitCell | None = None,
) -> tuple[TransitionOperand, PreparedCustomUnitCell | None]:
    """Adapt one configured lattice to the generic transition interface."""

    parameters.validate()
    frame_axes = tuple(
        tuple(float(value) for value in axis)
        for axis in cell_map.frame.axes
    )
    if isinstance(parameters, CustomUnitCellParameters):
        prepared = prepared_custom_cell
        if (
            prepared is None
            or prepared.report.source_path != parameters.source_path.resolve()
        ):
            prepared = prepare_stl_unit_cell(parameters.source_path)
        minimum_feature = parameters.effective_feature_mm(
            prepared.characteristic_feature_mm(cell_map)
        )

        def evaluate_custom(
            points: np.ndarray,
            _stage_reporter: EvaluationStageCallback | None,
        ) -> np.ndarray:
            return prepared.evaluate_periodic(
                points,
                cell_map,
                parameters.target_feature_mm,
                250_000,
                parameters.bridge_directions,
                parameters.bridge_depth_mm,
            )

        backend_name, supports_gpu = prepared.evaluation_backend
        raw_body = ImplicitBody(
            name="Custom",
            bounds=np.asarray(cell_map.bounds, dtype=np.float64),
            evaluate=evaluate_custom,
        )
        return (
            TransitionOperand(
                name="Custom",
                body=raw_body,
                characteristic_feature_mm=minimum_feature,
                cell_spacing_mm=tuple(float(value) for value in cell_map.spacing_mm),
                frame_axes_world=frame_axes,
                backend_name=backend_name,
                supports_gpu=supports_gpu,
            ),
            prepared,
        )

    status = get_default_tpms_backend().status
    gradient_bounds = (
        gradient_projection_range(
            mesh.vertices,
            cell_map.frame,
            parameters.gradient_axis,
        )
        if parameters.gradient_enabled
        else None
    )

    def evaluate_tpms(
        points: np.ndarray,
        _stage_reporter: EvaluationStageCallback | None,
    ) -> np.ndarray:
        return _cell_map_tpms_shell_field(
            parameters.kind,
            points,
            parameters,
            cell_map,
            max_tile_points=250_000,
            gradient_bounds_mm=gradient_bounds,
        )

    raw_body = ImplicitBody(
        name=parameters.kind,
        bounds=np.asarray(cell_map.bounds, dtype=np.float64),
        evaluate=evaluate_tpms,
    )
    return (
        TransitionOperand(
            name=parameters.kind,
            body=raw_body,
            characteristic_feature_mm=parameters.minimum_wall_thickness_mm,
            cell_spacing_mm=tuple(float(value) for value in cell_map.spacing_mm),
            frame_axes_world=frame_axes,
            backend_name=status.active_backend,
            supports_gpu=status.using_gpu,
        ),
        prepared_custom_cell,
    )


def _make_transition_body(
    mesh: trimesh.Trimesh,
    g_parameters: TPMSParameters,
    d_parameters: TPMSParameters,
    transition: TransitionParameters,
    sampling: SamplingParameters,
    g_cell_map: CellMap,
    d_cell_map: CellMap,
    cell_map: CellMap | None = None,
) -> ImplicitBody:
    """Create an evaluator-backed sigmoid-blended G/D transition."""

    if len(mesh.faces) == 0 or not mesh.is_watertight:
        raise ValueError("TPMS transition generation requires a non-empty watertight design domain")
    plane_point, plane_normal = transition_plane_geometry(mesh, transition)

    def evaluate(points: np.ndarray, stage_reporter: EvaluationStageCallback | None) -> np.ndarray:
        if stage_reporter:
            stage_reporter(PROGRESS_DOMAIN_SDF)
        domain_sdf = _compute_design_domain_sdf(mesh, points, None, sampling.use_cpp_sdf)
        if stage_reporter:
            stage_reporter(f"{PROGRESS_TPMS_FIELD} [{_tpms_backend_label()}]")
        return _tpms_transition_chunk_field(
            points,
            domain_sdf,
            g_parameters,
            d_parameters,
            plane_point,
            plane_normal,
            transition,
            g_cell_map,
            d_cell_map,
            max_tile_points=250_000,
            stage_progress=stage_reporter,
        )

    return ImplicitBody(
        name="Transition",
        bounds=np.asarray(
            cell_map.bounds if cell_map is not None else mesh.bounds,
            dtype=np.float64,
        ),
        evaluate=evaluate,
    )


def _batch_ranges(axis_length: int, batch_count: int) -> list[tuple[int, int]]:
    """Split the first grid axis into non-empty, nearly equal ranges."""

    count = min(max(int(batch_count), 1), axis_length)
    return [
        (
            (axis_length * index) // count,
            (axis_length * (index + 1)) // count,
        )
        for index in range(count)
    ]


def _grid_batch_points(
    axes: list[np.ndarray],
    shape: tuple[int, int, int],
    start: int,
    end: int,
) -> tuple[np.ndarray, tuple[int, int, int]]:
    """Construct one batch without materialising the complete 3-D grid.

    The point order is the same as ``meshgrid(..., indexing='ij').ravel()``:
    the last axis changes fastest, so the resulting field can be reshaped
    directly into ``(end - start, shape[1], shape[2])``.
    """

    nx, ny, nz = end - start, shape[1], shape[2]
    points = np.empty((nx * ny * nz, 3), dtype=np.float64)
    points[:, 0] = np.repeat(axes[0][start:end], ny * nz)
    points[:, 1] = np.tile(np.repeat(axes[1], nz), nx)
    points[:, 2] = np.tile(axes[2], nx * ny)
    return points, (nx, ny, nz)


def _sample_grid_points(
    axes: list[np.ndarray],
    shape: tuple[int, int, int],
    point_start: int,
    point_end: int,
    evaluate_points: Callable[[np.ndarray, int, int], np.ndarray],
    max_tile_points: int | None,
    out: np.ndarray | None = None,
) -> np.ndarray:
    """Evaluate a half-open grid range in point-bounded micro-slices."""

    if not 0 <= point_start < point_end <= shape[0]:
        raise ValueError("grid point range must be non-empty and inside the grid")
    x_count = point_end - point_start
    ny, nz = shape[1], shape[2]
    plane_point_count = ny * nz
    sample_count = x_count * plane_point_count
    if max_tile_points is None:
        tile_ranges = [(0, sample_count)]
    else:
        if max_tile_points < 1:
            raise ValueError("max_tile_points must be positive or None")
        tile_count = int(np.ceil(sample_count / max_tile_points))
        tile_ranges = _batch_ranges(sample_count, tile_count)

    expected_shape = (x_count, ny, nz)
    if out is None:
        result = np.empty(sample_count, dtype=np.float32)
    else:
        output = np.asarray(out)
        if output.shape != expected_shape or output.dtype != np.float32:
            raise ValueError(
                f"out must be a float32 array with shape {expected_shape}"
            )
        if not output.flags.c_contiguous:
            raise ValueError("out must be C-contiguous")
        result = output.reshape(-1)
    total_tiles = len(tile_ranges)
    for tile_index, (local_start, local_end) in enumerate(tile_ranges, start=1):
        flat_indices = np.arange(local_start, local_end, dtype=np.int64)
        local_x = flat_indices // plane_point_count
        remainder = flat_indices % plane_point_count
        local_y = remainder // nz
        local_z = remainder % nz
        points = np.empty((local_end - local_start, 3), dtype=np.float64)
        points[:, 0] = axes[0][point_start + local_x]
        points[:, 1] = axes[1][local_y]
        points[:, 2] = axes[2][local_z]
        values = np.asarray(
            evaluate_points(points, tile_index, total_tiles),
            dtype=np.float32,
        ).reshape(-1)
        if values.shape != (local_end - local_start,):
            raise ValueError(
                "evaluate_points returned "
                f"{values.shape}, expected {(local_end - local_start,)}"
            )
        result[local_start:local_end] = values
        del flat_indices, local_x, remainder, local_y, local_z, points, values
    return result.reshape(expected_shape)


def _display_batch_progress_text(
    batch_index: int,
    batch_count: int,
    tile_index: int,
    tile_count: int,
) -> str:
    if batch_count > 1:
        return (
            f"（批次 {batch_index}/{batch_count}，"
            f"微分片 {tile_index}/{tile_count}）"
        )
    return f"（微分片 {tile_index}/{tile_count}）"


def _sample_display_grid_points(
    axes: list[np.ndarray],
    shape: tuple[int, int, int],
    evaluate_points: Callable[
        [np.ndarray, int, int, int, int],
        np.ndarray,
    ],
    display_batch_count: int | None,
) -> np.ndarray:
    """Sample a display grid in optional outer batches and bounded micro-slices."""

    if display_batch_count is None:
        requested_batches = 1
    else:
        requested_batches = int(display_batch_count)
        if requested_batches != display_batch_count or requested_batches < 2:
            raise ValueError("display_batch_count must be None or an integer at least 2")
    batch_ranges = _batch_ranges(shape[0], requested_batches)
    values = np.empty(shape, dtype=np.float32)
    total_batches = len(batch_ranges)
    for batch_index, (point_start, point_end) in enumerate(
        batch_ranges,
        start=1,
    ):
        def evaluate_micro_slice(
            points: np.ndarray,
            tile_index: int,
            tile_count: int,
            current_batch: int = batch_index,
        ) -> np.ndarray:
            return evaluate_points(
                points,
                current_batch,
                total_batches,
                tile_index,
                tile_count,
            )

        _sample_grid_points(
            axes,
            shape,
            point_start,
            point_end,
            evaluate_micro_slice,
            max_tile_points=250_000,
            out=values[point_start:point_end],
        )
    return values


def _sample_grid_slab(
    axes: list[np.ndarray],
    shape: tuple[int, int, int],
    cell_start: int,
    cell_end: int,
    evaluate_points: Callable[[np.ndarray, int, int], np.ndarray],
) -> np.ndarray:
    """Evaluate one MC slab through bounded x-directed point tiles."""

    return _sample_grid_points(
        axes,
        shape,
        cell_start,
        cell_end + 1,
        evaluate_points,
        max_tile_points=250_000,
    )


def _stitch_mesh_chunks(
    chunks: list[tuple[np.ndarray, np.ndarray, np.ndarray | None]],
    merge_tolerance: float,
) -> trimesh.Trimesh:
    """Join Marching-Cubes chunks by edge identity or coordinate fallback."""

    if merge_tolerance <= 0 or not np.isfinite(merge_tolerance):
        raise ValueError("merge_tolerance must be a finite positive number")
    non_empty = [item for item in chunks if len(item[1])]
    if not non_empty:
        return trimesh.Trimesh(process=False)

    vertices = np.concatenate([item[0] for item in non_empty], axis=0)
    face_parts = []
    vertex_offset = 0
    for chunk_vertices, chunk_faces, _edge_ids in non_empty:
        face_parts.append(np.asarray(chunk_faces, dtype=np.int64) + vertex_offset)
        vertex_offset += len(chunk_vertices)
    faces = np.concatenate(face_parts, axis=0)

    if all(item[2] is not None for item in non_empty):
        vertex_keys = np.concatenate([item[2] for item in non_empty], axis=0)
        _, first_vertex_indices, inverse = np.unique(
            vertex_keys,
            return_index=True,
            return_inverse=True,
        )
    else:
        quantized = np.rint(vertices / merge_tolerance).astype(np.int64)
        _, first_vertex_indices, inverse = np.unique(
            quantized,
            axis=0,
            return_index=True,
            return_inverse=True,
        )
    vertices = vertices[first_vertex_indices]
    faces = inverse[faces]

    if any(item[2] is None for item in non_empty):
        non_degenerate = (
            (faces[:, 0] != faces[:, 1])
            & (faces[:, 1] != faces[:, 2])
            & (faces[:, 0] != faces[:, 2])
        )
        faces = faces[non_degenerate]
        if len(faces):
            face_keys = np.sort(faces, axis=1)
            _, unique_face_indices = np.unique(face_keys, axis=0, return_index=True)
            faces = faces[np.sort(unique_face_indices)]

    result = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    result.remove_unreferenced_vertices()
    if len(result.faces):
        result.fix_normals()
    return result


def marching_cubes_chunked(
    field_shape: tuple[int, int, int],
    spacing: tuple[float, float, float],
    origin: np.ndarray,
    batch_count: int,
    sample_chunk: Callable[[int, int], np.ndarray],
    level: float = 0.0,
    progress: ProgressCallback | None = None,
    progress_label: str | None = None,
    progress_start: float = 0.05,
    progress_end: float = 0.90,
) -> trimesh.Trimesh:
    """Extract and stitch an isosurface while keeping only one slab in memory.

    ``sample_chunk(start, end)`` must return the scalar samples for the cells
    in ``[start, end)`` including one shared endpoint, with shape
    ``(end - start + 1, ny, nz)``.  Adjacent chunks therefore share exactly
    one grid plane, while each cell belongs to only one Marching-Cubes call.
    """

    shape = tuple(int(value) for value in field_shape)
    if len(shape) != 3 or any(value < 2 for value in shape):
        raise ValueError("field_shape must contain three dimensions of at least 2")
    steps = np.asarray(spacing, dtype=float).reshape(-1)
    if steps.shape != (3,) or not np.isfinite(steps).all() or np.any(steps <= 0):
        raise ValueError("spacing must contain three finite positive values")
    offset = np.asarray(origin, dtype=float).reshape(-1)
    if offset.shape != (3,) or not np.isfinite(offset).all():
        raise ValueError("origin must contain three finite values")
    if batch_count < 1:
        raise ValueError("batch_count must be at least 1")
    if not 0.0 <= progress_start <= progress_end <= 1.0:
        raise ValueError("progress range must satisfy 0 <= start <= end <= 1")

    cell_ranges = _batch_ranges(shape[0] - 1, batch_count)
    chunks: list[tuple[np.ndarray, np.ndarray, np.ndarray | None]] = []
    for chunk_index, (cell_start, cell_end) in enumerate(cell_ranges, start=1):
        expected_shape = (cell_end - cell_start + 1, shape[1], shape[2])
        field_chunk = np.asarray(sample_chunk(cell_start, cell_end), dtype=np.float32)
        if field_chunk.shape != expected_shape:
            raise ValueError(
                f"sample_chunk returned {field_chunk.shape}, expected {expected_shape}"
            )
        if not np.isfinite(field_chunk).all():
            raise ValueError("sample_chunk returned non-finite scalar values")
        chunk_mesh, edge_ids = get_default_geometry_backend().marching_cubes_chunk(
            field_chunk,
            spacing=tuple(float(value) for value in steps),
            origin=np.zeros(3, dtype=np.float64),
            level=level,
            grid_offset=(cell_start, 0, 0),
            global_shape=shape,
        )
        vertices = np.asarray(chunk_mesh.vertices, dtype=float)
        faces = np.asarray(chunk_mesh.faces, dtype=np.int64)
        if len(faces):
            chunk_offset = offset + np.array(
                (cell_start * steps[0], 0.0, 0.0),
                dtype=float,
            )
            chunks.append((vertices + chunk_offset, faces, edge_ids))
        if progress:
            prefix = f"{progress_label}：" if progress_label else ""
            progress(
                f"{prefix}{PROGRESS_MARCHING_CUBES}（第 {chunk_index}/{len(cell_ranges)} 块）",
                progress_start
                + (progress_end - progress_start) * chunk_index / len(cell_ranges),
            )

    if progress:
        prefix = f"{progress_label}：" if progress_label else ""
        progress(f"{prefix}拼接分块网格", min(progress_end + 0.05, 0.98))
    tolerance = max(float(np.min(steps)) * 1.0e-5, 1.0e-9)
    return _stitch_mesh_chunks(chunks, tolerance)


def _marching_cubes_field(
    field: np.ndarray,
    spacing: tuple[float, float, float],
    origin: np.ndarray,
    level: float = 0.0,
) -> trimesh.Trimesh:
    """Extract one complete scalar field with one Marching-Cubes call."""

    scalar_field = np.asarray(field, dtype=np.float32)
    if scalar_field.ndim != 3 or any(size < 2 for size in scalar_field.shape):
        raise ValueError("field must be a three-dimensional grid with at least 2 samples per axis")
    if not np.isfinite(scalar_field).all():
        raise ValueError("field contains non-finite scalar values")
    return get_default_geometry_backend().marching_cubes(
        scalar_field,
        spacing=spacing,
        origin=origin,
        level=level,
    )


def _generate_mesh_by_processing_mode(
    axes: list[np.ndarray],
    field_shape: tuple[int, int, int],
    spacing: tuple[float, float, float],
    origin: np.ndarray,
    sampling: SamplingParameters,
    evaluate_points: Callable[
        [np.ndarray, int | None, EvaluationStageCallback | None],
        np.ndarray,
    ],
    progress: ProgressCallback | None,
    label: str,
    field_postprocessor: Callable[[np.ndarray], np.ndarray] | None = None,
) -> trimesh.Trimesh:
    """Route field evaluation and extraction through one of three modes."""

    sampling.validate()
    shape = tuple(int(value) for value in field_shape)
    if len(shape) != 3 or any(value < 2 for value in shape):
        raise ValueError("field_shape must contain three dimensions of at least 2")
    bounded_tile_points = 250_000

    if sampling.processing_mode == "single_pass":
        def report_single_stage(stage: str) -> None:
            if not progress:
                return
            stage_value = {
                PROGRESS_DOMAIN_SDF: 0.20,
                PROGRESS_TPMS_FIELD: 0.45,
                PROGRESS_INTERSECTION: 0.65,
            }.get(stage, 0.35)
            progress(f"{label}：{stage}", stage_value)

        points, point_shape = _grid_batch_points(axes, shape, 0, shape[0])
        values = np.asarray(
            evaluate_points(points, None, report_single_stage if progress else None),
            dtype=np.float32,
        )
        if values.shape == (int(np.prod(point_shape)),):
            values = values.reshape(point_shape)
        if values.shape != point_shape:
            raise ValueError(f"evaluate_points returned {values.shape}, expected {point_shape}")
        del points
        if field_postprocessor is not None:
            values = np.asarray(field_postprocessor(values), dtype=np.float32)
            if values.shape != point_shape:
                raise ValueError(
                    f"field_postprocessor returned {values.shape}, expected {point_shape}"
                )
        if progress:
            progress(f"{label}：{PROGRESS_MARCHING_CUBES}", 0.78)
        result = _marching_cubes_field(values, spacing, origin)
        del values
        return result

    if sampling.processing_mode == "batched_field":
        ranges = _batch_ranges(shape[0], sampling.batch_count)
        full_field = np.empty(shape, dtype=np.float32)
        for index, (point_start, point_end) in enumerate(ranges, start=1):
            def evaluate_batch_micro_slice(
                points: np.ndarray,
                micro_index: int,
                micro_count: int,
                batch_index: int = index,
            ) -> np.ndarray:
                def report_batch_stage(stage: str) -> None:
                    if not progress:
                        return
                    stage_offset = {
                        PROGRESS_DOMAIN_SDF: 0.0,
                        PROGRESS_TPMS_FIELD: 0.35,
                        PROGRESS_INTERSECTION: 0.70,
                    }.get(stage, 0.50)
                    batch_span = 0.55 / len(ranges)
                    micro_span = batch_span / micro_count
                    value = (
                        0.10
                        + batch_span * (batch_index - 1)
                        + micro_span * (micro_index - 1 + stage_offset)
                    )
                    progress(
                        f"{label}：第 {batch_index}/{len(ranges)} 批 · "
                        f"第 {micro_index}/{micro_count} 微分片 · {stage}",
                        value,
                    )

                return evaluate_points(
                    points,
                    bounded_tile_points,
                    report_batch_stage if progress else None,
                )

            slab = _sample_grid_points(
                axes,
                shape,
                point_start,
                point_end,
                evaluate_batch_micro_slice,
                max_tile_points=bounded_tile_points,
            )
            full_field[point_start:point_end] = slab
            del slab
        if field_postprocessor is not None:
            full_field = np.asarray(field_postprocessor(full_field), dtype=np.float32)
            if full_field.shape != shape:
                raise ValueError(
                    f"field_postprocessor returned {full_field.shape}, expected {shape}"
                )
        if progress:
            progress(f"{label}：{PROGRESS_MARCHING_CUBES}", 0.75)
        result = _marching_cubes_field(full_field, spacing, origin)
        del full_field
        return result

    if sampling.processing_mode == "chunked_marching_cubes":
        chunk_ranges = _batch_ranges(shape[0] - 1, sampling.batch_count)
        chunk_indices = {
            cell_range: index
            for index, cell_range in enumerate(chunk_ranges, start=1)
        }
        chunk_progress_start = 0.05
        chunk_progress_end = 0.90
        if progress:
            progress(f"{label}：划分计算分块（共 {len(chunk_ranges)} 块）", 0.02)

        def sample_chunk(cell_start: int, cell_end: int) -> np.ndarray:
            chunk_index = chunk_indices[(cell_start, cell_end)]
            chunk_span = (
                (chunk_progress_end - chunk_progress_start) / len(chunk_ranges)
            )
            chunk_base = chunk_progress_start + chunk_span * (chunk_index - 1)

            def evaluate_chunk_micro_slice(
                points: np.ndarray,
                micro_index: int,
                micro_count: int,
            ) -> np.ndarray:
                def report_chunk_stage(stage: str) -> None:
                    if not progress:
                        return
                    stage_offset = {
                        PROGRESS_DOMAIN_SDF: 0.05,
                        PROGRESS_TPMS_FIELD: 0.38,
                        PROGRESS_INTERSECTION: 0.70,
                    }.get(stage, 0.50)
                    micro_span = chunk_span / micro_count
                    value = (
                        chunk_base
                        + micro_span * (micro_index - 1 + stage_offset)
                    )
                    progress(
                        f"{label}：第 {chunk_index}/{len(chunk_ranges)} 块 · "
                        f"第 {micro_index}/{micro_count} 微分片 · {stage}",
                        value,
                    )

                return evaluate_points(
                    points,
                    bounded_tile_points,
                    report_chunk_stage if progress else None,
                )

            return _sample_grid_slab(
                axes,
                shape,
                cell_start,
                cell_end,
                evaluate_chunk_micro_slice,
            )

        return marching_cubes_chunked(
            shape,
            spacing,
            origin,
            sampling.batch_count,
            sample_chunk,
            progress=progress,
            progress_label=label,
            progress_start=chunk_progress_start,
            progress_end=chunk_progress_end,
        )

    raise ValueError(f"unsupported processing mode: {sampling.processing_mode}")


def _rotation_matrix(axis_index: int, angle_rad: float) -> np.ndarray:
    """Return the workbench's right-handed axis rotation."""

    c, s = np.cos(angle_rad), np.sin(angle_rad)
    if axis_index == 0:
        return np.array(((1, 0, 0), (0, c, -s), (0, s, c)), dtype=float)
    if axis_index == 1:
        return np.array(((c, 0, s), (0, 1, 0), (-s, 0, c)), dtype=float)
    if axis_index == 2:
        return np.array(((c, -s, 0), (s, c, 0), (0, 0, 1)), dtype=float)
    raise ValueError("axis_index must be 0, 1 or 2")


def transition_plane_geometry(
    mesh: trimesh.Trimesh,
    parameters: TransitionParameters,
) -> tuple[np.ndarray, np.ndarray]:
    """Return a transition-plane point and unit normal in world coordinates."""

    parameters.validate()
    axis_index = {"X": 0, "Y": 1, "Z": 2}[parameters.plane_axis]
    point = np.asarray(mesh.bounds.mean(axis=0), dtype=float)
    point[axis_index] = parameters.plane_position_mm
    normal = np.zeros(3, dtype=float)
    normal[axis_index] = 1.0

    # Preserve the established UI mapping: X -> (Y, Z), Y -> (X, Z),
    # Z -> (X, Y).
    if axis_index == 0:
        rotation_axes = (1, 2)
    elif axis_index == 1:
        rotation_axes = (0, 2)
    else:
        rotation_axes = (0, 1)
    normal = _rotation_matrix(rotation_axes[0], np.deg2rad(parameters.angle1_deg)) @ normal
    normal = _rotation_matrix(rotation_axes[1], np.deg2rad(parameters.angle2_deg)) @ normal
    normal /= max(np.linalg.norm(normal), 1e-12)
    return point, normal


def sigmoid_transition_weights(
    signed_plane_distance: np.ndarray,
    parameters: TransitionParameters,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(w_g, w_d)`` with independent width and curve sharpness.

    ``transition_width_mm`` is always the physical distance between the 10%
    and 90% D weights.  ``sigmoid_sharpness`` changes the shape inside and
    outside that band, while the 10%/90% locations remain fixed.
    """

    return compute_sigmoid_transition_weights(
        signed_plane_distance,
        ComputeTransitionBlendSpec(
            transition_width_mm=parameters.transition_width_mm,
            center_offset_mm=parameters.center_offset_mm,
            sigmoid_sharpness=parameters.sigmoid_sharpness,
            g_on_negative_side=parameters.g_on_negative_side,
        ),
    )


def _tpms_shell_field(
    kind: TPMSKind,
    points: np.ndarray,
    params: TPMSParameters,
    max_tile_points: int | None = 250_000,
) -> np.ndarray:
    params.validate()
    if kind != params.kind:
        raise ValueError("TPMS kind must match the parameter set")
    return get_default_tpms_backend().evaluate_shell(
        points,
        _tpms_field_spec(params, None),
        max_tile_points=max_tile_points,
    )


def _cell_map_tpms_shell_field(
    kind: TPMSKind,
    points: np.ndarray,
    params: TPMSParameters,
    cell_map: CellMap,
    max_tile_points: int | None = 250_000,
    gradient_bounds_mm: tuple[float, float] | None = None,
) -> np.ndarray:
    """Evaluate one family in coordinates owned by its Cell Map."""

    if kind != params.kind:
        raise ValueError("TPMS kind must match the parameter set")
    effective = _parameters_for_cell_map(params, cell_map)
    if effective.gradient_enabled:
        controls = _gradient_controls(effective)
        if gradient_bounds_mm is None:
            axis_index = "UVW".index(effective.gradient_axis)
            local_min = np.asarray(cell_map.index_min, dtype=np.float64) * np.asarray(
                cell_map.spacing_mm,
                dtype=np.float64,
            )
            local_max = local_min + cell_map.extent_mm
            gradient_bounds_mm = (
                float(local_min[axis_index]),
                float(local_max[axis_index]),
            )
        return evaluate_gradient_tpms_field(
            points,
            _tpms_field_spec(effective, cell_map),
            cell_map.frame,
            controls,
            gradient_bounds_mm,
            max_tile_points=max_tile_points,
        )
    return _tpms_shell_field(
        kind,
        points,
        effective,
        max_tile_points=max_tile_points,
    )


def _tpms_field_spec(
    parameters: TPMSParameters,
    cell_map: CellMap | None,
) -> ComputeTPMSFieldSpec:
    """Create a compute spec aligned to a Cell Map when one is available."""

    if cell_map is not None:
        frame = cell_map.frame
    else:
        frame = CellMapFrame.from_origin_axes(
            parameters.frame_origin_mm or (0.0, 0.0, 0.0),
            parameters.frame_u_axis,
            parameters.frame_v_axis,
        )
    return ComputeTPMSFieldSpec(
        kind=parameters.kind,
        cell_size_mm=(
            cell_map.spacing_mm if cell_map is not None else parameters.cell_size_mm
        ),
        wall_thickness_mm=parameters.wall_thickness_mm,
        level=parameters.level,
        origin_mm=tuple(float(value) for value in frame.origin),
        axes_world=tuple(tuple(float(value) for value in axis) for axis in frame.axes),
    )


def _gradient_controls(parameters: TPMSParameters) -> GradientControls:
    """Build the shared gradient-compute contract from one TPMS parameter set."""

    return GradientControls(
        enabled=parameters.gradient_enabled,
        axis=parameters.gradient_axis,
        mode=parameters.gradient_mode,
        power=parameters.gradient_power,
        layers=parameters.gradient_layers,
        sigmoid_sharpness=parameters.gradient_sigmoid_sharpness,
        use_thickness_gradient=parameters.use_thickness_gradient,
        thickness_soft_mm=parameters.thickness_soft_mm,
        thickness_stiff_mm=parameters.thickness_stiff_mm,
        use_offset_gradient=parameters.use_offset_gradient,
        offset_soft=parameters.offset_soft,
        offset_stiff=parameters.offset_stiff,
        wall_thickness_method=parameters.wall_thickness_method,
    )


def _tpms_transition_chunk_field(
    points: np.ndarray,
    domain_sdf: np.ndarray,
    g_parameters: TPMSParameters,
    d_parameters: TPMSParameters,
    plane_point: np.ndarray,
    plane_normal: np.ndarray,
    transition: TransitionParameters,
    g_cell_map: CellMap,
    d_cell_map: CellMap,
    max_tile_points: int | None = 250_000,
    stage_progress: EvaluationStageCallback | None = None,
    g_gradient_bounds_mm: tuple[float, float] | None = None,
    d_gradient_bounds_mm: tuple[float, float] | None = None,
) -> np.ndarray:
    """Blend G/D and intersect the domain in bounded evaluation tiles."""

    points = np.asarray(points)
    domain_sdf = np.asarray(domain_sdf, dtype=np.float32)
    if domain_sdf.shape != (len(points),):
        raise ValueError("domain_sdf must have one value per sample point")
    blended_sdf = _tpms_transition_blended_field(
        points,
        g_parameters,
        d_parameters,
        plane_point,
        plane_normal,
        transition,
        g_cell_map,
        d_cell_map,
        max_tile_points,
        g_gradient_bounds_mm,
        d_gradient_bounds_mm,
    )
    if stage_progress:
        stage_progress(PROGRESS_INTERSECTION)
    return get_default_geometry_backend().intersect_fields(
        domain_sdf,
        blended_sdf,
        max_tile_points=max_tile_points,
    )


def _tpms_transition_blended_field(
    points: np.ndarray,
    g_parameters: TPMSParameters,
    d_parameters: TPMSParameters,
    plane_point: np.ndarray,
    plane_normal: np.ndarray,
    transition: TransitionParameters,
    g_cell_map: CellMap,
    d_cell_map: CellMap,
    max_tile_points: int | None = 250_000,
    g_gradient_bounds_mm: tuple[float, float] | None = None,
    d_gradient_bounds_mm: tuple[float, float] | None = None,
) -> np.ndarray:
    """Evaluate the G/D blend without applying a design-domain intersection."""

    if g_parameters.gradient_enabled or d_parameters.gradient_enabled:
        g_sdf = _cell_map_tpms_shell_field(
            g_parameters.kind,
            points,
            g_parameters,
            g_cell_map,
            max_tile_points=max_tile_points,
            gradient_bounds_mm=g_gradient_bounds_mm,
        )
        d_sdf = _cell_map_tpms_shell_field(
            d_parameters.kind,
            points,
            d_parameters,
            d_cell_map,
            max_tile_points=max_tile_points,
            gradient_bounds_mm=d_gradient_bounds_mm,
        )
        g_weight, d_weight = sigmoid_transition_weights(
            (np.asarray(points, dtype=np.float32) - np.asarray(plane_point, dtype=np.float32))
            @ np.asarray(plane_normal, dtype=np.float32),
            transition,
        )
        return g_weight * g_sdf + d_weight * d_sdf
    return get_default_tpms_backend().evaluate_transition(
        points,
        _tpms_field_spec(g_parameters, g_cell_map),
        _tpms_field_spec(d_parameters, d_cell_map),
        plane_point,
        plane_normal,
        ComputeTransitionBlendSpec(
            transition.transition_width_mm,
            transition.center_offset_mm,
            transition.sigmoid_sharpness,
            transition.g_on_negative_side,
        ),
        max_tile_points=max_tile_points,
    )


def _compute_design_domain_sdf(
    mesh: trimesh.Trimesh,
    points: np.ndarray,
    progress: ProgressCallback | None,
    use_cpp: bool = True,
) -> np.ndarray:
    """Compute a signed STL distance field, negative inside the domain."""

    if use_cpp:
        try:
            if progress:
                progress(
                    f"{PROGRESS_DOMAIN_SDF} [{_geometry_backend_label()}]",
                    0.22,
                )
            return get_default_geometry_backend().signed_distance(
                mesh.vertices,
                mesh.faces,
                points,
                max_tile_points=250_000,
            )
        except (ImportError, AttributeError, RuntimeError, ValueError) as exc:
            if progress:
                progress(f"几何 SDF 后端不可用，回退到 PyVista：{exc}", 0.20)

    import pyvista as pv

    if progress:
        progress(f"{PROGRESS_DOMAIN_SDF}（PyVista）", 0.18)
    pv_mesh = pv.wrap(mesh)
    point_cloud = pv.PolyData(np.ascontiguousarray(points, dtype=np.float32))
    distances = point_cloud.compute_implicit_distance(pv_mesh, inplace=False)
    sdf = np.asarray(distances["implicit_distance"], dtype=np.float32)

    return sdf


def implicit_intersection(
    field_a: np.ndarray,
    field_b: np.ndarray,
    use_acceleration: bool = True,
    max_tile_points: int | None = 250_000,
) -> np.ndarray:
    """Hard implicit intersection for negative-inside signed fields."""

    if field_a.shape != field_b.shape:
        raise ValueError("implicit fields must have the same shape")
    if not use_acceleration:
        return np.maximum(field_a, field_b)
    return get_default_geometry_backend().intersect_fields(
        field_a,
        field_b,
        max_tile_points=max_tile_points,
    )


def generate_tpms_lattice(
    mesh: trimesh.Trimesh,
    parameters: TPMSParameters,
    sampling: SamplingParameters | None = None,
    voxel_size_mm: float | None = None,
    progress: ProgressCallback | None = None,
) -> tuple[trimesh.Trimesh, SamplingRecommendation]:
    """Generate one TPMS shell clipped to a watertight design-domain STL."""

    parameters.validate()
    sampling = sampling or SamplingParameters()
    sampling.validate()
    if len(mesh.faces) == 0 or not mesh.is_watertight:
        raise ValueError("design-domain STL must be a non-empty watertight mesh")

    cell_map = cell_map_for_parameters(mesh, parameters)
    effective_parameters = _parameters_for_cell_map(parameters, cell_map)
    gradient_bounds = (
        gradient_projection_range(
            mesh.vertices,
            cell_map.frame,
            parameters.gradient_axis,
        )
        if parameters.gradient_enabled
        else None
    )
    field_spec = _tpms_field_spec(effective_parameters, cell_map)
    gradient_spec = (
        make_tpms_gradient_spec(
            _gradient_controls(effective_parameters),
            gradient_bounds,
        )
        if gradient_bounds is not None
        else None
    )
    recommendation = recommend_voxel_size(
        mesh, effective_parameters, sampling, cell_map=cell_map
    )
    if voxel_size_mm is not None:
        recommendation = recommendation_for_voxel_size(
            mesh, effective_parameters, sampling, voxel_size_mm, cell_map=cell_map
        )

    shape = recommendation.grid_shape
    lo, axes = _grid_definition(mesh, recommendation, bounds=cell_map.bounds)
    def evaluate_points(
        points: np.ndarray,
        max_tile_points: int | None,
        stage_progress: EvaluationStageCallback | None,
    ) -> np.ndarray:
        if sampling.use_cpp_sdf:
            if stage_progress:
                stage_progress(
                    "Device-resident SDF + TPMS + intersection "
                    f"[{_lattice_pipeline_backend_label()}]"
                )
            return get_default_lattice_pipeline().evaluate_mesh_clipped_tpms(
                mesh.vertices,
                mesh.faces,
                points,
                field_spec,
                gradient_spec,
                max_tile_points=max_tile_points,
            )
        if stage_progress:
            stage_progress(PROGRESS_DOMAIN_SDF)
        domain_sdf = _compute_design_domain_sdf(
            mesh, points, None, sampling.use_cpp_sdf
        )
        if stage_progress:
            stage_progress(PROGRESS_TPMS_FIELD)
        tpms_sdf = _cell_map_tpms_shell_field(
            parameters.kind,
            points,
            parameters,
            cell_map=cell_map,
            max_tile_points=max_tile_points,
            gradient_bounds_mm=gradient_bounds,
        )
        if stage_progress:
            stage_progress(PROGRESS_INTERSECTION)
        result = implicit_intersection(domain_sdf, tpms_sdf)
        del domain_sdf, tpms_sdf
        return result

    result = _generate_mesh_by_processing_mode(
        axes,
        shape,
        (recommendation.voxel_size_mm,) * 3,
        lo,
        sampling,
        evaluate_points,
        progress,
        parameters.kind,
    )
    if progress:
        progress(f"{parameters.kind}：完成（{len(result.faces):,} 个面）", 1.0)
    return result, recommendation


def generate_tpms_transition(
    mesh: trimesh.Trimesh,
    g_parameters: TPMSParameters,
    d_parameters: TPMSParameters,
    transition: TransitionParameters,
    sampling: SamplingParameters | None = None,
    voxel_size_mm: float | None = None,
    progress: ProgressCallback | None = None,
) -> tuple[trimesh.Trimesh, SamplingRecommendation]:
    """Generate a sigmoid-blended G/D TPMS field inside a design domain."""

    g_parameters.validate()
    d_parameters.validate()
    if g_parameters.kind != "G" or d_parameters.kind != "D":
        raise ValueError("transition requires one G parameter set and one D parameter set")
    transition.validate()
    sampling = sampling or SamplingParameters()
    sampling.validate()
    if len(mesh.faces) == 0 or not mesh.is_watertight:
        raise ValueError("design-domain STL must be a non-empty watertight mesh")

    g_cell_map, d_cell_map, shared_cell_map = transition_cell_maps(
        mesh,
        g_parameters,
        d_parameters,
    )
    g_gradient_bounds = (
        gradient_projection_range(
            mesh.vertices,
            g_cell_map.frame,
            g_parameters.gradient_axis,
        )
        if g_parameters.gradient_enabled
        else None
    )
    d_gradient_bounds = (
        gradient_projection_range(
            mesh.vertices,
            d_cell_map.frame,
            d_parameters.gradient_axis,
        )
        if d_parameters.gradient_enabled
        else None
    )
    effective_g = _parameters_for_cell_map(g_parameters, g_cell_map)
    effective_d = _parameters_for_cell_map(d_parameters, d_cell_map)
    g_recommendation = recommend_voxel_size(
        mesh, effective_g, sampling, cell_map=shared_cell_map
    )
    d_recommendation = recommend_voxel_size(
        mesh, effective_d, sampling, cell_map=shared_cell_map
    )
    if voxel_size_mm is None:
        # One shared grid is required for a pointwise G/D blend. Use the finer
        # recommendation so neither lattice is sampled more coarsely than its
        # own physical feature constraints require.
        recommendation = min(
            (g_recommendation, d_recommendation),
            key=lambda item: item.voxel_size_mm,
        )
    else:
        recommendation = recommendation_for_voxel_size(
            mesh, effective_g, sampling, voxel_size_mm, cell_map=shared_cell_map
        )

    shape = recommendation.grid_shape
    plane_point, plane_normal = transition_plane_geometry(mesh, transition)
    lo, axes = _grid_definition(mesh, recommendation, bounds=shared_cell_map.bounds)
    def evaluate_points(
        points: np.ndarray,
        max_tile_points: int | None,
        stage_progress: EvaluationStageCallback | None,
    ) -> np.ndarray:
        if stage_progress:
            stage_progress(PROGRESS_DOMAIN_SDF)
        domain_sdf = _compute_design_domain_sdf(
            mesh, points, None, sampling.use_cpp_sdf
        )
        if stage_progress:
            stage_progress(PROGRESS_TPMS_FIELD)
        result = _tpms_transition_chunk_field(
            points,
            domain_sdf,
            g_parameters,
            d_parameters,
            plane_point,
            plane_normal,
            transition,
            g_cell_map,
            d_cell_map,
            max_tile_points=max_tile_points,
            stage_progress=stage_progress,
            g_gradient_bounds_mm=g_gradient_bounds,
            d_gradient_bounds_mm=d_gradient_bounds,
        )
        del domain_sdf
        return result

    result = _generate_mesh_by_processing_mode(
        axes,
        shape,
        (recommendation.voxel_size_mm,) * 3,
        lo,
        sampling,
        evaluate_points,
        progress,
        "过渡",
    )
    if progress:
        progress(f"过渡：完成（{len(result.faces):,} 个面）", 1.0)
    return result, recommendation


def generate_implicit_lattice(
    mesh: trimesh.Trimesh,
    parameters: TPMSParameters,
    sampling: SamplingParameters | None = None,
    voxel_size_mm: float | None = None,
    display_voxel_size_mm: float | None = None,
    display_memory_budget_mb: float = 512.0,
    visible_field_count: int = 1,
    display_batch_count: int | None = None,
    domain_display_field: SampledImplicitField | None = None,
    progress: ProgressCallback | None = None,
) -> ImplicitGenerationResult:
    """Build a TPMS evaluator and its bounded GPU preview cache."""

    parameters.validate()
    sampling = sampling or SamplingParameters()
    sampling.validate()
    if len(mesh.faces) == 0 or not mesh.is_watertight:
        raise ValueError("design-domain STL must be a non-empty watertight mesh")
    cell_map = cell_map_for_parameters(mesh, parameters)
    effective_parameters = _parameters_for_cell_map(parameters, cell_map)
    recommendation = recommend_voxel_size(
        mesh, effective_parameters, sampling, cell_map=cell_map
    )
    if voxel_size_mm is not None:
        recommendation = recommendation_for_voxel_size(
            mesh, effective_parameters, sampling, voxel_size_mm, cell_map=cell_map
        )
    feature_body = _make_lattice_feature_body(mesh, parameters, cell_map)
    body = _make_lattice_body(
        mesh,
        parameters,
        sampling,
        cell_map=cell_map,
        feature_body=feature_body,
    )
    if domain_display_field is None:
        display_recommendation = _display_recommendation(
            mesh,
            lattice_sampling_recommendation(mesh, parameters, sampling),
            display_voxel_size_mm,
            display_memory_budget_mb,
            visible_field_count,
        )
        field = _sample_implicit_body(
            body,
            mesh,
            display_recommendation,
            (0.72, 0.72, 0.72, 1.0),
            progress,
            parameters.kind,
            bounds=cell_map.bounds,
            display_batch_count=display_batch_count,
        )
    else:
        field = _sample_intersection_with_domain_cache(
            parameters.kind,
            domain_display_field,
            lambda points: _cell_map_tpms_shell_field(
                parameters.kind,
                points,
                parameters,
                cell_map=cell_map,
                max_tile_points=250_000,
            ),
            (0.72, 0.72, 0.72, 1.0),
            progress,
            display_batch_count=display_batch_count,
        )
    field, display_cleanup = _clean_display_field_components(
        field,
        cell_map,
        effective_parameters.minimum_wall_thickness_mm,
    )
    if progress:
        progress(f"{parameters.kind}：完成隐式显示结果", 1.0)
    return ImplicitGenerationResult(
        body,
        field,
        recommendation,
        cell_map,
        minimum_feature_mm=effective_parameters.minimum_wall_thickness_mm,
        display_fragment_cleanup=display_cleanup,
        unclipped_body=feature_body,
    )


def generate_implicit_custom_lattice(
    mesh: trimesh.Trimesh,
    parameters: CustomUnitCellParameters,
    sampling: SamplingParameters | None = None,
    voxel_size_mm: float | None = None,
    display_voxel_size_mm: float | None = None,
    display_memory_budget_mb: float = 512.0,
    visible_field_count: int = 1,
    display_batch_count: int | None = None,
    domain_display_field: SampledImplicitField | None = None,
    progress: ProgressCallback | None = None,
    prepared_unit_cell: PreparedCustomUnitCell | None = None,
) -> ImplicitGenerationResult:
    """Build a periodic custom-cell evaluator and interactive display cache."""

    parameters.validate()
    sampling = sampling or SamplingParameters()
    sampling.validate()
    if len(mesh.faces) == 0 or not mesh.is_watertight:
        raise ValueError("design-domain STL must be a non-empty watertight mesh")
    if progress:
        progress("自定义晶胞：导入并检查 STL", 0.02)
    prepared = prepared_unit_cell or prepare_stl_unit_cell(parameters.source_path)
    cell_map = cell_map_for_parameters(mesh, parameters)
    native_feature = prepared.characteristic_feature_mm(cell_map)
    minimum_feature = parameters.effective_feature_mm(native_feature)
    recommendation = custom_cell_sampling_recommendation(
        mesh,
        parameters,
        sampling,
        voxel_size_mm,
        cell_map=cell_map,
        minimum_feature_mm=minimum_feature,
    )
    design_domain = _make_design_domain_body(mesh, sampling)
    feature_body = _make_custom_lattice_feature_body(
        prepared,
        cell_map,
        parameters,
    )
    body = make_periodic_lattice(
        prepared,
        cell_map,
        design_domain,
        target_feature_mm=parameters.target_feature_mm,
        bridge_directions=parameters.bridge_directions,
        bridge_depth_mm=parameters.bridge_depth_mm,
    )
    display_recommendation = _display_recommendation(
        mesh,
        recommendation,
        display_voxel_size_mm,
        display_memory_budget_mb,
        visible_field_count,
    )
    if domain_display_field is None or not _display_field_matches_grid(
        domain_display_field,
        mesh,
        display_recommendation,
    ):
        field = _sample_implicit_body(
            body,
            mesh,
            display_recommendation,
            (0.72, 0.72, 0.72, 1.0),
            progress,
            "Custom",
            bounds=mesh.bounds,
            display_batch_count=display_batch_count,
        )
    else:
        field = _sample_intersection_with_domain_cache(
            "Custom",
            domain_display_field,
            lambda points: prepared.evaluate_periodic(
                points,
                cell_map,
                parameters.target_feature_mm,
                250_000,
                parameters.bridge_directions,
                parameters.bridge_depth_mm,
            ),
            (0.72, 0.72, 0.72, 1.0),
            progress,
            progress_stage="Custom：计算自定义晶胞隐式场",
            display_batch_count=display_batch_count,
        )
    field, display_cleanup = _clean_display_field_components(
        field,
        cell_map,
        minimum_feature,
    )
    if progress:
        progress("自定义晶胞：完成隐式显示结果", 1.0)
    return ImplicitGenerationResult(
        body,
        field,
        recommendation,
        cell_map,
        minimum_feature_mm=minimum_feature,
        metadata=prepared.report,
        display_fragment_cleanup=display_cleanup,
        unclipped_body=feature_body,
    )


def generate_implicit_lattice_transition(
    mesh: trimesh.Trimesh,
    first_parameters: LatticeParameters,
    second_parameters: LatticeParameters,
    transition: TransitionParameters,
    sampling: SamplingParameters | None = None,
    voxel_size_mm: float | None = None,
    display_voxel_size_mm: float | None = None,
    display_memory_budget_mb: float = 512.0,
    visible_field_count: int = 1,
    display_batch_count: int | None = None,
    domain_display_field: SampledImplicitField | None = None,
    progress: ProgressCallback | None = None,
    prepared_custom_cell: PreparedCustomUnitCell | None = None,
    transition_driver: ImplicitBody | None = None,
) -> ImplicitGenerationResult:
    """Build and validate a provider-agnostic implicit lattice transition."""

    first_parameters.validate()
    second_parameters.validate()
    transition.validate()
    sampling = sampling or SamplingParameters()
    sampling.validate()
    if len(mesh.faces) == 0 or not mesh.is_watertight:
        raise ValueError("design-domain STL must be a non-empty watertight mesh")
    if progress:
        progress("过渡：准备两侧 Cell Map", 0.01)
    first_map, second_map, shared_map = lattice_transition_cell_maps(
        mesh,
        first_parameters,
        second_parameters,
    )
    if progress:
        progress("过渡：准备晶胞隐式求值器", 0.025)
    first_operand, prepared_after_first = _make_transition_operand(
        mesh,
        first_parameters,
        first_map,
        prepared_custom_cell,
    )
    second_operand, _prepared_after_second = _make_transition_operand(
        mesh,
        second_parameters,
        second_map,
        prepared_after_first,
    )

    def operand_recommendation(
        parameters: LatticeParameters,
        cell_map: CellMap,
        minimum_feature_mm: float,
    ) -> SamplingRecommendation:
        if isinstance(parameters, CustomUnitCellParameters):
            return custom_cell_sampling_recommendation(
                mesh,
                parameters,
                sampling,
                voxel_size_mm,
                cell_map=cell_map,
                minimum_feature_mm=minimum_feature_mm,
            )
        effective = _parameters_for_cell_map(parameters, cell_map)
        if voxel_size_mm is None:
            return recommend_voxel_size(mesh, effective, sampling, cell_map=shared_map)
        return recommendation_for_voxel_size(
            mesh,
            effective,
            sampling,
            voxel_size_mm,
            cell_map=shared_map,
        )

    first_recommendation = operand_recommendation(
        first_parameters,
        first_map,
        first_operand.characteristic_feature_mm,
    )
    second_recommendation = operand_recommendation(
        second_parameters,
        second_map,
        second_operand.characteristic_feature_mm,
    )
    recommendation = min(
        (first_recommendation, second_recommendation),
        key=lambda item: item.voxel_size_mm,
    )
    design_domain = _make_design_domain_body(mesh, sampling)
    if transition.driver_mode == "field" and transition_driver is None:
        raise ValueError("field-driven transition requires the selected field object")
    if transition.driver_mode == "plane" and transition_driver is not None:
        raise ValueError("plane-driven transition must not receive a field object")
    plane_point: np.ndarray | None = None
    plane_normal: np.ndarray | None = None
    if transition.driver_mode == "plane":
        plane_point, plane_normal = transition_plane_geometry(mesh, transition)
    provider_spec = transition.provider_spec()
    if progress:
        stage = (
            "过渡：搜索局部 Cell Map 相位配准"
            if transition.driver_mode == "plane"
            else "过渡：准备独立场对象 Ramp"
        )
        progress(stage, 0.04)
    build_arguments: dict[str, object] = {"spec": provider_spec}
    if transition_driver is None:
        build_arguments.update(
            plane_point_mm=plane_point,
            plane_normal=plane_normal,
        )
    else:
        build_arguments["driver"] = transition_driver
    build = build_provider_transition_body(
        first_operand,
        second_operand,
        design_domain,
        **build_arguments,
    )
    if progress:
        progress(
            f"过渡：计算 {first_operand.name}/{second_operand.name} Ramp 场"
            f" [{build.diagnostics.execution_backend}]",
            0.08,
        )
    if domain_display_field is None:
        display_recommendation = _display_recommendation(
            mesh,
            recommendation,
            display_voxel_size_mm,
            display_memory_budget_mb,
            visible_field_count,
        )
        field = _sample_implicit_body(
            build.body,
            mesh,
            display_recommendation,
            (0.72, 0.32, 0.92, 1.0),
            progress,
            "过渡",
            bounds=shared_map.bounds,
            display_batch_count=display_batch_count,
        )
    else:
        field = _sample_intersection_with_domain_cache(
            "过渡",
            domain_display_field,
            lambda points: build.field_body.evaluate_points(points),
            (0.72, 0.32, 0.92, 1.0),
            progress,
            progress_stage="过渡：计算 Ramp 场与设计域交集",
            display_batch_count=display_batch_count,
        )
    field, display_cleanup = _clean_display_field_components(
        field,
        shared_map,
        build.diagnostics.minimum_feature_mm,
    )
    validation_spec = replace(
        provider_spec,
        minimum_feature_mm=build.diagnostics.minimum_feature_mm,
    )
    if transition.driver_mode == "plane":
        assert plane_point is not None and plane_normal is not None
        if progress:
            progress("过渡：检查跨带连通和最小特征厚度", 0.97)
        quality = validate_provider_transition(
            build.body,
            plane_point_mm=plane_point,
            plane_normal=plane_normal,
            spec=validation_spec,
            max_samples=750_000,
        )
    else:
        quality = TransitionQualityReport(
            cross_band_connected=False,
            minimum_feature_satisfied=False,
            isolated_fragment_count=0,
            component_count=0,
            inspection_spacing_mm=0.0,
            inspected_samples=0,
            inspection_supported=False,
            status_message="曲面场驱动不采用平面跨带检查；请使用逐层等值线和 STL 拓扑检查。",
        )
    metadata = TransitionGenerationMetadata(
        first_name=first_operand.name,
        second_name=second_operand.name,
        diagnostics=build.diagnostics,
        quality=quality,
    )
    if progress:
        status = "通过" if quality.passed else (
            "待 STL 拓扑检查" if not quality.inspection_supported else "存在风险"
        )
        progress(f"过渡：完成隐式显示，质量检查{status}", 1.0)
    return ImplicitGenerationResult(
        build.body,
        field,
        recommendation,
        shared_map,
        minimum_feature_mm=build.diagnostics.minimum_feature_mm,
        metadata=metadata,
        display_fragment_cleanup=display_cleanup,
        unclipped_body=build.field_body,
    )


def generate_implicit_transition(
    mesh: trimesh.Trimesh,
    g_parameters: TPMSParameters,
    d_parameters: TPMSParameters,
    transition: TransitionParameters,
    sampling: SamplingParameters | None = None,
    voxel_size_mm: float | None = None,
    display_voxel_size_mm: float | None = None,
    display_memory_budget_mb: float = 512.0,
    visible_field_count: int = 1,
    display_batch_count: int | None = None,
    domain_display_field: SampledImplicitField | None = None,
    progress: ProgressCallback | None = None,
) -> ImplicitGenerationResult:
    """Compatibility wrapper for the original G-to-D transition entry point."""

    if g_parameters.kind != "G" or d_parameters.kind != "D":
        raise ValueError("transition requires one G parameter set and one D parameter set")
    return generate_implicit_lattice_transition(
        mesh,
        g_parameters,
        d_parameters,
        transition,
        sampling,
        voxel_size_mm,
        display_voxel_size_mm,
        display_memory_budget_mb,
        visible_field_count,
        display_batch_count,
        domain_display_field,
        progress,
    )


def cell_map_for_design_domain(
    domain: DesignDomainValue,
    parameters: LatticeParameters,
) -> CellMap:
    """Build a Cell Map from a mesh or analytic Design domain authority."""

    parameters.validate()
    bounds = np.asarray(domain.bounds, dtype=np.float64)
    default_origin = bounds[0]
    frame = parameters.cell_map_frame(default_origin)
    uses_default_frame = frame.is_world_aligned and np.allclose(
        frame.origin,
        default_origin,
        rtol=0.0,
        atol=1.0e-9,
    )
    manual_counts = getattr(parameters, "manual_cell_counts", None)
    if manual_counts is not None:
        return CellMap.from_frame_counts(
            frame,
            parameters.cell_size_xyz_mm,
            manual_counts,
        )
    if uses_default_frame and parameters.cell_map_mode == "fit_bounds":
        return CellMap.from_bounds(
            bounds,
            parameters.cell_size_xyz_mm,
            boundary_mode="fit_bounds",
        )
    return CellMap.from_points(
        np.asarray(domain.frame_points, dtype=np.float64),
        parameters.cell_size_xyz_mm,
        frame,
    )


def _display_recommendation_for_bounds(
    bounds: np.ndarray,
    base: SamplingRecommendation,
    display_voxel_size_mm: float | None,
    display_memory_budget_mb: float,
    visible_field_count: int,
) -> SamplingRecommendation:
    """Apply display-cache limits without requiring a triangular Design domain."""

    source_bounds = np.asarray(bounds, dtype=np.float64)
    extents = source_bounds[1] - source_bounds[0]
    if source_bounds.shape != (2, 3) or np.any(~np.isfinite(extents)) or np.any(extents <= 0):
        raise ValueError("design-domain bounds must have positive finite extents")
    if display_voxel_size_mm is None:
        if display_memory_budget_mb <= 0.0 or not np.isfinite(display_memory_budget_mb):
            raise ValueError("display memory budget must be finite and positive")
        if visible_field_count < 1:
            raise ValueError("visible field count must be at least 1")
        step = float(base.voxel_size_mm)
        budget_bytes = float(display_memory_budget_mb) * 1024.0 * 1024.0
        max_voxels = max(8.0, budget_bytes / (4.0 * visible_field_count))
        for _ in range(8):
            shape = np.ceil(extents / step).astype(np.int64) + 3
            estimated = float(np.prod(shape, dtype=np.int64))
            if estimated <= max_voxels:
                break
            step *= (estimated / max_voxels) ** (1.0 / 3.0) * 1.02
    else:
        step = float(display_voxel_size_mm)
        if step <= 0.0 or not np.isfinite(step):
            raise ValueError("display voxel size must be finite and positive")
    shape = tuple(int(np.ceil(axis / step)) + 3 for axis in extents)
    return replace(
        base,
        voxel_size_mm=step,
        grid_shape=shape,
        estimated_voxels=int(np.prod(shape, dtype=np.int64)),
        samples_per_cell=base.samples_per_cell * base.voxel_size_mm / step,
        samples_per_wall=base.samples_per_wall * base.voxel_size_mm / step,
    )


def _sample_implicit_body_for_bounds(
    body: ImplicitBody,
    bounds: np.ndarray,
    recommendation: SamplingRecommendation,
    color: tuple[float, float, float, float],
    progress: ProgressCallback | None,
    label: str,
    display_batch_count: int | None,
) -> SampledImplicitField:
    """Sample a body using explicit bounds rather than a mesh proxy."""

    source_bounds = np.asarray(bounds, dtype=np.float64)
    if source_bounds.shape != (2, 3) or not np.all(source_bounds[0] < source_bounds[1]):
        raise ValueError("sampling bounds must have shape (2, 3) and positive extents")
    origin = source_bounds[0] - recommendation.voxel_size_mm
    axes = [
        origin[axis]
        + np.arange(recommendation.grid_shape[axis], dtype=np.float64)
        * recommendation.voxel_size_mm
        for axis in range(3)
    ]
    return _sample_implicit_body_on_grid(
        body,
        axes,
        origin,
        np.full(3, recommendation.voxel_size_mm, dtype=np.float64),
        color,
        progress,
        label,
        display_batch_count,
    )


def _make_lattice_feature_body_for_design_domain(
    domain: DesignDomainValue,
    parameters: TPMSParameters,
    cell_map: CellMap,
) -> ImplicitBody:
    """Create an unclipped TPMS evaluator using a domain's frame points."""

    gradient_bounds = (
        gradient_projection_range(
            np.asarray(domain.frame_points, dtype=np.float64),
            cell_map.frame,
            parameters.gradient_axis,
        )
        if parameters.gradient_enabled
        else None
    )

    def evaluate(
        points: np.ndarray,
        stage_reporter: EvaluationStageCallback | None,
    ) -> np.ndarray:
        if stage_reporter:
            stage_reporter(f"{PROGRESS_TPMS_FIELD} [{_tpms_backend_label()}]")
        return _cell_map_tpms_shell_field(
            parameters.kind,
            points,
            parameters,
            cell_map=cell_map,
            max_tile_points=250_000,
            gradient_bounds_mm=gradient_bounds,
        )

    return ImplicitBody(
        name=f"{parameters.kind} feature",
        bounds=np.asarray(cell_map.bounds, dtype=np.float64),
        evaluate=evaluate,
    )


def _make_lattice_body_for_design_domain(
    domain: DesignDomainValue,
    parameters: TPMSParameters,
    cell_map: CellMap,
    feature_body: ImplicitBody,
) -> ImplicitBody:
    """Intersect a TPMS evaluator with any watertight Design domain SDF."""

    if not domain.is_watertight:
        raise ValueError("lattice generation requires a watertight Design domain")
    effective = _parameters_for_cell_map(parameters, cell_map)
    gradient_bounds = (
        gradient_projection_range(
            np.asarray(domain.frame_points, dtype=np.float64),
            cell_map.frame,
            effective.gradient_axis,
        )
        if effective.gradient_enabled
        else None
    )
    gradient_spec = (
        make_tpms_gradient_spec(_gradient_controls(effective), gradient_bounds)
        if gradient_bounds is not None
        else None
    )
    field_spec = _tpms_field_spec(effective, cell_map)

    def evaluate(
        points: np.ndarray,
        stage_reporter: EvaluationStageCallback | None,
    ) -> np.ndarray:
        if isinstance(domain, MeshDesignDomain):
            if stage_reporter:
                stage_reporter(
                    "Device-resident SDF + TPMS + intersection "
                    f"[{_lattice_pipeline_backend_label()}]"
                )
            return get_default_lattice_pipeline().evaluate_mesh_clipped_tpms(
                domain.mesh.vertices,
                domain.mesh.faces,
                points,
                field_spec,
                gradient_spec,
                max_tile_points=250_000,
            )
        if stage_reporter:
            stage_reporter(PROGRESS_DOMAIN_SDF)
        domain_sdf = domain.evaluate_points(points)
        tpms_sdf = feature_body.evaluate_points(points, stage_reporter)
        if stage_reporter:
            stage_reporter(PROGRESS_INTERSECTION)
        return implicit_intersection(domain_sdf, tpms_sdf)

    return ImplicitBody(
        name=parameters.kind,
        bounds=np.asarray(cell_map.bounds, dtype=np.float64),
        evaluate=evaluate,
    )


def _domain_sampling_recommendation(
    domain: DesignDomainValue,
    parameters: LatticeParameters,
    sampling: SamplingParameters,
    *,
    voxel_size_mm: float | None = None,
    cell_map: CellMap | None = None,
    minimum_feature_mm: float | None = None,
) -> SamplingRecommendation:
    """Reuse the established physical sampling policy for an arbitrary domain."""

    map_value = cell_map or cell_map_for_design_domain(domain, parameters)
    proxy = domain.preview_mesh()
    if isinstance(parameters, CustomUnitCellParameters):
        return custom_cell_sampling_recommendation(
            proxy,
            parameters,
            sampling,
            voxel_size_mm,
            cell_map=map_value,
            minimum_feature_mm=minimum_feature_mm,
        )
    effective = _parameters_for_cell_map(parameters, map_value)
    if voxel_size_mm is None:
        return recommend_voxel_size(proxy, effective, sampling, cell_map=map_value)
    return recommendation_for_voxel_size(
        proxy,
        effective,
        sampling,
        voxel_size_mm,
        cell_map=map_value,
    )


def generate_implicit_domain_preview_for_design_domain(
    domain: DesignDomainValue,
    sampling: SamplingParameters,
    base_recommendation: SamplingRecommendation,
    display_voxel_size_mm: float | None = None,
    display_memory_budget_mb: float = 512.0,
    visible_field_count: int = 1,
    display_batch_count: int | None = None,
    progress: ProgressCallback | None = None,
) -> SampledImplicitField:
    """Build a display cache directly from a Design domain's authority."""

    display_recommendation = _display_recommendation_for_bounds(
        domain.bounds,
        base_recommendation,
        display_voxel_size_mm,
        display_memory_budget_mb,
        visible_field_count,
    )
    return _sample_implicit_body_for_bounds(
        domain.as_implicit_body(
            preview_shell_thickness_mm=max(display_recommendation.voxel_size_mm, 0.25)
        ),
        domain.bounds,
        display_recommendation,
        (0.72, 0.76, 0.82, 1.0),
        progress,
        "设计域",
        display_batch_count,
    )


def generate_implicit_lattice_for_design_domain(
    domain: DesignDomainValue,
    parameters: TPMSParameters,
    sampling: SamplingParameters | None = None,
    voxel_size_mm: float | None = None,
    display_voxel_size_mm: float | None = None,
    display_memory_budget_mb: float = 512.0,
    visible_field_count: int = 1,
    display_batch_count: int | None = None,
    domain_display_field: SampledImplicitField | None = None,
    progress: ProgressCallback | None = None,
) -> ImplicitGenerationResult:
    """Generate TPMS inside an analytic Design domain without STL conversion."""

    parameters.validate()
    sampling = sampling or SamplingParameters()
    sampling.validate()
    if not domain.is_watertight:
        raise ValueError("lattice generation requires a watertight Design domain")
    cell_map = cell_map_for_design_domain(domain, parameters)
    effective = _parameters_for_cell_map(parameters, cell_map)
    recommendation = _domain_sampling_recommendation(
        domain,
        parameters,
        sampling,
        voxel_size_mm=voxel_size_mm,
        cell_map=cell_map,
    )
    feature_body = _make_lattice_feature_body_for_design_domain(
        domain,
        parameters,
        cell_map,
    )
    body = _make_lattice_body_for_design_domain(
        domain,
        parameters,
        cell_map,
        feature_body,
    )
    if domain_display_field is None:
        display_recommendation = _display_recommendation_for_bounds(
            cell_map.bounds,
            recommendation,
            display_voxel_size_mm,
            display_memory_budget_mb,
            visible_field_count,
        )
        field = _sample_implicit_body_for_bounds(
            body,
            cell_map.bounds,
            display_recommendation,
            (0.72, 0.72, 0.72, 1.0),
            progress,
            parameters.kind,
            display_batch_count,
        )
    else:
        field = _sample_intersection_with_domain_cache(
            parameters.kind,
            domain_display_field,
            lambda points: _cell_map_tpms_shell_field(
                parameters.kind,
                points,
                parameters,
                cell_map=cell_map,
                max_tile_points=250_000,
            ),
            (0.72, 0.72, 0.72, 1.0),
            progress,
            display_batch_count=display_batch_count,
        )
    field, cleanup = _clean_display_field_components(
        field,
        cell_map,
        effective.minimum_wall_thickness_mm,
    )
    return ImplicitGenerationResult(
        body,
        field,
        recommendation,
        cell_map,
        minimum_feature_mm=effective.minimum_wall_thickness_mm,
        display_fragment_cleanup=cleanup,
        unclipped_body=feature_body,
    )


def generate_implicit_custom_lattice_for_design_domain(
    domain: DesignDomainValue,
    parameters: CustomUnitCellParameters,
    sampling: SamplingParameters | None = None,
    voxel_size_mm: float | None = None,
    display_voxel_size_mm: float | None = None,
    display_memory_budget_mb: float = 512.0,
    visible_field_count: int = 1,
    display_batch_count: int | None = None,
    domain_display_field: SampledImplicitField | None = None,
    progress: ProgressCallback | None = None,
    prepared_unit_cell: PreparedCustomUnitCell | None = None,
) -> ImplicitGenerationResult:
    """Generate a periodic custom-cell lattice in an analytic Design domain."""

    parameters.validate()
    sampling = sampling or SamplingParameters()
    sampling.validate()
    if not domain.is_watertight:
        raise ValueError("lattice generation requires a watertight Design domain")
    prepared = prepared_unit_cell or prepare_stl_unit_cell(parameters.source_path)
    cell_map = cell_map_for_design_domain(domain, parameters)
    minimum_feature = parameters.effective_feature_mm(
        prepared.characteristic_feature_mm(cell_map)
    )
    recommendation = _domain_sampling_recommendation(
        domain,
        parameters,
        sampling,
        voxel_size_mm=voxel_size_mm,
        cell_map=cell_map,
        minimum_feature_mm=minimum_feature,
    )
    feature_body = _make_custom_lattice_feature_body(prepared, cell_map, parameters)
    body = make_periodic_lattice(
        prepared,
        cell_map,
        domain.as_implicit_body(),
        target_feature_mm=parameters.target_feature_mm,
        bridge_directions=parameters.bridge_directions,
        bridge_depth_mm=parameters.bridge_depth_mm,
    )
    if domain_display_field is None:
        display_recommendation = _display_recommendation_for_bounds(
            domain.bounds,
            recommendation,
            display_voxel_size_mm,
            display_memory_budget_mb,
            visible_field_count,
        )
        field = _sample_implicit_body_for_bounds(
            body,
            domain.bounds,
            display_recommendation,
            (0.72, 0.72, 0.72, 1.0),
            progress,
            "Custom",
            display_batch_count,
        )
    else:
        field = _sample_intersection_with_domain_cache(
            "Custom",
            domain_display_field,
            lambda points: prepared.evaluate_periodic(
                points,
                cell_map,
                parameters.target_feature_mm,
                250_000,
                parameters.bridge_directions,
                parameters.bridge_depth_mm,
            ),
            (0.72, 0.72, 0.72, 1.0),
            progress,
            progress_stage="Custom：计算自定义晶胞隐式场",
            display_batch_count=display_batch_count,
        )
    field, cleanup = _clean_display_field_components(
        field,
        cell_map,
        minimum_feature,
    )
    return ImplicitGenerationResult(
        body,
        field,
        recommendation,
        cell_map,
        minimum_feature_mm=minimum_feature,
        metadata=prepared.report,
        display_fragment_cleanup=cleanup,
        unclipped_body=feature_body,
    )


def _transition_plane_geometry_for_bounds(
    bounds: np.ndarray,
    parameters: TransitionParameters,
) -> tuple[np.ndarray, np.ndarray]:
    """Resolve the plane driver without requiring a mesh Design domain."""

    parameters.validate()
    source_bounds = np.asarray(bounds, dtype=np.float64)
    if source_bounds.shape != (2, 3) or not np.all(source_bounds[0] < source_bounds[1]):
        raise ValueError("transition domain bounds must have positive extents")
    axis_index = {"X": 0, "Y": 1, "Z": 2}[parameters.plane_axis]
    point = source_bounds.mean(axis=0)
    point[axis_index] = parameters.plane_position_mm
    normal = np.zeros(3, dtype=np.float64)
    normal[axis_index] = 1.0
    rotation_axes = ((1, 2), (0, 2), (0, 1))[axis_index]
    normal = _rotation_matrix(
        rotation_axes[0], np.deg2rad(parameters.angle1_deg)
    ) @ normal
    normal = _rotation_matrix(
        rotation_axes[1], np.deg2rad(parameters.angle2_deg)
    ) @ normal
    normal /= max(float(np.linalg.norm(normal)), 1.0e-12)
    return point, normal


def _make_transition_operand_for_design_domain(
    domain: DesignDomainValue,
    parameters: LatticeParameters,
    cell_map: CellMap,
    prepared_custom_cell: PreparedCustomUnitCell | None,
) -> tuple[TransitionOperand, PreparedCustomUnitCell | None]:
    """Build a provider operand while retaining analytic-frame precision."""

    if isinstance(parameters, CustomUnitCellParameters):
        return _make_transition_operand(
            domain.preview_mesh(),
            parameters,
            cell_map,
            prepared_custom_cell,
        )
    parameters.validate()
    gradient_bounds = (
        gradient_projection_range(
            np.asarray(domain.frame_points, dtype=np.float64),
            cell_map.frame,
            parameters.gradient_axis,
        )
        if parameters.gradient_enabled
        else None
    )

    def evaluate(
        points: np.ndarray,
        _stage_reporter: EvaluationStageCallback | None,
    ) -> np.ndarray:
        return _cell_map_tpms_shell_field(
            parameters.kind,
            points,
            parameters,
            cell_map,
            max_tile_points=250_000,
            gradient_bounds_mm=gradient_bounds,
        )

    status = get_default_tpms_backend().status
    return (
        TransitionOperand(
            name=parameters.kind,
            body=ImplicitBody(
                name=parameters.kind,
                bounds=np.asarray(cell_map.bounds, dtype=np.float64),
                evaluate=evaluate,
            ),
            characteristic_feature_mm=parameters.minimum_wall_thickness_mm,
            cell_spacing_mm=tuple(float(value) for value in cell_map.spacing_mm),
            frame_axes_world=tuple(
                tuple(float(value) for value in axis)
                for axis in cell_map.frame.axes
            ),
            backend_name=status.active_backend,
            supports_gpu=status.using_gpu,
        ),
        prepared_custom_cell,
    )


def generate_implicit_lattice_transition_for_design_domain(
    domain: DesignDomainValue,
    first_parameters: LatticeParameters,
    second_parameters: LatticeParameters,
    transition: TransitionParameters,
    sampling: SamplingParameters | None = None,
    voxel_size_mm: float | None = None,
    display_voxel_size_mm: float | None = None,
    display_memory_budget_mb: float = 512.0,
    visible_field_count: int = 1,
    display_batch_count: int | None = None,
    domain_display_field: SampledImplicitField | None = None,
    progress: ProgressCallback | None = None,
    prepared_custom_cell: PreparedCustomUnitCell | None = None,
    transition_driver: ImplicitBody | None = None,
) -> ImplicitGenerationResult:
    """Blend arbitrary lattice providers inside an analytic Design domain."""

    first_parameters.validate()
    second_parameters.validate()
    transition.validate()
    sampling = sampling or SamplingParameters()
    sampling.validate()
    if not domain.is_watertight:
        raise ValueError("lattice generation requires a watertight Design domain")
    first_map = cell_map_for_design_domain(domain, first_parameters)
    second_map = cell_map_for_design_domain(domain, second_parameters)
    shared_map = CellMap.from_bounds(
        np.stack(
            (
                np.minimum(first_map.bounds[0], second_map.bounds[0]),
                np.maximum(first_map.bounds[1], second_map.bounds[1]),
            )
        ),
        np.minimum(first_map.spacing_mm, second_map.spacing_mm),
        boundary_mode=(
            "fit_bounds"
            if first_map.boundary_mode == second_map.boundary_mode == "fit_bounds"
            else "complete_cells"
        ),
    )
    first_operand, prepared_after_first = _make_transition_operand_for_design_domain(
        domain,
        first_parameters,
        first_map,
        prepared_custom_cell,
    )
    second_operand, _prepared_after_second = _make_transition_operand_for_design_domain(
        domain,
        second_parameters,
        second_map,
        prepared_after_first,
    )
    first_recommendation = _domain_sampling_recommendation(
        domain,
        first_parameters,
        sampling,
        voxel_size_mm=voxel_size_mm,
        cell_map=first_map,
        minimum_feature_mm=first_operand.characteristic_feature_mm,
    )
    second_recommendation = _domain_sampling_recommendation(
        domain,
        second_parameters,
        sampling,
        voxel_size_mm=voxel_size_mm,
        cell_map=second_map,
        minimum_feature_mm=second_operand.characteristic_feature_mm,
    )
    recommendation = min(
        (first_recommendation, second_recommendation),
        key=lambda item: item.voxel_size_mm,
    )
    if transition.driver_mode == "field" and transition_driver is None:
        raise ValueError("field-driven transition requires the selected field object")
    if transition.driver_mode == "plane" and transition_driver is not None:
        raise ValueError("plane-driven transition must not receive a field object")
    build_arguments: dict[str, object] = {"spec": transition.provider_spec()}
    plane_point: np.ndarray | None = None
    plane_normal: np.ndarray | None = None
    if transition_driver is None:
        plane_point, plane_normal = _transition_plane_geometry_for_bounds(
            domain.bounds,
            transition,
        )
        build_arguments.update(
            plane_point_mm=plane_point,
            plane_normal=plane_normal,
        )
    else:
        build_arguments["driver"] = transition_driver
    build = build_provider_transition_body(
        first_operand,
        second_operand,
        domain.as_implicit_body(),
        **build_arguments,
    )
    if domain_display_field is None:
        display_recommendation = _display_recommendation_for_bounds(
            shared_map.bounds,
            recommendation,
            display_voxel_size_mm,
            display_memory_budget_mb,
            visible_field_count,
        )
        field = _sample_implicit_body_for_bounds(
            build.body,
            shared_map.bounds,
            display_recommendation,
            (0.72, 0.32, 0.92, 1.0),
            progress,
            "过渡",
            display_batch_count,
        )
    else:
        field = _sample_intersection_with_domain_cache(
            "过渡",
            domain_display_field,
            lambda points: build.field_body.evaluate_points(points),
            (0.72, 0.32, 0.92, 1.0),
            progress,
            progress_stage="过渡：计算 Ramp 场与设计域交集",
            display_batch_count=display_batch_count,
        )
    field, cleanup = _clean_display_field_components(
        field,
        shared_map,
        build.diagnostics.minimum_feature_mm,
    )
    provider_spec = replace(
        transition.provider_spec(),
        minimum_feature_mm=build.diagnostics.minimum_feature_mm,
    )
    if plane_point is None or plane_normal is None:
        quality = TransitionQualityReport(
            cross_band_connected=False,
            minimum_feature_satisfied=False,
            isolated_fragment_count=0,
            component_count=0,
            inspection_spacing_mm=0.0,
            inspected_samples=0,
            inspection_supported=False,
            status_message="场驱动过渡不采用平面跨带检查。",
        )
    else:
        quality = validate_provider_transition(
            build.body,
            plane_point_mm=plane_point,
            plane_normal=plane_normal,
            spec=provider_spec,
            max_samples=750_000,
        )
    return ImplicitGenerationResult(
        build.body,
        field,
        recommendation,
        shared_map,
        minimum_feature_mm=build.diagnostics.minimum_feature_mm,
        metadata=TransitionGenerationMetadata(
            first_name=first_operand.name,
            second_name=second_operand.name,
            diagnostics=build.diagnostics,
            quality=quality,
        ),
        display_fragment_cleanup=cleanup,
        unclipped_body=build.field_body,
    )


def generate_implicit_domain_preview(
    mesh: trimesh.Trimesh,
    sampling: SamplingParameters,
    base_recommendation: SamplingRecommendation,
    display_voxel_size_mm: float | None = None,
    display_memory_budget_mb: float = 512.0,
    visible_field_count: int = 1,
    display_batch_count: int | None = None,
    progress: ProgressCallback | None = None,
) -> SampledImplicitField:
    """Build the independent design-domain preview field, including shells."""

    display_recommendation = _display_recommendation(
        mesh,
        base_recommendation,
        display_voxel_size_mm,
        display_memory_budget_mb,
        visible_field_count,
    )
    body = _make_design_domain_body(
        mesh,
        sampling,
        shell_thickness_mm=max(display_recommendation.voxel_size_mm, 0.25),
    )
    return _sample_implicit_body(
        body,
        mesh,
        display_recommendation,
        (0.72, 0.76, 0.82, 0.30),
        progress,
        "设计域",
        display_batch_count=display_batch_count,
    )


def design_domain_extraction_map(mesh: trimesh.Trimesh) -> CellMap:
    """Create the bounded grid reference used to reconstruct domain-derived bodies."""

    if len(mesh.faces) == 0:
        raise ValueError("design-domain STL must not be empty")
    bounds = np.asarray(mesh.bounds, dtype=np.float64)
    extents = bounds[1] - bounds[0]
    if not np.isfinite(extents).all() or np.any(extents <= 0.0):
        raise ValueError("design-domain STL must have positive finite extents")
    return CellMap.from_bounds(bounds, extents, boundary_mode="fit_bounds")


def shell_sampling_recommendation(
    mesh: trimesh.Trimesh,
    thickness_mm: float,
    sampling: SamplingParameters,
) -> SamplingRecommendation:
    """Recommend a physical sampling resolution for an SDF-derived shell."""

    thickness = float(thickness_mm)
    sampling.validate()
    if not np.isfinite(thickness) or thickness <= 0.0:
        raise ValueError("shell thickness must be finite and positive")
    extents = np.asarray(mesh.bounds[1] - mesh.bounds[0], dtype=np.float64)
    if not np.isfinite(extents).all() or np.any(extents <= 0.0):
        raise ValueError("design-domain STL must have positive finite extents")
    wall_step = thickness / sampling.min_samples_per_wall
    budget_step = (float(np.prod(extents)) / sampling.target_voxels) ** (1.0 / 3.0)
    voxel_size = min(wall_step, budget_step)
    limiting_constraint = (
        "壳厚采样" if wall_step <= budget_step else "目标体素数"
    )
    shape = tuple(int(np.ceil(value / voxel_size)) + 3 for value in extents)
    return SamplingRecommendation(
        voxel_size_mm=float(voxel_size),
        grid_shape=shape,
        estimated_voxels=int(np.prod(shape, dtype=np.int64)),
        samples_per_cell=float(np.min(extents) / voxel_size),
        samples_per_wall=float(thickness / voxel_size),
        limiting_constraint=limiting_constraint,
    )


def generate_implicit_shell(
    mesh: trimesh.Trimesh,
    thickness_mm: float,
    sampling: SamplingParameters,
    *,
    display_voxel_size_mm: float | None = None,
    display_memory_budget_mb: float = 512.0,
    visible_field_count: int = 1,
    display_batch_count: int | None = None,
    progress: ProgressCallback | None = None,
) -> ImplicitGenerationResult:
    """Create an inward design-domain shell and its disposable display field."""

    if len(mesh.faces) == 0 or not mesh.is_watertight:
        raise ValueError("抽壳需要非空且水密的设计域 STL")
    recommendation = shell_sampling_recommendation(mesh, thickness_mm, sampling)
    design_domain = _make_design_domain_body(mesh, sampling)
    body = make_interior_shell(design_domain, thickness_mm, name=SHELL_RESULT_KEY)
    display_recommendation = _display_recommendation(
        mesh,
        recommendation,
        display_voxel_size_mm,
        display_memory_budget_mb,
        visible_field_count,
    )
    if progress:
        progress("抽壳：计算设计域 SDF 并构造壳体", 0.05)
    field = _sample_implicit_body(
        body,
        mesh,
        display_recommendation,
        (0.22, 0.46, 0.82, 1.0),
        progress,
        "抽壳",
        display_batch_count=display_batch_count,
    )
    if progress:
        progress("抽壳：完成隐式体显示", 1.0)
    return ImplicitGenerationResult(
        body=body,
        display_field=field,
        recommendation=recommendation,
        cell_map=design_domain_extraction_map(mesh),
        minimum_feature_mm=float(thickness_mm),
    )


def generate_implicit_shell_lattice_union(
    mesh: trimesh.Trimesh,
    lattice_generation: ImplicitGenerationResult,
    thickness_mm: float,
    fusion_radius_mm: float,
    sampling: SamplingParameters,
    *,
    display_voxel_size_mm: float | None = None,
    display_memory_budget_mb: float = 512.0,
    visible_field_count: int = 1,
    display_batch_count: int | None = None,
    domain_display_field: SampledImplicitField | None = None,
    progress: ProgressCallback | None = None,
) -> ImplicitGenerationResult:
    """Create a design-limited smooth union between a shell and one lattice."""

    if len(mesh.faces) == 0 or not mesh.is_watertight:
        raise ValueError("壳体与晶格融合需要非空且水密的设计域 STL")
    if lattice_generation.minimum_feature_mm is None:
        raise ValueError("选中的晶格结果缺少最小特征厚度")
    if lattice_generation.cell_map is None:
        raise ValueError("选中的晶格结果缺少 Cell Map")
    shell_recommendation = shell_sampling_recommendation(
        mesh,
        thickness_mm,
        sampling,
    )
    recommendation = min(
        (shell_recommendation, lattice_generation.recommendation),
        key=lambda item: item.voxel_size_mm,
    )
    design_domain = _make_design_domain_body(mesh, sampling)
    if lattice_generation.unclipped_body is not None:
        body = make_coordinated_shell_lattice_union(
            design_domain,
            lattice_generation.unclipped_body,
            thickness_mm,
            fusion_radius_mm,
            name=SHELL_UNION_RESULT_KEY,
        )
    else:
        shell = make_interior_shell(
            design_domain,
            thickness_mm,
            name=SHELL_RESULT_KEY,
        )
        body = make_shell_lattice_union(
            shell,
            lattice_generation.body,
            design_domain,
            fusion_radius_mm,
            name=SHELL_UNION_RESULT_KEY,
        )
    minimum_feature = min(
        float(thickness_mm),
        float(lattice_generation.minimum_feature_mm),
    )
    display_recommendation = _display_recommendation(
        mesh,
        recommendation,
        display_voxel_size_mm,
        display_memory_budget_mb,
        visible_field_count,
    )
    lattice_field = lattice_generation.display_field
    fusion_domain_field: SampledImplicitField | None = None
    fusion_backend = "evaluator"
    if _display_field_meets_recommendation(lattice_field, display_recommendation):
        if (
            domain_display_field is not None
            and not domain_display_field.is_preview_only
            and _sampled_fields_share_grid(domain_display_field, lattice_field)
        ):
            fusion_domain_field = domain_display_field
            fusion_backend = "cached-fields-cpu"
            if progress:
                progress("壳体融合：复用设计域与晶格显示场", 0.60)
        else:
            if progress:
                progress("壳体融合：仅计算设计域 SDF", 0.10)
            axes = [
                lattice_field.origin[axis]
                + np.arange(lattice_field.values.shape[axis], dtype=np.float64)
                * lattice_field.spacing[axis]
                for axis in range(3)
            ]
            fusion_domain_field = _sample_implicit_body_on_grid(
                design_domain,
                axes,
                lattice_field.origin,
                lattice_field.spacing,
                (0.72, 0.72, 0.72, 1.0),
                progress,
                "壳体融合：设计域 SDF",
                display_batch_count,
            )
            fusion_backend = "cached-lattice-field-cpu"
        field = fuse_shell_lattice_fields(
            fusion_domain_field,
            lattice_field,
            thickness_mm,
            fusion_radius_mm,
            name=SHELL_UNION_RESULT_KEY,
            color=(0.12, 0.58, 0.48, 1.0),
        )
        if progress:
            progress("壳体融合：复用晶格场并完成融合", 0.95)
    else:
        if progress:
            progress("壳体融合：计算壳体与晶格隐式场", 0.05)
        fusion_backend = (
            "coordinated-resample"
            if lattice_generation.unclipped_body is not None
            else "evaluator"
        )
        field = _sample_implicit_body(
            body,
            mesh,
            display_recommendation,
            (0.12, 0.58, 0.48, 1.0),
            progress,
            "壳体融合",
            bounds=body.bounds,
            display_batch_count=display_batch_count,
        )
    if progress:
        progress("壳体融合：完成平滑布尔并集", 1.0)
    return ImplicitGenerationResult(
        body=body,
        display_field=field,
        recommendation=recommendation,
        cell_map=lattice_generation.cell_map,
        minimum_feature_mm=minimum_feature,
        metadata={
            "lattice_name": lattice_generation.body.name,
            "shell_thickness_mm": float(thickness_mm),
            "fusion_radius_mm": float(fusion_radius_mm),
            "display_fusion_backend": fusion_backend,
            "domain_display_field": fusion_domain_field,
        },
    )
