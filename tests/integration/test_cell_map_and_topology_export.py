"""Regression tests for the continuous Cell Map and topology-aware export path."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import trimesh

TESTS_DIR = Path(__file__).resolve().parent
if str(TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(TESTS_DIR))

from lattice_studio.presentation.qt import workbench as app
from lattice_studio.engine.implicit.cell_map import CellMap, CellMapFrame
from lattice_studio.engine.implicit.field import ImplicitBody
from lattice_studio.engine.implicit.tpms_compute import NumpyTPMSAdapter, TPMSFieldSpec


def test_cell_map_pads_to_complete_cells_and_keeps_domain_inside() -> None:
    cell_map = CellMap.from_bounds(
        np.array(((0.25, -1.0, 2.0), (5.25, 9.0, 8.5))),
        spacing_mm=(2.0, 3.0, 4.0),
    )

    assert cell_map.cell_counts == (3, 4, 2)
    assert np.all(cell_map.bounds[0] <= (0.25, -1.0, 2.0))
    assert np.all(cell_map.bounds[1] >= (5.25, 9.0, 8.5))
    np.testing.assert_allclose(
        cell_map.bounds[1] - cell_map.bounds[0],
        np.asarray(cell_map.cell_counts) * np.asarray(cell_map.spacing_mm),
    )


def test_fitted_cell_map_matches_bounds_with_uniform_complete_cells() -> None:
    bounds = np.array(((0.25, -1.0, 2.0), (5.25, 9.0, 8.5)))

    cell_map = CellMap.from_bounds(
        bounds,
        spacing_mm=(2.0, 3.0, 4.0),
        boundary_mode="fit_bounds",
    )

    assert cell_map.cell_counts == (3, 3, 2)
    assert cell_map.boundary_mode == "fit_bounds"
    np.testing.assert_allclose(cell_map.bounds, bounds)
    np.testing.assert_allclose(cell_map.requested_spacing_mm, (2.0, 3.0, 4.0))
    np.testing.assert_allclose(cell_map.spacing_mm, (5.0 / 3.0, 10.0 / 3.0, 3.25))
    np.testing.assert_allclose(cell_map.axes()[0][[0, -1]], bounds[:, 0])


def test_fitted_cell_map_drives_actual_tpms_period_and_phase_origin() -> None:
    domain = trimesh.creation.box(extents=(5.0, 10.0, 6.5))
    domain.apply_translation((3.0, -4.0, 2.0))
    parameters = app.TPMSParameters(
        "G",
        (2.0, 3.0, 4.0),
        0.4,
        cell_map_mode="fit_bounds",
    )
    cell_map = app.cell_map_for_parameters(domain, parameters)
    first = cell_map.bounds[0] + np.array((0.27, 0.41, 0.63))
    points = np.vstack((first, first + (cell_map.spacing_mm[0], 0.0, 0.0)))

    values = app._cell_map_tpms_shell_field(
        "G",
        points,
        parameters,
        cell_map=cell_map,
        max_tile_points=None,
    )

    np.testing.assert_allclose(values[0], values[1], rtol=1e-5, atol=1e-5)


def test_transition_keeps_independent_fitted_g_and_d_cell_maps() -> None:
    domain = trimesh.creation.box(extents=(61.5, 169.1, 11.5))
    g_parameters = app.TPMSParameters(
        "G",
        (7.0, 15.0, 4.5),
        1.2,
        cell_map_mode="fit_bounds",
    )
    d_parameters = app.TPMSParameters(
        "D",
        (16.0, 8.0, 6.0),
        1.3,
        cell_map_mode="fit_bounds",
    )

    g_map, d_map, sampling_map = app.transition_cell_maps(
        domain,
        g_parameters,
        d_parameters,
    )

    np.testing.assert_allclose(g_map.bounds, domain.bounds)
    np.testing.assert_allclose(d_map.bounds, domain.bounds)
    np.testing.assert_allclose(sampling_map.bounds, domain.bounds)
    assert g_map.cell_counts == (9, 11, 3)
    assert d_map.cell_counts == (4, 21, 2)
    assert g_map.spacing_mm != d_map.spacing_mm


def test_transition_intersection_returns_without_progress_callback() -> None:
    domain = trimesh.creation.box(extents=(12.0, 18.0, 6.0))
    g_parameters = app.TPMSParameters("G", (5.0, 7.0, 4.0), 0.8)
    d_parameters = app.TPMSParameters("D", (8.0, 4.0, 5.0), 0.9)
    transition = app.TransitionParameters(transition_width_mm=4.0)
    g_map, d_map, _sampling_map = app.transition_cell_maps(
        domain,
        g_parameters,
        d_parameters,
    )
    points = np.array(((0.0, 0.0, 0.0), (1.0, 2.0, 0.5)), dtype=np.float32)

    result = app._tpms_transition_chunk_field(
        points,
        domain_sdf=np.full(len(points), -1.0, dtype=np.float32),
        g_parameters=g_parameters,
        d_parameters=d_parameters,
        plane_point=np.zeros(3),
        plane_normal=np.array((1.0, 0.0, 0.0)),
        transition=transition,
        g_cell_map=g_map,
        d_cell_map=d_map,
        max_tile_points=None,
        stage_progress=None,
    )

    assert result.shape == (2,)
    assert np.isfinite(result).all()


def test_cell_map_uses_independent_periodic_coordinates() -> None:
    cell_map = CellMap.from_bounds(
        np.array(((0.0, 0.0, 0.0), (6.0, 8.0, 10.0))),
        spacing_mm=(2.0, 4.0, 5.0),
    )
    points = np.array(((0.2, 1.1, 2.3), (2.2, 1.1, 2.3)))

    local = cell_map.to_cell_coordinates(points)

    np.testing.assert_allclose(local[1] - local[0], (1.0, 0.0, 0.0))


def test_cell_map_frame_builds_a_right_handed_orthonormal_basis() -> None:
    frame = CellMapFrame.from_origin_axes(
        origin=(2.0, -3.0, 5.0),
        u_axis=(2.0, 0.0, 0.0),
        v_axis=(1.0, 3.0, 0.0),
    )

    np.testing.assert_allclose(frame.axes @ frame.axes.T, np.eye(3), atol=1e-12)
    np.testing.assert_allclose(frame.axes[2], np.cross(frame.axes[0], frame.axes[1]))
    np.testing.assert_allclose(frame.origin, (2.0, -3.0, 5.0))


def test_cell_map_frame_can_follow_rotated_domain_minimum() -> None:
    domain = trimesh.creation.box(extents=(10.0, 8.0, 4.0))
    domain.apply_translation((3.0, -2.0, 1.0))
    angle = np.deg2rad(37.0)
    u_axis = (np.cos(angle), np.sin(angle), 0.0)
    v_axis = (-np.sin(angle), np.cos(angle), 0.0)

    frame = CellMapFrame.aligned_to_points_minimum(
        domain.vertices,
        u_axis,
        v_axis,
    )
    local_points = frame.to_local(domain.vertices)
    cell_map = CellMap.from_points(
        domain.vertices,
        spacing_mm=(2.5, 2.0, 1.5),
        frame=frame,
    )

    np.testing.assert_allclose(local_points.min(axis=0), 0.0, atol=1.0e-12)
    assert not np.allclose(frame.origin, domain.bounds[0])
    assert cell_map.index_min == (0, 0, 0)
    assert np.all(cell_map.contains(domain.vertices, tolerance=1.0e-9))


def test_rotated_cell_map_uses_signed_indices_and_covers_domain_points() -> None:
    angle = np.deg2rad(45.0)
    frame = CellMapFrame.from_origin_axes(
        origin=(0.0, 0.0, 0.0),
        u_axis=(np.cos(angle), np.sin(angle), 0.0),
        v_axis=(-np.sin(angle), np.cos(angle), 0.0),
    )
    domain_points = np.array(
        (
            (-2.0, -1.0, -0.5),
            (4.0, -1.0, -0.5),
            (-2.0, 3.0, 2.5),
            (4.0, 3.0, 2.5),
        )
    )

    cell_map = CellMap.from_points(
        domain_points,
        spacing_mm=(2.0, 1.5, 1.0),
        frame=frame,
    )

    assert cell_map.boundary_mode == "complete_cells"
    assert any(index < 0 for index in cell_map.index_min)
    assert np.all(cell_map.contains(domain_points))
    local = cell_map.to_cell_coordinates(domain_points)
    round_trip = frame.to_world(local * np.asarray(cell_map.spacing_mm))
    np.testing.assert_allclose(round_trip, domain_points, atol=1e-12)


def test_cell_map_frame_rejects_parallel_input_axes() -> None:
    with np.testing.assert_raises_regex(ValueError, "parallel"):
        CellMapFrame.from_origin_axes(
            origin=(0.0, 0.0, 0.0),
            u_axis=(1.0, 0.0, 0.0),
            v_axis=(2.0, 0.0, 0.0),
        )


def test_tpms_parameters_build_a_rotated_complete_cell_map() -> None:
    domain = trimesh.creation.box(extents=(10.0, 8.0, 4.0))
    domain.apply_translation((3.0, -2.0, 1.0))
    angle = np.deg2rad(30.0)
    parameters = app.TPMSParameters(
        "G",
        (2.5, 2.0, 1.5),
        0.3,
        cell_map_mode="fit_bounds",
        frame_origin_mm=tuple(domain.bounds[0]),
        frame_u_axis=(np.cos(angle), np.sin(angle), 0.0),
        frame_v_axis=(-np.sin(angle), np.cos(angle), 0.0),
    )

    cell_map = app.cell_map_for_parameters(domain, parameters)

    assert cell_map.boundary_mode == "complete_cells"
    np.testing.assert_allclose(cell_map.frame.origin, domain.bounds[0])
    np.testing.assert_allclose(cell_map.frame.u_axis, parameters.frame_u_axis)
    assert np.all(cell_map.contains(domain.vertices))
    first = domain.centroid
    points = np.vstack((first, first + cell_map.frame.u_axis * cell_map.spacing_mm[0]))
    values = app._cell_map_tpms_shell_field(
        "G", points, parameters, cell_map, max_tile_points=None
    )
    np.testing.assert_allclose(values[0], values[1], rtol=1e-5, atol=1e-5)


def test_transition_preserves_independent_g_and_d_frames() -> None:
    domain = trimesh.creation.box(extents=(12.0, 9.0, 5.0))
    origin = tuple(domain.bounds[0])
    g_parameters = app.TPMSParameters(
        "G",
        (3.0, 2.0, 2.5),
        0.4,
        frame_origin_mm=origin,
        frame_u_axis=(0.0, 1.0, 0.0),
        frame_v_axis=(-1.0, 0.0, 0.0),
    )
    d_parameters = app.TPMSParameters(
        "D",
        (2.0, 3.0, 2.5),
        0.4,
        frame_origin_mm=origin,
        frame_u_axis=(1.0, 0.0, 0.0),
        frame_v_axis=(0.0, 0.0, 1.0),
    )

    g_map, d_map, sampling_map = app.transition_cell_maps(
        domain, g_parameters, d_parameters
    )

    assert not np.allclose(g_map.frame.axes, d_map.frame.axes)
    assert np.all(g_map.contains(domain.vertices))
    assert np.all(d_map.contains(domain.vertices))
    assert np.all(sampling_map.bounds[0] <= g_map.bounds[0])
    assert np.all(sampling_map.bounds[1] >= d_map.bounds[1])


def test_rotated_cell_map_sampling_grid_uses_its_world_aabb() -> None:
    domain = trimesh.creation.box(extents=(10.0, 8.0, 4.0))
    angle = np.deg2rad(45.0)
    parameters = app.TPMSParameters(
        "G",
        (2.5, 2.0, 1.5),
        0.3,
        frame_origin_mm=tuple(domain.bounds[0]),
        frame_u_axis=(np.cos(angle), np.sin(angle), 0.0),
        frame_v_axis=(-np.sin(angle), np.cos(angle), 0.0),
    )
    sampling = app.SamplingParameters(target_voxels=100_000)
    cell_map = app.cell_map_for_parameters(domain, parameters)

    recommendation = app.recommendation_for_voxel_size(
        domain,
        app._parameters_for_cell_map(parameters, cell_map),
        sampling,
        voxel_size_mm=0.5,
        cell_map=cell_map,
    )

    world_extent = cell_map.bounds[1] - cell_map.bounds[0]
    expected_shape = tuple(int(np.ceil(value / 0.5)) + 3 for value in world_extent)
    assert recommendation.grid_shape == expected_shape


def test_tpms_backend_accepts_independent_cell_periods() -> None:
    adapter = NumpyTPMSAdapter()
    spec = TPMSFieldSpec("G", (2.0, 3.0, 4.0), wall_thickness_mm=0.4)
    points = np.array(((0.3, 0.7, 1.1), (2.3, 0.7, 1.1), (0.3, 3.7, 1.1)))

    values = adapter.evaluate_shell(points, spec, max_tile_points=None)

    np.testing.assert_allclose(values[0], values[1], rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(values[0], values[2], rtol=1e-5, atol=1e-5)


def test_export_voxel_recommendation_is_independent_from_display_spacing() -> None:
    domain = trimesh.creation.box(extents=(12.0, 24.0, 6.0))
    cell_map = CellMap.from_bounds(domain.bounds, spacing_mm=(4.0, 6.0, 3.0))

    display_spacing = np.array((1.0, 1.0, 1.0))
    export_spacing = app.recommend_export_spacing(
        cell_map,
        tolerance_mm=0.1,
        wall_thickness_mm=0.8,
    )

    assert np.all(export_spacing < display_spacing)
    assert np.all(export_spacing <= 0.1)


def test_export_spacing_report_identifies_the_constraint_for_each_axis() -> None:
    cell_map = CellMap.from_bounds(
        np.array(((0.0, 0.0, 0.0), (8.0, 16.0, 32.0))),
        spacing_mm=(4.0, 8.0, 16.0),
    )
    spacing = app.recommend_export_spacing(
        cell_map,
        tolerance_mm=0.4,
        wall_thickness_mm=3.0,
    )

    np.testing.assert_allclose(spacing, (0.25, 0.4, 0.4))
    assert app.export_spacing_limit_labels(
        spacing,
        cell_map,
        tolerance_mm=0.4,
        minimum_feature_mm=3.0,
    ) == (
        "X 晶胞尺寸 / 16",
        "STL 容差",
        "STL 容差",
    )


def test_exact_export_spacing_bypasses_feature_and_period_limits() -> None:
    cell_map = CellMap.from_bounds(
        np.array(((-2.0, -2.0, -2.0), (2.0, 2.0, 2.0))),
        spacing_mm=(4.0, 4.0, 4.0),
    )

    recommended = app.resolve_export_spacing(
        cell_map,
        tolerance_mm=0.5,
        wall_thickness_mm=0.3,
        spacing_mode="recommended",
    )
    exact = app.resolve_export_spacing(
        cell_map,
        tolerance_mm=0.5,
        wall_thickness_mm=0.3,
        spacing_mode="exact",
    )

    np.testing.assert_allclose(recommended, (0.1, 0.1, 0.1))
    np.testing.assert_allclose(exact, (0.5, 0.5, 0.5))
    assert app.export_spacing_limit_labels(
        exact,
        cell_map,
        tolerance_mm=0.5,
        minimum_feature_mm=0.3,
        spacing_mode="exact",
    ) == ("用户指定 STL 间距",) * 3


def test_export_grid_estimate_matches_exact_reconstruction_grid() -> None:
    cell_map = CellMap.from_bounds(
        np.array(((-2.0, -2.0, -2.0), (2.0, 2.0, 2.0))),
        spacing_mm=(4.0, 4.0, 4.0),
    )

    def sphere(points, _reporter):
        return (
            np.linalg.norm(points.astype(np.float32), axis=1) - 1.25
        ).astype(np.float32)

    body = ImplicitBody("sphere", cell_map.bounds, sphere)
    estimate = app.estimate_export_grid(
        body,
        cell_map,
        tolerance_mm=0.5,
        wall_thickness_mm=0.3,
        spacing_mode="exact",
    )

    result = app.reconstruct_implicit_mesh(
        body,
        cell_map,
        tolerance_mm=0.5,
        wall_thickness_mm=0.3,
        processing_mode="single_pass",
        spacing_mode="exact",
    )

    assert estimate.grid_shape == (11, 11, 11)
    assert estimate.total_voxels == 1_331
    assert result.spacing_mm == (0.5, 0.5, 0.5)
    assert result.spacing_mode == "exact"
    assert result.spacing_limits == ("用户指定 STL 间距",) * 3
    assert result.extraction_statistics is not None
    assert result.extraction_statistics.dense_grid_points == estimate.total_voxels


def test_topology_aware_simplification_repairs_non_watertight_candidate(
    monkeypatch,
) -> None:
    source = trimesh.creation.icosphere(subdivisions=3, radius=1.0)
    candidate = trimesh.creation.icosphere(subdivisions=2, radius=1.0)
    candidate.update_faces(np.arange(len(candidate.faces)) != 0)
    candidate.remove_unreferenced_vertices()
    assert candidate.is_watertight is False

    monkeypatch.setattr(
        app,
        "_compiled_simplify_mesh",
        lambda _mesh, _target: (candidate.copy(), "stub"),
    )

    result, report = app.simplify_mesh_with_quality(
        source,
        target_faces=400,
        tolerance_mm=0.2,
    )

    assert report.accepted is True
    assert report.repair_attempted is True
    assert report.repaired is True
    assert len(result.faces) < len(source.faces)
    assert result.is_watertight
    assert result.body_count == 1


def test_pre_export_conditioning_removes_tiny_edges_without_overmerging() -> None:
    from skimage.measure import marching_cubes

    axis = np.arange(-2.0, 2.001, 0.1)
    x, y, z = np.meshgrid(axis, axis, axis, indexing="ij")
    field = np.sqrt((x - 0.03) ** 2 + y**2 + z**2) - 1.0
    vertices, faces, _normals, _values = marching_cubes(
        field,
        level=0.0,
        spacing=(0.1, 0.1, 0.1),
    )
    vertices += (-2.0, -2.0, -2.0)
    source = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)

    conditioned, report = app._condition_mesh_for_slicing(
        source,
        export_tolerance_mm=0.5,
        spacing_mm=np.array((0.1, 0.1, 0.1)),
    )

    assert report.applied is True
    assert report.topology_preserved is True
    assert report.tolerance_mm == 0.05
    assert report.max_deviation_mm is not None
    assert report.max_deviation_mm <= report.tolerance_mm
    assert report.poor_triangle_fraction_after <= report.poor_triangle_fraction_before
    assert report.edge_p01_mm_after >= report.edge_p01_mm_before
    assert len(conditioned.faces) < len(source.faces)
    assert len(conditioned.faces) >= len(source.faces) * 0.79
    assert conditioned.is_watertight
    assert conditioned.is_winding_consistent
    assert conditioned.body_count == 1


def test_pre_export_conditioning_rolls_back_when_candidate_is_too_aggressive(
    monkeypatch,
) -> None:
    source = trimesh.creation.icosphere(subdivisions=2, radius=1.0)
    monkeypatch.setattr(
        app,
        "_compiled_simplify_mesh",
        lambda _mesh, _target: (
            trimesh.creation.box(extents=(1.0, 1.0, 1.0)),
            "stub",
        ),
    )

    conditioned, report = app._condition_mesh_for_slicing(
        source,
        export_tolerance_mm=0.05,
        spacing_mm=np.array((0.1, 0.1, 0.1)),
    )

    assert report.applied is False
    assert "rollback" in report.message
    assert len(conditioned.faces) == len(source.faces)
    assert conditioned.is_watertight
    assert conditioned.body_count == 1


def test_stl_slicing_optimization_can_be_disabled_without_changing_mesh() -> None:
    cell_map = CellMap.from_bounds(
        np.array(((-2.0, -2.0, -2.0), (2.0, 2.0, 2.0))),
        spacing_mm=(4.0, 4.0, 4.0),
    )

    def sphere(points, _reporter):
        return (
            np.linalg.norm(points.astype(np.float32), axis=1) - 1.0
        ).astype(np.float32)

    result = app.reconstruct_implicit_mesh(
        ImplicitBody("sphere", cell_map.bounds, sphere),
        cell_map,
        tolerance_mm=0.2,
        wall_thickness_mm=0.5,
        optimize_for_slicing=False,
    )

    assert result.conditioning is None
    assert result.quality.watertight
    assert result.quality.connected_components == 1


def test_topology_aware_simplification_removes_numerical_component(
    monkeypatch,
) -> None:
    source = trimesh.creation.icosphere(subdivisions=3, radius=1.0)
    main = trimesh.creation.icosphere(subdivisions=2, radius=1.0)
    fragment = trimesh.creation.icosphere(subdivisions=1, radius=0.005)
    fragment.apply_translation((2.0, 0.0, 0.0))
    candidate = trimesh.util.concatenate((main, fragment))
    assert candidate.body_count == 2

    monkeypatch.setattr(
        app,
        "_compiled_simplify_mesh",
        lambda _mesh, _target: (candidate.copy(), "stub"),
    )

    result, report = app.simplify_mesh_with_quality(
        source,
        target_faces=400,
        tolerance_mm=0.2,
    )

    assert report.accepted is True
    assert report.repair_attempted is True
    assert report.repaired is True
    assert result.is_watertight
    assert result.body_count == 1


def test_topology_aware_simplification_rolls_back_when_repair_fails(
    monkeypatch,
) -> None:
    source = trimesh.creation.icosphere(subdivisions=2, radius=1.0)
    candidate = trimesh.util.concatenate(
        trimesh.creation.box(extents=(1.0, 1.0, 1.0)),
        trimesh.creation.box(extents=(1.0, 1.0, 1.0)).apply_translation(
            (3.0, 0.0, 0.0)
        ),
    )
    monkeypatch.setattr(
        app,
        "_compiled_simplify_mesh",
        lambda _mesh, _target: (candidate.copy(), "stub"),
    )

    result, report = app.simplify_mesh_with_quality(
        source,
        target_faces=max(4, len(source.faces) // 2),
        tolerance_mm=0.1,
    )

    assert len(result.faces) == len(source.faces)
    assert report.accepted is False
    assert report.repair_attempted is True
    assert report.repaired is False
    assert "回退" in report.message


def test_export_reconstruction_adds_a_closed_halo_and_reports_topology() -> None:
    cell_map = CellMap.from_bounds(
        np.array(((-2.0, -2.0, -2.0), (2.0, 2.0, 2.0))),
        spacing_mm=(4.0, 4.0, 4.0),
    )
    def sphere(points, _reporter):
        return (
            np.linalg.norm(points.astype(np.float32), axis=1) - 1.25
        ).astype(np.float32)

    body = ImplicitBody(
        "sphere",
        cell_map.bounds,
        sphere,
    )

    result = app.reconstruct_implicit_mesh(
        body,
        cell_map,
        tolerance_mm=0.2,
        wall_thickness_mm=0.8,
    )

    assert result.quality.watertight is True
    assert result.quality.connected_components == 1
    assert result.quality.winding_consistent is True
    assert len(result.mesh.faces) > 0
    assert result.extraction_backend == "dense-grid"
    assert result.extraction_statistics.total_bricks == 1
    assert result.extraction_statistics.active_bricks == 1
    assert result.extraction_statistics.skipped_bricks == 0
    assert result.extraction_fallback_reason is None
    assert result.spacing_mm == (0.2, 0.2, 0.2)
    assert result.spacing_limits == ("STL 容差",) * 3


def test_export_reconstruction_honors_each_processing_mode() -> None:
    cell_map = CellMap.from_bounds(
        np.array(((-2.0, -2.0, -2.0), (2.0, 2.0, 2.0))),
        spacing_mm=(4.0, 4.0, 4.0),
    )

    def sphere(points, _reporter):
        return (
            np.linalg.norm(points.astype(np.float32), axis=1) - 1.0
        ).astype(np.float32)

    body = ImplicitBody("sphere", cell_map.bounds, sphere)
    results = {}
    for mode, batch_count in (
        ("single_pass", 1),
        ("batched_field", 2),
        ("chunked_marching_cubes", 2),
    ):
        results[mode] = app.reconstruct_implicit_mesh(
            body,
            cell_map,
            tolerance_mm=0.2,
            wall_thickness_mm=0.5,
            processing_mode=mode,
            batch_count=batch_count,
        )

    assert results["single_pass"].extraction_backend == "dense-grid"
    assert results["batched_field"].extraction_backend == "batched-field-dense-grid"
    assert results["chunked_marching_cubes"].extraction_backend == "chunked-marching-cubes"
    assert results["single_pass"].extraction_statistics.total_bricks == 1
    assert results["batched_field"].extraction_statistics.total_bricks == 1
    assert results["chunked_marching_cubes"].extraction_statistics.total_bricks == 2
    assert results["chunked_marching_cubes"].extraction_statistics.evaluated_points > (
        results["single_pass"].extraction_statistics.evaluated_points
    )
    for result in results.values():
        assert result.quality.watertight is True
        assert result.quality.connected_components == 1


def test_mesh_inspection_counts_components_without_materializing_them(monkeypatch) -> None:
    mesh = trimesh.creation.icosphere(subdivisions=3, radius=1.0)

    def reject_split(*_args, **_kwargs):
        raise AssertionError("inspect_mesh materialized connected components")

    monkeypatch.setattr(trimesh.Trimesh, "split", reject_split)

    quality = app.inspect_mesh(mesh)

    assert quality.watertight is True
    assert quality.winding_consistent is True
    assert quality.connected_components == 1


def test_export_reconstruction_skips_repair_for_an_accepted_mc_mesh(monkeypatch) -> None:
    cell_map = CellMap.from_bounds(
        np.array(((-2.0, -2.0, -2.0), (2.0, 2.0, 2.0))),
        spacing_mm=(4.0, 4.0, 4.0),
    )
    body = ImplicitBody(
        "sphere",
        cell_map.bounds,
        lambda points, _reporter: (
            np.linalg.norm(points.astype(np.float32), axis=1) - 1.25
        ).astype(np.float32),
    )

    def reject_repair(*_args, **_kwargs):
        raise AssertionError("accepted Marching Cubes mesh was repaired again")

    monkeypatch.setattr(app, "repair_mesh", reject_repair)

    result = app.reconstruct_implicit_mesh(
        body,
        cell_map,
        tolerance_mm=0.2,
        wall_thickness_mm=0.8,
    )

    assert result.quality.watertight is True
    assert result.quality.winding_consistent is True
    assert result.quality.connected_components == 1


def test_topology_valid_simplification_still_rolls_back_above_tolerance(monkeypatch) -> None:
    source = trimesh.creation.icosphere(subdivisions=2, radius=1.0)
    monkeypatch.setattr(
        app,
        "_compiled_simplify_mesh",
        lambda _mesh, _target: (
            trimesh.creation.box(extents=(1.0, 1.0, 1.0)),
            "stub",
        ),
    )

    result, report = app.simplify_mesh_with_quality(
        source,
        target_faces=12,
        tolerance_mm=0.05,
    )

    assert len(result.faces) == len(source.faces)
    assert report.accepted is False
    assert report.max_deviation_mm is not None
    assert report.max_deviation_mm > 0.05


def test_vtk_cpp_simplifier_reduces_a_closed_mesh_within_tolerance() -> None:
    source = trimesh.creation.icosphere(subdivisions=3, radius=1.0)

    result, report = app.simplify_mesh_with_quality(
        source,
        target_faces=300,
        tolerance_mm=0.25,
    )

    assert report.accepted is True
    assert report.backend == "VTK C++"
    assert len(result.faces) < len(source.faces)
    assert result.is_watertight
    assert len(result.split(only_watertight=False)) == 1


def test_field_repair_connects_only_a_gap_within_repair_tolerance() -> None:
    cell_map = CellMap.from_bounds(
        np.array(((-3.0, -2.0, -2.0), (3.0, 2.0, 2.0))),
        spacing_mm=(6.0, 4.0, 4.0),
    )

    def two_spheres(points: np.ndarray, _reporter) -> np.ndarray:
        values = points.astype(np.float32)
        first = np.linalg.norm(values - (-1.1, 0.0, 0.0), axis=1) - 1.0
        second = np.linalg.norm(values - (1.1, 0.0, 0.0), axis=1) - 1.0
        return np.minimum(first, second).astype(np.float32)

    body = ImplicitBody("two-spheres", cell_map.bounds, two_spheres)
    original = app.reconstruct_implicit_mesh(
        body,
        cell_map,
        tolerance_mm=0.1,
        wall_thickness_mm=1.0,
    )
    repaired = app.reconstruct_implicit_mesh(
        body,
        cell_map,
        tolerance_mm=0.1,
        wall_thickness_mm=1.0,
        repair_tolerance_mm=0.3,
    )

    assert original.quality.watertight is True
    assert original.quality.connected_components == 2
    assert repaired.field_repaired is True
    assert repaired.quality.watertight is True
    assert repaired.quality.connected_components == 1


def test_numerical_fragment_cleanup_removes_only_sub_tolerance_artifact() -> None:
    main = trimesh.creation.icosphere(subdivisions=2, radius=2.0)
    artifact = trimesh.creation.box(extents=(0.02, 0.02, 0.02))
    artifact.apply_translation((3.0, 0.0, 0.0))
    source = trimesh.util.concatenate((main, artifact))

    cleaned, report = app.remove_numerical_fragments(
        source,
        tolerance_mm=0.1,
    )

    assert app.inspect_mesh(source).connected_components == 2
    assert app.inspect_mesh(cleaned).connected_components == 1
    assert report.removed_components == 1
    assert report.removed_faces == len(artifact.faces)
    assert report.removed_volume_mm3 > 0.0


def test_numerical_fragment_cleanup_preserves_real_independent_component() -> None:
    first = trimesh.creation.icosphere(subdivisions=2, radius=1.0)
    second = trimesh.creation.icosphere(subdivisions=2, radius=0.4)
    second.apply_translation((3.0, 0.0, 0.0))
    source = trimesh.util.concatenate((first, second))

    cleaned, report = app.remove_numerical_fragments(
        source,
        tolerance_mm=0.1,
    )

    assert app.inspect_mesh(cleaned).connected_components == 2
    assert report.removed_components == 0
    assert len(cleaned.faces) == len(source.faces)


def test_floating_component_cleanup_removes_small_physical_island() -> None:
    main = trimesh.creation.icosphere(subdivisions=2, radius=2.0)
    island = trimesh.creation.icosphere(subdivisions=2, radius=0.6)
    island.apply_translation((4.0, 0.0, 0.0))
    source = trimesh.util.concatenate((main, island))

    cleaned, report = app.remove_numerical_fragments(
        source,
        tolerance_mm=0.1,
        component_extent_limit_mm=3.0,
        relative_volume_limit=0.05,
    )

    assert app.inspect_mesh(source).connected_components == 2
    assert app.inspect_mesh(cleaned).connected_components == 1
    assert report.removed_components == 1
    assert report.removed_volume_mm3 > 0.0


def test_field_fragment_cleanup_removes_island_before_surface_extraction() -> None:
    axis = np.arange(-4.0, 4.001, 0.2)
    x, y, z = np.meshgrid(axis, axis, axis, indexing="ij")
    main = np.sqrt(x**2 + y**2 + z**2) - 2.0
    island = np.sqrt((x - 3.0) ** 2 + y**2 + z**2) - 0.65
    field = np.minimum(main, island).astype(np.float32)

    cleaned, report = app.remove_small_negative_field_components(
        field,
        spacing_mm=(0.2, 0.2, 0.2),
        tolerance_mm=0.1,
        relative_volume_limit=0.05,
        component_extent_limit_mm=2.0,
    )

    from scipy import ndimage

    _, original_components = ndimage.label(
        field < 0.0,
        structure=ndimage.generate_binary_structure(3, 3),
    )
    _, cleaned_components = ndimage.label(
        cleaned < 0.0,
        structure=ndimage.generate_binary_structure(3, 3),
    )
    assert original_components == 2
    assert cleaned_components == 1
    assert report.removed_components == 1
    assert report.removed_voxels > 0


def test_field_fragment_cleanup_preserves_boundary_component() -> None:
    field = np.ones((12, 12, 12), dtype=np.float32)
    field[0:2, 5:7, 5:7] = -1.0
    field[6:9, 6:9, 6:9] = -1.0

    cleaned, report = app.remove_small_negative_field_components(
        field,
        spacing_mm=(0.5, 0.5, 0.5),
        tolerance_mm=0.1,
        relative_volume_limit=0.5,
        component_extent_limit_mm=3.0,
    )

    assert np.count_nonzero(cleaned < 0.0) == np.count_nonzero(field < 0.0)
    assert report.removed_components == 0


def test_export_cleanup_runs_on_authoritative_field_before_marching_cubes() -> None:
    cell_map = CellMap.from_bounds(
        np.array(((-3.0, -3.0, -3.0), (3.0, 3.0, 3.0))),
        spacing_mm=(6.0, 6.0, 6.0),
    )

    def two_solids(points, _reporter):
        points = points.astype(np.float32)
        main = np.linalg.norm(points - (-1.0, 0.0, 0.0), axis=1) - 1.4
        island = np.linalg.norm(points - (2.0, 0.0, 0.0), axis=1) - 0.3
        return np.minimum(main, island).astype(np.float32)

    result = app.reconstruct_implicit_mesh(
        ImplicitBody("two-solids", cell_map.bounds, two_solids),
        cell_map,
        tolerance_mm=0.1,
        wall_thickness_mm=0.5,
        clean_numerical_fragments=True,
        spacing_mode="exact",
    )

    assert result.field_fragment_cleanup is not None
    assert result.field_fragment_cleanup.removed_components == 1
    assert result.quality.connected_components == 1


def test_numerical_fragment_cleanup_reuses_an_unchanged_mesh() -> None:
    source = trimesh.creation.icosphere(subdivisions=2, radius=1.0)

    cleaned, report = app.remove_numerical_fragments(source, tolerance_mm=0.1)

    assert cleaned is source
    assert report.removed_components == 0


def test_watertight_fragment_cleanup_does_not_split_component_meshes(monkeypatch) -> None:
    main = trimesh.creation.icosphere(subdivisions=2, radius=2.0)
    artifact = trimesh.creation.box(extents=(0.02, 0.02, 0.02))
    artifact.apply_translation((3.0, 0.0, 0.0))
    source = trimesh.util.concatenate((main, artifact))

    def reject_split(*_args, **_kwargs):
        raise AssertionError("fragment cleanup materialized component meshes")

    monkeypatch.setattr(trimesh.Trimesh, "split", reject_split)

    cleaned, report = app.remove_numerical_fragments(source, tolerance_mm=0.1)

    assert report.removed_components == 1
    assert app.inspect_mesh(cleaned).connected_components == 1
