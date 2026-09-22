"""Behavioral tests for STL-backed periodic custom unit cells."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import trimesh

from lattice_studio.engine.implicit.cell_map import CellMap, CellMapFrame
from lattice_studio.engine.implicit.custom_unit_cell import (
    CustomUnitCellError,
    UnitCellDomain,
    make_periodic_lattice,
    prepare_stl_unit_cell,
)
from lattice_studio.engine.implicit.field import (
    ImplicitBody,
    SampledImplicitField,
    remove_small_negative_islands,
)


class _AnalyticSphereBackend:
    """Stand-in for the CPU/CUDA mesh-distance adapter."""

    def signed_distance(self, vertices, faces, points, max_tile_points=250_000):
        del faces, max_tile_points
        bounds = np.vstack((vertices.min(axis=0), vertices.max(axis=0)))
        center = bounds.mean(axis=0)
        radius = float(np.min(bounds[1] - bounds[0]) * 0.3)
        return np.linalg.norm(np.asarray(points) - center, axis=1) - radius


class _AnalyticOffsetSphereBackend:
    """Asymmetric source field used to exercise mismatched periodic faces."""

    def signed_distance(self, vertices, faces, points, max_tile_points=250_000):
        del faces, max_tile_points
        bounds = np.vstack((vertices.min(axis=0), vertices.max(axis=0)))
        extent = bounds[1] - bounds[0]
        center = bounds.mean(axis=0) + np.array((0.2 * extent[0], 0.0, 0.0))
        radius = float(np.min(extent) * 0.3)
        return np.linalg.norm(np.asarray(points) - center, axis=1) - radius


def _write_stl(path: Path, mesh: trimesh.Trimesh) -> Path:
    mesh.export(path)
    return path


def _constant_domain(bounds, value: float = -100.0) -> ImplicitBody:
    return ImplicitBody(
        name="design-domain",
        bounds=np.asarray(bounds, dtype=np.float64),
        evaluate=lambda points, _stage: np.full(len(points), value, dtype=np.float32),
    )


def _sphere_domain(bounds, radius: float = 1.0) -> ImplicitBody:
    def sphere(points, _stage):
        return (
            np.linalg.norm(np.asarray(points, dtype=np.float32), axis=1) - radius
        ).astype(np.float32)

    return ImplicitBody(
        name="sphere-domain",
        bounds=np.asarray(bounds, dtype=np.float64),
        evaluate=sphere,
    )


def test_prepare_stl_unit_cell_removes_only_degenerate_faces(tmp_path):
    box = trimesh.creation.box(extents=(2.0, 3.0, 4.0))
    faces = np.vstack((box.faces, np.array(((0, 0, 1),), dtype=np.int64)))
    source = trimesh.Trimesh(vertices=box.vertices, faces=faces, process=False)
    path = _write_stl(tmp_path / "cell.stl", source)

    prepared = prepare_stl_unit_cell(path, geometry_backend=_AnalyticSphereBackend())

    assert prepared.report.original_faces == len(faces)
    assert prepared.report.removed_degenerate_faces == 1
    assert prepared.report.prepared_faces == len(box.faces)
    assert prepared.report.watertight is True
    assert prepared.report.connected_components == 1
    assert prepared.report.characteristic_thickness_mm > 0.0
    np.testing.assert_allclose(prepared.domain.extent_mm, (2.0, 3.0, 4.0))


def test_prepare_stl_unit_cell_rejects_open_mesh(tmp_path):
    box = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
    box.update_faces(np.arange(len(box.faces)) != 0)
    path = _write_stl(tmp_path / "open-cell.stl", box)

    with pytest.raises(CustomUnitCellError, match="watertight"):
        prepare_stl_unit_cell(path, geometry_backend=_AnalyticSphereBackend())


def test_unit_cell_domain_rejects_non_positive_extent():
    with pytest.raises(ValueError, match="positive extents"):
        UnitCellDomain(np.array(((0.0, 0.0, 0.0), (1.0, 0.0, 1.0))))


def test_periodic_lattice_repeats_along_rotated_frame_axes(tmp_path):
    source = trimesh.creation.icosphere(subdivisions=2, radius=0.5)
    path = _write_stl(tmp_path / "sphere-cell.stl", source)
    prepared = prepare_stl_unit_cell(path, geometry_backend=_AnalyticSphereBackend())
    frame = CellMapFrame.from_origin_axes(
        origin=(3.0, -2.0, 1.0),
        u_axis=(0.0, 1.0, 0.0),
        v_axis=(-1.0, 0.0, 0.0),
    )
    coverage = np.array(
        (
            (-5.0, -5.0, -5.0),
            (10.0, -5.0, -5.0),
            (-5.0, 10.0, -5.0),
            (-5.0, -5.0, 10.0),
            (10.0, 10.0, 10.0),
        )
    )
    cell_map = CellMap.from_points(coverage, (2.0, 3.0, 4.0), frame)
    design_domain = _constant_domain(cell_map.bounds)
    body = make_periodic_lattice(prepared, cell_map, design_domain)
    point = frame.to_world(np.array(((0.5, 1.5, 2.0),)))[0]
    translated = np.vstack(
        (
            point,
            point + frame.u_axis * 2.0,
            point + frame.v_axis * 3.0,
            point + frame.w_axis * 4.0,
        )
    )

    values = body.evaluate_points(translated)

    np.testing.assert_allclose(values, np.repeat(values[0], 4), atol=1.0e-6)
    assert values[0] < 0.0


def test_periodic_lattice_uses_trimmed_domain_bounds(tmp_path):
    source = trimesh.creation.icosphere(subdivisions=2, radius=0.5)
    path = _write_stl(tmp_path / "bounded-cell.stl", source)
    prepared = prepare_stl_unit_cell(path, geometry_backend=_AnalyticSphereBackend())
    cell_map = CellMap.from_points(
        np.array(((-4.0, -4.0, -4.0), (4.0, 4.0, 4.0))),
        (2.0, 2.0, 2.0),
        CellMapFrame.world((0.0, 0.0, 0.0)),
    )
    domain_bounds = np.array(((-2.0, -2.0, -2.0), (2.0, 2.0, 2.0)))
    body = make_periodic_lattice(
        prepared,
        cell_map,
        _sphere_domain(domain_bounds),
    )

    np.testing.assert_allclose(body.bounds, domain_bounds)


def test_periodic_lattice_uses_conservative_world_distance_scale(tmp_path):
    source = trimesh.creation.icosphere(subdivisions=2, radius=0.5)
    path = _write_stl(tmp_path / "scaled-cell.stl", source)
    prepared = prepare_stl_unit_cell(path, geometry_backend=_AnalyticSphereBackend())
    frame = CellMapFrame.world((0.0, 0.0, 0.0))
    cell_map = CellMap.from_points(
        np.array(((0.0, 0.0, 0.0), (2.0, 4.0, 6.0))),
        (2.0, 4.0, 6.0),
        frame,
    )
    body = make_periodic_lattice(
        prepared,
        cell_map,
        _constant_domain(cell_map.bounds),
    )
    source_fraction = np.array((0.9, 0.5, 0.5))
    world_point = source_fraction * np.asarray(cell_map.spacing_mm)
    source_point = prepared.domain.bounds_mm[0] + source_fraction * prepared.domain.extent_mm
    source_center = prepared.domain.bounds_mm.mean(axis=0)
    source_radius = float(np.min(prepared.domain.extent_mm) * 0.3)
    source_value = np.linalg.norm(source_point - source_center) - source_radius

    value = body.evaluate_points(world_point.reshape((1, 3)))[0]

    expected_scale = np.min(
        np.asarray(cell_map.spacing_mm) / prepared.domain.extent_mm
    )
    assert value == pytest.approx(source_value * expected_scale, abs=1.0e-6)


def test_periodic_seam_report_marks_empty_and_mismatched_faces_incompatible(tmp_path):
    source = trimesh.creation.icosphere(subdivisions=2, radius=0.5)
    path = _write_stl(tmp_path / "seam-report-cell.stl", source)
    prepared = prepare_stl_unit_cell(path, geometry_backend=_AnalyticSphereBackend())
    cell_map = CellMap.from_points(
        np.array(((-2.0, -2.0, -2.0), (2.0, 2.0, 2.0))),
        (2.0, 2.0, 2.0),
        CellMapFrame.world((0.0, 0.0, 0.0)),
    )

    report = prepared.seam_report(cell_map, samples_per_axis=12)

    assert report.samples_per_axis == 12
    assert {axis.direction for axis in report.axes} == {"U", "V", "W"}
    assert report.incompatible_directions == ("U", "V", "W")
    assert all(axis.lower_solid_samples == 0 for axis in report.axes)


def test_periodic_seam_report_uses_physical_cell_and_bridge_parameters(tmp_path):
    source = trimesh.creation.icosphere(subdivisions=2, radius=0.5)
    path = _write_stl(tmp_path / "physical-seam-report-cell.stl", source)
    prepared = prepare_stl_unit_cell(path, geometry_backend=_AnalyticSphereBackend())
    spacing = (2.0, 3.0, 4.0)
    cell_map = CellMap.from_points(
        np.array(((0.0, 0.0, 0.0), spacing)),
        spacing,
        CellMapFrame.world((0.0, 0.0, 0.0)),
    )
    native_feature = prepared.characteristic_feature_mm(cell_map)

    baseline = prepared.seam_report(cell_map, samples_per_axis=12)
    physical = prepared.seam_report(
        cell_map,
        samples_per_axis=12,
        target_feature_mm=native_feature + 1.0,
        bridge_directions=("V",),
        bridge_depth_mm=0.6,
    )

    baseline_by_axis = {axis.direction: axis for axis in baseline.axes}
    physical_by_axis = {axis.direction: axis for axis in physical.axes}
    assert tuple(axis.cell_spacing_mm for axis in physical.axes) == spacing
    assert physical_by_axis["U"].probe_depth_mm == pytest.approx(0.04)
    assert physical_by_axis["V"].probe_depth_mm == pytest.approx(0.6)
    assert physical_by_axis["W"].probe_depth_mm == pytest.approx(0.08)
    assert physical_by_axis["V"].bridge_enabled is True
    assert physical_by_axis["V"].effective_compatible is True
    assert physical_by_axis["U"].bridge_enabled is False
    assert physical_by_axis["U"].union_solid_samples > baseline_by_axis["U"].union_solid_samples
    expected_v_sample_area = spacing[0] * spacing[2] / (12 * 12)
    assert physical_by_axis["V"].union_solid_area_mm2 == pytest.approx(
        physical_by_axis["V"].union_solid_samples * expected_v_sample_area
    )


def test_periodic_seam_report_distinguishes_source_and_bridged_compatibility(tmp_path):
    source = trimesh.creation.icosphere(subdivisions=2, radius=0.5)
    path = _write_stl(tmp_path / "asymmetric-seam-cell.stl", source)
    prepared = prepare_stl_unit_cell(
        path,
        geometry_backend=_AnalyticOffsetSphereBackend(),
    )
    cell_map = CellMap.from_points(
        np.array(((0.0, 0.0, 0.0), (2.0, 2.0, 2.0))),
        (2.0, 2.0, 2.0),
        CellMapFrame.world((0.0, 0.0, 0.0)),
    )

    report = prepared.seam_report(
        cell_map,
        samples_per_axis=16,
        bridge_directions=("U",),
        bridge_depth_mm=0.5,
    )
    u_axis = next(axis for axis in report.axes if axis.direction == "U")

    assert u_axis.compatible is False
    assert u_axis.bridge_enabled is True
    assert u_axis.union_solid_samples > 0
    assert u_axis.effective_compatible is True


def test_periodic_bridge_adds_only_selected_direction_seam_material(tmp_path):
    source = trimesh.creation.icosphere(subdivisions=2, radius=0.5)
    path = _write_stl(tmp_path / "bridge-cell.stl", source)
    prepared = prepare_stl_unit_cell(path, geometry_backend=_AnalyticSphereBackend())
    cell_map = CellMap.from_points(
        np.array(((-2.0, -2.0, -2.0), (2.0, 2.0, 2.0))),
        (2.0, 2.0, 2.0),
        CellMapFrame.world((0.0, 0.0, 0.0)),
    )
    domain = _constant_domain(cell_map.bounds)
    baseline = make_periodic_lattice(prepared, cell_map, domain)
    bridged = make_periodic_lattice(
        prepared,
        cell_map,
        domain,
        bridge_directions=("U",),
        bridge_depth_mm=0.5,
    )
    points = np.array(
        (
            (0.0, 1.0, 1.0),
            (0.1, 1.0, 1.0),
            (1.0, 0.0, 1.0),
            (1.0, 1.0, 0.0),
        )
    )

    baseline_values = baseline.evaluate_points(points)
    bridged_values = bridged.evaluate_points(points)

    assert baseline_values[0] > 0.0
    assert bridged_values[0] < 0.0
    assert bridged_values[1] < 0.0
    np.testing.assert_allclose(bridged_values[2:], baseline_values[2:], atol=1.0e-6)


@pytest.mark.parametrize("depth", (0.0, -0.1, 1.0))
def test_periodic_bridge_rejects_invalid_depth(tmp_path, depth):
    source = trimesh.creation.icosphere(subdivisions=1, radius=0.5)
    path = _write_stl(tmp_path / "invalid-bridge-cell.stl", source)
    prepared = prepare_stl_unit_cell(path, geometry_backend=_AnalyticSphereBackend())
    cell_map = CellMap.from_points(
        np.array(((-2.0, -2.0, -2.0), (2.0, 2.0, 2.0))),
        (2.0, 2.0, 2.0),
        CellMapFrame.world((0.0, 0.0, 0.0)),
    )

    with pytest.raises(ValueError, match="bridge_depth_mm"):
        make_periodic_lattice(
            prepared,
            cell_map,
            _constant_domain(cell_map.bounds),
            bridge_directions=("U",),
            bridge_depth_mm=depth,
        )


def test_periodic_lattice_applies_target_feature_as_world_offset(tmp_path):
    source = trimesh.creation.icosphere(subdivisions=2, radius=0.5)
    path = _write_stl(tmp_path / "thickness-cell.stl", source)
    prepared = prepare_stl_unit_cell(path, geometry_backend=_AnalyticSphereBackend())
    frame = CellMapFrame.world((0.0, 0.0, 0.0))
    cell_map = CellMap.from_points(
        np.array(((0.0, 0.0, 0.0), (2.0, 2.0, 2.0))),
        (2.0, 2.0, 2.0),
        frame,
    )
    domain = _constant_domain(cell_map.bounds)
    native_feature = prepared.characteristic_feature_mm(cell_map)
    points = np.array(
        (
            (1.0, 1.0, 1.0),
            (1.7, 1.0, 1.0),
            (0.25, 1.0, 1.0),
        )
    )
    baseline = make_periodic_lattice(prepared, cell_map, domain)
    thicker = make_periodic_lattice(
        prepared,
        cell_map,
        domain,
        target_feature_mm=native_feature + 0.4,
    )
    thinner = make_periodic_lattice(
        prepared,
        cell_map,
        domain,
        target_feature_mm=native_feature - 0.2,
    )

    baseline_values = baseline.evaluate_points(points)

    np.testing.assert_allclose(
        thicker.evaluate_points(points),
        baseline_values - 0.2,
        atol=1.0e-6,
    )
    np.testing.assert_allclose(
        thinner.evaluate_points(points),
        baseline_values + 0.1,
        atol=1.0e-6,
    )


@pytest.mark.parametrize("target", (0.0, -0.1, np.nan, np.inf))
def test_periodic_lattice_rejects_invalid_target_feature(tmp_path, target):
    source = trimesh.creation.icosphere(subdivisions=1, radius=0.5)
    path = _write_stl(tmp_path / "invalid-thickness-cell.stl", source)
    prepared = prepare_stl_unit_cell(path, geometry_backend=_AnalyticSphereBackend())
    cell_map = CellMap.from_points(
        np.array(((0.0, 0.0, 0.0), (2.0, 2.0, 2.0))),
        (2.0, 2.0, 2.0),
        CellMapFrame.world((0.0, 0.0, 0.0)),
    )

    with pytest.raises(ValueError, match="target_feature_mm"):
        make_periodic_lattice(
            prepared,
            cell_map,
            _constant_domain(cell_map.bounds),
            target_feature_mm=target,
        )


def test_periodic_lattice_intersects_design_domain(tmp_path):
    source = trimesh.creation.icosphere(subdivisions=2, radius=0.5)
    path = _write_stl(tmp_path / "intersection-cell.stl", source)
    prepared = prepare_stl_unit_cell(path, geometry_backend=_AnalyticSphereBackend())
    frame = CellMapFrame.world((0.0, 0.0, 0.0))
    cell_map = CellMap.from_points(
        np.array(((0.0, 0.0, 0.0), (2.0, 2.0, 2.0))),
        (2.0, 2.0, 2.0),
        frame,
    )
    design_domain = ImplicitBody(
        name="half-domain",
        bounds=cell_map.bounds,
        evaluate=lambda points, _stage: np.asarray(points[:, 0] - 0.75, dtype=np.float32),
    )
    body = make_periodic_lattice(prepared, cell_map, design_domain)
    points = np.array(((0.5, 1.0, 1.0), (1.0, 1.0, 1.0)))

    values = body.evaluate_points(points)

    assert values[0] < 0.0
    assert values[1] > 0.0


def test_sampled_display_cleanup_removes_only_bounded_negative_islands():
    values = np.ones((28, 28, 28), dtype=np.float32)
    values[1:3, 1:3, 1:3] = -1.0
    values[5:11, 5, 5] = -0.001
    values[13:18, 13, 13] = -0.001
    values[20:27, 20, 20] = -0.75
    field = SampledImplicitField(
        "display",
        values,
        np.zeros(3),
        np.ones(3),
        (0.5, 0.5, 0.5, 1.0),
    )

    cleaned, report = remove_small_negative_islands(
        field,
        max_component_voxels=6,
        relative_volume_limit=1.0,
    )

    assert report.removed_components == 2
    assert report.removed_voxels == 11
    assert report.retained_components == 2
    assert report.removed_volume_mm3 <= report.volume_limit_mm3 * 2.0
    assert np.all(cleaned.values[5:11, 5, 5] > 0.0)
    assert np.all(cleaned.values[13:18, 13, 13] > 0.0)
    assert np.all(cleaned.values[1:3, 1:3, 1:3] < 0.0)
    assert np.all(cleaned.values[20:27, 20, 20] < 0.0)
    np.testing.assert_array_equal(field.values, values)
