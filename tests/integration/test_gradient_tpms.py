"""Regression tests for the MATLAB-compatible gradient TPMS path."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import trimesh

TESTS_DIR = Path(__file__).resolve().parent
if str(TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(TESTS_DIR))

from lattice_studio.presentation.qt import workbench as app
from lattice_studio.engine.implicit.cell_map import CellMap, CellMapFrame
from lattice_studio.engine.implicit.gradient_lattice import (
    GradientControls,
    estimate_layer_density,
    evaluate_gradient_tpms_field,
    gradient_profile,
    matlab_gradient_voxel_size,
)
from lattice_studio.engine.implicit.tpms_compute import NumpyTPMSAdapter, TPMSFieldSpec


@pytest.mark.parametrize("kind", ["G", "D", "IWP", "Primitive", "Neovius"])
def test_all_supported_tpms_kinds_return_finite_fields(kind: str) -> None:
    points = np.random.default_rng(12).normal(size=(257, 3)).astype(np.float32)
    spec = TPMSFieldSpec(kind, cell_size_mm=6.0, wall_thickness_mm=0.5)
    values = NumpyTPMSAdapter().evaluate_shell(points, spec, max_tile_points=31)

    assert values.shape == (257,)
    assert np.isfinite(values).all()


@pytest.mark.parametrize("mode", ["linear", "power", "sigmoid", "layered"])
def test_gradient_profile_is_bounded_and_monotone(mode: str) -> None:
    coordinate = np.linspace(0.0, 1.0, 101, dtype=np.float32)
    values = gradient_profile(coordinate, mode, power=5.0, layers=5)

    assert values[0] >= 0.0
    assert values[-1] <= 1.0
    assert np.all(np.diff(values) >= -1.0e-6)


def test_gradient_threshold_method_matches_matlab_zero_surface_for_cubic_cell() -> None:
    frame = CellMapFrame.world((0.0, 0.0, 0.0))
    field_spec = TPMSFieldSpec("IWP", 10.0, 0.8)
    controls = GradientControls(
        enabled=True,
        axis="W",
        mode="linear",
        thickness_soft_mm=0.8,
        thickness_stiff_mm=0.8,
        offset_soft=0.0,
        offset_stiff=0.0,
        wall_thickness_method="field_threshold",
    )
    points = np.array(
        ((1.0, 2.0, 0.0), (1.0, 2.0, 5.0), (1.0, 2.0, 10.0)),
        dtype=np.float32,
    )
    values = evaluate_gradient_tpms_field(
        points,
        field_spec,
        frame,
        controls,
        (0.0, 10.0),
        max_tile_points=2,
    )
    phase = points * (2.0 * np.pi / 10.0)
    field, _ = __import__(
        "lattice_studio.engine.implicit.tpms_compute",
        fromlist=["evaluate_tpms_field_and_gradient"],
    ).evaluate_tpms_field_and_gradient(phase, "IWP")
    expected = np.abs(field) - 0.8 * np.pi / 10.0
    np.testing.assert_allclose(values, expected, rtol=2.0e-5, atol=2.0e-5)


def test_gradient_projection_uses_design_domain_not_cell_map_padding() -> None:
    mesh = trimesh.creation.box(extents=(4.0, 4.0, 4.0))
    parameters = app.TPMSParameters(
        "G",
        cell_size_mm=2.0,
        wall_thickness_mm=0.2,
        gradient_enabled=True,
        gradient_axis="W",
        thickness_soft_mm=0.15,
        thickness_stiff_mm=0.3,
    )
    generation = app.generate_implicit_lattice(
        mesh,
        parameters,
        app.SamplingParameters(target_voxels=100_000, use_cpp_sdf=False),
        voxel_size_mm=0.5,
        display_voxel_size_mm=1.0,
        display_memory_budget_mb=64.0,
    )
    assert generation.body.evaluate_points(
        np.array(((0.0, 0.0, 0.0),), dtype=np.float32)
    ).shape == (1,)
    assert generation.minimum_feature_mm == pytest.approx(0.15)


def test_matlab_def_resolution_matches_half_cell_definition() -> None:
    assert matlab_gradient_voxel_size(10.0, 28) == pytest.approx(10.0 / 56.0)
    assert matlab_gradient_voxel_size((10.0, 20.0, 30.0), 10) == pytest.approx(0.5)


def test_matlab_def_strategy_controls_automatic_tpms_recommendation() -> None:
    domain = trimesh.creation.box(extents=(20.0, 20.0, 20.0))
    parameters = app.TPMSParameters(
        "G",
        cell_size_mm=(10.0, 20.0, 30.0),
        wall_thickness_mm=0.4,
        gradient_enabled=True,
        gradient_resolution_strategy="matlab_def",
        gradient_def_per_half_cell=10,
    )
    sampling = app.SamplingParameters(target_voxels=100_000)

    recommendation = app.recommend_voxel_size(domain, parameters, sampling)

    assert recommendation.voxel_size_mm == pytest.approx(0.5)
    assert recommendation.samples_per_cell == pytest.approx(20.0)
    assert recommendation.limiting_constraint == "MATLAB Def 分辨率"
    assert recommendation.estimated_voxels == 43 * 43 * 43


def test_matlab_def_strategy_is_inactive_without_gradient() -> None:
    domain = trimesh.creation.box(extents=(20.0, 20.0, 20.0))
    parameters = app.TPMSParameters(
        "G",
        cell_size_mm=10.0,
        wall_thickness_mm=0.4,
        gradient_enabled=False,
        gradient_resolution_strategy="matlab_def",
        gradient_def_per_half_cell=10,
    )
    recommendation = app.recommend_voxel_size(
        domain,
        parameters,
        app.SamplingParameters(target_voxels=100_000),
    )

    assert recommendation.limiting_constraint != "MATLAB Def 分辨率"


def test_explicit_sampling_spacing_overrides_matlab_def_strategy() -> None:
    domain = trimesh.creation.box(extents=(20.0, 20.0, 20.0))
    parameters = app.TPMSParameters(
        "G",
        cell_size_mm=10.0,
        wall_thickness_mm=0.4,
        gradient_enabled=True,
        gradient_resolution_strategy="matlab_def",
        gradient_def_per_half_cell=10,
    )
    recommendation = app.recommendation_for_voxel_size(
        domain,
        parameters,
        app.SamplingParameters(target_voxels=100_000),
        0.25,
    )

    assert recommendation.voxel_size_mm == pytest.approx(0.25)
    assert recommendation.limiting_constraint == "用户指定基础体素"


def test_manual_cell_counts_are_relative_to_frame_origin() -> None:
    frame = CellMapFrame.from_origin_axes((3.0, 4.0, 5.0), (0, 1, 0), (0, 0, 1))
    cell_map = CellMap.from_frame_counts(frame, (2.0, 3.0, 4.0), (2, 3, 4))

    assert cell_map.cell_counts == (2, 3, 4)
    np.testing.assert_allclose(cell_map.frame.origin, (3.0, 4.0, 5.0))
    assert cell_map.extent_mm.tolist() == [4.0, 9.0, 16.0]


def test_layer_density_reports_requested_number_of_layers() -> None:
    axis = np.linspace(0.0, 1.0, 100, dtype=np.float32)
    values = axis.reshape(10, 10)
    centers, density = estimate_layer_density(values, np.broadcast_to(axis.reshape(10, 10), values.shape), 5)

    assert centers.shape == (5,)
    assert density.shape == (5,)
    assert np.isfinite(density).all()
