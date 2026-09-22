"""Behavior tests for provider-agnostic topology-constrained transitions."""

from __future__ import annotations

import numpy as np
import pytest

from lattice_studio.engine.implicit.field import ImplicitBody
from lattice_studio.engine.implicit.transition import (
    TransitionOperand,
    TransitionSpec,
    build_transition_body,
    transition_weights,
    validate_transition,
)


def _body(name: str, evaluator, bounds=None) -> ImplicitBody:
    return ImplicitBody(
        name=name,
        bounds=np.asarray(
            bounds if bounds is not None else ((-4.0, -4.0, -4.0), (4.0, 4.0, 4.0)),
            dtype=np.float64,
        ),
        evaluate=lambda points, _reporter: evaluator(np.asarray(points, dtype=np.float32)),
    )


@pytest.mark.parametrize(
    "kind",
    [
        "automatic",
        "linear",
        "smoothstep",
        "smootherstep",
        "sigmoid",
        "cosine",
    ],
)
def test_transition_weights_have_exact_compact_support(kind: str) -> None:
    spec = TransitionSpec(width_mm=4.0, weight_kind=kind, sigmoid_sharpness=1.6)
    distances = np.array((-20.0, -2.0, 0.0, 2.0, 20.0), dtype=np.float32)

    first, second = transition_weights(distances, spec)

    np.testing.assert_array_equal(first[[0, 1]], np.ones(2, dtype=np.float32))
    np.testing.assert_array_equal(second[[0, 1]], np.zeros(2, dtype=np.float32))
    assert second[2] == pytest.approx(0.5)
    np.testing.assert_array_equal(first[[3, 4]], np.zeros(2, dtype=np.float32))
    np.testing.assert_array_equal(second[[3, 4]], np.ones(2, dtype=np.float32))
    assert np.all(np.diff(second) >= 0.0)


def test_transition_body_preserves_each_operand_exactly_outside_the_band() -> None:
    first_body = _body("first", lambda points: points[:, 1] - 0.25)
    second_body = _body("second", lambda points: -points[:, 2] + 0.75)
    domain = _body("domain", lambda points: np.full(len(points), -100.0, np.float32))
    first = TransitionOperand(
        name="analytic-a",
        body=first_body,
        characteristic_feature_mm=0.8,
        cell_spacing_mm=(6.0, 7.0, 8.0),
    )
    second = TransitionOperand(
        name="mesh-backed-b",
        body=second_body,
        characteristic_feature_mm=1.0,
        cell_spacing_mm=(9.0, 10.0, 11.0),
    )
    spec = TransitionSpec(
        width_mm=2.0,
        weight_kind="smootherstep",
        topology_correction=True,
        automatic_registration=False,
    )
    result = build_transition_body(
        first,
        second,
        domain,
        plane_point_mm=(0.0, 0.0, 0.0),
        plane_normal=(1.0, 0.0, 0.0),
        spec=spec,
    )
    points = np.array(
        ((-3.0, 0.1, 0.2), (-1.0, -0.4, 0.8), (1.0, 0.6, -0.2), (3.0, 0.3, 0.4)),
        dtype=np.float32,
    )

    values = result.body.evaluate_points(points)

    np.testing.assert_array_equal(values[:2], first_body.evaluate_points(points[:2]))
    np.testing.assert_array_equal(values[2:], second_body.evaluate_points(points[2:]))
    assert result.diagnostics.minimum_feature_mm == pytest.approx(0.8)
    assert result.diagnostics.resolved_weight_kind == "smootherstep"


def test_transition_quality_validation_detects_a_thick_cross_band_connection() -> None:
    cylinder = _body(
        "crossing-cylinder",
        lambda points: np.sqrt(points[:, 1] ** 2 + points[:, 2] ** 2) - 1.0,
        bounds=((-2.0, -1.5, -1.5), (2.0, 1.5, 1.5)),
    )
    spec = TransitionSpec(width_mm=2.0, minimum_feature_mm=0.8)

    report = validate_transition(
        cylinder,
        plane_point_mm=(0.0, 0.0, 0.0),
        plane_normal=(1.0, 0.0, 0.0),
        spec=spec,
        inspection_spacing_mm=0.2,
        max_samples=250_000,
    )

    assert report.cross_band_connected is True
    assert report.minimum_feature_satisfied is True
    assert report.isolated_fragment_count == 0
    assert report.passed is True


def test_transition_quality_validation_rejects_a_central_gap() -> None:
    disconnected = _body(
        "disconnected-cylinder",
        lambda points: np.maximum(
            np.sqrt(points[:, 1] ** 2 + points[:, 2] ** 2) - 1.0,
            0.25 - np.abs(points[:, 0]),
        ),
        bounds=((-2.0, -1.5, -1.5), (2.0, 1.5, 1.5)),
    )

    report = validate_transition(
        disconnected,
        plane_point_mm=(0.0, 0.0, 0.0),
        plane_normal=(1.0, 0.0, 0.0),
        spec=TransitionSpec(width_mm=2.0, minimum_feature_mm=0.8),
        inspection_spacing_mm=0.2,
        max_samples=250_000,
    )

    assert report.cross_band_connected is False
    assert report.issue_locations_mm
    assert report.passed is False


def test_transition_quality_validation_counts_an_isolated_band_fragment() -> None:
    def crossing_with_fragment(points: np.ndarray) -> np.ndarray:
        crossing = np.sqrt(points[:, 1] ** 2 + points[:, 2] ** 2) - 0.8
        fragment = np.linalg.norm(points - np.array((0.0, 2.0, 0.0)), axis=1) - 0.35
        return np.minimum(crossing, fragment)

    body = _body(
        "crossing-with-fragment",
        crossing_with_fragment,
        bounds=((-2.0, -3.0, -1.5), (2.0, 3.0, 1.5)),
    )

    report = validate_transition(
        body,
        plane_point_mm=(0.0, 0.0, 0.0),
        plane_normal=(1.0, 0.0, 0.0),
        spec=TransitionSpec(width_mm=2.0, minimum_feature_mm=0.6),
        inspection_spacing_mm=0.15,
        max_samples=300_000,
    )

    assert report.cross_band_connected is True
    assert report.isolated_fragment_count >= 1
    assert report.issue_locations_mm
    assert report.passed is False


def test_workbench_generates_same_provider_type_with_independent_parameters() -> None:
    from lattice_studio.presentation.qt import workbench as app

    design_domain = __import__("trimesh").creation.box(extents=(6.0, 6.0, 6.0))
    first = app.TPMSParameters(
        "G",
        cell_size_mm=(3.0, 3.5, 4.0),
        wall_thickness_mm=0.7,
    )
    second = app.TPMSParameters(
        "G",
        cell_size_mm=(4.5, 4.0, 3.5),
        wall_thickness_mm=0.9,
    )

    result = app.generate_implicit_lattice_transition(
        design_domain,
        first,
        second,
        app.TransitionParameters(
            plane_axis="X",
            transition_width_mm=3.0,
            weight_kind="smootherstep",
        ),
        app.SamplingParameters(target_voxels=100_000),
        display_voxel_size_mm=0.75,
        display_memory_budget_mb=64.0,
    )

    assert isinstance(result.metadata, app.TransitionGenerationMetadata)
    assert result.metadata.first_name == "G"
    assert result.metadata.second_name == "G"
    assert result.metadata.diagnostics.resolved_weight_kind == "smootherstep"
    assert result.display_field.values.min() < 0.0
    assert result.display_field.values.max() > 0.0


def test_workbench_generates_custom_to_analytic_transition(tmp_path) -> None:
    import trimesh

    from lattice_studio.engine.implicit.custom_unit_cell import prepare_stl_unit_cell
    from lattice_studio.presentation.qt import workbench as app

    source_path = tmp_path / "custom-cell.stl"
    trimesh.creation.icosphere(subdivisions=1, radius=0.5).export(source_path)
    prepared = prepare_stl_unit_cell(source_path)
    design_domain = trimesh.creation.box(extents=(6.0, 6.0, 6.0))
    custom = app.CustomUnitCellParameters(
        source_path=source_path,
        cell_size_mm=(3.0, 3.0, 3.0),
        target_feature_mm=0.8,
    )
    gyroid = app.TPMSParameters(
        "G",
        cell_size_mm=(3.5, 4.0, 4.5),
        wall_thickness_mm=0.75,
    )

    result = app.generate_implicit_lattice_transition(
        design_domain,
        custom,
        gyroid,
        app.TransitionParameters(
            plane_axis="X",
            transition_width_mm=3.5,
            weight_kind="automatic",
        ),
        app.SamplingParameters(target_voxels=100_000),
        display_voxel_size_mm=0.9,
        display_memory_budget_mb=64.0,
        prepared_custom_cell=prepared,
    )

    assert isinstance(result.metadata, app.TransitionGenerationMetadata)
    assert (result.metadata.first_name, result.metadata.second_name) == ("Custom", "G")
    assert result.metadata.diagnostics.minimum_feature_mm == pytest.approx(0.75)
    assert result.display_field.values.min() < 0.0
