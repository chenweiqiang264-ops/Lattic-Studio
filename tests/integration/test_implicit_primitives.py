"""Regression tests for analytic field objects and field-driven Ramps."""

from __future__ import annotations

import numpy as np
import pytest

from lattice_studio.engine.implicit.field import ImplicitBody
from lattice_studio.engine.implicit.primitives import ImplicitPrimitive
from lattice_studio.engine.implicit.transition import (
    TransitionOperand,
    TransitionSpec,
    build_transition_body,
    transition_weights,
)


def test_sphere_sdf_uses_standard_negative_inside_convention() -> None:
    sphere = ImplicitPrimitive("sphere-1", "Sphere 1", "sphere", radius_mm=2.0)
    values = sphere.evaluate_points(
        np.array(((0.0, 0.0, 0.0), (2.0, 0.0, 0.0), (3.5, 0.0, 0.0)))
    )

    np.testing.assert_allclose(values, (-2.0, 0.0, 1.5), atol=1.0e-6)


def test_oriented_finite_cylinder_has_caps_and_radial_wall() -> None:
    cylinder = ImplicitPrimitive(
        "cylinder-1",
        "Cylinder 1",
        "cylinder",
        rotation_euler_deg=(0.0, 90.0, 0.0),
        radius_mm=1.0,
        height_mm=4.0,
    )
    values = cylinder.evaluate_points(
        np.array(((0.0, 0.0, 0.0), (2.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.5)))
    )

    assert values[0] == pytest.approx(-1.0)
    assert values[1] == pytest.approx(0.0)
    assert values[2] == pytest.approx(0.0)
    assert values[3] == pytest.approx(0.5)
    bounds = cylinder.bounds
    assert np.all(bounds[0] <= (-2.0, -1.0, -1.0))
    assert np.all(bounds[1] >= (2.0, 1.0, 1.0))


def test_oriented_box_sdf_and_bounds_are_transform_aware() -> None:
    box = ImplicitPrimitive(
        "box-1",
        "Box 1",
        "box",
        center_mm=(4.0, -1.0, 2.0),
        rotation_euler_deg=(0.0, 0.0, 45.0),
        size_mm=(2.0, 4.0, 6.0),
    )

    assert box.evaluate_points(np.array(((4.0, -1.0, 2.0),)))[0] == pytest.approx(-1.0)
    corners = np.array(
        [
            (x, y, z)
            for x in (-1.0, 1.0)
            for y in (-2.0, 2.0)
            for z in (-3.0, 3.0)
        ]
    )
    world_corners = corners @ box.rotation_matrix.T + np.asarray(box.center_mm)
    assert np.all(world_corners >= box.bounds[0] - 1.0e-8)
    assert np.all(world_corners <= box.bounds[1] + 1.0e-8)


def test_primitive_preview_mesh_uses_a_smooth_default_resolution() -> None:
    sphere = ImplicitPrimitive("sphere-preview", "Sphere preview", "sphere")
    vertices, faces = sphere.preview_mesh()

    assert len(vertices) >= 4_000
    assert len(faces) >= 7_000


def test_field_interval_maps_to_exact_ramp_endpoints() -> None:
    first = ImplicitBody(
        "first",
        np.array(((-5.0, -5.0, -5.0), (5.0, 5.0, 5.0))),
        lambda points, _reporter: np.full(len(points), -1.0, dtype=np.float32),
    )
    second = ImplicitBody(
        "second",
        np.array(((-5.0, -5.0, -5.0), (5.0, 5.0, 5.0))),
        lambda points, _reporter: np.full(len(points), 1.0, dtype=np.float32),
    )
    domain = ImplicitBody(
        "domain",
        np.array(((-5.0, -5.0, -5.0), (5.0, 5.0, 5.0))),
        lambda points, _reporter: np.full(len(points), -100.0, dtype=np.float32),
    )
    driver = ImplicitPrimitive("sphere-2", "Sphere driver", "sphere", radius_mm=2.0).as_implicit_body()
    spec = TransitionSpec(
        width_mm=4.0,
        center_offset_mm=0.0,
        automatic_registration=True,
        topology_correction=False,
    )
    operand_a = TransitionOperand("A", first, 0.5, (3.0, 3.0, 3.0))
    operand_b = TransitionOperand("B", second, 0.5, (3.0, 3.0, 3.0))
    result = build_transition_body(operand_a, operand_b, domain, driver=driver, spec=spec)
    # Values -2 and +2 are exactly the requested [-2, 2] mm interval limits.
    points = np.array(((0.0, 0.0, 0.0), (2.0, 0.0, 0.0), (4.0, 0.0, 0.0)))
    np.testing.assert_allclose(result.body.evaluate_points(points), (-1.0, 0.0, 1.0))
    first_weight, second_weight = transition_weights(np.array((-2.0, 2.0)), spec)
    np.testing.assert_allclose(first_weight, (1.0, 0.0))
    np.testing.assert_allclose(second_weight, (0.0, 1.0))
    assert result.diagnostics.field_driven is True
    assert result.diagnostics.registration_translation_mm == (0.0, 0.0, 0.0)
    assert result.diagnostics.quality_validation_supported is False


def test_workbench_generates_transition_from_an_independent_primitive_field() -> None:
    """The workbench path must consume the selected primitive, not a plane."""

    import trimesh

    from lattice_studio.presentation.qt import workbench as app

    domain = trimesh.creation.box(extents=(8.0, 8.0, 8.0))
    sphere = ImplicitPrimitive(
        "transition-sphere",
        "Transition sphere",
        "sphere",
        radius_mm=2.0,
    )
    result = app.generate_implicit_lattice_transition(
        domain,
        app.TPMSParameters("G", cell_size_mm=4.0, wall_thickness_mm=0.8),
        app.TPMSParameters("D", cell_size_mm=4.0, wall_thickness_mm=0.8),
        app.TransitionParameters(
            driver_mode="field",
            driver_identifier=sphere.identifier,
            field_interval_lower_mm=-1.0,
            field_interval_upper_mm=1.0,
            automatic_registration=True,
            topology_correction=False,
        ),
        app.SamplingParameters(target_voxels=100_000),
        display_voxel_size_mm=1.0,
        display_memory_budget_mb=32.0,
        transition_driver=sphere.as_implicit_body(),
    )

    metadata = result.metadata
    assert isinstance(metadata, app.TransitionGenerationMetadata)
    assert metadata.diagnostics.field_driven is True
    assert metadata.diagnostics.driver_name == sphere.name
    assert metadata.diagnostics.quality_validation_supported is False
    assert metadata.quality.inspection_supported is False
    assert result.display_field.values.min() < 0.0 < result.display_field.values.max()


def test_field_transition_uses_its_interval_not_the_legacy_plane_width() -> None:
    from lattice_studio.presentation.qt import workbench as app

    parameters = app.TransitionParameters(
        transition_width_mm=0.0,
        driver_mode="field",
        driver_identifier="sphere-1",
        field_interval_lower_mm=-2.0,
        field_interval_upper_mm=3.0,
    )

    parameters.validate()
    spec = parameters.provider_spec()
    assert spec.width_mm == pytest.approx(5.0)
    assert spec.center_offset_mm == pytest.approx(0.5)
