"""Unit tests for implicit shelling and shell/lattice fusion."""

from __future__ import annotations

import numpy as np
import pytest
import trimesh

from lattice_studio.engine.implicit.field import ImplicitBody, SampledImplicitField
from lattice_studio.engine.implicit.shell import (
    ShellParameters,
    fuse_shell_lattice_fields,
    make_coordinated_shell_lattice_union,
    make_interior_shell,
    make_shell_lattice_union,
    smooth_union_values,
)
from lattice_studio.presentation.qt import workbench
from lattice_studio.presentation.qt.workbench import (
    SamplingParameters,
    design_domain_extraction_map,
    shell_sampling_recommendation,
)


def _sphere(name: str, radius: float) -> ImplicitBody:
    return ImplicitBody(
        name=name,
        bounds=np.array([[-3.0, -3.0, -3.0], [3.0, 3.0, 3.0]]),
        evaluate=lambda points, _reporter=None: (
            np.linalg.norm(np.asarray(points), axis=1) - radius
        ).astype(np.float32),
    )


def test_interior_shell_keeps_only_the_requested_boundary_band() -> None:
    shell = make_interior_shell(_sphere("domain", 2.0), 0.5)
    points = np.array(
        [[2.0, 0.0, 0.0], [1.75, 0.0, 0.0], [1.50, 0.0, 0.0], [1.0, 0.0, 0.0]]
    )
    values = shell.evaluate_points(points)
    assert values[0] == pytest.approx(0.0, abs=1.0e-6)
    assert values[1] < 0.0
    assert values[2] == pytest.approx(0.0, abs=1.0e-6)
    assert values[3] > 0.0


def test_hard_union_is_the_pointwise_minimum() -> None:
    first = np.array([-1.0, 0.2, 1.0], dtype=np.float32)
    second = np.array([0.5, -0.4, 1.5], dtype=np.float32)
    np.testing.assert_allclose(
        smooth_union_values(first, second, 0.0),
        np.minimum(first, second),
    )


def test_shell_lattice_union_cannot_expand_outside_domain() -> None:
    domain = _sphere("domain", 2.0)
    shell = make_interior_shell(domain, 0.4)
    lattice = _sphere("lattice", 2.2)
    fused = make_shell_lattice_union(shell, lattice, domain, 0.75)
    values = fused.evaluate_points(np.array([[2.1, 0.0, 0.0], [1.9, 0.0, 0.0]]))
    assert values[0] > 0.0
    assert values[1] <= 0.0


def test_coordinated_shell_union_matches_nested_evaluator_formula() -> None:
    domain = _sphere("domain", 2.0)
    feature = _sphere("feature", 1.6)

    def evaluate_clipped(points: np.ndarray, stage=None) -> np.ndarray:
        return np.maximum(
            domain.evaluate_points(points, stage),
            feature.evaluate_points(points, stage),
        )

    clipped = ImplicitBody("lattice", domain.bounds, evaluate_clipped)
    legacy = make_shell_lattice_union(
        make_interior_shell(domain, 0.5),
        clipped,
        domain,
        0.2,
    )
    coordinated = make_coordinated_shell_lattice_union(
        domain,
        feature,
        0.5,
        0.2,
    )
    axis = np.linspace(-2.5, 2.5, 13)
    points = np.column_stack(
        tuple(values.ravel() for values in np.meshgrid(axis, axis, axis, indexing="ij"))
    )

    np.testing.assert_allclose(
        coordinated.evaluate_points(points),
        legacy.evaluate_points(points),
        rtol=0.0,
        atol=0.0,
    )


def test_shell_parameters_reject_invalid_physical_values() -> None:
    with pytest.raises(ValueError):
        ShellParameters(0.0).validate()
    with pytest.raises(ValueError):
        ShellParameters(1.0, -0.1).validate()


def test_shell_sampling_uses_shell_thickness_as_a_physical_constraint() -> None:
    mesh = trimesh.creation.icosphere(subdivisions=1, radius=5.0)
    recommendation = shell_sampling_recommendation(
        mesh,
        0.6,
        SamplingParameters(target_voxels=100_000),
    )
    assert recommendation.voxel_size_mm <= 0.3
    assert recommendation.samples_per_wall >= 2.0
    extraction_map = design_domain_extraction_map(mesh)
    np.testing.assert_allclose(extraction_map.bounds, mesh.bounds)


def test_shell_and_lattice_union_enter_the_standard_implicit_result_pipeline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mesh = trimesh.creation.icosphere(subdivisions=1, radius=2.0)
    domain = _sphere("domain", 2.0)
    monkeypatch.setattr(
        workbench,
        "_make_design_domain_body",
        lambda _mesh, _sampling: domain,
    )
    sampling = SamplingParameters(target_voxels=100_000, use_cpp_sdf=False)
    shell = workbench.generate_implicit_shell(
        mesh,
        0.5,
        sampling,
        display_voxel_size_mm=1.0,
    )
    lattice = workbench.ImplicitGenerationResult(
        body=_sphere("lattice", 1.6),
        display_field=shell.display_field,
        recommendation=shell.recommendation,
        cell_map=shell.cell_map,
        minimum_feature_mm=0.4,
    )
    fused = workbench.generate_implicit_shell_lattice_union(
        mesh,
        lattice,
        0.5,
        0.2,
        sampling,
        display_voxel_size_mm=1.0,
    )
    assert shell.minimum_feature_mm == pytest.approx(0.5)
    assert fused.minimum_feature_mm == pytest.approx(0.4)
    assert fused.metadata["fusion_radius_mm"] == pytest.approx(0.2)
    assert fused.display_field.values.ndim == 3


def test_cached_shell_fusion_matches_the_sampled_field_formula() -> None:
    domain_field = SampledImplicitField(
        name="domain",
        values=np.array(
            [[[-0.8, -0.2], [0.1, 0.4]], [[-0.5, -0.1], [0.2, 0.7]]],
            dtype=np.float32,
        ),
        origin=np.zeros(3),
        spacing=np.ones(3),
        color=(1.0, 1.0, 1.0, 1.0),
    )
    lattice_field = SampledImplicitField(
        name="G",
        values=np.array(
            [[[-0.6, 0.3], [-0.4, 0.2]], [[-0.3, 0.4], [0.5, 0.8]]],
            dtype=np.float32,
        ),
        origin=np.zeros(3),
        spacing=np.ones(3),
        color=(0.7, 0.7, 0.7, 1.0),
    )

    fused = fuse_shell_lattice_fields(
        domain_field,
        lattice_field,
        thickness_mm=0.5,
        fusion_radius_mm=0.2,
        name="ShellUnion",
        color=(0.12, 0.58, 0.48, 1.0),
    )

    shell_values = np.maximum(
        domain_field.values,
        -domain_field.values - np.float32(0.5),
    )
    expected = np.maximum(
        domain_field.values,
        smooth_union_values(shell_values, lattice_field.values, 0.2),
    )
    np.testing.assert_allclose(fused.values, expected)
    assert fused.name == "ShellUnion"


def test_shell_union_reuses_matching_display_fields_without_evaluation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mesh = trimesh.creation.icosphere(subdivisions=1, radius=2.0)
    calls = {"domain": 0, "lattice": 0}

    def evaluate_domain(points: np.ndarray, _stage=None) -> np.ndarray:
        calls["domain"] += 1
        return (np.linalg.norm(points, axis=1) - 2.0).astype(np.float32)

    def evaluate_lattice(points: np.ndarray, _stage=None) -> np.ndarray:
        calls["lattice"] += 1
        return (np.linalg.norm(points, axis=1) - 1.5).astype(np.float32)

    domain = ImplicitBody(
        name="domain",
        bounds=mesh.bounds,
        evaluate=evaluate_domain,
    )
    lattice_body = ImplicitBody(
        name="G",
        bounds=mesh.bounds,
        evaluate=evaluate_lattice,
    )
    values = np.ones((3, 3, 3), dtype=np.float32)
    domain_field = SampledImplicitField(
        "domain", values, np.full(3, -1.0), np.ones(3), (1.0, 1.0, 1.0, 1.0)
    )
    lattice_field = SampledImplicitField(
        "G", values, np.full(3, -1.0), np.ones(3), (0.7, 0.7, 0.7, 1.0)
    )
    monkeypatch.setattr(
        workbench,
        "_make_design_domain_body",
        lambda _mesh, _sampling: domain,
    )
    lattice = workbench.ImplicitGenerationResult(
        body=lattice_body,
        display_field=lattice_field,
        recommendation=workbench.SamplingRecommendation(
            1.0,
            (3, 3, 3),
            27,
            2.0,
            2.0,
        ),
        cell_map=design_domain_extraction_map(mesh),
        minimum_feature_mm=0.5,
    )

    fused = workbench.generate_implicit_shell_lattice_union(
        mesh,
        lattice,
        0.5,
        0.0,
        SamplingParameters(target_voxels=100_000, use_cpp_sdf=False),
        display_voxel_size_mm=1.0,
        domain_display_field=domain_field,
    )

    assert calls == {"domain": 0, "lattice": 0}
    assert fused.metadata["display_fusion_backend"] == "cached-fields-cpu"


def test_shell_union_full_resample_evaluates_domain_once_per_point_batch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mesh = trimesh.creation.icosphere(subdivisions=1, radius=2.0)
    calls = {"domain": 0, "feature": 0, "clipped": 0}

    def evaluate_domain(points: np.ndarray, _stage=None) -> np.ndarray:
        calls["domain"] += 1
        return (np.linalg.norm(points, axis=1) - 2.0).astype(np.float32)

    def evaluate_feature(points: np.ndarray, _stage=None) -> np.ndarray:
        calls["feature"] += 1
        return (np.linalg.norm(points, axis=1) - 1.5).astype(np.float32)

    domain = ImplicitBody("domain", mesh.bounds, evaluate_domain)
    feature = ImplicitBody("feature", mesh.bounds, evaluate_feature)

    def evaluate_clipped(points: np.ndarray, stage=None) -> np.ndarray:
        calls["clipped"] += 1
        return np.maximum(
            domain.evaluate_points(points, stage),
            feature.evaluate_points(points, stage),
        )

    clipped = ImplicitBody("G", mesh.bounds, evaluate_clipped)
    coarse_field = SampledImplicitField(
        "G",
        np.ones((3, 3, 3), dtype=np.float32),
        np.full(3, -1.0),
        np.ones(3),
        (0.7, 0.7, 0.7, 1.0),
    )
    monkeypatch.setattr(
        workbench,
        "_make_design_domain_body",
        lambda _mesh, _sampling: domain,
    )
    lattice = workbench.ImplicitGenerationResult(
        body=clipped,
        display_field=coarse_field,
        recommendation=workbench.SamplingRecommendation(
            1.0,
            (3, 3, 3),
            27,
            2.0,
            2.0,
        ),
        cell_map=design_domain_extraction_map(mesh),
        minimum_feature_mm=0.5,
        unclipped_body=feature,
    )

    fused = workbench.generate_implicit_shell_lattice_union(
        mesh,
        lattice,
        0.5,
        0.2,
        SamplingParameters(target_voxels=100_000, use_cpp_sdf=False),
        display_voxel_size_mm=0.25,
    )

    assert calls["domain"] > 0
    assert calls["domain"] == calls["feature"]
    assert calls["clipped"] == 0
    assert fused.metadata["display_fusion_backend"] == "coordinated-resample"
