"""Regression tests for the layer-contour inspection API."""

from __future__ import annotations

import numpy as np

from lattice_studio.engine.implicit.field import ImplicitBody, SampledImplicitField
from lattice_studio.engine.implicit.layer_contours import (
    LayerContourCache,
    available_layer_positions,
    resolve_layer_spacing,
    sample_implicit_body_layer,
    sample_sampled_field_layer,
)
from lattice_studio.presentation.qt.workbench import vtk_bounds_from_implicit_body
from lattice_studio.presentation.qt.viewers.pyvista_viewer import ContourData, build_contour_polydata


def _sphere_body() -> ImplicitBody:
    def evaluate(points: np.ndarray, _stage=None) -> np.ndarray:
        return (np.linalg.norm(points - np.array([2.0, 2.0, 2.0]), axis=1) - 1.2).astype(
            np.float32
        )

    return ImplicitBody(
        name="sphere",
        bounds=np.array([[0.0, 0.0, 0.0], [4.0, 4.0, 4.0]]),
        evaluate=evaluate,
    )


def test_available_layers_include_both_bounds() -> None:
    positions = available_layer_positions(
        np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]]),
        "Z",
        1.0,
    )
    np.testing.assert_allclose(positions, [0.0, 1.0, 2.0, 3.0])


def test_fixed_layer_height_changes_only_layer_axis() -> None:
    spacing = resolve_layer_spacing((0.2, 0.3, 0.4), "Z", 0.75)
    np.testing.assert_allclose(spacing, [0.2, 0.3, 0.75])


def test_fixed_layer_height_does_not_mutate_sampled_field_spacing() -> None:
    """Layer enumeration must not rescale the source field in the renderer."""

    source_spacing = np.array([0.2, 0.3, 0.4], dtype=np.float64)
    resolved = resolve_layer_spacing(source_spacing, "Z", 0.75)

    np.testing.assert_allclose(source_spacing, [0.2, 0.3, 0.4])
    np.testing.assert_allclose(resolved, [0.2, 0.3, 0.75])
    assert not np.shares_memory(resolved, source_spacing)


def test_default_layer_spacing_is_preserved() -> None:
    spacing = resolve_layer_spacing((0.2, 0.3, 0.4), "Y")
    np.testing.assert_allclose(spacing, [0.2, 0.3, 0.4])


def test_fixed_layer_positions_are_independent_of_field_spacing() -> None:
    bounds = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    positions = available_layer_positions(
        bounds,
        "Z",
        resolve_layer_spacing((0.2, 0.3, 0.4), "Z", 0.25),
    )
    np.testing.assert_allclose(positions, [0.0, 0.25, 0.5, 0.75, 1.0])


def test_authoritative_body_layer_maps_to_world_coordinates() -> None:
    result = sample_implicit_body_layer(
        _sphere_body(), "Z", 2.0, 0.25, level=0.0
    )
    assert result.axis == "Z"
    assert result.grid_shape == (17, 17)
    assert result.point_count > 10
    points = np.concatenate([item.points for item in result.contours])
    assert np.allclose(points[:, 2], 2.0)
    assert points[:, 0].min() >= 0.0
    assert points[:, 0].max() <= 4.0


def test_non_cubic_body_layers_remain_on_the_selected_world_plane() -> None:
    """A 100 x 100 x 20 body must never be stretched through the view axis."""

    def evaluate(points: np.ndarray, _stage=None) -> np.ndarray:
        normalized = (points - np.array([50.0, 50.0, 10.0])) / np.array(
            [30.0, 30.0, 6.0]
        )
        return (np.linalg.norm(normalized, axis=1) - 1.0).astype(np.float32)

    body = ImplicitBody(
        name="rectangular",
        bounds=np.array([[0.0, 0.0, 0.0], [100.0, 100.0, 20.0]]),
        evaluate=evaluate,
    )
    for axis, position in (("X", 50.0), ("Y", 50.0), ("Z", 10.0)):
        result = sample_implicit_body_layer(body, axis, position, 1.0)
        assert result.contours
        points = np.concatenate([item.points for item in result.contours])
        fixed_index = "XYZ".index(axis)
        assert np.allclose(points[:, fixed_index], position)
        assert np.all(points >= body.bounds[0] - 1.0e-6)
        assert np.all(points <= body.bounds[1] + 1.0e-6)


def test_sampled_field_layer_interpolates_between_planes() -> None:
    coordinates = np.arange(5, dtype=np.float64)
    xx, yy, zz = np.meshgrid(coordinates, coordinates, coordinates, indexing="ij")
    values = (xx + yy + zz - 5.0).astype(np.float32)
    field = SampledImplicitField(
        name="linear",
        values=values,
        origin=np.zeros(3),
        spacing=np.ones(3),
        color=(0.2, 0.3, 0.4, 1.0),
    )
    result = sample_sampled_field_layer(field, "Z", 2.5, level=0.0)
    assert result.grid_shape == (5, 5)
    assert result.point_count > 0
    points = np.concatenate([item.points for item in result.contours])
    assert np.allclose(points[:, 2], 2.5)


def test_layer_contour_cache_reuses_results() -> None:
    cache = LayerContourCache(max_entries=1)
    result = sample_implicit_body_layer(_sphere_body(), "X", 2.0, 0.5)
    cache.put(("sphere", "X", 2.0), result)
    assert cache.get(("sphere", "X", 2.0)) is result
    assert len(cache) == 1
    cache.put(("sphere", "Y", 2.0), result)
    assert cache.get(("sphere", "X", 2.0)) is None


def test_empty_level_returns_valid_empty_result() -> None:
    result = sample_implicit_body_layer(
        _sphere_body(), "Y", 2.0, 0.5, level=100.0
    )
    assert result.contours == ()
    assert result.point_count == 0


def test_contour_polydata_preserves_polyline_cells() -> None:
    polydata = build_contour_polydata(
        ContourData(
            polylines=[
                np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 1.0]]),
                np.array([[0.0, 1.0, 1.0], [1.0, 1.0, 1.0], [2.0, 1.0, 1.0]]),
            ]
        )
    )
    assert polydata.n_points == 5
    assert polydata.n_lines == 2
    assert polydata.n_cells == 2


def test_contour_scene_does_not_retain_other_display_sources() -> None:
    """Contour inspection must not include stale overlays or field planes."""

    calls: list[str] = []
    owner = type("ViewerHarness", (), {})()
    owner.meshes = [object()]
    owner.implicit_fields = [object()]
    owner.scene_overlays = [object()]
    owner.contours = []
    owner._cell_map_preview = object()
    owner._field_plane_data = object()
    owner._update_scene = lambda: calls.append("scene")
    owner._fix_camera_clipping = lambda: calls.append("clipping")
    owner.render = lambda: calls.append("render")

    from lattice_studio.presentation.qt.viewers.pyvista_viewer import PyVistaRenderer

    PyVistaRenderer.set_contours(
        owner,
        [ContourData(polylines=[np.array([[0.0, 0.0, 10.0], [1.0, 0.0, 10.0]])])],
        reset_view=False,
    )

    assert owner.meshes == []
    assert owner.implicit_fields == []
    assert owner.scene_overlays == []
    assert owner._cell_map_preview is None
    assert owner._field_plane_data is None
    assert len(owner.contours) == 1
    assert calls == ["scene", "clipping", "render"]


def test_implicit_body_bounds_convert_to_vtk_order() -> None:
    body = _sphere_body()
    np.testing.assert_allclose(
        vtk_bounds_from_implicit_body(body),
        [0.0, 4.0, 0.0, 4.0, 0.0, 4.0],
    )
