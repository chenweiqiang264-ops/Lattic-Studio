"""Focused regression tests for the on-demand field-plane sampler."""

from __future__ import annotations

import numpy as np

from lattice_studio.engine.implicit.field import ImplicitBody, SampledImplicitField
from lattice_studio.engine.implicit.field_viewer import (
    FieldPlaneCache,
    FieldPlaneState,
    expand_implicit_body_bounds,
    extract_field_plane_isolines,
    field_surface_hit_from_ray,
    orthonormal_plane_axes,
    recommended_plane_resolution,
    resolve_field_value_range,
    sample_field_plane,
)
from lattice_studio.presentation.qt.viewers.pyvista_viewer import (
    FieldPlaneRenderData,
    PyVistaRenderer,
    build_field_plane_polydata,
)


def _sphere_body() -> ImplicitBody:
    return ImplicitBody(
        name="sphere",
        bounds=np.array([[-2.0, -2.0, -2.0], [2.0, 2.0, 2.0]]),
        evaluate=lambda points, _stage=None: (
            np.linalg.norm(points, axis=1) - 1.0
        ).astype(np.float32),
    )


def _xy_plane(size: float = 4.0) -> FieldPlaneState:
    return FieldPlaneState(
        center_mm=np.zeros(3),
        u_axis=np.array((1.0, 0.0, 0.0)),
        v_axis=np.array((0.0, 1.0, 0.0)),
        width_mm=size,
        height_mm=size,
    )


def test_plane_sampling_evaluates_the_authoritative_field() -> None:
    sample = sample_field_plane(_sphere_body(), _xy_plane(), (101, 101))

    assert sample.shape == (101, 101)
    assert sample.values[50, 50] < -0.9
    assert sample.values[0, 0] > 0.0
    assert np.allclose(sample.points[50, 50], (0.0, 0.0, 0.0))


def test_expanded_implicit_body_samples_real_exterior_distance() -> None:
    source = _sphere_body()
    expanded = expand_implicit_body_bounds(source, 3.0)
    sample = sample_field_plane(expanded, _xy_plane(size=8.0), 129)

    np.testing.assert_allclose(expanded.bounds, [[-5.0, -5.0, -5.0], [5.0, 5.0, 5.0]])
    assert sample.value_at(np.array((3.0, 0.0, 0.0))) == 2.0
    assert sample.value_at(np.array((4.0, 0.0, 0.0))) == 3.0


def test_plane_sampling_tri_linearly_interpolates_a_display_field() -> None:
    coordinates = np.arange(5, dtype=np.float64)
    xx, yy, zz = np.meshgrid(coordinates, coordinates, coordinates, indexing="ij")
    field = SampledImplicitField(
        name="linear",
        values=(xx + 2.0 * yy + 3.0 * zz).astype(np.float32),
        origin=np.zeros(3),
        spacing=np.ones(3),
        color=(0.2, 0.4, 0.8, 1.0),
    )
    state = FieldPlaneState(
        center_mm=np.array((2.0, 2.0, 2.0)),
        u_axis=np.array((1.0, 0.0, 0.0)),
        v_axis=np.array((0.0, 1.0, 0.0)),
        width_mm=1.0,
        height_mm=1.0,
    )
    sample = sample_field_plane(field, state, (3, 3))

    assert sample.value_at(np.array((2.0, 2.0, 2.0))) == 12.0
    assert sample.value_at(np.array((9.0, 2.0, 2.0))) is None


def test_field_plane_extracts_a_closed_zero_isoline() -> None:
    sample = sample_field_plane(_sphere_body(), _xy_plane(), 129)
    isolines = extract_field_plane_isolines(
        sample,
        value_range=(-1.5, 1.5),
        interval=0.0,
    )

    zero = min(isolines, key=lambda item: abs(item.value))
    assert abs(zero.value) < 1.0e-9
    assert len(zero.polylines) == 1
    radii = np.linalg.norm(zero.polylines[0][:, :2], axis=1)
    assert np.allclose(np.mean(radii), 1.0, atol=0.03)


def test_implicit_colormap_uses_a_symmetric_automatic_range() -> None:
    sample = sample_field_plane(_sphere_body(), _xy_plane(), 33)

    lower, upper = resolve_field_value_range(sample, "implicit")

    assert lower == -upper
    assert lower < 0.0 < upper


def test_rendered_field_plane_is_clipped_to_the_finite_source_bounds() -> None:
    sample = sample_field_plane(_sphere_body(), _xy_plane(size=8.0), 65)
    data = FieldPlaneRenderData(
        sample=sample,
        colormap="implicit",
        value_range=(-2.0, 2.0),
        opacity=0.6,
    )

    plane = build_field_plane_polydata(data)

    assert plane.n_cells > 0
    bounds = np.asarray(plane.bounds)
    # VTK retains cells intersecting the clip boundary.  Their one-sample
    # interpolation margin is bounded by the 0.125 mm plane spacing.
    np.testing.assert_allclose(bounds, [-2.125, 2.125, -2.125, 2.125, 0.0, 0.0])


def test_full_plane_marks_points_outside_the_field_as_no_data() -> None:
    sample = sample_field_plane(_sphere_body(), _xy_plane(size=8.0), 65)
    data = FieldPlaneRenderData(
        sample=sample,
        colormap="implicit",
        value_range=(-2.0, 2.0),
        opacity=0.6,
        clip_to_source_bounds=False,
    )

    plane = build_field_plane_polydata(data)

    assert plane.n_cells > 0
    assert np.isnan(np.asarray(plane["field_value"])).any()
    assert np.isfinite(np.asarray(plane["field_value"])).any()
    np.testing.assert_allclose(
        np.asarray(plane.bounds),
        (-4.0, 4.0, -4.0, 4.0, 0.0, 0.0),
    )


def test_full_plane_excludes_no_data_from_colormap_and_isolines() -> None:
    sample = sample_field_plane(_sphere_body(), _xy_plane(size=8.0), 129)

    lower, upper = resolve_field_value_range(sample, "implicit")
    isolines = extract_field_plane_isolines(
        sample,
        value_range=(-4.0, 4.0),
        interval=0.5,
    )

    # The synthetic positive values used for safe out-of-bounds evaluation are
    # not physical field data and must not widen the display scale.
    assert lower == -upper
    assert upper < 2.0
    for isoline in isolines:
        for polyline in isoline.polylines:
            assert np.all(np.abs(polyline[:, :2]) <= 2.0 + 1.0e-9)


def test_setting_a_field_plane_refreshes_camera_bounds() -> None:
    """A larger inspection plane must not inherit an old model clip range."""

    calls: list[str] = []
    sample = sample_field_plane(_sphere_body(), _xy_plane(size=12.0), 17)
    data = FieldPlaneRenderData(
        sample=sample,
        colormap="implicit",
        value_range=(-2.0, 2.0),
        opacity=0.6,
        clip_to_source_bounds=False,
    )
    owner = type("ViewerHarness", (), {})()
    owner._field_plane_data = None
    owner._remove_field_plane_actors = lambda: calls.append("remove")
    owner._add_field_plane_to_scene = lambda received: (
        calls.append("add") if received is data else None
    )
    owner._update_bbox = lambda: calls.append("bounds")
    owner._fix_camera_clipping = lambda: calls.append("clipping")
    owner.render = lambda: calls.append("render")

    PyVistaRenderer.set_field_plane(owner, data)

    assert owner._field_plane_data is data
    assert calls == ["remove", "add", "bounds", "clipping", "render"]


def test_field_viewer_can_hide_model_layers_without_clearing_its_plane() -> None:
    """Object visibility is a render-only Field Viewer setting."""

    from lattice_studio.presentation.qt.workbench import _build_qt_app

    class _Flag:
        def __init__(self, checked: bool) -> None:
            self._checked = checked

        def isChecked(self) -> bool:
            return self._checked

    class _Viewer:
        def __init__(self) -> None:
            self.field_plane = object()
            self.calls: list[tuple[list[object], bool]] = []

        def set_implicit_fields(
            self,
            fields: list[object],
            *,
            reset_view: bool,
        ) -> None:
            self.calls.append((fields, reset_view))

    viewer = _Viewer()
    harness = type("WindowHarness", (), {})()
    harness.field_viewer_enabled = _Flag(True)
    harness.field_viewer_show_object = _Flag(False)
    harness.viewer = viewer
    harness._refresh_render_backend_info = lambda: None
    harness._update_transition_plane = lambda: None

    window_class = _build_qt_app()
    window_class._refresh_scene(harness, reset_view=False)

    assert viewer.calls == [([], False)]
    assert viewer.field_plane is not None


def test_surface_pick_returns_nearest_zero_crossing_and_normal() -> None:
    hit = field_surface_hit_from_ray(
        _sphere_body(),
        ray_origin_mm=np.array((0.0, 0.0, 4.0)),
        ray_direction=np.array((0.0, 0.0, -1.0)),
        sample_spacing_mm=0.05,
    )

    assert hit is not None
    assert np.allclose(hit.point_mm, (0.0, 0.0, 1.0), atol=1.0e-3)
    assert np.allclose(hit.normal, (0.0, 0.0, 1.0), atol=1.0e-3)


def test_plane_axes_fall_back_when_the_preferred_axis_is_parallel() -> None:
    u_axis, v_axis = orthonormal_plane_axes(
        np.array((0.0, 0.0, 1.0)),
        preferred_u_axis=np.array((0.0, 0.0, 2.0)),
    )

    assert abs(float(np.dot(u_axis, v_axis))) < 1.0e-12
    assert abs(float(np.dot(u_axis, (0.0, 0.0, 1.0)))) < 1.0e-12


def test_plane_cache_is_lru_bounded_and_resolution_is_capped() -> None:
    field = _sphere_body()
    state = _xy_plane(size=100.0)
    cache = FieldPlaneCache(max_entries=1)
    sample = sample_field_plane(field, _xy_plane(), 9)
    cache.put("first", sample)
    cache.put("second", sample)

    resolution = recommended_plane_resolution(state, field, maximum=200)
    assert cache.get("first") is None
    assert cache.get("second") is sample
    assert resolution == (200, 200)


def test_probe_uses_known_plane_geometry_without_vtk_cell_pick() -> None:
    """Dense Field Viewer planes must use ray-plane evaluation on hover."""

    class _Interactor:
        def GetEventPosition(self):
            return (100, 100)

    sample = sample_field_plane(_sphere_body(), _xy_plane(), 17)
    owner = type("ViewerHarness", (), {})()
    owner._field_plane_data = FieldPlaneRenderData(
        sample=sample,
        colormap="implicit",
        value_range=(-2.0, 2.0),
        opacity=0.6,
    )
    owner._field_plane_actor = object()
    owner.interactor = _Interactor()
    owner._field_plane_value_at = PyVistaRenderer._field_plane_value_at
    owner._field_plane_point_from_display = lambda _position, _data: np.zeros(3)
    owner.probed = None
    owner.cleared = False
    owner._set_field_plane_probe = lambda point, value, _display_position: setattr(
        owner,
        "probed",
        (np.asarray(point), value),
    )
    owner._queue_field_plane_probe_update = lambda point, value, position: owner._set_field_plane_probe(
        point,
        value,
        position,
    )
    owner._clear_field_plane_probe = lambda: setattr(owner, "cleared", True)

    PyVistaRenderer._on_field_plane_mouse_move(owner, None, None)

    assert owner.probed is not None
    assert owner.cleared is False
