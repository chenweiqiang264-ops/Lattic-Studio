"""Regression tests for bounded, cached STL display geometry."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pyvista as pv


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _load_renderer_module():
    from lattice_studio.presentation.qt.viewers import pyvista_viewer

    return pyvista_viewer


def _dense_sphere(subdivisions: int = 5):
    surface = pv.Sphere(theta_resolution=220, phi_resolution=220).triangulate()
    vertices = np.asarray(surface.points).copy()
    faces = np.asarray(surface.faces).reshape(-1, 4)[:, 1:].copy()
    return vertices, faces


def test_display_proxy_reduces_faces_without_changing_source_arrays() -> None:
    renderer = _load_renderer_module()
    vertices, faces = _dense_sphere()
    original_vertices = vertices.copy()
    original_faces = faces.copy()

    proxy = renderer.build_mesh_display_polydata(vertices, faces, face_limit=8_000)

    assert 0 < proxy.n_cells <= 8_400
    assert proxy.n_open_edges == 0
    np.testing.assert_array_equal(vertices, original_vertices)
    np.testing.assert_array_equal(faces, original_faces)


def test_display_cache_reuses_proxy_for_the_same_mesh_and_quality() -> None:
    renderer = _load_renderer_module()
    vertices, faces = _dense_sphere()
    harness = SimpleNamespace(_mesh_display_cache={})
    mesh = renderer.MeshData(vertices, faces, cache_key=("result", 7))

    first = renderer.PyVistaRenderer._display_polydata(harness, mesh, 8_000)
    second = renderer.PyVistaRenderer._display_polydata(harness, mesh, 8_000)

    assert first is second
    assert len(harness._mesh_display_cache) == 1


def test_mesh_material_refresh_keeps_actor_mapper_and_geometry_resident() -> None:
    from PyQt5 import QtWidgets

    renderer = _load_renderer_module()
    qt_app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    viewer = renderer.PyVistaRenderer()
    surface = pv.Sphere(theta_resolution=24, phi_resolution=24).triangulate()
    vertices = np.asarray(surface.points).copy()
    faces = np.asarray(surface.faces).reshape(-1, 4)[:, 1:].copy()
    first_layer = renderer.MeshData(
        vertices,
        faces,
        color=(0.2, 0.3, 0.4, 1.0),
        cache_key=("resident", 1),
    )
    second_layer = renderer.MeshData(
        vertices,
        faces,
        color=(0.8, 0.1, 0.2, 1.0),
        cache_key=("resident", 1),
    )

    try:
        viewer.set_meshes([first_layer], reset_view=False)
        qt_app.processEvents()
        first_actor = viewer._pv_actors[0]
        first_mapper = first_actor.GetMapper()
        first_input = first_mapper.GetInput()

        viewer.set_meshes([second_layer], reset_view=False)
        qt_app.processEvents()

        assert viewer._pv_actors[0] is first_actor
        assert viewer._pv_actors[0].GetMapper() is first_mapper
        assert viewer._pv_actors[0].GetMapper().GetInput() is first_input
        np.testing.assert_allclose(
            viewer._pv_actors[0].GetProperty().GetColor(),
            (0.8, 0.1, 0.2),
            atol=1.0 / 255.0,
        )

        viewer.set_meshes([], reset_view=False)
        assert bool(first_actor.GetVisibility()) is False
        viewer.set_meshes([second_layer], reset_view=False)
        assert viewer._pv_actors[0] is first_actor
        assert bool(first_actor.GetVisibility()) is True
    finally:
        viewer.close()
        qt_app.processEvents()


def test_camera_interaction_keeps_the_target_quality_actor() -> None:
    from PyQt5 import QtWidgets

    renderer = _load_renderer_module()
    qt_app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    viewer = renderer.PyVistaRenderer()
    vertices, faces = _dense_sphere()
    layer = renderer.MeshData(
        vertices,
        faces,
        cache_key=("interactive", 1),
    )

    try:
        viewer.set_meshes([layer], reset_view=False)
        qt_app.processEvents()
        main_actor = viewer._pv_actors[0]
        main_mapper = main_actor.GetMapper()
        record = viewer._mesh_layer_records[("interactive", 1)]
        assert getattr(record, "interaction_actor", None) is None

        for button in ("Right", "Middle"):
            viewer.interactor.InvokeEvent(f"{button}ButtonPressEvent")
            qt_app.processEvents()
            assert bool(main_actor.GetVisibility()) is True

            viewer.interactor.InvokeEvent(f"{button}ButtonReleaseEvent")
            qt_app.processEvents()
            assert bool(main_actor.GetVisibility()) is True
            assert main_actor.GetMapper() is main_mapper
    finally:
        viewer.close()
        qt_app.processEvents()


def test_tpms_display_quality_never_collapses_to_a_tiny_absolute_face_budget() -> None:
    renderer = _load_renderer_module()
    source_faces = 5_134_712

    medium_limit = renderer.mesh_display_face_limit(source_faces, "medium")
    ultra_limit = renderer.mesh_display_face_limit(source_faces, "ultra")

    assert medium_limit >= int(source_faces * 0.70)
    assert ultra_limit == source_faces


def test_mesh_section_updates_mapper_without_rebuilding_geometry() -> None:
    renderer = _load_renderer_module()

    class Mapper:
        def __init__(self) -> None:
            self.removed = 0
            self.planes = None

        def RemoveAllClippingPlanes(self) -> None:
            self.removed += 1

        def SetClippingPlanes(self, planes) -> None:
            self.planes = planes

    class Actor:
        def __init__(self, mapper) -> None:
            self.mapper = mapper

        def GetMapper(self):
            return self.mapper

    class Plane:
        def EvaluateFunction(self, *_args):
            return 0.0

        def GetOrigin(self):
            return (1.0, 2.0, 3.0)

        def GetNormal(self):
            return (1.0, 0.0, 0.0)

    class Harness:
        def __init__(self) -> None:
            self.meshes = [object()]
            self._pv_actors = [Actor(Mapper())]
            self._implicit_actors = []
            self.section_enabled = False
            self.section_bounds = None
            self.section_planes = None
            self.section_plane = None
            self.render_count = 0

        def _update_scene(self) -> None:
            raise AssertionError("section interaction rebuilt mesh geometry")

        def _update_mesh_clipping(self) -> None:
            renderer.PyVistaRenderer._update_mesh_clipping(self)

        def _apply_mesh_clipping(self, mapper) -> None:
            renderer.PyVistaRenderer._apply_mesh_clipping(self, mapper)

        def _mesh_clipping_plane_collection(self):
            return renderer.PyVistaRenderer._mesh_clipping_plane_collection(self)

        def render(self) -> None:
            self.render_count += 1

    harness = Harness()
    plane = Plane()

    renderer.PyVistaRenderer.set_section_plane(harness, plane)

    mapper = harness._pv_actors[0].GetMapper()
    assert mapper.planes.GetNumberOfItems() == 1
    np.testing.assert_allclose(mapper.planes.GetItem(0).GetOrigin(), (1.0, 2.0, 3.0))
    np.testing.assert_allclose(mapper.planes.GetItem(0).GetNormal(), (-1.0, 0.0, 0.0))
    assert mapper.removed == 1
    assert harness.render_count == 1
