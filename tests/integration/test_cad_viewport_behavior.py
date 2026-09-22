"""Regression tests for the bright CAD-style implicit viewport."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lattice_studio.engine.implicit.cell_map import CellMap, CellMapFrame


def _load_renderer_module():
    from lattice_studio.presentation.qt.viewers import pyvista_viewer

    return pyvista_viewer


class _Camera:
    def __init__(self) -> None:
        self.position = np.array((0.0, 0.0, 10.0), dtype=float)
        self.focal_point = np.zeros(3, dtype=float)
        self.up = (0.0, 0.0, 1.0)
        self.parallel_projection = False
        self.parallel_scale = 1.0
        self.clipping_range = (0.1, 100.0)

    @property
    def distance(self) -> float:
        return float(np.linalg.norm(self.position - self.focal_point))


class _ResetHarness:
    def __init__(self, renderer_module) -> None:
        self.meshes = []
        self.implicit_fields = [object()]
        self._scene_bounds = np.array(((10.0, 20.0, -2.0), (30.0, 80.0, 8.0)))
        self._bbox_center = self._scene_bounds.mean(axis=0)
        self._bbox_size = 60.0
        self.camera = _Camera()
        self.reset_camera_called = False
        self.render_count = 0
        self._renderer_module = renderer_module

    def reset_camera(self) -> None:
        self.reset_camera_called = True

    def _has_scene_content(self) -> bool:
        return self._renderer_module.PyVistaRenderer._has_scene_content(self)

    def _set_camera_orientation(self, direction, view_up) -> None:
        self._renderer_module.PyVistaRenderer._set_camera_orientation(
            self,
            direction,
            view_up,
        )

    def _fix_camera_clipping(self) -> None:
        self._renderer_module.PyVistaRenderer._fix_camera_clipping(self)

    def render(self) -> None:
        self.render_count += 1


def test_reset_view_fits_implicit_bounds_without_resetting_to_auxiliary_props() -> None:
    renderer = _load_renderer_module()
    harness = _ResetHarness(renderer)

    renderer.PyVistaRenderer.reset_view(harness)

    assert harness.reset_camera_called is False
    np.testing.assert_allclose(harness.camera.focal_point, (20.0, 50.0, 3.0))
    assert harness.camera.parallel_projection is True
    assert 65.0 <= harness.camera.parallel_scale <= 80.0


def test_close_zoom_keeps_near_clip_before_the_focal_plane() -> None:
    renderer = _load_renderer_module()
    harness = _ResetHarness(renderer)
    harness.camera.position = np.array((0.0, 0.0, 0.01))
    harness.camera.focal_point = np.zeros(3)

    renderer.PyVistaRenderer._fix_camera_clipping(harness)

    near_clip, far_clip = harness.camera.clipping_range
    assert 0.0 < near_clip < harness.camera.distance
    assert far_clip > harness._bbox_size


def test_parallel_zoom_supports_deep_close_inspection() -> None:
    renderer = _load_renderer_module()
    harness = _ResetHarness(renderer)
    initial_scale = harness.camera.parallel_scale

    for _ in range(160):
        renderer.PyVistaRenderer._apply_wheel_zoom(harness, 1.0)

    assert 0.0 < harness.camera.parallel_scale < initial_scale * 1.0e-5
    assert harness.camera.parallel_scale <= harness._bbox_size * 1.01e-7
    assert harness.render_count == 160


def test_interactive_quality_avoids_unstable_per_pixel_jitter() -> None:
    renderer = _load_renderer_module()

    assert renderer.DEFAULT_RENDER_QUALITY == "high"
    assert renderer.RENDER_QUALITY_SETTINGS["medium"].use_jittering is False
    assert renderer.RENDER_QUALITY_SETTINGS["high"].use_jittering is False
    assert renderer.RENDER_QUALITY_SETTINGS["ultra"].use_jittering is False
    assert renderer.RENDER_QUALITY_SETTINGS["high"].use_ssao is False
    assert renderer.RENDER_QUALITY_SETTINGS["ultra"].use_ssao is True


def test_studio_lighting_is_neutral_on_a_white_cad_background() -> None:
    renderer = _load_renderer_module()

    lights = renderer.build_studio_lights(np.zeros(3), 100.0)

    assert len(lights) >= 4
    colors = [light.GetDiffuseColor() for light in lights]
    assert all(max(color) - min(color) <= 0.04 for color in colors)
    assert 0.9 <= max(light.intensity for light in lights) <= 1.15


def test_studio_lighting_uses_one_dominant_key_without_flattening_fill() -> None:
    renderer = _load_renderer_module()

    lights = renderer.build_studio_lights(np.zeros(3), 100.0)
    intensities = sorted((light.intensity for light in lights), reverse=True)

    assert intensities[0] >= 0.9
    assert sum(intensities[1:]) <= 0.25
    assert intensities[0] >= 6.0 * intensities[1]


def test_studio_environment_texture_contains_softbox_contrast() -> None:
    renderer = _load_renderer_module()

    texture = renderer.build_studio_environment_texture()
    pixels = np.asarray(texture.to_array(), dtype=np.float64)

    assert pixels.ndim == 3
    assert pixels.shape[2] == 3
    assert float(pixels.max() - pixels.min()) >= 120.0
    assert 25.0 <= float(np.median(pixels)) <= 90.0


def test_cad_defaults_start_with_white_background() -> None:
    from lattice_studio.presentation.qt import workbench as module

    assert next(iter(module.BACKGROUND_PRESETS.values())) == ("#F1F3F6", "#FFFFFF")
    assert module.MATERIAL_PRESETS["CAD 银白"] == (0.68, 0.70, 0.74)


def test_cell_map_wireframe_contains_all_three_axis_line_families() -> None:
    renderer = _load_renderer_module()
    cell_map = CellMap.from_bounds(
        np.array(((0.0, 0.0, 0.0), (4.0, 6.0, 8.0))),
        spacing_mm=(2.0, 3.0, 4.0),
    )

    grid = renderer.PyVistaRenderer._cell_map_wireframe(cell_map)

    nx, ny, nz = cell_map.grid_shape
    assert grid.n_lines == nx * ny + nx * nz + ny * nz
    np.testing.assert_allclose(grid.bounds, (0.0, 4.0, 0.0, 6.0, 0.0, 8.0))


def test_cell_map_wireframe_uses_rotated_world_space_segments() -> None:
    renderer = _load_renderer_module()
    frame = CellMapFrame.from_origin_axes(
        origin=(1.0, 2.0, 3.0),
        u_axis=(0.0, 1.0, 0.0),
        v_axis=(-1.0, 0.0, 0.0),
    )
    points = frame.to_world(
        np.array(((0.0, 0.0, 0.0), (4.0, 6.0, 8.0)), dtype=np.float64)
    )
    cell_map = CellMap.from_points(points, (2.0, 3.0, 4.0), frame)

    grid = renderer.PyVistaRenderer._cell_map_wireframe(cell_map)

    nx, ny, nz = cell_map.grid_shape
    assert grid.n_lines == nx * ny + nx * nz + ny * nz
    expected_bounds = (
        cell_map.bounds[0, 0],
        cell_map.bounds[1, 0],
        cell_map.bounds[0, 1],
        cell_map.bounds[1, 1],
        cell_map.bounds[0, 2],
        cell_map.bounds[1, 2],
    )
    np.testing.assert_allclose(grid.bounds, expected_bounds)
