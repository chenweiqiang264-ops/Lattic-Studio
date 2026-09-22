"""Standalone NTopology-style interactive section gizmo for STL meshes.

The gizmo follows the common three-axis transform convention:

* X is red, Y is green, and Z is blue;
* the three arrow handles translate the section plane along their axes;
* the three coloured arcs rotate the section plane around their axes.

The section is a single implicit plane.  Its positive normal side is hidden
for display only; the source STL and the stored mesh data are never modified.

Example::

    python -m lattice_studio.presentation.qt.tools.interactive_section_viewer
    python -m lattice_studio.presentation.qt.tools.interactive_section_viewer --mesh path/to/model.stl
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Callable

import numpy as np
import trimesh
from PyQt5 import QtWidgets
from vtkmodules.vtkCommonCore import vtkCommand, vtkPoints
from vtkmodules.vtkCommonDataModel import (
    vtkCellArray,
    vtkPlane,
    vtkPolyData,
    vtkPolyLine,
)
from vtkmodules.vtkCommonMath import vtkMatrix4x4
from vtkmodules.vtkCommonTransforms import vtkTransform
from vtkmodules.vtkFiltersCore import vtkTubeFilter
from vtkmodules.vtkFiltersSources import vtkArrowSource, vtkCubeSource, vtkPlaneSource
from vtkmodules.vtkRenderingCore import vtkActor, vtkCellPicker, vtkPolyDataMapper, vtkRenderer


PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _load_renderer_classes():
    """Load the production renderer without importing optional Open3D support."""

    from lattice_studio.presentation.qt.viewers.pyvista_viewer import (
        MeshData,
        PyVistaRenderer,
    )

    return MeshData, PyVistaRenderer


MeshData, PyVistaRenderer = _load_renderer_classes()


def load_triangle_mesh(path: Path) -> trimesh.Trimesh:
    """Load an STL path and normalize a possible scene to one triangle mesh."""

    loaded = trimesh.load(path, force="mesh")
    if not isinstance(loaded, trimesh.Trimesh):
        loaded = trimesh.util.concatenate(tuple(loaded.dump()))
    if len(loaded.faces) == 0:
        raise ValueError(f"STL contains no faces: {path}")
    return loaded


def vtk_bounds_from_mesh(mesh: trimesh.Trimesh) -> np.ndarray:
    """Convert Trimesh ``[[min], [max]]`` bounds to VTK's six-value order."""

    mesh_bounds = np.asarray(mesh.bounds, dtype=float)
    if mesh_bounds.shape != (2, 3) or not np.isfinite(mesh_bounds).all():
        raise ValueError("mesh bounds must be a finite 2x3 array")
    values = np.array(
        (
            mesh_bounds[0, 0],
            mesh_bounds[1, 0],
            mesh_bounds[0, 1],
            mesh_bounds[1, 1],
            mesh_bounds[0, 2],
            mesh_bounds[1, 2],
        ),
        dtype=float,
    )
    if not (values[0] < values[1] and values[2] < values[3] and values[4] < values[5]):
        raise ValueError("mesh bounds must have positive extents")
    return values


def _unit_vector(vector: np.ndarray, name: str) -> np.ndarray:
    value = np.asarray(vector, dtype=float).reshape(3)
    length = np.linalg.norm(value)
    if not np.isfinite(length) or length <= np.finfo(float).eps:
        raise ValueError(f"{name} must be a finite non-zero 3D vector")
    return value / length


def _basis_from_normal(normal: np.ndarray) -> np.ndarray:
    """Build a right-handed local XYZ basis whose Z axis is ``normal``."""

    local_z = _unit_vector(normal, "section normal")
    reference = np.array((1.0, 0.0, 0.0))
    if abs(float(np.dot(reference, local_z))) > 0.9:
        reference = np.array((0.0, 1.0, 0.0))
    local_x = _unit_vector(np.cross(reference, local_z), "section X axis")
    local_y = _unit_vector(np.cross(local_z, local_x), "section Y axis")
    return np.column_stack((local_x, local_y, local_z))


def _axis_angle_matrix(axis: np.ndarray, angle: float) -> np.ndarray:
    """Return a right-handed 3-D rotation matrix."""

    axis = _unit_vector(axis, "rotation axis")
    x, y, z = axis
    c = float(np.cos(angle))
    s = float(np.sin(angle))
    one_minus_c = 1.0 - c
    return np.array(
        (
            (c + x * x * one_minus_c, x * y * one_minus_c - z * s, x * z * one_minus_c + y * s),
            (y * x * one_minus_c + z * s, c + y * y * one_minus_c, y * z * one_minus_c - x * s),
            (z * x * one_minus_c - y * s, z * y * one_minus_c + x * s, c + z * z * one_minus_c),
        ),
        dtype=float,
    )


def _signed_screen_angle(before: np.ndarray, after: np.ndarray) -> float:
    """Return the signed cursor angle in the renderer's screen plane."""

    first = np.asarray(before, dtype=float).reshape(2)
    second = np.asarray(after, dtype=float).reshape(2)
    cross = float(first[0] * second[1] - first[1] * second[0])
    dot = float(np.dot(first, second))
    return float(np.arctan2(cross, dot))


class _SectionGizmo:
    """Three-axis arrow and rotation-arc manipulator for one section plane."""

    AXIS_COLORS = ((0.95, 0.15, 0.15), (0.15, 0.82, 0.35), (0.15, 0.48, 0.98))
    AXIS_NAMES = ("X", "Y", "Z")
    BLOCK_LEFT_VIEW_MODE = ("block_left_view", None)
    # vtkCellPicker expresses tolerance as a fraction of the render-window
    # diagonal.  Keep only its numerical baseline; do not add a screen-space
    # halo around the handles.
    PICK_TOLERANCE = 1.0e-6
    ROTATION_ARC_SEGMENTS = 144

    def __init__(
        self,
        viewer: "InteractiveSectionViewer",
        bounds: np.ndarray,
        origin: np.ndarray,
        normal: np.ndarray,
        changed_callback,
    ):
        self.viewer = viewer
        self.renderer = viewer.renderer
        self._overlay_renderer = vtkRenderer()
        self._overlay_renderer.SetLayer(1)
        self._overlay_renderer.SetInteractive(False)
        # The manipulator is a screen-space interaction aid, not scene
        # geometry.  Do not inherit the scene depth buffer: otherwise a large
        # primitive or implicit volume can hide its handles.  Keeping this in
        # a dedicated renderer preserves normal depth testing for all actual
        # objects while making the gizmo unconditionally foreground.
        # Preserve the already rendered color buffer, but erase the inherited
        # depth buffer.  ``EraseOff`` leaves the scene depth values active and
        # causes large objects to cut holes in the gizmo arcs.
        self._overlay_renderer.SetPreserveColorBuffer(True)
        self._overlay_renderer.SetErase(True)
        self._overlay_renderer.SetPreserveDepthBuffer(False)
        try:
            self._overlay_renderer.SetUseDepthPeeling(False)
        except AttributeError:
            pass
        self._overlay_renderer.SetViewport(*self.renderer.GetViewport())
        self._overlay_renderer.SetActiveCamera(self.renderer.GetActiveCamera())
        viewer.ren_win.SetNumberOfLayers(2)
        viewer.ren_win.AddRenderer(self._overlay_renderer)
        self.pick_renderer = self._overlay_renderer
        self.interactor = viewer.iren.interactor
        self.bounds = np.asarray(bounds, dtype=float).copy()
        self.origin = np.asarray(origin, dtype=float).reshape(3).copy()
        self.basis = _basis_from_normal(normal)
        self.initial_origin = self.origin.copy()
        self.initial_basis = self.basis.copy()
        self.changed_callback = changed_callback
        # Keep the manipulator readable around large primitives.  The old
        # 0.32 factor placed the complete arrow inside a sphere or box, so
        # the geometry visually covered the handle.  A handle reaching past
        # the half-extent remains usable while the small margin keeps the
        # rotation arcs close to the object.
        self.gizmo_length = max(float(np.max(self._extent())), 1.0) * 0.90
        self.arc_radius = self.gizmo_length * 0.78
        self.arc_tube_radius = max(self.gizmo_length * 0.018, 0.03)
        self._active_mode: tuple[str, int | None] | None = None
        self._last_mouse: np.ndarray | None = None
        self._observer_ids: list[int] = []
        self._actor_modes: dict[str, tuple[str, int | None]] = {}
        self._actors_added = False
        self._hover_mode: tuple[str, int | None] | None = None
        self._picker = vtkCellPicker()
        # Keep hover hit-testing strict: the cursor must be over the rendered
        # arrow/arc geometry rather than merely close to it.
        self._picker.SetTolerance(self.PICK_TOLERANCE)

        self._arrow_source = vtkArrowSource()
        self._arrow_source.SetTipLength(0.22)
        self._arrow_source.SetTipRadius(0.075)
        self._arrow_source.SetShaftRadius(0.026)
        self._arrow_source.Update()
        self._arrow_mapper = vtkPolyDataMapper()
        self._arrow_mapper.SetInputConnection(self._arrow_source.GetOutputPort())
        self._arrow_actors = self._create_arrow_actors()
        self._arc_actors, self._arc_points, self._arc_tubes = self._create_arc_actors()
        self._center_source = vtkCubeSource()
        self._center_source.SetXLength(self.gizmo_length * 0.16)
        self._center_source.SetYLength(self.gizmo_length * 0.16)
        self._center_source.SetZLength(self.gizmo_length * 0.16)
        self._center_source.Update()
        self._center_mapper = vtkPolyDataMapper()
        self._center_mapper.SetInputConnection(self._center_source.GetOutputPort())
        self._center_actor = vtkActor()
        self._center_actor.SetMapper(self._center_mapper)
        self._center_actor.GetProperty().SetColor(0.92, 0.92, 0.92)
        self._center_actor.GetProperty().SetOpacity(0.92)
        self._plane_source = vtkPlaneSource()
        self._plane_mapper = vtkPolyDataMapper()
        self._plane_mapper.SetInputConnection(self._plane_source.GetOutputPort())
        self._plane_actor = vtkActor()
        self._plane_actor.SetMapper(self._plane_mapper)
        self._plane_actor.GetProperty().SetColor(0.95, 0.62, 0.18)
        self._plane_actor.GetProperty().SetOpacity(0.08)
        self._plane_actor.PickableOff()
        self._connect_interaction_events()
        self.update_geometry()

    def _extent(self) -> np.ndarray:
        return np.array(
            (
                self.bounds[1] - self.bounds[0],
                self.bounds[3] - self.bounds[2],
                self.bounds[5] - self.bounds[4],
            ),
            dtype=float,
        )

    @staticmethod
    def _actor_key(actor) -> str:
        return actor.GetAddressAsString("")

    @classmethod
    def _rotation_arc_angles(cls) -> np.ndarray:
        """Return a closed, evenly sampled rotation-handle ring."""

        return np.linspace(0.0, 2.0 * np.pi, cls.ROTATION_ARC_SEGMENTS + 1)

    def _create_arrow_actors(self) -> list[vtkActor]:
        actors = []
        for axis, color in enumerate(self.AXIS_COLORS):
            actor = vtkActor()
            actor.SetMapper(self._arrow_mapper)
            actor.GetProperty().SetColor(*color)
            actor.PickableOn()
            self._actor_modes[self._actor_key(actor)] = ("translate", axis)
            actors.append(actor)
        return actors

    def _create_arc_actors(self) -> tuple[list[vtkActor], list[vtkPoints], list[vtkTubeFilter]]:
        actors = []
        point_sets = []
        tubes = []
        theta = self._rotation_arc_angles()
        for axis, color in enumerate(self.AXIS_COLORS):
            points = vtkPoints()
            for _ in theta:
                points.InsertNextPoint(0.0, 0.0, 0.0)
            polyline = vtkPolyLine()
            polyline.GetPointIds().SetNumberOfIds(len(theta))
            for index in range(len(theta)):
                polyline.GetPointIds().SetId(index, index)
            lines = vtkCellArray()
            lines.InsertNextCell(polyline)
            polydata = vtkPolyData()
            polydata.SetPoints(points)
            polydata.SetLines(lines)
            tube = vtkTubeFilter()
            tube.SetInputData(polydata)
            tube.SetRadius(self.arc_tube_radius)
            tube.SetNumberOfSides(12)
            tube.CappingOn()
            tube.Update()
            mapper = vtkPolyDataMapper()
            mapper.SetInputConnection(tube.GetOutputPort())
            actor = vtkActor()
            actor.SetMapper(mapper)
            actor.GetProperty().SetColor(*color)
            actor.PickableOn()
            self._actor_modes[self._actor_key(actor)] = ("rotate", axis)
            actors.append(actor)
            point_sets.append(points)
            tubes.append(tube)
        return actors, point_sets, tubes

    def _connect_interaction_events(self) -> None:
        self._observer_ids.extend(
            (
                self.interactor.AddObserver(vtkCommand.LeftButtonPressEvent, self._on_left_press, 1.0),
                self.interactor.AddObserver(vtkCommand.MouseMoveEvent, self._on_move, 1.0),
                self.interactor.AddObserver(vtkCommand.LeftButtonReleaseEvent, self._on_left_release, 1.0),
                self.interactor.AddObserver(vtkCommand.LeaveEvent, self._on_leave, 1.0),
            )
        )

    def disconnect(self) -> None:
        self._clear_active_drag()
        self._set_highlight_mode(None, render=False)
        for observer_id in self._observer_ids:
            try:
                self.interactor.RemoveObserver(observer_id)
            except Exception:
                pass
        self._observer_ids.clear()
        self._overlay_renderer.RemoveAllViewProps()
        try:
            self.viewer.ren_win.RemoveRenderer(self._overlay_renderer)
            self.viewer.ren_win.SetNumberOfLayers(1)
        except Exception:
            pass

    def add_to_scene(self) -> None:
        if self._actors_added:
            return
        for actor in (*self._arrow_actors, *self._arc_actors, self._center_actor, self._plane_actor):
            self._overlay_renderer.AddActor(actor)
        self._actors_added = True

    def reset(self) -> None:
        self.origin = self.initial_origin.copy()
        self.basis = self.initial_basis.copy()
        self.update_geometry()
        self.changed_callback()

    def synchronize_transform(
        self,
        bounds: np.ndarray,
        origin: np.ndarray,
        basis: np.ndarray,
    ) -> None:
        """Update an idle gizmo without rebuilding its native VTK resources."""

        bounds_array = np.asarray(bounds, dtype=float).reshape(-1)
        if (
            bounds_array.shape != (6,)
            or not np.isfinite(bounds_array).all()
            or not (
                bounds_array[0] < bounds_array[1]
                and bounds_array[2] < bounds_array[3]
                and bounds_array[4] < bounds_array[5]
            )
        ):
            raise ValueError("gizmo bounds must contain three positive finite ranges")
        origin_array = np.asarray(origin, dtype=float).reshape(3)
        basis_array = np.asarray(basis, dtype=float)
        if not np.isfinite(origin_array).all():
            raise ValueError("gizmo origin must contain three finite values")
        if basis_array.shape != (3, 3) or not np.isfinite(basis_array).all():
            raise ValueError("gizmo basis must be a finite 3 by 3 matrix")
        if not np.allclose(basis_array.T @ basis_array, np.eye(3), atol=1.0e-6):
            raise ValueError("gizmo basis must be orthonormal")
        if not np.isclose(np.linalg.det(basis_array), 1.0, atol=1.0e-6):
            raise ValueError("gizmo basis must be right-handed")

        self._clear_active_drag()
        self.bounds = bounds_array.copy()
        self.origin = origin_array.copy()
        self.basis = basis_array.copy()
        self.initial_origin = self.origin.copy()
        self.initial_basis = self.basis.copy()
        self.gizmo_length = max(float(np.max(self._extent())), 1.0) * 0.90
        self.arc_radius = self.gizmo_length * 0.78
        self.arc_tube_radius = max(self.gizmo_length * 0.018, 0.03)
        for tube in self._arc_tubes:
            tube.SetRadius(self.arc_tube_radius)
            tube.Update()
        center_size = self.gizmo_length * 0.16
        self._center_source.SetXLength(center_size)
        self._center_source.SetYLength(center_size)
        self._center_source.SetZLength(center_size)
        self.update_geometry()

    def update_geometry(self) -> None:
        self._update_arrow_transforms()
        self._update_arc_geometry()
        self._center_source.SetCenter(*self.origin)
        self._center_source.Update()
        half = max(float(np.max(self._extent())) * 0.58, self.gizmo_length * 0.8)
        plane_origin = self.origin - half * self.basis[:, 0] - half * self.basis[:, 1]
        plane_point1 = self.origin + half * self.basis[:, 0] - half * self.basis[:, 1]
        plane_point2 = self.origin - half * self.basis[:, 0] + half * self.basis[:, 1]
        self._plane_source.SetOrigin(*plane_origin)
        self._plane_source.SetPoint1(*plane_point1)
        self._plane_source.SetPoint2(*plane_point2)
        self._plane_source.Update()

    def _update_arrow_transforms(self) -> None:
        for axis, actor in enumerate(self._arrow_actors):
            matrix = vtkMatrix4x4()
            for row in range(3):
                for local_column in range(3):
                    world_axis = (axis + local_column) % 3
                    matrix.SetElement(
                        row,
                        local_column,
                        float(self.basis[row, world_axis] * self.gizmo_length),
                    )
                matrix.SetElement(row, 3, float(self.origin[row]))
            matrix.SetElement(3, 0, 0.0)
            matrix.SetElement(3, 1, 0.0)
            matrix.SetElement(3, 2, 0.0)
            matrix.SetElement(3, 3, 1.0)
            transform = vtkTransform()
            transform.SetMatrix(matrix)
            actor.SetUserTransform(transform)

    def _update_arc_geometry(self) -> None:
        theta = self._rotation_arc_angles()
        for axis, points in enumerate(self._arc_points):
            first = (axis + 1) % 3
            second = (axis + 2) % 3
            for index, angle in enumerate(theta):
                point = self.origin + self.arc_radius * (
                    np.cos(angle) * self.basis[:, first]
                    + np.sin(angle) * self.basis[:, second]
                )
                points.SetPoint(index, *point)
            points.Modified()

    def _world_to_display(self, point: np.ndarray) -> np.ndarray:
        coordinate = self.viewer._world_to_display_coordinate
        coordinate.SetValue(*point)
        value = coordinate.GetComputedDoubleDisplayValue(self.pick_renderer)
        return np.asarray(value, dtype=float)

    def _axis_screen_vector(self, axis: int) -> np.ndarray:
        start = self._world_to_display(self.origin)
        end = self._world_to_display(self.origin + self.basis[:, axis] * self.gizmo_length)
        return end[:2] - start[:2]

    def _screen_rotation_sign(self, axis: int) -> float:
        """Map a positive world-axis rotation to its screen direction.

        A right-handed world rotation can appear clockwise or counterclockwise
        depending on the camera and the axis direction.  Probe one of the
        arc's local radial vectors after a small positive world rotation so a
        mouse drag always keeps the same visual direction.
        """

        center = self._world_to_display(self.origin)[:2]
        probe_angle = 1.0e-3
        rotation = _axis_angle_matrix(self.basis[:, axis], probe_angle)
        for local_column in ((axis + 1) % 3, (axis + 2) % 3):
            radial = self.basis[:, local_column] * self.arc_radius
            before = self._world_to_display(self.origin + radial)[:2] - center
            after = self._world_to_display(self.origin + rotation @ radial)[:2] - center
            cross = float(before[0] * after[1] - before[1] * after[0])
            if abs(cross) > 1.0e-10:
                return 1.0 if cross > 0.0 else -1.0
        # An axis exactly edge-on to the camera has no uniquely visible
        # rotation direction. Keep the right-handed sign as a stable fallback.
        return 1.0

    @staticmethod
    def _highlight_color(color: tuple[float, float, float]) -> tuple[float, float, float]:
        return tuple(min(1.0, 0.5 + 0.5 * value) for value in color)

    def _pick_mode(self, position: np.ndarray) -> tuple[str, int | None] | None:
        self._picker.Pick(
            float(position[0]),
            float(position[1]),
            0.0,
            self.pick_renderer,
        )
        actor = self._picker.GetActor()
        if actor is None:
            return None
        return self._actor_modes.get(self._actor_key(actor))

    def _set_highlight_mode(
        self,
        mode: tuple[str, int | None] | None,
        render: bool = True,
    ) -> None:
        if mode == self._hover_mode:
            return
        self._hover_mode = mode
        for axis, actor in enumerate(self._arrow_actors):
            selected = mode == ("translate", axis)
            color = self._highlight_color(self.AXIS_COLORS[axis]) if selected else self.AXIS_COLORS[axis]
            actor.GetProperty().SetColor(*color)
            actor.GetProperty().SetSpecular(0.85 if selected else 0.2)
            actor.GetProperty().SetSpecularPower(80.0 if selected else 30.0)
        for axis, actor in enumerate(self._arc_actors):
            selected = mode == ("rotate", axis)
            color = self._highlight_color(self.AXIS_COLORS[axis]) if selected else self.AXIS_COLORS[axis]
            actor.GetProperty().SetColor(*color)
            actor.GetProperty().SetSpecular(0.85 if selected else 0.2)
            actor.GetProperty().SetSpecularPower(80.0 if selected else 30.0)
            # Keep the pickable tube geometry unchanged while highlighting.
            # Enlarging the tube would enlarge the hit area after the first
            # hover, even though the picker tolerance is strict.
        if render:
            self.viewer.render()

    def _on_left_press(self, _caller, _event) -> None:
        position = np.asarray(self.interactor.GetEventPosition(), dtype=float)
        mode = self._pick_mode(position)
        self._clear_active_drag()
        # Section mode owns the left button for the entire press interval.
        # A miss must not fall through to TrackballCamera, otherwise the same
        # gesture rotates the camera instead of leaving the section unchanged.
        self._active_mode = mode if mode is not None else self.BLOCK_LEFT_VIEW_MODE
        self._last_mouse = position
        self._set_highlight_mode(mode)

    def _on_move(self, _caller, _event) -> None:
        current = np.asarray(self.interactor.GetEventPosition(), dtype=float)
        if self._active_mode is None or self._last_mouse is None:
            self._set_highlight_mode(self._pick_mode(current))
            return
        previous = self._last_mouse
        delta = current - previous
        mode, axis = self._active_mode
        if mode == self.BLOCK_LEFT_VIEW_MODE[0]:
            self._last_mouse = current
            return
        if axis is not None and mode == "translate":
            screen_axis = self._axis_screen_vector(axis)
            denominator = float(np.dot(screen_axis, screen_axis))
            if denominator > 1e-8:
                amount = float(np.dot(delta[:2], screen_axis) / denominator * self.gizmo_length)
                self.origin += self.basis[:, axis] * amount
        elif axis is not None and mode == "rotate":
            center = self._world_to_display(self.origin)[:2]
            before = previous[:2] - center
            after = current[:2] - center
            if np.linalg.norm(before) > 2.0 and np.linalg.norm(after) > 2.0:
                screen_angle = _signed_screen_angle(before, after)
                angle = screen_angle * self._screen_rotation_sign(axis)
                rotation = _axis_angle_matrix(self.basis[:, axis], angle)
                self.basis = rotation @ self.basis
                self.basis = np.column_stack(
                    [_unit_vector(self.basis[:, column], f"gizmo axis {column}") for column in range(3)]
                )
        self._last_mouse = current
        self.update_geometry()
        self.changed_callback()

    def _clear_active_drag(self) -> None:
        """Clear the temporary left-button selection/drag state."""

        self._active_mode = None
        self._last_mouse = None

    def _on_left_release(self, _caller, _event) -> None:
        # Always process the release event.  VTK can deliver a release after
        # a press that did not hit a handle, and that event is also the
        # definitive cancellation point for a handle drag.
        self._clear_active_drag()
        position = np.asarray(self.interactor.GetEventPosition(), dtype=float)
        self._set_highlight_mode(self._pick_mode(position))

    def _on_leave(self, _caller, _event) -> None:
        if self._active_mode is None:
            self._set_highlight_mode(None)


class InteractiveSectionViewer(PyVistaRenderer):
    """PyVista renderer with a mouse-driven three-axis section gizmo."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._section_gizmo: _SectionGizmo | None = None
        self._field_plane_gizmo: _SectionGizmo | None = None
        self._primitive_gizmo: _SectionGizmo | None = None
        self._cell_map_gizmo: _SectionGizmo | None = None
        self._primitive_changed_callback: Callable[[np.ndarray, np.ndarray], None] | None = None
        self._field_plane_changed_callback: Callable[[np.ndarray, np.ndarray], None] | None = None
        self._cell_map_changed_callback: Callable[[np.ndarray, np.ndarray], None] | None = None
        self._field_surface_pick_callback: Callable[[np.ndarray, np.ndarray], None] | None = None
        self._field_pick_press_position: np.ndarray | None = None
        self._scene_mesh_pick_callback: Callable[[object], None] | None = None
        self._scene_mesh_pick_press_position: np.ndarray | None = None
        self._scene_mesh_picker = vtkCellPicker()
        self._scene_mesh_picker.SetTolerance(1.0e-5)

        coordinate = __import__("vtkmodules.vtkRenderingCore", fromlist=["vtkCoordinate"])
        self._world_to_display_coordinate = coordinate.vtkCoordinate()
        self._world_to_display_coordinate.SetCoordinateSystemToWorld()
        self.interactor.AddObserver(
            vtkCommand.RightButtonPressEvent,
            self._on_field_pick_press,
            1.5,
        )
        self.interactor.AddObserver(
            vtkCommand.RightButtonReleaseEvent,
            self._on_field_pick_release,
            1.5,
        )
        self.interactor.AddObserver(
            vtkCommand.LeftButtonPressEvent,
            self._on_scene_mesh_pick_press,
            0.4,
        )
        self.interactor.AddObserver(
            vtkCommand.LeftButtonReleaseEvent,
            self._on_scene_mesh_pick_release,
            0.4,
        )

    def close(self) -> None:
        """Disconnect custom observers before the render window is finalized."""

        if getattr(self, "_closed", False):
            return
        self.disable_interactive_section()
        self.disable_field_plane_gizmo()
        self.disable_primitive_transform()
        self.disable_cell_map_transform()
        self._primitive_changed_callback = None
        self._field_plane_changed_callback = None
        self._cell_map_changed_callback = None
        self._field_surface_pick_callback = None
        self._scene_mesh_pick_callback = None
        super().close()
        self._scene_mesh_picker = None
        self._world_to_display_coordinate = None

    def enable_interactive_section(
        self,
        bounds: np.ndarray | None = None,
        origin: np.ndarray | None = None,
        normal: np.ndarray | None = None,
    ) -> None:
        """Create the draggable XYZ translation and rotation gizmo."""

        self.disable_field_plane_gizmo()
        self.disable_primitive_transform()
        self.disable_cell_map_transform()
        self.disable_interactive_section()
        bounds_array = self._resolve_bounds(bounds)
        default_origin = np.array(
            (
                (bounds_array[0] + bounds_array[1]) * 0.5,
                (bounds_array[2] + bounds_array[3]) * 0.5,
                (bounds_array[4] + bounds_array[5]) * 0.5,
            ),
            dtype=float,
        )
        origin_array = default_origin if origin is None else np.asarray(origin, dtype=float).reshape(3)
        if not np.isfinite(origin_array).all():
            raise ValueError("section origin must contain three finite values")
        normal_array = np.array((0.0, 0.0, 1.0)) if normal is None else _unit_vector(normal, "section normal")
        self._section_gizmo = _SectionGizmo(
            self,
            bounds_array,
            origin_array,
            normal_array,
            self._on_gizmo_changed,
        )
        self._section_gizmo.update_geometry()
        self._on_gizmo_changed()

    def disable_interactive_section(self) -> None:
        """Remove the gizmo and restore the complete displayed mesh."""

        if self._section_gizmo is not None:
            self._section_gizmo.disconnect()
            self._section_gizmo = None
        self.clear_section_box()

    def reset_interactive_section(self) -> None:
        """Restore the initial gizmo position and orientation."""

        if self._section_gizmo is not None:
            self._section_gizmo.reset()

    def set_field_surface_pick_callback(
        self,
        callback: Callable[[np.ndarray, np.ndarray], None] | None,
    ) -> None:
        """Register a right-click surface callback while Field Viewer is active."""

        self._field_surface_pick_callback = callback
        self._field_pick_press_position = None

    def set_scene_mesh_pick_callback(
        self,
        callback: Callable[[object], None] | None,
    ) -> None:
        """Select a persistent scene mesh after a short left click."""

        self._scene_mesh_pick_callback = callback
        self._scene_mesh_pick_press_position = None

    def enable_field_plane_gizmo(
        self,
        bounds: np.ndarray,
        origin: np.ndarray,
        u_axis: np.ndarray,
        v_axis: np.ndarray,
        changed_callback: Callable[[np.ndarray, np.ndarray], None],
    ) -> None:
        """Show a non-clipping XYZ plane manipulator for Field Viewer."""

        if not callable(changed_callback):
            raise TypeError("changed_callback must be callable")
        self.disable_interactive_section()
        self.disable_primitive_transform()
        self.disable_cell_map_transform()
        self.disable_field_plane_gizmo()
        bounds_array = self._resolve_bounds(bounds)
        origin_array = np.asarray(origin, dtype=float).reshape(3)
        u = _unit_vector(u_axis, "field-plane U axis")
        v_candidate = np.asarray(v_axis, dtype=float).reshape(3)
        v_candidate -= u * float(np.dot(v_candidate, u))
        v = _unit_vector(v_candidate, "field-plane V axis")
        normal = _unit_vector(np.cross(u, v), "field-plane normal")
        basis = np.column_stack((u, v, normal))
        self._field_plane_changed_callback = changed_callback
        self._field_plane_gizmo = _SectionGizmo(
            self,
            bounds_array,
            origin_array,
            normal,
            self._on_field_plane_gizmo_changed,
        )
        # The colored scalar plane is already the Field Viewer reference
        # plane.  Do not stack the section viewer's translucent orange helper
        # plane on top of it.
        self._field_plane_gizmo._plane_actor.SetVisibility(False)
        self._field_plane_gizmo.basis = basis.copy()
        self._field_plane_gizmo.initial_basis = basis.copy()
        self._field_plane_gizmo.update_geometry()
        self._on_field_plane_gizmo_changed()

    def disable_field_plane_gizmo(self) -> None:
        """Remove only the Field Viewer manipulator and keep scene geometry."""

        if self._field_plane_gizmo is not None:
            self._field_plane_gizmo.disconnect()
            self._field_plane_gizmo = None
        self._field_plane_changed_callback = None

    def reset_field_plane_gizmo(self) -> None:
        """Restore the Field Viewer plane manipulator's initial transform."""

        if self._field_plane_gizmo is not None:
            self._field_plane_gizmo.reset()

    def enable_primitive_transform(
        self,
        bounds: np.ndarray,
        origin: np.ndarray,
        basis: np.ndarray,
        changed_callback: Callable[[np.ndarray, np.ndarray], None],
    ) -> None:
        """Show the strict-drag transform gizmo for one field object."""

        if not callable(changed_callback):
            raise TypeError("changed_callback must be callable")
        self.disable_interactive_section()
        self.disable_field_plane_gizmo()
        self.disable_cell_map_transform()
        bounds_array = self._resolve_bounds(bounds)
        origin_array = np.asarray(origin, dtype=float).reshape(3)
        basis_array = np.asarray(basis, dtype=float)
        if basis_array.shape != (3, 3):
            raise ValueError("primitive basis must have shape (3, 3)")
        normal = _unit_vector(basis_array[:, 2], "primitive Z axis")
        self._primitive_changed_callback = changed_callback
        if self._primitive_gizmo is not None:
            self._primitive_gizmo.changed_callback = self._on_primitive_gizmo_changed
            self._primitive_gizmo.synchronize_transform(
                bounds_array,
                origin_array,
                basis_array,
            )
            self._primitive_gizmo._plane_actor.SetVisibility(False)
            self._primitive_gizmo.add_to_scene()
            self.render()
            return
        self._primitive_gizmo = _SectionGizmo(
            self,
            bounds_array,
            origin_array,
            normal,
            self._on_primitive_gizmo_changed,
        )
        self._primitive_gizmo._plane_actor.SetVisibility(False)
        self._primitive_gizmo.basis = basis_array.copy()
        self._primitive_gizmo.initial_basis = basis_array.copy()
        self._primitive_gizmo.update_geometry()
        # Creating a manipulator is a view-state synchronization, not a user
        # edit.  Calling the change callback here re-enters the window's
        # scene-refresh path for analytic design domains.
        self._primitive_gizmo.add_to_scene()
        self.render()

    def disable_primitive_transform(self) -> None:
        """Remove the field-object manipulator without affecting geometry."""

        if self._primitive_gizmo is not None:
            self._primitive_gizmo.disconnect()
            self._primitive_gizmo = None
        self._primitive_changed_callback = None

    def reset_primitive_transform(self) -> None:
        if self._primitive_gizmo is not None:
            self._primitive_gizmo.reset()

    def enable_cell_map_transform(
        self,
        bounds: np.ndarray,
        origin: np.ndarray,
        basis: np.ndarray,
        changed_callback: Callable[[np.ndarray, np.ndarray], None],
    ) -> None:
        """Show a foreground UVW transform gizmo for the current Cell Map."""

        if not callable(changed_callback):
            raise TypeError("changed_callback must be callable")
        self.disable_interactive_section()
        self.disable_field_plane_gizmo()
        self.disable_primitive_transform()
        self.disable_cell_map_transform()
        bounds_array = self._resolve_bounds(bounds)
        origin_array = np.asarray(origin, dtype=float).reshape(3)
        basis_array = np.asarray(basis, dtype=float)
        if basis_array.shape != (3, 3) or not np.isfinite(basis_array).all():
            raise ValueError("Cell Map basis must be a finite 3 by 3 matrix")
        normal = _unit_vector(basis_array[:, 2], "Cell Map W axis")
        self._cell_map_changed_callback = changed_callback
        self._cell_map_gizmo = _SectionGizmo(
            self,
            bounds_array,
            origin_array,
            normal,
            self._on_cell_map_gizmo_changed,
        )
        self._cell_map_gizmo._plane_actor.SetVisibility(False)
        self._cell_map_gizmo.basis = basis_array.copy()
        self._cell_map_gizmo.initial_basis = basis_array.copy()
        self._cell_map_gizmo.update_geometry()
        self._cell_map_gizmo.add_to_scene()
        self.render()

    def disable_cell_map_transform(self) -> None:
        """Remove the Cell Map manipulator without changing its parameters."""

        if self._cell_map_gizmo is not None:
            self._cell_map_gizmo.disconnect()
            self._cell_map_gizmo = None
        self._cell_map_changed_callback = None

    def reset_cell_map_transform(self) -> None:
        if self._cell_map_gizmo is not None:
            self._cell_map_gizmo.reset()

    def display_ray(self, display_position: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Map one display-space position to a finite world-space camera ray."""

        position = np.asarray(display_position, dtype=float).reshape(2)
        renderer = self.renderer
        endpoints = []
        for depth in (0.0, 1.0):
            renderer.SetDisplayPoint(float(position[0]), float(position[1]), depth)
            renderer.DisplayToWorld()
            world = np.asarray(renderer.GetWorldPoint(), dtype=float)
            if abs(float(world[3])) <= np.finfo(float).eps:
                raise RuntimeError("unable to resolve display ray")
            endpoints.append(world[:3] / world[3])
        direction = _unit_vector(endpoints[1] - endpoints[0], "display ray")
        return endpoints[0], direction

    def _on_field_pick_press(self, _caller, _event) -> None:
        if self._field_surface_pick_callback is None:
            self._field_pick_press_position = None
            return
        self._field_pick_press_position = np.asarray(
            self.interactor.GetEventPosition(),
            dtype=float,
        )

    def _on_scene_mesh_pick_press(self, _caller, _event) -> None:
        if self._scene_mesh_pick_callback is None:
            self._scene_mesh_pick_press_position = None
            return
        self._scene_mesh_pick_press_position = np.asarray(
            self.interactor.GetEventPosition(),
            dtype=float,
        )

    def _on_scene_mesh_pick_release(self, _caller, _event) -> None:
        callback = self._scene_mesh_pick_callback
        pressed = self._scene_mesh_pick_press_position
        self._scene_mesh_pick_press_position = None
        if callback is None or pressed is None:
            return
        released = np.asarray(self.interactor.GetEventPosition(), dtype=float)
        if float(np.linalg.norm(released - pressed)) > 3.0:
            return
        try:
            self._scene_mesh_picker.Pick(
                float(released[0]),
                float(released[1]),
                0.0,
                self.renderer,
            )
            actor = self._scene_mesh_picker.GetActor()
            if actor is None:
                return
            for key, record in self._mesh_layer_records.items():
                if self._same_vtk_actor(actor, record.actor):
                    callback(key)
                    return
        except (AttributeError, RuntimeError):
            return

    def _on_field_pick_release(self, _caller, _event) -> None:
        callback = self._field_surface_pick_callback
        press_position = self._field_pick_press_position
        self._field_pick_press_position = None
        if callback is None or press_position is None:
            return
        release_position = np.asarray(self.interactor.GetEventPosition(), dtype=float)
        if float(np.linalg.norm(release_position - press_position)) > 3.0:
            return
        try:
            origin, direction = self.display_ray(release_position)
            callback(origin, direction)
        except (RuntimeError, ValueError, AttributeError):
            return

    def _on_field_plane_gizmo_changed(self) -> None:
        gizmo = self._field_plane_gizmo
        callback = self._field_plane_changed_callback
        if gizmo is None or callback is None:
            return
        gizmo.add_to_scene()
        callback(gizmo.origin.copy(), gizmo.basis.copy())
        self.render()

    def _on_primitive_gizmo_changed(self) -> None:
        gizmo = self._primitive_gizmo
        callback = self._primitive_changed_callback
        if gizmo is None or callback is None:
            return
        gizmo.add_to_scene()
        callback(gizmo.origin.copy(), gizmo.basis.copy())
        self.render()

    def _on_cell_map_gizmo_changed(self) -> None:
        gizmo = self._cell_map_gizmo
        callback = self._cell_map_changed_callback
        if gizmo is None or callback is None:
            return
        gizmo.add_to_scene()
        callback(gizmo.origin.copy(), gizmo.basis.copy())
        self.render()

    def _resolve_bounds(self, bounds: np.ndarray | None) -> np.ndarray:
        if bounds is not None:
            values = np.asarray(bounds, dtype=float).reshape(-1)
            if values.shape != (6,) or not np.isfinite(values).all():
                raise ValueError("section bounds must contain six finite values")
            if not (values[0] < values[1] and values[2] < values[3] and values[4] < values[5]):
                raise ValueError("section bounds must have positive extents")
            return values
        if not self.meshes and self.implicit_fields:
            scene_bounds = np.asarray(self._scene_bounds, dtype=float)
            return np.array(
                (
                    scene_bounds[0, 0], scene_bounds[1, 0],
                    scene_bounds[0, 1], scene_bounds[1, 1],
                    scene_bounds[0, 2], scene_bounds[1, 2],
                ),
                dtype=float,
            )
        if not self.meshes:
            raise ValueError("load a mesh or implicit field before enabling the section gizmo")
        vertices = np.concatenate(
            [mesh.vertices for mesh in self.meshes if len(mesh.vertices)],
            axis=0,
        )
        if len(vertices) == 0:
            raise ValueError("the displayed meshes contain no vertices")
        return np.array(
            (
                vertices[:, 0].min(),
                vertices[:, 0].max(),
                vertices[:, 1].min(),
                vertices[:, 1].max(),
                vertices[:, 2].min(),
                vertices[:, 2].max(),
            ),
            dtype=float,
        )

    def _on_gizmo_changed(self) -> None:
        if self._section_gizmo is None:
            return
        plane = vtkPlane()
        plane.SetOrigin(*self._section_gizmo.origin)
        plane.SetNormal(*self._section_gizmo.basis[:, 2])
        self.set_section_plane(plane)
        self._section_gizmo.add_to_scene()
        self.render()


class SectionViewerWindow(QtWidgets.QMainWindow):
    """Small host window for the standalone interactive section viewer."""

    def __init__(self, mesh_path: Path):
        super().__init__()
        self.setWindowTitle("Interactive STL Section Gizmo")
        self.resize(1400, 900)
        self.viewer = InteractiveSectionViewer(self)
        self.viewer.show_ground_plane = False
        self.viewer.show_ground_grid = False
        self.viewer.enable_pbr = True
        self.viewer.enable_ssao_flag = True
        self.viewer.set_background("#FFFFFF")

        mesh = load_triangle_mesh(mesh_path)
        self._mesh_bounds = vtk_bounds_from_mesh(mesh)
        self.viewer.set_meshes(
            [
                MeshData(
                    mesh.vertices,
                    mesh.faces,
                    color=(0.72, 0.76, 0.82, 1.0),
                    metallic=0.15,
                    roughness=0.42,
                )
            ],
            reset_view=True,
        )
        self.viewer.enable_interactive_section(self._mesh_bounds)

        root = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(root)
        toolbar = QtWidgets.QHBoxLayout()
        enable = QtWidgets.QPushButton("启用交互剖切")
        enable.clicked.connect(lambda: self.viewer.enable_interactive_section(self._mesh_bounds))
        reset = QtWidgets.QPushButton("恢复剖切姿态")
        reset.clicked.connect(self.viewer.reset_interactive_section)
        disable = QtWidgets.QPushButton("显示完整模型")
        disable.clicked.connect(self.viewer.disable_interactive_section)
        toolbar.addWidget(enable)
        toolbar.addWidget(reset)
        toolbar.addWidget(disable)
        toolbar.addStretch()
        layout.addLayout(toolbar)
        hint = QtWidgets.QLabel(
            "左键命中并按住红/绿/蓝箭头可沿 X/Y/Z 轴移动，按住对应颜色弧线可绕该轴旋转；"
            "右键拖动旋转视角，中键拖动平移，滚轮缩放。剖切面正法向一侧被隐藏。"
        )
        hint.setWordWrap(True)
        layout.addWidget(hint)
        layout.addWidget(self.viewer, 1)
        self.setCentralWidget(root)


def _resolve_mesh_argument(path: Path) -> Path:
    candidate = path if path.is_absolute() else Path.cwd() / path
    if candidate.is_file():
        return candidate.resolve()
    project_candidate = PROJECT_ROOT / path
    if project_candidate.is_file():
        return project_candidate.resolve()
    return candidate.resolve()


def main() -> int:
    parser = argparse.ArgumentParser(description="Mouse-driven XYZ section gizmo for STL meshes")
    parser.add_argument(
        "--mesh",
        type=Path,
        default=(
            PROJECT_ROOT
            / "resources"
            / "examples"
            / "design_domains"
            / "sole-1.stl"
        ),
        help="STL file to inspect (default: resources/examples/design_domains/sole-1.stl)",
    )
    args = parser.parse_args()
    mesh_path = _resolve_mesh_argument(args.mesh)
    if not mesh_path.is_file():
        parser.error(f"mesh does not exist: {mesh_path}")

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = SectionViewerWindow(mesh_path)
    window.show()
    return app.exec_()


if __name__ == "__main__":
    raise SystemExit(main())
