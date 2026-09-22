"""Regression tests for the interactive section gizmo mouse state."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest import mock

import numpy as np
from pyvista.plotting.render_window_interactor import RenderWindowInteractor
from vtkmodules.vtkCommonCore import vtkCommand
from vtkmodules.vtkRenderingCore import vtkRenderer, vtkRenderWindow, vtkRenderWindowInteractor

TESTS_DIR = Path(__file__).resolve().parent
if str(TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(TESTS_DIR))

from lattice_studio.presentation.qt.tools import interactive_section_viewer as viewer_module
from lattice_studio.presentation.qt.tools.interactive_section_viewer import (
    InteractiveSectionViewer,
    PyVistaRenderer,
    _SectionGizmo,
    _signed_screen_angle,
)


class _FakeStyle:
    def __init__(self) -> None:
        self.enabled = 1
        self.removed_observers = []

    def GetEnabled(self) -> int:
        return self.enabled

    def SetEnabled(self, value: int) -> None:
        self.enabled = int(value)

    def RemoveObservers(self, event) -> None:
        self.removed_observers.append(event)


class _FakeInteractor:
    def __init__(self) -> None:
        self.position = (20, 30)
        self.style = _FakeStyle()
        self.observers = []
        self.style_history = []

    def GetEventPosition(self):
        return self.position

    def GetInteractorStyle(self):
        return self.style

    def SetInteractorStyle(self, style) -> None:
        self.style_history.append(style)
        self.style = style

    def AddObserver(self, event, callback, priority):
        self.observers.append((event, callback, priority))
        return len(self.observers)


class _FakeProperty:
    def SetColor(self, *_color) -> None:
        pass

    def SetSpecular(self, *_value) -> None:
        pass

    def SetSpecularPower(self, *_value) -> None:
        pass


class _FakeActor:
    def __init__(self) -> None:
        self.property = _FakeProperty()

    def GetProperty(self):
        return self.property


def _make_gizmo():
    gizmo = object.__new__(_SectionGizmo)
    gizmo.interactor = _FakeInteractor()
    gizmo._active_mode = None
    gizmo._last_mouse = None
    gizmo._hover_mode = None
    gizmo.highlight_history = []
    gizmo.pick_mode = ("translate", 0)
    gizmo._pick_mode = lambda _position: gizmo.pick_mode
    gizmo._set_highlight_mode = lambda mode: gizmo.highlight_history.append(mode)
    return gizmo


def test_left_press_locks_only_until_left_release():
    gizmo = _make_gizmo()

    gizmo._on_left_press(None, None)

    assert gizmo._active_mode == ("translate", 0)
    assert np.array_equal(gizmo._last_mouse, np.array((20.0, 30.0)))
    assert gizmo.interactor.style.enabled == 1

    gizmo._on_left_release(None, None)

    assert gizmo._active_mode is None
    assert gizmo._last_mouse is None
    assert gizmo.interactor.style.enabled == 1
    assert gizmo.highlight_history[-1] == ("translate", 0)


def test_left_press_away_from_handle_does_not_lock_or_leave_highlight():
    gizmo = _make_gizmo()
    gizmo._hover_mode = ("translate", 0)
    gizmo.pick_mode = None

    gizmo._on_left_press(None, None)

    assert gizmo._active_mode == _SectionGizmo.BLOCK_LEFT_VIEW_MODE
    assert np.array_equal(gizmo._last_mouse, np.array((20.0, 30.0)))
    assert gizmo.interactor.style.enabled == 1
    assert gizmo.highlight_history[-1] is None

    gizmo._on_left_release(None, None)

    assert gizmo._active_mode is None
    assert gizmo._last_mouse is None
    assert gizmo.interactor.style.enabled == 1


def test_picker_tolerance_is_strict():
    assert _SectionGizmo.PICK_TOLERANCE == 1.0e-6


def test_rotation_handles_are_complete_closed_rings():
    """Rotation handles must not deliberately omit a visible arc segment."""

    angles = _SectionGizmo._rotation_arc_angles()

    assert len(angles) == _SectionGizmo.ROTATION_ARC_SEGMENTS + 1
    assert np.isclose(angles[0], 0.0)
    assert np.isclose(angles[-1], 2.0 * np.pi)


def test_highlight_does_not_enlarge_pickable_arc_geometry():
    gizmo = _make_gizmo()
    gizmo._arrow_actors = [_FakeActor(), _FakeActor(), _FakeActor()]
    gizmo._arc_actors = [_FakeActor(), _FakeActor(), _FakeActor()]

    class _Tube:
        def __init__(self) -> None:
            self.radius_changes = 0

        def SetRadius(self, *_value) -> None:
            self.radius_changes += 1

        def Update(self) -> None:
            self.radius_changes += 1

    gizmo._arc_tubes = [_Tube(), _Tube(), _Tube()]
    _SectionGizmo._set_highlight_mode(gizmo, ("rotate", 1), render=False)

    assert [tube.radius_changes for tube in gizmo._arc_tubes] == [0, 0, 0]


def test_signed_screen_angle_follows_drag_direction():
    assert _signed_screen_angle((1.0, 0.0), (0.0, 1.0)) > 0.0
    assert _signed_screen_angle((1.0, 0.0), (0.0, -1.0)) < 0.0


def test_primitive_transform_sync_reuses_existing_gizmo():
    """Parameter edits must not churn native VTK renderers and observers."""

    class _PlaneActor:
        def SetVisibility(self, _visible) -> None:
            pass

    class _ReusableGizmo:
        instances = []

        def __init__(self, _viewer, bounds, origin, _normal, callback) -> None:
            self.bounds = np.asarray(bounds, dtype=float).copy()
            self.origin = np.asarray(origin, dtype=float).copy()
            self.basis = np.eye(3)
            self.initial_origin = self.origin.copy()
            self.initial_basis = self.basis.copy()
            self.changed_callback = callback
            self._plane_actor = _PlaneActor()
            self.sync_calls = 0
            self.disconnected = False
            self.__class__.instances.append(self)

        def synchronize_transform(self, bounds, origin, basis) -> None:
            self.bounds = np.asarray(bounds, dtype=float).copy()
            self.origin = np.asarray(origin, dtype=float).copy()
            self.basis = np.asarray(basis, dtype=float).copy()
            self.initial_origin = self.origin.copy()
            self.initial_basis = self.basis.copy()
            self.sync_calls += 1

        def update_geometry(self) -> None:
            pass

        def add_to_scene(self) -> None:
            pass

        def disconnect(self) -> None:
            self.disconnected = True

    class _ViewerHarness:
        def __init__(self) -> None:
            self._primitive_gizmo = None
            self._primitive_changed_callback = None
            self.render_count = 0

        def disable_interactive_section(self) -> None:
            pass

        def disable_field_plane_gizmo(self) -> None:
            pass

        def disable_cell_map_transform(self) -> None:
            pass

        def disable_primitive_transform(self) -> None:
            InteractiveSectionViewer.disable_primitive_transform(self)

        def _on_primitive_gizmo_changed(self) -> None:
            InteractiveSectionViewer._on_primitive_gizmo_changed(self)

        def _resolve_bounds(self, bounds):
            return np.asarray(bounds, dtype=float)

        def render(self) -> None:
            self.render_count += 1

    viewer = _ViewerHarness()
    first_bounds = np.array((-5.0, 5.0, -5.0, 5.0, -5.0, 5.0))
    second_bounds = np.array((-8.0, 8.0, -8.0, 8.0, -8.0, 8.0))
    second_origin = np.array((1.0, 2.0, 3.0))
    second_basis = np.array(
        ((0.0, -1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0))
    )

    with mock.patch.object(viewer_module, "_SectionGizmo", _ReusableGizmo):
        InteractiveSectionViewer.enable_primitive_transform(
            viewer,
            first_bounds,
            np.zeros(3),
            np.eye(3),
            lambda *_args: None,
        )
        first_gizmo = viewer._primitive_gizmo
        InteractiveSectionViewer.enable_primitive_transform(
            viewer,
            second_bounds,
            second_origin,
            second_basis,
            lambda *_args: None,
        )

    assert viewer._primitive_gizmo is first_gizmo
    assert len(_ReusableGizmo.instances) == 1
    assert first_gizmo.sync_calls == 1
    assert first_gizmo.disconnected is False
    np.testing.assert_allclose(first_gizmo.bounds, second_bounds)
    np.testing.assert_allclose(first_gizmo.origin, second_origin)
    np.testing.assert_allclose(first_gizmo.basis, second_basis)


def test_world_rotation_sign_tracks_camera_screen_orientation():
    gizmo = object.__new__(_SectionGizmo)
    gizmo.origin = np.zeros(3)
    gizmo.basis = np.eye(3)
    gizmo.arc_radius = 1.0
    gizmo._world_to_display = lambda point: np.asarray((point[0], point[1], 0.0))

    assert _SectionGizmo._screen_rotation_sign(gizmo, 2) == 1.0

    gizmo._world_to_display = lambda point: np.asarray((point[0], -point[1], 0.0))
    assert _SectionGizmo._screen_rotation_sign(gizmo, 2) == -1.0


def test_renderer_navigation_maps_right_to_rotate_and_middle_to_pan():
    class _NavigationHarness:
        def __init__(self):
            self.interactor = _FakeInteractor()
            self._left_navigation_style = None
            self._left_navigation_suspended = False
            self._suspend_left_camera_navigation = (
                lambda caller, event: PyVistaRenderer._suspend_left_camera_navigation(self, caller, event)
            )
            self._restore_left_camera_navigation = (
                lambda caller, event: PyVistaRenderer._restore_left_camera_navigation(self, caller, event)
            )

    viewer = _NavigationHarness()
    calls = []
    viewer.enable_custom_trackball_style = lambda **kwargs: calls.append(kwargs)

    PyVistaRenderer._configure_navigation_controls(viewer)
    assert calls[0]["right"] == "rotate"
    assert calls[0]["middle"] == "pan"
    assert len(viewer.interactor.observers) == 4
    observed_events = [item[0] for item in viewer.interactor.observers]
    assert "LeftButtonPressEvent" in observed_events
    assert "LeftButtonReleaseEvent" in observed_events
    assert "MouseWheelForwardEvent" in observed_events
    assert "MouseWheelBackwardEvent" in observed_events
    assert all(
        observer[2] == 2.0
        for observer in viewer.interactor.observers
        if observer[0] in {
            "LeftButtonPressEvent",
            "LeftButtonReleaseEvent",
            "MouseWheelForwardEvent",
            "MouseWheelBackwardEvent",
        }
    )
    assert all(observer[2] == 2.0 for observer in viewer.interactor.observers)
    assert viewer.interactor.style.removed_observers == [
        "LeftButtonPressEvent",
        "LeftButtonReleaseEvent",
        "MouseWheelForwardEvent",
        "MouseWheelBackwardEvent",
    ]

    original_style = viewer.interactor.style
    viewer.interactor.observers[0][1](viewer.interactor, "LeftButtonPressEvent")
    assert viewer.interactor.style is None

    viewer.interactor.observers[1][1](viewer.interactor, "LeftButtonReleaseEvent")
    assert viewer.interactor.style is original_style


def test_left_drag_does_not_move_camera_in_vtk_event_loop():
    class _PlotterHarness:
        click_position = None
        renderers = []

        def store_click_position(self) -> None:
            self.click_position = (0, 0)

    class _NavigationHarness(RenderWindowInteractor):
        _suspend_left_camera_navigation = PyVistaRenderer._suspend_left_camera_navigation
        _restore_left_camera_navigation = PyVistaRenderer._restore_left_camera_navigation

    def invoke(iren, event, x, y):
        iren.SetEventInformation(x, y, 0, 0, "a", 0, "")
        iren.InvokeEvent(event)

    renderer = vtkRenderer()
    renderer.GetActiveCamera().SetPosition(0.0, 0.0, 10.0)
    renderer.GetActiveCamera().SetFocalPoint(0.0, 0.0, 0.0)
    window = vtkRenderWindow()
    window.SetOffScreenRendering(1)
    window.SetSize(400, 400)
    window.AddRenderer(renderer)
    interactor = vtkRenderWindowInteractor()
    interactor.SetRenderWindow(window)
    plotter = _PlotterHarness()
    plotter.renderers = [renderer]
    viewer = _NavigationHarness(plotter, interactor=interactor)
    viewer._left_navigation_style = None
    viewer._left_navigation_suspended = False

    PyVistaRenderer._configure_navigation_controls(viewer)
    left_observer_styles = []
    interactor.AddObserver(
        "LeftButtonPressEvent",
        lambda caller, _event: left_observer_styles.append(caller.GetInteractorStyle()),
        1.0,
    )
    interactor.Initialize()
    original_style = interactor.GetInteractorStyle()
    original_position = tuple(renderer.GetActiveCamera().GetPosition())

    invoke(interactor, vtkCommand.LeftButtonPressEvent, 100, 100)
    assert interactor.GetInteractorStyle() is None
    assert left_observer_styles == [None]
    invoke(interactor, vtkCommand.MouseMoveEvent, 160, 130)
    assert tuple(renderer.GetActiveCamera().GetPosition()) == original_position
    invoke(interactor, vtkCommand.LeftButtonReleaseEvent, 160, 130)
    assert interactor.GetInteractorStyle() is original_style

    right_start_position = tuple(renderer.GetActiveCamera().GetPosition())
    invoke(interactor, vtkCommand.RightButtonPressEvent, 100, 100)
    invoke(interactor, vtkCommand.MouseMoveEvent, 160, 130)
    assert not np.allclose(renderer.GetActiveCamera().GetPosition(), right_start_position)
    invoke(interactor, vtkCommand.RightButtonReleaseEvent, 160, 130)
