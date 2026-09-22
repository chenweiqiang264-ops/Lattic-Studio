"""UI regressions for interactive Cell Map frame manipulation."""

from __future__ import annotations

from unittest import mock

import numpy as np
import pytest
import trimesh
from PyQt5 import QtWidgets

from lattice_studio.presentation.qt.tools import interactive_section_viewer
from lattice_studio.presentation.qt.workbench import _build_qt_app
from tests.integration.implicit.test_field_viewer_primitive import _FakeViewer


class _FakeCellMapGizmo:
    def __init__(self, origin: np.ndarray, basis: np.ndarray) -> None:
        self.origin = np.asarray(origin, dtype=np.float64).copy()
        self.basis = np.asarray(basis, dtype=np.float64).copy()

    def update_geometry(self) -> None:
        return None

    def add_to_scene(self) -> None:
        return None


class _CellMapViewer(_FakeViewer):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._cell_map_preview = None
        self._cell_map_gizmo = None
        self.cell_map_callback = None

    def set_cell_map_preview(self, cell_map, _domain, reset_view=False) -> None:
        self._cell_map_preview = cell_map

    def enable_cell_map_transform(
        self,
        _bounds,
        origin,
        basis,
        callback,
    ) -> None:
        self._cell_map_gizmo = _FakeCellMapGizmo(origin, basis)
        self.cell_map_callback = callback

    def disable_cell_map_transform(self) -> None:
        self._cell_map_gizmo = None
        self.cell_map_callback = None


@pytest.fixture(scope="module")
def application():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.fixture
def window(application):
    with mock.patch.object(
        interactive_section_viewer,
        "InteractiveSectionViewer",
        _CellMapViewer,
    ):
        widget = _build_qt_app()()
        widget.sole_mesh = trimesh.creation.box(extents=(100.0, 100.0, 20.0))
        yield widget
        widget.close()
        application.processEvents()


def test_interactive_cell_map_updates_origin_and_uvw_axes(window) -> None:
    window.cell_map_interactive.setChecked(True)

    assert window.viewer._cell_map_preview is not None
    assert window.viewer._cell_map_gizmo is not None
    assert window.viewer.cell_map_callback is not None
    np.testing.assert_allclose(
        window.viewer._cell_map_gizmo.basis,
        window.viewer._cell_map_preview.frame.axes.T,
    )

    origin = np.array((5.0, 6.0, 7.0))
    basis = np.array(
        (
            (0.0, -1.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 0.0, 1.0),
        )
    )
    window.viewer.cell_map_callback(origin, basis)

    widgets = window.tpms_widgets
    assert not widgets["frame_origin_follow"].isChecked()
    np.testing.assert_allclose(
        [control.value() for control in widgets["frame_inputs"]["origin"]],
        origin,
    )
    np.testing.assert_allclose(
        [control.value() for control in widgets["frame_inputs"]["u"]],
        basis[:, 0],
    )
    np.testing.assert_allclose(
        [control.value() for control in widgets["frame_inputs"]["v"]],
        basis[:, 1],
    )
    parameters = window._params("G")
    np.testing.assert_allclose(parameters.frame_origin_mm, origin)
    np.testing.assert_allclose(parameters.frame_u_axis, basis[:, 0])
    np.testing.assert_allclose(parameters.frame_v_axis, basis[:, 1])
    np.testing.assert_allclose(window.viewer._cell_map_preview.frame.origin, origin)
    np.testing.assert_allclose(window.viewer._cell_map_preview.frame.axes.T, basis)


def test_interactive_cell_map_switches_shared_tpms_editor_and_resets(window) -> None:
    window.cell_map_interactive.setChecked(True)
    window.cell_map_kind.setCurrentIndex(window.cell_map_kind.findData("D"))

    assert window.tpms_widgets["kind"].currentData() == "D"
    assert window._active_tpms_kind == "D"
    assert window.viewer._cell_map_gizmo is not None

    window.cell_map_reset_button.click()

    widgets = window.tpms_widgets
    assert widgets["frame_origin_follow"].isChecked()
    np.testing.assert_allclose(
        [control.value() for control in widgets["frame_inputs"]["u"]],
        (1.0, 0.0, 0.0),
    )
    np.testing.assert_allclose(
        [control.value() for control in widgets["frame_inputs"]["v"]],
        (0.0, 1.0, 0.0),
    )


def test_normal_scene_refresh_disables_cell_map_manipulation(window) -> None:
    window.cell_map_interactive.setChecked(True)
    assert window.viewer._cell_map_gizmo is not None

    window._refresh_scene(reset_view=False)

    assert not window.cell_map_interactive.isChecked()
    assert window.viewer._cell_map_gizmo is None
