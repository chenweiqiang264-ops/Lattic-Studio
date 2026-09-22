"""Regression coverage for per-family TPMS settings in the shared panel."""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap


def test_tpms_types_restore_independent_parameter_states() -> None:
    script = textwrap.dedent(
        """
        from PyQt5 import QtWidgets
        from lattice_studio.presentation.qt.tools import interactive_section_viewer
        from lattice_studio.presentation.qt import workbench as app

        class ViewerStub(QtWidgets.QWidget):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
            def set_background(self, _color): pass
            def set_meshes(self, _meshes, reset_view=False): pass
            def hide_plane(self): pass
            def set_plane(self, _point, _normal, _size): pass
            def disable_interactive_section(self): pass
            def set_scene_mesh_pick_callback(self, _callback): pass
            def set_scene_overlay_meshes(self, _meshes, *, rebuild_scene=True): pass
            def enable_primitive_transform(self, *_args): pass
            def disable_primitive_transform(self): pass

        interactive_section_viewer.InteractiveSectionViewer = ViewerStub
        application = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        window = app._build_qt_app()()
        widgets = window.tpms_widgets

        widgets["cell_x"].setValue(8.5)
        widgets["cell_y"].setValue(9.5)
        widgets["cell_z"].setValue(10.5)
        widgets["wall"].setValue(0.73)
        widgets["level"].setValue(0.21)
        widgets["gradient_enabled"].setChecked(True)
        widgets["gradient_axis"].setCurrentIndex(
            widgets["gradient_axis"].findData("U")
        )
        widgets["gradient_sharpness"].setValue(7.5)
        widgets["manual_counts_enabled"].setChecked(True)
        for control, value in zip(widgets["manual_counts"], (3, 4, 5)):
            control.setValue(value)

        widgets["kind"].setCurrentIndex(widgets["kind"].findData("D"))
        assert tuple(control.value() for control in (
            widgets["cell_x"], widgets["cell_y"], widgets["cell_z"]
        )) == (12.0, 12.0, 12.0)
        assert widgets["wall"].value() == 1.0
        assert window._params("G").cell_size_xyz_mm == (8.5, 9.5, 10.5)
        assert window._params("G").wall_thickness_mm == 0.73
        assert window._params("G").gradient_axis == "U"
        assert window._params("G").manual_cell_counts == (3, 4, 5)

        widgets["cell_x"].setValue(15.0)
        widgets["wall"].setValue(1.25)
        widgets["gradient_enabled"].setChecked(False)
        widgets["kind"].setCurrentIndex(widgets["kind"].findData("G"))
        assert tuple(control.value() for control in (
            widgets["cell_x"], widgets["cell_y"], widgets["cell_z"]
        )) == (8.5, 9.5, 10.5)
        assert widgets["wall"].value() == 0.73
        assert widgets["gradient_enabled"].isChecked()
        assert widgets["gradient_axis"].currentData() == "U"
        assert tuple(control.value() for control in widgets["manual_counts"]) == (3, 4, 5)

        widgets["kind"].setCurrentIndex(widgets["kind"].findData("D"))
        assert widgets["cell_x"].value() == 15.0
        assert widgets["wall"].value() == 1.25
        assert not widgets["gradient_enabled"].isChecked()
        window.close()
        application.processEvents()
        """
    )
    environment = os.environ.copy()
    environment["QT_QPA_PLATFORM"] = "offscreen"
    environment["PYTHONIOENCODING"] = "utf-8"
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=os.getcwd(),
        env=environment,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=60,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
