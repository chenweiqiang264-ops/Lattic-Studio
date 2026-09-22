"""UI wiring tests for independent analytic field objects."""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap


def test_workbench_registers_primitives_for_transition_and_field_viewing() -> None:
    script = textwrap.dedent(
        """
        from PyQt5 import QtWidgets
        import numpy as np
        from lattice_studio.engine.implicit.field_viewer import FieldPlaneState
        from lattice_studio.presentation.qt.tools import interactive_section_viewer
        from lattice_studio.presentation.qt import workbench as app

        class ViewerStub(QtWidgets.QWidget):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.overlay_layers = []
                self.implicit_layers = []
                self.gizmo = None
            def set_background(self, _color): pass
            def set_meshes(self, _meshes, reset_view=False): pass
            def set_implicit_fields(self, fields, reset_view=False):
                self.implicit_layers = list(fields)
            def hide_plane(self): pass
            def set_plane(self, _point, _normal, _size): pass
            def disable_interactive_section(self): pass
            def set_scene_mesh_pick_callback(self, callback): self.pick_callback = callback
            def set_scene_overlay_meshes(self, meshes, *, rebuild_scene=True): self.overlay_layers = meshes
            def enable_primitive_transform(self, bounds, origin, basis, callback):
                self.gizmo = (bounds, origin, basis, callback)
            def disable_primitive_transform(self): self.gizmo = None

        interactive_section_viewer.InteractiveSectionViewer = ViewerStub
        application = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        window = app._build_qt_app()()
        window._add_field_primitive("cylinder")
        identifier = window.field_primitive_combo.currentData()
        primitive = window.field_primitives[identifier]
        assert primitive.kind == "cylinder"
        assert len(window.viewer.overlay_layers) == 1
        assert window.viewer.overlay_layers[0].color[3] >= 0.84
        assert window.viewer.gizmo is not None

        for mode in (
            window.contour_enabled,
            window.section_enabled,
        ):
            mode.blockSignals(True)
            mode.setChecked(True)
            mode.blockSignals(False)
            window._refresh_primitive_preview(sync_gizmo=True)
            assert window.viewer.overlay_layers == []
            assert window.viewer.gizmo is None
            mode.blockSignals(True)
            mode.setChecked(False)
            mode.blockSignals(False)
        window._refresh_primitive_preview(sync_gizmo=True)
        assert len(window.viewer.overlay_layers) == 1
        assert window.viewer.gizmo is not None

        window._on_scene_mesh_selected(("field-primitive", identifier, primitive))
        assert window._selected_primitive_identifier == identifier
        assert window.transition_driver_combo.findData(identifier) >= 0
        assert window.field_viewer_target_combo.findData(f"field:{identifier}") >= 0

        window.transition_enabled.setChecked(True)
        window.transition_driver_mode.setCurrentIndex(
            window.transition_driver_mode.findData("field")
        )
        transition_form = window._transition_form
        for control in (
            window.transition_axis,
            window.transition_position_container,
            window.transition_angle1,
            window.transition_angle2,
            window.transition_width,
            window.transition_width_status,
            window.transition_center_offset,
        ):
            assert control.isHidden()
            assert transition_form.labelForField(control).isHidden()
        assert window.transition_registration.isHidden()
        assert window.show_transition_plane.isHidden()
        assert not window.transition_driver_combo.isHidden()
        assert not window.transition_field_interval_container.isHidden()
        window.transition_driver_mode.setCurrentIndex(
            window.transition_driver_mode.findData("plane")
        )
        assert not window.transition_axis.isHidden()
        assert not window.transition_position_container.isHidden()
        assert not window.transition_registration.isHidden()
        assert not window.show_transition_plane.isHidden()
        window.transition_driver_mode.setCurrentIndex(
            window.transition_driver_mode.findData("field")
        )
        window.transition_driver_combo.setCurrentIndex(
            window.transition_driver_combo.findData(identifier)
        )
        window.transition_field_lower.setValue(-2.0)
        window.transition_field_upper.setValue(2.0)
        first, second, parameters, driver = window._selected_transition_job()
        assert first.kind == "G" and second.kind == "D"
        assert driver.name == primitive.name
        spec = parameters.provider_spec()
        assert spec.width_mm == 4.0 and spec.center_offset_mm == 0.0

        window.field_viewer_target_combo.setCurrentIndex(
            window.field_viewer_target_combo.findData(f"field:{identifier}")
        )
        window.field_viewer_enabled.blockSignals(True)
        window.field_viewer_enabled.setChecked(True)
        window.field_viewer_enabled.blockSignals(False)
        assert window.field_viewer_show_object.text() == "显示基本几何体"
        layers = window._primitive_preview_layers()
        assert len(layers) == 1
        window.field_viewer_show_object.blockSignals(True)
        window.field_viewer_show_object.setChecked(False)
        window.field_viewer_show_object.blockSignals(False)
        assert window._primitive_preview_layers() == []
        window.field_viewer_show_object.blockSignals(True)
        window.field_viewer_show_object.setChecked(True)
        window.field_viewer_show_object.blockSignals(False)
        window.field_viewer_enabled.blockSignals(True)
        window.field_viewer_enabled.setChecked(False)
        window.field_viewer_enabled.blockSignals(False)
        assert window.field_viewer_source_combo.currentData() == "authoritative", (window.field_viewer_target_combo.currentData(), window.field_viewer_source_combo.currentData(), window.field_viewer_source_combo.findData("authoritative"), [window.field_viewer_source_combo.itemData(index) for index in range(window.field_viewer_source_combo.count())])
        assert window.field_viewer_source_combo.isEnabled() is False, "source combo remained enabled"
        assert window._field_viewer_target_body().name == primitive.name, window._field_viewer_target_body()
        window._field_plane_state = FieldPlaneState(
            np.array((30.0, 0.0, 0.0)),
            np.array((1.0, 0.0, 0.0)),
            np.array((0.0, 1.0, 0.0)),
            20.0,
            20.0,
        )
        expanded = window._field_viewer_source_data()
        assert expanded.bounds[1, 0] >= 40.0
        assert expanded.bounds[0, 1] <= -10.0

        window.field_primitive_visible.setChecked(False)
        assert window.viewer.overlay_layers == [], len(window.viewer.overlay_layers)

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
