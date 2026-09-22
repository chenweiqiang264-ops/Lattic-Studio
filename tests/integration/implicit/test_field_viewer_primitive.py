"""Controller regressions for Field Viewer analytic field objects."""

from __future__ import annotations

import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
import trimesh


os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5 import QtWidgets

from lattice_studio.presentation.qt.tools import interactive_section_viewer
from lattice_studio.presentation.qt.workbench import (
    ImplicitGenerationResult,
    SamplingRecommendation,
    _build_qt_app,
    reconstruct_implicit_mesh,
)


class _FakeFieldPlaneGizmo:
    def __init__(self) -> None:
        self.origin = None
        self.basis = None

    def update_geometry(self) -> None:
        return None

    def add_to_scene(self) -> None:
        return None


class _FakeViewer(QtWidgets.QWidget):
    """Record scene commands without creating a native VTK render window."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._field_plane_gizmo = None
        self.scene_rebuild_count = 0
        self.field_plane_data = None
        self.implicit_field_updates = []
        self.scene_geometry_updates = []

    def set_background(self, _color) -> None:
        return None

    def set_field_surface_pick_callback(self, _callback) -> None:
        return None

    def set_scene_mesh_pick_callback(self, _callback) -> None:
        return None

    def set_field_plane_probe_callback(self, _callback) -> None:
        return None

    def set_implicit_fields(self, fields, reset_view=False) -> None:
        self.implicit_field_updates.append(list(fields))
        self.scene_rebuild_count += 1

    def set_scene_geometry(self, meshes, fields, reset_view=False) -> None:
        self.scene_geometry_updates.append((list(meshes), list(fields)))
        self.implicit_field_updates.append(list(fields))
        self.scene_rebuild_count += 1

    def set_meshes(self, _meshes, reset_view=False) -> None:
        self.scene_rebuild_count += 1

    def set_scene_overlay_meshes(self, _meshes, *, rebuild_scene=True) -> None:
        if rebuild_scene:
            self.scene_rebuild_count += 1

    def set_field_plane(self, data) -> None:
        self.field_plane_data = data

    def clear_field_plane(self) -> None:
        self.field_plane_data = None

    def enable_field_plane_gizmo(self, *_args) -> None:
        self._field_plane_gizmo = _FakeFieldPlaneGizmo()

    def disable_field_plane_gizmo(self) -> None:
        self._field_plane_gizmo = None

    def disable_interactive_section(self) -> None:
        return None

    def disable_primitive_transform(self) -> None:
        return None

    def hide_plane(self) -> None:
        return None


class _ImmediatePrimitiveCallbackViewer(_FakeViewer):
    """Model the real viewer's initial primitive-gizmo callback."""

    def enable_primitive_transform(self, _bounds, origin, basis, callback) -> None:
        callback(np.asarray(origin, dtype=float), np.asarray(basis, dtype=float))


class FieldViewerPrimitiveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.application = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def test_enabling_primitive_field_viewer_rebuilds_scene_once(self) -> None:
        """Field-plane activation must not synchronously rebuild VTK twice."""

        with mock.patch.object(
            interactive_section_viewer,
            "InteractiveSectionViewer",
            _FakeViewer,
        ):
            window_type = _build_qt_app()
            window_type._refresh_domain_implicit_preview = lambda self: None
            window = window_type()
            try:
                window._create_analytic_design("sphere")
                document = window.design_workspace.active_document
                assert document is not None
                window.domain_implicit_body = document.domain.as_implicit_body()
                window._add_field_primitive("box")
                window._sync_field_viewer_options()
                target = "field:primitive-1"
                target_index = window.field_viewer_target_combo.findData(target)
                self.assertGreaterEqual(target_index, 0)
                window.field_viewer_target_combo.setCurrentIndex(target_index)
                window.viewer.scene_rebuild_count = 0

                window.field_viewer_enabled.setChecked(True)
                self.application.processEvents()

                self.assertTrue(window.field_viewer_enabled.isChecked())
                self.assertIsNotNone(window.viewer.field_plane_data)
                self.assertLessEqual(window.viewer.scene_rebuild_count, 1)
            finally:
                window.close()

    def test_analytic_design_has_one_field_viewer_target(self) -> None:
        """A dual-role analytic design must not be listed as two field targets."""

        with mock.patch.object(
            interactive_section_viewer,
            "InteractiveSectionViewer",
            _FakeViewer,
        ):
            window_type = _build_qt_app()
            window_type._refresh_domain_implicit_preview = lambda self: None
            window = window_type()
            try:
                window._create_analytic_design("box")
                document = window.design_workspace.active_document
                assert document is not None
                window.domain_implicit_body = document.domain.as_implicit_body()
                window._sync_field_viewer_options()

                targets = [
                    window.field_viewer_target_combo.itemData(index)
                    for index in range(window.field_viewer_target_combo.count())
                ]

                self.assertEqual(targets, ["domain"])
            finally:
                window.close()

    def test_analytic_design_gizmo_initialization_does_not_reenter_scene_refresh(self) -> None:
        """Creating an analytic domain must not recurse through its gizmo callback."""

        with mock.patch.object(
            interactive_section_viewer,
            "InteractiveSectionViewer",
            _ImmediatePrimitiveCallbackViewer,
        ):
            window_type = _build_qt_app()
            window_type._refresh_domain_implicit_preview = lambda self: None
            window = window_type()
            try:
                for kind in ("sphere", "cylinder", "box"):
                    with self.subTest(kind=kind):
                        window._create_analytic_design(kind)
                        self.assertIsNotNone(window.design_workspace.active_document)
            finally:
                window.close()

    def test_new_analytic_design_refreshes_its_implicit_scene(self) -> None:
        """A newly sampled analytic Design domain must be rendered immediately."""

        with mock.patch.object(
            interactive_section_viewer,
            "InteractiveSectionViewer",
            _FakeViewer,
        ):
            window_type = _build_qt_app()
            window = window_type()
            try:
                window._create_analytic_design("sphere")

                self.assertIsNotNone(window.domain_implicit_field)
                self.assertTrue(window.viewer.scene_geometry_updates)
                _meshes, fields = window.viewer.scene_geometry_updates[-1]
                self.assertEqual(len(fields), 1)
                self.assertEqual(fields[0].name, window.domain_implicit_field.name)
            finally:
                window.close()

    def test_editing_analytic_design_dimension_refreshes_main_scene(self) -> None:
        """Replacing a dual-role analytic Design domain cannot leave its old layer visible."""

        with mock.patch.object(
            interactive_section_viewer,
            "InteractiveSectionViewer",
            _FakeViewer,
        ):
            window_type = _build_qt_app()
            window = window_type()
            try:
                window._create_analytic_design("box")
                original_width = window.field_primitive_size[0].value()
                old_bounds = tuple(
                    tuple(float(value) for value in row)
                    for row in window.domain_implicit_field.bounds
                )
                window.viewer.implicit_field_updates.clear()

                window.field_primitive_size[0].setValue(original_width + 2.0)
                window._flush_pending_analytic_domain_preview()
                self.application.processEvents()

                self.assertTrue(window.viewer.implicit_field_updates)
                visible_fields = window.viewer.implicit_field_updates[-1]
                self.assertEqual(len(visible_fields), 1)
                new_bounds = tuple(
                    tuple(float(value) for value in row)
                    for row in window.domain_implicit_field.bounds
                )
                self.assertNotEqual(new_bounds, old_bounds)
                self.assertEqual(
                    tuple(
                        tuple(float(value) for value in row)
                        for row in visible_fields[0].bounds
                    ),
                    new_bounds,
                )
            finally:
                window.close()

    def test_rapid_dimension_edits_coalesce_to_one_implicit_preview(self) -> None:
        """Spin-box bursts must not churn native GPU volume resources."""

        with mock.patch.object(
            interactive_section_viewer,
            "InteractiveSectionViewer",
            _FakeViewer,
        ):
            window_type = _build_qt_app()
            window = window_type()
            try:
                window._create_analytic_design("box")
                with mock.patch.object(
                    window,
                    "_refresh_domain_implicit_preview",
                    wraps=window._refresh_domain_implicit_preview,
                ) as refresh:
                    for index in range(40):
                        window.field_primitive_size[2].setValue(
                            20.0 if index % 2 == 0 else 60.0
                        )

                    self.assertEqual(refresh.call_count, 0)
                    self.assertTrue(window._analytic_domain_preview_timer.isActive())
                    window._flush_pending_analytic_domain_preview()
                    self.assertEqual(refresh.call_count, 1)
            finally:
                window.close()

    def test_new_stl_design_builds_implicit_preview_and_has_no_legacy_selector(self) -> None:
        """New STL Designs must enter through one command and render their SDF immediately."""

        with mock.patch.object(
            interactive_section_viewer,
            "InteractiveSectionViewer",
            _FakeViewer,
        ):
            window_type = _build_qt_app()
            window = window_type()
            try:
                with tempfile.TemporaryDirectory() as directory:
                    path = Path(directory) / "domain.stl"
                    trimesh.creation.box(extents=(8.0, 10.0, 12.0)).export(path)
                    window._load_sole(path, create_design=True)

                self.assertIsNotNone(window.domain_implicit_body)
                self.assertIsNotNone(window.domain_implicit_field)
                self.assertEqual(
                    window.domain_display_source_combo.currentData(), "implicit"
                )
                self.assertTrue(window.viewer.scene_geometry_updates)
                self.assertEqual(
                    len(window.viewer.scene_geometry_updates[-1][1]), 1
                )
                mesh_index = window.domain_display_source_combo.findData("mesh")
                window.domain_display_source_combo.setCurrentIndex(mesh_index)
                self.application.processEvents()
                mesh_layers, implicit_layers = window.viewer.scene_geometry_updates[-1]
                self.assertEqual(len(mesh_layers), 1)
                self.assertEqual(implicit_layers, [])
                implicit_index = window.domain_display_source_combo.findData("implicit")
                window.domain_display_source_combo.setCurrentIndex(implicit_index)
                self.application.processEvents()
                mesh_layers, implicit_layers = window.viewer.scene_geometry_updates[-1]
                self.assertEqual(mesh_layers, [])
                self.assertEqual(len(implicit_layers), 1)
                domain_group = next(
                    group
                    for group in window.findChildren(QtWidgets.QGroupBox)
                    if group.title() == "设计域"
                )
                button_texts = [
                    button.text()
                    for button in domain_group.findChildren(QtWidgets.QPushButton)
                ]
                self.assertNotIn("选择 STL", button_texts)
            finally:
                window.close()

    def test_analytic_design_is_a_stl_reconstruction_target(self) -> None:
        """An analytic Design domain must be reconstructed from its authoritative SDF."""

        with mock.patch.object(
            interactive_section_viewer,
            "InteractiveSectionViewer",
            _FakeViewer,
        ):
            window_type = _build_qt_app()
            window = window_type()
            try:
                window._create_analytic_design("box")
                target_index = window.stl_reconstruction_target_combo.findData("domain")

                self.assertGreaterEqual(target_index, 0)
                window.stl_reconstruction_target_combo.setCurrentIndex(target_index)
                selected = window._selected_stl_reconstruction_results()
                self.assertIn("domain", selected)
                self.assertIs(
                    selected["domain"].body,
                    window.domain_implicit_body,
                )
                self.assertTrue(window.build_stl_button.isEnabled())
                export = reconstruct_implicit_mesh(
                    selected["domain"].body,
                    selected["domain"].extraction_map,
                    tolerance_mm=1.0,
                    wall_thickness_mm=selected["domain"].minimum_feature_mm,
                    spacing_mode="recommended",
                    enforce_feature_limits=False,
                )
                self.assertEqual(export.spacing_mm, (1.0, 1.0, 1.0))
                self.assertTrue(export.quality.watertight)
            finally:
                window.close()

    def test_generation_exits_field_object_inspection_and_shows_result(self) -> None:
        """Generation completion cannot leave Field Viewer masking the new lattice."""

        with mock.patch.object(
            interactive_section_viewer,
            "InteractiveSectionViewer",
            _FakeViewer,
        ):
            window_type = _build_qt_app()
            window = window_type()
            try:
                window._create_analytic_design("sphere")
                window._add_field_primitive("box")
                target_index = window.field_viewer_target_combo.findData("field:primitive-1")
                self.assertGreaterEqual(target_index, 0)
                window.field_viewer_target_combo.setCurrentIndex(target_index)
                window.field_viewer_enabled.setChecked(True)

                document = window.design_workspace.active_document
                assert document is not None
                field = window.domain_implicit_field
                assert field is not None
                recommendation = SamplingRecommendation(
                    voxel_size_mm=1.0,
                    grid_shape=tuple(int(value) for value in field.values.shape),
                    estimated_voxels=int(np.prod(field.values.shape)),
                    samples_per_cell=8.0,
                    samples_per_wall=4.0,
                )
                result = ImplicitGenerationResult(
                    body=document.domain.as_implicit_body(),
                    display_field=field,
                    recommendation=recommendation,
                )

                window._generation_finished(
                    document.identifier,
                    document.revision,
                    {"domain": (field, recommendation), "G": (result, recommendation)},
                )

                self.assertFalse(window.field_viewer_enabled.isChecked())
                self.assertTrue(window.viewer.scene_geometry_updates)
                self.assertEqual(
                    len(window.viewer.scene_geometry_updates[-1][1]), 1
                )
            finally:
                window.close()


if __name__ == "__main__":
    unittest.main()
