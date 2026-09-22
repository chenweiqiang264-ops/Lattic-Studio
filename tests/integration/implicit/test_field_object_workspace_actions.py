"""Regressions for display-only field edits and workspace commands."""

from unittest import mock

import numpy as np
import pytest
from PyQt5 import QtCore, QtGui, QtWidgets

from lattice_studio.application.workspace import ManagedDesignWorkspace as DesignWorkspace
from lattice_studio.engine.implicit.field_viewer import sample_field_plane
from tests.integration.implicit.test_field_viewer_primitive import _FakeViewer
from lattice_studio.presentation.qt.tools import interactive_section_viewer
from lattice_studio.presentation.qt.workbench import (
    ImplicitGenerationResult,
    SamplingRecommendation,
    _build_qt_app,
)


@pytest.fixture(scope="module")
def application():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.fixture
def window(application):
    with mock.patch.object(interactive_section_viewer, "InteractiveSectionViewer", _FakeViewer):
        widget = _build_qt_app()()
        widget._create_analytic_design("sphere")
        widget.show()
        application.processEvents()
        yield widget
        widget.close()
        application.processEvents()


def seed_transition(window):
    document = window.design_workspace.active_document
    field = window.domain_implicit_field
    recommendation = SamplingRecommendation(
        voxel_size_mm=1.0,
        grid_shape=field.values.shape,
        estimated_voxels=field.values.size,
        samples_per_cell=8.0,
        samples_per_wall=4.0,
    )
    result = ImplicitGenerationResult(
        body=document.domain.as_implicit_body(),
        display_field=field,
        recommendation=recommendation,
    )
    window._generation_finished(
        document.identifier, document.revision,
        {"domain": (field, recommendation), "Transition": (result, recommendation)},
    )
    return result


@pytest.mark.parametrize("dual_role", [False, True])
def test_preview_toggle_preserves_transition_and_revision(window, dual_role):
    if not dual_role:
        window._add_field_primitive("box")
    identifier = window.field_primitive_combo.currentData()
    window.transition_driver_mode.setCurrentIndex(window.transition_driver_mode.findData("field"))
    window.transition_driver_combo.setCurrentIndex(window.transition_driver_combo.findData(identifier))
    result = seed_transition(window)
    document = window.design_workspace.active_document
    revision = document.revision
    domain = document.domain
    with mock.patch.object(window, "_refresh_domain_implicit_preview") as sample_domain:
        for visible in (False, True, False, True):
            window.field_primitive_visible.setChecked(visible)
            assert window.implicit_generation_results.get("Transition") is result
            assert document.runtime.implicit_generation_results.get("Transition") is result
            assert document.domain is domain
            assert document.revision == revision
            assert document.field_primitive_visibility[identifier] is visible
            snapshot = window.backend_client.get_workspace_snapshot(
                window.backend_workspace_id
            )
            backend_document = next(
                item
                for item in snapshot["documents"]
                if item["document_id"] == document.identifier
            )
            backend_field = next(
                item
                for item in backend_document["field_primitives"]
                if item["primitive"]["identifier"] == identifier
            )
            assert backend_field["visible"] is visible
            window._refresh_scene()
            assert window.viewer.scene_geometry_updates[-1][1]
        sample_domain.assert_not_called()
    # Display-only edits must not disable genuine dependency invalidation.
    window.field_primitive_center[0].setValue(window.field_primitive_center[0].value() + 1.0)
    assert "Transition" not in window.implicit_generation_results


@pytest.mark.parametrize("kind", ["sphere", "cylinder", "box"])
def test_primitive_exterior_extension_queries_true_sdf(window, kind):
    window._add_field_primitive(kind)
    identifier = window.field_primitive_combo.currentData()
    window.field_viewer_target_combo.setCurrentIndex(
        window.field_viewer_target_combo.findData(f"field:{identifier}")
    )
    assert window.field_viewer_domain_extension.isEnabled()
    body = window._field_viewer_target_body()
    window.field_viewer_domain_extension.setValue(12.0)
    expanded = window._field_viewer_source_data()
    np.testing.assert_allclose(expanded.bounds, body.bounds + np.array([[-12.0], [12.0]]))
    state = window._initialize_field_viewer_plane()
    sample = sample_field_plane(window._field_viewer_source_data(), state, (25, 25))
    assert np.max(sample.points[..., 0]) > body.bounds[1, 0]
    assert np.isfinite(sample.values).all()
    primitive = window.field_primitives[identifier]
    np.testing.assert_allclose(
        sample.values.ravel(), primitive.evaluate_points(sample.points.reshape(-1, 3)),
        atol=1e-5,
    )


def test_rename_requires_confirmation_and_preserves_results(window, tmp_path):
    document = window.design_workspace.active_document
    result = seed_transition(window)
    revision = document.revision
    original = document.name
    with mock.patch.object(QtWidgets.QInputDialog, "getText", return_value=("Cancelled", False)):
        window.rename_design_button.click()
    assert document.name == original
    with (
        mock.patch.object(QtWidgets.QInputDialog, "getText", return_value=("   ", True)),
        mock.patch.object(QtWidgets.QMessageBox, "warning") as warning,
    ):
        window.rename_design_button.click()
        warning.assert_called_once()
    assert document.name == original
    with mock.patch.object(QtWidgets.QInputDialog, "getText", return_value=("  Named design  ", True)):
        window.rename_design_button.click()
    assert document.name == "Named design"
    assert window.design_selector.currentText() == "Named design"
    assert document.revision == revision
    assert window.implicit_generation_results["Transition"] is result
    path = tmp_path / "workspace.json"
    window.design_workspace.save(path)
    assert DesignWorkspace.load(path).active_document.name == "Named design"


def test_clear_selected_archive_requires_confirmation_and_preserves_other_designs(window, tmp_path):
    archived = window.design_workspace.active_document
    window._archive_active_design()
    window._create_analytic_design("box")
    active = window.design_workspace.active_document
    window._create_analytic_design("cylinder")
    another = window.design_workspace.active_document
    window._archive_active_design()
    window.archived_design_selector.setCurrentIndex(
        window.archived_design_selector.findData(archived.identifier)
    )
    with mock.patch.object(QtWidgets.QMessageBox, "question", return_value=QtWidgets.QMessageBox.No):
        window.clear_archived_design_button.click()
    assert archived.identifier in window.design_workspace.archived_documents
    with mock.patch.object(QtWidgets.QMessageBox, "question", return_value=QtWidgets.QMessageBox.Yes):
        window.clear_archived_design_button.click()
    assert set(window.design_workspace.archived_documents) == {another.identifier}
    assert window.design_workspace.active_document is active
    path = tmp_path / "workspace.json"
    window.design_workspace.save(path)
    restored = DesignWorkspace.load(path)
    assert set(restored.archived_documents) == {another.identifier}
    assert restored.active_design_id == active.identifier


def test_empty_workspace_disables_rename_and_empty_recycle_actions(window):
    window._archive_active_design()
    assert not window.rename_design_button.isEnabled()
    assert window.clear_archived_design_button.isEnabled()
    with mock.patch.object(QtWidgets.QMessageBox, "question", return_value=QtWidgets.QMessageBox.Yes):
        window.clear_archived_design_button.click()
    assert not window.clear_archived_design_button.isEnabled()
    assert not window.restore_design_button.isEnabled()
    assert window.design_workspace.is_empty


def test_sampled_render_field_cannot_be_extrapolated(window):
    window.field_viewer_target_combo.setCurrentIndex(window.field_viewer_target_combo.findData("domain"))
    window.field_viewer_source_combo.setCurrentIndex(window.field_viewer_source_combo.findData("render"))
    assert not window.field_viewer_domain_extension.isEnabled()
    assert window._field_viewer_domain_extension_mm() == 0.0
    assert window._field_viewer_source_data() is window.domain_implicit_field


@pytest.mark.parametrize("source_kind", ["authoritative", "stl_reconstruction"])
def test_generated_lattice_field_supports_exterior_extension(window, source_kind):
    result = seed_transition(window)
    window.field_viewer_target_combo.setCurrentIndex(
        window.field_viewer_target_combo.findData("Transition")
    )
    window.field_viewer_source_combo.setCurrentIndex(
        window.field_viewer_source_combo.findData(source_kind)
    )

    assert window.field_viewer_domain_extension.isEnabled()
    window.field_viewer_domain_extension.setValue(9.0)
    expanded = window._field_viewer_source_data()

    np.testing.assert_allclose(
        expanded.bounds,
        result.body.bounds + np.array([[-9.0], [9.0]]),
    )
    outside = np.asarray([expanded.bounds[1] - 0.5], dtype=np.float64)
    assert np.isfinite(expanded.evaluate_points(outside)).all()


def test_spin_boxes_ignore_wheel_and_scroll_the_parameter_page(window):
    window.stl_processing_group.set_expanded(True)
    QtWidgets.QApplication.processEvents()
    control = window.export_tolerance
    control.setValue(0.5)
    scroll_area = next(
        area
        for area in window.findChildren(QtWidgets.QScrollArea)
        if area.isAncestorOf(control)
    )
    scroll_bar = scroll_area.verticalScrollBar()
    scroll_bar.setValue(min(scroll_bar.maximum(), max(1, scroll_bar.maximum() // 2)))
    original_scroll = scroll_bar.value()
    original_value = control.value()
    control.setFocus(QtCore.Qt.MouseFocusReason)
    QtWidgets.QApplication.processEvents()
    assert control.hasFocus()
    local = QtCore.QPointF(control.rect().center())
    global_position = QtCore.QPointF(control.mapToGlobal(control.rect().center()))
    event = QtGui.QWheelEvent(
        local,
        global_position,
        QtCore.QPoint(0, 0),
        QtCore.QPoint(0, -120),
        QtCore.Qt.NoButton,
        QtCore.Qt.NoModifier,
        QtCore.Qt.ScrollUpdate,
        False,
    )

    QtWidgets.QApplication.sendEvent(control, event)

    assert control.value() == original_value
    if scroll_bar.maximum() > 0:
        assert scroll_bar.value() > original_scroll


def test_combo_boxes_ignore_wheel_and_scroll_the_parameter_page(window):
    window.stl_processing_group.set_expanded(True)
    QtWidgets.QApplication.processEvents()
    control = window.processing_mode
    control.setCurrentIndex(0)
    scroll_area = next(
        area
        for area in window.findChildren(QtWidgets.QScrollArea)
        if area.isAncestorOf(control)
    )
    scroll_bar = scroll_area.verticalScrollBar()
    scroll_bar.setValue(min(scroll_bar.maximum(), max(1, scroll_bar.maximum() // 2)))
    original_scroll = scroll_bar.value()
    original_index = control.currentIndex()
    control.setFocus(QtCore.Qt.MouseFocusReason)
    QtWidgets.QApplication.processEvents()
    assert control.hasFocus()
    local = QtCore.QPointF(control.rect().center())
    global_position = QtCore.QPointF(control.mapToGlobal(control.rect().center()))
    event = QtGui.QWheelEvent(
        local,
        global_position,
        QtCore.QPoint(0, 0),
        QtCore.QPoint(0, -120),
        QtCore.Qt.NoButton,
        QtCore.Qt.NoModifier,
        QtCore.Qt.ScrollUpdate,
        False,
    )

    QtWidgets.QApplication.sendEvent(control, event)

    assert control.currentIndex() == original_index
    if scroll_bar.maximum() > 0:
        assert scroll_bar.value() > original_scroll


def test_sliders_ignore_wheel_and_scroll_the_parameter_page(window):
    control = window.transition_position_slider
    control.setValue(control.minimum())
    scroll_area = next(
        area
        for area in window.findChildren(QtWidgets.QScrollArea)
        if area.isAncestorOf(control)
    )
    scroll_bar = scroll_area.verticalScrollBar()
    scroll_bar.setValue(min(scroll_bar.maximum(), max(1, scroll_bar.maximum() // 2)))
    original_scroll = scroll_bar.value()
    original_value = control.value()
    local = QtCore.QPointF(control.rect().center())
    global_position = QtCore.QPointF(control.mapToGlobal(control.rect().center()))
    event = QtGui.QWheelEvent(
        local,
        global_position,
        QtCore.QPoint(0, 0),
        QtCore.QPoint(0, -120),
        QtCore.Qt.NoButton,
        QtCore.Qt.NoModifier,
        QtCore.Qt.ScrollUpdate,
        False,
    )

    QtWidgets.QApplication.sendEvent(control, event)

    assert control.value() == original_value
    if scroll_bar.maximum() > 0:
        assert scroll_bar.value() > original_scroll
