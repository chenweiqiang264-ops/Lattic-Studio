"""Workbench integration tests for STL-backed custom unit cells."""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest
import trimesh


TESTS_DIR = Path(__file__).resolve().parent
if str(TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(TESTS_DIR))

from lattice_studio.presentation.qt import workbench as app
from lattice_studio.engine.implicit.custom_unit_cell import prepare_stl_unit_cell


class _AnalyticSphereBackend:
    def signed_distance(self, vertices, faces, points, max_tile_points=250_000):
        del faces, max_tile_points
        bounds = np.vstack((vertices.min(axis=0), vertices.max(axis=0)))
        center = bounds.mean(axis=0)
        radius = float(np.min(bounds[1] - bounds[0]) * 0.3)
        return np.linalg.norm(np.asarray(points) - center, axis=1) - radius


def _source_cell(path: Path) -> Path:
    trimesh.creation.icosphere(subdivisions=2, radius=0.5).export(path)
    return path


def test_complete_workbench_exposes_custom_cell_without_tpms_controls():
    script = textwrap.dedent(
        """
        from PyQt5 import QtWidgets
        import numpy as np
        import trimesh
        from lattice_studio.presentation.qt.tools import interactive_section_viewer
        from lattice_studio.presentation.qt import workbench as app

        class ViewerStub(QtWidgets.QWidget):
            def set_background(self, _color):
                pass
            def set_meshes(self, _meshes, reset_view=False):
                pass
            def hide_plane(self):
                pass
            def set_plane(self, _point, _normal, _size):
                pass
            def disable_interactive_section(self):
                pass

        interactive_section_viewer.InteractiveSectionViewer = ViewerStub
        application = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        window = app._build_qt_app()()
        widgets = window.custom_widgets
        assert window.custom_group.title() == "自定义 STL 晶胞"
        assert any(
            group.title() == "隐式体渲染"
            for group in window.findChildren(QtWidgets.QGroupBox)
        )
        assert widgets["enabled"].isChecked() is False
        assert widgets["source_path"].text() == ""
        assert "wall" not in widgets
        assert "level" not in widgets
        assert widgets["target_feature"].specialValueText() == "使用源模型厚度"
        assert set(widgets["bridge_checks"]) == {"U", "V", "W"}
        assert window._params("Custom").bridge_directions == ()
        widgets["bridge_checks"]["V"].setChecked(True)
        widgets["bridge_depth"].setValue(0.4)
        assert window._params("Custom").bridge_directions == ("V",)
        assert window._params("Custom").bridge_depth_mm == 0.4
        assert window._params("Custom").target_feature_mm is None
        widgets["target_feature"].setValue(1.25)
        assert window._params("Custom").target_feature_mm == 1.25
        assert set(widgets["frame_inputs"]) == {"origin", "u", "v"}
        custom_index = window.cell_map_kind.findData("Custom")
        assert custom_index >= 0
        assert window.cell_map_kind.itemText(custom_index) == "自定义晶胞 Cell Map"
        assert window.show_custom.isChecked() is True
        assert window.export_custom_button.isEnabled() is False
        assert window.transition_group.title() == "晶胞空间过渡（Ramp）"
        assert window.transition_weight.currentData() == "automatic"
        assert window.transition_sharpness.isEnabled() is False
        sigmoid_index = window.transition_weight.findData("sigmoid")
        window.transition_weight.setCurrentIndex(sigmoid_index)
        assert window.transition_sharpness.isEnabled() is True
        first_controls = window.transition_first_controls
        second_controls = window.transition_second_controls
        first_controls["kind"].setCurrentIndex(first_controls["kind"].findData("G"))
        second_controls["kind"].setCurrentIndex(second_controls["kind"].findData("G"))
        second_controls["inherit"].setChecked(False)
        for control, value in zip(second_controls["sizes"], (8.0, 9.0, 10.0)):
            control.setValue(value)
        second_controls["feature"].setValue(1.4)
        window.transition_enabled.setChecked(True)
        first_transition, second_transition, transition = window._selected_transition_job()
        assert first_transition.kind == second_transition.kind == "G"
        assert first_transition.cell_size_xyz_mm != second_transition.cell_size_xyz_mm
        assert second_transition.cell_size_xyz_mm == (8.0, 9.0, 10.0)
        assert second_transition.wall_thickness_mm == 1.4
        assert transition.weight_kind == "sigmoid"
        window.transition_enabled.setChecked(False)
        assert window.display_batched_sampling.isChecked() is False
        assert window.display_batch_count.isEnabled() is False
        window.sole_mesh = trimesh.creation.box(extents=(20.0, 40.0, 5.0))
        g_frame = window.g_widgets
        assert g_frame["frame_origin_follow"].isChecked() is True
        assert all(not control.isEnabled() for control in g_frame["frame_inputs"]["origin"])
        angle = np.deg2rad(35.0)
        for key, values in {
            "u": (np.cos(angle), np.sin(angle), 0.0),
            "v": (-np.sin(angle), np.cos(angle), 0.0),
        }.items():
            for control, value in zip(g_frame["frame_inputs"][key], values):
                control.blockSignals(True)
                control.setValue(float(value))
                control.blockSignals(False)
        window._on_cell_map_frame_changed("G")
        followed = window._params("G")
        expected_frame = app.CellMapFrame.aligned_to_points_minimum(
            window.sole_mesh.vertices,
            followed.frame_u_axis,
            followed.frame_v_axis,
        )
        np.testing.assert_allclose(
            followed.frame_origin_mm,
            expected_frame.origin,
            atol=1.0e-6,
        )
        followed_map = app.cell_map_for_parameters(window.sole_mesh, followed)
        assert followed_map.index_min == (0, 0, 0)
        g_frame["frame_origin_follow"].setChecked(False)
        assert all(control.isEnabled() for control in g_frame["frame_inputs"]["origin"])
        window._update_recommendation()
        assert "G：U/V/W" in window.cell_sample_info.text()
        assert "D：U/V/W" in window.cell_sample_info.text()
        assert "单晶胞约" in window.cell_sample_info.text()
        assert "voxel/samples" in window.cell_sample_info.text()
        window.display_batched_sampling.setChecked(True)
        assert window.display_batch_count.isEnabled() is True
        assert window._display_sampling_batch_count() == 4
        assert "单晶胞约" in window.cell_sample_info.text()
        assert window.processing_mode.currentData() == "single_pass"
        window.processing_mode.setCurrentIndex(2)
        assert window.processing_mode.currentData() == "chunked_marching_cubes"
        assert window.batch_count.isEnabled() is True
        window.display_batched_sampling.setChecked(False)
        assert window._display_sampling_batch_count() is None
        window.close()
        application.processEvents()
        """
    )
    environment = os.environ.copy()
    environment["QT_QPA_PLATFORM"] = "offscreen"
    environment["PYTHONIOENCODING"] = "utf-8"
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=app.PROJECT_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=60,
        check=False,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_workbench_switches_between_recommended_and_exact_stl_spacing():
    script = textwrap.dedent(
        """
        import numpy as np
        from PyQt5 import QtWidgets
        from lattice_studio.presentation.qt.tools import interactive_section_viewer
        from lattice_studio.presentation.qt import workbench as app

        class ViewerStub(QtWidgets.QWidget):
            def set_background(self, _color):
                pass
            def set_meshes(self, _meshes, reset_view=False):
                pass
            def hide_plane(self):
                pass
            def disable_interactive_section(self):
                pass

        interactive_section_viewer.InteractiveSectionViewer = ViewerStub
        application = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        window = app._build_qt_app()()
        cell_map = app.CellMap.from_bounds(
            np.array(((-2.0, -2.0, -2.0), (2.0, 2.0, 2.0))),
            spacing_mm=(4.0, 4.0, 4.0),
        )
        body = app.ImplicitBody(
            "sphere",
            cell_map.bounds,
            lambda points, _reporter: np.linalg.norm(points, axis=1) - 1.0,
        )
        field = app.SampledImplicitField(
            "sphere",
            np.zeros((3, 3, 3), dtype=np.float32),
            np.full(3, -2.0),
            np.full(3, 2.0),
            (0.7, 0.7, 0.7, 1.0),
        )
        recommendation = app.SamplingRecommendation(
            0.1, (43, 43, 43), 79_507, 40.0, 3.0
        )
        window.implicit_generation_results = {
            "Custom": app.ImplicitGenerationResult(
                body,
                field,
                recommendation,
                cell_map=cell_map,
                minimum_feature_mm=0.3,
            )
        }
        window._sync_stl_reconstruction_target_options()
        window.stl_reconstruction_target_combo.setCurrentIndex(
            window.stl_reconstruction_target_combo.findData("Custom")
        )
        window.export_tolerance.setValue(0.5)
        window.export_spacing_auto.setChecked(False)
        window._update_stl_grid_estimate()
        exact_text = window.export_grid_info.text()
        assert "11×11×11 = 1,331 体素" in exact_text
        assert "实际间距 0.5000 mm" in exact_text
        assert "安全建议不大于 0.1000 mm" in exact_text

        window.export_spacing_auto.setChecked(True)
        recommended_text = window.export_grid_info.text()
        assert "44×44×44 = 85,184 体素" in recommended_text
        assert "实际间距 0.1000 mm" in recommended_text
        window.close()
        application.processEvents()
        """
    )
    environment = os.environ.copy()
    environment["QT_QPA_PLATFORM"] = "offscreen"
    environment["PYTHONIOENCODING"] = "utf-8"
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=app.PROJECT_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=60,
        check=False,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_custom_parameters_build_an_independent_cell_map(tmp_path):
    source = _source_cell(tmp_path / "cell.stl")
    domain = trimesh.creation.box(extents=(12.0, 8.0, 4.0))
    parameters = app.CustomUnitCellParameters(
        source,
        cell_size_mm=(3.0, 2.0, 2.0),
        cell_map_mode="fit_bounds",
    )

    cell_map = app.cell_map_for_parameters(domain, parameters)

    assert cell_map.boundary_mode == "fit_bounds"
    assert cell_map.cell_counts == (4, 4, 2)
    np.testing.assert_allclose(cell_map.bounds, domain.bounds)
    np.testing.assert_allclose(cell_map.spacing_mm, (3.0, 2.0, 2.0))


def test_custom_implicit_generation_carries_source_report_and_feature_size(
    tmp_path,
):
    source = _source_cell(tmp_path / "cell.stl")
    prepared = prepare_stl_unit_cell(
        source,
        geometry_backend=_AnalyticSphereBackend(),
    )
    domain = trimesh.creation.box(extents=(4.0, 4.0, 4.0))
    parameters = app.CustomUnitCellParameters(
        source,
        cell_size_mm=(2.0, 2.0, 2.0),
    )
    sampling = app.SamplingParameters(target_voxels=100_000)

    result = app.generate_implicit_custom_lattice(
        domain,
        parameters,
        sampling,
        voxel_size_mm=0.5,
        display_voxel_size_mm=0.5,
        display_memory_budget_mb=64.0,
        prepared_unit_cell=prepared,
    )

    assert result.cell_map is not None
    assert result.metadata is prepared.report
    assert result.minimum_feature_mm is not None
    assert result.minimum_feature_mm > 0.0
    assert result.display_fragment_cleanup is not None
    assert result.display_field.values.size > 0
    assert np.any(result.display_field.values < 0.0)
    assert np.any(result.display_field.values > 0.0)


def test_custom_target_feature_controls_generation_and_sampling(tmp_path):
    source = _source_cell(tmp_path / "target-feature-cell.stl")
    prepared = prepare_stl_unit_cell(
        source,
        geometry_backend=_AnalyticSphereBackend(),
    )
    domain = trimesh.creation.box(extents=(4.0, 4.0, 4.0))
    target_feature_mm = 0.3
    parameters = app.CustomUnitCellParameters(
        source,
        cell_size_mm=(2.0, 2.0, 2.0),
        target_feature_mm=target_feature_mm,
    )
    sampling = app.SamplingParameters(
        target_voxels=100_000,
        min_samples_per_wall=3.0,
    )

    result = app.generate_implicit_custom_lattice(
        domain,
        parameters,
        sampling,
        display_voxel_size_mm=0.25,
        display_memory_budget_mb=64.0,
        prepared_unit_cell=prepared,
    )

    assert result.minimum_feature_mm == pytest.approx(target_feature_mm)
    assert result.recommendation.samples_per_wall == pytest.approx(
        target_feature_mm / result.recommendation.voxel_size_mm
    )


def test_custom_display_reuses_a_matching_domain_field(tmp_path, monkeypatch):
    source = _source_cell(tmp_path / "cached-domain-cell.stl")
    prepared = prepare_stl_unit_cell(
        source,
        geometry_backend=_AnalyticSphereBackend(),
    )
    domain = trimesh.creation.box(extents=(4.0, 4.0, 4.0))
    shape = (11, 11, 11)
    cached_domain = app.SampledImplicitField(
        "domain",
        np.full(shape, -10.0, dtype=np.float32),
        np.full(3, -2.5),
        np.full(3, 0.5),
        (0.7, 0.7, 0.7, 1.0),
    )

    def fail_if_resampled(_points, _stage):
        raise AssertionError("matching design-domain display SDF was not reused")

    monkeypatch.setattr(
        app,
        "_make_design_domain_body",
        lambda *_args, **_kwargs: app.ImplicitBody(
            "domain",
            domain.bounds,
            fail_if_resampled,
        ),
    )

    result = app.generate_implicit_custom_lattice(
        domain,
        app.CustomUnitCellParameters(source, cell_size_mm=(2.0, 2.0, 2.0)),
        app.SamplingParameters(target_voxels=100_000),
        voxel_size_mm=0.5,
        display_voxel_size_mm=0.5,
        display_memory_budget_mb=64.0,
        domain_display_field=cached_domain,
        prepared_unit_cell=prepared,
    )

    assert result.display_field.values.shape == shape
    assert np.any(result.display_field.values < 0.0)
    assert np.any(result.display_field.values > 0.0)


def test_custom_display_rejects_an_incompatible_domain_cache(tmp_path):
    source = _source_cell(tmp_path / "stale-domain-cell.stl")
    prepared = prepare_stl_unit_cell(
        source,
        geometry_backend=_AnalyticSphereBackend(),
    )
    domain = trimesh.creation.box(extents=(4.0, 4.0, 4.0))
    stale_domain = app.SampledImplicitField(
        "stale-domain",
        np.full((5, 5, 5), -10.0, dtype=np.float32),
        np.full(3, -1.0),
        np.full(3, 1.0),
        (0.7, 0.7, 0.7, 1.0),
    )

    result = app.generate_implicit_custom_lattice(
        domain,
        app.CustomUnitCellParameters(source, cell_size_mm=(2.0, 2.0, 2.0)),
        app.SamplingParameters(target_voxels=100_000),
        voxel_size_mm=0.5,
        display_voxel_size_mm=0.5,
        display_memory_budget_mb=64.0,
        domain_display_field=stale_domain,
        prepared_unit_cell=prepared,
    )

    assert result.display_field.values.shape == (11, 11, 11)
    assert result.display_field.values.shape != stale_domain.values.shape


def test_custom_reconstruction_grid_trims_bounds_without_changing_phase(tmp_path):
    source = _source_cell(tmp_path / "aligned-grid-cell.stl")
    prepared = prepare_stl_unit_cell(
        source,
        geometry_backend=_AnalyticSphereBackend(),
    )
    domain = trimesh.creation.box(extents=(4.0, 12.0, 2.0))
    angle = np.deg2rad(45.0)
    parameters = app.CustomUnitCellParameters(
        source,
        cell_size_mm=(2.0, 2.0, 2.0),
        cell_map_mode="complete_cells",
        frame_origin_mm=tuple(domain.bounds[0]),
        frame_u_axis=(np.cos(angle), np.sin(angle), 0.0),
        frame_v_axis=(-np.sin(angle), np.cos(angle), 0.0),
    )
    cell_map = app.cell_map_for_parameters(domain, parameters)
    body = app.make_periodic_lattice(
        prepared,
        cell_map,
        app._make_design_domain_body(
            domain,
            app.SamplingParameters(target_voxels=100_000),
        ),
    )
    spacing = app.recommend_export_spacing(
        cell_map,
        tolerance_mm=0.25,
        wall_thickness_mm=0.6,
    )
    anchor = np.asarray(cell_map.bounds[0], dtype=np.float64)
    new_origin, new_shape, new_axes = app.extraction_grid_definition(
        body,
        spacing,
        anchor,
    )
    legacy_body = app.ImplicitBody("legacy", cell_map.bounds, body.evaluate)
    old_origin, old_shape, old_axes = app.extraction_grid_definition(
        legacy_body,
        spacing,
        anchor,
    )

    assert int(np.prod(new_shape)) < int(np.prod(old_shape))
    np.testing.assert_allclose(
        (new_origin - old_origin) / spacing,
        np.rint((new_origin - old_origin) / spacing),
        atol=1.0e-12,
    )
    for axis in range(3):
        old_indices = np.rint((new_axes[axis] - old_origin[axis]) / spacing[axis])
        np.testing.assert_allclose(
            new_axes[axis],
            old_axes[axis][old_indices.astype(np.int64)],
            atol=1.0e-12,
        )


@pytest.mark.parametrize("target", (0.0, -0.1, np.nan, np.inf))
def test_custom_parameters_reject_invalid_target_feature(tmp_path, target):
    source = _source_cell(tmp_path / "invalid-target-cell.stl")
    parameters = app.CustomUnitCellParameters(
        source,
        target_feature_mm=target,
    )

    with pytest.raises(ValueError, match="target_feature_mm"):
        parameters.validate()


def test_rotated_custom_cell_display_samples_the_trimmed_design_domain(tmp_path):
    source = _source_cell(tmp_path / "rotated-cell.stl")
    prepared = prepare_stl_unit_cell(
        source,
        geometry_backend=_AnalyticSphereBackend(),
    )
    domain = trimesh.creation.box(extents=(4.0, 12.0, 2.0))
    angle = np.deg2rad(45.0)
    parameters = app.CustomUnitCellParameters(
        source,
        cell_size_mm=(2.0, 2.0, 2.0),
        cell_map_mode="complete_cells",
        frame_origin_mm=tuple(domain.bounds[0]),
        frame_u_axis=(np.cos(angle), np.sin(angle), 0.0),
        frame_v_axis=(-np.sin(angle), np.cos(angle), 0.0),
    )

    result = app.generate_implicit_custom_lattice(
        domain,
        parameters,
        app.SamplingParameters(target_voxels=100_000),
        voxel_size_mm=0.25,
        display_voxel_size_mm=0.25,
        display_memory_budget_mb=64.0,
        prepared_unit_cell=prepared,
    )

    assert result.cell_map is not None
    assert np.any(result.cell_map.extent_mm > domain.extents)
    assert np.any(result.display_field.values <= 0.0)
    assert np.any(result.display_field.values > 0.0)
    assert np.all(result.display_field.bounds[0] <= domain.bounds[0])
    assert np.all(result.display_field.bounds[1] >= domain.bounds[1])
    expected_shape = tuple(
        int(np.ceil(extent / 0.25)) + 3 for extent in domain.extents
    )
    assert result.display_field.values.shape == expected_shape
