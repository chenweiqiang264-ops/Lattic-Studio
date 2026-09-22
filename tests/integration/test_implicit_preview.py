"""Regression tests for evaluator-backed implicit preview generation."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pytest
import trimesh

from lattice_studio.engine.implicit.tpms_compute import get_default_tpms_backend

TESTS_DIR = Path(__file__).resolve().parent
if str(TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(TESTS_DIR))

from lattice_studio.presentation.qt import workbench as app


def test_manual_base_voxel_controls_are_not_exposed_in_the_workbench() -> None:
    source = Path(app.__file__).read_text(encoding="utf-8")

    assert "使用自定义体素大小" not in source
    assert 'sample_form.addRow("推荐体素大小"' not in source
    assert "self.manual_voxel_size" not in source


def _load_renderer_module():
    from lattice_studio.presentation.qt.viewers import pyvista_viewer

    return pyvista_viewer


def test_display_budget_coarsens_preview_without_changing_base_spacing() -> None:
    domain = trimesh.creation.box(extents=(60.0, 170.0, 12.0))

    step = app.recommend_display_voxel_size(
        domain,
        base_voxel_size_mm=0.05,
        display_memory_budget_mb=64.0,
        visible_field_count=3,
    )

    assert step > 0.05
    shape = np.ceil(domain.extents / step).astype(np.int64) + 3
    assert int(np.prod(shape)) * 4 * 3 <= 64 * 1024 * 1024


def test_display_base_recommendation_reports_its_limiting_constraint() -> None:
    domain = trimesh.creation.box(extents=(20.0, 20.0, 5.0))
    sampling = app.SamplingParameters(target_voxels=100_000)
    parameters = app.TPMSParameters(
        "G",
        cell_size_mm=10.0,
        wall_thickness_mm=0.2,
    )

    recommendation = app.recommend_voxel_size(domain, parameters, sampling)

    assert recommendation.voxel_size_mm == pytest.approx(0.08)
    assert recommendation.limiting_constraint == "壁厚采样"


def test_display_sampling_report_states_the_applied_grid_and_reason() -> None:
    field = app.SampledImplicitField(
        "G",
        np.zeros((9, 11, 13), dtype=np.float32),
        np.zeros(3),
        np.full(3, 0.25),
        (0.7, 0.7, 0.7, 1.0),
    )
    base = app.SamplingRecommendation(0.10, (21, 21, 21), 9_261, 20.0, 4.0)

    automatic = app.describe_display_sampling("G", field, base, None)
    manual = app.describe_display_sampling("G", field, base, 0.25)

    assert "实际间距 0.2500 mm" in automatic
    assert "网格 9×11×13 (1,287 体素)" in automatic
    assert "显示预算调整" in automatic
    assert "手动设置 0.2500 mm" in manual


def test_manual_display_spacing_is_used_without_the_display_budget_clamp() -> None:
    domain = trimesh.creation.box(extents=(20.0, 20.0, 5.0))
    base = app.SamplingRecommendation(0.1, (203, 203, 53), 2_184_077, 40.0, 3.0)

    manual = app._display_recommendation(
        domain,
        base,
        display_voxel_size_mm=0.05,
        display_memory_budget_mb=1.0,
        visible_field_count=4,
    )

    assert manual.voxel_size_mm == 0.05
    assert manual.grid_shape == (403, 403, 103)


def test_display_batch_sampling_preserves_the_cached_implicit_field() -> None:
    domain = trimesh.creation.box(extents=(4.0, 4.0, 4.0))
    recommendation = app.SamplingRecommendation(
        0.5,
        (11, 11, 11),
        1_331,
        4.0,
        1.6,
    )
    progress_messages = []

    def evaluate(points, _reporter):
        return (np.linalg.norm(points, axis=1) - 1.0).astype(np.float32)

    body = app.ImplicitBody("sphere", domain.bounds, evaluate)
    single = app._sample_implicit_body(
        body,
        domain,
        recommendation,
        (0.7, 0.7, 0.7, 1.0),
        display_batch_count=None,
    )
    batched = app._sample_implicit_body(
        body,
        domain,
        recommendation,
        (0.7, 0.7, 0.7, 1.0),
        progress=lambda message, _value: progress_messages.append(message),
        display_batch_count=4,
    )

    np.testing.assert_array_equal(single.values, batched.values)
    assert any("批次 2/4" in message for message in progress_messages)
    assert batched.values.shape == recommendation.grid_shape


def test_cell_display_estimate_reports_voxel_samples_per_unit_cell() -> None:
    cell_map = app.CellMap.from_bounds(
        np.array(((0.0, 0.0, 0.0), (10.0, 12.0, 8.0))),
        spacing_mm=(10.0, 12.0, 8.0),
    )

    estimate = app.estimate_cell_display_samples(cell_map, 0.5)

    assert estimate.samples_per_axis == (20.0, 24.0, 16.0)
    assert estimate.voxel_samples_per_cell == 7_680


def test_automatic_display_cache_can_be_finer_than_user_generation_voxel(monkeypatch) -> None:
    """The interactive cache must preserve wall sampling independently."""

    domain = trimesh.creation.box(extents=(6.0, 12.0, 2.0))
    sampling = app.SamplingParameters(target_voxels=100_000, use_cpp_sdf=False)
    parameters = app.TPMSParameters("G", cell_size_mm=10.0, wall_thickness_mm=0.8)
    generation_recommendation = app.recommendation_for_voxel_size(
        domain,
        parameters,
        sampling,
        voxel_size_mm=1.0,
    )

    monkeypatch.setattr(
        app,
        "_compute_design_domain_sdf",
        lambda _mesh, points, _progress, _use_cpp: np.full(
            len(points), -1.0, dtype=np.float32
        ),
    )
    display_base_recommendation = app.recommend_voxel_size(domain, parameters, sampling)
    preview = app.generate_implicit_domain_preview(
        domain,
        sampling,
        display_base_recommendation,
        display_memory_budget_mb=64.0,
    )

    expected_display_step = app.recommend_display_voxel_size(
        domain,
        display_base_recommendation.voxel_size_mm,
        display_memory_budget_mb=64.0,
        visible_field_count=1,
    )
    assert generation_recommendation.voxel_size_mm == 1.0
    assert preview.spacing[0] == expected_display_step
    assert preview.spacing[0] < generation_recommendation.voxel_size_mm


def test_implicit_generation_does_not_extract_a_triangle_mesh(monkeypatch) -> None:
    domain = trimesh.creation.box(extents=(4.0, 4.0, 4.0))

    def fake_domain(_mesh, points, _progress, _use_cpp):
        return np.full(len(points), -10.0, dtype=np.float32)

    def fake_tpms(_kind, points, _params, max_tile_points=None):
        center = np.zeros(3, dtype=np.float32)
        return (np.linalg.norm(points.astype(np.float32) - center, axis=1) - 1.0).astype(np.float32)

    monkeypatch.setattr(app, "_compute_design_domain_sdf", fake_domain)
    monkeypatch.setattr(app, "_tpms_shell_field", fake_tpms)
    monkeypatch.setattr(
        app,
        "_marching_cubes_field",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("Marching Cubes must be deferred")),
    )

    result = app.generate_implicit_lattice(
        domain,
        app.TPMSParameters("G", cell_size_mm=2.0, wall_thickness_mm=0.2),
        app.SamplingParameters(target_voxels=100_000, use_cpp_sdf=False),
        voxel_size_mm=0.5,
        display_voxel_size_mm=0.5,
        display_memory_budget_mb=64.0,
    )

    assert result.body.name == "G"
    assert result.display_field.values.shape == (11, 11, 11)
    assert result.display_field.values.min() < 0 < result.display_field.values.max()
    assert result.display_fragment_cleanup is not None


def test_tpms_display_generation_removes_a_small_negative_island(monkeypatch) -> None:
    domain = trimesh.creation.box(extents=(4.0, 4.0, 4.0))

    def fake_sample(*_args, **_kwargs):
        values = np.ones((21, 21, 21), dtype=np.float32)
        values[6:14, 6:14, 6:14] = -1.0
        # A shallow isolated negative excursion represents a display artifact;
        # a full-depth negative voxel is intentionally retained by the
        # conservative default relative-volume threshold.
        values[2, 2, 2] = -0.01
        return app.SampledImplicitField(
            "G",
            values,
            np.full(3, -5.0),
            np.full(3, 0.5),
            (0.7, 0.7, 0.7, 1.0),
        )

    monkeypatch.setattr(app, "_sample_implicit_body", fake_sample)
    result = app.generate_implicit_lattice(
        domain,
        app.TPMSParameters("G", cell_size_mm=2.0, wall_thickness_mm=0.2),
        app.SamplingParameters(target_voxels=100_000, use_cpp_sdf=False),
        voxel_size_mm=0.5,
        display_voxel_size_mm=0.5,
        display_memory_budget_mb=64.0,
    )

    assert result.display_fragment_cleanup is not None
    assert result.display_fragment_cleanup.removed_components == 1
    assert result.display_field.values[2, 2, 2] > 0.0


def test_tpms_display_generation_cleans_cached_domain_path(monkeypatch) -> None:
    domain = trimesh.creation.box(extents=(4.0, 4.0, 4.0))
    values = np.ones((21, 21, 21), dtype=np.float32)
    values[6:14, 6:14, 6:14] = -1.0
    values[2, 2, 2] = -0.01
    sampled = app.SampledImplicitField(
        "G",
        values,
        np.full(3, -5.0),
        np.full(3, 0.5),
        (0.7, 0.7, 0.7, 1.0),
    )
    cached_domain = app.SampledImplicitField(
        "domain",
        np.full((21, 21, 21), -10.0, dtype=np.float32),
        np.full(3, -5.0),
        np.full(3, 0.5),
        (0.7, 0.7, 0.7, 0.3),
    )

    monkeypatch.setattr(
        app,
        "_sample_intersection_with_domain_cache",
        lambda *_args, **_kwargs: sampled,
    )
    result = app.generate_implicit_lattice(
        domain,
        app.TPMSParameters("G", cell_size_mm=2.0, wall_thickness_mm=0.2),
        app.SamplingParameters(target_voxels=100_000, use_cpp_sdf=False),
        voxel_size_mm=0.5,
        domain_display_field=cached_domain,
    )

    assert result.display_fragment_cleanup is not None
    assert result.display_fragment_cleanup.removed_components == 1
    assert result.display_field.values[2, 2, 2] > 0.0


def test_lattice_preview_reuses_a_matching_domain_field(monkeypatch) -> None:
    domain = trimesh.creation.box(extents=(4.0, 4.0, 4.0))
    shape = (11, 11, 11)
    cached_domain = app.SampledImplicitField(
        "domain",
        np.full(shape, -10.0, dtype=np.float32),
        np.full(3, -2.5),
        np.full(3, 0.5),
        (0.7, 0.7, 0.7, 0.3),
    )

    monkeypatch.setattr(
        app,
        "_compute_design_domain_sdf",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("the cached design-domain SDF should be reused")
        ),
    )
    monkeypatch.setattr(
        app,
        "_tpms_shell_field",
        lambda _kind, points, _params, max_tile_points=None: (
            np.linalg.norm(points.astype(np.float32), axis=1) - 1.0
        ).astype(np.float32),
    )

    result = app.generate_implicit_lattice(
        domain,
        app.TPMSParameters("G", cell_size_mm=2.0, wall_thickness_mm=0.2),
        app.SamplingParameters(target_voxels=100_000, use_cpp_sdf=False),
        voxel_size_mm=0.5,
        domain_display_field=cached_domain,
    )

    assert result.display_field.values.shape == shape
    assert np.shares_memory(result.display_field.values, cached_domain.values) is False


def test_implicit_generation_uses_automatic_cuda_backend() -> None:
    backend = get_default_tpms_backend()
    if not backend.status.using_gpu:
        pytest.skip("CUDA backend is unavailable")
    domain = trimesh.creation.box(extents=(4.0, 4.0, 4.0))
    shape = (11, 11, 11)
    cached_domain = app.SampledImplicitField(
        "domain",
        np.full(shape, -10.0, dtype=np.float32),
        np.full(3, -2.5),
        np.full(3, 0.5),
        (0.5, 0.5, 0.5, 1.0),
    )

    result = app.generate_implicit_lattice(
        domain,
        app.TPMSParameters("G", cell_size_mm=2.0, wall_thickness_mm=0.2),
        app.SamplingParameters(target_voxels=100_000, use_cpp_sdf=False),
        voxel_size_mm=0.5,
        domain_display_field=cached_domain,
    )

    assert result.display_field.values.shape == shape
    assert get_default_tpms_backend().status.using_gpu is True
    assert get_default_tpms_backend().status.active_backend == "Numba CUDA"


def test_non_watertight_domain_becomes_display_only_shell(monkeypatch) -> None:
    domain = trimesh.creation.box(extents=(4.0, 4.0, 4.0))
    domain.update_faces(np.arange(len(domain.faces) - 1))
    domain.remove_unreferenced_vertices()

    def fake_distance(_mesh, points, _progress, _use_cpp):
        return (np.linalg.norm(points.astype(np.float32), axis=1) - 1.0).astype(np.float32)

    monkeypatch.setattr(app, "_compute_design_domain_sdf", fake_distance)
    sampling = app.SamplingParameters(target_voxels=100_000, use_cpp_sdf=False)
    base = app.recommend_voxel_size(
        trimesh.creation.box(extents=(4.0, 4.0, 4.0)),
        app.TPMSParameters("G", cell_size_mm=2.0, wall_thickness_mm=0.2),
        sampling,
    )

    preview = app.generate_implicit_domain_preview(
        domain,
        sampling,
        base,
        display_voxel_size_mm=0.5,
        display_memory_budget_mb=64.0,
    )

    assert preview.is_preview_only is True
    assert preview.values.min() < 0 < preview.values.max()


def test_gpu_actor_uses_zero_isosurface_mapper() -> None:
    renderer = _load_renderer_module()
    axis = np.linspace(-1.5, 1.5, 24, dtype=np.float32)
    x, y, z = np.meshgrid(axis, axis, axis, indexing="ij")
    values = np.sqrt(x * x + y * y + z * z) - 1.0
    field = app.SampledImplicitField(
        "sphere",
        values,
        np.full(3, -1.5),
        np.full(3, axis[1] - axis[0]),
        (0.9, 0.3, 0.2, 1.0),
    )

    image, volume = renderer.build_implicit_volume_actor(field)

    assert image.dimensions == field.values.shape
    assert volume.GetMapper().GetBlendMode() == 5
    assert volume.GetProperty().GetIsoSurfaceValues().GetValue(0) == 0.0


def test_hidden_implicit_field_reuses_its_resident_render_cache() -> None:
    """Hiding a layer must not force another CPU-to-GPU field upload."""

    from PyQt5 import QtWidgets

    renderer = _load_renderer_module()
    qt_app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    viewer = renderer.PyVistaRenderer()
    axis = np.linspace(-1.5, 1.5, 24, dtype=np.float32)
    x, y, z = np.meshgrid(axis, axis, axis, indexing="ij")
    field = app.SampledImplicitField(
        "cached-sphere",
        np.sqrt(x * x + y * y + z * z) - 1.0,
        np.full(3, -1.5),
        np.full(3, axis[1] - axis[0]),
        (0.9, 0.3, 0.2, 1.0),
    )

    try:
        viewer.set_implicit_fields([field], reset_view=False)
        first = viewer.get_implicit_cache_statistics()

        viewer.set_implicit_fields([], reset_view=False)
        viewer.set_implicit_fields([field], reset_view=False)
        reused = viewer.get_implicit_cache_statistics()

        assert first.cache_misses == 1
        assert reused.cache_misses == 1
        assert reused.cache_hits == 1
        assert reused.resident_layers == 1
        assert reused.resident_bytes == field.estimated_bytes
    finally:
        viewer.close()
        qt_app.processEvents()


def test_implicit_render_cache_survives_a_mesh_mode_round_trip() -> None:
    from PyQt5 import QtWidgets

    renderer = _load_renderer_module()
    qt_app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    viewer = renderer.PyVistaRenderer()
    values = np.linspace(-1.0, 1.0, 1_000, dtype=np.float32).reshape(10, 10, 10)
    field = app.SampledImplicitField(
        "round-trip",
        values,
        np.zeros(3),
        np.full(3, 0.4),
        (0.3, 0.6, 0.8, 1.0),
    )
    mesh = renderer.MeshData(
        np.array(
            ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
            dtype=np.float32,
        ),
        np.array(((0, 1, 2),), dtype=np.int32),
        cache_key=("round-trip-mesh", 1),
    )

    try:
        viewer.set_implicit_fields([field], reset_view=False)
        viewer.set_meshes([mesh], reset_view=False)
        viewer.set_implicit_fields([field], reset_view=False)

        statistics = viewer.get_implicit_cache_statistics()
        assert statistics.cache_misses == 1
        assert statistics.cache_hits == 1
        assert statistics.resident_layers == 1
    finally:
        viewer.close()
        qt_app.processEvents()


def test_implicit_render_cache_evicts_only_inactive_layers_over_budget() -> None:
    from PyQt5 import QtWidgets

    renderer = _load_renderer_module()
    qt_app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    viewer = renderer.PyVistaRenderer()
    first = app.SampledImplicitField(
        "first",
        np.linspace(-1.0, 1.0, 8_000, dtype=np.float32).reshape(20, 20, 20),
        np.zeros(3),
        np.ones(3),
        (0.8, 0.2, 0.2, 1.0),
    )
    second = app.SampledImplicitField(
        "second",
        np.linspace(-1.0, 1.0, 8_000, dtype=np.float32).reshape(20, 20, 20),
        np.zeros(3),
        np.ones(3),
        (0.2, 0.2, 0.8, 1.0),
    )

    try:
        viewer.set_implicit_cache_budget_mb(0.05)
        viewer.set_implicit_fields([first], reset_view=False)
        viewer.set_implicit_fields([], reset_view=False)
        viewer.set_implicit_fields([second], reset_view=False)

        statistics = viewer.get_implicit_cache_statistics()
        assert statistics.cache_misses == 2
        assert statistics.resident_layers == 1
        assert statistics.resident_bytes == second.estimated_bytes
    finally:
        viewer.close()
        qt_app.processEvents()


def test_authoritative_edit_can_discard_all_inactive_implicit_layers() -> None:
    """Obsolete geometry revisions must not remain resident after an edit."""

    from PyQt5 import QtWidgets

    renderer = _load_renderer_module()
    qt_app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    viewer = renderer.PyVistaRenderer()
    first = app.SampledImplicitField(
        "domain-before-edit",
        np.linspace(-1.0, 1.0, 1_000, dtype=np.float32).reshape(10, 10, 10),
        np.zeros(3),
        np.ones(3),
        (0.7, 0.7, 0.7, 1.0),
    )
    second = app.SampledImplicitField(
        "domain-after-edit",
        np.linspace(-2.0, 2.0, 1_331, dtype=np.float32).reshape(11, 11, 11),
        np.zeros(3),
        np.ones(3),
        (0.7, 0.7, 0.7, 1.0),
    )

    try:
        viewer.set_implicit_fields([first], reset_view=False)
        viewer.set_implicit_fields([second], reset_view=False)
        assert viewer.get_implicit_cache_statistics().resident_layers == 2

        viewer.discard_inactive_implicit_layers()

        statistics = viewer.get_implicit_cache_statistics()
        assert statistics.resident_layers == 1
        assert statistics.resident_bytes == second.estimated_bytes
    finally:
        viewer.close()
        qt_app.processEvents()


def test_stale_display_refinement_cannot_refresh_a_newer_design_revision() -> None:
    """A completed field for old primitive dimensions must be ignored."""

    window_class = app._build_qt_app()
    workspace = app.DesignWorkspace()
    primitive = app.ImplicitPrimitive(
        "domain-box",
        "Domain box",
        "box",
        size_mm=(40.0, 40.0, 60.0),
    )
    document = workspace.create_analytic_document(primitive, "Box design")
    document.touch()
    stale_revision = document.revision - 1
    field = app.SampledImplicitField(
        "stale-refinement",
        np.linspace(-1.0, 1.0, 1_000, dtype=np.float32).reshape(10, 10, 10),
        np.zeros(3),
        np.ones(3),
        (0.7, 0.7, 0.7, 1.0),
    )
    report = app.DisplayRefinementReport(
        requested_spacing_mm=1.0,
        applied_spacing_mm=1.0,
        estimated_voxels=1_000,
        refined=True,
        budget_limited=False,
    )

    class _Control:
        def __init__(self, value=None) -> None:
            self.value = value
            self.history = []

        def currentData(self):
            return self.value

        def setValue(self, value) -> None:
            self.history.append(value)

        def setText(self, value) -> None:
            self.history.append(value)

    class _Harness:
        def __init__(self) -> None:
            self.design_workspace = workspace
            self.render_quality_combo = _Control("high")
            self.progress = _Control()
            self.status = _Control()
            self.refresh_count = 0

        def _prune_display_refinement_cache(self, _cache, _protected) -> None:
            pass

        def _refresh_scene(self, *, reset_view=False) -> None:
            del reset_view
            self.refresh_count += 1

    harness = _Harness()
    results = {("high", "domain", "stale"): (field, report)}

    window_class._display_refinement_progress(
        harness,
        document.identifier,
        stale_revision,
        "stale progress",
        0.5,
    )
    window_class._display_refinement_finished(
        harness,
        document.identifier,
        stale_revision,
        "high",
        results,
    )
    window_class._display_refinement_failed(
        harness,
        document.identifier,
        stale_revision,
        "high",
        "stale failure",
    )

    assert document.runtime.renderer_cache == {}
    assert harness.refresh_count == 0
    assert harness.progress.history == []
    assert harness.status.history == []


def test_implicit_section_updates_do_not_rebuild_render_resources() -> None:
    from PyQt5 import QtWidgets
    from vtkmodules.vtkCommonDataModel import vtkPlane

    renderer = _load_renderer_module()
    qt_app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    viewer = renderer.PyVistaRenderer()
    field = app.SampledImplicitField(
        "sectioned",
        np.linspace(-1.0, 1.0, 1_000, dtype=np.float32).reshape(10, 10, 10),
        np.zeros(3),
        np.full(3, 0.4),
        (0.4, 0.7, 0.3, 1.0),
    )
    plane = vtkPlane()
    plane.SetOrigin(1.0, 1.0, 1.0)
    plane.SetNormal(1.0, 0.0, 0.0)

    try:
        viewer.set_implicit_fields([field], reset_view=False)
        before = viewer.get_implicit_cache_statistics()

        viewer.set_section_plane(plane)
        plane.SetOrigin(1.5, 1.0, 1.0)
        viewer.set_section_plane(plane)

        after = viewer.get_implicit_cache_statistics()
        assert after.cache_hits == before.cache_hits
        assert after.cache_misses == before.cache_misses
        assert after.resident_layers == before.resident_layers
    finally:
        viewer.close()
        qt_app.processEvents()


def test_render_quality_changes_gpu_sampling_settings() -> None:
    renderer = _load_renderer_module()
    values = np.linspace(-1.0, 1.0, 1_000, dtype=np.float32).reshape(10, 10, 10)
    field = app.SampledImplicitField(
        "quality",
        values,
        np.zeros(3),
        np.full(3, 0.4),
        (0.4, 0.5, 0.6, 1.0),
    )

    _low_image, low = renderer.build_implicit_volume_actor(field, "low")
    _ultra_image, ultra = renderer.build_implicit_volume_actor(field, "ultra")

    assert low.GetMapper().GetSampleDistance() > ultra.GetMapper().GetSampleDistance()
    assert low.GetProperty().GetInterpolationType() != ultra.GetProperty().GetInterpolationType()


def test_ultra_quality_uses_volume_safe_antialiasing() -> None:
    renderer = _load_renderer_module()

    assert renderer.RENDER_QUALITY_SETTINGS["ultra"].anti_aliasing == "fxaa"


def test_ultra_display_refinement_resamples_the_authoritative_evaluator() -> None:
    source = app.SampledImplicitField(
        "coarse-sphere",
        np.zeros((9, 9, 9), dtype=np.float32),
        np.full(3, -2.0),
        np.full(3, 0.5),
        (0.7, 0.7, 0.7, 1.0),
    )
    evaluated_sizes = []

    def evaluate(points, _reporter):
        evaluated_sizes.append(len(points))
        return (np.linalg.norm(points, axis=1) - 1.0).astype(np.float32)

    body = app.ImplicitBody("sphere", source.bounds, evaluate)
    refined, report = app.refine_implicit_display_field(
        body,
        source,
        quality="ultra",
        display_memory_budget_mb=8.0,
    )

    assert np.all(refined.spacing < source.spacing)
    assert refined.values.shape[0] > source.values.shape[0]
    assert sum(evaluated_sizes) == refined.values.size
    assert refined.values.min() < 0.0 < refined.values.max()
    assert report.refined
    assert report.applied_spacing_mm == pytest.approx(refined.spacing[0])


def test_high_display_refinement_uses_the_same_pipeline_at_lower_resolution() -> None:
    source = app.SampledImplicitField(
        "quality-sphere",
        np.zeros((9, 9, 9), dtype=np.float32),
        np.full(3, -2.0),
        np.full(3, 0.5),
        (0.7, 0.7, 0.7, 1.0),
    )

    def evaluate(points, _reporter):
        return (np.linalg.norm(points, axis=1) - 1.0).astype(np.float32)

    body = app.ImplicitBody("sphere", source.bounds, evaluate)
    high, high_report = app.refine_implicit_display_field(
        body,
        source,
        quality="high",
        display_memory_budget_mb=8.0,
    )
    ultra, ultra_report = app.refine_implicit_display_field(
        body,
        source,
        quality="ultra",
        display_memory_budget_mb=8.0,
    )

    assert high_report.refined
    assert ultra_report.refined
    assert source.spacing[0] > high.spacing[0] > ultra.spacing[0]
    assert high.spacing[0] == pytest.approx(0.375)
    assert ultra.spacing[0] == pytest.approx(0.25)


def test_ultra_display_refinement_obeys_the_display_memory_budget() -> None:
    source = app.SampledImplicitField(
        "large-field",
        np.zeros((65, 65, 65), dtype=np.float32),
        np.zeros(3),
        np.ones(3),
        (0.7, 0.7, 0.7, 1.0),
    )

    def evaluate(points, _reporter):
        return (points[:, 0] - 32.0).astype(np.float32)

    body = app.ImplicitBody("plane", source.bounds, evaluate)
    refined, report = app.refine_implicit_display_field(
        body,
        source,
        quality="ultra",
        display_memory_budget_mb=1.25,
    )

    assert refined.estimated_bytes <= int(1.25 * 1024 * 1024)
    assert report.budget_limited
    assert report.applied_spacing_mm > report.requested_spacing_mm


def test_implicit_material_responds_to_metallic_and_roughness() -> None:
    renderer = _load_renderer_module()
    values = np.linspace(-1.0, 1.0, 1_000, dtype=np.float32).reshape(10, 10, 10)
    dielectric = app.SampledImplicitField(
        "dielectric",
        values,
        np.zeros(3),
        np.full(3, 0.4),
        (0.72, 0.34, 0.08, 1.0),
        metallic=0.0,
        roughness=0.75,
    )
    metal = app.SampledImplicitField(
        "metal",
        values,
        np.zeros(3),
        np.full(3, 0.4),
        (0.72, 0.34, 0.08, 1.0),
        metallic=0.8,
        roughness=0.25,
    )

    _dielectric_image, dielectric_actor = renderer.build_implicit_volume_actor(
        dielectric,
        "high",
    )
    _metal_image, metal_actor = renderer.build_implicit_volume_actor(metal, "high")

    dielectric_property = dielectric_actor.GetProperty()
    metal_property = metal_actor.GetProperty()
    assert metal_property.GetDiffuse() < dielectric_property.GetDiffuse()
    assert metal_property.GetSpecular() > dielectric_property.GetSpecular()
    assert metal_property.GetSpecularPower() > dielectric_property.GetSpecularPower()


def test_high_quality_implicit_render_enables_gpu_lighting_features() -> None:
    renderer = _load_renderer_module()
    values = np.linspace(-1.0, 1.0, 1_000, dtype=np.float32).reshape(10, 10, 10)
    field = app.SampledImplicitField(
        "quality",
        values,
        np.zeros(3),
        np.full(3, 0.4),
        (0.4, 0.5, 0.6, 1.0),
    )

    _low_image, low = renderer.build_implicit_volume_actor(field, "low")
    _high_image, high = renderer.build_implicit_volume_actor(field, "high")

    assert low.GetMapper().GetUseJittering() == 0
    assert high.GetMapper().GetUseJittering() == 0
    assert high.GetMapper().GetGlobalIlluminationReach() > 0.0
    assert high.GetMapper().GetVolumetricScatteringBlending() > 0.0


def test_render_backend_status_identifies_hardware_opengl() -> None:
    renderer = _load_renderer_module()
    report = "\n".join(
        (
            "OpenGL vendor string: NVIDIA Corporation",
            "OpenGL renderer string: NVIDIA GeForce RTX 4060 Laptop GPU/PCIe/SSE2",
            "OpenGL version string: 4.6.0 NVIDIA 560.94",
        )
    )

    status = renderer.RenderBackendStatus.from_capabilities(
        report,
        supports_opengl=True,
        mapper_supported=True,
    )

    assert status.using_gpu is True
    assert status.active_backend == "OpenGL GPU 光线投射"
    assert "RTX 4060" in status.device_name
    assert status.fallback_reason is None


def test_render_backend_status_does_not_misreport_software_opengl() -> None:
    renderer = _load_renderer_module()
    status = renderer.RenderBackendStatus.from_capabilities(
        "OpenGL renderer string: llvmpipe (LLVM 18.1.8, 256 bits)",
        supports_opengl=True,
        mapper_supported=True,
    )

    assert status.using_gpu is False
    assert "软件 OpenGL" in status.active_backend
    assert status.fallback_reason is not None


def test_studio_lights_are_centered_and_include_a_rim_light() -> None:
    renderer = _load_renderer_module()
    center = np.array((100.0, -25.0, 12.0))
    lights = renderer.build_studio_lights(center, scene_size=80.0)

    scene_lights = [light for light in lights if light.light_type != "Headlight"]
    assert len(scene_lights) >= 3
    assert all(np.allclose(light.focal_point, center) for light in scene_lights)
    assert any(light.position[1] < center[1] for light in scene_lights)
    assert max(light.intensity for light in lights) <= 1.0


def test_qt_domain_actor_contributes_visible_pixels() -> None:
    """The default design-domain layer must be visible in the embedded viewer."""

    from PyQt5 import QtWidgets

    qt_app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window_class = app._build_qt_app()
    window = window_class()
    window.show()
    try:
        window._create_analytic_design("sphere")

        def wait_for_refined_quality(quality: str) -> None:
            deadline = time.monotonic() + 10.0
            while time.monotonic() < deadline:
                qt_app.processEvents()
                cached_qualities = {
                    key[0]
                    for key in window._display_refinement_cache()
                    if isinstance(key, tuple) and key
                }
                if (
                    quality in cached_qualities
                    and window.display_refinement_thread is None
                    and not window._display_refinement_refresh_pending
                ):
                    return
            raise AssertionError(f"{quality} display refinement did not finish")

        wait_for_refined_quality("high")
        for _ in range(3):
            qt_app.processEvents()
        window.viewer.reset_view()
        window.viewer.render()
        visible = np.asarray(window.viewer.screenshot(return_img=True), dtype=np.int16)

        assert len(window.viewer._implicit_actors) == 1
        actor = window.viewer._implicit_actors[0]
        assert window.domain_implicit_field.values.min() < 0
        assert window.domain_implicit_field.values.max() > 0
        assert np.isclose(actor.GetProperty().GetScalarOpacity().GetValue(0.0), 1.0)
        actor.SetVisibility(False)
        window.viewer.render()
        hidden = np.asarray(window.viewer.screenshot(return_img=True), dtype=np.int16)

        changed_pixels = np.count_nonzero(np.any(np.abs(visible - hidden) >= 4, axis=2))
        assert changed_pixels >= 500

        cached_domain = window.domain_implicit_field
        before_toggle = window.viewer.get_implicit_cache_statistics()
        window.show_domain.setChecked(False)
        qt_app.processEvents()
        window.show_domain.setChecked(True)
        qt_app.processEvents()
        after_toggle = window.viewer.get_implicit_cache_statistics()
        assert after_toggle.cache_misses == before_toggle.cache_misses
        assert after_toggle.cache_hits == before_toggle.cache_hits + 1

        window.render_quality_combo.setCurrentIndex(3)
        qt_app.processEvents()
        assert window.viewer.render_quality == "ultra"
        assert window.domain_implicit_field is cached_domain
        assert len(window.viewer._implicit_actors) == 1
        window.viewer.render()
        ultra_visible = np.asarray(
            window.viewer.screenshot(return_img=True), dtype=np.int16
        )
        window.viewer._implicit_actors[0].SetVisibility(False)
        window.viewer.render()
        ultra_hidden = np.asarray(
            window.viewer.screenshot(return_img=True), dtype=np.int16
        )
        ultra_changed_pixels = np.count_nonzero(
            np.any(np.abs(ultra_visible - ultra_hidden) >= 4, axis=2)
        )
        assert ultra_changed_pixels >= 500

        wait_for_refined_quality("ultra")
        assert window.display_refinement_thread is None
        cache = window._display_refinement_cache()
        assert {key[0] for key in cache} == {"high", "ultra"}
        high_field, high_report = next(
            value for key, value in cache.items() if key[0] == "high"
        )
        ultra_field, ultra_report = next(
            value for key, value in cache.items() if key[0] == "ultra"
        )
        assert high_report.refined
        assert ultra_report.refined
        assert np.all(
            cached_domain.spacing > high_field.spacing
        )
        assert np.all(high_field.spacing > ultra_field.spacing)

        window.render_quality_combo.setCurrentIndex(
            window.render_quality_combo.findData("high")
        )
        window.render_quality_combo.setCurrentIndex(
            window.render_quality_combo.findData("ultra")
        )
        qt_app.processEvents()
        assert window.display_refinement_thread is None
        assert len(window._display_refinement_cache()) == 2
    finally:
        window.close()
        qt_app.processEvents()
