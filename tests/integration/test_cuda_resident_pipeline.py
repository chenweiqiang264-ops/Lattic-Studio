"""Behavior tests for the device-resident mesh/TPMS field pipeline."""

from __future__ import annotations

import numpy as np
import pytest
import trimesh

import lattice_studio.engine.implicit.lattice_pipeline as lattice_pipeline
from lattice_studio.engine.implicit.lattice_pipeline import (
    AutomaticLatticeFieldPipeline,
    CpuLatticeFieldPipeline,
    NumbaCudaLatticeFieldPipeline,
)
from lattice_studio.engine.implicit.tpms_compute import TPMSFieldSpec, TPMSGradientSpec

from lattice_studio.engine import workflows as app


def _points() -> np.ndarray:
    return np.random.default_rng(73).uniform(-1.4, 1.4, (10_003, 3)).astype(
        np.float32
    )


def test_cuda_resident_pipeline_matches_cpu_reference() -> None:
    gpu = NumbaCudaLatticeFieldPipeline()
    if not gpu.available:
        pytest.skip("CUDA resident pipeline is unavailable")
    cpu = CpuLatticeFieldPipeline()
    domain = trimesh.creation.box(extents=(2.0, 2.0, 2.0))
    points = _points()
    spec = TPMSFieldSpec("D", 1.25, 0.22, level=0.06)

    expected = cpu.evaluate_mesh_clipped_tpms(
        domain.vertices, domain.faces, points, spec, None, 2_501
    )
    actual = gpu.evaluate_mesh_clipped_tpms(
        domain.vertices, domain.faces, points, spec, None, 2_501
    )

    np.testing.assert_allclose(actual, expected, rtol=2e-4, atol=8e-5)


def test_cuda_resident_pipeline_supports_gradient_tpms() -> None:
    gpu = NumbaCudaLatticeFieldPipeline()
    if not gpu.available:
        pytest.skip("CUDA resident pipeline is unavailable")
    cpu = CpuLatticeFieldPipeline()
    domain = trimesh.creation.icosphere(subdivisions=2, radius=1.0)
    points = _points()
    spec = TPMSFieldSpec("IWP", 1.4, 0.2)
    gradient = TPMSGradientSpec(
        axis_index=0,
        coordinate_min_mm=-1.0,
        coordinate_max_mm=1.0,
        mode="sigmoid",
        thickness_soft_mm=0.14,
        thickness_stiff_mm=0.28,
        offset_soft=-0.1,
        offset_stiff=0.1,
    )

    expected = cpu.evaluate_mesh_clipped_tpms(
        domain.vertices, domain.faces, points, spec, gradient, 2_501
    )
    actual = gpu.evaluate_mesh_clipped_tpms(
        domain.vertices, domain.faces, points, spec, gradient, 2_501
    )

    np.testing.assert_allclose(actual, expected, rtol=3e-4, atol=1e-4)


def test_cuda_resident_pipeline_uploads_each_point_tile_once(monkeypatch) -> None:
    gpu = NumbaCudaLatticeFieldPipeline()
    if not gpu.available:
        pytest.skip("CUDA resident pipeline is unavailable")
    domain = trimesh.creation.box(extents=(2.0, 2.0, 2.0))
    points = _points()[:4_097]
    spec = TPMSFieldSpec("G", 1.5, 0.2)
    original = lattice_pipeline._numba_cuda.to_device
    uploaded_point_tiles = 0

    def counting_to_device(value, *args, **kwargs):
        nonlocal uploaded_point_tiles
        array = np.asarray(value)
        if array.ndim == 2 and array.shape[1] == 3 and array.dtype == np.float32:
            uploaded_point_tiles += 1
        return original(value, *args, **kwargs)

    monkeypatch.setattr(lattice_pipeline._numba_cuda, "to_device", counting_to_device)

    gpu.evaluate_mesh_clipped_tpms(
        domain.vertices,
        domain.faces,
        points,
        spec,
        None,
        max_tile_points=2_048,
    )

    assert uploaded_point_tiles == 3


def test_cuda_resident_pipeline_reuses_mesh_bvh_between_calls(monkeypatch) -> None:
    gpu = NumbaCudaLatticeFieldPipeline()
    if not gpu.available:
        pytest.skip("CUDA resident pipeline is unavailable")
    domain = trimesh.creation.box(extents=(2.0, 2.0, 2.0))
    points = _points()[:257]
    spec = TPMSFieldSpec("G", 1.5, 0.2)
    geometry_module = __import__(
        "lattice_studio.engine.implicit.geometry_compute", fromlist=["_build_flat_bvh"]
    )
    original = geometry_module._build_flat_bvh
    prepare_calls = 0

    def counting_build(*args, **kwargs):
        nonlocal prepare_calls
        prepare_calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(geometry_module, "_build_flat_bvh", counting_build)
    gpu.evaluate_mesh_clipped_tpms(
        domain.vertices, domain.faces, points, spec, None, 128
    )
    gpu.evaluate_mesh_clipped_tpms(
        domain.vertices, domain.faces, points, spec, None, 128
    )

    assert prepare_calls == 1


class _PipelineStub:
    def __init__(self, name, *, available=True, error=None):
        self.name = name
        self.available = available
        self.device_name = name
        self.unavailable_reason = None if available else "unavailable"
        self.error = error

    def evaluate_mesh_clipped_tpms(self, *_args, **_kwargs):
        if self.error is not None:
            raise self.error
        return np.zeros(3, dtype=np.float32)


def test_invalid_pipeline_request_does_not_disable_gpu() -> None:
    automatic = AutomaticLatticeFieldPipeline(
        cpu_pipeline=_PipelineStub("cpu"),
        gpu_pipeline=_PipelineStub("gpu", error=ValueError("bad request")),
    )

    with pytest.raises(ValueError, match="bad request"):
        automatic.evaluate_mesh_clipped_tpms()

    assert automatic.status.using_gpu is True
    assert automatic.status.fallback_reason is None


def test_workbench_routes_gradient_mesh_lattice_through_resident_pipeline(
    monkeypatch,
) -> None:
    domain = trimesh.creation.box(extents=(4.0, 4.0, 4.0))
    parameters = app.TPMSParameters(
        "IWP",
        cell_size_mm=(2.0, 2.5, 3.0),
        wall_thickness_mm=0.25,
        gradient_enabled=True,
        gradient_axis="V",
        gradient_mode="power",
        gradient_power=3.0,
        thickness_soft_mm=0.18,
        thickness_stiff_mm=0.32,
    )
    sampling = app.SamplingParameters(target_voxels=100_000, use_cpp_sdf=True)
    recorded = {}

    class RecordingPipeline:
        def evaluate_mesh_clipped_tpms(
            self, vertices, faces, points, spec, gradient, max_tile_points
        ):
            recorded.update(
                vertices=vertices,
                faces=faces,
                spec=spec,
                gradient=gradient,
                max_tile_points=max_tile_points,
            )
            return np.full(len(points), -0.25, dtype=np.float32)

    monkeypatch.setattr(
        app, "get_default_lattice_pipeline", lambda: RecordingPipeline()
    )
    body = app._make_lattice_body(domain, parameters, sampling)
    actual = body.evaluate_points(np.zeros((7, 3), dtype=np.float32))

    np.testing.assert_array_equal(actual, np.full(7, -0.25, dtype=np.float32))
    assert recorded["vertices"] is domain.vertices
    assert recorded["faces"] is domain.faces
    assert recorded["spec"].kind == "IWP"
    assert recorded["gradient"].axis_index == 1
    assert recorded["gradient"].mode == "power"
    assert recorded["gradient"].thickness_soft_mm == pytest.approx(0.18)
    assert recorded["gradient"].thickness_stiff_mm == pytest.approx(0.32)
    assert recorded["max_tile_points"] == 250_000


def test_direct_stl_generation_routes_through_resident_pipeline(monkeypatch) -> None:
    domain = trimesh.creation.box(extents=(4.0, 4.0, 4.0))
    parameters = app.TPMSParameters(
        "G", cell_size_mm=2.0, wall_thickness_mm=0.2
    )
    sampling = app.SamplingParameters(
        target_voxels=100_000,
        use_cpp_sdf=True,
        processing_mode="single_pass",
    )
    calls = 0

    class RecordingPipeline:
        def evaluate_mesh_clipped_tpms(
            self, _vertices, _faces, points, spec, gradient, max_tile_points
        ):
            nonlocal calls
            calls += 1
            assert spec.kind == "G"
            assert gradient is None
            assert max_tile_points is None
            return (
                np.linalg.norm(np.asarray(points, dtype=np.float32), axis=1) - 1.5
            ).astype(np.float32)

    monkeypatch.setattr(
        app, "get_default_lattice_pipeline", lambda: RecordingPipeline()
    )
    result, _recommendation = app.generate_tpms_lattice(
        domain,
        parameters,
        sampling,
        voxel_size_mm=1.0,
    )

    assert calls == 1
    assert len(result.faces) > 0
