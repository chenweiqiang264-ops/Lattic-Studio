"""Behavior tests for automatic GPU geometry computation."""

from __future__ import annotations

import numpy as np
import pytest
import trimesh

import lattice_studio.engine.implicit.geometry_compute as geometry_compute
from lattice_studio.engine.implicit.geometry_compute import (
    AutomaticGeometryBackend,
    CpuGeometryAdapter,
    NumbaCudaGeometryAdapter,
    _local_edge_ids_to_global,
)


def test_chunk_edge_identity_mapping_is_stable_across_boundaries() -> None:
    actual = _local_edge_ids_to_global(
        np.array((0, 39, 40, 84, 85, 132), dtype=np.int64),
        local_shape=(3, 4, 5),
        grid_offset=(2, 3, 4),
        global_shape=(10, 20, 30),
    )

    np.testing.assert_array_equal(
        actual,
        np.array((1294, 1988, 6634, 7838, 12351, 13601), dtype=np.int64),
    )


class _StubGeometryAdapter:
    def __init__(self, name: str, available: bool, error: Exception | None = None):
        self.name = name
        self.available = available
        self.device_name = name
        self.error = error

    def signed_distance(self, vertices, faces, points, max_tile_points):
        if self.error is not None:
            raise self.error
        return np.full(len(points), 2.0 if self.name == "gpu" else -1.0, np.float32)

    def intersect_fields(self, field_a, field_b, max_tile_points):
        if self.error is not None:
            raise self.error
        return np.maximum(field_a, field_b)

    def marching_cubes(self, field, spacing, origin, level):
        if self.error is not None:
            raise self.error
        return trimesh.creation.box(extents=(1.0, 1.0, 1.0))

    def marching_cubes_chunk(
        self,
        field,
        spacing,
        origin,
        level,
        grid_offset,
        global_shape,
    ):
        del grid_offset, global_shape
        return self.marching_cubes(field, spacing, origin, level), None


def test_automatic_geometry_backend_uses_gpu_and_falls_back() -> None:
    gpu = _StubGeometryAdapter("gpu", True)
    cpu = _StubGeometryAdapter("cpu", True)
    backend = AutomaticGeometryBackend(cpu_adapter=cpu, gpu_adapter=gpu)
    points = np.zeros((3, 3), dtype=np.float32)

    values = backend.signed_distance(
        np.zeros((3, 3)),
        np.array(((0, 1, 2),)),
        points,
    )

    np.testing.assert_array_equal(values, np.full(3, 2.0, np.float32))
    assert backend.status.using_gpu is True

    failing = _StubGeometryAdapter("gpu", True, RuntimeError("CUDA failed"))
    backend = AutomaticGeometryBackend(cpu_adapter=cpu, gpu_adapter=failing)
    values = backend.signed_distance(
        np.zeros((3, 3)),
        np.array(((0, 1, 2),)),
        points,
    )

    np.testing.assert_array_equal(values, np.full(3, -1.0, np.float32))
    assert backend.status.using_gpu is False
    assert backend.status.fallback_reason == "RuntimeError: CUDA failed"


def test_geometry_backend_reports_startup_gpu_probe_failure() -> None:
    cpu = _StubGeometryAdapter("cpu", True)
    gpu = _StubGeometryAdapter("gpu", False)
    gpu.unavailable_reason = "OSError: CUDA context unavailable"

    backend = AutomaticGeometryBackend(cpu_adapter=cpu, gpu_adapter=gpu)

    assert backend.status.using_gpu is False
    assert backend.status.fallback_reason == "OSError: CUDA context unavailable"


def test_invalid_input_does_not_permanently_disable_gpu_backend() -> None:
    cpu = _StubGeometryAdapter("cpu", True)
    gpu = _StubGeometryAdapter("gpu", True)
    backend = AutomaticGeometryBackend(cpu_adapter=cpu, gpu_adapter=gpu)

    with pytest.raises(ValueError, match="matching finite values"):
        backend.intersect_fields(
            np.zeros(4, dtype=np.float32),
            np.zeros(3, dtype=np.float32),
        )

    assert backend.status.using_gpu is True
    assert backend.status.fallback_reason is None
    np.testing.assert_array_equal(
        backend.intersect_fields(
            np.zeros(4, dtype=np.float32),
            np.ones(4, dtype=np.float32),
        ),
        np.ones(4, dtype=np.float32),
    )


def test_chunked_marching_cubes_falls_back_to_cpu() -> None:
    cpu = _StubGeometryAdapter("cpu", True)
    gpu = _StubGeometryAdapter("gpu", True, RuntimeError("CUDA failed"))
    backend = AutomaticGeometryBackend(cpu_adapter=cpu, gpu_adapter=gpu)

    mesh, edge_ids = backend.marching_cubes_chunk(
        np.ones((3, 3, 3), dtype=np.float32),
        spacing=(1.0, 1.0, 1.0),
        origin=np.zeros(3),
        level=0.0,
        grid_offset=(0, 0, 0),
        global_shape=(3, 3, 3),
    )

    assert len(mesh.faces) == 12
    assert edge_ids is None
    assert backend.status.using_gpu is False
    assert backend.status.fallback_reason == "RuntimeError: CUDA failed"


def test_cuda_signed_distance_matches_known_box_distances() -> None:
    gpu = NumbaCudaGeometryAdapter()
    if not gpu.available:
        pytest.skip("CUDA geometry backend is unavailable")
    box = trimesh.creation.box(extents=(2.0, 2.0, 2.0))
    points = np.array(
        (
            (0.0, 0.0, 0.0),
            (0.5, 0.5, 0.5),
            (1.5, 0.0, 0.0),
            (0.0, 0.0, 1.25),
            (2.0, 2.0, 2.0),
        ),
        dtype=np.float32,
    )

    actual = gpu.signed_distance(box.vertices, box.faces, points, 3)

    expected = np.array((-1.0, -0.5, 0.5, 0.25, np.sqrt(3.0)), dtype=np.float32)
    np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-5)


def test_cuda_signed_distance_does_not_move_near_surface_sign_rays() -> None:
    gpu = NumbaCudaGeometryAdapter()
    if not gpu.available:
        pytest.skip("CUDA geometry backend is unavailable")
    cylinder = trimesh.creation.cylinder(radius=1.0, height=0.4, sections=512)
    points = np.column_stack(
        (
            np.full(13, -0.8, dtype=np.float32),
            np.full(13, -0.6, dtype=np.float32),
            np.linspace(-0.15, 0.15, 13, dtype=np.float32),
        )
    )

    values = gpu.signed_distance(
        cylinder.vertices,
        cylinder.faces,
        points,
        max_tile_points=32,
    )

    assert np.all(values > 0.0), values
    np.testing.assert_allclose(values, values[0], rtol=2e-3, atol=2e-6)


def test_cuda_intersection_matches_numpy_reference() -> None:
    gpu = NumbaCudaGeometryAdapter()
    if not gpu.available:
        pytest.skip("CUDA geometry backend is unavailable")
    rng = np.random.default_rng(20260806)
    field_a = rng.normal(size=10_013).astype(np.float32).reshape(17, 19, 31)
    field_b = rng.normal(size=10_013).astype(np.float32).reshape(17, 19, 31)

    actual = gpu.intersect_fields(field_a, field_b, max_tile_points=1_024)

    np.testing.assert_array_equal(actual, np.maximum(field_a, field_b))


def test_cuda_marching_cubes_extracts_closed_sphere() -> None:
    gpu = NumbaCudaGeometryAdapter()
    if not gpu.available:
        pytest.skip("CUDA geometry backend is unavailable")
    axis = np.linspace(-3.0, 3.0, 25, dtype=np.float32)
    x, y, z = np.meshgrid(axis, axis, axis, indexing="ij")
    field = (x * x + y * y + z * z - 4.0).astype(np.float32)

    mesh = gpu.marching_cubes(
        field,
        spacing=(0.25, 0.25, 0.25),
        origin=np.array((-3.0, -3.0, -3.0)),
        level=0.0,
    )

    assert len(mesh.faces) > 0
    assert mesh.is_watertight
    np.testing.assert_allclose(mesh.bounds, ((-2, -2, -2), (2, 2, 2)), atol=0.08)
    assert np.isclose(abs(mesh.volume), 4.0 / 3.0 * np.pi * 8.0, rtol=0.08)


def test_cuda_marching_cubes_does_not_sort_duplicate_edges_on_host(monkeypatch) -> None:
    """CUDA extraction must compact grid edges before copying mesh data home."""

    gpu = NumbaCudaGeometryAdapter()
    if not gpu.available:
        pytest.skip("CUDA geometry backend is unavailable")
    axis = np.linspace(-2.0, 2.0, 19, dtype=np.float32)
    x, y, z = np.meshgrid(axis, axis, axis, indexing="ij")
    field = (x * x + y * y + z * z - 1.5**2).astype(np.float32)

    # Initialize and compile the CUDA path before replacing the host helper.
    gpu.marching_cubes(
        field[:5, :5, :5],
        spacing=(axis[1] - axis[0],) * 3,
        origin=np.full(3, axis[0]),
        level=0.0,
    )

    def reject_host_unique(*_args, **_kwargs):
        raise AssertionError("CUDA Marching Cubes used host np.unique")

    monkeypatch.setattr(geometry_compute.np, "unique", reject_host_unique)
    mesh = gpu.marching_cubes(
        field,
        spacing=(axis[1] - axis[0],) * 3,
        origin=np.full(3, axis[0]),
        level=0.0,
    )

    assert len(mesh.faces) > 0
    assert len(mesh.vertices) < len(mesh.faces) * 3
    assert mesh.is_watertight


def test_cuda_marching_cubes_matches_cpu_lorensen_topology() -> None:
    gpu = NumbaCudaGeometryAdapter()
    if not gpu.available:
        pytest.skip("CUDA geometry backend is unavailable")
    rng = np.random.default_rng(20260806)
    field = rng.normal(size=(12, 13, 14)).astype(np.float32)
    field[[0, -1], :, :] = 1.0
    field[:, [0, -1], :] = 1.0
    field[:, :, [0, -1]] = 1.0
    spacing = (0.4, 0.5, 0.6)
    origin = np.array((-2.0, 3.0, 1.0))

    expected = CpuGeometryAdapter().marching_cubes(field, spacing, origin, 0.0)
    actual = gpu.marching_cubes(field, spacing, origin, 0.0)

    assert len(actual.vertices) == len(expected.vertices)
    assert len(actual.faces) == len(expected.faces)
    assert actual.is_watertight == expected.is_watertight
    np.testing.assert_allclose(actual.bounds, expected.bounds, rtol=1e-6, atol=1e-6)


def test_cuda_marching_cubes_stays_closed_after_degenerate_face_cleanup() -> None:
    gpu = NumbaCudaGeometryAdapter()
    if not gpu.available:
        pytest.skip("CUDA geometry backend is unavailable")
    x_axis = -3.1 + np.arange(63, dtype=np.float64) * 0.1
    yz_axis = -2.1 + np.arange(43, dtype=np.float64) * 0.1
    x, y, z = np.meshgrid(x_axis, yz_axis, yz_axis, indexing="ij")
    first = np.sqrt((x + 1.1) ** 2 + y**2 + z**2) - 1.0
    second = np.sqrt((x - 1.1) ** 2 + y**2 + z**2) - 1.0
    field = np.minimum(first, second).astype(np.float32)

    mesh = gpu.marching_cubes(
        field,
        spacing=(0.1, 0.1, 0.1),
        origin=np.array((-3.1, -2.1, -2.1)),
        level=0.0,
    )
    mesh.merge_vertices()
    mesh.update_faces(mesh.unique_faces())
    mesh.update_faces(mesh.nondegenerate_faces())
    mesh.remove_unreferenced_vertices()

    assert mesh.is_watertight
    assert len(mesh.split(only_watertight=False)) == 2


def test_cpu_geometry_adapter_remains_a_reference_path() -> None:
    cpu = CpuGeometryAdapter()
    box = trimesh.creation.box(extents=(2.0, 2.0, 2.0))
    points = np.array(((0.0, 0.0, 0.0), (1.5, 0.0, 0.0)))

    values = cpu.signed_distance(box.vertices, box.faces, points, None)

    np.testing.assert_allclose(values, (-1.0, 0.5), atol=1e-8)
