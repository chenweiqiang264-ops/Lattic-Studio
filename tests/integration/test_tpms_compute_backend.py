"""Behavior tests for automatic TPMS numerical backend selection."""

from __future__ import annotations

import numpy as np
import pytest

from lattice_studio.engine.implicit.tpms_compute import (
    AutomaticTPMSBackend,
    NumbaCudaTPMSAdapter,
    NumpyTPMSAdapter,
    TPMSFieldSpec,
    TPMSGradientSpec,
    TransitionBlendSpec,
)


class _StubAdapter:
    def __init__(
        self,
        name: str,
        available: bool,
        value: float,
        error: Exception | None = None,
    ):
        self.name = name
        self.available = available
        self.device_name = name
        self.value = value
        self.error = error

    def evaluate_shell(
        self,
        points: np.ndarray,
        _spec: TPMSFieldSpec,
        _max_tile_points: int | None,
    ) -> np.ndarray:
        if self.error is not None:
            raise self.error
        return np.full(len(points), self.value, dtype=np.float32)


def test_automatic_backend_uses_gpu_when_available() -> None:
    cpu = _StubAdapter("numpy-cpu", True, -1.0)
    gpu = _StubAdapter("cuda-gpu", True, 2.0)
    backend = AutomaticTPMSBackend(cpu_adapter=cpu, gpu_adapter=gpu)
    points = np.zeros((4, 3), dtype=np.float32)

    values = backend.evaluate_shell(
        points,
        TPMSFieldSpec("G", cell_size_mm=10.0, wall_thickness_mm=1.0),
    )

    np.testing.assert_array_equal(values, np.full(4, 2.0, dtype=np.float32))
    assert backend.status.active_backend == "cuda-gpu"
    assert backend.status.using_gpu is True


def test_automatic_backend_falls_back_after_gpu_failure() -> None:
    cpu = _StubAdapter("numpy-cpu", True, -3.0)
    gpu = _StubAdapter("cuda-gpu", True, 2.0, RuntimeError("CUDA out of memory"))
    backend = AutomaticTPMSBackend(cpu_adapter=cpu, gpu_adapter=gpu)

    values = backend.evaluate_shell(
        np.zeros((3, 3), dtype=np.float32),
        TPMSFieldSpec("D", cell_size_mm=8.0, wall_thickness_mm=0.8),
    )

    np.testing.assert_array_equal(values, np.full(3, -3.0, dtype=np.float32))
    assert backend.status.active_backend == "numpy-cpu"
    assert backend.status.using_gpu is False
    assert backend.status.fallback_reason == "RuntimeError: CUDA out of memory"


def test_automatic_backend_reports_startup_gpu_probe_failure() -> None:
    cpu = _StubAdapter("numpy-cpu", True, -1.0)
    gpu = _StubAdapter("cuda-gpu", False, 2.0)
    gpu.unavailable_reason = "OSError: CUDA context unavailable"

    backend = AutomaticTPMSBackend(cpu_adapter=cpu, gpu_adapter=gpu)

    assert backend.status.using_gpu is False
    assert backend.status.fallback_reason == "OSError: CUDA context unavailable"


@pytest.mark.parametrize("kind", ["G", "D"])
def test_cuda_shell_matches_numpy_reference(kind: str) -> None:
    cpu = NumpyTPMSAdapter()
    gpu = NumbaCudaTPMSAdapter()
    if not gpu.available:
        pytest.skip("CUDA backend is unavailable")
    points = np.random.default_rng(20260805).uniform(
        low=(-12.0, -8.0, -3.0),
        high=(15.0, 21.0, 5.0),
        size=(4_097, 3),
    ).astype(np.float32)
    spec = TPMSFieldSpec(
        kind,
        cell_size_mm=9.5 if kind == "G" else 11.0,
        wall_thickness_mm=0.85,
        level=0.13,
    )

    expected = cpu.evaluate_shell(points, spec, max_tile_points=1_000)
    actual = gpu.evaluate_shell(points, spec, max_tile_points=1_000)

    np.testing.assert_allclose(actual, expected, rtol=4e-5, atol=4e-5)


def test_cuda_transition_matches_numpy_reference() -> None:
    cpu = NumpyTPMSAdapter()
    gpu = NumbaCudaTPMSAdapter()
    if not gpu.available:
        pytest.skip("CUDA backend is unavailable")
    points = np.random.default_rng(42).normal(size=(5_003, 3)).astype(np.float32) * 8.0
    g_spec = TPMSFieldSpec("G", 9.0, 0.8, level=-0.08)
    d_spec = TPMSFieldSpec("D", 12.0, 1.1, level=0.12)
    transition = TransitionBlendSpec(
        transition_width_mm=7.5,
        center_offset_mm=0.4,
        sigmoid_sharpness=1.7,
        g_on_negative_side=False,
    )
    plane_point = np.array((1.0, -2.0, 0.5), dtype=np.float32)
    plane_normal = np.array((0.3, 0.9, -0.1), dtype=np.float32)
    plane_normal /= np.linalg.norm(plane_normal)

    expected = cpu.evaluate_transition(
        points,
        g_spec,
        d_spec,
        plane_point,
        plane_normal,
        transition,
        max_tile_points=1_024,
    )
    actual = gpu.evaluate_transition(
        points,
        g_spec,
        d_spec,
        plane_point,
        plane_normal,
        transition,
        max_tile_points=1_024,
    )

    np.testing.assert_allclose(actual, expected, rtol=6e-5, atol=6e-5)


def test_cuda_shell_matches_independent_axis_periods() -> None:
    cpu = NumpyTPMSAdapter()
    gpu = NumbaCudaTPMSAdapter()
    if not gpu.available:
        pytest.skip("CUDA backend is unavailable")
    points = np.random.default_rng(20260807).uniform(
        low=(-8.0, -9.0, -4.0),
        high=(11.0, 13.0, 6.0),
        size=(4_099, 3),
    ).astype(np.float32)
    spec = TPMSFieldSpec(
        "G",
        cell_size_mm=(8.0, 11.0, 5.5),
        wall_thickness_mm=0.75,
        level=0.08,
        origin_mm=(-3.2, 1.7, -0.4),
    )

    expected = cpu.evaluate_shell(points, spec, max_tile_points=1_024)
    actual = gpu.evaluate_shell(points, spec, max_tile_points=1_024)

    np.testing.assert_allclose(actual, expected, rtol=5e-5, atol=5e-5)


@pytest.mark.parametrize("mode", ["linear", "power", "sigmoid", "layered"])
@pytest.mark.parametrize("wall_method", ["gradient_normalized", "field_threshold"])
def test_cuda_gradient_shell_matches_numpy_reference(
    mode: str,
    wall_method: str,
) -> None:
    cpu = NumpyTPMSAdapter()
    gpu = NumbaCudaTPMSAdapter()
    if not gpu.available:
        pytest.skip("CUDA backend is unavailable")
    points = np.random.default_rng(20260915).uniform(
        low=(-7.0, -5.0, -3.0),
        high=(12.0, 9.0, 8.0),
        size=(8_193, 3),
    ).astype(np.float32)
    spec = TPMSFieldSpec(
        "G",
        cell_size_mm=(7.5, 9.0, 6.0),
        wall_thickness_mm=0.75,
        level=0.08,
        origin_mm=(-1.0, 2.0, 0.5),
        axes_world=((0.0, 1.0, 0.0), (-1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
    )
    gradient = TPMSGradientSpec(
        axis_index=2,
        coordinate_min_mm=-3.0,
        coordinate_max_mm=8.0,
        mode=mode,
        power=3.5,
        layers=6,
        sigmoid_sharpness=8.0,
        use_thickness_gradient=True,
        thickness_soft_mm=0.45,
        thickness_stiff_mm=1.05,
        use_offset_gradient=True,
        offset_soft=-0.15,
        offset_stiff=0.2,
        wall_thickness_method=wall_method,
    )

    expected = cpu.evaluate_gradient_shell(points, spec, gradient, 2_047)
    actual = gpu.evaluate_gradient_shell(points, spec, gradient, 2_047)

    np.testing.assert_allclose(actual, expected, rtol=2e-4, atol=8e-5)


def test_shell_field_period_is_relative_to_cell_map_origin() -> None:
    adapter = NumpyTPMSAdapter()
    spec = TPMSFieldSpec(
        "D",
        cell_size_mm=(4.0, 6.0, 8.0),
        wall_thickness_mm=0.6,
        origin_mm=(1.25, -2.5, 0.75),
    )
    first = np.array((1.7, -1.9, 1.2), dtype=np.float32)
    points = np.vstack((first, first + (4.0, 0.0, 0.0)))

    values = adapter.evaluate_shell(points, spec, max_tile_points=None)

    np.testing.assert_allclose(values[0], values[1], rtol=1e-5, atol=1e-5)


def test_shell_field_periods_follow_the_cell_map_frame_axes() -> None:
    adapter = NumpyTPMSAdapter()
    axes = (
        (0.0, 1.0, 0.0),
        (-1.0, 0.0, 0.0),
        (0.0, 0.0, 1.0),
    )
    spec = TPMSFieldSpec(
        "G",
        cell_size_mm=(4.0, 6.0, 8.0),
        wall_thickness_mm=0.6,
        origin_mm=(1.25, -2.5, 0.75),
        axes_world=axes,
    )
    first = np.array((1.7, -1.9, 1.2), dtype=np.float32)
    points = np.vstack(
        (
            first,
            first + np.asarray(axes[0]) * 4.0,
            first + np.asarray(axes[1]) * 6.0,
            first + np.asarray(axes[2]) * 8.0,
        )
    )

    values = adapter.evaluate_shell(points, spec, max_tile_points=None)

    np.testing.assert_allclose(values, np.repeat(values[0], 4), rtol=1e-5, atol=1e-5)
