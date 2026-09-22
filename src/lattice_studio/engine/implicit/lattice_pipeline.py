"""Device-resident evaluation of mesh-clipped TPMS fields.

The public contract accepts and returns host arrays.  The CUDA implementation
owns all device details and keeps each point tile, mesh SDF, TPMS field, and
intersection on one ordered CUDA stream until the final clipped field is ready.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from lattice_studio.engine.implicit.geometry_compute import (
    CpuGeometryAdapter,
    NumbaCudaGeometryAdapter,
    _cuda_intersection_kernel,
    _cuda_sdf_kernel,
    _numba_cuda,
    _validate_mesh,
    _validate_points,
)
from lattice_studio.engine.implicit.tpms_compute import (
    NumbaCudaTPMSAdapter,
    NumpyTPMSAdapter,
    TPMSFieldSpec,
    TPMSGradientSpec,
)


@dataclass(frozen=True)
class LatticePipelineStatus:
    active_backend: str
    using_gpu: bool
    device_name: str
    fallback_reason: str | None = None


class LatticeFieldPipeline(Protocol):
    name: str
    available: bool
    device_name: str

    def evaluate_mesh_clipped_tpms(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
        points: np.ndarray,
        spec: TPMSFieldSpec,
        gradient: TPMSGradientSpec | None,
        max_tile_points: int | None = 250_000,
    ) -> np.ndarray: ...


def _tile_ranges(size: int, max_tile_points: int | None):
    if max_tile_points is not None and max_tile_points < 1:
        raise ValueError("max_tile_points must be positive or None")
    step = size if max_tile_points is None else max_tile_points
    if size == 0:
        return []
    return [
        (start, min(start + step, size))
        for start in range(0, size, step)
    ]


def _validate_request(vertices, faces, points, spec, gradient):
    mesh = _validate_mesh(vertices, faces)
    point_array = _validate_points(points)
    spec.validate()
    if gradient is not None:
        gradient.validate()
    return mesh, point_array


class CpuLatticeFieldPipeline:
    """Reference implementation with the same host-facing contract."""

    name = "CPU lattice pipeline"
    available = True
    device_name = "CPU"

    def __init__(self) -> None:
        self._geometry = CpuGeometryAdapter()
        self._tpms = NumpyTPMSAdapter()

    def evaluate_mesh_clipped_tpms(
        self,
        vertices,
        faces,
        points,
        spec,
        gradient=None,
        max_tile_points=250_000,
    ) -> np.ndarray:
        (vertex_array, face_array), point_array = _validate_request(
            vertices, faces, points, spec, gradient
        )
        _tile_ranges(len(point_array), max_tile_points)
        domain = self._geometry.signed_distance(
            vertex_array, face_array, point_array, max_tile_points
        )
        if gradient is None:
            feature = self._tpms.evaluate_shell(point_array, spec, max_tile_points)
        else:
            feature = self._tpms.evaluate_gradient_shell(
                point_array, spec, gradient, max_tile_points
            )
        return np.maximum(domain, feature, dtype=np.float32)


class NumbaCudaLatticeFieldPipeline:
    """Fuse mesh SDF, TPMS, and clipping without intermediate host copies."""

    name = "Numba CUDA resident lattice pipeline"

    def __init__(self) -> None:
        self._geometry = NumbaCudaGeometryAdapter()
        self._tpms = NumbaCudaTPMSAdapter()
        self.available = bool(
            self._geometry.available
            and self._tpms.available
            and _numba_cuda is not None
            and _cuda_sdf_kernel is not None
            and _cuda_intersection_kernel is not None
        )
        if self.available:
            self.device_name = self._geometry.device_name
            self.unavailable_reason = None
        else:
            self.device_name = "CUDA unavailable"
            self.unavailable_reason = (
                getattr(self._geometry, "unavailable_reason", None)
                or getattr(self._tpms, "unavailable_reason", None)
                or "required CUDA kernels are unavailable"
            )

    def evaluate_mesh_clipped_tpms(
        self,
        vertices,
        faces,
        points,
        spec,
        gradient=None,
        max_tile_points=250_000,
    ) -> np.ndarray:
        if not self.available:
            raise RuntimeError("CUDA resident lattice pipeline is unavailable")
        (vertex_array, face_array), point_array = _validate_request(
            vertices, faces, points, spec, gradient
        )
        tiles = _tile_ranges(len(point_array), max_tile_points)
        # Keep the caller-owned array identity so display and extraction
        # micro-slices reuse one host/device BVH.
        _bvh, device_bvh = self._geometry._prepare_bvh(vertices, faces)
        result = np.empty(len(point_array), dtype=np.float32)
        threads = 256

        for start, end in tiles:
            host_points = np.ascontiguousarray(point_array[start:end], dtype=np.float32)
            device_points = _numba_cuda.to_device(host_points)
            device_domain = _numba_cuda.device_array(len(host_points), dtype=np.float32)
            device_result = _numba_cuda.device_array(len(host_points), dtype=np.float32)
            blocks = (len(host_points) + threads - 1) // threads

            _cuda_sdf_kernel[blocks, threads](
                device_points, device_domain, *device_bvh
            )
            if gradient is None:
                self._tpms._launch_shell_device(device_points, device_result, spec)
            else:
                self._tpms._launch_gradient_shell_device(
                    device_points, device_result, spec, gradient
                )
            _cuda_intersection_kernel[blocks, threads](
                device_domain, device_result, device_result
            )
            _numba_cuda.synchronize()
            device_result.copy_to_host(result[start:end])
            del device_points, device_domain, device_result
        return result


class AutomaticLatticeFieldPipeline:
    """Prefer the resident CUDA pipeline and preserve a reliable CPU fallback."""

    def __init__(self, cpu_pipeline=None, gpu_pipeline=None) -> None:
        self._cpu_pipeline = cpu_pipeline or CpuLatticeFieldPipeline()
        self._gpu_pipeline = gpu_pipeline or NumbaCudaLatticeFieldPipeline()
        self._active_pipeline = (
            self._gpu_pipeline
            if self._gpu_pipeline.available
            else self._cpu_pipeline
        )
        self._fallback_reason = (
            None
            if self._gpu_pipeline.available
            else getattr(self._gpu_pipeline, "unavailable_reason", None)
        )

    @property
    def status(self) -> LatticePipelineStatus:
        pipeline = self._active_pipeline
        return LatticePipelineStatus(
            pipeline.name,
            pipeline is self._gpu_pipeline,
            pipeline.device_name,
            self._fallback_reason,
        )

    def evaluate_mesh_clipped_tpms(self, *args, **kwargs) -> np.ndarray:
        try:
            return self._active_pipeline.evaluate_mesh_clipped_tpms(*args, **kwargs)
        except (TypeError, ValueError):
            raise
        except Exception as exc:
            if self._active_pipeline is not self._gpu_pipeline:
                raise
            self._fallback_reason = f"{type(exc).__name__}: {exc}"
            self._active_pipeline = self._cpu_pipeline
            return self._cpu_pipeline.evaluate_mesh_clipped_tpms(*args, **kwargs)


_DEFAULT_PIPELINE = None
_DEFAULT_PIPELINE_LOCK = threading.Lock()


def get_default_lattice_pipeline() -> AutomaticLatticeFieldPipeline:
    global _DEFAULT_PIPELINE
    if _DEFAULT_PIPELINE is None:
        with _DEFAULT_PIPELINE_LOCK:
            if _DEFAULT_PIPELINE is None:
                _DEFAULT_PIPELINE = AutomaticLatticeFieldPipeline()
    return _DEFAULT_PIPELINE


__all__ = [
    "AutomaticLatticeFieldPipeline",
    "CpuLatticeFieldPipeline",
    "LatticePipelineStatus",
    "NumbaCudaLatticeFieldPipeline",
    "get_default_lattice_pipeline",
]
