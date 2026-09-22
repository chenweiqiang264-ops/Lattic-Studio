"""Automatic GPU/CPU geometry operations for implicit generation.

The CUDA path keeps the mesh BVH resident on the device between point tiles.
Marching Cubes classifies cells and compacts active grid edges on the device,
so only shared vertices and final triangle indices are copied to the host.
"""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from lattice_studio.engine.implicit.cuda_probe import probe_numba_cuda


@dataclass(frozen=True)
class GeometryBackendStatus:
    active_backend: str
    using_gpu: bool
    device_name: str
    fallback_reason: str | None = None


class GeometryComputeAdapter(Protocol):
    name: str
    available: bool
    device_name: str

    def signed_distance(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
        points: np.ndarray,
        max_tile_points: int | None,
    ) -> np.ndarray: ...

    def intersect_fields(
        self,
        field_a: np.ndarray,
        field_b: np.ndarray,
        max_tile_points: int | None,
    ) -> np.ndarray: ...

    def marching_cubes(
        self,
        field: np.ndarray,
        spacing: tuple[float, float, float],
        origin: np.ndarray,
        level: float,
    ): ...

    def marching_cubes_chunk(
        self,
        field: np.ndarray,
        spacing: tuple[float, float, float],
        origin: np.ndarray,
        level: float,
        grid_offset: tuple[int, int, int],
        global_shape: tuple[int, int, int],
    ): ...


def _tile_ranges(size: int, max_tile_points: int | None):
    if size == 0:
        return []
    if max_tile_points is None:
        return [(0, size)]
    if max_tile_points < 1:
        raise ValueError("max_tile_points must be positive or None")
    return [
        (start, min(start + max_tile_points, size))
        for start in range(0, size, max_tile_points)
    ]


def _validate_mesh(
    vertices: np.ndarray,
    faces: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    vertex_array = np.ascontiguousarray(vertices, dtype=np.float32)
    face_array = np.ascontiguousarray(faces, dtype=np.int32)
    if vertex_array.ndim != 2 or vertex_array.shape[1] != 3:
        raise ValueError("vertices must have shape (N, 3)")
    if face_array.ndim != 2 or face_array.shape[1] != 3 or len(face_array) == 0:
        raise ValueError("faces must have shape (M, 3) and not be empty")
    if not np.isfinite(vertex_array).all():
        raise ValueError("vertices must contain only finite values")
    if face_array.min() < 0 or face_array.max() >= len(vertex_array):
        raise ValueError("faces contain an out-of-range vertex index")
    return vertex_array, face_array


def _validate_points(points: np.ndarray) -> np.ndarray:
    point_array = np.ascontiguousarray(points, dtype=np.float32)
    if point_array.ndim != 2 or point_array.shape[1] != 3:
        raise ValueError("points must have shape (N, 3)")
    if not np.isfinite(point_array).all():
        raise ValueError("points must contain only finite values")
    return point_array


def _stabilize_isosurface_samples(
    field: np.ndarray,
    level: float,
) -> np.ndarray:
    """Move near-level samples outside the float32 ambiguity band.

    Samples numerically indistinguishable from the selected level can make
    several Marching Cubes vertices collapse onto one grid point. Removing the
    resulting degenerate triangles may then open an otherwise closed surface.
    The deterministic nudge is far below the requested geometric tolerance and
    is applied identically by CPU, GPU, global, and chunked extraction paths.
    """

    scalar = np.ascontiguousarray(field, dtype=np.float32)
    if scalar.ndim != 3 or any(size < 2 for size in scalar.shape):
        raise ValueError("field must be a 3-D grid with at least two samples")
    if not np.isfinite(scalar).all() or not np.isfinite(level):
        raise ValueError("field and level must contain only finite values")
    scale = max(float(np.max(np.abs(scalar), initial=0.0)), abs(float(level)), 1.0)
    epsilon = np.finfo(np.float32).eps * scale * 8.0
    distance = scalar - np.float32(level)
    ambiguous = np.abs(distance) < epsilon
    if not np.any(ambiguous):
        return scalar
    stabilized = scalar.copy()
    stabilized[ambiguous] = np.float32(level) + np.where(
        distance[ambiguous] < 0.0,
        -epsilon,
        epsilon,
    )
    return stabilized


def _local_edge_ids_to_global(
    local_edge_ids: np.ndarray,
    local_shape: tuple[int, int, int],
    grid_offset: tuple[int, int, int],
    global_shape: tuple[int, int, int],
) -> np.ndarray:
    """Map compact chunk-grid edge identifiers to whole-grid identifiers."""

    ids = np.asarray(local_edge_ids, dtype=np.int64).reshape(-1)
    nx, ny, nz = (int(value) for value in local_shape)
    global_nx, global_ny, global_nz = (int(value) for value in global_shape)
    offset_i, offset_j, offset_k = (int(value) for value in grid_offset)
    if any(value < 2 for value in (nx, ny, nz, global_nx, global_ny, global_nz)):
        raise ValueError("local_shape and global_shape must be at least two per axis")
    local_x_count = (nx - 1) * ny * nz
    local_y_count = nx * (ny - 1) * nz
    local_edge_count = local_x_count + local_y_count + nx * ny * (nz - 1)
    if ids.size and (ids.min() < 0 or ids.max() >= local_edge_count):
        raise ValueError("local_edge_ids contain an out-of-range edge")

    result = np.empty(ids.shape, dtype=np.int64)
    global_x_count = (global_nx - 1) * global_ny * global_nz
    global_y_count = global_nx * (global_ny - 1) * global_nz

    x_mask = ids < local_x_count
    if np.any(x_mask):
        selected = ids[x_mask]
        i = selected // (ny * nz)
        remainder = selected % (ny * nz)
        j = remainder // nz
        k = remainder % nz
        result[x_mask] = (
            ((offset_i + i) * global_ny + offset_j + j) * global_nz
            + offset_k
            + k
        )

    y_mask = (ids >= local_x_count) & (ids < local_x_count + local_y_count)
    if np.any(y_mask):
        selected = ids[y_mask] - local_x_count
        i = selected // ((ny - 1) * nz)
        remainder = selected % ((ny - 1) * nz)
        j = remainder // nz
        k = remainder % nz
        result[y_mask] = (
            global_x_count
            + ((offset_i + i) * (global_ny - 1) + offset_j + j) * global_nz
            + offset_k
            + k
        )

    z_mask = ids >= local_x_count + local_y_count
    if np.any(z_mask):
        selected = ids[z_mask] - local_x_count - local_y_count
        i = selected // (ny * (nz - 1))
        remainder = selected % (ny * (nz - 1))
        j = remainder // (nz - 1)
        k = remainder % (nz - 1)
        result[z_mask] = (
            global_x_count
            + global_y_count
            + ((offset_i + i) * global_ny + offset_j + j) * (global_nz - 1)
            + offset_k
            + k
        )
    return result


def _vertices_to_local_edge_ids(
    vertices: np.ndarray,
    spacing: tuple[float, float, float],
    local_shape: tuple[int, int, int],
) -> np.ndarray:
    """Recover canonical grid-edge IDs from a CPU Marching Cubes result.

    ``skimage.measure.marching_cubes`` does not expose the grid edge that
    generated each returned vertex.  Chunk stitching nevertheless needs that
    identity: welding by rounded world coordinates is not robust enough when
    two independently extracted triangles meet on a brick boundary.  Every
    ordinary Marching Cubes vertex lies on exactly one grid edge, so the
    fractional index coordinate identifies its orientation and its integer
    coordinates identify the edge origin.

    The isosurface sampler is stabilized before extraction, which prevents a
    vertex from landing exactly on a grid point.  A defensive fallback still
    raises a clear error if a backend ever violates that invariant instead of
    silently producing incorrect topology.
    """

    points = np.asarray(vertices, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("vertices must have shape (N, 3)")
    dimensions = tuple(int(value) for value in local_shape)
    if len(dimensions) != 3 or any(value < 2 for value in dimensions):
        raise ValueError("local_shape must contain three dimensions of at least two")
    scale = np.asarray(spacing, dtype=np.float64)
    if scale.shape != (3,) or not np.isfinite(scale).all() or np.any(scale <= 0.0):
        raise ValueError("spacing must contain three finite positive values")
    if not np.isfinite(points).all():
        raise ValueError("vertices must contain only finite values")

    index_points = points / scale
    rounded = np.rint(index_points)
    fractional = np.abs(index_points - rounded)
    orientation = np.argmax(fractional, axis=1)
    # A stabilized field should keep this comfortably above machine noise.
    # The tolerance only absorbs skimage's floating-point round-off.
    ambiguous = np.max(fractional, axis=1) <= 1.0e-7
    if np.any(ambiguous):
        raise ValueError(
            "Marching Cubes returned a vertex on a grid point; "
            "canonical chunk-edge identity is undefined"
        )

    edge_ids = np.empty(len(points), dtype=np.int64)
    nx, ny, nz = dimensions
    x_count = (nx - 1) * ny * nz
    y_count = nx * (ny - 1) * nz
    for axis in range(3):
        selected = np.flatnonzero(orientation == axis)
        if not len(selected):
            continue
        integer = np.rint(index_points[selected]).astype(np.int64)
        # The coordinate along the oriented edge is interpolated and must be
        # anchored at the lower endpoint.  ``rint`` is correct for the two
        # fixed coordinates, but can move an interpolated coordinate at the
        # upper grid boundary from ``nx - 1 - epsilon`` to ``nx - 1`` and
        # incorrectly classify a valid edge as out of range.
        integer[:, axis] = np.floor(index_points[selected, axis]).astype(np.int64)
        if axis == 0:
            valid = (
                (integer[:, 0] >= 0)
                & (integer[:, 0] < nx - 1)
                & (integer[:, 1] >= 0)
                & (integer[:, 1] < ny)
                & (integer[:, 2] >= 0)
                & (integer[:, 2] < nz)
            )
            if not np.all(valid):
                raise ValueError("x-oriented Marching Cubes edge is out of range")
            edge_ids[selected] = (
                (integer[:, 0] * ny + integer[:, 1]) * nz + integer[:, 2]
            )
        elif axis == 1:
            valid = (
                (integer[:, 0] >= 0)
                & (integer[:, 0] < nx)
                & (integer[:, 1] >= 0)
                & (integer[:, 1] < ny - 1)
                & (integer[:, 2] >= 0)
                & (integer[:, 2] < nz)
            )
            if not np.all(valid):
                raise ValueError("y-oriented Marching Cubes edge is out of range")
            edge_ids[selected] = x_count + (
                (integer[:, 0] * (ny - 1) + integer[:, 1]) * nz + integer[:, 2]
            )
        else:
            valid = (
                (integer[:, 0] >= 0)
                & (integer[:, 0] < nx)
                & (integer[:, 1] >= 0)
                & (integer[:, 1] < ny)
                & (integer[:, 2] >= 0)
                & (integer[:, 2] < nz - 1)
            )
            if not np.all(valid):
                raise ValueError("z-oriented Marching Cubes edge is out of range")
            edge_ids[selected] = x_count + y_count + (
                (integer[:, 0] * ny + integer[:, 1]) * (nz - 1)
                + integer[:, 2]
            )
    return edge_ids


class CpuGeometryAdapter:
    """Reference geometry implementation using existing CPU libraries."""

    name = "CPU geometry"
    available = True
    device_name = "CPU"

    def signed_distance(self, vertices, faces, points, max_tile_points):
        vertex_array, face_array = _validate_mesh(vertices, faces)
        point_array = _validate_points(points)
        try:
            from lattice_studio.infrastructure.native import CPP_MESH_SDF_AVAILABLE, cpp_mesh_sdf

            if CPP_MESH_SDF_AVAILABLE and cpp_mesh_sdf is not None:
                return np.asarray(
                    cpp_mesh_sdf.compute_signed_distance(
                        np.asarray(vertex_array, dtype=np.float64),
                        face_array,
                        np.asarray(point_array, dtype=np.float64),
                    ),
                    dtype=np.float32,
                )
        except (ImportError, AttributeError, RuntimeError, ValueError):
            pass

        import pyvista as pv

        mesh = pv.PolyData(vertex_array, np.column_stack((
            np.full(len(face_array), 3, dtype=np.int32),
            face_array,
        )))
        values = np.empty(len(point_array), dtype=np.float32)
        for start, end in _tile_ranges(len(point_array), max_tile_points):
            cloud = pv.PolyData(point_array[start:end])
            values[start:end] = np.asarray(
                cloud.compute_implicit_distance(mesh, inplace=False)[
                    "implicit_distance"
                ],
                dtype=np.float32,
            )
        return values

    def intersect_fields(self, field_a, field_b, max_tile_points):
        a = np.asarray(field_a, dtype=np.float32)
        b = np.asarray(field_b, dtype=np.float32)
        if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
            raise ValueError("intersection fields must have matching finite values")
        return np.maximum(a, b)

    def marching_cubes(self, field, spacing, origin, level):
        scalar_field = _stabilize_isosurface_samples(field, level)
        from skimage import measure

        import trimesh

        try:
            vertices, faces, _, _ = measure.marching_cubes(
                scalar_field,
                level=float(level),
                spacing=tuple(float(value) for value in spacing),
                method="lorensen",
            )
        except ValueError:
            return trimesh.Trimesh(process=False)
        result = trimesh.Trimesh(
            vertices=np.asarray(vertices, dtype=np.float64)
            + np.asarray(origin, dtype=np.float64),
            faces=np.asarray(faces, dtype=np.int64),
            process=False,
        )
        result.fix_normals()
        return result

    def marching_cubes_chunk(
        self,
        field,
        spacing,
        origin,
        level,
        grid_offset,
        global_shape,
    ):
        mesh = self.marching_cubes(field, spacing, origin, level)
        local_edge_ids = _vertices_to_local_edge_ids(
            mesh.vertices,
            spacing,
            tuple(np.asarray(field).shape),
        )
        global_edge_ids = _local_edge_ids_to_global(
            local_edge_ids,
            tuple(np.asarray(field).shape),
            grid_offset,
            global_shape,
        )
        return mesh, global_edge_ids


@dataclass
class _FlatBvh:
    triangle_vertices: np.ndarray
    node_lo: np.ndarray
    node_hi: np.ndarray
    left: np.ndarray
    right: np.ndarray
    begin: np.ndarray
    end: np.ndarray
    order: np.ndarray


def _build_flat_bvh(vertices: np.ndarray, faces: np.ndarray) -> _FlatBvh:
    triangles = np.asarray(vertices[faces], dtype=np.float32)
    triangle_lo = triangles.min(axis=1)
    triangle_hi = triangles.max(axis=1)
    centroids = triangles.mean(axis=1)
    order = np.arange(len(triangles), dtype=np.int32)
    node_lo: list[np.ndarray] = []
    node_hi: list[np.ndarray] = []
    left: list[int] = []
    right: list[int] = []
    begin: list[int] = []
    end: list[int] = []

    def build(start: int, stop: int) -> int:
        indices = order[start:stop]
        node_index = len(node_lo)
        node_lo.append(triangle_lo[indices].min(axis=0))
        node_hi.append(triangle_hi[indices].max(axis=0))
        left.append(-1)
        right.append(-1)
        begin.append(start)
        end.append(stop)
        if stop - start > 8:
            span = np.ptp(centroids[indices], axis=0)
            axis = int(np.argmax(span))
            sorted_indices = indices[
                np.argsort(centroids[indices, axis], kind="mergesort")
            ]
            order[start:stop] = sorted_indices
            middle = start + (stop - start) // 2
            left[node_index] = build(start, middle)
            right[node_index] = build(middle, stop)
        return node_index

    build(0, len(order))
    return _FlatBvh(
        triangle_vertices=np.ascontiguousarray(triangles),
        node_lo=np.ascontiguousarray(np.asarray(node_lo, dtype=np.float32)),
        node_hi=np.ascontiguousarray(np.asarray(node_hi, dtype=np.float32)),
        left=np.asarray(left, dtype=np.int32),
        right=np.asarray(right, dtype=np.int32),
        begin=np.asarray(begin, dtype=np.int32),
        end=np.asarray(end, dtype=np.int32),
        order=np.asarray(order, dtype=np.int32),
    )


try:
    from numba import cuda as _numba_cuda

    @_numba_cuda.jit(device=True)
    def _box_distance_sq(px, py, pz, lo, hi):
        dx = lo[0] - px
        if dx < 0.0:
            dx = 0.0
        if px > hi[0]:
            dx = px - hi[0]
        dy = lo[1] - py
        if dy < 0.0:
            dy = 0.0
        if py > hi[1]:
            dy = py - hi[1]
        dz = lo[2] - pz
        if dz < 0.0:
            dz = 0.0
        if pz > hi[2]:
            dz = pz - hi[2]
        return dx * dx + dy * dy + dz * dz

    @_numba_cuda.jit(device=True)
    def _triangle_distance_sq(px, py, pz, triangle):
        ax, ay, az = triangle[0, 0], triangle[0, 1], triangle[0, 2]
        bx, by, bz = triangle[1, 0], triangle[1, 1], triangle[1, 2]
        cx, cy, cz = triangle[2, 0], triangle[2, 1], triangle[2, 2]
        abx, aby, abz = bx - ax, by - ay, bz - az
        acx, acy, acz = cx - ax, cy - ay, cz - az
        apx, apy, apz = px - ax, py - ay, pz - az
        d1 = abx * apx + aby * apy + abz * apz
        d2 = acx * apx + acy * apy + acz * apz
        if d1 <= 0.0 and d2 <= 0.0:
            return apx * apx + apy * apy + apz * apz

        bpx, bpy, bpz = px - bx, py - by, pz - bz
        d3 = abx * bpx + aby * bpy + abz * bpz
        d4 = acx * bpx + acy * bpy + acz * bpz
        if d3 >= 0.0 and d4 <= d3:
            return bpx * bpx + bpy * bpy + bpz * bpz

        vc = d1 * d4 - d3 * d2
        if vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:
            t = d1 / (d1 - d3)
            qx, qy, qz = ax + t * abx, ay + t * aby, az + t * abz
            dx, dy, dz = px - qx, py - qy, pz - qz
            return dx * dx + dy * dy + dz * dz

        cpx, cpy, cpz = px - cx, py - cy, pz - cz
        d5 = abx * cpx + aby * cpy + abz * cpz
        d6 = acx * cpx + acy * cpy + acz * cpz
        if d6 >= 0.0 and d5 <= d6:
            return cpx * cpx + cpy * cpy + cpz * cpz

        vb = d5 * d2 - d1 * d6
        if vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0:
            t = d2 / (d2 - d6)
            qx, qy, qz = ax + t * acx, ay + t * acy, az + t * acz
            dx, dy, dz = px - qx, py - qy, pz - qz
            return dx * dx + dy * dy + dz * dz

        va = d3 * d6 - d5 * d4
        if va <= 0.0 and (d4 - d3) >= 0.0 and (d5 - d6) >= 0.0:
            t = (d4 - d3) / ((d4 - d3) + (d5 - d6))
            bcx, bcy, bcz = cx - bx, cy - by, cz - bz
            qx, qy, qz = bx + t * bcx, by + t * bcy, bz + t * bcz
            dx, dy, dz = px - qx, py - qy, pz - qz
            return dx * dx + dy * dy + dz * dz

        nx = aby * acz - abz * acy
        ny = abz * acx - abx * acz
        nz = abx * acy - aby * acx
        n2 = nx * nx + ny * ny + nz * nz
        if n2 <= 1e-12:
            first = apx * apx + apy * apy + apz * apz
            second = bpx * bpx + bpy * bpy + bpz * bpz
            third = cpx * cpx + cpy * cpy + cpz * cpz
            if second < first:
                first = second
            if third < first:
                first = third
            return first
        distance = (apx * nx + apy * ny + apz * nz) / math.sqrt(n2)
        return distance * distance

    @_numba_cuda.jit(device=True)
    def _ray_intersects_box(px, py, pz, dx, dy, dz, lo, hi, ray_length):
        t_min = 0.0
        t_max = ray_length

        if abs(dx) < 1e-12:
            if px < lo[0] or px > hi[0]:
                return False
        else:
            first = (lo[0] - px) / dx
            second = (hi[0] - px) / dx
            if first > second:
                first, second = second, first
            if first > t_min:
                t_min = first
            if second < t_max:
                t_max = second
            if t_max < t_min:
                return False

        if abs(dy) < 1e-12:
            if py < lo[1] or py > hi[1]:
                return False
        else:
            first = (lo[1] - py) / dy
            second = (hi[1] - py) / dy
            if first > second:
                first, second = second, first
            if first > t_min:
                t_min = first
            if second < t_max:
                t_max = second
            if t_max < t_min:
                return False

        if abs(dz) < 1e-12:
            if pz < lo[2] or pz > hi[2]:
                return False
        else:
            first = (lo[2] - pz) / dz
            second = (hi[2] - pz) / dz
            if first > second:
                first, second = second, first
            if first > t_min:
                t_min = first
            if second < t_max:
                t_max = second
            if t_max < t_min:
                return False

        return t_max > 1e-7

    @_numba_cuda.jit(device=True)
    def _ray_hits_triangle(
        px,
        py,
        pz,
        dx,
        dy,
        dz,
        triangle,
        ray_length,
    ):
        e1x = triangle[1, 0] - triangle[0, 0]
        e1y = triangle[1, 1] - triangle[0, 1]
        e1z = triangle[1, 2] - triangle[0, 2]
        e2x = triangle[2, 0] - triangle[0, 0]
        e2y = triangle[2, 1] - triangle[0, 1]
        e2z = triangle[2, 2] - triangle[0, 2]
        hx = dy * e2z - dz * e2y
        hy = dz * e2x - dx * e2z
        hz = dx * e2y - dy * e2x
        determinant = e1x * hx + e1y * hy + e1z * hz
        if abs(determinant) < 1e-7:
            return 0
        inverse = 1.0 / determinant
        sx = px - triangle[0, 0]
        sy = py - triangle[0, 1]
        sz = pz - triangle[0, 2]
        u = inverse * (sx * hx + sy * hy + sz * hz)
        if u < -1e-6 or u > 1.0 + 1e-6:
            return 0
        qx = sy * e1z - sz * e1y
        qy = sz * e1x - sx * e1z
        qz = sx * e1y - sy * e1x
        v = inverse * (dx * qx + dy * qy + dz * qz)
        distance = inverse * (e2x * qx + e2y * qy + e2z * qz)
        if (
            v < -1e-6
            or u + v > 1.0 + 1e-6
            or distance <= 1e-7
            or distance >= ray_length
        ):
            return 0
        w = 1.0 - u - v
        if abs(u) <= 2e-6 or abs(v) <= 2e-6 or abs(w) <= 2e-6:
            return 2
        return 1

    @_numba_cuda.jit(device=True)
    def _ray_parity(
        px,
        py,
        pz,
        dx,
        dy,
        dz,
        triangle_vertices,
        node_lo,
        node_hi,
        left,
        right,
        begin,
        end,
        order,
    ):
        stack = _numba_cuda.local.array(64, dtype=np.int32)
        stack[0] = 0
        top = 1
        hits = 0
        ambiguous = False
        ray_length = 1e12
        while top > 0:
            top -= 1
            node = stack[top]
            if not _ray_intersects_box(
                px,
                py,
                pz,
                dx,
                dy,
                dz,
                node_lo[node],
                node_hi[node],
                ray_length,
            ):
                continue
            if left[node] < 0:
                for cursor in range(begin[node], end[node]):
                    code = _ray_hits_triangle(
                        px,
                        py,
                        pz,
                        dx,
                        dy,
                        dz,
                        triangle_vertices[order[cursor]],
                        ray_length,
                    )
                    if code:
                        hits += 1
                        if code == 2:
                            ambiguous = True
            elif top < 62:
                stack[top] = left[node]
                top += 1
                stack[top] = right[node]
                top += 1
        return hits & 1, ambiguous

    @_numba_cuda.jit
    def _cuda_sdf_kernel(
        points, result, triangle_vertices, node_lo, node_hi, left, right, begin, end, order
    ):
        index = _numba_cuda.grid(1)
        if index >= points.shape[0]:
            return
        px, py, pz = points[index, 0], points[index, 1], points[index, 2]
        stack = _numba_cuda.local.array(64, dtype=np.int32)
        stack[0] = 0
        top = 1
        best = 1e30
        hits = 0
        # Keep the sign ray anchored at the query point. Transverse origin
        # offsets can cross a nearby surface and invert isolated samples.
        ray_dx = 1.0
        ray_dy = 0.0
        ray_dz = 0.0
        ray_length = 1e12
        ray_ambiguous = False
        while top > 0:
            top -= 1
            node = stack[top]
            distance_candidate = (
                _box_distance_sq(px, py, pz, node_lo[node], node_hi[node]) < best
            )
            ray_candidate = (
                py >= node_lo[node, 1]
                and py <= node_hi[node, 1]
                and pz >= node_lo[node, 2]
                and pz <= node_hi[node, 2]
                and node_hi[node, 0] > px + 1e-7
            )
            if not distance_candidate and not ray_candidate:
                continue
            if left[node] < 0:
                for cursor in range(begin[node], end[node]):
                    triangle = triangle_vertices[order[cursor]]
                    if distance_candidate:
                        distance_sq = _triangle_distance_sq(px, py, pz, triangle)
                        if distance_sq < best:
                            best = distance_sq
                    if ray_candidate:
                        code = _ray_hits_triangle(
                            px,
                            py,
                            pz,
                            ray_dx,
                            ray_dy,
                            ray_dz,
                            triangle,
                            ray_length,
                        )
                        if code:
                            hits += 1
                            if code == 2:
                                ray_ambiguous = True
            else:
                if top < 62:
                    stack[top] = left[node]
                    top += 1
                    stack[top] = right[node]
                    top += 1
        first_inside = hits & 1
        second_inside, second_ambiguous = _ray_parity(
            px,
            py,
            pz,
            0.3713906763541037,
            1.0,
            0.6947465906068658,
            triangle_vertices,
            node_lo,
            node_hi,
            left,
            right,
            begin,
            end,
            order,
        )
        inside = first_inside
        if (
            ray_ambiguous
            or second_ambiguous
            or first_inside != second_inside
        ):
            third_inside, _third_ambiguous = _ray_parity(
                px,
                py,
                pz,
                0.6947465906068658,
                0.3713906763541037,
                1.0,
                triangle_vertices,
                node_lo,
                node_hi,
                left,
                right,
                begin,
                end,
                order,
            )
            inside = (first_inside + second_inside + third_inside) >= 2
        distance = math.sqrt(best)
        result[index] = -distance if inside else distance

    @_numba_cuda.jit
    def _cuda_intersection_kernel(field_a, field_b, result):
        index = _numba_cuda.grid(1)
        if index < result.size:
            first = field_a[index]
            second = field_b[index]
            result[index] = first if first >= second else second

    @_numba_cuda.jit
    def _cuda_mc_count_kernel(field, counts, active_edges, table, nx, ny, nz, level):
        cell = _numba_cuda.grid(1)
        total = (nx - 1) * (ny - 1) * (nz - 1)
        if cell >= total:
            return
        cells_y, cells_z = ny - 1, nz - 1
        i = cell // (cells_y * cells_z)
        rem = cell - i * cells_y * cells_z
        j, k = rem // cells_z, rem - (rem // cells_z) * cells_z
        values = _numba_cuda.local.array(8, dtype=np.float32)
        offsets = (
            (0, 0, 0),
            (0, 0, 1),
            (0, 1, 1),
            (0, 1, 0),
            (1, 0, 0),
            (1, 0, 1),
            (1, 1, 1),
            (1, 1, 0),
        )
        for corner in range(8):
            oi, oj, ok = offsets[corner]
            values[corner] = field[(i + oi) * ny * nz + (j + oj) * nz + k + ok]
        cube = 0
        for corner in range(8):
            if values[corner] > level:
                cube |= 1 << corner
        triangles = 0
        while triangles < 5 and table[cube, triangles * 3] >= 0:
            edge_axis = (2, 1, 2, 1, 2, 1, 2, 1, 0, 0, 0, 0)
            edge_i = (0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0)
            edge_j = (0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1)
            edge_k = (0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0)
            x_edge_count = (nx - 1) * ny * nz
            y_edge_count = nx * (ny - 1) * nz
            for local in range(3):
                edge = table[cube, triangles * 3 + local]
                edge_grid_i = i + edge_i[edge]
                edge_grid_j = j + edge_j[edge]
                edge_grid_k = k + edge_k[edge]
                if edge_axis[edge] == 0:
                    edge_id = (edge_grid_i * ny + edge_grid_j) * nz + edge_grid_k
                elif edge_axis[edge] == 1:
                    edge_id = (
                        x_edge_count
                        + (edge_grid_i * (ny - 1) + edge_grid_j) * nz
                        + edge_grid_k
                    )
                else:
                    edge_id = (
                        x_edge_count
                        + y_edge_count
                        + (edge_grid_i * ny + edge_grid_j) * (nz - 1)
                        + edge_grid_k
                    )
                active_edges[edge_id] = 1
            triangles += 1
        counts[cell] = triangles

    @_numba_cuda.jit
    def _cuda_mc_emit_kernel(
        field,
        offsets,
        vertices,
        edge_to_vertex,
        faces,
        table,
        nx,
        ny,
        nz,
        spacing,
        origin,
        level,
    ):
        cell = _numba_cuda.grid(1)
        total = (nx - 1) * (ny - 1) * (nz - 1)
        if cell >= total:
            return
        cells_y, cells_z = ny - 1, nz - 1
        i = cell // (cells_y * cells_z)
        rem = cell - i * cells_y * cells_z
        j, k = rem // cells_z, rem - (rem // cells_z) * cells_z
        values = _numba_cuda.local.array(8, dtype=np.float32)
        coords = _numba_cuda.local.array((8, 3), dtype=np.float32)
        corner_offsets = (
            (0, 0, 0),
            (0, 0, 1),
            (0, 1, 1),
            (0, 1, 0),
            (1, 0, 0),
            (1, 0, 1),
            (1, 1, 1),
            (1, 1, 0),
        )
        for corner in range(8):
            oi, oj, ok = corner_offsets[corner]
            grid_index = (i + oi) * ny * nz + (j + oj) * nz + k + ok
            values[corner] = field[grid_index]
            coords[corner, 0] = origin[0] + (i + oi) * spacing[0]
            coords[corner, 1] = origin[1] + (j + oj) * spacing[1]
            coords[corner, 2] = origin[2] + (k + ok) * spacing[2]
        cube = 0
        for corner in range(8):
            if values[corner] > level:
                cube |= 1 << corner
        edge_a = (0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3)
        edge_b = (1, 2, 3, 0, 5, 6, 7, 4, 4, 5, 6, 7)
        edge_axis = (2, 1, 2, 1, 2, 1, 2, 1, 0, 0, 0, 0)
        edge_i = (0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0)
        edge_j = (0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1)
        edge_k = (0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0)
        x_edge_count = (nx - 1) * ny * nz
        y_edge_count = nx * (ny - 1) * nz
        triangle_index = 0
        while triangle_index < 5 and table[cube, triangle_index * 3] >= 0:
            for local in range(3):
                edge = table[cube, triangle_index * 3 + local]
                a, b = edge_a[edge], edge_b[edge]
                denominator = values[b] - values[a]
                fraction = 0.5 if abs(denominator) < 1e-7 else (level - values[a]) / denominator
                if fraction < 0.0:
                    fraction = 0.0
                elif fraction > 1.0:
                    fraction = 1.0
                grid_i = i + edge_i[edge]
                grid_j = j + edge_j[edge]
                grid_k = k + edge_k[edge]
                if edge_axis[edge] == 0:
                    edge_id = (grid_i * ny + grid_j) * nz + grid_k
                elif edge_axis[edge] == 1:
                    edge_id = (
                        x_edge_count
                        + (grid_i * (ny - 1) + grid_j) * nz
                        + grid_k
                    )
                else:
                    edge_id = (
                        x_edge_count
                        + y_edge_count
                        + (grid_i * ny + grid_j) * (nz - 1)
                        + grid_k
                    )
                vertex_index = edge_to_vertex[edge_id]
                for axis in range(3):
                    vertices[vertex_index, axis] = coords[a, axis] + fraction * (
                        coords[b, axis] - coords[a, axis]
                    )
                faces[offsets[cell] + triangle_index, local] = vertex_index
            triangle_index += 1

except ImportError:
    _numba_cuda = None
    _cuda_sdf_kernel = None
    _cuda_intersection_kernel = None
    _cuda_mc_count_kernel = None
    _cuda_mc_emit_kernel = None


class NumbaCudaGeometryAdapter:
    """CUDA geometry adapter with cached mesh BVH and bounded field tiles."""

    name = "Numba CUDA geometry"

    def __init__(self):
        self.available = False
        self.device_name = "CUDA unavailable"
        self.unavailable_reason = "Numba CUDA is not installed"
        self._bvh_key = None
        self._bvh = None
        self._device_bvh = None
        self._device_mc_table = None
        if _numba_cuda is None:
            return
        probe = probe_numba_cuda()
        if not probe.available:
            self.unavailable_reason = probe.reason
            return
        try:
            self.available = bool(_numba_cuda.is_available())
            if not self.available:
                self.unavailable_reason = "CUDA runtime is unavailable"
            if self.available:
                self.device_name = probe.device_name
                # ``is_available`` can be true while the WDDM context cannot
                # allocate or copy device memory.  Probe the actual operation
                # used by every CUDA path before advertising GPU availability.
                probe = _numba_cuda.to_device(np.zeros(1, dtype=np.float32))
                _numba_cuda.synchronize()
                del probe
                self.unavailable_reason = None
        except Exception as exc:
            self.available = False
            self.device_name = "CUDA unavailable"
            self.unavailable_reason = f"{type(exc).__name__}: {exc}"

    def _prepare_bvh(self, vertices, faces):
        vertex_array, face_array = _validate_mesh(vertices, faces)
        key = (id(vertices), id(faces), vertex_array.shape, face_array.shape)
        if self._bvh_key == key and self._bvh is not None:
            return self._bvh, self._device_bvh
        bvh = _build_flat_bvh(vertex_array, face_array)
        self._bvh_key = key
        self._bvh = bvh
        def upload_cached(array):
            device = _numba_cuda.device_array(array.shape, dtype=array.dtype)
            device.copy_to_device(array)
            return device

        self._device_bvh = tuple(
            upload_cached(array)
            for array in (
                bvh.triangle_vertices,
                bvh.node_lo,
                bvh.node_hi,
                bvh.left,
                bvh.right,
                bvh.begin,
                bvh.end,
                bvh.order,
            )
        )
        return bvh, self._device_bvh

    def signed_distance(self, vertices, faces, points, max_tile_points=250_000):
        if not self.available or _cuda_sdf_kernel is None:
            raise RuntimeError("CUDA geometry backend is unavailable")
        point_array = _validate_points(points)
        _bvh, device_bvh = self._prepare_bvh(vertices, faces)
        result = np.empty(len(point_array), dtype=np.float32)
        threads = 256
        for start, end in _tile_ranges(len(point_array), max_tile_points):
            host_points = np.ascontiguousarray(point_array[start:end])
            device_points = _numba_cuda.to_device(host_points)
            device_result = _numba_cuda.device_array(len(host_points), dtype=np.float32)
            blocks = (len(host_points) + threads - 1) // threads
            _cuda_sdf_kernel[blocks, threads](device_points, device_result, *device_bvh)
            _numba_cuda.synchronize()
            device_result.copy_to_host(result[start:end])
        return result

    def intersect_fields(self, field_a, field_b, max_tile_points=250_000):
        if not self.available or _cuda_intersection_kernel is None:
            raise RuntimeError("CUDA geometry backend is unavailable")
        a = np.asarray(field_a, dtype=np.float32)
        b = np.asarray(field_b, dtype=np.float32)
        if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
            raise ValueError("intersection fields must have matching finite values")
        flat_a = a.ravel()
        flat_b = b.ravel()
        result = np.empty(a.size, dtype=np.float32)
        threads = 256
        for start, end in _tile_ranges(a.size, max_tile_points):
            da = _numba_cuda.to_device(np.ascontiguousarray(flat_a[start:end]))
            db = _numba_cuda.to_device(np.ascontiguousarray(flat_b[start:end]))
            dr = _numba_cuda.device_array(end - start, dtype=np.float32)
            blocks = (end - start + threads - 1) // threads
            _cuda_intersection_kernel[blocks, threads](da, db, dr)
            _numba_cuda.synchronize()
            dr.copy_to_host(result[start:end])
        return result.reshape(a.shape)

    def marching_cubes(self, field, spacing, origin, level=0.0):
        result, _edge_ids = self._marching_cubes(
            field,
            spacing,
            origin,
            level,
            grid_offset=(0, 0, 0),
            global_shape=tuple(np.asarray(field).shape),
        )
        return result

    def marching_cubes_chunk(
        self,
        field,
        spacing,
        origin,
        level,
        grid_offset,
        global_shape,
    ):
        return self._marching_cubes(
            field,
            spacing,
            origin,
            level,
            grid_offset=grid_offset,
            global_shape=global_shape,
        )

    def _marching_cubes(
        self,
        field,
        spacing,
        origin,
        level,
        grid_offset,
        global_shape,
    ):
        if not self.available or _cuda_mc_count_kernel is None:
            raise RuntimeError("CUDA geometry backend is unavailable")
        import trimesh
        from skimage.measure import _marching_cubes_lewiner as lewiner

        scalar = _stabilize_isosurface_samples(field, level)
        nx, ny, nz = scalar.shape
        global_shape = tuple(int(value) for value in global_shape)
        grid_offset = tuple(int(value) for value in grid_offset)
        if len(global_shape) != 3 or any(size < 2 for size in global_shape):
            raise ValueError("global_shape must contain three dimensions of at least two")
        if len(grid_offset) != 3 or any(value < 0 for value in grid_offset):
            raise ValueError("grid_offset must contain three non-negative values")
        if any(
            grid_offset[axis] + scalar.shape[axis] > global_shape[axis]
            for axis in range(3)
        ):
            raise ValueError("field chunk lies outside global_shape")
        table = lewiner._to_array(lewiner.mcluts.CASESCLASSIC).astype(np.int8)
        device_field = _numba_cuda.to_device(scalar.ravel())
        if self._device_mc_table is None:
            self._device_mc_table = _numba_cuda.to_device(table)
        table_device = self._device_mc_table
        cell_count = (nx - 1) * (ny - 1) * (nz - 1)
        local_x_edge_count = (nx - 1) * ny * nz
        local_y_edge_count = nx * (ny - 1) * nz
        local_edge_count = local_x_edge_count + local_y_edge_count + nx * ny * (nz - 1)
        device_counts = _numba_cuda.device_array(cell_count, dtype=np.int32)
        device_active_edges = _numba_cuda.to_device(
            np.zeros(local_edge_count, dtype=np.uint8)
        )
        threads = 256
        blocks = (cell_count + threads - 1) // threads
        _cuda_mc_count_kernel[blocks, threads](
            device_field,
            device_counts,
            device_active_edges,
            table_device,
            nx,
            ny,
            nz,
            np.float32(level),
        )
        _numba_cuda.synchronize()
        counts = device_counts.copy_to_host()
        active_edge_ids = np.flatnonzero(
            device_active_edges.copy_to_host()
        ).astype(np.int64, copy=False)
        offsets64 = np.zeros(cell_count, dtype=np.int64)
        if cell_count > 1:
            offsets64[1:] = np.cumsum(counts[:-1], dtype=np.int64)
        triangle_count = int(offsets64[-1] + counts[-1]) if cell_count else 0
        if triangle_count == 0:
            return trimesh.Trimesh(process=False), np.empty(0, dtype=np.int64)
        if triangle_count > np.iinfo(np.int32).max:
            raise MemoryError("Marching Cubes output exceeds 32-bit face indexing")
        vertex_count = len(active_edge_ids)
        if vertex_count > np.iinfo(np.int32).max:
            raise MemoryError("Marching Cubes output exceeds 32-bit vertex indexing")
        offsets = offsets64.astype(np.int32)
        device_offsets = _numba_cuda.to_device(offsets)
        edge_to_vertex = np.full(local_edge_count, -1, dtype=np.int32)
        edge_to_vertex[active_edge_ids] = np.arange(vertex_count, dtype=np.int32)
        device_edge_to_vertex = _numba_cuda.to_device(edge_to_vertex)
        device_vertices = _numba_cuda.device_array((vertex_count, 3), dtype=np.float32)
        device_faces = _numba_cuda.device_array((triangle_count, 3), dtype=np.int32)
        device_spacing = _numba_cuda.to_device(np.asarray(spacing, dtype=np.float32))
        device_origin = _numba_cuda.to_device(np.asarray(origin, dtype=np.float32))
        _cuda_mc_emit_kernel[blocks, threads](
            device_field,
            device_offsets,
            device_vertices,
            device_edge_to_vertex,
            device_faces,
            table_device,
            nx,
            ny,
            nz,
            device_spacing,
            device_origin,
            np.float32(level),
        )
        _numba_cuda.synchronize()
        vertices = device_vertices.copy_to_host().astype(np.float64)
        faces = device_faces.copy_to_host().astype(np.int64)
        global_edge_ids = _local_edge_ids_to_global(
            active_edge_ids,
            scalar.shape,
            grid_offset,
            global_shape,
        )
        result = trimesh.Trimesh(
            vertices=vertices,
            faces=faces,
            process=False,
        )
        return result, global_edge_ids


class AutomaticGeometryBackend:
    """Select CUDA geometry operations and permanently fall back on failure."""

    def __init__(self, cpu_adapter: GeometryComputeAdapter, gpu_adapter):
        self._cpu_adapter = cpu_adapter
        self._gpu_adapter = gpu_adapter
        self._active_adapter = (
            gpu_adapter if gpu_adapter is not None and gpu_adapter.available else cpu_adapter
        )
        self._fallback_reason = (
            None
            if gpu_adapter is None or gpu_adapter.available
            else getattr(gpu_adapter, "unavailable_reason", None)
        )

    @property
    def status(self) -> GeometryBackendStatus:
        adapter = self._active_adapter
        return GeometryBackendStatus(
            adapter.name,
            adapter is self._gpu_adapter,
            adapter.device_name,
            self._fallback_reason,
        )

    def _run(self, name, *args):
        try:
            return getattr(self._active_adapter, name)(*args)
        except (TypeError, ValueError):
            # Invalid caller input is not a CUDA backend failure.  Let the
            # caller correct it without poisoning the process-wide backend.
            raise
        except Exception as exc:
            if self._active_adapter is not self._gpu_adapter:
                raise
            self._fallback_reason = f"{type(exc).__name__}: {exc}"
            self._active_adapter = self._cpu_adapter
            return getattr(self._cpu_adapter, name)(*args)

    def signed_distance(self, vertices, faces, points, max_tile_points=250_000):
        return np.asarray(
            self._run("signed_distance", vertices, faces, points, max_tile_points),
            dtype=np.float32,
        )

    def intersect_fields(self, field_a, field_b, max_tile_points=250_000):
        a = np.asarray(field_a, dtype=np.float32)
        b = np.asarray(field_b, dtype=np.float32)
        if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
            raise ValueError("intersection fields must have matching finite values")
        return np.asarray(
            self._run("intersect_fields", a, b, max_tile_points),
            dtype=np.float32,
        )

    def marching_cubes(self, field, spacing, origin, level=0.0):
        return self._run("marching_cubes", field, spacing, origin, level)

    def marching_cubes_chunk(
        self,
        field,
        spacing,
        origin,
        level,
        grid_offset,
        global_shape,
    ):
        return self._run(
            "marching_cubes_chunk",
            field,
            spacing,
            origin,
            level,
            grid_offset,
            global_shape,
        )


_DEFAULT_BACKEND = None
_DEFAULT_BACKEND_LOCK = threading.Lock()


def get_default_geometry_backend() -> AutomaticGeometryBackend:
    global _DEFAULT_BACKEND
    if _DEFAULT_BACKEND is None:
        with _DEFAULT_BACKEND_LOCK:
            if _DEFAULT_BACKEND is None:
                _DEFAULT_BACKEND = AutomaticGeometryBackend(
                    cpu_adapter=CpuGeometryAdapter(),
                    gpu_adapter=NumbaCudaGeometryAdapter(),
                )
    return _DEFAULT_BACKEND
