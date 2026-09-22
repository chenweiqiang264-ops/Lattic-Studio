"""Analytic implicit primitives used as independent scalar-field objects.

The preview mesh exported here is deliberately derived data.  The SDF
evaluator is the authoritative definition consumed by field inspection and
field-driven transition generation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from lattice_studio.engine.implicit.field import ImplicitBody


PrimitiveKind = Literal["sphere", "cylinder", "box"]


def rotation_matrix_from_euler_deg(euler_deg: tuple[float, float, float]) -> np.ndarray:
    """Return a right-handed XYZ intrinsic rotation matrix."""

    angles = np.deg2rad(np.asarray(euler_deg, dtype=np.float64))
    if angles.shape != (3,) or not np.isfinite(angles).all():
        raise ValueError("Euler angles must contain three finite values")
    cx, cy, cz = np.cos(angles)
    sx, sy, sz = np.sin(angles)
    rotate_x = np.array(((1.0, 0.0, 0.0), (0.0, cx, -sx), (0.0, sx, cx)))
    rotate_y = np.array(((cy, 0.0, sy), (0.0, 1.0, 0.0), (-sy, 0.0, cy)))
    rotate_z = np.array(((cz, -sz, 0.0), (sz, cz, 0.0), (0.0, 0.0, 1.0)))
    return rotate_z @ rotate_y @ rotate_x


def euler_deg_from_rotation_matrix(matrix: np.ndarray) -> tuple[float, float, float]:
    """Return the XYZ intrinsic Euler representation of a rotation matrix."""

    rotation = np.asarray(matrix, dtype=np.float64)
    if rotation.shape != (3, 3) or not np.isfinite(rotation).all():
        raise ValueError("rotation matrix must have shape (3, 3)")
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1.0e-6):
        raise ValueError("rotation matrix must be orthonormal")
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=1.0e-6):
        raise ValueError("rotation matrix must be right-handed")
    sy = float(-rotation[2, 0])
    pitch = float(np.arcsin(np.clip(sy, -1.0, 1.0)))
    cosine = float(np.cos(pitch))
    if abs(cosine) > 1.0e-8:
        roll = float(np.arctan2(rotation[2, 1], rotation[2, 2]))
        yaw = float(np.arctan2(rotation[1, 0], rotation[0, 0]))
    else:
        roll = float(np.arctan2(-rotation[1, 2], rotation[1, 1]))
        yaw = 0.0
    return tuple(float(value) for value in np.rad2deg((roll, pitch, yaw)))


@dataclass(frozen=True)
class ImplicitPrimitive:
    """One named analytic SDF scene object in millimetres.

    Cylinders use their local Z axis and ``height_mm`` is their complete
    end-to-end length.  Boxes use full X/Y/Z dimensions, not half extents.
    """

    identifier: str
    name: str
    kind: PrimitiveKind
    center_mm: tuple[float, float, float] = (0.0, 0.0, 0.0)
    rotation_euler_deg: tuple[float, float, float] = (0.0, 0.0, 0.0)
    radius_mm: float = 5.0
    height_mm: float = 10.0
    size_mm: tuple[float, float, float] = (10.0, 10.0, 10.0)

    def __post_init__(self) -> None:
        if not self.identifier.strip():
            raise ValueError("primitive identifier must not be empty")
        if not self.name.strip():
            raise ValueError("primitive name must not be empty")
        if self.kind not in ("sphere", "cylinder", "box"):
            raise ValueError("unsupported primitive kind")
        center = np.asarray(self.center_mm, dtype=np.float64)
        size = np.asarray(self.size_mm, dtype=np.float64)
        if center.shape != (3,) or not np.isfinite(center).all():
            raise ValueError("primitive center must contain three finite values")
        if size.shape != (3,) or not np.isfinite(size).all() or np.any(size <= 0.0):
            raise ValueError("primitive box dimensions must be positive")
        if not np.isfinite(self.radius_mm) or self.radius_mm <= 0.0:
            raise ValueError("primitive radius must be positive")
        if not np.isfinite(self.height_mm) or self.height_mm <= 0.0:
            raise ValueError("primitive height must be positive")
        rotation_matrix_from_euler_deg(self.rotation_euler_deg)
        object.__setattr__(self, "center_mm", tuple(float(value) for value in center))
        object.__setattr__(self, "size_mm", tuple(float(value) for value in size))
        object.__setattr__(
            self,
            "rotation_euler_deg",
            tuple(float(value) for value in self.rotation_euler_deg),
        )

    @property
    def rotation_matrix(self) -> np.ndarray:
        return rotation_matrix_from_euler_deg(self.rotation_euler_deg)

    @property
    def local_half_extents_mm(self) -> np.ndarray:
        if self.kind == "sphere":
            return np.full(3, self.radius_mm, dtype=np.float64)
        if self.kind == "cylinder":
            return np.array(
                (self.radius_mm, self.radius_mm, 0.5 * self.height_mm),
                dtype=np.float64,
            )
        return np.asarray(self.size_mm, dtype=np.float64) * 0.5

    @property
    def bounds(self) -> np.ndarray:
        """Return a tight axis-aligned world-space bound for the primitive."""

        center = np.asarray(self.center_mm, dtype=np.float64)
        if self.kind == "sphere":
            extent = np.full(3, self.radius_mm, dtype=np.float64)
        elif self.kind == "cylinder":
            rotation = self.rotation_matrix
            radial = self.radius_mm * np.sqrt(rotation[:, 0] ** 2 + rotation[:, 1] ** 2)
            extent = radial + 0.5 * self.height_mm * np.abs(rotation[:, 2])
        else:
            extent = np.abs(self.rotation_matrix) @ self.local_half_extents_mm
        return np.vstack((center - extent, center + extent))

    def evaluate_points(self, points: np.ndarray) -> np.ndarray:
        """Evaluate the exact signed distance at world-space points."""

        world = np.asarray(points, dtype=np.float32)
        if world.ndim != 2 or world.shape[1] != 3:
            raise ValueError("points must have shape (N, 3)")
        local = (world - np.asarray(self.center_mm, dtype=np.float32)) @ self.rotation_matrix.astype(np.float32)
        if self.kind == "sphere":
            return (np.linalg.norm(local, axis=1) - np.float32(self.radius_mm)).astype(np.float32)
        if self.kind == "cylinder":
            radial = np.linalg.norm(local[:, :2], axis=1) - np.float32(self.radius_mm)
            caps = np.abs(local[:, 2]) - np.float32(0.5 * self.height_mm)
            distance = np.column_stack((radial, caps))
        else:
            distance = np.abs(local) - self.local_half_extents_mm.astype(np.float32)
        outside = np.linalg.norm(np.maximum(distance, 0.0), axis=1)
        inside = np.minimum(np.max(distance, axis=1), 0.0)
        return (outside + inside).astype(np.float32, copy=False)

    def as_implicit_body(self) -> ImplicitBody:
        """Create the authoritative evaluator consumed by implicit workflows."""

        return ImplicitBody(
            name=self.name,
            bounds=self.bounds,
            evaluate=lambda points, _reporter=None: self.evaluate_points(points),
        )

    def preview_mesh(self, subdivisions: int = 64) -> tuple[np.ndarray, np.ndarray]:
        """Create a lightweight local-space mesh for selection and placement."""

        resolution = int(subdivisions)
        if resolution < 8:
            raise ValueError("preview subdivisions must be at least 8")
        import trimesh

        if self.kind == "sphere":
            mesh = trimesh.creation.uv_sphere(radius=self.radius_mm, count=[resolution, resolution])
        elif self.kind == "cylinder":
            mesh = trimesh.creation.cylinder(
                radius=self.radius_mm,
                height=self.height_mm,
                sections=resolution,
            )
        else:
            mesh = trimesh.creation.box(extents=self.size_mm)
        return (
            np.asarray(mesh.vertices, dtype=np.float32),
            np.asarray(mesh.faces, dtype=np.int32),
        )
