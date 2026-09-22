"""Geometry-backed implementations of design-domain interfaces."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from itertools import product
from pathlib import Path
from typing import Literal, TypeAlias

import numpy as np
import trimesh

from lattice_studio.engine.implicit.field import ImplicitBody
from lattice_studio.engine.implicit.geometry_compute import get_default_geometry_backend
from lattice_studio.engine.implicit.primitives import ImplicitPrimitive


DesignDomainKind = Literal["mesh", "analytic"]


def _validate_points(points: np.ndarray) -> np.ndarray:
    values = np.ascontiguousarray(points, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != 3 or not np.isfinite(values).all():
        raise ValueError("points must have shape (N, 3) and contain finite values")
    return values


def _validate_bounds(bounds: np.ndarray) -> np.ndarray:
    values = np.asarray(bounds, dtype=np.float64)
    if values.shape != (2, 3) or not np.isfinite(values).all():
        raise ValueError("design-domain bounds must have shape (2, 3) and be finite")
    if not np.all(values[0] < values[1]):
        raise ValueError("design-domain bounds must have positive extents")
    return values


@dataclass
class MeshDesignDomain:
    mesh: trimesh.Trimesh
    name: str = "Mesh design domain"
    asset_path: Path | None = None
    kind: DesignDomainKind = field(default="mesh", init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.mesh, trimesh.Trimesh):
            raise TypeError("mesh design domain requires a trimesh.Trimesh")
        if not self.name.strip() or len(self.mesh.vertices) == 0 or len(self.mesh.faces) == 0:
            raise ValueError("mesh design domain must have a name and geometry")
        self.mesh = self.mesh.copy()
        _validate_bounds(self.mesh.bounds)
        if self.asset_path is not None:
            self.asset_path = Path(self.asset_path)

    @property
    def bounds(self) -> np.ndarray:
        return _validate_bounds(self.mesh.bounds)

    @property
    def frame_points(self) -> np.ndarray:
        return np.asarray(self.mesh.vertices, dtype=np.float64)

    @property
    def is_watertight(self) -> bool:
        return bool(self.mesh.is_watertight)

    def evaluate_points(self, points: np.ndarray) -> np.ndarray:
        return get_default_geometry_backend().signed_distance(
            self.mesh.vertices,
            self.mesh.faces,
            _validate_points(points),
            max_tile_points=250_000,
        )

    def as_implicit_body(self, preview_shell_thickness_mm: float = 0.5) -> ImplicitBody:
        shell_thickness = float(preview_shell_thickness_mm)
        if not np.isfinite(shell_thickness) or shell_thickness <= 0.0:
            raise ValueError("preview_shell_thickness_mm must be finite and positive")

        def evaluate(points: np.ndarray, _reporter=None) -> np.ndarray:
            sdf = self.evaluate_points(points)
            return sdf if self.is_watertight else np.abs(sdf) - shell_thickness

        return ImplicitBody(self.name, self.bounds, evaluate, is_preview_only=not self.is_watertight)

    def preview_mesh(self) -> trimesh.Trimesh:
        return self.mesh.copy()

    def duplicate_definition(self) -> "MeshDesignDomain":
        return MeshDesignDomain(self.mesh, self.name, self.asset_path)


@dataclass(frozen=True)
class AnalyticDesignDomain:
    primitive: ImplicitPrimitive
    name: str = "Analytic design domain"
    kind: DesignDomainKind = field(default="analytic", init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.primitive, ImplicitPrimitive):
            raise TypeError("analytic design domain requires an ImplicitPrimitive")
        if not self.name.strip():
            raise ValueError("design-domain name must not be empty")

    @property
    def bounds(self) -> np.ndarray:
        return self.primitive.bounds

    @property
    def frame_points(self) -> np.ndarray:
        bounds = _validate_bounds(self.bounds)
        return np.asarray(tuple(product(*zip(bounds[0], bounds[1]))), dtype=np.float64)

    @property
    def is_watertight(self) -> bool:
        return True

    def evaluate_points(self, points: np.ndarray) -> np.ndarray:
        return self.primitive.evaluate_points(_validate_points(points))

    def as_implicit_body(self, preview_shell_thickness_mm: float = 0.5) -> ImplicitBody:
        del preview_shell_thickness_mm
        return replace(self.primitive.as_implicit_body(), name=self.name)

    def preview_mesh(self) -> trimesh.Trimesh:
        vertices, faces = self.primitive.preview_mesh()
        vertices = vertices @ self.primitive.rotation_matrix.T + np.asarray(self.primitive.center_mm)
        return trimesh.Trimesh(vertices=vertices, faces=faces, process=False)

    def duplicate_definition(self) -> "AnalyticDesignDomain":
        return AnalyticDesignDomain(replace(self.primitive), self.name)


DesignDomainValue: TypeAlias = MeshDesignDomain | AnalyticDesignDomain

__all__ = ["AnalyticDesignDomain", "DesignDomainKind", "DesignDomainValue", "MeshDesignDomain"]
