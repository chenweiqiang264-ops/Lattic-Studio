"""Implicit design-domain shelling and shell/lattice fusion.

All functions use the project's negative-inside SDF convention.  The shell
is constructed from the signed design-domain field rather than by offsetting
an STL mesh, which keeps it compatible with implicit rendering and later
Marching-Cubes reconstruction.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .field import ImplicitBody, SampledImplicitField, StageReporter


@dataclass(frozen=True)
class ShellParameters:
    """Physical controls for an inward design-domain shell."""

    thickness_mm: float
    fusion_radius_mm: float = 0.0

    def validate(self) -> None:
        thickness = float(self.thickness_mm)
        fusion_radius = float(self.fusion_radius_mm)
        if not np.isfinite(thickness) or thickness <= 0.0:
            raise ValueError("thickness_mm must be finite and positive")
        if not np.isfinite(fusion_radius) or fusion_radius < 0.0:
            raise ValueError("fusion_radius_mm must be finite and non-negative")


def _validate_solid_body(body: ImplicitBody, name: str) -> None:
    if not isinstance(body, ImplicitBody):
        raise TypeError(f"{name} must be an ImplicitBody")
    if body.is_preview_only:
        raise ValueError(f"{name} is preview-only and cannot define a solid")


def _shared_bounds(first: ImplicitBody, second: ImplicitBody) -> np.ndarray:
    bounds = np.vstack(
        (
            np.minimum(first.bounds[0], second.bounds[0]),
            np.maximum(first.bounds[1], second.bounds[1]),
        )
    )
    if np.any(bounds[0] >= bounds[1]):
        raise ValueError("implicit bodies must have finite positive bounds")
    return bounds


def smooth_union_values(
    first_values: np.ndarray,
    second_values: np.ndarray,
    fusion_radius_mm: float,
) -> np.ndarray:
    """Return hard or polynomial-smooth SDF union values.

    ``fusion_radius_mm=0`` is an exact hard union.  A positive radius only
    smooths the join; callers should apply a limiting SDF afterwards when the
    union must remain inside a design domain.
    """

    first = np.asarray(first_values, dtype=np.float32)
    second = np.asarray(second_values, dtype=np.float32)
    if first.shape != second.shape:
        raise ValueError("union inputs must have matching shapes")
    radius = float(fusion_radius_mm)
    if not np.isfinite(radius) or radius < 0.0:
        raise ValueError("fusion_radius_mm must be finite and non-negative")
    if radius == 0.0:
        return np.minimum(first, second).astype(np.float32, copy=False)
    radius32 = np.float32(radius)
    weight = np.clip(
        0.5 + 0.5 * (second - first) / radius32,
        0.0,
        1.0,
    )
    return (
        second * (1.0 - weight)
        + first * weight
        - radius32 * weight * (1.0 - weight)
    ).astype(np.float32, copy=False)


def fuse_shell_lattice_fields(
    domain_field: SampledImplicitField,
    lattice_field: SampledImplicitField,
    thickness_mm: float,
    fusion_radius_mm: float = 0.0,
    *,
    name: str = "ShellUnion",
    color: tuple[float, float, float, float] = (0.12, 0.58, 0.48, 1.0),
) -> SampledImplicitField:
    """Fuse shell and lattice samples already defined on one display grid.

    This only builds a disposable display cache. The authoritative implicit
    evaluators remain unchanged for later inspection and STL reconstruction.
    """

    if not isinstance(domain_field, SampledImplicitField):
        raise TypeError("domain_field must be a SampledImplicitField")
    if not isinstance(lattice_field, SampledImplicitField):
        raise TypeError("lattice_field must be a SampledImplicitField")
    if domain_field.is_preview_only:
        raise ValueError("a preview-only domain field cannot define a shell union")
    if (
        domain_field.values.shape != lattice_field.values.shape
        or not np.allclose(
            domain_field.origin,
            lattice_field.origin,
            rtol=0.0,
            atol=1.0e-9,
        )
        or not np.allclose(
            domain_field.spacing,
            lattice_field.spacing,
            rtol=0.0,
            atol=1.0e-9,
        )
    ):
        raise ValueError("domain and lattice fields must use the same sampling grid")
    parameters = ShellParameters(thickness_mm, fusion_radius_mm)
    parameters.validate()
    domain_values = domain_field.values
    shell_values = np.maximum(
        domain_values,
        -domain_values - np.float32(parameters.thickness_mm),
    )
    union_values = smooth_union_values(
        shell_values,
        lattice_field.values,
        parameters.fusion_radius_mm,
    )
    return SampledImplicitField(
        name=name,
        values=np.maximum(domain_values, union_values),
        origin=domain_field.origin,
        spacing=domain_field.spacing,
        color=color,
    )


def make_interior_shell(
    design_domain: ImplicitBody,
    thickness_mm: float,
    *,
    name: str = "Shell",
) -> ImplicitBody:
    """Build an inward shell that preserves the design-domain exterior.

    For a negative-inside domain field ``d``, shell material is the band
    ``-thickness <= d <= 0``.  Its implicit representation is
    ``max(d, -d - thickness)``.
    """

    _validate_solid_body(design_domain, "design_domain")
    parameters = ShellParameters(thickness_mm)
    parameters.validate()
    thickness = np.float32(parameters.thickness_mm)

    def evaluate(
        points: np.ndarray,
        reporter: StageReporter | None = None,
    ) -> np.ndarray:
        domain_values = design_domain.evaluate_points(points, reporter)
        return np.maximum(domain_values, -domain_values - thickness).astype(
            np.float32,
            copy=False,
        )

    return ImplicitBody(name=name, bounds=design_domain.bounds, evaluate=evaluate)


def make_shell_lattice_union(
    shell: ImplicitBody,
    lattice: ImplicitBody,
    design_domain: ImplicitBody,
    fusion_radius_mm: float = 0.0,
    *,
    name: str = "ShellUnion",
) -> ImplicitBody:
    """Fuse a shell and lattice while strictly retaining the design domain."""

    _validate_solid_body(shell, "shell")
    _validate_solid_body(lattice, "lattice")
    _validate_solid_body(design_domain, "design_domain")
    parameters = ShellParameters(1.0, fusion_radius_mm)
    parameters.validate()
    bounds = _shared_bounds(shell, lattice)
    bounds[0] = np.maximum(bounds[0], design_domain.bounds[0])
    bounds[1] = np.minimum(bounds[1], design_domain.bounds[1])
    if np.any(bounds[0] >= bounds[1]):
        raise ValueError("shell, lattice and design domain do not overlap")

    def evaluate(
        points: np.ndarray,
        reporter: StageReporter | None = None,
    ) -> np.ndarray:
        shell_values = shell.evaluate_points(points, reporter)
        lattice_values = lattice.evaluate_points(points, reporter)
        union_values = smooth_union_values(
            shell_values,
            lattice_values,
            parameters.fusion_radius_mm,
        )
        domain_values = design_domain.evaluate_points(points, reporter)
        return np.maximum(domain_values, union_values).astype(np.float32, copy=False)

    return ImplicitBody(name=name, bounds=bounds, evaluate=evaluate)


def make_coordinated_shell_lattice_union(
    design_domain: ImplicitBody,
    lattice_feature: ImplicitBody,
    thickness_mm: float,
    fusion_radius_mm: float = 0.0,
    *,
    name: str = "ShellUnion",
) -> ImplicitBody:
    """Build a shell/lattice union with one domain evaluation per point batch.

    ``lattice_feature`` is the periodic lattice field before design-domain
    clipping. Sharing one sampled domain array across shell construction,
    lattice clipping, and the final bound avoids three identical mesh-SDF
    queries in nested evaluators.
    """

    _validate_solid_body(design_domain, "design_domain")
    _validate_solid_body(lattice_feature, "lattice_feature")
    parameters = ShellParameters(thickness_mm, fusion_radius_mm)
    parameters.validate()
    bounds = _shared_bounds(design_domain, lattice_feature)
    bounds[0] = np.maximum(bounds[0], design_domain.bounds[0])
    bounds[1] = np.minimum(bounds[1], design_domain.bounds[1])
    if np.any(bounds[0] >= bounds[1]):
        raise ValueError("lattice feature and design domain do not overlap")
    thickness = np.float32(parameters.thickness_mm)

    def evaluate(
        points: np.ndarray,
        reporter: StageReporter | None = None,
    ) -> np.ndarray:
        domain_values = design_domain.evaluate_points(points, reporter)
        feature_values = lattice_feature.evaluate_points(points, reporter)
        lattice_values = np.maximum(domain_values, feature_values)
        shell_values = np.maximum(
            domain_values,
            -domain_values - thickness,
        )
        union_values = smooth_union_values(
            shell_values,
            lattice_values,
            parameters.fusion_radius_mm,
        )
        return np.maximum(domain_values, union_values).astype(
            np.float32,
            copy=False,
        )

    return ImplicitBody(name=name, bounds=bounds, evaluate=evaluate)
