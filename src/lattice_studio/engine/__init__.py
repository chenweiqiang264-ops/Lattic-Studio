"""Numerical engine interfaces and implementations."""

from lattice_studio.engine.design_domains import (
    AnalyticDesignDomain,
    MeshDesignDomain,
)
from lattice_studio.engine.contracts import (
    ExportMeshResult,
    ImplicitGenerationResult,
    SamplingRecommendation,
)
from lattice_studio.engine.mesh_quality import inspect_mesh, repair_mesh
from lattice_studio.engine.reconstruction_grid import (
    estimate_export_grid,
    recommend_export_spacing,
    resolve_export_spacing,
)

__all__ = [
    "AnalyticDesignDomain",
    "ExportMeshResult",
    "ImplicitGenerationResult",
    "MeshDesignDomain",
    "inspect_mesh",
    "repair_mesh",
    "estimate_export_grid",
    "recommend_export_spacing",
    "resolve_export_spacing",
    "SamplingRecommendation",
]
