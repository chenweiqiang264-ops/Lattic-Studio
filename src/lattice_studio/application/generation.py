"""Application use cases for implicit lattice generation.

This module is the seam between presentation code and the numerical engine.
It deliberately returns engine results and does not expose Qt workers or
widgets.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import trimesh

from lattice_studio.domain.parameters import (
    CustomUnitCellParameters,
    LatticeParameters,
    SamplingParameters,
    TPMSParameters,
    TransitionParameters,
)

ProgressCallback = Callable[[str, float], None]


class LatticeGenerationService:
    """Coordinate supported lattice generation use cases."""

    def generate_lattice(
        self,
        design_domain: trimesh.Trimesh,
        parameters: TPMSParameters | CustomUnitCellParameters,
        *,
        sampling: SamplingParameters | None = None,
        progress: ProgressCallback | None = None,
    ):
        from lattice_studio.engine import workflows

        if isinstance(parameters, CustomUnitCellParameters):
            return workflows.generate_implicit_custom_lattice(
                design_domain, parameters, sampling, progress=progress
            )
        return workflows.generate_implicit_lattice(
            design_domain, parameters, sampling, progress=progress
        )

    def generate_transition(
        self,
        design_domain: trimesh.Trimesh,
        first: LatticeParameters,
        second: LatticeParameters,
        transition: TransitionParameters,
        *,
        sampling: SamplingParameters | None = None,
        progress: ProgressCallback | None = None,
    ):
        from lattice_studio.engine import workflows

        return workflows.generate_implicit_lattice_transition(
            design_domain,
            first,
            second,
            transition,
            sampling,
            progress=progress,
        )

    def reconstruct_stl(
        self,
        body,
        cell_map,
        *,
        tolerance_mm: float,
        wall_thickness_mm: float,
        sampling: SamplingParameters,
        progress: ProgressCallback | None = None,
    ):
        from lattice_studio.engine import workflows

        return workflows.reconstruct_implicit_mesh(
            body,
            cell_map,
            tolerance_mm,
            wall_thickness_mm,
            progress=progress,
            processing_mode=sampling.processing_mode,
            batch_count=sampling.batch_count,
        )


__all__ = ["LatticeGenerationService", "ProgressCallback"]
