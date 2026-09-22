"""Backend-owned implementations of serializable implicit-workflow tasks."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from threading import RLock
from uuid import uuid4

import numpy as np
import trimesh

from lattice_studio.application.tasks import TaskContext, TaskOutcome
from lattice_studio.domain.parameters import SamplingParameters, TPMSParameters
from lattice_studio.engine.contracts import ImplicitGenerationResult
from lattice_studio.engine.workflows import (
    generate_implicit_lattice,
    reconstruct_implicit_mesh,
)


class ImplicitWorkflowTasks:
    """Run TPMS and STL tasks while retaining authority in the backend process."""

    def __init__(self) -> None:
        self._lock = RLock()
        self._generations: dict[str, ImplicitGenerationResult] = {}

    def handlers(self) -> dict[str, object]:
        return {
            "tpms.generate": self.generate_tpms,
            "stl.reconstruct": self.reconstruct_stl,
        }

    def generate_tpms(
        self,
        payload: Mapping[str, object],
        context: TaskContext,
    ) -> TaskOutcome:
        context.report("loading design domain", 0.02)
        mesh = _load_mesh(payload["mesh_path"])
        parameters = TPMSParameters(**_object(payload["parameters"], "parameters"))
        sampling = SamplingParameters(**_object(payload.get("sampling", {}), "sampling"))
        display_voxel_size = _optional_positive_float(
            payload.get("display_voxel_size_mm"), "display_voxel_size_mm"
        )
        display_memory_budget = float(payload.get("display_memory_budget_mb", 512.0))
        display_batch_count = _optional_positive_int(
            payload.get("display_batch_count"), "display_batch_count"
        )
        if display_memory_budget <= 0.0:
            raise ValueError("display_memory_budget_mb must be positive")

        def progress(message: str, value: float) -> None:
            context.report(message, 0.02 + 0.96 * min(max(float(value), 0.0), 1.0))

        result = generate_implicit_lattice(
            mesh,
            parameters,
            sampling,
            display_voxel_size_mm=display_voxel_size,
            display_memory_budget_mb=display_memory_budget,
            display_batch_count=display_batch_count,
            progress=progress,
        )
        context.raise_if_cancelled()
        generation_id = uuid4().hex
        with self._lock:
            self._generations[generation_id] = result
        field_path = context.work_dir / "display-field.npz"
        _write_field(field_path, result)
        context.report("display field ready", 1.0)
        return TaskOutcome(
            result={
                "generation_id": generation_id,
                "kind": parameters.kind,
                "recommendation": _recommendation_payload(result),
                "cell_map": _cell_map_payload(result),
                "minimum_feature_mm": result.minimum_feature_mm,
            },
            artifacts=(field_path,),
        )

    def reconstruct_stl(
        self,
        payload: Mapping[str, object],
        context: TaskContext,
    ) -> TaskOutcome:
        generation_id = str(payload["generation_id"])
        with self._lock:
            result = self._generations.get(generation_id)
        if result is None:
            raise KeyError(f"unknown backend generation: {generation_id}")
        if result.cell_map is None or result.minimum_feature_mm is None:
            raise ValueError("generation cannot be reconstructed as STL")
        tolerance_mm = _positive_float(payload.get("tolerance_mm", 0.2), "tolerance_mm")
        repair_tolerance_mm = float(payload.get("repair_tolerance_mm", 0.0))
        processing_mode = str(payload.get("processing_mode", "single_pass"))
        batch_count = int(payload.get("batch_count", 1))
        spacing_mode = str(payload.get("spacing_mode", "recommended"))
        clean_numerical_fragments = bool(payload.get("clean_numerical_fragments", True))
        optimize_for_slicing = bool(payload.get("optimize_for_slicing", True))

        def progress(message: str, value: float) -> None:
            context.report(message, min(max(float(value), 0.0), 1.0))

        mesh_result = reconstruct_implicit_mesh(
            result.body,
            result.cell_map,
            tolerance_mm,
            result.minimum_feature_mm,
            repair_tolerance_mm=repair_tolerance_mm,
            clean_numerical_fragments=clean_numerical_fragments,
            processing_mode=processing_mode,  # validated by the workflow
            batch_count=batch_count,
            spacing_mode=spacing_mode,  # validated by the workflow
            optimize_for_slicing=optimize_for_slicing,
            progress=progress,
        )
        context.raise_if_cancelled()
        output = context.work_dir / "lattice.stl"
        mesh_result.mesh.export(output)
        context.report("STL ready", 1.0)
        return TaskOutcome(
            result={
                "generation_id": generation_id,
                "triangle_count": len(mesh_result.mesh.faces),
                "spacing_mm": list(mesh_result.spacing_mm),
                "tolerance_mm": mesh_result.tolerance_mm,
                "extraction_backend": mesh_result.extraction_backend,
            },
            artifacts=(output,),
        )


def _load_mesh(value: object) -> trimesh.Trimesh:
    path = Path(str(value)).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"design-domain mesh does not exist: {path}")
    # STL has no shared vertex table.  Let trimesh merge its duplicated face
    # vertices at this file boundary so watertightness has its geometric meaning.
    mesh = trimesh.load_mesh(path, force="mesh", process=True)
    if not isinstance(mesh, trimesh.Trimesh) or len(mesh.faces) == 0:
        raise ValueError("design-domain mesh must contain triangular faces")
    return mesh


def _write_field(path: Path, result: ImplicitGenerationResult) -> None:
    field = result.display_field
    np.savez_compressed(
        path,
        name=np.asarray(field.name),
        values=field.values,
        origin=field.origin,
        spacing=field.spacing,
        color=np.asarray(field.color),
        is_preview_only=np.asarray(field.is_preview_only),
    )


def _recommendation_payload(result: ImplicitGenerationResult) -> dict[str, object]:
    recommendation = result.recommendation
    return {
        "voxel_size_mm": recommendation.voxel_size_mm,
        "grid_shape": list(recommendation.grid_shape),
        "estimated_voxels": recommendation.estimated_voxels,
        "samples_per_cell": recommendation.samples_per_cell,
        "samples_per_wall": recommendation.samples_per_wall,
        "limiting_constraint": recommendation.limiting_constraint,
    }


def _cell_map_payload(result: ImplicitGenerationResult) -> dict[str, object] | None:
    cell_map = result.cell_map
    if cell_map is None:
        return None
    return {
        "bounds": cell_map.bounds.tolist(),
        "spacing_mm": list(cell_map.spacing_mm),
        "cell_counts": list(cell_map.cell_counts),
        "boundary_mode": cell_map.boundary_mode,
        "index_min": list(cell_map.index_min),
        "frame_origin": cell_map.frame.origin.tolist(),
        "frame_axes": cell_map.frame.axes.tolist(),
    }


def _object(value: object, name: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a JSON object")
    return value


def _positive_float(value: object, name: str) -> float:
    numeric = float(value)
    if numeric <= 0.0:
        raise ValueError(f"{name} must be positive")
    return numeric


def _optional_positive_float(value: object, name: str) -> float | None:
    return None if value is None else _positive_float(value, name)


def _optional_positive_int(value: object, name: str) -> int | None:
    if value is None:
        return None
    numeric = int(value)
    if numeric < 1:
        raise ValueError(f"{name} must be at least one")
    return numeric


__all__ = ["ImplicitWorkflowTasks"]
