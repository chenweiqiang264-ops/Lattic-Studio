"""Execute one implicit-pipeline stress scenario in an isolated process.

This module is intentionally not a pytest test.  The matrix runner launches it
in a fresh process so an OpenGL, VTK, CUDA, or native geometry crash cannot
abort the remaining scenarios.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
import traceback
from pathlib import Path
from typing import Callable

import numpy as np
import psutil
import trimesh


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(PROJECT_ROOT))

from lattice_studio.engine.implicit.custom_unit_cell import prepare_stl_unit_cell
from lattice_studio.presentation.qt import workbench as app


ProgressCallback = Callable[[str, float], None]


class ProgressRecorder:
    def __init__(self) -> None:
        self.events = 0
        self.maximum = 0.0
        self.last_message = ""

    def __call__(self, message: str, value: float) -> None:
        self.events += 1
        self.maximum = max(self.maximum, float(value))
        self.last_message = str(message)

    def as_dict(self) -> dict[str, object]:
        if self.events == 0 or self.maximum < 0.999:
            raise AssertionError(
                f"progress did not complete: events={self.events}, maximum={self.maximum:.3f}"
            )
        return {
            "events": self.events,
            "maximum": self.maximum,
            "last_message": self.last_message,
        }


def _load_domain(path: Path | None, extents: tuple[float, float, float]) -> trimesh.Trimesh:
    if path is None:
        return trimesh.creation.box(extents=extents)
    loaded = trimesh.load(path, force="mesh", process=True)
    if not isinstance(loaded, trimesh.Trimesh) or len(loaded.faces) == 0:
        raise ValueError(f"design domain is not a non-empty mesh: {path}")
    if not loaded.is_watertight:
        raise ValueError(f"design domain must be watertight: {path}")
    return loaded


def _custom_cell_path(path: Path | None, directory: Path) -> Path:
    if path is not None:
        return path
    generated = directory / "synthetic_custom_cell.stl"
    trimesh.creation.icosphere(subdivisions=2, radius=5.0).export(generated)
    return generated


def _assert_field(result: app.ImplicitGenerationResult) -> dict[str, object]:
    values = np.asarray(result.display_field.values)
    if values.ndim != 3 or values.size == 0:
        raise AssertionError("display field is empty or not three-dimensional")
    if not np.isfinite(values).all():
        raise AssertionError("display field contains non-finite values")
    if not (np.any(values < 0.0) and np.any(values > 0.0)):
        raise AssertionError("display field does not resolve a zero crossing")
    return {
        "grid_shape": list(values.shape),
        "voxels": int(values.size),
        "value_min": float(values.min()),
        "value_max": float(values.max()),
        "spacing_mm": [float(value) for value in result.display_field.spacing],
    }


def _custom_generation(
    custom_cell: Path,
    domain: trimesh.Trimesh,
    progress: ProgressCallback,
    *,
    display_voxel_mm: float,
) -> app.ImplicitGenerationResult:
    prepared = prepare_stl_unit_cell(custom_cell)
    parameters = app.CustomUnitCellParameters(
        source_path=custom_cell,
        cell_size_mm=(10.0, 10.0, 8.0),
        target_feature_mm=1.0,
    )
    return app.generate_implicit_custom_lattice(
        domain,
        parameters,
        app.SamplingParameters(target_voxels=2_000_000),
        display_voxel_size_mm=display_voxel_mm,
        display_memory_budget_mb=1024.0,
        display_batch_count=8,
        progress=progress,
        prepared_unit_cell=prepared,
    )


def scenario_custom_smoke(args: argparse.Namespace, directory: Path) -> dict[str, object]:
    custom = _custom_cell_path(args.custom_cell, directory)
    domain = _load_domain(None, (24.0, 20.0, 12.0))
    progress = ProgressRecorder()
    result = _custom_generation(custom, domain, progress, display_voxel_mm=0.75)
    return {"field": _assert_field(result), "progress": progress.as_dict()}


def scenario_custom_large(args: argparse.Namespace, directory: Path) -> dict[str, object]:
    custom = _custom_cell_path(args.custom_cell, directory)
    domain = _load_domain(args.domain_stl, (100.0, 100.0, 20.0))
    progress = ProgressRecorder()
    result = _custom_generation(custom, domain, progress, display_voxel_mm=0.55)
    field = _assert_field(result)
    return {
        "field": field,
        "source_faces": int(result.metadata.prepared_faces),
        "source_bytes": int(custom.stat().st_size),
        "domain_faces": int(len(domain.faces)),
        "progress": progress.as_dict(),
    }


def scenario_custom_reconstruct_chunked(
    args: argparse.Namespace,
    directory: Path,
) -> dict[str, object]:
    custom = _custom_cell_path(args.custom_cell, directory)
    domain = _load_domain(args.domain_stl, (80.0, 80.0, 20.0))
    generation_progress = ProgressRecorder()
    generated = _custom_generation(
        custom,
        domain,
        generation_progress,
        display_voxel_mm=0.75,
    )
    export_progress = ProgressRecorder()
    rebuilt = app.reconstruct_implicit_mesh(
        generated.body,
        generated.cell_map,
        tolerance_mm=0.45,
        wall_thickness_mm=float(generated.minimum_feature_mm),
        clean_numerical_fragments=True,
        progress=export_progress,
        processing_mode="chunked_marching_cubes",
        batch_count=8,
    )
    if len(rebuilt.mesh.faces) == 0:
        raise AssertionError("custom-cell reconstruction produced an empty mesh")
    return {
        "field": _assert_field(generated),
        "faces": int(len(rebuilt.mesh.faces)),
        "vertices": int(len(rebuilt.mesh.vertices)),
        "watertight": bool(rebuilt.quality.watertight),
        "components": int(rebuilt.quality.connected_components),
        "extraction_backend": rebuilt.extraction_backend,
        "generation_progress": generation_progress.as_dict(),
        "export_progress": export_progress.as_dict(),
    }


def scenario_transition_smoke(args: argparse.Namespace, directory: Path) -> dict[str, object]:
    del args, directory
    domain = trimesh.creation.box(extents=(30.0, 24.0, 12.0))
    progress = ProgressRecorder()
    generated = app.generate_implicit_lattice_transition(
        domain,
        app.TPMSParameters("G", (6.0, 6.0, 6.0), 0.9),
        app.TPMSParameters("D", (7.0, 6.0, 5.0), 0.8),
        app.TransitionParameters(
            plane_axis="X",
            transition_width_mm=3.0,
            weight_kind="smootherstep",
        ),
        app.SamplingParameters(target_voxels=500_000),
        display_voxel_size_mm=0.65,
        display_memory_budget_mb=256.0,
        display_batch_count=4,
        progress=progress,
    )
    if not isinstance(generated.metadata, app.TransitionGenerationMetadata):
        raise AssertionError("transition diagnostics were not retained")
    return {
        "field": _assert_field(generated),
        "first": generated.metadata.first_name,
        "second": generated.metadata.second_name,
        "weight": generated.metadata.diagnostics.resolved_weight_kind,
        "progress": progress.as_dict(),
    }


def scenario_transition_custom_large(
    args: argparse.Namespace,
    directory: Path,
) -> dict[str, object]:
    custom = _custom_cell_path(args.custom_cell, directory)
    prepared = prepare_stl_unit_cell(custom)
    domain = _load_domain(args.domain_stl, (90.0, 70.0, 20.0))
    progress = ProgressRecorder()
    generated = app.generate_implicit_lattice_transition(
        domain,
        app.CustomUnitCellParameters(
            custom,
            cell_size_mm=(10.0, 10.0, 8.0),
            target_feature_mm=1.0,
        ),
        app.TPMSParameters("G", (8.0, 8.0, 8.0), 1.0),
        app.TransitionParameters(
            plane_axis="X",
            transition_width_mm=3.0,
            weight_kind="automatic",
        ),
        app.SamplingParameters(target_voxels=3_000_000),
        display_voxel_size_mm=0.6,
        display_memory_budget_mb=1024.0,
        display_batch_count=8,
        progress=progress,
        prepared_custom_cell=prepared,
    )
    return {
        "field": _assert_field(generated),
        "first": generated.metadata.first_name,
        "second": generated.metadata.second_name,
        "source_faces": int(prepared.report.prepared_faces),
        "source_bytes": int(custom.stat().st_size),
        "progress": progress.as_dict(),
    }


def _tpms_generation(
    domain: trimesh.Trimesh,
    progress: ProgressCallback,
    display_voxel_mm: float,
) -> app.ImplicitGenerationResult:
    return app.generate_implicit_lattice(
        domain,
        app.TPMSParameters("G", (8.0, 8.0, 8.0), 1.0),
        app.SamplingParameters(target_voxels=2_000_000),
        display_voxel_size_mm=display_voxel_mm,
        display_memory_budget_mb=1024.0,
        display_batch_count=8,
        progress=progress,
    )


def _shell_fusion(
    domain: trimesh.Trimesh,
    display_voxel_mm: float,
) -> dict[str, object]:
    lattice_progress = ProgressRecorder()
    lattice = _tpms_generation(domain, lattice_progress, display_voxel_mm)
    fusion_progress = ProgressRecorder()
    fused = app.generate_implicit_shell_lattice_union(
        domain,
        lattice,
        thickness_mm=1.2,
        fusion_radius_mm=0.6,
        sampling=app.SamplingParameters(target_voxels=2_000_000),
        display_voxel_size_mm=display_voxel_mm,
        display_memory_budget_mb=1024.0,
        display_batch_count=8,
        progress=fusion_progress,
    )
    return {
        "field": _assert_field(fused),
        "fusion_backend": fused.metadata["display_fusion_backend"],
        "lattice_progress": lattice_progress.as_dict(),
        "fusion_progress": fusion_progress.as_dict(),
    }


def scenario_shell_fusion_smoke(args: argparse.Namespace, directory: Path) -> dict[str, object]:
    del args, directory
    return _shell_fusion(trimesh.creation.box(extents=(24.0, 20.0, 12.0)), 0.75)


def scenario_shell_fusion_large(args: argparse.Namespace, directory: Path) -> dict[str, object]:
    del directory
    domain = _load_domain(args.domain_stl, (100.0, 100.0, 20.0))
    result = _shell_fusion(domain, 0.5)
    result["domain_faces"] = int(len(domain.faces))
    return result


def _memory_reconstruction(mode: str, batch_count: int) -> dict[str, object]:
    domain = trimesh.creation.box(extents=(100.0, 100.0, 20.0))
    progress = ProgressRecorder()
    generated = _tpms_generation(domain, progress, 0.75)
    export_progress = ProgressRecorder()
    rebuilt = app.reconstruct_implicit_mesh(
        generated.body,
        generated.cell_map,
        tolerance_mm=0.35,
        wall_thickness_mm=1.0,
        clean_numerical_fragments=True,
        progress=export_progress,
        processing_mode=mode,
        batch_count=batch_count,
    )
    if len(rebuilt.mesh.faces) == 0:
        raise AssertionError("memory-boundary reconstruction produced an empty mesh")
    return {
        "faces": int(len(rebuilt.mesh.faces)),
        "vertices": int(len(rebuilt.mesh.vertices)),
        "watertight": bool(rebuilt.quality.watertight),
        "components": int(rebuilt.quality.connected_components),
        "extraction_backend": rebuilt.extraction_backend,
        "generation_progress": progress.as_dict(),
        "export_progress": export_progress.as_dict(),
    }


def scenario_memory_dense(args: argparse.Namespace, directory: Path) -> dict[str, object]:
    del args, directory
    return _memory_reconstruction("single_pass", 1)


def scenario_memory_chunked(args: argparse.Namespace, directory: Path) -> dict[str, object]:
    del args, directory
    return _memory_reconstruction("chunked_marching_cubes", 10)


SCENARIOS = {
    "custom_smoke": scenario_custom_smoke,
    "transition_smoke": scenario_transition_smoke,
    "shell_fusion_smoke": scenario_shell_fusion_smoke,
    "custom_large": scenario_custom_large,
    "custom_reconstruct_chunked": scenario_custom_reconstruct_chunked,
    "transition_custom_large": scenario_transition_custom_large,
    "shell_fusion_large": scenario_shell_fusion_large,
    "memory_dense": scenario_memory_dense,
    "memory_chunked": scenario_memory_chunked,
}


def _backend_status() -> dict[str, object]:
    geometry = app.get_default_geometry_backend().status
    tpms = app.get_default_tpms_backend().status
    return {
        "geometry": dict(geometry.__dict__),
        "tpms": dict(tpms.__dict__),
    }


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", required=True, choices=tuple(SCENARIOS))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--custom-cell", type=Path)
    parser.add_argument("--domain-stl", type=Path)
    args = parser.parse_args()
    for value, label in (
        (args.custom_cell, "custom-cell STL"),
        (args.domain_stl, "design-domain STL"),
    ):
        if value is not None and not value.is_file():
            parser.error(f"{label} does not exist: {value}")

    started = time.perf_counter()
    process = psutil.Process()
    baseline_rss = process.memory_info().rss
    payload: dict[str, object] = {
        "scenario": args.scenario,
        "pid": os.getpid(),
        "status": "failed",
        "started_at_epoch_s": time.time(),
    }
    try:
        with tempfile.TemporaryDirectory(prefix=f"shoe-stress-{args.scenario}-") as temp:
            details = SCENARIOS[args.scenario](args, Path(temp))
        payload.update(
            {
                "status": "passed",
                "details": details,
                "backends": _backend_status(),
            }
        )
        return_code = 0
    except BaseException as exc:
        payload.update(
            {
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
        )
        return_code = 1
    finally:
        payload["elapsed_s"] = time.perf_counter() - started
        payload["worker_rss_delta_mib"] = (
            process.memory_info().rss - baseline_rss
        ) / 1024**2
        _write_json(args.output, payload)
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
