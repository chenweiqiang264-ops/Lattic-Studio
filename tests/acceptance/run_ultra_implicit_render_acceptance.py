"""Compare normal and evaluator-refined ultra implicit rendering on 1.stl."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import sys
import time

import numpy as np
import trimesh


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lattice_studio.presentation.qt import workbench as app
from lattice_studio.presentation.qt.viewers.pyvista_viewer import PyVistaRenderer


def _render(field: app.SampledImplicitField, output_path: Path) -> None:
    from PyQt5 import QtWidgets

    qt_app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    viewer = PyVistaRenderer()
    try:
        viewer.set_render_quality("ultra")
        viewer.set_background("#F5F7FA", top="#FFFFFF")
        display_field = replace(
            field,
            color=(0.68, 0.70, 0.74, 1.0),
            metallic=0.08,
            roughness=0.52,
        )
        viewer.set_implicit_fields([display_field], reset_view=True)
        viewer.camera_position = "iso"
        viewer.reset_view()
        viewer.camera.zoom(5.0)
        viewer.render()
        viewer.screenshot(str(output_path))
    finally:
        viewer.close()
        qt_app.processEvents()


def _field_error_against_evaluator(
    body: app.ImplicitBody,
    normal: app.SampledImplicitField,
    high: app.SampledImplicitField,
    refined: app.SampledImplicitField,
) -> dict[str, float | int]:
    from scipy.interpolate import RegularGridInterpolator

    lower = np.maximum(normal.bounds[0], refined.bounds[0])
    upper = np.minimum(normal.bounds[1], refined.bounds[1])
    rng = np.random.default_rng(20260915)
    points = rng.uniform(lower, upper, size=(60_000, 3))
    truth = body.evaluate_points(points)
    near_surface = np.abs(truth) <= 2.0 * float(np.max(normal.spacing))
    if np.count_nonzero(near_surface) < 1_000:
        raise RuntimeError("acceptance sample did not cover enough near-surface points")
    points = points[near_surface]
    truth = truth[near_surface]

    def interpolate(field: app.SampledImplicitField) -> np.ndarray:
        axes = tuple(
            field.origin[axis]
            + np.arange(field.values.shape[axis], dtype=np.float64)
            * field.spacing[axis]
            for axis in range(3)
        )
        return RegularGridInterpolator(
            axes,
            field.values,
            method="linear",
            bounds_error=True,
        )(points)

    normal_error = interpolate(normal) - truth
    high_error = interpolate(high) - truth
    refined_error = interpolate(refined) - truth
    normal_rmse = float(np.sqrt(np.mean(normal_error * normal_error)))
    high_rmse = float(np.sqrt(np.mean(high_error * high_error)))
    refined_rmse = float(np.sqrt(np.mean(refined_error * refined_error)))
    if not normal_rmse > high_rmse > refined_rmse:
        raise RuntimeError(
            "display-field error did not decrease from normal through High to Ultra"
        )
    return {
        "near_surface_samples": int(len(points)),
        "normal_rmse": normal_rmse,
        "high_rmse": high_rmse,
        "ultra_rmse": refined_rmse,
        "high_rmse_reduction_percent": 100.0 * (1.0 - high_rmse / normal_rmse),
        "rmse_reduction_percent": 100.0 * (1.0 - refined_rmse / normal_rmse),
    }


def main() -> int:
    output_dir = PROJECT_ROOT / "build" / "test-artifacts" / "ultra-render"
    output_dir.mkdir(parents=True, exist_ok=True)
    source_path = app._default_sole_path(PROJECT_ROOT)
    domain = trimesh.load(source_path, force="mesh", process=True)
    parameters = app.TPMSParameters(
        "G",
        cell_size_mm=(10.0, 10.0, 10.0),
        wall_thickness_mm=1.0,
        cell_map_mode="complete_cells",
    )
    sampling = app.SamplingParameters(
        target_voxels=1_000_000,
        use_cpp_sdf=True,
        processing_mode="single_pass",
        batch_count=1,
    )

    def progress(message: str, value: float) -> None:
        print(f"{value:6.1%} {message}", flush=True)

    start = time.perf_counter()
    generation = app.generate_implicit_lattice(
        domain,
        parameters,
        sampling,
        display_voxel_size_mm=0.70,
        display_memory_budget_mb=384.0,
        display_batch_count=4,
        progress=progress,
    )
    generation_seconds = time.perf_counter() - start

    start = time.perf_counter()
    high, high_refinement = app.refine_implicit_display_field(
        generation.body,
        generation.display_field,
        quality="high",
        display_memory_budget_mb=192.0,
        display_batch_count=4,
        progress=progress,
    )
    high_refinement_seconds = time.perf_counter() - start

    start = time.perf_counter()
    refined, refinement = app.refine_implicit_display_field(
        generation.body,
        generation.display_field,
        quality="ultra",
        display_memory_budget_mb=192.0,
        display_batch_count=4,
        progress=progress,
    )
    refinement_seconds = time.perf_counter() - start
    accuracy = _field_error_against_evaluator(
        generation.body,
        generation.display_field,
        high,
        refined,
    )

    normal_path = output_dir / "1stl_G_ultra_ray_only.png"
    high_path = output_dir / "1stl_G_high_refined_field.png"
    refined_path = output_dir / "1stl_G_ultra_refined_field.png"
    _render(generation.display_field, normal_path)
    _render(high, high_path)
    _render(refined, refined_path)

    report = {
        "source": str(source_path),
        "parameters": {
            "cell_size_mm": parameters.cell_size_xyz_mm,
            "wall_thickness_mm": parameters.wall_thickness_mm,
        },
        "normal_display_field": {
            "spacing_mm": generation.display_field.spacing.tolist(),
            "shape": generation.display_field.values.shape,
            "voxels": generation.display_field.values.size,
            "memory_mb": generation.display_field.estimated_bytes / 1024**2,
        },
        "high_display_field": {
            "requested_spacing_mm": high_refinement.requested_spacing_mm,
            "applied_spacing_mm": high_refinement.applied_spacing_mm,
            "shape": high.values.shape,
            "voxels": high.values.size,
            "memory_mb": high.estimated_bytes / 1024**2,
            "refined": high_refinement.refined,
            "budget_limited": high_refinement.budget_limited,
        },
        "ultra_display_field": {
            "requested_spacing_mm": refinement.requested_spacing_mm,
            "applied_spacing_mm": refinement.applied_spacing_mm,
            "shape": refined.values.shape,
            "voxels": refined.values.size,
            "memory_mb": refined.estimated_bytes / 1024**2,
            "refined": refinement.refined,
            "budget_limited": refinement.budget_limited,
        },
        "timing_seconds": {
            "initial_generation": generation_seconds,
            "high_refinement": high_refinement_seconds,
            "ultra_refinement": refinement_seconds,
        },
        "evaluator_accuracy": accuracy,
        "artifacts": {
            "ray_step_only": str(normal_path),
            "high_evaluator_refined": str(high_path),
            "evaluator_refined": str(refined_path),
        },
    }
    report_path = output_dir / "report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
