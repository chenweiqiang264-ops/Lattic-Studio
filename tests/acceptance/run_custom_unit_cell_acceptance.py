"""Run the real custom-unit-cell acceptance on the example sole domain."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pyvista as pv
import trimesh


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lattice_studio.presentation.qt import workbench as app
from lattice_studio.engine.implicit.custom_unit_cell import prepare_stl_unit_cell


DEFAULT_UNIT_CELL = (
    PROJECT_ROOT
    / "resources"
    / "examples"
    / "unit_cells"
    / "custom-cell.stl"
)
CELL_SIZE_MM = (12.0, 12.0, 8.0)
DISPLAY_VOXEL_MM = 0.8
EXPORT_TOLERANCE_MM = 0.45
CHUNK_COUNT = 12


def _polydata(mesh: trimesh.Trimesh) -> pv.PolyData:
    faces = np.column_stack(
        (np.full(len(mesh.faces), 3, dtype=np.int32), mesh.faces.astype(np.int32))
    ).ravel()
    return pv.PolyData(np.asarray(mesh.vertices, dtype=np.float64), faces)


def _wireframe(cell_map) -> pv.PolyData:
    segments = cell_map.wireframe_segments()
    points = np.ascontiguousarray(segments.reshape((-1, 3)), dtype=np.float64)
    lines = np.column_stack(
        (
            np.full(len(segments), 2, dtype=np.int32),
            np.arange(len(points), dtype=np.int32).reshape((-1, 2)),
        )
    ).ravel()
    return pv.PolyData(points, lines=lines)


def _new_plotter() -> pv.Plotter:
    plotter = pv.Plotter(off_screen=True, window_size=(1400, 900))
    plotter.set_background("white")
    plotter.enable_anti_aliasing("ssaa")
    return plotter


def _finish_render(plotter: pv.Plotter, path: Path) -> None:
    plotter.enable_parallel_projection()
    plotter.camera_position = "iso"
    plotter.reset_camera()
    plotter.show(screenshot=str(path), auto_close=True)


def _render_source(mesh: trimesh.Trimesh, path: Path) -> None:
    plotter = _new_plotter()
    plotter.add_mesh(
        _polydata(mesh),
        color="#8C939B",
        pbr=True,
        metallic=0.02,
        roughness=0.55,
        smooth_shading=True,
    )
    plotter.add_axes()
    _finish_render(plotter, path)


def _render_cell_map(
    domain: trimesh.Trimesh,
    cell_map,
    path: Path,
) -> None:
    plotter = _new_plotter()
    plotter.add_mesh(
        _polydata(domain),
        color="#B1B5BA",
        opacity=1.0,
        smooth_shading=True,
    )
    plotter.add_mesh(
        _wireframe(cell_map),
        color="#69727D",
        line_width=1.2,
        lighting=False,
    )
    plotter.add_axes()
    _finish_render(plotter, path)


def _render_result(mesh: trimesh.Trimesh, path: Path) -> None:
    plotter = _new_plotter()
    plotter.add_mesh(
        _polydata(mesh),
        color="#858C94",
        pbr=True,
        metallic=0.03,
        roughness=0.52,
        smooth_shading=True,
    )
    plotter.add_axes()
    _finish_render(plotter, path)


def _progress(message: str, value: float) -> None:
    print(f"[{value:6.1%}] {message}", flush=True)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--unit-cell", type=Path, default=DEFAULT_UNIT_CELL)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "build" / "test-artifacts" / "custom-unit-cell",
    )
    return parser.parse_args()


def main() -> int:
    arguments = _parse_args()
    output = arguments.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    domain_path = (
        PROJECT_ROOT / "resources" / "examples" / "design_domains" / "sole-1.stl"
    )
    domain = trimesh.load(domain_path, force="mesh", process=True)
    if not isinstance(domain, trimesh.Trimesh) or not domain.is_watertight:
        raise RuntimeError("acceptance design domain must be a watertight mesh")

    timings = {}
    started = time.perf_counter()
    prepared = prepare_stl_unit_cell(arguments.unit_cell)
    timings["prepare_unit_cell_seconds"] = time.perf_counter() - started
    parameters = app.CustomUnitCellParameters(
        source_path=arguments.unit_cell,
        cell_size_mm=CELL_SIZE_MM,
        cell_map_mode="fit_bounds",
    )
    sampling = app.SamplingParameters(
        target_voxels=2_000_000,
        processing_mode="chunked_marching_cubes",
        batch_count=CHUNK_COUNT,
    )

    started = time.perf_counter()
    generation = app.generate_implicit_custom_lattice(
        domain,
        parameters,
        sampling,
        voxel_size_mm=DISPLAY_VOXEL_MM,
        display_voxel_size_mm=DISPLAY_VOXEL_MM,
        display_memory_budget_mb=256.0,
        progress=_progress,
        prepared_unit_cell=prepared,
    )
    timings["implicit_generation_seconds"] = time.perf_counter() - started
    if not np.any(generation.display_field.values <= 0.0):
        raise RuntimeError("custom-cell implicit display field has no interior")

    started = time.perf_counter()
    reconstruction = app.reconstruct_implicit_mesh(
        generation.body,
        generation.cell_map,
        tolerance_mm=EXPORT_TOLERANCE_MM,
        wall_thickness_mm=float(generation.minimum_feature_mm),
        clean_numerical_fragments=True,
        progress=_progress,
        processing_mode="chunked_marching_cubes",
        batch_count=CHUNK_COUNT,
    )
    timings["stl_reconstruction_seconds"] = time.perf_counter() - started
    if len(reconstruction.mesh.faces) == 0:
        raise RuntimeError("custom-cell STL reconstruction is empty")

    stl_path = output / "1_custom_unit_cell_lattice.stl"
    reconstruction.mesh.export(stl_path)
    reloaded = trimesh.load(stl_path, force="mesh", process=True)
    quality = app.inspect_mesh(reloaded)
    component_labels = trimesh.graph.connected_component_labels(
        reloaded.face_adjacency,
        node_count=len(reloaded.faces),
    )
    component_face_counts = sorted(
        np.bincount(component_labels).astype(int).tolist(),
        reverse=True,
    )
    source_mesh = trimesh.load(arguments.unit_cell, force="mesh", process=True)
    _render_source(source_mesh, output / "source_unit_cell.png")
    _render_cell_map(domain, generation.cell_map, output / "custom_cell_map.png")
    _render_result(reloaded, output / "1_custom_unit_cell_lattice.png")

    import_report = prepared.report
    report = {
        "design_domain": str(domain_path),
        "unit_cell": str(arguments.unit_cell.resolve()),
        "parameters": {
            "target_cell_size_mm": CELL_SIZE_MM,
            "actual_cell_size_mm": generation.cell_map.spacing_mm,
            "boundary_mode": generation.cell_map.boundary_mode,
            "display_voxel_mm": DISPLAY_VOXEL_MM,
            "export_tolerance_mm": EXPORT_TOLERANCE_MM,
            "minimum_feature_mm": generation.minimum_feature_mm,
            "processing_mode": "chunked_marching_cubes",
            "chunk_count": CHUNK_COUNT,
        },
        "source_cleanup": {
            "original_vertices": import_report.original_vertices,
            "original_faces": import_report.original_faces,
            "prepared_vertices": import_report.prepared_vertices,
            "prepared_faces": import_report.prepared_faces,
            "removed_degenerate_faces": import_report.removed_degenerate_faces,
            "removed_duplicate_faces": import_report.removed_duplicate_faces,
            "watertight": import_report.watertight,
            "winding_consistent": import_report.winding_consistent,
            "connected_components": import_report.connected_components,
        },
        "cell_map": {
            "counts": generation.cell_map.cell_counts,
            "index_min": generation.cell_map.index_min,
            "bounds": generation.cell_map.bounds.tolist(),
        },
        "display_grid": {
            "shape": generation.display_field.values.shape,
            "estimated_bytes": generation.display_field.estimated_bytes,
        },
        "stl": {
            "vertices": len(reloaded.vertices),
            "faces": len(reloaded.faces),
            "watertight": quality.watertight,
            "winding_consistent": quality.winding_consistent,
            "connected_components": quality.connected_components,
            "component_face_counts": component_face_counts,
            "main_component_face_fraction": (
                component_face_counts[0] / len(reloaded.faces)
            ),
            "extraction_backend": reconstruction.extraction_backend,
            "spacing_mm": reconstruction.spacing_mm,
        },
        "backend": {
            "geometry": app._geometry_backend_label(),
        },
        "timings_seconds": timings,
        "artifacts": {
            "stl": str(stl_path),
            "source_image": str(output / "source_unit_cell.png"),
            "cell_map_image": str(output / "custom_cell_map.png"),
            "result_image": str(output / "1_custom_unit_cell_lattice.png"),
        },
    }
    report_path = output / "acceptance_report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
