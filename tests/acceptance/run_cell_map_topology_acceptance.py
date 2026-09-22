"""Generate repeatable Cell Map topology acceptance artifacts from 1.stl."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pyvista as pv
import trimesh

TESTS_DIR = Path(__file__).resolve().parent
if str(TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(TESTS_DIR))

from lattice_studio.presentation.qt import workbench as app


PARAMETERS = {
    "G": app.TPMSParameters("G", (7.0, 15.0, 4.5), 1.2, 0.0),
    "D": app.TPMSParameters("D", (16.0, 8.0, 6.0), 1.3, 0.0),
}
EXPORT_TOLERANCE_MM = 0.45
REPAIR_TOLERANCE_MM = 0.45


def _mesh_polydata(mesh: trimesh.Trimesh) -> pv.PolyData:
    faces = np.column_stack(
        (np.full(len(mesh.faces), 3, dtype=np.int32), mesh.faces.astype(np.int32))
    ).ravel()
    return pv.PolyData(np.asarray(mesh.vertices, dtype=np.float64), faces)


def _cell_map_wireframe(cell_map) -> pv.PolyData:
    axes = cell_map.axes()
    points = []
    lines = []

    def add_line(first, second):
        start = len(points)
        points.extend((first, second))
        lines.append((2, start, start + 1))

    for x in axes[0]:
        for y in axes[1]:
            add_line((x, y, axes[2][0]), (x, y, axes[2][-1]))
    for x in axes[0]:
        for z in axes[2]:
            add_line((x, axes[1][0], z), (x, axes[1][-1], z))
    for y in axes[1]:
        for z in axes[2]:
            add_line((axes[0][0], y, z), (axes[0][-1], y, z))
    return pv.PolyData(
        np.asarray(points, dtype=np.float64),
        lines=np.asarray(lines, dtype=np.int64).ravel(),
    )


def _render_result(mesh: trimesh.Trimesh, path: Path) -> None:
    plotter = pv.Plotter(off_screen=True, window_size=(1280, 800))
    plotter.set_background("white")
    plotter.add_mesh(
        _mesh_polydata(mesh),
        color=(0.50, 0.52, 0.56),
        pbr=True,
        metallic=0.0,
        roughness=0.56,
        smooth_shading=True,
    )
    plotter.add_axes(
        x_color="#D33F49",
        y_color="#1B9E62",
        z_color="#2563EB",
    )
    plotter.enable_parallel_projection()
    plotter.camera_position = "iso"
    plotter.reset_camera()
    plotter.screenshot(path)
    plotter.close()


def _render_cell_map(domain: trimesh.Trimesh, cell_map, path: Path) -> None:
    plotter = pv.Plotter(off_screen=True, window_size=(1280, 800))
    plotter.set_background("white")
    plotter.add_mesh(
        _mesh_polydata(domain),
        color=(0.68, 0.70, 0.73),
        opacity=1.0,
        smooth_shading=True,
    )
    plotter.add_mesh(
        _cell_map_wireframe(cell_map),
        color="#9AA2AC",
        line_width=1.0,
        lighting=False,
    )
    plotter.add_axes(
        x_color="#D33F49",
        y_color="#1B9E62",
        z_color="#2563EB",
    )
    plotter.enable_parallel_projection()
    plotter.view_xy()
    plotter.reset_camera()
    plotter.screenshot(path)
    plotter.close()


def main() -> int:
    sole_path = app._default_sole_path(app.PROJECT_ROOT)
    domain = trimesh.load(sole_path, force="mesh", process=True)
    sampling = app.SamplingParameters(
        target_voxels=4_000_000,
        use_cpp_sdf=True,
        processing_mode="single_pass",
        batch_count=1,
    )
    output_dir = app.PROJECT_ROOT / "build" / "test-artifacts" / "cell-map-topology"
    output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "source": str(sole_path),
        "export_tolerance_mm": EXPORT_TOLERANCE_MM,
        "repair_tolerance_mm": REPAIR_TOLERANCE_MM,
        "results": {},
    }
    for kind, parameters in PARAMETERS.items():
        cell_map = app.cell_map_for_parameters(domain, parameters)
        body = app._make_lattice_body(domain, parameters, sampling, cell_map=cell_map)
        result = app.reconstruct_implicit_mesh(
            body,
            cell_map,
            EXPORT_TOLERANCE_MM,
            parameters.wall_thickness_mm,
            repair_tolerance_mm=REPAIR_TOLERANCE_MM,
            clean_numerical_fragments=True,
            optimize_for_slicing=True,
            progress=lambda message, value, label=kind: print(
                f"[{label} {value:6.1%}] {message}", flush=True
            ),
        )
        stl_path = output_dir / f"1_{kind}_cell_map_topology.stl"
        result.mesh.export(stl_path)
        reloaded = trimesh.load(stl_path, force="mesh", process=True)
        quality = app.inspect_mesh(reloaded)
        image_path = output_dir / f"1_{kind}_cell_map_topology.png"
        cell_map_path = output_dir / f"1_{kind}_cell_map.png"
        _render_result(reloaded, image_path)
        _render_cell_map(domain, cell_map, cell_map_path)
        report["results"][kind] = {
            "parameters": {
                "target_cell_spacing_mm": parameters.cell_size_xyz_mm,
                "actual_cell_spacing_mm": cell_map.spacing_mm,
                "cell_map_boundary_mode": cell_map.boundary_mode,
                "wall_thickness_mm": parameters.wall_thickness_mm,
            },
            "cell_counts": cell_map.cell_counts,
            "cell_map_bounds": cell_map.bounds.tolist(),
            "sampling_spacing_mm": result.spacing_mm,
            "conditioning": (
                None
                if result.conditioning is None
                else {
                    "attempted": result.conditioning.attempted,
                    "applied": result.conditioning.applied,
                    "backend": result.conditioning.backend,
                    "tolerance_mm": result.conditioning.tolerance_mm,
                    "removed_vertices": result.conditioning.removed_vertices,
                    "removed_faces": result.conditioning.removed_faces,
                    "topology_preserved": result.conditioning.topology_preserved,
                    "bounds_deviation_mm": result.conditioning.bounds_deviation_mm,
                    "message": result.conditioning.message,
                }
            ),
            "field_repaired": result.field_repaired,
            "field_fragment_cleanup": (
                None
                if result.field_fragment_cleanup is None
                else {
                    "removed_components": result.field_fragment_cleanup.removed_components,
                    "removed_voxels": result.field_fragment_cleanup.removed_voxels,
                    "removed_volume_mm3": result.field_fragment_cleanup.removed_volume_mm3,
                    "retained_components": result.field_fragment_cleanup.retained_components,
                }
            ),
            "numerical_fragment_cleanup": (
                None
                if result.fragment_cleanup is None
                else {
                    "removed_components": result.fragment_cleanup.removed_components,
                    "removed_faces": result.fragment_cleanup.removed_faces,
                    "removed_volume_mm3": result.fragment_cleanup.removed_volume_mm3,
                    "volume_limit_mm3": result.fragment_cleanup.volume_limit_mm3,
                    "relative_volume_limit": result.fragment_cleanup.relative_volume_limit,
                }
            ),
            "vertices": len(reloaded.vertices),
            "faces": len(reloaded.faces),
            "watertight": quality.watertight,
            "winding_consistent": quality.winding_consistent,
            "connected_components": quality.connected_components,
            "stl": str(stl_path),
            "image": str(image_path),
            "cell_map_image": str(cell_map_path),
        }
        print(json.dumps(report["results"][kind], ensure_ascii=False, indent=2))
    report_path = output_dir / "acceptance_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
