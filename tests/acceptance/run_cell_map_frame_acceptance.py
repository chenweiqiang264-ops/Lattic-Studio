"""Generate reproducible rotated Cell Map acceptance artifacts for 1.stl."""

from __future__ import annotations

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


def _line_polydata(cell_map) -> pv.PolyData:
    segments = cell_map.wireframe_segments()
    points = np.ascontiguousarray(segments.reshape((-1, 3)), dtype=np.float64)
    cells = np.column_stack(
        (
            np.full(len(segments), 2, dtype=np.int32),
            np.arange(len(points), dtype=np.int32).reshape((-1, 2)),
        )
    ).ravel()
    return pv.PolyData(points, lines=cells)


def _polydata(mesh: trimesh.Trimesh) -> pv.PolyData:
    faces = np.column_stack(
        (np.full(len(mesh.faces), 3, dtype=np.int32), mesh.faces.astype(np.int32))
    ).ravel()
    return pv.PolyData(np.asarray(mesh.vertices, dtype=np.float64), faces)


def _render_cell_map(path: Path, domain: trimesh.Trimesh, cell_map) -> None:
    plotter = pv.Plotter(off_screen=True, window_size=(1400, 900))
    plotter.set_background("white")
    plotter.add_mesh(
        _polydata(domain),
        color="#A7ADB5",
        opacity=0.82,
        smooth_shading=True,
    )
    plotter.add_mesh(_line_polydata(cell_map), color="#AAB2BD", line_width=1.0)
    arrow_length = max(
        float(np.min(cell_map.spacing_mm)) * 1.6,
        float(np.max(cell_map.extent_mm)) * 0.12,
    )
    for axis, color in (
        (cell_map.frame.u_axis, "#E85B65"),
        (cell_map.frame.v_axis, "#52C788"),
        (cell_map.frame.w_axis, "#5BA8E8"),
    ):
        plotter.add_mesh(
            pv.Arrow(
                start=cell_map.frame.origin,
                direction=axis,
                scale=arrow_length,
            ),
            color=color,
        )
    plotter.camera_position = "iso"
    plotter.reset_camera()
    plotter.show(screenshot=str(path), auto_close=True)


def _render_result(path: Path, mesh: trimesh.Trimesh) -> None:
    plotter = pv.Plotter(off_screen=True, window_size=(1400, 900))
    plotter.set_background("white")
    plotter.add_mesh(
        _polydata(mesh),
        color="#8E969F",
        pbr=True,
        metallic=0.05,
        roughness=0.58,
        smooth_shading=True,
    )
    plotter.camera_position = "iso"
    plotter.reset_camera()
    plotter.show(screenshot=str(path), auto_close=True)


def main() -> int:
    source = PROJECT_ROOT / "resources" / "examples" / "design_domains" / "sole-1.stl"
    output = PROJECT_ROOT / "build" / "test-artifacts" / "cell-map-frame"
    output.mkdir(parents=True, exist_ok=True)
    domain = trimesh.load(source, force="mesh")
    if not isinstance(domain, trimesh.Trimesh) or not domain.is_watertight:
        raise RuntimeError("acceptance design domain must be a watertight triangle mesh")

    g_angle = np.deg2rad(37.0)
    d_angle = np.deg2rad(-53.0)
    parameters = {
        "G": app.TPMSParameters(
            "G",
            (14.0, 9.0, 5.0),
            1.1,
            frame_origin_mm=tuple(domain.bounds[0]),
            frame_u_axis=(np.cos(g_angle), np.sin(g_angle), 0.0),
            frame_v_axis=(-np.sin(g_angle), np.cos(g_angle), 0.0),
        ),
        "D": app.TPMSParameters(
            "D",
            (7.0, 16.0, 6.0),
            1.0,
            frame_origin_mm=tuple(domain.centroid),
            frame_u_axis=(np.cos(d_angle), np.sin(d_angle), 0.0),
            frame_v_axis=(-np.sin(d_angle), np.cos(d_angle), 0.0),
        ),
    }
    sampling = app.SamplingParameters(
        target_voxels=2_000_000,
        processing_mode="batched_field",
        batch_count=4,
    )
    report = {}
    for kind, spec in parameters.items():
        cell_map = app.cell_map_for_parameters(domain, spec)
        if not np.all(cell_map.contains(domain.vertices)):
            raise RuntimeError(f"{kind} Cell Map does not cover the design domain")
        _render_cell_map(output / f"{kind.lower()}_cell_map.png", domain, cell_map)
        started = time.perf_counter()
        result, recommendation = app.generate_tpms_lattice(
            domain,
            spec,
            sampling,
            voxel_size_mm=0.7,
        )
        elapsed = time.perf_counter() - started
        if len(result.faces) == 0:
            raise RuntimeError(f"{kind} generation returned an empty mesh")
        result.export(output / f"{kind.lower()}_rotated_tpms.stl")
        _render_result(output / f"{kind.lower()}_rotated_tpms.png", result)
        quality = app.inspect_mesh(result)
        report[kind] = {
            "frame_origin_mm": [float(value) for value in cell_map.frame.origin],
            "frame_axes": cell_map.frame.axes.tolist(),
            "signed_index_min": list(cell_map.index_min),
            "cell_counts": list(cell_map.cell_counts),
            "world_bounds": cell_map.bounds.tolist(),
            "voxel_size_mm": recommendation.voxel_size_mm,
            "grid_shape": list(recommendation.grid_shape),
            "vertices": len(result.vertices),
            "faces": len(result.faces),
            "watertight": quality.watertight,
            "connected_components": quality.connected_components,
            "winding_consistent": quality.winding_consistent,
            "elapsed_seconds": elapsed,
        }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
