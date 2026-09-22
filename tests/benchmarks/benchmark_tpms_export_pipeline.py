"""Benchmark the authoritative TPMS export and derived-mesh display pipeline."""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import psutil
import trimesh


TESTS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = TESTS_DIR.parents[1]
if str(TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(TESTS_DIR))

from lattice_studio.presentation.qt import workbench as app


PARAMETERS = app.TPMSParameters("G", (7.0, 15.0, 4.5), 1.2, 0.0)
EXPORT_TOLERANCE_MM = 0.45
REPAIR_TOLERANCE_MM = 0.45
PREVIOUS_BASELINE = {
    "measured_on": "2026-08-06",
    "field_sampling_s": 3.79,
    "marching_cubes_s": 15.44,
    "reconstruction_total_s": 65.41,
    "stl_write_s": 0.31,
    "stl_reload_s": 5.52,
    "reload_topology_s": 5.43,
    "pipeline_total_s": 76.66,
    "peak_rss_gib": 4.18,
    "faces": 2_047_280,
}


def _load_renderer_module():
    from lattice_studio.presentation.qt.viewers import pyvista_viewer

    return pyvista_viewer


class _PeakRssSampler:
    def __init__(self) -> None:
        self._process = psutil.Process()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._sample, daemon=True)
        self.peak_bytes = self._process.memory_info().rss

    def _sample(self) -> None:
        while not self._stop.wait(0.02):
            self.peak_bytes = max(
                self.peak_bytes,
                self._process.memory_info().rss,
            )

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *_args) -> None:
        self._stop.set()
        self._thread.join()
        self.peak_bytes = max(self.peak_bytes, self._process.memory_info().rss)


def _install_timers(names: tuple[str, ...]):
    elapsed = defaultdict(float)
    calls = defaultdict(int)
    originals = {}
    for name in names:
        original = getattr(app, name)
        originals[name] = original

        def wrapper(*args, _name=name, _original=original, **kwargs):
            started = time.perf_counter()
            try:
                return _original(*args, **kwargs)
            finally:
                elapsed[_name] += time.perf_counter() - started
                calls[_name] += 1

        setattr(app, name, wrapper)
    return elapsed, calls, originals


def _restore_timers(originals) -> None:
    for name, original in originals.items():
        setattr(app, name, original)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--disable-slicing-optimization",
        action="store_true",
        help="run the same export benchmark without topology-preserving slicer optimization",
    )
    arguments = parser.parse_args()
    output_dir = PROJECT_ROOT / "build" / "test-artifacts" / "tpms-export-benchmark"
    output_dir.mkdir(parents=True, exist_ok=True)
    domain_path = app._default_sole_path(PROJECT_ROOT)
    domain = trimesh.load(domain_path, force="mesh", process=True)
    sampling = app.SamplingParameters(
        target_voxels=4_000_000,
        use_cpp_sdf=True,
        processing_mode="single_pass",
        batch_count=1,
    )
    cell_map = app.cell_map_for_parameters(domain, PARAMETERS)
    body = app._make_lattice_body(domain, PARAMETERS, sampling, cell_map=cell_map)
    timer_names = (
        "_sample_grid_points",
        "_marching_cubes_field",
        "repair_mesh",
        "remove_numerical_fragments",
        "inspect_mesh",
    )
    elapsed, calls, originals = _install_timers(timer_names)
    started = time.perf_counter()
    try:
        with _PeakRssSampler() as memory:
            result = app.reconstruct_implicit_mesh(
                body,
                cell_map,
                EXPORT_TOLERANCE_MM,
                PARAMETERS.wall_thickness_mm,
                repair_tolerance_mm=REPAIR_TOLERANCE_MM,
                clean_numerical_fragments=True,
                optimize_for_slicing=not arguments.disable_slicing_optimization,
                progress=lambda message, value: print(
                    f"[{value:6.1%}] {message}",
                    flush=True,
                ),
            )
        reconstruction_s = time.perf_counter() - started
    finally:
        _restore_timers(originals)

    variant = (
        "unoptimized"
        if arguments.disable_slicing_optimization
        else "optimized"
    )
    stl_path = output_dir / f"1_G_{variant}.stl"
    started = time.perf_counter()
    result.mesh.export(stl_path)
    stl_write_s = time.perf_counter() - started
    started = time.perf_counter()
    reloaded = trimesh.load(stl_path, force="mesh", process=True)
    stl_reload_s = time.perf_counter() - started
    started = time.perf_counter()
    reloaded_quality = app.inspect_mesh(reloaded)
    reload_topology_s = time.perf_counter() - started

    renderer = _load_renderer_module()
    display_data = renderer.MeshData(
        result.mesh.vertices,
        result.mesh.faces,
        cache_key=("G", "benchmark"),
    )
    display_harness = SimpleNamespace(_mesh_display_cache={})
    display_face_limit = renderer.mesh_display_face_limit(
        len(result.mesh.faces),
        "medium",
    )
    started = time.perf_counter()
    proxy = renderer.PyVistaRenderer._display_polydata(
        display_harness,
        display_data,
        display_face_limit,
    )
    display_proxy_first_s = time.perf_counter() - started
    started = time.perf_counter()
    cached_proxy = renderer.PyVistaRenderer._display_polydata(
        display_harness,
        display_data,
        display_face_limit,
    )
    display_proxy_cached_s = time.perf_counter() - started
    from vtkmodules.vtkCommonDataModel import vtkPlanes

    plotter = renderer.pv.Plotter(off_screen=True, window_size=(900, 600))
    actor = plotter.add_mesh(proxy, smooth_shading=True)
    plotter.camera_position = "iso"
    plotter.show(auto_close=False, interactive=False)
    mapper = actor.GetMapper()
    proxy_bounds = np.asarray(proxy.bounds).reshape(3, 2)
    mapper_clip_samples = []
    for ratio in np.linspace(0.15, 0.85, 20):
        clip_bounds = proxy_bounds.copy()
        clip_bounds[0, 1] = proxy_bounds[0, 0] + ratio * (
            proxy_bounds[0, 1] - proxy_bounds[0, 0]
        )
        planes = vtkPlanes()
        planes.SetBounds(clip_bounds.ravel())
        started = time.perf_counter()
        mapper.SetClippingPlanes(planes)
        plotter.render()
        mapper_clip_samples.append(time.perf_counter() - started)
    plotter.close()

    geometry_status = app.get_default_geometry_backend().status
    tpms_status = app.get_default_tpms_backend().status
    report = {
        "source": str(domain_path),
        "parameters": {
            "cell_size_xyz_mm": PARAMETERS.cell_size_xyz_mm,
            "wall_thickness_mm": PARAMETERS.wall_thickness_mm,
            "export_tolerance_mm": EXPORT_TOLERANCE_MM,
            "repair_tolerance_mm": REPAIR_TOLERANCE_MM,
            "optimize_for_slicing": not arguments.disable_slicing_optimization,
            "cell_counts": cell_map.cell_counts,
            "actual_cell_spacing_mm": cell_map.spacing_mm,
        },
        "backends": {
            "geometry": geometry_status.__dict__,
            "tpms": tpms_status.__dict__,
        },
        "previous_baseline": PREVIOUS_BASELINE,
        "optimized": {
            "field_sampling_s": elapsed["_sample_grid_points"],
            "marching_cubes_s": elapsed["_marching_cubes_field"],
            "repair_s": elapsed["repair_mesh"],
            "fragment_cleanup_s": elapsed["remove_numerical_fragments"],
            "topology_s": elapsed["inspect_mesh"],
            "reconstruction_total_s": reconstruction_s,
            "stl_write_s": stl_write_s,
            "stl_reload_s": stl_reload_s,
            "reload_topology_s": reload_topology_s,
            "pipeline_total_s": (
                reconstruction_s + stl_write_s + stl_reload_s + reload_topology_s
            ),
            "peak_rss_gib": memory.peak_bytes / 1024**3,
            "instrumented_calls": dict(calls),
            "vertices": len(reloaded.vertices),
            "faces": len(reloaded.faces),
            "watertight": reloaded_quality.watertight,
            "winding_consistent": reloaded_quality.winding_consistent,
            "connected_components": reloaded_quality.connected_components,
            "display_proxy_first_s": display_proxy_first_s,
            "display_proxy_cached_s": display_proxy_cached_s,
            "display_proxy_faces": proxy.n_cells,
            "display_proxy_open_edges": proxy.n_open_edges,
            "display_cache_reused": proxy is cached_proxy,
            "display_mapper_clip_render_average_s": float(
                np.mean(mapper_clip_samples)
            ),
            "display_mapper_clip_render_max_s": float(
                np.max(mapper_clip_samples)
            ),
            "numerical_fragment_cleanup": (
                None
                if result.fragment_cleanup is None
                else result.fragment_cleanup.__dict__
            ),
            "conditioning": (
                None
                if result.conditioning is None
                else result.conditioning.__dict__
            ),
        },
        "artifacts": {"stl": str(stl_path)},
    }
    report_path = output_dir / f"benchmark_{variant}_report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
