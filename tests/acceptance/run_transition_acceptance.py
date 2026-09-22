"""Generate representative 1.stl artifacts for the generic transition pipeline."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pyvista as pv
import trimesh


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lattice_studio.presentation.qt import workbench as app


def _render(mesh: trimesh.Trimesh, path: Path) -> None:
    faces = np.column_stack(
        (np.full(len(mesh.faces), 3, dtype=np.int64), mesh.faces)
    ).ravel()
    surface = pv.PolyData(np.asarray(mesh.vertices), faces)
    plotter = pv.Plotter(off_screen=True, window_size=(1400, 900))
    plotter.set_background((0.98, 0.98, 0.98))
    plotter.add_mesh(
        surface,
        color=(0.54, 0.57, 0.62),
        pbr=True,
        metallic=0.08,
        roughness=0.42,
        smooth_shading=True,
    )
    for position, intensity in (
        ((1.5, -1.0, 2.0), 0.9),
        ((-1.0, 0.5, 1.2), 0.55),
        ((0.0, 1.5, 0.5), 0.35),
    ):
        light = pv.Light(position=position, focal_point=(0.0, 0.0, 0.0))
        light.intensity = intensity
        light.positional = False
        plotter.add_light(light)
    plotter.camera_position = "iso"
    plotter.reset_camera()
    plotter.show(screenshot=str(path), auto_close=True)


def main() -> int:
    output_dir = app.PROJECT_ROOT / "build" / "test-artifacts" / "transition"
    output_dir.mkdir(parents=True, exist_ok=True)
    print("验收：查找 1.stl", flush=True)
    source_path = app._default_sole_path(app.PROJECT_ROOT)
    print(f"验收：载入 {source_path}", flush=True)
    design_domain = trimesh.load(source_path, force="mesh", process=True)
    print(
        f"验收：设计域 {len(design_domain.vertices):,} 顶点 / "
        f"{len(design_domain.faces):,} 面",
        flush=True,
    )
    g_parameters = app.TPMSParameters(
        "G",
        cell_size_mm=(11.0, 13.0, 9.0),
        wall_thickness_mm=1.5,
        cell_map_mode="complete_cells",
    )
    d_parameters = app.TPMSParameters(
        "D",
        cell_size_mm=(16.0, 14.0, 10.0),
        wall_thickness_mm=1.8,
        level=0.08,
        cell_map_mode="complete_cells",
    )
    transition = app.TransitionParameters(
        plane_axis="Y",
        plane_position_mm=float(design_domain.bounds[:, 1].mean()),
        angle1_deg=8.0,
        angle2_deg=-6.0,
        transition_width_mm=0.5,
        weight_kind="smootherstep",
        minimum_feature_mm=1.5,
        automatic_registration=True,
        topology_correction=True,
    )
    sampling = app.SamplingParameters(
        target_voxels=4_000_000,
        processing_mode="chunked_marching_cubes",
        batch_count=6,
    )

    def progress(message: str, value: float) -> None:
        print(f"{value:6.1%} {message}", flush=True)

    generated = app.generate_implicit_lattice_transition(
        design_domain,
        g_parameters,
        d_parameters,
        transition,
        sampling,
        display_voxel_size_mm=1.0,
        display_memory_budget_mb=256.0,
        display_batch_count=4,
        progress=progress,
    )
    reconstructed = app.reconstruct_implicit_mesh(
        generated.body,
        generated.cell_map,
        tolerance_mm=0.55,
        wall_thickness_mm=generated.minimum_feature_mm,
        repair_tolerance_mm=0.0,
        clean_numerical_fragments=True,
        progress=progress,
        processing_mode="chunked_marching_cubes",
        batch_count=6,
        spacing_mode="exact",
    )
    stl_path = output_dir / "1stl_G_D_ramp_transition_0p5mm.stl"
    image_path = output_dir / "1stl_G_D_ramp_transition_0p5mm.png"
    report_path = output_dir / "1stl_G_D_ramp_transition_0p5mm.json"
    reconstructed.mesh.export(stl_path)
    _render(reconstructed.mesh, image_path)
    metadata = generated.metadata
    report = {
        "source": str(source_path),
        "parameters": {
            "first": "G 11x13x9 mm, wall 1.5 mm",
            "second": "D 16x14x10 mm, wall 1.8 mm, level 0.08",
            "transition_width_mm": 0.5,
            "weight": "smootherstep",
            "minimum_feature_mm": generated.minimum_feature_mm,
        },
        "transition_quality": {
            "passed": metadata.quality.passed,
            "cross_band_connected": metadata.quality.cross_band_connected,
            "minimum_feature_satisfied": metadata.quality.minimum_feature_satisfied,
            "isolated_fragment_count": metadata.quality.isolated_fragment_count,
            "component_count": metadata.quality.component_count,
            "inspection_spacing_mm": metadata.quality.inspection_spacing_mm,
            "issue_locations_mm": metadata.quality.issue_locations_mm,
        },
        "transition_build": {
            "recommended_width_mm": metadata.diagnostics.recommended_width_mm,
            "registration_translation_mm": metadata.diagnostics.registration_translation_mm,
            "execution_backend": metadata.diagnostics.execution_backend,
        },
        "stl": {
            "vertices": len(reconstructed.mesh.vertices),
            "faces": len(reconstructed.mesh.faces),
            "watertight": reconstructed.quality.watertight,
            "winding_consistent": reconstructed.quality.winding_consistent,
            "connected_components": reconstructed.quality.connected_components,
            "spacing_mm": reconstructed.spacing_mm,
            "backend": reconstructed.extraction_backend,
        },
        "artifacts": {
            "stl": str(stl_path),
            "image": str(image_path),
        },
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
