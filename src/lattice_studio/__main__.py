"""Command-line entry point for Lattice Studio."""

from __future__ import annotations


def _print_acceleration_status() -> None:
    """Print the selected numerical backends before the Qt event loop starts."""

    try:
        from lattice_studio.infrastructure.native import (
            CPP_DISTANCE_FIELD_AVAILABLE,
            CPP_MASK_AVAILABLE,
            CPP_MESH_SDF_AVAILABLE,
            CPP_MESH_SLICER_AVAILABLE,
        )

        native_modules = {
            "cpp_distance_field": CPP_DISTANCE_FIELD_AVAILABLE,
            "cpp_mesh_sdf": CPP_MESH_SDF_AVAILABLE,
            "cpp_mesh_slicer": CPP_MESH_SLICER_AVAILABLE,
            "cpp_mask": CPP_MASK_AVAILABLE,
        }
        loaded = [name for name, available in native_modules.items() if available]
        missing = [name for name, available in native_modules.items() if not available]
        print(
            "[Acceleration] C++ native: "
            + (", ".join(loaded) if loaded else "unavailable")
            + (f"; fallback: {', '.join(missing)}" if missing else ""),
            flush=True,
        )
    except Exception as exc:  # pragma: no cover - defensive startup reporting
        print(f"[Acceleration] C++ native probe failed: {type(exc).__name__}: {exc}", flush=True)

    backends = (
        ("geometry/SDF", "lattice_studio.engine.implicit.geometry_compute", "get_default_geometry_backend"),
        ("TPMS", "lattice_studio.engine.implicit.tpms_compute", "get_default_tpms_backend"),
        ("resident lattice pipeline", "lattice_studio.engine.implicit.lattice_pipeline", "get_default_lattice_pipeline"),
    )
    for label, module_name, factory_name in backends:
        try:
            module = __import__(module_name, fromlist=[factory_name])
            status = getattr(module, factory_name)().status
            state = "GPU" if status.using_gpu else "CPU fallback"
            detail = f"; reason: {status.fallback_reason}" if status.fallback_reason else ""
            print(
                f"[Acceleration] {label}: {state} / {status.active_backend} "
                f"({status.device_name}){detail}",
                flush=True,
            )
        except Exception as exc:  # pragma: no cover - defensive startup reporting
            print(
                f"[Acceleration] {label}: unavailable; "
                f"{type(exc).__name__}: {exc}",
                flush=True,
            )


def main() -> int:
    import sys
    import multiprocessing

    multiprocessing.freeze_support()
    if len(sys.argv) >= 2 and sys.argv[1] == "--serve-backend":
        from lattice_studio.presentation.http.local_backend import main as run_backend

        return run_backend(sys.argv[2:])
    if len(sys.argv) == 3 and sys.argv[1] == "--cuda-probe-worker":
        from lattice_studio.engine.implicit.cuda_probe import run_probe_worker
        return run_probe_worker(sys.argv[2])
    if len(sys.argv) == 3 and sys.argv[1] == "--verify-runtime":
        from lattice_studio.infrastructure.release_check import verify_runtime
        return verify_runtime(sys.argv[2])
    _print_acceleration_status()
    from lattice_studio.presentation.qt.workbench import main as run_workbench

    return int(run_workbench())


if __name__ == "__main__":
    raise SystemExit(main())
