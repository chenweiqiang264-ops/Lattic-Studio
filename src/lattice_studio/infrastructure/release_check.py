"""Small, machine-readable acceptance check run by the installed executable."""

from __future__ import annotations

import importlib
import json
from pathlib import Path
import sys
import traceback
from dataclasses import asdict


def verify_runtime(report_path: str) -> int:
    report = {"ok": False, "executable": sys.executable, "checks": {}}
    checks = report["checks"]
    try:
        for name in ("_ctypes", "_ssl", "_bz2", "_lzma", "_sqlite3", "pyexpat"):
            module = importlib.import_module(name)
            checks[name] = str(module.__file__)
        import numpy as np
        import trimesh
        from skimage.measure import marching_cubes
        from lattice_studio.infrastructure import native
        from lattice_studio.engine.implicit.tpms_compute import (
            get_default_tpms_backend, NumpyTPMSAdapter, TPMSFieldSpec,
        )

        checks["native"] = {name: bool(getattr(native, name)) for name in native.__all__}
        if not all(checks["native"].values()):
            raise RuntimeError("A required C++ extension failed to load")
        # Offset the grid from analytical critical points where gradient-based
        # distance normalization is singular (not part of the zero surface).
        axis = np.linspace(-6, 6, 33, dtype=np.float32) + 0.137
        points = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1).reshape(-1, 3)
        backend = get_default_tpms_backend()
        spec = TPMSFieldSpec("G", 6.0, 0.8)
        values = backend.evaluate_shell(points, spec)
        expected = NumpyTPMSAdapter().evaluate_shell(points, spec, None)
        np.testing.assert_allclose(values, expected, atol=2e-4, rtol=2e-4)
        checks["tpms"] = asdict(backend.status)
        checks["sample_count"] = len(values)
        vertices, faces, _, _ = marching_cubes(values.reshape(33, 33, 33), 0, spacing=(0.375,) * 3)
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        export_path = Path(report_path).with_suffix(".stl")
        mesh.export(export_path)
        loaded = trimesh.load(export_path, force="mesh")
        if not len(loaded.faces):
            raise RuntimeError("STL roundtrip produced an empty mesh")
        checks["stl_faces"] = len(loaded.faces)
        # Exercise the actual installed Qt/VTK window and OpenGL context.
        from PyQt5 import QtWidgets
        from lattice_studio.presentation.qt.workbench import _build_qt_app
        app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        window = _build_qt_app()()
        window.show()
        app.processEvents()
        checks["main_window"] = window.isVisible()
        window.close()
        app.processEvents()
        checks["window_closed"] = not window.isVisible()
        if not checks["main_window"] or not checks["window_closed"]:
            raise RuntimeError("Workbench did not open and close successfully")
        report["ok"] = True
    except Exception:
        report["error"] = traceback.format_exc()
    Path(report_path).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0 if report["ok"] else 1
