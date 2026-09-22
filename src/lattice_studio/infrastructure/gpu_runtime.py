"""Locate optional CUDA runtime files in source and PyInstaller layouts."""

from __future__ import annotations

import os
import sys
from pathlib import Path


def _bundle_roots() -> tuple[Path, ...]:
    roots: list[Path] = []
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        roots.append(Path(meipass))
    executable = Path(sys.executable).resolve()
    roots.extend(
        (
            executable.parent,
            executable.parent / "_internal",
            Path(__file__).resolve().parents[3],
        )
    )
    unique: list[Path] = []
    for root in roots:
        resolved = root.resolve()
        if resolved not in unique:
            unique.append(resolved)
    return tuple(unique)


def configure_bundled_cuda_runtime() -> Path | None:
    """Expose a bundled CUDA toolkit to Numba before CUDA is imported."""

    candidates: list[Path] = []
    for root in _bundle_roots():
        candidates.extend((root / "cuda", root / "runtime" / "cuda"))
    configured = os.environ.get("CUDA_PATH")
    if configured:
        candidates.insert(0, Path(configured))

    for candidate in candidates:
        nvvm = candidate / "nvvm"
        if not nvvm.is_dir():
            continue
        bin_dirs = [candidate / "bin", nvvm / "bin"]
        existing = [str(path) for path in bin_dirs if path.is_dir()]
        if existing:
            current_path = os.environ.get("PATH", "")
            os.environ["PATH"] = os.pathsep.join(
                existing + ([current_path] if current_path else [])
            )
        os.environ.setdefault("CUDA_PATH", str(candidate))
        return candidate
    return None


__all__ = ["configure_bundled_cuda_runtime"]
