"""Qt viewport implementations, loaded only when explicitly requested."""

from importlib import import_module

__all__ = [
    "GLMeshViewer",
    "MeshData",
    "OPEN3D_AVAILABLE",
    "Open3DRenderer",
    "PyVistaRenderer",
]


def __getattr__(name: str):
    if name in {"GLMeshViewer", "MeshData", "PyVistaRenderer"}:
        module = import_module(
            "lattice_studio.presentation.qt.viewers.pyvista_viewer"
        )
        value = module.PyVistaRenderer if name == "GLMeshViewer" else getattr(module, name)
    elif name in {"OPEN3D_AVAILABLE", "Open3DRenderer"}:
        module = import_module(
            "lattice_studio.presentation.qt.viewers.open3d_viewer"
        )
        value = getattr(module, name)
    else:
        raise AttributeError(name)
    globals()[name] = value
    return value
