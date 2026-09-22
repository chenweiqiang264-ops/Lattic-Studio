"""Persistence, native backends, mesh I/O, and distribution adapters.

Heavy adapters are loaded lazily. This package is imported by the isolated
CUDA probe, so eager persistence imports would create a cycle through the
geometry engine during backend startup.
"""

from importlib import import_module

__all__ = [
    "CURRENT_VERSION",
    "configure_bundled_cuda_runtime",
    "load_workspace",
    "save_workspace",
]


def __getattr__(name: str):
    if name == "configure_bundled_cuda_runtime":
        value = import_module("lattice_studio.infrastructure.gpu_runtime").configure_bundled_cuda_runtime
    elif name in {"CURRENT_VERSION", "load_workspace", "save_workspace"}:
        module = import_module("lattice_studio.infrastructure.workspace_persistence")
        value = getattr(module, name)
    else:
        raise AttributeError(name)
    globals()[name] = value
    return value
