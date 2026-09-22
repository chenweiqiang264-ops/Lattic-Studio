"""PyInstaller runtime hook for the optional bundled CUDA toolkit."""

from lattice_studio.infrastructure.gpu_runtime import configure_bundled_cuda_runtime

bundled_cuda = configure_bundled_cuda_runtime()
if bundled_cuda is not None:
    # Make the bundled NVVM/libdevice location visible before any Numba CUDA
    # module is imported by the application.
    print(f"[Acceleration] bundled CUDA runtime: {bundled_cuda}", flush=True)
