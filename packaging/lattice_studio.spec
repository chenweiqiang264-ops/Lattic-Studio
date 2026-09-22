# PyInstaller onedir build. Run from repository root:
#   python -m PyInstaller --clean --noconfirm packaging/lattice_studio.spec

import os
import sys
from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs, collect_submodules


PACKAGING_ROOT = Path(SPECPATH)
PROJECT_ROOT = PACKAGING_ROOT.parent
SRC = PROJECT_ROOT / "src"
datas = []
binaries = []
hiddenimports = [
    "lattice_studio",
    # These modules are loaded dynamically by Numba's CUDA dispatcher and
    # therefore are not all visible to PyInstaller's normal import graph.
    "numba.cuda",
    "numba.cuda.compiler",
    "numba.cuda.cuda_paths",
    "numba.cuda.cudadrv.driver",
    "numba.cuda.cudadrv.devicearray",
    "numba.cuda.cudadrv.devices",
    "numba.cuda.cudadrv.libs",
    "numba.cuda.cudadrv.nvvm",
]

# Use the maintained package hooks for Qt, VTK, NumPy, SciPy and Numba.  The
# previous collect_all() loop pulled every Numba/PyQt test module into the
# application and made the distribution unnecessarily large.
for package in ("numba", "llvmlite", "pyvista"):
    try:
        datas.extend(collect_data_files(package))
        binaries.extend(collect_dynamic_libs(package))
    except Exception:
        pass

for package in ("lattice_studio",):
    try:
        hiddenimports.extend(collect_submodules(package))
    except Exception:
        pass


def _python_runtime_roots() -> tuple[Path, ...]:
    """Return the runtime directories used by this Python installation.

    The project environment is a venv created from Anaconda.  Python's
    standard extension modules therefore come from ``base_prefix`` (for
    example ``_ctypes.pyd``), while their DLLs live in Anaconda's
    ``Library\\bin`` directory.  PyInstaller does not reliably infer those
    DLLs from the extension import graph, so the small set of non-system
    runtime libraries is collected explicitly below.
    """

    roots: list[Path] = []
    prefixes = (
        Path(sys.prefix),
        Path(getattr(sys, "base_prefix", sys.prefix)),
        PROJECT_ROOT / ".venv",
    )
    configured_prefix = os.environ.get("CONDA_PREFIX")
    if configured_prefix:
        prefixes += (Path(configured_prefix),)
    for prefix in prefixes:
        roots.extend((prefix / "DLLs", prefix / "Library" / "bin", prefix / "bin"))
    unique: list[Path] = []
    for root in roots:
        resolved = root.resolve()
        if resolved.is_dir() and resolved not in unique:
            unique.append(resolved)
    return tuple(unique)


def _collect_python_runtime_dlls() -> None:
    """Bundle non-system DLLs required by Python's standard extensions."""

    required = (
        "ffi.dll",              # _ctypes.pyd
        "libcrypto-3-x64.dll",  # _ssl.pyd
        "libssl-3-x64.dll",     # _ssl.pyd
        "libbz2.dll",           # _bz2.pyd (Anaconda uses this casing)
        "liblzma.dll",          # _lzma.pyd
        "libexpat.dll",         # pyexpat.pyd
        "sqlite3.dll",          # _sqlite3.pyd
    )
    roots = _python_runtime_roots()
    found_names: set[str] = set()
    for name in required:
        matches = [
            path
            for root in roots
            for path in root.iterdir()
            if path.is_file() and path.name.casefold() == name.casefold()
        ]
        if not matches:
            continue
        source = matches[0]
        destination_name = source.name.casefold()
        if destination_name in found_names:
            continue
        binaries.append((str(source), "."))
        found_names.add(destination_name)
    missing = [
        name
        for name in required
        if name.casefold() not in found_names
    ]
    if missing:
        if (Path(sys.base_prefix) / "conda-meta").is_dir():
            raise RuntimeError("Missing Conda runtime DLLs: " + ", ".join(missing))


_collect_python_runtime_dlls()

resources = PROJECT_ROOT / "resources"
if resources.is_dir():
    datas.append((str(resources), "resources"))


def _cuda_root() -> Path | None:
    """Find the CUDA Toolkit used to build this distribution."""

    candidates = []
    configured = os.environ.get("CUDA_PATH")
    if configured:
        candidates.append(Path(configured))
    candidates.extend(
        Path(path)
        for path in (
            r"C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8",
            r"C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.6",
            r"C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.4",
            r"C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.2",
        )
    )
    for candidate in candidates:
        if (candidate / "nvvm" / "libdevice").is_dir():
            return candidate
    return None


cuda_root = _cuda_root()
if cuda_root is not None:
    # Numba needs NVVM and libdevice to JIT compile kernels.  The driver DLL
    # (nvcuda.dll) is intentionally not bundled: it must come from the target
    # computer's installed NVIDIA driver.
    nvvm_bin = cuda_root / "nvvm" / "bin"
    for path in nvvm_bin.glob("*.dll"):
        binaries.append((str(path), "cuda/nvvm/bin"))
    for path in (cuda_root / "nvvm" / "libdevice").glob("*.bc"):
        datas.append((str(path), "cuda/nvvm/libdevice"))

    # Keep the optional CUDA libraries used by Numba discoverable without
    # copying the entire Toolkit (which would add several gigabytes).
    cuda_bin = cuda_root / "bin"
    for pattern in ("cudart*.dll", "nvrtc*.dll", "nvJitLink*.dll"):
        for path in cuda_bin.glob(pattern):
            binaries.append((str(path), "cuda/bin"))
else:
    print("WARNING: CUDA Toolkit not found; the package will use CPU fallback.")

design_tokens = PROJECT_ROOT / "design.md"
if design_tokens.is_file():
    datas.append((str(design_tokens), "."))

runtime_hooks = [str(PACKAGING_ROOT / "hooks" / "lattice_studio_cuda.py")]

a = Analysis(
    [str(SRC / "lattice_studio" / "__main__.py")],
    pathex=[str(PROJECT_ROOT), str(SRC)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    runtime_hooks=runtime_hooks,
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="LatticeStudio",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name="LatticeStudio",
)
