"""Crash-isolated CUDA runtime probing for optional numerical backends."""

from __future__ import annotations

import os
import subprocess
import sys
import json
import tempfile
from pathlib import Path
from dataclasses import dataclass

from lattice_studio.infrastructure.gpu_runtime import configure_bundled_cuda_runtime


_TRUE_VALUES = frozenset(("1", "true", "yes", "on"))
_FALSE_VALUES = frozenset(("0", "false", "no", "off"))


@dataclass(frozen=True)
class CudaProbeResult:
    available: bool
    device_name: str = "CUDA unavailable"
    reason: str = "CUDA runtime is unavailable"


def probe_numba_cuda() -> CudaProbeResult:
    """Probe a Numba CUDA context in a child process.

    A broken Windows WDDM/driver combination can terminate the process with a
    native access violation while creating a Numba context.  Running the probe
    out of process turns that failure into an ordinary unavailable result and
    keeps the host application alive for CPU fallback.
    """

    # Numba's Windows driver binding has a known class of configurations in
    # which a child process can enumerate the device but the host process
    # crashes on its first context allocation.  The probe remains isolated so
    # that this is safe to do automatically when the application starts.
    # ``nvidia-smi`` is intentionally not used as a prerequisite: it is a
    # driver diagnostic utility, not the CUDA runtime API that the numerical
    # backend actually needs, and it may be absent from PATH on a valid setup.
    configure_bundled_cuda_runtime()
    disable_value = os.environ.get("TPMS_DISABLE_NUMBA_CUDA", "").strip().lower()
    legacy_enable_value = os.environ.get("TPMS_ENABLE_NUMBA_CUDA")
    legacy_disable_requested = (
        legacy_enable_value is not None
        and legacy_enable_value.strip().lower() in _FALSE_VALUES
    )
    if disable_value in _TRUE_VALUES:
        return CudaProbeResult(
            False,
            reason="Numba CUDA disabled by TPMS_DISABLE_NUMBA_CUDA=1",
        )
    if legacy_disable_requested:
        return CudaProbeResult(
            False,
            reason="Numba CUDA disabled by TPMS_ENABLE_NUMBA_CUDA=0",
        )
    if getattr(sys, "frozen", False):
        # A frozen executable cannot interpret `-c`; that launches the GUI
        # again and recursively spawns probes. Use an early worker entry point
        # and a file result, since windowed executables have no stdout.
        try:
            with tempfile.TemporaryDirectory(prefix="LatticeCudaProbe-") as directory:
                result_path = Path(directory) / "result.json"
                completed = subprocess.run(
                    [sys.executable, "--cuda-probe-worker", str(result_path)],
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                    text=True, check=False, timeout=30,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                )
                if completed.returncode != 0 or not result_path.is_file():
                    return CudaProbeResult(False, reason=f"CUDA worker exited with {completed.returncode}")
                return CudaProbeResult(**json.loads(result_path.read_text(encoding="utf-8")))
        except (OSError, ValueError, TypeError, subprocess.SubprocessError) as exc:
            return CudaProbeResult(False, reason=f"{type(exc).__name__}: {exc}")
    # Device enumeration alone is insufficient: Numba additionally needs
    # NVVM to JIT compile CUDA kernels.  Validate the complete chain in the
    # isolated process so the application never advertises a GPU that must
    # immediately fall back to CPU.
    script = """
from numba import cuda
from numba.cuda.cudadrv import driver, nvvm

if not driver.driver.is_available:
    raise RuntimeError("CUDA driver is unavailable")
if not nvvm.is_available():
    raise RuntimeError("NVVM is unavailable: install CUDA Toolkit and set CUDA_PATH")
if not cuda.is_available():
    raise RuntimeError("Numba CUDA runtime is unavailable")
device = cuda.get_current_device()
name = device.name
print(name.decode() if isinstance(name, bytes) else str(name))
"""
    try:
        completed = subprocess.run(
            [sys.executable, "-c", script],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
            timeout=20,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return CudaProbeResult(False, reason=f"{type(exc).__name__}: {exc}")
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).strip().splitlines()
        reason = detail[-1] if detail else f"child probe exited with {completed.returncode}"
        return CudaProbeResult(False, reason=reason)
    name = completed.stdout.strip()
    if not name:
        return CudaProbeResult(False, reason="CUDA probe returned no device name")
    return CudaProbeResult(True, device_name=name, reason="")


__all__ = ["CudaProbeResult", "probe_numba_cuda"]


def run_probe_worker(result_path: str) -> int:
    """Run only in the isolated child, never initialize the workbench."""
    from dataclasses import asdict

    try:
        configure_bundled_cuda_runtime()
        from numba import cuda
        from numba.cuda.cudadrv import nvvm
        import numpy as np

        if not cuda.is_available() or not nvvm.is_available():
            raise RuntimeError("CUDA driver or bundled NVVM is unavailable")
        cuda.to_device(np.zeros(1, dtype=np.float32))
        cuda.synchronize()
        name = cuda.get_current_device().name
        result = CudaProbeResult(True, name.decode() if isinstance(name, bytes) else str(name), "")
    except Exception as exc:
        result = CudaProbeResult(False, reason=f"{type(exc).__name__}: {exc}")
    Path(result_path).write_text(json.dumps(asdict(result)), encoding="utf-8")
    return 0
