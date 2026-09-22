# Lattice Studio engineering refactor

## Objective

Turn the current TPMS workbench into an installable Windows application with a clear package structure, stable Python interface, isolated domain state, automatic NVIDIA acceleration, CPU fallback, and maintainable tests. Preserve current user-visible behaviour and numerical results while migrating, then remove obsolete implementations and compatibility code.

## Current state

- `tests/intermediate_tests/tpms_filling_app.py` is the real production application.
- The file contains about 6,100 lines of models and workflows plus a Qt window with 262 methods.
- Tests and stress tools import production functions directly from the test-hosted application.
- `core/implicit` already contains reusable numerical modules, but orchestration, reconstruction, sampling, repair, and UI state remain coupled in the workbench.
- `ui_app.py`, the old project/domain packages, stale native artifacts, and generated files require usage verification before removal.
- CUDA selection works locally but currently requires NVVM from a separately installed CUDA Toolkit.

## Target structure

```text
src/lattice_studio/
  __init__.py
  __main__.py
  public.py
  domain/
  application/
  engine/
  infrastructure/
  presentation/qt/
tests/
  unit/
  integration/
  acceptance/
  stress/
```

Dependencies point inward: presentation calls application use cases; application owns workflow orchestration and depends on domain models and engine interfaces; infrastructure supplies persistence, native, CUDA, mesh I/O, and packaging adapters. Domain code does not import Qt, VTK, filesystem, CUDA, or persistence modules.

## Compatibility

- Keep the existing test-hosted entry only while imports migrate.
- Preserve current GUI behaviour during structural extraction.
- Preserve numerical results within explicit tolerances.
- Load old workspace data through versioned migrations.
- Expose a small supported Python interface for opening designs, generating lattices, reconstructing meshes, and exporting results.

## Distribution

- Build a Windows PyInstaller `onedir` application.
- Users do not install Python or the CUDA Toolkit.
- Detect compatible NVIDIA devices automatically and use bundled CUDA runtime components.
- Fall back to CPU on unsupported hardware, insufficient memory, or runtime failure and report the reason.
- AMD and Intel GPUs remain rendering devices only in the first release.

## Cleanup policy

Delete code only after static references, dynamic import paths, tests, representative workflows, and packaged startup demonstrate that it is unused. Do not retain backup copies, stale binaries, object files, caches, screenshots, or production modules under `tests` in the final tree. Source files required to rebuild shipped native extensions remain tracked; transient compiler outputs do not.

## Completion criteria

- `python -m lattice_studio` and the packaged executable start successfully.
- Existing automated tests pass after migration to supported interfaces.
- G, D, custom-cell, gradient, and transition results match the baseline within documented tolerances.
- Existing workspace data loads and migrates.
- STL reconstruction, shell fusion, field viewing, section viewing, and layer contours pass representative acceptance tests.
- NVIDIA detection and CPU fallback are verified in isolated child processes.
- PyInstaller `onedir` builds and starts without a development environment.
- A representative stress matrix completes.
- The final dependency and dead-code audits find no production dependency on legacy entry files.
- The engineering structure document describes ownership, interfaces, runtime flow, persistence, acceleration, packaging, testing, and extension procedures.

