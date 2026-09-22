# Lattice Studio Engineering Architecture

## Purpose

Lattice Studio is a desktop lattice-design application with a supported
headless Python interface. The production entrypoint is `python -m
lattice_studio`; the Qt workbench is one presentation adapter, not the owner of
the numerical model.

## Package ownership

```text
src/lattice_studio/
  domain/             state, value objects, invariants, no I/O or Qt
  application/        user-facing use cases and scripting services
  engine/             numerical workflows and geometry adapters
    implicit/         TPMS, SDF, transitions, shell, extraction
  infrastructure/     persistence, mesh I/O, native/GPU/distribution adapters
    native/           optional C++ extensions and build sources
  presentation/qt/    Qt composition, widgets, workers, view state
    viewers/          PyVista/Open3D rendering adapters
  presentation/http/  loopback HTTP adapter and desktop-client transport
```

The intended dependency direction is:

```text
presentation -> application -> domain
                         -> engine interfaces/implementations
infrastructure --------> domain/application contracts
```

Domain code must not import Qt, VTK, CUDA, filesystem persistence, or window
state. Engine code owns numerical kernels and mesh evaluation. Infrastructure
owns effects such as reading an STL, copying managed assets, and writing a
workspace manifest.

## Runtime flow

1. `lattice_studio.__main__` parses launch options and creates the Qt
   workbench.
2. The workbench collects parameters and calls application services.
3. Application services validate the request and dispatch engine workflows.
4. Engine workflows evaluate authoritative implicit fields, reconstruct meshes,
   and return result objects.
5. Presentation workers publish progress and render derived data. Renderer
   caches and sampled fields remain runtime-only.

The stable scripting facade is `lattice_studio.public.LatticeStudio`. New
headless operations belong in `application/`, then are exposed deliberately
through this facade; callers must not import Qt widgets or the historical test
entrypoint.

## Local backend migration

`application.backend.LocalBackend` is the interface for a standalone local
backend process. It owns short-lived workspace sessions and delegates workspace
behavior to application use cases; it does not expose Qt, VTK, `trimesh`, CUDA
buffers, or renderer caches. `presentation.http.local_backend` is its loopback
HTTP adapter, bound only to `127.0.0.1`, with a Python client adapter for the
desktop front end.

The transport exposes workspace health, creation, loading, lookup, and explicit
save. It also exposes a fixed asynchronous task registry: submit, status,
cancellation, and artifact download. A task snapshot contains an ID, stable
state (`queued`, `running`, `succeeded`, `failed`, or `cancelled`), progress,
message, JSON-safe metadata, and artifact descriptors. Artifact files are held
under backend-managed task directories and may not escape those directories.

The first complete numerical chain is `tpms.generate` followed by
`stl.reconstruct`. The backend retains the authoritative implicit evaluator,
returns an opaque generation ID, publishes only a disposable NPZ display field,
and reconstructs the STL by consuming that generation ID. The desktop process
owns only a private backend child process and starts it on a dynamically chosen
loopback port. It stops only that child during workbench shutdown.

Existing Qt workflows remain compatibility paths while their consumers are
migrated to backend handles and renderable artifacts. New remote-capable
operations must first be expressed through the application backend interface,
use serializable request and result data, and add a representative transport
contract test. Evaluators, VTK objects, CUDA buffers, and renderer caches must
never cross the seam.

## Workspace and persistence

`domain.workspace` owns `DesignWorkspace`, `DesignDocument`, runtime-cache
boundaries, multi-design isolation, and recoverable archive transitions. A
document contains editable definitions and references to derived assets; it
does not persist sampled fields, renderer actors, CUDA buffers, or UI objects.

`infrastructure.workspace_persistence` owns the JSON schema. The current schema
is version 2. Version 1 manifests are migrated on read, and saves always write
the current version. Meshes are referenced through workspace-relative managed
assets under `assets/`; absolute paths and parent traversal are rejected.

The former root-level `core`, `ui`, and `domain` packages are not runtime
packages. Their active implementations now live under `src/lattice_studio`.

## GPU and fallback boundary

GPU selection is an infrastructure concern. The application must receive a
backend capability and a fallback reason, not inspect Qt or CUDA from domain
models. NVIDIA CUDA is preferred for supported numerical kernels; CPU remains
the deterministic fallback for missing drivers, unavailable packaged runtime,
insufficient memory, or kernel failure. AMD/Intel are rendering devices in the
first distribution target.

PyInstaller uses `onedir` so bundled CUDA runtime components and native mesh
libraries can be discovered beside the executable. GPU detection must be
tested in child processes for both supported and failure paths.

## Testing strategy

- `tests/unit/`: domain invariants, serialization, migrations, and pure kernels.
- `tests/integration/`: numerical workflows plus service and adapter contracts.
- `tests/acceptance/`: executable Qt workflows and representative user scenarios.
- `tests/benchmarks/`: opt-in performance measurements; never part of the fast suite.
- `tests/stress/`: subprocess-isolated VTK/GPU/large-mesh workloads.

Generated reports, screenshots, logs, and meshes are written under
`build/test-artifacts/`. Test source directories never own generated output.

Native OpenGL/VTK scenarios must not be repeatedly created in one pytest
process on Windows. They run in child processes with explicit cleanup and
timeouts. Numerical tests report tolerances and backend selection.

## Adding a feature

Define or refine the domain vocabulary first. Add a domain invariant or value
object without UI imports, implement the use case in `application/`, put
numerical work in a cohesive `engine/` module, and add I/O or device handling
through `infrastructure/`. The Qt layer should only translate widgets/events
to commands and render returned state. Add a unit contract and one
representative acceptance test before removing a compatibility path.

## Migration completion criteria

The refactor is complete when no production module imports the historical test
entrypoint, PyInstaller `onedir` starts on a clean machine, and the subprocess
acceptance/stress matrix covers TPMS, custom cells, gradients, transitions,
shell fusion, field viewing, layer contours, and GPU fallback.
