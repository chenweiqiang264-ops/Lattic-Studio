# Local Backend Separation

## Goal

Move long-running, authoritative numerical computation behind a private local
backend process while preserving the PyQt/PyVista desktop user experience.

## Constraints

- Bind only to `127.0.0.1`.
- Never serialize Qt, VTK, CUDA, trimesh, or `ImplicitBody` objects as JSON.
- Expose only fixed task kinds, not arbitrary executable names or code.
- Keep backend result authority for the process lifetime and return serializable
  status plus derived artifacts to the frontend.

## Delivered

- Workspace-session loopback interface.
- Asynchronous task registry with state, progress, cancellation, and artifacts.
- Desktop-owned child-process lifecycle.
- A tested `tpms.generate` -> `stl.reconstruct` vertical path.

## Open Scope

Move every remaining workbench workflow and workspace mutation to task/result
handles before claiming complete frontend/backend separation.
