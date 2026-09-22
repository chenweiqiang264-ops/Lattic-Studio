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
- Fixed backend handlers for custom cells, plane-driven transitions, shells,
  shell unions, display refinement, and precise rendering.
- Qt adapters for supported mesh-domain single-result generation paths and
  downstream handle-based STL, shell, refinement, and render operations.
- Serializable workspace/document commands, including atomic command batches,
  snapshots, and backend-owned persistence.
- Batch generation for multiple TPMS/custom/plane or field-driven transition
  requests, plus analytic design-domain DTOs.
- Qt workspace editing through the command client: document lifecycle,
  domain import/replacement, settings, field scenes, save/load reconciliation,
  and backend-snapshot projection restore.
- A rebuilt Windows onedir and installer validated in an isolated install,
  including GPU, forced CPU fallback, Qt lifecycle, and genuine uninstall.

## Open Scope

Remove or explicitly retain the remaining in-process UI compatibility workers
after representative Qt integration coverage. The backend owns the
serializable editable-document record, while Qt deliberately retains a
disposable projection for widgets, preview actors, and display-only runtime
state.
