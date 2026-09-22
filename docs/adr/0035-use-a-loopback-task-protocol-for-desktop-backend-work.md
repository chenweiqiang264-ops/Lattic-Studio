# ADR 0035: Use a Loopback Task Protocol for Desktop Backend Work

## Status

Accepted.

## Context

The Qt workbench previously owned long-running numerical work, mutable
implicit evaluators, renderer caches, and worker lifecycle in one process.
That makes the user interface the authority for calculation and prevents a
separate local backend from being introduced safely.

Implicit evaluators, VTK actors, Qt objects, CUDA allocations, and trimesh
instances are process-local objects. Sending them as JSON is neither stable nor
safe. At the same time, a desktop client needs progress, cancellation, and
files such as display fields, PNGs, and STL exports.

## Decision

Run a separate backend process bound exclusively to `127.0.0.1`. The desktop
client starts and stops only its own child process. The HTTP interface exposes
fixed task kinds, task snapshots, cancellation, and artifact download. It does
not accept arbitrary function names or code for execution.

The backend owns authoritative implicit results for its process lifetime.
Clients receive:

- task IDs, state, progress, and error text;
- JSON-safe result metadata and opaque result handles;
- files held under a backend-managed task directory.

The task registry includes `tpms.generate`, `custom.generate`,
`transition.generate`, `generation.batch`, `shell.generate`, `shell.union`,
`display.refine`, `render.precise`, and `stl.reconstruct`. Generation tasks
retain the evaluator in the backend, return a generation handle, and publish
the disposable display field as NPZ. Subsequent tasks consume the handle and
publish an NPZ, PNG, or STL artifact as appropriate. Analytic domains and
field-driven transitions use primitive-definition DTOs so the backend builds
its own evaluator; no client evaluator or sampled field is promoted across the
boundary.

The workspace API additionally provides a JSON-only editable-document snapshot
and atomic fixed command batches. These commands mutate backend-owned
definitions, not rendering/runtime state, and are the migration target for Qt
workspace editing.

## Consequences

New asynchronous numerical features must be implemented as backend task
handlers before a Qt adapter is added. Their contracts must have a transport
test covering success and an appropriate failure or cancellation path.

The Qt workbench uses those handles for single and combined mesh or analytic
generation requests, including field-driven transitions, and downstream STL,
shell/fusion, display-refinement, and precise-render operations. It commits
editable document mutations through the workspace command API and restores its
projection from backend snapshots. Qt still keeps a disposable
`DesignWorkspace` mirror for widget state and display-only runtime data. A
sampled display field must never be promoted to an authoritative evaluator.

Some legacy direct-worker compatibility paths remain while their replacement
coverage is completed. Their existence means this decision separates local
processes and authority boundaries; it does not turn the desktop UI into a
remote web frontend.

The loopback backend is private to the local desktop application. This ADR does
not introduce remote networking, multi-user access, authentication, or a web
frontend.
