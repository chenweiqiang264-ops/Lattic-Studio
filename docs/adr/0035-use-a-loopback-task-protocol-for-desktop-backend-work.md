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

The first end-to-end numerical task chain is `tpms.generate` followed by
`stl.reconstruct`. The TPMS task retains the evaluator in the backend, returns
a generation handle, and publishes the disposable display field as NPZ. STL
reconstruction consumes that handle and returns a managed STL artifact.

## Consequences

New asynchronous numerical features must be implemented as backend task
handlers before a Qt adapter is added. Their contracts must have a transport
test covering success and an appropriate failure or cancellation path.

The current workbench still contains in-process workers for several workflows.
They remain compatibility paths until each consumer uses backend result handles
and renderable artifacts. A sampled display field must never be promoted to an
authoritative evaluator merely to avoid completing that migration.

The loopback backend is private to the local desktop application. This ADR does
not introduce remote networking, multi-user access, authentication, or a web
frontend.
