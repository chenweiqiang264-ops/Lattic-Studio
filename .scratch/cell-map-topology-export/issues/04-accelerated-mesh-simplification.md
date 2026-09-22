Status: resolved
Type: task

# Compiled Mesh Simplification

Replace the current unconstrained simplification path with a compiled VTK or
Open3D backend driven by export tolerance. Validate every candidate and roll
back if water-tightness, connectivity, winding, or deviation constraints regress.

Acceptance criteria:

- The selected backend is reported to the UI and has a deterministic fallback.
- Simplification is bounded by the requested geometric tolerance.
- A topology regression returns the pre-simplified mesh.
- A small closed fixture demonstrates reduced face count without losing topology.

## Answer

Implemented VTK C++ simplification with an Open3D C++ fallback, symmetric
surface-deviation checking, and rollback on topology or tolerance regression.
Regression tests verify both successful closed-mesh reduction and rollback for
invalid topology or excessive geometric deviation.
