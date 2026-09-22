# Stabilize Custom-cell SDF Signs and Display Islands

## Context

Small custom cells repeatedly map regular display samples to the same source
mesh locations. The CUDA mesh SDF previously shifted +X sign-ray origins in Y
and Z to avoid shared triangle edges. A shift can cross a nearby surface and
replicate one wrong sign through every Cell Map period as a bead-like string.

Even with correct signs, intersecting a cell that touches a periodic face with
the design-domain boundary can leave legitimate but visually distracting
sub-voxel slivers. STL reconstruction already has an explicit tolerance- and
volume-bounded numerical-fragment cleanup, but implicit display caches did not.

## Decision

CUDA mesh SDF sign classification keeps the ray origin at the requested point.
Two independent ray directions must agree; edge hits or parity disagreement use
a third direction and majority classification.

Custom-cell display caches consider negative 26-neighbour components containing
at most six samples, always retaining the largest component. A candidate is
removed only when a local Marching Cubes volume is no greater than one display
voxel and no greater than one millionth of the largest occupancy proxy. The
operation is reported and applies only to the disposable sampled display field.
It does not modify the authoritative evaluator, STL reconstruction, or export.

## Consequences

Periodic custom cells no longer amplify near-surface sign-ray offsets into
repeating particles. Display-only sub-voxel boundary islands are suppressed
without deleting full sampled cells. The cleanup temporarily allocates a label
grid comparable in size to the display field; authoritative extraction memory
and geometry remain unchanged.
