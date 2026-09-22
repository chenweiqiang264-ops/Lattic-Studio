# Evaluate Custom STL Cells through Cell Maps

## Context

G and D are parametric field families, but a general lattice workflow must
also accept a user-authored unit-cell body. Copying a large STL into every
Cell Map cell would multiply mesh memory by the cell count, make arbitrary
UVW frames expensive, and create a separate Boolean operation for every
copy. The supplied custom cell alone contains 383,648 source triangles.

## Decision

Prepare a custom STL once as a non-parametric unit cell. Import performs only
deterministic cleanup: merge processing, duplicate and degenerate face
removal, unreferenced vertex removal, and normal repair. An input that remains
non-watertight or winding-inconsistent is rejected because its signed field is
not reliable. Valid multiple closed source components are preserved and
reported.

The cleaned source bounding box is the default three-dimensional Unit-cell
Domain. Source X/Y/Z map to Cell Map U/V/W. World points are transformed into
Cell Map coordinates, folded into one period, mapped into the source Domain,
and evaluated against one private resident CPU/CUDA mesh-distance backend.
The source STL defines the default thickness. The UI exposes target U/V/W cell
dimensions, the Cell-map frame, and an optional target feature thickness. The
target is implemented as a world-space signed-distance offset after Cell Map
scaling: changing thickness does not scale the source body, alter its period,
or move its Frame. The control is not a TPMS wall-thickness or level parameter.
For a requested change `delta_t`, the zero set moves by `delta_t / 2` on each
exposed side. Complex intersections and curvature mean this global
characteristic target is not a guarantee of exact local minimum thickness.

The periodic custom-cell field is intersected once with the authoritative
design-domain field. Interactive implicit display, STL reconstruction,
topology inspection, repair, simplification, and export reuse the existing
lattice pipeline. For non-uniform source-to-cell scaling, signed-distance
magnitude uses the minimum scale factor; this preserves the zero set and is
conservative, but it is not an exact anisotropically transformed distance.

## Consequences

Mesh memory is independent of Cell Map cell count, UVW frame changes rotate
and phase-shift the custom cell through the same mechanism as TPMS families,
and future non-TPMS providers can reuse the placement and export pipeline.
The first implementation evaluates the exact source mesh BVH; a sampled local
template may optimize it later without changing the public preparation and
placement interfaces.

Periodic occupancy is authoritative, but field magnitude can be discontinuous
at opposite source-Domain faces when the authored cell does not match there.
Disconnected solids or design-domain boundary slivers remain observable
topology and are not silently deleted.

Automatic sampling and derived-STL extraction use the effective feature
thickness: the scaled source characteristic when the target is unset, or the
positive user target when it is set. Invalid zero, negative, or non-finite
targets are rejected before generation.
