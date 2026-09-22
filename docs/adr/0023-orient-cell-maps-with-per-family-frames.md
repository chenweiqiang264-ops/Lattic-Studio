# Orient Cell Maps with Per-Family Frames

## Context

The original Cell Map was aligned to the world XYZ axes and used its
axis-aligned minimum bound as the TPMS phase origin. Rotating a unit cell
therefore required changing family-specific equations, and G and D could not
have independent orientations in a transition. A rotated map also cannot, in
general, preserve an exact fitted envelope, a requested cell size, and a fixed
phase origin at the same time.

## Decision

Each lattice family owns an independent right-handed UVW Cell-map frame. The
frame is entered as an origin plus U and V directions; U is normalized, V is
orthogonalized against U, and W is their cross product. Parallel or degenerate
input axes are rejected.

The workbench offers two origin modes. By default, the origin follows U/V: it
is recomputed as the design domain's minimum corner in the current UVW frame,
so changing orientation also changes the Cell Map starting point. In manual
mode, the frame origin remains an explicit periodic phase anchor. Cell Maps
continue to support signed integer indices for programmatic or manual frames
whose origin is not the frame-aligned minimum.

A custom frame always preserves the requested U/V/W cell sizes and uses
complete-cell coverage. The design-domain implicit intersection trims cells
outside the target volume. Fitted bounds remain available only for the default
world-aligned frame.

G and D retain separate frames when their fields are blended in a transition.
Frame edits update only the selected Cell Map preview and invalidate dependent
generated results; expensive implicit generation remains an explicit user
action.

## Consequences

The default interaction matches a Frame-driven starting point: rotating U/V
also relocates the visible UVW origin. Users who require phase continuity can
disable origin following and edit the phase anchor directly. Unit-cell
orientation and phase remain independent between lattice families. Rotated
Cell Maps may sample an axis-aligned world envelope larger than the design
domain, but they do not silently distort cell sizes. Future non-TPMS unit-cell
providers can consume the same UVW coordinates without family-specific
rotation controls.
