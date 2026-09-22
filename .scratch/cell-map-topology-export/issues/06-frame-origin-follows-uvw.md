Status: resolved
Type: task
Blocked by: 05

# Make the Frame Origin Follow UVW

The previous workbench retained the world-axis bounding-box minimum when U/V
changed. The Cell Map orientation updated, but its visible UVW origin did not
become the starting point of the newly oriented map.

## Decision

Origin following is enabled by default. The origin is recomputed as the
design-domain point whose coordinates are the independent minima in the
current UVW frame. Users can disable origin following and edit the origin as a
manual periodic phase anchor.

## Answer

`CellMapFrame.aligned_to_points_minimum()` now computes the full-precision
Frame start. G, D, and Custom panels expose an `原点随 U/V 自动定位` checkbox.
Automatic values are displayed at six decimal places but generation resolves
the origin directly from the design domain, so UI quantization cannot create
an unintended negative cell index. Reset restores world XYZ and automatic
origin following.

Verification completed with 48 Cell Map/workbench tests and 28 viewport/custom
tests. On `resources/鞋底/1.stl`, a 37-degree rotation moved the origin from
`[-26.8225, -83.4561, 0.0]` to `[0.2705, -91.1351, 0.0]`; the resulting
11 x 13 x 2 map starts at index `(0, 0, 0)` and covers every design-domain
vertex.
