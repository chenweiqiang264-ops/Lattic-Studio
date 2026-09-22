Status: resolved
Type: task

# Render Rotated Custom Unit Cells

Rotating a custom-cell Frame expands the Cell Map world AABB. The display
recommendation uses the design-domain grid shape while sampling starts at the
Cell Map lower bound, so the sampled cache can miss the design domain and
contain no zero isosurface.

Acceptance criteria:

- A rotated custom cell produces a display cache containing inside and outside
  field values.
- The display cache covers the complete design-domain bounds.
- Display cache dimensions remain based on the trimmed design domain rather
  than the larger rotated Cell Map AABB.
- The supplied `胞元.stl` renders after a 37-degree Frame rotation.

## Answer

`generate_implicit_custom_lattice()` now samples its display cache over the
design-domain bounds. Previously, the cache shape came from the design domain
while its origin came from the rotated Cell Map AABB; that mismatched grid
could miss the design domain and contain no zero crossing.

Regression coverage was added for a rotated custom-cell frame. The complete
intermediate suite passes with 116 tests. A real acceptance run using
`resources/鞋底/1.stl`, the supplied `胞元.stl`, a 37-degree Z rotation,
12 x 12 x 8 mm cells, and a 0.8 mm display voxel produced a `(80, 215, 18)`
display grid with 39,754 inside voxels and 269,846 outside voxels. Its scalar
range was `[-0.4717086, 18.940954]`, and the sampled bounds covered the complete
design domain.
