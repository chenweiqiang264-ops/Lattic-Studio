Status: resolved
Type: task

# Optimize Custom-cell Field and STL Reconstruction

Custom STL cells repeatedly evaluate a large source-mesh BVH during display
sampling and STL reconstruction. Rotated Cell Maps also enlarge the extraction
AABB, while custom bodies currently lack a conservative exclusion field and
therefore fall back from adaptive extraction to a complete dense grid.

Acceptance criteria:

- The authoritative custom-cell result bounds are the trimmed design-domain
  bounds, not the larger rotated Cell Map AABB.
- Custom cells provide a conservative exclusion field that can skip only
  space proven outside the design domain.
- Adaptive/chunked STL reconstruction no longer falls back solely because the
  custom body lacks an exclusion field.
- Interactive custom-cell sampling can reuse a compatible design-domain SDF
  display cache without changing the authoritative evaluator.
- Geometry, topology, processing-mode, display-cache, and full regression
  tests pass.
- Real `1.stl` plus supplied `胞元.stl` timing and sample-count comparisons are
  recorded before and after the change.

## Answer

Custom-cell bodies now use design-domain bounds, preserve Cell Map sampling
phase through an extraction-grid anchor, expose a conservative domain-exterior
field, request 48-cell CUDA-friendly adaptive bricks, and reuse compatible
design-domain display caches.

Real acceptance used `1.stl`, the supplied `胞元.stl`, a 37-degree Z Frame,
12 x 12 x 8 mm cells, 3.0 mm target feature thickness, 0.8 mm export tolerance,
and the same 0.5 mm isotropic extraction spacing before and after:

- STL reconstruction: 18.719 s -> 10.325 s (44.8% faster).
- Full-range grid points: 5,796,840 -> 1,176,147 (79.7% fewer).
- Actual adaptive evaluations including overlap: 1,219,074.
- Display generation with a resident design-domain cache: 7.599 s -> 3.403 s
  (55.2% faster).
- Display shape and inside voxels remained `(80, 215, 18)` and 39,754.
- Both STL meshes had 381,452 faces, 190,626 vertices, watertight and
  winding-consistent topology, and 83 connected components.
- Symmetric 100,000-sample surface deviation was 0.01032 mm, below the
  0.8 mm export tolerance and without changing voxel spacing.

Verification completed with 55 focused tests and 131 complete intermediate
tests. CUDA low-occupancy warnings came only from deliberately small test
grids.
