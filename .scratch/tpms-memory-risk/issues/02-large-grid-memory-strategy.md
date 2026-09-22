Status: needs-triage
Type: task

# Large-grid memory strategy

TPMS-002: design and validate a memory-bounded generation path for fields and Marching Cubes output without imposing a hidden maximum voxel limit. The solution must preserve shared-boundary topology, support GPU and CPU fallback, and report allocations clearly to the user.

Acceptance criteria:

- peak host and device memory are measured for each processing mode;
- chunked extraction remains water-tight on representative closed fields;
- no implicit maximum voxel limit is introduced;
- out-of-memory failures produce an actionable message and do not leave stale render results.

## Comments

2026-08-08 transition acceptance reproduced the remaining extraction defect on
`resources/鞋底/1.stl`. The authoritative 0.5 mm G-to-D Ramp transition was one
cross-band component with no isolated implicit fragments, but 0.55 mm
`adaptive-bricks` STL extraction produced a winding-consistent, non-watertight
mesh with 5 connected components (178,100 faces). The 36 mm control likewise
passed authoritative transition validation but exported as non-watertight with
14 components. Reports and meshes are under
`tests/results/transition_acceptance/`. This evidence belongs to chunk seam and
domain-boundary reconstruction, not the transition-field connectivity check.
