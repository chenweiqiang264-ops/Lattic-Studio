Type: task
Status: resolved

# Analytic design-domain Field Viewer recursion

Using a basic analytic primitive as the active design domain can freeze the
application when Field Viewer is enabled. The observed exception terminates in
`trimesh.base.Trimesh.update_faces()` with `RecursionError`.

## Acceptance criteria

- A sphere, cylinder, and box analytic design domain can enable Field Viewer.
- No analytic design domain is routed through mesh repair during Field Viewer
  refresh.
- The regression loop covers the real Qt/VTK Field Viewer activation path.

## Comments

- 2026-09-14: Claimed. Direct analytic SDF sampling and primitive preview mesh
  construction both pass; full GUI path remains to be reproduced.
- 2026-09-14: Resolved. `repair_mesh()` no longer calls
  `Trimesh.update_faces()`, avoiding the observed recursive cache query.
  Regression coverage forces that third-party call to raise `RecursionError`
  and verifies both normal cleanup and optional hole closure remain valid.
  The analytic Design-domain Field Viewer controller path also passed while
  asserting no mesh repair can occur.
