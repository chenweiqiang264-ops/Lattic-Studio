Type: task
Status: resolved

# Keep analytic field-object viewing responsive

Viewing a sphere, cylinder, or box Field object can freeze the UI. The Field
Viewer currently rebuilds the VTK scene more than once when primitive overlays
change, and hover handling performs a VTK cell pick for every mouse movement.

## Acceptance criteria

- Selecting a Field object among multiple primitives rebuilds the scene once.
- Hover field probes use the known plane geometry rather than VTK cell picks.
- Existing Field Viewer controller and mesh-repair tests still pass.

## Comments

- 2026-09-14: Resolved. Primitive-field activation batches overlay changes
  into the single main-scene rebuild. Field Viewer now owns the active gizmo
  while enabled. Hover queries use ray-plane intersection and coalesce UI
  redraws to one update per 33 ms. The field-viewer regressions and all
  `tests/test_implicit` checks pass.
