Status: resolved
Type: task

# Configure Custom-cell Feature Thickness

Allow users to retain the imported STL thickness or request a target physical
feature thickness for a periodic custom unit cell.

Acceptance criteria:

- The default preserves the imported custom-cell zero set.
- A positive target thickness applies a resolution-independent signed-distance
  offset after Cell Map scaling without changing cell size, Frame, or phase.
- Increasing the target expands the custom-cell solid by half the requested
  thickness difference on each exposed side; decreasing it erodes the solid.
- Automatic sampling and STL extraction use the effective target feature
  thickness.
- The UI exposes the optional target in millimetres and reports the effective
  value.
- Core, workbench, UI, full-regression, and real-model checks pass.

## Answer

The custom-cell workbench now exposes an optional target feature thickness.
Its zero value means “use source-model thickness”; a positive value is passed
through `CustomUnitCellParameters` to `make_periodic_lattice()`, which applies
the signed-distance offset after Cell Map scaling. Sampling recommendations,
implicit result metadata, and STL reconstruction receive the effective target.

Verification:

- Focused custom-cell suite: 20 passed.
- Complete intermediate suite: 126 passed.
- Compilation checks passed.
- Real `1.stl` plus supplied `胞元.stl`, with a 37-degree Z Frame rotation:
  increasing feature thickness from 0.754916 mm to 1.132374 mm increased
  inside display voxels from 39,754 to 59,562. All 192,314 changed samples
  moved monotonically toward a larger solid, while Cell Map bounds and Frame
  remained unchanged and the field retained a renderable zero crossing.
