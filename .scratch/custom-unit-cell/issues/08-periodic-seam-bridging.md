Status: resolved
Type: task
Blocked by: 07

# Periodic Seam Bridging

Custom STL unit cells may be watertight and connected individually while their
opposite source-domain face sections do not match under periodic repetition.
The workbench must report this as a warning and optionally add local material
at explicitly selected U/V/W periodic seams.

## Decision

The user manually selects bridge directions. The system never automatically
adds material in an unselected direction. A bridge uses the union of the two
opposite-face sections and a bounded physical depth; the default depth is
conservative and can be overridden in millimetres.

## Answer

Added `PeriodicSeamReport` and `PeriodicSeamAxisReport` to the prepared custom
unit cell. The report now evaluates the current Cell Map scale, target feature
offset, physical probe depth, and solid face areas in mm². It reports both the
source compatibility and the expected post-bridge compatibility, while
remaining warning-only. `PreparedCustomUnitCell.evaluate_periodic()` now accepts
`bridge_directions` and `bridge_depth_mm`; selected directions receive a local
union-based bridge field, while the empty tuple preserves the prior evaluator.
The workbench exposes U/V/W checkboxes, an automatic or explicit bridge depth,
and includes seam status in preflight details. The same parameters flow into
implicit display sampling and later STL reconstruction through the shared body.

Focused regression: 18 core custom-cell tests and 1 Qt workbench smoke test
pass with CUDA probing disabled; compilation passes. The supplied high-face
count cell was prepared as 383,644 faces and one component. At 12 samples per
face its seam report marked U/V/W incompatible with mismatch ratios 1.000,
1.000, and 0.714. A CPU field probe with U bridging changed 20 of 512 signs
and increased occupied samples from 84 to 104; V and W were not bridged.
