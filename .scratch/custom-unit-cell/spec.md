# Custom Unit Cell Specification

## Scope

Import a closed STL body as a reusable non-parametric unit cell, map its source
domain to a regular Cell Map, generate one continuous periodic implicit field,
intersect that field with the design domain, and expose the result through the
existing implicit display and STL reconstruction workflow.

## Requirements

- Import STL in millimetres and use its cleaned bounding box as the default
  three-dimensional unit-cell domain.
- Merge duplicate vertices and remove duplicate or degenerate faces without
  inventing geometry. Reject the input if it remains non-watertight.
- Preserve valid multiple closed components and report their count.
- Map source X/Y/Z to Cell Map U/V/W, including arbitrary right-handed Frame
  orientation and independent target cell dimensions.
- Evaluate one source body periodically. Never duplicate its triangle mesh for
  every Cell Map cell.
- Use negative-inside fields and intersect once with the design-domain field.
- Preserve source thickness by default. Optionally expose a target feature
  thickness as a world-space signed-distance offset; do not present it as a
  TPMS wall-thickness or level parameter.
- Reuse processing modes, implicit rendering, section viewing, reconstruction,
  repair, simplification, and export.

## Acceptance

- A synthetic closed cell repeats with matching field values one period apart.
- Rotating the Cell Map rotates the custom cell field in world coordinates.
- Invalid source domains and non-watertight cells fail with explicit errors.
- The supplied `胞元.stl` is deterministically cleaned from 383,648 to 383,644
  faces and becomes watertight with one connected component.
- The supplied cell fills `resources/鞋底/1.stl`, produces a visible implicit
  result and a non-empty derived STL, and emits a topology report.
