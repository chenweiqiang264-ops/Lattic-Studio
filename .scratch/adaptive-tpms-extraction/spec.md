Status: resolved
Type: task

# Real TPMS adaptive extraction

The G and D evaluator-backed implicit bodies must provide a conservative
surface-exclusion field so the user-facing adaptive extraction switch reaches
the adaptive-bricks backend on real TPMS data.  The adaptive result must
match the dense Marching Cubes baseline for the same evaluator and sampling
parameters.  This work does not claim to solve the existing multi-component
G topology defect.

Acceptance criteria:

- The exclusion field has the same sign and zero set as the authoritative G
  or D shell field.
- Its declared Lipschitz bound is conservative for rectangular Cell Maps.
- `resources/鞋底/1.stl`, after the explicit existing repair fixture, enters
  `adaptive-bricks` for both G and D in automatic mode.
- Adaptive and dense outputs agree in face count, bounds, volume, and topology
  status for the regression fixture.
- Auto mode retains a dense fallback for cases whose adaptive boundary/sample
  overhead is materially larger than the dense path.
