Status: needs-triage
Type: research

# Transition topology correction creates voxel-like blocks

## Problem

The user reports regular pixel/voxel-like material blocks after enabling the
transition topology correction. The same visual defect is not accepted as a
valid G/D/custom-cell transition result.

The authoritative implementation currently blends the base transition toward
a smooth union with compact strength
`16 * t^2 * (1 - t)^2`, then offsets that union by
`0.125 * minimum_feature_mm`. This formula is continuous, so the visible grid
pattern must be localized to one of three boundaries before changing it:

1. the authoritative corrected field;
2. the sampled implicit display cache;
3. derived-surface extraction and stitching.

The exact screenshot, complete operand parameters, Frame values, display
spacing, and STL spacing are not stored as a reproducible fixture yet.

## Required diagnosis

- Capture one deterministic fixture with design domain, A/B providers, Cell
  Maps, Frames, transition plane, width, minimum feature, display spacing, and
  export spacing.
- Save topology correction off/on field slices and their scalar difference on
  the same grid.
- Compare the authoritative evaluator at 0.5x, 1x, and 2x sampling spacing.
- Compare implicit display, dense Marching Cubes, fixed chunking, and adaptive
  bricks without changing the evaluator.
- Record whether every artifact aligns to sampling cells, adaptive bricks, or
  the physical Cell Map.

## Acceptance criteria

- Topology correction does not create grid-aligned isolated blocks absent from
  the uncorrected operands.
- The corrected result converges under finer sampling rather than changing
  topology unpredictably.
- The correction remains exactly zero outside the user transition band.
- Cross-band connectivity and configured minimum-feature validation continue
  to pass on the accepted fixture.
- A regression test fails on the current implementation and passes on the fix.

## Comments

Do not close this issue by increasing display quality alone. A quality change
may hide a sampling artifact but cannot repair a defective authoritative field.

