Status: ready-for-agent
Type: task

# Derived transition STL loses watertight topology

## Problem

The authoritative transition validator and the adaptive-brick STL extractor
currently disagree on `resources/鞋底/1.stl`.

Evidence under `tests/results/transition_acceptance/`:

- The 36 mm G/D Ramp passes authoritative validation with one component, but
  the 0.55 mm adaptive-brick STL is non-watertight with 14 components.
- The 0.5 mm G/D Ramp is cross-band connected but fails minimum feature; its
  0.55 mm adaptive-brick STL is non-watertight with 5 components.

Both STL outputs are winding-consistent. This points to open boundaries,
brick stitching, degeneracy cleanup, or design-domain halo closure rather than
only a global face-orientation error.

## Investigation plan

- Reconstruct the same authoritative evaluator at the same spacing with dense,
  fixed chunked, and adaptive-brick paths.
- Report open/non-manifold edges and their distance to brick boundaries and
  the global design-domain boundary.
- Verify CUDA global edge IDs on both sides of every adjacent brick.
- Check whether duplicate/degenerate-face cleanup removes one side of a valid
  seam.
- Compare bounds, volume, component count, and symmetric surface deviation
  against the dense reference.

## Acceptance criteria

- Adaptive and fixed-chunk extraction are watertight whenever the same-spacing
  dense reference is watertight.
- Winding consistency, connected-component count, bounds, and volume match the
  dense reference within an explicit numerical tolerance.
- The 0.5 mm and 36 mm transition fixtures have regression coverage.
- Peak RAM remains measured; the fix does not silently force every request to
  a full dense field.
- Invalid results remain exportable with a visible warning.

## Related

- `docs/issues/TPMS-001-connectivity-and-watertightness.md`
- `.scratch/tpms-memory-risk/issues/02-large-grid-memory-strategy.md`
- `core/implicit/surface_extraction.py::_stitch_chunks`
- `core/implicit/geometry_compute.py::marching_cubes_chunk`

