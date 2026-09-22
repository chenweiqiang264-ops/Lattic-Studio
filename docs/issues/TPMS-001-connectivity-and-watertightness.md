# TPMS-001: Multiple components and non-watertight results

- Status: Open
- Priority: Critical
- Scope: TPMS implicit-field generation and final mesh extraction
- Discovered in: `resources/examples/design_domains/sole-1.stl`

## Problem

Generated G/D TPMS results can contain multiple connected components and may be non-watertight. A component count greater than one is not by itself an error, because several disconnected closed components can still form a watertight mesh. The actual defect is that at least one component may contain open boundary edges or non-manifold topology.

## Known contributing factors

- Thin local regions of the shoe sole relative to lattice wall thickness;
- Insufficient wall sampling in the voxel grid;
- Discrete loss of narrow TPMS connections during sampling;
- Approximation error in the TPMS signed-distance-like field;
- Topology changes introduced by the implicit intersection and Marching Cubes extraction.

## Required future work

1. Report per-component size, boundary-edge count, manifold status, and watertightness.
2. Distinguish valid disconnected closed components from broken components.
3. Define an explicit policy for removing tiny components and bridging small gaps; never silently discard valid lattice volume.
4. Add field-level connectivity repair before Marching Cubes, followed by conservative mesh-level repair.
5. Validate topology, volume deviation, domain containment, and export suitability after every repair.

## Current workaround

Use a smaller manually selected voxel size and verify the generated result. The existing repair action handles simple holes and mesh cleanup but is not a complete solution for large connectivity defects.
