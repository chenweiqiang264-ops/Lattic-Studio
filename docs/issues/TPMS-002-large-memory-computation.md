# TPMS-002: Large-grid memory usage and scalable extraction

- Status: Open
- Priority: High
- Scope: Voxel field evaluation, mesh extraction, and result export
- Discovered in: custom voxel generation on the shoe-sole design domain

## Problem

Very fine user-selected voxels can produce tens of millions of grid points.
The former all-at-once path materialised the complete point array and several
intermediate scalar arrays at the same time. A single allocation such as
`(29,899,896, 3) float64` requires approximately 684 MiB, while the complete
calculation requires substantially more memory.

## Current mitigation

The UI now provides optional batch evaluation. When enabled, the grid is
split into user-selected slabs, and each slab evaluates the design-domain
SDF, TPMS fields, and Boolean/blended field before releasing its temporary
arrays. The C++ mesh-to-SDF backend remains enabled per batch. The final
continuous scalar field is retained for the current Marching Cubes step.

## Required future work

1. Add an explicit memory estimate to the generation preflight and report the
   expected peak for the selected voxel size and batch count.
2. Store very large scalar fields in a memory-mapped or disk-backed format
   when RAM is insufficient.
3. Implement overlapping slab Marching Cubes with deterministic vertex/face
   stitching, so the final scalar field does not have to remain fully in RAM.
4. Add cancellation and progress reporting at the batch and extraction
   stages, including cleanup of temporary storage after interruption.
5. Benchmark batch count, C++ SDF throughput, peak RSS, extraction time, and
   output equivalence on representative shoe domains.

