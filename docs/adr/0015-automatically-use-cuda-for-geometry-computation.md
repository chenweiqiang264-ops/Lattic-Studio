# Automatically use CUDA for geometry computation

The geometry compute boundary owns three operations used by sampled implicit generation: signed distance from a design-domain triangle mesh, pointwise implicit intersection, and Marching Cubes extraction.

When a working CUDA device is available, the Numba CUDA adapter is selected for all three operations. The design-domain mesh is converted once into a flat CPU BVH and its arrays are kept on the device while point tiles are evaluated. GPU Marching Cubes performs cell classification, active-edge marking, interpolation, and shared grid-edge vertex emission. Prefix sums, compact edge-mask indexing, and STL-facing mesh objects remain on the CPU as bounded coordination or topology operations. See ADR-0019 for the shared-edge transfer and display-proxy decisions.

The CPU adapter is the reference path. It uses the existing C++ BVH SDF
extension when available, otherwise PyVista, and uses the same Lorensen
Marching Cubes case convention as the CUDA adapter. A CUDA runtime failure
permanently switches the process to the CPU adapter and records the reason for
the UI. Caller input errors (`TypeError` and `ValueError`) propagate without
changing backend state. This avoids retrying a known-bad device path without
allowing one invalid request to poison later GPU work.

The GPU SDF sign is computed with a point-to-triangle distance plus an independent positive-X ray parity test. Distance pruning never suppresses ray traversal; the two concerns use separate BVH predicates. Marching Cubes vertices are stitched by global grid-edge identity, rather than coordinate-only tolerance, so iso-values passing through grid vertices do not create topology holes.

This backend does not remove the large-grid memory risk. Field slabs and GPU tiles are bounded, but single-pass extraction still requires a complete sampled field and the current Marching Cubes prefix-sum arrays. That risk remains tracked as TPMS-002.
