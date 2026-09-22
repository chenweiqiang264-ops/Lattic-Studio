Status: resolved
Type: task

# Continuous Cell Map Generation

Introduce a regular Cell Map covering the padded design-domain bounding box.
Expose independent X/Y/Z spacings, use one continuous evaluator, and extract
the clipped field with one global Marching Cubes path. Add an independent Cell
Map preview and preserve a unit-cell provider boundary for future families.

Acceptance criteria:

- Cell-map geometry is independently inspectable.
- Changing one axis spacing changes only that axis' period and derived map metadata.
- The TPMS field is evaluated continuously across cell boundaries.
- Complete-cell padding is visible in metadata and does not leak outside the design domain.
- G and D regression tests cover non-empty output and stable field periodicity.

## Answer

Implemented continuous, complete-cell-padded Cell Maps with independent X/Y/Z
spacing, per-family G/D maps, shared transition bounds, and an independent
Cell Map view. Regression tests cover map padding, periodic coordinates, and
G/D field evaluation. The `1.stl` acceptance run produced visible G and D Cell
Map artifacts under `tests/results/cell_map_topology_acceptance/`.
