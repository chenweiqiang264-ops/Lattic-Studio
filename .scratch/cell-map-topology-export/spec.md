# Cell Map, Topology Repair, and Tolerance Export

## Scope

Replace the current TPMS AABB-only generation assumptions with a continuous
cell-map workflow, rebuild generated STL results from the authoritative
implicit field when repair is requested, and add tolerance-driven export and
compiled mesh simplification.

## Decisions

- Cell Map is a regular hexahedral parameterization, not independently meshed STL blocks.
- X, Y, and Z cell spacings are independent user controls. The UI calls them cell-map dimensions and does not expose the internal term "anisotropic".
- The TPMS evaluator receives normalized periodic coordinates derived from the three physical spacings.
- The map is expanded to complete cells before the design-domain SDF clips the field.
- Cell Map has an independent preview showing the design-domain relationship and grid.
- G and D are the only concrete lattice families in this iteration; the pipeline uses a reusable unit-cell provider boundary.
- Generated results are repaired from the authoritative evaluator. Imported external meshes use mesh-only repair.
- Repair is explicit and preserves the original derived mesh for comparison.
- STL export always runs topology inspection and warns, but does not block export.
- Export tolerance is a physical distance in millimetres. It controls extraction and simplification quality rather than directly requesting a face count.
- Export reconstruction never reuses the interactive display cache.
- VTK/Open3D compiled simplification is used with topology validation and rollback on regression.
- No automatic component deletion or invented bridge is allowed.

## Acceptance scenarios

1. `1.stl` generates G and D through one continuous cell-map field and produces a non-empty derived mesh.
2. Cell Map view displays a regular 3D grid with independently changed X/Y/Z spacing and no TPMS result visible.
3. Explicit repair reconstructs a generated result from its evaluator, then reports topology status before and after.
4. Export with a smaller tolerance creates a denser surface; a simplification regression returns to the pre-simplified mesh and emits a warning.
5. An invalid result can still be exported after an explicit warning confirmation.

## Out of scope

Additional non-TPMS cell families, automatic deletion of disconnected components,
and fabricated geometric bridges.
