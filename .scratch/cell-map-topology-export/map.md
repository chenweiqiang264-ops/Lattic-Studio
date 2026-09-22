# Cell Map, Topology Repair, and Tolerance Export

## Notes

This effort addresses the historical multi-component and non-watertight STL
problem at the continuous-field boundary first, then makes repair and export
quality observable and reversible.

## Decisions-so-far

- [spec](spec.md): continuous global Cell Map field with complete-cell padding.
- [spec](spec.md): topology warnings do not block STL export.
- [spec](spec.md): generated results use implicit-field reconstruction for repair and export.
- [spec](spec.md): G/D only now, with a reusable unit-cell provider boundary.
- [01](issues/01-cell-map-continuous-generation.md): continuous per-family Cell Maps and independent preview are implemented.
- [02](issues/02-implicit-field-topology-repair.md): field repair and explicit, tolerance-bounded numerical-fragment cleanup are implemented.
- [03](issues/03-tolerance-driven-stl-export.md): authoritative tolerance reconstruction and reload validation are implemented.
- [04](issues/04-accelerated-mesh-simplification.md): compiled simplification with topology/deviation rollback is implemented.
- [05](issues/05-fitted-cell-map.md): fitted Cell Map uses exact design-domain bounds, uniform actual spacing, and map-aligned TPMS phase.
- [06](issues/06-frame-origin-follows-uvw.md): the default Frame origin follows
  U/V to the design domain's UVW minimum; manual phase-origin control remains
  available.

## Frontier

All implementation tickets are resolved. The current `1.stl` fitted-map
acceptance report is
`tests/results/cell_map_topology_acceptance/acceptance_report.json`.
