Status: resolved
Type: task

# Tolerance-Driven STL Export

Reconstruct export fields at an export-specific resolution derived from a
millimetre tolerance, independent of the interactive display cache. Run global
Marching Cubes, validate topology, and allow export with a visible warning when
the result is not watertight or singly connected.

Acceptance criteria:

- Export tolerance is configurable in millimetres.
- Export does not reuse display-field spacing.
- Smaller tolerance produces a denser or equal derived mesh for the same field.
- Invalid topology produces a warning and still permits confirmed export.
- Exported files are reloaded and inspected before success is reported.

## Answer

Implemented tolerance-driven reconstruction from the authoritative implicit
evaluator, independent of display spacing. Export remains available after an
explicit topology warning, and every written STL is reloaded with vertex
processing enabled before its water-tightness and component count are reported.
The `1.stl` G/D acceptance exports are both watertight, winding-consistent, and
single-component.
