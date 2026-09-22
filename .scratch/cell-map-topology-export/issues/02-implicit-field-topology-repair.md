Status: resolved
Type: task

# Implicit-Field Topology Repair

Replace ineffective generic mesh repair for generated TPMS results with an
explicit evaluator-backed reconstruction and a bounded field repair pass.
Keep imported external STL repair as a separate mesh-only path.

Acceptance criteria:

- Original and repaired derived meshes remain independently available.
- Repair does not delete disconnected components or invent long bridges.
- Small bounded gaps can be field-repaired within the selected tolerance.
- Before/after topology status reports watertightness, winding consistency, and component count.
- Regression fixtures prove the old repair failure now has an effective outcome or an explicit diagnostic.

## Answer

Implemented evaluator-backed reconstruction, bounded field closing, and
separate original/repaired result selection. Numerical-fragment cleanup is a
separate explicit option: it removes only closed components below both
`export_tolerance^3` and one millionth of the main component volume, and reports
every removal. Tests prove a sub-tolerance artifact is removed while a real
independent component remains.
