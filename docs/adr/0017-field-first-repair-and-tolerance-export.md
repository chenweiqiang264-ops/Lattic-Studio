# Use Field-First Repair and Tolerance-Based Export

Generated lattice results are repaired and exported by reconstructing from the
authoritative implicit evaluator at an export-specific millimetre tolerance;
interactive display caches are not export sources. Mesh repair remains separate
for imported STL files, while explicit numerical-fragment cleanup may remove
only closed components below both the tolerance scale and a strict main-volume
ratio. Topology checks warn without blocking confirmed STL export. When
compiled simplification introduces a topology defect, the candidate first goes
through conservative mesh repair and tolerance-bounded numerical-fragment
cleanup. The simplified candidate is accepted only after topology and
geometric-deviation checks pass; unsuccessful repair still rolls back to the
source mesh.
