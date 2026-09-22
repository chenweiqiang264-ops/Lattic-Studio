Status: resolved
Type: task

# Eliminate Periodic Custom-cell SDF Speckles

Small custom cells can repeat isolated floating particles or bead-like strings
through the implicit preview and derived mesh. The supplied source cell makes
the symptom denser because the same source-space samples recur in every Cell
Map period.

Diagnosis found that the CUDA mesh SDF offsets every +X sign ray in Y and Z.
For points close to a surface, that transverse offset can move the sign-query
origin across the surface. A structured cylinder reproduces the exact failure:
an exterior vertical sample line is classified as interior along its length.
Periodic evaluation then replicates that wrong sign in every cell.

Acceptance criteria:

- CUDA sign rays preserve the original query point.
- A structured cylindrical surface classifies the known exterior vertical
  sample line as outside without isolated negative samples.
- Known box distances and existing CPU/CUDA geometry tests still pass.
- The supplied custom cell no longer disagrees with the VTK reference at the
  previously captured near-surface probes.
- Small-cell reconstruction of the supplied custom cell does not contain
  sub-voxel components created by isolated SDF sign errors.
- Focused and complete intermediate regression suites pass.

## Comments

- The source cell also has point-only X/Y contacts and nonmatching Z faces.
  Those are separate authored-cell topology constraints and must not be hidden
  by deleting legitimate components.

## Answer

The CUDA SDF now keeps every sign ray anchored at the query point. It compares
two independent ray directions and uses a third-direction majority whenever
the first two disagree or encounter a triangle boundary. A structured 512-side
cylinder regression reproduces the former exterior-to-interior sign inversion
and now passes.

The supplied cell was checked on 520,251 regular source-space samples. CUDA and
the C++ BVH reference had zero finite-distance sign disagreements; hot timings
were 0.914 s and 12.045 s respectively. The four originally captured
near-surface GPU/VTK disagreements now match both VTK and C++.

Correct signs leave a second, distinct case at the design-domain boundary:
sub-voxel intersections between a periodic-face contact and the shoe surface.
Custom-cell display caches now inspect negative 26-neighbour components with at
most six samples, estimate each local volume with Marching Cubes, and remove it
only below both one display-voxel volume and one millionth of the main occupancy
proxy. This changes only the disposable display cache.

Real acceptance used `resources/鞋底/1.stl`, the supplied `胞元.stl`, fitted
4 x 4 x 2.5 mm cells, and a 0.25 mm display voxel:

- Raw display-derived mesh: 6,773,272 faces, watertight, winding-consistent,
  81 components, and 23 tolerance-bounded fragments.
- Cleaned display-derived mesh: 6,772,740 faces, watertight,
  winding-consistent, 58 components, and zero fragments on a second probe.
- Display cleanup removed exactly the 23 qualified islands and retained six
  small candidates whose reconstructed volumes exceeded the physical limit.
- Complete intermediate regression suite: 133 passed.
- Python compilation and whitespace checks passed; no debug instrumentation
  remains.
