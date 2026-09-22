Status: resolved
Type: task
Blocked by: 01, 02

# Real Custom Cell Acceptance

Fill `resources/鞋底/1.stl` with the supplied `胞元.stl`, render the Cell Map and
result, reconstruct an STL, record timings and topology, and run regression
tests.

Acceptance criteria:

- Reproducible PNG, STL, and JSON artifacts are written under `results/`.
- The report includes source cleanup, backend, grid, timings, watertightness,
  winding, and connected-component count.

## Answer

Acceptance artifacts are under `results/custom_unit_cell_acceptance/`. The
supplied source is deterministically reduced from 383,648 to 383,644 faces,
is watertight and winding-consistent, and has one source component. With a
12 x 12 x 8 mm target Cell Map, 0.8 mm display spacing, and 12-way chunked STL
reconstruction, the RTX 4060 run took 4.74 s to prepare the source, 4.44 s for
implicit generation, and 29.06 s for STL reconstruction. A repeated run took
4.91 s, 4.53 s, and 30.04 s respectively and reproduced the same topology.

The exported result has 2,010,184 faces and is watertight and
winding-consistent. It has eight connected components: the main component
contains 99.773% of all faces and the remaining seven are small solids created
where cells are clipped by the design-domain boundary. They are reported and
preserved rather than silently removed.
