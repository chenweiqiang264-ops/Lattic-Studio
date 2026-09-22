Status: resolved
Type: task

# Fitted Cell Map

Add an nTop-style regular Cell Map whose outer bounds match the design-domain
bounding box. Treat user spacing as a target, derive uniform actual spacing
from integer cell counts, and use the map's actual spacing and origin for TPMS
period and phase. Preserve complete-cell padding as an explicit compatibility
mode.

Acceptance criteria:

- Fitted map bounds equal the design-domain bounds on all axes.
- Every axis contains a positive integer number of uniform cells.
- UI reports target size, actual size, and cell count.
- G and D use independent fitted maps in transition generation.
- Cell Map preview grid lines represent the actual TPMS period and phase.
- `1.stl` G/D output remains watertight and singly connected.

## Answer

Implemented fitted and complete-cell boundary modes. Fitted mode selects the
nearest positive integer counts, derives uniform actual spacing, and aligns the
CPU/CUDA TPMS phase to the map origin. G and D retain independent maps during
transition evaluation. The UI defaults to fitted mode and reports target and
actual sizes. On `1.stl`, G produced a `9 x 11 x 3` map and D produced a
`4 x 21 x 2` map; both bounds exactly equal the design-domain bounds, and both
exported STL files reload as watertight, winding-consistent, single-component
meshes.
