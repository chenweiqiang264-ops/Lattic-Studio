# Extract engine workflows

Type: task
Status: claimed
Blocked by: 02

Move sampling, reconstruction, repair, Cell Map, lattice generation, transition, and shell workflows into cohesive engine modules with small supported interfaces.

## Comments

Initial extraction is complete: the numerical implementation is under
`src/lattice_studio/engine/workflows.py`, while shared result and quality
contracts now live in `engine/contracts.py`. Design-domain adapters are in
`engine/design_domains.py`. Remaining work is to split sampling, reconstruction,
mesh conditioning, and shell/transition modules without changing numerical
contracts.

The first contract migration is verified by the TPMS, gradient, transition,
shell, Cell Map, workspace, and processing-mode suites (`85 passed` in the
focused run). `workflows.py` re-exports the contract classes during migration.

Mesh validation/repair and field-fragment cleanup now live in
`engine/mesh_quality.py`. STL reconstruction grid planning now lives in
`engine/reconstruction_grid.py`; both modules contain the implementations and
are consumed directly by the Qt adapter.
