Status: resolved
Type: task

# Prepare Periodic Custom Unit Cell

Implement the deep custom-unit-cell module. It owns STL import, deterministic
cleanup, source-domain validation, an independent resident geometry backend,
periodic Cell Map evaluation, and design-domain implicit intersection.

Acceptance criteria:

- Public callers use two entry points rather than geometry-backend details.
- Unit tests cover cleanup, rejection, periodicity, scaling, Frame rotation,
  and implicit intersection.
- Mesh memory is independent of Cell Map cell count.

## Answer

Implemented `core.implicit.custom_unit_cell` with two public entry points,
deterministic STL cleanup, explicit source Domain validation, a private
resident CPU/CUDA geometry backend, periodic UVW evaluation, conservative
non-uniform distance scaling, and design-domain intersection. Six behavioral
tests cover the public interface. The supplied cell is cleaned to 383,644
watertight faces and one connected component.
