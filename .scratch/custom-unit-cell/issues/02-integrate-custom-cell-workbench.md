Status: resolved
Type: task
Blocked by: 01

# Integrate Custom Cell Workbench

Add custom-cell import, parameters, Frame controls, Cell Map preview, generation,
visibility, materials, reconstruction, and export to the existing workbench.

Acceptance criteria:

- Custom is selectable independently from G, D, and Transition.
- Cell dimensions and UVW Frame are editable; no wall-thickness input is shown.
- Existing G/D behavior and processing modes remain unchanged.

## Answer

Added an independent custom-cell panel with STL preparation status, U/V/W
target cell dimensions, boundary mode, and shared Frame controls. Custom is
available in Cell Map preview, generation selection, implicit and STL
visibility, material selection, preflight inspection, tolerance-based STL
reconstruction, repair, simplification, and individual or bulk export. STL
reconstruction now consumes each implicit result's physical minimum feature
instead of a TPMS-only wall-thickness dictionary. Existing G/D behavior and
all three processing modes remain covered by the 115-test regression suite.
