Status: needs-triage
Type: task

# Promote the intermediate TPMS workbench

## Problem

The complete G/D/custom-cell/transition workflow currently lives in
`tests/intermediate_tests/tpms_filling_app.py`, a roughly 6,900-line module
containing parameter models, sampling, reconstruction, repair, workers, UI,
and export orchestration. The root `ui_app.py` uses the enhanced renderer but
does not contain the complete new workflow.

The current code is suitable for experimentation and acceptance testing but is
not yet a stable application boundary or reusable public API.

## Decision required

Choose one product direction before implementation:

1. integrate the TPMS workbench into the existing `ui_app.py`; or
2. promote it to a new formal application entry point and share only renderer,
   project, and domain services with the legacy app.

## Proposed module seams

- generation/application service;
- display-sampling service;
- STL reconstruction/export service;
- Qt worker/controller layer;
- parameter/view-model layer;
- thin window composition.

## Acceptance criteria

- No production application code remains under `tests/`.
- Existing 166 intermediate regression tests remain valid or are migrated
  without reducing coverage.
- Loading a new design domain invalidates all dependent implicit, mesh, Cell
  Map, and render-cache layers.
- The promoted entry point has a documented launch command and dependency set.
- Existing user files and unrelated dirty-worktree changes are preserved.

