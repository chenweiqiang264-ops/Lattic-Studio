# Migrate Remaining Workbench Workflows

Type: task
Status: ready-for-agent

## Problem

The task protocol has a tested TPMS-to-STL chain, but the Qt workbench still
executes custom-cell, transition, shell/fusion, display refinement, precise
rendering, and runtime workspace mutation in-process.

## Acceptance Criteria

- Each workflow is a whitelisted backend task with serializable input,
  progress, cancellation, a result handle, and derived artifacts where needed.
- Qt consumes handles and artifacts rather than direct engine results.
- No authoritative evaluator crosses HTTP.
- Add a contract test for each new task and a representative workbench
  integration test.
- Rebuild and run `packaging/verify_installer.ps1` after UI migration.

## Comments

2026-09-22: Initial protocol and TPMS-to-STL vertical path delivered. See ADR
0035 for the selected boundary.
