# Migrate Remaining Workbench Workflows

Type: task
Status: needs-triage
Priority: deferred

## Problem

The task protocol and workspace command client now cover the supported
generation workflows and editable document mutations. The Qt workbench still
contains legacy direct-worker compatibility paths, which must either be
removed after representative UI coverage or explicitly retained as a product
decision.

## Acceptance Criteria

- Each workflow is a whitelisted backend task with serializable input,
  progress, cancellation, a result handle, and derived artifacts where needed.
- Qt consumes handles and artifacts rather than direct engine results.
- No authoritative evaluator crosses HTTP.
- Add a contract test for each new task and a representative workbench
  integration test.
- Rebuild and run `packaging/verify_installer.ps1` after UI migration.

## Remaining Scope

- Remove legacy direct-worker fallback once the replacement paths have
  representative Qt integration coverage.

## Deferral Decision

The remaining direct-worker compatibility paths are architectural debt, not a
current release blocker. The workspace command path, backend task contract,
isolated installer runtime check, GPU path, CPU fallback, Qt lifecycle, and
uninstall validation have passed.

Defer this work until one of these conditions applies:

- a remote, headless, multi-client, or automated client is required;
- duplicate Qt/backend result state causes a reproducible defect;
- a new numerical workflow would otherwise add another in-process worker; or
- the team schedules a dedicated architecture-maintenance iteration.

When resumed, preserve Qt-only presentation caches but remove their ability to
act as numerical inputs or editable-state authority. The backend must own
document definitions, generation handles, and derived artifacts; Qt must use
DTOs, snapshots, and display artifacts only.

## Comments

2026-09-22: Initial protocol and TPMS-to-STL vertical path delivered. See ADR
0035 for the selected boundary.

2026-09-22: Added whitelisted task handlers for custom cells, plane-driven
transitions, shell/fusion, display refinement, and precise rendering. The Qt
workbench now uses backend generation handles for supported mesh-domain
single-result paths and their downstream operations. Analytic domains,
field-driven transitions, combined requests, and workspace mutation remain
open.

2026-09-22: Added `generation.batch`, serializable analytic-domain DTOs and
field-driver primitive DTOs. The workbench now routes normal combined and
analytic/field-driven generation requests to backend tasks. Added atomic
`workspace.commands` and `workspace.snapshot` endpoints with document create,
replace, lifecycle, settings, and field-object mutations. Qt has not yet been
changed to use those document endpoints as its editable projection.

2026-09-22: Qt now creates a private backend workspace and commits editable
document operations through `workspace.commands` before changing its local
projection. Save/load reconciles through the backend snapshot. An analytic
domain's dual-role field primitive is included in its initial atomic command
batch; the field-visibility regression covers both local and backend state.
The onedir and installer were rebuilt, then `verify_installer.ps1` passed GPU,
forced CPU fallback, window lifecycle, and genuine uninstall validation in
`build/installer-acceptance-workspace-projection-20260922-145200`.

2026-09-22: Maintainer decision: defer removal of the remaining in-process
compatibility workers. Status changed to `needs-triage`; this is not a current
functional or packaging blocker. Resume only when a stated deferral trigger
applies, then implement the remaining ownership split with dedicated Qt and
transport coverage.
