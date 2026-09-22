# Extract domain models

Type: task
Status: resolved
Blocked by: 01

Move parameter, result, target, and quality models behind domain-owned modules without importing Qt or platform backends.

## Comments

The parameter, Cell Map, gradient, transition, and workspace models now live
under `src/lattice_studio/domain/`. The workspace aggregate is persistence
agnostic; geometry implementations are in `engine/design_domains.py`, and JSON
schema/asset handling is in `infrastructure/workspace_persistence.py`. The old
`core.implicit.design_workspace` path is now a compatibility facade only.

## Answer

Completed the domain extraction and added unit coverage for workspace
isolation, managed assets, runtime-cache exclusion, and v1 manifest migration.
