# Establish the package and behavioural baseline

Type: task
Status: resolved

Create the installable package skeleton, define the supported entry points, capture dependency and test baselines, and keep the current workbench operational while migration starts.

## Acceptance

- `lattice_studio` imports without Qt initialization.
- `python -m lattice_studio` owns runtime startup.
- Existing application can still be launched during migration.
- Baseline tests and import consumers are recorded.

## Comments

## Answer

Created the `src/lattice_studio` package, module and console entry points, a
Qt-free public import, and a temporary compatibility module at the historical
test path. The production workbench now lives under the package, the root
launcher delegates to it, and the package/entrypoint baseline tests pass.
