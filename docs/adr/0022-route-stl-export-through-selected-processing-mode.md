# Route STL export through the selected processing mode

## Context

The UI exposed three field-processing modes, but STL reconstruction used a
separate implicit-surface extractor and therefore ignored that selection.
This made the control misleading in the main workflow.

## Decision

STL reconstruction receives the selected `processing_mode` and `batch_count`
from the UI and routes STL generation through the same three strategies used
by direct mesh generation:

- `single_pass`: complete field and one Marching Cubes call;
- `batched_field`: slab evaluation, complete field retained, one Marching
  Cubes call;
- `chunked_marching_cubes`: slab evaluation, one Marching Cubes call per slab,
  followed by shared-edge welding and mesh cleanup.

Global implicit-field repair still requires a complete regular field.  If
repair is requested, chunked extraction is safely replaced by the full-field
repair path and the result reports that fallback reason.

## Consequences

The UI's processing-mode selection now affects STL output.  Chunked STL
generation has lower scalar-field peak memory but pays for repeated shared
boundary samples and final mesh welding.  Global field repair retains its
existing safety constraints.
