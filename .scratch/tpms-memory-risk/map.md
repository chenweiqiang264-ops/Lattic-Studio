# TPMS-002 Large-grid memory risk

## Notes

The generation pipeline now bounds point evaluation by micro-slices and supports batched field and chunked Marching Cubes modes. This reduces peak memory but does not eliminate allocations required by single-pass fields or by mesh extraction.

## Decisions-so-far

- GPU geometry work is tile-bounded, while the selected processing mode controls whether the complete scalar field is retained.
- The risk remains open until a streaming extraction design is validated against topology and export requirements.

## Frontier

- [02-large-grid-memory-strategy](issues/02-large-grid-memory-strategy.md)
