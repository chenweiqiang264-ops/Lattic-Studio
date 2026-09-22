# Custom Unit Cell

## Notes

Add nTop-style non-parametric unit cells prepared from an STL body and a
three-dimensional source domain. The prepared body is evaluated periodically
through the existing Cell Map; triangles are never copied per map cell.

## Decisions-so-far

- The public workflow has two entry points: prepare a custom unit cell, then
  fill it through a Cell Map and a design-domain implicit body.
- The source mesh bounding box is the default unit-cell domain.
- Import performs only deterministic cleanup. Inputs that remain non-watertight
  are rejected because their signed field is not reliable.
- Source X/Y/Z map to Cell Map U/V/W. The source geometry owns its thickness.
- The first implementation keeps an independent CPU/CUDA BVH resident for the
  prepared cell. A sampled template may replace this implementation later
  without changing the public interface.
- Workbench integration and real `胞元.stl` / `1.stl` acceptance are recorded
  in [02](issues/02-integrate-custom-cell-workbench.md) and
  [03](issues/03-real-custom-cell-acceptance.md).
- Rotated custom-cell display sampling now uses the trimmed design-domain
  bounds; diagnosis and acceptance evidence are recorded in
  [04](issues/04-render-rotated-custom-cell.md).
- Optional custom-cell target feature thickness is implemented as a
  world-space signed-distance offset; interface and acceptance evidence are
  recorded in [05](issues/05-configurable-feature-thickness.md).
- Custom-field display and STL reconstruction now trim external work, preserve
  Cell Map sampling phase, use conservative domain-exterior exclusion, and
  schedule larger CUDA bricks; evidence is recorded in
  [06](issues/06-optimize-custom-field-and-stl-reconstruction.md).
- Repeated bead-like particles in small custom cells are traced to CUDA SDF
  sign rays moving near-surface query points; remediation is tracked in
  [07](issues/07-eliminate-periodic-sdf-speckles.md).

## Frontier

- [01](issues/01-prepare-periodic-custom-unit-cell.md): resolved
- [02](issues/02-integrate-custom-cell-workbench.md): resolved
- [03](issues/03-real-custom-cell-acceptance.md): resolved
- [04](issues/04-render-rotated-custom-cell.md): resolved
- [05](issues/05-configurable-feature-thickness.md): resolved
- [06](issues/06-optimize-custom-field-and-stl-reconstruction.md): resolved
- [07](issues/07-eliminate-periodic-sdf-speckles.md): resolved
- [08](issues/08-periodic-seam-bridging.md): resolved
