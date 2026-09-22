Status: resolved
Type: task

# Connect real G/D fields to conservative adaptive extraction

The previous tests covered only synthetic spheres and UI parameter forwarding.
Real G/D bodies had no `surface_exclusion`, so automatic extraction always
fell back to a dense grid.

## Answer

Implemented a sign-equivalent TPMS shell exclusion field from
`abs(f-level) - wall/2 * max(norm(physical_gradient), epsilon)`.  Its global
gradient/Hessian bounds are used to normalize the field to Lipschitz one.
The intersection exclusion combines that field with a conservatively scaled
design-domain SDF.  Automatic extraction permits a bounded 10% sampling
overhead when it buys lower-peak adaptive processing, while retaining dense
fallback for materially worse plans.

Regression coverage uses the repaired `resources/鞋底/1.stl` fixture and
compares real G and D adaptive output with dense output.
