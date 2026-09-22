# Use Topology-Constrained Provider-Agnostic Transitions

## Context

The existing G-to-D transition computes two signed-distance-like shell fields
and blends them with a sigmoid weight before intersecting the result with the
design-domain field. A smooth scalar weight does not guarantee that material
features meet across the transition plane, retain a minimum thickness, or
avoid isolated zero-level components. Different Cell Map periods, Frames, and
phases make those failures more likely. Increasing display or export sampling
can reveal the geometry more accurately, but cannot correct a disconnected
authoritative field.

The pair-specific API also prevents the same transition workflow from being
used by custom STL cells and future Unit-cell providers.

## Decision

Transitions operate on two common Unit-cell provider operands. G, D, and
custom STL providers are the initial supported operands; equal provider types
with different parameters or Cell Maps are valid. New providers integrate
through evaluator capabilities rather than pair-specific transition branches.

The transition pipeline has three distinct stages:

1. A base morphology blend uses automatic weighting by default. Advanced
   selection supports linear (C0), smoothstep (C1), smootherstep (C2),
   sigmoid/tanh, and cosine curves. Like nTopology's Ramp block, plane distance
   is mapped across a finite input range and clamped to the endpoint values
   outside it. Physical transition width is the only normal-mode shape control;
   sigmoid sharpness appears only when that advanced curve is selected.
2. Inside the transition band, automatic local registration may smoothly
   adjust Cell Map coordinates to reconcile period, Frame, and phase mismatch.
   A topology-constrained operator may then add or remove bounded material to
   establish cross-band connectivity, enforce the configured transition
   minimum feature thickness, and eliminate isolated fragments.
3. Independent adaptive validation checks those constraints directly against
   the authoritative implicit evaluator. Display quality and STL export
   tolerance do not determine transition acceptance.

Each source evaluator remains exactly unchanged outside the transition band,
and local registration converges to the corresponding original Cell Map at
the band boundaries. Transition validation guarantees only cross-band
connectivity; it reports whole-result component count without modifying
out-of-band geometry to force a globally single component.

Transition minimum feature thickness is independently configurable. Its
automatic default is the smaller characteristic thickness of the operands.
The system recommends a transition width from cell dimensions, Frame mismatch,
and feature thickness, but accepts narrower values without overriding them.

Execution is capability-driven. GPU evaluation is preferred for operands that
support it; otherwise the system uses batched CPU. The initial generic
implementation composes Ramp weights and topology correction on CPU arrays,
so even two GPU operands report `GPU operand evaluation + CPU Ramp/topology`
rather than claiming full device residency. A result that still fails a
transition constraint remains inspectable and exportable. Failures are shown
as lightweight inline status and export metadata, never as a modal warning.

## Consequences

Weight curves become controllable morphology tools rather than implicit claims
of structural validity. Transition quality is reproducible across display and
STL settings, and the same architecture supports analytic and geometry-backed
unit cells.

The implementation is more expensive than weighted field interpolation. It
requires provider capability metadata, compact-support coordinate warping,
topology-aware local correction, adaptive validation, and deterministic
failure reporting. Custom STL combinations may execute more slowly when a GPU
evaluator is unavailable. Exact source preservation at band boundaries also
limits how aggressively local registration can resolve a very narrow or highly
mismatched transition.
