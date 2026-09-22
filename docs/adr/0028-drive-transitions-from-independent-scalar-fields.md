# Drive Transitions From Independent Scalar Fields

## Context

The transition pipeline accepts a plane point and plane normal directly. This
hard-codes one geometric field type and prevents a transition from being
defined by independently editable spatial objects or later analysis fields.
At the same time, overloading the transition operands with this responsibility
would make lattice sources, driver fields, and display objects indistinguishable.

## Decision

A transition has an explicit region source. The existing plane remains one
source and retains its current behaviour. A selected transition driver field is
the other source. Its ordered scalar interval defines the exact two Ramp
endpoints: values at or below the lower endpoint select the negative-side
operand; values at or above the upper endpoint select the positive-side
operand; values between them receive the configured continuous Ramp weight.

Analytic spheres, cylinders, and boxes are initial transition driver fields.
They are persistent implicit primitives in an independent field-object scene,
not lattice operands or generated results. Each owns a standard signed-distance
evaluator, an editable world transform, and an unclipped display preview. A
transition only references one object by stable identifier, so editing or
selecting a different object updates the dependent transition without taking
ownership of the object. The design-domain evaluator remains the sole final
spatial restriction on generated material.

## Consequences

Plane-defined transitions remain compatible, while a driver can be inspected,
reused, and later replaced by stress, temperature, or other scalar fields.
The current field viewer can inspect a primitive's authoritative SDF before
generation. The transition core must accept a scalar-driver abstraction rather
than assuming a signed plane distance, and dependency invalidation must track
referenced field-object changes.

The first version deliberately supports one selected driver field rather than
field Boolean graphs. Multi-field algebra is a future composition layer; it
must not be embedded as ad hoc transition-specific branches.
