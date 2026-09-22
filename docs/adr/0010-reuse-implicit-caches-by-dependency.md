# Reuse implicit caches by dependency

Design-domain evaluation and each G, D, or transition display cache are tracked as separate dependencies. A domain change invalidates all dependents; a lattice-specific or transition-specific parameter change invalidates only the affected result, preserving reusable evaluation and display data elsewhere.

Each lattice generation result also retains its unclipped feature evaluator. Shell/lattice fusion coordinates that evaluator with one design-domain evaluation per point batch, then reuses the resulting domain values for lattice clipping, shell construction, and the final domain restriction. This avoids repeated mesh-SDF queries without making a sampled display field authoritative.
