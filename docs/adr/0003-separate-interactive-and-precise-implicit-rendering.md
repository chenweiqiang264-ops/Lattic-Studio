# Separate interactive and precise implicit rendering

Implicit display uses two paths: a bounded-fidelity interactive view for responsive camera and section-plane manipulation, followed later by a separate precise render for high-confidence visual inspection. This keeps display cost independent from mesh export quality while making ordinary preview artifacts distinguishable from defects in the authoritative implicit geometry.
