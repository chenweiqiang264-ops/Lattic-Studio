# Cache implicit render layers without camera LOD

Sampled implicit fields retain their VTK image, GPU volume mapper, and actor
while hidden. Re-showing a field, changing its display material, changing the
render-quality preset, or temporarily switching to a mesh view reuses those
resources instead of rebuilding and uploading the scalar grid. The renderer
owns cache identity, hit/miss diagnostics, and least-recent inactive eviction.
The user-selected display-memory budget bounds inactive resident layers; active
layers are never evicted while visible.

STL camera rotation and panning keep the target-quality resident mesh actor.
They do not build or switch to a separate interaction-only LOD. Interactive
section inspection remains responsive by changing clipping planes directly on
the existing polygon or implicit mapper, without rebuilding mesh geometry or
resampling an implicit field.

This decision supersedes only the camera-interaction LOD paragraph in ADR-0019.
Its topology-preserving display proxy, immutable cache-key, and bounded resident
mesh-layer decisions remain in force.
