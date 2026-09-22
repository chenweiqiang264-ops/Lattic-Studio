# Clip implicit previews on the GPU

The interactive section gizmo updates GPU clipping planes attached to implicit preview actors. Sections remain open and do not generate cap geometry, so plane translation or rotation does not recompute the implicit field or rebuild a triangle mesh.
