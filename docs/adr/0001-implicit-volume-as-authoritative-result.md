# Use the implicit volume as the authoritative generated result

Generated TPMS results are owned as implicit volumes because interactive display and section inspection must not depend on high-polygon STL meshes. Surface meshes are derived on demand for mesh-specific operations and STL export; imported design-domain meshes remain source inputs from which display and computation fields may be derived and cached.
