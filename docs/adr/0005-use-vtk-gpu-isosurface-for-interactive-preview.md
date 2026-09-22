# Use VTK GPU isosurface rendering for the interactive preview

The first interactive implicit renderer samples any implicit evaluator into a VTK image-data cache and renders its zero isosurface with the VTK GPU volume mapper. The backend remains behind the implicit-view boundary so a later precise renderer or custom shader can replace it without changing implicit geometry ownership or mesh export.
