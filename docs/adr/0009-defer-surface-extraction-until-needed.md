# Defer surface extraction until needed

Generation returns evaluator-backed implicit results and builds only the sampled cache required for interactive display. Marching Cubes is deferred until STL export, mesh inspection, simplification, or another operation explicitly requires a triangle mesh.
