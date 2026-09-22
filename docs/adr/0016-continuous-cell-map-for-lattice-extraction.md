# Use a Continuous Cell Map for Lattice Extraction

Status: superseded by ADR-0018

The lattice pipeline uses a regular cell map with complete-cell padding and a
single continuous implicit evaluator before one global Marching Cubes extraction.
Cell map cells are never independently meshed and concatenated, because that
would create coincident faces and seam topology defects; the design-domain SDF
performs the final spatial restriction.
