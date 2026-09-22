# Support Fitted and Padded Cell Maps

The lattice pipeline supports a fitted regular Cell Map whose bounds equal the
design-domain bounding box and a compatibility mode that pads to the requested
cell size. In fitted mode, requested sizes select the nearest positive integer
cell counts and the resulting uniform actual sizes and map origin drive the
lattice evaluator; this preserves complete regular cells and phase alignment
without paying for padded sampling space.
