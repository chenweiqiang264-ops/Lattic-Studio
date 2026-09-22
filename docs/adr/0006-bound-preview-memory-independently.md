# Bound preview memory independently from generation

Interactive sampled-field caches obey a separate display memory budget and may automatically use a coarser display voxel size when necessary. Authoritative implicit evaluation and mesh-export resolution remain unaffected, so preview safeguards do not reintroduce a maximum voxel count into generation or export.
