# Author implicit bodies as evaluators

An implicit body's authoritative representation is a bounded point evaluator rather than a dense voxel array. Display and mesh extraction consume disposable sampled-field caches at their own resolutions, which prevents geometric authority from being tied to one voxel size and allows large evaluations to remain chunked.
