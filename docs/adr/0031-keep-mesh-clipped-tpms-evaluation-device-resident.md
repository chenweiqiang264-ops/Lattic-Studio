# Keep mesh-clipped TPMS evaluation device-resident

Mesh-backed TPMS generation uses one deep compute operation for design-domain
SDF evaluation, ordinary or gradient TPMS evaluation, and implicit
intersection. Its host-facing contract accepts points and returns only the
final clipped scalar field.

The CUDA adapter uploads each bounded point tile once, evaluates the cached
mesh BVH and TPMS kernels on the ordered default CUDA stream, intersects both
fields on the device, and downloads only the final result. The design-domain
BVH remains cached across display and extraction micro-slices. This removes the
previous TPMS download, SDF download, and two-field re-upload before clipping.

The CPU adapter implements the same contract and is the reference result. A
CUDA execution failure selects the CPU adapter for later requests and records
the reason for the UI. `TypeError` and `ValueError` identify invalid requests;
they propagate without changing backend state. Explicitly disabling accelerated
SDF evaluation retains the existing PyVista and NumPy path.
