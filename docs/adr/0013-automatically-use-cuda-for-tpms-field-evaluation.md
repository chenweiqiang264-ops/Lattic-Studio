# Automatically use CUDA for TPMS field evaluation

All supported TPMS families, sigmoid transition fields, and optional gradient
TPMS fields use an automatic numerical backend. A Numba CUDA adapter is
selected when a working CUDA device is available; otherwise the same request
uses the NumPy CPU adapter. Runtime CUDA failures permanently switch the
process to CPU and retain the failure reason for the user interface. Invalid
caller input propagates without changing backend state. CUDA is enabled by
default after a crash-isolated child-process probe; set
`TPMS_DISABLE_NUMBA_CUDA=1` when a machine must force the CPU path. The legacy
`TPMS_ENABLE_NUMBA_CUDA=0` setting remains an explicit CPU override.

GPU work is bounded by the existing micro-slice size so device memory does not
scale with the complete sampled field. Linear, power, sigmoid, and layered
gradient profiles support both gradient-normalized physical thickness and the
MATLAB-compatible field-threshold method. Mesh-clipped TPMS requests use the
resident pipeline in ADR-0031; standalone geometry operations and Marching
Cubes use the automatic geometry backend described in ADR-0015.
