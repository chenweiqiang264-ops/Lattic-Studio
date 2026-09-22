# Ship automatic NVIDIA acceleration with CPU fallback

The Windows distribution will detect compatible NVIDIA CUDA devices at runtime, carry the CUDA runtime components required by the numerical kernels, and select acceleration without requiring users to install Python or the CUDA Toolkit. Unsupported devices and runtime failures fall back to the CPU adapter with a visible reason; AMD and Intel GPUs may render through OpenGL but are not numerical compute targets in the first packaged release.

