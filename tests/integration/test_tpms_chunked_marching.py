"""Behavioral tests for memory-bounded marching-cubes extraction."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

TESTS_DIR = Path(__file__).resolve().parent
if str(TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(TESTS_DIR))

from lattice_studio.presentation.qt.workbench import marching_cubes_chunked


def _sphere_field(shape: tuple[int, int, int]) -> np.ndarray:
    axes = [np.linspace(-4.0, 4.0, size, dtype=np.float32) for size in shape]
    x, y, z = np.meshgrid(*axes, indexing="ij")
    return (x * x + y * y + z * z - 8.0).astype(np.float32)


def test_chunked_marching_cubes_stitches_shared_boundary_vertices():
    shape = (18, 20, 22)
    spacing = (0.5, 0.5, 0.5)
    origin = np.array((-4.0, -4.0, -4.0), dtype=float)
    source = _sphere_field(shape)

    single_calls: list[tuple[int, int]] = []

    def single_provider(start: int, end: int) -> np.ndarray:
        single_calls.append((start, end))
        return source[start : end + 1]

    single = marching_cubes_chunked(
        shape,
        spacing,
        origin,
        batch_count=1,
        sample_chunk=single_provider,
    )

    chunk_calls: list[tuple[int, int]] = []

    def chunk_provider(start: int, end: int) -> np.ndarray:
        chunk_calls.append((start, end))
        return source[start : end + 1]

    chunked = marching_cubes_chunked(
        shape,
        spacing,
        origin,
        batch_count=4,
        sample_chunk=chunk_provider,
    )

    assert single_calls == [(0, shape[0] - 1)]
    assert len(chunk_calls) == 4
    assert all(end > start for start, end in chunk_calls)
    assert all(end - start + 1 < shape[0] for start, end in chunk_calls)
    assert chunked.is_watertight
    assert len(chunked.faces) == len(single.faces)
    assert np.isclose(chunked.volume, single.volume, rtol=1e-6, atol=1e-6)
