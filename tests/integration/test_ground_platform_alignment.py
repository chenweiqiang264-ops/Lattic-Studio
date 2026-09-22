"""Regression tests for keeping the render grid under the visible mesh."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

TESTS_DIR = Path(__file__).resolve().parent
if str(TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(TESTS_DIR))

from lattice_studio.presentation.qt.tools.interactive_section_viewer import MeshData, PyVistaRenderer


class _GroundPlatformHarness:
    def __init__(self, mesh: MeshData) -> None:
        self.meshes = [mesh]
        lower = mesh.vertices.min(axis=0)
        upper = mesh.vertices.max(axis=0)
        self._bbox_center = (lower + upper) * 0.5
        self._bbox_size = max(float(np.max(upper - lower)), 1.0)
        self.show_ground_plane = False
        self.show_ground_grid = True
        self._ground_grid_actor = None
        self.added_datasets = []

    def add_mesh(self, dataset, **_kwargs):
        self.added_datasets.append(dataset)
        return dataset


def test_ground_grid_uses_visible_mesh_minimum_z() -> None:
    mesh = MeshData(
        vertices=np.array(
            (
                (10.0, -4.0, 20.0),
                (14.0, -4.0, 20.0),
                (10.0, 2.0, 32.0),
            ),
            dtype=float,
        ),
        faces=np.array(((0, 1, 2),), dtype=int),
    )
    harness = _GroundPlatformHarness(mesh)

    PyVistaRenderer._add_ground_platform(harness)

    grid_bounds = np.asarray(harness.added_datasets[0].bounds, dtype=float)
    assert grid_bounds[4] == 20.0
    assert grid_bounds[5] == 20.0
