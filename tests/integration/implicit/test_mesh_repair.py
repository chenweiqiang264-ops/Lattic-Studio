"""Regression coverage for conservative mesh repair helpers."""

from __future__ import annotations

import unittest
from unittest import mock

import trimesh

from lattice_studio.presentation.qt.workbench import repair_mesh


class MeshRepairTests(unittest.TestCase):
    def test_repair_does_not_depend_on_trimesh_update_faces(self) -> None:
        """Avoid the third-party cache recursion raised by ``update_faces``."""

        source = trimesh.Trimesh(
            vertices=(
                (0.0, 0.0, 0.0),
                (1.0, 0.0, 0.0),
                (0.0, 1.0, 0.0),
                (0.0, 0.0, 1.0),
                (0.0, 0.0, 0.0),
            ),
            faces=((0, 1, 2), (0, 1, 2), (4, 1, 3), (0, 0, 1)),
            process=False,
        )

        with mock.patch.object(
            trimesh.Trimesh,
            "update_faces",
            side_effect=RecursionError("maximum recursion depth exceeded"),
        ):
            repaired = repair_mesh(source, fill_holes=False)

        self.assertEqual(len(repaired.faces), 2)
        self.assertEqual(len(repaired.vertices), 4)
        self.assertTrue(repaired.faces.dtype.kind in {"i", "u"})

    def test_repair_hole_closure_does_not_depend_on_trimesh_update_faces(self) -> None:
        """Keep the optional conservative hole closure on the safe path too."""

        source = trimesh.Trimesh(
            vertices=(
                (0.0, 0.0, 0.0),
                (1.0, 0.0, 0.0),
                (0.0, 1.0, 0.0),
                (0.0, 0.0, 1.0),
            ),
            faces=((0, 2, 1), (0, 1, 3), (1, 2, 3)),
            process=False,
        )

        with mock.patch.object(
            trimesh.Trimesh,
            "update_faces",
            side_effect=RecursionError("maximum recursion depth exceeded"),
        ):
            repaired = repair_mesh(source, fill_holes=True)

        self.assertTrue(repaired.is_watertight)
        self.assertTrue(repaired.is_winding_consistent)


if __name__ == "__main__":
    unittest.main()
