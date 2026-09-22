"""Performance regressions for Field Viewer hover probing."""

from __future__ import annotations

from types import SimpleNamespace
import unittest

import numpy as np

from lattice_studio.engine.implicit.field_viewer import FieldPlaneSample, FieldPlaneState
from lattice_studio.presentation.qt.viewers.pyvista_viewer import FieldPlaneRenderData, PyVistaRenderer


class _Picker:
    def __init__(self) -> None:
        self.pick_count = 0

    def Pick(self, *_args) -> None:
        self.pick_count += 1


class FieldViewerHoverTests(unittest.TestCase):
    def test_hover_uses_known_plane_geometry_without_vtk_cell_pick(self) -> None:
        """A dense field plane must not be cell-picked on every mouse move."""

        state = FieldPlaneState(
            center_mm=np.zeros(3),
            u_axis=np.array((1.0, 0.0, 0.0)),
            v_axis=np.array((0.0, 1.0, 0.0)),
            width_mm=2.0,
            height_mm=2.0,
        )
        sample = FieldPlaneSample(
            state=state,
            values=np.zeros((2, 2), dtype=np.float32),
            points=np.zeros((2, 2, 3), dtype=np.float64),
            source_bounds_mm=np.array(((-1.0, -1.0, -1.0), (1.0, 1.0, 1.0))),
        )
        data = FieldPlaneRenderData(
            sample=sample,
            colormap="implicit",
            value_range=(-1.0, 1.0),
            opacity=0.7,
        )
        picker = _Picker()
        queued: list[tuple[np.ndarray, float, tuple[int, int]]] = []
        viewer = SimpleNamespace(
            _field_plane_data=data,
            _field_plane_picker=picker,
            _field_plane_actor=object(),
            interactor=SimpleNamespace(GetEventPosition=lambda: (12, 18)),
            _field_plane_point_from_display=lambda _position, _data: np.zeros(3),
            _field_plane_value_at=lambda _data, _point: 0.25,
            _queue_field_plane_probe_update=lambda point, value, position: queued.append(
                (point, value, position)
            ),
            _clear_field_plane_probe=lambda: self.fail("valid plane point was cleared"),
        )

        PyVistaRenderer._on_field_plane_mouse_move(viewer, None, None)

        self.assertEqual(picker.pick_count, 0)
        self.assertEqual(len(queued), 1)
        np.testing.assert_allclose(queued[0][0], (0.0, 0.0, 0.0))
        self.assertEqual(queued[0][1:], (0.25, (12, 18)))


if __name__ == "__main__":
    unittest.main()
