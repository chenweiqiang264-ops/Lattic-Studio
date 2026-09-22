"""Regression test for replacing the TPMS design-domain STL."""

from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import trimesh

from lattice_studio.application.workspace import (
    ManagedDesignWorkspace as DesignWorkspace,
    MeshDesignDomain,
)

TESTS_DIR = Path(__file__).resolve().parent
if str(TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(TESTS_DIR))

from lattice_studio.presentation.qt import workbench as app


class _FakeValueControl:
    def __init__(self, value: float) -> None:
        self._value = float(value)

    def value(self) -> float:
        return self._value


class _FakeComboBox:
    def __init__(self, current_data: str) -> None:
        self.items = ("fit_bounds", "complete_cells")
        self.current_index = self.items.index(current_data)
        self.enabled = True
        self.tooltip = ""

    def findData(self, value: str) -> int:
        return self.items.index(value) if value in self.items else -1

    def currentIndex(self) -> int:
        return self.current_index

    def setCurrentIndex(self, index: int) -> None:
        self.current_index = int(index)

    def blockSignals(self, _blocked: bool) -> None:
        pass

    def setEnabled(self, enabled: bool) -> None:
        self.enabled = bool(enabled)

    def setToolTip(self, tooltip: str) -> None:
        self.tooltip = tooltip


def _frame_controls(origin, u=(1.0, 0.0, 0.0), v=(0.0, 1.0, 0.0)):
    return {
        "origin": [_FakeValueControl(value) for value in origin],
        "u": [_FakeValueControl(value) for value in u],
        "v": [_FakeValueControl(value) for value in v],
    }


def test_loading_a_new_domain_discards_previous_generated_layers():
    """Replacing a Design domain clears results tied to its old geometry."""

    workspace = DesignWorkspace()
    document = workspace.create_mesh_document(
        trimesh.creation.box(extents=(10.0, 10.0, 2.0)),
        "original",
    )
    new_domain = trimesh.creation.box(extents=(2.0, 3.0, 1.0))
    document.runtime.results = {"G": new_domain.copy()}
    document.runtime.raw_results = {"G": new_domain.copy()}

    document.replace_domain(MeshDesignDomain(new_domain, "replacement"))

    assert document.runtime.results == {}
    assert document.runtime.raw_results == {}
    assert np.array_equal(document.domain.preview_mesh().extents, (2.0, 3.0, 1.0))


def test_default_frame_accepts_origin_precision_exposed_by_spin_boxes():
    window_class = app._build_qt_app()
    window = window_class.__new__(window_class)
    window.design_workspace = DesignWorkspace()
    window.sole_mesh = trimesh.creation.box(extents=(2.0, 3.0, 1.0))
    window.sole_mesh.apply_translation((-26.8224697113, -83.4561157227, 0.5))
    displayed_origin = np.round(window.sole_mesh.bounds[0], decimals=4)
    boundary_mode = _FakeComboBox("complete_cells")
    window.g_widgets = {
        "frame_inputs": _frame_controls(displayed_origin),
        "cell_map_mode": boundary_mode,
    }

    assert window._frame_is_default("G") is True
    window._sync_frame_boundary_mode("G")
    assert boundary_mode.enabled is True


def test_custom_frame_still_forces_complete_cell_coverage():
    window_class = app._build_qt_app()
    window = window_class.__new__(window_class)
    window.design_workspace = DesignWorkspace()
    window.sole_mesh = trimesh.creation.box(extents=(2.0, 3.0, 1.0))
    custom_origin = window.sole_mesh.bounds[0] + (0.001, 0.0, 0.0)
    boundary_mode = _FakeComboBox("fit_bounds")
    window.g_widgets = {
        "frame_inputs": _frame_controls(custom_origin),
        "cell_map_mode": boundary_mode,
    }

    window._sync_frame_boundary_mode("G")

    assert window._frame_is_default("G") is False
    assert boundary_mode.enabled is False
    assert boundary_mode.items[boundary_mode.current_index] == "complete_cells"
