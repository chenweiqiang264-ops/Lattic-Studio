"""Regression tests for explicit implicit-body target selection."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from lattice_studio.engine.implicit.field import ImplicitBody
from lattice_studio.presentation.qt import workbench


class _TargetCombo:
    """Minimal combo-box substitute for target-selection method tests."""

    def __init__(self, value: str | None) -> None:
        self.value = value

    def currentData(self) -> str | None:
        return self.value


class _CheckBox:
    def __init__(self, checked: bool) -> None:
        self.checked = checked

    def isChecked(self) -> bool:
        return self.checked


def _body(name: str) -> ImplicitBody:
    return ImplicitBody(
        name=name,
        bounds=np.array([[0.0, 0.0, 0.0], [2.0, 2.0, 2.0]]),
        evaluate=lambda points, _stage=None: np.ones(len(points), dtype=np.float32),
    )


def test_stl_reconstruction_selection_returns_only_the_requested_body() -> None:
    window_class = workbench._build_qt_app()
    body = _body("ShellUnion")
    cell_map = workbench.CellMap.from_bounds(
        np.array(((0.0, 0.0, 0.0), (2.0, 2.0, 2.0))),
        spacing_mm=(2.0, 2.0, 2.0),
    )
    owner = SimpleNamespace(
        stl_reconstruction_target_combo=_TargetCombo("ShellUnion"),
        implicit_generation_results={
            "G": SimpleNamespace(),
            "ShellUnion": SimpleNamespace(
                body=body,
                cell_map=cell_map,
                minimum_feature_mm=0.5,
            ),
        },
    )

    selected = window_class._selected_stl_reconstruction_results(owner)

    assert list(selected) == ["ShellUnion"]
    assert selected["ShellUnion"].body is body
    assert selected["ShellUnion"].extraction_map is cell_map
    assert selected["ShellUnion"].minimum_feature_mm == 0.5


def test_section_selection_resolves_domain_or_the_selected_implicit_body() -> None:
    window_class = workbench._build_qt_app()
    lattice_body = _body("G")
    owner = SimpleNamespace(
        section_target_combo=_TargetCombo("domain"),
        domain_implicit_body=_body("domain"),
        implicit_generation_results={"G": SimpleNamespace(body=lattice_body)},
    )

    assert window_class._section_target_body(owner) is owner.domain_implicit_body
    owner.section_target_combo.value = "G"
    assert window_class._section_target_body(owner) is lattice_body


def test_section_render_source_is_independent_from_the_main_view() -> None:
    window_class = workbench._build_qt_app()
    owner = SimpleNamespace(
        view_stl_mesh=_CheckBox(True),
        section_render_source_combo=_TargetCombo("implicit"),
    )

    assert window_class._scene_uses_stl_mesh(owner, None)
    assert not window_class._scene_uses_stl_mesh(owner, "G")
    owner.section_render_source_combo.value = "mesh"
    assert window_class._scene_uses_stl_mesh(owner, "G")
