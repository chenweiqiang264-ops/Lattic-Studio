"""Regression coverage for the workbench visual system and disclosure UI."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from lattice_studio.presentation.qt.workbench import (
    UI_THEME,
    _application_stylesheet,
    _build_collapsible_group_box_class,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_design_document_and_qss_share_semantic_tokens() -> None:
    design_document = (PROJECT_ROOT / "design.md").read_text(encoding="utf-8")
    stylesheet = _application_stylesheet()

    assert UI_THEME["primary"] == "#7C3AED"
    assert UI_THEME["background"] == "#17181D"
    assert UI_THEME["danger"] == "#EF4444"
    assert all(value in design_document for value in UI_THEME.values())
    assert 'QPushButton[role="primary"]' in stylesheet
    assert 'QPushButton[role="danger"]' in stylesheet
    assert 'QGroupBox[collapsed="true"]' in stylesheet
    assert "QMenu::item:selected" in stylesheet
    assert "QFrame#viewportToolbar" in stylesheet


def test_secondary_groups_start_collapsed_and_restore_control_state() -> None:
    from PyQt5 import QtCore, QtWidgets

    application = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    group_class = _build_collapsible_group_box_class(QtCore, QtWidgets)
    group = group_class("高级参数", collapsed=True)
    group.setMinimumHeight(212)
    layout = QtWidgets.QFormLayout(group)
    control = QtWidgets.QLineEdit("保持参数")
    layout.addRow("参数", control)
    try:
        group.finalize_initial_state()
        assert group.property("collapsed") is True
        assert group.maximumHeight() == 40
        assert control.isHidden()

        group.set_expanded(True)
        assert group.property("collapsed") is False
        assert group.minimumHeight() == 212
        assert not control.isHidden()
        assert control.text() == "保持参数"

        group.set_expanded(False)
        assert control.isHidden()
        assert control.text() == "保持参数"
    finally:
        group.close()
        application.processEvents()
