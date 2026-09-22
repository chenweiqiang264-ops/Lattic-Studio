"""Capture the complete workbench chrome for isolated visual acceptance."""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from PyQt5 import QtWidgets

from lattice_studio.presentation.qt.workbench import _build_qt_app


def main() -> int:
    application = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = _build_qt_app()()
    try:
        window.resize(1480, 900)
        window.show()
        application.processEvents()
        window.viewer.render()
        application.processEvents()
        output_path = (
            PROJECT_ROOT
            / "build"
            / "test-artifacts"
            / "ui-style-acceptance.png"
        )
        viewport_path = (
            PROJECT_ROOT
            / "build"
            / "test-artifacts"
            / "ui-style-viewport-acceptance.png"
        )
        collapsed_path = (
            PROJECT_ROOT
            / "build"
            / "test-artifacts"
            / "ui-style-collapsed-sections.png"
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if not window.grab().save(str(output_path)):
            raise RuntimeError("failed to save UI acceptance screenshot")
        window.viewer.screenshot(str(viewport_path))
        collapsed_groups = (
            window.sampling_group,
            window.cell_map_group,
            window.advanced_render_group,
            window.stl_processing_group,
        )
        if not all(group.property("collapsed") is True for group in collapsed_groups):
            raise AssertionError("secondary UI groups must start collapsed")
        generation_scroll = next(
            area
            for area in window.findChildren(QtWidgets.QScrollArea)
            if area.widget() is not None
            and area.widget().isAncestorOf(window.sampling_group)
        )
        generation_scroll.verticalScrollBar().setValue(
            generation_scroll.verticalScrollBar().maximum()
        )
        application.processEvents()
        if not window.grab().save(str(collapsed_path)):
            raise RuntimeError("failed to save collapsed-section screenshot")
        print(output_path)
        print(viewport_path)
        print(collapsed_path)
        return 0
    finally:
        window.close()
        application.processEvents()


if __name__ == "__main__":
    raise SystemExit(main())
