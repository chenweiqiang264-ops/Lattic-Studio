"""Rewrite restored pre-package tests to the production package imports."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests"

REPLACEMENTS = (
    ("from core.implicit.", "from lattice_studio.engine.implicit."),
    ("import core.implicit.", "import lattice_studio.engine.implicit."),
    ("from ui.viewers.", "from lattice_studio.presentation.qt.viewers."),
    (
        "from tests.intermediate_tests import tpms_filling_app as app",
        "from lattice_studio.presentation.qt import workbench as app",
    ),
    (
        "from tests.intermediate_tests import tpms_filling_app as workbench",
        "from lattice_studio.presentation.qt import workbench",
    ),
    (
        "import tpms_filling_app as app",
        "from lattice_studio.presentation.qt import workbench as app",
    ),
    (
        "from tpms_filling_app import ",
        "from lattice_studio.presentation.qt.workbench import ",
    ),
    (
        "from tests.intermediate_tests import interactive_section_viewer",
        "from lattice_studio.presentation.qt.tools import interactive_section_viewer",
    ),
    (
        "from tests.intermediate_tests.tpms_filling_app import ",
        "from lattice_studio.presentation.qt.workbench import ",
    ),
    (
        "from tests.test_implicit.",
        "from tests.integration.implicit.",
    ),
    (
        "import interactive_section_viewer as viewer_module",
        "from lattice_studio.presentation.qt.tools import "
        "interactive_section_viewer as viewer_module",
    ),
    (
        "from interactive_section_viewer import ",
        "from lattice_studio.presentation.qt.tools.interactive_section_viewer import ",
    ),
    ("core.implicit.geometry_compute", "lattice_studio.engine.implicit.geometry_compute"),
    ("core.implicit.tpms_compute", "lattice_studio.engine.implicit.tpms_compute"),
)


def main() -> None:
    for path in TESTS.rglob("*.py"):
        source = path.read_text(encoding="utf-8")
        migrated = source
        for old, new in REPLACEMENTS:
            migrated = migrated.replace(old, new)
        if migrated != source:
            path.write_text(migrated, encoding="utf-8")


if __name__ == "__main__":
    main()
