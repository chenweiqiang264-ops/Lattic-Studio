"""One-time extraction of user-facing parameter value objects."""

from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "src/lattice_studio/engine/workflows.py"
TARGET = ROOT / "src/lattice_studio/domain/parameters.py"

ASSIGNMENTS = {
    "TPMSKind",
    "LatticeKind",
    "TPMS_KINDS",
    "CellSize",
    "CellMapMode",
    "PlaneAxis",
    "TransitionDriverMode",
    "ProcessingMode",
    "ExportSpacingMode",
    "FRAME_INPUT_DECIMALS",
    "FRAME_ORIGIN_DEFAULT_ATOL_MM",
    "LatticeParameters",
}
CLASSES = {
    "TPMSParameterState",
    "TPMSParameters",
    "CustomUnitCellParameters",
    "SamplingParameters",
    "TransitionParameters",
}


def names_for(node: ast.Assign | ast.AnnAssign) -> set[str]:
    targets = node.targets if isinstance(node, ast.Assign) else [node.target]
    return {target.id for target in targets if isinstance(target, ast.Name)}


source = SOURCE.read_text(encoding="utf-8")
lines = source.splitlines(keepends=True)
tree = ast.parse(source)
moved: list[ast.AST] = []
exports: list[str] = []
for node in tree.body:
    names: set[str] = set()
    if isinstance(node, (ast.Assign, ast.AnnAssign)):
        names = names_for(node)
        selected = bool(names & ASSIGNMENTS)
    elif isinstance(node, ast.ClassDef):
        names = {node.name}
        selected = node.name in CLASSES
    else:
        selected = False
    if selected:
        moved.append(node)
        exports.extend(sorted(names))

if {node.name for node in moved if isinstance(node, ast.ClassDef)} != CLASSES:
    raise RuntimeError("parameter class selection is incomplete")

header = '''"""User-facing lattice design parameters and validation rules."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypedDict

import numpy as np

from lattice_studio.domain.cell_map import CellMapFrame
from lattice_studio.domain.gradient import (
    GRADIENT_RESOLUTION_STRATEGIES,
    GradientAxis,
    GradientControls,
    GradientMode,
    GradientResolutionStrategy,
    WallThicknessMethod,
)
from lattice_studio.domain.transition import (
    TransitionSpec as ProviderTransitionSpec,
    TransitionWeightKind,
)

'''
content = [header]
for node in moved:
    start = node.lineno
    decorators = getattr(node, "decorator_list", ())
    if decorators:
        start = min(start, *(decorator.lineno for decorator in decorators))
    content.extend(lines[start - 1 : node.end_lineno])
    content.append("\n\n")
content.append("__all__ = [\n")
content.extend(f'    "{name}",\n' for name in dict.fromkeys(exports))
content.append("]\n")
TARGET.write_text("".join(content), encoding="utf-8")

for node in sorted(moved, key=lambda item: item.lineno, reverse=True):
    start = node.lineno
    decorators = getattr(node, "decorator_list", ())
    if decorators:
        start = min(start, *(decorator.lineno for decorator in decorators))
    del lines[start - 1 : node.end_lineno]

ordered = list(dict.fromkeys(exports))
imports = ["from lattice_studio.domain.parameters import (\n"]
imports.extend(f"    {name},\n" for name in ordered)
imports.append(")\n\n")
insert_at = min(node.lineno for node in moved) - 1
lines[insert_at:insert_at] = imports
SOURCE.write_text("".join(lines), encoding="utf-8")
print(f"extracted {len(CLASSES)} parameter classes")
