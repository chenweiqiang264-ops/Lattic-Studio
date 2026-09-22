"""One-time source-preserving extraction of workbench numerical workflows."""

from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "src/lattice_studio/presentation/qt/workbench.py"
TARGET = ROOT / "src/lattice_studio/engine/workflows.py"

MOVED_ASSIGNMENTS = {
    "TPMSKind",
    "LatticeKind",
    "TPMS_KINDS",
    "CellSize",
    "CellMapMode",
    "PlaneAxis",
    "TransitionDriverMode",
    "ProcessingMode",
    "ExportSpacingMode",
    "ProgressCallback",
    "EvaluationStageCallback",
    "FRAME_INPUT_DECIMALS",
    "FRAME_ORIGIN_DEFAULT_ATOL_MM",
    "PROGRESS_DOMAIN_SDF",
    "PROGRESS_TPMS_FIELD",
    "PROGRESS_INTERSECTION",
    "PROGRESS_MARCHING_CUBES",
    "LatticeParameters",
    "DISPLAY_FIELD_REFINEMENT_FACTORS",
    "DISPLAY_QUALITY_LABELS",
}

MOVED_EARLY_DEFINITIONS = {
    "TPMSParameterState",
    "_tpms_backend_label",
    "_geometry_backend_label",
    "_lattice_pipeline_backend_label",
}


def assigned_names(node: ast.Assign | ast.AnnAssign) -> set[str]:
    targets = node.targets if isinstance(node, ast.Assign) else [node.target]
    return {target.id for target in targets if isinstance(target, ast.Name)}


source = SOURCE.read_text(encoding="utf-8")
lines = source.splitlines(keepends=True)
tree = ast.parse(source)

import_nodes = [
    node
    for node in tree.body
    if isinstance(node, (ast.Import, ast.ImportFrom)) and node.lineno <= 167
]

moved_nodes: list[ast.AST] = []
moved_names: list[str] = []
for node in tree.body:
    move = False
    names: set[str] = set()
    if isinstance(node, (ast.Assign, ast.AnnAssign)):
        names = assigned_names(node)
        move = bool(names & MOVED_ASSIGNMENTS)
    elif isinstance(node, (ast.FunctionDef, ast.ClassDef)):
        names = {node.name}
        move = node.name in MOVED_EARLY_DEFINITIONS or 629 <= node.lineno <= 6068
    if move:
        moved_nodes.append(node)
        moved_names.extend(sorted(names))

if not moved_nodes:
    raise RuntimeError("no workflow definitions selected")

header = [
    '"""Implicit lattice generation, sampling, reconstruction, and repair workflows.\n\n',
    "This module is the compatibility-preserving first extraction from the Qt\n",
    "workbench. Its public surface will be narrowed as cohesive engine modules\n",
    "replace direct function imports.\n",
    '"""\n\n',
]
for node in import_nodes:
    header.extend(lines[node.lineno - 1 : node.end_lineno])
header.append("\n")
for node in moved_nodes:
    header.extend(lines[node.lineno - 1 : node.end_lineno])
    header.append("\n\n")
TARGET.write_text("".join(header).rstrip() + "\n", encoding="utf-8")

for node in sorted(moved_nodes, key=lambda item: item.lineno, reverse=True):
    start_line = node.lineno
    decorators = getattr(node, "decorator_list", ())
    if decorators:
        start_line = min(start_line, *(decorator.lineno for decorator in decorators))
    del lines[start_line - 1 : node.end_lineno]

unique_names = list(dict.fromkeys(moved_names))
import_block = ["from lattice_studio.engine.workflows import (\n"]
import_block.extend(f"    {name},\n" for name in unique_names)
import_block.append(")\n\n")
insert_at = min(node.lineno for node in moved_nodes) - 1
lines[insert_at:insert_at] = import_block
SOURCE.write_text("".join(lines), encoding="utf-8")

print(f"extracted {len(moved_nodes)} definitions into {TARGET}")
