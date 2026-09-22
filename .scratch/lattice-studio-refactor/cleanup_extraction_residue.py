"""Remove decorators and blank lines left by the first workflow extraction."""

from pathlib import Path


root = Path(__file__).resolve().parents[2]
path = root / "src/lattice_studio/presentation/qt/workbench.py"
lines = path.read_text(encoding="utf-8").splitlines(keepends=True)

configure_call = next(
    index for index, line in enumerate(lines) if line.strip() == "_configure_console_encoding()"
)
next_definition = next(
    index
    for index, line in enumerate(lines[configure_call + 1 :], configure_call + 1)
    if line.startswith("def _default_sole_path")
)
residue = lines[configure_call + 1 : next_definition]
if any(line.strip() and line.strip() != "@dataclass(frozen=True)" for line in residue):
    raise RuntimeError("unexpected source found in extraction residue")
lines[configure_call + 1 : next_definition] = ["\n\n"]
path.write_text("".join(lines), encoding="utf-8")
