"""Repository launcher for Lattice Studio.

Installed distributions use the ``lattice-studio`` console command. This file
keeps ``python main.py`` convenient for a source checkout.
"""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SOURCE_ROOT = ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from lattice_studio.__main__ import main


if __name__ == "__main__":
    raise SystemExit(main())
