"""Build the optional C++ design-domain SDF extension.

Run from the project root:
    python src/lattice_studio/infrastructure/native/build_mesh_sdf.py build_ext --inplace
"""

from pathlib import Path
import sys

from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup


ROOT = Path(__file__).resolve().parent
compile_args = ["/O2", "/openmp"] if sys.platform == "win32" else ["-O3", "-fopenmp"]
link_args = [] if sys.platform == "win32" else ["-fopenmp"]


setup(
    name="cpp_mesh_sdf",
    version="1.0.0",
    ext_modules=[
        Pybind11Extension(
            "lattice_studio.infrastructure.native.cpp_mesh_sdf",
            [str(ROOT / "cpp_mesh_sdf.cpp")],
            cxx_std=17,
            extra_compile_args=compile_args,
            extra_link_args=link_args,
        )
    ],
    cmdclass={"build_ext": build_ext},
)
