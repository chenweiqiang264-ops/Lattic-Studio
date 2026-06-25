"""
Core module - 核心算法层
包含几何操作、隐式曲面等核心算法
"""

# 导出 geometry 模块中的主要函数
from core.geometry import (
    load_mesh,
    simplify_mesh,
    normalize_lattice_unit,
    prepare_lattice_unit_for_planar_tiling,
    generate_lattice_in_sole,
    generate_lattice_single_in_sole_bbox,
    generate_tile_lattice_with_cache,
    clip_lattice_to_sole,
    trim_lattice_to_sole_volume,
    generate_gyroid_lattice,
    generate_voronoi_lattice_in_sole,
    generate_voronoi_lattice_implicit,
    build_sole_with_lattice,
)

# 导出 implicit 子模块
from core import implicit


__all__ = [
    'load_mesh',
    'simplify_mesh',
    'normalize_lattice_unit',
    'prepare_lattice_unit_for_planar_tiling',
    'generate_lattice_in_sole',
    'generate_lattice_single_in_sole_bbox',
    'generate_tile_lattice_with_cache',
    'clip_lattice_to_sole',
    'trim_lattice_to_sole_volume',
    'generate_gyroid_lattice',
    'generate_voronoi_lattice_in_sole',
    'generate_voronoi_lattice_implicit',
    'build_sole_with_lattice',
    'implicit',
]
