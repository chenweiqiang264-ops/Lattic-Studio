"""
C++ 扩展模块
包含用于加速计算的 C++ 扩展
"""

# 尝试导入 C++ 扩展模块
# 如果导入失败，程序会自动回退到 Python 实现

# cpp_distance_field - Voronoi 隐式曲面距离场计算
try:
    from core.cpp.cpp_distance_field import *
    CPP_DISTANCE_FIELD_AVAILABLE = True
except ImportError:
    CPP_DISTANCE_FIELD_AVAILABLE = False

# cpp_mesh_slicer - 网格分割加速
try:
    from core.cpp.cpp_mesh_slicer import *
    CPP_MESH_SLICER_AVAILABLE = True
except ImportError:
    CPP_MESH_SLICER_AVAILABLE = False

# cpp_mask - 点内判定加速（Gyroid、Tile、Voronoi）
try:
    from core.cpp.cpp_mask import *
    CPP_MASK_AVAILABLE = True
except ImportError:
    CPP_MASK_AVAILABLE = False

__all__ = [
    'CPP_DISTANCE_FIELD_AVAILABLE',
    'CPP_MESH_SLICER_AVAILABLE',
    'CPP_MASK_AVAILABLE',
]
