"""
Viewers module - 3D 查看器模块
"""

from ui.viewers.pyvista_viewer import PyVistaRenderer, MeshData
from ui.viewers.open3d_viewer import Open3DRenderer, OPEN3D_AVAILABLE

# 为了兼容性，提供 GLMeshViewer 别名
GLMeshViewer = PyVistaRenderer

__all__ = [
    'PyVistaRenderer',
    'Open3DRenderer',
    'MeshData',
    'GLMeshViewer',
    'OPEN3D_AVAILABLE',
]
