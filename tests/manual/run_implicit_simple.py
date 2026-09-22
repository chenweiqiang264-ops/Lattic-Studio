"""
简化版隐式建模测试 - 确保窗口能正常显示
"""

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

import numpy as np
import trimesh
from skimage import measure
import time
import pyvista as pv
from PyQt5 import QtWidgets

def select_model():
    """选择模型"""
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication(sys.argv)
    
    file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
        None,
        "选择 STL 模型文件",
        str(project_root / "resources"),
        "STL Files (*.stl);;All Files (*.*)"
    )
    
    if not file_path:
        print("未选择文件")
        return None
    
    return trimesh.load(file_path)

def compute_sdf(mesh, resolution=48):
    """计算 SDF"""
    print(f"\n计算 SDF（分辨率: {resolution}³）...")
    
    pv_mesh = pv.wrap(mesh)
    bounds = mesh.bounds
    margin = (bounds[1] - bounds[0]) * 0.05
    bounds_min = bounds[0] - margin
    bounds_max = bounds[1] + margin
    
    x = np.linspace(bounds_min[0], bounds_max[0], resolution)
    y = np.linspace(bounds_min[1], bounds_max[1], resolution)
    z = np.linspace(bounds_min[2], bounds_max[2], resolution)
    
    points = np.stack(np.meshgrid(x, y, z, indexing='ij'), axis=-1)
    points_flat = points.reshape(-1, 3)
    
    point_cloud = pv.PolyData(points_flat)
    distances = point_cloud.compute_implicit_distance(pv_mesh, inplace=False)
    sdf_values = distances['implicit_distance']
    
    sdf_grid = sdf_values.reshape(resolution, resolution, resolution)
    spacing = (bounds_max - bounds_min) / resolution
    
    print(f"✓ SDF 计算完成")
    return sdf_grid, spacing, bounds_min

def reconstruct(sdf_grid, spacing, offset):
    """重建网格"""
    print("重建表面...")
    vertices, faces, _, _ = measure.marching_cubes(sdf_grid, level=0.0, spacing=spacing)
    vertices += offset
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=True)
    print(f"✓ 重建完成: {len(vertices)} 顶点, {len(faces)} 面")
    return mesh

def visualize(original, reconstructed):
    """可视化对比"""
    print("\n启动可视化...")
    
    # 创建 PyVista plotter（不使用 Qt）
    plotter = pv.Plotter(shape=(1, 2), window_size=(1600, 800))
    
    # 左侧：原始模型
    plotter.subplot(0, 0)
    plotter.add_text("原始模型", font_size=14, color='orange')
    plotter.add_mesh(pv.wrap(original), color='orange', show_edges=False)
    
    # 右侧：重建模型
    plotter.subplot(0, 1)
    plotter.add_text("重建模型", font_size=14, color='lightblue')
    plotter.add_mesh(pv.wrap(reconstructed), color='lightblue', show_edges=False)
    
    # 链接相机
    plotter.link_views()
    
    print("✓ 窗口已打开")
    print("  - 鼠标左键: 旋转")
    print("  - 鼠标右键: 平移")
    print("  - 滚轮: 缩放")
    print("  - Q 键: 退出")
    
    plotter.show()

def main():
    print("=" * 60)
    print("隐式建模测试（简化版）")
    print("=" * 60)
    
    # 1. 选择模型
    print("\n[1/4] 选择模型...")
    original = select_model()
    if original is None:
        return
    
    print(f"✓ 已加载: {len(original.vertices)} 顶点, {len(original.faces)} 面")
    
    # 2. 计算 SDF
    print("\n[2/4] 计算 SDF...")
    sdf_grid, spacing, offset = compute_sdf(original, resolution=48)
    
    # 3. 重建
    print("\n[3/4] 重建网格...")
    reconstructed = reconstruct(sdf_grid, spacing, offset)
    
    # 4. 对比
    print("\n[4/4] 质量评估...")
    volume_error = abs(original.volume - reconstructed.volume) / original.volume * 100
    area_error = abs(original.area - reconstructed.area) / original.area * 100
    print(f"  体积误差: {volume_error:.2f}%")
    print(f"  表面积误差: {area_error:.2f}%")
    
    # 5. 可视化
    visualize(original, reconstructed)

if __name__ == "__main__":
    main()
