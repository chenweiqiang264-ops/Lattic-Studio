"""
验证三个方向的晶胞连接性

测试 X, Y, Z 三个方向上两个晶胞的连接情况
使用已确定的最佳包围盒缩小系数 0.90
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import trimesh
from skimage import measure
from scipy.ndimage import gaussian_filter
import pyvista as pv
from PyQt5 import QtWidgets
import time


def select_unit_cell():
    """选择晶胞"""
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication(sys.argv)
    
    file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
        None,
        "选择晶胞 STL",
        str(project_root / "resources" / "晶格"),
        "STL Files (*.stl);;All Files (*.*)"
    )
    
    if not file_path:
        return None
    
    return trimesh.load(file_path)


def compute_sdf_with_custom_bbox(
    mesh: trimesh.Trimesh,
    bbox_min: np.ndarray,
    bbox_max: np.ndarray,
    resolution: int = 32
):
    """使用自定义包围盒计算 SDF"""
    
    pv_mesh = pv.wrap(mesh)
    
    # 使用自定义包围盒
    x = np.linspace(bbox_min[0], bbox_max[0], resolution)
    y = np.linspace(bbox_min[1], bbox_max[1], resolution)
    z = np.linspace(bbox_min[2], bbox_max[2], resolution)
    
    points = np.stack(np.meshgrid(x, y, z, indexing='ij'), axis=-1)
    points_flat = points.reshape(-1, 3)
    
    # 计算 SDF
    point_cloud = pv.PolyData(points_flat)
    distances = point_cloud.compute_implicit_distance(pv_mesh, inplace=False)
    sdf_values = distances['implicit_distance']
    
    sdf_grid = sdf_values.reshape(resolution, resolution, resolution)
    
    return sdf_grid, bbox_max - bbox_min


def generate_two_cells_direction(
    mesh: trimesh.Trimesh,
    bbox_min: np.ndarray,
    bbox_max: np.ndarray,
    unit_sdf: np.ndarray,
    direction: str = 'x',
    resolution_multiplier: float = 1.5,
    smoothing_options: dict = None
):
    """
    在指定方向生成两个晶胞
    
    参数:
        mesh: 晶胞网格
        bbox_min: 调整后的包围盒最小点
        bbox_max: 调整后的包围盒最大点
        unit_sdf: 单个晶胞的 SDF
        direction: 方向 ('x', 'y', 或 'z')
        resolution_multiplier: 分辨率倍增系数
        smoothing_options: 平滑选项字典
    """
    if smoothing_options is None:
        smoothing_options = {
            'sdf_enabled': False,
            'sdf_sigma': 1.0,
            'mesh_enabled': False,
            'mesh_iterations': 10,
            'mesh_lambda': 0.5
        }
    
    print(f"\n[生成] 两个晶胞 - {direction.upper()} 方向...")
    
    axis_map = {'x': 0, 'y': 1, 'z': 2}
    axis_idx = axis_map[direction.lower()]
    
    cell_size = bbox_max - bbox_min
    
    # 目标区域：两个晶胞
    target_size = cell_size.copy()
    target_size[axis_idx] *= 2
    
    target_bounds = np.array([bbox_min, bbox_min + target_size])
    
    # 计算目标分辨率
    unit_resolution = unit_sdf.shape[0]
    target_resolution = np.array([unit_resolution, unit_resolution, unit_resolution])
    target_resolution[axis_idx] = int(unit_resolution * 2 * resolution_multiplier)
    
    for i in range(3):
        if i != axis_idx:
            target_resolution[i] = int(unit_resolution * resolution_multiplier)
    
    print(f"  目标分辨率: {target_resolution}")
    
    spacing = target_size / target_resolution
    
    # 生成 SDF
    combined_sdf = np.zeros(tuple(target_resolution))
    
    total_voxels = np.prod(target_resolution)
    progress_step = max(1, total_voxels // 20)
    
    t_start = time.time()
    
    for idx in range(total_voxels):
        i = idx // (target_resolution[1] * target_resolution[2])
        j = (idx % (target_resolution[1] * target_resolution[2])) // target_resolution[2]
        k = idx % target_resolution[2]
        
        if idx % progress_step == 0:
            progress = idx / total_voxels * 100
            elapsed = time.time() - t_start
            if elapsed > 0:
                eta = elapsed / (idx + 1) * (total_voxels - idx - 1)
                print(f"  进度: {progress:.0f}% (预计剩余: {eta:.1f}s)", end='\r')
            else:
                print(f"  进度: {progress:.0f}%", end='\r')
        
        # 世界坐标
        world_pos = target_bounds[0] + np.array([i, j, k]) * spacing
        
        # 映射到晶胞内（周期性）
        local_pos = world_pos - bbox_min
        local_pos_mod = np.mod(local_pos, cell_size)
        
        # 转换为 SDF 索引
        unit_indices = (local_pos_mod / cell_size * unit_resolution).astype(int)
        unit_indices = np.clip(unit_indices, 0, unit_resolution - 1)
        
        combined_sdf[i, j, k] = unit_sdf[unit_indices[0], unit_indices[1], unit_indices[2]]
    
    t_end = time.time()
    print(f"  进度: 100% (总耗时: {t_end - t_start:.1f}s)")
    
    # 应用 SDF 平滑
    if smoothing_options['sdf_enabled']:
        print(f"  应用 SDF 高斯平滑 (sigma={smoothing_options['sdf_sigma']:.1f})...")
        t_start = time.time()
        combined_sdf = gaussian_filter(combined_sdf, sigma=smoothing_options['sdf_sigma'])
        t_smooth = time.time() - t_start
        print(f"  ✓ SDF 平滑完成 (耗时: {t_smooth:.2f}s)")
    
    # 重建
    print(f"  重建表面...")
    vertices, faces, _, _ = measure.marching_cubes(
        combined_sdf, level=0.0, spacing=spacing
    )
    vertices += target_bounds[0]
    
    result_mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=True)
    
    # 应用网格平滑
    if smoothing_options['mesh_enabled']:
        print(f"  应用网格拉普拉斯平滑 (iterations={smoothing_options['mesh_iterations']})...")
        t_start = time.time()
        
        pv_mesh = pv.wrap(result_mesh)
        smoothed_pv = pv_mesh.smooth(
            n_iter=smoothing_options['mesh_iterations'],
            relaxation_factor=smoothing_options['mesh_lambda']
        )
        
        result_mesh = trimesh.Trimesh(
            vertices=np.array(smoothed_pv.points),
            faces=np.array(smoothed_pv.faces).reshape(-1, 4)[:, 1:],
            process=False
        )
        
        t_smooth = time.time() - t_start
        print(f"  ✓ 网格平滑完成 (耗时: {t_smooth:.2f}s)")
    
    print(f"  ✓ 完成: {len(vertices)} 顶点, {len(faces)} 面")
    
    # 检查连通性
    components = result_mesh.split(only_watertight=False)
    print(f"  连通分量数: {len(components)}")
    
    connected = len(components) == 1
    
    if connected:
        print(f"  ✓✓✓ {direction.upper()} 方向：两个晶胞完美连接！")
    else:
        print(f"  ✗ {direction.upper()} 方向：晶胞分离")
        # 分析各分量大小
        volumes = [comp.volume for comp in components]
        print(f"  分量体积: {volumes}")
    
    return result_mesh, connected


def analyze_boundary_sdf(unit_sdf: np.ndarray, direction: str):
    """分析指定方向边界的 SDF 值"""
    axis_map = {'x': 0, 'y': 1, 'z': 2}
    axis_idx = axis_map[direction.lower()]
    
    # 提取边界
    if axis_idx == 0:
        left_boundary = unit_sdf[0, :, :]
        right_boundary = unit_sdf[-1, :, :]
    elif axis_idx == 1:
        left_boundary = unit_sdf[:, 0, :]
        right_boundary = unit_sdf[:, -1, :]
    else:  # axis_idx == 2
        left_boundary = unit_sdf[:, :, 0]
        right_boundary = unit_sdf[:, :, -1]
    
    left_mean = left_boundary.mean()
    left_min = left_boundary.min()
    left_has_structure = np.any(left_boundary < 0)
    
    right_mean = right_boundary.mean()
    right_min = right_boundary.min()
    right_has_structure = np.any(right_boundary < 0)
    
    print(f"\n  {direction.upper()} 方向边界 SDF:")
    print(f"    最小边界: 平均 {left_mean:.3f}, 最小 {left_min:.3f}, 有结构: {'✓' if left_has_structure else '✗'}")
    print(f"    最大边界: 平均 {right_mean:.3f}, 最小 {right_min:.3f}, 有结构: {'✓' if right_has_structure else '✗'}")
    
    return left_has_structure and right_has_structure


def visualize_results(results, test_directions):
    """可视化结果（动态布局）"""
    print("\n[可视化] 启动渲染器...")
    
    # 根据测试方向数量决定布局
    num_directions = len(test_directions)
    
    if num_directions == 1:
        # 1个方向：1x2 布局（原始 + 结果）
        plotter = pv.Plotter(shape=(1, 2), window_size=(1600, 800))
        
        # 左侧：原始晶胞
        plotter.subplot(0, 0)
        plotter.add_text("原始晶胞", font_size=14, color='gray')
        plotter.add_mesh(pv.wrap(results['original']), color='lightgray', opacity=0.5, show_edges=True)
        
        # 右侧：结果
        plotter.subplot(0, 1)
        direction = test_directions[0]
        title = f"{direction.upper()} 方向 - {'✓ 已连接' if results[f'{direction}_connected'] else '✗ 未连接'}"
        color = 'green' if results[f'{direction}_connected'] else 'red'
        plotter.add_text(title, font_size=14, color=color)
        plotter.add_mesh(pv.wrap(results[direction]), color=color, show_edges=True, line_width=1)
        
    elif num_directions == 2:
        # 2个方向：1x3 布局（原始 + 2个结果）
        plotter = pv.Plotter(shape=(1, 3), window_size=(2000, 700))
        
        # 左侧：原始晶胞
        plotter.subplot(0, 0)
        plotter.add_text("原始晶胞", font_size=12, color='gray')
        plotter.add_mesh(pv.wrap(results['original']), color='lightgray', opacity=0.5, show_edges=True)
        
        # 中间和右侧：结果
        for idx, direction in enumerate(test_directions):
            plotter.subplot(0, idx + 1)
            title = f"{direction.upper()} 方向 - {'✓ 已连接' if results[f'{direction}_connected'] else '✗ 未连接'}"
            color = 'green' if results[f'{direction}_connected'] else 'red'
            plotter.add_text(title, font_size=12, color=color)
            plotter.add_mesh(pv.wrap(results[direction]), color=color, show_edges=True, line_width=1)
    
    else:  # 3个方向
        # 3个方向：2x2 布局（原始 + 3个结果）
        plotter = pv.Plotter(shape=(2, 2), window_size=(1600, 1200))
        
        # 左上：原始晶胞
        plotter.subplot(0, 0)
        plotter.add_text("原始晶胞", font_size=12, color='gray')
        plotter.add_mesh(pv.wrap(results['original']), color='lightgray', opacity=0.5, show_edges=True)
        
        # 其他位置：结果
        positions = [(0, 1), (1, 0), (1, 1)]
        for idx, direction in enumerate(test_directions):
            row, col = positions[idx]
            plotter.subplot(row, col)
            title = f"{direction.upper()} 方向 - {'✓ 已连接' if results[f'{direction}_connected'] else '✗ 未连接'}"
            color = 'green' if results[f'{direction}_connected'] else 'red'
            plotter.add_text(title, font_size=12, color=color)
            plotter.add_mesh(pv.wrap(results[direction]), color=color, show_edges=True, line_width=1)
    
    plotter.link_views()
    plotter.show()


def visualize_three_directions(meshes_dict):
    """可视化三个方向的结果（保留用于向后兼容）"""
    print("\n[可视化] 启动渲染器...")
    
    plotter = pv.Plotter(shape=(2, 2), window_size=(1600, 1200))
    
    # 原始晶胞（左上）
    plotter.subplot(0, 0)
    plotter.add_text("原始晶胞", font_size=12, color='gray')
    plotter.add_mesh(pv.wrap(meshes_dict['original']), color='lightgray', opacity=0.5, show_edges=True)
    
    # X 方向（右上）
    plotter.subplot(0, 1)
    title = f"X 方向 - {'✓ 已连接' if meshes_dict['x_connected'] else '✗ 未连接'}"
    color = 'green' if meshes_dict['x_connected'] else 'red'
    plotter.add_text(title, font_size=12, color=color)
    plotter.add_mesh(pv.wrap(meshes_dict['x']), color=color, show_edges=True, line_width=1)
    
    # Y 方向（左下）
    plotter.subplot(1, 0)
    title = f"Y 方向 - {'✓ 已连接' if meshes_dict['y_connected'] else '✗ 未连接'}"
    color = 'green' if meshes_dict['y_connected'] else 'red'
    plotter.add_text(title, font_size=12, color=color)
    plotter.add_mesh(pv.wrap(meshes_dict['y']), color=color, show_edges=True, line_width=1)
    
    # Z 方向（右下）
    plotter.subplot(1, 1)
    title = f"Z 方向 - {'✓ 已连接' if meshes_dict['z_connected'] else '✗ 未连接'}"
    color = 'green' if meshes_dict['z_connected'] else 'red'
    plotter.add_text(title, font_size=12, color=color)
    plotter.add_mesh(pv.wrap(meshes_dict['z']), color=color, show_edges=True, line_width=1)
    
    plotter.link_views()
    plotter.show()


def select_resolution():
    """选择分辨率"""
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication(sys.argv)
    
    resolution, ok = QtWidgets.QInputDialog.getInt(
        None,
        "选择分辨率",
        "单个晶胞的 SDF 分辨率（建议: 32-128）:\n"
        "- 32: 快速预览（~1秒/晶胞）\n"
        "- 64: 平衡质量（~8秒/晶胞）\n"
        "- 96: 高质量（~27秒/晶胞）\n"
        "- 128: 超高质量（~64秒/晶胞）",
        64,  # 默认值
        16,  # 最小值
        256, # 最大值
        16   # 步长
    )
    
    if not ok:
        return None
    
    return resolution


def select_smoothing_options():
    """选择平滑选项"""
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication(sys.argv)
    
    dialog = QtWidgets.QDialog()
    dialog.setWindowTitle("平滑选项")
    dialog.setMinimumWidth(400)
    
    layout = QtWidgets.QVBoxLayout()
    
    # SDF 平滑
    sdf_group = QtWidgets.QGroupBox("SDF 高斯平滑")
    sdf_layout = QtWidgets.QVBoxLayout()
    
    sdf_checkbox = QtWidgets.QCheckBox("启用 SDF 平滑")
    sdf_checkbox.setChecked(True)
    sdf_layout.addWidget(sdf_checkbox)
    
    sdf_sigma_layout = QtWidgets.QHBoxLayout()
    sdf_sigma_layout.addWidget(QtWidgets.QLabel("Sigma:"))
    sdf_sigma_spin = QtWidgets.QDoubleSpinBox()
    sdf_sigma_spin.setRange(0.1, 3.0)
    sdf_sigma_spin.setSingleStep(0.1)
    sdf_sigma_spin.setValue(1.0)
    sdf_sigma_layout.addWidget(sdf_sigma_spin)
    sdf_layout.addLayout(sdf_sigma_layout)
    
    sdf_group.setLayout(sdf_layout)
    layout.addWidget(sdf_group)
    
    # 网格平滑
    mesh_group = QtWidgets.QGroupBox("网格拉普拉斯平滑")
    mesh_layout = QtWidgets.QVBoxLayout()
    
    mesh_checkbox = QtWidgets.QCheckBox("启用网格平滑")
    mesh_checkbox.setChecked(False)
    mesh_layout.addWidget(mesh_checkbox)
    
    mesh_iter_layout = QtWidgets.QHBoxLayout()
    mesh_iter_layout.addWidget(QtWidgets.QLabel("迭代次数:"))
    mesh_iter_spin = QtWidgets.QSpinBox()
    mesh_iter_spin.setRange(1, 50)
    mesh_iter_spin.setValue(10)
    mesh_iter_layout.addWidget(mesh_iter_spin)
    mesh_layout.addLayout(mesh_iter_layout)
    
    mesh_lambda_layout = QtWidgets.QHBoxLayout()
    mesh_lambda_layout.addWidget(QtWidgets.QLabel("松弛因子:"))
    mesh_lambda_spin = QtWidgets.QDoubleSpinBox()
    mesh_lambda_spin.setRange(0.0, 1.0)
    mesh_lambda_spin.setSingleStep(0.1)
    mesh_lambda_spin.setValue(0.5)
    mesh_lambda_layout.addWidget(mesh_lambda_spin)
    mesh_layout.addLayout(mesh_lambda_layout)
    
    mesh_group.setLayout(mesh_layout)
    layout.addWidget(mesh_group)
    
    # 说明
    info_label = QtWidgets.QLabel(
        "推荐设置:\n"
        "• SDF 平滑 (sigma=1.0): 连接处自然过渡\n"
        "• 网格平滑: 可选，用于最终抛光"
    )
    info_label.setStyleSheet("color: gray; font-size: 10px;")
    layout.addWidget(info_label)
    
    button_box = QtWidgets.QDialogButtonBox(
        QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
    )
    button_box.accepted.connect(dialog.accept)
    button_box.rejected.connect(dialog.reject)
    layout.addWidget(button_box)
    
    dialog.setLayout(layout)
    
    if dialog.exec_() == QtWidgets.QDialog.Accepted:
        return {
            'sdf_enabled': sdf_checkbox.isChecked(),
            'sdf_sigma': sdf_sigma_spin.value(),
            'mesh_enabled': mesh_checkbox.isChecked(),
            'mesh_iterations': mesh_iter_spin.value(),
            'mesh_lambda': mesh_lambda_spin.value()
        }
    
    return None


def select_directions():
    """选择要测试的方向"""
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication(sys.argv)
    
    dialog = QtWidgets.QDialog()
    dialog.setWindowTitle("选择测试方向")
    dialog.setMinimumWidth(300)
    
    layout = QtWidgets.QVBoxLayout()
    
    label = QtWidgets.QLabel("选择要测试的方向（可多选）:")
    layout.addWidget(label)
    
    checkbox_x = QtWidgets.QCheckBox("X 方向")
    checkbox_x.setChecked(True)
    layout.addWidget(checkbox_x)
    
    checkbox_y = QtWidgets.QCheckBox("Y 方向")
    checkbox_y.setChecked(True)
    layout.addWidget(checkbox_y)
    
    checkbox_z = QtWidgets.QCheckBox("Z 方向")
    checkbox_z.setChecked(True)
    layout.addWidget(checkbox_z)
    
    button_box = QtWidgets.QDialogButtonBox(
        QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
    )
    button_box.accepted.connect(dialog.accept)
    button_box.rejected.connect(dialog.reject)
    layout.addWidget(button_box)
    
    dialog.setLayout(layout)
    
    if dialog.exec_() == QtWidgets.QDialog.Accepted:
        directions = []
        if checkbox_x.isChecked():
            directions.append('x')
        if checkbox_y.isChecked():
            directions.append('y')
        if checkbox_z.isChecked():
            directions.append('z')
        return directions if directions else None
    
    return None


def main():
    """主流程"""
    print("=" * 70)
    print("三方向连接性验证")
    print("=" * 70)
    
    # 1. 选择晶胞
    print("\n请选择晶胞 STL 文件...")
    mesh = select_unit_cell()
    if mesh is None:
        print("✗ 未选择晶胞")
        return
    
    print(f"✓ 晶胞已加载: {len(mesh.vertices)} 顶点, {len(mesh.faces)} 面")
    
    # 2. 选择分辨率
    print("\n请选择分辨率...")
    resolution = select_resolution()
    if resolution is None:
        print("✗ 未选择分辨率")
        return
    
    print(f"✓ 分辨率: {resolution}³")
    
    # 3. 选择测试方向
    print("\n请选择要测试的方向...")
    test_directions = select_directions()
    if test_directions is None:
        print("✗ 未选择方向")
        return
    
    print(f"✓ 测试方向: {', '.join([d.upper() for d in test_directions])}")
    
    # 4. 选择平滑选项
    print("\n请选择平滑选项...")
    smoothing_options = select_smoothing_options()
    if smoothing_options is None:
        print("✗ 未选择平滑选项")
        return
    
    print(f"✓ 平滑选项:")
    if smoothing_options['sdf_enabled']:
        print(f"  - SDF 高斯平滑: sigma={smoothing_options['sdf_sigma']:.1f}")
    if smoothing_options['mesh_enabled']:
        print(f"  - 网格拉普拉斯平滑: iterations={smoothing_options['mesh_iterations']}, lambda={smoothing_options['mesh_lambda']:.1f}")
    if not smoothing_options['sdf_enabled'] and not smoothing_options['mesh_enabled']:
        print(f"  - 无平滑")
    
    # 5. 使用最佳缩小系数计算 SDF
    shrink_factor = 0.85
    
    print(f"\n[准备] 使用缩小系数 {shrink_factor:.2f}, 分辨率 {resolution}³")
    
    original_bounds = mesh.bounds
    original_center = (original_bounds[0] + original_bounds[1]) / 2
    original_size = original_bounds[1] - original_bounds[0]
    
    new_size = original_size * shrink_factor
    bbox_min = original_center - new_size / 2
    bbox_max = original_center + new_size / 2
    
    print(f"  调整后包围盒: {bbox_min} → {bbox_max}")
    print(f"  晶胞尺寸: {new_size}")
    
    # 计算 SDF
    print(f"\n[计算] SDF...")
    t_start = time.time()
    unit_sdf, cell_size = compute_sdf_with_custom_bbox(
        mesh, bbox_min, bbox_max, resolution
    )
    t_end = time.time()
    
    print(f"  ✓ SDF 计算完成 (耗时: {t_end - t_start:.2f}s)")
    print(f"  SDF 范围: [{unit_sdf.min():.3f}, {unit_sdf.max():.3f}]")
    
    # 3. 分析边界 SDF（仅分析要测试的方向）
    print("\n" + "=" * 70)
    print("边界 SDF 分析")
    print("=" * 70)
    
    boundary_predictions = {}
    for direction in test_directions:
        boundary_predictions[direction] = analyze_boundary_sdf(unit_sdf, direction)
    
    print(f"\n  预测:")
    for direction in test_directions:
        status = '✓' if boundary_predictions[direction] else '✗'
        print(f"    {direction.upper()} 方向可连接: {status}")
    
    # 4. 生成选定方向的两晶胞模型
    print("\n" + "=" * 70)
    print("生成两晶胞模型")
    print("=" * 70)
    
    results = {}
    
    for direction in test_directions:
        mesh_result, connected = generate_two_cells_direction(
            mesh, bbox_min, bbox_max, unit_sdf, direction=direction, 
            resolution_multiplier=1.5, smoothing_options=smoothing_options
        )
        results[direction] = mesh_result
        results[f'{direction}_connected'] = connected
    
    results['original'] = mesh
    
    # 5. 保存结果
    print("\n" + "=" * 70)
    print("保存结果")
    print("=" * 70)
    
    output_dir = project_root / "tests"
    
    for direction in test_directions:
        output_path = output_dir / f"two_cells_{direction}_direction_res{resolution}.stl"
        results[direction].export(str(output_path))
        status = "✓ 已连接" if results[f'{direction}_connected'] else "✗ 未连接"
        print(f"  {direction.upper()} 方向: {output_path.name} ({status})")
    
    # 6. 总结
    print("\n" + "=" * 70)
    print("总结")
    print("=" * 70)
    
    all_connected = all(results[f'{d}_connected'] for d in test_directions)
    
    for direction in test_directions:
        status = '✓ 已连接' if results[f'{direction}_connected'] else '✗ 未连接'
        print(f"  {direction.upper()} 方向: {status}")
    
    if all_connected:
        print(f"\n  ✓✓✓ 完美！所有测试方向都能正确连接")
        print(f"  可以安全地应用到周期性晶格生成")
    else:
        print(f"\n  ⚠ 部分方向无法连接")
        failed_directions = [d.upper() for d in test_directions if not results[f'{d}_connected']]
        print(f"  失败方向: {', '.join(failed_directions)}")
        print(f"\n  可能的原因:")
        print(f"    1. 晶胞在不同方向的几何特征不对称")
        print(f"    2. 需要针对不同方向使用不同的缩小系数")
        print(f"    3. 晶胞设计本身在某些方向不支持周期性")
    
    # 7. 可视化（根据测试的方向调整布局）
    if len(test_directions) > 0:
        visualize_results(results, test_directions)


if __name__ == "__main__":
    main()
