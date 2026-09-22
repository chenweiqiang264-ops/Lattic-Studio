"""
立方体区域周期性晶格填充测试

使用简单的立方体区域进行实验，便于：
1. 验证周期性连接
2. 调试参数
3. 理解工作原理
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

# 尝试导入 C++ 加速模块
try:
    from core.cpp.cpp_periodic_sdf import sample_periodic_sdf as cpp_sample_periodic_sdf
    from core.cpp.cpp_periodic_sdf import compute_mesh_distance_accurate as cpp_compute_mesh_distance
    CPP_AVAILABLE = True
    print("[加速] C++ 周期性 SDF 采样模块已加载")
except ImportError:
    CPP_AVAILABLE = False
    print("[警告] C++ 加速模块未找到，将使用 Python 版本（较慢）")
    print("       运行 'python core/cpp/build_periodic_sdf.py build_ext --inplace' 来编译")

# 尝试导入 PyMCubes（支持 Dual Contouring）
try:
    import mcubes
    PYMCUBES_AVAILABLE = True
    print("[加速] PyMCubes 已加载（支持 Dual Contouring）")
except ImportError:
    PYMCUBES_AVAILABLE = False
    print("[提示] PyMCubes 未安装，将使用传统 Marching Cubes")
    print("       运行 'pip install PyMCubes' 以启用 Dual Contouring（更好的锐特征保留）")

# 尝试导入八叉树 SDF
try:
    from core.octree_sdf import OctreeSDF, build_octree_from_grid
    OCTREE_AVAILABLE = True
    print("[优化] 八叉树 SDF 已加载（大幅减少内存占用）")
except ImportError:
    OCTREE_AVAILABLE = False
    print("[提示] 八叉树 SDF 模块未找到")

# 导入八叉树 SDF 模块
try:
    from core.octree_sdf import OctreeSDF, build_octree_from_grid
    OCTREE_AVAILABLE = True
    print("[优化] 八叉树 SDF 模块已加载（节省内存）")
except ImportError as e:
    OCTREE_AVAILABLE = False
    print(f"[警告] 八叉树 SDF 模块加载失败: {e}")
    print("       将使用传统密集网格（内存占用较大）")


def marching_cubes_adaptive(
    sdf_grid,
    level=0.0,
    spacing=(1.0, 1.0, 1.0),
    use_dual=False
):
    """
    自适应 Marching Cubes
    优先使用 PyMCubes（支持 Dual Contouring），回退到 scikit-image
    
    参数:
        sdf_grid: SDF网格
        level: 等值面level
        spacing: 体素间距
        use_dual: 是否尝试使用 Dual Contouring
    
    返回:
        vertices, faces
    """
    if use_dual and PYMCUBES_AVAILABLE:
        try:
            print(f"    使用 PyMCubes Dual Contouring（保留锐特征）...")
            t_start = time.time()
            
            # PyMCubes 使用不同的参数
            vertices, triangles = mcubes.marching_cubes(sdf_grid, level)
            
            # 应用spacing
            vertices = vertices * np.array(spacing)
            
            t_end = time.time()
            print(f"    ✓ Dual Contouring 完成 (耗时: {t_end - t_start:.1f}s)")
            
            return vertices, triangles
        except Exception as e:
            print(f"    ⚠ PyMCubes 失败: {e}")
            print(f"    回退到传统 Marching Cubes...")
    
    # 使用传统 Marching Cubes
    if use_dual and not PYMCUBES_AVAILABLE:
        print(f"    PyMCubes 未安装，使用传统 Marching Cubes")
        print(f"    提示: pip install PyMCubes 以启用 Dual Contouring")
    
    vertices, faces, _, _ = measure.marching_cubes(
        sdf_grid,
        level=level,
        spacing=spacing
    )
    
    return vertices, faces


def simplify_mesh(mesh, target_faces=10000):
    """
    简化网格（减面）
    
    参数:
        mesh: 输入网格
        target_faces: 目标面数
    
    返回:
        simplified_mesh: 简化后的网格
    """
    original_faces = len(mesh.faces)
    
    if original_faces <= target_faces:
        print(f"  网格面数（{original_faces:,}）已经小于目标（{target_faces:,}），无需简化")
        return mesh
    
    print(f"\n[简化网格]")
    print(f"  原始面数: {original_faces:,}")
    print(f"  目标面数: {target_faces:,}")
    print(f"  简化比例: {target_faces/original_faces*100:.1f}%")
    
    t_start = time.time()
    
    # 方法1：尝试使用 fast_simplification（最好的质量）
    try:
        if hasattr(mesh, 'simplify_quadric_decimation'):
            target_vertices = target_faces // 2
            simplified = mesh.simplify_quadric_decimation(target_vertices)
            t_end = time.time()
            
            print(f"  ✓ 使用 Fast Simplification (Quadric Decimation) 简化完成 (耗时: {t_end - t_start:.2f}s)")
            print(f"  简化后面数: {len(simplified.faces):,}")
            print(f"  简化后顶点数: {len(simplified.vertices):,}")
            print(f"  实际简化比例: {len(simplified.faces)/original_faces*100:.1f}%")
            
            return simplified
        else:
            raise AttributeError("simplify_quadric_decimation not available")
            
    except Exception as e:
        print(f"  ⚠ Fast Simplification 不可用")
    
    # 方法2：尝试使用 Open3D（推荐的 fallback）
    try:
        import open3d as o3d
        
        print(f"  使用 Open3D 简化...")
        
        # 转换为 Open3D 网格
        o3d_mesh = o3d.geometry.TriangleMesh()
        o3d_mesh.vertices = o3d.utility.Vector3dVector(mesh.vertices)
        o3d_mesh.triangles = o3d.utility.Vector3iVector(mesh.faces)
        
        # 计算法向量（某些简化算法需要）
        o3d_mesh.compute_vertex_normals()
        
        # 使用 Quadric Decimation 简化
        simplified_o3d = o3d_mesh.simplify_quadric_decimation(target_number_of_triangles=target_faces)
        
        # 转换回 trimesh
        vertices = np.asarray(simplified_o3d.vertices)
        faces = np.asarray(simplified_o3d.triangles)
        simplified = trimesh.Trimesh(vertices=vertices, faces=faces, process=True)
        
        t_end = time.time()
        
        print(f"  ✓ Open3D 简化完成 (耗时: {t_end - t_start:.2f}s)")
        print(f"  简化后面数: {len(simplified.faces):,}")
        print(f"  简化后顶点数: {len(simplified.vertices):,}")
        print(f"  实际简化比例: {len(simplified.faces)/original_faces*100:.1f}%")
        
        return simplified
        
    except ImportError:
        print(f"  ⚠ Open3D 未安装")
        print(f"  提示: pip install open3d")
    except Exception as e2:
        print(f"  ⚠ Open3D 简化失败: {e2}")
    
    # 方法3：使用 trimesh 的顶点合并（基础方法）
    try:
        print(f"  尝试使用顶点合并方法...")
        
        # 计算合适的合并距离
        bounds_size = mesh.bounds[1] - mesh.bounds[0]
        avg_size = bounds_size.mean()
        reduction_ratio = target_faces / original_faces
        merge_distance = avg_size * 0.005 * (1.0 / reduction_ratio) ** 0.5
        
        print(f"  合并距离: {merge_distance:.4f}")
        
        # 合并接近的顶点
        simplified = mesh.copy()
        simplified.merge_vertices(merge_distance)
        
        # 清理网格
        if hasattr(simplified, 'remove_degenerate_faces'):
            simplified.remove_degenerate_faces()
        if hasattr(simplified, 'remove_duplicate_faces'):
            simplified.remove_duplicate_faces()
        
        t_end = time.time()
        
        result_faces = len(simplified.faces)
        
        print(f"  ✓ 简化完成 (耗时: {t_end - t_start:.2f}s)")
        print(f"  简化后面数: {result_faces:,}")
        print(f"  简化后顶点数: {len(simplified.vertices):,}")
        print(f"  实际简化比例: {result_faces/original_faces*100:.1f}%")
        
        # 如果简化不够，再次简化
        if result_faces > target_faces * 1.5 and result_faces < original_faces * 0.9:
            print(f"  简化不够，再次简化...")
            merge_distance *= 1.5
            simplified.merge_vertices(merge_distance)
            if hasattr(simplified, 'remove_degenerate_faces'):
                simplified.remove_degenerate_faces()
            if hasattr(simplified, 'remove_duplicate_faces'):
                simplified.remove_duplicate_faces()
            print(f"  二次简化后面数: {len(simplified.faces):,}")
        
        if len(simplified.faces) < original_faces * 0.9:
            return simplified
        else:
            print(f"  ⚠ 简化效果不明显")
            raise Exception("Insufficient simplification")
        
    except Exception as e3:
        print(f"  ⚠ 顶点合并失败: {e3}")
    
    # 方法4：如果都失败，返回原始网格
    print(f"\n  ⚠ 所有简化方法都失败，将使用原始网格")
    print(f"  ")
    print(f"  建议安装以下任一库以启用网格简化：")
    print(f"    1. pip install fast-simplification  (推荐，最快最好)")
    print(f"    2. pip install open3d  (推荐，功能强大)")
    print(f"  ")
    print(f"  或者在外部软件（如 MeshLab, Blender）中手动简化网格")
    return mesh


def select_simplification_options(mesh):
    """选择网格简化选项"""
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication(sys.argv)
    
    dialog = QtWidgets.QDialog()
    dialog.setWindowTitle("网格简化选项")
    dialog.setMinimumWidth(450)
    
    layout = QtWidgets.QVBoxLayout()
    
    # 显示当前网格信息
    info_group = QtWidgets.QGroupBox("当前网格信息")
    info_layout = QtWidgets.QVBoxLayout()
    
    info_text = f"顶点数: {len(mesh.vertices):,}\n"
    info_text += f"面数: {len(mesh.faces):,}\n"
    
    # 估算 SDF 计算时间
    estimated_time = len(mesh.faces) * 0.0001  # 粗略估算
    info_text += f"\n预计 SDF 计算时间: ~{estimated_time:.1f}秒/晶胞"
    
    info_label = QtWidgets.QLabel(info_text)
    info_label.setStyleSheet("color: #555; font-size: 10px;")
    info_layout.addWidget(info_label)
    
    info_group.setLayout(info_layout)
    layout.addWidget(info_group)
    
    # 简化选项
    simplify_group = QtWidgets.QGroupBox("网格简化")
    simplify_layout = QtWidgets.QVBoxLayout()
    
    simplify_check = QtWidgets.QCheckBox("启用网格简化（减面）")
    simplify_check.setChecked(len(mesh.faces) > 20000)  # 超过2万面默认启用
    simplify_layout.addWidget(simplify_check)
    
    # 目标面数
    target_layout = QtWidgets.QHBoxLayout()
    target_layout.addWidget(QtWidgets.QLabel("目标面数:"))
    target_spin = QtWidgets.QSpinBox()
    target_spin.setRange(1000, 100000)
    target_spin.setSingleStep(1000)
    target_spin.setValue(10000)
    target_layout.addWidget(target_spin)
    simplify_layout.addLayout(target_layout)
    
    # 预设选项
    preset_layout = QtWidgets.QHBoxLayout()
    preset_layout.addWidget(QtWidgets.QLabel("快速预设:"))
    
    preset_low = QtWidgets.QPushButton("低质量 (5K)")
    preset_low.clicked.connect(lambda: target_spin.setValue(5000))
    preset_layout.addWidget(preset_low)
    
    preset_medium = QtWidgets.QPushButton("中等 (10K)")
    preset_medium.clicked.connect(lambda: target_spin.setValue(10000))
    preset_layout.addWidget(preset_medium)
    
    preset_high = QtWidgets.QPushButton("高质量 (20K)")
    preset_high.clicked.connect(lambda: target_spin.setValue(20000))
    preset_layout.addWidget(preset_high)
    
    simplify_layout.addLayout(preset_layout)
    
    # 说明
    help_text = QtWidgets.QLabel(
        "建议:\n"
        "• 面数 < 10K: 无需简化\n"
        "• 面数 10K-50K: 简化到 10K\n"
        "• 面数 > 50K: 简化到 5K-10K\n"
        "\n"
        "简化会加快 SDF 计算，但可能损失细节"
    )
    help_text.setStyleSheet("color: gray; font-size: 10px;")
    simplify_layout.addWidget(help_text)
    
    simplify_group.setLayout(simplify_layout)
    layout.addWidget(simplify_group)
    
    # 按钮
    button_box = QtWidgets.QDialogButtonBox(
        QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
    )
    button_box.accepted.connect(dialog.accept)
    button_box.rejected.connect(dialog.reject)
    layout.addWidget(button_box)
    
    dialog.setLayout(layout)
    
    if dialog.exec_() == QtWidgets.QDialog.Accepted:
        return {
            'enabled': simplify_check.isChecked(),
            'target_faces': target_spin.value()
        }
    
    return None


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


def select_fill_space():
    """选择填充空间类型"""
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication(sys.argv)
    
    dialog = QtWidgets.QDialog()
    dialog.setWindowTitle("选择填充空间")
    dialog.setMinimumWidth(350)
    
    layout = QtWidgets.QVBoxLayout()
    
    label = QtWidgets.QLabel("选择填充空间类型:")
    layout.addWidget(label)
    
    # 单选按钮
    radio_cubic = QtWidgets.QRadioButton("立方体区域（参数化）")
    radio_cubic.setChecked(True)
    layout.addWidget(radio_cubic)
    
    radio_stl = QtWidgets.QRadioButton("STL 模型（如鞋底）")
    layout.addWidget(radio_stl)
    
    # 按钮
    button_box = QtWidgets.QDialogButtonBox(
        QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
    )
    button_box.accepted.connect(dialog.accept)
    button_box.rejected.connect(dialog.reject)
    layout.addWidget(button_box)
    
    dialog.setLayout(layout)
    
    if dialog.exec_() == QtWidgets.QDialog.Accepted:
        if radio_cubic.isChecked():
            return 'cubic'
        else:
            return 'stl'
    
    return None


def select_stl_file():
    """选择 STL 填充空间"""
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication(sys.argv)
    
    file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
        None,
        "选择填充空间 STL",
        str(project_root / "resources" / "鞋底"),
        "STL Files (*.stl);;All Files (*.*)"
    )
    
    if not file_path:
        return None
    
    return trimesh.load(file_path)


def select_post_simplification_options():
    """选择后处理简化选项"""
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication(sys.argv)
    
    dialog = QtWidgets.QDialog()
    dialog.setWindowTitle("后处理减面选项")
    dialog.setMinimumWidth(450)
    
    layout = QtWidgets.QVBoxLayout()
    
    # 说明
    info_label = QtWidgets.QLabel(
        "后处理减面：在生成最终网格后进行简化\n"
        "优点：保留高分辨率生成的细节，然后减少面数\n"
        "适用于：网格太大无法导出或打开的情况"
    )
    info_label.setStyleSheet("color: #555; font-size: 10px; padding: 10px;")
    layout.addWidget(info_label)
    
    # 启用选项
    enable_group = QtWidgets.QGroupBox("后处理减面")
    enable_layout = QtWidgets.QVBoxLayout()
    
    enable_check = QtWidgets.QCheckBox("启用后处理减面")
    enable_check.setChecked(False)
    enable_layout.addWidget(enable_check)
    
    enable_group.setLayout(enable_layout)
    layout.addWidget(enable_group)
    
    # 目标面数
    target_group = QtWidgets.QGroupBox("目标面数")
    target_layout = QtWidgets.QVBoxLayout()
    
    target_layout.addWidget(QtWidgets.QLabel("目标面数（百万）:"))
    target_spin = QtWidgets.QDoubleSpinBox()
    target_spin.setRange(0.1, 100.0)
    target_spin.setSingleStep(0.5)
    target_spin.setValue(5.0)  # 默认5M面
    target_spin.setDecimals(1)
    target_spin.setSuffix(" M")
    target_layout.addWidget(target_spin)
    
    # 预设
    preset_layout = QtWidgets.QHBoxLayout()
    preset_layout.addWidget(QtWidgets.QLabel("快速预设:"))
    
    preset_1m = QtWidgets.QPushButton("1M")
    preset_1m.clicked.connect(lambda: target_spin.setValue(1.0))
    preset_layout.addWidget(preset_1m)
    
    preset_5m = QtWidgets.QPushButton("5M")
    preset_5m.clicked.connect(lambda: target_spin.setValue(5.0))
    preset_layout.addWidget(preset_5m)
    
    preset_10m = QtWidgets.QPushButton("10M")
    preset_10m.clicked.connect(lambda: target_spin.setValue(10.0))
    preset_layout.addWidget(preset_10m)
    
    preset_20m = QtWidgets.QPushButton("20M")
    preset_20m.clicked.connect(lambda: target_spin.setValue(20.0))
    preset_layout.addWidget(preset_20m)
    
    target_layout.addLayout(preset_layout)
    
    # 说明
    help_text = QtWidgets.QLabel(
        "建议:\n"
        "• 1M: 快速预览，文件小\n"
        "• 5M: 平衡质量和大小（推荐）\n"
        "• 10M: 高质量\n"
        "• 20M: 超高质量（文件较大）"
    )
    help_text.setStyleSheet("color: gray; font-size: 10px;")
    target_layout.addWidget(help_text)
    
    target_group.setLayout(target_layout)
    target_group.setEnabled(False)
    layout.addWidget(target_group)
    
    # 切换启用/禁用
    def on_enable_changed():
        target_group.setEnabled(enable_check.isChecked())
    
    enable_check.toggled.connect(on_enable_changed)
    
    # 按钮
    button_box = QtWidgets.QDialogButtonBox(
        QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
    )
    button_box.accepted.connect(dialog.accept)
    button_box.rejected.connect(dialog.reject)
    layout.addWidget(button_box)
    
    dialog.setLayout(layout)
    
    if dialog.exec_() == QtWidgets.QDialog.Accepted:
        return {
            'enabled': enable_check.isChecked(),
            'target_faces': int(target_spin.value() * 1_000_000)
        }
    
    return None


def select_parameters(space_type='cubic', unit_cell_mesh=None, fill_space_mesh=None):
    """选择参数"""
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication(sys.argv)
    
    dialog = QtWidgets.QDialog()
    dialog.setWindowTitle("填充参数设置")
    dialog.setMinimumWidth(450)
    
    layout = QtWidgets.QVBoxLayout()
    
    # 晶胞尺寸控制
    scale_group = QtWidgets.QGroupBox("晶胞尺寸控制")
    scale_layout = QtWidgets.QVBoxLayout()
    
    # 显示原始尺寸
    if unit_cell_mesh is not None:
        original_size = unit_cell_mesh.bounds[1] - unit_cell_mesh.bounds[0]
        original_size_text = f"原始晶胞尺寸: {original_size[0]:.2f} × {original_size[1]:.2f} × {original_size[2]:.2f} mm"
        
        # 如果是 STL 模式，显示填充空间尺寸
        if space_type == 'stl' and fill_space_mesh is not None:
            fill_size = fill_space_mesh.bounds[1] - fill_space_mesh.bounds[0]
            original_size_text += f"\n填充空间尺寸: {fill_size[0]:.2f} × {fill_size[1]:.2f} × {fill_size[2]:.2f} mm"
            
            # 计算推荐缩放比例（让最小维度能容纳 3-5 个晶胞）
            min_fill_dim = fill_size.min()
            max_original_dim = original_size.max()
            recommended_scale = (min_fill_dim / 4.0) / max_original_dim  # 目标：4个晶胞
            original_size_text += f"\n\n推荐缩放比例: {recommended_scale:.3f} (让最小维度容纳约4个晶胞)"
        
        size_label = QtWidgets.QLabel(original_size_text)
        size_label.setStyleSheet("color: #555; font-size: 10px;")
        scale_layout.addWidget(size_label)
    
    # 缩放比例控制
    scale_method_layout = QtWidgets.QHBoxLayout()
    scale_method_layout.addWidget(QtWidgets.QLabel("缩放方式:"))
    
    scale_ratio_radio = QtWidgets.QRadioButton("按比例")
    scale_ratio_radio.setChecked(True)
    scale_method_layout.addWidget(scale_ratio_radio)
    
    scale_absolute_radio = QtWidgets.QRadioButton("指定尺寸")
    scale_method_layout.addWidget(scale_absolute_radio)
    
    scale_layout.addLayout(scale_method_layout)
    
    # 比例缩放
    ratio_layout = QtWidgets.QHBoxLayout()
    ratio_layout.addWidget(QtWidgets.QLabel("缩放比例:"))
    scale_ratio_spin = QtWidgets.QDoubleSpinBox()
    scale_ratio_spin.setRange(0.1, 5.0)
    scale_ratio_spin.setSingleStep(0.1)
    scale_ratio_spin.setValue(1.0)
    scale_ratio_spin.setDecimals(3)
    ratio_layout.addWidget(scale_ratio_spin)
    ratio_layout.addWidget(QtWidgets.QLabel("(1.0 = 原始尺寸)"))
    scale_layout.addLayout(ratio_layout)
    
    # 绝对尺寸
    absolute_layout = QtWidgets.QHBoxLayout()
    absolute_layout.addWidget(QtWidgets.QLabel("目标尺寸 (mm):"))
    scale_absolute_spin = QtWidgets.QDoubleSpinBox()
    scale_absolute_spin.setRange(1.0, 200.0)
    scale_absolute_spin.setSingleStep(1.0)
    scale_absolute_spin.setValue(10.0)
    scale_absolute_spin.setDecimals(2)
    absolute_layout.addWidget(scale_absolute_spin)
    absolute_layout.addWidget(QtWidgets.QLabel("(最大维度)"))
    scale_layout.addLayout(absolute_layout)
    
    # 默认禁用绝对尺寸输入
    scale_absolute_spin.setEnabled(False)
    
    # 切换启用/禁用
    def on_scale_method_changed():
        scale_ratio_spin.setEnabled(scale_ratio_radio.isChecked())
        scale_absolute_spin.setEnabled(scale_absolute_radio.isChecked())
    
    scale_ratio_radio.toggled.connect(on_scale_method_changed)
    
    scale_group.setLayout(scale_layout)
    layout.addWidget(scale_group)
    
    # Shrink Factor（包围盒缩小系数）
    shrink_group = QtWidgets.QGroupBox("包围盒缩小系数（周期性连接）")
    shrink_layout = QtWidgets.QVBoxLayout()
    
    shrink_info = QtWidgets.QLabel(
        "控制 SDF 计算时的包围盒大小：\n"
        "• 0.90: 适用于大多数晶格（默认）\n"
        "• 1.00: 适用于数学生成的周期性晶格（如 Gyroid）"
    )
    shrink_info.setStyleSheet("color: #555; font-size: 10px;")
    shrink_layout.addWidget(shrink_info)
    
    shrink_spin_layout = QtWidgets.QHBoxLayout()
    shrink_spin_layout.addWidget(QtWidgets.QLabel("Shrink Factor:"))
    shrink_spin = QtWidgets.QDoubleSpinBox()
    shrink_spin.setRange(0.80, 1.00)
    shrink_spin.setSingleStep(0.05)
    shrink_spin.setValue(0.90)
    shrink_spin.setDecimals(2)
    shrink_spin_layout.addWidget(shrink_spin)
    shrink_layout.addLayout(shrink_spin_layout)
    
    shrink_group.setLayout(shrink_layout)
    layout.addWidget(shrink_group)
    
    # 晶胞数量（仅立方体模式）
    if space_type == 'cubic':
        cells_group = QtWidgets.QGroupBox("立方体尺寸（晶胞数量）")
        cells_layout = QtWidgets.QVBoxLayout()
        
        cells_x_layout = QtWidgets.QHBoxLayout()
        cells_x_layout.addWidget(QtWidgets.QLabel("X 方向:"))
        cells_x_spin = QtWidgets.QSpinBox()
        cells_x_spin.setRange(2, 10)
        cells_x_spin.setValue(3)
        cells_x_layout.addWidget(cells_x_spin)
        cells_layout.addLayout(cells_x_layout)
        
        cells_y_layout = QtWidgets.QHBoxLayout()
        cells_y_layout.addWidget(QtWidgets.QLabel("Y 方向:"))
        cells_y_spin = QtWidgets.QSpinBox()
        cells_y_spin.setRange(2, 10)
        cells_y_spin.setValue(3)
        cells_y_layout.addWidget(cells_y_spin)
        cells_layout.addLayout(cells_y_layout)
        
        cells_z_layout = QtWidgets.QHBoxLayout()
        cells_z_layout.addWidget(QtWidgets.QLabel("Z 方向:"))
        cells_z_spin = QtWidgets.QSpinBox()
        cells_z_spin.setRange(2, 10)
        cells_z_spin.setValue(3)
        cells_z_layout.addWidget(cells_z_spin)
        cells_layout.addLayout(cells_z_layout)
        
        cells_group.setLayout(cells_layout)
        layout.addWidget(cells_group)
    else:
        # STL 模式：显示信息
        info_label = QtWidgets.QLabel("填充空间：STL 模型\n将自动计算所需晶胞数量")
        info_label.setStyleSheet("color: gray;")
        layout.addWidget(info_label)
        
        # 虚拟的 spin boxes（不显示，但需要返回值）
        cells_x_spin = QtWidgets.QSpinBox()
        cells_y_spin = QtWidgets.QSpinBox()
        cells_z_spin = QtWidgets.QSpinBox()
        cells_x_spin.setValue(0)  # 0 表示自动计算
        cells_y_spin.setValue(0)
        cells_z_spin.setValue(0)
    
    # 分辨率
    res_group = QtWidgets.QGroupBox("分辨率")
    res_layout = QtWidgets.QVBoxLayout()
    
    res_layout.addWidget(QtWidgets.QLabel("单个晶胞的分辨率:"))
    res_spin = QtWidgets.QSpinBox()
    res_spin.setRange(16, 128)
    res_spin.setSingleStep(16)
    res_spin.setValue(48)
    res_layout.addWidget(res_spin)
    
    res_info = QtWidgets.QLabel(
        "推荐:\n"
        "• 32: 快速预览\n"
        "• 48: 平衡质量（推荐）\n"
        "• 64: 高质量"
    )
    res_info.setStyleSheet("color: gray; font-size: 10px;")
    res_layout.addWidget(res_info)
    
    res_group.setLayout(res_layout)
    layout.addWidget(res_group)
    
    # 平滑选项
    smooth_group = QtWidgets.QGroupBox("平滑选项")
    smooth_layout = QtWidgets.QVBoxLayout()
    
    sdf_smooth_check = QtWidgets.QCheckBox("启用 SDF 高斯平滑")
    sdf_smooth_check.setChecked(True)
    smooth_layout.addWidget(sdf_smooth_check)
    
    sigma_layout = QtWidgets.QHBoxLayout()
    sigma_layout.addWidget(QtWidgets.QLabel("Sigma:"))
    sigma_spin = QtWidgets.QDoubleSpinBox()
    sigma_spin.setRange(0.0, 3.0)
    sigma_spin.setSingleStep(0.1)
    sigma_spin.setValue(1.0)
    sigma_layout.addWidget(sigma_spin)
    smooth_layout.addLayout(sigma_layout)
    
    smooth_group.setLayout(smooth_layout)
    layout.addWidget(smooth_group)
    
    # Marching Cubes 选项
    mc_group = QtWidgets.QGroupBox("网格重建算法")
    mc_layout = QtWidgets.QVBoxLayout()
    
    use_dual_check = QtWidgets.QCheckBox("使用 Dual Contouring（保留锐特征）")
    use_dual_check.setChecked(PYMCUBES_AVAILABLE)  # 如果可用则默认启用
    use_dual_check.setEnabled(PYMCUBES_AVAILABLE)  # 如果不可用则禁用
    mc_layout.addWidget(use_dual_check)
    
    if not PYMCUBES_AVAILABLE:
        mc_info = QtWidgets.QLabel(
            "⚠ PyMCubes 未安装\n"
            "运行: pip install PyMCubes\n"
            "以启用 Dual Contouring（更好的锐特征保留）"
        )
        mc_info.setStyleSheet("color: orange; font-size: 10px;")
        mc_layout.addWidget(mc_info)
    else:
        mc_info = QtWidgets.QLabel(
            "✓ Dual Contouring 可用\n"
            "优点：更好地保留锐边和角\n"
            "适合：有明显边角的晶格结构"
        )
        mc_info.setStyleSheet("color: green; font-size: 10px;")
        mc_layout.addWidget(mc_info)
    
    mc_group.setLayout(mc_layout)
    layout.addWidget(mc_group)
    
    # 八叉树 SDF 选项
    octree_group = QtWidgets.QGroupBox("内存优化（八叉树 SDF）")
    octree_layout = QtWidgets.QVBoxLayout()
    
    use_octree_check = QtWidgets.QCheckBox("使用八叉树 SDF 存储（节省内存）")
    use_octree_check.setChecked(False)  # 默认不启用（实验性功能）
    use_octree_check.setEnabled(OCTREE_AVAILABLE)  # 如果不可用则禁用
    octree_layout.addWidget(use_octree_check)
    
    if not OCTREE_AVAILABLE:
        octree_info = QtWidgets.QLabel(
            "⚠ 八叉树模块未加载\n"
            "检查 core/octree_sdf.py 是否存在"
        )
        octree_info.setStyleSheet("color: orange; font-size: 10px;")
        octree_layout.addWidget(octree_info)
    else:
        octree_info = QtWidgets.QLabel(
            "✓ 八叉树 SDF 可用\n"
            "优点：节省 80-95% 内存\n"
            "适合：大规模晶格阵列（>5×5×5）\n"
            "注意：会增加计算时间"
        )
        octree_info.setStyleSheet("color: green; font-size: 10px;")
        octree_layout.addWidget(octree_info)
    
    octree_group.setLayout(octree_layout)
    layout.addWidget(octree_group)
    
    # 按钮
    button_box = QtWidgets.QDialogButtonBox(
        QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
    )
    button_box.accepted.connect(dialog.accept)
    button_box.rejected.connect(dialog.reject)
    layout.addWidget(button_box)
    
    dialog.setLayout(layout)
    
    if dialog.exec_() == QtWidgets.QDialog.Accepted:
        # 计算最终缩放比例
        if scale_ratio_radio.isChecked():
            final_scale = scale_ratio_spin.value()
        else:
            # 根据目标尺寸计算缩放比例
            if unit_cell_mesh is not None:
                original_size = unit_cell_mesh.bounds[1] - unit_cell_mesh.bounds[0]
                max_original_dim = original_size.max()
                final_scale = scale_absolute_spin.value() / max_original_dim
            else:
                final_scale = 1.0
        
        return {
            'num_cells': np.array([cells_x_spin.value(), cells_y_spin.value(), cells_z_spin.value()]),
            'unit_resolution': res_spin.value(),
            'sdf_smooth': sdf_smooth_check.isChecked(),
            'sdf_sigma': sigma_spin.value(),
            'scale_factor': final_scale,
            'shrink_factor': shrink_spin.value(),
            'use_dual_contouring': use_dual_check.isChecked(),
            'use_octree': use_octree_check.isChecked()
        }
    
    return None


def compute_unit_cell_sdf(mesh, resolution, shrink_factor=0.90, scale_factor=1.0):
    """
    计算单个晶胞的 SDF
    
    参数:
        mesh: 晶胞网格
        resolution: SDF 分辨率
        shrink_factor: 包围盒缩小系数（用于周期性连接）
                      - 0.90: 适用于大多数晶格
                      - 1.00: 适用于数学生成的周期性晶格（如 Gyroid）
        scale_factor: 晶胞整体缩放比例
    """
    print(f"\n[计算] 晶胞 SDF (分辨率: {resolution}³, 缩放: {scale_factor:.3f}x, shrink: {shrink_factor:.2f})...")
    
    # 记录原始尺寸
    original_bounds = mesh.bounds
    original_size = original_bounds[1] - original_bounds[0]
    print(f"  输入 STL 尺寸: {original_size}")
    
    # 如果需要缩放，先缩放网格
    if scale_factor != 1.0:
        mesh = mesh.copy()
        mesh.apply_scale(scale_factor)
        print(f"  ✓ 晶胞已缩放 {scale_factor:.3f}x")
    
    bounds = mesh.bounds
    center = (bounds[0] + bounds[1]) / 2
    size = bounds[1] - bounds[0]
    
    if scale_factor != 1.0:
        print(f"  缩放后尺寸: {size}")
    
    # 调整包围盒
    if shrink_factor < 1.0:
        adjusted_size = size * shrink_factor
        bbox_min = center - adjusted_size / 2
        bbox_max = center + adjusted_size / 2
        print(f"  SDF 包围盒尺寸: {adjusted_size} (shrink_factor={shrink_factor})")
    else:
        # shrink_factor = 1.0，不缩小（用于数学生成的周期性晶格）
        bbox_min = bounds[0]
        bbox_max = bounds[1]
        adjusted_size = size
        print(f"  SDF 包围盒尺寸: {adjusted_size} (无缩小，周期性晶格)")
    
    # 生成网格点
    x = np.linspace(bbox_min[0], bbox_max[0], resolution)
    y = np.linspace(bbox_min[1], bbox_max[1], resolution)
    z = np.linspace(bbox_min[2], bbox_max[2], resolution)
    
    points = np.stack(np.meshgrid(x, y, z, indexing='ij'), axis=-1)
    points_flat = points.reshape(-1, 3)
    
    # 计算 SDF
    t_start = time.time()
    pv_mesh = pv.wrap(mesh)
    point_cloud = pv.PolyData(points_flat)
    distances = point_cloud.compute_implicit_distance(pv_mesh, inplace=False)
    sdf_values = distances['implicit_distance']
    t_end = time.time()
    
    sdf_grid = sdf_values.reshape(resolution, resolution, resolution)
    
    print(f"  ✓ SDF 计算完成 (耗时: {t_end - t_start:.2f}s)")
    print(f"  SDF 范围: [{sdf_grid.min():.3f}, {sdf_grid.max():.3f}]")
    
    # 检查SDF是否有效
    if sdf_grid.min() >= 0:
        print(f"  ⚠⚠⚠ 警告：SDF 全是正值，没有内部结构！")
        print(f"  可能原因：")
        print(f"    1. 分辨率太低（{resolution}³），采样点错过了晶格杆件")
        print(f"    2. shrink_factor 太小（{shrink_factor}），包围盒缩得太小")
        print(f"    3. 晶胞本身是空心的或杆件太细")
    
    # 生成单个晶胞的预览网格
    print(f"\n  [预览] 从SDF重建单个晶胞...")
    try:
        t_preview_start = time.time()
        
        # 计算spacing
        spacing_preview = adjusted_size / resolution
        
        # Marching Cubes重建
        vertices_unit, faces_unit = marching_cubes_adaptive(
            sdf_grid, level=0.0, spacing=spacing_preview, use_dual=False  # 单个晶胞用传统MC
        )
        vertices_unit += bbox_min
        
        t_preview = time.time() - t_preview_start
        
        print(f"  ✓ 单个晶胞重建完成 (耗时: {t_preview:.2f}s)")
        print(f"  顶点数: {len(vertices_unit):,}, 面数: {len(faces_unit):,}")
        
        # 创建网格
        unit_cell_reconstructed = trimesh.Trimesh(
            vertices=vertices_unit,
            faces=faces_unit,
            process=False
        )
        
        # 分析连通性
        components_unit = unit_cell_reconstructed.split(only_watertight=False)
        num_components_unit = len(components_unit)
        print(f"  连通分量数: {num_components_unit}")
        
        if num_components_unit == 1:
            print(f"  ✓ 单个晶胞完整")
        else:
            print(f"  ⚠ 单个晶胞有 {num_components_unit} 个分量（破碎）")
            print(f"  这会导致周期性阵列中每个晶胞都破碎！")
        
        # 保存单个晶胞重建结果
        output_filename = f"unit_cell_reconstructed_res{resolution}.stl"
        output_path = project_root / "tests" / output_filename
        try:
            unit_cell_reconstructed.export(str(output_path))
            print(f"  ✓ 单个晶胞重建已保存: {output_path}")
        except Exception as e:
            print(f"  ⚠ 保存失败: {e}")
        
        # 可视化对比：原始 vs 重建
        print(f"\n  [可视化] 原始晶胞 vs SDF重建...")
        plotter = pv.Plotter(shape=(1, 2), window_size=(1600, 800))
        
        # 左侧：原始晶胞
        plotter.subplot(0, 0)
        plotter.add_text(f"原始晶胞\n({len(mesh.vertices):,} 顶点, {len(mesh.faces):,} 面)", font_size=12)
        plotter.add_mesh(pv.wrap(mesh), color='lightblue', opacity=0.8, show_edges=True)
        
        # 右侧：SDF重建
        plotter.subplot(0, 1)
        title = f"SDF重建 (分辨率{resolution}³)\n({len(vertices_unit):,} 顶点, {len(faces_unit):,} 面)"
        if num_components_unit > 1:
            title += f"\n⚠ {num_components_unit} 个分量（破碎）"
            color = 'red'
        else:
            title += f"\n✓ 完整"
            color = 'green'
        
        plotter.add_text(title, font_size=12)
        plotter.add_mesh(pv.wrap(unit_cell_reconstructed), color=color, opacity=0.8, show_edges=True)
        
        plotter.link_views()
        plotter.show()
        
        print(f"\n  提示：如果单个晶胞重建就破碎，需要：")
        print(f"    1. 提高分辨率（当前 {resolution}³，建议 ≥48³）")
        print(f"    2. 调整 shrink_factor（当前 {shrink_factor}，试试 0.95 或 1.0）")
        print(f"    3. 检查原始晶胞STL是否完整")
        
    except Exception as e:
        print(f"  ✗ 单个晶胞重建失败: {e}")
        import traceback
        traceback.print_exc()
    
    return sdf_grid, adjusted_size, bbox_min, bbox_max, mesh


def generate_periodic_lattice_sdf(
    unit_sdf,
    target_resolution,
    target_bbox_min,
    spacing,
    unit_bbox_min,
    cell_size,
    unit_resolution,
    total_voxels,
    use_octree=False
):
    """
    生成周期性晶格 SDF（使用 C++ 加速或八叉树）
    
    参数:
        unit_sdf: 单个晶胞的 SDF
        target_resolution: 目标分辨率 [nx, ny, nz]
        target_bbox_min: 目标包围盒最小点
        spacing: 体素间距
        unit_bbox_min: 晶胞包围盒最小点
        cell_size: 晶胞尺寸
        unit_resolution: 晶胞 SDF 分辨率
        total_voxels: 总体素数
        use_octree: 是否使用八叉树存储（节省内存）
    
    返回:
        lattice_sdf: 周期性晶格 SDF（密集网格或八叉树对象）
    """
    t_start = time.time()
    
    # 如果使用八叉树
    if use_octree and OCTREE_AVAILABLE:
        print(f"  使用八叉树 SDF 存储（节省内存）...")
        
        # 计算包围盒
        target_bbox_max = target_bbox_min + target_resolution * spacing
        
        # 根据晶胞数量自动调整深度
        num_cells_total = np.prod(target_resolution // unit_resolution)
        if num_cells_total > 1000:
            max_depth = 10  # 超大阵列
        elif num_cells_total > 100:
            max_depth = 9   # 大阵列
        elif num_cells_total > 10:
            max_depth = 8   # 中等阵列
        else:
            max_depth = 7   # 小阵列
        
        print(f"  八叉树深度: {max_depth} (晶胞总数: {num_cells_total})")
        
        # 叶节点分辨率：根据单元分辨率调整
        leaf_resolution = min(8, unit_resolution // 4)  # 通常 4-8
        uniform_threshold = 0.01  # SDF 变化阈值
        
        # 尝试使用 C++ 加速版本
        try:
            from core.octree_sdf import build_octree_from_periodic_sdf_cpp, CPP_OCTREE_AVAILABLE
            
            if CPP_OCTREE_AVAILABLE:
                print(f"  使用 C++ 加速构建八叉树...")
                
                # 确保 cell_size 是标量
                cell_size_scalar = float(cell_size) if np.isscalar(cell_size) else float(np.max(cell_size))
                
                octree = build_octree_from_periodic_sdf_cpp(
                    unit_sdf,
                    target_bbox_min,
                    target_bbox_max,
                    cell_size_scalar,
                    unit_bbox_min,
                    unit_resolution,
                    max_depth=max_depth,
                    leaf_resolution=leaf_resolution,
                    uniform_threshold=uniform_threshold,
                    verbose=True
                )
                
                t_lattice = time.time() - t_start
                print(f"  ✓ C++ 加速八叉树构建完成 (总耗时: {t_lattice:.1f}s)")
                
                return octree
            else:
                print(f"  ⚠ C++ 八叉树模块未加载，使用 Python 版本...")
        except Exception as e:
            print(f"  ⚠ C++ 加速失败: {e}")
            print(f"  回退到 Python 版本...")
            import traceback
            traceback.print_exc()
        
        # Python 版本（回退）
        print(f"  使用 Python 版本构建八叉树（较慢）...")
        
        # 确保 cell_size 是标量
        cell_size_scalar = float(cell_size) if np.isscalar(cell_size) else float(np.max(cell_size))
        
        # 创建 SDF 查询函数（周期性采样）
        inv_cell_size = 1.0 / cell_size_scalar
        
        def periodic_sdf_func(x, y, z):
            """周期性 SDF 查询函数"""
            world_pos = np.array([x, y, z])
            
            # 映射到晶胞内（周期性）
            local_pos = world_pos - unit_bbox_min
            local_pos_mod = np.mod(local_pos, cell_size_scalar)
            
            # 转换为单元 SDF 的索引（最近邻）
            unit_indices = (local_pos_mod * inv_cell_size * unit_resolution).astype(int)
            unit_indices = np.clip(unit_indices, 0, unit_resolution - 1)
            
            return unit_sdf[unit_indices[0], unit_indices[1], unit_indices[2]]
        
        octree = OctreeSDF(target_bbox_min, target_bbox_max, max_depth=max_depth)
        
        # 构建八叉树
        octree.build_from_function(
            periodic_sdf_func,
            leaf_resolution=leaf_resolution,
            uniform_threshold=uniform_threshold,
            verbose=True
        )
        
        t_lattice = time.time() - t_start
        print(f"  ✓ Python 八叉树构建完成 (耗时: {t_lattice:.1f}s)")
        
        # 返回八叉树对象（而不是密集网格）
        return octree
    
    # 否则使用传统方法（密集网格）
    if CPP_AVAILABLE:
        print(f"  使用 C++ 加速版本（最近邻采样）...")
        try:
            lattice_sdf = cpp_sample_periodic_sdf(
                unit_sdf.astype(np.float32),
                target_resolution.astype(np.int32),
                target_bbox_min,
                spacing,
                unit_bbox_min,
                cell_size,
                unit_resolution
            )
            t_lattice = time.time() - t_start
            print(f"  ✓ C++ 加速完成 (耗时: {t_lattice:.1f}s)")
            return lattice_sdf
        except Exception as e:
            print(f"  ⚠ C++ 加速失败: {e}")
            print(f"  回退到 Python 版本...")
    else:
        print(f"  C++ 加速模块未加载，使用 Python 版本...")
    
    # Python 版本（回退）- 使用最近邻采样（与C++一致）
    print(f"  使用 Python 版本（最近邻采样，较慢）...")
    
    lattice_sdf = np.zeros(tuple(target_resolution), dtype=np.float32)
    
    progress_step = max(1, total_voxels // 20)
    
    # 预计算常量
    inv_cell_size = 1.0 / cell_size
    
    for idx in range(total_voxels):
        i = idx // (target_resolution[1] * target_resolution[2])
        j = (idx % (target_resolution[1] * target_resolution[2])) // target_resolution[2]
        k = idx % target_resolution[2]
        
        if idx % progress_step == 0:
            progress = idx / total_voxels * 100
            elapsed = time.time() - t_start
            if elapsed > 0 and idx > 0:
                eta = elapsed / idx * (total_voxels - idx)
                print(f"  进度: {progress:.0f}% (预计剩余: {eta:.0f}s)", end='\r')
        
        # 世界坐标
        world_pos = target_bbox_min + np.array([i, j, k]) * spacing
        
        # 映射到晶胞内（周期性）
        local_pos = world_pos - unit_bbox_min
        local_pos_mod = np.mod(local_pos, cell_size)
        
        # 转换为单元 SDF 的索引（最近邻）
        unit_indices = (local_pos_mod * inv_cell_size * unit_resolution).astype(int)
        unit_indices = np.clip(unit_indices, 0, unit_resolution - 1)
        
        lattice_sdf[i, j, k] = unit_sdf[unit_indices[0], unit_indices[1], unit_indices[2]]
    
    t_lattice = time.time() - t_start
    print(f"  进度: 100% (耗时: {t_lattice:.1f}s)          ")
    
    return lattice_sdf


def fill_stl_space_with_lattice(
    unit_sdf,
    cell_size,
    unit_bbox_min,
    fill_space_mesh,
    unit_resolution,
    sdf_smooth_sigma=1.0,
    scaled_unit_cell=None,
    num_cells_hint=None,
    use_dual_contouring=False,
    use_octree=False
):
    """
    在 STL 模型空间内填充周期性晶格
    
    参数:
        unit_sdf: 单个晶胞的 SDF
        cell_size: 晶胞尺寸
        unit_bbox_min: 晶胞包围盒最小点
        fill_space_mesh: 填充空间网格
        unit_resolution: 单个晶胞的分辨率
        sdf_smooth_sigma: SDF 平滑参数
        use_octree: 是否使用八叉树存储（节省内存）
    """
    print("\n" + "=" * 70)
    print("STL 空间周期性填充")
    print("=" * 70)
    
    # 1. 分析填充空间
    fill_bounds = fill_space_mesh.bounds
    fill_center = (fill_bounds[0] + fill_bounds[1]) / 2
    fill_size = fill_bounds[1] - fill_bounds[0]
    
    print(f"\n[步骤 1] 分析填充空间")
    print(f"  填充空间尺寸: {fill_size}")
    print(f"  晶胞尺寸: {cell_size}")
    
    # 计算需要的晶胞数量
    num_cells = np.ceil(fill_size / cell_size).astype(int) + 2
    print(f"  需要晶胞数: {num_cells} (总计 {np.prod(num_cells)} 个)")
    
    # 2. 设置目标区域
    target_size = num_cells * cell_size
    target_bbox_min = fill_center - target_size / 2
    target_bbox_max = fill_center + target_size / 2
    
    # 目标分辨率（步骤3：用户设置，步骤4：自动降低）
    num_cells_total = np.prod(num_cells)
    
    # 步骤3：使用用户设置的分辨率（unit_resolution，如 48）
    target_resolution_per_cell = unit_resolution  # 来自用户设置
    target_resolution = (num_cells * target_resolution_per_cell).astype(int)
    total_voxels = np.prod(target_resolution)
    
    print(f"\n[步骤 2] 分辨率设置")
    print(f"  晶胞总数: {num_cells_total}")
    print(f"  目标分辨率（晶格，步骤3）: {target_resolution} (每晶胞{target_resolution_per_cell}³)")
    print(f"  填充空间分辨率（裁剪，步骤4）: 固定16³/晶胞 + 插值")
    print(f"  总体素数: {total_voxels:,}")
    
    # 内存估算
    memory_mb = total_voxels * 4 / (1024 * 1024)
    print(f"  预计内存: {memory_mb:.1f} MB")
    
    # 估算最终网格大小（粗略估计）
    # 假设每个体素平均产生 0.5 个三角形（经验值）
    estimated_faces = total_voxels * 0.5
    estimated_faces_millions = estimated_faces / 1_000_000
    
    print(f"\n  ⚠ 预估最终网格大小:")
    print(f"    预计面数: ~{estimated_faces_millions:.1f}M 面")
    
    if estimated_faces > 100_000_000:
        print(f"    ⚠⚠⚠ 警告：预计网格非常大（>100M 面）！")
        print(f"    这会导致：")
        print(f"      1. 内存溢出（计算体积、导出STL时）")
        print(f"      2. 文件巨大（可能几 GB）")
        print(f"      3. 无法在大多数软件中打开")
        print(f"      4. 3D 打印切片软件可能崩溃")
        print(f"    强烈建议：")
        print(f"      1. 降低分辨率（当前每晶胞 {target_resolution_per_cell}³，建议降到 24³ 或更低）")
        print(f"      2. 减少晶胞数量")
        print(f"      3. 使用更简单的晶胞模型（减面）")
        response = input(f"\n  是否仍要继续？(y/n): ")
        if response.lower() != 'y':
            print(f"  已取消")
            return None
    elif estimated_faces > 50_000_000:
        print(f"    ⚠⚠ 警告：预计网格很大（>50M 面）")
        print(f"    可能会遇到内存问题")
        print(f"    建议降低分辨率或减少晶胞数量")
        response = input(f"\n  是否继续？(y/n): ")
        if response.lower() != 'y':
            print(f"  已取消")
            return None
    elif estimated_faces > 20_000_000:
        print(f"    ⚠ 提示：预计网格较大（>20M 面）")
        print(f"    处理时间可能较长，但应该可以完成")
    else:
        print(f"    ✓ 预计网格大小合理")
    
    if memory_mb > 2000:
        print(f"\n  ⚠ 警告：内存需求较大")
        response = input("  是否继续？(y/n): ")
        if response.lower() != 'y':
            return None
    
    spacing = target_size / target_resolution
    
    # 3. 生成周期性晶格 SDF
    print(f"\n[步骤 3] 生成周期性晶格 SDF...")
    
    lattice_sdf = generate_periodic_lattice_sdf(
        unit_sdf,
        target_resolution,
        target_bbox_min,
        spacing,
        unit_bbox_min,
        cell_size,
        unit_resolution,
        total_voxels,
        use_octree=use_octree
    )
    
    # 检查是否返回八叉树对象
    is_octree = isinstance(lattice_sdf, OctreeSDF) if OCTREE_AVAILABLE else False
    
    if is_octree:
        print(f"  ✓ 使用八叉树存储（节省内存）")
        print(f"  八叉树节点数: {lattice_sdf.num_nodes:,}")
        print(f"  内存节省: {lattice_sdf.memory_saved_ratio * 100:.1f}%")
        
        # 从八叉树查询 SDF 范围（采样一些点）
        sample_points = 1000
        sample_values = []
        for _ in range(sample_points):
            pos = target_bbox_min + np.random.rand(3) * (target_bbox_max - target_bbox_min)
            sample_values.append(lattice_sdf.query(*pos))
        
        sdf_min = min(sample_values)
        sdf_max = max(sample_values)
        print(f"  ✓ 晶格 SDF 范围（采样）: [{sdf_min:.3f}, {sdf_max:.3f}]")
        
        # 统计内部体素（采样估算）
        interior_count = sum(1 for v in sample_values if v < 0)
        interior_ratio = interior_count / sample_points * 100
        print(f"  内部体素（估算）: {interior_ratio:.1f}%")
    else:
        print(f"  ✓ 晶格 SDF 范围: [{lattice_sdf.min():.3f}, {lattice_sdf.max():.3f}]")
        
        # 统计内部体素
        interior_voxels = np.sum(lattice_sdf < 0)
        interior_ratio = interior_voxels / total_voxels * 100
        print(f"  内部体素: {interior_voxels:,} ({interior_ratio:.1f}%)")
    
    # 检查是否有有效的晶格结构
    if is_octree:
        # 八叉树：通过采样检查
        if sdf_min >= 0:
            print(f"\n  ⚠⚠⚠ 警告：晶格 SDF 全是正值（外部），没有内部结构！")
            print(f"  可能原因：")
            print(f"    1. 分辨率太低（当前 {unit_resolution}³），采样点错过了晶格杆件")
            print(f"    2. shrink_factor 太小，包围盒缩得太小")
            print(f"    3. 晶胞本身是空心的或杆件太细")
            print(f"  建议：")
            print(f"    1. 提高分辨率到至少 32³ 或 48³")
            print(f"    2. 增大 shrink_factor 到 1.0")
            print(f"    3. 检查晶胞 STL 文件")
            
            # 询问是否继续生成预览
            response = input(f"\n  是否仍要生成预览（可能是空的）？(y/n): ")
            if response.lower() != 'y':
                print(f"  已取消")
                return None
    else:
        # 密集网格：直接检查
        if lattice_sdf.min() >= 0:
            print(f"\n  ⚠⚠⚠ 警告：晶格 SDF 全是正值（外部），没有内部结构！")
            print(f"  可能原因：")
            print(f"    1. 分辨率太低（当前 {unit_resolution}³），采样点错过了晶格杆件")
            print(f"    2. shrink_factor 太小，包围盒缩得太小")
            print(f"    3. 晶胞本身是空心的或杆件太细")
            print(f"  建议：")
            print(f"    1. 提高分辨率到至少 32³ 或 48³")
            print(f"    2. 增大 shrink_factor 到 1.0")
            print(f"    3. 检查晶胞 STL 文件")
            
            # 询问是否继续生成预览
            response = input(f"\n  是否仍要生成预览（可能是空的）？(y/n): ")
            if response.lower() != 'y':
                print(f"  已取消")
                return None
    
    # 生成步骤3的晶格预览
    print(f"\n[步骤 3 预览] 生成周期性晶格网格...")
    t_start = time.time()
    
    # 如果使用八叉树，需要先转换为密集网格
    if is_octree:
        print(f"  将八叉树转换为密集网格（用于 Marching Cubes）...")
        lattice_sdf_dense = lattice_sdf.to_dense_grid(tuple(target_resolution))
        print(f"  ✓ 转换完成")
    else:
        lattice_sdf_dense = lattice_sdf
    
    # 确保 spacing 是 tuple
    spacing_tuple = tuple(spacing) if hasattr(spacing, '__iter__') else (spacing, spacing, spacing)
    
    try:
        # 使用自适应 Marching Cubes（支持 Dual Contouring）
        vertices_preview, faces_preview = marching_cubes_adaptive(
            lattice_sdf_dense, level=0.0, spacing=spacing_tuple, use_dual=use_dual_contouring
        )
        vertices_preview += target_bbox_min
        
        t_mc = time.time() - t_start
        print(f"  ✓ Marching Cubes 完成 (耗时: {t_mc:.1f}s)")
        print(f"  顶点数: {len(vertices_preview):,}, 面数: {len(faces_preview):,}")
        
        # 创建网格
        # 先检查面数，如果太大需要简化
        needs_simplification = len(faces_preview) > 10_000_000
        
        if needs_simplification:
            print(f"\n  ⚠ 网格很大（{len(faces_preview)/1_000_000:.1f}M 面），需要简化以避免内存溢出")
            
            # 使用 Open3D 直接简化（不创建 Trimesh 对象）
            try:
                import open3d as o3d
                
                # 目标面数：统一减少到原始面数的50%
                target_faces = int(len(faces_preview) * 0.5)
                
                print(f"  原始面数: {len(faces_preview)/1_000_000:.1f}M")
                print(f"  目标面数: {target_faces/1_000_000:.1f}M (50%)")
                print(f"  使用 Open3D Quadric Decimation 简化...")
                
                t_simp_start = time.time()
                
                # 创建 Open3D 网格（直接从顶点和面）
                o3d_mesh = o3d.geometry.TriangleMesh()
                o3d_mesh.vertices = o3d.utility.Vector3dVector(vertices_preview)
                o3d_mesh.triangles = o3d.utility.Vector3iVector(faces_preview)
                
                # 计算法向量
                o3d_mesh.compute_vertex_normals()
                
                # Quadric Decimation 简化
                simplified_o3d = o3d_mesh.simplify_quadric_decimation(target_number_of_triangles=target_faces)
                
                # 转换回 numpy 数组
                vertices_preview = np.asarray(simplified_o3d.vertices)
                faces_preview = np.asarray(simplified_o3d.triangles)
                
                t_simp = time.time() - t_simp_start
                
                print(f"  ✓ 简化完成 (耗时: {t_simp:.1f}s)")
                print(f"  简化后: {len(vertices_preview):,} 顶点, {len(faces_preview):,} 面")
                print(f"  实际简化率: {(1 - len(faces_preview)/int(len(faces_preview)/0.5))*100:.1f}%")
                
            except ImportError:
                print(f"  ✗ Open3D 未安装，无法简化")
                print(f"  建议: pip install open3d")
                print(f"  将尝试继续（可能会内存溢出）...")
            except Exception as e:
                print(f"  ✗ 简化失败: {e}")
                print(f"  将尝试继续（可能会内存溢出）...")
        
        # 创建 Trimesh 对象
        try:
            lattice_preview = trimesh.Trimesh(
                vertices=vertices_preview, 
                faces=faces_preview, 
                process=False
            )
            print(f"  ✓ Trimesh 对象创建成功")
        except MemoryError as e:
            print(f"  ✗ 内存不足: {e}")
            print(f"  建议：降低分辨率或安装 Open3D")
            return None
        
        # 分析连通性
        components = lattice_preview.split(only_watertight=False)
        num_components = len(components)
        print(f"  连通分量数: {num_components}")
        
        if num_components == 1:
            print(f"  ✓✓✓ 完美！所有晶胞连接成一个整体")
        else:
            print(f"  ⚠ 有 {num_components} 个分量")
            if num_components <= 10:
                volumes = [comp.volume for comp in components[:10]]
                print(f"  各分量体积: {[f'{v:.2f}' for v in volumes]}")
        
        # 保存预览
        output_filename = f"lattice_preview_res{unit_resolution}.stl"
        output_path = project_root / "tests" / output_filename
        
        if len(faces_preview) < 50_000_000:
            try:
                lattice_preview.export(str(output_path))
                print(f"\n  ✓ 预览已保存: {output_path}")
            except Exception as e:
                print(f"\n  ⚠ 保存失败: {e}")
        else:
            print(f"\n  ⚠ 网格太大（{len(faces_preview):,} 面），跳过保存")
        
        # 可视化预览
        print(f"\n[可视化] 步骤3晶格预览...")
        plotter = pv.Plotter(shape=(1, 2), window_size=(1600, 800))
        
        # 左侧：单个晶胞
        plotter.subplot(0, 0)
        plotter.add_text("单个晶胞", font_size=12)
        
        # 使用传入的 scaled_unit_cell，如果没有则创建一个简单的占位符
        if scaled_unit_cell is not None:
            plotter.add_mesh(pv.wrap(scaled_unit_cell), color='lightgray', opacity=0.5, show_edges=True)
        else:
            # 如果没有 scaled_unit_cell，显示包围盒
            from trimesh.primitives import Box
            bbox_center = (unit_bbox_min + cell_size / 2)
            unit_box = Box(center=bbox_center, extents=cell_size)
            plotter.add_mesh(pv.wrap(unit_box), color='lightgray', opacity=0.2, show_edges=True, style='wireframe')
        
        # 右侧：周期性晶格预览
        plotter.subplot(0, 1)
        title = f"周期性晶格预览\n({num_cells[0]}×{num_cells[1]}×{num_cells[2]} 晶胞, {unit_resolution}³/晶胞)"
        if needs_simplification:
            title += f"\n(已简化到 {len(faces_preview)/1_000_000:.1f}M 面)"
        plotter.add_text(title, font_size=12)
        
        if num_components == 1:
            color = 'green'
        elif num_components <= 5:
            color = 'yellow'
        else:
            color = 'red'
        
        plotter.add_mesh(pv.wrap(lattice_preview), color=color, opacity=0.9, show_edges=True, line_width=0.5)
        
        plotter.link_views()
        plotter.show()
        
        print(f"\n✓ 步骤3预览完成")
        if needs_simplification:
            print(f"  注意：预览已简化，实际生成时可能需要更高分辨率")
        
        # 询问是否继续到步骤4-7（裁剪）
        print(f"\n" + "=" * 70)
        response = input("是否继续步骤4-7（裁剪到填充空间）？(y/n): ")
        if response.lower() != 'y':
            print(f"已停止，仅生成步骤3预览")
            return None
        
        # 继续执行步骤4-7
        print(f"\n继续执行步骤4-7...")
        
    except Exception as e:
        print(f"  ✗ 预览生成失败: {e}")
        print(f"  可能原因：SDF 不包含 0 值（无法提取等值面）")
        return None
    
    # 4. 计算填充空间 SDF
    print(f"\n[步骤 4] 计算填充空间 SDF...")
    
    t_start = time.time()
    
    # 策略：使用固定低分辨率（16³/晶胞）+ 插值
    # 优点：快速，内存占用小
    # 缺点：边界精度略有损失（但通过插值可以接受）
    
    fill_sdf_resolution_per_cell = 16  # 固定使用16³/晶胞
    fill_sdf_resolution = (num_cells * fill_sdf_resolution_per_cell).astype(int)
    
    print(f"  策略：低分辨率（{fill_sdf_resolution_per_cell}³/晶胞）+ 插值")
    print(f"  低分辨率: {fill_sdf_resolution} ({np.prod(fill_sdf_resolution):,} 点)")
    print(f"  目标分辨率: {target_resolution} ({np.prod(target_resolution):,} 点)")
    
    fill_grid = np.stack(
        np.meshgrid(
            np.linspace(target_bbox_min[0], target_bbox_max[0], fill_sdf_resolution[0]),
            np.linspace(target_bbox_min[1], target_bbox_max[1], fill_sdf_resolution[1]),
            np.linspace(target_bbox_min[2], target_bbox_max[2], fill_sdf_resolution[2]),
            indexing='ij'
        ),
        axis=-1
    )
    fill_grid_flat = fill_grid.reshape(-1, 3)
    
    # 使用 PyVista 计算有符号距离场（SDF）
    print(f"  使用 PyVista 计算有符号距离场（{len(fill_grid_flat):,} 点）...")
    pv_fill = pv.wrap(fill_space_mesh)
    point_cloud = pv.PolyData(fill_grid_flat)
    distances = point_cloud.compute_implicit_distance(pv_fill, inplace=False)
    fill_sdf_flat = distances['implicit_distance']
    
    t_fill = time.time() - t_start
    print(f"  ✓ 低分辨率计算完成 (耗时: {t_fill:.1f}s)")
    
    # 插值到目标分辨率
    fill_sdf_lowres = fill_sdf_flat.reshape(tuple(fill_sdf_resolution))
    
    if tuple(fill_sdf_resolution) != tuple(target_resolution):
        print(f"  插值到目标分辨率...")
        t_interp_start = time.time()
        from scipy.ndimage import zoom
        zoom_factors = target_resolution / fill_sdf_resolution
        fill_sdf = zoom(fill_sdf_lowres, zoom_factors, order=1)
        
        # 确保形状完全匹配（处理浮点精度问题）
        if fill_sdf.shape != tuple(target_resolution):
            print(f"  调整形状: {fill_sdf.shape} → {tuple(target_resolution)}")
            actual_zoom = np.array(target_resolution) / np.array(fill_sdf.shape)
            fill_sdf = zoom(fill_sdf, actual_zoom, order=1)
        
        t_interp = time.time() - t_interp_start
        print(f"  ✓ 插值完成 (耗时: {t_interp:.1f}s)")
    else:
        fill_sdf = fill_sdf_lowres
    
    # 调试：检查最终 SDF
    print(f"  最终 fill_sdf 范围: [{fill_sdf.min():.3f}, {fill_sdf.max():.3f}]")
    print(f"  内部点数（<0）: {np.sum(fill_sdf < 0):,} ({np.sum(fill_sdf < 0)/np.prod(fill_sdf.shape)*100:.1f}%)")
    
    print(f"  填充空间 SDF 形状: {fill_sdf.shape}")
    print(f"  填充空间 SDF 范围: [{fill_sdf.min():.3f}, {fill_sdf.max():.3f}]")
    
    # 5. 求交
    print(f"\n[步骤 5] 裁剪到填充空间内...")
    
    # 如果使用八叉树，需要先转换为密集网格
    if is_octree:
        print(f"  将八叉树转换为密集网格（用于裁剪）...")
        lattice_sdf_dense = lattice_sdf.to_dense_grid(tuple(target_resolution))
        print(f"  ✓ 转换完成")
    else:
        lattice_sdf_dense = lattice_sdf
    
    # 确保形状匹配
    print(f"  lattice_sdf 形状: {lattice_sdf_dense.shape}")
    print(f"  fill_sdf 形状: {fill_sdf.shape}")
    
    if lattice_sdf_dense.shape != fill_sdf.shape:
        print(f"  ⚠ 形状不匹配！调整 fill_sdf...")
        from scipy.ndimage import zoom
        zoom_factors = np.array(lattice_sdf_dense.shape) / np.array(fill_sdf.shape)
        fill_sdf = zoom(fill_sdf, zoom_factors, order=1)
        print(f"  调整后 fill_sdf 形状: {fill_sdf.shape}")
    
    # 调试信息
    print(f"  lattice_sdf 范围: [{lattice_sdf_dense.min():.3f}, {lattice_sdf_dense.max():.3f}]")
    print(f"  fill_sdf 范围: [{fill_sdf.min():.3f}, {fill_sdf.max():.3f}]")
    
    combined_sdf = np.maximum(lattice_sdf_dense, fill_sdf)
    
    interior_voxels = np.sum(combined_sdf < 0)
    interior_ratio = interior_voxels / total_voxels * 100
    print(f"  ✓ 内部体素: {interior_voxels:,} ({interior_ratio:.1f}%)")
    print(f"  最终 SDF 范围: [{combined_sdf.min():.3f}, {combined_sdf.max():.3f}]")
    
    # 检查是否包含 0 值
    if combined_sdf.min() > 0 or combined_sdf.max() < 0:
        print(f"  ⚠ 警告：SDF 范围不包含 0，无法提取等值面！")
        print(f"  这通常意味着晶格和填充空间没有交集")
        return None
    
    # 6. 平滑
    if sdf_smooth_sigma > 0:
        print(f"\n[步骤 6] SDF 平滑 (sigma={sdf_smooth_sigma:.1f})...")
        t_start = time.time()
        combined_sdf = gaussian_filter(combined_sdf, sigma=sdf_smooth_sigma)
        t_smooth = time.time() - t_start
        print(f"  ✓ 完成 (耗时: {t_smooth:.1f}s)")
    else:
        print(f"\n[步骤 6] SDF 平滑: 已跳过（sigma=0）")
    
    # 确保 spacing 是 tuple（在步骤7之前定义，供步骤7a和7b使用）
    spacing_tuple = tuple(spacing) if hasattr(spacing, '__iter__') else (spacing, spacing, spacing)
    
    # 7. Marching Cubes 重建 - 裁剪前的晶格
    print(f"\n[步骤 7a] 重建裁剪前的完整晶格...")
    
    # 预估面数
    estimated_faces_before = np.prod(target_resolution) * 0.5
    
    if estimated_faces_before > 100_000_000:
        print(f"  ⚠ 预估面数过大（~{estimated_faces_before/1_000_000:.1f}M 面），跳过裁剪前网格生成")
        print(f"  （避免内存溢出）")
        lattice_before_clip = None
    else:
        t_start = time.time()
        
        try:
            # 如果使用八叉树，需要先转换为密集网格（如果还没转换）
            if is_octree and 'lattice_sdf_dense' not in locals():
                print(f"  将八叉树转换为密集网格...")
                lattice_sdf_dense = lattice_sdf.to_dense_grid(tuple(target_resolution))
                print(f"  ✓ 转换完成")
            
            vertices_before, faces_before, _, _ = measure.marching_cubes(
                lattice_sdf_dense, level=0.0, spacing=spacing_tuple
            )
            vertices_before += target_bbox_min
            
            t_mc = time.time() - t_start
            print(f"  ✓ Marching Cubes 完成 (耗时: {t_mc:.1f}s)")
            print(f"  顶点数: {len(vertices_before):,}, 面数: {len(faces_before):,}")
            
            # 检查面数，如果太大就不创建 trimesh 对象（避免内存溢出）
            if len(faces_before) > 50_000_000:
                print(f"  ⚠ 面数过大（{len(faces_before)/1_000_000:.1f}M），跳过 Trimesh 对象创建")
                print(f"  （Trimesh 处理大网格时会内存溢出）")
                lattice_before_clip = None
            else:
                # 使用 process=False 避免自动处理导致的内存问题
                lattice_before_clip = trimesh.Trimesh(
                    vertices=vertices_before, 
                    faces=faces_before, 
                    process=False  # 关键：不自动处理
                )
                print(f"  ✓ Trimesh 对象创建成功")
        except MemoryError as e:
            print(f"  ✗ 内存不足，跳过裁剪前网格: {e}")
            lattice_before_clip = None
        except Exception as e:
            print(f"  ✗ 创建失败: {e}")
            lattice_before_clip = None
    
    # 8. Marching Cubes 重建 - 裁剪后的晶格
    print(f"\n[步骤 7b] 重建裁剪后的晶格...")
    
    # 再次检查 SDF 范围
    print(f"  combined_sdf 形状: {combined_sdf.shape}")
    print(f"  combined_sdf 范围: [{combined_sdf.min():.3f}, {combined_sdf.max():.3f}]")
    print(f"  包含 0 的体素数: {np.sum((combined_sdf >= -0.1) & (combined_sdf <= 0.1))}")
    
    # 如果 SDF 不包含 0，无法提取等值面
    if combined_sdf.min() > 0:
        print(f"  ✗ 错误：所有 SDF 值都是正数（外部），没有内部结构！")
        print(f"  可能原因：晶格和填充空间没有交集")
        return None
    
    if combined_sdf.max() < 0:
        print(f"  ✗ 错误：所有 SDF 值都是负数（内部），没有表面！")
        print(f"  可能原因：填充空间完全包含晶格")
        return None
    
    t_start = time.time()
    
    vertices, faces, _, _ = measure.marching_cubes(
        combined_sdf, level=0.0, spacing=spacing_tuple
    )
    vertices += target_bbox_min
    
    t_mc = time.time() - t_start
    print(f"  ✓ 完成 (耗时: {t_mc:.1f}s)")
    
    result_mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=True)
    
    # 9. 分析结果
    print(f"\n[结果分析]")
    
    if lattice_before_clip is not None:
        print(f"\n  裁剪前:")
        print(f"    顶点: {len(lattice_before_clip.vertices):,}")
        print(f"    面: {len(lattice_before_clip.faces):,}")
        
        # 警告：网格太大
        if len(lattice_before_clip.faces) > 50_000_000:
            print(f"    ⚠⚠⚠ 警告：网格非常大（{len(lattice_before_clip.faces)/1_000_000:.1f}M 面）！")
            print(f"    这会导致：")
            print(f"      1. 文件巨大（可能几 GB）")
            print(f"      2. 无法在大多数软件中打开")
            print(f"      3. 3D 打印切片软件可能崩溃")
            print(f"    建议：")
            print(f"      1. 降低分辨率（当前每晶胞 48³）")
            print(f"      2. 减少晶胞数量")
            print(f"      3. 使用后处理减面")
        
        # 跳过大网格的体积计算（避免内存溢出）
        if len(lattice_before_clip.faces) < 10_000_000:
            try:
                print(f"    体积: {lattice_before_clip.volume:.2f} mm³")
            except Exception as e:
                print(f"    体积: 无法计算（{e}）")
        else:
            print(f"    体积: 跳过（网格太大，避免内存溢出）")
    else:
        print(f"\n  裁剪前:")
        print(f"    ⚠ 网格太大，已跳过生成（避免内存溢出）")
    
    print(f"\n  裁剪后:")
    print(f"    顶点: {len(result_mesh.vertices):,}")
    print(f"    面: {len(result_mesh.faces):,}")
    
    # 同样跳过大网格的体积计算
    if len(result_mesh.faces) < 10_000_000:
        try:
            print(f"    体积: {result_mesh.volume:.2f} mm³")
            print(f"    表面积: {result_mesh.area:.2f} mm²")
            print(f"    水密性: {result_mesh.is_watertight}")
        except Exception as e:
            print(f"    体积: 无法计算（{e}）")
            print(f"    表面积: 无法计算")
            print(f"    水密性: 无法检查")
    else:
        print(f"    体积: 跳过（网格太大）")
        print(f"    表面积: 跳过（网格太大）")
        print(f"    水密性: 跳过（网格太大）")
    
    # 保留率（如果两个体积都能计算）
    if lattice_before_clip is not None and len(lattice_before_clip.faces) < 10_000_000 and len(result_mesh.faces) < 10_000_000:
        try:
            volume_ratio = result_mesh.volume / lattice_before_clip.volume * 100 if lattice_before_clip.volume > 0 else 0
            print(f"\n  保留率: {volume_ratio:.1f}%")
        except:
            pass
    
    # 连通性分析
    components = result_mesh.split(only_watertight=False)
    num_components = len(components)
    print(f"  连通分量数: {num_components}")
    
    if num_components == 1:
        print(f"  ✓✓✓ 完美！所有晶胞连接成一个整体")
    else:
        print(f"  ⚠ 有 {num_components} 个分量")
        
        # 跳过大分量的体积计算（避免内存溢出）
        try:
            # 只计算前几个分量的体积
            volumes = []
            for i, comp in enumerate(components[:10]):  # 最多计算前10个
                if len(comp.faces) < 5_000_000:  # 小于500万面才计算
                    volumes.append(comp.volume)
                else:
                    print(f"  分量 {i+1} 太大（{len(comp.faces):,} 面），跳过体积计算")
                    volumes.append(0)  # 占位
            
            volumes = sorted(volumes, reverse=True)
            
            if len(result_mesh.faces) < 10_000_000 and volumes[0] > 0:
                try:
                    total_volume = result_mesh.volume
                    print(f"  主分量体积: {volumes[0]:.2f} mm³ ({volumes[0]/total_volume*100:.1f}%)")
                except:
                    print(f"  主分量体积: {volumes[0]:.2f} mm³")
            else:
                print(f"  主分量体积: 无法计算（网格太大）")
        except Exception as e:
            print(f"  体积计算失败: {e}")
    
    return result_mesh, num_components, fill_space_mesh, lattice_before_clip


def post_process_simplify(mesh, target_faces, mesh_name="网格"):
    """
    后处理减面
    
    参数:
        mesh: 输入网格
        target_faces: 目标面数
        mesh_name: 网格名称（用于显示）
    
    返回:
        simplified_mesh: 简化后的网格
    """
    original_faces = len(mesh.faces)
    
    if original_faces <= target_faces:
        print(f"\n[后处理] {mesh_name}面数（{original_faces:,}）已经小于目标（{target_faces:,}），无需简化")
        return mesh
    
    print(f"\n" + "=" * 70)
    print(f"后处理减面 - {mesh_name}")
    print("=" * 70)
    print(f"  原始面数: {original_faces:,} ({original_faces/1_000_000:.1f}M)")
    print(f"  目标面数: {target_faces:,} ({target_faces/1_000_000:.1f}M)")
    print(f"  简化比例: {target_faces/original_faces*100:.1f}%")
    
    # 使用与前处理相同的简化函数
    simplified = simplify_mesh(mesh, target_faces)
    
    result_faces = len(simplified.faces)
    reduction = (1 - result_faces/original_faces) * 100
    
    print(f"\n  ✓ 后处理完成")
    print(f"  最终面数: {result_faces:,} ({result_faces/1_000_000:.1f}M)")
    print(f"  减少: {original_faces - result_faces:,} 面 ({reduction:.1f}%)")
    
    # 检查质量
    if result_faces < 10_000_000:
        try:
            print(f"  水密性: {simplified.is_watertight}")
            print(f"  体积: {simplified.volume:.2f} mm³")
        except:
            pass
    
    return simplified


def fill_cubic_region(
    unit_sdf,
    cell_size,
    unit_bbox_min,
    num_cells,
    unit_resolution,
    sdf_smooth_sigma=1.0,
    use_octree=False
):
    """
    在立方体区域内填充周期性晶格
    
    参数:
        unit_sdf: 单个晶胞的 SDF
        cell_size: 晶胞尺寸
        unit_bbox_min: 晶胞包围盒最小点
        num_cells: 每个方向的晶胞数量 [nx, ny, nz]
        unit_resolution: 单个晶胞的分辨率
        sdf_smooth_sigma: SDF 平滑参数
        use_octree: 是否使用八叉树存储（节省内存）
    """
    print("\n" + "=" * 70)
    print("立方体区域周期性填充")
    print("=" * 70)
    
    # 1. 计算目标区域
    target_size = num_cells * cell_size
    target_bbox_min = unit_bbox_min.copy()
    target_bbox_max = target_bbox_min + target_size
    
    print(f"\n[步骤 1] 目标区域设置")
    print(f"  晶胞数量: {num_cells} (总计 {np.prod(num_cells)} 个)")
    print(f"  晶胞尺寸: {cell_size}")
    print(f"  目标区域尺寸: {target_size}")
    print(f"  目标包围盒: {target_bbox_min} → {target_bbox_max}")
    
    # 2. 计算目标分辨率
    target_resolution = (num_cells * unit_resolution).astype(int)
    total_voxels = np.prod(target_resolution)
    
    print(f"\n[步骤 2] 分辨率设置")
    print(f"  单个晶胞: {unit_resolution}³")
    print(f"  目标分辨率: {target_resolution}")
    print(f"  总体素数: {total_voxels:,}")
    
    # 内存估算
    memory_mb = total_voxels * 4 / (1024 * 1024)
    print(f"  预计内存: {memory_mb:.1f} MB")
    
    if memory_mb > 2000:
        print(f"  ⚠ 警告：内存需求较大")
        response = input("  是否继续？(y/n): ")
        if response.lower() != 'y':
            return None
    
    spacing = target_size / target_resolution
    
    # 3. 生成周期性晶格 SDF
    print(f"\n[步骤 3] 生成周期性晶格 SDF...")
    
    lattice_sdf = generate_periodic_lattice_sdf(
        unit_sdf,
        target_resolution,
        target_bbox_min,
        spacing,
        unit_bbox_min,
        cell_size,
        unit_resolution,
        total_voxels,
        use_octree=use_octree
    )
    
    # 检查是否返回八叉树对象
    is_octree = isinstance(lattice_sdf, OctreeSDF) if OCTREE_AVAILABLE else False
    
    if is_octree:
        print(f"  ✓ 使用八叉树存储（节省内存）")
        print(f"  八叉树节点数: {lattice_sdf.num_nodes:,}")
        print(f"  内存节省: {lattice_sdf.memory_saved_ratio * 100:.1f}%")
        
        # 需要转换为密集网格用于后续处理
        print(f"  将八叉树转换为密集网格...")
        lattice_sdf = lattice_sdf.to_dense_grid(tuple(target_resolution))
        print(f"  ✓ 转换完成")
    
    print(f"  ✓ 晶格 SDF 范围: [{lattice_sdf.min():.3f}, {lattice_sdf.max():.3f}]")
    
    # 统计
    interior_voxels = np.sum(lattice_sdf < 0)
    interior_ratio = interior_voxels / total_voxels * 100
    print(f"  内部体素: {interior_voxels:,} ({interior_ratio:.1f}%)")
    
    # 4. 应用平滑
    if sdf_smooth_sigma > 0:
        print(f"\n[步骤 4] SDF 平滑 (sigma={sdf_smooth_sigma:.1f})...")
        t_start = time.time()
        lattice_sdf = gaussian_filter(lattice_sdf, sigma=sdf_smooth_sigma)
        t_smooth = time.time() - t_start
        print(f"  ✓ 完成 (耗时: {t_smooth:.1f}s)")
        print(f"  平滑后 SDF 范围: [{lattice_sdf.min():.3f}, {lattice_sdf.max():.3f}]")
    
    # 5. Marching Cubes 重建
    print(f"\n[步骤 5] Marching Cubes 重建...")
    t_start = time.time()
    
    vertices, faces, _, _ = measure.marching_cubes(
        lattice_sdf, level=0.0, spacing=spacing
    )
    vertices += target_bbox_min
    
    t_mc = time.time() - t_start
    print(f"  ✓ 完成 (耗时: {t_mc:.1f}s)")
    
    result_mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=True)
    
    # 6. 分析结果
    print(f"\n[结果分析]")
    print(f"  顶点: {len(result_mesh.vertices):,}")
    print(f"  面: {len(result_mesh.faces):,}")
    print(f"  体积: {result_mesh.volume:.2f} mm³")
    print(f"  表面积: {result_mesh.area:.2f} mm²")
    print(f"  水密性: {result_mesh.is_watertight}")
    
    # 连通性分析
    components = result_mesh.split(only_watertight=False)
    num_components = len(components)
    print(f"  连通分量数: {num_components}")
    
    if num_components == 1:
        print(f"  ✓✓✓ 完美！所有晶胞连接成一个整体")
    else:
        print(f"  ⚠ 有 {num_components} 个分量")
        
        # 跳过大分量的体积计算（避免内存溢出）
        try:
            volumes = []
            for i, comp in enumerate(components[:10]):  # 最多计算前10个
                if len(comp.faces) < 5_000_000:  # 小于500万面才计算
                    volumes.append(comp.volume)
                else:
                    print(f"  分量 {i+1} 太大（{len(comp.faces):,} 面），跳过体积计算")
                    volumes.append(0)
            
            volumes = sorted(volumes, reverse=True)
            
            if len(result_mesh.faces) < 10_000_000 and volumes[0] > 0:
                try:
                    total_volume = result_mesh.volume
                    print(f"  主分量体积: {volumes[0]:.2f} mm³ ({volumes[0]/total_volume*100:.1f}%)")
                except:
                    print(f"  主分量体积: {volumes[0]:.2f} mm³")
            else:
                print(f"  主分量体积: 无法计算（网格太大）")
            
            if num_components <= 10 and any(v > 0 for v in volumes):
                print(f"  各分量体积: {[f'{v:.2f}' if v > 0 else 'N/A' for v in volumes[:10]]}")
        except Exception as e:
            print(f"  体积计算失败: {e}")
    
    return result_mesh, num_components


def create_wireframe_box(bbox_min, bbox_max):
    """创建立方体线框用于可视化"""
    # 8个顶点
    vertices = np.array([
        [bbox_min[0], bbox_min[1], bbox_min[2]],
        [bbox_max[0], bbox_min[1], bbox_min[2]],
        [bbox_max[0], bbox_max[1], bbox_min[2]],
        [bbox_min[0], bbox_max[1], bbox_min[2]],
        [bbox_min[0], bbox_min[1], bbox_max[2]],
        [bbox_max[0], bbox_min[1], bbox_max[2]],
        [bbox_max[0], bbox_max[1], bbox_max[2]],
        [bbox_min[0], bbox_max[1], bbox_max[2]],
    ])
    
    # 12条边
    edges = np.array([
        [0, 1], [1, 2], [2, 3], [3, 0],  # 底面
        [4, 5], [5, 6], [6, 7], [7, 4],  # 顶面
        [0, 4], [1, 5], [2, 6], [3, 7],  # 竖边
    ])
    
    return vertices, edges


def visualize_result(lattice_mesh, unit_cell_mesh, fill_space_mesh=None, bbox_min=None, bbox_max=None, num_cells=None, lattice_before_clip=None):
    """可视化结果"""
    print("\n[可视化] 启动渲染器...")
    
    if fill_space_mesh is not None:
        # STL 模式：根据是否有裁剪前网格决定布局
        if lattice_before_clip is not None:
            # 4列布局（有裁剪前网格）
            plotter = pv.Plotter(shape=(1, 4), window_size=(2400, 600))
            
            # 第1列：单个晶胞
            plotter.subplot(0, 0)
            plotter.add_text("单个晶胞", font_size=10)
            plotter.add_mesh(pv.wrap(unit_cell_mesh), color='lightgray', opacity=0.5, show_edges=True)
            
            # 第2列：裁剪前的完整晶格
            plotter.subplot(0, 1)
            plotter.add_text("裁剪前\n(完整周期性晶格)", font_size=10)
            plotter.add_mesh(pv.wrap(lattice_before_clip), color='yellow', opacity=0.7, show_edges=True, line_width=0.3)
            plotter.add_mesh(pv.wrap(fill_space_mesh), color='lightblue', opacity=0.2, show_edges=True)
            
            # 第3列：填充空间
            plotter.subplot(0, 2)
            plotter.add_text("填充空间", font_size=10)
            plotter.add_mesh(pv.wrap(fill_space_mesh), color='lightblue', opacity=0.3, show_edges=True)
            
            # 第4列：裁剪后的结果
            plotter.subplot(0, 3)
            plotter.add_text("裁剪后\n(最终结果)", font_size=10)
            plotter.add_mesh(pv.wrap(fill_space_mesh), color='lightblue', opacity=0.15)
            plotter.add_mesh(pv.wrap(lattice_mesh), color='green', opacity=0.9, show_edges=True, line_width=0.5)
        else:
            # 3列布局（无裁剪前网格）
            plotter = pv.Plotter(shape=(1, 3), window_size=(1800, 600))
            
            # 第1列：单个晶胞
            plotter.subplot(0, 0)
            plotter.add_text("单个晶胞", font_size=10)
            plotter.add_mesh(pv.wrap(unit_cell_mesh), color='lightgray', opacity=0.5, show_edges=True)
            
            # 第2列：填充空间
            plotter.subplot(0, 1)
            plotter.add_text("填充空间", font_size=10)
            plotter.add_mesh(pv.wrap(fill_space_mesh), color='lightblue', opacity=0.3, show_edges=True)
            
            # 第3列：裁剪后的结果
            plotter.subplot(0, 2)
            plotter.add_text("裁剪后\n(最终结果)", font_size=10)
            plotter.add_mesh(pv.wrap(fill_space_mesh), color='lightblue', opacity=0.15)
            plotter.add_mesh(pv.wrap(lattice_mesh), color='green', opacity=0.9, show_edges=True, line_width=0.5)
    
    else:
        # 立方体模式：3列布局
        plotter = pv.Plotter(shape=(1, 3), window_size=(2000, 700))
        
        # 左侧：单个晶胞
        plotter.subplot(0, 0)
        plotter.add_text("单个晶胞", font_size=12)
        plotter.add_mesh(pv.wrap(unit_cell_mesh), color='lightgray', opacity=0.5, show_edges=True)
        
        # 中间：填充结果
        plotter.subplot(0, 1)
        plotter.add_text(f"填充结果 ({num_cells[0]}×{num_cells[1]}×{num_cells[2]} 晶胞)", font_size=12)
        plotter.add_mesh(pv.wrap(lattice_mesh), color='lightgreen', show_edges=True, line_width=0.5)
        
        # 右侧：结果 + 边界框
        plotter.subplot(0, 2)
        plotter.add_text("结果 + 边界", font_size=12)
        plotter.add_mesh(pv.wrap(lattice_mesh), color='green', opacity=0.8, show_edges=True, line_width=0.5)
        
        # 添加边界框线框
        box_vertices, box_edges = create_wireframe_box(bbox_min, bbox_max)
        for edge in box_edges:
            line = pv.Line(box_vertices[edge[0]], box_vertices[edge[1]])
            plotter.add_mesh(line, color='red', line_width=3)
    
    plotter.link_views()
    plotter.show()


def main():
    """主流程"""
    print("=" * 70)
    print("立方体/STL 空间周期性晶格填充测试")
    print("=" * 70)
    
    # 1. 选择晶胞
    print("\n请选择晶胞 STL 文件...")
    unit_cell = select_unit_cell()
    if unit_cell is None:
        print("✗ 未选择晶胞")
        return
    
    print(f"✓ 晶胞已加载: {len(unit_cell.vertices)} 顶点, {len(unit_cell.faces)} 面")
    
    # 诊断：检查晶胞复杂度
    if len(unit_cell.faces) > 50000:
        print(f"  ⚠ 警告：晶胞面数较多（{len(unit_cell.faces):,} 面）")
        print(f"  这会显著增加 SDF 计算时间")
        print(f"  建议：使用网格简化工具减少面数")
    
    # 2. 网格简化选项
    print("\n请选择网格简化选项...")
    simplify_options = select_simplification_options(unit_cell)
    if simplify_options is None:
        print("✗ 未设置简化选项")
        return
    
    if simplify_options['enabled']:
        unit_cell = simplify_mesh(unit_cell, simplify_options['target_faces'])
    else:
        print("✓ 跳过网格简化")
    
    # 3. 选择填充空间类型
    print("\n请选择填充空间类型...")
    space_type = select_fill_space()
    if space_type is None:
        print("✗ 未选择填充空间类型")
        return
    
    print(f"✓ 填充空间类型: {space_type}")
    
    # 4. 加载填充空间（如果是 STL）
    fill_space_mesh = None
    if space_type == 'stl':
        print("\n请选择填充空间 STL 文件...")
        fill_space_mesh = select_stl_file()
        if fill_space_mesh is None:
            print("✗ 未选择填充空间")
            return
        print(f"✓ 填充空间已加载: {len(fill_space_mesh.vertices)} 顶点")
        
        # 诊断：检查填充空间复杂度
        num_faces = len(fill_space_mesh.faces)
        print(f"  三角面数: {num_faces:,}")
        if num_faces > 10000:
            print(f"  ⚠ 警告：填充空间面数较多")
            print(f"  这会显著增加步骤4的计算时间")
            print(f"  建议：使用网格简化工具减少面数")
    
    # 5. 选择参数
    print("\n请设置填充参数...")
    params = select_parameters(space_type, unit_cell, fill_space_mesh)
    if params is None:
        print("✗ 未设置参数")
        return
    
    print(f"✓ 参数设置:")
    if space_type == 'cubic':
        print(f"  晶胞数量: {params['num_cells']}")
    print(f"  晶胞缩放: {params['scale_factor']:.3f}x")
    print(f"  单元分辨率: {params['unit_resolution']}³")
    print(f"  SDF 平滑: {'是' if params['sdf_smooth'] else '否'} (sigma={params['sdf_sigma']:.1f})")
    
    # 6. 计算晶胞 SDF
    unit_sdf, cell_size, unit_bbox_min, unit_bbox_max, scaled_unit_cell = compute_unit_cell_sdf(
        unit_cell, params['unit_resolution'], 
        shrink_factor=params['shrink_factor'], 
        scale_factor=params['scale_factor']
    )
    
    # 7. 执行填充
    if space_type == 'cubic':
        # 立方体填充
        result = fill_cubic_region(
            unit_sdf,
            cell_size,
            unit_bbox_min,
            params['num_cells'],
            params['unit_resolution'],
            sdf_smooth_sigma=params['sdf_sigma'] if params['sdf_smooth'] else 0.0,
            use_octree=params.get('use_octree', False)
        )
        
        if result is None:
            return
        
        lattice_mesh, num_components = result
        
        # 保存结果
        output_filename = f"cubic_filling_{params['num_cells'][0]}x{params['num_cells'][1]}x{params['num_cells'][2]}_res{params['unit_resolution']}.stl"
        output_path = project_root / "tests" / output_filename
        lattice_mesh.export(str(output_path))
        print(f"\n✓ 结果已保存: {output_path}")
        
        # 可视化
        target_bbox_min = unit_bbox_min
        target_bbox_max = unit_bbox_min + params['num_cells'] * cell_size
        
        visualize_result(lattice_mesh, scaled_unit_cell, None, target_bbox_min, target_bbox_max, params['num_cells'])
    
    else:
        # STL 填充
        result = fill_stl_space_with_lattice(
            unit_sdf,
            cell_size,
            unit_bbox_min,
            fill_space_mesh,
            params['unit_resolution'],
            sdf_smooth_sigma=params['sdf_sigma'] if params['sdf_smooth'] else 0.0,
            scaled_unit_cell=scaled_unit_cell,
            num_cells_hint=None,  # 将在函数内部计算
            use_dual_contouring=params.get('use_dual_contouring', False),
            use_octree=params.get('use_octree', False)
        )
        
        if result is None:
            return
        
        lattice_mesh, num_components, fill_space, lattice_before_clip = result
        
        # 后处理减面（如果启用）
        print("\n请选择后处理减面选项...")
        post_simplify_options = select_post_simplification_options()
        if post_simplify_options is None:
            print("✗ 未设置后处理选项")
            return
        
        lattice_mesh_simplified = None
        lattice_before_clip_simplified = None
        
        if post_simplify_options['enabled']:
            target_faces = post_simplify_options['target_faces']
            
            # 简化裁剪后的网格
            lattice_mesh_simplified = post_process_simplify(
                lattice_mesh, 
                target_faces, 
                "裁剪后网格"
            )
            
            # 简化裁剪前的网格（如果存在且不是太大）
            if lattice_before_clip is not None and len(lattice_before_clip.faces) < 100_000_000:
                lattice_before_clip_simplified = post_process_simplify(
                    lattice_before_clip, 
                    target_faces, 
                    "裁剪前网格"
                )
            else:
                if lattice_before_clip is None:
                    print(f"\n⚠ 裁剪前网格未生成，跳过简化")
                else:
                    print(f"\n⚠ 裁剪前网格太大（{len(lattice_before_clip.faces)/1_000_000:.1f}M 面），跳过简化")
        else:
            print("✓ 跳过后处理减面")
        
        # 保存结果（裁剪后的网格）
        output_filename = f"stl_filling_res{params['unit_resolution']}.stl"
        output_path = project_root / "tests" / output_filename
        
        # 决定保存哪个版本
        mesh_to_save = lattice_mesh_simplified if lattice_mesh_simplified is not None else lattice_mesh
        
        # 检查网格大小，避免导出时内存溢出
        if len(mesh_to_save.faces) < 50_000_000:
            try:
                mesh_to_save.export(str(output_path))
                print(f"\n✓ 结果已保存: {output_path}")
                if lattice_mesh_simplified is not None:
                    print(f"  （已简化版本：{len(mesh_to_save.faces):,} 面）")
            except Exception as e:
                print(f"\n⚠ 结果保存失败: {e}")
                print(f"  网格可能太大（{len(mesh_to_save.faces):,} 面）")
        else:
            print(f"\n⚠ 结果网格太大（{len(mesh_to_save.faces):,} 面），跳过保存")
            print(f"  （导出会导致内存溢出）")
            print(f"  建议：")
            print(f"    1. 启用后处理减面并设置更小的目标面数")
            print(f"    2. 降低分辨率（当前每晶胞 {params['unit_resolution']}³）")
            print(f"    3. 减少晶胞数量")
        
        # 保存裁剪前的晶格（用于对比）
        output_filename_before = f"stl_filling_before_clip_res{params['unit_resolution']}.stl"
        output_path_before = project_root / "tests" / output_filename_before
        
        # 决定保存哪个版本
        if lattice_before_clip is not None:
            mesh_before_to_save = lattice_before_clip_simplified if lattice_before_clip_simplified is not None else lattice_before_clip
            
            # 跳过超大网格的导出（避免内存溢出）
            if len(mesh_before_to_save.faces) < 50_000_000:
                try:
                    mesh_before_to_save.export(str(output_path_before))
                    print(f"✓ 裁剪前晶格已保存: {output_path_before}")
                    if lattice_before_clip_simplified is not None:
                        print(f"  （已简化版本：{len(mesh_before_to_save.faces):,} 面）")
                except Exception as e:
                    print(f"⚠ 裁剪前晶格保存失败: {e}")
            else:
                print(f"⚠ 裁剪前晶格太大（{len(mesh_before_to_save.faces):,} 面），跳过保存")
                print(f"  （导出会导致内存溢出）")
        else:
            print(f"⚠ 裁剪前晶格未生成，跳过保存")
        
        # 可视化（使用简化后的版本，如果有的话）
        mesh_to_visualize = lattice_mesh_simplified if lattice_mesh_simplified is not None else lattice_mesh
        
        if lattice_before_clip is not None:
            mesh_before_to_visualize = lattice_before_clip_simplified if lattice_before_clip_simplified is not None else lattice_before_clip
        else:
            mesh_before_to_visualize = None
        
        visualize_result(mesh_to_visualize, scaled_unit_cell, fill_space, lattice_before_clip=mesh_before_to_visualize)
    
    # 8. 总结
    print("\n" + "=" * 70)
    print("总结")
    print("=" * 70)
    
    if num_components == 1:
        print("\n✓✓✓ 成功！晶格完美连接")
        if space_type == 'stl':
            print("\n✓ STL 空间填充成功")
            print("  • 晶格自动适应曲面边界")
            print("  • 边界处自然裁剪")
    else:
        print(f"\n⚠ 有 {num_components} 个分量，需要优化")
        print("\n建议:")
        print("  1. 提高分辨率")
        print("  2. 增加 SDF 平滑 (sigma)")
        print("  3. 检查晶胞本身是否支持周期性")


if __name__ == "__main__":
    main()
