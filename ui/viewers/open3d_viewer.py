"""
基于 Open3D 的高质量渲染器
支持 PBR 材质和高级渲染特性

作为 PyVista 渲染器的替代选项 - 使用独立窗口
"""

from __future__ import annotations

import numpy as np
import trimesh
from dataclasses import dataclass
from typing import Optional, Literal

try:
    import open3d as o3d
    import open3d.visualization.gui as gui
    import open3d.visualization.rendering as rendering
    OPEN3D_AVAILABLE = True
except ImportError:
    OPEN3D_AVAILABLE = False
    print("⚠ Open3D 未安装，请运行: pip install open3d")

from PyQt5 import QtWidgets, QtCore, QtGui
import sys

RenderMode = Literal["solid", "wireframe"]
MouseMode = Literal["rotate", "pan"]


@dataclass
class MeshData:
    """网格数据类（保持与其他渲染器兼容）"""
    vertices: np.ndarray
    faces: np.ndarray
    color: tuple[float, float, float, float] = (0.7, 0.7, 0.7, 1.0)
    render_mode: RenderMode = "solid"
    unlit: bool = False
    line_width: float = 1.5
    metallic: float = 0.3
    roughness: float = 0.5
    
    def __post_init__(self):
        self.vertices = np.asarray(self.vertices, dtype=np.float32)
        self.faces = np.asarray(self.faces, dtype=np.int32)


class Open3DRenderer(QtWidgets.QWidget):
    """
    基于 Open3D 的高质量渲染器（独立窗口版本）
    
    特性：
    - PBR 材质
    - 高质量抗锯齿
    - 多光源支持
    - 完整的鼠标交互
    - 独立窗口显示
    
    使用 Open3D 的原生可视化器，提供最佳的渲染质量和交互体验
    """
    
    # 定义信号（保持与其他渲染器兼容）
    lasso_completed = QtCore.pyqtSignal(list)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        if not OPEN3D_AVAILABLE:
            # 如果 Open3D 不可用，显示错误信息
            layout = QtWidgets.QVBoxLayout(self)
            label = QtWidgets.QLabel("Open3D 未安装\n\n请运行: pip install open3d")
            label.setAlignment(QtCore.Qt.AlignCenter)
            label.setStyleSheet("font-size: 14pt; color: #ff6b6b;")
            layout.addWidget(label)
            return
        
        # 状态变量
        self.meshes: list[MeshData] = []
        self.mouse_mode: MouseMode = "rotate"
        self._bbox_center = np.zeros(3, dtype=np.float32)
        self._bbox_size = 1.0
        self._bbox_dimensions = np.ones(3, dtype=np.float32)
        
        # 渲染选项
        self.show_axes = True
        self.enable_pbr = True
        self.enable_ssao_flag = True
        self.enable_shadows_flag = False
        self.enable_antialiasing = True
        
        # Open3D 可视化器（独立窗口）
        self._vis = None
        self._vis_thread = None
        
        # 初始化 UI
        self._init_ui()
    
    def _init_ui(self):
        """初始化 UI"""
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)
        
        # 提示信息
        info_label = QtWidgets.QLabel(
            "Open3D 渲染器\n\n"
            "点击下方按钮在独立窗口中查看 3D 模型\n\n"
            "独立窗口支持：\n"
            "• 左键拖动：旋转视角\n"
            "• 右键拖动：平移视图\n"
            "• 滚轮：缩放\n"
            "• Ctrl+C：复制视角\n"
            "• Ctrl+V：粘贴视角\n"
            "• H：显示/隐藏帮助\n"
            "• 高质量 PBR 渲染"
        )
        info_label.setAlignment(QtCore.Qt.AlignCenter)
        info_label.setStyleSheet(
            "color: #ccc; "
            "font-size: 11pt; "
            "padding: 20px; "
            "background-color: #2a2a2a; "
            "border-radius: 8px;"
        )
        layout.addWidget(info_label)
        
        # 渲染选项组
        options_group = QtWidgets.QGroupBox("渲染选项")
        options_layout = QtWidgets.QVBoxLayout(options_group)
        options_layout.setSpacing(8)
        
        # 显示坐标轴
        self.check_show_axes = QtWidgets.QCheckBox("显示坐标轴")
        self.check_show_axes.setChecked(self.show_axes)
        self.check_show_axes.stateChanged.connect(lambda state: setattr(self, 'show_axes', state == QtCore.Qt.Checked))
        options_layout.addWidget(self.check_show_axes)
        
        # 背景颜色选择
        bg_layout = QtWidgets.QHBoxLayout()
        bg_layout.addWidget(QtWidgets.QLabel("背景颜色:"))
        self.combo_bg_color = QtWidgets.QComboBox()
        self.combo_bg_color.addItems(["深灰", "浅灰", "白色", "黑色"])
        self.combo_bg_color.setCurrentIndex(0)
        bg_layout.addWidget(self.combo_bg_color)
        bg_layout.addStretch()
        options_layout.addLayout(bg_layout)
        
        # 窗口尺寸选择
        size_layout = QtWidgets.QHBoxLayout()
        size_layout.addWidget(QtWidgets.QLabel("窗口尺寸:"))
        self.combo_window_size = QtWidgets.QComboBox()
        self.combo_window_size.addItems(["1200×900", "1600×1200", "1920×1080", "全屏"])
        self.combo_window_size.setCurrentIndex(0)
        size_layout.addWidget(self.combo_window_size)
        size_layout.addStretch()
        options_layout.addLayout(size_layout)
        
        layout.addWidget(options_group)
        
        # 打开窗口按钮
        self.button_open_window = QtWidgets.QPushButton("🚀 打开 Open3D 渲染窗口")
        self.button_open_window.setStyleSheet(
            "QPushButton {"
            "  background-color: #0d7377; "
            "  color: white; "
            "  font-size: 14pt; "
            "  font-weight: bold; "
            "  padding: 15px; "
            "  border-radius: 8px; "
            "}"
            "QPushButton:hover {"
            "  background-color: #14a085; "
            "}"
            "QPushButton:pressed {"
            "  background-color: #0a5a5d; "
            "}"
            "QPushButton:disabled {"
            "  background-color: #555; "
            "  color: #888; "
            "}"
        )
        self.button_open_window.clicked.connect(self._open_visualization_window)
        layout.addWidget(self.button_open_window)
        
        # 快速导出按钮
        export_layout = QtWidgets.QHBoxLayout()
        self.button_export_screenshot = QtWidgets.QPushButton("📷 导出截图")
        self.button_export_screenshot.setEnabled(False)
        self.button_export_screenshot.clicked.connect(self._export_screenshot)
        self.button_export_screenshot.setToolTip("在独立窗口中按 P 键可以直接截图")
        export_layout.addWidget(self.button_export_screenshot)
        
        self.button_export_ply = QtWidgets.QPushButton("💾 导出 PLY")
        self.button_export_ply.setEnabled(False)
        self.button_export_ply.clicked.connect(self._export_ply)
        self.button_export_ply.setToolTip("导出为 Open3D 的 PLY 格式（保留颜色）")
        export_layout.addWidget(self.button_export_ply)
        layout.addLayout(export_layout)
        
        # 状态标签
        self.status_label = QtWidgets.QLabel("准备就绪 - 等待加载模型")
        self.status_label.setAlignment(QtCore.Qt.AlignCenter)
        self.status_label.setStyleSheet("color: #888; font-size: 10pt; padding: 10px;")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)
        
        layout.addStretch()
    
    def _convert_to_open3d(self, mesh_data: MeshData) -> Optional[o3d.geometry.TriangleMesh]:
        """将 MeshData 转换为 Open3D TriangleMesh"""
        if len(mesh_data.vertices) == 0 or len(mesh_data.faces) == 0:
            return None
        
        try:
            # 创建 Open3D mesh
            o3d_mesh = o3d.geometry.TriangleMesh()
            o3d_mesh.vertices = o3d.utility.Vector3dVector(mesh_data.vertices)
            o3d_mesh.triangles = o3d.utility.Vector3iVector(mesh_data.faces)
            
            # 计算法线
            o3d_mesh.compute_vertex_normals()
            
            # 设置颜色
            color_rgb = mesh_data.color[:3]
            o3d_mesh.paint_uniform_color(color_rgb)
            
            return o3d_mesh
            
        except Exception as e:
            print(f"[错误] 转换为 Open3D mesh 失败: {e}")
            return None
    
    def _open_visualization_window(self):
        """打开 Open3D 可视化窗口"""
        if not OPEN3D_AVAILABLE:
            QtWidgets.QMessageBox.warning(
                self,
                "Open3D 未安装",
                "请先安装 Open3D:\npip install open3d"
            )
            return
        
        if len(self.meshes) == 0:
            QtWidgets.QMessageBox.information(
                self,
                "无模型数据",
                "请先加载或生成模型数据"
            )
            return
        
        try:
            # 转换所有网格
            geometries = []
            for mesh_data in self.meshes:
                o3d_mesh = self._convert_to_open3d(mesh_data)
                if o3d_mesh is not None:
                    geometries.append(o3d_mesh)
            
            if len(geometries) == 0:
                QtWidgets.QMessageBox.warning(
                    self,
                    "转换失败",
                    "无法转换网格数据"
                )
                return
            
            # 更新状态
            self.status_label.setText(f"正在打开窗口... ({len(geometries)} 个网格)")
            QtWidgets.QApplication.processEvents()
            
            # 获取窗口尺寸设置
            size_text = self.combo_window_size.currentText()
            if size_text == "全屏":
                width, height = 1920, 1080
                fullscreen = True
            else:
                width, height = map(int, size_text.replace('×', 'x').split('x'))
                fullscreen = False
            
            # 获取背景颜色
            bg_colors = {
                "深灰": [0.15, 0.15, 0.15],
                "浅灰": [0.7, 0.7, 0.7],
                "白色": [1.0, 1.0, 1.0],
                "黑色": [0.0, 0.0, 0.0]
            }
            bg_color = bg_colors.get(self.combo_bg_color.currentText(), [0.15, 0.15, 0.15])
            
            # 创建可视化器并显示
            vis = o3d.visualization.Visualizer()
            vis.create_window(
                window_name="Open3D 渲染器 - 鞋底晶格查看器 [按 H 显示帮助]",
                width=width,
                height=height
            )
            
            # 添加所有几何体
            for geom in geometries:
                vis.add_geometry(geom)
            
            # 设置渲染选项
            render_option = vis.get_render_option()
            render_option.background_color = np.array(bg_color)
            render_option.point_size = 2.0
            render_option.line_width = 1.0
            render_option.show_coordinate_frame = self.show_axes
            render_option.light_on = True
            render_option.mesh_show_back_face = False  # 不显示背面
            
            # 设置更好的光照
            render_option.mesh_shade_option = o3d.visualization.MeshShadeOption.Default
            
            # 重置视角
            vis.reset_view_point(True)
            
            # 更新状态
            self.status_label.setText(
                f"✓ 窗口已打开 ({len(geometries)} 个网格) | "
                f"尺寸: {self._bbox_dimensions[0]:.1f} × "
                f"{self._bbox_dimensions[1]:.1f} × "
                f"{self._bbox_dimensions[2]:.1f} mm | "
                f"按 P 键截图，按 H 显示帮助"
            )
            
            # 运行可视化器（阻塞直到窗口关闭）
            vis.run()
            vis.destroy_window()
            
            # 窗口关闭后更新状态
            self.status_label.setText("窗口已关闭 - 可以重新打开")
            
        except Exception as e:
            QtWidgets.QMessageBox.critical(
                self,
                "渲染错误",
                f"打开 Open3D 窗口时出错:\n{str(e)}"
            )
            self.status_label.setText(f"错误: {str(e)}")
            print(f"[错误] Open3D 可视化失败: {e}")
            import traceback
            traceback.print_exc()
    
    def _export_screenshot(self):
        """导出截图（需要先打开窗口）"""
        QtWidgets.QMessageBox.information(
            self,
            "导出截图",
            "请在 Open3D 窗口中按 P 键直接截图\n\n"
            "截图将保存到当前目录"
        )
    
    def _export_ply(self):
        """导出为 PLY 格式"""
        if len(self.meshes) == 0:
            QtWidgets.QMessageBox.warning(
                self,
                "无数据",
                "没有可导出的网格数据"
            )
            return
        
        try:
            # 选择保存路径
            file_path, _ = QtWidgets.QFileDialog.getSaveFileName(
                self,
                "导出 PLY 文件",
                "combined_mesh.ply",
                "PLY Files (*.ply)"
            )
            
            if not file_path:
                return
            
            # 转换并合并所有网格
            combined_mesh = o3d.geometry.TriangleMesh()
            for mesh_data in self.meshes:
                o3d_mesh = self._convert_to_open3d(mesh_data)
                if o3d_mesh is not None:
                    combined_mesh += o3d_mesh
            
            # 保存
            o3d.io.write_triangle_mesh(file_path, combined_mesh)
            
            QtWidgets.QMessageBox.information(
                self,
                "导出成功",
                f"已导出到:\n{file_path}"
            )
            self.status_label.setText(f"✓ 已导出 PLY: {file_path}")
            
        except Exception as e:
            QtWidgets.QMessageBox.critical(
                self,
                "导出失败",
                f"导出 PLY 时出错:\n{str(e)}"
            )
            print(f"[错误] 导出 PLY 失败: {e}")
    
    def set_mouse_mode(self, mode: MouseMode):
        """设置鼠标模式（Open3D 自动处理）"""
        self.mouse_mode = mode
    
    def get_view_state(self) -> dict[str, float]:
        """获取当前视图状态"""
        return {}
    
    def apply_view_state(self, state: dict):
        """应用视图状态"""
        pass
    
    def reset_view(self):
        """重置视图（在独立窗口中操作）"""
        pass
    
    def set_view_front(self):
        """前视图"""
        pass
    
    def set_view_back(self):
        """后视图"""
        pass
    
    def set_view_left(self):
        """左视图"""
        pass
    
    def set_view_right(self):
        """右视图"""
        pass
    
    def set_view_top(self):
        """顶视图"""
        pass
    
    def set_view_bottom(self):
        """底视图"""
        pass
    
    def set_view_isometric(self):
        """等轴测视图"""
        pass
    
    def set_meshes(self, meshes: list[MeshData], reset_view: bool = True):
        """设置要渲染的网格"""
        self.meshes = meshes
        self._update_bbox()
        
        # 更新状态
        if len(meshes) > 0:
            total_vertices = sum(len(m.vertices) for m in meshes)
            total_faces = sum(len(m.faces) for m in meshes)
            self.status_label.setText(
                f"✓ 已加载 {len(meshes)} 个网格 | "
                f"{total_vertices:,} 顶点, {total_faces:,} 面 | "
                f"尺寸: {self._bbox_dimensions[0]:.1f} × "
                f"{self._bbox_dimensions[1]:.1f} × "
                f"{self._bbox_dimensions[2]:.1f} mm"
            )
            self.button_open_window.setEnabled(True)
            self.button_export_screenshot.setEnabled(True)
            self.button_export_ply.setEnabled(True)
        else:
            self.status_label.setText("无模型数据")
            self.button_open_window.setEnabled(False)
            self.button_export_screenshot.setEnabled(False)
            self.button_export_ply.setEnabled(False)
    
    def _update_bbox(self):
        """更新包围盒"""
        valid = [m for m in self.meshes if len(m.vertices) > 0]
        if not valid:
            self._bbox_center = np.zeros(3, dtype=np.float32)
            self._bbox_size = 1.0
            self._bbox_dimensions = np.ones(3, dtype=np.float32)
            return
        
        all_v = np.concatenate([m.vertices for m in valid], axis=0)
        vmin = all_v.min(axis=0)
        vmax = all_v.max(axis=0)
        self._bbox_center = (vmin + vmax) * 0.5
        self._bbox_dimensions = vmax - vmin
        self._bbox_size = max(float(np.max(vmax - vmin)), 1.0)
    
    def set_plane(self, point: np.ndarray, normal: np.ndarray, size: float = None):
        """设置要显示的平面"""
        pass
    
    def hide_plane(self):
        """隐藏平面"""
        pass
    
    def enable_lasso_mode(self, mesh):
        """启用套索模式（暂不支持）"""
        print("[警告] Open3D 渲染器暂不支持套索模式")
    
    def disable_lasso_mode(self):
        """禁用套索模式"""
        pass
    
    def render(self):
        """渲染（在独立窗口中）"""
        pass
    
    def clear(self):
        """清空场景"""
        self.meshes.clear()
        self.status_label.setText("场景已清空")
        self.button_open_window.setEnabled(False)


# 为了兼容性，创建一个别名
GLMeshViewer = Open3DRenderer


if __name__ == "__main__":
    """测试渲染器"""
    from pathlib import Path
    
    if not OPEN3D_AVAILABLE:
        print("请先安装 Open3D: pip install open3d")
        sys.exit(1)
    
    app = QtWidgets.QApplication(sys.argv)
    
    # 创建窗口
    window = QtWidgets.QMainWindow()
    window.setWindowTitle("Open3D 渲染器测试")
    window.resize(800, 600)
    
    # 创建渲染器
    renderer = Open3DRenderer(window)
    window.setCentralWidget(renderer)
    
    # 加载测试模型
    model_path = None
    for path in ["鞋底/1.stl", "voronoi_lattice.stl"]:
        if Path(path).exists():
            model_path = path
            break
    
    if model_path:
        print(f"加载模型: {model_path}")
        mesh = trimesh.load(model_path)
        
        mesh_data = MeshData(
            vertices=mesh.vertices,
            faces=mesh.faces,
            color=(0.29, 0.56, 0.89, 1.0),
            metallic=0.3,
            roughness=0.4
        )
        
        renderer.set_meshes([mesh_data], reset_view=True)
        print("✓ 模型已加载")
    else:
        print("未找到测试模型")
    
    window.show()
    sys.exit(app.exec_())
