from __future__ import annotations

"""带参数面板的 Qt 鞋底晶格工作台。"""

from pathlib import Path
import shutil
from typing import Optional

import numpy as np
import trimesh
from PyQt5 import QtWidgets, QtGui, QtCore

import core.geometry as geometry
from config import (
    SOLE_STL_PATH,
    LATTICE_STL_PATH,
    OUTPUT_STL_PATH,
    RESOURCES_DIR,
    RESULTS_DIR,
    LATTICE_METHOD,
    GYROID_CELL_SIZE,
    GYROID_ISOVALUE,
    GYROID_RESOLUTION,
    GYROID_MASK_CACHE_DIR,
    GYROID_RESULT_CACHE_DIR,
    TILE_RESULT_CACHE_DIR,
    VORONOI_CELL_SIZE,
    VORONOI_STRUT_THICKNESS,
    VORONOI_Z_LAYERS,
    VORONOI_IMPLICIT_CELL_SIZE,
    VORONOI_IMPLICIT_WALL_THICKNESS,
    VORONOI_IMPLICIT_RESOLUTION,
    VORONOI_IMPLICIT_SEED_DENSITY_FACTOR,
    LATTICE_TILE_SHRINK,
    TILE_SPACING_FACTOR,
    TILE_BOUNDARY_MARGIN,
    LATTICE_UNIT_TARGET_FACES,
)
from ui.viewers.pyvista_viewer import PyVistaRenderer as GLMeshViewer, MeshData
from ui.viewers.open3d_viewer import Open3DRenderer, OPEN3D_AVAILABLE
from domain.slicer.mesh_slicer import MeshSlicer, is_cpp_available
from domain.region.region_manager import Region, RegionManager
from domain.region.plane_region_manager import PlaneRegion, PlaneRegionManager, clip_lattice_by_planes
from domain.project.project_manager import ProjectManager, Project


class LatticeWorkbenchWindow(QtWidgets.QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("鞋底晶格生成系统")
        self.resize(1400, 860)
        self.setMinimumSize(800, 600)  # 增加最小尺寸，防止缩放太小崩溃
        
        # 不使用无边框窗口，使用系统标准窗口（更稳定）
        # 移除 FramelessWindowHint，恢复标准标题栏
        
        # 设置深色标题栏（Windows 10/11）
        self._set_dark_title_bar()

        # 初始化渲染器（默认使用 PyVista）
        self.current_renderer_type = "pyvista"
        self.viewer = GLMeshViewer(self)
        self.current_view_mode = "combined"
        self.current_sole_mesh = trimesh.Trimesh()
        self.current_lattice_mesh = trimesh.Trimesh()
        self.current_combined_mesh = trimesh.Trimesh()
        self.selected_lattice_unit_mesh = trimesh.Trimesh()
        
        # 存储生成的最终结果（与预览分离）
        self.generated_sole_mesh = trimesh.Trimesh()
        self.generated_lattice_mesh = trimesh.Trimesh()
        self.generated_combined_mesh = trimesh.Trimesh()
        self._suspend_auto_recommend = False
        self._loading_project = False  # 标记是否正在加载工程
        self._preview_view_states: dict[str, dict[str, float]] = {}
        self._current_preview_key: str | None = None
        self._mouse_mode = "rotate"
        
        # 工程管理器
        self.project_manager = ProjectManager()
        self.current_project_modified = False
        
        # 分割相关状态
        self.original_sole_mesh = trimesh.Trimesh()  # 保存原始鞋底
        self.slicer = None
        self.slicing_enabled = False
        self.positive_mesh = None
        self.negative_mesh = None
        
        # 区域晶格相关状态
        self.region_manager = RegionManager()
        self.selected_region_id = None
        self.combined_region_lattice = None
        
        # 平面区域晶格相关状态（方案B）
        self.plane_region_manager = PlaneRegionManager()
        self.selected_plane_region_id = None
        self.combined_plane_region_lattice = None
        
        # 减面配置
        self.lattice_unit_target_faces = LATTICE_UNIT_TARGET_FACES
        
        # 材质颜色设置
        self.sole_material_colors = {
            0: (0.75, 0.75, 0.75, 1.0),  # 灰色
            1: (0.95, 0.95, 0.95, 1.0),  # 白色
            2: (0.25, 0.25, 0.25, 1.0),  # 黑色
            3: (0.3, 0.5, 0.8, 1.0),     # 蓝色
            4: (1.0, 0.6, 0.2, 1.0),     # 橙色
        }
        self.lattice_material_colors = {
            0: (1.0, 0.6, 0.2, 1.0),     # 橙色
            1: (1.0, 0.9, 0.2, 1.0),     # 黄色
            2: (0.3, 0.8, 0.3, 1.0),     # 绿色
            3: (0.9, 0.2, 0.2, 1.0),     # 红色
            4: (0.2, 0.5, 0.9, 1.0),     # 蓝色
        }
        self.current_sole_material = 0
        self.current_lattice_material = 0
        self._current_bg_color_index = 0  # 默认深灰

        self._build_ui()
        self._create_actions()
        self._create_menu()
        self._sync_method_fields()
        
        # 连接参数变化信号，标记工程已修改
        self._connect_parameter_signals()
        
        # 不恢复设置，使用默认值
        self._apply_recommended_params(show_message=False)
        self._preview_selected_sole(log_message=False, reset_view_mode=True)
        self._preview_selected_lattice(log_message=False, reset_view_mode=False, update_main_view=False, inset_visible=True)
        
        # 创建新工程
        self.project_manager.create_new_project()
        self.current_project_modified = False
        self._update_window_title()
        
        # 设置全局样式（包括所有子窗口）
        self._set_global_dark_style()
        
        # 延迟刷新，确保界面完全初始化后再渲染
        QtCore.QTimer.singleShot(100, self._initial_render)
        
        # 延迟设置深色标题栏（需要在窗口显示后）
        QtCore.QTimer.singleShot(200, self._set_dark_title_bar)
    
    def _initial_render(self) -> None:
        """初始渲染"""
        if hasattr(self, 'viewer'):
            try:
                self.viewer.render()
            except:
                pass
    
    def _set_dark_title_bar(self) -> None:
        """设置深色标题栏（Windows 10/11）"""
        try:
            # Windows 10/11 深色标题栏
            from ctypes import windll, c_int, byref, sizeof
            
            # 获取窗口句柄
            hwnd = int(self.winId())
            
            # DWMWA_USE_IMMERSIVE_DARK_MODE = 20 (Windows 11)
            # DWMWA_USE_IMMERSIVE_DARK_MODE = 19 (Windows 10 build 19041+)
            DWMWA_USE_IMMERSIVE_DARK_MODE = 20
            value = c_int(1)  # 1 = 深色模式
            
            try:
                windll.dwmapi.DwmSetWindowAttribute(
                    hwnd,
                    DWMWA_USE_IMMERSIVE_DARK_MODE,
                    byref(value),
                    sizeof(value)
                )
            except:
                # 尝试 Windows 10 的值
                DWMWA_USE_IMMERSIVE_DARK_MODE = 19
                windll.dwmapi.DwmSetWindowAttribute(
                    hwnd,
                    DWMWA_USE_IMMERSIVE_DARK_MODE,
                    byref(value),
                    sizeof(value)
                )
        except Exception as e:
            # 如果失败（非Windows或不支持），忽略
            pass
    
    def _set_global_dark_style(self) -> None:
        """设置全局深色样式（包括所有子窗口）"""
        app = QtWidgets.QApplication.instance()
        if app:
            app.setStyleSheet("""
                /* QMessageBox 样式 */
                QMessageBox {
                    background-color: #2c2c2c;
                    color: #ffffff;
                }
                QMessageBox QLabel {
                    color: #ffffff;
                }
                QMessageBox QPushButton {
                    background-color: #3a3a3a;
                    border: 1px solid #4a4a4a;
                    border-radius: 4px;
                    padding: 8px 20px;
                    color: #ffffff;
                    min-width: 80px;
                }
                QMessageBox QPushButton:hover {
                    background-color: #4a4a4a;
                    border-color: #2196F3;
                }
                QMessageBox QPushButton:pressed {
                    background-color: #2a2a2a;
                }
                
                /* QFileDialog 样式 */
                QFileDialog {
                    background-color: #2c2c2c;
                    color: #ffffff;
                }
                QFileDialog QLabel {
                    color: #ffffff;
                }
                QFileDialog QPushButton {
                    background-color: #3a3a3a;
                    border: 1px solid #4a4a4a;
                    border-radius: 4px;
                    padding: 6px 16px;
                    color: #ffffff;
                }
                QFileDialog QPushButton:hover {
                    background-color: #4a4a4a;
                }
                QFileDialog QTreeView, QFileDialog QListView {
                    background-color: #3a3a3a;
                    border: 1px solid #4a4a4a;
                    color: #ffffff;
                }
                QFileDialog QTreeView::item:selected, QFileDialog QListView::item:selected {
                    background-color: #2196F3;
                }
                QFileDialog QLineEdit, QFileDialog QComboBox {
                    background-color: #3a3a3a;
                    border: 1px solid #4a4a4a;
                    border-radius: 4px;
                    padding: 5px;
                    color: #ffffff;
                }
                
                /* QInputDialog 样式 */
                QInputDialog {
                    background-color: #2c2c2c;
                    color: #ffffff;
                }
                QInputDialog QLabel {
                    color: #ffffff;
                }
                QInputDialog QLineEdit {
                    background-color: #3a3a3a;
                    border: 1px solid #4a4a4a;
                    border-radius: 4px;
                    padding: 5px;
                    color: #ffffff;
                }
                QInputDialog QPushButton {
                    background-color: #3a3a3a;
                    border: 1px solid #4a4a4a;
                    border-radius: 4px;
                    padding: 8px 20px;
                    color: #ffffff;
                }
                QInputDialog QPushButton:hover {
                    background-color: #4a4a4a;
                }
            """)

    def _build_ui(self) -> None:
        root = QtWidgets.QWidget(self)
        self.setCentralWidget(root)
        # 使用垂直布局作为主布局
        main_layout = QtWidgets.QVBoxLayout(root)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # 创建工具栏
        self._create_toolbar()
        main_layout.addWidget(self.toolbar)
        
        # 主内容区域：使用 QSplitter 实现可调整大小的侧边栏
        self.content_splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        self.content_splitter.setHandleWidth(4)  # 增加拖动条宽度，更容易识别
        self.content_splitter.setChildrenCollapsible(False)  # 防止子控件被完全折叠
        self.content_splitter.setOpaqueResize(True)  # 实时调整大小，提高响应速度
        self.content_splitter.setStyleSheet("""
            QSplitter::handle {
                background-color: #4a4a4a;
                border-left: 1px solid #3a3a3a;
                border-right: 1px solid #3a3a3a;
            }
            QSplitter::handle:hover {
                background-color: #2196F3;
            }
            QSplitter {
                background-color: #2c2c2c;
            }
        """)
        
        # 左侧：侧边面板容器
        self.side_panel_container = QtWidgets.QFrame()
        self.side_panel_container.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.side_panel_container.setStyleSheet("""
            QFrame {
                background-color: #2c2c2c;
                border-right: 1px solid #1a1a1a;
            }
        """)
        self.side_panel_container.setMinimumWidth(250)
        self.side_panel_container.setMaximumWidth(600)
        side_panel_layout = QtWidgets.QVBoxLayout(self.side_panel_container)
        side_panel_layout.setContentsMargins(0, 0, 0, 0)
        side_panel_layout.setSpacing(0)
        
        # 创建 QStackedWidget 来管理不同的面板内容
        self.panel_stack = QtWidgets.QStackedWidget()
        self.panel_stack.setStyleSheet("""
            QStackedWidget {
                background-color: #2c2c2c;
            }
            /* 面板内容区域的文字颜色 */
            QLabel {
                color: #ffffff;
            }
            QGroupBox {
                color: #ffffff;
                font-weight: bold;
            }
            QCheckBox {
                color: #ffffff;
            }
            QRadioButton {
                color: #ffffff;
            }
        """)
        
        # 创建各个面板
        self._create_panels()
        
        side_panel_layout.addWidget(self.panel_stack)
        
        # 默认隐藏侧边栏
        self.side_panel_container.hide()
        self.current_panel_index = -1  # -1 表示没有面板打开
        
        # 设置固定宽度策略，避免拖动时内容重排
        self.side_panel_container.setSizePolicy(
            QtWidgets.QSizePolicy.Preferred,
            QtWidgets.QSizePolicy.Expanding
        )
        
        self.content_splitter.addWidget(self.side_panel_container)

        # 右侧区域：视图工具栏 + 视图
        right_area = QtWidgets.QWidget()
        right_area.setStyleSheet("QWidget { background-color: #3a3a3a; }")  # 设置为与PyVista相同的深灰色
        right_area.setAutoFillBackground(True)  # 确保背景填充
        right_area_layout = QtWidgets.QVBoxLayout(right_area)
        right_area_layout.setContentsMargins(0, 0, 0, 0)
        right_area_layout.setSpacing(0)
        
        # 视图工具栏 - 现代扁平化设计
        view_toolbar = QtWidgets.QFrame()
        view_toolbar.setFrameShape(QtWidgets.QFrame.NoFrame)
        view_toolbar.setStyleSheet("""
            QFrame { 
                background-color: #3a3a3a;
                border: none;
                border-bottom: 1px solid #2a2a2a;
            }
        """)
        view_toolbar.setFixedHeight(50)  # 增加高度以容纳按钮
        view_toolbar_layout = QtWidgets.QHBoxLayout(view_toolbar)
        view_toolbar_layout.setContentsMargins(10, 8, 10, 8)
        view_toolbar_layout.setSpacing(8)
        
        # 添加弹簧，让按钮居中
        view_toolbar_layout.addStretch()
        
        # 现代扁平化按钮样式（适配深色背景）
        button_style = """
            QPushButton {
                background-color: #4a4a4a;
                border: 1px solid #5a5a5a;
                border-radius: 4px;
                padding: 0px;
                font-size: 16px;
                min-width: 36px;
                max-width: 36px;
                min-height: 36px;
                max-height: 36px;
                color: #ccc;
            }
            QPushButton:hover {
                background-color: #5a5a5a;
                border-color: #2196F3;
                color: #2196F3;
            }
            QPushButton:pressed {
                background-color: #3a3a3a;
                border-color: #1976D2;
                color: #1976D2;
            }
        """
        
        # 前视图
        self.btn_view_front = QtWidgets.QPushButton("↑")
        self.btn_view_front.setStyleSheet(button_style)
        self.btn_view_front.setToolTip("前视图")
        self.btn_view_front.clicked.connect(lambda: self._set_camera_view("front"))
        view_toolbar_layout.addWidget(self.btn_view_front)
        
        # 后视图
        self.btn_view_back = QtWidgets.QPushButton("↓")
        self.btn_view_back.setStyleSheet(button_style)
        self.btn_view_back.setToolTip("后视图")
        self.btn_view_back.clicked.connect(lambda: self._set_camera_view("back"))
        view_toolbar_layout.addWidget(self.btn_view_back)
        
        # 左视图
        self.btn_view_left = QtWidgets.QPushButton("←")
        self.btn_view_left.setStyleSheet(button_style)
        self.btn_view_left.setToolTip("左视图")
        self.btn_view_left.clicked.connect(lambda: self._set_camera_view("left"))
        view_toolbar_layout.addWidget(self.btn_view_left)
        
        # 右视图
        self.btn_view_right = QtWidgets.QPushButton("→")
        self.btn_view_right.setStyleSheet(button_style)
        self.btn_view_right.setToolTip("右视图")
        self.btn_view_right.clicked.connect(lambda: self._set_camera_view("right"))
        view_toolbar_layout.addWidget(self.btn_view_right)
        
        # 顶视图
        self.btn_view_top = QtWidgets.QPushButton("⊤")
        self.btn_view_top.setStyleSheet(button_style)
        self.btn_view_top.setToolTip("顶视图")
        self.btn_view_top.clicked.connect(lambda: self._set_camera_view("top"))
        view_toolbar_layout.addWidget(self.btn_view_top)
        
        # 底视图
        self.btn_view_bottom = QtWidgets.QPushButton("⊥")
        self.btn_view_bottom.setStyleSheet(button_style)
        self.btn_view_bottom.setToolTip("底视图")
        self.btn_view_bottom.clicked.connect(lambda: self._set_camera_view("bottom"))
        view_toolbar_layout.addWidget(self.btn_view_bottom)
        
        # 小间距
        view_toolbar_layout.addSpacing(8)
        
        # 等轴测视图
        self.btn_view_isometric = QtWidgets.QPushButton("◇")
        self.btn_view_isometric.setStyleSheet(button_style)
        self.btn_view_isometric.setToolTip("等轴测视图")
        self.btn_view_isometric.clicked.connect(lambda: self._set_camera_view("isometric"))
        view_toolbar_layout.addWidget(self.btn_view_isometric)
        
        # 重置视图
        self.btn_view_reset = QtWidgets.QPushButton("⟲")
        self.btn_view_reset.setStyleSheet(button_style)
        self.btn_view_reset.setToolTip("重置视图")
        self.btn_view_reset.clicked.connect(lambda: self._set_camera_view("reset"))
        view_toolbar_layout.addWidget(self.btn_view_reset)
        
        # 添加弹簧，让按钮居中
        view_toolbar_layout.addStretch()
        
        # 添加工具栏到右侧区域
        right_area_layout.addWidget(view_toolbar)

        # 右侧视图区域
        self.viewer_host = QtWidgets.QFrame(self)
        self.viewer_host.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.viewer_host.setStyleSheet("QFrame { background-color: #3a3a3a; border: none; }")
        self.viewer_host.setAutoFillBackground(True)  # 确保背景填充
        viewer_host_layout = QtWidgets.QVBoxLayout(self.viewer_host)
        viewer_host_layout.setContentsMargins(0, 0, 0, 0)
        viewer_host_layout.setSpacing(0)
        viewer_host_layout.addWidget(self.viewer)
        
        # 添加viewer_host到右侧区域
        right_area_layout.addWidget(self.viewer_host, stretch=1)
        
        # 添加右侧区域到splitter
        self.content_splitter.addWidget(right_area)
        
        # 设置splitter的初始大小（侧边栏默认350px）
        self.content_splitter.setSizes([350, 1050])  # 默认宽度分配
        
        # 设置右侧区域的大小策略，确保它能快速响应
        right_area.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding,
            QtWidgets.QSizePolicy.Expanding
        )
        
        # 连接splitter的移动信号，强制重绘面板
        self.content_splitter.splitterMoved.connect(self._on_splitter_moved)
        
        # 使用垂直splitter分隔3D视图和日志区域
        self.vertical_splitter = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        self.vertical_splitter.setHandleWidth(4)
        self.vertical_splitter.setChildrenCollapsible(False)
        self.vertical_splitter.setOpaqueResize(True)
        self.vertical_splitter.setStyleSheet("""
            QSplitter::handle {
                background-color: #4a4a4a;
                border-top: 1px solid #3a3a3a;
                border-bottom: 1px solid #3a3a3a;
            }
            QSplitter::handle:hover {
                background-color: #2196F3;
            }
            QSplitter {
                background-color: #2c2c2c;
            }
        """)
        
        # 将content_splitter添加到vertical_splitter
        self.vertical_splitter.addWidget(self.content_splitter)
        
        # 底部：状态日志（可调整高度）
        log_container = QtWidgets.QWidget()
        log_container.setStyleSheet("background-color: #2c2c2c; border-top: 1px solid #1a1a1a;")
        log_container.setMinimumHeight(100)  # 最小高度
        log_container_layout = QtWidgets.QVBoxLayout(log_container)
        log_container_layout.setContentsMargins(10, 8, 10, 8)
        log_container_layout.setSpacing(5)
        
        log_label = QtWidgets.QLabel("运行日志")
        log_label.setStyleSheet("font-size: 12px; font-weight: bold; color: #aaa; padding: 2px;")
        log_container_layout.addWidget(log_label)
        
        self.status_box = QtWidgets.QPlainTextEdit()
        self.status_box.setReadOnly(True)
        self.status_box.setPlaceholderText("运行日志将显示在这里...")
        # 设置文本换行模式
        self.status_box.setLineWrapMode(QtWidgets.QPlainTextEdit.WidgetWidth)
        # 设置最大行数限制，避免日志过多导致性能问题
        self.status_box.setMaximumBlockCount(1000)
        self.status_box.setStyleSheet("""
            QPlainTextEdit {
                background-color: #1e1e1e;
                border: 1px solid #3a3a3a;
                border-radius: 4px;
                padding: 8px;
                font-family: Consolas, Monaco, monospace;
                font-size: 13px;
                line-height: 1.5;
                color: #e0e0e0;
            }
            QPlainTextEdit::placeholder {
                color: #666;
            }
        """)
        log_container_layout.addWidget(self.status_box)
        
        # 将日志容器添加到vertical_splitter
        self.vertical_splitter.addWidget(log_container)
        
        # 设置初始大小比例（3D视图占大部分，日志占180px）
        self.vertical_splitter.setSizes([680, 180])
        
        # 将vertical_splitter添加到主布局
        main_layout.addWidget(self.vertical_splitter, stretch=1)
        
        # 连接垂直splitter的移动信号
        self.vertical_splitter.splitterMoved.connect(self._on_vertical_splitter_moved)
        
        # 初始化所有参数控件（隐藏，通过对话框访问）
        self._init_parameter_widgets()
        
        self._set_mouse_mode("rotate")
        
        # 为中央控件启用鼠标追踪（在 setCentralWidget 之后）
        root.setMouseTracking(True)
    
    def _create_toolbar(self) -> None:
        """创建功能工具栏"""
        self.toolbar = QtWidgets.QFrame()
        self.toolbar.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.toolbar.setStyleSheet("""
            QFrame {
                background-color: #2c2c2c;
                border-bottom: 1px solid #1a1a1a;
            }
        """)
        self.toolbar.setFixedHeight(70)
        
        toolbar_layout = QtWidgets.QHBoxLayout(self.toolbar)
        toolbar_layout.setContentsMargins(10, 8, 10, 8)
        toolbar_layout.setSpacing(8)
        
        # 工具栏按钮样式 - 正方形，图标在上，文字在下
        toolbar_button_style = """
            QPushButton {
                background-color: #3a3a3a;
                border: 1px solid #4a4a4a;
                border-radius: 5px;
                padding: 6px;
                font-size: 10px;
                color: #ffffff;
                min-width: 56px;
                max-width: 56px;
                min-height: 54px;
                max-height: 54px;
            }
            QPushButton:hover {
                background-color: #4a4a4a;
                border-color: #2196F3;
            }
            QPushButton:pressed {
                background-color: #2a2a2a;
            }
            QPushButton:checked {
                background-color: #2196F3;
                border-color: #2196F3;
                color: #ffffff;
            }
        """
        
        # 创建4个功能按钮（使用垂直布局：图标在上，文字在下）
        self.btn_panel_lattice = self._create_toolbar_button("🔷", "晶格生成", toolbar_button_style)
        self.btn_panel_lattice.clicked.connect(lambda: self._toggle_panel(0))
        toolbar_layout.addWidget(self.btn_panel_lattice)
        
        self.btn_panel_slice = self._create_toolbar_button("✂️", "模型分割", toolbar_button_style)
        self.btn_panel_slice.clicked.connect(lambda: self._toggle_panel(1))
        toolbar_layout.addWidget(self.btn_panel_slice)
        
        self.btn_panel_region_a = self._create_toolbar_button("🎯", "区域设计A", toolbar_button_style)
        self.btn_panel_region_a.clicked.connect(lambda: self._toggle_panel(2))
        toolbar_layout.addWidget(self.btn_panel_region_a)
        
        self.btn_panel_region_b = self._create_toolbar_button("🎯", "区域设计B", toolbar_button_style)
        self.btn_panel_region_b.clicked.connect(lambda: self._toggle_panel(3))
        toolbar_layout.addWidget(self.btn_panel_region_b)
        
        # 保存按钮引用到列表
        self.panel_buttons = [
            self.btn_panel_lattice,
            self.btn_panel_slice,
            self.btn_panel_region_a,
            self.btn_panel_region_b
        ]
        
        toolbar_layout.addStretch()
    
    def _create_toolbar_button(self, icon: str, text: str, style: str) -> QtWidgets.QPushButton:
        """创建工具栏按钮（图标在上，文字在下）"""
        button = QtWidgets.QPushButton()
        button.setCheckable(True)
        button.setStyleSheet(style)
        
        # 创建垂直布局
        layout = QtWidgets.QVBoxLayout(button)
        layout.setContentsMargins(3, 4, 3, 4)
        layout.setSpacing(3)
        
        # 图标标签
        icon_label = QtWidgets.QLabel(icon)
        icon_label.setAlignment(QtCore.Qt.AlignCenter)
        icon_label.setStyleSheet("font-size: 20px; background: transparent; border: none; color: #ffffff; font-weight: bold;")
        layout.addWidget(icon_label)
        
        # 文字标签
        text_label = QtWidgets.QLabel(text)
        text_label.setAlignment(QtCore.Qt.AlignCenter)
        text_label.setStyleSheet("font-size: 10px; background: transparent; border: none; color: #ffffff; font-weight: bold;")
        text_label.setWordWrap(True)
        layout.addWidget(text_label)
        
        return button
    
    def _toggle_panel(self, panel_index: int) -> None:
        """切换面板显示/隐藏"""
        # 如果点击的是当前已打开的面板，则关闭它
        if self.current_panel_index == panel_index:
            self.side_panel_container.hide()
            self.current_panel_index = -1
            self.panel_buttons[panel_index].setChecked(False)
        else:
            # 取消其他按钮的选中状态
            for i, btn in enumerate(self.panel_buttons):
                btn.setChecked(i == panel_index)
            
            # 显示侧边栏并切换到对应面板
            self.panel_stack.setCurrentIndex(panel_index)
            self.side_panel_container.show()
            self.current_panel_index = panel_index
            
            # 强制更新布局和重绘
            self.side_panel_container.update()
            self.panel_stack.currentWidget().update()
    
    def _on_splitter_moved(self, pos: int, index: int) -> None:
        """splitter移动时强制重绘面板"""
        if self.current_panel_index >= 0:
            # 强制更新当前面板
            self.side_panel_container.update()
            current_widget = self.panel_stack.currentWidget()
            if current_widget:
                current_widget.update()
                # 强制处理所有待处理的事件，确保立即重绘
                QtWidgets.QApplication.processEvents()
    
    def _on_vertical_splitter_moved(self, pos: int, index: int) -> None:
        """垂直splitter移动时强制重绘日志区域"""
        # 强制更新日志容器
        if hasattr(self, 'status_box'):
            self.status_box.update()
            # 强制处理所有待处理的事件，确保立即重绘
            QtWidgets.QApplication.processEvents()
    
    def _on_toggle_log(self, checked: bool) -> None:
        """切换运行日志显示/隐藏"""
        if hasattr(self, 'vertical_splitter'):
            # 获取日志容器（vertical_splitter的第二个子控件）
            log_widget = self.vertical_splitter.widget(1)
            if log_widget:
                if checked:
                    log_widget.show()
                    # 恢复之前的大小，如果没有保存则使用默认值
                    if hasattr(self, '_saved_splitter_sizes'):
                        self.vertical_splitter.setSizes(self._saved_splitter_sizes)
                    else:
                        self.vertical_splitter.setSizes([680, 180])
                else:
                    # 保存当前大小
                    self._saved_splitter_sizes = self.vertical_splitter.sizes()
                    log_widget.hide()
    
    def _create_panels(self) -> None:
        """创建所有面板（复用原选项卡内容）"""
        # 面板1：晶格生成
        panel1 = QtWidgets.QWidget()
        panel1_layout = QtWidgets.QVBoxLayout(panel1)
        panel1_layout.setContentsMargins(10, 10, 10, 10)
        panel1_layout.setSpacing(10)
        self._build_lattice_tab(panel1_layout)
        self.panel_stack.addWidget(panel1)
        
        # 面板2：模型分割
        panel2 = QtWidgets.QWidget()
        panel2_layout = QtWidgets.QVBoxLayout(panel2)
        panel2_layout.setContentsMargins(10, 10, 10, 10)
        panel2_layout.setSpacing(10)
        self._build_slice_tab(panel2_layout)
        self.panel_stack.addWidget(panel2)
        
        # 面板3：区域晶格设计
        panel3 = QtWidgets.QWidget()
        panel3_layout = QtWidgets.QVBoxLayout(panel3)
        panel3_layout.setContentsMargins(10, 10, 10, 10)
        panel3_layout.setSpacing(10)
        self._build_region_lattice_tab(panel3_layout)
        self.panel_stack.addWidget(panel3)
        
        # 面板4：平面区域晶格设计
        panel4 = QtWidgets.QWidget()
        panel4_layout = QtWidgets.QVBoxLayout(panel4)
        panel4_layout.setContentsMargins(10, 10, 10, 10)
        panel4_layout.setSpacing(10)
        self._build_plane_region_tab(panel4_layout)
        self.panel_stack.addWidget(panel4)
    
    def _create_tabs(self) -> None:
        """创建所有选项卡（已废弃，保留用于向后兼容）"""
        # 此方法已被 _create_panels 替代
        # 保留空方法以防止旧代码引用时出错
        pass
    
    def _init_parameter_widgets(self) -> None:
        """初始化所有参数控件（不显示在主界面）"""
        # 注意：大部分参数控件现在在 _build_lattice_tab() 中创建
        # 这里只创建不在选项卡中的控件
        
        # 分割相关（稍后在对话框中创建）
        self.check_enable_slice = QtWidgets.QCheckBox()
    
    def _build_lattice_tab(self, layout: QtWidgets.QVBoxLayout) -> None:
        """构建晶格生成选项卡"""
        # 鞋底和晶格路径 - 初始为空，让用户手动选择
        self.edit_sole_path = QtWidgets.QLineEdit("")
        self.edit_lattice_path = QtWidgets.QLineEdit("")
        self.edit_output_path = QtWidgets.QLineEdit("")
        self.edit_sole_path.editingFinished.connect(self._on_sole_path_changed)
        self.edit_lattice_path.editingFinished.connect(self._on_lattice_path_changed)
        layout.addWidget(self._path_row("鞋底 STL", self.edit_sole_path, self._browse_sole, self._preview_selected_sole))
        layout.addWidget(self._path_row("晶格 STL", self.edit_lattice_path, self._browse_lattice, self._preview_selected_lattice))
        layout.addWidget(self._path_row("输出 STL", self.edit_output_path, self._browse_output, None))

        # 晶格方法选择
        method_layout = QtWidgets.QHBoxLayout()
        method_layout.addWidget(QtWidgets.QLabel("晶格方法:"))
        self.combo_method = QtWidgets.QComboBox()
        # 设置下拉列表的最大可见项数
        self.combo_method.setMaxVisibleItems(10)
        self.combo_method.addItems(["gyroid", "voronoi_2_5d", "voronoi_implicit", "tile_unit"])
        self.combo_method.setCurrentText(LATTICE_METHOD)
        self.combo_method.currentTextChanged.connect(self._on_method_changed)
        method_layout.addWidget(self.combo_method, stretch=1)
        layout.addLayout(method_layout)

        # 参数策略（紧凑布局）
        strategy_layout = QtWidgets.QHBoxLayout()
        
        self.check_auto_recommend = QtWidgets.QCheckBox("自动推荐")
        self.check_auto_recommend.setChecked(True)
        self.check_auto_recommend.setToolTip("切换模型/方法时自动推荐参数")
        strategy_layout.addWidget(self.check_auto_recommend)
        
        self.button_recommend = QtWidgets.QPushButton("💡 推荐")
        self.button_recommend.setToolTip("根据鞋底尺寸推荐当前方法的参数")
        self.button_recommend.clicked.connect(lambda: self._apply_recommended_params(force=True, reason="已重新推荐当前方法参数"))
        self.button_recommend.setMaximumWidth(80)
        strategy_layout.addWidget(self.button_recommend)
        
        strategy_layout.addStretch()
        
        layout.addLayout(strategy_layout)
        
        layout.addSpacing(5)

        # === 创建参数组（使用 QStackedWidget 实现动态切换）===
        self.params_stack = QtWidgets.QStackedWidget()
        
        # Gyroid 参数页
        gyroid_page = QtWidgets.QWidget()
        gyroid_layout = QtWidgets.QVBoxLayout(gyroid_page)
        gyroid_layout.setContentsMargins(0, 0, 0, 0)
        
        self.spin_gyroid_cell = QtWidgets.QDoubleSpinBox()
        self.spin_gyroid_cell.setRange(0.5, 50.0)
        self.spin_gyroid_cell.setDecimals(3)
        self.spin_gyroid_cell.setSingleStep(0.5)
        self.spin_gyroid_cell.setValue(float(GYROID_CELL_SIZE))

        self.spin_gyroid_iso = QtWidgets.QDoubleSpinBox()
        self.spin_gyroid_iso.setRange(-1.5, 1.5)
        self.spin_gyroid_iso.setDecimals(3)
        self.spin_gyroid_iso.setSingleStep(0.05)
        self.spin_gyroid_iso.setValue(float(GYROID_ISOVALUE))

        self.spin_gyroid_res = QtWidgets.QSpinBox()
        self.spin_gyroid_res.setRange(8, 128)
        self.spin_gyroid_res.setValue(int(GYROID_RESOLUTION))

        self.gyroid_group = QtWidgets.QGroupBox("Gyroid 参数")
        gyroid_form = QtWidgets.QFormLayout(self.gyroid_group)
        gyroid_form.addRow("单胞尺寸 (mm):", self.spin_gyroid_cell)
        gyroid_form.addRow("等值面:", self.spin_gyroid_iso)
        gyroid_form.addRow("分辨率:", self.spin_gyroid_res)
        gyroid_layout.addWidget(self.gyroid_group)
        gyroid_layout.addStretch()
        self.params_stack.addWidget(gyroid_page)

        # Voronoi 2.5D 参数页
        voronoi_page = QtWidgets.QWidget()
        voronoi_layout = QtWidgets.QVBoxLayout(voronoi_page)
        voronoi_layout.setContentsMargins(0, 0, 0, 0)
        
        self.spin_voronoi_cell = QtWidgets.QDoubleSpinBox()
        self.spin_voronoi_cell.setRange(1.0, 100.0)
        self.spin_voronoi_cell.setDecimals(3)
        self.spin_voronoi_cell.setValue(float(VORONOI_CELL_SIZE))
        
        self.spin_voronoi_thickness = QtWidgets.QDoubleSpinBox()
        self.spin_voronoi_thickness.setRange(0.1, 20.0)
        self.spin_voronoi_thickness.setDecimals(3)
        self.spin_voronoi_thickness.setValue(float(VORONOI_STRUT_THICKNESS))
        
        self.spin_voronoi_layers = QtWidgets.QSpinBox()
        self.spin_voronoi_layers.setRange(2, 30)
        self.spin_voronoi_layers.setValue(int(VORONOI_Z_LAYERS))

        self.voronoi_group = QtWidgets.QGroupBox("Voronoi 2.5D 参数")
        voronoi_form = QtWidgets.QFormLayout(self.voronoi_group)
        voronoi_form.addRow("胞元尺寸 (mm):", self.spin_voronoi_cell)
        voronoi_form.addRow("杆径 (mm):", self.spin_voronoi_thickness)
        voronoi_form.addRow("Z 向层数:", self.spin_voronoi_layers)
        voronoi_layout.addWidget(self.voronoi_group)
        voronoi_layout.addStretch()
        self.params_stack.addWidget(voronoi_page)

        # Voronoi Implicit 参数页
        voronoi_implicit_page = QtWidgets.QWidget()
        voronoi_implicit_layout = QtWidgets.QVBoxLayout(voronoi_implicit_page)
        voronoi_implicit_layout.setContentsMargins(0, 0, 0, 0)
        
        self.spin_voronoi_implicit_cell = QtWidgets.QDoubleSpinBox()
        self.spin_voronoi_implicit_cell.setRange(1.0, 100.0)
        self.spin_voronoi_implicit_cell.setDecimals(3)
        self.spin_voronoi_implicit_cell.setValue(float(VORONOI_IMPLICIT_CELL_SIZE))
        
        self.spin_voronoi_implicit_wall = QtWidgets.QDoubleSpinBox()
        self.spin_voronoi_implicit_wall.setRange(0.1, 5.0)
        self.spin_voronoi_implicit_wall.setDecimals(3)
        self.spin_voronoi_implicit_wall.setValue(float(VORONOI_IMPLICIT_WALL_THICKNESS))
        
        self.spin_voronoi_implicit_res = QtWidgets.QSpinBox()
        self.spin_voronoi_implicit_res.setRange(10, 50)
        self.spin_voronoi_implicit_res.setValue(int(VORONOI_IMPLICIT_RESOLUTION))
        
        self.spin_voronoi_implicit_density = QtWidgets.QDoubleSpinBox()
        self.spin_voronoi_implicit_density.setRange(0.5, 2.0)
        self.spin_voronoi_implicit_density.setDecimals(3)
        self.spin_voronoi_implicit_density.setSingleStep(0.1)
        self.spin_voronoi_implicit_density.setValue(float(VORONOI_IMPLICIT_SEED_DENSITY_FACTOR))

        self.voronoi_implicit_group = QtWidgets.QGroupBox("Voronoi Implicit 参数（3D 隐函数）")
        voronoi_implicit_form = QtWidgets.QFormLayout(self.voronoi_implicit_group)
        voronoi_implicit_form.addRow("胞元尺寸 (mm):", self.spin_voronoi_implicit_cell)
        voronoi_implicit_form.addRow("壁厚 (mm):", self.spin_voronoi_implicit_wall)
        voronoi_implicit_form.addRow("分辨率:", self.spin_voronoi_implicit_res)
        voronoi_implicit_form.addRow("种子密度系数:", self.spin_voronoi_implicit_density)
        voronoi_implicit_layout.addWidget(self.voronoi_implicit_group)
        voronoi_implicit_layout.addStretch()
        self.params_stack.addWidget(voronoi_implicit_page)

        # Tile 参数页
        tile_page = QtWidgets.QWidget()
        tile_layout = QtWidgets.QVBoxLayout(tile_page)
        tile_layout.setContentsMargins(0, 0, 0, 0)
        
        self.spin_tile_shrink = QtWidgets.QDoubleSpinBox()
        self.spin_tile_shrink.setRange(0.05, 2.0)
        self.spin_tile_shrink.setDecimals(3)
        self.spin_tile_shrink.setSingleStep(0.02)
        self.spin_tile_shrink.setValue(float(LATTICE_TILE_SHRINK))
        
        self.spin_tile_spacing = QtWidgets.QDoubleSpinBox()
        self.spin_tile_spacing.setRange(0.5, 1.5)
        self.spin_tile_spacing.setDecimals(3)
        self.spin_tile_spacing.setSingleStep(0.02)
        self.spin_tile_spacing.setValue(float(TILE_SPACING_FACTOR))
        
        self.spin_tile_margin = QtWidgets.QDoubleSpinBox()
        self.spin_tile_margin.setRange(0.0, 5.0)
        self.spin_tile_margin.setDecimals(3)
        self.spin_tile_margin.setSingleStep(0.1)
        self.spin_tile_margin.setValue(float(TILE_BOUNDARY_MARGIN))
        
        self.spin_tile_decimate = QtWidgets.QSpinBox()
        self.spin_tile_decimate.setRange(0, 10000)
        self.spin_tile_decimate.setValue(int(LATTICE_UNIT_TARGET_FACES))
        self.spin_tile_decimate.setSingleStep(100)
        self.spin_tile_decimate.setSpecialValueText("禁用减面")
        self.spin_tile_decimate.setToolTip("目标面数（0 = 禁用减面，保持最高精度；>0 = 减面到指定面数）")
        self.spin_tile_decimate.valueChanged.connect(self._on_tile_decimate_changed)

        self.tile_group = QtWidgets.QGroupBox("Tile Unit 参数")
        tile_form = QtWidgets.QFormLayout(self.tile_group)
        tile_form.addRow("单元尺寸缩放:", self.spin_tile_shrink)
        tile_form.addRow("平铺间距系数:", self.spin_tile_spacing)
        tile_form.addRow("边界安全距离 (mm):", self.spin_tile_margin)
        tile_form.addRow("晶格单元减面:", self.spin_tile_decimate)
        tile_layout.addWidget(self.tile_group)
        
        # 减面提示
        tile_decimate_hint = QtWidgets.QLabel("💡 减面会影响生成精度，建议设为 0 保持最高精度")
        tile_decimate_hint.setStyleSheet("color: #aaaaaa; font-size: 10px; padding: 4px;")
        tile_decimate_hint.setWordWrap(True)
        tile_layout.addWidget(tile_decimate_hint)
        
        tile_layout.addStretch()
        self.params_stack.addWidget(tile_page)
        
        # 添加参数堆栈到布局
        layout.addWidget(self.params_stack)

        # 全局选项（紧跟在参数页面后面）
        global_options_group = QtWidgets.QGroupBox("全局选项")
        global_options_layout = QtWidgets.QVBoxLayout(global_options_group)
        global_options_layout.setContentsMargins(8, 8, 8, 8)
        global_options_layout.setSpacing(6)
        
        self.check_enable_final_trim = QtWidgets.QCheckBox("启用最后精细裁剪（推荐）")
        self.check_enable_final_trim.setChecked(True)
        self.check_enable_final_trim.setToolTip(
            "启用后，在晶格生成的最后一步会进行精细裁剪，确保晶格完全在鞋底内部。\n"
            "禁用后，可能会有少量晶格超出鞋底边界（用于对比测试）。\n"
            "此选项影响所有选项卡的晶格生成。"
        )
        self.check_enable_final_trim.setStyleSheet("padding: 5px; font-size: 12px;")
        global_options_layout.addWidget(self.check_enable_final_trim)
        
        layout.addWidget(global_options_group)

        # 生成和导出按钮
        self.button_generate = QtWidgets.QPushButton("生成并预览")
        self.button_generate.clicked.connect(self._generate)
        self.button_export = QtWidgets.QPushButton("导出当前结果")
        self.button_export.clicked.connect(self._export_current)
        layout.addWidget(self.button_generate)
        layout.addWidget(self.button_export)
        
        # 添加弹性空间，让上面的内容紧凑排列
        layout.addStretch()
    
    def _build_slice_tab(self, layout: QtWidgets.QVBoxLayout) -> None:
        """构建模型分割选项卡"""
        # 提示信息
        slice_hint = QtWidgets.QLabel("💡 对鞋底模型进行平面分割")
        slice_hint.setStyleSheet("color: #ffffff; font-size: 11px; padding: 8px; background: #3a3a3a; border-radius: 4px;")
        slice_hint.setWordWrap(True)
        layout.addWidget(slice_hint)
        
        # 启用分割
        self.check_enable_slice = QtWidgets.QCheckBox("启用分割功能")
        self.check_enable_slice.setToolTip("勾选后可以对鞋底模型进行平面分割")
        self.check_enable_slice.setStyleSheet("font-weight: bold; padding: 6px;")
        self.check_enable_slice.stateChanged.connect(self._on_slice_enabled_changed)
        layout.addWidget(self.check_enable_slice)
        
        # 分割参数组
        params_group = QtWidgets.QGroupBox("分割参数")
        params_layout = QtWidgets.QVBoxLayout(params_group)
        params_layout.setSpacing(6)
        params_layout.setContentsMargins(8, 8, 8, 8)
        
        # 分割轴和显示平面（合并到一行）
        axis_plane_layout = QtWidgets.QHBoxLayout()
        axis_plane_layout.addWidget(QtWidgets.QLabel("分割轴:"))
        self.slice_axis_combo = QtWidgets.QComboBox()
        # 设置下拉列表的最大可见项数
        self.slice_axis_combo.setMaxVisibleItems(10)
        self.slice_axis_combo.addItems(["X 轴", "Y 轴", "Z 轴"])
        self.slice_axis_combo.setCurrentIndex(1)
        self.slice_axis_combo.currentIndexChanged.connect(self._on_slice_axis_changed)
        axis_plane_layout.addWidget(self.slice_axis_combo)
        axis_plane_layout.addSpacing(20)
        self.check_show_plane = QtWidgets.QCheckBox("显示平面")
        self.check_show_plane.setToolTip("在 3D 视图中显示分割平面")
        self.check_show_plane.stateChanged.connect(self._on_show_plane_changed)
        axis_plane_layout.addWidget(self.check_show_plane)
        axis_plane_layout.addStretch()
        params_layout.addLayout(axis_plane_layout)
        
        # 位置滑块
        self.slice_position_label = QtWidgets.QLabel("位置: 0.0 mm")
        params_layout.addWidget(self.slice_position_label)
        
        self.slice_position_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slice_position_slider.setMinimum(0)
        self.slice_position_slider.setMaximum(1000)
        self.slice_position_slider.setValue(500)
        self.slice_position_slider.valueChanged.connect(self._on_slice_position_changed)
        params_layout.addWidget(self.slice_position_slider)
        
        # 角度调整（紧凑布局）
        angles_layout = QtWidgets.QHBoxLayout()
        self.slice_angle1_label = QtWidgets.QLabel("角度1:")
        angles_layout.addWidget(self.slice_angle1_label)
        self.slice_angle1_spin = QtWidgets.QDoubleSpinBox()
        self.slice_angle1_spin.setRange(-180.0, 180.0)
        self.slice_angle1_spin.setValue(0.0)
        self.slice_angle1_spin.setSingleStep(5.0)
        self.slice_angle1_spin.setSuffix("°")
        self.slice_angle1_spin.setToolTip("绕第一个轴旋转分割平面")
        self.slice_angle1_spin.valueChanged.connect(self._on_slice_angle_changed)
        self.slice_angle1_spin.setMaximumWidth(100)
        angles_layout.addWidget(self.slice_angle1_spin)
        
        angles_layout.addSpacing(10)
        self.slice_angle2_label = QtWidgets.QLabel("角度2:")
        angles_layout.addWidget(self.slice_angle2_label)
        self.slice_angle2_spin = QtWidgets.QDoubleSpinBox()
        self.slice_angle2_spin.setRange(-180.0, 180.0)
        self.slice_angle2_spin.setValue(0.0)
        self.slice_angle2_spin.setSingleStep(5.0)
        self.slice_angle2_spin.setSuffix("°")
        self.slice_angle2_spin.setToolTip("绕第二个轴旋转分割平面")
        self.slice_angle2_spin.valueChanged.connect(self._on_slice_angle_changed)
        self.slice_angle2_spin.setMaximumWidth(100)
        angles_layout.addWidget(self.slice_angle2_spin)
        angles_layout.addStretch()
        params_layout.addLayout(angles_layout)
        
        # 网格细化（合并到一行）
        subdivide_layout = QtWidgets.QHBoxLayout()
        subdivide_layout.addWidget(QtWidgets.QLabel("网格细化:"))
        self.slice_subdivide_spin = QtWidgets.QSpinBox()
        self.slice_subdivide_spin.setRange(0, 5)
        self.slice_subdivide_spin.setValue(0)
        self.slice_subdivide_spin.setSuffix(" 次")
        self.slice_subdivide_spin.setToolTip("细化次数，每次将面数增加4倍\n0=不细化，1=4倍，2=16倍，3=64倍...")
        self.slice_subdivide_spin.setMaximumWidth(100)
        subdivide_layout.addWidget(self.slice_subdivide_spin)
        subdivide_layout.addStretch()
        params_layout.addLayout(subdivide_layout)
        
        layout.addWidget(params_group)
        
        # 显示选项组（折叠不常用选项）
        display_group = QtWidgets.QGroupBox("显示选项")
        display_layout = QtWidgets.QVBoxLayout(display_group)
        display_layout.setSpacing(6)
        display_layout.setContentsMargins(8, 8, 8, 8)
        
        # 显示内容和区域间距（合并到一行）
        content_spacing_layout = QtWidgets.QHBoxLayout()
        content_spacing_layout.addWidget(QtWidgets.QLabel("显示:"))
        self.slice_display_combo = QtWidgets.QComboBox()
        # 设置下拉列表的最大可见项数
        self.slice_display_combo.setMaxVisibleItems(10)
        self.slice_display_combo.addItems(["原始", "分割", "正侧", "负侧"])
        self.slice_display_combo.currentIndexChanged.connect(self._on_slice_display_changed)
        content_spacing_layout.addWidget(self.slice_display_combo)
        
        content_spacing_layout.addSpacing(10)
        content_spacing_layout.addWidget(QtWidgets.QLabel("间距:"))
        self.slice_spacing_spin = QtWidgets.QDoubleSpinBox()
        self.slice_spacing_spin.setRange(0.0, 50.0)
        self.slice_spacing_spin.setValue(0.0)
        self.slice_spacing_spin.setSingleStep(1.0)
        self.slice_spacing_spin.setSuffix(" mm")
        self.slice_spacing_spin.setToolTip("分割结果显示时，两部分之间的距离")
        self.slice_spacing_spin.valueChanged.connect(self._on_slice_spacing_changed)
        self.slice_spacing_spin.setMaximumWidth(100)
        content_spacing_layout.addWidget(self.slice_spacing_spin)
        content_spacing_layout.addStretch()
        display_layout.addLayout(content_spacing_layout)
        
        # 颜色选择（合并到一行）
        color_layout = QtWidgets.QHBoxLayout()
        color_layout.addWidget(QtWidgets.QLabel("正侧:"))
        self.button_positive_color = QtWidgets.QPushButton()
        self.button_positive_color.setFixedSize(50, 22)
        self.positive_color = QtGui.QColor(0, 230, 230)  # 青色
        self.button_positive_color.setStyleSheet(f"background-color: {self.positive_color.name()}; border: 1px solid #999;")
        self.button_positive_color.clicked.connect(lambda: self._choose_color('positive'))
        color_layout.addWidget(self.button_positive_color)
        
        color_layout.addSpacing(10)
        color_layout.addWidget(QtWidgets.QLabel("负侧:"))
        self.button_negative_color = QtWidgets.QPushButton()
        self.button_negative_color.setFixedSize(50, 22)
        self.negative_color = QtGui.QColor(255, 128, 0)  # 橙色
        self.button_negative_color.setStyleSheet(f"background-color: {self.negative_color.name()}; border: 1px solid #999;")
        self.button_negative_color.clicked.connect(lambda: self._choose_color('negative'))
        color_layout.addWidget(self.button_negative_color)
        color_layout.addStretch()
        display_layout.addLayout(color_layout)
        
        # 其他选项（合并到一行）
        options_layout = QtWidgets.QHBoxLayout()
        self.check_slice_lighting = QtWidgets.QCheckBox("启用光照")
        self.check_slice_lighting.setChecked(True)
        self.check_slice_lighting.setToolTip("启用光照可以更好地显示模型的立体感")
        self.check_slice_lighting.stateChanged.connect(self._on_slice_lighting_changed)
        options_layout.addWidget(self.check_slice_lighting)
        options_layout.addStretch()
        display_layout.addLayout(options_layout)
        
        layout.addWidget(display_group)
        
        # 操作按钮组
        actions_group = QtWidgets.QGroupBox("操作")
        actions_layout = QtWidgets.QVBoxLayout(actions_group)
        actions_layout.setSpacing(6)
        actions_layout.setContentsMargins(8, 8, 8, 8)
        
        # 执行分割
        self.button_perform_slice = QtWidgets.QPushButton("执行分割")
        self.button_perform_slice.clicked.connect(self._perform_slice)
        self.button_perform_slice.setStyleSheet("background: #4CAF50; color: white; padding: 8px; font-weight: bold;")
        actions_layout.addWidget(self.button_perform_slice)
        
        # 重置
        self.button_reset_slice = QtWidgets.QPushButton("重置分割")
        self.button_reset_slice.clicked.connect(self._reset_slice)
        actions_layout.addWidget(self.button_reset_slice)
        
        # 导出按钮
        export_layout = QtWidgets.QHBoxLayout()
        self.button_export_positive = QtWidgets.QPushButton("导出正侧")
        self.button_export_positive.setEnabled(False)
        self.button_export_positive.clicked.connect(lambda: self._export_sliced_mesh('positive'))
        export_layout.addWidget(self.button_export_positive)
        
        self.button_export_negative = QtWidgets.QPushButton("导出负侧")
        self.button_export_negative.setEnabled(False)
        self.button_export_negative.clicked.connect(lambda: self._export_sliced_mesh('negative'))
        export_layout.addWidget(self.button_export_negative)
        actions_layout.addLayout(export_layout)
        
        layout.addWidget(actions_group)
        
        layout.addStretch()
        
        # 初始化时禁用所有分割控件
        params_group.setEnabled(False)
        display_group.setEnabled(False)
        actions_group.setEnabled(False)
        
        # 保存组引用以便后续启用/禁用
        self.slice_params_group = params_group
        self.slice_display_group = display_group
        self.slice_actions_group = actions_group

    def _create_actions(self) -> None:
        self.action_quit = QtWidgets.QAction("退出", self)
        self.action_clear_cache = QtWidgets.QAction("清空缓存", self)
        self.action_quit.triggered.connect(QtWidgets.qApp.quit)
        self.action_clear_cache.triggered.connect(self._clear_cache)
        self.action_show_sole = QtWidgets.QAction("显示鞋底", self)
        self.action_show_lattice = QtWidgets.QAction("显示晶格", self)
        self.action_show_combined = QtWidgets.QAction("显示鞋底+晶格", self)
        self.action_toggle_mesh_edges = QtWidgets.QAction("显示三角面片边缘", self)
        self.action_toggle_mesh_edges.setCheckable(True)
        self.action_toggle_mesh_edges.setChecked(False)
        
        # 网格平台选项
        self.action_show_ground_platform = QtWidgets.QAction("显示网格平台", self)
        self.action_show_ground_platform.setCheckable(True)
        self.action_show_ground_platform.setChecked(True)  # 默认显示
        self.action_show_ground_platform.toggled.connect(self._on_ground_platform_toggled)
        
        # 渲染器选择动作
        self.action_renderer_pyvista = QtWidgets.QAction("PyVista (VTK)", self)
        self.action_renderer_pyvista.setCheckable(True)
        self.action_renderer_pyvista.setChecked(True)
        self.action_renderer_pyvista.triggered.connect(lambda: self._switch_to_pyvista())
        
        self.action_renderer_open3d = QtWidgets.QAction("Open3D", self)
        self.action_renderer_open3d.setCheckable(True)
        self.action_renderer_open3d.setChecked(False)
        self.action_renderer_open3d.triggered.connect(lambda: self._switch_to_open3d())
        
        # 创建渲染器动作组（互斥）
        self.renderer_action_group = QtWidgets.QActionGroup(self)
        self.renderer_action_group.addAction(self.action_renderer_pyvista)
        self.renderer_action_group.addAction(self.action_renderer_open3d)
        self.renderer_action_group.setExclusive(True)
        
        # 添加标记：是否有生成的结果
        self.has_generated_result = False

    def _create_menu(self) -> None:
        menubar = self.menuBar()
        
        # 设置菜单栏样式 - 深色主题
        menubar.setStyleSheet("""
            QMenuBar {
                background-color: #1e1e1e;
                color: #ffffff;
                border: none;
                padding: 4px;
                font-size: 13px;
                font-weight: bold;
            }
            QMenuBar::item {
                background-color: transparent;
                padding: 6px 12px;
                border-radius: 4px;
                font-weight: bold;
            }
            QMenuBar::item:selected {
                background-color: #3d3d3d;
            }
            QMenuBar::item:pressed {
                background-color: #4d4d4d;
            }
            
            QMenu {
                background-color: #2c2c2c;
                color: #ffffff;
                border: 1px solid #3a3a3a;
                padding: 4px;
            }
            QMenu::item {
                background-color: transparent;
                padding: 6px 30px 6px 20px;
                border-radius: 3px;
            }
            QMenu::item:selected {
                background-color: #3d3d3d;
            }
            QMenu::item:disabled {
                color: #666666;
            }
            QMenu::separator {
                height: 1px;
                background-color: #3a3a3a;
                margin: 4px 10px;
            }
            QMenu::indicator {
                width: 16px;
                height: 16px;
                margin-left: 4px;
            }
            QMenu::indicator:checked {
                image: none;
                background-color: #2196F3;
                border-radius: 2px;
            }
        """)
        
        file_menu = menubar.addMenu("文件")
        
        # 工程管理
        self.action_new_project = QtWidgets.QAction("新建工程", self)
        self.action_new_project.setShortcut("Ctrl+N")
        self.action_new_project.triggered.connect(self._new_project)
        file_menu.addAction(self.action_new_project)
        
        self.action_open_project = QtWidgets.QAction("打开工程...", self)
        self.action_open_project.setShortcut("Ctrl+O")
        self.action_open_project.triggered.connect(self._open_project)
        file_menu.addAction(self.action_open_project)
        
        # 最近工程子菜单
        self.recent_projects_menu = file_menu.addMenu("最近工程")
        self._update_recent_projects_menu()
        
        file_menu.addSeparator()
        
        self.action_save_project = QtWidgets.QAction("保存工程", self)
        self.action_save_project.setShortcut("Ctrl+S")
        self.action_save_project.triggered.connect(self._save_project)
        file_menu.addAction(self.action_save_project)
        
        self.action_save_project_as = QtWidgets.QAction("另存为...", self)
        self.action_save_project_as.setShortcut("Ctrl+Shift+S")
        self.action_save_project_as.triggered.connect(self._save_project_as)
        file_menu.addAction(self.action_save_project_as)
        
        file_menu.addSeparator()
        file_menu.addAction(self.action_clear_cache)
        file_menu.addSeparator()
        file_menu.addAction(self.action_quit)
        
        view_menu = menubar.addMenu("视图")
        view_menu.addAction(self.action_show_sole)
        view_menu.addAction(self.action_show_lattice)
        view_menu.addAction(self.action_show_combined)
        view_menu.addSeparator()
        view_menu.addAction(self.action_toggle_mesh_edges)
        view_menu.addAction(self.action_show_ground_platform)
        
        # 添加相机视角子菜单
        view_menu.addSeparator()
        camera_view_menu = view_menu.addMenu("相机视角")
        
        # 创建视角动作
        self.action_view_front = QtWidgets.QAction("前视图", self)
        self.action_view_front.setShortcut("Num1")
        self.action_view_front.triggered.connect(lambda: self._set_camera_view("front"))
        camera_view_menu.addAction(self.action_view_front)
        
        self.action_view_back = QtWidgets.QAction("后视图", self)
        self.action_view_back.setShortcut("Ctrl+Num1")
        self.action_view_back.triggered.connect(lambda: self._set_camera_view("back"))
        camera_view_menu.addAction(self.action_view_back)
        
        self.action_view_right = QtWidgets.QAction("右视图", self)
        self.action_view_right.setShortcut("Num3")
        self.action_view_right.triggered.connect(lambda: self._set_camera_view("right"))
        camera_view_menu.addAction(self.action_view_right)
        
        self.action_view_left = QtWidgets.QAction("左视图", self)
        self.action_view_left.setShortcut("Ctrl+Num3")
        self.action_view_left.triggered.connect(lambda: self._set_camera_view("left"))
        camera_view_menu.addAction(self.action_view_left)
        
        self.action_view_top = QtWidgets.QAction("顶视图", self)
        self.action_view_top.setShortcut("Num7")
        self.action_view_top.triggered.connect(lambda: self._set_camera_view("top"))
        camera_view_menu.addAction(self.action_view_top)
        
        self.action_view_bottom = QtWidgets.QAction("底视图", self)
        self.action_view_bottom.setShortcut("Ctrl+Num7")
        self.action_view_bottom.triggered.connect(lambda: self._set_camera_view("bottom"))
        camera_view_menu.addAction(self.action_view_bottom)
        
        camera_view_menu.addSeparator()
        
        self.action_view_isometric = QtWidgets.QAction("等轴测视图", self)
        self.action_view_isometric.setShortcut("Num0")
        self.action_view_isometric.triggered.connect(lambda: self._set_camera_view("isometric"))
        camera_view_menu.addAction(self.action_view_isometric)
        
        self.action_view_reset = QtWidgets.QAction("重置视图", self)
        self.action_view_reset.setShortcut("Home")
        self.action_view_reset.triggered.connect(lambda: self._set_camera_view("reset"))
        camera_view_menu.addAction(self.action_view_reset)
        
        # 添加材质菜单
        material_menu = menubar.addMenu("材质")
        
        # 鞋底材质子菜单
        sole_material_menu = material_menu.addMenu("鞋底材质")
        self.sole_material_actions = []
        sole_materials = ["灰色", "白色", "黑色", "蓝色", "橙色"]
        sole_material_group = QtWidgets.QActionGroup(self)
        sole_material_group.setExclusive(True)
        for i, name in enumerate(sole_materials):
            action = QtWidgets.QAction(name, self)
            action.setCheckable(True)
            if i == 0:  # 默认选中灰色
                action.setChecked(True)
            action.triggered.connect(lambda checked, idx=i: self._on_sole_material_changed(idx))
            sole_material_group.addAction(action)
            sole_material_menu.addAction(action)
            self.sole_material_actions.append(action)
        
        # 晶格材质子菜单
        lattice_material_menu = material_menu.addMenu("晶格材质")
        self.lattice_material_actions = []
        lattice_materials = ["橙色", "黄色", "绿色", "红色", "蓝色"]
        lattice_material_group = QtWidgets.QActionGroup(self)
        lattice_material_group.setExclusive(True)
        for i, name in enumerate(lattice_materials):
            action = QtWidgets.QAction(name, self)
            action.setCheckable(True)
            if i == 0:  # 默认选中橙色
                action.setChecked(True)
            action.triggered.connect(lambda checked, idx=i: self._on_lattice_material_changed(idx))
            lattice_material_group.addAction(action)
            lattice_material_menu.addAction(action)
            self.lattice_material_actions.append(action)
        
        # 添加分隔符
        material_menu.addSeparator()
        
        # PBR渲染选项
        self.action_enable_pbr = QtWidgets.QAction("启用 PBR 渲染", self)
        self.action_enable_pbr.setCheckable(True)
        self.action_enable_pbr.setChecked(True)  # 默认启用PBR
        self.action_enable_pbr.setToolTip("基于物理的渲染（PBR）提供更真实的材质效果")
        self.action_enable_pbr.toggled.connect(self._on_pbr_toggled)
        material_menu.addAction(self.action_enable_pbr)
        
        # SSAO选项
        self.action_enable_ssao = QtWidgets.QAction("启用 SSAO (环境光遮蔽)", self)
        self.action_enable_ssao.setCheckable(True)
        self.action_enable_ssao.setChecked(False)  # 默认禁用SSAO（避免在特定距离产生过度阴影）
        self.action_enable_ssao.setToolTip("环境光遮蔽（SSAO）增加深度感和真实感，但可能在某些距离下产生过度阴影")
        self.action_enable_ssao.toggled.connect(self._on_ssao_toggled)
        material_menu.addAction(self.action_enable_ssao)
        
        # 添加视图设置菜单
        view_settings_menu = menubar.addMenu("视图设置")
        
        # 背景颜色子菜单
        bg_color_menu = view_settings_menu.addMenu("背景颜色")
        self.bg_color_actions = []
        bg_colors = ["深灰", "中灰", "浅灰", "白色", "黑色"]
        bg_color_group = QtWidgets.QActionGroup(self)
        bg_color_group.setExclusive(True)
        for i, name in enumerate(bg_colors):
            action = QtWidgets.QAction(name, self)
            action.setCheckable(True)
            if i == 0:  # 默认选中深灰
                action.setChecked(True)
            action.triggered.connect(lambda checked, idx=i: self._on_background_color_changed(idx))
            bg_color_group.addAction(action)
            bg_color_menu.addAction(action)
            self.bg_color_actions.append(action)
        
        # 添加渲染器菜单
        renderer_menu = menubar.addMenu("渲染器")
        renderer_menu.addAction(self.action_renderer_pyvista)
        renderer_menu.addAction(self.action_renderer_open3d)
        
        # 添加窗口菜单（最右侧）
        window_menu = menubar.addMenu("窗口")
        
        # 显示/隐藏运行日志
        self.action_toggle_log = QtWidgets.QAction("显示运行日志", self)
        self.action_toggle_log.setCheckable(True)
        self.action_toggle_log.setChecked(True)  # 默认显示
        self.action_toggle_log.setShortcut("Ctrl+L")
        self.action_toggle_log.setToolTip("显示或隐藏底部运行日志窗口")
        self.action_toggle_log.toggled.connect(self._on_toggle_log)
        window_menu.addAction(self.action_toggle_log)

        self.action_show_sole.triggered.connect(lambda: self._set_view_mode("sole"))
        self.action_show_lattice.triggered.connect(lambda: self._set_view_mode("lattice"))
        self.action_show_combined.triggered.connect(lambda: self._set_view_mode("combined"))
        self.action_toggle_mesh_edges.toggled.connect(lambda _: self._refresh_view())

    def _path_row(self, label: str, edit: QtWidgets.QLineEdit, browse_fn, preview_fn=None) -> QtWidgets.QWidget:
        wrapper = QtWidgets.QWidget(self)
        row = QtWidgets.QHBoxLayout(wrapper)
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(QtWidgets.QLabel(label))
        row.addWidget(edit, stretch=1)
        if preview_fn is not None:
            preview_button = QtWidgets.QPushButton("预览")
            preview_button.clicked.connect(preview_fn)
            row.addWidget(preview_button)
        button = QtWidgets.QPushButton("浏览")
        button.clicked.connect(browse_fn)
        row.addWidget(button)
        return wrapper

    def _field_row(self, label: str, widget: QtWidgets.QWidget) -> QtWidgets.QWidget:
        wrapper = QtWidgets.QWidget(self)
        row = QtWidgets.QHBoxLayout(wrapper)
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(QtWidgets.QLabel(label))
        row.addWidget(widget, stretch=1)
        return wrapper

    def _browse_sole(self) -> None:
        # 确定默认目录：如果当前路径为空，使用 resources/ 文件夹
        current_path = self.edit_sole_path.text().strip()
        if not current_path:
            default_dir = str(RESOURCES_DIR) if RESOURCES_DIR.exists() else ""
        else:
            default_dir = current_path
        
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "选择鞋底 STL", default_dir, "STL Files (*.stl)"
        )
        if path:
            self.edit_sole_path.setText(path)
            self._on_sole_path_changed()

    def _browse_lattice(self) -> None:
        # 确定默认目录：如果当前路径为空，使用 resources/ 文件夹
        current_path = self.edit_lattice_path.text().strip()
        if not current_path:
            default_dir = str(RESOURCES_DIR) if RESOURCES_DIR.exists() else ""
        else:
            default_dir = current_path
        
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "选择晶格 STL", default_dir, "STL Files (*.stl)"
        )
        if path:
            self.edit_lattice_path.setText(path)
            self._on_lattice_path_changed()
    
    def _on_slice_enabled_changed(self, state) -> None:
        """启用/禁用分割功能"""
        self.slicing_enabled = (state == QtCore.Qt.Checked)
        
        # 启用/禁用分割控制组
        if hasattr(self, 'slice_params_group'):
            self.slice_params_group.setEnabled(self.slicing_enabled)
        if hasattr(self, 'slice_display_group'):
            self.slice_display_group.setEnabled(self.slicing_enabled)
        if hasattr(self, 'slice_actions_group'):
            self.slice_actions_group.setEnabled(self.slicing_enabled)
        
        if self.slicing_enabled:
            # 初始化分割器
            if self.current_sole_mesh.vertices.size == 0:
                self._show_warning("提示", "请先加载鞋底模型")
                self.check_enable_slice.setChecked(False)
                return
            
            self.original_sole_mesh = self.current_sole_mesh.copy()
            self.slicer = MeshSlicer(self.original_sole_mesh)
            self._update_slice_slider_range()
            self._append_status("✓ 分割功能已启用")
            if is_cpp_available():
                self._append_status("✓ C++ 加速可用")
        else:
            # 恢复原始模型
            if self.original_sole_mesh.vertices.size > 0:
                self.current_sole_mesh = self.original_sole_mesh.copy()
                self._refresh_view()
            self.positive_mesh = None
            self.negative_mesh = None
            self.button_export_positive.setEnabled(False)
            self.button_export_negative.setEnabled(False)
            self.viewer.hide_plane()  # 隐藏平面
            self._append_status("分割功能已禁用")
    
    def _update_slice_slider_range(self) -> None:
        """更新滑块范围"""
        if self.slicer is None:
            return
        
        # 检查是否有旧UI的控件
        if not hasattr(self, 'slice_axis_combo'):
            return
        
        axis_map = {0: 'x', 1: 'y', 2: 'z'}
        axis = axis_map[self.slice_axis_combo.currentIndex()]
        
        min_val, max_val = self.slicer.get_axis_range(axis)
        self.slice_slider_min = min_val
        self.slice_slider_max = max_val
        
        # 如果正在加载工程，保持已设置的滑块值（不重置）
        if hasattr(self, '_loading_project') and self._loading_project:
            # 只更新标签，不改变滑块位置
            current_position = self._get_slice_position()
            self._update_slice_position_label(current_position)
        # 如果滑块已经有值（非默认值），保持当前位置
        elif hasattr(self, 'slice_position_slider') and self.slice_position_slider.value() != 500:
            # 保持当前滑块值，只更新标签
            current_position = self._get_slice_position()
            self._update_slice_position_label(current_position)
        else:
            # 首次初始化时设置为中心
            center = (min_val + max_val) / 2
            self._update_slice_position_label(center)
            if hasattr(self, 'slice_position_slider'):
                self.slice_position_slider.setValue(500)  # 中心位置
    
    def _on_slice_axis_changed(self) -> None:
        """分割轴改变"""
        if not hasattr(self, 'slice_axis_combo'):
            return
            
        if self.slicing_enabled and self.slicer:
            self._update_slice_slider_range()
        
        # 更新角度标签以反映当前轴（使用简洁的标签）
        axis_idx = self.slice_axis_combo.currentIndex()
        if axis_idx == 0:  # X 轴
            self.slice_angle1_label.setText("绕Y:")
            self.slice_angle2_label.setText("绕Z:")
            self.slice_angle1_spin.setToolTip("绕 Y 轴旋转分割平面")
            self.slice_angle2_spin.setToolTip("绕 Z 轴旋转分割平面")
        elif axis_idx == 1:  # Y 轴
            self.slice_angle1_label.setText("绕Z:")
            self.slice_angle2_label.setText("绕X:")
            self.slice_angle1_spin.setToolTip("绕 Z 轴旋转分割平面")
            self.slice_angle2_spin.setToolTip("绕 X 轴旋转分割平面")
        else:  # Z 轴
            self.slice_angle1_label.setText("绕X:")
            self.slice_angle2_label.setText("绕Y:")
            self.slice_angle1_spin.setToolTip("绕 X 轴旋转分割平面")
            self.slice_angle2_spin.setToolTip("绕 Y 轴旋转分割平面")
    
    def _on_slice_angle_changed(self) -> None:
        """角度改变"""
        if not self.slicing_enabled:
            return
        
        # 如果显示平面，更新平面显示
        if hasattr(self, 'check_show_plane') and self.check_show_plane.isChecked():
            self._on_show_plane_changed(QtCore.Qt.Checked)
    
    def _on_slice_position_changed(self) -> None:
        """位置滑块改变"""
        if not self.slicing_enabled:
            return
        
        slider_value = self.slice_position_slider.value()
        position = self.slice_slider_min + (slider_value / 1000.0) * (self.slice_slider_max - self.slice_slider_min)
        self._update_slice_position_label(position)
        
        # 如果显示平面，更新平面位置
        if hasattr(self, 'check_show_plane') and self.check_show_plane.isChecked():
            self._on_show_plane_changed(QtCore.Qt.Checked)
    
    def _update_slice_position_label(self, position: float) -> None:
        """更新位置标签"""
        if not hasattr(self, 'slice_axis_combo') or not hasattr(self, 'slice_position_label'):
            return
            
        axis_names = {0: 'X', 1: 'Y', 2: 'Z'}
        axis_name = axis_names[self.slice_axis_combo.currentIndex()]
        self.slice_position_label.setText(f"位置: {position:.2f} mm ({axis_name} 轴)")
    
    def _get_slice_position(self) -> float:
        """获取当前分割位置"""
        if not hasattr(self, 'slice_position_slider'):
            return 0.0
            
        slider_value = self.slice_position_slider.value()
        return self.slice_slider_min + (slider_value / 1000.0) * (self.slice_slider_max - self.slice_slider_min)
    
    def _perform_slice(self) -> None:
        """执行分割"""
        if not self.slicing_enabled or self.slicer is None:
            return
        
        if not hasattr(self, 'slice_axis_combo'):
            return
        
        try:
            axis_map = {0: 'x', 1: 'y', 2: 'z'}
            axis = axis_map[self.slice_axis_combo.currentIndex()]
            position = self._get_slice_position()
            subdivide_level = self.slice_subdivide_spin.value()
            angle1 = self.slice_angle1_spin.value()
            angle2 = self.slice_angle2_spin.value()
            
            angle_info = ""
            if angle1 != 0 or angle2 != 0:
                angle_info = f", 角度 = ({angle1:.1f}°, {angle2:.1f}°)"
            
            self._append_status(f"正在分割: {axis.upper()} 轴, 位置 = {position:.2f} mm{angle_info}, 细化 = {subdivide_level} 次")
            QtWidgets.QApplication.processEvents()
            
            # 执行分割
            self.positive_mesh, self.negative_mesh = self.slicer.slice_by_plane(
                axis, position, subdivide=subdivide_level, angle1=angle1, angle2=angle2
            )
            
            self._append_status(f"✓ 分割完成: 正侧 {len(self.positive_mesh.faces)} 面, 负侧 {len(self.negative_mesh.faces)} 面")
            
            # 清理晶格显示，避免叠加
            self.current_lattice_mesh = trimesh.Trimesh()
            self.current_combined_mesh = trimesh.Trimesh()
            self.has_generated_result = False
            
            # 启用导出按钮
            self.button_export_positive.setEnabled(True)
            self.button_export_negative.setEnabled(True)
            
            # 更新显示
            self._on_slice_display_changed()
            
        except Exception as e:
            self._show_critical("错误", f"分割失败:\n{e}")
            self._append_status(f"✗ 分割失败: {e}")
            import traceback
            traceback.print_exc()
    
    def _reset_slice(self) -> None:
        """重置分割"""
        if self.original_sole_mesh.vertices.size > 0:
            self.current_sole_mesh = self.original_sole_mesh.copy()
            self.positive_mesh = None
            self.negative_mesh = None
            
            # 清理晶格显示
            self.current_lattice_mesh = trimesh.Trimesh()
            self.current_combined_mesh = trimesh.Trimesh()
            self.has_generated_result = False
            
            self.button_export_positive.setEnabled(False)
            self.button_export_negative.setEnabled(False)
            self.slice_display_combo.setCurrentIndex(0)
            self._refresh_view()
            self._append_status("已重置分割")
    
    def _on_slice_display_changed(self) -> None:
        """显示模式改变"""
        if not self.slicing_enabled:
            return
        
        display_mode = self.slice_display_combo.currentIndex()
        spacing = self.slice_spacing_spin.value() if hasattr(self, 'slice_spacing_spin') else 0.0
        
        # 检查是否需要上色（通过按钮颜色判断是否使用自定义颜色）
        use_custom_colors = True  # 默认使用自定义颜色
        
        if display_mode == 0:  # 原始
            self.current_sole_mesh = self.original_sole_mesh.copy()
            self._refresh_view()
        elif display_mode == 1:  # 分割结果（显示两部分，带颜色）
            if self.positive_mesh and self.negative_mesh:
                meshes = []
                show_edges = self.action_toggle_mesh_edges.isChecked()
                
                # 获取分割轴
                axis_idx = self.slice_axis_combo.currentIndex()
                
                # 复制网格
                pos_mesh = self.positive_mesh.copy()
                neg_mesh = self.negative_mesh.copy()
                
                # 应用间距
                if spacing > 0:
                    pos_offset = np.zeros(3)
                    pos_offset[axis_idx] = spacing / 2
                    pos_mesh.vertices += pos_offset
                    
                    neg_offset = np.zeros(3)
                    neg_offset[axis_idx] = -spacing / 2
                    neg_mesh.vertices += neg_offset
                
                # 转换颜色为 0-1 范围
                pos_color = (
                    self.positive_color.redF(),
                    self.positive_color.greenF(),
                    self.positive_color.blueF(),
                    1.0
                )
                neg_color = (
                    self.negative_color.redF(),
                    self.negative_color.greenF(),
                    self.negative_color.blueF(),
                    1.0
                )
                
                # 添加正侧网格
                meshes.append(MeshData(
                    np.asarray(pos_mesh.vertices),
                    np.asarray(pos_mesh.faces),
                    color=pos_color
                ))
                if show_edges:
                    meshes.append(MeshData(
                        np.asarray(pos_mesh.vertices),
                        np.asarray(pos_mesh.faces),
                        color=(0.2, 0.2, 0.2, 1.0),
                        render_mode="wireframe",
                        unlit=True,
                        line_width=1.0
                    ))
                
                # 添加负侧网格
                meshes.append(MeshData(
                    np.asarray(neg_mesh.vertices),
                    np.asarray(neg_mesh.faces),
                    color=neg_color
                ))
                if show_edges:
                    meshes.append(MeshData(
                        np.asarray(neg_mesh.vertices),
                        np.asarray(neg_mesh.faces),
                        color=(0.2, 0.2, 0.2, 1.0),
                        render_mode="wireframe",
                        unlit=True,
                        line_width=1.0
                    ))
                
                # 直接设置网格，不通过 _mesh_layers
                self.viewer.set_meshes(meshes, reset_view=False)
                return  # 直接返回，不调用 _refresh_view
        elif display_mode == 2:  # 仅正侧
            if self.positive_mesh:
                meshes = []
                show_edges = self.action_toggle_mesh_edges.isChecked()
                
                pos_color = (
                    self.positive_color.redF(),
                    self.positive_color.greenF(),
                    self.positive_color.blueF(),
                    1.0
                )
                
                meshes.append(MeshData(
                    np.asarray(self.positive_mesh.vertices),
                    np.asarray(self.positive_mesh.faces),
                    color=pos_color
                ))
                if show_edges:
                    meshes.append(MeshData(
                        np.asarray(self.positive_mesh.vertices),
                        np.asarray(self.positive_mesh.faces),
                        color=(0.2, 0.2, 0.2, 1.0),
                        render_mode="wireframe",
                        unlit=True,
                        line_width=1.0
                    ))
                
                self.viewer.set_meshes(meshes, reset_view=False)
                return
        elif display_mode == 3:  # 仅负侧
            if self.negative_mesh:
                meshes = []
                show_edges = self.action_toggle_mesh_edges.isChecked()
                
                neg_color = (
                    self.negative_color.redF(),
                    self.negative_color.greenF(),
                    self.negative_color.blueF(),
                    1.0
                )
                
                meshes.append(MeshData(
                    np.asarray(self.negative_mesh.vertices),
                    np.asarray(self.negative_mesh.faces),
                    color=neg_color
                ))
                if show_edges:
                    meshes.append(MeshData(
                        np.asarray(self.negative_mesh.vertices),
                        np.asarray(self.negative_mesh.faces),
                        color=(0.2, 0.2, 0.2, 1.0),
                        render_mode="wireframe",
                        unlit=True,
                        line_width=1.0
                    ))
                
                self.viewer.set_meshes(meshes, reset_view=False)
                return
        
        self._refresh_view()
    
    def _export_sliced_mesh(self, which: str) -> None:
        """导出分割后的网格"""
        if which == 'positive' and self.positive_mesh is None:
            return
        if which == 'negative' and self.negative_mesh is None:
            return
        
        suffix = 'positive' if which == 'positive' else 'negative'
        default_name = f"sole_{suffix}.stl"
        
        # 默认保存到 results/ 文件夹
        if RESULTS_DIR.exists():
            default_path = str(RESULTS_DIR / default_name)
        else:
            default_path = default_name
        
        file_path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            f"导出{suffix}部分",
            default_path,
            "STL Files (*.stl)"
        )
        
        if file_path:
            try:
                mesh = self.positive_mesh if which == 'positive' else self.negative_mesh
                mesh.export(file_path)
                self._append_status(f"✓ 导出成功: {Path(file_path).name}")
                self._show_information("导出成功", f"已导出: {Path(file_path).name}")
            except Exception as e:
                self._show_critical("错误", f"导出失败:\n{e}")
                self._append_status(f"✗ 导出失败: {e}")
    
    def _on_show_plane_changed(self, state) -> None:
        """显示/隐藏分割平面"""
        if not self.slicing_enabled or self.slicer is None:
            return
        
        if not hasattr(self, 'slice_axis_combo') or not hasattr(self, 'slice_angle1_spin') or not hasattr(self, 'slice_angle2_spin'):
            return
        
        show_plane = (state == QtCore.Qt.Checked)
        
        if show_plane:
            # 显示分割平面
            position = self._get_slice_position()
            axis_idx = self.slice_axis_combo.currentIndex()
            angle1 = self.slice_angle1_spin.value()
            angle2 = self.slice_angle2_spin.value()
            
            # 创建平面原点 - 必须与 mesh_slicer.py 中的逻辑一致
            # 使用模型包围盒的中心，只在分割轴上使用指定位置
            point = self.original_sole_mesh.bounds.mean(axis=0).astype(np.float32)
            point[axis_idx] = position
            
            # 创建初始法向量（垂直于分割轴）
            normal = np.zeros(3, dtype=np.float32)
            normal[axis_idx] = 1.0
            
            # 如果有角度旋转，应用旋转变换
            if angle1 != 0.0 or angle2 != 0.0:
                # 确定旋转轴的顺序 - 必须与 mesh_slicer.py 一致
                if axis_idx == 0:  # X 轴
                    rot_axis1, rot_axis2 = 1, 2  # Y, Z
                elif axis_idx == 1:  # Y 轴
                    rot_axis1, rot_axis2 = 0, 2  # X, Z
                else:  # Z 轴
                    rot_axis1, rot_axis2 = 0, 1  # X, Y
                
                # 应用第一个旋转
                if angle1 != 0.0:
                    angle1_rad = np.radians(angle1)
                    rotation_matrix1 = self._get_rotation_matrix(rot_axis1, angle1_rad)
                    normal = rotation_matrix1 @ normal
                
                # 应用第二个旋转
                if angle2 != 0.0:
                    angle2_rad = np.radians(angle2)
                    rotation_matrix2 = self._get_rotation_matrix(rot_axis2, angle2_rad)
                    normal = rotation_matrix2 @ normal
                
                # 归一化法向量
                normal = normal / np.linalg.norm(normal)
            
            # 计算平面大小（增大到1.5倍，使平面更明显）
            extents = self.original_sole_mesh.extents
            plane_size = float(np.max(extents)) * 1.5
            
            self.viewer.set_plane(point, normal, plane_size)
        else:
            self.viewer.hide_plane()
    
    def _get_rotation_matrix(self, axis_idx: int, angle_rad: float) -> np.ndarray:
        """
        获取绕指定轴的旋转矩阵
        
        参数:
            axis_idx: 旋转轴索引 (0=X, 1=Y, 2=Z)
            angle_rad: 旋转角度（弧度）
        
        返回:
            3x3 旋转矩阵
        """
        c = np.cos(angle_rad)
        s = np.sin(angle_rad)
        
        if axis_idx == 0:  # 绕 X 轴
            return np.array([
                [1, 0, 0],
                [0, c, -s],
                [0, s, c]
            ], dtype=np.float32)
        elif axis_idx == 1:  # 绕 Y 轴
            return np.array([
                [c, 0, s],
                [0, 1, 0],
                [-s, 0, c]
            ], dtype=np.float32)
        else:  # 绕 Z 轴
            return np.array([
                [c, -s, 0],
                [s, c, 0],
                [0, 0, 1]
            ], dtype=np.float32)
    
    def _on_slice_lighting_changed(self, state) -> None:
        """启用/禁用光照"""
        # 重新绘制视图
        self._refresh_view()
    
    def _choose_color(self, which: str) -> None:
        """选择颜色"""
        if which == 'positive':
            color = QtWidgets.QColorDialog.getColor(self.positive_color, self, "选择正侧颜色")
            if color.isValid():
                self.positive_color = color
                self.button_positive_color.setStyleSheet(f"background-color: {color.name()}; border: 1px solid #999;")
                self._refresh_view()
        elif which == 'negative':
            color = QtWidgets.QColorDialog.getColor(self.negative_color, self, "选择负侧颜色")
            if color.isValid():
                self.negative_color = color
                self.button_negative_color.setStyleSheet(f"background-color: {color.name()}; border: 1px solid #999;")
                self._refresh_view()
    
    def _on_slice_spacing_changed(self) -> None:
        """区域间距改变"""
        if self.slicing_enabled and self.positive_mesh and self.negative_mesh:
            self._on_slice_display_changed()

    def _browse_output(self) -> None:
        # 确定默认目录：如果当前路径为空，使用 results/ 文件夹
        current_path = self.edit_output_path.text().strip()
        if not current_path:
            default_dir = str(RESULTS_DIR) if RESULTS_DIR.exists() else ""
        else:
            default_dir = current_path
        
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "选择导出路径", default_dir, "STL Files (*.stl)"
        )
        if path:
            if not path.lower().endswith(".stl"):
                path += ".stl"
            self.edit_output_path.setText(path)

    def _on_sole_path_changed(self) -> None:
        self._apply_recommended_params(reason="鞋底模型已更新")
        self._preview_selected_sole(reset_view_mode=True)

    def _on_lattice_path_changed(self) -> None:
        self._apply_recommended_params(reason="晶格模型已更新")
        self._preview_selected_lattice(reset_view_mode=True)

    def _preview_selected_sole(self, log_message: bool = True, reset_view_mode: bool = True) -> None:
        sole_path = self.edit_sole_path.text().strip()
        if not sole_path or not Path(sole_path).exists():
            if log_message:
                self._append_status("鞋底预览失败: 文件路径无效或文件不存在")
            return
        try:
            sole_mesh = geometry.load_mesh(sole_path)
            if sole_mesh.faces.size == 0:
                if log_message:
                    self._append_status("鞋底预览失败: 网格为空")
                return
        except Exception as e:
            if log_message:
                self._append_status(f"鞋底预览失败: {type(e).__name__}: {e}")
            return
        
        # 更新当前鞋底网格
        self.current_sole_mesh = sole_mesh
        self.current_combined_mesh = sole_mesh.copy()
        
        # 同时更新原始鞋底网格（用于分割和视图切换）
        self.original_sole_mesh = sole_mesh.copy()
        
        # 如果分割功能已启用，重新初始化分割器
        if self.slicing_enabled and hasattr(self, 'slicer'):
            self.slicer = MeshSlicer(self.original_sole_mesh)
            self._update_slice_slider_range()
        
        self._show_preview_meshes(f"sole::{Path(sole_path).resolve()}", reset_view_mode, "sole")
        if log_message:
            self._append_status(f"已预览鞋底模型: {Path(sole_path).name} (顶点: {len(sole_mesh.vertices)}, 面: {len(sole_mesh.faces)})")

    def _preview_selected_lattice(
        self,
        log_message: bool = True,
        reset_view_mode: bool = True,
        update_main_view: bool = True,
        inset_visible: bool = False,
    ) -> None:
        lattice_path = self.edit_lattice_path.text().strip()
        if not lattice_path or not Path(lattice_path).exists():
            return
        try:
            lattice_mesh = geometry.load_mesh(lattice_path)
        except Exception as e:
            if log_message:
                self._append_status(f"晶格预览失败: {type(e).__name__}: {e}")
            return
        self.selected_lattice_unit_mesh = lattice_mesh.copy()
        self.current_lattice_mesh = lattice_mesh
        if update_main_view:
            self._show_preview_meshes(f"lattice::{Path(lattice_path).resolve()}", reset_view_mode, "lattice")
        if log_message:
            self._append_status(f"已预览晶格模型: {Path(lattice_path).name}")

    def _on_method_changed(self) -> None:
        self._sync_method_fields()
        self._apply_recommended_params(reason="生成方式已切换")
    
    def _on_tile_decimate_changed(self, value: int) -> None:
        """Tile减面参数变化时更新内部变量"""
        self.lattice_unit_target_faces = value

    def _sync_method_fields(self) -> None:
        """同步方法字段 - 根据选择的方法显示对应的参数页"""
        if not hasattr(self, 'params_stack'):
            return
        
        method = self.combo_method.currentText().strip().lower()
        
        # 根据方法切换到对应的参数页
        method_to_index = {
            'gyroid': 0,
            'voronoi_2_5d': 1,
            'voronoi_implicit': 2,
            'tile_unit': 3
        }
        
        index = method_to_index.get(method, 0)
        self.params_stack.setCurrentIndex(index)

    def _append_status(self, text: str) -> None:
        """添加状态日志，确保文本正确显示和滚动"""
        self.status_box.appendPlainText(text)
        
        # 强制刷新文本框
        self.status_box.update()
        
        # 确保滚动到底部
        cursor = self.status_box.textCursor()
        cursor.movePosition(QtGui.QTextCursor.End)
        self.status_box.setTextCursor(cursor)
        self.status_box.ensureCursorVisible()
        
        # 处理待处理的事件，确保UI及时更新
        QtWidgets.QApplication.processEvents()

    def _set_mouse_mode(self, mode: str) -> None:
        """设置鼠标模式"""
        self._mouse_mode = "pan" if mode == "pan" else "rotate"
        self.viewer.set_mouse_mode(self._mouse_mode)
    
    def _on_sole_material_changed(self, index: int) -> None:
        """鞋底材质改变时的回调"""
        self.current_sole_material = index
        material_names = ["灰色", "白色", "黑色", "蓝色", "橙色"]
        self._append_status(f"鞋底材质已切换为: {material_names[index]}")
        self._refresh_view()
    
    def _on_lattice_material_changed(self, index: int) -> None:
        """晶格材质改变时的回调"""
        self.current_lattice_material = index
        material_names = ["橙色", "黄色", "绿色", "红色", "蓝色"]
        self._append_status(f"晶格材质已切换为: {material_names[index]}")
        self._refresh_view()
    
    def _on_background_color_changed(self, index: int) -> None:
        """背景颜色改变时的回调"""
        # 定义背景颜色映射
        bg_colors = {
            0: '#3a3a3a',  # 深灰（默认）
            1: '#5a5a5a',  # 中灰
            2: '#8a8a8a',  # 浅灰
            3: '#ffffff',  # 白色
            4: '#1a1a1a',  # 黑色
        }
        color_names = ["深灰", "中灰", "浅灰", "白色", "黑色"]
        
        color_hex = bg_colors.get(index, '#3a3a3a')
        
        # 只对 PyVista 渲染器设置背景颜色
        if self.current_renderer_type == "pyvista" and hasattr(self.viewer, 'set_background'):
            self.viewer.set_background(color_hex)
            self._append_status(f"背景颜色已切换为: {color_names[index]}")
        
        # 保存当前选择的索引
        self._current_bg_color_index = index
    
    def _on_pbr_toggled(self, checked: bool) -> None:
        """PBR渲染切换"""
        if self.current_renderer_type == "pyvista":
            if hasattr(self.viewer, 'enable_pbr'):
                self.viewer.enable_pbr = checked
                self._refresh_view()
                status = "启用" if checked else "禁用"
                self._append_status(f"已{status} PBR 渲染")
        else:
            self._append_status("PBR 渲染仅支持 PyVista 渲染器")
    
    def _on_ssao_toggled(self, checked: bool) -> None:
        """SSAO切换"""
        if self.current_renderer_type == "pyvista":
            if hasattr(self.viewer, 'enable_ssao_flag'):
                old_value = self.viewer.enable_ssao_flag
                self.viewer.enable_ssao_flag = checked
                
                if old_value != checked:
                    status = "启用" if checked else "禁用"
                    
                    # PyVista 的 SSAO 是渲染器级别的设置，无法动态切换
                    # 需要重新加载模型才能生效
                    self._append_status(f"SSAO 设置已更改为: {status}")
                    self._append_status("⚠️ 请重新加载模型以应用 SSAO 设置")
                    
                    # 提示用户：如果是启用 SSAO，需要重新加载
                    # 如果是禁用 SSAO，也需要重新加载才能移除效果
                    if checked:
                        self._append_status("提示: 重新加载模型后，SSAO 将会启用")
                    else:
                        self._append_status("提示: 重新加载模型后，SSAO 将会禁用")
        else:
            self._append_status("SSAO 仅支持 PyVista 渲染器")
    
    def _on_ground_platform_toggled(self, checked: bool) -> None:
        """网格平台显示切换"""
        if self.current_renderer_type == "pyvista":
            if hasattr(self.viewer, 'show_ground_plane'):
                self.viewer.show_ground_plane = checked
                self.viewer.show_ground_grid = checked
                self._refresh_view()
                status = "显示" if checked else "隐藏"
                self._append_status(f"已{status}网格平台")
    
    def _set_camera_view(self, view_type: str) -> None:
        """设置相机视角"""
        if self.current_renderer_type != "pyvista":
            self._append_status("相机视角切换仅支持 PyVista 渲染器")
            return
        
        view_names = {
            "front": "前视图",
            "back": "后视图",
            "left": "左视图",
            "right": "右视图",
            "top": "顶视图",
            "bottom": "底视图",
            "isometric": "等轴测视图",
            "reset": "重置视图"
        }
        
        try:
            if view_type == "front":
                self.viewer.set_view_front()
            elif view_type == "back":
                self.viewer.set_view_back()
            elif view_type == "left":
                self.viewer.set_view_left()
            elif view_type == "right":
                self.viewer.set_view_right()
            elif view_type == "top":
                self.viewer.set_view_top()
            elif view_type == "bottom":
                self.viewer.set_view_bottom()
            elif view_type == "isometric":
                self.viewer.set_view_isometric()
            elif view_type == "reset":
                self.viewer.reset_view()
            
            view_name = view_names.get(view_type, view_type)
            self._append_status(f"已切换到{view_name}")
            
        except Exception as e:
            self._append_status(f"切换视角失败: {str(e)}")
    
    def _switch_to_pyvista(self):
        """切换到 PyVista 渲染器"""
        if self.current_renderer_type == "pyvista":
            return
        
        self._append_status("切换到 PyVista 渲染器...")
        
        # 保存当前网格数据
        current_meshes = self.viewer.meshes if hasattr(self.viewer, 'meshes') else []
        
        # 移除旧的渲染器
        viewer_host_layout = self.viewer_host.layout()
        if self.viewer:
            viewer_host_layout.removeWidget(self.viewer)
            self.viewer.deleteLater()
        
        # 创建新的 PyVista 渲染器
        from ui.viewers.pyvista_viewer import PyVistaRenderer
        self.viewer = PyVistaRenderer(self.viewer_host)
        viewer_host_layout.insertWidget(0, self.viewer)
        
        # 应用当前背景颜色设置
        bg_colors = {
            0: '#3a3a3a',  # 深灰
            1: '#5a5a5a',  # 中灰
            2: '#8a8a8a',  # 浅灰
            3: '#ffffff',  # 白色
            4: '#1a1a1a',  # 黑色
        }
        current_bg_index = getattr(self, '_current_bg_color_index', 0)
        color_hex = bg_colors.get(current_bg_index, '#3a3a3a')
        self.viewer.set_background(color_hex)
        
        # 同步菜单选中状态
        if hasattr(self, 'bg_color_actions') and 0 <= current_bg_index < len(self.bg_color_actions):
            self.bg_color_actions[current_bg_index].setChecked(True)
        
        # 恢复网格数据
        if current_meshes:
            self.viewer.set_meshes(current_meshes, reset_view=True)
        
        self.current_renderer_type = "pyvista"
        self._append_status("✓ 已切换到 PyVista 渲染器")
    
    def _switch_to_open3d(self):
        """切换到 Open3D 渲染器"""
        if not OPEN3D_AVAILABLE:
            self._show_warning("Open3D 未安装", "Open3D 渲染器需要安装 Open3D 库。\n\n"
                "请运行以下命令安装：\n"
                "pip install open3d\n\n"
                "安装后重启程序即可使用。"
            )
            # 恢复到 PyVista
            self.action_renderer_pyvista.setChecked(True)
            return
        
        if self.current_renderer_type == "open3d":
            return
        
        self._append_status("切换到 Open3D 渲染器...")
        
        # 保存当前网格数据
        current_meshes = self.viewer.meshes if hasattr(self.viewer, 'meshes') else []
        
        # 移除旧的渲染器
        viewer_host_layout = self.viewer_host.layout()
        if self.viewer:
            viewer_host_layout.removeWidget(self.viewer)
            self.viewer.deleteLater()
        
        # 创建新的 Open3D 渲染器
        self.viewer = Open3DRenderer(self.viewer_host)
        viewer_host_layout.insertWidget(0, self.viewer)
        
        # 恢复网格数据
        if current_meshes:
            self.viewer.set_meshes(current_meshes, reset_view=True)
        
        self.current_renderer_type = "open3d"
        self._append_status("✓ 已切换到 Open3D 渲染器（嵌入式）")

    def _clear_cache(self) -> None:
        cache_dirs = [
            Path(GYROID_MASK_CACHE_DIR),
            Path(GYROID_RESULT_CACHE_DIR),
            Path(TILE_RESULT_CACHE_DIR),
        ]
        unique_dirs: list[Path] = []
        seen: set[str] = set()
        for cache_dir in cache_dirs:
            key = str(cache_dir.resolve()) if cache_dir.exists() else str(cache_dir)
            if key not in seen:
                seen.add(key)
                unique_dirs.append(cache_dir)

        msgbox = QtWidgets.QMessageBox(self)
        msgbox.setWindowTitle("清空缓存")
        msgbox.setText("确定要清空当前项目的缓存文件吗？\n这会删除 .cache 目录下已有的结果缓存。")
        msgbox.setStandardButtons(QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No)
        msgbox.setDefaultButton(QtWidgets.QMessageBox.No)
        msgbox.setIcon(QtWidgets.QMessageBox.Question)
        self._apply_dark_messagebox_style(msgbox)
        
        reply = msgbox.exec_()
        if reply != QtWidgets.QMessageBox.Yes:
            return

        deleted_files = 0
        removed_dirs = 0
        for cache_dir in unique_dirs:
            if cache_dir.exists():
                for child in cache_dir.iterdir():
                    try:
                        if child.is_file() or child.is_symlink():
                            child.unlink()
                            deleted_files += 1
                        elif child.is_dir():
                            shutil.rmtree(child)
                            removed_dirs += 1
                    except Exception as e:
                        self._append_status(f"清理缓存失败: {child} ({type(e).__name__}: {e})")

        summary = f"已清空缓存：删除文件 {deleted_files} 个，目录 {removed_dirs} 个"
        self._append_status(summary)
        self._show_information("清空缓存", summary)
        
        # 强制垃圾回收，释放内存
        import gc
        gc.collect()
        self._append_status("已释放内存")

    def _remember_current_preview_view(self) -> None:
        if self._current_preview_key:
            self._preview_view_states[self._current_preview_key] = self.viewer.get_view_state()

    def _show_preview_meshes(self, preview_key: str, reset_view_mode: bool, target_mode: str) -> None:
        self._remember_current_preview_view()
        self._current_preview_key = preview_key
        meshes = self._mesh_layers()
        saved = self._preview_view_states.get(preview_key)
        if reset_view_mode or self.current_view_mode not in {"sole", "lattice", "combined"}:
            self.current_view_mode = target_mode
            meshes = self._mesh_layers()
        if saved is not None:
            self.viewer.set_meshes(meshes, reset_view=False)
            self.viewer.apply_view_state(saved)
        else:
            self.viewer.set_meshes(meshes, reset_view=True)

    def _load_mesh_extents(self, path_text: str) -> np.ndarray | None:
        path = Path(path_text)
        if not path.exists():
            return None
        try:
            mesh = geometry.load_mesh(str(path))
            extents = np.asarray(mesh.extents, dtype=np.float64)
            if extents.shape != (3,) or not np.isfinite(extents).all():
                return None
            return np.maximum(extents, 1e-6)
        except Exception:
            return None

    def _recommended_gyroid_params(self, sole_extents: np.ndarray | None) -> dict[str, float | int]:
        if sole_extents is None:
            return {"gyroid_cell": float(GYROID_CELL_SIZE), "gyroid_iso": float(GYROID_ISOVALUE), "gyroid_res": int(GYROID_RESOLUTION)}
        thickness = float(np.min(sole_extents))
        cell = float(np.clip(thickness * 0.55, 3.0, 7.5))
        iso = 0.26 if thickness < 10.0 else 0.32 if thickness < 16.0 else 0.38
        res = int(np.clip(round(140.0 / max(cell, 1e-6)), 20, 40))
        return {"gyroid_cell": cell, "gyroid_iso": iso, "gyroid_res": res}

    def _recommended_voronoi_params(self, sole_extents: np.ndarray | None) -> dict[str, float | int]:
        if sole_extents is None:
            return {"voronoi_cell": float(VORONOI_CELL_SIZE), "voronoi_thickness": float(VORONOI_STRUT_THICKNESS), "voronoi_layers": int(VORONOI_Z_LAYERS)}
        thickness = float(np.min(sole_extents))
        cell = float(np.clip(thickness * 0.60, 3.5, 8.0))
        strut = float(np.clip(cell * 0.13, 0.45, 1.10))
        layers = int(np.clip(round(thickness / 1.8), 3, 8))
        return {"voronoi_cell": cell, "voronoi_thickness": strut, "voronoi_layers": layers}

    def _recommended_tile_params(self, sole_extents: np.ndarray | None, lattice_extents: np.ndarray | None) -> dict[str, float | int]:
        if sole_extents is None or lattice_extents is None:
            return {"tile_shrink": float(LATTICE_TILE_SHRINK), "tile_spacing": float(TILE_SPACING_FACTOR), "tile_margin": float(TILE_BOUNDARY_MARGIN)}
        sole_xy = np.sort(np.asarray(sole_extents[:2], dtype=np.float64))
        lattice_xy = np.sort(np.asarray(lattice_extents[:2], dtype=np.float64))
        sole_width = float(sole_xy[0])
        sole_thickness = float(np.min(sole_extents))
        unit_width = float(max(lattice_xy[0], 1e-6))
        target_unit_width = float(np.clip(sole_width / 7.0, 6.0, 22.0))
        shrink = float(np.clip(target_unit_width / unit_width, 0.12, 1.10))
        spacing = 0.98 if shrink <= 0.22 else 0.96 if shrink <= 0.35 else 0.92
        margin = float(np.clip(sole_thickness * 0.08, 0.6, 1.4))
        return {"tile_shrink": shrink, "tile_spacing": spacing, "tile_margin": margin}

    def _recommended_voronoi_implicit_params(self, sole_extents: np.ndarray | None) -> dict[str, float | int]:
        """根据鞋底尺寸推荐 Voronoi Implicit 参数"""
        if sole_extents is None:
            return {
                "voronoi_implicit_cell": float(VORONOI_IMPLICIT_CELL_SIZE),
                "voronoi_implicit_wall": float(VORONOI_IMPLICIT_WALL_THICKNESS),
                "voronoi_implicit_res": int(VORONOI_IMPLICIT_RESOLUTION),
                "voronoi_implicit_density": float(VORONOI_IMPLICIT_SEED_DENSITY_FACTOR)
            }
        # 根据鞋底尺寸推荐参数
        sole_width = float(np.min(sole_extents[:2]))
        cell = float(np.clip(sole_width / 10.0, 8.0, 25.0))
        wall = float(np.clip(cell * 0.1, 0.5, 2.0))
        res = 30
        density = 1.0
        return {
            "voronoi_implicit_cell": cell,
            "voronoi_implicit_wall": wall,
            "voronoi_implicit_res": res,
            "voronoi_implicit_density": density
        }

    def _apply_recommended_params(self, show_message: bool = True, reason: str = "", force: bool = False) -> None:
        if self._suspend_auto_recommend:
            return
        if not force and hasattr(self, "check_auto_recommend") and not self.check_auto_recommend.isChecked():
            return
        method = self.combo_method.currentText().strip().lower()
        sole_extents = self._load_mesh_extents(self.edit_sole_path.text().strip())
        lattice_extents = self._load_mesh_extents(self.edit_lattice_path.text().strip())

        self._suspend_auto_recommend = True
        try:
            if method == "gyroid":
                params = self._recommended_gyroid_params(sole_extents)
                self.spin_gyroid_cell.setValue(float(params["gyroid_cell"]))
                self.spin_gyroid_iso.setValue(float(params["gyroid_iso"]))
                self.spin_gyroid_res.setValue(int(params["gyroid_res"]))
            elif method == "voronoi_2_5d":
                params = self._recommended_voronoi_params(sole_extents)
                self.spin_voronoi_cell.setValue(float(params["voronoi_cell"]))
                self.spin_voronoi_thickness.setValue(float(params["voronoi_thickness"]))
                self.spin_voronoi_layers.setValue(int(params["voronoi_layers"]))
            elif method == "voronoi_implicit":
                params = self._recommended_voronoi_implicit_params(sole_extents)
                self.spin_voronoi_implicit_cell.setValue(float(params["voronoi_implicit_cell"]))
                self.spin_voronoi_implicit_wall.setValue(float(params["voronoi_implicit_wall"]))
                self.spin_voronoi_implicit_res.setValue(int(params["voronoi_implicit_res"]))
                self.spin_voronoi_implicit_density.setValue(float(params["voronoi_implicit_density"]))
            elif method == "tile_unit":
                params = self._recommended_tile_params(sole_extents, lattice_extents)
                self.spin_tile_shrink.setValue(float(params["tile_shrink"]))
                self.spin_tile_spacing.setValue(float(params["tile_spacing"]))
                self.spin_tile_margin.setValue(float(params["tile_margin"]))
        finally:
            self._suspend_auto_recommend = False

        if show_message and reason:
            self._append_status(f"[推荐参数] {reason}，已更新 {method} 的推荐初始值")

    def _apply_default_params(self) -> None:
        method = self.combo_method.currentText().strip().lower()
        self._suspend_auto_recommend = True
        try:
            if method == "gyroid":
                self.spin_gyroid_cell.setValue(float(GYROID_CELL_SIZE))
                self.spin_gyroid_iso.setValue(float(GYROID_ISOVALUE))
                self.spin_gyroid_res.setValue(int(GYROID_RESOLUTION))
            elif method == "voronoi_2_5d":
                self.spin_voronoi_cell.setValue(float(VORONOI_CELL_SIZE))
                self.spin_voronoi_thickness.setValue(float(VORONOI_STRUT_THICKNESS))
                self.spin_voronoi_layers.setValue(int(VORONOI_Z_LAYERS))
            elif method == "tile_unit":
                self.spin_tile_shrink.setValue(float(LATTICE_TILE_SHRINK))
                self.spin_tile_spacing.setValue(float(TILE_SPACING_FACTOR))
                self.spin_tile_margin.setValue(float(TILE_BOUNDARY_MARGIN))
        finally:
            self._suspend_auto_recommend = False
        self._append_status(f"[默认参数] 已恢复 {method} 的 config 默认值")

    def _apply_runtime_settings(self) -> None:
        geometry.LATTICE_METHOD = self.combo_method.currentText().strip().lower()
        geometry.GYROID_CELL_SIZE = float(self.spin_gyroid_cell.value())
        geometry.GYROID_ISOVALUE = float(self.spin_gyroid_iso.value())
        geometry.GYROID_RESOLUTION = int(self.spin_gyroid_res.value())
        geometry.VORONOI_CELL_SIZE = float(self.spin_voronoi_cell.value())
        geometry.VORONOI_STRUT_THICKNESS = float(self.spin_voronoi_thickness.value())
        geometry.VORONOI_Z_LAYERS = int(self.spin_voronoi_layers.value())
        geometry.VORONOI_IMPLICIT_CELL_SIZE = float(self.spin_voronoi_implicit_cell.value())
        geometry.VORONOI_IMPLICIT_WALL_THICKNESS = float(self.spin_voronoi_implicit_wall.value())
        geometry.VORONOI_IMPLICIT_RESOLUTION = int(self.spin_voronoi_implicit_res.value())
        geometry.VORONOI_IMPLICIT_SEED_DENSITY_FACTOR = float(self.spin_voronoi_implicit_density.value())
        geometry.LATTICE_TILE_SHRINK = float(self.spin_tile_shrink.value())
        geometry.TILE_SPACING_FACTOR = float(self.spin_tile_spacing.value())
        geometry.TILE_BOUNDARY_MARGIN = float(self.spin_tile_margin.value())
        geometry.LATTICE_UNIT_TARGET_FACES = int(self.lattice_unit_target_faces)

    def _mesh_layers(self) -> list[MeshData]:
        """生成要显示的网格层。
        
        注意：分割对话框已通过viewer.set_meshes()直接处理分割网格的显示，
        所以这里只需要处理正常的鞋底和晶格显示。
        """
        meshes: list[MeshData] = []
        show_edges = self.action_toggle_mesh_edges.isChecked()
        
        # 显示鞋底
        if self.current_view_mode in {"sole", "combined"} and self.current_sole_mesh.faces.size > 0:
            sole_color = self.sole_material_colors.get(self.current_sole_material, (0.75, 0.75, 0.75, 1.0))
            meshes.append(MeshData(
                np.asarray(self.current_sole_mesh.vertices), 
                np.asarray(self.current_sole_mesh.faces), 
                color=sole_color
            ))
            if show_edges:
                meshes.append(MeshData(
                    np.asarray(self.current_sole_mesh.vertices), 
                    np.asarray(self.current_sole_mesh.faces), 
                    color=(0.2, 0.2, 0.2, 1.0), 
                    render_mode="wireframe", 
                    unlit=True, 
                    line_width=1.0
                ))
        
        # 显示晶格
        if self.current_view_mode in {"lattice", "combined"} and self.current_lattice_mesh.faces.size > 0:
            lattice_color = self.lattice_material_colors.get(self.current_lattice_material, (1.0, 0.6, 0.2, 1.0))
            meshes.append(MeshData(
                np.asarray(self.current_lattice_mesh.vertices), 
                np.asarray(self.current_lattice_mesh.faces), 
                color=lattice_color
            ))
            if show_edges:
                meshes.append(MeshData(
                    np.asarray(self.current_lattice_mesh.vertices), 
                    np.asarray(self.current_lattice_mesh.faces), 
                    color=(0.3, 0.3, 0.3, 1.0), 
                    render_mode="wireframe", 
                    unlit=True, 
                    line_width=1.0
                ))
        
        return meshes

    def _set_view_mode(self, mode: str) -> None:
        """切换视图模式"""
        # 如果切换到 combined 但没有生成结果，提示用户
        if mode == "combined" and not self.has_generated_result:
            self._show_information("提示", '还没有生成最终结果。\n请先点击"生成"按钮生成晶格。'
            )
            return
        
        # 如果切换到"显示鞋底"，始终显示原始鞋底
        if mode == "sole":
            # 如果有原始鞋底，显示原始鞋底
            if self.original_sole_mesh.vertices.size > 0:
                self.current_sole_mesh = self.original_sole_mesh.copy()
            # 否则，如果有生成的鞋底，显示生成的鞋底
            elif self.has_generated_result:
                self.current_sole_mesh = self.generated_sole_mesh
            # 清空晶格显示
            self.current_lattice_mesh = trimesh.Trimesh()
            self.current_combined_mesh = trimesh.Trimesh()
        
        # 如果有生成的结果，恢复生成的网格（避免被预览覆盖）
        elif self.has_generated_result:
            self.current_sole_mesh = self.generated_sole_mesh
            self.current_lattice_mesh = self.generated_lattice_mesh
            self.current_combined_mesh = self.generated_combined_mesh
        
        self.current_view_mode = mode
        self._refresh_view()

    def _refresh_view(self) -> None:
        self.viewer.set_meshes(self._mesh_layers(), reset_view=False)

    def _generate(self) -> None:
        """生成晶格 - 支持全局参数和区域设计两种模式"""
        sole_path = self.edit_sole_path.text().strip()
        lattice_path = self.edit_lattice_path.text().strip()
        output_path = self.edit_output_path.text().strip()
        if not sole_path or not Path(sole_path).exists():
            self._show_warning("提示", "请先选择存在的鞋底 STL 文件")
            return

        # 检查是否在区域设计模式
        # 如果有选中的区域或平面区域，则使用区域设计生成
        if self.region_manager.regions or self.plane_region_manager.regions:
            # 询问用户选择生成模式
            dialog = QtWidgets.QDialog(self)
            dialog.setWindowTitle("选择生成模式")
            dialog.setModal(True)
            dialog.resize(350, 200)
            self._apply_dark_dialog_style(dialog)
            
            layout = QtWidgets.QVBoxLayout(dialog)
            
            label = QtWidgets.QLabel("检测到已定义的区域，请选择生成模式：")
            label.setStyleSheet("font-weight: bold;")
            layout.addWidget(label)
            
            layout.addSpacing(10)
            
            radio_global = QtWidgets.QRadioButton("全局参数生成（使用晶格参数设置）")
            radio_global.setChecked(True)
            layout.addWidget(radio_global)
            
            radio_region = QtWidgets.QRadioButton("区域设计生成（使用已定义的区域）")
            layout.addWidget(radio_region)
            
            layout.addStretch()
            
            btn_layout = QtWidgets.QHBoxLayout()
            btn_ok = QtWidgets.QPushButton("确定")
            btn_cancel = QtWidgets.QPushButton("取消")
            btn_layout.addStretch()
            btn_layout.addWidget(btn_ok)
            btn_layout.addWidget(btn_cancel)
            layout.addLayout(btn_layout)
            
            btn_ok.clicked.connect(dialog.accept)
            btn_cancel.clicked.connect(dialog.reject)
            
            if dialog.exec_() != QtWidgets.QDialog.Accepted:
                return
            
            if radio_region.isChecked():
                # 使用区域设计生成
                if self.region_manager.regions:
                    self._on_generate_region_lattices()
                elif self.plane_region_manager.regions:
                    self._on_generate_plane_region_lattices()
                return
        
        # 全局参数生成
        # 主动清理旧的网格对象，释放内存
        self._clear_previous_meshes()

        self._apply_runtime_settings()
        
        # 问题1：清空日志
        self.status_box.clear()
        
        self._append_status("开始生成鞋底内部晶格...")
        self._append_status(f"当前方法: {geometry.LATTICE_METHOD}")

        old_log = geometry._log
        geometry._log = self._append_status
        try:
            # 检查全局精细裁剪开关
            enable_final_trim = self.check_enable_final_trim.isChecked()
            sole_mesh, lattice_mesh, combined_mesh = geometry.build_sole_with_lattice(
                sole_path=sole_path, 
                lattice_path=lattice_path,
                enable_final_trim=enable_final_trim
            )
        except Exception as e:
            self._append_status(f"生成失败: {type(e).__name__}: {e}")
            self._show_critical("生成失败", str(e))
            return
        finally:
            geometry._log = old_log

        # 保存到生成结果变量（与预览分离）
        self.generated_sole_mesh = sole_mesh
        self.generated_lattice_mesh = lattice_mesh
        self.generated_combined_mesh = combined_mesh
        
        # 同时更新当前显示的网格
        self.current_sole_mesh = sole_mesh
        self.current_lattice_mesh = lattice_mesh
        self.current_combined_mesh = combined_mesh
        
        # 问题2：标记已生成结果
        self.has_generated_result = True
        
        # 标记工程已修改（生成了新的晶格结果）
        self._mark_project_modified()
        
        self.current_view_mode = "combined"
        self._remember_current_preview_view()
        self._current_preview_key = None
        self.viewer.set_meshes(self._mesh_layers(), reset_view=True)

        # 问题3：不自动保存，只在用户点击导出时才保存
        # 移除自动导出逻辑
        self._append_status('生成完成！可以在"视图"菜单切换显示模式。')

    def _clear_previous_meshes(self) -> None:
        """主动清理旧的网格对象，释放内存"""
        import gc
        
        # 清空旧的生成结果
        if hasattr(self, 'generated_sole_mesh') and self.generated_sole_mesh.faces.size > 0:
            del self.generated_sole_mesh
        if hasattr(self, 'generated_lattice_mesh') and self.generated_lattice_mesh.faces.size > 0:
            del self.generated_lattice_mesh
        if hasattr(self, 'generated_combined_mesh') and self.generated_combined_mesh.faces.size > 0:
            del self.generated_combined_mesh
        
        # 清空当前显示的网格对象
        if hasattr(self, 'current_sole_mesh') and self.current_sole_mesh.faces.size > 0:
            del self.current_sole_mesh
        if hasattr(self, 'current_lattice_mesh') and self.current_lattice_mesh.faces.size > 0:
            del self.current_lattice_mesh
        if hasattr(self, 'current_combined_mesh') and self.current_combined_mesh.faces.size > 0:
            del self.current_combined_mesh
        
        # 重置为空网格
        self.generated_sole_mesh = trimesh.Trimesh()
        self.generated_lattice_mesh = trimesh.Trimesh()
        self.generated_combined_mesh = trimesh.Trimesh()
        self.current_sole_mesh = trimesh.Trimesh()
        self.current_lattice_mesh = trimesh.Trimesh()
        self.current_combined_mesh = trimesh.Trimesh()
        
        # 强制垃圾回收
        gc.collect()


    def _export_meshes(self, output_path: str) -> None:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        
        # 确定要导出的网格
        export_mesh = None
        if self.current_combined_mesh.faces.size > 0:
            export_mesh = self.current_combined_mesh
        elif self.current_sole_mesh.faces.size > 0:
            export_mesh = self.current_sole_mesh
        
        if export_mesh is not None:
            # 导出主网格
            export_mesh.export(str(out))
        
        # 导出晶格单独文件
        if self.current_lattice_mesh.faces.size > 0:
            lattice_only = out.with_name(out.stem + "_lattice_only.stl")
            self.current_lattice_mesh.export(str(lattice_only))

    def _export_current(self) -> None:
        # 检查是否有生成的结果
        if not self.has_generated_result:
            self._show_warning("提示", '还没有生成最终结果。\n请先点击"生成"按钮生成晶格。'
            )
            return
        
        output_path = self.edit_output_path.text().strip()
        if not output_path:
            self._show_warning("提示", "请先设置输出 STL 路径")
            return
        try:
            self._export_meshes(output_path)
            self._append_status(f"手动导出完成: {output_path}")
        except Exception as e:
            self._append_status(f"导出失败: {type(e).__name__}: {e}")
            self._show_critical("导出失败", str(e))
    
    # ========================================================================
    # 区域晶格设计相关方法
    # ========================================================================
    
    def _build_region_lattice_tab(self, layout: QtWidgets.QVBoxLayout) -> None:
        """构建区域晶格设计选项卡"""
        
        # 提示信息
        hint = QtWidgets.QLabel("💡 为不同区域配置独立的晶格参数")
        hint.setStyleSheet("color: #ffffff; font-size: 11px; padding: 8px; background: #3a3a3a; border-radius: 4px;")
        hint.setWordWrap(True)
        layout.addWidget(hint)
        
        # === 区域管理 ===
        region_mgmt_group = QtWidgets.QGroupBox("区域管理")
        region_mgmt_layout = QtWidgets.QVBoxLayout(region_mgmt_group)
        
        # 区域统计
        stats_layout = QtWidgets.QHBoxLayout()
        stats_layout.addWidget(QtWidgets.QLabel("区域总数:"))
        self.region_count_label = QtWidgets.QLabel("0")
        self.region_count_label.setStyleSheet("font-weight: bold; font-size: 14px; color: #2196F3;")
        stats_layout.addWidget(self.region_count_label)
        stats_layout.addStretch()
        region_mgmt_layout.addLayout(stats_layout)
        
        # 管理按钮
        self.btn_open_region_manager = QtWidgets.QPushButton("📋 打开区域管理器")
        self.btn_open_region_manager.setToolTip("打开区域管理窗口，添加/删除/配置区域")
        self.btn_open_region_manager.setStyleSheet("padding: 10px; font-size: 13px;")
        self.btn_open_region_manager.clicked.connect(self._open_region_manager_dialog)
        region_mgmt_layout.addWidget(self.btn_open_region_manager)
        
        layout.addWidget(region_mgmt_group)
        
        # === 操作按钮 ===
        actions_group = QtWidgets.QGroupBox("操作")
        actions_layout = QtWidgets.QVBoxLayout(actions_group)
        
        self.btn_generate_regions = QtWidgets.QPushButton("🚀 生成所有区域晶格")
        self.btn_generate_regions.setStyleSheet("background: #4CAF50; color: white; padding: 12px; font-weight: bold; font-size: 14px;")
        self.btn_generate_regions.clicked.connect(self._on_generate_region_lattices)
        actions_layout.addWidget(self.btn_generate_regions)
        
        self.btn_view_region_result = QtWidgets.QPushButton("👁️ 查看生成结果")
        self.btn_view_region_result.setEnabled(False)
        self.btn_view_region_result.setStyleSheet("padding: 10px; font-size: 13px;")
        self.btn_view_region_result.setToolTip("在主视图中显示生成的区域晶格结果")
        self.btn_view_region_result.clicked.connect(self._display_region_result)
        actions_layout.addWidget(self.btn_view_region_result)
        
        self.btn_select_view_region = QtWidgets.QPushButton("🔍 选择查看区域")
        self.btn_select_view_region.setEnabled(False)
        self.btn_select_view_region.setStyleSheet("padding: 10px; font-size: 13px;")
        self.btn_select_view_region.setToolTip("选择要查看的单个区域或完整晶格")
        self.btn_select_view_region.clicked.connect(self._open_select_region_view_dialog)
        actions_layout.addWidget(self.btn_select_view_region)
        
        self.btn_export_combined = QtWidgets.QPushButton("💾 导出合并晶格")
        self.btn_export_combined.setEnabled(False)
        self.btn_export_combined.setStyleSheet("padding: 10px; font-size: 13px;")
        self.btn_export_combined.clicked.connect(self._on_export_combined_lattice)
        actions_layout.addWidget(self.btn_export_combined)
        
        layout.addWidget(actions_group)
        
        layout.addStretch()  # 添加弹性空间
    
    def _open_region_manager_dialog(self) -> None:
        """打开区域管理器对话框"""
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle("区域管理器")
        dialog.setModal(False)  # 非模态，可以同时操作主窗口
        dialog.resize(800, 700)
        
        # 应用深色主题
        self._apply_dark_dialog_style(dialog)
        
        main_layout = QtWidgets.QVBoxLayout(dialog)
        
        # === 顶部：操作按钮 ===
        btn_layout = QtWidgets.QHBoxLayout()
        btn_add = QtWidgets.QPushButton("➕ 添加区域")
        btn_remove = QtWidgets.QPushButton("➖ 删除区域")
        btn_add.clicked.connect(lambda: self._on_add_region_from_dialog(dialog))
        btn_remove.clicked.connect(lambda: self._on_remove_region_from_dialog(dialog))
        btn_layout.addWidget(btn_add)
        btn_layout.addWidget(btn_remove)
        btn_layout.addStretch()
        main_layout.addLayout(btn_layout)
        
        # === 中部：左右分栏 ===
        content_layout = QtWidgets.QHBoxLayout()
        
        # 左侧：区域列表
        left_panel = QtWidgets.QGroupBox("区域列表")
        left_layout = QtWidgets.QVBoxLayout(left_panel)
        
        region_list = QtWidgets.QListWidget()
        region_list.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        left_layout.addWidget(region_list)
        
        content_layout.addWidget(left_panel, stretch=1)
        
        # 右侧：参数配置
        right_panel = QtWidgets.QGroupBox("晶格参数配置")
        right_layout = QtWidgets.QVBoxLayout(right_panel)
        
        # 推荐参数按钮
        btn_recommend = QtWidgets.QPushButton("💡 推荐参数")
        btn_recommend.setToolTip("根据鞋底尺寸推荐当前区域的晶格参数")
        btn_recommend.clicked.connect(self._on_region_recommend_params)
        btn_recommend.setEnabled(False)
        right_layout.addWidget(btn_recommend)
        
        # 晶格方法选择
        method_layout = QtWidgets.QHBoxLayout()
        method_layout.addWidget(QtWidgets.QLabel("晶格方法:"))
        method_combo = QtWidgets.QComboBox()
        # 设置下拉列表的最大可见项数
        method_combo.setMaxVisibleItems(10)
        method_combo.addItems(["gyroid", "voronoi_2_5d", "tile_unit"])
        method_layout.addWidget(method_combo)
        method_layout.addStretch()
        right_layout.addLayout(method_layout)
        
        # Gyroid 参数组
        gyroid_group = self._create_region_gyroid_params_for_dialog()
        right_layout.addWidget(gyroid_group)
        
        # Voronoi 参数组
        voronoi_group = self._create_region_voronoi_params_for_dialog()
        right_layout.addWidget(voronoi_group)
        
        # Tile 参数组
        tile_group = self._create_region_tile_params_for_dialog()
        right_layout.addWidget(tile_group)
        
        # 提示标签
        params_hint = QtWidgets.QLabel("请先选择一个区域")
        params_hint.setStyleSheet("color: #cccccc; font-style: italic; padding: 10px;")
        params_hint.setAlignment(QtCore.Qt.AlignCenter)
        right_layout.addWidget(params_hint)
        
        right_layout.addStretch()
        content_layout.addWidget(right_panel, stretch=2)
        
        main_layout.addLayout(content_layout)
        
        # === 底部：配置管理和关闭按钮 ===
        bottom_layout = QtWidgets.QHBoxLayout()
        
        btn_save = QtWidgets.QPushButton("💾 保存配置")
        btn_load = QtWidgets.QPushButton("📂 加载配置")
        btn_save.clicked.connect(self._on_save_region_config)
        btn_load.clicked.connect(lambda: [self._on_load_region_config(), refresh_list()])
        bottom_layout.addWidget(btn_save)
        bottom_layout.addWidget(btn_load)
        bottom_layout.addStretch()
        
        btn_close = QtWidgets.QPushButton("关闭")
        btn_close.clicked.connect(dialog.close)
        bottom_layout.addWidget(btn_close)
        
        main_layout.addLayout(bottom_layout)
        
        # === 辅助函数 ===
        def refresh_list():
            """刷新区域列表"""
            region_list.clear()
            for region in self.region_manager.regions:
                item = QtWidgets.QListWidgetItem(f"{region.name} ({region.source_type})")
                item.setData(QtCore.Qt.UserRole, region.id)
                region_list.addItem(item)
                if region.id == self.selected_region_id:
                    item.setSelected(True)
            self._update_region_display()
        
        def set_params_enabled(enabled: bool):
            """启用/禁用参数配置"""
            method_combo.setEnabled(enabled)
            gyroid_group.setEnabled(enabled)
            voronoi_group.setEnabled(enabled)
            tile_group.setEnabled(enabled)
            btn_recommend.setEnabled(enabled)
            
            if enabled:
                params_hint.hide()
                sync_method_fields()
            else:
                params_hint.show()
        
        def sync_method_fields():
            """同步参数组显示"""
            method = method_combo.currentText().strip().lower()
            gyroid_group.setVisible(method == "gyroid")
            voronoi_group.setVisible(method == "voronoi_2_5d")
            tile_group.setVisible(method == "tile_unit")
        
        def on_selection_changed():
            """选择变化"""
            current_item = region_list.currentItem()
            if current_item:
                region_id = current_item.data(QtCore.Qt.UserRole)
                self.selected_region_id = region_id
                load_parameters(region_id, method_combo, gyroid_group, voronoi_group, tile_group)
                set_params_enabled(True)
            else:
                self.selected_region_id = None
                set_params_enabled(False)
            self._update_region_display()
        
        def load_parameters(region_id, method_combo, gyroid_group, voronoi_group, tile_group):
            """加载区域参数"""
            region = self.region_manager.get_region(region_id)
            if region is None:
                return
            
            # 获取对话框中的控件
            gyroid_widgets = {
                'cell': gyroid_group.findChild(QtWidgets.QDoubleSpinBox, 'gyroid_cell'),
                'iso': gyroid_group.findChild(QtWidgets.QDoubleSpinBox, 'gyroid_iso'),
                'res': gyroid_group.findChild(QtWidgets.QSpinBox, 'gyroid_res')
            }
            voronoi_widgets = {
                'cell': voronoi_group.findChild(QtWidgets.QDoubleSpinBox, 'voronoi_cell'),
                'thickness': voronoi_group.findChild(QtWidgets.QDoubleSpinBox, 'voronoi_thickness'),
                'layers': voronoi_group.findChild(QtWidgets.QSpinBox, 'voronoi_layers')
            }
            tile_widgets = {
                'shrink': tile_group.findChild(QtWidgets.QDoubleSpinBox, 'tile_shrink'),
                'spacing': tile_group.findChild(QtWidgets.QDoubleSpinBox, 'tile_spacing'),
                'margin': tile_group.findChild(QtWidgets.QDoubleSpinBox, 'tile_margin'),
                'decimate': tile_group.findChild(QtWidgets.QSpinBox, 'tile_decimate')
            }
            
            # 阻止信号
            method_combo.blockSignals(True)
            for w in gyroid_widgets.values():
                if w: w.blockSignals(True)
            for w in voronoi_widgets.values():
                if w: w.blockSignals(True)
            for w in tile_widgets.values():
                if w: w.blockSignals(True)
            
            # 设置值
            method_combo.setCurrentText(region.lattice_method)
            if gyroid_widgets['cell']: gyroid_widgets['cell'].setValue(region.gyroid_cell_size)
            if gyroid_widgets['iso']: gyroid_widgets['iso'].setValue(region.gyroid_isovalue)
            if gyroid_widgets['res']: gyroid_widgets['res'].setValue(region.gyroid_resolution)
            if voronoi_widgets['cell']: voronoi_widgets['cell'].setValue(region.voronoi_cell_size)
            if voronoi_widgets['thickness']: voronoi_widgets['thickness'].setValue(region.voronoi_strut_thickness)
            if voronoi_widgets['layers']: voronoi_widgets['layers'].setValue(region.voronoi_z_layers)
            if tile_widgets['shrink']: tile_widgets['shrink'].setValue(region.tile_shrink)
            if tile_widgets['spacing']: tile_widgets['spacing'].setValue(region.tile_spacing)
            if tile_widgets['margin']: tile_widgets['margin'].setValue(region.tile_margin)
            if tile_widgets['decimate']: tile_widgets['decimate'].setValue(region.tile_decimate_target_faces)
            
            # 恢复信号
            method_combo.blockSignals(False)
            for w in gyroid_widgets.values():
                if w: w.blockSignals(False)
            for w in voronoi_widgets.values():
                if w: w.blockSignals(False)
            for w in tile_widgets.values():
                if w: w.blockSignals(False)
            
            sync_method_fields()
        
        def on_param_changed():
            """参数改变"""
            if self.selected_region_id is None:
                return
            
            region = self.region_manager.get_region(self.selected_region_id)
            if region is None:
                return
            
            # 获取对话框中的控件
            gyroid_widgets = {
                'cell': gyroid_group.findChild(QtWidgets.QDoubleSpinBox, 'gyroid_cell'),
                'iso': gyroid_group.findChild(QtWidgets.QDoubleSpinBox, 'gyroid_iso'),
                'res': gyroid_group.findChild(QtWidgets.QSpinBox, 'gyroid_res')
            }
            voronoi_widgets = {
                'cell': voronoi_group.findChild(QtWidgets.QDoubleSpinBox, 'voronoi_cell'),
                'thickness': voronoi_group.findChild(QtWidgets.QDoubleSpinBox, 'voronoi_thickness'),
                'layers': voronoi_group.findChild(QtWidgets.QSpinBox, 'voronoi_layers')
            }
            tile_widgets = {
                'shrink': tile_group.findChild(QtWidgets.QDoubleSpinBox, 'tile_shrink'),
                'spacing': tile_group.findChild(QtWidgets.QDoubleSpinBox, 'tile_spacing'),
                'margin': tile_group.findChild(QtWidgets.QDoubleSpinBox, 'tile_margin'),
                'decimate': tile_group.findChild(QtWidgets.QSpinBox, 'tile_decimate')
            }
            
            # 更新区域参数
            region.lattice_method = method_combo.currentText()
            if gyroid_widgets['cell']: region.gyroid_cell_size = gyroid_widgets['cell'].value()
            if gyroid_widgets['iso']: region.gyroid_isovalue = gyroid_widgets['iso'].value()
            if gyroid_widgets['res']: region.gyroid_resolution = gyroid_widgets['res'].value()
            if voronoi_widgets['cell']: region.voronoi_cell_size = voronoi_widgets['cell'].value()
            if voronoi_widgets['thickness']: region.voronoi_strut_thickness = voronoi_widgets['thickness'].value()
            if voronoi_widgets['layers']: region.voronoi_z_layers = voronoi_widgets['layers'].value()
            if tile_widgets['shrink']: region.tile_shrink = tile_widgets['shrink'].value()
            if tile_widgets['spacing']: region.tile_spacing = tile_widgets['spacing'].value()
            if tile_widgets['margin']: region.tile_margin = tile_widgets['margin'].value()
            if tile_widgets['decimate']: region.tile_decimate_target_faces = tile_widgets['decimate'].value()
        
        # 连接信号
        region_list.itemSelectionChanged.connect(on_selection_changed)
        method_combo.currentTextChanged.connect(lambda: [sync_method_fields(), on_param_changed()])
        
        # 连接参数变化信号
        for group in [gyroid_group, voronoi_group, tile_group]:
            for widget in group.findChildren(QtWidgets.QDoubleSpinBox):
                widget.valueChanged.connect(on_param_changed)
            for widget in group.findChildren(QtWidgets.QSpinBox):
                widget.valueChanged.connect(on_param_changed)
        
        # 初始刷新
        refresh_list()
        set_params_enabled(False)
        
        # 保存引用
        dialog.refresh_list = refresh_list
        self._region_manager_dialog = dialog
        
        dialog.show()
    
    def _create_region_gyroid_params_for_dialog(self) -> QtWidgets.QGroupBox:
        """为对话框创建 Gyroid 参数组"""
        group = QtWidgets.QGroupBox("Gyroid 参数")
        layout = QtWidgets.QFormLayout(group)
        
        cell = QtWidgets.QDoubleSpinBox()
        cell.setObjectName('gyroid_cell')
        cell.setRange(0.5, 50.0)
        cell.setDecimals(3)
        cell.setValue(4.0)
        
        iso = QtWidgets.QDoubleSpinBox()
        iso.setObjectName('gyroid_iso')
        iso.setRange(-1.5, 1.5)
        iso.setDecimals(3)
        iso.setValue(0.3)
        
        res = QtWidgets.QSpinBox()
        res.setObjectName('gyroid_res')
        res.setRange(8, 128)
        res.setValue(30)
        
        layout.addRow("单胞尺寸", cell)
        layout.addRow("等值面", iso)
        layout.addRow("分辨率", res)
        
        return group
    
    def _create_region_voronoi_params_for_dialog(self) -> QtWidgets.QGroupBox:
        """为对话框创建 Voronoi 参数组"""
        group = QtWidgets.QGroupBox("Voronoi 参数")
        layout = QtWidgets.QFormLayout(group)
        
        cell = QtWidgets.QDoubleSpinBox()
        cell.setObjectName('voronoi_cell')
        cell.setRange(1.0, 100.0)
        cell.setDecimals(3)
        cell.setValue(4.2)
        
        thickness = QtWidgets.QDoubleSpinBox()
        thickness.setObjectName('voronoi_thickness')
        thickness.setRange(0.1, 20.0)
        thickness.setDecimals(3)
        thickness.setValue(0.55)
        
        layers = QtWidgets.QSpinBox()
        layers.setObjectName('voronoi_layers')
        layers.setRange(2, 30)
        layers.setValue(7)
        
        layout.addRow("胞元尺寸", cell)
        layout.addRow("杆径", thickness)
        layout.addRow("Z 向层数", layers)
        
        return group
    
    def _create_region_tile_params_for_dialog(self) -> QtWidgets.QGroupBox:
        """为对话框创建 Tile 参数组"""
        group = QtWidgets.QGroupBox("Tile 参数")
        layout = QtWidgets.QFormLayout(group)
        
        shrink = QtWidgets.QDoubleSpinBox()
        shrink.setObjectName('tile_shrink')
        shrink.setRange(0.05, 2.0)
        shrink.setDecimals(3)
        shrink.setValue(0.28)
        
        spacing = QtWidgets.QDoubleSpinBox()
        spacing.setObjectName('tile_spacing')
        spacing.setRange(0.5, 1.5)
        spacing.setDecimals(3)
        spacing.setValue(0.96)
        
        margin = QtWidgets.QDoubleSpinBox()
        margin.setObjectName('tile_margin')
        margin.setRange(0.0, 5.0)
        margin.setDecimals(3)
        margin.setValue(1.0)
        
        decimate = QtWidgets.QSpinBox()
        decimate.setObjectName('tile_decimate')
        decimate.setRange(0, 10000)
        decimate.setValue(int(self.lattice_unit_target_faces))
        decimate.setSingleStep(100)
        decimate.setSpecialValueText("禁用减面")
        decimate.setToolTip("减面目标面数（0=禁用，保持最高精度）\n减面会降低晶格单元的面数，影响精度")
        
        layout.addRow("单元尺寸缩放", shrink)
        layout.addRow("平铺间距系数", spacing)
        layout.addRow("边界安全距离", margin)
        layout.addRow("减面目标面数", decimate)
        
        return group
    
    def _on_add_region_from_dialog(self, dialog) -> None:
        """从对话框添加区域"""
        # 防止双击
        sender = self.sender()
        if sender and hasattr(sender, 'setEnabled'):
            sender.setEnabled(False)
            QtCore.QTimer.singleShot(500, lambda: sender.setEnabled(True))
        
        self._on_add_region()
        if hasattr(dialog, 'refresh_list'):
            dialog.refresh_list()
        self._update_region_display()
        # 标记工程已修改（添加了区域）
        self._mark_project_modified()
    
    def _on_remove_region_from_dialog(self, dialog) -> None:
        """从对话框删除区域"""
        if self.selected_region_id is None:
            msgbox = QtWidgets.QMessageBox(dialog)
            msgbox.setWindowTitle("提示")
            msgbox.setText("请先选择一个区域")
            msgbox.setIcon(QtWidgets.QMessageBox.Warning)
            self._apply_dark_messagebox_style(msgbox)
            msgbox.exec_()
            return
        
        region = self.region_manager.get_region(self.selected_region_id)
        if region is None:
            return
        
        msgbox = QtWidgets.QMessageBox(dialog)
        msgbox.setWindowTitle("确认删除")
        msgbox.setText(f"确定要删除区域 '{region.name}' 吗？")
        msgbox.setStandardButtons(QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No)
        msgbox.setDefaultButton(QtWidgets.QMessageBox.No)
        msgbox.setIcon(QtWidgets.QMessageBox.Question)
        self._apply_dark_messagebox_style(msgbox)
        
        reply = msgbox.exec_()
        
        if reply == QtWidgets.QMessageBox.Yes:
            self.region_manager.remove_region(self.selected_region_id)
            self.selected_region_id = None
            if hasattr(dialog, 'refresh_list'):
                dialog.refresh_list()
            self._update_region_display()
            # 标记工程已修改（删除了区域）
            self._mark_project_modified()
    
    def _update_region_display(self) -> None:
        """更新区域显示信息"""
        # 更新区域总数（如果标签存在）
        if hasattr(self, 'region_count_label'):
            self.region_count_label.setText(str(len(self.region_manager.regions)))
    
    def _on_add_region(self) -> None:
        """添加区域"""
        # 创建对话框
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle("添加区域")
        dialog.setModal(True)
        dialog.resize(350, 250)
        
        # 应用深色主题
        self._apply_dark_dialog_style(dialog)
        
        layout = QtWidgets.QVBoxLayout(dialog)
        
        # 提示
        label = QtWidgets.QLabel("请选择区域来源:")
        layout.addWidget(label)
        
        # 单选按钮组
        radio_group = QtWidgets.QButtonGroup(dialog)
        
        radio_whole = QtWidgets.QRadioButton("使用整体鞋底")
        radio_whole.setChecked(True)
        radio_group.addButton(radio_whole)
        layout.addWidget(radio_whole)
        
        layout.addSpacing(10)
        
        radio_split = QtWidgets.QRadioButton("使用分割结果")
        radio_group.addButton(radio_split)
        layout.addWidget(radio_split)
        
        # 分割选项（缩进）
        split_options = QtWidgets.QWidget()
        split_layout = QtWidgets.QVBoxLayout(split_options)
        split_layout.setContentsMargins(30, 0, 0, 0)
        
        split_radio_group = QtWidgets.QButtonGroup(dialog)
        radio_positive = QtWidgets.QRadioButton("正侧部分")
        radio_positive.setChecked(True)
        split_radio_group.addButton(radio_positive)
        split_layout.addWidget(radio_positive)
        
        radio_negative = QtWidgets.QRadioButton("负侧部分")
        split_radio_group.addButton(radio_negative)
        split_layout.addWidget(radio_negative)
        
        layout.addWidget(split_options)
        split_options.setEnabled(False)
        
        # 连接信号
        radio_split.toggled.connect(split_options.setEnabled)
        
        layout.addSpacing(10)
        
        # 提示信息
        hint = QtWidgets.QLabel("提示: 使用分割结果前，请先在\n\"模型分割\"选项卡中执行分割")
        hint.setStyleSheet("color: #cccccc; font-size: 10px; font-style: italic;")
        layout.addWidget(hint)
        
        layout.addStretch()
        
        # 按钮
        button_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
        )
        button_box.accepted.connect(dialog.accept)
        button_box.rejected.connect(dialog.reject)
        layout.addWidget(button_box)
        
        # 显示对话框
        if dialog.exec_() == QtWidgets.QDialog.Accepted:
            # 确定来源类型
            if radio_whole.isChecked():
                source_type = "whole_sole"
            elif radio_positive.isChecked():
                source_type = "split_positive"
            else:
                source_type = "split_negative"
            
            # 获取网格
            mesh = self._get_region_mesh(source_type)
            if mesh is None:
                self._show_warning("错误", "无法获取区域网格。\n请确保已加载鞋底模型或已执行分割。")
                return
            
            # 添加区域
            region = self.region_manager.add_region(source_type, mesh)
            self._append_status(f"已添加区域: {region.get_display_name()}")
            
            # 更新显示
            self._update_region_display()
    
    def _get_region_mesh(self, source_type: str) -> Optional[trimesh.Trimesh]:
        """根据来源类型获取区域网格"""
        if source_type == "whole_sole":
            if self.current_sole_mesh.faces.size == 0:
                return None
            return self.current_sole_mesh.copy()
        
        elif source_type == "split_positive":
            if not self.slicing_enabled or self.positive_mesh is None:
                return None
            return self.positive_mesh.copy()
        
        elif source_type == "split_negative":
            if not self.slicing_enabled or self.negative_mesh is None:
                return None
            return self.negative_mesh.copy()
        
        return None
    
    def _on_region_recommend_params(self) -> None:
        """为当前区域推荐参数"""
        if self.selected_region_id is None:
            self._show_warning("提示", "请先选择一个区域")
            return
        
        region = self.region_manager.get_region(self.selected_region_id)
        if region is None:
            return
        
        # 获取区域网格尺寸（从 RegionManager 中获取）
        region_mesh = self.region_manager.get_region_mesh(self.selected_region_id)
        if region_mesh is None or region_mesh.faces.size == 0:
            self._show_warning("提示", "区域网格为空，无法推荐参数")
            return
        
        # 计算区域网格的尺寸
        region_extents = np.asarray(region_mesh.extents, dtype=np.float64)
        if region_extents.shape != (3,) or not np.isfinite(region_extents).all():
            self._show_warning("提示", "无法获取区域网格尺寸")
            return
        
        region_extents = np.maximum(region_extents, 1e-6)
        
        # 获取晶格单元尺寸（用于tile方法）
        lattice_extents = self._load_mesh_extents(self.edit_lattice_path.text().strip())
        
        method = region.lattice_method
        
        try:
            if method == "gyroid":
                params = self._recommended_gyroid_params(region_extents)
                
                # 直接更新区域参数
                self.region_manager.update_region_parameters(
                    self.selected_region_id,
                    gyroid_cell_size=float(params["gyroid_cell"]),
                    gyroid_isovalue=float(params["gyroid_iso"]),
                    gyroid_resolution=int(params["gyroid_res"])
                )
                
            elif method == "voronoi_2_5d":
                params = self._recommended_voronoi_params(region_extents)
                
                # 直接更新区域参数
                self.region_manager.update_region_parameters(
                    self.selected_region_id,
                    voronoi_cell_size=float(params["voronoi_cell"]),
                    voronoi_strut_thickness=float(params["voronoi_thickness"]),
                    voronoi_z_layers=int(params["voronoi_layers"])
                )
                
            elif method == "tile_unit":
                if lattice_extents is None:
                    self._show_warning("提示", "Tile方法需要晶格单元文件，请先加载晶格STL文件")
                    return
                
                params = self._recommended_tile_params(region_extents, lattice_extents)
                
                # 直接更新区域参数
                self.region_manager.update_region_parameters(
                    self.selected_region_id,
                    tile_shrink=float(params["tile_shrink"]),
                    tile_spacing=float(params["tile_spacing"]),
                    tile_margin=float(params["tile_margin"])
                )
            
            self._append_status(f"[推荐参数] 已为区域 \"{region.name}\" 推荐 {method} 参数")
            
            # 如果对话框打开，刷新参数显示
            if hasattr(self, '_region_manager_dialog') and self._region_manager_dialog is not None:
                # 触发对话框刷新（通过重新选择当前区域）
                if hasattr(self._region_manager_dialog, 'refresh_list'):
                    self._region_manager_dialog.refresh_list()
            
            self._show_information("推荐参数", f"已为区域 \"{region.name}\" 推荐 {method} 参数")
            
        except Exception as e:
            self._show_critical("推荐失败", f"推荐参数时出错:\n{e}")
    
    def _on_generate_region_lattices(self) -> None:
        """生成所有区域的晶格"""
        # 验证区域列表
        if len(self.region_manager.regions) == 0:
            self._show_warning("提示", "请先添加至少一个区域")
            return
        
        # 清空之前的结果
        self.region_manager.clear_all_lattices()
        self.combined_region_lattice = None
        
        self._append_status("=" * 50)
        self._append_status("开始生成区域晶格...")
        self._append_status(f"区域数量: {len(self.region_manager.regions)}")
        
        try:
            # 遍历所有区域
            for i, region in enumerate(self.region_manager.regions, 1):
                self._append_status(f"\n[{i}/{len(self.region_manager.regions)}] 处理区域: {region.name}")
                self._append_status(f"  来源: {region.source_type}")
                self._append_status(f"  方法: {region.lattice_method}")
                
                # 获取区域网格（从 RegionManager 中获取）
                region_mesh = self.region_manager.get_region_mesh(region.id)
                if region_mesh is None or region_mesh.faces.size == 0:
                    self._append_status(f"  ✗ 区域 {region.name} 网格为空")
                    continue
                
                self._append_status(f"  区域网格: {len(region_mesh.faces)} 面")
                
                # 为区域生成晶格
                lattice_mesh = self._generate_lattice_for_split_region(region, region_mesh)
                
                if lattice_mesh is None or lattice_mesh.faces.size == 0:
                    self._append_status(f"  ✗ 区域 {region.name} 晶格生成失败")
                    continue
                
                self._append_status(f"  ✓ 生成晶格: {len(lattice_mesh.faces)} 面")
                
                # 保存到区域管理器
                self.region_manager.set_region_lattice(region.id, lattice_mesh)
            
            # 合并所有区域的晶格
            self.combined_region_lattice = self.region_manager.get_combined_lattice()
            
            if self.combined_region_lattice is None or self.combined_region_lattice.faces.size == 0:
                self._append_status("\n✗ 没有生成任何晶格")
                self._show_warning("生成失败", "没有生成任何晶格，请检查参数设置")
                return
            
            self._append_status(f"\n✓ 合并完成: {len(self.combined_region_lattice.faces)} 面")
            
            # 启用查看和导出按钮
            if hasattr(self, 'btn_view_region_result'):
                self.btn_view_region_result.setEnabled(True)
            if hasattr(self, 'btn_select_view_region'):
                self.btn_select_view_region.setEnabled(True)
            if hasattr(self, 'btn_export_combined'):
                self.btn_export_combined.setEnabled(True)
            
            # 在 3D 视图中显示结果
            self._display_region_result()
            
            # 启用导出按钮
            if hasattr(self, 'btn_export_combined'):
                self.btn_export_combined.setEnabled(True)
            
            self._append_status("=" * 50)
            self._append_status("✓ 区域晶格生成完成！")
            
            # 标记工程已修改（生成了新的区域晶格结果）
            self._mark_project_modified()
            
            self._show_information("生成成功", f"已成功生成 {len(self.region_manager.regions)} 个区域的晶格\n"
                f"总面数: {len(self.combined_region_lattice.faces)}"
            )
            
        except Exception as e:
            self._append_status(f"\n✗ 生成失败: {e}")
            self._show_critical("生成失败", f"生成晶格时出错:\n{e}")
            import traceback
            traceback.print_exc()
    
    def _generate_lattice_for_split_region(self, region: Region, region_mesh: trimesh.Trimesh) -> trimesh.Trimesh:
        """为单个分割区域生成晶格"""
        
        # 计算区域包围盒信息
        if self.current_sole_mesh is not None:
            region_bounds = region_mesh.bounds
            region_extents = region_bounds[1] - region_bounds[0]
            
            sole_bounds = self.current_sole_mesh.bounds
            sole_extents = sole_bounds[1] - sole_bounds[0]
            
            # 计算包围盒体积比例
            region_bbox_volume = np.prod(region_extents)
            sole_bbox_volume = np.prod(sole_extents)
            
            if sole_bbox_volume > 0:
                ratio = region_bbox_volume / sole_bbox_volume * 100
                self._append_status(
                    f"  区域包围盒: {region_extents[0]:.1f}×{region_extents[1]:.1f}×{region_extents[2]:.1f} mm ({ratio:.1f}% 完整包围盒)"
                )
        
        if region.lattice_method == "gyroid":
            # 生成 Gyroid 晶格
            lattice = geometry.generate_gyroid_lattice(
                sole_mesh=region_mesh,
                cell_size=region.gyroid_cell_size,
                isovalue=region.gyroid_isovalue,
                resolution=region.gyroid_resolution,
            )
        
        elif region.lattice_method == "voronoi_2_5d":
            # 生成 Voronoi 晶格（需要临时修改全局配置）
            import config
            old_cell_size = config.VORONOI_CELL_SIZE
            old_thickness = config.VORONOI_STRUT_THICKNESS
            old_layers = config.VORONOI_Z_LAYERS
            
            try:
                config.VORONOI_CELL_SIZE = region.voronoi_cell_size
                config.VORONOI_STRUT_THICKNESS = region.voronoi_strut_thickness
                config.VORONOI_Z_LAYERS = region.voronoi_z_layers
                
                lattice = geometry.generate_voronoi_lattice_in_sole(
                    sole_mesh=region_mesh,
                    cell_size=region.voronoi_cell_size,
                    strut_thickness=region.voronoi_strut_thickness,
                )
            finally:
                # 恢复原始配置
                config.VORONOI_CELL_SIZE = old_cell_size
                config.VORONOI_STRUT_THICKNESS = old_thickness
                config.VORONOI_Z_LAYERS = old_layers
        
        elif region.lattice_method == "tile_unit":
            # 生成 Tile 晶格（需要晶格单元文件）
            lattice_path = self.edit_lattice_path.text()
            if not Path(lattice_path).exists():
                self._append_status(f"  ✗ 晶格单元文件不存在: {lattice_path}")
                return trimesh.Trimesh()
            
            # 使用带缓存的生成函数
            lattice = geometry.generate_tile_lattice_with_cache(
                sole_mesh=region_mesh,
                lattice_path=lattice_path,
                shrink=region.tile_shrink,
                spacing=region.tile_spacing,
                margin=region.tile_margin,
                unit_target_faces=region.tile_decimate_target_faces,
                use_cache=True
            )
        
        else:
            self._append_status(f"  ✗ 未知的晶格方法: {region.lattice_method}")
            return trimesh.Trimesh()
        
        # === 第二次精细裁剪：使用多点采样裁剪到区域体积 ===
        # 检查全局精细裁剪开关
        enable_final_trim = self.check_enable_final_trim.isChecked() if hasattr(self, 'check_enable_final_trim') else True
        
        if enable_final_trim and lattice.faces.size > 0:
            faces_before = len(lattice.faces)
            lattice = geometry.trim_lattice_to_sole_volume(region_mesh, lattice)
            faces_after = len(lattice.faces)
            if faces_before != faces_after:
                self._append_status(f"  精细裁剪: {faces_before} -> {faces_after} 面")
        elif not enable_final_trim:
            self._append_status(f"  跳过精细裁剪（全局开关已禁用）")
        
        return lattice
    
    def _display_region_result(self) -> None:
        """在 3D 视图中显示区域晶格结果"""
        if self.combined_region_lattice is None:
            return
        
        # 创建合并视图（鞋底 + 晶格）
        combined = trimesh.util.concatenate([
            self.current_sole_mesh,
            self.combined_region_lattice
        ])
        
        # 保存生成的网格，使其可以通过视图菜单切换
        self.generated_sole_mesh = self.current_sole_mesh
        self.generated_lattice_mesh = self.combined_region_lattice
        self.generated_combined_mesh = combined
        self.has_generated_result = True
        
        # 更新当前视图为只显示晶格
        self.current_lattice_mesh = self.combined_region_lattice
        self.current_view_mode = "lattice"
        
        # 刷新视图
        self._refresh_view()
    
    def _open_select_region_view_dialog(self) -> None:
        """打开选择查看区域的对话框（选项卡3）"""
        if self.combined_region_lattice is None:
            self._show_warning("提示", "请先生成区域晶格")
            return
        
        # 创建对话框
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle("选择查看区域")
        dialog.setModal(True)
        dialog.resize(400, 500)
        
        # 应用深色主题
        self._apply_dark_dialog_style(dialog)
        
        layout = QtWidgets.QVBoxLayout(dialog)
        
        # 提示
        hint = QtWidgets.QLabel("选择要在主视图中显示的内容：")
        hint.setStyleSheet("font-weight: bold; font-size: 12px; padding: 5px;")
        layout.addWidget(hint)
        
        # 列表
        list_widget = QtWidgets.QListWidget()
        list_widget.setStyleSheet("font-size: 12px;")
        
        # 添加"完整晶格"选项
        item_all = QtWidgets.QListWidgetItem("🌐 完整晶格（所有区域合并）")
        item_all.setData(QtCore.Qt.UserRole, "all")
        list_widget.addItem(item_all)
        
        # 添加各个区域选项
        for region in self.region_manager.regions:
            lattice = self.region_manager.get_region_lattice(region.id)
            if lattice is not None and lattice.faces.size > 0:
                face_count = len(lattice.faces)
                item = QtWidgets.QListWidgetItem(f"📦 {region.name} ({face_count:,} 面)")
                item.setData(QtCore.Qt.UserRole, region.id)
                list_widget.addItem(item)
        
        # 默认选中第一项
        if list_widget.count() > 0:
            list_widget.setCurrentRow(0)
        
        layout.addWidget(list_widget)
        
        # 按钮
        button_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
        )
        button_box.accepted.connect(dialog.accept)
        button_box.rejected.connect(dialog.reject)
        layout.addWidget(button_box)
        
        # 显示对话框
        if dialog.exec_() == QtWidgets.QDialog.Accepted:
            current_item = list_widget.currentItem()
            if current_item is None:
                return
            
            selection = current_item.data(QtCore.Qt.UserRole)
            
            self._display_selected_region(selection)
    
    def _display_selected_region(self, selection: str) -> None:
        """显示选中的区域（选项卡3）"""
        if selection == "all":
            # 显示完整晶格
            lattice = self.combined_region_lattice
            self._append_status("显示: 完整晶格（所有区域）")
        else:
            # 显示单个区域
            region = self.region_manager.get_region(selection)
            if region is None:
                return
            
            lattice = self.region_manager.get_region_lattice(selection)
            if lattice is None or lattice.faces.size == 0:
                self._show_warning("提示", f"区域 '{region.name}' 没有生成晶格")
                return
            
            self._append_status(f"显示: 区域 '{region.name}'")
        
        # 更新视图（只显示晶格，不显示鞋底）
        self.current_lattice_mesh = lattice
        self.current_view_mode = "lattice"
        
        # 刷新视图
        self._refresh_view()
    
    def _on_export_combined_lattice(self) -> None:
        """导出合并的晶格"""
        if self.combined_region_lattice is None:
            self._show_warning("提示", "请先生成区域晶格")
            return
        
        # 默认保存到 results/ 文件夹
        default_name = "region_lattice.stl"
        if RESULTS_DIR.exists():
            default_path = str(RESULTS_DIR / default_name)
        else:
            default_path = default_name
        
        # 选择导出文件路径
        file_path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "导出区域晶格",
            default_path,
            "STL Files (*.stl)"
        )
        
        if file_path:
            try:
                # 导出晶格
                self.combined_region_lattice.export(file_path)
                self._append_status(f"✓ 区域晶格已导出: {Path(file_path).name}")
                self._append_status(f"  面数: {len(self.combined_region_lattice.faces)}")
                self._append_status(f"  顶点数: {len(self.combined_region_lattice.vertices)}")
                
                self._show_information("导出成功", f"晶格已导出到:\n{file_path}\n\n"
                    f"面数: {len(self.combined_region_lattice.faces)}\n"
                    f"顶点数: {len(self.combined_region_lattice.vertices)}"
                )
            except Exception as e:
                self._show_critical("导出失败", f"导出晶格失败:\n{e}")
                self._append_status(f"✗ 导出失败: {e}")
    
    def _on_save_region_config(self) -> None:
        """保存区域配置"""
        if len(self.region_manager.regions) == 0:
            self._show_warning("提示", "没有可保存的区域配置")
            return
        
        file_path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "保存区域配置",
            "region_config.json",
            "JSON Files (*.json)"
        )
        
        if file_path:
            try:
                self.region_manager.save_to_json(file_path)
                self._append_status(f"✓ 配置已保存: {Path(file_path).name}")
                self._show_information("保存成功", f"配置已保存到:\n{file_path}")
            except Exception as e:
                self._show_critical("保存失败", f"保存配置失败:\n{e}")
                self._append_status(f"✗ 保存配置失败: {e}")
    
    def _on_load_region_config(self) -> None:
        """加载区域配置"""
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "加载区域配置",
            "",
            "JSON Files (*.json)"
        )
        
        if file_path:
            try:
                # 定义网格提供函数
                def mesh_provider(source_type: str) -> Optional[trimesh.Trimesh]:
                    return self._get_region_mesh(source_type)
                
                self.region_manager.load_from_json(file_path, mesh_provider)
                self._update_region_display()
                self._append_status(f"✓ 配置已加载: {Path(file_path).name}")
                self._show_information("加载成功", f"已加载 {len(self.region_manager.regions)} 个区域"
                )
            except Exception as e:
                self._show_critical("加载失败", f"加载配置失败:\n{e}")
                self._append_status(f"✗ 加载配置失败: {e}")
    
    # ========================================================================
    # 平面区域晶格设计相关方法（方案B - 推荐）
    # ========================================================================
    
    def _build_plane_region_tab(self, layout: QtWidgets.QVBoxLayout) -> None:
        """构建平面区域晶格设计选项卡（方案B）"""
        
        # 提示信息
        hint = QtWidgets.QLabel(
            "⭐ 推荐方案：使用平面边界定义区域\n"
            "✅ 边界光滑（无锯齿）\n"
            "✅ 灵活调整\n"
            "✅ 支持未来的过渡功能"
        )
        hint.setStyleSheet(
            "color: #2c5aa0; font-size: 11px; padding: 10px; "
            "background: #e3f2fd; border-radius: 4px; border-left: 4px solid #2196F3;"
        )
        hint.setWordWrap(True)
        layout.addWidget(hint)
        
        # === 区域管理 ===
        region_mgmt_group = QtWidgets.QGroupBox("区域管理")
        region_mgmt_layout = QtWidgets.QVBoxLayout(region_mgmt_group)
        
        # 区域统计
        stats_layout = QtWidgets.QHBoxLayout()
        stats_layout.addWidget(QtWidgets.QLabel("区域总数:"))
        self.plane_region_count_label = QtWidgets.QLabel("0")
        self.plane_region_count_label.setStyleSheet("font-weight: bold; font-size: 14px; color: #2196F3;")
        stats_layout.addWidget(self.plane_region_count_label)
        stats_layout.addStretch()
        region_mgmt_layout.addLayout(stats_layout)
        
        # 管理按钮
        self.btn_open_plane_region_manager = QtWidgets.QPushButton("📋 打开区域管理器")
        self.btn_open_plane_region_manager.setToolTip("打开区域管理窗口，添加/删除/配置区域")
        self.btn_open_plane_region_manager.setStyleSheet("padding: 10px; font-size: 13px;")
        self.btn_open_plane_region_manager.clicked.connect(self._open_plane_region_manager_dialog)
        region_mgmt_layout.addWidget(self.btn_open_plane_region_manager)
        
        layout.addWidget(region_mgmt_group)
        
        # === 操作按钮 ===
        actions_group = QtWidgets.QGroupBox("操作")
        actions_layout = QtWidgets.QVBoxLayout(actions_group)
        
        # 包围盒优化选项
        self.check_enable_bbox_optimization = QtWidgets.QCheckBox("启用包围盒优化（推荐）")
        self.check_enable_bbox_optimization.setChecked(True)
        self.check_enable_bbox_optimization.setToolTip(
            "启用后，只在区域范围内生成候选，可节省约50%的内存和计算时间。\n"
            "禁用后，在整个鞋底范围内生成候选（用于对比测试）。"
        )
        self.check_enable_bbox_optimization.setStyleSheet("padding: 5px; font-size: 12px;")
        actions_layout.addWidget(self.check_enable_bbox_optimization)
        
        # 严格平面裁剪选项
        self.check_strict_plane_clip = QtWidgets.QCheckBox("严格平面裁剪")
        self.check_strict_plane_clip.setChecked(False)
        self.check_strict_plane_clip.setToolTip(
            "平面裁剪模式：\n"
            "• 未勾选（默认）：平衡模式 - 面中心+至少2个顶点在区域内\n"
            "  效果：轻微拖尾，结构连续\n"
            "• 勾选：严格模式 - 所有顶点都在区域内\n"
            "  效果：无拖尾，但边界可能有空隙\n\n"
            "推荐：Voronoi方法且禁用包围盒优化时勾选"
        )
        self.check_strict_plane_clip.setStyleSheet("padding: 5px; font-size: 12px;")
        actions_layout.addWidget(self.check_strict_plane_clip)
        
        self.btn_generate_plane_regions = QtWidgets.QPushButton("🚀 生成所有区域晶格")
        self.btn_generate_plane_regions.setStyleSheet("background: #4CAF50; color: white; padding: 12px; font-weight: bold; font-size: 14px;")
        self.btn_generate_plane_regions.clicked.connect(self._on_generate_plane_region_lattices)
        actions_layout.addWidget(self.btn_generate_plane_regions)
        
        self.btn_view_plane_region_result = QtWidgets.QPushButton("👁️ 查看生成结果")
        self.btn_view_plane_region_result.setEnabled(False)
        self.btn_view_plane_region_result.setStyleSheet("padding: 10px; font-size: 13px;")
        self.btn_view_plane_region_result.setToolTip("在主视图中显示生成的平面区域晶格结果")
        self.btn_view_plane_region_result.clicked.connect(self._display_plane_region_result)
        actions_layout.addWidget(self.btn_view_plane_region_result)
        
        self.btn_select_view_plane_region = QtWidgets.QPushButton("🔍 选择查看区域")
        self.btn_select_view_plane_region.setEnabled(False)
        self.btn_select_view_plane_region.setStyleSheet("padding: 10px; font-size: 13px;")
        self.btn_select_view_plane_region.setToolTip("选择要查看的单个区域或完整晶格")
        self.btn_select_view_plane_region.clicked.connect(self._open_select_plane_region_view_dialog)
        actions_layout.addWidget(self.btn_select_view_plane_region)
        
        self.btn_export_plane_combined = QtWidgets.QPushButton("💾 导出合并晶格")
        self.btn_export_plane_combined.setEnabled(False)
        self.btn_export_plane_combined.setStyleSheet("padding: 10px; font-size: 13px;")
        self.btn_export_plane_combined.clicked.connect(self._on_export_plane_combined_lattice)
        actions_layout.addWidget(self.btn_export_plane_combined)
        
        layout.addWidget(actions_group)
        
        layout.addStretch()  # 添加弹性空间
    
    
    def _open_plane_region_manager_dialog(self) -> None:
        """打开平面区域管理器对话框"""
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle("平面区域管理器")
        dialog.setModal(False)  # 非模态，可以同时操作主窗口
        dialog.resize(800, 700)
        
        # 应用深色主题
        self._apply_dark_dialog_style(dialog)
        
        main_layout = QtWidgets.QVBoxLayout(dialog)
        
        # === 顶部：操作按钮 ===
        btn_layout = QtWidgets.QHBoxLayout()
        btn_add = QtWidgets.QPushButton("➕ 添加区域")
        btn_remove = QtWidgets.QPushButton("➖ 删除区域")
        btn_add.clicked.connect(lambda: self._on_add_plane_region_from_dialog(dialog))
        btn_remove.clicked.connect(lambda: self._on_remove_plane_region_from_dialog(dialog))
        btn_layout.addWidget(btn_add)
        btn_layout.addWidget(btn_remove)
        btn_layout.addStretch()
        main_layout.addLayout(btn_layout)
        
        # === 中部：左右分栏 ===
        content_layout = QtWidgets.QHBoxLayout()
        
        # 左侧：区域列表
        left_panel = QtWidgets.QGroupBox("区域列表")
        left_layout = QtWidgets.QVBoxLayout(left_panel)
        
        region_list = QtWidgets.QListWidget()
        region_list.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        left_layout.addWidget(region_list)
        
        content_layout.addWidget(left_panel, stretch=1)
        
        # 右侧：参数配置
        right_panel = QtWidgets.QGroupBox("晶格参数配置")
        right_layout = QtWidgets.QVBoxLayout(right_panel)
        
        # 推荐参数按钮
        btn_recommend = QtWidgets.QPushButton("💡 推荐参数")
        btn_recommend.setToolTip("根据鞋底尺寸推荐当前区域的晶格参数")
        btn_recommend.clicked.connect(self._on_plane_region_recommend_params)
        btn_recommend.setEnabled(False)
        right_layout.addWidget(btn_recommend)
        
        # 晶格方法选择
        method_layout = QtWidgets.QHBoxLayout()
        method_layout.addWidget(QtWidgets.QLabel("晶格方法:"))
        method_combo = QtWidgets.QComboBox()
        # 设置下拉列表的最大可见项数
        method_combo.setMaxVisibleItems(10)
        method_combo.addItems(["gyroid", "voronoi_2_5d", "tile_unit"])
        method_layout.addWidget(method_combo)
        method_layout.addStretch()
        right_layout.addLayout(method_layout)
        
        # Gyroid 参数组
        gyroid_group = self._create_plane_region_gyroid_params_for_dialog()
        right_layout.addWidget(gyroid_group)
        
        # Voronoi 参数组
        voronoi_group = self._create_plane_region_voronoi_params_for_dialog()
        right_layout.addWidget(voronoi_group)
        
        # Tile 参数组
        tile_group = self._create_plane_region_tile_params_for_dialog()
        right_layout.addWidget(tile_group)
        
        # 提示标签
        params_hint = QtWidgets.QLabel("请先选择一个区域")
        params_hint.setStyleSheet("color: #cccccc; font-style: italic; padding: 10px;")
        params_hint.setAlignment(QtCore.Qt.AlignCenter)
        right_layout.addWidget(params_hint)
        
        right_layout.addStretch()
        content_layout.addWidget(right_panel, stretch=2)
        
        main_layout.addLayout(content_layout)
        
        # === 底部：配置管理和关闭按钮 ===
        bottom_layout = QtWidgets.QHBoxLayout()
        
        btn_save = QtWidgets.QPushButton("💾 保存配置")
        btn_load = QtWidgets.QPushButton("📂 加载配置")
        btn_save.clicked.connect(self._on_save_plane_region_config)
        btn_load.clicked.connect(lambda: [self._on_load_plane_region_config(), refresh_list()])
        bottom_layout.addWidget(btn_save)
        bottom_layout.addWidget(btn_load)
        bottom_layout.addStretch()
        
        btn_close = QtWidgets.QPushButton("关闭")
        btn_close.clicked.connect(dialog.close)
        bottom_layout.addWidget(btn_close)
        
        main_layout.addLayout(bottom_layout)
        
        # === 辅助函数 ===
        def refresh_list():
            """刷新区域列表"""
            region_list.clear()
            for region in self.plane_region_manager.regions:
                item = QtWidgets.QListWidgetItem(region.get_display_name())
                item.setData(QtCore.Qt.UserRole, region.id)
                item.setToolTip(region.get_boundary_description())
                region_list.addItem(item)
                if region.id == self.selected_plane_region_id:
                    item.setSelected(True)
            self._update_plane_region_display()
        
        def set_params_enabled(enabled: bool):
            """启用/禁用参数配置"""
            method_combo.setEnabled(enabled)
            gyroid_group.setEnabled(enabled)
            voronoi_group.setEnabled(enabled)
            tile_group.setEnabled(enabled)
            btn_recommend.setEnabled(enabled)
            
            if enabled:
                params_hint.hide()
                sync_method_fields()
            else:
                params_hint.show()
        
        def sync_method_fields():
            """同步参数组显示"""
            method = method_combo.currentText().strip().lower()
            gyroid_group.setVisible(method == "gyroid")
            voronoi_group.setVisible(method == "voronoi_2_5d")
            tile_group.setVisible(method == "tile_unit")
        
        def on_selection_changed():
            """选择变化"""
            current_item = region_list.currentItem()
            if current_item:
                region_id = current_item.data(QtCore.Qt.UserRole)
                self.selected_plane_region_id = region_id
                load_parameters(region_id, method_combo, gyroid_group, voronoi_group, tile_group)
                set_params_enabled(True)
            else:
                self.selected_plane_region_id = None
                set_params_enabled(False)
            self._update_plane_region_display()
        
        def load_parameters(region_id, method_combo, gyroid_group, voronoi_group, tile_group):
            """加载区域参数"""
            region = self.plane_region_manager.get_region(region_id)
            if region is None:
                return
            
            # 获取对话框中的控件
            gyroid_widgets = {
                'cell': gyroid_group.findChild(QtWidgets.QDoubleSpinBox, 'gyroid_cell'),
                'iso': gyroid_group.findChild(QtWidgets.QDoubleSpinBox, 'gyroid_iso'),
                'res': gyroid_group.findChild(QtWidgets.QSpinBox, 'gyroid_res')
            }
            voronoi_widgets = {
                'cell': voronoi_group.findChild(QtWidgets.QDoubleSpinBox, 'voronoi_cell'),
                'thickness': voronoi_group.findChild(QtWidgets.QDoubleSpinBox, 'voronoi_thickness'),
                'layers': voronoi_group.findChild(QtWidgets.QSpinBox, 'voronoi_layers')
            }
            tile_widgets = {
                'shrink': tile_group.findChild(QtWidgets.QDoubleSpinBox, 'tile_shrink'),
                'spacing': tile_group.findChild(QtWidgets.QDoubleSpinBox, 'tile_spacing'),
                'margin': tile_group.findChild(QtWidgets.QDoubleSpinBox, 'tile_margin'),
                'decimate': tile_group.findChild(QtWidgets.QSpinBox, 'tile_decimate')
            }
            
            # 阻止信号
            method_combo.blockSignals(True)
            for w in gyroid_widgets.values():
                if w: w.blockSignals(True)
            for w in voronoi_widgets.values():
                if w: w.blockSignals(True)
            for w in tile_widgets.values():
                if w: w.blockSignals(True)
            
            # 设置值
            method_combo.setCurrentText(region.lattice_method)
            if gyroid_widgets['cell']: gyroid_widgets['cell'].setValue(region.gyroid_cell_size)
            if gyroid_widgets['iso']: gyroid_widgets['iso'].setValue(region.gyroid_isovalue)
            if gyroid_widgets['res']: gyroid_widgets['res'].setValue(region.gyroid_resolution)
            if voronoi_widgets['cell']: voronoi_widgets['cell'].setValue(region.voronoi_cell_size)
            if voronoi_widgets['thickness']: voronoi_widgets['thickness'].setValue(region.voronoi_strut_thickness)
            if voronoi_widgets['layers']: voronoi_widgets['layers'].setValue(region.voronoi_z_layers)
            if tile_widgets['shrink']: tile_widgets['shrink'].setValue(region.tile_shrink)
            if tile_widgets['spacing']: tile_widgets['spacing'].setValue(region.tile_spacing)
            if tile_widgets['margin']: tile_widgets['margin'].setValue(region.tile_margin)
            if tile_widgets['decimate']: tile_widgets['decimate'].setValue(region.tile_decimate_target_faces)
            
            # 恢复信号
            method_combo.blockSignals(False)
            for w in gyroid_widgets.values():
                if w: w.blockSignals(False)
            for w in voronoi_widgets.values():
                if w: w.blockSignals(False)
            for w in tile_widgets.values():
                if w: w.blockSignals(False)
            
            sync_method_fields()
        
        def on_param_changed():
            """参数改变"""
            if self.selected_plane_region_id is None:
                return
            
            region = self.plane_region_manager.get_region(self.selected_plane_region_id)
            if region is None:
                return
            
            # 获取对话框中的控件
            gyroid_widgets = {
                'cell': gyroid_group.findChild(QtWidgets.QDoubleSpinBox, 'gyroid_cell'),
                'iso': gyroid_group.findChild(QtWidgets.QDoubleSpinBox, 'gyroid_iso'),
                'res': gyroid_group.findChild(QtWidgets.QSpinBox, 'gyroid_res')
            }
            voronoi_widgets = {
                'cell': voronoi_group.findChild(QtWidgets.QDoubleSpinBox, 'voronoi_cell'),
                'thickness': voronoi_group.findChild(QtWidgets.QDoubleSpinBox, 'voronoi_thickness'),
                'layers': voronoi_group.findChild(QtWidgets.QSpinBox, 'voronoi_layers')
            }
            tile_widgets = {
                'shrink': tile_group.findChild(QtWidgets.QDoubleSpinBox, 'tile_shrink'),
                'spacing': tile_group.findChild(QtWidgets.QDoubleSpinBox, 'tile_spacing'),
                'margin': tile_group.findChild(QtWidgets.QDoubleSpinBox, 'tile_margin'),
                'decimate': tile_group.findChild(QtWidgets.QSpinBox, 'tile_decimate')
            }
            
            # 更新区域参数
            region.lattice_method = method_combo.currentText()
            if gyroid_widgets['cell']: region.gyroid_cell_size = gyroid_widgets['cell'].value()
            if gyroid_widgets['iso']: region.gyroid_isovalue = gyroid_widgets['iso'].value()
            if gyroid_widgets['res']: region.gyroid_resolution = gyroid_widgets['res'].value()
            if voronoi_widgets['cell']: region.voronoi_cell_size = voronoi_widgets['cell'].value()
            if voronoi_widgets['thickness']: region.voronoi_strut_thickness = voronoi_widgets['thickness'].value()
            if voronoi_widgets['layers']: region.voronoi_z_layers = voronoi_widgets['layers'].value()
            if tile_widgets['shrink']: region.tile_shrink = tile_widgets['shrink'].value()
            if tile_widgets['spacing']: region.tile_spacing = tile_widgets['spacing'].value()
            if tile_widgets['margin']: region.tile_margin = tile_widgets['margin'].value()
            if tile_widgets['decimate']:
                region.tile_decimate_target_faces = tile_widgets['decimate'].value()
                # 同步到全局设置（与选项卡1保持一致）
                self.lattice_unit_target_faces = tile_widgets['decimate'].value()
                if hasattr(self, 'spin_tile_decimate'):
                    self.spin_tile_decimate.blockSignals(True)
                    self.spin_tile_decimate.setValue(tile_widgets['decimate'].value())
                    self.spin_tile_decimate.blockSignals(False)
        
        # 连接信号
        region_list.itemSelectionChanged.connect(on_selection_changed)
        method_combo.currentTextChanged.connect(lambda: [sync_method_fields(), on_param_changed()])
        
        # 连接参数变化信号
        for group in [gyroid_group, voronoi_group, tile_group]:
            for widget in group.findChildren(QtWidgets.QDoubleSpinBox):
                widget.valueChanged.connect(on_param_changed)
            for widget in group.findChildren(QtWidgets.QSpinBox):
                widget.valueChanged.connect(on_param_changed)
        
        # 初始刷新
        refresh_list()
        set_params_enabled(False)
        
        # 保存引用
        dialog.refresh_list = refresh_list
        self._plane_region_manager_dialog = dialog
        
        dialog.show()
    
    def _create_plane_region_gyroid_params_for_dialog(self) -> QtWidgets.QGroupBox:
        """为对话框创建 Gyroid 参数组"""
        group = QtWidgets.QGroupBox("Gyroid 参数")
        layout = QtWidgets.QFormLayout(group)
        
        cell = QtWidgets.QDoubleSpinBox()
        cell.setObjectName('gyroid_cell')
        cell.setRange(0.5, 50.0)
        cell.setDecimals(2)
        cell.setValue(4.0)
        
        iso = QtWidgets.QDoubleSpinBox()
        iso.setObjectName('gyroid_iso')
        iso.setRange(-1.5, 1.5)
        iso.setDecimals(3)
        iso.setValue(0.3)
        
        res = QtWidgets.QSpinBox()
        res.setObjectName('gyroid_res')
        res.setRange(8, 128)
        res.setValue(30)
        
        layout.addRow("单胞尺寸", cell)
        layout.addRow("等值面", iso)
        layout.addRow("分辨率", res)
        
        return group
    
    def _create_plane_region_voronoi_params_for_dialog(self) -> QtWidgets.QGroupBox:
        """为对话框创建 Voronoi 参数组"""
        group = QtWidgets.QGroupBox("Voronoi 参数")
        layout = QtWidgets.QFormLayout(group)
        
        cell = QtWidgets.QDoubleSpinBox()
        cell.setObjectName('voronoi_cell')
        cell.setRange(1.0, 100.0)
        cell.setDecimals(2)
        cell.setValue(4.2)
        
        thickness = QtWidgets.QDoubleSpinBox()
        thickness.setObjectName('voronoi_thickness')
        thickness.setRange(0.1, 20.0)
        thickness.setDecimals(2)
        thickness.setValue(0.55)
        
        layers = QtWidgets.QSpinBox()
        layers.setObjectName('voronoi_layers')
        layers.setRange(2, 30)
        layers.setValue(7)
        
        layout.addRow("胞元尺寸", cell)
        layout.addRow("杆径", thickness)
        layout.addRow("Z 向层数", layers)
        
        return group
    
    def _create_plane_region_tile_params_for_dialog(self) -> QtWidgets.QGroupBox:
        """为对话框创建 Tile 参数组"""
        group = QtWidgets.QGroupBox("Tile 参数")
        layout = QtWidgets.QFormLayout(group)
        
        shrink = QtWidgets.QDoubleSpinBox()
        shrink.setObjectName('tile_shrink')
        shrink.setRange(0.05, 2.0)
        shrink.setDecimals(3)
        shrink.setValue(0.28)
        
        spacing = QtWidgets.QDoubleSpinBox()
        spacing.setObjectName('tile_spacing')
        spacing.setRange(0.5, 1.5)
        spacing.setDecimals(3)
        spacing.setValue(0.96)
        
        margin = QtWidgets.QDoubleSpinBox()
        margin.setObjectName('tile_margin')
        margin.setRange(0.0, 5.0)
        margin.setDecimals(3)
        margin.setValue(1.0)
        
        decimate = QtWidgets.QSpinBox()
        decimate.setObjectName('tile_decimate')
        decimate.setRange(0, 10000)
        decimate.setValue(int(self.lattice_unit_target_faces))
        decimate.setSingleStep(100)
        decimate.setSpecialValueText("禁用减面")
        decimate.setToolTip("减面目标面数（0=禁用，保持最高精度）\n减面会降低晶格单元的面数，影响精度")
        
        layout.addRow("单元尺寸缩放", shrink)
        layout.addRow("平铺间距系数", spacing)
        layout.addRow("边界安全距离", margin)
        layout.addRow("减面目标面数", decimate)
        
        return group
    
    def _on_add_plane_region_from_dialog(self, dialog) -> None:
        """从对话框添加平面区域"""
        # 防止双击
        sender = self.sender()
        if sender and hasattr(sender, 'setEnabled'):
            sender.setEnabled(False)
            QtCore.QTimer.singleShot(500, lambda: sender.setEnabled(True))
        
        self._on_add_plane_region()
        if hasattr(dialog, 'refresh_list'):
            dialog.refresh_list()
        # 标记工程已修改（添加了平面区域）
        self._mark_project_modified()
    
    def _on_remove_plane_region_from_dialog(self, dialog) -> None:
        """从对话框删除平面区域"""
        if self.selected_plane_region_id is None:
            msgbox = QtWidgets.QMessageBox(dialog)
            msgbox.setWindowTitle("提示")
            msgbox.setText("请先选择一个区域")
            msgbox.setIcon(QtWidgets.QMessageBox.Warning)
            self._apply_dark_messagebox_style(msgbox)
            msgbox.exec_()
            return
        
        region = self.plane_region_manager.get_region(self.selected_plane_region_id)
        if region is None:
            return
        
        msgbox = QtWidgets.QMessageBox(dialog)
        msgbox.setWindowTitle("确认删除")
        msgbox.setText(f"确定要删除区域 '{region.name}' 吗？")
        msgbox.setStandardButtons(QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No)
        msgbox.setDefaultButton(QtWidgets.QMessageBox.No)
        msgbox.setIcon(QtWidgets.QMessageBox.Question)
        self._apply_dark_messagebox_style(msgbox)
        
        reply = msgbox.exec_()
        
        if reply == QtWidgets.QMessageBox.Yes:
            self.plane_region_manager.remove_region(self.selected_plane_region_id)
            self.selected_plane_region_id = None
            if hasattr(dialog, 'refresh_list'):
                dialog.refresh_list()
            # 标记工程已修改（删除了平面区域）
            self._mark_project_modified()
    
    def _update_plane_region_display(self) -> None:
        """更新平面区域显示信息"""
        # 更新区域总数（如果标签存在）
        if hasattr(self, 'plane_region_count_label'):
            self.plane_region_count_label.setText(str(len(self.plane_region_manager.regions)))
    
    def _on_add_plane_region(self) -> None:
        """添加平面区域"""
        # 创建对话框
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle("添加平面区域")
        dialog.setModal(True)
        dialog.resize(400, 300)
        
        # 应用深色主题
        self._apply_dark_dialog_style(dialog)
        
        layout = QtWidgets.QVBoxLayout(dialog)
        
        # 提示
        label = QtWidgets.QLabel("请选择区域边界定义方式:")
        layout.addWidget(label)
        
        # 单选按钮组
        radio_group = QtWidgets.QButtonGroup(dialog)
        
        radio_whole = QtWidgets.QRadioButton("无边界（使用整体鞋底）")
        radio_whole.setChecked(True)
        radio_group.addButton(radio_whole)
        layout.addWidget(radio_whole)
        
        layout.addSpacing(10)
        
        radio_plane = QtWidgets.QRadioButton("使用当前分割平面作为边界")
        radio_group.addButton(radio_plane)
        layout.addWidget(radio_plane)
        
        # 平面选项（缩进）
        plane_widget = QtWidgets.QWidget()
        plane_layout = QtWidgets.QVBoxLayout(plane_widget)
        plane_layout.setContentsMargins(30, 0, 0, 0)
        
        if self.slicing_enabled:
            if hasattr(self, 'slice_axis_combo'):
                axis = ['X', 'Y', 'Z'][self.slice_axis_combo.currentIndex()]
            else:
                axis = 'Y'
            position = self._get_slice_position()
            if hasattr(self, 'slice_angle1_spin'):
                angle1 = self.slice_angle1_spin.value()
            else:
                angle1 = 0.0
            if hasattr(self, 'slice_angle2_spin'):
                angle2 = self.slice_angle2_spin.value()
            else:
                angle2 = 0.0
            
            plane_info = f"当前分割平面: {axis} 轴, 位置 = {position:.2f} mm"
            if angle1 != 0.0 or angle2 != 0.0:
                plane_info += f"\n角度: {angle1:.1f}° / {angle2:.1f}°"
            
            plane_layout.addWidget(QtWidgets.QLabel(plane_info))
        else:
            plane_layout.addWidget(QtWidgets.QLabel(
                "⚠️ 请先在'模型分割'选项卡中设置分割平面"
            ))
        
        plane_side_group = QtWidgets.QButtonGroup(dialog)
        radio_positive = QtWidgets.QRadioButton(f"保留正侧（≥ 分割位置）")
        radio_positive.setChecked(True)
        radio_negative = QtWidgets.QRadioButton(f"保留负侧（< 分割位置）")
        plane_side_group.addButton(radio_positive)
        plane_side_group.addButton(radio_negative)
        plane_layout.addWidget(radio_positive)
        plane_layout.addWidget(radio_negative)
        
        layout.addWidget(plane_widget)
        plane_widget.setEnabled(False)
        radio_plane.toggled.connect(plane_widget.setEnabled)
        
        layout.addSpacing(10)
        
        # 说明
        info = QtWidgets.QLabel(
            "💡 提示：\n"
            "• 平面边界不会实际分割 STL 模型\n"
            "• 只在生成晶格时用平面裁剪\n"
            "• 边界光滑，无锯齿"
        )
        info.setStyleSheet("color: #ffffff; font-size: 10px; padding: 8px; background: #3a3a3a; border-radius: 4px;")
        info.setWordWrap(True)
        layout.addWidget(info)
        
        layout.addStretch()
        
        # 按钮
        button_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
        )
        button_box.accepted.connect(dialog.accept)
        button_box.rejected.connect(dialog.reject)
        layout.addWidget(button_box)
        
        # 显示对话框
        if dialog.exec_() == QtWidgets.QDialog.Accepted:
            planes = []
            
            if radio_plane.isChecked():
                # 使用平面边界
                # 检查是否有分割信息（可能来自分割对话框）
                if not self.slicing_enabled or self.slicer is None:
                    self._show_warning("错误", "请先在分割对话框中启用分割功能并设置分割平面")
                    return
                
                if not hasattr(self, 'slice_axis_combo'):
                    # 如果没有旧UI的控件，使用默认值
                    axis = 'y'
                    position = self.current_sole_mesh.bounds.mean(axis=0)[1]
                    angle1 = 0.0
                    angle2 = 0.0
                else:
                    axis = ['x', 'y', 'z'][self.slice_axis_combo.currentIndex()]
                    position = self._get_slice_position()
                    angle1 = self.slice_angle1_spin.value() if hasattr(self, 'slice_angle1_spin') else 0.0
                    angle2 = self.slice_angle2_spin.value() if hasattr(self, 'slice_angle2_spin') else 0.0
                
                side = "positive" if radio_positive.isChecked() else "negative"
                
                planes = [{
                    "axis": axis,
                    "position": position,
                    "side": side,
                    "angle1": angle1,
                    "angle2": angle2
                }]
            
            # 添加区域
            region = self.plane_region_manager.add_region(planes)
            self._append_status(f"已添加平面区域: {region.get_display_name()}")
            
            # 更新显示
            self._update_plane_region_display()
    
    def _on_plane_region_recommend_params(self) -> None:
        """为当前平面区域推荐参数"""
        if self.selected_plane_region_id is None:
            return
        
        region = self.plane_region_manager.get_region(self.selected_plane_region_id)
        if region is None:
            return
        
        # 获取鞋底尺寸
        sole_extents = self._load_mesh_extents(self.edit_sole_path.text().strip())
        if sole_extents is None:
            self._show_warning("提示", "无法加载鞋底模型，请先加载鞋底STL文件")
            return
        
        lattice_extents = self._load_mesh_extents(self.edit_lattice_path.text().strip())
        
        method = region.lattice_method
        
        try:
            if method == "gyroid":
                params = self._recommended_gyroid_params(sole_extents)
                
                # 直接更新区域参数
                self.plane_region_manager.update_region_parameters(
                    self.selected_plane_region_id,
                    gyroid_cell_size=float(params["gyroid_cell"]),
                    gyroid_isovalue=float(params["gyroid_iso"]),
                    gyroid_resolution=int(params["gyroid_res"])
                )
                
            elif method == "voronoi_2_5d":
                params = self._recommended_voronoi_params(sole_extents)
                
                # 直接更新区域参数
                self.plane_region_manager.update_region_parameters(
                    self.selected_plane_region_id,
                    voronoi_cell_size=float(params["voronoi_cell"]),
                    voronoi_strut_thickness=float(params["voronoi_thickness"]),
                    voronoi_z_layers=int(params["voronoi_layers"])
                )
                
            elif method == "tile_unit":
                if lattice_extents is None:
                    self._show_warning("提示", "Tile方法需要晶格单元文件，请先加载晶格STL文件")
                    return
                
                params = self._recommended_tile_params(sole_extents, lattice_extents)
                
                # 直接更新区域参数
                self.plane_region_manager.update_region_parameters(
                    self.selected_plane_region_id,
                    tile_shrink=float(params["tile_shrink"]),
                    tile_spacing=float(params["tile_spacing"]),
                    tile_margin=float(params["tile_margin"])
                )
            
            else:
                self._show_warning("提示", f"未知的晶格方法: {method}")
                return
            
            self._append_status(f"已为区域 '{region.name}' 推荐参数（{method}）")
            
            # 如果对话框打开，刷新参数显示
            if hasattr(self, '_plane_region_manager_dialog') and self._plane_region_manager_dialog is not None:
                # 触发对话框刷新（通过重新选择当前区域）
                if hasattr(self._plane_region_manager_dialog, 'refresh_list'):
                    self._plane_region_manager_dialog.refresh_list()
            
            self._show_information("成功", f"已为区域 '{region.name}' 推荐参数")
            
        except Exception as e:
            self._append_status(f"推荐参数失败: {e}")
            self._show_warning("错误", f"推荐参数失败: {e}")
    
    def _on_generate_plane_region_lattices(self) -> None:
        """生成所有平面区域的晶格"""
        # 验证区域列表
        if len(self.plane_region_manager.regions) == 0:
            self._show_warning("提示", "请先添加至少一个区域")
            return
        
        # 获取用于生成晶格的鞋底模型（优先使用原始完整鞋底）
        if self.original_sole_mesh.vertices.size > 0:
            # 如果已启用分割功能，使用原始完整鞋底
            sole_mesh_for_lattice = self.original_sole_mesh
        else:
            # 如果未启用分割功能，使用当前鞋底
            sole_mesh_for_lattice = self.current_sole_mesh
        
        # 验证鞋底模型
        if sole_mesh_for_lattice.faces.size == 0:
            self._show_warning("提示", "请先加载鞋底模型")
            return
        
        # 清空之前的结果
        self.plane_region_manager.clear_all_lattices()
        self.combined_plane_region_lattice = None
        
        self._append_status("=" * 50)
        self._append_status("开始生成平面区域晶格...")
        self._append_status(f"区域数量: {len(self.plane_region_manager.regions)}")
        
        # 设置平面裁剪模式
        import config
        strict_mode = self.check_strict_plane_clip.isChecked() if hasattr(self, 'check_strict_plane_clip') else False
        config.PLANE_CLIP_STRICT_MODE = strict_mode
        
        if strict_mode:
            self._append_status("平面裁剪模式: 严格（无拖尾，可能有空隙）")
        else:
            self._append_status("平面裁剪模式: 平衡（面中心+至少2个顶点）")
        
        try:
            # 遍历所有区域
            for i, region in enumerate(self.plane_region_manager.regions, 1):
                self._append_status(f"\n[{i}/{len(self.plane_region_manager.regions)}] 处理区域: {region.name}")
                self._append_status(f"  边界: {region.get_boundary_description()}")
                self._append_status(f"  方法: {region.lattice_method}")
                
                # 在完整鞋底上生成晶格（使用原始完整鞋底）
                lattice_mesh = self._generate_lattice_for_region(region, sole_mesh=sole_mesh_for_lattice)
                
                if lattice_mesh is None or lattice_mesh.faces.size == 0:
                    self._append_status(f"  ✗ 区域 {region.name} 晶格生成失败")
                    continue
                
                self._append_status(f"  ✓ 生成晶格: {len(lattice_mesh.faces)} 面")
                
                # 用平面裁剪晶格
                if len(region.boundary_planes) > 0:
                    clipped_lattice = clip_lattice_by_planes(lattice_mesh, region.boundary_planes)
                    self._append_status(f"  ✓ 平面裁剪: {len(lattice_mesh.faces)} -> {len(clipped_lattice.faces)} 面")
                    lattice_mesh = clipped_lattice
                
                # 保存到区域管理器
                self.plane_region_manager.set_region_lattice(region.id, lattice_mesh)
            
            # 合并所有区域的晶格
            self.combined_plane_region_lattice = self.plane_region_manager.get_combined_lattice()
            
            if self.combined_plane_region_lattice is None or self.combined_plane_region_lattice.faces.size == 0:
                self._append_status("\n✗ 没有生成任何晶格")
                self._show_warning("生成失败", "没有生成任何晶格，请检查参数设置")
                return
            
            self._append_status(f"\n✓ 合并完成: {len(self.combined_plane_region_lattice.faces)} 面")
            
            # 启用查看和导出按钮
            if hasattr(self, 'btn_view_plane_region_result'):
                self.btn_view_plane_region_result.setEnabled(True)
            if hasattr(self, 'btn_select_view_plane_region'):
                self.btn_select_view_plane_region.setEnabled(True)
            if hasattr(self, 'btn_export_plane_combined'):
                self.btn_export_plane_combined.setEnabled(True)
            
            # 在 3D 视图中显示结果
            self._display_plane_region_result()
            
            self._append_status("=" * 50)
            self._append_status("✓ 平面区域晶格生成完成！")
            
            # 标记工程已修改（生成了新的平面区域晶格结果）
            self._mark_project_modified()
            
            self._show_information("生成成功", f"已成功生成 {len(self.plane_region_manager.regions)} 个区域的晶格\n"
                f"总面数: {len(self.combined_plane_region_lattice.faces)}"
            )
            
        except Exception as e:
            self._append_status(f"\n✗ 生成失败: {e}")
            self._show_critical("生成失败", f"生成晶格时出错:\n{e}")
            import traceback
            traceback.print_exc()
    
    def _create_region_bounding_box(
        self,
        boundary_planes: list,
        sole_mesh: trimesh.Trimesh,
        margin: float = 2.0
    ) -> tuple[trimesh.Trimesh, np.ndarray]:
        """
        为平面区域创建包围盒
        
        策略：
        - 返回完整鞋底网格（水密，用于体内判断）
        - 同时返回区域AABB包围盒（用于减少候选数量）
        
        参数:
            boundary_planes: 平面边界列表
            sole_mesh: 完整鞋底网格
            margin: 包围盒扩展边距（mm）
        
        返回:
            (完整鞋底网格, 区域AABB包围盒)
        """
        # 获取完整鞋底的包围盒
        sole_bounds = sole_mesh.bounds.copy()
        region_bounds = sole_bounds.copy()
        
        axis_map = {'x': 0, 'y': 1, 'z': 2}
        
        # 根据平面调整区域包围盒
        for plane in boundary_planes:
            axis = plane['axis'].lower()
            position = plane['position']
            side = plane['side']
            angle1 = plane.get('angle1', 0.0)
            angle2 = plane.get('angle2', 0.0)
            
            axis_idx = axis_map[axis]
            
            if angle1 == 0.0 and angle2 == 0.0:
                # 无旋转：精确的轴对齐包围盒
                if side == 'positive':
                    region_bounds[0, axis_idx] = max(region_bounds[0, axis_idx], position)
                else:  # negative
                    region_bounds[1, axis_idx] = min(region_bounds[1, axis_idx], position)
            else:
                # 有旋转：保守估计，扩展包围盒
                max_angle = max(abs(angle1), abs(angle2))
                angle_rad = np.radians(max_angle)
                
                # 计算其他轴的最大范围
                other_axes = [i for i in range(3) if i != axis_idx]
                max_other_extent = max(
                    region_bounds[1, other_axes[0]] - region_bounds[0, other_axes[0]],
                    region_bounds[1, other_axes[1]] - region_bounds[0, other_axes[1]]
                )
                
                # 估算扩展量：tan(angle) * max_other_extent
                expansion = abs(np.tan(angle_rad)) * max_other_extent
                
                if side == 'positive':
                    region_bounds[0, axis_idx] = max(
                        region_bounds[0, axis_idx], 
                        position - expansion - margin
                    )
                else:  # negative
                    region_bounds[1, axis_idx] = min(
                        region_bounds[1, axis_idx], 
                        position + expansion + margin
                    )
        
        # 添加小的边距，避免边界数值误差
        if len(boundary_planes) > 0:
            region_bounds[0] -= margin
            region_bounds[1] += margin
            
            # 确保不超出鞋底范围
            region_bounds[0] = np.maximum(region_bounds[0], sole_bounds[0])
            region_bounds[1] = np.minimum(region_bounds[1], sole_bounds[1])
        
        # 验证包围盒有效性
        if np.any(region_bounds[1] <= region_bounds[0]):
            # 如果无效，使用完整鞋底包围盒
            region_bounds = sole_bounds.copy()
        
        # 返回完整鞋底网格和区域包围盒
        return sole_mesh, region_bounds
    

    def _generate_lattice_for_region(
        self, 
        region: PlaneRegion, 
        sole_mesh: trimesh.Trimesh = None
    ) -> trimesh.Trimesh:
        """为单个平面区域生成晶格（可选包围盒优化）
        
        参数:
            region: 平面区域对象
            sole_mesh: 用于生成晶格的鞋底网格（可选，默认使用 current_sole_mesh）
        """
        # 使用传入的鞋底网格，如果没有则使用当前鞋底
        if sole_mesh is None:
            sole_mesh = self.current_sole_mesh
        
        # === 包围盒优化：根据复选框决定是否启用 ===
        region_bounds = None
        use_bbox_optimization = self.check_enable_bbox_optimization.isChecked() if hasattr(self, 'check_enable_bbox_optimization') else True
        
        if use_bbox_optimization and len(region.boundary_planes) > 0:
            # 启用包围盒优化：创建区域包围盒
            try:
                bbox_mesh, region_bounds = self._create_region_bounding_box(
                    boundary_planes=region.boundary_planes,
                    sole_mesh=sole_mesh,
                    margin=0.5
                )
                
                # 计算包围盒信息
                bbox_extents = region_bounds[1] - region_bounds[0]
                sole_bounds = sole_mesh.bounds
                sole_extents = sole_bounds[1] - sole_bounds[0]
                
                # 计算包围盒体积比例
                bbox_bbox_volume = np.prod(bbox_extents)
                sole_bbox_volume = np.prod(sole_extents)
                
                # 检查是否有旋转平面
                has_rotation = any(
                    plane.get('angle1', 0.0) != 0.0 or plane.get('angle2', 0.0) != 0.0
                    for plane in region.boundary_planes
                )
                
                if sole_bbox_volume > 0:
                    ratio = bbox_bbox_volume / sole_bbox_volume * 100
                    
                    if has_rotation:
                        self._append_status(
                            f"  区域包围盒: {bbox_extents[0]:.1f}×{bbox_extents[1]:.1f}×{bbox_extents[2]:.1f} mm "
                            f"({ratio:.1f}% 完整包围盒，有旋转）"
                        )
                    else:
                        self._append_status(
                            f"  区域包围盒: {bbox_extents[0]:.1f}×{bbox_extents[1]:.1f}×{bbox_extents[2]:.1f} mm "
                            f"({ratio:.1f}% 完整包围盒)"
                        )
            except Exception as e:
                self._append_status(f"  ⚠ 创建包围盒失败: {e}，使用完整鞋底")
                region_bounds = None
        elif not use_bbox_optimization:
            # 用户禁用了包围盒优化
            self._append_status(f"  使用完整鞋底（包围盒优化已禁用）")
            region_bounds = None
        else:
            # 无边界限制：使用完整鞋底
            self._append_status(f"  使用完整鞋底（无边界限制）")
            region_bounds = None
        
        # === 在区域包围盒内生成晶格 ===
        if region.lattice_method == "gyroid":
            # 生成 Gyroid 晶格（在区域包围盒内，用完整鞋底判断）
            lattice = geometry.generate_gyroid_lattice(
                sole_mesh=sole_mesh,  # 完整鞋底（用于体内判断）
                cell_size=region.gyroid_cell_size,
                isovalue=region.gyroid_isovalue,
                resolution=region.gyroid_resolution,
                bounds=region_bounds,  # 区域包围盒（用于减少候选）
            )
        
        elif region.lattice_method == "voronoi_2_5d":
            # 生成 Voronoi 晶格（在区域包围盒内，用完整鞋底判断）
            import config
            old_cell_size = config.VORONOI_CELL_SIZE
            old_thickness = config.VORONOI_STRUT_THICKNESS
            old_layers = config.VORONOI_Z_LAYERS
            
            try:
                config.VORONOI_CELL_SIZE = region.voronoi_cell_size
                config.VORONOI_STRUT_THICKNESS = region.voronoi_strut_thickness
                config.VORONOI_Z_LAYERS = region.voronoi_z_layers
                
                lattice = geometry.generate_voronoi_lattice_in_sole(
                    sole_mesh=sole_mesh,  # 完整鞋底（用于体内判断）
                    cell_size=region.voronoi_cell_size,
                    strut_thickness=region.voronoi_strut_thickness,
                    bounds=region_bounds,  # 区域包围盒（用于减少候选）
                )
            finally:
                # 恢复原始配置
                config.VORONOI_CELL_SIZE = old_cell_size
                config.VORONOI_STRUT_THICKNESS = old_thickness
                config.VORONOI_Z_LAYERS = old_layers
        
        elif region.lattice_method == "tile_unit":
            # 生成 Tile 晶格（在区域包围盒内，用完整鞋底判断）
            lattice_path = self.edit_lattice_path.text()
            if not Path(lattice_path).exists():
                self._append_status(f"  ✗ 晶格单元文件不存在: {lattice_path}")
                return trimesh.Trimesh()
            
            # 使用带缓存的生成函数
            lattice = geometry.generate_tile_lattice_with_cache(
                sole_mesh=sole_mesh,  # 完整鞋底（用于体内判断）
                lattice_path=lattice_path,
                shrink=region.tile_shrink,
                spacing=region.tile_spacing,
                margin=region.tile_margin,
                unit_target_faces=region.tile_decimate_target_faces,
                use_cache=True,
                bounds=region_bounds,  # 区域包围盒（用于减少候选）
            )
        
        else:
            self._append_status(f"  ✗ 未知的晶格方法: {region.lattice_method}")
            return trimesh.Trimesh()
        
        # === 第二次精细裁剪：使用多点采样裁剪到鞋底体积 ===
        # 无论是否有边界平面，都执行精细裁剪以保持与选项卡1一致
        # 检查全局精细裁剪开关
        enable_final_trim = self.check_enable_final_trim.isChecked() if hasattr(self, 'check_enable_final_trim') else True
        
        if enable_final_trim and lattice.faces.size > 0:
            faces_before = len(lattice.faces)
            lattice = geometry.trim_lattice_to_sole_volume(sole_mesh, lattice)
            faces_after = len(lattice.faces)
            if faces_before != faces_after:
                self._append_status(f"  精细裁剪: {faces_before} -> {faces_after} 面")
        elif not enable_final_trim:
            self._append_status(f"  跳过精细裁剪（全局开关已禁用）")
        
        return lattice
    
    def _display_plane_region_result(self) -> None:
        """在 3D 视图中显示平面区域晶格结果"""
        if self.combined_plane_region_lattice is None:
            return
        
        # 创建合并视图（鞋底 + 晶格）
        combined = trimesh.util.concatenate([
            self.current_sole_mesh,
            self.combined_plane_region_lattice
        ])
        
        # 保存生成的网格，使其可以通过视图菜单切换
        self.generated_sole_mesh = self.current_sole_mesh
        self.generated_lattice_mesh = self.combined_plane_region_lattice
        self.generated_combined_mesh = combined
        self.has_generated_result = True
        
        # 更新当前视图为只显示晶格
        self.current_lattice_mesh = self.combined_plane_region_lattice
        self.current_view_mode = "lattice"
        
        # 刷新视图
        self._refresh_view()
    
    def _open_select_plane_region_view_dialog(self) -> None:
        """打开选择查看区域的对话框（选项卡4）"""
        if self.combined_plane_region_lattice is None:
            self._show_warning("提示", "请先生成区域晶格")
            return
        
        # 创建对话框
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle("选择查看区域")
        dialog.setModal(True)
        dialog.resize(400, 500)
        
        # 应用深色主题
        self._apply_dark_dialog_style(dialog)
        
        layout = QtWidgets.QVBoxLayout(dialog)
        
        # 提示
        hint = QtWidgets.QLabel("选择要在主视图中显示的内容：")
        hint.setStyleSheet("font-weight: bold; font-size: 12px; padding: 5px;")
        layout.addWidget(hint)
        
        # 列表
        list_widget = QtWidgets.QListWidget()
        list_widget.setStyleSheet("font-size: 12px;")
        
        # 添加"完整晶格"选项
        item_all = QtWidgets.QListWidgetItem("🌐 完整晶格（所有区域合并）")
        item_all.setData(QtCore.Qt.UserRole, "all")
        list_widget.addItem(item_all)
        
        # 添加各个区域选项
        for region in self.plane_region_manager.regions:
            lattice = self.plane_region_manager.get_region_lattice(region.id)
            if lattice is not None and lattice.faces.size > 0:
                face_count = len(lattice.faces)
                item = QtWidgets.QListWidgetItem(f"📦 {region.name} ({face_count:,} 面)")
                item.setData(QtCore.Qt.UserRole, region.id)
                list_widget.addItem(item)
        
        # 默认选中第一项
        if list_widget.count() > 0:
            list_widget.setCurrentRow(0)
        
        layout.addWidget(list_widget)
        
        # 按钮
        button_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
        )
        button_box.accepted.connect(dialog.accept)
        button_box.rejected.connect(dialog.reject)
        layout.addWidget(button_box)
        
        # 显示对话框
        if dialog.exec_() == QtWidgets.QDialog.Accepted:
            current_item = list_widget.currentItem()
            if current_item is None:
                return
            
            selection = current_item.data(QtCore.Qt.UserRole)
            
            self._display_selected_plane_region(selection)
    
    def _display_selected_plane_region(self, selection: str) -> None:
        """显示选中的平面区域（选项卡4）"""
        if selection == "all":
            # 显示完整晶格
            lattice = self.combined_plane_region_lattice
            self._append_status("显示: 完整晶格（所有区域）")
        else:
            # 显示单个区域
            region = self.plane_region_manager.get_region(selection)
            if region is None:
                return
            
            lattice = self.plane_region_manager.get_region_lattice(selection)
            if lattice is None or lattice.faces.size == 0:
                self._show_warning("提示", f"区域 '{region.name}' 没有生成晶格")
                return
            
            self._append_status(f"显示: 区域 '{region.name}'")
        
        # 更新视图（只显示晶格，不显示鞋底）
        self.current_lattice_mesh = lattice
        self.current_view_mode = "lattice"
        
        # 刷新视图
        self._refresh_view()
    
    def _on_export_plane_combined_lattice(self) -> None:
        """导出合并的平面区域晶格"""
        if self.combined_plane_region_lattice is None:
            self._show_warning("提示", "请先生成区域晶格")
            return
        
        # 选择导出文件路径
        file_path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "导出平面区域晶格",
            "plane_region_lattice.stl",
            "STL Files (*.stl)"
        )
        
        if file_path:
            try:
                # 导出晶格
                self.combined_plane_region_lattice.export(file_path)
                self._append_status(f"✓ 平面区域晶格已导出: {Path(file_path).name}")
                self._append_status(f"  面数: {len(self.combined_plane_region_lattice.faces)}")
                self._append_status(f"  顶点数: {len(self.combined_plane_region_lattice.vertices)}")
                
                self._show_information("导出成功", f"晶格已导出到:\n{file_path}\n\n"
                    f"面数: {len(self.combined_plane_region_lattice.faces)}\n"
                    f"顶点数: {len(self.combined_plane_region_lattice.vertices)}"
                )
            except Exception as e:
                self._show_critical("导出失败", f"导出晶格失败:\n{e}")
                self._append_status(f"✗ 导出失败: {e}")
    
    def _on_save_plane_region_config(self) -> None:
        """保存平面区域配置"""
        if len(self.plane_region_manager.regions) == 0:
            self._show_warning("提示", "没有可保存的区域配置")
            return
        
        file_path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "保存平面区域配置",
            "plane_region_config.json",
            "JSON Files (*.json)"
        )
        
        if file_path:
            try:
                self.plane_region_manager.save_to_json(file_path)
                self._append_status(f"✓ 平面区域配置已保存: {Path(file_path).name}")
                self._show_information("保存成功", f"配置已保存到:\n{file_path}")
            except Exception as e:
                self._show_critical("保存失败", f"保存配置失败:\n{e}")
                self._append_status(f"✗ 保存配置失败: {e}")
    
    def _on_load_plane_region_config(self) -> None:
        """加载平面区域配置"""
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "加载平面区域配置",
            "",
            "JSON Files (*.json)"
        )
        
        if file_path:
            try:
                self.plane_region_manager.load_from_json(file_path)
                self._update_plane_region_display()
                self._append_status(f"✓ 平面区域配置已加载: {Path(file_path).name}")
                self._show_information("加载成功", f"已加载 {len(self.plane_region_manager.regions)} 个平面区域"
                )
            except Exception as e:
                self._show_critical("加载失败", f"加载配置失败:\n{e}")
                self._append_status(f"✗ 加载配置失败: {e}")

    def resizeEvent(self, event) -> None:
        """窗口大小改变事件"""
        super().resizeEvent(event)
        
        # 刷新日志栏显示
        if hasattr(self, 'status_box'):
            try:
                self.status_box.update()
                # 确保滚动条位置正确
                cursor = self.status_box.textCursor()
                cursor.movePosition(QtGui.QTextCursor.End)
                self.status_box.setTextCursor(cursor)
            except Exception:
                pass
        
        # 立即强制重绘
        if hasattr(self, 'viewer'):
            try:
                self.viewer.update()
                self.viewer.render()
                # 强制处理所有待处理的事件
                QtWidgets.QApplication.processEvents()
            except Exception as e:
                print(f"[调试] 立即渲染失败: {e}")
        
        # 强制重绘所有区域
        if hasattr(self, 'viewer_host'):
            self.viewer_host.update()
            QtWidgets.QApplication.processEvents()
        
        # 额外的延迟重绘作为备份
        QtCore.QTimer.singleShot(10, self._delayed_repaint)
    
    def showEvent(self, event) -> None:
        """窗口显示事件"""
        super().showEvent(event)
        # 立即重绘
        if hasattr(self, 'viewer'):
            try:
                self.viewer.render()
                QtWidgets.QApplication.processEvents()
            except:
                pass
        # 延迟重绘作为备份
        QtCore.QTimer.singleShot(50, self._delayed_repaint)
    
    def changeEvent(self, event) -> None:
        """窗口状态改变事件（最大化、最小化、还原等）"""
        super().changeEvent(event)
        if event.type() == QtCore.QEvent.WindowStateChange:
            # 立即重绘
            if hasattr(self, 'viewer'):
                try:
                    self.viewer.update()
                    self.viewer.render()
                    QtWidgets.QApplication.processEvents()
                except:
                    pass
            # 延迟重绘作为备份
            QtCore.QTimer.singleShot(50, self._delayed_repaint)
    
    def _delayed_repaint(self) -> None:
        """延迟重绘（作为备份）"""
        if hasattr(self, 'viewer'):
            try:
                self.viewer.update()
                self.viewer.render()
            except:
                pass
    
    def _connect_parameter_signals(self) -> None:
        """连接所有参数变化信号"""
        # 文件路径
        self.edit_sole_path.textChanged.connect(self._mark_project_modified)
        self.edit_lattice_path.textChanged.connect(self._mark_project_modified)
        self.edit_output_path.textChanged.connect(self._mark_project_modified)
        
        # 晶格方法
        self.combo_method.currentTextChanged.connect(self._mark_project_modified)
        self.check_auto_recommend.stateChanged.connect(self._mark_project_modified)
        
        # Gyroid 参数
        self.spin_gyroid_cell.valueChanged.connect(self._mark_project_modified)
        self.spin_gyroid_iso.valueChanged.connect(self._mark_project_modified)
        self.spin_gyroid_res.valueChanged.connect(self._mark_project_modified)
        
        # Voronoi 参数
        self.spin_voronoi_cell.valueChanged.connect(self._mark_project_modified)
        self.spin_voronoi_thickness.valueChanged.connect(self._mark_project_modified)
        self.spin_voronoi_layers.valueChanged.connect(self._mark_project_modified)
        
        # Voronoi Implicit 参数
        self.spin_voronoi_implicit_cell.valueChanged.connect(self._mark_project_modified)
        self.spin_voronoi_implicit_wall.valueChanged.connect(self._mark_project_modified)
        self.spin_voronoi_implicit_res.valueChanged.connect(self._mark_project_modified)
        self.spin_voronoi_implicit_density.valueChanged.connect(self._mark_project_modified)
        
        # Tile 参数
        self.spin_tile_shrink.valueChanged.connect(self._mark_project_modified)
        self.spin_tile_spacing.valueChanged.connect(self._mark_project_modified)
        self.spin_tile_margin.valueChanged.connect(self._mark_project_modified)
        self.spin_tile_decimate.valueChanged.connect(self._mark_project_modified)  # Tile减面参数
        
        # 全局选项
        self.check_enable_final_trim.stateChanged.connect(self._mark_project_modified)
        
        # 模型分割设置
        if hasattr(self, 'check_enable_slice'):
            self.check_enable_slice.stateChanged.connect(self._mark_project_modified)
        if hasattr(self, 'slice_axis_combo'):
            self.slice_axis_combo.currentIndexChanged.connect(self._mark_project_modified)
        if hasattr(self, 'slice_position_slider'):
            self.slice_position_slider.valueChanged.connect(self._mark_project_modified)
        if hasattr(self, 'slice_angle1_spin'):
            self.slice_angle1_spin.valueChanged.connect(self._mark_project_modified)
        if hasattr(self, 'slice_angle2_spin'):
            self.slice_angle2_spin.valueChanged.connect(self._mark_project_modified)
        if hasattr(self, 'slice_subdivide_spin'):
            self.slice_subdivide_spin.valueChanged.connect(self._mark_project_modified)
        if hasattr(self, 'slice_display_combo'):
            self.slice_display_combo.currentIndexChanged.connect(self._mark_project_modified)
        if hasattr(self, 'slice_spacing_spin'):
            self.slice_spacing_spin.valueChanged.connect(self._mark_project_modified)
        if hasattr(self, 'check_show_plane'):
            self.check_show_plane.stateChanged.connect(self._mark_project_modified)
        if hasattr(self, 'check_slice_lighting'):
            self.check_slice_lighting.stateChanged.connect(self._mark_project_modified)
    
    # ==================== 工程管理方法 ====================
    
    def _update_window_title(self) -> None:
        """更新窗口标题"""
        title = "鞋底晶格生成系统"
        if self.project_manager.current_project:
            project_name = self.project_manager.current_project.get_display_name()
            if self.current_project_modified:
                title = f"{project_name}* - {title}"
            else:
                title = f"{project_name} - {title}"
        self.setWindowTitle(title)
    
    def _mark_project_modified(self) -> None:
        """标记工程已修改"""
        self.current_project_modified = True
        self._update_window_title()
    
    def _new_project(self) -> None:
        """创建新工程"""
        # 检查当前工程是否需要保存
        if not self._check_save_current_project():
            return
        
        # 创建新工程
        self.project_manager.create_new_project()
        self.current_project_modified = False
        
        # 重置所有参数到默认值
        self._apply_recommended_params(show_message=False)
        # 清空所有路径，让用户手动选择
        self.edit_sole_path.setText("")
        self.edit_lattice_path.setText("")
        self.edit_output_path.setText("")
        
        self._update_window_title()
        self._append_status("✓ 已创建新工程")
    
    def _open_project(self) -> None:
        """打开工程"""
        # 检查当前工程是否需要保存
        if not self._check_save_current_project():
            return
        
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "打开工程",
            str(Path.home()),
            "工程文件 (*.slp);;所有文件 (*.*)"
        )
        
        if file_path:
            self._load_project_from_file(Path(file_path))
    
    def _load_project_from_file(self, file_path: Path) -> None:
        """从文件加载工程"""
        try:
            project = self.project_manager.load_project(file_path)
            self._apply_project_to_ui(project)
            self.current_project_modified = False
            self._update_window_title()
            self._update_recent_projects_menu()
            self._append_status(f"✓ 已加载工程: {file_path.name}")
        except Exception as e:
            self._show_critical("加载失败", f"加载工程失败:\n{e}")
            self._append_status(f"✗ 加载工程失败: {e}")
    
    def _save_project(self) -> None:
        """保存工程"""
        if self.project_manager.current_project is None:
            self.project_manager.create_new_project()
        
        if self.project_manager.current_project.file_path is None:
            self._save_project_as()
        else:
            self._save_project_to_file(self.project_manager.current_project.file_path)
    
    def _save_project_as(self) -> None:
        """另存为工程"""
        if self.project_manager.current_project is None:
            self.project_manager.create_new_project()
        
        file_path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "保存工程",
            str(Path.home() / "未命名工程.slp"),
            "工程文件 (*.slp);;所有文件 (*.*)"
        )
        
        if file_path:
            self._save_project_to_file(Path(file_path))
    
    def _save_project_to_file(self, file_path: Path) -> None:
        """保存工程到文件"""
        try:
            # 从UI收集当前参数
            self._collect_project_from_ui()
            
            # 保存工程
            self.project_manager.save_project(file_path)
            self.current_project_modified = False
            self._update_window_title()
            self._update_recent_projects_menu()
            self._append_status(f"✓ 工程已保存: {file_path.name}")
        except Exception as e:
            self._show_critical("保存失败", f"保存工程失败:\n{e}")
            self._append_status(f"✗ 保存工程失败: {e}")
    
    def _collect_project_from_ui(self) -> None:
        """从UI收集参数到当前工程"""
        if self.project_manager.current_project is None:
            return
        
        project = self.project_manager.current_project
        
        # 文件路径
        project.sole_path = self.edit_sole_path.text().strip()
        project.lattice_path = self.edit_lattice_path.text().strip()
        project.output_path = self.edit_output_path.text().strip()
        
        # 晶格方法和参数
        project.method = self.combo_method.currentText().strip()
        project.auto_recommend = self.check_auto_recommend.isChecked()
        
        # Gyroid 参数
        project.gyroid_cell = self.spin_gyroid_cell.value()
        project.gyroid_iso = self.spin_gyroid_iso.value()
        project.gyroid_res = self.spin_gyroid_res.value()
        
        # Voronoi 参数
        project.voronoi_cell = self.spin_voronoi_cell.value()
        project.voronoi_thickness = self.spin_voronoi_thickness.value()
        project.voronoi_layers = self.spin_voronoi_layers.value()
        
        # Voronoi Implicit 参数
        project.voronoi_implicit_cell = self.spin_voronoi_implicit_cell.value()
        project.voronoi_implicit_wall = self.spin_voronoi_implicit_wall.value()
        project.voronoi_implicit_res = self.spin_voronoi_implicit_res.value()
        project.voronoi_implicit_density = self.spin_voronoi_implicit_density.value()
        
        # Tile 参数
        project.tile_shrink = self.spin_tile_shrink.value()
        project.tile_spacing = self.spin_tile_spacing.value()
        project.tile_margin = self.spin_tile_margin.value()
        
        # 减面配置
        project.lattice_unit_target_faces = self.lattice_unit_target_faces
        
        # 全局选项
        project.enable_final_trim = self.check_enable_final_trim.isChecked()
        
        # 视图设置
        project.view_mode = self.current_view_mode
        project.mouse_mode = self._mouse_mode
        project.show_mesh_edges = self.action_toggle_mesh_edges.isChecked()
        project.current_sole_material = self.current_sole_material
        project.current_lattice_material = self.current_lattice_material
        project.current_bg_color = self._current_bg_color_index
        
        # 分割设置
        project.slicing_enabled = self.check_enable_slice.isChecked() if hasattr(self, 'check_enable_slice') else False
        if hasattr(self, 'slice_axis_combo'):
            project.slice_axis = self.slice_axis_combo.currentIndex()
        if hasattr(self, 'slice_position_slider'):
            project.slice_position = self.slice_position_slider.value() / 1000.0
        if hasattr(self, 'slice_angle1_spin'):
            project.slice_angle1 = self.slice_angle1_spin.value()
        if hasattr(self, 'slice_angle2_spin'):
            project.slice_angle2 = self.slice_angle2_spin.value()
        if hasattr(self, 'slice_subdivide_spin'):
            project.slice_subdivide = self.slice_subdivide_spin.value()
        if hasattr(self, 'slice_display_combo'):
            project.slice_display_mode = self.slice_display_combo.currentIndex()
        if hasattr(self, 'slice_spacing_spin'):
            project.slice_spacing = self.slice_spacing_spin.value()
        if hasattr(self, 'check_show_plane'):
            project.show_slice_plane = self.check_show_plane.isChecked()
        if hasattr(self, 'check_slice_lighting'):
            project.slice_lighting = self.check_slice_lighting.isChecked()
        
        # 保存生成结果和分割结果到工程数据目录
        self._save_generated_results_to_project(project)
        
        # 保存区域设置
        self._save_regions_to_project(project)
    
    def _save_generated_results_to_project(self, project: Project) -> None:
        """保存生成的晶格结果和分割结果到工程数据目录"""
        import trimesh
        
        # 如果工程还没有保存过，无法确定数据目录
        if project.file_path is None:
            return
        
        # 创建数据目录
        data_dir = project.get_data_dir()
        if data_dir is None:
            return
        data_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存全局生成结果
        if hasattr(self, 'generated_lattice_mesh') and self.generated_lattice_mesh.vertices.size > 0:
            lattice_path = data_dir / "generated_lattice.stl"
            self.generated_lattice_mesh.export(str(lattice_path))
            project.generated_lattice_result_path = f"{project.file_path.stem}_data/generated_lattice.stl"
            project.generation_type = "global"
        
        if hasattr(self, 'generated_sole_mesh') and self.generated_sole_mesh.vertices.size > 0:
            sole_path = data_dir / "generated_sole.stl"
            self.generated_sole_mesh.export(str(sole_path))
            project.generated_sole_result_path = f"{project.file_path.stem}_data/generated_sole.stl"
        
        if hasattr(self, 'generated_combined_mesh') and self.generated_combined_mesh.vertices.size > 0:
            combined_path = data_dir / "generated_combined.stl"
            self.generated_combined_mesh.export(str(combined_path))
            project.generated_combined_result_path = f"{project.file_path.stem}_data/generated_combined.stl"
        
        # 保存区域晶格生成结果
        if hasattr(self, 'region_lattice_meshes') and self.region_lattice_meshes:
            project.region_lattice_results = {}
            for region_id, mesh in self.region_lattice_meshes.items():
                if mesh.vertices.size > 0:
                    region_lattice_path = data_dir / f"region_{region_id}_lattice.stl"
                    mesh.export(str(region_lattice_path))
                    project.region_lattice_results[region_id] = f"{project.file_path.stem}_data/region_{region_id}_lattice.stl"
            project.generation_type = "region"
        
        # 保存平面区域晶格生成结果
        if hasattr(self, 'plane_region_lattice_meshes') and self.plane_region_lattice_meshes:
            project.plane_region_lattice_results = {}
            for region_id, mesh in self.plane_region_lattice_meshes.items():
                if mesh.vertices.size > 0:
                    plane_region_lattice_path = data_dir / f"plane_region_{region_id}_lattice.stl"
                    mesh.export(str(plane_region_lattice_path))
                    project.plane_region_lattice_results[region_id] = f"{project.file_path.stem}_data/plane_region_{region_id}_lattice.stl"
            project.generation_type = "plane_region"
        
        # 保存分割结果
        if hasattr(self, 'positive_mesh') and self.positive_mesh is not None and self.positive_mesh.vertices.size > 0:
            positive_path = data_dir / "sliced_positive.stl"
            self.positive_mesh.export(str(positive_path))
            project.sliced_positive_mesh_path = f"{project.file_path.stem}_data/sliced_positive.stl"
        
        if hasattr(self, 'negative_mesh') and self.negative_mesh is not None and self.negative_mesh.vertices.size > 0:
            negative_path = data_dir / "sliced_negative.stl"
            self.negative_mesh.export(str(negative_path))
            project.sliced_negative_mesh_path = f"{project.file_path.stem}_data/sliced_negative.stl"
    
    def _save_regions_to_project(self, project: Project) -> None:
        """保存区域设置到工程"""
        # 保存Tab 3的区域设置
        project.regions = []
        if hasattr(self, 'region_manager') and self.region_manager.regions:
            for region in self.region_manager.regions:
                region_dict = {
                    'id': region.id,
                    'name': region.name,
                    'source_type': region.source_type,
                    'source_file': region.source_file,
                    'lattice_method': region.lattice_method,
                    'gyroid_cell': region.gyroid_cell_size,
                    'gyroid_iso': region.gyroid_isovalue,
                    'gyroid_res': region.gyroid_resolution,
                    'voronoi_cell': region.voronoi_cell_size,
                    'voronoi_thickness': region.voronoi_strut_thickness,
                    'voronoi_layers': region.voronoi_z_layers,
                    'tile_shrink': region.tile_shrink,
                    'tile_spacing': region.tile_spacing,
                    'tile_margin': region.tile_margin,
                    'tile_decimate_target_faces': region.tile_decimate_target_faces,
                }
                project.regions.append(region_dict)
        
        # 保存Tab 4的平面区域设置
        project.plane_regions = []
        if hasattr(self, 'plane_region_manager') and self.plane_region_manager.regions:
            for region in self.plane_region_manager.regions:
                region_dict = {
                    'id': region.id,
                    'name': region.name,
                    'boundary_planes': region.boundary_planes,  # 直接保存，已经是正确格式
                    'lattice_method': region.lattice_method,
                    'gyroid_cell': region.gyroid_cell_size,
                    'gyroid_iso': region.gyroid_isovalue,
                    'gyroid_res': region.gyroid_resolution,
                    'voronoi_cell': region.voronoi_cell_size,
                    'voronoi_thickness': region.voronoi_strut_thickness,
                    'voronoi_layers': region.voronoi_z_layers,
                    'tile_shrink': region.tile_shrink,
                    'tile_spacing': region.tile_spacing,
                    'tile_margin': region.tile_margin,
                    'tile_decimate_target_faces': region.tile_decimate_target_faces,
                }
                project.plane_regions.append(region_dict)
    
    def _apply_project_to_ui(self, project: Project) -> None:
        """将工程参数应用到UI"""
        self._suspend_auto_recommend = True
        self._loading_project = True  # 标记正在加载工程
        try:
            # 文件路径
            self.edit_sole_path.setText(project.sole_path)
            self.edit_lattice_path.setText(project.lattice_path)
            self.edit_output_path.setText(project.output_path)
            
            # 晶格方法和参数
            self.combo_method.setCurrentText(project.method)
            self.check_auto_recommend.setChecked(project.auto_recommend)
            
            # Gyroid 参数
            self.spin_gyroid_cell.setValue(project.gyroid_cell)
            self.spin_gyroid_iso.setValue(project.gyroid_iso)
            self.spin_gyroid_res.setValue(project.gyroid_res)
            
            # Voronoi 参数
            self.spin_voronoi_cell.setValue(project.voronoi_cell)
            self.spin_voronoi_thickness.setValue(project.voronoi_thickness)
            self.spin_voronoi_layers.setValue(project.voronoi_layers)
            
            # Voronoi Implicit 参数
            self.spin_voronoi_implicit_cell.setValue(project.voronoi_implicit_cell)
            self.spin_voronoi_implicit_wall.setValue(project.voronoi_implicit_wall)
            self.spin_voronoi_implicit_res.setValue(project.voronoi_implicit_res)
            self.spin_voronoi_implicit_density.setValue(project.voronoi_implicit_density)
            
            # Tile 参数
            self.spin_tile_shrink.setValue(project.tile_shrink)
            self.spin_tile_spacing.setValue(project.tile_spacing)
            self.spin_tile_margin.setValue(project.tile_margin)
            
            # 减面配置
            self.lattice_unit_target_faces = getattr(project, 'lattice_unit_target_faces', LATTICE_UNIT_TARGET_FACES)
            
            # 全局选项
            self.check_enable_final_trim.setChecked(project.enable_final_trim)
            
            # 视图设置
            self.current_view_mode = project.view_mode
            self._mouse_mode = project.mouse_mode
            self.action_toggle_mesh_edges.setChecked(project.show_mesh_edges)
            self.current_sole_material = project.current_sole_material
            self.current_lattice_material = project.current_lattice_material
            self._current_bg_color_index = project.current_bg_color
            
            # 应用材质
            if project.current_sole_material < len(self.sole_material_actions):
                self.sole_material_actions[project.current_sole_material].setChecked(True)
            if project.current_lattice_material < len(self.lattice_material_actions):
                self.lattice_material_actions[project.current_lattice_material].setChecked(True)
            
            # 应用背景颜色
            if hasattr(self, 'bg_color_actions') and 0 <= project.current_bg_color < len(self.bg_color_actions):
                self.bg_color_actions[project.current_bg_color].setChecked(True)
                # 如果是PyVista渲染器，应用背景颜色
                if self.current_renderer_type == "pyvista" and hasattr(self.viewer, 'set_background'):
                    bg_colors = {
                        0: '#3a3a3a',  # 深灰
                        1: '#5a5a5a',  # 中灰
                        2: '#8a8a8a',  # 浅灰
                        3: '#ffffff',  # 白色
                        4: '#1a1a1a',  # 黑色
                    }
                    color_hex = bg_colors.get(project.current_bg_color, '#3a3a3a')
                    self.viewer.set_background(color_hex)
            
            # 加载分割设置
            if hasattr(self, 'check_enable_slice'):
                self.check_enable_slice.setChecked(project.slicing_enabled)
            if hasattr(self, 'slice_axis_combo'):
                self.slice_axis_combo.setCurrentIndex(project.slice_axis)
            if hasattr(self, 'slice_position_slider'):
                # 注意：slider值是0-1000，需要转换
                slider_value = int(project.slice_position * 1000)
                self.slice_position_slider.setValue(slider_value)
            if hasattr(self, 'slice_angle1_spin'):
                self.slice_angle1_spin.setValue(project.slice_angle1)
            if hasattr(self, 'slice_angle2_spin'):
                self.slice_angle2_spin.setValue(project.slice_angle2)
            if hasattr(self, 'slice_subdivide_spin'):
                self.slice_subdivide_spin.setValue(project.slice_subdivide)
            if hasattr(self, 'slice_display_combo'):
                self.slice_display_combo.setCurrentIndex(project.slice_display_mode)
            if hasattr(self, 'slice_spacing_spin'):
                self.slice_spacing_spin.setValue(project.slice_spacing)
            if hasattr(self, 'check_show_plane'):
                self.check_show_plane.setChecked(project.show_slice_plane)
            if hasattr(self, 'check_slice_lighting'):
                self.check_slice_lighting.setChecked(project.slice_lighting)
            
            # 加载生成结果和分割结果
            self._load_generated_results_from_project(project)
            
            # 加载区域设置
            self._load_regions_from_project(project)
            
            # 预览模型
            self._preview_selected_sole(log_message=False, reset_view_mode=True)
            self._preview_selected_lattice(log_message=False, reset_view_mode=False, update_main_view=False, inset_visible=True)
            
        finally:
            self._suspend_auto_recommend = False
            self._loading_project = False  # 加载完成
        
        self._sync_method_fields()
        self._set_mouse_mode(self._mouse_mode)
    
    def _load_generated_results_from_project(self, project: Project) -> None:
        """从工程加载生成的晶格结果和分割结果"""
        import trimesh
        
        if project.file_path is None:
            return
        
        project_dir = project.file_path.parent
        
        try:
            # 加载全局生成结果
            if project.generated_lattice_result_path:
                lattice_path = project_dir / project.generated_lattice_result_path
                if lattice_path.exists():
                    self.generated_lattice_mesh = trimesh.load(str(lattice_path))
                    self._append_status(f"✓ 已加载生成的晶格结果")
            
            if project.generated_sole_result_path:
                sole_path = project_dir / project.generated_sole_result_path
                if sole_path.exists():
                    self.generated_sole_mesh = trimesh.load(str(sole_path))
            
            if project.generated_combined_result_path:
                combined_path = project_dir / project.generated_combined_result_path
                if combined_path.exists():
                    self.generated_combined_mesh = trimesh.load(str(combined_path))
                    self.has_generated_result = True
            
            # 加载区域晶格生成结果
            if project.region_lattice_results:
                self.region_lattice_meshes = {}
                for region_id, rel_path in project.region_lattice_results.items():
                    region_lattice_path = project_dir / rel_path
                    if region_lattice_path.exists():
                        self.region_lattice_meshes[region_id] = trimesh.load(str(region_lattice_path))
                self._append_status(f"✓ 已加载 {len(self.region_lattice_meshes)} 个区域晶格结果")
            
            # 加载平面区域晶格生成结果
            if project.plane_region_lattice_results:
                self.plane_region_lattice_meshes = {}
                for region_id, rel_path in project.plane_region_lattice_results.items():
                    plane_region_lattice_path = project_dir / rel_path
                    if plane_region_lattice_path.exists():
                        self.plane_region_lattice_meshes[region_id] = trimesh.load(str(plane_region_lattice_path))
                self._append_status(f"✓ 已加载 {len(self.plane_region_lattice_meshes)} 个平面区域晶格结果")
            
            # 加载分割结果
            if project.sliced_positive_mesh_path:
                positive_path = project_dir / project.sliced_positive_mesh_path
                if positive_path.exists():
                    self.positive_mesh = trimesh.load(str(positive_path))
                    self._append_status(f"✓ 已加载分割结果（正侧）")
            
            if project.sliced_negative_mesh_path:
                negative_path = project_dir / project.sliced_negative_mesh_path
                if negative_path.exists():
                    self.negative_mesh = trimesh.load(str(negative_path))
                    self._append_status(f"✓ 已加载分割结果（负侧）")
                    
        except Exception as e:
            self._append_status(f"⚠ 加载生成结果时出错: {e}")
    
    def _load_regions_from_project(self, project: Project) -> None:
        """从工程加载区域设置"""
        import numpy as np
        from domain.region.region_manager import Region
        from domain.region.plane_region_manager import PlaneRegion
        
        try:
            # 加载Tab 3的区域设置
            if project.regions:
                self.region_manager.regions = []
                for region_dict in project.regions:
                    region = Region(
                        id=region_dict['id'],
                        name=region_dict['name'],
                        source_type=region_dict['source_type'],
                        source_file=region_dict.get('source_file', '')
                    )
                    region.lattice_method = region_dict.get('lattice_method', 'gyroid')
                    region.gyroid_cell_size = region_dict.get('gyroid_cell', 8.0)
                    region.gyroid_isovalue = region_dict.get('gyroid_iso', 0.0)
                    region.gyroid_resolution = region_dict.get('gyroid_res', 32)
                    region.voronoi_cell_size = region_dict.get('voronoi_cell', 15.0)
                    region.voronoi_strut_thickness = region_dict.get('voronoi_thickness', 2.0)
                    region.voronoi_z_layers = region_dict.get('voronoi_layers', 5)
                    region.tile_shrink = region_dict.get('tile_shrink', 0.85)
                    region.tile_spacing = region_dict.get('tile_spacing', 1.0)
                    region.tile_margin = region_dict.get('tile_margin', 2.0)
                    region.tile_decimate_target_faces = region_dict.get('tile_decimate_target_faces', 0)
                    self.region_manager.regions.append(region)
                
                self._update_region_display()
                self._append_status(f"✓ 已加载 {len(self.region_manager.regions)} 个区域设置")
            
            # 加载Tab 4的平面区域设置
            if project.plane_regions:
                self.plane_region_manager.regions = []
                for region_dict in project.plane_regions:
                    region = PlaneRegion(
                        id=region_dict['id'],
                        name=region_dict['name'],
                        boundary_planes=region_dict.get('boundary_planes', [])  # 直接使用，已经是正确格式
                    )
                    region.lattice_method = region_dict.get('lattice_method', 'gyroid')
                    region.gyroid_cell_size = region_dict.get('gyroid_cell', 8.0)
                    region.gyroid_isovalue = region_dict.get('gyroid_iso', 0.0)
                    region.gyroid_resolution = region_dict.get('gyroid_res', 32)
                    region.voronoi_cell_size = region_dict.get('voronoi_cell', 15.0)
                    region.voronoi_strut_thickness = region_dict.get('voronoi_thickness', 2.0)
                    region.voronoi_z_layers = region_dict.get('voronoi_layers', 5)
                    region.tile_shrink = region_dict.get('tile_shrink', 0.85)
                    region.tile_spacing = region_dict.get('tile_spacing', 1.0)
                    region.tile_margin = region_dict.get('tile_margin', 2.0)
                    region.tile_decimate_target_faces = region_dict.get('tile_decimate_target_faces', 0)
                    self.plane_region_manager.regions.append(region)
                
                self._update_plane_region_display()
                self._append_status(f"✓ 已加载 {len(self.plane_region_manager.regions)} 个平面区域设置")
                
        except Exception as e:
            self._append_status(f"⚠ 加载区域设置时出错: {e}")
    
    def _check_save_current_project(self) -> bool:
        """检查当前工程是否需要保存，返回True表示可以继续"""
        if not self.current_project_modified:
            return True
        
        msgbox = QtWidgets.QMessageBox(self)
        msgbox.setWindowTitle("保存工程")
        msgbox.setText("当前工程已修改，是否保存？")
        msgbox.setStandardButtons(
            QtWidgets.QMessageBox.Save | 
            QtWidgets.QMessageBox.Discard | 
            QtWidgets.QMessageBox.Cancel
        )
        msgbox.setDefaultButton(QtWidgets.QMessageBox.Save)
        msgbox.setIcon(QtWidgets.QMessageBox.Question)
        
        # 应用深色主题
        self._apply_dark_messagebox_style(msgbox)
        
        reply = msgbox.exec_()
        
        if reply == QtWidgets.QMessageBox.Save:
            self._save_project()
            return True
        elif reply == QtWidgets.QMessageBox.Discard:
            return True
        else:
            return False
    
    def _update_recent_projects_menu(self) -> None:
        """更新最近工程菜单"""
        self.recent_projects_menu.clear()
        
        recent_projects = self.project_manager.get_recent_projects()
        
        if not recent_projects:
            action = QtWidgets.QAction("(无最近工程)", self)
            action.setEnabled(False)
            self.recent_projects_menu.addAction(action)
        else:
            for project_path in recent_projects:
                action = QtWidgets.QAction(project_path.name, self)
                action.setToolTip(str(project_path))
                action.triggered.connect(lambda checked, p=project_path: self._load_project_from_file(p))
                self.recent_projects_menu.addAction(action)
            
            self.recent_projects_menu.addSeparator()
            clear_action = QtWidgets.QAction("清除列表", self)
            clear_action.triggered.connect(self._clear_recent_projects)
            self.recent_projects_menu.addAction(clear_action)
    
    def _clear_recent_projects(self) -> None:
        """清除最近工程列表"""
        self.project_manager.recent_projects.clear()
        self.project_manager._save_recent_projects()
        self._update_recent_projects_menu()
    
    # ==================== 对话框方法 ====================
    
    def _apply_dark_dialog_style(self, dialog: QtWidgets.QDialog) -> None:
        """为对话框应用深色主题样式"""
        # 先设置样式表
        dialog.setStyleSheet("""
            QDialog {
                background-color: #2c2c2c;
                color: #ffffff;
            }
            QLabel {
                color: #ffffff;
            }
            QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox, QTextEdit, QPlainTextEdit {
                background-color: #3a3a3a;
                border: 1px solid #4a4a4a;
                border-radius: 4px;
                padding: 5px;
                color: #ffffff;
            }
            QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {
                border-color: #2196F3;
            }
            QComboBox::drop-down {
                border: none;
                padding-right: 5px;
            }
            QComboBox::down-arrow {
                image: none;
                border-left: 4px solid transparent;
                border-right: 4px solid transparent;
                border-top: 6px solid #ffffff;
                margin-right: 5px;
            }
            QComboBox QAbstractItemView {
                background-color: #3a3a3a;
                border: 1px solid #4a4a4a;
                selection-background-color: #2196F3;
                color: #ffffff;
            }
            QPushButton {
                background-color: #3a3a3a;
                border: 1px solid #4a4a4a;
                border-radius: 4px;
                padding: 8px 20px;
                color: #ffffff;
                font-size: 13px;
            }
            QPushButton:hover {
                background-color: #4a4a4a;
                border-color: #2196F3;
            }
            QPushButton:pressed {
                background-color: #2a2a2a;
            }
            QCheckBox {
                color: #ffffff;
                spacing: 8px;
            }
            QCheckBox::indicator {
                width: 18px;
                height: 18px;
                border: 2px solid #4a4a4a;
                border-radius: 3px;
                background-color: #3a3a3a;
            }
            QCheckBox::indicator:checked {
                background-color: #2196F3;
                border-color: #2196F3;
            }
            QCheckBox::indicator:checked:after {
                content: "✓";
                color: white;
            }
            QGroupBox {
                color: #ffffff;
                border: 1px solid #4a4a4a;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 0 5px;
                color: #2196F3;
            }
            QTabWidget::pane {
                border: 1px solid #4a4a4a;
                background-color: #2c2c2c;
            }
            QTabBar::tab {
                background-color: #3a3a3a;
                color: #ffffff;
                padding: 8px 16px;
                border: 1px solid #4a4a4a;
                border-bottom: none;
            }
            QTabBar::tab:selected {
                background-color: #2c2c2c;
                border-bottom: 2px solid #2196F3;
            }
            QTabBar::tab:hover {
                background-color: #4a4a4a;
            }
        """)
        
        # 延迟设置深色标题栏（需要在对话框显示后）
        def set_dark_title():
            try:
                from ctypes import windll, c_int, byref, sizeof
                hwnd = int(dialog.winId())
                DWMWA_USE_IMMERSIVE_DARK_MODE = 20
                value = c_int(1)
                try:
                    windll.dwmapi.DwmSetWindowAttribute(hwnd, DWMWA_USE_IMMERSIVE_DARK_MODE, byref(value), sizeof(value))
                except:
                    DWMWA_USE_IMMERSIVE_DARK_MODE = 19
                    windll.dwmapi.DwmSetWindowAttribute(hwnd, DWMWA_USE_IMMERSIVE_DARK_MODE, byref(value), sizeof(value))
            except:
                pass
        
        # 使用定时器延迟设置
        QtCore.QTimer.singleShot(50, set_dark_title)
    
    def _apply_dark_messagebox_style(self, msgbox: QtWidgets.QMessageBox) -> None:
        """为消息框应用深色主题样式"""
        msgbox.setStyleSheet("""
            QMessageBox {
                background-color: #2c2c2c;
                color: #ffffff;
            }
            QMessageBox QLabel {
                color: #ffffff;
                font-size: 13px;
            }
            QMessageBox QPushButton {
                background-color: #3a3a3a;
                border: 1px solid #4a4a4a;
                border-radius: 4px;
                padding: 8px 20px;
                color: #ffffff;
                min-width: 80px;
                font-size: 13px;
            }
            QMessageBox QPushButton:hover {
                background-color: #4a4a4a;
                border-color: #2196F3;
            }
            QMessageBox QPushButton:pressed {
                background-color: #2a2a2a;
            }
        """)
        
        # 延迟设置深色标题栏
        def set_dark_title():
            try:
                from ctypes import windll, c_int, byref, sizeof
                hwnd = int(msgbox.winId())
                DWMWA_USE_IMMERSIVE_DARK_MODE = 20
                value = c_int(1)
                try:
                    windll.dwmapi.DwmSetWindowAttribute(hwnd, DWMWA_USE_IMMERSIVE_DARK_MODE, byref(value), sizeof(value))
                except:
                    DWMWA_USE_IMMERSIVE_DARK_MODE = 19
                    windll.dwmapi.DwmSetWindowAttribute(hwnd, DWMWA_USE_IMMERSIVE_DARK_MODE, byref(value), sizeof(value))
            except:
                pass
        
        QtCore.QTimer.singleShot(50, set_dark_title)
    
    def _show_information(self, title: str, message: str) -> None:
        """显示深色主题的信息对话框"""
        msgbox = QtWidgets.QMessageBox(self)
        msgbox.setWindowTitle(title)
        msgbox.setText(message)
        msgbox.setIcon(QtWidgets.QMessageBox.Information)
        self._apply_dark_messagebox_style(msgbox)
        msgbox.exec_()
    
    def _show_warning(self, title: str, message: str) -> None:
        """显示深色主题的警告对话框"""
        msgbox = QtWidgets.QMessageBox(self)
        msgbox.setWindowTitle(title)
        msgbox.setText(message)
        msgbox.setIcon(QtWidgets.QMessageBox.Warning)
        self._apply_dark_messagebox_style(msgbox)
        msgbox.exec_()
    
    def _show_critical(self, title: str, message: str) -> None:
        """显示深色主题的错误对话框"""
        msgbox = QtWidgets.QMessageBox(self)
        msgbox.setWindowTitle(title)
        msgbox.setText(message)
        msgbox.setIcon(QtWidgets.QMessageBox.Critical)
        self._apply_dark_messagebox_style(msgbox)
        msgbox.exec_()

    def closeEvent(self, event) -> None:
        # 检查是否需要保存工程
        if not self._check_save_current_project():
            event.ignore()
            return
        
        self._remember_current_preview_view()
        # 不再自动保存UI设置
        # self._save_ui_settings()
        super().closeEvent(event)


def run_app() -> None:
    import os
    # 抑制Qt在Windows多显示器环境下的几何警告
    os.environ['QT_LOGGING_RULES'] = '*.debug=false;qt.qpa.*=false'
    
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    
    # 显示启动对话框
    from ui.dialogs.startup_dialog import StartupDialog
    from domain.project.project_manager import ProjectManager
    
    temp_project_manager = ProjectManager()
    startup_dialog = StartupDialog(temp_project_manager)
    
    if startup_dialog.exec_() != QtWidgets.QDialog.Accepted:
        # 用户取消，退出程序
        return
    
    # 创建主窗口
    window = LatticeWorkbenchWindow()
    
    # 根据用户选择加载工程
    if startup_dialog.selected_action == "new":
        # 新建工程（已在__init__中创建）
        pass
    elif startup_dialog.selected_action in ("open", "recent"):
        # 加载工程
        if startup_dialog.selected_project_path:
            window._load_project_from_file(startup_dialog.selected_project_path)
    
    window.show()
    app.exec()


if __name__ == "__main__":
    run_app()

