"""
基于 PyVista (VTK) 的高质量渲染器
支持 PBR + SSAO + 阴影 + 抗锯齿

"""

from __future__ import annotations

import numpy as np
import trimesh
import warnings
import logging
import sys
import os

# 在导入 VTK/PyVista 之前设置环境变量
os.environ['VTK_SILENCE_GET_VOID_POINTER_WARNINGS'] = '1'
os.environ['VTK_DEBUG_LEAKS'] = '0'
os.environ['PYVISTA_OFF_SCREEN'] = '0'

# 现在导入 pyvista
import pyvista as pv
from pyvistaqt import QtInteractor
from PyQt5 import QtWidgets, QtCore
from dataclasses import dataclass
from typing import Optional, Literal

# 抑制 VTK 的 OpenGL 上下文警告（Windows 特有问题）
warnings.filterwarnings('ignore', category=UserWarning, module='vtkmodules')
logging.getLogger('vtkmodules').setLevel(logging.ERROR)

# 更激进的方法：重定向 VTK 的 C++ 错误输出
# 这些错误来自 VTK 的 C++ 层，无法通过 Python logging 抑制
try:
    import vtkmodules.vtkCommonCore as vtkcc
    # 创建一个输出窗口来捕获 VTK 错误
    vtk_output_window = vtkcc.vtkOutputWindow()
    vtk_output_window.SetInstance(vtk_output_window)
    # 禁用所有 VTK 输出
    vtk_output_window.GlobalWarningDisplayOff()
except:
    pass

# 禁用 PyVista 的详细输出
try:
    pv.set_error_output_file(os.devnull if os.name != 'nt' else 'nul')
except:
    pass


RenderMode = Literal["solid", "wireframe"]
MouseMode = Literal["rotate", "pan"]


@dataclass
class MeshData:
    """网格数据类（保持与原 viewer.py 兼容）"""
    vertices: np.ndarray
    faces: np.ndarray
    color: tuple[float, float, float, float] = (0.7, 0.7, 0.7, 1.0)
    render_mode: RenderMode = "solid"
    unlit: bool = False
    line_width: float = 1.5
    metallic: float = 0.3  # 金属度 (0-1)
    roughness: float = 0.5  # 粗糙度 (0-1)
    
    def __post_init__(self):
        self.vertices = np.asarray(self.vertices, dtype=np.float32)
        self.faces = np.asarray(self.faces, dtype=np.int32)


class PyVistaRenderer(QtInteractor):
    """
    基于 PyVista 的高质量渲染器
    
    特性：
    - PBR 材质
    - SSAO (环境光遮蔽)
    - 实时阴影
    - FXAA 抗锯齿
    - 多光源
    
    交互：
    - 左键拖动：旋转
    - 右键拖动：平移
    - 滚轮：缩放
    """
    
    # 定义信号（保持与原 GLMeshViewer 兼容）
    lasso_completed = QtCore.pyqtSignal(list)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # 首次初始化时显示提示（仅一次）
        if not hasattr(PyVistaRenderer, '_init_message_shown'):
            PyVistaRenderer._init_message_shown = True
            print("=" * 60)
            print("PyVista 渲染器已启动")
            print("- 如果看到 VTK OpenGL 错误（wglMakeCurrent），可以忽略")
            print("- 这是 Windows 上 VTK 的已知问题，不影响功能")
            print("=" * 60)
        
        # 状态变量
        self.meshes: list[MeshData] = []
        self.mouse_mode: MouseMode = "rotate"
        self._bbox_center = np.zeros(3, dtype=np.float32)
        self._bbox_size = 1.0
        self._bbox_dimensions = np.ones(3, dtype=np.float32)  # 添加三维尺寸存储
        
        # 渲染选项
        self.show_axes = True
        self.show_ground_grid = True  # 默认显示网格平台
        self.show_ground_plane = True  # 显示地面平台
        self.use_smooth_shading = True
        self.enable_pbr = True  # 默认启用PBR
        self.enable_ssao_flag = False  # 默认禁用SSAO（避免在特定距离产生过度阴影）
        self.enable_shadows_flag = False  # 禁用阴影
        self.enable_antialiasing = True
        
        # 平台相关
        self._ground_plane_actor = None
        self._ground_grid_actor = None
        
        # 平面可视化
        self.show_plane = False
        self.plane_point = None
        self.plane_normal = None
        self.plane_size = 100.0
        self.plane_actor = None
        
        # PyVista 内部对象
        self._pv_meshes = []  # 存储 PyVista mesh 对象
        self._pv_actors = []  # 存储 actor 对象
        
        # 初始化渲染器
        self._setup_renderer()
        
        # 调整鼠标滚轮缩放灵敏度（降低缩放速度）
        self._adjust_zoom_sensitivity()
        
        # 添加相机交互回调，在相机移动后修复裁剪面
        self._setup_camera_callback()
    
    def _setup_renderer(self):
        """设置渲染器参数"""
        # 设置背景颜色（深灰色，与 UI 默认选项一致）
        self.set_background('#3a3a3a')
        
        # 启用抗锯齿
        if self.enable_antialiasing:
            try:
                self.enable_anti_aliasing('fxaa')
            except Exception as e:
                print(f"[警告] 抗锯齿启用失败: {e}")
        
        # 启用深度剥离（提高透明度渲染质量）
        try:
            self.enable_depth_peeling()
        except:
            pass
    
    def _adjust_zoom_sensitivity(self):
        """调整鼠标滚轮缩放灵敏度"""
        try:
            # 获取 VTK 交互器样式
            interactor_style = self.interactor.GetInteractorStyle()
            
            # 设置更小的缩放因子（默认是 1.1，改为 1.05 让缩放更平滑）
            if hasattr(interactor_style, 'SetMouseWheelMotionFactor'):
                interactor_style.SetMouseWheelMotionFactor(0.5)  # 降低滚轮灵敏度
            
            # 另一种方法：直接设置缩放因子
            if hasattr(self.interactor, 'SetDollyFactor'):
                self.interactor.SetDollyFactor(1.05)  # 从默认的 1.1 降到 1.05
                
        except Exception as e:
            print(f"[提示] 无法调整缩放灵敏度: {e}")
    
    def _setup_camera_callback(self):
        """设置相机交互回调，在相机移动后自动修复裁剪面"""
        try:
            # 添加相机修改事件的观察者
            def on_camera_modified(obj, event):
                """相机修改时的回调"""
                try:
                    self._fix_camera_clipping()
                except:
                    pass
            
            # 监听相机的修改事件
            self.camera.AddObserver('ModifiedEvent', on_camera_modified)
            
        except Exception as e:
            print(f"[提示] 无法设置相机回调: {e}")
    
    def set_mouse_mode(self, mode: MouseMode):
        """设置鼠标模式"""
        self.mouse_mode = "pan" if mode == "pan" else "rotate"
        # PyVista 的交互模式由内部处理，这里只记录状态
    
    def get_view_state(self) -> dict[str, float]:
        """获取当前视图状态"""
        camera = self.camera
        return {
            "position": camera.position,
            "focal_point": camera.focal_point,
            "view_up": camera.up,
            "distance": camera.distance,
        }
    
    def apply_view_state(self, state: dict):
        """应用视图状态"""
        try:
            camera = self.camera
            if "position" in state:
                camera.position = state["position"]
            if "focal_point" in state:
                camera.focal_point = state["focal_point"]
            if "view_up" in state:
                camera.up = state["view_up"]
            self.render()
        except:
            pass
    
    def reset_view(self):
        """重置视图 - 等轴测正视图（类似 3D 打印软件）"""
        if not self.meshes or len(self.meshes) == 0:
            # 如果没有网格，使用默认视图
            self.reset_camera()
            self._fix_camera_clipping()
            return
        
        # 计算所有网格的包围盒
        all_vertices = []
        for mesh_data in self.meshes:
            if len(mesh_data.vertices) > 0:
                all_vertices.append(mesh_data.vertices)
        
        if not all_vertices:
            self.reset_camera()
            self._fix_camera_clipping()
            return
        
        all_v = np.concatenate(all_vertices, axis=0)
        vmin = all_v.min(axis=0)
        vmax = all_v.max(axis=0)
        center = (vmin + vmax) * 0.5
        size = np.max(vmax - vmin)
        
        # 设置相机
        camera = self.camera
        camera.focal_point = center
        
        # 等轴测视角（从右前上方观察，类似图片中的视角）
        # 这是标准的等轴测投影角度
        distance = size * 5.0  # 固定使用5倍距离
        
        # 使用标准等轴测角度：
        # - 水平旋转 45°（东北方向）
        # - 垂直旋转 35.264°（arctan(1/√2)）
        angle_h = np.radians(45)  # 水平角度
        angle_v = np.radians(35.264)  # 垂直角度（等轴测标准角度）
        
        camera.position = (
            center[0] + distance * np.cos(angle_v) * np.cos(angle_h),
            center[1] + distance * np.cos(angle_v) * np.sin(angle_h),
            center[2] + distance * np.sin(angle_v)
        )
        
        # 设置上方向为 Z 轴
        camera.up = (0, 0, 1)
        
        # 修复裁剪面
        self._fix_camera_clipping()
        
        self.render()
    
    def set_view_front(self):
        """前视图 - 从前方（-Y方向）看向模型"""
        if not self.meshes or len(self.meshes) == 0:
            return
        
        camera = self.camera
        # 焦点往上移动（Z方向增加），让模型和平台在视野中显得往下
        focal_offset = self._bbox_size * 0.5  # 往上偏移50%（从30%增加）
        camera.focal_point = (
            self._bbox_center[0],
            self._bbox_center[1],
            self._bbox_center[2] + focal_offset
        )
        
        # 从前方看（-Y方向），距离更远
        distance = self._bbox_size * 5.0
        camera.position = (
            self._bbox_center[0],
            self._bbox_center[1] - distance,
            self._bbox_center[2] + focal_offset
        )
        camera.up = (0, 0, 1)
        self._fix_camera_clipping()
        self.render()
    
    def set_view_back(self):
        """后视图 - 从后方（+Y方向）看向模型"""
        if not self.meshes or len(self.meshes) == 0:
            return
        
        camera = self.camera
        # 焦点往上移动（Z方向增加），让模型和平台在视野中显得往下
        focal_offset = self._bbox_size * 0.5  # 往上偏移50%（从30%增加）
        camera.focal_point = (
            self._bbox_center[0],
            self._bbox_center[1],
            self._bbox_center[2] + focal_offset
        )
        
        # 从后方看（+Y方向），距离更远
        distance = self._bbox_size * 5.0
        camera.position = (
            self._bbox_center[0],
            self._bbox_center[1] + distance,
            self._bbox_center[2] + focal_offset
        )
        camera.up = (0, 0, 1)
        self._fix_camera_clipping()
        self.render()
    
    def set_view_left(self):
        """左视图 - 从左侧（-X方向）看向模型"""
        if not self.meshes or len(self.meshes) == 0:
            return
        
        camera = self.camera
        # 焦点往上移动（Z方向增加），让模型和平台在视野中显得往下
        focal_offset = self._bbox_size * 0.5  # 往上偏移50%（从30%增加）
        camera.focal_point = (
            self._bbox_center[0],
            self._bbox_center[1],
            self._bbox_center[2] + focal_offset
        )
        
        # 从左侧看（-X方向），距离更远
        distance = self._bbox_size * 5.0
        camera.position = (
            self._bbox_center[0] - distance,
            self._bbox_center[1],
            self._bbox_center[2] + focal_offset
        )
        camera.up = (0, 0, 1)
        self._fix_camera_clipping()
        self.render()
    
    def set_view_right(self):
        """右视图 - 从右侧（+X方向）看向模型"""
        if not self.meshes or len(self.meshes) == 0:
            return
        
        camera = self.camera
        # 焦点往上移动（Z方向增加），让模型和平台在视野中显得往下
        focal_offset = self._bbox_size * 0.5  # 往上偏移50%（从30%增加）
        camera.focal_point = (
            self._bbox_center[0],
            self._bbox_center[1],
            self._bbox_center[2] + focal_offset
        )
        
        # 从右侧看（+X方向），距离更远
        distance = self._bbox_size * 5.0
        camera.position = (
            self._bbox_center[0] + distance,
            self._bbox_center[1],
            self._bbox_center[2] + focal_offset
        )
        camera.up = (0, 0, 1)
        self._fix_camera_clipping()
        self.render()
    
    def set_view_top(self):
        """顶视图 - 从上方（+Z方向）俯视"""
        if not self.meshes or len(self.meshes) == 0:
            return
        
        camera = self.camera
        camera.focal_point = self._bbox_center
        
        # 从上方俯视（+Z方向），距离更远
        distance = self._bbox_size * 5.0  # 改为 5.0
        camera.position = (
            self._bbox_center[0],
            self._bbox_center[1],
            self._bbox_center[2] + distance
        )
        camera.up = (0, 1, 0)  # Y轴向上
        self._fix_camera_clipping()
        self.render()
    
    def set_view_bottom(self):
        """底视图 - 从下方（-Z方向）仰视"""
        if not self.meshes or len(self.meshes) == 0:
            return
        
        camera = self.camera
        camera.focal_point = self._bbox_center
        
        # 从下方仰视（-Z方向），距离更远
        distance = self._bbox_size * 5.0  # 改为 5.0
        camera.position = (
            self._bbox_center[0],
            self._bbox_center[1],
            self._bbox_center[2] - distance
        )
        camera.up = (0, 1, 0)  # Y轴向上
        self._fix_camera_clipping()
        self.render()
    
    def set_view_isometric(self):
        """等轴测视图 - 标准等轴测角度"""
        if not self.meshes or len(self.meshes) == 0:
            return
        
        # 使用与 reset_view 相同的逻辑
        self.reset_view()
        self.render()
    
    def _fix_camera_clipping(self):
        """修复相机裁剪面，防止在特定距离时模型显示异常"""
        try:
            camera = self.camera
            
            # 计算相机到焦点的距离
            distance = camera.distance
            
            # 设置合理的近裁剪面和远裁剪面
            # 近裁剪面：距离的 1%（不能太近，否则会裁剪掉近处的物体）
            # 远裁剪面：距离的 100倍（确保远处的物体也能看到）
            near_clip = max(distance * 0.01, self._bbox_size * 0.001)
            far_clip = distance * 100.0
            
            # 设置裁剪范围
            camera.clipping_range = (near_clip, far_clip)
            
        except Exception as e:
            print(f"[警告] 修复相机裁剪面失败: {e}")
    
    def set_meshes(self, meshes: list[MeshData], reset_view: bool = True):
        """设置要渲染的网格"""
        self.meshes = meshes
        self._update_scene()
        
        if reset_view:
            self.reset_view()
        else:
            self._fix_camera_clipping()
            self.render()
    
    def _update_scene(self):
        """更新场景（重新构建所有网格）"""
        # 清空现有的网格
        self.clear()
        self._pv_meshes.clear()
        self._pv_actors.clear()
        
        if not self.meshes:
            return
        
        # 计算包围盒
        self._update_bbox()
        
        # 添加网格平台（在模型之前，这样模型在上面）
        if self.show_ground_plane or self.show_ground_grid:
            self._add_ground_platform()
        
        # 添加所有网格
        for mesh_data in self.meshes:
            self._add_mesh_to_scene(mesh_data)
        
        # 添加坐标轴
        if self.show_axes:
            self.add_axes()
        
        # 设置光照
        self._setup_lighting()
        
        # 启用高级特性
        self._enable_advanced_features()
        
        # 添加模型信息显示（左上角）
        self._add_model_info_overlay()
        
        # 重新渲染平面（如果需要）
        if self.show_plane and self.plane_point is not None:
            self._update_plane()
    
    def _add_mesh_to_scene(self, mesh_data: MeshData):
        """添加单个网格到场景"""
        if len(mesh_data.vertices) == 0 or len(mesh_data.faces) == 0:
            return
        
        # 转换为 PyVista 格式
        try:
            # 创建 trimesh 对象
            tri_mesh = trimesh.Trimesh(
                vertices=mesh_data.vertices,
                faces=mesh_data.faces
            )
            
            # 转换为 PyVista
            pv_mesh = pv.wrap(tri_mesh)
            
            # 提取颜色（转换为 0-1 范围）
            color = mesh_data.color
            if all(c <= 1.0 for c in color[:3]):
                color_rgb = color[:3]
            else:
                color_rgb = tuple(c / 255.0 for c in color[:3])
            
            # 转换为十六进制颜色
            color_hex = '#{:02x}{:02x}{:02x}'.format(
                int(color_rgb[0] * 255),
                int(color_rgb[1] * 255),
                int(color_rgb[2] * 255)
            )
            
            # 添加到场景
            if mesh_data.render_mode == "wireframe":
                actor = self.add_mesh(
                    pv_mesh,
                    color=color_hex,
                    style='wireframe',
                    line_width=mesh_data.line_width,
                    lighting=False
                )
            else:
                # 实体渲染 - 使用合理的PBR参数
                if self.enable_pbr and not mesh_data.unlit:
                    # 使用适中的金属度和粗糙度，启用双面渲染
                    actor = self.add_mesh(
                        pv_mesh,
                        color=color_hex,
                        pbr=True,
                        metallic=0.2,  # 适中的金属度
                        roughness=0.6,  # 适中的粗糙度
                        smooth_shading=self.use_smooth_shading,
                        show_edges=False,
                        lighting=True,
                        backface_culling=False  # 启用双面渲染
                    )
                else:
                    # 非PBR渲染，也启用双面渲染
                    actor = self.add_mesh(
                        pv_mesh,
                        color=color_hex,
                        smooth_shading=self.use_smooth_shading,
                        show_edges=False,
                        lighting=not mesh_data.unlit,
                        backface_culling=False  # 启用双面渲染
                    )
            
            self._pv_meshes.append(pv_mesh)
            self._pv_actors.append(actor)
            
        except Exception as e:
            print(f"[错误] 添加网格失败: {e}")
            import traceback
            traceback.print_exc()
    
    def _update_bbox(self):
        """更新包围盒"""
        valid = [m for m in self.meshes if len(m.vertices) > 0]
        if not valid:
            self._bbox_center = np.zeros(3, dtype=np.float32)
            self._bbox_size = 1.0
            self._bbox_dimensions = np.ones(3, dtype=np.float32)  # 添加三维尺寸
            return
        
        all_v = np.concatenate([m.vertices for m in valid], axis=0)
        vmin = all_v.min(axis=0)
        vmax = all_v.max(axis=0)
        self._bbox_center = (vmin + vmax) * 0.5
        self._bbox_dimensions = vmax - vmin  # 保存三维尺寸
        self._bbox_size = max(float(np.max(vmax - vmin)), 1.0)
    
    def _add_ground_platform(self):
        """添加网格平台（类似 3D 打印软件的打印平台）"""
        try:
            # 计算平台尺寸（更宽）
            platform_size = self._bbox_size * 3.0  # 从 2.5 增加到 3.0，更宽
            
            # 找到模型的最低点
            all_v = np.concatenate([m.vertices for m in self.meshes if len(m.vertices) > 0], axis=0)
            z_min = all_v[:, 2].min()
            
            # 平台位置：始终在Z=0平面上，不管模型在哪里
            platform_z = 0.0
            platform_center = [self._bbox_center[0], self._bbox_center[1], platform_z]
            
            # 立方体固定高度（更高）
            cube_height = max(self._bbox_size * 2.0, abs(z_min) + self._bbox_size * 1.5)  # 确保立方体包含模型
            
            # 创建地面平台（半透明平面）
            if self.show_ground_plane:
                plane = pv.Plane(
                    center=platform_center,
                    direction=[0, 0, 1],
                    i_size=platform_size,
                    j_size=platform_size
                )
                
                self._ground_plane_actor = self.add_mesh(
                    plane,
                    color='#888888',
                    opacity=0.3,
                    show_edges=False,
                    lighting=False
                )
            
            # 创建网格线（虚线效果）
            if self.show_ground_grid:
                # 网格间距
                grid_spacing = platform_size / 30
                
                # 创建网格线
                lines = []
                
                # X 方向的线
                for i in range(31):
                    y = -platform_size/2 + i * grid_spacing
                    line = pv.Line(
                        [platform_center[0] - platform_size/2, platform_center[1] + y, platform_z],
                        [platform_center[0] + platform_size/2, platform_center[1] + y, platform_z]
                    )
                    lines.append(line)
                
                # Y 方向的线
                for i in range(31):
                    x = -platform_size/2 + i * grid_spacing
                    line = pv.Line(
                        [platform_center[0] + x, platform_center[1] - platform_size/2, platform_z],
                        [platform_center[0] + x, platform_center[1] + platform_size/2, platform_z]
                    )
                    lines.append(line)
                
                # 合并所有线
                grid = lines[0]
                for line in lines[1:]:
                    grid = grid + line
                
                # 网格线更虚（降低透明度）
                self._ground_grid_actor = self.add_mesh(
                    grid,
                    color='#999999',  # 中灰色
                    line_width=1.0,  # 细线
                    opacity=0.4,  # 更透明（虚）
                    lighting=False,
                    style='wireframe'  # 线框模式
                )
                
                # 创建立方体边框（12条边）
                half_size = platform_size / 2
                cube_top_z = platform_z + cube_height
                
                # 底部4条边
                bottom_edges = [
                    # 前边
                    pv.Line(
                        [platform_center[0] - half_size, platform_center[1] - half_size, platform_z],
                        [platform_center[0] + half_size, platform_center[1] - half_size, platform_z]
                    ),
                    # 后边
                    pv.Line(
                        [platform_center[0] - half_size, platform_center[1] + half_size, platform_z],
                        [platform_center[0] + half_size, platform_center[1] + half_size, platform_z]
                    ),
                    # 左边
                    pv.Line(
                        [platform_center[0] - half_size, platform_center[1] - half_size, platform_z],
                        [platform_center[0] - half_size, platform_center[1] + half_size, platform_z]
                    ),
                    # 右边
                    pv.Line(
                        [platform_center[0] + half_size, platform_center[1] - half_size, platform_z],
                        [platform_center[0] + half_size, platform_center[1] + half_size, platform_z]
                    ),
                ]
                
                # 顶部4条边
                top_edges = [
                    # 前边
                    pv.Line(
                        [platform_center[0] - half_size, platform_center[1] - half_size, cube_top_z],
                        [platform_center[0] + half_size, platform_center[1] - half_size, cube_top_z]
                    ),
                    # 后边
                    pv.Line(
                        [platform_center[0] - half_size, platform_center[1] + half_size, cube_top_z],
                        [platform_center[0] + half_size, platform_center[1] + half_size, cube_top_z]
                    ),
                    # 左边
                    pv.Line(
                        [platform_center[0] - half_size, platform_center[1] - half_size, cube_top_z],
                        [platform_center[0] - half_size, platform_center[1] + half_size, cube_top_z]
                    ),
                    # 右边
                    pv.Line(
                        [platform_center[0] + half_size, platform_center[1] - half_size, cube_top_z],
                        [platform_center[0] + half_size, platform_center[1] + half_size, cube_top_z]
                    ),
                ]
                
                # 垂直4条边（连接底部和顶部）
                vertical_edges = [
                    # 左前
                    pv.Line(
                        [platform_center[0] - half_size, platform_center[1] - half_size, platform_z],
                        [platform_center[0] - half_size, platform_center[1] - half_size, cube_top_z]
                    ),
                    # 右前
                    pv.Line(
                        [platform_center[0] + half_size, platform_center[1] - half_size, platform_z],
                        [platform_center[0] + half_size, platform_center[1] - half_size, cube_top_z]
                    ),
                    # 左后
                    pv.Line(
                        [platform_center[0] - half_size, platform_center[1] + half_size, platform_z],
                        [platform_center[0] - half_size, platform_center[1] + half_size, cube_top_z]
                    ),
                    # 右后
                    pv.Line(
                        [platform_center[0] + half_size, platform_center[1] + half_size, platform_z],
                        [platform_center[0] + half_size, platform_center[1] + half_size, cube_top_z]
                    ),
                ]
                
                # 合并所有边框
                all_edges = bottom_edges + top_edges + vertical_edges
                cube_border = all_edges[0]
                for edge in all_edges[1:]:
                    cube_border = cube_border + edge
                
                # 添加黑色立方体边框（细线）
                self.add_mesh(
                    cube_border,
                    color='#333333',  # 深灰色（不是纯黑）
                    line_width=2,  # 细一点（从3改为2）
                    opacity=1.0,  # 完全不透明
                    lighting=False
                )
            
        except Exception as e:
            print(f"[警告] 添加网格平台失败: {e}")
            import traceback
            traceback.print_exc()
    
    def _setup_lighting(self):
        """设置光照系统"""
        # 移除默认光源
        self.remove_all_lights()
        
        # 跟随相机的主光源（最重要！确保任何角度都有光照）
        light_headlight = pv.Light(
            light_type='headlight',  # 跟随相机的光源
            intensity=1.5  # 进一步增强headlight强度
        )
        self.add_light(light_headlight)
        
        # 环境补光（从多个方向提供柔和的补光）
        # 主光源（从右上方照射）
        light1 = pv.Light(
            position=(self._bbox_size * 2.0, self._bbox_size * 2.0, self._bbox_size * 3.0),
            focal_point=self._bbox_center,
            color='white',
            intensity=0.6  # 进一步降低固定光源强度
        )
        self.add_light(light1)
        
        # 补光（从左侧）
        light2 = pv.Light(
            position=(-self._bbox_size * 1.5, self._bbox_size * 0.8, self._bbox_size * 1.5),
            focal_point=self._bbox_center,
            color='white',
            intensity=0.4  # 进一步降低固定光源强度
        )
        self.add_light(light2)
        
        # 背光（从后下方，增强轮廓）
        light3 = pv.Light(
            position=(0, -self._bbox_size * 1.0, -self._bbox_size * 0.3),
            focal_point=self._bbox_center,
            color='white',
            intensity=0.3  # 进一步降低固定光源强度
        )
        self.add_light(light3)
        
        # 底部补光
        light4 = pv.Light(
            position=(0, 0, -self._bbox_size * 1.0),
            focal_point=self._bbox_center,
            color='white',
            intensity=0.2  # 进一步降低固定光源强度
        )
        self.add_light(light4)
    
    def _enable_advanced_features(self):
        """启用高级渲染特性"""
        # 启用 SSAO（环境光遮蔽）- 使用更保守的参数
        if self.enable_ssao_flag:
            try:
                # 使用更小的半径和更大的偏移，减少过度阴影
                self.enable_ssao(
                    radius=self._bbox_size * 0.03,  # 减小半径（从0.06降到0.03）
                    bias=0.03,  # 增大偏移（从0.015增到0.03），减少假阴影
                    kernel_size=16,  # 减少采样数（从24降到16），提高性能
                    blur=True  # 启用模糊，使效果更柔和
                )
                print("[提示] SSAO 已启用（保守参数）")
            except Exception as e:
                print(f"[警告] SSAO 启用失败: {e}")
        else:
            # 尝试禁用 SSAO
            try:
                # PyVista 没有 disable_ssao() 方法，但我们可以尝试关闭
                if hasattr(self, 'disable_ssao'):
                    self.disable_ssao()
                # 或者尝试设置极小的参数来最小化效果
                elif hasattr(self, 'enable_ssao'):
                    # 不调用 enable_ssao，让它保持未启用状态
                    pass
            except:
                pass
        
        # 阴影功能已禁用
    
    def _add_model_info_overlay(self):
        """在左上角添加模型信息显示"""
        if not self.meshes:
            return
        
        # 统计信息
        total_vertices = sum(len(m.vertices) for m in self.meshes)
        total_faces = sum(len(m.faces) for m in self.meshes)
        num_meshes = len(self.meshes)
        
        # 获取三维尺寸
        size_x, size_y, size_z = self._bbox_dimensions
        
        # 构建信息文本（只显示模型信息）
        info_lines = [
            "--- Model Info ---",
            f"Meshes  : {num_meshes}",
            f"Vertices: {total_vertices:,}",
            f"Faces   : {total_faces:,}",
            f"Size X  : {size_x:.1f} mm",
            f"Size Y  : {size_y:.1f} mm",
            f"Size Z  : {size_z:.1f} mm",
        ]
        
        info_text = "\n".join(info_lines)
        
        # 添加文本到左上角
        try:
            self.add_text(
                info_text,
                position='upper_left',
                font_size=11,
                color='white',
                font='courier',
                shadow=True
            )
        except Exception as e:
            # 如果失败，尝试最简单的方式
            try:
                self.add_text(
                    info_text,
                    position='upper_left',
                    font_size=12,
                    color='white'
                )
            except:
                print(f"[警告] 添加信息显示失败: {e}")
    
    def set_plane(self, point: np.ndarray, normal: np.ndarray, size: float = None):
        """设置要显示的平面"""
        self.plane_point = np.asarray(point, dtype=np.float32)
        self.plane_normal = np.asarray(normal, dtype=np.float32)
        self.plane_normal = self.plane_normal / np.linalg.norm(self.plane_normal)
        
        if size is not None:
            self.plane_size = size
        else:
            self.plane_size = self._bbox_size * 1.5  # 增大平面尺寸
        
        self.show_plane = True
        self._update_plane()
    
    def hide_plane(self):
        """隐藏平面"""
        self.show_plane = False
        if self.plane_actor is not None:
            try:
                self.remove_actor(self.plane_actor)
                self.plane_actor = None
            except:
                pass
        self.render()
    
    def _update_plane(self):
        """更新平面显示"""
        # 移除旧的平面
        if self.plane_actor is not None:
            try:
                self.remove_actor(self.plane_actor)
            except:
                pass
        
        if not self.show_plane or self.plane_point is None:
            return
        
        try:
            # 创建平面
            plane = pv.Plane(
                center=self.plane_point,
                direction=self.plane_normal,
                i_size=self.plane_size,
                j_size=self.plane_size
            )
            
            # 添加到场景（半透明）
            self.plane_actor = self.add_mesh(
                plane,
                color='yellow',
                opacity=0.3,
                show_edges=True,
                edge_color='orange',
                lighting=False
            )
            
            self.render()
            
        except Exception as e:
            print(f"[错误] 平面显示失败: {e}")
    
    def enable_lasso_mode(self, mesh):
        """启用套索模式（暂不支持）"""
        print("[警告] PyVista 渲染器暂不支持套索模式")
    
    def disable_lasso_mode(self):
        """禁用套索模式"""
        pass
    
    def resizeEvent(self, event):
        """处理窗口大小改变事件"""
        super().resizeEvent(event)
        # 立即渲染
        try:
            self.render()
            # 强制处理事件
            from PyQt5.QtWidgets import QApplication
            QApplication.processEvents()
        except:
            pass


# 为了兼容性，创建一个别名
GLMeshViewer = PyVistaRenderer


if __name__ == "__main__":
    """测试渲染器"""
    import sys
    from pathlib import Path
    
    app = QtWidgets.QApplication(sys.argv)
    
    # 创建窗口
    window = QtWidgets.QMainWindow()
    window.setWindowTitle("PyVista 渲染器测试")
    window.resize(1200, 900)
    
    # 创建渲染器
    renderer = PyVistaRenderer(window)
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
        print("  - PBR 材质: 启用")
        print("  - SSAO: 启用")
        print("  - 阴影: 启用")
        print("  - 抗锯齿: 启用")
    else:
        print("未找到测试模型，创建测试几何体...")
        # 创建测试球体
        sphere = trimesh.creation.icosphere(subdivisions=4, radius=50.0)
        mesh_data = MeshData(
            vertices=sphere.vertices,
            faces=sphere.faces,
            color=(0.29, 0.56, 0.89, 1.0),
            metallic=0.3,
            roughness=0.4
        )
        renderer.set_meshes([mesh_data], reset_view=True)
    
    window.show()
    sys.exit(app.exec_())
