from __future__ import annotations

"""基于 PyQt5 的图形渲染器。
   目前主工程并未使用
"""

import sys
from dataclasses import dataclass, field
from typing import Optional, Literal

import numpy as np
from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtOpenGL import QGLWidget
from OpenGL.GL import *
from OpenGL.GLU import *


RenderMode = Literal["solid", "wireframe"]
MouseMode = Literal["rotate", "pan"]


@dataclass
class MeshData:
    vertices: np.ndarray
    faces: np.ndarray
    color: tuple[float, float, float, float] = (0.7, 0.7, 0.7, 1.0)
    render_mode: RenderMode = "solid"
    unlit: bool = False
    line_width: float = 1.5
    metallic: float = 0.0  # 金属度 (0-1)
    roughness: float = 0.5  # 粗糙度 (0-1)

    _tri_vertices: Optional[np.ndarray] = field(default=None, init=False, repr=False)
    _tri_normals: Optional[np.ndarray] = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self.vertices = self.vertices.astype(np.float32, copy=False)
        self.faces = self.faces.astype(np.int32, copy=False)
        self.line_width = float(self.line_width)
        self.metallic = max(0.0, min(1.0, float(self.metallic)))
        self.roughness = max(0.0, min(1.0, float(self.roughness)))

        if not (self.vertices.size and self.faces.size):
            return

        try:
            tris = self.vertices[self.faces.reshape(-1)]
            v0, v1, v2 = tris[0::3], tris[1::3], tris[2::3]
            n = np.cross(v1 - v0, v2 - v0)
            norm = np.linalg.norm(n, axis=1, keepdims=True)
            n = n / np.maximum(norm, 1e-12)
            tri_normals = np.repeat(n.astype(np.float32, copy=False), 3, axis=0)
            self._tri_vertices = np.ascontiguousarray(tris.astype(np.float32, copy=False))
            self._tri_normals = np.ascontiguousarray(tri_normals)
        except Exception:
            self._tri_vertices = None
            self._tri_normals = None


class GLMeshViewer(QGLWidget):
    """
    3D 网格查看器交互说明：
    - 左键：选择对象
    - 右键拖动：旋转视角
    - 中键拖动：平移视角
    - 滚轮：缩放
    """
    
    # 定义信号
    lasso_completed = QtCore.pyqtSignal(list)  # 发送选中的面索引列表

    def __init__(self, parent=None):
        super().__init__(parent)
        self.meshes: list[MeshData] = []
        self.last_pos = QtCore.QPoint()
        self.rot_x = -30.0
        self.rot_y = -45.0
        self.trans_x = 0.0
        self.trans_y = 0.0
        self.distance = 500.0
        self.mouse_mode: MouseMode = "rotate"
        self._bbox_center = np.zeros(3, dtype=np.float32)
        self._bbox_min = np.zeros(3, dtype=np.float32)
        self._bbox_size = 1.0
        
        # 渲染选项
        self.show_axes = True
        self.show_ground_grid = True
        self.use_smooth_shading = True
        self.ambient_occlusion = False  # 简化版 AO（性能考虑）
        
        # 套索工具
        self.lasso_tool = None  # SurfaceLassoTool 实例
        self.lasso_mode = False  # 是否启用套索模式
        self.current_lasso_curve = None  # 当前正在绘制的曲线
        
        # 平面可视化
        self.show_plane = False
        self.plane_point = None  # 平面上的点
        self.plane_normal = None  # 平面法向量
        self.plane_size = 100.0  # 平面显示大小

    def get_view_state(self) -> dict[str, float]:
        return {
            "rot_x": float(self.rot_x),
            "rot_y": float(self.rot_y),
            "trans_x": float(self.trans_x),
            "trans_y": float(self.trans_y),
            "distance": float(self.distance),
        }

    def apply_view_state(self, state: dict[str, float]) -> None:
        self.rot_x = float(state.get("rot_x", -30.0))
        self.rot_y = float(state.get("rot_y", -45.0))
        self.trans_x = float(state.get("trans_x", 0.0))
        self.trans_y = float(state.get("trans_y", 0.0))
        self.distance = float(state.get("distance", 2.4))
        self.update()

    def reset_view(self) -> None:
        self.rot_x = -30.0
        self.rot_y = -45.0
        self.trans_x = 0.0
        self.trans_y = 0.0
        self.distance = max(5.0, 0.2)  # 固定使用5.0
        self.update()
    
    def set_view_front(self) -> None:
        """前视图（沿 -Y 轴看向 +Y）"""
        self.rot_x = 0.0
        self.rot_y = 0.0
        self.trans_x = 0.0
        self.trans_y = 0.0
        self.update()
    
    def set_view_back(self) -> None:
        """后视图（沿 +Y 轴看向 -Y）"""
        self.rot_x = 0.0
        self.rot_y = 180.0
        self.trans_x = 0.0
        self.trans_y = 0.0
        self.update()
    
    def set_view_left(self) -> None:
        """左视图（沿 +X 轴看向 -X）"""
        self.rot_x = 0.0
        self.rot_y = 90.0
        self.trans_x = 0.0
        self.trans_y = 0.0
        self.update()
    
    def set_view_right(self) -> None:
        """右视图（沿 -X 轴看向 +X）"""
        self.rot_x = 0.0
        self.rot_y = -90.0
        self.trans_x = 0.0
        self.trans_y = 0.0
        self.update()
    
    def set_view_top(self) -> None:
        """顶视图（沿 -Z 轴看向 +Z）"""
        self.rot_x = -90.0
        self.rot_y = 0.0
        self.trans_x = 0.0
        self.trans_y = 0.0
        self.update()
    
    def set_view_bottom(self) -> None:
        """底视图（沿 +Z 轴看向 -Z）"""
        self.rot_x = 90.0
        self.rot_y = 0.0
        self.trans_x = 0.0
        self.trans_y = 0.0
        self.update()
    
    def set_view_isometric(self) -> None:
        """等轴测视图（默认 3D 视角）"""
        self.rot_x = -30.0
        self.rot_y = -45.0
        self.trans_x = 0.0
        self.trans_y = 0.0
        self.update()

    def set_mouse_mode(self, mode: MouseMode) -> None:
        self.mouse_mode = "pan" if mode == "pan" else "rotate"
    
    def enable_lasso_mode(self, mesh: trimesh.Trimesh) -> None:
        """启用套索模式"""
        from surface_lasso import SurfaceLassoTool
        self.lasso_mode = True
        self.lasso_tool = SurfaceLassoTool(self, mesh)
        self.setCursor(QtCore.Qt.CrossCursor)
    
    def disable_lasso_mode(self) -> None:
        """禁用套索模式"""
        self.lasso_mode = False
        self.lasso_tool = None
        self.current_lasso_curve = None
        self.setCursor(QtCore.Qt.ArrowCursor)
        self.update()
    
    def set_plane(self, point: np.ndarray, normal: np.ndarray, size: float = None) -> None:
        """设置要显示的平面"""
        self.plane_point = np.asarray(point, dtype=np.float32)
        self.plane_normal = np.asarray(normal, dtype=np.float32)
        self.plane_normal = self.plane_normal / np.linalg.norm(self.plane_normal)
        if size is not None:
            self.plane_size = size
        else:
            # 默认使用模型大小（增大到1.5倍）
            self.plane_size = self._bbox_size * 1.5
        self.show_plane = True
        self.update()
    
    def hide_plane(self) -> None:
        """隐藏平面"""
        self.show_plane = False
        self.update()
    
    def get_view_direction(self) -> np.ndarray:
        """获取当前视角方向"""
        modelview = glGetDoublev(GL_MODELVIEW_MATRIX)
        view_direction = -np.array([modelview[2], modelview[6], modelview[10]], dtype=np.float64)
        view_direction = view_direction / np.linalg.norm(view_direction)
        return view_direction

    def _apply_pan(self, dx: float, dy: float) -> None:
        width = max(1, self.width())
        height = max(1, self.height())
        aspect = width / float(height)
        fov_y = np.deg2rad(45.0)
        view_h = 2.0 * max(self.distance, 0.2) * np.tan(fov_y * 0.5)
        view_w = view_h * aspect
        self.trans_x += (dx / float(width)) * view_w * 0.85
        self.trans_y -= (dy / float(height)) * view_h * 0.85

    def set_meshes(self, meshes: list[MeshData], reset_view: bool = True):
        self.meshes = meshes
        self._update_bbox()
        if reset_view:
            self.reset_view()
        else:
            self.update()

    def _update_bbox(self):
        valid = [m for m in self.meshes if m.vertices.size > 0 and m.faces.size > 0]
        if not valid:
            self._bbox_center = np.zeros(3, dtype=np.float32)
            self._bbox_min = np.zeros(3, dtype=np.float32)
            self._bbox_size = 1.0
            return
        all_v = np.concatenate([m.vertices for m in valid], axis=0)
        vmin = all_v.min(axis=0)
        vmax = all_v.max(axis=0)
        self._bbox_min = vmin
        self._bbox_center = (vmin + vmax) * 0.5
        self._bbox_size = max(float(np.max(vmax - vmin)), 1.0)

    def initializeGL(self):
        # 深黑色背景
        glClearColor(0.05, 0.05, 0.05, 1.0)
        
        # 深度测试和面剔除
        glEnable(GL_DEPTH_TEST)
        glDepthFunc(GL_LEQUAL)
        glEnable(GL_CULL_FACE)
        glCullFace(GL_BACK)
        
        # 抗锯齿
        glEnable(GL_MULTISAMPLE)
        glHint(GL_LINE_SMOOTH_HINT, GL_NICEST)
        glHint(GL_POLYGON_SMOOTH_HINT, GL_NICEST)
        
        # 禁用透明混合（确保模型完全不透明）
        glDisable(GL_BLEND)
        
        # 光照系统 - 大幅降低光照强度，避免颜色被冲成白色
        glEnable(GL_LIGHTING)
        glEnable(GL_COLOR_MATERIAL)
        glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)
        
        # 主光源 - 大幅降低强度
        glEnable(GL_LIGHT0)
        glLightfv(GL_LIGHT0, GL_POSITION, (1.5, 1.8, 2.0, 0.0))
        glLightfv(GL_LIGHT0, GL_DIFFUSE, (0.35, 0.35, 0.35, 1.0))  # 进一步降低
        glLightfv(GL_LIGHT0, GL_SPECULAR, (0.15, 0.15, 0.15, 1.0))  # 进一步降低
        glLightfv(GL_LIGHT0, GL_AMBIENT, (0.0, 0.0, 0.0, 1.0))
        
        # 补光 - 大幅降低强度
        glEnable(GL_LIGHT1)
        glLightfv(GL_LIGHT1, GL_POSITION, (-1.2, 0.5, 1.0, 0.0))
        glLightfv(GL_LIGHT1, GL_DIFFUSE, (0.15, 0.15, 0.15, 1.0))  # 进一步降低
        glLightfv(GL_LIGHT1, GL_SPECULAR, (0.05, 0.05, 0.05, 1.0))  # 进一步降低
        glLightfv(GL_LIGHT1, GL_AMBIENT, (0.0, 0.0, 0.0, 1.0))
        
        # 全局环境光 - 提高，让颜色主要来自环境光而非直射光
        glLightModelfv(GL_LIGHT_MODEL_AMBIENT, (0.55, 0.55, 0.55, 1.0))

    def resizeGL(self, w: int, h: int):
        glViewport(0, 0, w, max(1, h))
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(45.0, w / float(max(1, h)), 0.1, 10000.0)
        glMatrixMode(GL_MODELVIEW)

    def paintGL(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()
        
        # 相机设置
        gluLookAt(0.0, 0.0, self.distance, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0)
        glTranslatef(self.trans_x, self.trans_y, 0.0)
        glRotatef(self.rot_x, 1.0, 0.0, 0.0)
        glRotatef(self.rot_y, 0.0, 1.0, 0.0)
        glScalef(1.0 / self._bbox_size, 1.0 / self._bbox_size, 1.0 / self._bbox_size)
        glTranslatef(-self._bbox_center[0], -self._bbox_center[1], -self._bbox_center[2])
        
        # 绘制场景
        if self.show_axes:
            self._draw_axes()
        self._draw_meshes()
        
        # 绘制切割平面
        if self.show_plane and self.plane_point is not None and self.plane_normal is not None:
            self._draw_plane()
        
        # 绘制套索曲线
        if self.lasso_mode and self.lasso_tool and self.lasso_tool.current_curve:
            self.lasso_tool.draw_current_curve()
        
        # 绘制信息叠加层（可选）
        self._draw_overlay_info()

    def _draw_axes(self):
        """绘制清晰的坐标轴"""
        s = self._bbox_size * 0.4
        arrow_size = s * 0.08
        
        glDisable(GL_LIGHTING)
        glEnable(GL_LINE_SMOOTH)
        glLineWidth(3.5)
        
        # X 轴 - 鲜红色
        glColor3f(1.0, 0.2, 0.2)
        glBegin(GL_LINES)
        glVertex3f(0, 0, 0)
        glVertex3f(s, 0, 0)
        glEnd()
        # X 轴箭头
        glBegin(GL_TRIANGLES)
        glVertex3f(s, 0, 0)
        glVertex3f(s - arrow_size, arrow_size * 0.4, 0)
        glVertex3f(s - arrow_size, -arrow_size * 0.4, 0)
        glEnd()
        
        # Y 轴 - 鲜绿色
        glColor3f(0.2, 1.0, 0.2)
        glBegin(GL_LINES)
        glVertex3f(0, 0, 0)
        glVertex3f(0, s, 0)
        glEnd()
        # Y 轴箭头
        glBegin(GL_TRIANGLES)
        glVertex3f(0, s, 0)
        glVertex3f(arrow_size * 0.4, s - arrow_size, 0)
        glVertex3f(-arrow_size * 0.4, s - arrow_size, 0)
        glEnd()
        
        # Z 轴 - 鲜蓝色
        glColor3f(0.2, 0.5, 1.0)
        glBegin(GL_LINES)
        glVertex3f(0, 0, 0)
        glVertex3f(0, 0, s)
        glEnd()
        # Z 轴箭头
        glBegin(GL_TRIANGLES)
        glVertex3f(0, 0, s)
        glVertex3f(0, arrow_size * 0.4, s - arrow_size)
        glVertex3f(0, -arrow_size * 0.4, s - arrow_size)
        glEnd()
        
        glDisable(GL_LINE_SMOOTH)
        glLineWidth(1.0)
        glEnable(GL_LIGHTING)
    
    def _draw_plane(self):
        """绘制切割平面"""
        glDisable(GL_LIGHTING)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glDisable(GL_CULL_FACE)
        
        # 计算平面的两个切向量
        normal = self.plane_normal
        
        # 选择一个不平行于法向量的向量
        if abs(normal[2]) < 0.9:
            up = np.array([0, 0, 1], dtype=np.float32)
        else:
            up = np.array([1, 0, 0], dtype=np.float32)
        
        # 计算两个切向量
        tangent1 = np.cross(normal, up)
        tangent1 = tangent1 / np.linalg.norm(tangent1)
        tangent2 = np.cross(normal, tangent1)
        tangent2 = tangent2 / np.linalg.norm(tangent2)
        
        # 平面大小
        size = self.plane_size
        
        # 计算平面的四个角点
        p1 = self.plane_point + tangent1 * size + tangent2 * size
        p2 = self.plane_point - tangent1 * size + tangent2 * size
        p3 = self.plane_point - tangent1 * size - tangent2 * size
        p4 = self.plane_point + tangent1 * size - tangent2 * size
        
        # 绘制半透明平面（黄色）
        glColor4f(1.0, 0.8, 0.0, 0.3)
        glBegin(GL_QUADS)
        glVertex3f(p1[0], p1[1], p1[2])
        glVertex3f(p2[0], p2[1], p2[2])
        glVertex3f(p3[0], p3[1], p3[2])
        glVertex3f(p4[0], p4[1], p4[2])
        glEnd()
        
        # 绘制平面边框（橙色）
        glLineWidth(3.0)
        glColor4f(1.0, 0.5, 0.0, 0.8)
        glBegin(GL_LINE_LOOP)
        glVertex3f(p1[0], p1[1], p1[2])
        glVertex3f(p2[0], p2[1], p2[2])
        glVertex3f(p3[0], p3[1], p3[2])
        glVertex3f(p4[0], p4[1], p4[2])
        glEnd()
        
        # 绘制法向量箭头（红色）
        arrow_length = size * 0.3
        arrow_end = self.plane_point + normal * arrow_length
        
        glLineWidth(4.0)
        glColor4f(1.0, 0.0, 0.0, 0.9)
        glBegin(GL_LINES)
        glVertex3f(self.plane_point[0], self.plane_point[1], self.plane_point[2])
        glVertex3f(arrow_end[0], arrow_end[1], arrow_end[2])
        glEnd()
        
        # 绘制箭头头部
        arrow_head_size = size * 0.05
        glBegin(GL_TRIANGLES)
        head_base = arrow_end - normal * arrow_head_size * 2
        side1 = head_base + tangent1 * arrow_head_size
        side2 = head_base + tangent2 * arrow_head_size
        side3 = head_base - tangent1 * arrow_head_size
        side4 = head_base - tangent2 * arrow_head_size
        
        glVertex3f(arrow_end[0], arrow_end[1], arrow_end[2])
        glVertex3f(side1[0], side1[1], side1[2])
        glVertex3f(side2[0], side2[1], side2[2])
        
        glVertex3f(arrow_end[0], arrow_end[1], arrow_end[2])
        glVertex3f(side2[0], side2[1], side2[2])
        glVertex3f(side3[0], side3[1], side3[2])
        
        glVertex3f(arrow_end[0], arrow_end[1], arrow_end[2])
        glVertex3f(side3[0], side3[1], side3[2])
        glVertex3f(side4[0], side4[1], side4[2])
        
        glVertex3f(arrow_end[0], arrow_end[1], arrow_end[2])
        glVertex3f(side4[0], side4[1], side4[2])
        glVertex3f(side1[0], side1[1], side1[2])
        glEnd()
        
        glDisable(GL_BLEND)
        glEnable(GL_CULL_FACE)
        glLineWidth(1.0)
        glEnable(GL_LIGHTING)
    
    def _draw_ground_grid(self):
        """绘制地面网格"""
        grid_size = self._bbox_size * 0.7
        grid_step = grid_size / 10
        center_x = float(self._bbox_center[0])
        center_y = float(self._bbox_center[1])
        z_ground = float(self._bbox_min[2])
        
        glColor4f(0.6, 0.6, 0.6, 0.3)
        glLineWidth(0.8)
        glBegin(GL_LINES)
        
        # X 方向网格线
        for i in range(-5, 6):
            x = center_x + i * grid_step
            glVertex3f(x, center_y - grid_size / 2, z_ground)
            glVertex3f(x, center_y + grid_size / 2, z_ground)
        
        # Y 方向网格线
        for i in range(-5, 6):
            y = center_y + i * grid_step
            glVertex3f(center_x - grid_size / 2, y, z_ground)
            glVertex3f(center_x + grid_size / 2, y, z_ground)
        
        glEnd()
    
    def _draw_overlay_info(self):
        """绘制叠加信息（简洁版）"""
        # 切换到 2D 正交投影
        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()
        glOrtho(0, self.width(), self.height(), 0, -1, 1)
        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glLoadIdentity()
        
        glDisable(GL_DEPTH_TEST)
        glDisable(GL_LIGHTING)
        
        # 绘制半透明背景（深灰色，在黑背景上可见）
        glColor4f(0.15, 0.15, 0.15, 0.8)
        glBegin(GL_QUADS)
        glVertex2f(10, 10)
        glVertex2f(220, 10)
        glVertex2f(220, 100)
        glVertex2f(10, 100)
        glEnd()
        
        # 统计信息（白色文字）
        glColor3f(1.0, 1.0, 1.0)
        total_verts = sum(len(m.vertices) for m in self.meshes if m.vertices.size > 0)
        total_faces = sum(len(m.faces) for m in self.meshes if m.faces.size > 0)
        
        self.renderText(20, 30, f"顶点: {total_verts:,}", QtGui.QFont("Arial", 10, QtGui.QFont.Bold))
        self.renderText(20, 50, f"面数: {total_faces:,}", QtGui.QFont("Arial", 10, QtGui.QFont.Bold))
        self.renderText(20, 70, f"网格: {len(self.meshes)}", QtGui.QFont("Arial", 10))
        self.renderText(20, 90, f"模式: {self.mouse_mode.upper()}", QtGui.QFont("Arial", 9))
        
        # 恢复投影
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)
        glPopMatrix()
        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)

    def _draw_meshes(self):
        """绘制网格（优化材质以便观察模型细节）"""
        for mesh in self.meshes:
            r, g, b, a = mesh.color
            
            if mesh.unlit:
                # 无光照模式 - 使用纯色渲染，不受光照影响
                glDisable(GL_LIGHTING)
                glColor4f(r, g, b, a)
            else:
                # 有光照模式 - 增强立体感的材质
                glEnable(GL_LIGHTING)
                
                # 关键修复：使用 glColor 设置颜色，因为 GL_COLOR_MATERIAL 已启用
                # GL_COLOR_MATERIAL 会让 glColor 自动设置 GL_AMBIENT_AND_DIFFUSE
                glColor4f(r, g, b, 1.0)
                
                # 镜面反射（降低，避免白色高光覆盖颜色）
                specular = (0.2, 0.2, 0.2, 1.0)
                glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, specular)
                
                # 光泽度（降低，减少高光区域）
                glMaterialf(GL_FRONT_AND_BACK, GL_SHININESS, 20.0)
            
            # 渲染模式
            if mesh.render_mode == "wireframe":
                glDisable(GL_CULL_FACE)
                glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
                glLineWidth(mesh.line_width)
                glEnable(GL_LINE_SMOOTH)
            else:
                glEnable(GL_CULL_FACE)
                glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
                
                # 启用多边形偏移（避免 Z-fighting）
                glEnable(GL_POLYGON_OFFSET_FILL)
                glPolygonOffset(1.0, 1.0)
            
            # 绘制网格
            if mesh._tri_vertices is not None and mesh._tri_normals is not None:
                glEnableClientState(GL_VERTEX_ARRAY)
                glEnableClientState(GL_NORMAL_ARRAY)
                glVertexPointerf(mesh._tri_vertices)
                glNormalPointerf(mesh._tri_normals)
                glDrawArrays(GL_TRIANGLES, 0, int(mesh._tri_vertices.shape[0]))
                glDisableClientState(GL_NORMAL_ARRAY)
                glDisableClientState(GL_VERTEX_ARRAY)
            else:
                # 回退模式（慢）
                glBegin(GL_TRIANGLES)
                verts = mesh.vertices
                for f in mesh.faces:
                    v0, v1, v2 = verts[f[0]], verts[f[1]], verts[f[2]]
                    n = np.cross(v1 - v0, v2 - v0)
                    norm = np.linalg.norm(n)
                    if norm > 1e-9:
                        n /= norm
                    glNormal3f(*n)
                    glVertex3f(*v0)
                    glVertex3f(*v1)
                    glVertex3f(*v2)
                glEnd()
            
            # 恢复默认状态
            glDisable(GL_POLYGON_OFFSET_FILL)
            glDisable(GL_LINE_SMOOTH)
            glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
            glLineWidth(1.0)
            glEnable(GL_CULL_FACE)
            glEnable(GL_LIGHTING)

    def mousePressEvent(self, event: QtGui.QMouseEvent):
        self.last_pos = event.pos()
        
        # 套索模式处理（点击模式）
        if self.lasso_mode and self.lasso_tool and event.button() == QtCore.Qt.LeftButton:
            success = self.lasso_tool.start_drawing(event.x(), event.y())
            if success:
                # 检查是否完成闭合
                if self.lasso_tool.current_curve is None and len(self.lasso_tool.completed_curves) > 0:
                    # 曲线已闭合
                    curve = self.lasso_tool.completed_curves[-1]
                    selected_faces = self.lasso_tool.select_faces_inside_curve(curve)
                    self.lasso_completed.emit(selected_faces)
                self.update()
            return

    def mouseMoveEvent(self, event: QtGui.QMouseEvent):
        dx = event.x() - self.last_pos.x()
        dy = event.y() - self.last_pos.y()
        
        # 套索模式下：右键旋转，中键平移
        if self.lasso_mode and self.lasso_tool:
            if event.buttons() & QtCore.Qt.RightButton:
                # 右键旋转
                self.rot_x += dy * 0.5
                self.rot_y += dx * 0.5
                self.last_pos = event.pos()
                self.update()
            elif event.buttons() & QtCore.Qt.MiddleButton:
                # 中键平移
                self._apply_pan(dx, dy)
                self.last_pos = event.pos()
                self.update()
            return
        
        # 正常模式下左键只保留给对象选择，不参与相机操作。
        if event.buttons() & QtCore.Qt.LeftButton:
            self.last_pos = event.pos()
            return
        if event.buttons() & QtCore.Qt.RightButton:
            self.rot_x += dy * 0.5
            self.rot_y += dx * 0.5
        elif event.buttons() & QtCore.Qt.MiddleButton:
            self._apply_pan(dx, dy)
        self.last_pos = event.pos()
        self.update()
    
    def mouseReleaseEvent(self, event: QtGui.QMouseEvent):
        """鼠标释放事件"""
        # 点击模式下不需要处理释放事件
        pass
    
    def keyPressEvent(self, event: QtGui.QKeyEvent):
        """键盘事件处理"""
        if self.lasso_mode and self.lasso_tool:
            if event.key() == QtCore.Qt.Key_Escape:
                # ESC 取消当前绘制
                self.lasso_tool.cancel_drawing()
                self.update()
                return
            elif event.key() == QtCore.Qt.Key_Backspace or event.key() == QtCore.Qt.Key_Z:
                # Backspace 或 Z 撤销最后一个点
                if self.lasso_tool.undo_last_point():
                    self.update()
                return
            elif event.key() == QtCore.Qt.Key_Return or event.key() == QtCore.Qt.Key_Enter:
                # Enter 强制闭合（如果点数足够）
                if self.lasso_tool.current_curve and len(self.lasso_tool.current_curve.points_3d) >= self.lasso_tool.min_points:
                    self.lasso_tool.current_curve.close()
                    curve = self.lasso_tool.current_curve
                    self.lasso_tool.completed_curves.append(curve)
                    self.lasso_tool.current_curve = None
                    self.lasso_tool.is_drawing = False
                    selected_faces = self.lasso_tool.select_faces_inside_curve(curve)
                    self.lasso_completed.emit(selected_faces)
                    self.update()
                return
        
        # 调用父类处理
        super().keyPressEvent(event)

    def wheelEvent(self, event: QtGui.QWheelEvent):
        delta = event.angleDelta().y()
        self.distance *= 0.9 if delta > 0 else 1.1
        self.distance = max(self._bbox_size * 0.001, min(self._bbox_size * 10.0, self.distance))
        self.update()


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("鞋底晶格查看器")
        self.resize(1024, 768)
        self.viewer = GLMeshViewer(self)
        self.setCentralWidget(self.viewer)
        self._create_actions()
        self._create_menu()

    def _create_actions(self):
        self.action_quit = QtWidgets.QAction("退出", self)
        self.action_quit.triggered.connect(QtWidgets.qApp.quit)
        self.action_show_sole = QtWidgets.QAction("显示鞋底", self)
        self.action_show_lattice = QtWidgets.QAction("显示晶格", self)
        self.action_show_combined = QtWidgets.QAction("显示组合", self)
        self.action_toggle_mesh_edges = QtWidgets.QAction("显示三角面片边缘", self)
        self.action_toggle_mesh_edges.setCheckable(True)
        self.action_toggle_mesh_edges.setChecked(False)
        
        # 标准视图动作
        self.action_view_front = QtWidgets.QAction("前视图 (Front)", self)
        self.action_view_front.setShortcut("1")
        self.action_view_front.triggered.connect(self.viewer.set_view_front)
        
        self.action_view_back = QtWidgets.QAction("后视图 (Back)", self)
        self.action_view_back.setShortcut("Ctrl+1")
        self.action_view_back.triggered.connect(self.viewer.set_view_back)
        
        self.action_view_left = QtWidgets.QAction("左视图 (Left)", self)
        self.action_view_left.setShortcut("3")
        self.action_view_left.triggered.connect(self.viewer.set_view_left)
        
        self.action_view_right = QtWidgets.QAction("右视图 (Right)", self)
        self.action_view_right.setShortcut("Ctrl+3")
        self.action_view_right.triggered.connect(self.viewer.set_view_right)
        
        self.action_view_top = QtWidgets.QAction("顶视图 (Top)", self)
        self.action_view_top.setShortcut("7")
        self.action_view_top.triggered.connect(self.viewer.set_view_top)
        
        self.action_view_bottom = QtWidgets.QAction("底视图 (Bottom)", self)
        self.action_view_bottom.setShortcut("Ctrl+7")
        self.action_view_bottom.triggered.connect(self.viewer.set_view_bottom)
        
        self.action_view_isometric = QtWidgets.QAction("等轴测视图 (Isometric)", self)
        self.action_view_isometric.setShortcut("0")
        self.action_view_isometric.triggered.connect(self.viewer.set_view_isometric)
        
        self.action_reset_view = QtWidgets.QAction("重置视图", self)
        self.action_reset_view.setShortcut("Home")
        self.action_reset_view.triggered.connect(lambda: self.viewer.reset_view())

    def _create_menu(self):
        menubar = self.menuBar()
        file_menu = menubar.addMenu("文件")
        file_menu.addAction(self.action_quit)
        
        view_menu = menubar.addMenu("视图")
        view_menu.addAction(self.action_show_sole)
        view_menu.addAction(self.action_show_lattice)
        view_menu.addAction(self.action_show_combined)
        view_menu.addSeparator()
        view_menu.addAction(self.action_toggle_mesh_edges)
        view_menu.addSeparator()
        
        # 标准视图子菜单
        standard_views_menu = view_menu.addMenu("标准视图")
        standard_views_menu.addAction(self.action_view_front)
        standard_views_menu.addAction(self.action_view_back)
        standard_views_menu.addSeparator()
        standard_views_menu.addAction(self.action_view_left)
        standard_views_menu.addAction(self.action_view_right)
        standard_views_menu.addSeparator()
        standard_views_menu.addAction(self.action_view_top)
        standard_views_menu.addAction(self.action_view_bottom)
        standard_views_menu.addSeparator()
        standard_views_menu.addAction(self.action_view_isometric)
        
        view_menu.addAction(self.action_reset_view)


def run_viewer(
    sole_vertices: np.ndarray,
    sole_faces: np.ndarray,
    lattice_vertices: np.ndarray,
    lattice_faces: np.ndarray,
) -> None:
    """启动查看器并显示鞋底与晶格两层网格。"""
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()

    sole_mesh = MeshData(sole_vertices, sole_faces, color=(0.78, 0.80, 0.82, 1.0))
    sole_edges = MeshData(sole_vertices, sole_faces, color=(0.2, 0.2, 0.2, 1.0), render_mode="wireframe", unlit=True, line_width=1.0)
    lattice_mesh = MeshData(
        lattice_vertices,
        lattice_faces,
        color=(0.72, 0.82, 0.92, 0.92),
        render_mode="solid",
        unlit=False,
    )
    lattice_edges = MeshData(
        lattice_vertices,
        lattice_faces,
        color=(0.3, 0.3, 0.3, 1.0),
        render_mode="wireframe",
        unlit=True,
        line_width=1.0,
    )
    lattice_wire = MeshData(
        lattice_vertices,
        lattice_faces,
        color=(0.12, 0.22, 0.34, 1.0),
        render_mode="wireframe",
        unlit=True,
        line_width=1.4,
    )

    def current_sole_meshes() -> list[MeshData]:
        meshes = [sole_mesh]
        if window.action_toggle_mesh_edges.isChecked():
            meshes.append(sole_edges)
        return meshes

    def current_lattice_meshes() -> list[MeshData]:
        meshes = [lattice_mesh]
        if window.action_toggle_mesh_edges.isChecked():
            meshes.append(lattice_edges)
        return meshes

    def show_sole() -> None:
        window.viewer.set_meshes(current_sole_meshes())

    def show_lattice() -> None:
        window.viewer.set_meshes(current_lattice_meshes())

    def show_combined() -> None:
        window.viewer.set_meshes([*current_sole_meshes(), *current_lattice_meshes()])

    def refresh_view() -> None:
        if window.viewer.meshes and all(mesh is sole_mesh for mesh in window.viewer.meshes):
            show_sole()
            return
        if any(mesh is sole_mesh for mesh in window.viewer.meshes):
            show_combined()
        else:
            show_lattice()

    window.action_show_sole.triggered.connect(show_sole)
    window.action_show_lattice.triggered.connect(show_lattice)
    window.action_show_combined.triggered.connect(show_combined)
    window.action_toggle_mesh_edges.toggled.connect(lambda _: refresh_view())

    show_combined()
    window.show()
    sys.exit(app.exec_())
