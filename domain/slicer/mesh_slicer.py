"""
网格分割模块
提供精准的平面分割功能，支持 C++ 加速
"""

import numpy as np
import trimesh
from typing import Tuple

# 尝试导入 C++ 扩展
CPP_AVAILABLE = False
try:
    from core.cpp import cpp_mesh_slicer
    CPP_AVAILABLE = True
    print("[加速] C++ 网格分割模块已加载 ✓")
except ImportError as e:
    # 尝试添加项目根目录到路径（当文件被直接运行时）
    try:
        import sys
        from pathlib import Path
        project_root = Path(__file__).parent.parent.parent
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))
        from core.cpp import cpp_mesh_slicer
        CPP_AVAILABLE = True
        print("[加速] C++ 网格分割模块已加载 ✓")
    except ImportError as e2:
        print("[提示] C++ 网格分割模块未找到，使用 Python 版本（较慢）")
        print(f"[调试] 导入错误: {e2}")


class MeshSlicer:
    """网格分割器"""
    
    def __init__(self, mesh: trimesh.Trimesh):
        """
        初始化分割器
        
        参数:
            mesh: 要分割的 trimesh.Trimesh 对象
        """
        self.mesh = mesh
        self.bounds = mesh.bounds
        self.use_cpp = CPP_AVAILABLE
    
    def get_axis_range(self, axis: str) -> Tuple[float, float]:
        """
        获取指定轴的范围
        
        参数:
            axis: 'x', 'y', 或 'z'
        
        返回:
            (min_value, max_value)
        """
        axis_idx = {'x': 0, 'y': 1, 'z': 2}[axis.lower()]
        return float(self.bounds[0][axis_idx]), float(self.bounds[1][axis_idx])
    
    def subdivide_mesh(self, iterations: int = 1) -> trimesh.Trimesh:
        """
        细化网格（增加三角形密度）
        
        参数:
            iterations: 细化迭代次数，每次迭代会将每个三角形分成4个
        
        返回:
            细化后的网格
        """
        mesh = self.mesh.copy()
        
        for i in range(iterations):
            mesh = mesh.subdivide()
        
        return mesh
    
    def slice_by_plane(
        self, 
        axis: str, 
        position: float,
        subdivide: int = 0,
        angle1: float = 0.0,
        angle2: float = 0.0
    ) -> Tuple[trimesh.Trimesh, trimesh.Trimesh]:
        """
        按轴对齐平面分割网格（支持旋转）
        
        参数:
            axis: 分割轴 ('x', 'y', 或 'z')
            position: 平面在该轴上的位置
            subdivide: 细化迭代次数（0=不细化）
            angle1: 绕第一个旋转轴的角度（度）
            angle2: 绕第二个旋转轴的角度（度）
        
        返回:
            (positive_mesh, negative_mesh)
            - positive_mesh: 在平面正侧的部分（坐标值 >= position）
            - negative_mesh: 在平面负侧的部分（坐标值 < position）
        """
        # 如果需要细化
        if subdivide > 0:
            mesh_to_slice = self.subdivide_mesh(iterations=subdivide)
            temp_slicer = MeshSlicer(mesh_to_slice)
            return temp_slicer._slice_precise(axis, position, angle1, angle2)
        else:
            return self._slice_precise(axis, position, angle1, angle2)
    
    def _slice_precise(
        self, 
        axis: str, 
        position: float,
        angle1: float = 0.0,
        angle2: float = 0.0
    ) -> Tuple[trimesh.Trimesh, trimesh.Trimesh]:
        """
        精准分割（内部方法）
        
        参数:
            axis: 分割轴 ('x', 'y', 或 'z')
            position: 平面在该轴上的位置
            angle1: 绕第一个旋转轴的角度（度）
            angle2: 绕第二个旋转轴的角度（度）
        
        返回:
            (positive_mesh, negative_mesh)
        """
        axis_idx = {'x': 0, 'y': 1, 'z': 2}[axis.lower()]
        
        # 如果有角度旋转，使用通用平面分割
        if angle1 != 0.0 or angle2 != 0.0:
            return self._slice_by_rotated_plane(axis_idx, position, angle1, angle2)
        
        # 否则使用原来的轴对齐分割（更快）
        if self.use_cpp and CPP_AVAILABLE:
            # 使用 C++ 加速版本
            return self._slice_cpp(axis_idx, position)
        else:
            # 使用纯 Python 版本
            return self._slice_python(axis_idx, position)
    
    def _slice_cpp(
        self, 
        axis_idx: int, 
        position: float
    ) -> Tuple[trimesh.Trimesh, trimesh.Trimesh]:
        """使用 C++ 加速的精准分割"""
        print("[分割] 使用 C++ 加速版本 ⚡")
        
        vertices = np.asarray(self.mesh.vertices, dtype=np.float64)
        faces = np.asarray(self.mesh.faces, dtype=np.int32)
        
        import time
        t_start = time.time()
        
        # 调用 C++ 扩展
        pos_verts, pos_faces, neg_verts, neg_faces = cpp_mesh_slicer.slice_mesh_precise(
            vertices, faces, axis_idx, position
        )
        
        t_end = time.time()
        print(f"[分割] C++ 计算耗时: {t_end - t_start:.3f}s")
        
        # 创建网格
        if len(pos_verts) > 0 and len(pos_faces) > 0:
            positive_mesh = trimesh.Trimesh(
                vertices=pos_verts,
                faces=pos_faces,
                process=True
            )
        else:
            positive_mesh = trimesh.Trimesh()
        
        if len(neg_verts) > 0 and len(neg_faces) > 0:
            negative_mesh = trimesh.Trimesh(
                vertices=neg_verts,
                faces=neg_faces,
                process=True
            )
        else:
            negative_mesh = trimesh.Trimesh()
        
        return positive_mesh, negative_mesh
    
    def _slice_python(
        self, 
        axis_idx: int, 
        position: float
    ) -> Tuple[trimesh.Trimesh, trimesh.Trimesh]:
        """纯 Python 实现的精准分割"""
        print("[分割] 使用 Python 版本（较慢）")
        
        import time
        t_start = time.time()
        
        # 创建平面
        plane_origin = np.zeros(3)
        plane_origin[axis_idx] = position
        plane_normal = np.zeros(3)
        plane_normal[axis_idx] = 1.0
        
        # 计算所有顶点到平面的有向距离
        vertices = self.mesh.vertices
        distances = np.dot(vertices - plane_origin, plane_normal)
        
        # 容差
        eps = 1e-10
        
        # 分类顶点
        vertex_signs = np.sign(distances)
        vertex_signs[np.abs(distances) < eps] = 0
        
        # 存储结果
        pos_verts_list = []
        pos_faces_list = []
        neg_verts_list = []
        neg_faces_list = []
        
        # 顶点映射
        pos_vert_map = {}
        neg_vert_map = {}
        
        def add_pos_vert(v):
            key = tuple(v)
            if key not in pos_vert_map:
                pos_vert_map[key] = len(pos_verts_list)
                pos_verts_list.append(v.copy())
            return pos_vert_map[key]
        
        def add_neg_vert(v):
            key = tuple(v)
            if key not in neg_vert_map:
                neg_vert_map[key] = len(neg_verts_list)
                neg_verts_list.append(v.copy())
            return neg_vert_map[key]
        
        # 处理每个三角面
        for face in self.mesh.faces:
            v0, v1, v2 = vertices[face]
            s0, s1, s2 = vertex_signs[face]
            
            # 情况1：所有顶点在正侧或平面上
            if s0 >= 0 and s1 >= 0 and s2 >= 0:
                i0 = add_pos_vert(v0)
                i1 = add_pos_vert(v1)
                i2 = add_pos_vert(v2)
                pos_faces_list.append([i0, i1, i2])
            
            # 情况2：所有顶点在负侧或平面上
            elif s0 <= 0 and s1 <= 0 and s2 <= 0:
                i0 = add_neg_vert(v0)
                i1 = add_neg_vert(v1)
                i2 = add_neg_vert(v2)
                neg_faces_list.append([i0, i1, i2])
            
            # 情况3：三角面跨越平面
            else:
                verts_list = [v0, v1, v2]
                signs_list = [s0, s1, s2]
                
                # 找交点
                intersections = []
                for i in range(3):
                    j = (i + 1) % 3
                    si, sj = signs_list[i], signs_list[j]
                    
                    if si * sj < 0:
                        vi, vj = verts_list[i], verts_list[j]
                        di, dj = distances[face[i]], distances[face[j]]
                        
                        t = di / (di - dj)
                        intersection = vi + t * (vj - vi)
                        intersection[axis_idx] = position
                        
                        intersections.append(intersection)
                
                if len(intersections) != 2:
                    continue
                
                p1 = intersections[0]
                p2 = intersections[1]
                
                pos_verts = [verts_list[i] for i in range(3) if signs_list[i] > 0]
                neg_verts = [verts_list[i] for i in range(3) if signs_list[i] < 0]
                
                if len(pos_verts) == 1:
                    i0 = add_pos_vert(pos_verts[0])
                    i1 = add_pos_vert(p1)
                    i2 = add_pos_vert(p2)
                    pos_faces_list.append([i0, i1, i2])
                    
                    i0 = add_neg_vert(neg_verts[0])
                    i1 = add_neg_vert(neg_verts[1])
                    i2 = add_neg_vert(p1)
                    i3 = add_neg_vert(p2)
                    neg_faces_list.append([i0, i1, i2])
                    neg_faces_list.append([i1, i3, i2])
                
                elif len(neg_verts) == 1:
                    i0 = add_neg_vert(neg_verts[0])
                    i1 = add_neg_vert(p1)
                    i2 = add_neg_vert(p2)
                    neg_faces_list.append([i0, i1, i2])
                    
                    i0 = add_pos_vert(pos_verts[0])
                    i1 = add_pos_vert(pos_verts[1])
                    i2 = add_pos_vert(p1)
                    i3 = add_pos_vert(p2)
                    pos_faces_list.append([i0, i1, i2])
                    pos_faces_list.append([i1, i3, i2])
        
        # 创建网格
        if len(pos_verts_list) > 0 and len(pos_faces_list) > 0:
            positive_mesh = trimesh.Trimesh(
                vertices=np.array(pos_verts_list),
                faces=np.array(pos_faces_list),
                process=True
            )
        else:
            positive_mesh = trimesh.Trimesh()
        
        if len(neg_verts_list) > 0 and len(neg_faces_list) > 0:
            negative_mesh = trimesh.Trimesh(
                vertices=np.array(neg_verts_list),
                faces=np.array(neg_faces_list),
                process=True
            )
        else:
            negative_mesh = trimesh.Trimesh()
        
        t_end = time.time()
        print(f"[分割] Python 计算耗时: {t_end - t_start:.3f}s")
        
        return positive_mesh, negative_mesh
    
    def _slice_by_rotated_plane(
        self,
        axis_idx: int,
        position: float,
        angle1: float,
        angle2: float
    ) -> Tuple[trimesh.Trimesh, trimesh.Trimesh]:
        """
        使用旋转后的平面进行分割
        
        参数:
            axis_idx: 主分割轴索引 (0=X, 1=Y, 2=Z)
            position: 平面在主轴上的位置
            angle1: 绕第一个旋转轴的角度（度）
            angle2: 绕第二个旋转轴的角度（度）
        
        返回:
            (positive_mesh, negative_mesh)
        
        注意：
            旋转是绕着分割轴上的指定位置点进行的，该点位于模型包围盒中心的投影位置。
            这样可以确保旋转后的平面仍然通过用户指定的位置。
        """
        # 确定旋转轴的顺序
        # 对于主轴 X(0): angle1绕Y(1)，angle2绕Z(2)
        # 对于主轴 Y(1): angle1绕X(0)，angle2绕Z(2)
        # 对于主轴 Z(2): angle1绕X(0)，angle2绕Y(1)
        if axis_idx == 0:  # X 轴
            rot_axis1, rot_axis2 = 1, 2  # Y, Z
        elif axis_idx == 1:  # Y 轴
            rot_axis1, rot_axis2 = 0, 2  # X, Z
        else:  # Z 轴
            rot_axis1, rot_axis2 = 0, 1  # X, Y
        
        # 创建初始平面（垂直于主轴）
        # 平面原点：在分割轴上使用指定位置，其他轴使用包围盒中心
        # 这样旋转时，平面会绕着这个点旋转，保持在指定位置附近
        plane_origin = self.mesh.bounds.mean(axis=0)
        plane_origin[axis_idx] = position
        
        # 初始法向量（垂直于主轴）
        plane_normal = np.zeros(3)
        plane_normal[axis_idx] = 1.0
        
        # 应用旋转到法向量
        # 注意：我们只旋转法向量，不旋转原点
        # 这样平面仍然通过 plane_origin，但方向改变了
        
        # 先绕第一个轴旋转
        if angle1 != 0.0:
            angle1_rad = np.radians(angle1)
            rotation_matrix1 = self._get_rotation_matrix(rot_axis1, angle1_rad)
            plane_normal = rotation_matrix1 @ plane_normal
        
        # 再绕第二个轴旋转
        if angle2 != 0.0:
            angle2_rad = np.radians(angle2)
            rotation_matrix2 = self._get_rotation_matrix(rot_axis2, angle2_rad)
            plane_normal = rotation_matrix2 @ plane_normal
        
        # 归一化法向量
        plane_normal = plane_normal / np.linalg.norm(plane_normal)
        
        # 使用通用平面分割
        return self._slice_by_general_plane(plane_origin, plane_normal)
    
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
            ])
        elif axis_idx == 1:  # 绕 Y 轴
            return np.array([
                [c, 0, s],
                [0, 1, 0],
                [-s, 0, c]
            ])
        else:  # 绕 Z 轴
            return np.array([
                [c, -s, 0],
                [s, c, 0],
                [0, 0, 1]
            ])
    
    def _slice_by_general_plane(
        self,
        plane_origin: np.ndarray,
        plane_normal: np.ndarray
    ) -> Tuple[trimesh.Trimesh, trimesh.Trimesh]:
        """
        使用任意平面进行分割
        
        参数:
            plane_origin: 平面上的一点
            plane_normal: 平面法向量（已归一化）
        
        返回:
            (positive_mesh, negative_mesh)
        """
        # 计算所有顶点到平面的有向距离
        vertices = self.mesh.vertices
        distances = np.dot(vertices - plane_origin, plane_normal)
        
        # 容差
        eps = 1e-10
        
        # 分类顶点
        vertex_signs = np.sign(distances)
        vertex_signs[np.abs(distances) < eps] = 0
        
        # 存储结果
        pos_verts_list = []
        pos_faces_list = []
        neg_verts_list = []
        neg_faces_list = []
        
        # 顶点映射
        pos_vert_map = {}
        neg_vert_map = {}
        
        def add_pos_vert(v):
            key = tuple(v)
            if key not in pos_vert_map:
                pos_vert_map[key] = len(pos_verts_list)
                pos_verts_list.append(v.copy())
            return pos_vert_map[key]
        
        def add_neg_vert(v):
            key = tuple(v)
            if key not in neg_vert_map:
                neg_vert_map[key] = len(neg_verts_list)
                neg_verts_list.append(v.copy())
            return neg_vert_map[key]
        
        # 处理每个三角面
        for face in self.mesh.faces:
            v0, v1, v2 = vertices[face]
            s0, s1, s2 = vertex_signs[face]
            
            # 情况1：所有顶点在正侧或平面上
            if s0 >= 0 and s1 >= 0 and s2 >= 0:
                i0 = add_pos_vert(v0)
                i1 = add_pos_vert(v1)
                i2 = add_pos_vert(v2)
                pos_faces_list.append([i0, i1, i2])
            
            # 情况2：所有顶点在负侧或平面上
            elif s0 <= 0 and s1 <= 0 and s2 <= 0:
                i0 = add_neg_vert(v0)
                i1 = add_neg_vert(v1)
                i2 = add_neg_vert(v2)
                neg_faces_list.append([i0, i1, i2])
            
            # 情况3：三角面跨越平面
            else:
                verts_list = [v0, v1, v2]
                signs_list = [s0, s1, s2]
                
                # 找交点
                intersections = []
                for i in range(3):
                    j = (i + 1) % 3
                    si, sj = signs_list[i], signs_list[j]
                    
                    if si * sj < 0:
                        vi, vj = verts_list[i], verts_list[j]
                        di, dj = distances[face[i]], distances[face[j]]
                        
                        t = di / (di - dj)
                        intersection = vi + t * (vj - vi)
                        
                        # 将交点投影到平面上以确保精度
                        dist_to_plane = np.dot(intersection - plane_origin, plane_normal)
                        intersection = intersection - dist_to_plane * plane_normal
                        
                        intersections.append(intersection)
                
                if len(intersections) != 2:
                    continue
                
                p1 = intersections[0]
                p2 = intersections[1]
                
                pos_verts = [verts_list[i] for i in range(3) if signs_list[i] > 0]
                neg_verts = [verts_list[i] for i in range(3) if signs_list[i] < 0]
                
                if len(pos_verts) == 1:
                    i0 = add_pos_vert(pos_verts[0])
                    i1 = add_pos_vert(p1)
                    i2 = add_pos_vert(p2)
                    pos_faces_list.append([i0, i1, i2])
                    
                    i0 = add_neg_vert(neg_verts[0])
                    i1 = add_neg_vert(neg_verts[1])
                    i2 = add_neg_vert(p1)
                    i3 = add_neg_vert(p2)
                    neg_faces_list.append([i0, i1, i2])
                    neg_faces_list.append([i1, i3, i2])
                
                elif len(neg_verts) == 1:
                    i0 = add_neg_vert(neg_verts[0])
                    i1 = add_neg_vert(p1)
                    i2 = add_neg_vert(p2)
                    neg_faces_list.append([i0, i1, i2])
                    
                    i0 = add_pos_vert(pos_verts[0])
                    i1 = add_pos_vert(pos_verts[1])
                    i2 = add_pos_vert(p1)
                    i3 = add_pos_vert(p2)
                    pos_faces_list.append([i0, i1, i2])
                    pos_faces_list.append([i1, i3, i2])
        
        # 创建网格
        if len(pos_verts_list) > 0 and len(pos_faces_list) > 0:
            positive_mesh = trimesh.Trimesh(
                vertices=np.array(pos_verts_list),
                faces=np.array(pos_faces_list),
                process=True
            )
        else:
            positive_mesh = trimesh.Trimesh()
        
        if len(neg_verts_list) > 0 and len(neg_faces_list) > 0:
            negative_mesh = trimesh.Trimesh(
                vertices=np.array(neg_verts_list),
                faces=np.array(neg_faces_list),
                process=True
            )
        else:
            negative_mesh = trimesh.Trimesh()
        
        return positive_mesh, negative_mesh


def is_cpp_available() -> bool:
    """检查 C++ 扩展是否可用"""
    return CPP_AVAILABLE
