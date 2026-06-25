"""
基于平面边界的区域管理模块（方案B）
不分割 STL 模型，只用平面定义区域边界
"""

from __future__ import annotations

import json
import uuid
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Dict, Any

import numpy as np
import trimesh


@dataclass
class PlaneRegion:
    """基于平面边界的区域数据模型"""
    id: str                    # 唯一标识符 (UUID)
    name: str                  # 区域名称（用户可编辑）
    
    # 平面边界定义
    boundary_planes: List[Dict[str, Any]] = field(default_factory=list)
    # 例如: [
    #   {"axis": "y", "position": 50.0, "side": "negative"},  # Y < 50
    #   {"axis": "z", "position": 5.0, "side": "positive"}    # Z >= 5
    # ]
    
    # 晶格参数
    lattice_method: str = "gyroid"
    
    # Gyroid 参数
    gyroid_cell_size: float = 4.0
    gyroid_isovalue: float = 0.3
    gyroid_resolution: int = 30
    
    # Voronoi 参数
    voronoi_cell_size: float = 4.2
    voronoi_strut_thickness: float = 0.55
    voronoi_z_layers: int = 7
    
    # Tile 参数
    tile_shrink: float = 0.28
    tile_spacing: float = 0.96
    tile_margin: float = 1.0
    tile_decimate: bool = False  # 是否启用减面操作
    tile_decimate_target_faces: int = 500  # 晶格单元目标面数（0=禁用）
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典（用于 JSON 序列化）"""
        return {
            'id': self.id,
            'name': self.name,
            'boundary_planes': self.boundary_planes,
            'lattice_method': self.lattice_method,
            'gyroid_cell_size': float(self.gyroid_cell_size),
            'gyroid_isovalue': float(self.gyroid_isovalue),
            'gyroid_resolution': int(self.gyroid_resolution),
            'voronoi_cell_size': float(self.voronoi_cell_size),
            'voronoi_strut_thickness': float(self.voronoi_strut_thickness),
            'voronoi_z_layers': int(self.voronoi_z_layers),
            'tile_shrink': float(self.tile_shrink),
            'tile_spacing': float(self.tile_spacing),
            'tile_margin': float(self.tile_margin),
            'tile_decimate': bool(self.tile_decimate),
            'tile_decimate_target_faces': int(self.tile_decimate_target_faces),
        }
    
    @staticmethod
    def from_dict(data: Dict[str, Any]) -> PlaneRegion:
        """从字典创建 PlaneRegion 对象"""
        return PlaneRegion(
            id=data['id'],
            name=data['name'],
            boundary_planes=data.get('boundary_planes', []),
            lattice_method=data['lattice_method'],
            gyroid_cell_size=float(data.get('gyroid_cell_size', 4.0)),
            gyroid_isovalue=float(data.get('gyroid_isovalue', 0.3)),
            gyroid_resolution=int(data.get('gyroid_resolution', 30)),
            voronoi_cell_size=float(data.get('voronoi_cell_size', 4.2)),
            voronoi_strut_thickness=float(data.get('voronoi_strut_thickness', 0.55)),
            voronoi_z_layers=int(data.get('voronoi_z_layers', 7)),
            tile_shrink=float(data.get('tile_shrink', 0.28)),
            tile_spacing=float(data.get('tile_spacing', 0.96)),
            tile_margin=float(data.get('tile_margin', 1.0)),
            tile_decimate=bool(data.get('tile_decimate', False)),
            tile_decimate_target_faces=int(data.get('tile_decimate_target_faces', 500)),
        )
    
    def get_display_name(self) -> str:
        """获取显示名称（包含边界信息）"""
        if len(self.boundary_planes) == 0:
            return f"{self.name} (整体)"
        
        # 显示第一个边界平面的信息
        plane = self.boundary_planes[0]
        axis = plane['axis'].upper()
        pos = plane['position']
        side = '≥' if plane['side'] == 'positive' else '<'
        
        if len(self.boundary_planes) == 1:
            return f"{self.name} ({axis} {side} {pos:.1f})"
        else:
            return f"{self.name} ({len(self.boundary_planes)} 个边界)"
    
    def get_boundary_description(self) -> str:
        """获取边界描述"""
        if len(self.boundary_planes) == 0:
            return "无边界限制"
        
        descriptions = []
        for plane in self.boundary_planes:
            axis = plane['axis'].upper()
            pos = plane['position']
            side = plane['side']
            
            if side == 'positive':
                descriptions.append(f"{axis} ≥ {pos:.2f} mm")
            else:
                descriptions.append(f"{axis} < {pos:.2f} mm")
        
        return " 且 ".join(descriptions)


class PlaneRegionManager:
    """基于平面边界的区域管理器"""
    
    def __init__(self):
        self.regions: List[PlaneRegion] = []
        self.region_counter: int = 0
        # 存储生成的晶格（不序列化到 JSON）
        self._region_lattices: Dict[str, trimesh.Trimesh] = {}
    
    def add_region(
        self, 
        boundary_planes: List[Dict[str, Any]],
        lattice_method: str = "gyroid"
    ) -> PlaneRegion:
        """
        添加新区域
        
        参数:
            boundary_planes: 边界平面列表
            lattice_method: 默认晶格方法
        
        返回:
            新创建的 PlaneRegion 对象
        """
        self.region_counter += 1
        region_id = str(uuid.uuid4())
        region_name = f"区域 {self.region_counter}"
        
        region = PlaneRegion(
            id=region_id,
            name=region_name,
            boundary_planes=boundary_planes,
            lattice_method=lattice_method,
        )
        
        self.regions.append(region)
        
        return region
    
    def remove_region(self, region_id: str) -> bool:
        """删除区域"""
        region = self.get_region(region_id)
        if region is None:
            return False
        
        self.regions.remove(region)
        
        # 清理关联数据
        if region_id in self._region_lattices:
            del self._region_lattices[region_id]
        
        return True
    
    def get_region(self, region_id: str) -> Optional[PlaneRegion]:
        """获取区域"""
        for region in self.regions:
            if region.id == region_id:
                return region
        return None
    
    def get_region_lattice(self, region_id: str) -> Optional[trimesh.Trimesh]:
        """获取区域生成的晶格"""
        return self._region_lattices.get(region_id)
    
    def set_region_lattice(self, region_id: str, lattice: trimesh.Trimesh) -> None:
        """设置区域生成的晶格"""
        self._region_lattices[region_id] = lattice
    
    def update_region_name(self, region_id: str, name: str) -> bool:
        """更新区域名称"""
        region = self.get_region(region_id)
        if region is None:
            return False
        
        region.name = name
        return True
    
    def update_region_parameters(self, region_id: str, **params) -> bool:
        """更新区域参数"""
        region = self.get_region(region_id)
        if region is None:
            return False
        
        # 更新参数
        for key, value in params.items():
            if hasattr(region, key):
                setattr(region, key, value)
        
        # 清除该区域的晶格缓存（参数变化需要重新生成）
        if region_id in self._region_lattices:
            del self._region_lattices[region_id]
        
        return True
    
    def get_all_regions(self) -> List[PlaneRegion]:
        """获取所有区域"""
        return self.regions.copy()
    
    def clear_all_lattices(self) -> None:
        """清除所有生成的晶格"""
        self._region_lattices.clear()
    
    def has_generated_lattices(self) -> bool:
        """检查是否有已生成的晶格"""
        return len(self._region_lattices) > 0
    
    def get_combined_lattice(self) -> Optional[trimesh.Trimesh]:
        """获取合并后的晶格"""
        if len(self._region_lattices) == 0:
            return None
        
        lattices = []
        for region in self.regions:
            lattice = self._region_lattices.get(region.id)
            if lattice is not None and lattice.faces.size > 0:
                lattices.append(lattice)
        
        if len(lattices) == 0:
            return None
        
        if len(lattices) == 1:
            return lattices[0].copy()
        
        # 合并所有晶格
        combined = trimesh.util.concatenate(lattices)
        
        # 移除重复顶点
        try:
            combined.merge_vertices()
        except Exception:
            pass
        
        return combined
    
    def save_to_json(self, filepath: str) -> None:
        """保存配置到 JSON 文件"""
        config = {
            'version': '1.0',
            'type': 'plane_boundary',
            'region_counter': self.region_counter,
            'regions': [region.to_dict() for region in self.regions]
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
    
    def load_from_json(self, filepath: str) -> None:
        """从 JSON 文件加载配置"""
        with open(filepath, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        # 验证版本
        version = config.get('version', '1.0')
        if version != '1.0':
            raise ValueError(f"不支持的配置文件版本: {version}")
        
        # 验证类型
        config_type = config.get('type', 'plane_boundary')
        if config_type != 'plane_boundary':
            raise ValueError(f"配置文件类型不匹配: {config_type}")
        
        # 清空当前数据
        self.regions.clear()
        self._region_lattices.clear()
        
        # 加载区域计数器
        self.region_counter = config.get('region_counter', 0)
        
        # 加载区域
        regions_data = config.get('regions', [])
        for region_data in regions_data:
            region = PlaneRegion.from_dict(region_data)
            self.regions.append(region)


def clip_lattice_by_plane(
    lattice: trimesh.Trimesh,
    axis: str,
    position: float,
    side: str,
    angle1: float = 0.0,
    angle2: float = 0.0,
    reference_bounds: np.ndarray = None
) -> trimesh.Trimesh:
    """
    用平面裁剪晶格（保留指定侧）
    
    参数:
        lattice: 要裁剪的晶格网格
        axis: 平面法向轴 ('x', 'y', 'z')
        position: 平面在该轴上的位置（可以是绝对坐标或相对值0-1）
        side: 保留哪一侧 ('positive' 表示 ≥, 'negative' 表示 <)
        angle1: 绕第一个旋转轴的角度（度）
        angle2: 绕第二个旋转轴的角度（度）
        reference_bounds: 参考包围盒，用于转换相对位置（如果为None，使用晶格自身的bounds）
    
    返回:
        裁剪后的晶格
    """
    if lattice.faces.size == 0:
        return lattice
    
    axis_idx = {'x': 0, 'y': 1, 'z': 2}[axis.lower()]
    
    # 如果 position 是相对值（0-1），转换为绝对坐标
    if 0 <= position <= 1:
        # 使用参考包围盒（如果提供），否则使用晶格自身的bounds
        bounds = reference_bounds if reference_bounds is not None else lattice.bounds
        min_val = bounds[0][axis_idx]
        max_val = bounds[1][axis_idx]
        position = min_val + position * (max_val - min_val)
    
    # 如果有角度旋转，使用通用平面裁剪
    if angle1 != 0.0 or angle2 != 0.0:
        return clip_lattice_by_rotated_plane(
            lattice, axis_idx, position, side, angle1, angle2, reference_bounds
        )
    
    # 否则使用快速的轴对齐裁剪
    # 使用更宽松的策略：检查面的顶点而不只是面中心
    vertices = lattice.vertices
    faces = lattice.faces
    
    # 获取每个面的三个顶点在指定轴上的坐标
    face_vertex_coords = vertices[faces][:, :, axis_idx]  # (n_faces, 3)
    
    # 根据平面位置和侧面筛选
    # 使用较大的容差以避免因数值误差导致的晶格缺失
    tolerance = 0.1  # 0.1mm 容差
    if side == 'positive':
        # 保留：至少有一个顶点 >= position（更宽松）
        keep = (face_vertex_coords >= position - tolerance).any(axis=1)
    else:  # negative
        # 保留：至少有一个顶点 < position（更宽松）
        keep = (face_vertex_coords < position + tolerance).any(axis=1)
    
    # 保留符合条件的面
    clipped = lattice.copy()
    clipped.update_faces(keep)
    clipped.remove_unreferenced_vertices()
    
    return clipped


def clip_lattice_by_rotated_plane(
    lattice: trimesh.Trimesh,
    axis_idx: int,
    position: float,
    side: str,
    angle1: float,
    angle2: float,
    reference_bounds: np.ndarray = None
) -> trimesh.Trimesh:
    """
    用旋转后的平面裁剪晶格
    
    参数:
        lattice: 要裁剪的晶格网格
        axis_idx: 主轴索引 (0=X, 1=Y, 2=Z)
        position: 平面在主轴上的位置（绝对坐标）
        side: 保留哪一侧 ('positive' 或 'negative')
        angle1: 绕第一个旋转轴的角度（度）
        angle2: 绕第二个旋转轴的角度（度）
        reference_bounds: 参考包围盒，用于确定平面原点（如果为None，使用晶格自身的bounds）
    
    返回:
        裁剪后的晶格
    """
    if lattice.faces.size == 0:
        return lattice
    
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
    
    # 创建平面原点和法向量
    # 平面原点应该在参考包围盒的中心，只在分割轴上使用指定位置
    # 使用参考包围盒（如果提供），否则使用晶格自身的bounds
    bounds = reference_bounds if reference_bounds is not None else lattice.bounds
    plane_origin = bounds.mean(axis=0)
    plane_origin[axis_idx] = position
    plane_normal = np.zeros(3)
    plane_normal[axis_idx] = 1.0
    
    # 应用旋转
    if angle1 != 0.0:
        angle1_rad = np.radians(angle1)
        rotation_matrix1 = _get_rotation_matrix(rot_axis1, angle1_rad)
        plane_normal = rotation_matrix1 @ plane_normal
    
    if angle2 != 0.0:
        angle2_rad = np.radians(angle2)
        rotation_matrix2 = _get_rotation_matrix(rot_axis2, angle2_rad)
        plane_normal = rotation_matrix2 @ plane_normal
    
    # 归一化法向量
    plane_normal = plane_normal / np.linalg.norm(plane_normal)
    
    # === 平面裁剪策略 ===
    # 支持两种模式：
    # 1. 平衡模式（默认）：面中心 + 至少2个顶点在保留侧
    # 2. 严格模式：所有顶点都在保留侧
    
    # 从配置中获取裁剪模式（如果有的话）
    try:
        from config import PLANE_CLIP_STRICT_MODE
        strict_mode = PLANE_CLIP_STRICT_MODE
    except (ImportError, AttributeError):
        strict_mode = False  # 默认使用平衡模式
    
    # 1. 计算所有顶点到平面的有向距离
    vertices = lattice.vertices
    vertex_distances = np.dot(vertices - plane_origin, plane_normal)
    
    # 2. 获取每个三角面的三个顶点的距离
    face_vertex_distances = vertex_distances[lattice.faces]  # (n_faces, 3)
    
    # 3. 根据模式选择裁剪策略
    # 使用较大的容差以避免因数值误差导致的晶格缺失
    tolerance = 0.1  # 0.1mm 容差，避免边界附近的晶格被错误裁剪
    
    if strict_mode:
        # === 严格模式：只保留所有顶点都在保留侧的面 ===
        if side == 'positive':
            keep_faces = (face_vertex_distances >= -tolerance).all(axis=1)
        else:
            keep_faces = (face_vertex_distances < tolerance).all(axis=1)
    else:
        # === 平衡模式：面中心在保留侧 OR 至少2个顶点在保留侧 ===
        if side == 'positive':
            # 计算每个面有多少个顶点在正侧
            vertices_in_side = (face_vertex_distances >= -tolerance).sum(axis=1)
            # 计算面中心到平面的距离
            face_centers = lattice.triangles_center
            center_distances = np.dot(face_centers - plane_origin, plane_normal)
            # 保留条件：面中心在正侧 OR 至少2个顶点在正侧
            keep_faces = (center_distances >= -tolerance) | (vertices_in_side >= 2)
        else:
            # 计算每个面有多少个顶点在负侧
            vertices_in_side = (face_vertex_distances < tolerance).sum(axis=1)
            # 计算面中心到平面的距离
            face_centers = lattice.triangles_center
            center_distances = np.dot(face_centers - plane_origin, plane_normal)
            # 保留条件：面中心在负侧 OR 至少2个顶点在负侧
            keep_faces = (center_distances < tolerance) | (vertices_in_side >= 2)
    
    # 4. 保留符合条件的面
    clipped = lattice.copy()
    clipped.update_faces(keep_faces)
    clipped.remove_unreferenced_vertices()
    
    return clipped


def _get_rotation_matrix(axis_idx: int, angle_rad: float) -> np.ndarray:
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


def clip_lattice_by_planes(
    lattice: trimesh.Trimesh,
    planes: List[Dict[str, Any]],
    reference_bounds: np.ndarray = None
) -> trimesh.Trimesh:
    """
    用多个平面裁剪晶格
    
    参数:
        lattice: 要裁剪的晶格网格
        planes: 平面列表，每个平面包含 axis, position, side, angle1 (可选), angle2 (可选)
        reference_bounds: 参考包围盒，用于转换相对位置（如果为None，使用晶格自身的bounds）
    
    返回:
        裁剪后的晶格
    """
    result = lattice
    
    for plane in planes:
        angle1 = plane.get('angle1', 0.0)
        angle2 = plane.get('angle2', 0.0)
        
        result = clip_lattice_by_plane(
            result,
            axis=plane['axis'],
            position=plane['position'],
            side=plane['side'],
            angle1=angle1,
            angle2=angle2,
            reference_bounds=reference_bounds
        )
    
    return result
