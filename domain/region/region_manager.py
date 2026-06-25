"""
区域管理模块
提供区域数据模型和管理功能
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import List, Optional, Dict, Any

import numpy as np
import trimesh


@dataclass
class Region:
    """区域数据模型"""
    id: str                    # 唯一标识符 (UUID)
    name: str                  # 区域名称（用户可编辑）
    source_type: str           # 来源类型: "plane_bounded", "whole_sole"
    
    # 平面边界定义（用于 plane_bounded 类型）
    boundary_planes: List[Dict[str, Any]] = field(default_factory=list)
    # 每个平面包含: axis, position, side, angle1 (可选), angle2 (可选)
    
    # 晶格参数
    lattice_method: str = "gyroid"  # "gyroid", "voronoi_2_5d", "tile_unit"
    
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
            'source_type': self.source_type,
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
    def from_dict(data: Dict[str, Any]) -> Region:
        """从字典创建 Region 对象"""
        return Region(
            id=data['id'],
            name=data['name'],
            source_type=data['source_type'],
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
        """获取显示名称（包含来源类型）"""
        if self.source_type == 'whole_sole':
            return f"{self.name} (整体鞋底)"
        elif self.source_type == 'plane_bounded':
            if len(self.boundary_planes) == 0:
                return f"{self.name} (无边界)"
            elif len(self.boundary_planes) == 1:
                plane = self.boundary_planes[0]
                axis = plane['axis'].upper()
                pos = plane['position']
                side = '正侧' if plane['side'] == 'positive' else '负侧'
                angle1 = plane.get('angle1', 0.0)
                angle2 = plane.get('angle2', 0.0)
                
                if angle1 != 0.0 or angle2 != 0.0:
                    return f"{self.name} ({axis}={pos:.1f}mm {side}, ∠{angle1:.0f}°/{angle2:.0f}°)"
                else:
                    return f"{self.name} ({axis}={pos:.1f}mm {side})"
            else:
                return f"{self.name} ({len(self.boundary_planes)}个边界)"
        else:
            return f"{self.name} ({self.source_type})"


class RegionManager:
    """区域管理器，负责区域的增删改查"""
    
    def __init__(self):
        self.regions: List[Region] = []
        self.region_counter: int = 0
        # 存储区域对应的网格（不序列化到 JSON）
        self._region_meshes: Dict[str, trimesh.Trimesh] = {}
        # 存储生成的晶格（不序列化到 JSON）
        self._region_lattices: Dict[str, trimesh.Trimesh] = {}
    
    def add_region(
        self, 
        source_type: str, 
        mesh: trimesh.Trimesh,
        boundary_planes: List[Dict[str, Any]] = None,
        lattice_method: str = "gyroid"
    ) -> Region:
        """
        添加新区域
        
        参数:
            source_type: 来源类型 ("whole_sole", "plane_bounded")
            mesh: 区域网格（完整鞋底）
            boundary_planes: 平面边界列表（可选）
            lattice_method: 默认晶格方法
        
        返回:
            新创建的 Region 对象
        """
        self.region_counter += 1
        region_id = str(uuid.uuid4())
        region_name = f"区域 {self.region_counter}"
        
        if boundary_planes is None:
            boundary_planes = []
        
        region = Region(
            id=region_id,
            name=region_name,
            source_type=source_type,
            boundary_planes=boundary_planes,
            lattice_method=lattice_method,
        )
        
        self.regions.append(region)
        self._region_meshes[region_id] = mesh.copy()
        
        return region
    
    def remove_region(self, region_id: str) -> bool:
        """
        删除区域
        
        参数:
            region_id: 区域 ID
        
        返回:
            是否删除成功
        """
        region = self.get_region(region_id)
        if region is None:
            return False
        
        self.regions.remove(region)
        
        # 清理关联数据
        if region_id in self._region_meshes:
            del self._region_meshes[region_id]
        if region_id in self._region_lattices:
            del self._region_lattices[region_id]
        
        return True
    
    def get_region(self, region_id: str) -> Optional[Region]:
        """
        获取区域
        
        参数:
            region_id: 区域 ID
        
        返回:
            Region 对象，如果不存在则返回 None
        """
        for region in self.regions:
            if region.id == region_id:
                return region
        return None
    
    def get_region_mesh(self, region_id: str) -> Optional[trimesh.Trimesh]:
        """
        获取区域网格
        
        参数:
            region_id: 区域 ID
        
        返回:
            区域网格，如果不存在则返回 None
        """
        return self._region_meshes.get(region_id)
    
    def get_region_lattice(self, region_id: str) -> Optional[trimesh.Trimesh]:
        """
        获取区域生成的晶格
        
        参数:
            region_id: 区域 ID
        
        返回:
            区域晶格，如果未生成则返回 None
        """
        return self._region_lattices.get(region_id)
    
    def set_region_lattice(self, region_id: str, lattice: trimesh.Trimesh) -> None:
        """
        设置区域生成的晶格
        
        参数:
            region_id: 区域 ID
            lattice: 晶格网格
        """
        self._region_lattices[region_id] = lattice
    
    def update_region_name(self, region_id: str, name: str) -> bool:
        """
        更新区域名称
        
        参数:
            region_id: 区域 ID
            name: 新名称
        
        返回:
            是否更新成功
        """
        region = self.get_region(region_id)
        if region is None:
            return False
        
        region.name = name
        return True
    
    def update_region_parameters(self, region_id: str, **params) -> bool:
        """
        更新区域参数
        
        参数:
            region_id: 区域 ID
            **params: 参数键值对
        
        返回:
            是否更新成功
        """
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
    
    def get_all_regions(self) -> List[Region]:
        """获取所有区域"""
        return self.regions.copy()
    
    def clear_all_lattices(self) -> None:
        """清除所有生成的晶格"""
        self._region_lattices.clear()
    
    def has_generated_lattices(self) -> bool:
        """检查是否有已生成的晶格"""
        return len(self._region_lattices) > 0
    
    def get_combined_lattice(self) -> Optional[trimesh.Trimesh]:
        """
        获取合并后的晶格
        
        返回:
            合并的晶格网格，如果没有生成的晶格则返回 None
        """
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
        """
        保存配置到 JSON 文件
        
        参数:
            filepath: 文件路径
        """
        config = {
            'version': '1.0',
            'region_counter': self.region_counter,
            'regions': [region.to_dict() for region in self.regions]
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
    
    def load_from_json(
        self, 
        filepath: str,
        mesh_provider: callable
    ) -> None:
        """
        从 JSON 文件加载配置
        
        参数:
            filepath: 文件路径
            mesh_provider: 网格提供函数，接受 source_type 返回对应的网格
                          例如: lambda source_type: get_mesh_by_source(source_type)
        
        异常:
            ValueError: 配置文件格式错误
            FileNotFoundError: 文件不存在
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        # 验证版本
        version = config.get('version', '1.0')
        if version != '1.0':
            raise ValueError(f"不支持的配置文件版本: {version}")
        
        # 清空当前数据
        self.regions.clear()
        self._region_meshes.clear()
        self._region_lattices.clear()
        
        # 加载区域计数器
        self.region_counter = config.get('region_counter', 0)
        
        # 加载区域
        regions_data = config.get('regions', [])
        for region_data in regions_data:
            region = Region.from_dict(region_data)
            self.regions.append(region)
            
            # 通过 mesh_provider 获取网格
            try:
                mesh = mesh_provider(region.source_type)
                if mesh is not None:
                    self._region_meshes[region.id] = mesh
            except Exception as e:
                print(f"警告: 无法加载区域 '{region.name}' 的网格: {e}")
