"""
工程管理模块
支持创建、保存、加载工程文件
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional
from datetime import datetime


class Project:
    """工程类 - 保存所有参数和设置"""
    
    def __init__(self, name: str = "未命名工程"):
        self.name = name
        self.created_time = datetime.now().isoformat()
        self.modified_time = datetime.now().isoformat()
        self.file_path: Optional[Path] = None
        
        # 文件路径
        self.sole_path = ""
        self.lattice_path = ""
        self.output_path = ""
        
        # 晶格方法和参数
        self.method = "gyroid"
        self.auto_recommend = True
        
        # Gyroid 参数
        self.gyroid_cell = 8.0
        self.gyroid_iso = 0.0
        self.gyroid_res = 32
        
        # Voronoi 参数
        self.voronoi_cell = 15.0
        self.voronoi_thickness = 2.0
        self.voronoi_layers = 5
        
        # Voronoi Implicit 参数
        self.voronoi_implicit_cell = 15.0
        self.voronoi_implicit_wall = 1.5
        self.voronoi_implicit_res = 20
        self.voronoi_implicit_density = 1.0
        
        # Tile 参数
        self.tile_shrink = 0.85
        self.tile_spacing = 1.0
        self.tile_margin = 2.0
        
        # 全局选项
        self.enable_final_trim = True
        
        # 视图设置
        self.view_mode = "sole"
        self.mouse_mode = "rotate"
        self.show_mesh_edges = False
        self.current_sole_material = 0
        self.current_lattice_material = 0
        self.current_bg_color = 0
        
        # 分割设置
        self.slicing_enabled = False
        self.slice_axis = 1  # Y轴
        self.slice_position = 0.5
        self.slice_angle1 = 0.0
        self.slice_angle2 = 0.0
        self.slice_subdivide = 0
        self.slice_display_mode = 0  # 原始
        self.slice_spacing = 0.0
        self.show_slice_plane = False
        self.slice_lighting = True
        
        # 生成结果（保存到工程目录的相对路径）
        self.generated_lattice_result_path = ""  # 生成的晶格结果STL文件
        self.generated_sole_result_path = ""  # 生成的鞋底结果STL文件
        self.generated_combined_result_path = ""  # 生成的合并结果STL文件
        self.generation_type = ""  # 生成类型：global, region, plane_region
        
        # 分割结果（保存到工程目录的相对路径）
        self.sliced_positive_mesh_path = ""  # 分割后的正侧网格
        self.sliced_negative_mesh_path = ""  # 分割后的负侧网格
        
        # 区域设计（Tab 3 - 基于分割的区域设计）
        self.regions = []  # 区域列表，每个区域是一个字典
        
        # 平面区域设计（Tab 4 - 基于平面的区域设计）
        self.plane_regions = []  # 平面区域列表，每个区域是一个字典
        
        # 区域晶格生成结果（保存每个区域的晶格STL文件路径）
        self.region_lattice_results = {}  # {region_id: stl_file_path}
        self.plane_region_lattice_results = {}  # {region_id: stl_file_path}
    
    def to_dict(self) -> dict[str, Any]:
        """转换为字典"""
        return {
            "name": self.name,
            "created_time": self.created_time,
            "modified_time": self.modified_time,
            "sole_path": self.sole_path,
            "lattice_path": self.lattice_path,
            "output_path": self.output_path,
            "method": self.method,
            "auto_recommend": self.auto_recommend,
            "gyroid_cell": self.gyroid_cell,
            "gyroid_iso": self.gyroid_iso,
            "gyroid_res": self.gyroid_res,
            "voronoi_cell": self.voronoi_cell,
            "voronoi_thickness": self.voronoi_thickness,
            "voronoi_layers": self.voronoi_layers,
            "voronoi_implicit_cell": self.voronoi_implicit_cell,
            "voronoi_implicit_wall": self.voronoi_implicit_wall,
            "voronoi_implicit_res": self.voronoi_implicit_res,
            "voronoi_implicit_density": self.voronoi_implicit_density,
            "tile_shrink": self.tile_shrink,
            "tile_spacing": self.tile_spacing,
            "tile_margin": self.tile_margin,
            "enable_final_trim": self.enable_final_trim,
            "view_mode": self.view_mode,
            "mouse_mode": self.mouse_mode,
            "show_mesh_edges": self.show_mesh_edges,
            "current_sole_material": self.current_sole_material,
            "current_lattice_material": self.current_lattice_material,
            "current_bg_color": self.current_bg_color,
            "slicing_enabled": self.slicing_enabled,
            "slice_axis": self.slice_axis,
            "slice_position": self.slice_position,
            "slice_angle1": self.slice_angle1,
            "slice_angle2": self.slice_angle2,
            "slice_subdivide": self.slice_subdivide,
            "slice_display_mode": self.slice_display_mode,
            "slice_spacing": self.slice_spacing,
            "show_slice_plane": self.show_slice_plane,
            "slice_lighting": self.slice_lighting,
            "generated_lattice_result_path": self.generated_lattice_result_path,
            "generated_sole_result_path": self.generated_sole_result_path,
            "generated_combined_result_path": self.generated_combined_result_path,
            "generation_type": self.generation_type,
            "sliced_positive_mesh_path": self.sliced_positive_mesh_path,
            "sliced_negative_mesh_path": self.sliced_negative_mesh_path,
            "regions": self.regions,
            "plane_regions": self.plane_regions,
            "region_lattice_results": self.region_lattice_results,
            "plane_region_lattice_results": self.plane_region_lattice_results,
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Project:
        """从字典创建工程"""
        project = cls(data.get("name", "未命名工程"))
        
        # 基本信息
        project.created_time = data.get("created_time", project.created_time)
        project.modified_time = data.get("modified_time", project.modified_time)
        
        # 文件路径
        project.sole_path = data.get("sole_path", "")
        project.lattice_path = data.get("lattice_path", "")
        project.output_path = data.get("output_path", "")
        
        # 晶格方法和参数
        project.method = data.get("method", "gyroid")
        project.auto_recommend = data.get("auto_recommend", True)
        
        # Gyroid 参数
        project.gyroid_cell = data.get("gyroid_cell", 8.0)
        project.gyroid_iso = data.get("gyroid_iso", 0.0)
        project.gyroid_res = data.get("gyroid_res", 32)
        
        # Voronoi 参数
        project.voronoi_cell = data.get("voronoi_cell", 15.0)
        project.voronoi_thickness = data.get("voronoi_thickness", 2.0)
        project.voronoi_layers = data.get("voronoi_layers", 5)
        
        # Voronoi Implicit 参数
        project.voronoi_implicit_cell = data.get("voronoi_implicit_cell", 15.0)
        project.voronoi_implicit_wall = data.get("voronoi_implicit_wall", 1.5)
        project.voronoi_implicit_res = data.get("voronoi_implicit_res", 20)
        project.voronoi_implicit_density = data.get("voronoi_implicit_density", 1.0)
        
        # Tile 参数
        project.tile_shrink = data.get("tile_shrink", 0.85)
        project.tile_spacing = data.get("tile_spacing", 1.0)
        project.tile_margin = data.get("tile_margin", 2.0)
        
        # 全局选项
        project.enable_final_trim = data.get("enable_final_trim", True)
        
        # 视图设置
        project.view_mode = data.get("view_mode", "sole")
        project.mouse_mode = data.get("mouse_mode", "rotate")
        project.show_mesh_edges = data.get("show_mesh_edges", False)
        project.current_sole_material = data.get("current_sole_material", 0)
        project.current_lattice_material = data.get("current_lattice_material", 0)
        project.current_bg_color = data.get("current_bg_color", 0)
        
        # 分割设置
        project.slicing_enabled = data.get("slicing_enabled", False)
        project.slice_axis = data.get("slice_axis", 1)
        project.slice_position = data.get("slice_position", 0.5)
        project.slice_angle1 = data.get("slice_angle1", 0.0)
        project.slice_angle2 = data.get("slice_angle2", 0.0)
        project.slice_subdivide = data.get("slice_subdivide", 0)
        project.slice_display_mode = data.get("slice_display_mode", 0)
        project.slice_spacing = data.get("slice_spacing", 0.0)
        project.show_slice_plane = data.get("show_slice_plane", False)
        project.slice_lighting = data.get("slice_lighting", True)
        
        # 生成结果
        project.generated_lattice_result_path = data.get("generated_lattice_result_path", "")
        project.generated_sole_result_path = data.get("generated_sole_result_path", "")
        project.generated_combined_result_path = data.get("generated_combined_result_path", "")
        project.generation_type = data.get("generation_type", "")
        
        # 分割结果
        project.sliced_positive_mesh_path = data.get("sliced_positive_mesh_path", "")
        project.sliced_negative_mesh_path = data.get("sliced_negative_mesh_path", "")
        
        # 区域设计
        project.regions = data.get("regions", [])
        project.plane_regions = data.get("plane_regions", [])
        project.region_lattice_results = data.get("region_lattice_results", {})
        project.plane_region_lattice_results = data.get("plane_region_lattice_results", {})
        
        return project
    
    def save(self, file_path: Path) -> None:
        """保存工程到文件
        
        工程文件结构：
        project_name.slp (JSON文件)
        project_name_data/ (数据目录)
            ├── generated_lattice.stl (生成的晶格)
            ├── generated_sole.stl (生成的鞋底)
            ├── generated_combined.stl (生成的合并结果)
            ├── sliced_positive.stl (分割正侧)
            ├── sliced_negative.stl (分割负侧)
            ├── region_<id>_lattice.stl (区域晶格)
            └── plane_region_<id>_lattice.stl (平面区域晶格)
        """
        self.file_path = file_path
        self.modified_time = datetime.now().isoformat()
        
        data = self.to_dict()
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    def get_data_dir(self) -> Optional[Path]:
        """获取工程数据目录路径"""
        if self.file_path is None:
            return None
        return self.file_path.parent / f"{self.file_path.stem}_data"
    
    @classmethod
    def load(cls, file_path: Path) -> Project:
        """从文件加载工程"""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        project = cls.from_dict(data)
        project.file_path = file_path
        return project
    
    def get_display_name(self) -> str:
        """获取显示名称"""
        if self.file_path:
            return self.file_path.stem
        return self.name


class ProjectManager:
    """工程管理器"""
    
    def __init__(self):
        self.current_project: Optional[Project] = None
        self.recent_projects: list[Path] = []
        self.max_recent = 10
        
        # 加载最近工程列表
        self._load_recent_projects()
    
    def _get_recent_file_path(self) -> Path:
        """获取最近工程列表文件路径"""
        return Path.home() / ".kiro" / "recent_projects.json"
    
    def _load_recent_projects(self) -> None:
        """加载最近工程列表"""
        recent_file = self._get_recent_file_path()
        if recent_file.exists():
            try:
                with open(recent_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.recent_projects = [Path(p) for p in data.get("recent", [])]
                    # 过滤不存在的文件
                    self.recent_projects = [p for p in self.recent_projects if p.exists()]
            except Exception as e:
                print(f"加载最近工程列表失败: {e}")
                self.recent_projects = []
    
    def _save_recent_projects(self) -> None:
        """保存最近工程列表"""
        recent_file = self._get_recent_file_path()
        recent_file.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            data = {
                "recent": [str(p) for p in self.recent_projects[:self.max_recent]]
            }
            with open(recent_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"保存最近工程列表失败: {e}")
    
    def add_to_recent(self, file_path: Path) -> None:
        """添加到最近工程列表"""
        file_path = file_path.resolve()
        
        # 如果已存在，先移除
        if file_path in self.recent_projects:
            self.recent_projects.remove(file_path)
        
        # 添加到列表开头
        self.recent_projects.insert(0, file_path)
        
        # 限制列表长度
        self.recent_projects = self.recent_projects[:self.max_recent]
        
        # 保存
        self._save_recent_projects()
    
    def create_new_project(self, name: str = "未命名工程") -> Project:
        """创建新工程"""
        self.current_project = Project(name)
        return self.current_project
    
    def save_project(self, file_path: Path) -> None:
        """保存当前工程"""
        if self.current_project is None:
            raise ValueError("没有当前工程")
        
        self.current_project.save(file_path)
        self.add_to_recent(file_path)
    
    def load_project(self, file_path: Path) -> Project:
        """加载工程"""
        self.current_project = Project.load(file_path)
        self.add_to_recent(file_path)
        return self.current_project
    
    def get_recent_projects(self) -> list[Path]:
        """获取最近工程列表"""
        return self.recent_projects.copy()
