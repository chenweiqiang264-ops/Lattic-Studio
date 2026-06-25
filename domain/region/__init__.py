"""
Region management module - 区域管理模块
"""

from domain.region.region_manager import Region, RegionManager
from domain.region.plane_region_manager import (
    PlaneRegion,
    PlaneRegionManager,
    clip_lattice_by_plane,
    clip_lattice_by_planes,
)

__all__ = [
    'Region',
    'RegionManager',
    'PlaneRegion',
    'PlaneRegionManager',
    'clip_lattice_by_plane',
    'clip_lattice_by_planes',
]
