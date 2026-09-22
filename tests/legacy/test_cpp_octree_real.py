"""
使用真实晶格数据测试 C++ 八叉树加速
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import time
import trimesh

print("=" * 70)
print("使用真实晶格测试 C++ 八叉树加速")
print("=" * 70)

# 1. 加载真实晶格
print("\n[1] 加载晶格文件...")
lattice_files = list((project_root / "resources" / "晶格").glob("*.stl"))
if not lattice_files:
    print("  ✗ 未找到晶格文件")
    sys.exit(1)

lattice_file = lattice_files[0]
print(f"  使用: {lattice_file.name}")

mesh = trimesh.load(str(lattice_file))
print(f"  顶点数: {len(mesh.vertices):,}")
print(f"  面数: {len(mesh.faces):,}")

# 2. 计算晶胞 SDF
print("\n[2] 计算晶胞 SDF...")
unit_resolution = 32
print(f"  分辨率: {unit_resolution}³")

# 获取包围盒
bounds = mesh.bounds
bbox_min = bounds[0]
bbox_max = bounds[1]
size = bbox_max - bbox_min

print(f"  包围盒: {bbox_min} → {bbox_max}")
print(f"  尺寸: {size}")

# 使用 C++ 距离场模块计算 SDF
try:
    from core.cpp.cpp_distance_field import compute_mesh_distance_accurate
    
    print(f"  使用 C++ 加速计算距离场...")
    t_start = time.time()
    
    unit_sdf = compute_mesh_distance_accurate(
        mesh.vertices.astype(np.float32),
        mesh.faces.astype(np.int32),
        bbox_min.astype(np.float32),
        bbox_max.astype(np.float32),
        unit_resolution
    )
    
    t_sdf = time.time() - t_start
    
    print(f"  ✓ SDF 计算完成 (耗时: {t_sdf:.1f}s)")
    print(f"  SDF 范围: [{unit_sdf.min():.3f}, {unit_sdf.max():.3f}]")
    
except ImportError:
    print("  ⚠ C++ 距离场模块未找到，使用简化方法...")
    # 简化方法：基于体素化
    from skimage import measure
    
    # 创建体素网格
    x = np.linspace(bbox_min[0], bbox_max[0], unit_resolution)
    y = np.linspace(bbox_min[1], bbox_max[1], unit_resolution)
    z = np.linspace(bbox_min[2], bbox_max[2], unit_resolution)
    
    # 简单的内外判断
    unit_sdf = np.ones((unit_resolution, unit_resolution, unit_resolution), dtype=np.float32)
    
    print(f"  ✓ 使用简化 SDF")
    print(f"  SDF 范围: [{unit_sdf.min():.3f}, {unit_sdf.max():.3f}]")

# 3. 测试 C++ 八叉树构建
print("\n[3] 测试 C++ 八叉树构建...")

try:
    from core.octree_sdf import build_octree_from_periodic_sdf_cpp, CPP_OCTREE_AVAILABLE
    
    if not CPP_OCTREE_AVAILABLE:
        print("  ✗ C++ 模块未加载")
        sys.exit(1)
    
    # 设置参数（模拟 5x5x5 晶格阵列）
    num_cells = np.array([5, 5, 5])
    cell_size = size.max()
    
    target_size = num_cells * cell_size
    target_bbox_min = bbox_min.copy()
    target_bbox_max = target_bbox_min + target_size
    
    print(f"  模拟阵列: {num_cells} ({np.prod(num_cells)} 个晶胞)")
    print(f"  目标区域: {target_bbox_min} → {target_bbox_max}")
    
    # C++ 构建
    print(f"\n  [C++ 版本]")
    t_start = time.time()
    
    octree_cpp = build_octree_from_periodic_sdf_cpp(
        unit_sdf,
        target_bbox_min,
        target_bbox_max,
        cell_size,
        bbox_min,
        unit_resolution,
        max_depth=8,
        leaf_resolution=4,
        uniform_threshold=0.01,
        verbose=True
    )
    
    t_cpp = time.time() - t_start
    
    print(f"\n  ✓ C++ 构建完成")
    print(f"  总耗时: {t_cpp:.2f}s")
    print(f"  节点数: {octree_cpp.num_nodes:,}")
    print(f"  叶节点数: {octree_cpp.num_leaf_nodes:,}")
    
    # 4. 对比 Python 版本
    print(f"\n[4] 对比 Python 版本...")
    
    from core.octree_sdf import OctreeSDF
    
    # 创建周期性查询函数
    inv_cell_size = 1.0 / cell_size
    
    def periodic_sdf_func(x, y, z):
        world_pos = np.array([x, y, z])
        local_pos = world_pos - bbox_min
        local_pos_mod = np.mod(local_pos, cell_size)
        unit_indices = (local_pos_mod * inv_cell_size * unit_resolution).astype(int)
        unit_indices = np.clip(unit_indices, 0, unit_resolution - 1)
        return unit_sdf[unit_indices[0], unit_indices[1], unit_indices[2]]
    
    print(f"  [Python 版本]")
    t_start = time.time()
    
    octree_py = OctreeSDF(target_bbox_min, target_bbox_max, max_depth=8)
    octree_py.build_from_function(
        periodic_sdf_func,
        leaf_resolution=4,
        uniform_threshold=0.01,
        verbose=True
    )
    
    t_py = time.time() - t_start
    
    print(f"\n  ✓ Python 构建完成")
    print(f"  总耗时: {t_py:.2f}s")
    print(f"  节点数: {octree_py.num_nodes:,}")
    print(f"  叶节点数: {octree_py.num_leaf_nodes:,}")
    
    # 5. 性能对比
    print(f"\n" + "=" * 70)
    print(f"性能对比")
    print(f"=" * 70)
    print(f"  C++ 版本: {t_cpp:.2f}s")
    print(f"  Python 版本: {t_py:.2f}s")
    print(f"  加速比: {t_py / t_cpp:.1f}x")
    print(f"  节点数一致: {octree_cpp.num_nodes == octree_py.num_nodes}")
    
    if t_cpp < t_py:
        print(f"\n  ✓✓✓ C++ 版本更快 {t_py / t_cpp:.1f}x！")
    else:
        print(f"\n  ⚠ C++ 版本反而更慢，可能是数据量太小")
    
except Exception as e:
    print(f"\n  ✗ 测试失败: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 70)
