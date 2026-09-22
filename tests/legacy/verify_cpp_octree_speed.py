"""
验证 C++ 八叉树加速效果
使用合成的复杂 SDF 数据
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import time

print("=" * 70)
print("验证 C++ 八叉树加速效果")
print("=" * 70)

# 1. 创建复杂的 SDF 数据（Gyroid 晶格）
print("\n[1] 创建复杂 SDF 数据（Gyroid 晶格）...")
unit_resolution = 48
print(f"  分辨率: {unit_resolution}³")

# 创建 Gyroid SDF
x = np.linspace(0, 2*np.pi, unit_resolution)
y = np.linspace(0, 2*np.pi, unit_resolution)
z = np.linspace(0, 2*np.pi, unit_resolution)

xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
unit_sdf = (np.sin(xx) * np.cos(yy) + 
            np.sin(yy) * np.cos(zz) + 
            np.sin(zz) * np.cos(xx)).astype(np.float32)

print(f"  SDF 范围: [{unit_sdf.min():.3f}, {unit_sdf.max():.3f}]")
print(f"  SDF 标准差: {unit_sdf.std():.3f} (复杂度指标)")

# 2. 设置参数
print("\n[2] 设置参数...")
num_cells = np.array([8, 8, 8])  # 8x8x8 = 512 个晶胞
cell_size = 10.0
bbox_min = np.array([0.0, 0.0, 0.0], dtype=np.float32)

target_size = num_cells * cell_size
target_bbox_min = bbox_min
target_bbox_max = target_bbox_min + target_size

print(f"  阵列大小: {num_cells} ({np.prod(num_cells)} 个晶胞)")
print(f"  目标分辨率: {num_cells * unit_resolution}")
print(f"  目标区域: {target_bbox_min} → {target_bbox_max}")

# 3. 测试 C++ 版本
print("\n[3] 测试 C++ 八叉树构建...")

try:
    from core.octree_sdf import build_octree_from_periodic_sdf_cpp, CPP_OCTREE_AVAILABLE
    
    if not CPP_OCTREE_AVAILABLE:
        print("  ✗ C++ 模块未加载")
        sys.exit(1)
    
    print("  [C++ 版本]")
    t_start = time.time()
    
    octree_cpp = build_octree_from_periodic_sdf_cpp(
        unit_sdf,
        target_bbox_min,
        target_bbox_max,
        cell_size,
        bbox_min,
        unit_resolution,
        max_depth=9,
        leaf_resolution=4,
        uniform_threshold=0.05,  # 稍微放宽以加快速度
        verbose=True
    )
    
    t_cpp = time.time() - t_start
    
    print(f"\n  ✓ C++ 构建完成")
    print(f"  总耗时: {t_cpp:.2f}s")
    print(f"  节点数: {octree_cpp.num_nodes:,}")
    print(f"  叶节点数: {octree_cpp.num_leaf_nodes:,}")
    
    # 4. 测试 Python 版本
    print(f"\n[4] 测试 Python 八叉树构建...")
    
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
    
    print("  [Python 版本]")
    t_start = time.time()
    
    octree_py = OctreeSDF(target_bbox_min, target_bbox_max, max_depth=9)
    octree_py.build_from_function(
        periodic_sdf_func,
        leaf_resolution=4,
        uniform_threshold=0.05,
        verbose=True
    )
    
    t_py = time.time() - t_start
    
    print(f"\n  ✓ Python 构建完成")
    print(f"  总耗时: {t_py:.2f}s")
    print(f"  节点数: {octree_py.num_nodes:,}")
    print(f"  叶节点数: {octree_py.num_leaf_nodes:,}")
    
    # 5. 性能对比
    print(f"\n" + "=" * 70)
    print(f"性能对比总结")
    print(f"=" * 70)
    print(f"  C++ 版本:     {t_cpp:6.2f}s  ({octree_cpp.num_nodes:,} 节点)")
    print(f"  Python 版本:  {t_py:6.2f}s  ({octree_py.num_nodes:,} 节点)")
    print(f"  加速比:       {t_py / t_cpp:6.1f}x")
    print(f"  节点数一致:   {'✓' if octree_cpp.num_nodes == octree_py.num_nodes else '✗'}")
    
    if t_cpp < t_py:
        speedup = t_py / t_cpp
        print(f"\n  ✓✓✓ C++ 版本成功加速 {speedup:.1f}x！")
        print(f"  节省时间: {t_py - t_cpp:.2f}s")
    else:
        print(f"\n  ⚠ C++ 版本未能加速")
    
    # 6. 测试查询性能
    print(f"\n[5] 测试查询性能...")
    
    test_points = np.random.rand(1000, 3) * target_size + target_bbox_min
    
    # C++ 查询
    t_start = time.time()
    for pt in test_points:
        octree_cpp.query(*pt)
    t_query_cpp = time.time() - t_start
    
    # Python 查询
    t_start = time.time()
    for pt in test_points:
        octree_py.query(*pt)
    t_query_py = time.time() - t_start
    
    print(f"  查询 1000 个点:")
    print(f"    C++ 版本:    {t_query_cpp:.3f}s")
    print(f"    Python 版本: {t_query_py:.3f}s")
    print(f"    查询加速比:  {t_query_py / t_query_cpp:.1f}x")
    
except Exception as e:
    print(f"\n  ✗ 测试失败: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 70)
print("测试完成")
print("=" * 70)
