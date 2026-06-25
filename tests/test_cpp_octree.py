"""
测试 C++ 加速的八叉树构建
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import time

print("=" * 70)
print("测试 C++ 加速的八叉树构建")
print("=" * 70)

# 1. 检查 C++ 模块
print("\n[1] 检查 C++ 模块...")
try:
    from core.cpp.cpp_octree_sdf import build_octree_from_periodic_sdf
    print("  ✓ C++ 八叉树模块已加载")
    
    import core.cpp.cpp_octree_sdf as cpp_module
    print(f"  OpenMP 启用: {cpp_module.OPENMP_ENABLED}")
    print(f"  线程数: {cpp_module.NUM_THREADS}")
except ImportError as e:
    print(f"  ✗ C++ 模块未找到: {e}")
    print(f"  请运行: python core/cpp/build_octree_sdf.py build_ext --inplace")
    sys.exit(1)

# 2. 创建测试数据
print("\n[2] 创建测试数据...")

# 创建一个简单的晶胞 SDF（球体）
unit_resolution = 32
unit_sdf = np.zeros((unit_resolution, unit_resolution, unit_resolution), dtype=np.float32)

center = unit_resolution // 2
radius = unit_resolution // 4

for i in range(unit_resolution):
    for j in range(unit_resolution):
        for k in range(unit_resolution):
            dx = i - center
            dy = j - center
            dz = k - center
            dist = np.sqrt(dx**2 + dy**2 + dz**2)
            unit_sdf[i, j, k] = dist - radius

print(f"  单元 SDF 形状: {unit_sdf.shape}")
print(f"  SDF 范围: [{unit_sdf.min():.2f}, {unit_sdf.max():.2f}]")

# 3. 测试 C++ 构建
print("\n[3] 测试 C++ 八叉树构建...")

target_bbox_min = np.array([0.0, 0.0, 0.0], dtype=np.float32)
target_bbox_max = np.array([10.0, 10.0, 10.0], dtype=np.float32)
cell_size = 1.0
unit_bbox_min = np.array([0.0, 0.0, 0.0], dtype=np.float32)

t_start = time.time()

result = build_octree_from_periodic_sdf(
    unit_sdf,
    target_bbox_min,
    target_bbox_max,
    cell_size,
    unit_bbox_min,
    unit_resolution,
    max_depth=6,
    leaf_resolution=4,
    uniform_threshold=0.01,
    verbose=True
)

t_end = time.time()

print(f"\n  构建时间: {t_end - t_start:.2f}s")
print(f"  节点总数: {result['total_nodes']:,}")
print(f"  叶节点数: {result['leaf_nodes']:,}")
print(f"  内存占用: {result['memory_bytes'] / (1024**2):.1f} MB")

# 4. 测试 Python 包装
print("\n[4] 测试 Python 包装...")

try:
    from core.octree_sdf import build_octree_from_periodic_sdf_cpp
    
    t_start = time.time()
    
    octree = build_octree_from_periodic_sdf_cpp(
        unit_sdf,
        target_bbox_min,
        target_bbox_max,
        cell_size,
        unit_bbox_min,
        unit_resolution,
        max_depth=6,
        leaf_resolution=4,
        uniform_threshold=0.01,
        verbose=True
    )
    
    t_end = time.time()
    
    print(f"\n  ✓ Python 包装成功")
    print(f"  构建时间: {t_end - t_start:.2f}s")
    print(f"  节点总数: {octree.num_nodes:,}")
    print(f"  叶节点数: {octree.num_leaf_nodes:,}")
    
    # 测试查询
    print(f"\n[5] 测试查询...")
    test_points = [
        [5.0, 5.0, 5.0],
        [0.5, 0.5, 0.5],
        [9.5, 9.5, 9.5],
    ]
    
    for pos in test_points:
        sdf_value = octree.query(*pos)
        print(f"  位置 {pos}: SDF = {sdf_value:.4f}")
    
    print(f"\n✓ 所有测试通过！")
    
except Exception as e:
    print(f"  ✗ 测试失败: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 70)
print("测试完成")
print("=" * 70)
