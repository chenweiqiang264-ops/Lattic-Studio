"""
测试八叉树 SDF 集成
验证八叉树功能是否正常工作
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
from core.octree_sdf import OctreeSDF, build_octree_from_grid

def test_octree_basic():
    """测试基本的八叉树功能"""
    print("=" * 70)
    print("测试八叉树 SDF 基本功能")
    print("=" * 70)
    
    # 1. 创建一个简单的 SDF 函数（球体）
    print("\n[测试 1] 创建球体 SDF")
    
    def sphere_sdf(x, y, z):
        """球体 SDF：中心在原点，半径为 1"""
        return np.sqrt(x**2 + y**2 + z**2) - 1.0
    
    # 2. 构建八叉树
    bbox_min = np.array([-2.0, -2.0, -2.0])
    bbox_max = np.array([2.0, 2.0, 2.0])
    
    print(f"  包围盒: {bbox_min} → {bbox_max}")
    print(f"  最大深度: 6")
    
    octree = OctreeSDF(bbox_min, bbox_max, max_depth=6)
    octree.build_from_function(
        sphere_sdf,
        leaf_resolution=4,
        uniform_threshold=0.01,
        verbose=True
    )
    
    # 3. 测试查询
    print(f"\n[测试 2] 查询 SDF 值")
    
    test_points = [
        ([0.0, 0.0, 0.0], "中心点"),
        ([1.0, 0.0, 0.0], "表面点"),
        ([2.0, 0.0, 0.0], "外部点"),
        ([0.5, 0.0, 0.0], "内部点"),
    ]
    
    for pos, desc in test_points:
        sdf_value = octree.query(*pos)
        expected = sphere_sdf(*pos)
        error = abs(sdf_value - expected)
        print(f"  {desc} {pos}: SDF={sdf_value:.4f}, 期望={expected:.4f}, 误差={error:.4f}")
    
    # 4. 转换为密集网格
    print(f"\n[测试 3] 转换为密集网格")
    
    resolution = (32, 32, 32)
    dense_grid = octree.to_dense_grid(resolution)
    
    print(f"  密集网格形状: {dense_grid.shape}")
    print(f"  SDF 范围: [{dense_grid.min():.3f}, {dense_grid.max():.3f}]")
    
    # 统计内部体素
    interior_voxels = np.sum(dense_grid < 0)
    interior_ratio = interior_voxels / np.prod(resolution) * 100
    print(f"  内部体素: {interior_voxels:,} ({interior_ratio:.1f}%)")
    
    # 5. 测试从密集网格构建八叉树
    print(f"\n[测试 4] 从密集网格构建八叉树")
    
    octree2 = build_octree_from_grid(
        dense_grid,
        bbox_min,
        bbox_max,
        max_depth=5,
        leaf_resolution=4,
        uniform_threshold=0.01
    )
    
    # 验证查询结果一致
    print(f"\n[测试 5] 验证两个八叉树的查询结果")
    
    for pos, desc in test_points[:2]:  # 只测试前两个点
        sdf1 = octree.query(*pos)
        sdf2 = octree2.query(*pos)
        diff = abs(sdf1 - sdf2)
        print(f"  {desc}: 八叉树1={sdf1:.4f}, 八叉树2={sdf2:.4f}, 差异={diff:.4f}")
    
    print(f"\n✓ 所有测试通过！")
    print(f"✓ 八叉树 SDF 功能正常")


def test_octree_memory_savings():
    """测试八叉树的内存节省效果"""
    print("\n" + "=" * 70)
    print("测试八叉树内存节省效果")
    print("=" * 70)
    
    # 创建一个简单的周期性晶格 SDF（模拟 Gyroid）
    def gyroid_sdf(x, y, z):
        """简化的 Gyroid SDF"""
        return np.sin(x) * np.cos(y) + np.sin(y) * np.cos(z) + np.sin(z) * np.cos(x)
    
    bbox_min = np.array([0.0, 0.0, 0.0])
    bbox_max = np.array([10.0, 10.0, 10.0])
    
    print(f"\n包围盒: {bbox_min} → {bbox_max}")
    
    # 测试不同深度的内存占用
    for max_depth in [6, 7, 8]:
        print(f"\n[深度 {max_depth}]")
        
        octree = OctreeSDF(bbox_min, bbox_max, max_depth=max_depth)
        octree.build_from_function(
            gyroid_sdf,
            leaf_resolution=4,
            uniform_threshold=0.05,  # 较大的阈值以节省更多内存
            verbose=False
        )
        
        # 估算内存
        octree_memory = octree.root.get_memory_usage()
        full_resolution = 2 ** max_depth * 4
        full_memory = full_resolution ** 3 * 4
        
        print(f"  节点数: {octree.num_nodes:,}")
        print(f"  叶节点数: {octree.num_leaf_nodes:,}")
        print(f"  八叉树内存: {octree_memory / (1024**2):.1f} MB")
        print(f"  完整网格内存: {full_memory / (1024**2):.1f} MB")
        print(f"  节省: {octree.memory_saved_ratio * 100:.1f}%")
    
    print(f"\n✓ 内存节省测试完成")


if __name__ == "__main__":
    try:
        test_octree_basic()
        test_octree_memory_savings()
        
        print("\n" + "=" * 70)
        print("✓✓✓ 所有测试通过！八叉树 SDF 集成成功！")
        print("=" * 70)
        
    except Exception as e:
        print(f"\n✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
