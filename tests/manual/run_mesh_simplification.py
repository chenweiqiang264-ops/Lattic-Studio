"""
测试网格简化功能

验证 vertex clustering 方法是否正常工作
"""

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

import numpy as np
import trimesh
from PyQt5 import QtWidgets
import time


def test_vertex_clustering():
    """测试顶点聚类简化方法"""
    print("=" * 70)
    print("测试网格简化 - Vertex Clustering 方法")
    print("=" * 70)
    
    # 1. 选择测试文件
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication(sys.argv)
    
    file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
        None,
        "选择测试 STL 文件",
        str(project_root / "resources" / "examples" / "unit_cells"),
        "STL Files (*.stl);;All Files (*.*)"
    )
    
    if not file_path:
        print("✗ 未选择文件")
        return
    
    # 2. 加载网格
    print(f"\n[加载] {Path(file_path).name}")
    mesh = trimesh.load(file_path)
    
    original_faces = len(mesh.faces)
    original_vertices = len(mesh.vertices)
    
    print(f"  原始面数: {original_faces:,}")
    print(f"  原始顶点数: {original_vertices:,}")
    
    # 3. 测试不同的简化比例
    target_ratios = [0.5, 0.3, 0.1]  # 50%, 30%, 10%
    
    for ratio in target_ratios:
        target_faces = int(original_faces * ratio)
        
        print(f"\n" + "-" * 70)
        print(f"测试简化到 {ratio*100:.0f}% ({target_faces:,} 面)")
        print("-" * 70)
        
        # 计算聚类大小
        bounds_size = mesh.bounds[1] - mesh.bounds[0]
        avg_size = bounds_size.mean()
        
        reduction_ratio = target_faces / original_faces
        cluster_size = avg_size * 0.01 * (1.0 / reduction_ratio) ** 0.5
        
        print(f"  包围盒尺寸: {bounds_size}")
        print(f"  平均尺寸: {avg_size:.3f}")
        print(f"  计算的聚类大小: {cluster_size:.4f}")
        
        # 执行简化
        try:
            t_start = time.time()
            simplified = mesh.simplify_vertex_clustering(voxel_size=cluster_size)
            t_end = time.time()
            
            result_faces = len(simplified.faces)
            result_vertices = len(simplified.vertices)
            actual_ratio = result_faces / original_faces
            
            print(f"\n  ✓ 简化成功 (耗时: {t_end - t_start:.2f}s)")
            print(f"  简化后面数: {result_faces:,}")
            print(f"  简化后顶点数: {result_vertices:,}")
            print(f"  实际简化比例: {actual_ratio*100:.1f}%")
            print(f"  面数减少: {original_faces - result_faces:,} ({(1-actual_ratio)*100:.1f}%)")
            
            # 检查网格质量
            print(f"\n  网格质量:")
            print(f"    水密性: {simplified.is_watertight}")
            print(f"    体积: {simplified.volume:.2f} mm³")
            print(f"    表面积: {simplified.area:.2f} mm²")
            
            # 如果简化不够，尝试二次简化
            if result_faces > target_faces * 1.5:
                print(f"\n  简化不够（目标 {target_faces:,}，实际 {result_faces:,}）")
                print(f"  尝试二次简化...")
                
                cluster_size_2 = cluster_size * 1.5
                print(f"  新聚类大小: {cluster_size_2:.4f}")
                
                t_start = time.time()
                simplified_2 = simplified.simplify_vertex_clustering(voxel_size=cluster_size_2)
                t_end = time.time()
                
                result_faces_2 = len(simplified_2.faces)
                actual_ratio_2 = result_faces_2 / original_faces
                
                print(f"  ✓ 二次简化完成 (耗时: {t_end - t_start:.2f}s)")
                print(f"  最终面数: {result_faces_2:,}")
                print(f"  最终比例: {actual_ratio_2*100:.1f}%")
            
        except Exception as e:
            print(f"  ✗ 简化失败: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 70)
    print("测试完成")
    print("=" * 70)


def test_simplification_methods():
    """测试所有可用的简化方法"""
    print("=" * 70)
    print("测试所有简化方法")
    print("=" * 70)
    
    # 1. 选择测试文件
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication(sys.argv)
    
    file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
        None,
        "选择测试 STL 文件",
        str(project_root / "resources" / "examples" / "unit_cells"),
        "STL Files (*.stl);;All Files (*.*)"
    )
    
    if not file_path:
        print("✗ 未选择文件")
        return
    
    # 2. 加载网格
    print(f"\n[加载] {Path(file_path).name}")
    mesh = trimesh.load(file_path)
    
    original_faces = len(mesh.faces)
    target_faces = original_faces // 2  # 简化到一半
    
    print(f"  原始面数: {original_faces:,}")
    print(f"  目标面数: {target_faces:,}")
    
    # 3. 测试各种方法
    methods = []
    
    # 方法1: Quadric Decimation
    print(f"\n" + "-" * 70)
    print("方法1: Quadric Decimation (需要 fast_simplification)")
    print("-" * 70)
    
    if hasattr(mesh, 'simplify_quadric_decimation'):
        try:
            t_start = time.time()
            simplified = mesh.simplify_quadric_decimation(target_faces // 2)
            t_end = time.time()
            
            print(f"  ✓ 可用")
            print(f"  简化后面数: {len(simplified.faces):,}")
            print(f"  耗时: {t_end - t_start:.2f}s")
            methods.append('quadric_decimation')
        except Exception as e:
            print(f"  ✗ 失败: {e}")
    else:
        print(f"  ✗ 不可用（需要安装 fast_simplification）")
        print(f"  提示: pip install fast-simplification")
    
    # 方法2: Vertex Clustering
    print(f"\n" + "-" * 70)
    print("方法2: Vertex Clustering (Trimesh 内置)")
    print("-" * 70)
    
    try:
        bounds_size = mesh.bounds[1] - mesh.bounds[0]
        avg_size = bounds_size.mean()
        reduction_ratio = target_faces / original_faces
        cluster_size = avg_size * 0.01 * (1.0 / reduction_ratio) ** 0.5
        
        t_start = time.time()
        simplified = mesh.simplify_vertex_clustering(voxel_size=cluster_size)
        t_end = time.time()
        
        print(f"  ✓ 可用")
        print(f"  简化后面数: {len(simplified.faces):,}")
        print(f"  耗时: {t_end - t_start:.2f}s")
        methods.append('vertex_clustering')
    except Exception as e:
        print(f"  ✗ 失败: {e}")
    
    # 方法3: Quadratic Decimation
    print(f"\n" + "-" * 70)
    print("方法3: Quadratic Decimation (某些 Trimesh 版本)")
    print("-" * 70)
    
    if hasattr(mesh, 'simplify_quadratic_decimation'):
        try:
            t_start = time.time()
            simplified = mesh.simplify_quadratic_decimation(target_faces)
            t_end = time.time()
            
            print(f"  ✓ 可用")
            print(f"  简化后面数: {len(simplified.faces):,}")
            print(f"  耗时: {t_end - t_start:.2f}s")
            methods.append('quadratic_decimation')
        except Exception as e:
            print(f"  ✗ 失败: {e}")
    else:
        print(f"  ✗ 不可用")
    
    # 总结
    print(f"\n" + "=" * 70)
    print("总结")
    print("=" * 70)
    
    if methods:
        print(f"\n可用的简化方法:")
        for i, method in enumerate(methods, 1):
            print(f"  {i}. {method}")
        
        if 'vertex_clustering' in methods:
            print(f"\n✓ Vertex Clustering 可用！")
            print(f"  这是 Trimesh 内置方法，不需要额外依赖")
            print(f"  可以作为 fallback 方法使用")
    else:
        print(f"\n✗ 没有可用的简化方法")
        print(f"  建议安装: pip install fast-simplification")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "all":
        test_simplification_methods()
    else:
        test_vertex_clustering()
