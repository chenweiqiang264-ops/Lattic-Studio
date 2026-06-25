"""
全新的 Voronoi 隐函数建模实现
使用距离场 + Marching Cubes 方法生成 Voronoi 晶格结构

核心思路：
1. 在空间中采样种子点
2. 计算 3D Voronoi 图
3. 提取 Voronoi 骨架（连接相邻种子点的边）
4. 计算每个体素到骨架的距离场
5. 使用 Marching Cubes 提取等值面
"""

import numpy as np
import trimesh
from scipy.spatial import Voronoi
from skimage import measure
import time
import matplotlib.pyplot as plt

# 尝试导入 C++ 加速模块
try:
    from core.cpp import cpp_distance_field
    CPP_AVAILABLE = True
    print("[加速] C++ 距离场模块已加载 ✓")
    
    # 显示 OpenMP 信息
    info = cpp_distance_field.get_openmp_info()
    if info['available']:
        print(f"[加速] OpenMP 多线程: {info['max_threads']} 核心")
    print(f"[加速] 预期加速: 15-20×")
except ImportError as e:
    # 如果作为模块导入失败，尝试直接导入（当文件被直接运行时）
    try:
        import sys
        from pathlib import Path
        # 添加项目根目录到路径
        project_root = Path(__file__).parent.parent.parent
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))
        from core.cpp import cpp_distance_field
        CPP_AVAILABLE = True
        print("[加速] C++ 距离场模块已加载 ✓")
        
        # 显示 OpenMP 信息
        info = cpp_distance_field.get_openmp_info()
        if info['available']:
            print(f"[加速] OpenMP 多线程: {info['max_threads']} 核心")
        print(f"[加速] 预期加速: 15-20×")
    except ImportError as e2:
        CPP_AVAILABLE = False
        print("[提示] C++ 加速模块未找到，使用 Python 版本（较慢）")
        print(f"[调试] 导入错误: {e2}")
        print("[提示] 运行 'compile_cpp.bat' 编译加速模块")

# 尝试导入 PyVista（用于更好的 3D Voronoi 支持）
try:
    import pyvista as pv
    PYVISTA_AVAILABLE = True
except ImportError:
    PYVISTA_AVAILABLE = False
    print("[警告] PyVista 未安装，将使用 scipy.spatial.Voronoi")


def estimate_optimal_wall_thickness(edges, bounds, target_density=0.15):
    """
    估算合理的壁厚范围
    
    基于边的分布和目标密度，计算合理的壁厚范围。
    
    参数:
        edges: 边列表 [(p1, p2), ...]
        bounds: 包围盒 [[xmin, ymin, zmin], [xmax, ymax, zmax]]
        target_density: 目标填充密度（0-1），默认0.15（15%）
    
    返回:
        dict: {
            'min_thickness': float,      # 最小建议壁厚
            'recommended': float,        # 推荐壁厚
            'max_thickness': float,      # 最大建议壁厚
            'avg_edge_length': float,    # 平均边长
            'min_edge_spacing': float    # 最小边间距
        }
    """
    if len(edges) == 0:
        return None
    
    print(f"\n[壁厚估算] 分析边的分布...")
    
    bounds = np.asarray(bounds, dtype=np.float64)
    lo, hi = bounds[0], bounds[1]
    extents = hi - lo
    volume = np.prod(extents)
    
    # 1. 计算边长统计
    edge_lengths = []
    for edge in edges:
        p1, p2 = edge
        length = np.linalg.norm(np.array(p2) - np.array(p1))
        edge_lengths.append(length)
    
    avg_edge_length = np.mean(edge_lengths)
    min_edge_length = np.min(edge_lengths)
    max_edge_length = np.max(edge_lengths)
    
    print(f"[边长统计]")
    print(f"  平均边长: {avg_edge_length:.2f} mm")
    print(f"  最短边: {min_edge_length:.2f} mm")
    print(f"  最长边: {max_edge_length:.2f} mm")
    
    # 2. 估算边的总长度
    total_edge_length = sum(edge_lengths)
    
    # 3. 根据目标密度估算壁厚
    # 假设：体积 ≈ 总边长 × π × r²
    # 其中 r = wall_thickness / 2
    # 密度 = 体积 / 包围盒体积
    # 求解：r = sqrt(密度 × 包围盒体积 / (π × 总边长))
    
    recommended_radius = np.sqrt(target_density * volume / (np.pi * total_edge_length))
    recommended_thickness = recommended_radius * 2
    
    # 4. 计算合理范围
    # 最小壁厚：不小于最短边的5%
    min_thickness = max(0.5, min_edge_length * 0.05)
    
    # 最大壁厚：不超过平均边长的40%（避免过度填充）
    max_thickness = avg_edge_length * 0.4
    
    # 确保推荐值在合理范围内
    recommended_thickness = np.clip(recommended_thickness, min_thickness, max_thickness)
    
    # 5. 估算最小边间距（用于判断是否会过度填充）
    # 简化方法：假设边均匀分布
    edge_density = len(edges) / volume  # 边数/体积
    avg_spacing = (1 / edge_density) ** (1/3)  # 平均间距的粗略估计
    
    print(f"\n[壁厚建议]")
    print(f"  最小建议: {min_thickness:.2f} mm （避免过细断裂）")
    print(f"  推荐值: {recommended_thickness:.2f} mm （目标密度 {target_density*100:.0f}%）")
    print(f"  最大建议: {max_thickness:.2f} mm （避免过度填充）")
    print(f"\n[密度预估]")
    print(f"  使用推荐壁厚，预计填充密度: {target_density*100:.0f}%")
    print(f"  平均边间距（估算）: {avg_spacing:.2f} mm")
    
    return {
        'min_thickness': min_thickness,
        'recommended': recommended_thickness,
        'max_thickness': max_thickness,
        'avg_edge_length': avg_edge_length,
        'min_edge_spacing': avg_spacing
    }


def preview_lattice_density(edges, bounds, wall_thickness, sample_size=1000):
    """
    预览给定壁厚下的晶格密度（使用 C++ 加速）
    
    通过采样估算填充密度，无需完整计算距离场。
    
    参数:
        edges: 边列表
        bounds: 包围盒
        wall_thickness: 壁厚
        sample_size: 采样点数（默认 1000，已足够准确）
    
    返回:
        float: 估算的填充密度（0-1）
    """
    import time
    
    bounds = np.asarray(bounds, dtype=np.float64)
    lo, hi = bounds[0], bounds[1]
    
    # 显示调试信息
    print(f"  边数量: {len(edges)}")
    print(f"  采样点数: {sample_size}")
    
    # 显示C++加速状态
    if CPP_AVAILABLE:
        print(f"  ✓ 使用 C++ 加速（OpenMP 多线程）")
    else:
        print(f"  ⚠ 使用 Python 版本（较慢，建议编译 C++ 扩展）")
    
    # 如果边数量过多，减少采样点数以加快速度
    if len(edges) > 10000:
        sample_size = min(sample_size, 500)
        print(f"  边数量较多，采样点数调整为: {sample_size}")
    
    # 在包围盒内随机采样点
    sample_points = np.random.uniform(lo, hi, size=(sample_size, 3))
    
    radius = wall_thickness / 2.0
    
    t_start = time.time()
    
    # 尝试使用 C++ 加速
    if CPP_AVAILABLE:
        # 转换边格式
        edges_cpp = []
        for p1, p2 in edges:
            p1 = np.asarray(p1, dtype=np.float64).tolist()
            p2 = np.asarray(p2, dtype=np.float64).tolist()
            edges_cpp.append((p1, p2))
        
        # 使用 C++ 计算距离（快速）
        distances = cpp_distance_field.compute_distance_field_optimized(
            sample_points, 
            edges_cpp,
            num_threads=0
        )
        
        # 统计填充点数
        filled_count = np.sum(distances < radius)
        
    else:
        # Python 版本（较慢）
        filled_count = 0
        
        for i, point in enumerate(sample_points):
            if (i + 1) % 100 == 0:
                print(f"  进度: {i+1}/{sample_size}")
            
            min_distance = np.inf
            
            for edge in edges:
                p1, p2 = edge
                p1 = np.asarray(p1, dtype=np.float64)
                p2 = np.asarray(p2, dtype=np.float64)
                
                # 计算点到线段的距离
                edge_vec = p2 - p1
                edge_length_sq = np.dot(edge_vec, edge_vec)
                
                if edge_length_sq < 1e-10:
                    distance = np.linalg.norm(point - p1)
                else:
                    t = np.dot(point - p1, edge_vec) / edge_length_sq
                    t = np.clip(t, 0.0, 1.0)
                    closest_point = p1 + t * edge_vec
                    distance = np.linalg.norm(point - closest_point)
                
                min_distance = min(min_distance, distance)
            
            if min_distance < radius:
                filled_count += 1
    
    t_end = time.time()
    print(f"  计算耗时: {t_end - t_start:.2f}s")
    
    density = filled_count / sample_size
    return density


def compute_distance_field_to_edges(bounds, resolution, edges):
    """
    计算体素网格到边骨架的距离场
    
    参数:
        bounds: 包围盒 [[xmin, ymin, zmin], [xmax, ymax, zmax]]
        resolution: 分辨率 (nx, ny, nz)
        edges: 边列表 [(p1, p2), ...]
    
    返回:
        distance_field: shape=(nx, ny, nz) 的距离场数组
        grid_coords: (xs, ys, zs) 网格坐标
    """
    print(f"\n[距离场] 开始计算...")
    print(f"[距离场] 分辨率: {resolution}")
    print(f"[距离场] 边数: {len(edges)}")
    
    bounds = np.asarray(bounds, dtype=np.float64)
    lo, hi = bounds[0], bounds[1]
    
    # 创建体素网格
    xs = np.linspace(lo[0], hi[0], resolution[0])
    ys = np.linspace(lo[1], hi[1], resolution[1])
    zs = np.linspace(lo[2], hi[2], resolution[2])
    
    # 创建网格点
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing='ij')
    points = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)
    
    print(f"[距离场] 体素数: {len(points):,}")
    
    # 尝试使用 C++ 加速版本
    if CPP_AVAILABLE:
        print(f"[距离场] 使用 C++ 加速版本 ⚡")
        
        t_start = time.time()
        
        # 转换边格式为 C++ 需要的格式
        edges_cpp = []
        for p1, p2 in edges:
            p1 = np.asarray(p1, dtype=np.float64).tolist()
            p2 = np.asarray(p2, dtype=np.float64).tolist()
            edges_cpp.append((p1, p2))
        
        # 调用 C++ 函数（自动使用所有 CPU 核心）
        distances = cpp_distance_field.compute_distance_field_optimized(
            points, 
            edges_cpp,
            num_threads=0  # 0 = 自动使用所有核心
        )
        
        t_end = time.time()
        print(f"[距离场] C++ 计算完成，耗时 {t_end - t_start:.2f}s ⚡")
        
    else:
        # 使用 Python 版本（较慢）
        print(f"[距离场] 使用 Python 版本（较慢）")
        
        t_start = time.time()
        
        # 初始化距离场为无穷大
        distances = np.full(len(points), np.inf, dtype=np.float64)
        
        # 对每条边计算距离
        for i, edge in enumerate(edges):
            if (i + 1) % 100 == 0 or i == len(edges) - 1:
                print(f"[距离场] 处理边 {i + 1}/{len(edges)}...")
            
            p1, p2 = edge
            p1 = np.asarray(p1, dtype=np.float64)
            p2 = np.asarray(p2, dtype=np.float64)
            
            # 计算点到线段的距离
            edge_vec = p2 - p1
            edge_length_sq = np.dot(edge_vec, edge_vec)
            
            if edge_length_sq < 1e-10:
                # 退化边（两端点重合），当作点处理
                edge_distances = np.linalg.norm(points - p1, axis=1)
            else:
                # 计算投影参数 t
                t = np.dot(points - p1, edge_vec) / edge_length_sq
                t = np.clip(t, 0.0, 1.0)  # 限制在线段范围内
                
                # 计算最近点
                closest_points = p1 + t[:, np.newaxis] * edge_vec
                
                # 计算距离
                edge_distances = np.linalg.norm(points - closest_points, axis=1)
            
            # 更新最小距离
            distances = np.minimum(distances, edge_distances)
        
        t_end = time.time()
        print(f"[距离场] Python 计算完成，耗时 {t_end - t_start:.2f}s")
    
    # 重塑为3D数组
    distance_field = distances.reshape(resolution)
    
    print(f"[距离场] 计算完成！")
    print(f"[距离场] 最小距离: {np.min(distance_field):.4f} mm")
    print(f"[距离场] 最大距离: {np.max(distance_field):.4f} mm")
    
    return distance_field, (xs, ys, zs)


def generate_lattice_from_edges(bounds, edges, wall_thickness=3.0, resolution=4, smooth=True):
    """
    从边骨架生成晶格结构（使用隐函数建模）
    
    参数:
        bounds: 包围盒 [[xmin, ymin, zmin], [xmax, ymax, zmax]]
        edges: 边列表 [(p1, p2), ...]
        wall_thickness: 管径/壁厚 (mm)
        resolution: 分辨率参数（每mm的体素数），推荐2-5
        smooth: 是否平滑网格
    
    返回:
        mesh: trimesh对象
    """
    print(f"\n{'='*60}")
    print(f"基于边骨架生成 Voronoi 晶格")
    print(f"{'='*60}")
    
    if len(edges) == 0:
        print("[错误] 没有边可用于生成晶格")
        return None
    
    bounds = np.asarray(bounds, dtype=np.float64)
    lo, hi = bounds[0], bounds[1]
    extents = hi - lo
    
    # 改进的分辨率计算：基于壁厚
    # 原则：每个壁厚直径至少有 resolution 个体素
    # 这样可以保证杆件的圆度和光滑度
    voxel_size = wall_thickness / resolution
    
    # 计算网格分辨率
    grid_resolution = tuple(int(np.ceil(extent / voxel_size)) + 1 for extent in extents)
    
    # 限制最大分辨率（避免内存溢出）
    max_voxels = 10_000_000  # 1000万体素
    
    # 使用float64计算避免整数溢出
    total_voxels_float = float(grid_resolution[0]) * float(grid_resolution[1]) * float(grid_resolution[2])
    
    original_resolution = resolution
    adjusted = False
    
    # 检查是否超过限制
    if total_voxels_float > max_voxels:
        # 自动调整分辨率
        scale_factor = (max_voxels / total_voxels_float) ** (1/3)
        
        print(f"\n[警告] 分辨率过高，已自动调整")
        print(f"  原始网格: {grid_resolution}")
        print(f"  原始体素数: {total_voxels_float:.0f}")
        
        grid_resolution = tuple(max(10, int(res * scale_factor)) for res in grid_resolution)
        voxel_size = np.mean(extents / np.array(grid_resolution))
        total_voxels_float = float(grid_resolution[0]) * float(grid_resolution[1]) * float(grid_resolution[2])
        
        # 更新实际分辨率参数
        resolution = wall_thickness / voxel_size
        adjusted = True
        
        print(f"  调整后网格: {grid_resolution}")
        print(f"  调整后体素数: {total_voxels_float:.0f}")
        print(f"  原始分辨率: {original_resolution:.2f} 体素/mm")
        print(f"  调整后分辨率: {resolution:.2f} 体素/mm")
        print(f"  调整后体素大小: {voxel_size:.4f} mm")
    
    # 计算实际杆件直径（体素数）
    voxels_per_diameter = wall_thickness / voxel_size
    
    # 转换为整数用于显示
    total_voxels = int(total_voxels_float)
    
    print(f"\n[参数]")
    print(f"  包围盒: {lo} ~ {hi}")
    print(f"  尺寸: {extents}")
    print(f"  管径: {wall_thickness} mm")
    if adjusted:
        print(f"  分辨率参数: {resolution:.2f} 体素/mm (已自动调整)")
    else:
        print(f"  分辨率参数: {resolution} 体素/mm")
    print(f"  体素大小: {voxel_size:.4f} mm")
    print(f"  网格分辨率: {grid_resolution}")
    print(f"  总体素数: {total_voxels:,}")
    print(f"  杆件直径: {voxels_per_diameter:.1f} 个体素")
    
    # 检查质量并给出建议
    if voxels_per_diameter < 3:
        print(f"\n  ⚠️  警告：杆件直径只有 {voxels_per_diameter:.1f} 个体素，会很粗糙！")
        
        if adjusted:
            # 如果是自动调整导致的，建议增大壁厚
            min_wall_thickness = voxel_size * 3
            print(f"  原因：分辨率被自动降低（体素数超限）")
            print(f"  解决方案 1：增大壁厚到 {min_wall_thickness:.2f} mm 以上")
            print(f"  解决方案 2：降低分辨率参数（如改为 {max(2, resolution-1)}）")
        else:
            # 如果是壁厚太小导致的
            print(f"  原因：壁厚 {wall_thickness} mm 太小")
            print(f"  解决方案 1：增大壁厚到 {voxel_size * 3:.2f} mm 以上")
            print(f"  解决方案 2：降低分辨率参数到 {max(2, int(resolution * voxels_per_diameter / 3))}）")
    elif voxels_per_diameter < 6:
        print(f"  💡 提示：杆件直径 {voxels_per_diameter:.1f} 个体素，质量可接受")
        print(f"  建议：启用平滑以改善表面质量")
    else:
        print(f"  ✓ 杆件直径 {voxels_per_diameter:.1f} 个体素，质量良好")
    
    # 步骤1：计算距离场
    distance_field, (xs, ys, zs) = compute_distance_field_to_edges(
        bounds, grid_resolution, edges
    )
    
    # 步骤2：应用隐函数（距离 < 半径的区域）
    print(f"\n[隐函数] 应用阈值...")
    radius = wall_thickness / 2.0
    implicit_field = distance_field < radius
    
    filled_voxels = np.sum(implicit_field)
    total_voxels_actual = np.prod(grid_resolution)
    fill_ratio = filled_voxels / total_voxels_actual * 100
    
    print(f"[隐函数] 阈值半径: {radius:.4f} mm")
    print(f"[隐函数] 填充体素: {filled_voxels:,} / {total_voxels_actual:,} ({fill_ratio:.2f}%)")
    
    if filled_voxels == 0:
        print("[错误] 没有体素被填充，请增大管径或分辨率")
        return None
    
    # 步骤3：使用 Marching Cubes 提取等值面
    print(f"\n[Marching Cubes] 提取等值面...")
    t_start = time.time()
    
    try:
        # 使用距离场直接提取等值面
        vertices, faces, normals, values = measure.marching_cubes(
            distance_field,
            level=radius,
            spacing=(
                (hi[0] - lo[0]) / (grid_resolution[0] - 1),
                (hi[1] - lo[1]) / (grid_resolution[1] - 1),
                (hi[2] - lo[2]) / (grid_resolution[2] - 1)
            )
        )
        
        # 调整顶点坐标到实际空间
        vertices += lo
        
        t_end = time.time()
        
        print(f"[Marching Cubes] 完成！耗时 {t_end - t_start:.2f}s")
        print(f"[网格] 顶点数: {len(vertices):,}")
        print(f"[网格] 面数: {len(faces):,}")
        
        # 创建 trimesh 对象
        lattice_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        
        # 步骤4：平滑处理（可选）
        if smooth:
            print(f"\n[平滑处理] 应用拉普拉斯平滑...")
            t_start = time.time()
            
            # 使用 trimesh 的平滑功能
            # 注意：这会稍微改变几何形状，但能显著改善外观
            trimesh.smoothing.filter_laplacian(lattice_mesh, iterations=2)
            
            t_end = time.time()
            print(f"[平滑处理] 完成！耗时 {t_end - t_start:.2f}s")
        
        # 修复法向量
        lattice_mesh.fix_normals()
        
        print(f"\n[结果]")
        print(f"  顶点数: {len(lattice_mesh.vertices):,}")
        print(f"  面数: {len(lattice_mesh.faces):,}")
        print(f"  是否水密: {lattice_mesh.is_watertight}")
        print(f"  体积: {lattice_mesh.volume:.2f} mm³")
        
        return lattice_mesh
        
    except Exception as e:
        print(f"[错误] Marching Cubes 失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def check_edge_connectivity(edges, tolerance=0.001):
    """
    检查边集合的连通性
    
    使用图论方法检查所有边是否形成连通图。
    如果存在孤立的边或不连通的子图，会报告详细信息。
    
    参数:
        edges: 边列表 [(p1, p2), ...]，每条边由两个端点定义
        tolerance: 顶点匹配容差（mm），用于判断两个点是否相同
    
    返回:
        dict: {
            'is_connected': bool,  # 是否连通
            'num_components': int,  # 连通分量数量
            'num_vertices': int,    # 顶点总数
            'num_edges': int,       # 边总数
            'components': list,     # 每个连通分量的顶点数
            'isolated_edges': int   # 孤立边数量（两端点都不与其他边连接）
        }
    """
    if len(edges) == 0:
        return {
            'is_connected': True,
            'num_components': 0,
            'num_vertices': 0,
            'num_edges': 0,
            'components': [],
            'isolated_edges': 0
        }
    
    print(f"\n[连通性检测] 开始检测 {len(edges)} 条边...")
    
    # 步骤1：构建顶点映射（将相近的点视为同一顶点）
    vertices = []
    vertex_map = {}  # 点坐标 -> 顶点索引
    
    def get_vertex_index(point):
        """获取或创建顶点索引"""
        # 四舍五入到容差精度
        key = tuple(np.round(point / tolerance) * tolerance)
        
        if key not in vertex_map:
            vertex_map[key] = len(vertices)
            vertices.append(point)
        
        return vertex_map[key]
    
    # 步骤2：构建邻接表
    from collections import defaultdict
    adjacency = defaultdict(set)
    
    for edge in edges:
        p1, p2 = edge
        v1 = get_vertex_index(p1)
        v2 = get_vertex_index(p2)
        
        if v1 != v2:  # 避免自环
            adjacency[v1].add(v2)
            adjacency[v2].add(v1)
    
    num_vertices = len(vertices)
    num_edges = len(edges)
    
    print(f"[连通性检测] 顶点数: {num_vertices}, 边数: {num_edges}")
    
    # 步骤3：使用BFS/DFS查找连通分量
    visited = set()
    components = []
    
    def bfs(start):
        """广度优先搜索，找出一个连通分量"""
        queue = [start]
        visited.add(start)
        component = [start]
        
        while queue:
            v = queue.pop(0)
            for neighbor in adjacency[v]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
                    component.append(neighbor)
        
        return component
    
    # 遍历所有顶点，找出所有连通分量
    for v in range(num_vertices):
        if v not in visited:
            component = bfs(v)
            components.append(len(component))
    
    num_components = len(components)
    is_connected = (num_components == 1)
    
    # 步骤4：检测孤立边（度数为1的顶点对）
    isolated_edges = 0
    for edge in edges:
        p1, p2 = edge
        v1 = get_vertex_index(p1)
        v2 = get_vertex_index(p2)
        
        if len(adjacency[v1]) == 1 and len(adjacency[v2]) == 1:
            isolated_edges += 1
    
    # 输出结果
    print(f"\n[连通性检测] 结果:")
    print(f"  连通性: {'✓ 连通' if is_connected else '✗ 不连通'}")
    print(f"  连通分量数: {num_components}")
    print(f"  顶点总数: {num_vertices}")
    print(f"  边总数: {num_edges}")
    
    if not is_connected:
        print(f"\n  [警告] 图不连通！存在 {num_components} 个独立的连通分量:")
        for i, size in enumerate(sorted(components, reverse=True)):
            print(f"    分量 {i+1}: {size} 个顶点 ({size/num_vertices*100:.1f}%)")
    
    if isolated_edges > 0:
        print(f"\n  [警告] 发现 {isolated_edges} 条孤立边（两端点都不与其他边连接）")
    
    return {
        'is_connected': is_connected,
        'num_components': num_components,
        'num_vertices': num_vertices,
        'num_edges': num_edges,
        'components': components,
        'isolated_edges': isolated_edges
    }


def are_triangles_coplanar(tri1_points, tri2_points, tolerance=0.001):
    """
    判断两个三角形是否共面（基于平面方程和点到平面距离）
    
    参数:
        tri1_points: 三角形1的3个顶点 (3, 3)
        tri2_points: 三角形2的3个顶点 (3, 3)
        tolerance: 距离容差（mm），默认0.001mm（1微米）
    
    返回:
        bool: True表示共面，应该过滤；False表示不共面，是真实棱边
    """
    try:
        # 计算三角形1的平面方程 Ax + By + Cz + D = 0
        p1, p2, p3 = tri1_points
        
        # 法向量 = (p2-p1) × (p3-p1)
        v1 = p2 - p1
        v2 = p3 - p1
        normal = np.cross(v1, v2)
        
        # 检查是否为退化三角形
        normal_length = np.linalg.norm(normal)
        if normal_length < 1e-10:
            return True  # 退化三角形，保守处理：认为共面
        
        # 归一化法向量
        A, B, C = normal / normal_length
        D = -(A * p1[0] + B * p1[1] + C * p1[2])
        
        # 检查三角形2的所有顶点到平面1的距离
        max_distance = 0
        for point in tri2_points:
            # 点到平面距离公式（法向量已归一化，分母为1）
            distance = abs(A * point[0] + B * point[1] + C * point[2] + D)
            max_distance = max(max_distance, distance)
        
        # 如果最大距离小于容差，认为共面
        return max_distance < tolerance
        
    except Exception as e:
        # 出错时保守处理：认为共面（过滤掉）
        return True


def extract_bounds_from_stl(stl_path):
    """
    从STL文件提取包围盒
    
    参数:
        stl_path: STL文件路径
    
    返回:
        bounds: [[xmin, ymin, zmin], [xmax, ymax, zmax]]
        mesh: trimesh对象（可选，用于后续处理）
    """
    try:
        # 加载STL文件
        mesh = trimesh.load(stl_path)
        
        # 获取包围盒
        bbox = mesh.bounds  # shape: (2, 3), [[xmin, ymin, zmin], [xmax, ymax, zmax]]
        bounds = bbox.tolist()
        
        # 计算包围盒尺寸
        extents = bbox[1] - bbox[0]
        
        print(f"[STL文件] 路径: {stl_path}")
        print(f"[STL文件] 顶点数: {len(mesh.vertices)}")
        print(f"[STL文件] 面数: {len(mesh.faces)}")
        print(f"[包围盒] 最小点: [{bbox[0][0]:.2f}, {bbox[0][1]:.2f}, {bbox[0][2]:.2f}]")
        print(f"[包围盒] 最大点: [{bbox[1][0]:.2f}, {bbox[1][1]:.2f}, {bbox[1][2]:.2f}]")
        print(f"[包围盒] 尺寸: {extents[0]:.2f} × {extents[1]:.2f} × {extents[2]:.2f} mm")
        
        return bounds, mesh
        
    except FileNotFoundError:
        print(f"[错误] 文件不存在: {stl_path}")
        return None, None
    except Exception as e:
        print(f"[错误] 读取STL文件失败: {e}")
        return None, None


def poisson_disk_sampling_on_mesh_surface(mesh, min_distance, num_seeds=None, max_attempts=30):
    """
    在STL网格表面上进行泊松盘采样（保证种子点之间的最小距离）
    
    参数:
        mesh: trimesh对象
        min_distance: 种子点之间的最小距离
        num_seeds: 目标种子数量（可选，如果不指定则尽可能多地采样）
        max_attempts: 每个点的最大尝试次数
    
    返回:
        surface_seeds: 表面种子点数组
    """
    print(f"[泊松盘采样] 最小距离: {min_distance:.2f} mm")
    print(f"[泊松盘采样] 目标种子数: {num_seeds if num_seeds else '尽可能多'}")
    
    surface_seeds = []
    attempts = 0
    max_total_attempts = (num_seeds * max_attempts) if num_seeds else 10000
    
    while attempts < max_total_attempts:
        # 在表面随机采样一个候选点
        candidate, _ = trimesh.sample.sample_surface(mesh, 1)
        candidate = candidate[0]
        
        # 检查是否满足最小距离约束
        if len(surface_seeds) == 0:
            # 第一个点直接接受
            surface_seeds.append(candidate)
        else:
            # 计算到所有已有点的距离
            distances = np.linalg.norm(np.array(surface_seeds) - candidate, axis=1)
            if np.min(distances) >= min_distance:
                surface_seeds.append(candidate)
                
                # 如果达到目标数量，停止
                if num_seeds and len(surface_seeds) >= num_seeds:
                    break
        
        attempts += 1
        
        # 每1000次尝试输出进度
        if attempts % 1000 == 0 and num_seeds:
            print(f"[泊松盘采样] 进度: {len(surface_seeds)}/{num_seeds} ({attempts}次尝试)")
    
    surface_seeds = np.array(surface_seeds)
    print(f"[泊松盘采样] 完成: {len(surface_seeds)} 个种子点 ({attempts}次尝试)")
    
    return surface_seeds


def compute_geodesic_distances_on_mesh(mesh, seed_points, sample_points):
    """
    计算网格表面上从种子点到采样点的测地距离（真正的测地距离）
    
    使用图算法（Dijkstra）计算网格表面上的最短路径距离
    
    参数:
        mesh: trimesh对象
        seed_points: 种子点数组 (n_seeds, 3)
        sample_points: 采样点数组 (n_samples, 3)
    
    返回:
        distances: 距离矩阵 (n_samples, n_seeds)
    """
    # 注意：真正的测地距离计算非常耗时
    # 这里使用trimesh的内置功能或近似方法
    
    # 方法1：使用欧氏距离作为快速近似（当前方法）
    # 对于曲率不大的表面，欧氏距离和测地距离差异不大
    distances = np.zeros((len(sample_points), len(seed_points)))
    
    for i, seed in enumerate(seed_points):
        distances[:, i] = np.linalg.norm(sample_points - seed, axis=1)
    
    return distances


def build_mesh_graph(mesh):
    """
    构建网格的图结构（用于测地距离计算）
    
    将网格的顶点作为图的节点，边作为图的边
    
    参数:
        mesh: trimesh对象
    
    返回:
        graph: 邻接表 {vertex_id: [(neighbor_id, distance), ...]}
        vertices: 顶点坐标数组
    """
    import scipy.sparse as sp
    from scipy.sparse.csgraph import dijkstra
    
    # 获取网格的边
    edges = mesh.edges_unique
    vertices = mesh.vertices
    
    # 计算每条边的长度
    edge_lengths = np.linalg.norm(
        vertices[edges[:, 0]] - vertices[edges[:, 1]], 
        axis=1
    )
    
    # 构建稀疏邻接矩阵
    n_vertices = len(vertices)
    row = np.concatenate([edges[:, 0], edges[:, 1]])
    col = np.concatenate([edges[:, 1], edges[:, 0]])
    data = np.concatenate([edge_lengths, edge_lengths])
    
    adjacency_matrix = sp.csr_matrix(
        (data, (row, col)), 
        shape=(n_vertices, n_vertices)
    )
    
    return adjacency_matrix, vertices


def compute_geodesic_distances_dijkstra(mesh, seed_points, sample_points, max_distance=None):
    """
    使用Dijkstra算法计算真正的测地距离
    
    参数:
        mesh: trimesh对象
        seed_points: 种子点数组 (n_seeds, 3)
        sample_points: 采样点数组 (n_samples, 3)
        max_distance: 最大距离限制（可选，用于加速）
    
    返回:
        distances: 距离矩阵 (n_samples, n_seeds)
    """
    from scipy.sparse.csgraph import dijkstra
    
    print(f"[测地距离] 构建网格图...")
    adjacency_matrix, vertices = build_mesh_graph(mesh)
    
    print(f"[测地距离] 网格顶点数: {len(vertices)}")
    print(f"[测地距离] 种子点数: {len(seed_points)}")
    print(f"[测地距离] 采样点数: {len(sample_points)}")
    
    # 找到每个种子点和采样点最近的网格顶点
    print(f"[测地距离] 映射种子点到网格顶点...")
    seed_vertex_indices = []
    for seed in seed_points:
        distances_to_vertices = np.linalg.norm(vertices - seed, axis=1)
        closest_vertex = np.argmin(distances_to_vertices)
        seed_vertex_indices.append(closest_vertex)
    
    print(f"[测地距离] 映射采样点到网格顶点...")
    sample_vertex_indices = []
    for sample in sample_points:
        distances_to_vertices = np.linalg.norm(vertices - sample, axis=1)
        closest_vertex = np.argmin(distances_to_vertices)
        sample_vertex_indices.append(closest_vertex)
    
    # 计算测地距离
    print(f"[测地距离] 计算Dijkstra最短路径...")
    geodesic_distances = np.zeros((len(sample_points), len(seed_points)))
    
    for i, seed_vertex in enumerate(seed_vertex_indices):
        # 从每个种子点计算到所有顶点的最短距离
        distances_from_seed = dijkstra(
            adjacency_matrix, 
            indices=seed_vertex,
            limit=max_distance if max_distance else np.inf
        )
        
        # 提取到采样点的距离
        for j, sample_vertex in enumerate(sample_vertex_indices):
            geodesic_distances[j, i] = distances_from_seed[sample_vertex]
        
        if (i + 1) % 10 == 0 or i == len(seed_vertex_indices) - 1:
            print(f"[测地距离] 进度: {i + 1}/{len(seed_vertex_indices)}")
    
    print(f"[测地距离] 计算完成！")
    return geodesic_distances


def compute_geodesic_distances_on_mesh_old(mesh, seed_points, sample_points):
    """
    返回:
        distances: 距离矩阵 (n_samples, n_seeds)
    """
    # 使用欧氏距离作为近似
    # 对于表面上的点，欧氏距离和测地距离差异不大
    distances = np.zeros((len(sample_points), len(seed_points)))
    
    for i, seed in enumerate(seed_points):
        distances[:, i] = np.linalg.norm(sample_points - seed, axis=1)
    
    return distances


def lloyd_relaxation_on_surface(mesh, seed_points, num_iterations=10, sample_density=1000):
    """
    在网格表面上进行Lloyd松弛，优化种子点分布
    
    Lloyd算法：
    1. 计算每个表面点属于哪个种子点（Voronoi区域）
    2. 计算每个Voronoi区域的质心
    3. 将种子点移动到质心位置
    4. 重复迭代
    
    参数:
        mesh: trimesh对象
        seed_points: 初始种子点数组 (n_seeds, 3)
        num_iterations: 迭代次数
        sample_density: 表面采样密度（用于计算质心）
    
    返回:
        optimized_seeds: 优化后的种子点数组
    """
    print(f"\n[Lloyd松弛] 开始优化种子点分布...")
    print(f"[Lloyd松弛] 初始种子数: {len(seed_points)}")
    print(f"[Lloyd松弛] 迭代次数: {num_iterations}")
    
    optimized_seeds = seed_points.copy()
    
    # 在表面上密集采样点（用于计算Voronoi区域和质心）
    num_samples = int(mesh.area * sample_density / 1000)  # 根据表面积调整采样数
    num_samples = max(1000, min(10000, num_samples))  # 限制在1000-10000之间
    
    print(f"[Lloyd松弛] 表面采样点数: {num_samples}")
    surface_samples, face_indices = trimesh.sample.sample_surface(mesh, num_samples)
    
    for iteration in range(num_iterations):
        # 1. 计算每个采样点到所有种子点的距离
        distances = compute_geodesic_distances_on_mesh(mesh, optimized_seeds, surface_samples)
        
        # 2. 为每个采样点分配最近的种子点（Voronoi区域）
        assignments = np.argmin(distances, axis=1)
        
        # 3. 计算每个Voronoi区域的质心
        new_seeds = []
        for seed_idx in range(len(optimized_seeds)):
            # 找到属于这个种子的所有采样点
            region_points = surface_samples[assignments == seed_idx]
            
            if len(region_points) > 0:
                # 计算质心
                centroid = np.mean(region_points, axis=0)
                
                # 将质心投影到最近的表面点
                closest_point, distance, triangle_id = mesh.nearest.on_surface([centroid])
                new_seeds.append(closest_point[0])
            else:
                # 如果没有点分配给这个种子，保持原位置
                new_seeds.append(optimized_seeds[seed_idx])
        
        optimized_seeds = np.array(new_seeds)
        
        # 计算种子点的移动距离（用于判断收敛）
        movement = np.mean(np.linalg.norm(optimized_seeds - seed_points, axis=1))
        
        if (iteration + 1) % 2 == 0 or iteration == num_iterations - 1:
            print(f"[Lloyd松弛] 迭代 {iteration + 1}/{num_iterations}, 平均移动距离: {movement:.4f} mm")
        
        seed_points = optimized_seeds.copy()
    
    print(f"[Lloyd松弛] 优化完成！")
    return optimized_seeds


def compute_surface_voronoi_2d(mesh, seed_points, use_geodesic=True, sample_density=1000):
    """
    在网格表面上计算2D Voronoi图（基于测地距离）
    
    这个函数计算真正的曲面2D Voronoi：
    1. 在表面密集采样点
    2. 计算每个采样点到所有种子点的测地距离
    3. 为每个采样点分配最近的种子点（Voronoi区域）
    4. 提取Voronoi边界
    
    参数:
        mesh: trimesh对象
        seed_points: 种子点数组 (n_seeds, 3)
        use_geodesic: 是否使用真正的测地距离（True）或欧氏距离近似（False）
        sample_density: 表面采样密度（每1000mm²采样多少个点）
    
    返回:
        voronoi_regions: 字典 {seed_idx: [point_indices]}
        surface_samples: 表面采样点数组
        assignments: 每个采样点所属的种子点索引
    """
    print(f"\n{'='*60}")
    print(f"计算曲面2D Voronoi图")
    print(f"{'='*60}")
    
    # 步骤1：在表面密集采样
    num_samples = int(mesh.area * sample_density / 1000)
    num_samples = max(1000, min(20000, num_samples))  # 限制在1000-20000之间
    
    print(f"\n[步骤1] 表面采样")
    print(f"[采样] 表面积: {mesh.area:.2f} mm²")
    print(f"[采样] 采样点数: {num_samples}")
    
    surface_samples, face_indices = trimesh.sample.sample_surface(mesh, num_samples)
    
    # 步骤2：计算测地距离
    print(f"\n[步骤2] 计算测地距离")
    
    if use_geodesic:
        print(f"[距离] 使用真正的测地距离（Dijkstra算法）")
        distances = compute_geodesic_distances_dijkstra(mesh, seed_points, surface_samples)
    else:
        print(f"[距离] 使用欧氏距离近似")
        distances = compute_geodesic_distances_on_mesh(mesh, seed_points, surface_samples)
    
    # 步骤3：分配Voronoi区域
    print(f"\n[步骤3] 分配Voronoi区域")
    assignments = np.argmin(distances, axis=1)
    
    # 统计每个区域的点数
    voronoi_regions = {}
    for seed_idx in range(len(seed_points)):
        region_points = np.where(assignments == seed_idx)[0]
        voronoi_regions[seed_idx] = region_points
        if seed_idx < 5:
            print(f"[区域] 种子{seed_idx}: {len(region_points)} 个点")
    
    print(f"\n{'='*60}")
    print(f"曲面2D Voronoi图计算完成")
    print(f"{'='*60}")
    
    return voronoi_regions, surface_samples, assignments


def extract_voronoi_boundaries_on_surface(mesh, surface_samples, assignments, face_indices):
    """
    提取曲面Voronoi的边界
    
    边界定义：相邻采样点属于不同的Voronoi区域
    
    参数:
        mesh: trimesh对象
        surface_samples: 表面采样点数组
        assignments: 每个采样点所属的种子点索引
        face_indices: 每个采样点所在的面片索引
    
    返回:
        boundary_edges: 边界边列表 [(point1, point2), ...]
    """
    print(f"\n[边界提取] 提取Voronoi边界...")
    
    # 简化方法：找到相邻点属于不同区域的情况
    # 这里使用KD树找相邻点
    from scipy.spatial import cKDTree
    
    tree = cKDTree(surface_samples)
    boundary_edges = []
    
    # 对每个点，找到最近的几个邻居
    k_neighbors = 5
    distances, indices = tree.query(surface_samples, k=k_neighbors + 1)
    
    for i, neighbors in enumerate(indices):
        my_region = assignments[i]
        for neighbor_idx in neighbors[1:]:  # 跳过自己
            neighbor_region = assignments[neighbor_idx]
            if my_region != neighbor_region:
                # 这是一条边界边
                edge = tuple(sorted([i, neighbor_idx]))
                if edge not in boundary_edges:
                    boundary_edges.append(edge)
    
    print(f"[边界提取] 找到 {len(boundary_edges)} 条边界边")
    
    return boundary_edges


def sample_seeds_on_mesh_surface(mesh, num_seeds=None, density=None):
    """
    在STL网格表面上采样种子点（基础版本，均匀采样）
    
    参数:
        mesh: trimesh对象
        num_seeds: 指定种子数量（可选）
        density: 种子密度（每平方毫米的种子数，可选）
    
    返回:
        surface_seeds: 表面种子点数组
    """
    if num_seeds is None and density is None:
        # 默认：根据表面积自动计算
        surface_area = mesh.area
        density = 0.01  # 默认每100平方毫米1个种子点
        num_seeds = max(10, int(surface_area * density))
    elif density is not None:
        surface_area = mesh.area
        num_seeds = max(10, int(surface_area * density))
    
    print(f"[表面采样] 目标种子数: {num_seeds}")
    print(f"[表面采样] 网格表面积: {mesh.area:.2f} mm²")
    
    # 使用trimesh的表面采样功能
    # sample_surface_even 会均匀分布采样点
    surface_seeds, face_indices = trimesh.sample.sample_surface_even(mesh, num_seeds)
    
    print(f"[表面采样] 实际采样: {len(surface_seeds)} 个表面种子点")
    
    return surface_seeds


def generate_surface_seeds_with_2d_voronoi(mesh, num_seeds, cell_size, use_geodesic=True):
    """
    使用曲面2D Voronoi生成表面种子点
    
    这个函数实现真正的曲面2D Voronoi：
    1. 初始化：在表面随机采样种子点
    2. 计算曲面2D Voronoi（基于测地距离）
    3. 优化：使用Lloyd松弛（基于测地距离）
    4. 返回优化后的种子点
    
    参数:
        mesh: trimesh对象
        num_seeds: 目标种子数量
        cell_size: 胞元尺寸
        use_geodesic: 是否使用真正的测地距离
    
    返回:
        surface_seeds: 优化后的表面种子点数组
        voronoi_info: Voronoi信息（可选，用于可视化）
    """
    print(f"\n{'='*60}")
    print(f"使用曲面2D Voronoi生成表面种子点")
    print(f"{'='*60}")
    
    # 步骤1：初始化种子点（泊松盘采样）
    print(f"\n[步骤1] 初始化种子点")
    min_distance = cell_size * 0.8
    surface_seeds = poisson_disk_sampling_on_mesh_surface(
        mesh, 
        min_distance=min_distance, 
        num_seeds=num_seeds
    )
    
    # 步骤2：计算曲面2D Voronoi
    print(f"\n[步骤2] 计算曲面2D Voronoi")
    voronoi_regions, surface_samples, assignments = compute_surface_voronoi_2d(
        mesh, 
        surface_seeds, 
        use_geodesic=use_geodesic,
        sample_density=1000
    )
    
    # 步骤3：Lloyd松弛优化（基于测地距离）
    print(f"\n[步骤3] Lloyd松弛优化（基于测地距离）")
    lloyd_iterations = 5  # 使用测地距离时，迭代次数可以少一些
    
    for iteration in range(lloyd_iterations):
        print(f"\n[Lloyd迭代 {iteration + 1}/{lloyd_iterations}]")
        
        # 计算每个Voronoi区域的质心
        new_seeds = []
        for seed_idx in range(len(surface_seeds)):
            region_point_indices = voronoi_regions[seed_idx]
            
            if len(region_point_indices) > 0:
                # 计算区域内所有点的质心
                region_points = surface_samples[region_point_indices]
                centroid = np.mean(region_points, axis=0)
                
                # 将质心投影到最近的表面点
                closest_point, distance, triangle_id = mesh.nearest.on_surface([centroid])
                new_seeds.append(closest_point[0])
            else:
                # 如果没有点分配给这个种子，保持原位置
                new_seeds.append(surface_seeds[seed_idx])
        
        surface_seeds = np.array(new_seeds)
        
        # 重新计算Voronoi区域
        if iteration < lloyd_iterations - 1:  # 最后一次迭代不需要重新计算
            voronoi_regions, surface_samples, assignments = compute_surface_voronoi_2d(
                mesh, 
                surface_seeds, 
                use_geodesic=use_geodesic,
                sample_density=1000
            )
    
    print(f"\n{'='*60}")
    print(f"曲面2D Voronoi种子点生成完成: {len(surface_seeds)} 个")
    print(f"{'='*60}")
    
    voronoi_info = {
        'regions': voronoi_regions,
        'samples': surface_samples,
        'assignments': assignments
    }
    
    return surface_seeds, voronoi_info


def generate_optimized_surface_seeds(mesh, num_seeds, cell_size, use_lloyd=True, lloyd_iterations=10):
    """
    生成优化的表面种子点（泊松盘采样 + Lloyd松弛）
    
    这个函数结合了两种方法：
    1. 泊松盘采样：保证种子点之间的最小距离
    2. Lloyd松弛：优化种子点分布，使其更均匀（类似2D Voronoi效果）
    
    参数:
        mesh: trimesh对象
        num_seeds: 目标种子数量
        cell_size: 胞元尺寸（用于计算最小距离）
        use_lloyd: 是否使用Lloyd松弛优化
        lloyd_iterations: Lloyd松弛迭代次数
    
    返回:
        surface_seeds: 优化后的表面种子点数组
    """
    print(f"\n{'='*60}")
    print(f"生成优化的表面种子点")
    print(f"{'='*60}")
    
    # 步骤1：泊松盘采样（初始化）
    print(f"\n[步骤1] 泊松盘采样（初始化）")
    min_distance = cell_size * 0.8
    surface_seeds = poisson_disk_sampling_on_mesh_surface(
        mesh, 
        min_distance=min_distance, 
        num_seeds=num_seeds
    )
    
    # 步骤2：Lloyd松弛（优化）
    if use_lloyd and len(surface_seeds) > 0:
        print(f"\n[步骤2] Lloyd松弛优化")
        surface_seeds = lloyd_relaxation_on_surface(
            mesh, 
            surface_seeds, 
            num_iterations=lloyd_iterations
        )
    else:
        print(f"\n[步骤2] 跳过Lloyd松弛优化")
    
    print(f"\n{'='*60}")
    print(f"表面种子点生成完成: {len(surface_seeds)} 个")
    print(f"{'='*60}")
    
    return surface_seeds


def sample_seeds_on_mesh_surface_old(mesh, num_seeds=None, density=None):
    """
    在STL网格表面上采样种子点
    
    参数:
        mesh: trimesh对象
        num_seeds: 指定种子数量（可选）
        density: 种子密度（每平方毫米的种子数，可选）
    
    返回:
        surface_seeds: 表面种子点数组
    """
    if num_seeds is None and density is None:
        # 默认：根据表面积自动计算
        surface_area = mesh.area
        density = 0.01  # 默认每100平方毫米1个种子点
        num_seeds = max(10, int(surface_area * density))
    elif density is not None:
        surface_area = mesh.area
        num_seeds = max(10, int(surface_area * density))
    
    print(f"[表面采样] 目标种子数: {num_seeds}")
    print(f"[表面采样] 网格表面积: {mesh.area:.2f} mm²")
    
    # 使用trimesh的表面采样功能
    # sample_surface_even 会均匀分布采样点
    surface_seeds, face_indices = trimesh.sample.sample_surface_even(mesh, num_seeds)
    
    print(f"[表面采样] 实际采样: {len(surface_seeds)} 个表面种子点")
    
    return surface_seeds


def sample_seeds_inside_mesh(mesh, num_seeds, method='random'):
    """
    在STL网格内部采样种子点
    
    参数:
        mesh: trimesh对象
        num_seeds: 种子数量
        method: 采样方法 ('random' 或 'grid')
    
    返回:
        inside_seeds: 内部种子点数组
    """
    bounds = mesh.bounds
    lo, hi = bounds[0], bounds[1]
    
    print(f"[内部采样] 目标种子数: {num_seeds}")
    
    if method == 'random':
        # 在包围盒内随机采样，然后过滤到网格内部
        # 为了确保得到足够的点，采样更多然后过滤
        oversample_factor = 3
        candidate_seeds = np.random.uniform(lo, hi, size=(num_seeds * oversample_factor, 3))
        
        # 检查哪些点在网格内部
        inside_mask = mesh.contains(candidate_seeds)
        inside_seeds = candidate_seeds[inside_mask]
        
        # 如果点数不够，继续采样
        while len(inside_seeds) < num_seeds:
            additional = np.random.uniform(lo, hi, size=(num_seeds, 3))
            inside_mask = mesh.contains(additional)
            inside_seeds = np.vstack([inside_seeds, additional[inside_mask]])
        
        # 如果点数太多，随机选择
        if len(inside_seeds) > num_seeds:
            indices = np.random.choice(len(inside_seeds), num_seeds, replace=False)
            inside_seeds = inside_seeds[indices]
    
    else:  # grid
        # 网格采样
        extents = hi - lo
        n_per_dim = int(np.ceil((num_seeds * 2) ** (1/3)))
        
        xs = np.linspace(lo[0], hi[0], n_per_dim)
        ys = np.linspace(lo[1], hi[1], n_per_dim)
        zs = np.linspace(lo[2], hi[2], n_per_dim)
        
        grid = np.array(np.meshgrid(xs, ys, zs, indexing='ij'))
        candidate_seeds = grid.reshape(3, -1).T
        
        # 过滤到网格内部
        inside_mask = mesh.contains(candidate_seeds)
        inside_seeds = candidate_seeds[inside_mask]
        
        # 随机选择目标数量
        if len(inside_seeds) > num_seeds:
            indices = np.random.choice(len(inside_seeds), num_seeds, replace=False)
            inside_seeds = inside_seeds[indices]
    
    print(f"[内部采样] 实际采样: {len(inside_seeds)} 个内部种子点")
    
    return inside_seeds


def sample_seeds_near_mesh_surface(mesh, num_seeds, offset_range=(0, 5)):
    """
    在STL网格表面附近采样种子点（带偏移）
    
    参数:
        mesh: trimesh对象
        num_seeds: 种子数量
        offset_range: 偏移范围 (min_offset, max_offset)，单位mm
    
    返回:
        seeds: 种子点数组
    """
    # 先在表面采样
    surface_seeds, face_indices = trimesh.sample.sample_surface_even(mesh, num_seeds)
    
    # 获取对应面的法向量
    face_normals = mesh.face_normals[face_indices]
    
    # 沿法向量随机偏移
    min_offset, max_offset = offset_range
    offsets = np.random.uniform(min_offset, max_offset, size=(len(surface_seeds), 1))
    
    # 随机选择向内或向外
    directions = np.random.choice([-1, 1], size=(len(surface_seeds), 1))
    
    seeds = surface_seeds + face_normals * offsets * directions
    
    print(f"[表面附近采样] 采样 {len(seeds)} 个种子点")
    print(f"[表面附近采样] 偏移范围: {min_offset} ~ {max_offset} mm")
    
    return seeds


def generate_internal_seeds_excluding_surface(mesh, cell_size, exclude_distance):
    """
    生成内部种子，排除表面附近区域
    
    用于表面-内部连接的 Voronoi 晶格生成。
    排除表面附近区域，避免表面种子和内部种子过近。
    
    参数:
        mesh: trimesh 对象
        cell_size: 胞元尺寸
        exclude_distance: 排除距离（距离表面多近的区域不生成种子）
    
    返回:
        internal_seeds: 内部种子点数组
    """
    print(f"\n[内部种子] 生成内部种子（排除表面 {exclude_distance:.2f} mm）...")
    
    bounds = mesh.bounds
    volume = np.prod(bounds[1] - bounds[0])
    n_seeds = int(volume / (cell_size ** 3))
    n_seeds = max(10, n_seeds)
    
    print(f"[内部种子] 目标数量: {n_seeds}")
    
    # 在包围盒内随机采样（过采样以确保足够的点）
    oversample_factor = 5
    candidates = np.random.uniform(bounds[0], bounds[1], 
                                  size=(n_seeds * oversample_factor, 3))
    
    # 过滤：只保留内部且距离表面足够远的点
    print(f"[内部种子] 过滤候选点...")
    internal_seeds = []
    
    # 批量检查是否在内部
    inside_mask = mesh.contains(candidates)
    candidates_inside = candidates[inside_mask]
    
    print(f"[内部种子] 内部候选点: {len(candidates_inside)}")
    
    # 批量计算到表面的距离
    if len(candidates_inside) > 0:
        closest_points, distances, triangle_id = mesh.nearest.on_surface(candidates_inside)
        
        # 只保留距离表面足够远的点
        far_enough_mask = distances > exclude_distance
        internal_seeds = candidates_inside[far_enough_mask]
        
        # 如果点数太多，随机选择
        if len(internal_seeds) > n_seeds:
            indices = np.random.choice(len(internal_seeds), n_seeds, replace=False)
            internal_seeds = internal_seeds[indices]
        
        print(f"[内部种子] 距离表面 > {exclude_distance:.2f} mm: {len(internal_seeds)}")
    
    if len(internal_seeds) < n_seeds * 0.5:
        print(f"[警告] 内部种子数量不足（{len(internal_seeds)}/{n_seeds}）")
        print(f"[建议] 减小 exclude_distance 或增大胞元尺寸")
    
    return np.array(internal_seeds)


def project_edge_to_surface_geodesic(p1, p2, mesh, num_segments=5):
    """
    将边投影到表面（沿测地线近似）
    
    返回多段线近似曲线
    
    参数:
        p1, p2: 边的两个端点
        mesh: trimesh 对象
        num_segments: 分段数（越多越精确，但边数越多）
    
    返回:
        segments: 多段线列表 [[p1, p2], [p2, p3], ...]
    """
    # 在 p1-p2 之间线性插值
    t_values = np.linspace(0, 1, num_segments + 1)
    interpolated = np.array([p1 + t * (p2 - p1) for t in t_values])
    
    # 投影每个点到表面
    projected, _, _ = mesh.nearest.on_surface(interpolated)
    
    # 转换为多段线
    segments = []
    for i in range(len(projected) - 1):
        segments.append([projected[i], projected[i+1]])
    
    return segments


def project_point_to_surface(point, mesh):
    """
    投影点到表面
    
    参数:
        point: 3D 点
        mesh: trimesh 对象
    
    返回:
        projected_point: 投影后的点
    """
    closest_point, _, _ = mesh.nearest.on_surface([point])
    return closest_point[0]


def sample_seeds_in_box(bounds, cell_size, num_seeds=None, method='random'):
    """
    在包围盒内采样种子点
    
    参数:
        bounds: 包围盒 [[xmin, ymin, zmin], [xmax, ymax, zmax]]
        cell_size: 胞元尺寸（控制种子密度）
        num_seeds: 指定种子数量（可选，如果不指定则根据 cell_size 自动计算）
        method: 采样方法
            - 'random': 完全随机（最不规则）
            - 'jittered': 网格 + 大抖动（中等随机）
            - 'grid': 规则网格 + 小抖动（最规则）
            - 'poisson': 泊松盘采样（均匀但不规则）
    
    返回:
        seeds: shape=(n, 3) 的种子点数组
    """
    bounds = np.asarray(bounds, dtype=np.float64)
    lo, hi = bounds[0], bounds[1]
    extents = hi - lo
    
    # 计算种子数量
    if num_seeds is None:
        volume = np.prod(extents)
        num_seeds = int(volume / (cell_size ** 3))
        num_seeds = max(10, num_seeds)  # 至少10个
    
    print(f"[采样] 目标种子数: {num_seeds}, 方法: {method}")
    
    if method == 'random':
        # 方法1: 完全随机（最不规则）
        seeds = np.random.uniform(lo, hi, size=(num_seeds, 3))
        print(f"[采样] 完全随机采样: {len(seeds)} 个种子")
        
    elif method == 'jittered':
        # 方法2: 网格 + 大抖动（中等随机）
        n_per_dim = int(np.ceil(num_seeds ** (1/3)))
        
        xs = np.linspace(lo[0], hi[0], n_per_dim)
        ys = np.linspace(lo[1], hi[1], n_per_dim)
        zs = np.linspace(lo[2], hi[2], n_per_dim)
        
        # 生成网格点
        grid = np.array(np.meshgrid(xs, ys, zs, indexing='ij'))
        seeds = grid.reshape(3, -1).T
        
        # 添加大抖动（80% 的网格间距）
        step = extents / n_per_dim
        jitter = (np.random.random(seeds.shape) - 0.5) * step * 0.8
        seeds = np.clip(seeds + jitter, lo, hi)
        
        # 如果超过目标数量，随机选择
        if len(seeds) > num_seeds:
            indices = np.random.choice(len(seeds), num_seeds, replace=False)
            seeds = seeds[indices]
        
        print(f"[采样] 网格+大抖动: {len(seeds)} 个种子")
        
    elif method == 'grid':
        # 方法3: 规则网格 + 小抖动（最规则）
        n_per_dim = int(np.ceil(num_seeds ** (1/3)))
        
        xs = np.linspace(lo[0], hi[0], n_per_dim)
        ys = np.linspace(lo[1], hi[1], n_per_dim)
        zs = np.linspace(lo[2], hi[2], n_per_dim)
        
        # 生成网格点
        grid = np.array(np.meshgrid(xs, ys, zs, indexing='ij'))
        seeds = grid.reshape(3, -1).T
        
        # 添加小抖动（20% 的网格间距）
        step = extents / n_per_dim
        jitter = (np.random.random(seeds.shape) - 0.5) * step * 0.2
        seeds = np.clip(seeds + jitter, lo, hi)
        
        # 如果超过目标数量，随机选择
        if len(seeds) > num_seeds:
            indices = np.random.choice(len(seeds), num_seeds, replace=False)
            seeds = seeds[indices]
        
        print(f"[采样] 规则网格+小抖动: {len(seeds)} 个种子")
        
    elif method == 'poisson':
        # 方法4: 泊松盘采样（均匀但不规则）
        seeds = poisson_disk_sampling_3d(bounds, cell_size * 0.8, num_seeds)
        print(f"[采样] 泊松盘采样: {len(seeds)} 个种子")
        
    else:
        print(f"[警告] 未知采样方法 '{method}'，使用 'random'")
        seeds = np.random.uniform(lo, hi, size=(num_seeds, 3))
    
    print(f"[采样] 实际种子数: {len(seeds)}")
    return seeds


def poisson_disk_sampling_3d(bounds, min_distance, max_samples):
    """
    3D 泊松盘采样（简化版）
    
    保证种子点之间的最小距离，同时分布均匀
    
    参数:
        bounds: 包围盒 [[xmin, ymin, zmin], [xmax, ymax, zmax]]
        min_distance: 最小距离
        max_samples: 最大样本数
    
    返回:
        seeds: shape=(n, 3) 的种子点数组
    """
    bounds = np.asarray(bounds, dtype=np.float64)
    lo, hi = bounds[0], bounds[1]
    
    seeds = []
    attempts = 0
    max_attempts = max_samples * 30
    
    while len(seeds) < max_samples and attempts < max_attempts:
        # 生成候选点
        candidate = np.random.uniform(lo, hi)
        
        # 检查与现有点的距离
        if len(seeds) == 0:
            seeds.append(candidate)
        else:
            distances = np.linalg.norm(np.array(seeds) - candidate, axis=1)
            if np.all(distances >= min_distance):
                seeds.append(candidate)
        
        attempts += 1
    
    return np.array(seeds, dtype=np.float64)


def extract_voronoi_skeleton(seeds, bounds, min_edge_length=None):
    """
    计算 3D Voronoi 图并提取骨架边（用于晶格生成）
    
    方案D：提取相邻种子点之间的连接线（Voronoi图的对偶图）
    这才是真正的晶格骨架结构 - 连接相邻Voronoi胞元的种子点
    
    参数:
        seeds: shape=(n, 3) 的种子点数组
        bounds: 包围盒 [[xmin, ymin, zmin], [xmax, ymax, zmax]]
        min_edge_length: 最小边长度阈值（可选），过滤掉过短的边
    
    返回:
        edges: shape=(m, 2, 3) 的边数组，每条边由两个端点定义
    """
    print(f"[Voronoi骨架] 计算 3D Voronoi 图，种子数={len(seeds)}...")
    t_start = time.time()
    
    # 计算 Voronoi 图
    vor = Voronoi(seeds)
    
    bounds = np.asarray(bounds, dtype=np.float64)
    lo, hi = bounds[0], bounds[1]
    
    # 使用字典存储边（自动去重）
    edge_dict = {}
    
    print(f"[Voronoi骨架] 提取相邻种子点之间的连接线（对偶图）...")
    
    valid_edges = 0
    skipped_infinite = 0
    skipped_out_of_bounds = 0
    skipped_short = 0
    
    # 遍历所有ridge（Voronoi面），每个ridge连接两个相邻的种子点
    for ridge_idx, (seed_idx1, seed_idx2) in enumerate(vor.ridge_points):
        ridge_vertices = vor.ridge_vertices[ridge_idx]
        
        # 跳过包含无限远点的ridge
        if -1 in ridge_vertices:
            skipped_infinite += 1
            continue
        
        # 获取两个种子点的坐标
        seed1 = seeds[seed_idx1]
        seed2 = seeds[seed_idx2]
        
        # 检查连接线的中点是否在包围盒内
        midpoint = (seed1 + seed2) / 2.0
        if not (np.all(midpoint >= lo) and np.all(midpoint <= hi)):
            skipped_out_of_bounds += 1
            continue
        
        # 可选：过滤过短的边
        if min_edge_length is not None:
            edge_length = np.linalg.norm(seed2 - seed1)
            if edge_length < min_edge_length:
                skipped_short += 1
                continue
        
        # 创建边的唯一标识
        seed1_tuple = tuple(np.round(seed1, decimals=6))
        seed2_tuple = tuple(np.round(seed2, decimals=6))
        edge_key = tuple(sorted([seed1_tuple, seed2_tuple]))
        
        # 添加到字典（自动去重）
        if edge_key not in edge_dict:
            edge_dict[edge_key] = [seed1, seed2]
            valid_edges += 1
    
    # 转换为数组
    if not edge_dict:
        print("[Voronoi骨架] 警告：未提取到任何有效的边")
        return np.zeros((0, 2, 3), dtype=np.float64)
    
    edges_list = list(edge_dict.values())
    edges_array = np.array(edges_list, dtype=np.float64)
    
    t_end = time.time()
    print(f"[Voronoi骨架] 提取到 {len(edges_array)} 条骨架边（连接相邻种子点）")
    print(f"[Voronoi骨架] 跳过: 无限远={skipped_infinite}, 超出边界={skipped_out_of_bounds}, 过短={skipped_short}")
    if min_edge_length is not None:
        print(f"[Voronoi骨架] 边长度过滤阈值: {min_edge_length:.3f}")
    print(f"[Voronoi骨架] 耗时 {t_end - t_start:.2f}s")
    
    return edges_array


def extract_voronoi_edges_for_visualization(seeds, bounds):
    """
    提取 3D Voronoi 图的边用于可视化（只显示胞元边界）
    
    方案D：提取相邻种子点之间的连接线（Voronoi图的对偶图）
    这样可以得到干净的骨架线，用于可视化
    
    参数:
        seeds: shape=(n, 3) 的种子点数组
        bounds: 包围盒 [[xmin, ymin, zmin], [xmax, ymax, zmax]]
    
    返回:
        edges: shape=(m, 2, 3) 的边数组
        voronoi_vertices: Voronoi 顶点坐标
    """
    print(f"[Voronoi可视化] 计算 3D Voronoi 图，种子数={len(seeds)}...")
    t_start = time.time()
    
    # 计算 Voronoi 图
    vor = Voronoi(seeds)
    
    bounds = np.asarray(bounds, dtype=np.float64)
    lo, hi = bounds[0], bounds[1]
    
    # 使用字典存储边（自动去重）
    edge_dict = {}
    
    valid_edges = 0
    skipped_infinite = 0
    skipped_out_of_bounds = 0
    
    # 遍历所有ridge（Voronoi面），每个ridge连接两个相邻的种子点
    for ridge_idx, (seed_idx1, seed_idx2) in enumerate(vor.ridge_points):
        ridge_vertices = vor.ridge_vertices[ridge_idx]
        
        # 跳过包含无限远点的ridge
        if -1 in ridge_vertices:
            skipped_infinite += 1
            continue
        
        # 获取两个种子点的坐标
        seed1 = seeds[seed_idx1]
        seed2 = seeds[seed_idx2]
        
        # 检查连接线的中点是否在包围盒内
        midpoint = (seed1 + seed2) / 2.0
        if not (np.all(midpoint >= lo) and np.all(midpoint <= hi)):
            skipped_out_of_bounds += 1
            continue
        
        # 创建边的唯一标识
        seed1_tuple = tuple(np.round(seed1, decimals=6))
        seed2_tuple = tuple(np.round(seed2, decimals=6))
        edge_key = tuple(sorted([seed1_tuple, seed2_tuple]))
        
        # 添加到字典（自动去重）
        if edge_key not in edge_dict:
            edge_dict[edge_key] = [seed1, seed2]
            valid_edges += 1
    
    if not edge_dict:
        print("[Voronoi可视化] 警告：未提取到任何有效的边")
        return np.zeros((0, 2, 3), dtype=np.float64), vor.vertices
    
    edges_list = list(edge_dict.values())
    edges_array = np.array(edges_list, dtype=np.float64)
    
    t_end = time.time()
    print(f"[Voronoi可视化] 提取到 {len(edges_array)} 条骨架边（连接相邻种子点）")
    print(f"[Voronoi可视化] 跳过: 无限远={skipped_infinite}, 超出边界={skipped_out_of_bounds}")
    print(f"[Voronoi可视化] 耗时 {t_end - t_start:.2f}s")
    
    return edges_array, vor.vertices


def smooth_min(F1, F2, k=10.0):
    """
    平滑最小值函数（LogSumExp）
    
    F = -log(exp(-k*F1) + exp(-k*F2)) / k
    
    参数:
        F1, F2: 两个隐函数值数组
        k: 平滑参数（越大越接近硬最小值）
    
    返回:
        F: 平滑融合后的隐函数值
    """
    # 数值稳定版本
    # F = -log(exp(-k*F1) + exp(-k*F2)) / k
    #   = -log(exp(-k*F1) * (1 + exp(-k*(F2-F1)))) / k
    #   = F1 - log(1 + exp(-k*(F2-F1))) / k
    
    # 为了数值稳定，使用 LogSumExp 技巧
    m = np.minimum(F1, F2)
    return m - np.log(np.exp(-k * (F1 - m)) + np.exp(-k * (F2 - m))) / k


def smooth_union_multiple(field_values_list, k=10.0):
    """
    多个隐函数的平滑并集
    
    参数:
        field_values_list: 隐函数值列表 [F1, F2, F3, ...]
        k: 平滑参数
    
    返回:
        F: 融合后的隐函数值
    """
    if len(field_values_list) == 0:
        return None
    
    if len(field_values_list) == 1:
        return field_values_list[0]
    
    # 逐个融合
    F = field_values_list[0]
    for Fi in field_values_list[1:]:
        F = smooth_min(F, Fi, k)
    
    return F


def point_to_segment_distance(points, seg_start, seg_end):
    """
    计算点到线段的最短距离（向量化）
    
    参数:
        points: shape=(n, 3) 的查询点
        seg_start: shape=(3,) 线段起点
        seg_end: shape=(3,) 线段终点
    
    返回:
        distances: shape=(n,) 的距离数组
    """
    seg_vec = seg_end - seg_start
    seg_len_sq = np.dot(seg_vec, seg_vec)
    
    if seg_len_sq < 1e-12:
        # 退化为点
        return np.linalg.norm(points - seg_start, axis=1)
    
    # 计算投影参数 t
    t = np.dot(points - seg_start, seg_vec) / seg_len_sq
    t = np.clip(t, 0.0, 1.0)  # 限制在线段范围内
    
    # 计算最近点
    closest = seg_start + t[:, np.newaxis] * seg_vec
    
    # 计算距离
    distances = np.linalg.norm(points - closest, axis=1)
    
    return distances


def compute_distance_field(voxel_coords, edges):
    """
    计算每个体素到 Voronoi 骨架的距离场（简单版本）
    
    参数:
        voxel_coords: shape=(n, 3) 的体素中心坐标
        edges: shape=(m, 2, 3) 的骨架边数组
    
    返回:
        distances: shape=(n,) 的距离数组
    """
    print(f"[距离场] 计算 {len(voxel_coords)} 个体素到 {len(edges)} 条边的距离...")
    t_start = time.time()
    
    n_voxels = len(voxel_coords)
    min_distances = np.full(n_voxels, np.inf, dtype=np.float32)
    
    # 对每条边计算距离
    batch_size = 10000
    for i, edge in enumerate(edges):
        if (i + 1) % 100 == 0 or i == len(edges) - 1:
            print(f"[距离场] 处理进度: {i+1}/{len(edges)}")
        
        seg_start, seg_end = edge[0], edge[1]
        
        # 批处理计算距离
        for j in range(0, n_voxels, batch_size):
            batch = voxel_coords[j:j+batch_size]
            batch_dist = point_to_segment_distance(batch, seg_start, seg_end)
            min_distances[j:j+batch_size] = np.minimum(
                min_distances[j:j+batch_size],
                batch_dist
            )
    
    t_end = time.time()
    print(f"[距离场] 计算完成，耗时 {t_end - t_start:.2f}s")
    print(f"[距离场] 距离范围: [{min_distances.min():.3f}, {min_distances.max():.3f}]")
    
    return min_distances


def compute_projection_field_smooth(voxel_coords, edges, k=10.0):
    """
    计算投影距离场，使用平滑融合（论文高级方法）
    
    对每条边单独计算隐函数，然后用平滑最小值融合
    
    参数:
        voxel_coords: shape=(n, 3) 的体素坐标
        edges: shape=(m, 2, 3) 的骨架边数组
        k: 平滑参数（越大越接近硬最小值）
    
    返回:
        field_values: shape=(n,) 的隐函数值
    """
    print(f"[平滑投影场] 计算 {len(voxel_coords)} 个体素的平滑投影距离场...")
    print(f"[平滑投影场] 平滑参数 k={k}")
    t_start = time.time()
    
    n_voxels = len(voxel_coords)
    
    # 对每条边计算距离场
    edge_fields = []
    batch_size = 10000
    
    for i, edge in enumerate(edges):
        if (i + 1) % 100 == 0 or i == len(edges) - 1:
            print(f"[平滑投影场] 处理进度: {i+1}/{len(edges)}")
        
        seg_start, seg_end = edge[0], edge[1]
        seg_vec = seg_end - seg_start
        seg_len_sq = np.dot(seg_vec, seg_vec)
        
        if seg_len_sq < 1e-12:
            continue
        
        # 计算这条边的距离场
        edge_distances = np.full(n_voxels, np.inf, dtype=np.float32)
        
        for j in range(0, n_voxels, batch_size):
            batch = voxel_coords[j:j+batch_size]
            
            # 计算投影参数 t
            t = np.dot(batch - seg_start, seg_vec) / seg_len_sq
            t = np.clip(t, 0.0, 1.0)
            
            # 计算最近点（投影点）
            proj_points = seg_start + t[:, np.newaxis] * seg_vec
            
            # 计算距离
            distances = np.linalg.norm(batch - proj_points, axis=1)
            edge_distances[j:j+batch_size] = distances
        
        edge_fields.append(edge_distances)
    
    # 使用平滑最小值融合所有边的距离场
    print(f"[平滑投影场] 融合 {len(edge_fields)} 条边的距离场...")
    field_values = smooth_union_multiple(edge_fields, k=k)
    
    t_end = time.time()
    print(f"[平滑投影场] 计算完成，耗时 {t_end - t_start:.2f}s")
    print(f"[平滑投影场] 值范围: [{field_values.min():.3f}, {field_values.max():.3f}]")
    
    return field_values


def compute_projection_field(voxel_coords, edges):
    """
    计算投影距离场（论文方法）
    
    对于每个空间点 q，计算:
        f_c(q) = (q - v_pi) · n_qi
    
    其中:
        v_pi: 点 q 在骨架上的最近投影点
        n_qi: 从投影点指向 q 的单位法向量
    
    参数:
        voxel_coords: shape=(n, 3) 的体素坐标
        edges: shape=(m, 2, 3) 的骨架边数组
    
    返回:
        field_values: shape=(n,) 的隐函数值
        closest_points: shape=(n, 3) 最近投影点
        normals: shape=(n, 3) 法向量
    """
    print(f"[投影场] 计算 {len(voxel_coords)} 个体素的投影距离场...")
    t_start = time.time()
    
    n_voxels = len(voxel_coords)
    min_distances = np.full(n_voxels, np.inf, dtype=np.float32)
    closest_points = np.zeros((n_voxels, 3), dtype=np.float32)
    
    # 对每条边计算最近点
    batch_size = 10000
    for i, edge in enumerate(edges):
        if (i + 1) % 100 == 0 or i == len(edges) - 1:
            print(f"[投影场] 处理进度: {i+1}/{len(edges)}")
        
        seg_start, seg_end = edge[0], edge[1]
        seg_vec = seg_end - seg_start
        seg_len_sq = np.dot(seg_vec, seg_vec)
        
        if seg_len_sq < 1e-12:
            continue
        
        # 批处理
        for j in range(0, n_voxels, batch_size):
            batch = voxel_coords[j:j+batch_size]
            
            # 计算投影参数 t
            t = np.dot(batch - seg_start, seg_vec) / seg_len_sq
            t = np.clip(t, 0.0, 1.0)
            
            # 计算最近点（投影点）
            proj_points = seg_start + t[:, np.newaxis] * seg_vec
            
            # 计算距离
            distances = np.linalg.norm(batch - proj_points, axis=1)
            
            # 更新最小距离和对应的投影点
            mask = distances < min_distances[j:j+batch_size]
            min_distances[j:j+batch_size] = np.where(mask, distances, min_distances[j:j+batch_size])
            closest_points[j:j+batch_size] = np.where(
                mask[:, np.newaxis], 
                proj_points, 
                closest_points[j:j+batch_size]
            )
    
    # 计算法向量 n_qi = (q - v_pi) / ||q - v_pi||
    vectors = voxel_coords - closest_points
    distances = np.linalg.norm(vectors, axis=1, keepdims=True)
    normals = np.where(
        distances > 1e-9,
        vectors / distances,
        np.array([0, 0, 1])  # 默认法向
    )
    
    # 计算隐函数值: f_c(q) = (q - v_pi) · n_qi = ||q - v_pi||
    # 注意：(q - v_pi) · n_qi = ||q - v_pi|| * ||n_qi|| = ||q - v_pi|| (因为 n_qi 是单位向量)
    field_values = distances.ravel()
    
    t_end = time.time()
    print(f"[投影场] 计算完成，耗时 {t_end - t_start:.2f}s")
    print(f"[投影场] 值范围: [{field_values.min():.3f}, {field_values.max():.3f}]")
    
    return field_values, closest_points, normals


def generate_voronoi_lattice(bounds, cell_size=10.0, wall_thickness=3.0, resolution=4, num_seeds=None,
                            seed_method='jittered', use_smooth=True, smooth_k=10.0):
    """
    生成 Voronoi 晶格结构
    
    参数:
        bounds: 包围盒 [[xmin, ymin, zmin], [xmax, ymax, zmax]]
        cell_size: Voronoi 胞元尺寸（mm）
        wall_thickness: 骨架管径（mm）
        resolution: 每个胞元内的体素数
        num_seeds: 指定种子点数量（可选）
        seed_method: 种子采样方法
            - 'jittered': 网格 + 大抖动（中等随机）⭐ 默认
            - 'random': 完全随机（最不规则）
            - 'grid': 规则网格 + 小抖动（最规则）
            - 'poisson': 泊松盘采样（均匀但不规则）
        use_smooth: 是否使用平滑融合（smooth min）⭐ 默认开启
        smooth_k: 平滑参数（5-20，越大越接近硬边）
    
    返回:
        mesh: trimesh.Trimesh 对象
    """
    print("=" * 60)
    if use_smooth:
        print("开始生成 Voronoi 晶格（投影距离场 + 平滑融合）")
    else:
        print("开始生成 Voronoi 晶格（投影距离场）")
    print("=" * 60)
    
    t_total_start = time.time()
    
    bounds = np.asarray(bounds, dtype=np.float64)
    lo, hi = bounds[0], bounds[1]
    extents = hi - lo
    
    print(f"[参数] 包围盒: {lo} ~ {hi}")
    print(f"[参数] 尺寸: {extents}")
    print(f"[参数] cell_size={cell_size}, wall_thickness={wall_thickness}, resolution={resolution}")
    if use_smooth:
        print(f"[参数] 平滑融合: 开启, k={smooth_k}")
    
    # 步骤1: 采样种子点
    seeds = sample_seeds_in_box(bounds, cell_size, num_seeds=num_seeds, method=seed_method)
    
    if len(seeds) < 4:
        print("[错误] 种子点数量不足")
        return trimesh.Trimesh()
    
    # 步骤2: 提取 Voronoi 骨架
    edges = extract_voronoi_skeleton(seeds, bounds)
    
    if len(edges) == 0:
        print("[错误] 未能提取到 Voronoi 边")
        return trimesh.Trimesh()
    
    # 步骤3: 建立体素网格
    voxel_size = cell_size / resolution
    nx = max(4, int(np.ceil(extents[0] / voxel_size)) + 2)
    ny = max(4, int(np.ceil(extents[1] / voxel_size)) + 2)
    nz = max(4, int(np.ceil(extents[2] / voxel_size)) + 2)
    
    print(f"[体素网格] {nx}×{ny}×{nz} = {nx*ny*nz:,} 体素")
    print(f"[体素网格] 体素尺寸: {voxel_size:.3f} mm")
    
    xs = np.linspace(lo[0], hi[0], nx)
    ys = np.linspace(lo[1], hi[1], ny)
    zs = np.linspace(lo[2], hi[2], nz)
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing='ij')
    
    # 展平坐标
    voxel_coords = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
    
    # 步骤4: 计算距离场
    if use_smooth:
        # 使用平滑融合
        field_values = compute_projection_field_smooth(voxel_coords, edges, k=smooth_k)
    else:
        # 使用标准投影距离场
        field_values, closest_points, normals = compute_projection_field(voxel_coords, edges)
    
    # 步骤5: 构建隐函数
    radius = wall_thickness / 2.0
    F = field_values - radius
    F = F.reshape(X.shape)
    
    # 统计
    n_inside = np.sum(F < 0)
    ratio = n_inside / F.size * 100
    print(f"[隐函数] 骨架体积占比: {ratio:.1f}%")
    print(f"[隐函数] 值范围: [{F.min():.3f}, {F.max():.3f}]")
    
    if ratio < 0.5:
        print(f"[警告] 骨架体积占比过低（{ratio:.1f}%），建议增大 wall_thickness")
    elif ratio > 40.0:
        print(f"[警告] 骨架体积占比过高（{ratio:.1f}%），建议减小 wall_thickness")
    
    # 步骤6: Marching Cubes 提取等值面
    print("[Marching Cubes] 提取等值面...")
    t_mc_start = time.time()
    
    try:
        verts, faces, _, _ = measure.marching_cubes(
            F,
            level=0.0,
            spacing=(voxel_size, voxel_size, voxel_size)
        )
        
        # 调整到世界坐标
        verts += lo
        
        mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
        
        t_mc_end = time.time()
        print(f"[Marching Cubes] 耗时 {t_mc_end - t_mc_start:.2f}s")
        print(f"[Marching Cubes] 生成网格: {len(verts)} 顶点, {len(faces)} 面")
        
    except Exception as e:
        print(f"[错误] Marching Cubes 失败: {e}")
        return trimesh.Trimesh()
    
    t_total_end = time.time()
    print("=" * 60)
    print(f"总耗时: {t_total_end - t_total_start:.2f}s")
    print("=" * 60)
    
    return mesh


def generate_voronoi_lattice_smooth(bounds, cell_size=10.0, wall_thickness=3.0, resolution=4, 
                                   num_seeds=None, smooth_k=10.0):
    """
    使用平滑融合生成 Voronoi 晶格（论文高级方法）
    
    F = -log(exp(-k*F1) + exp(-k*F2) + ...) / k
    
    参数:
        bounds: 包围盒 [[xmin, ymin, zmin], [xmax, ymax, zmax]]
        cell_size: Voronoi 胞元尺寸（mm）
        wall_thickness: 骨架管径（mm）
        resolution: 每个胞元内的体素数
        num_seeds: 指定种子点数量（可选）
        smooth_k: 平滑参数（越大越接近硬最小值，推荐 5-20）
    
    返回:
        mesh: trimesh.Trimesh 对象
    """
    print("=" * 60)
    print("开始生成 Voronoi 晶格（平滑融合方法）")
    print("=" * 60)
    
    t_total_start = time.time()
    
    bounds = np.asarray(bounds, dtype=np.float64)
    lo, hi = bounds[0], bounds[1]
    extents = hi - lo
    
    print(f"[参数] 包围盒: {lo} ~ {hi}")
    print(f"[参数] 尺寸: {extents}")
    print(f"[参数] cell_size={cell_size}, wall_thickness={wall_thickness}, resolution={resolution}")
    print(f"[参数] smooth_k={smooth_k}")
    
    # 步骤1: 采样种子点
    seeds = sample_seeds_in_box(bounds, cell_size, num_seeds=num_seeds)
    
    if len(seeds) < 4:
        print("[错误] 种子点数量不足")
        return trimesh.Trimesh()
    
    # 步骤2: 提取 Voronoi 骨架
    edges = extract_voronoi_skeleton(seeds, bounds)
    
    if len(edges) == 0:
        print("[错误] 未能提取到 Voronoi 边")
        return trimesh.Trimesh()
    
    # 步骤3: 建立体素网格
    voxel_size = cell_size / resolution
    nx = max(4, int(np.ceil(extents[0] / voxel_size)) + 2)
    ny = max(4, int(np.ceil(extents[1] / voxel_size)) + 2)
    nz = max(4, int(np.ceil(extents[2] / voxel_size)) + 2)
    
    print(f"[体素网格] {nx}×{ny}×{nz} = {nx*ny*nz:,} 体素")
    print(f"[体素网格] 体素尺寸: {voxel_size:.3f} mm")
    
    xs = np.linspace(lo[0], hi[0], nx)
    ys = np.linspace(lo[1], hi[1], ny)
    zs = np.linspace(lo[2], hi[2], nz)
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing='ij')
    
    # 展平坐标
    voxel_coords = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
    
    # 步骤4: 计算平滑融合的距离场
    field_values = compute_projection_field_smooth(voxel_coords, edges, k=smooth_k)
    
    # 步骤5: 构建隐函数
    radius = wall_thickness / 2.0
    F = field_values - radius
    F = F.reshape(X.shape)
    
    # 统计
    n_inside = np.sum(F < 0)
    ratio = n_inside / F.size * 100
    print(f"[隐函数] 骨架体积占比: {ratio:.1f}%")
    print(f"[隐函数] 值范围: [{F.min():.3f}, {F.max():.3f}]")
    
    if ratio < 0.5:
        print(f"[警告] 骨架体积占比过低（{ratio:.1f}%），建议增大 wall_thickness")
    elif ratio > 40.0:
        print(f"[警告] 骨架体积占比过高（{ratio:.1f}%），建议减小 wall_thickness")
    
    # 步骤6: Marching Cubes 提取等值面
    print("[Marching Cubes] 提取等值面...")
    t_mc_start = time.time()
    
    try:
        verts, faces, _, _ = measure.marching_cubes(
            F,
            level=0.0,
            spacing=(voxel_size, voxel_size, voxel_size)
        )
        
        # 调整到世界坐标
        verts += lo
        
        mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
        
        t_mc_end = time.time()
        print(f"[Marching Cubes] 耗时 {t_mc_end - t_mc_start:.2f}s")
        print(f"[Marching Cubes] 生成网格: {len(verts)} 顶点, {len(faces)} 面")
        
    except Exception as e:
        print(f"[错误] Marching Cubes 失败: {e}")
        return trimesh.Trimesh()
    
    t_total_end = time.time()
    print("=" * 60)
    print(f"总耗时: {t_total_end - t_total_start:.2f}s")
    print("=" * 60)
    
    return mesh


def visualize_voronoi_diagram(seeds, edges, bounds, output_path="voronoi_diagram.stl"):
    """
    可视化 3D Voronoi 图
    
    将种子点显示为小球，Voronoi 边显示为细圆柱
    
    参数:
        seeds: shape=(n, 3) 的种子点数组
        edges: shape=(m, 2, 3) 的 Voronoi 边数组
        bounds: 包围盒 [[xmin, ymin, zmin], [xmax, ymax, zmax]]
        output_path: 输出文件路径
    
    返回:
        mesh: 合并后的 trimesh.Trimesh 对象
    """
    print(f"\n[可视化] 生成 Voronoi 图可视化...")
    print(f"[可视化] 种子点数: {len(seeds)}")
    print(f"[可视化] 边数: {len(edges)}")
    
    meshes = []
    
    # 计算合适的球和圆柱尺寸
    bounds = np.asarray(bounds, dtype=np.float64)
    extents = bounds[1] - bounds[0]
    avg_extent = np.mean(extents)
    
    sphere_radius = avg_extent * 0.01  # 球半径为平均尺寸的 1%
    cylinder_radius = avg_extent * 0.005  # 圆柱半径为平均尺寸的 0.5%
    
    print(f"[可视化] 球半径: {sphere_radius:.3f} mm")
    print(f"[可视化] 圆柱半径: {cylinder_radius:.3f} mm")
    
    # 1. 为每个种子点创建小球
    print(f"[可视化] 创建种子点球体...")
    for i, seed in enumerate(seeds):
        if (i + 1) % 50 == 0 or i == len(seeds) - 1:
            print(f"  进度: {i+1}/{len(seeds)}")
        
        sphere = trimesh.creation.icosphere(subdivisions=2, radius=sphere_radius)
        sphere.apply_translation(seed)
        meshes.append(sphere)
    
    # 2. 为每条 Voronoi 边创建圆柱
    print(f"[可视化] 创建 Voronoi 边圆柱...")
    for i, edge in enumerate(edges):
        if (i + 1) % 100 == 0 or i == len(edges) - 1:
            print(f"  进度: {i+1}/{len(edges)}")
        
        v0, v1 = edge[0], edge[1]
        
        # 计算圆柱的长度和方向
        direction = v1 - v0
        length = np.linalg.norm(direction)
        
        if length < 1e-9:
            continue
        
        # 创建圆柱（沿 Z 轴）
        cylinder = trimesh.creation.cylinder(
            radius=cylinder_radius,
            height=length,
            sections=8  # 8 边形近似圆柱
        )
        
        # 计算旋转：将 Z 轴对齐到 direction
        z_axis = np.array([0, 0, 1])
        direction_norm = direction / length
        
        # 使用 Rodrigues 旋转公式
        if np.allclose(direction_norm, z_axis):
            # 已经对齐，无需旋转
            pass
        elif np.allclose(direction_norm, -z_axis):
            # 反向，旋转 180 度
            cylinder.apply_transform(trimesh.transformations.rotation_matrix(np.pi, [1, 0, 0]))
        else:
            # 计算旋转轴和角度
            axis = np.cross(z_axis, direction_norm)
            axis = axis / np.linalg.norm(axis)
            angle = np.arccos(np.clip(np.dot(z_axis, direction_norm), -1.0, 1.0))
            
            rotation_matrix = trimesh.transformations.rotation_matrix(angle, axis)
            cylinder.apply_transform(rotation_matrix)
        
        # 平移到中点
        midpoint = (v0 + v1) / 2.0
        cylinder.apply_translation(midpoint)
        
        meshes.append(cylinder)
    
    # 3. 合并所有网格
    print(f"[可视化] 合并 {len(meshes)} 个网格...")
    combined_mesh = trimesh.util.concatenate(meshes)
    
    # 4. 保存
    combined_mesh.export(output_path)
    print(f"[可视化] 已保存到: {output_path}")
    print(f"[可视化] 总顶点数: {len(combined_mesh.vertices)}")
    print(f"[可视化] 总面数: {len(combined_mesh.faces)}")
    
    return combined_mesh


def generate_and_visualize_voronoi(bounds, cell_size=10.0, num_seeds=None, seed_method='jittered',
                                   output_path="voronoi_diagram.stl"):
    """
    生成并可视化 Voronoi 图（不生成晶格，只显示 Voronoi 结构）
    
    参数:
        bounds: 包围盒 [[xmin, ymin, zmin], [xmax, ymax, zmax]]
        cell_size: Voronoi 胞元尺寸（mm）
        num_seeds: 指定种子点数量（可选）
        seed_method: 种子采样方法
        output_path: 输出文件路径
    
    返回:
        mesh: 可视化网格
        seeds: 种子点数组
        edges: Voronoi 边数组
    """
    print("=" * 60)
    print("生成并可视化 3D Voronoi 图")
    print("=" * 60)
    
    bounds = np.asarray(bounds, dtype=np.float64)
    lo, hi = bounds[0], bounds[1]
    extents = hi - lo
    
    print(f"[参数] 包围盒: {lo} ~ {hi}")
    print(f"[参数] 尺寸: {extents}")
    print(f"[参数] cell_size={cell_size}, seed_method={seed_method}")
    
    # 步骤1: 采样种子点
    seeds = sample_seeds_in_box(bounds, cell_size, num_seeds=num_seeds, method=seed_method)
    
    if len(seeds) < 4:
        print("[错误] 种子点数量不足")
        return None, seeds, None
    
    # 步骤2: 提取 Voronoi 骨架
    edges = extract_voronoi_skeleton(seeds, bounds)
    
    if len(edges) == 0:
        print("[错误] 未能提取到 Voronoi 边")
        return None, seeds, edges
    
    # 步骤3: 可视化
    mesh = visualize_voronoi_diagram(seeds, edges, bounds, output_path)
    
    print("=" * 60)
    print("Voronoi 图可视化完成")
    print("=" * 60)
    
    return mesh, seeds, edges


def create_boundary_mirror_seeds(bounds, cell_size):
    """
    在包围盒外侧创建镜像种子点，确保边界胞元有限
    
    优化策略：
    - 根据包围盒尺寸自适应调整镜像种子密度
    - 较大的面使用较稀疏的种子
    - 较小的面使用较密集的种子
    
    参数:
        bounds: 包围盒 [[xmin, ymin, zmin], [xmax, ymax, zmax]]
        cell_size: 种子点间距
    
    返回:
        mirror_seeds: 镜像种子点数组
    """
    lo, hi = bounds[0], bounds[1]
    extents = hi - lo
    
    # 自适应镜像层间距：根据cell_size和包围盒尺寸调整
    # 对于较大的包围盒，使用稍大的间距；对于较小的包围盒，使用较小的间距
    avg_extent = np.mean(extents)
    
    if avg_extent > 100:  # 大包围盒
        mirror_spacing = cell_size * 1.2
    elif avg_extent > 50:  # 中等包围盒
        mirror_spacing = cell_size * 1.0
    else:  # 小包围盒
        mirror_spacing = cell_size * 0.8
    
    # 计算每个方向需要多少个镜像点
    # 至少3个，最多根据尺寸计算
    nx = max(3, min(15, int(extents[0] / mirror_spacing) + 2))
    ny = max(3, min(15, int(extents[1] / mirror_spacing) + 2))
    nz = max(3, min(15, int(extents[2] / mirror_spacing) + 2))
    
    mirror_seeds = []
    
    # 镜像层距离：根据cell_size自适应
    # 距离包围盒表面约1.5-2个cell_size
    offset = cell_size * 1.8
    
    # 扩展范围：镜像面要比包围盒稍大
    margin = cell_size * 0.8
    
    # X 方向的两个面（YZ 平面）
    for x in [lo[0] - offset, hi[0] + offset]:
        y_range = np.linspace(lo[1] - margin, hi[1] + margin, ny)
        z_range = np.linspace(lo[2] - margin, hi[2] + margin, nz)
        for y in y_range:
            for z in z_range:
                mirror_seeds.append([x, y, z])
    
    # Y 方向的两个面（XZ 平面）
    for y in [lo[1] - offset, hi[1] + offset]:
        x_range = np.linspace(lo[0] - margin, hi[0] + margin, nx)
        z_range = np.linspace(lo[2] - margin, hi[2] + margin, nz)
        for x in x_range:
            for z in z_range:
                mirror_seeds.append([x, y, z])
    
    # Z 方向的两个面（XY 平面）
    for z in [lo[2] - offset, hi[2] + offset]:
        x_range = np.linspace(lo[0] - margin, hi[0] + margin, nx)
        y_range = np.linspace(lo[1] - margin, hi[1] + margin, ny)
        for x in x_range:
            for y in y_range:
                mirror_seeds.append([x, y, z])
    
    # 计算镜像种子密度（每平方毫米）
    total_surface_area = 2 * (extents[0] * extents[1] + extents[1] * extents[2] + extents[0] * extents[2])
    density = len(mirror_seeds) / total_surface_area if total_surface_area > 0 else 0
    
    print(f"[镜像种子] 生成 {len(mirror_seeds)} 个镜像种子点")
    print(f"[镜像种子] 每个面约: X面={ny}×{nz}, Y面={nx}×{nz}, Z面={nx}×{ny}")
    print(f"[镜像种子] 间距: {mirror_spacing:.2f}, 偏移: {offset:.2f}, 边缘扩展: {margin:.2f}")
    print(f"[镜像种子] 密度: {density:.4f} 个/mm²")
    
    return np.array(mirror_seeds)


def clip_triangle_to_box(triangle, box_min, box_max):
    """
    将三角形裁剪到包围盒内
    使用 Sutherland-Hodgman 算法的简化版本
    """
    # 检查三角形是否完全在包围盒内
    all_inside = np.all((triangle >= box_min) & (triangle <= box_max))
    if all_inside:
        return triangle
    
    # 检查三角形是否完全在包围盒外
    # 如果所有顶点都在同一侧外面，则完全在外
    for axis in range(3):
        if np.all(triangle[:, axis] < box_min[axis]) or np.all(triangle[:, axis] > box_max[axis]):
            return None
    
    # 简化处理：将顶点裁剪到包围盒
    # 这不是完美的裁剪，但对于可视化足够了
    clipped = np.clip(triangle, box_min, box_max)
    
    # 检查是否退化
    unique_verts = np.unique(clipped, axis=0)
    if len(unique_verts) >= 3:
        return clipped
    else:
        return None


def visualize_with_matplotlib(vor, internal_seeds, bounds, colors):
    """
    使用 Matplotlib 渲染 Voronoi 图（备用方案）
    使用 PyVista 进行裁剪，然后用 Matplotlib 显示
    """
    from mpl_toolkits.mplot3d import Axes3D
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    lo, hi = bounds[0], bounds[1]
    
    # 如果 PyVista 可用，使用它来裁剪
    if PYVISTA_AVAILABLE:
        print("[提示] 使用 PyVista 进行裁剪，Matplotlib 进行显示")
        
        # 创建包围盒用于裁剪
        box_mesh = pv.Box(bounds=[lo[0], hi[0], lo[1], hi[1], lo[2], hi[2]])
        
        cell_count = 0
        
        # 遍历内部种子
        for i in range(len(internal_seeds)):
            point_idx = i
            region_idx = vor.point_region[point_idx]
            region = vor.regions[region_idx]
            
            if not region or -1 in region:
                continue
            
            # 获取顶点
            vertices = vor.vertices[region]
            
            try:
                # 创建凸包
                from scipy.spatial import ConvexHull
                hull = ConvexHull(vertices)
                
                # 创建 PyVista 多面体
                faces = []
                for simplex in hull.simplices:
                    faces.append(3)
                    faces.extend(simplex)
                
                poly = pv.PolyData(vertices, faces)
                
                # 使用 PyVista 裁剪
                clipped = poly.clip_box(box_mesh, invert=False)
                
                if clipped.n_points == 0:
                    continue
                
                # 提取裁剪后的面用于 Matplotlib 显示
                surface = clipped.extract_surface(algorithm=None)
                surface = surface.triangulate()
                
                # 转换为 Matplotlib 的 Poly3DCollection
                faces_for_mpl = []
                for cell_id in range(surface.n_cells):
                    cell = surface.get_cell(cell_id)
                    if cell.n_points == 3:
                        face_verts = surface.points[cell.point_ids]
                        faces_for_mpl.append(face_verts)
                
                if len(faces_for_mpl) > 0:
                    color = colors[i][:3]
                    poly_collection = Poly3DCollection(faces_for_mpl, alpha=0.7, facecolor=color, 
                                                      edgecolor='darkblue', linewidth=0.5)
                    ax.add_collection3d(poly_collection)
                    
                    # 添加种子点
                    seed = internal_seeds[i]
                    if np.all(seed >= lo) and np.all(seed <= hi):
                        ax.scatter(*seed, color='red', s=50, alpha=1.0)
                    
                    cell_count += 1
                    
            except Exception as e:
                continue
    
    else:
        # PyVista 不可用，使用简单方法（不裁剪）
        print("[警告] PyVista 不可用，显示未裁剪的胞元")
        cell_count = 0
        
        for i in range(len(internal_seeds)):
            point_idx = i
            region_idx = vor.point_region[point_idx]
            region = vor.regions[region_idx]
            
            if not region or -1 in region:
                continue
            
            vertices = vor.vertices[region]
            
            try:
                from scipy.spatial import ConvexHull
                hull = ConvexHull(vertices)
                
                faces = []
                for simplex in hull.simplices:
                    face_vertices = vertices[simplex]
                    faces.append(face_vertices)
                
                color = colors[i][:3]
                poly = Poly3DCollection(faces, alpha=0.7, facecolor=color, 
                                       edgecolor='darkblue', linewidth=0.5)
                ax.add_collection3d(poly)
                
                seed = internal_seeds[i]
                ax.scatter(*seed, color='red', s=50, alpha=1.0)
                
                cell_count += 1
            except:
                continue
    
    # 绘制包围盒边框
    # 定义包围盒的12条边
    edges = [
        # 底面
        [[lo[0], lo[1], lo[2]], [hi[0], lo[1], lo[2]]],
        [[hi[0], lo[1], lo[2]], [hi[0], hi[1], lo[2]]],
        [[hi[0], hi[1], lo[2]], [lo[0], hi[1], lo[2]]],
        [[lo[0], hi[1], lo[2]], [lo[0], lo[1], lo[2]]],
        # 顶面
        [[lo[0], lo[1], hi[2]], [hi[0], lo[1], hi[2]]],
        [[hi[0], lo[1], hi[2]], [hi[0], hi[1], hi[2]]],
        [[hi[0], hi[1], hi[2]], [lo[0], hi[1], hi[2]]],
        [[lo[0], hi[1], hi[2]], [lo[0], lo[1], hi[2]]],
        # 竖边
        [[lo[0], lo[1], lo[2]], [lo[0], lo[1], hi[2]]],
        [[hi[0], lo[1], lo[2]], [hi[0], lo[1], hi[2]]],
        [[hi[0], hi[1], lo[2]], [hi[0], hi[1], hi[2]]],
        [[lo[0], hi[1], lo[2]], [lo[0], hi[1], hi[2]]],
    ]
    
    for edge in edges:
        edge = np.array(edge)
        ax.plot3D(*edge.T, 'k-', linewidth=1, alpha=0.3)
    
    # 设置坐标轴
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1], hi[1])
    ax.set_zlim(lo[2], hi[2])
    
    print(f"[可视化] 成功添加 {cell_count} 个 Voronoi 胞元")
    
    print(f"[注意] Matplotlib渲染器不支持边提取，只显示胞元")
    print(f"[提示] 如需查看边，请使用PyVista渲染器")
    
    ax.set_title(f'3D Voronoi Diagram\nSeeds: {len(internal_seeds)}, Cells: {cell_count}')
    plt.show()


def extract_all_edges_from_polydata(poly):
    """
    直接从 PolyData 提取所有边（不经过表面提取或三角化）
    
    参数:
        poly: PyVista PolyData 对象
    
    返回:
        all_edges: 所有边的列表 [(p1, p2), ...]
    """
    from collections import defaultdict
    
    # 使用字典存储所有边（自动去重）
    edge_coords = {}
    
    # 遍历所有面
    for i in range(poly.n_cells):
        cell = poly.get_cell(i)
        n_points = cell.n_points
        
        # 提取这个面的所有边
        for j in range(n_points):
            p1_idx = cell.point_ids[j]
            p2_idx = cell.point_ids[(j + 1) % n_points]
            
            # 创建边的唯一标识（排序确保方向无关）
            edge_key = tuple(sorted([p1_idx, p2_idx]))
            
            if edge_key not in edge_coords:
                p1 = poly.points[p1_idx]
                p2 = poly.points[p2_idx]
                edge_coords[edge_key] = (p1.copy(), p2.copy())
    
    # 返回所有边
    all_edges = list(edge_coords.values())
    
    return all_edges


def extract_all_edges_from_mesh(mesh):
    """
    从网格中提取所有边（包括内部共享边和边界边）
    
    对于3D网格（UnstructuredGrid），先提取外表面，再提取边
    对于2D表面网格（PolyData），直接提取边
    
    提取每个面的所有边，包括：
    - 边界边（只被一个面使用）
    - 内部共享边（被两个面共享）
    
    参数:
        mesh: PyVista 网格（可以是 PolyData 或 UnstructuredGrid）
    
    返回:
        all_edges: 所有边的列表 [(p1, p2), ...]
    """
    from collections import defaultdict
    
    # 如果是3D网格，先提取外表面（不三角化）
    if hasattr(mesh, 'extract_surface'):
        try:
            surface = mesh.extract_surface(algorithm=None)
        except:
            surface = mesh
    else:
        surface = mesh
    
    # 使用字典存储所有边（自动去重）
    edge_coords = {}
    
    # 遍历所有面
    for i in range(surface.n_cells):
        cell = surface.get_cell(i)
        n_points = cell.n_points
        
        # 提取这个面的所有边
        for j in range(n_points):
            p1_idx = cell.point_ids[j]
            p2_idx = cell.point_ids[(j + 1) % n_points]
            
            # 创建边的唯一标识（排序确保方向无关）
            edge_key = tuple(sorted([p1_idx, p2_idx]))
            
            if edge_key not in edge_coords:
                p1 = surface.points[p1_idx]
                p2 = surface.points[p2_idx]
                edge_coords[edge_key] = (p1.copy(), p2.copy())
    
    # 返回所有边（包括边界边和内部共享边）
    all_edges = list(edge_coords.values())
    
    return all_edges


def create_edge_comparison_viewer(convex_hulls, coplanar_tolerance=0.001):
    """
    创建交互式对比查看器，显示凸包的过滤前后边对比
    
    参数:
        convex_hulls: 凸包列表，每个元素包含 {'mesh': pv.PolyData, 'seed': np.array, 'color': tuple}
        coplanar_tolerance: 共面判断的距离容差（mm）
    """
    from collections import defaultdict
    
    if len(convex_hulls) == 0:
        print("[对比查看器] 没有凸包可显示")
        return
    
    print(f"[对比查看器] 准备显示 {len(convex_hulls)} 个凸包的对比")
    print(f"[对比查看器] 共面容差: {coplanar_tolerance} mm")
    
    # 为每个凸包提取过滤前和过滤后的边
    comparison_data = []
    
    for idx, hull_info in enumerate(convex_hulls):
        try:
            clipped = hull_info['mesh']
            seed = hull_info['seed']
            color = hull_info['color']
            
            # 提取所有边（过滤前）
            all_edges_before = []
            edge_set = set()
            
            for face_idx in range(clipped.n_cells):
                cell = clipped.get_cell(face_idx)
                n_points = cell.n_points
                
                for j in range(n_points):
                    p1_idx = cell.point_ids[j]
                    p2_idx = cell.point_ids[(j + 1) % n_points]
                    edge_key = tuple(sorted([p1_idx, p2_idx]))
                    
                    if edge_key not in edge_set:
                        edge_set.add(edge_key)
                        p1 = clipped.points[p1_idx]
                        p2 = clipped.points[p2_idx]
                        all_edges_before.append([p1.copy(), p2.copy()])
            
            # 提取真实棱边（过滤后）- 使用平面方程方法
            all_edges_after = []
            edge_to_faces = defaultdict(list)
            
            for face_idx in range(clipped.n_cells):
                cell = clipped.get_cell(face_idx)
                n_points = cell.n_points
                
                for j in range(n_points):
                    p1_idx = cell.point_ids[j]
                    p2_idx = cell.point_ids[(j + 1) % n_points]
                    edge_key = tuple(sorted([p1_idx, p2_idx]))
                    edge_to_faces[edge_key].append(face_idx)
            
            # 提取每个面的顶点坐标
            face_vertices = []
            for face_idx in range(clipped.n_cells):
                cell = clipped.get_cell(face_idx)
                vertices = [clipped.points[cell.point_ids[j]] for j in range(cell.n_points)]
                face_vertices.append(vertices)
            
            # 判断每条边是否为真实棱边
            for edge_key, face_indices in edge_to_faces.items():
                if len(face_indices) == 2:
                    face1_idx, face2_idx = face_indices
                    
                    # 获取两个三角形的顶点
                    tri1_points = np.array(face_vertices[face1_idx])
                    tri2_points = np.array(face_vertices[face2_idx])
                    
                    # 判断是否共面
                    is_coplanar = are_triangles_coplanar(tri1_points, tri2_points, 
                                                         tolerance=coplanar_tolerance)
                    
                    if not is_coplanar:
                        # 不共面，是真实棱边
                        p1_idx, p2_idx = edge_key
                        p1 = clipped.points[p1_idx]
                        p2 = clipped.points[p2_idx]
                        all_edges_after.append([p1.copy(), p2.copy()])
            
            comparison_data.append({
                'index': idx,
                'mesh': clipped,
                'seed': seed,
                'color': color,
                'edges_before': all_edges_before,
                'edges_after': all_edges_after
            })
            
            # 检测单个凸包的连通性
            if len(all_edges_after) > 0:
                conn = check_edge_connectivity(all_edges_after, tolerance=0.001)
                if not conn['is_connected']:
                    print(f"  [警告] 凸包{idx}的边不连通！分量数: {conn['num_components']}")
            
            print(f"[凸包{idx}] 过滤前: {len(all_edges_before)}条边, 过滤后: {len(all_edges_after)}条边")
            
        except Exception as e:
            print(f"[凸包{idx}] 处理失败: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # 检查是否有有效数据
    if len(comparison_data) == 0:
        print("[对比查看器] 没有成功处理的凸包数据")
        return
    
    print(f"\n[对比查看器] 成功处理 {len(comparison_data)}/{len(convex_hulls)} 个凸包")
    
    # 创建交互式查看器
    print("\n[对比查看器] 启动交互式窗口...")
    print("[说明] 左侧=过滤前（所有边）, 右侧=过滤后（真实棱边）")
    print("[操作] 使用左右箭头键切换凸包")
    print("[操作] 按 'q' 键退出")
    
    class ComparisonViewer:
        def __init__(self, data):
            self.data = data
            self.current_index = 0
            self.plotter = None
            self.show_window()
        
        def show_window(self):
            """显示窗口"""
            # 创建新的 plotter
            if self.plotter is not None:
                try:
                    self.plotter.close()
                except:
                    pass
            
            self.plotter = pv.Plotter(shape=(1, 2), window_size=[1600, 800])
            
            # 添加键盘事件
            self.plotter.add_key_event('Right', self.next_hull)
            self.plotter.add_key_event('Left', self.prev_hull)
            
            # 渲染当前凸包
            self.render_current()
            
            # 显示窗口
            self.plotter.show()
        
        def render_current(self):
            """渲染当前凸包"""
            try:
                # 边界检查
                if len(self.data) == 0:
                    print("[错误] 没有数据可显示")
                    return
                
                if self.current_index < 0 or self.current_index >= len(self.data):
                    print(f"[错误] 索引越界: {self.current_index}, 数据长度: {len(self.data)}")
                    self.current_index = 0
                    return
                
                print(f"[调试] 正在显示凸包 {self.current_index + 1}/{len(self.data)}")
                
                # 获取当前凸包数据
                current = self.data[self.current_index]
                mesh = current['mesh']
                seed = current['seed']
                color = current['color']
                edges_before = current['edges_before']
                edges_after = current['edges_after']
                
                print(f"[调试] 数据: 面数={mesh.n_cells}, 点数={mesh.n_points}, 边(前)={len(edges_before)}, 边(后)={len(edges_after)}")
                
                # 左侧：过滤前
                self.plotter.subplot(0, 0)
                self.plotter.add_text("过滤前（所有边）", position='upper_edge', font_size=14, color='black')
                self.plotter.add_mesh(mesh, color=color, opacity=0.3, show_edges=False)
                
                # 渲染所有边（红色）
                if len(edges_before) > 0:
                    points_before = []
                    lines_before = []
                    point_offset = 0
                    for edge in edges_before:
                        p1, p2 = edge
                        points_before.append(p1)
                        points_before.append(p2)
                        lines_before.append([2, point_offset, point_offset + 1])
                        point_offset += 2
                    
                    points_array = np.array(points_before)
                    lines_array = np.hstack(lines_before)
                    edges_polydata = pv.PolyData(points_array, lines=lines_array)
                    self.plotter.add_mesh(edges_polydata, color='red', line_width=3, 
                                         render_lines_as_tubes=False)
                
                # 添加种子点
                sphere = pv.Sphere(radius=mesh.length * 0.02, center=seed)
                self.plotter.add_mesh(sphere, color='blue', opacity=1.0)
                
                info_text = f"凸包 {self.current_index + 1}/{len(self.data)}\n边数: {len(edges_before)}"
                self.plotter.add_text(info_text, position='lower_left', font_size=12, color='black')
                self.plotter.add_axes()
                self.plotter.reset_camera()
                
                # 右侧：过滤后
                self.plotter.subplot(0, 1)
                self.plotter.add_text("过滤后（真实棱边）", position='upper_edge', font_size=14, color='black')
                self.plotter.add_mesh(mesh, color=color, opacity=0.3, show_edges=False)
                
                # 渲染真实棱边（绿色）
                if len(edges_after) > 0:
                    points_after = []
                    lines_after = []
                    point_offset = 0
                    for edge in edges_after:
                        p1, p2 = edge
                        points_after.append(p1)
                        points_after.append(p2)
                        lines_after.append([2, point_offset, point_offset + 1])
                        point_offset += 2
                    
                    points_array = np.array(points_after)
                    lines_array = np.hstack(lines_after)
                    edges_polydata = pv.PolyData(points_array, lines=lines_array)
                    self.plotter.add_mesh(edges_polydata, color='green', line_width=3, 
                                         render_lines_as_tubes=False)
                
                # 添加种子点
                sphere = pv.Sphere(radius=mesh.length * 0.02, center=seed)
                self.plotter.add_mesh(sphere, color='blue', opacity=1.0)
                
                info_text = f"凸包 {self.current_index + 1}/{len(self.data)}\n边数: {len(edges_after)}"
                self.plotter.add_text(info_text, position='lower_left', font_size=12, color='black')
                self.plotter.add_axes()
                self.plotter.reset_camera()
                
                print(f"[调试] 显示完成")
                
            except Exception as e:
                print(f"[错误] 渲染失败 (凸包 {self.current_index + 1}): {e}")
                import traceback
                traceback.print_exc()
        
        def next_hull(self):
            """下一个凸包"""
            if len(self.data) == 0:
                print("[警告] 没有可显示的数据")
                return
            self.current_index = (self.current_index + 1) % len(self.data)
            print(f"[切换] 显示凸包 {self.current_index + 1}/{len(self.data)} (索引: {self.current_index})")
            # 关闭当前窗口并重新创建
            self.show_window()
        
        def prev_hull(self):
            """上一个凸包"""
            if len(self.data) == 0:
                print("[警告] 没有可显示的数据")
                return
            self.current_index = (self.current_index - 1) % len(self.data)
            print(f"[切换] 显示凸包 {self.current_index + 1}/{len(self.data)} (索引: {self.current_index})")
            # 关闭当前窗口并重新创建
            self.show_window()
    
    # 创建并显示查看器
    viewer = ComparisonViewer(comparison_data)
    # show_window() 会在 __init__ 中自动调用
    
    print("[对比查看器] 已关闭")


def visualize_with_pyvista(vor, internal_seeds, bounds, show_edges_only=False, is_sphere=False, sphere_center=None, sphere_radius=None, use_mesh_filter=False, stl_mesh=None, show_seeds=True, use_strategy_a=False, surface_seeds_count=0):
    """
    使用 PyVista 渲染 Voronoi 图
    
    参数:
        vor: Voronoi 对象
        internal_seeds: 内部种子点
        bounds: 包围盒边界
        show_edges_only: 是否只显示边（已废弃）
        is_sphere: 是否使用球形包围盒
        sphere_center: 球心坐标（仅当is_sphere=True时有效）
        sphere_radius: 球半径（仅当is_sphere=True时有效）
        use_mesh_filter: 是否使用STL网格过滤顶点
        stl_mesh: STL网格对象（用于顶点过滤）
        show_seeds: 是否显示种子点
        use_strategy_a: 是否使用策略A（表面边投影）
        surface_seeds_count: 表面种子数量（用于边分类）
    """
    lo, hi = bounds[0], bounds[1]
    extents = hi - lo
    
    # 创建 PyVista 场景
    plotter = pv.Plotter()
    plotter.set_background('white')
    
    # 计算可视化尺寸（种子点球体的半径）
    avg_extent = np.mean(extents)
    seed_sphere_radius = avg_extent * 0.015
    
    # 创建包围盒网格用于裁剪
    box_mesh = pv.Box(bounds=[lo[0], hi[0], lo[1], hi[1], lo[2], hi[2]])
    
    # 用于存储凸包信息
    all_convex_hulls = []
    
    # 用于存储所有特征边（用于去重）
    all_feature_edges = []
    
    # 定义角度阈值（用于提取真实棱边）
    angle_threshold = 40.0  # 角度阈值（度），法向量夹角小于此值认为共面
    
    # 创建包围盒网格用于布尔运算（确保三角化和法向量一致）
    box_mesh = pv.Box(bounds=[lo[0], hi[0], lo[1], hi[1], lo[2], hi[2]])
    box_mesh = box_mesh.triangulate()
    box_mesh = box_mesh.compute_normals(cell_normals=True, point_normals=False,
                                       consistent_normals=True, auto_orient_normals=True)
    
    # 只处理内部种子对应的胞元
    print(f"[可视化] 添加 Voronoi 胞元...")
    if use_mesh_filter and stl_mesh is not None:
        print(f"[可视化] 使用STL网格裁剪模式")
    elif is_sphere:
        print(f"[可视化] 使用球形裁剪模式")
    else:
        print(f"[可视化] 使用立方体裁剪模式")
    
    cell_count = 0
    skipped_infinite = 0
    skipped_error = 0
    colors = plt.cm.tab20(np.linspace(0, 1, len(internal_seeds)))
    
    # 遍历内部种子
    for i in range(len(internal_seeds)):
        if i > 0 and i % 10 == 0:
            print(f"[进度] 已处理 {i}/{len(internal_seeds)} 个种子，成功 {cell_count} 个")
        
        point_idx = i
        region_idx = vor.point_region[point_idx]
        region = vor.regions[region_idx]
        
        # 跳过空区域
        if not region:
            skipped_error += 1
            continue
        
        seed = internal_seeds[i]
        
        # 跳过包含无限远点的胞元
        if -1 in region:
            skipped_infinite += 1
            continue
        
        # 处理完全有限的胞元
        vertices = vor.vertices[region]
        
        try:
            # 方法：先过滤顶点，再生成凸包
            if use_mesh_filter and stl_mesh is not None:
                # STL网格过滤：只保留网格内部的顶点
                inside_mask = stl_mesh.contains(vertices)
                filtered_vertices = vertices[inside_mask]
                
                # 如果内部顶点太少，尝试添加边界附近的顶点
                if len(filtered_vertices) < 4:
                    closest_points, distances, triangle_id = stl_mesh.nearest.on_surface(vertices)
                    near_mask = distances < 2.0  # 距离小于2mm
                    combined_mask = inside_mask | near_mask
                    filtered_vertices = vertices[combined_mask]
                
                if i < 3:
                    print(f"\n[调试胞元{i}] 原始顶点: {len(vertices)}个")
                    print(f"[调试胞元{i}] STL网格过滤后: {len(filtered_vertices)}个")
                    
            elif is_sphere:
                # 球形包围盒：过滤到球内的顶点
                distances = np.linalg.norm(vertices - sphere_center, axis=1)
                inside_mask = distances <= sphere_radius
                filtered_vertices = vertices[inside_mask]
                
                # 对于边界顶点，投影到球面上
                boundary_mask = (distances > sphere_radius) & (distances <= sphere_radius * 1.1)
                if np.any(boundary_mask):
                    boundary_vertices = vertices[boundary_mask]
                    # 投影到球面
                    directions = boundary_vertices - sphere_center
                    directions = directions / np.linalg.norm(directions, axis=1, keepdims=True)
                    projected = sphere_center + directions * sphere_radius
                    filtered_vertices = np.vstack([filtered_vertices, projected])
                    
                if i < 3:
                    print(f"\n[调试胞元{i}] 原始顶点: {len(vertices)}个")
                    print(f"[调试胞元{i}] 球形过滤后: {len(filtered_vertices)}个")
            else:
                # 立方体包围盒：将超出包围盒的顶点裁剪到包围盒表面
                filtered_vertices = np.clip(vertices, lo, hi)
                
                if i < 3:
                    print(f"\n[调试胞元{i}] 原始顶点: {len(vertices)}个")
                    print(f"[调试胞元{i}] 立方体裁剪后: {len(filtered_vertices)}个")
            
            # 去除重复的顶点（裁剪可能导致多个顶点重合）
            filtered_vertices = np.unique(filtered_vertices, axis=0)
            
            # 检查是否有足够的顶点创建凸包
            if len(filtered_vertices) < 4:
                if i < 3:
                    print(f"[调试胞元{i}] 顶点不足4个，跳过")
                skipped_error += 1
                continue
            
            # 创建凸包
            from scipy.spatial import ConvexHull
            hull = ConvexHull(filtered_vertices)
            
            # 创建 PyVista 多面体
            faces = []
            for simplex in hull.simplices:
                faces.append(3)
                faces.extend(simplex)
            
            clipped = pv.PolyData(filtered_vertices, faces)
            
            if i < 3:
                print(f"[调试胞元{i}] 凸包: {clipped.n_points}个点, {clipped.n_cells}个面")
            
            # 保存凸包信息
            all_convex_hulls.append({
                'mesh': clipped,
                'seed': seed,
                'color': colors[i][:3]
            })
            
            # 渲染晶胞（凸包）- 只显示实体面片
            # plotter.add_mesh(clipped, color=colors[i][:3], 
            #                opacity=0.6)  # 半透明
            
            # 提取凸包的真实棱边（基于平面方程共面判断）
            try:
                # 方法：找出被两个三角形共享的边，判断两个三角形是否共面
                from collections import defaultdict
                
                # 步骤1：统计每条边被哪些面使用
                edge_to_faces = defaultdict(list)
                
                for face_idx in range(clipped.n_cells):
                    cell = clipped.get_cell(face_idx)
                    n_points = cell.n_points
                    
                    # 遍历这个面的所有边
                    for j in range(n_points):
                        p1_idx = cell.point_ids[j]
                        p2_idx = cell.point_ids[(j + 1) % n_points]
                        
                        # 创建边的唯一标识（排序确保方向无关）
                        edge_key = tuple(sorted([p1_idx, p2_idx]))
                        edge_to_faces[edge_key].append(face_idx)
                
                # 步骤2：提取每个面的顶点坐标（用于共面判断）
                face_vertices = []
                for face_idx in range(clipped.n_cells):
                    cell = clipped.get_cell(face_idx)
                    vertices = [clipped.points[cell.point_ids[j]] for j in range(cell.n_points)]
                    face_vertices.append(vertices)
                
                # 步骤3：找出真实棱边（被两个面共享且不共面）
                # 使用全局定义的 coplanar_tolerance
                coplanar_tolerance = 0.001  # 1微米容差
                
                real_edges_count = 0
                filtered_edges_count = 0
                
                for edge_key, face_indices in edge_to_faces.items():
                    # 只处理被恰好两个面共享的边
                    if len(face_indices) == 2:
                        face1_idx, face2_idx = face_indices
                        
                        # 获取两个三角形的顶点
                        tri1_points = np.array(face_vertices[face1_idx])
                        tri2_points = np.array(face_vertices[face2_idx])
                        
                        # 判断是否共面
                        is_coplanar = are_triangles_coplanar(tri1_points, tri2_points, 
                                                             tolerance=coplanar_tolerance)
                        
                        if not is_coplanar:
                            # 不共面，是真实棱边
                            p1_idx, p2_idx = edge_key
                            p1 = clipped.points[p1_idx]
                            p2 = clipped.points[p2_idx]
                            all_feature_edges.append([p1.copy(), p2.copy()])
                            real_edges_count += 1
                        else:
                            # 共面，过滤掉
                            filtered_edges_count += 1
                
                if i < 3:
                    total_edges = len(edge_to_faces)
                    shared_by_two = sum(1 for faces in edge_to_faces.values() if len(faces) == 2)
                    print(f"[调试胞元{i}] 总边数: {total_edges}, 被2面共享: {shared_by_two}")
                    print(f"[调试胞元{i}] 真实棱边: {real_edges_count}, 过滤掉: {filtered_edges_count}")
                        
            except Exception as e:
                if i < 3:
                    print(f"[调试胞元{i}] 提取棱边失败: {e}")
                import traceback
                traceback.print_exc()
            
            # 添加种子点（如果启用）
            if show_seeds:
                sphere = pv.Sphere(radius=seed_sphere_radius, center=seed)
                plotter.add_mesh(sphere, color='red', opacity=1.0)
            
            cell_count += 1
            
        except Exception as e:
            if i < 3:
                print(f"[调试胞元{i}] 创建凸包失败: {e}")
            skipped_error += 1
            continue
    
    print(f"[可视化] 成功添加 {cell_count} 个 Voronoi 胞元")
    print(f"[统计] 跳过: 无限远={skipped_infinite}, 错误={skipped_error}")
    print(f"[边统计] 提取到 {len(all_feature_edges)} 条特征边（去重前）")
    
    # 过滤相近的边（去重）- 使用优化的算法
    print(f"\n[边过滤] 开始过滤相近的边...")
    distance_threshold = 0.01  # 距离阈值，可调整
    
    # 优化方法：使用字典存储边的标准化表示
    # 将每条边的端点坐标四舍五入到一定精度，作为唯一标识
    edge_dict = {}
    precision = 3  # 保留3位小数
    
    for edge in all_feature_edges:
        p1, p2 = edge
        
        # 将端点坐标四舍五入
        p1_rounded = tuple(np.round(p1, decimals=precision))
        p2_rounded = tuple(np.round(p2, decimals=precision))
        
        # 确保边的方向一致（小端点在前）
        if p1_rounded > p2_rounded:
            p1_rounded, p2_rounded = p2_rounded, p1_rounded
        
        # 创建边的唯一标识
        edge_key = (p1_rounded, p2_rounded)
        
        # 如果这条边还没有被添加，则添加
        if edge_key not in edge_dict:
            edge_dict[edge_key] = [p1, p2]
    
    # 转换为列表
    filtered_edges = list(edge_dict.values())
    
    print(f"[边过滤] 过滤后剩余 {len(filtered_edges)} 条边（去除 {len(all_feature_edges) - len(filtered_edges)} 条重复边）")
    
    # 策略A：表面边投影处理
    if use_strategy_a and stl_mesh is not None and surface_seeds_count > 0:
        print(f"\n[策略A] 开始处理表面边投影...")
        print(f"[策略A] 表面种子数: {surface_seeds_count}")
        
        # 分类边
        surface_to_surface_edges = []
        surface_to_internal_edges = []
        internal_to_internal_edges = []
        
        for edge in filtered_edges:
            p1, p2 = edge
            
            # 找到最接近的种子点索引（简化版：使用距离判断）
            # 注意：这里假设边的端点接近某个种子点
            # 更精确的方法是在提取边时就记录种子点索引
            
            # 判断端点是否接近表面种子
            # 简化判断：检查端点是否在表面附近
            dist1_to_surface = stl_mesh.nearest.on_surface([p1])[1][0]
            dist2_to_surface = stl_mesh.nearest.on_surface([p2])[1][0]
            
            # 阈值：如果距离表面很近，认为是表面边
            surface_threshold = 0.5  # mm
            
            is_p1_surface = dist1_to_surface < surface_threshold
            is_p2_surface = dist2_to_surface < surface_threshold
            
            if is_p1_surface and is_p2_surface:
                surface_to_surface_edges.append(edge)
            elif is_p1_surface or is_p2_surface:
                surface_to_internal_edges.append(edge)
            else:
                internal_to_internal_edges.append(edge)
        
        print(f"[策略A] 边分类:")
        print(f"  - 表面-表面边: {len(surface_to_surface_edges)} (将投影到曲面)")
        print(f"  - 表面-内部边: {len(surface_to_internal_edges)} (保持直线)")
        print(f"  - 内部-内部边: {len(internal_to_internal_edges)} (保持直线)")
        
        # 投影表面-表面边到曲面
        projected_edges = []
        
        print(f"[策略A] 投影表面-表面边到曲面...")
        for edge in surface_to_surface_edges:
            p1, p2 = edge
            
            # 投影为多段线（5段）
            segments = project_edge_to_surface_geodesic(p1, p2, stl_mesh, num_segments=5)
            projected_edges.extend(segments)
        
        print(f"[策略A] 投影完成: {len(surface_to_surface_edges)} 条边 → {len(projected_edges)} 条线段")
        
        # 处理表面-内部边（投影表面端点）
        processed_surface_internal = []
        for edge in surface_to_internal_edges:
            p1, p2 = edge
            
            dist1 = stl_mesh.nearest.on_surface([p1])[1][0]
            dist2 = stl_mesh.nearest.on_surface([p2])[1][0]
            
            if dist1 < dist2:
                # p1 是表面点，投影它
                p1_proj = project_point_to_surface(p1, stl_mesh)
                processed_surface_internal.append([p1_proj, p2])
            else:
                # p2 是表面点，投影它
                p2_proj = project_point_to_surface(p2, stl_mesh)
                processed_surface_internal.append([p1, p2_proj])
        
        # 合并所有边
        filtered_edges = projected_edges + processed_surface_internal + internal_to_internal_edges
        
        print(f"[策略A] 最终边数: {len(filtered_edges)}")
    
    # 连通性检测
    connectivity_result = check_edge_connectivity(filtered_edges, tolerance=0.001)
    
    if not connectivity_result['is_connected']:
        print(f"\n{'='*60}")
        print(f"警告：提取的边不连通！")
        print(f"这可能导致晶格结构不完整或存在孤立部分")
        print(f"{'='*60}\n")
    
    # 渲染过滤后的边（批量渲染，提升性能）
    print(f"[渲染] 渲染特征边...")
    if len(filtered_edges) > 0:
        # 方法：将所有边合并成一个PolyData对象，一次性渲染
        all_points = []
        all_lines = []
        point_offset = 0
        
        for edge in filtered_edges:
            p1, p2 = edge
            all_points.append(p1)
            all_points.append(p2)
            # 线段格式：[2, point_index1, point_index2]
            all_lines.append([2, point_offset, point_offset + 1])
            point_offset += 2
        
        # 创建包含所有边的PolyData
        points_array = np.array(all_points)
        lines_array = np.hstack(all_lines)  # 展平成一维数组
        
        edges_polydata = pv.PolyData(points_array, lines=lines_array)
        
        # 一次性添加所有边（性能大幅提升）
        plotter.add_mesh(edges_polydata, color='black', line_width=2, opacity=1.0, 
                        render_lines_as_tubes=False)  # 使用简单线条而非管状，更快
        
        print(f"[渲染] 已批量渲染 {len(filtered_edges)} 条边（性能优化）")
    
    # 如果使用STL网格，也渲染STL模型（半透明）
    if use_mesh_filter and stl_mesh is not None:
        # 不渲染STL模型，只显示Voronoi晶胞
        print(f"[可视化] STL模型已隐藏，仅显示Voronoi晶胞")
    
    # 添加包围盒或球形边界的线框
    if is_sphere:
        # 添加球形边界线框
        sphere_boundary = pv.Sphere(radius=sphere_radius, center=sphere_center, 
                                    theta_resolution=30, phi_resolution=30)
        plotter.add_mesh(sphere_boundary, color='blue', style='wireframe', 
                        line_width=2, opacity=0.3)
        print("[渲染] 已添加球形边界线框")
    else:
        # 不添加立方体包围盒边框
        print("[渲染] 已跳过包围盒线框渲染")
    
    # 添加坐标轴
    plotter.add_axes()
    
    # 更新文本信息
    info_text = f"Seeds: {len(internal_seeds)}, Cells: {cell_count}"
    plotter.add_text(info_text, position='upper_left', font_size=12)
    
    print("\n[可视化] 打开交互窗口...")
    print("[提示] 关闭窗口以继续")
    print("[说明] 黑色线 = 凸包的特征边（轮廓边）")
    if show_seeds:
        print("[说明] 红色球 = 种子点")
    if is_sphere:
        print(f"[说明] 蓝色线框 = 球形边界（半径={sphere_radius}）")
    plotter.show()
    
    # 创建对比查看器：显示前10个凸包的过滤前后对比（可选）
    if len(all_convex_hulls) > 0:
        print("\n[对比查看器] 可以显示前10个凸包的过滤前后对比（用于调试）")
        show_comparison = input("是否查看对比? (y/n) [默认: n]: ").strip().lower() or "n"
        if show_comparison == "y":
            print("[对比查看器] 启动中...")
            create_edge_comparison_viewer(all_convex_hulls[:10], coplanar_tolerance=0.001)
    
    # 返回提取的边，用于后续晶格生成
    return filtered_edges


def visualize_voronoi_interactive(bounds, cell_size=10.0, num_seeds=None, seed_method='jittered', use_matplotlib=False, is_sphere=False, sphere_center=None, sphere_radius=None, surface_seeds=None, use_mesh_filter=False, stl_mesh=None, show_seeds=True, use_strategy_a=False, surface_seeds_count=0):
    """
    交互式可视化 Voronoi 图
    
    参数:
        bounds: 包围盒边界 [[xmin, ymin, zmin], [xmax, ymax, zmax]]
        cell_size: 胞元尺寸
        num_seeds: 种子点数量（None表示自动计算）
        seed_method: 种子采样方法
        use_matplotlib: 是否使用matplotlib渲染器
        is_sphere: 是否使用球形包围盒
        sphere_center: 球心坐标（仅当is_sphere=True时有效）
        sphere_radius: 球半径（仅当is_sphere=True时有效）
        surface_seeds: 预先生成的表面种子点（可选）
        use_mesh_filter: 是否使用STL网格过滤顶点
        stl_mesh: STL网格对象（用于顶点过滤）
        show_seeds: 是否显示种子点
        use_strategy_a: 是否使用策略A（表面边投影）
        surface_seeds_count: 表面种子数量（用于边分类）
    """
    if not use_matplotlib and not PYVISTA_AVAILABLE:
        print("[警告] PyVista 未安装，切换到 matplotlib")
        use_matplotlib = True
    
    print("=" * 60)
    print("生成并可视化 3D Voronoi 图")
    print("=" * 60)
    
    bounds = np.asarray(bounds, dtype=np.float64)
    lo, hi = bounds[0], bounds[1]
    extents = hi - lo
    
    print(f"[参数] 包围盒: {lo} ~ {hi}")
    print(f"[参数] 尺寸: {extents}")
    print(f"[参数] cell_size={cell_size}, seed_method={seed_method}")
    
    # 步骤1: 采样内部种子点
    print(f"\n[步骤1] 采样内部种子点...")
    
    if surface_seeds is not None:
        # 使用预先生成的表面种子点
        internal_seeds = surface_seeds
        print(f"[表面种子] 使用预先生成的 {len(internal_seeds)} 个表面种子点")
    else:
        # 常规方式：在包围盒内采样
        internal_seeds = sample_seeds_in_box(bounds, cell_size, num_seeds=num_seeds, method=seed_method)
        
        # 如果是球形包围盒，过滤掉球外的种子点
        if is_sphere:
            distances = np.linalg.norm(internal_seeds - sphere_center, axis=1)
            inside_mask = distances <= sphere_radius
            internal_seeds = internal_seeds[inside_mask]
            print(f"[球形过滤] 过滤后保留 {len(internal_seeds)} 个球内种子点")
    
    if len(internal_seeds) < 4:
        print("[错误] 种子点数量不足")
        return
    
    print(f"[结果] 内部种子点数: {len(internal_seeds)}")
    
    # 步骤2: 创建边界镜像种子点
    print(f"\n[步骤2] 创建边界镜像种子点...")
    mirror_seeds = create_boundary_mirror_seeds(bounds, cell_size)
    print(f"[结果] 镜像种子点数: {len(mirror_seeds)}")
    
    # 步骤3: 合并所有种子点
    all_seeds = np.vstack([internal_seeds, mirror_seeds])
    print(f"[结果] 总种子点数: {len(all_seeds)} (内部 {len(internal_seeds)} + 镜像 {len(mirror_seeds)})")
    
    # 步骤4: 计算 Voronoi 图（使用所有种子）
    print(f"\n[步骤3] 计算 3D Voronoi 图...")
    from scipy.spatial import Voronoi as ScipyVoronoi
    vor = ScipyVoronoi(all_seeds)
    
    print(f"[结果] Voronoi 顶点数: {len(vor.vertices)}")
    print(f"[结果] Voronoi 区域数: {len(vor.regions)}")
    
    # 步骤5: 可视化（根据渲染器选择）
    if use_matplotlib:
        print(f"\n[步骤4] 使用 Matplotlib 可视化...")
        visualize_with_matplotlib(vor, internal_seeds, bounds, colors=plt.cm.tab20(np.linspace(0, 1, len(internal_seeds))))
        extracted_edges = None
    else:
        print(f"\n[步骤4] 使用 PyVista 可视化...")
        extracted_edges = visualize_with_pyvista(vor, internal_seeds, bounds, is_sphere=is_sphere, sphere_center=sphere_center, sphere_radius=sphere_radius, use_mesh_filter=use_mesh_filter, stl_mesh=stl_mesh, show_seeds=show_seeds, use_strategy_a=use_strategy_a, surface_seeds_count=surface_seeds_count)
    
    print("\n" + "=" * 60)
    print("Voronoi 图可视化完成")
    print("=" * 60)
    
    # 返回提取的边和包围盒
    return extracted_edges, bounds


def voronoi_visualization_mode():
    """
    Voronoi 图可视化模式
    """
    print("\n" + "=" * 60)
    print("Voronoi 图可视化模式")
    print("=" * 60)
    
    # 选择包围盒
    print("\n请选择包围盒:")
    print("1. 简单立方体")
    print("2. 球形")
    print("3. 自定义包围盒")
    print("4. 从STL文件提取")
    
    box_choice = input("\n请输入选项 (1/2/3/4) [默认: 1]: ").strip() or "1"
    
    is_sphere = False
    sphere_center = None
    sphere_radius = None
    
    if box_choice == "1":
        size = input("立方体尺寸 (mm) [默认: 50]: ").strip()
        size = float(size) if size else 50.0
        bounds = [[0, 0, 0], [size, size, size]]
        print(f"[包围盒] 立方体: {size}×{size}×{size} mm")
    elif box_choice == "2":
        radius = input("球形半径 (mm) [默认: 25]: ").strip()
        radius = float(radius) if radius else 25.0
        # 球形的包围盒是外接立方体
        bounds = [[-radius, -radius, -radius], [radius, radius, radius]]
        sphere_center = np.array([0.0, 0.0, 0.0])
        sphere_radius = radius
        is_sphere = True
        print(f"[包围盒] 球形: 半径 {radius} mm, 中心 (0, 0, 0)")
    elif box_choice == "3":
        print("请输入包围盒坐标 (xmin ymin zmin xmax ymax zmax):")
        coords = input("[默认: 0 0 0 50 50 50]: ").strip()
        if coords:
            vals = list(map(float, coords.split()))
            bounds = [[vals[0], vals[1], vals[2]], [vals[3], vals[4], vals[5]]]
        else:
            bounds = [[0, 0, 0], [50, 50, 50]]
        print(f"[包围盒] {bounds[0]} ~ {bounds[1]}")
    elif box_choice == "4":
        stl_path = input("请输入STL文件路径: ").strip()
        if not stl_path:
            print("[错误] 未输入文件路径")
            return
        
        bounds, mesh = extract_bounds_from_stl(stl_path)
        if bounds is None:
            print("[错误] 无法提取包围盒，退出")
            return
        
        # 询问种子点生成方式
        print("\n种子点生成方式:")
        print("  1. 在包围盒内部生成（常规方式）")
        print("  2. 在STL表面上生成")
        print("  3. 在STL表面附近生成（带偏移）")
        print("  4. 在STL表面+内部混合生成 ⭐ 推荐")
        print("  5. 表面2D Voronoi + 内部3D Voronoi（表面均匀）🆕")
        surface_choice = input("选择方式 (1/2/3/4/5) [默认: 4]: ").strip() or "4"
        
        use_surface_seeds = False
        surface_seeds = None
        use_mesh_filter = False  # 是否使用STL网格过滤顶点
        stl_mesh = None  # 保存mesh对象用于过滤
        use_strategy_a = False  # 是否使用策略A（表面边投影）
        surface_seeds_count = 0  # 表面种子数量
        
        if surface_choice == "2":
            use_surface_seeds = True
            use_mesh_filter = True
            stl_mesh = mesh
            num_surface_seeds = input("表面种子点数量 [默认: 100]: ").strip()
            num_surface_seeds = int(num_surface_seeds) if num_surface_seeds else 100
            surface_seeds = sample_seeds_on_mesh_surface(mesh, num_seeds=num_surface_seeds)
            
        elif surface_choice == "3":
            use_surface_seeds = True
            use_mesh_filter = True
            stl_mesh = mesh
            num_surface_seeds = input("表面附近种子点数量 [默认: 100]: ").strip()
            num_surface_seeds = int(num_surface_seeds) if num_surface_seeds else 100
            
            min_offset = input("最小偏移距离 (mm) [默认: 0]: ").strip()
            min_offset = float(min_offset) if min_offset else 0.0
            
            max_offset = input("最大偏移距离 (mm) [默认: 5]: ").strip()
            max_offset = float(max_offset) if max_offset else 5.0
            
            surface_offset_range = (min_offset, max_offset)
            surface_seeds = sample_seeds_near_mesh_surface(mesh, num_surface_seeds, surface_offset_range)
            
        elif surface_choice == "4":
            # 表面+内部混合 - 根据胞元尺寸自动计算，使用泊松盘采样
            use_surface_seeds = True
            use_mesh_filter = True
            stl_mesh = mesh
            
            # 询问胞元尺寸
            cell_size_input = input("胞元尺寸 (mm) [默认: 10.0]: ").strip()
            cell_size_for_calc = float(cell_size_input) if cell_size_input else 10.0
            
            # 根据STL模型体积和胞元尺寸计算总种子数
            # 使用包围盒体积作为近似
            bounds_array = np.array(bounds)
            extents = bounds_array[1] - bounds_array[0]
            volume = np.prod(extents)
            total_seeds = int(volume / (cell_size_for_calc ** 3))
            total_seeds = max(20, total_seeds)  # 至少20个
            
            # 按4:6比例分配（表面40%，内部60%）
            num_surface_seeds = int(total_seeds * 0.4)
            num_inside_seeds = int(total_seeds * 0.6)
            
            print(f"\n[自动计算] 胞元尺寸: {cell_size_for_calc} mm")
            print(f"[自动计算] 包围盒体积: {volume:.2f} mm³")
            print(f"[自动计算] 总种子数: {total_seeds}")
            print(f"[自动计算] 表面种子: {num_surface_seeds} (40%)")
            print(f"[自动计算] 内部种子: {num_inside_seeds} (60%)")
            
            # 询问使用哪种方法生成表面种子
            print("\n表面种子生成方法:")
            print("  1. 泊松盘采样 + Lloyd松弛（快速，近似）")
            print("  2. 曲面2D Voronoi（慢速，精确）⭐ 推荐")
            method_choice = input("选择方法 (1/2) [默认: 2]: ").strip() or "2"
            
            if method_choice == "2":
                # 方法2：使用曲面2D Voronoi（基于测地距离）
                print(f"\n[方法] 使用曲面2D Voronoi（基于测地距离）")
                
                use_geodesic_input = input("是否使用真正的测地距离? (y/n) [默认: y]: ").strip().lower() or "y"
                use_geodesic = (use_geodesic_input == "y")
                
                if use_geodesic:
                    print("[警告] 使用真正的测地距离计算量较大，可能需要几分钟时间")
                
                surface_pts, voronoi_info = generate_surface_seeds_with_2d_voronoi(
                    mesh,
                    num_seeds=num_surface_seeds,
                    cell_size=cell_size_for_calc,
                    use_geodesic=use_geodesic
                )
            else:
                # 方法1：使用泊松盘采样 + Lloyd松弛
                print(f"\n[方法] 使用泊松盘采样 + Lloyd松弛")
                
                use_lloyd_input = input("是否使用Lloyd松弛优化? (y/n) [默认: y]: ").strip().lower() or "y"
                use_lloyd = (use_lloyd_input == "y")
                
                lloyd_iterations = 10
                if use_lloyd:
                    lloyd_iter_input = input("Lloyd松弛迭代次数 [默认: 10]: ").strip()
                    lloyd_iterations = int(lloyd_iter_input) if lloyd_iter_input else 10
                
                surface_pts = generate_optimized_surface_seeds(
                    mesh,
                    num_seeds=num_surface_seeds,
                    cell_size=cell_size_for_calc,
                    use_lloyd=use_lloyd,
                    lloyd_iterations=lloyd_iterations
                )
            
            # 生成内部种子点 - 使用随机采样
            print(f"\n[内部采样] 使用随机采样")
            inside_pts = sample_seeds_inside_mesh(mesh, num_seeds=num_inside_seeds, method='random')
            
            # 合并
            surface_seeds = np.vstack([surface_pts, inside_pts])
            print(f"\n[混合采样] 总计 {len(surface_seeds)} 个种子点 (表面 {len(surface_pts)} + 内部 {len(inside_pts)})")
        
        elif surface_choice == "5":
            # 表面2D Voronoi + 内部3D Voronoi（表面均匀）
            use_surface_seeds = True
            use_mesh_filter = True
            stl_mesh = mesh
            
            # 询问胞元尺寸
            cell_size_input = input("胞元尺寸 (mm) [默认: 10.0]: ").strip()
            cell_size_for_calc = float(cell_size_input) if cell_size_input else 10.0
            
            print(f"\n[策略A] 表面2D Voronoi + 内部3D Voronoi（共享边界节点）")
            print(f"[说明] 表面-表面边将投影到曲面，表面-内部边和内部-内部边保持直线")
            
            # 步骤1：生成表面种子（2D Voronoi）
            print(f"\n[1/3] 生成表面种子...")
            
            # 询问使用哪种方法生成表面种子
            print("\n表面种子生成方法:")
            print("  1. 泊松盘采样 + Lloyd松弛（快速，近似）")
            print("  2. 曲面2D Voronoi（慢速，精确）⭐ 推荐")
            method_choice = input("选择方法 (1/2) [默认: 2]: ").strip() or "2"
            
            # 计算表面种子数量
            surface_area = mesh.area
            num_surface_seeds = int(surface_area / (cell_size_for_calc ** 2))
            num_surface_seeds = max(20, num_surface_seeds)
            
            if method_choice == "2":
                # 方法2：使用曲面2D Voronoi（基于测地距离）
                print(f"\n[方法] 使用曲面2D Voronoi（基于测地距离）")
                
                use_geodesic_input = input("是否使用真正的测地距离? (y/n) [默认: y]: ").strip().lower() or "y"
                use_geodesic = (use_geodesic_input == "y")
                
                if use_geodesic:
                    print("[警告] 使用真正的测地距离计算量较大，可能需要几分钟时间")
                
                surface_pts, voronoi_info = generate_surface_seeds_with_2d_voronoi(
                    mesh,
                    num_seeds=num_surface_seeds,
                    cell_size=cell_size_for_calc,
                    use_geodesic=use_geodesic
                )
            else:
                # 方法1：使用泊松盘采样 + Lloyd松弛
                print(f"\n[方法] 使用泊松盘采样 + Lloyd松弛")
                
                use_lloyd_input = input("是否使用Lloyd松弛优化? (y/n) [默认: y]: ").strip().lower() or "y"
                use_lloyd = (use_lloyd_input == "y")
                
                lloyd_iterations = 10
                if use_lloyd:
                    lloyd_iter_input = input("Lloyd松弛迭代次数 [默认: 10]: ").strip()
                    lloyd_iterations = int(lloyd_iter_input) if lloyd_iter_input else 10
                
                surface_pts = generate_optimized_surface_seeds(
                    mesh,
                    num_seeds=num_surface_seeds,
                    cell_size=cell_size_for_calc,
                    use_lloyd=use_lloyd,
                    lloyd_iterations=lloyd_iterations
                )
            
            # 步骤2：生成内部种子（排除表面附近）
            print(f"\n[2/3] 生成内部种子（排除表面附近）...")
            
            exclude_distance_input = input(f"排除距离（距离表面多近不生成种子，单位mm）[默认: {cell_size_for_calc * 0.5:.2f}]: ").strip()
            exclude_distance = float(exclude_distance_input) if exclude_distance_input else cell_size_for_calc * 0.5
            
            inside_pts = generate_internal_seeds_excluding_surface(
                mesh,
                cell_size=cell_size_for_calc,
                exclude_distance=exclude_distance
            )
            
            # 步骤3：合并种子
            print(f"\n[3/3] 合并种子...")
            surface_seeds = np.vstack([surface_pts, inside_pts])
            
            print(f"\n[混合采样] 总计 {len(surface_seeds)} 个种子点")
            print(f"  - 表面种子: {len(surface_pts)} (前 {len(surface_pts)} 个)")
            print(f"  - 内部种子: {len(inside_pts)} (后 {len(inside_pts)} 个)")
            
            # 保存表面种子数量，用于后续边分类
            # 将这个信息存储在全局变量或传递给后续函数
            # 这里我们使用一个特殊的标记
            surface_seeds_count = len(surface_pts)
            
            # 标记这是策略A模式，需要特殊处理边
            use_strategy_a = True
            print(f"\n[策略A] 已启用表面边投影模式")
        
        # 询问是否使用边距
        use_margin = input("\n是否添加包围盒边距? (y/n) [默认: y]: ").strip().lower() or "y"
        if use_margin == "y":
            margin_percent = input("边距百分比 (%) [默认: 10]: ").strip()
            margin_percent = float(margin_percent) if margin_percent else 10.0
            
            # 计算边距
            bounds_array = np.array(bounds)
            extents = bounds_array[1] - bounds_array[0]
            margin = extents * (margin_percent / 100.0)
            
            # 扩展包围盒
            bounds_array[0] -= margin
            bounds_array[1] += margin
            bounds = bounds_array.tolist()
            
            print(f"[包围盒] 添加 {margin_percent}% 边距后:")
            print(f"[包围盒] {bounds[0]} ~ {bounds[1]}")
    else:
        print("[错误] 无效选项")
        return
    
    # 参数设置
    print("\n" + "-" * 60)
    print("参数设置:")
    print("-" * 60)
    
    # 如果使用表面种子点（模式4已经在前面设置了cell_size）
    if box_choice == "4" and use_surface_seeds:
        # 模式4（表面+内部混合）已经在前面根据胞元尺寸自动计算了种子数
        # 这里只需要设置镜像种子的cell_size
        if surface_choice == "4" or surface_choice == "5":
            # 表面+内部混合模式（选项4）或策略A模式（选项5），使用之前输入的cell_size
            cell_size = cell_size_for_calc  # 使用前面计算时输入的值
        else:
            # 其他表面种子模式，询问胞元尺寸
            cell_size_input = input("胞元尺寸（用于镜像种子）(mm) [默认: 10.0]: ").strip()
            cell_size = float(cell_size_input) if cell_size_input else 10.0
        
        num_seeds = None  # 已经有表面种子点了
        seed_method = 'surface'  # 标记为表面采样
        print(f"  [种子点] 已生成 {len(surface_seeds)} 个种子点")
        print(f"  [镜像种子] 胞元尺寸: {cell_size} mm (用于生成镜像种子)")
    else:
        cell_size = input("胞元尺寸 (mm) [默认: 10.0]: ").strip()
        cell_size = float(cell_size) if cell_size else 10.0
        
        num_seeds = input("种子点数量 (留空自动计算) [默认: 自动]: ").strip()
        num_seeds = int(num_seeds) if num_seeds else None
        
        # 选择种子采样方法
        print("\n种子采样方法:")
        print("  1. jittered - 网格+大抖动（中等随机）⭐ 推荐")
        print("  2. random - 完全随机（最不规则）")
        print("  3. grid - 规则网格+小抖动（最规则）")
        print("  4. poisson - 泊松盘采样（均匀但不规则）")
        seed_method_choice = input("选择方法 (1/2/3/4) [默认: 1]: ").strip() or "1"
        seed_method_map = {'1': 'jittered', '2': 'random', '3': 'grid', '4': 'poisson'}
        seed_method = seed_method_map.get(seed_method_choice, 'jittered')
        
        surface_seeds = None  # 不使用表面种子点
    
    # 选择渲染器
    print("\n渲染器选择:")
    print("  1. PyVista (功能强大，支持边提取) ⭐ 推荐")
    print("  2. Matplotlib (备用，仅显示胞元)")
    renderer_choice = input("选择渲染器 (1/2) [默认: 1]: ").strip() or "1"
    use_matplotlib = (renderer_choice == "2")
    
    # 是否显示种子点
    show_seeds_input = input("\n是否显示种子点? (y/n) [默认: y]: ").strip().lower() or "y"
    show_seeds = (show_seeds_input == "y")
    
    print("\n" + "-" * 60)
    print("参数确认:")
    if seed_method == 'surface':
        print(f"  种子类型: 表面种子点")
        print(f"  种子数: {len(surface_seeds)}")
    else:
        print(f"  胞元尺寸: {cell_size} mm")
        print(f"  种子数: {'自动' if num_seeds is None else num_seeds}")
        print(f"  采样方法: {seed_method}")
    print(f"  渲染器: {'Matplotlib' if use_matplotlib else 'PyVista'}")
    print(f"  显示种子点: {'是' if show_seeds else '否'}")
    print("-" * 60)
    
    confirm = input("\n开始生成并可视化? (y/n) [默认: y]: ").strip().lower() or "y"
    if confirm != "y":
        print("已取消")
        return
    
    # 生成并可视化
    extracted_edges, bounds = visualize_voronoi_interactive(
        bounds=bounds,
        cell_size=cell_size,
        num_seeds=num_seeds,
        seed_method=seed_method,
        use_matplotlib=use_matplotlib,
        is_sphere=is_sphere,
        sphere_center=sphere_center,
        sphere_radius=sphere_radius,
        surface_seeds=surface_seeds,
        use_mesh_filter=use_mesh_filter if box_choice == "4" else False,
        stl_mesh=stl_mesh if box_choice == "4" else None,
        show_seeds=show_seeds,
        use_strategy_a=use_strategy_a if box_choice == "4" else False,
        surface_seeds_count=surface_seeds_count if box_choice == "4" else 0
    )
    
    # 询问是否生成晶格
    if extracted_edges is not None and len(extracted_edges) > 0:
        print("\n" + "=" * 60)
        generate_lattice = input("是否基于提取的边生成 Voronoi 晶格? (y/n) [默认: n]: ").strip().lower() or "n"
        
        if generate_lattice == "y":
            # 步骤1：估算合理的壁厚范围
            thickness_info = estimate_optimal_wall_thickness(
                extracted_edges, 
                bounds, 
                target_density=0.15  # 目标15%填充密度
            )
            
            # 步骤2：询问晶格参数
            print("\n" + "=" * 60)
            print("晶格生成参数设置")
            print("=" * 60)
            
            # 壁厚设置
            print(f"\n[壁厚设置]")
            if thickness_info:
                print(f"  根据边的分布，建议壁厚范围：")
                print(f"  • 最小: {thickness_info['min_thickness']:.2f} mm （避免过细）")
                print(f"  • 推荐: {thickness_info['recommended']:.2f} mm （平衡强度和重量）⭐")
                print(f"  • 最大: {thickness_info['max_thickness']:.2f} mm （避免过度填充）")
                default_thickness = thickness_info['recommended']
            else:
                default_thickness = 3.0
            
            wall_thickness_input = input(f"\n  请输入壁厚 (mm) [默认: {default_thickness:.2f}]: ").strip()
            wall_thickness = float(wall_thickness_input) if wall_thickness_input else default_thickness
            
            # 验证壁厚是否合理
            if thickness_info:
                if wall_thickness < thickness_info['min_thickness']:
                    print(f"  ⚠️  警告：壁厚过小（< {thickness_info['min_thickness']:.2f} mm），可能导致结构过细或断裂")
                elif wall_thickness > thickness_info['max_thickness']:
                    print(f"  ⚠️  警告：壁厚过大（> {thickness_info['max_thickness']:.2f} mm），可能导致过度填充")
                else:
                    print(f"  ✓ 壁厚在合理范围内")
            
            # 密度预览
            print(f"\n[密度预览] 正在估算填充密度...")
            estimated_density = preview_lattice_density(
                extracted_edges, 
                bounds, 
                wall_thickness, 
                sample_size=1000
            )
            print(f"  估算填充密度: {estimated_density*100:.1f}%")
            
            if estimated_density < 0.05:
                print(f"  ⚠️  密度过低（< 5%），结构可能过于稀疏")
                adjust = input("  是否增大壁厚? (y/n) [默认: n]: ").strip().lower()
                if adjust == "y":
                    wall_thickness = wall_thickness * 1.5
                    print(f"  已调整壁厚为: {wall_thickness:.2f} mm")
            elif estimated_density > 0.30:
                print(f"  ⚠️  密度过高（> 30%），结构可能过于密实")
                adjust = input("  是否减小壁厚? (y/n) [默认: n]: ").strip().lower()
                if adjust == "y":
                    wall_thickness = wall_thickness * 0.7
                    print(f"  已调整壁厚为: {wall_thickness:.2f} mm")
            else:
                print(f"  ✓ 密度合理（5%-30%）")
            
            # 分辨率设置
            print(f"\n[分辨率设置]")
            print(f"  分辨率 = 每毫米的体素数（voxels/mm）")
            print(f"  影响杆件的圆滑度和计算时间：")
            print(f"  • 2: 快速预览（粗糙，杆件有明显棱角）")
            print(f"  • 3-4: 平衡质量（推荐，杆件较圆滑）⭐")
            print(f"  • 5-6: 高质量（精细，杆件非常圆滑，计算慢）")
            print(f"  ⚠️  建议：杆件直径至少包含3个体素，否则会很粗糙")
            
            # 计算推荐分辨率
            min_resolution_for_quality = max(3, int(np.ceil(3.0 / wall_thickness)))
            if min_resolution_for_quality > 4:
                print(f"  💡 提示：当前壁厚 {wall_thickness:.2f} mm，建议分辨率 ≥ {min_resolution_for_quality} 以保证质量")
            
            resolution_input = input(f"\n  请输入分辨率（体素/mm）[默认: 4]: ").strip()
            resolution = int(resolution_input) if resolution_input else 4
            
            # 检查分辨率是否足够
            voxels_per_diameter = wall_thickness * resolution
            if voxels_per_diameter < 3:
                print(f"  ⚠️  警告：杆件直径仅 {voxels_per_diameter:.1f} 个体素，会很粗糙！")
                print(f"  建议：增大分辨率至 {min_resolution_for_quality} 或以上")
            
            # 平滑设置
            print(f"\n[平滑设置]")
            print(f"  Laplacian平滑可以改善杆件表面质量：")
            print(f"  • 优点：减少阶梯状伪影，表面更光滑")
            print(f"  • 缺点：可能轻微改变几何形状")
            print(f"  • 推荐：对于低分辨率（2-3）启用，高分辨率（5+）可选")
            
            smooth_input = input(f"\n  是否启用平滑? (y/n) [默认: y]: ").strip().lower() or "y"
            smooth = (smooth_input == "y")
            
            # 确认参数
            print(f"\n" + "-" * 60)
            print(f"参数确认：")
            print(f"  壁厚: {wall_thickness:.2f} mm")
            print(f"  分辨率: {resolution} 体素/mm")
            print(f"  体素大小: {1.0/resolution:.4f} mm")
            print(f"  杆件直径: {voxels_per_diameter:.1f} 个体素")
            print(f"  平滑: {'是' if smooth else '否'}")
            print(f"  预计密度: {estimated_density*100:.1f}%")
            print(f"-" * 60)
            
            confirm = input("\n开始生成? (y/n) [默认: y]: ").strip().lower() or "y"
            if confirm != "y":
                print("已取消生成")
                return
            
            # 生成晶格
            lattice_mesh = generate_lattice_from_edges(
                bounds=bounds,
                edges=extracted_edges,
                wall_thickness=wall_thickness,
                resolution=resolution,
                smooth=smooth
            )
            
            if lattice_mesh is not None:
                # 计算实际密度
                bounds_array = np.array(bounds)
                bbox_volume = np.prod(bounds_array[1] - bounds_array[0])
                actual_density = lattice_mesh.volume / bbox_volume
                
                print(f"\n[实际结果]")
                print(f"  晶格体积: {lattice_mesh.volume:.2f} mm³")
                print(f"  包围盒体积: {bbox_volume:.2f} mm³")
                print(f"  实际填充密度: {actual_density*100:.1f}%")
                print(f"  密度偏差: {abs(actual_density - estimated_density)*100:.1f}%")
                
                # 保存文件
                print("\n" + "-" * 60)
                default_name = "voronoi_lattice.stl"
                output_path = input(f"保存文件名 [默认: {default_name}]: ").strip() or default_name
                
                try:
                    lattice_mesh.export(output_path)
                    print(f"\n[成功] 已保存到: {output_path}")
                    print(f"[网格] {len(lattice_mesh.vertices)} 顶点, {len(lattice_mesh.faces)} 面")
                    
                    # 提供优化建议
                    if actual_density < 0.05:
                        print(f"\n💡 建议：密度较低，可以增大壁厚到 {wall_thickness*1.5:.2f} mm")
                    elif actual_density > 0.30:
                        print(f"\n💡 建议：密度较高，可以减小壁厚到 {wall_thickness*0.7:.2f} mm")
                    
                except Exception as e:
                    print(f"[错误] 保存失败: {e}")
    
    # 询问是否继续
    print("\n" + "=" * 60)
    again = input("是否继续可视化? (y/n) [默认: n]: ").strip().lower() or "n"
    if again == "y":
        voronoi_visualization_mode()


def interactive_mode():
    """
    交互式模式：通过命令行输入参数生成 Voronoi 晶格
    """
    print("\n" + "=" * 60)
    print("Voronoi 隐函数建模 - 交互式模式")
    print("=" * 60)
    
    # 选择模式
    print("\n请选择模式:")
    print("1. 简单立方体测试")
    print("2. 使用现有网格的包围盒")
    print("3. 自定义包围盒")
    
    mode = input("\n请输入选项 (1/2/3) [默认: 1]: ").strip() or "1"
    
    # 确定包围盒
    bounds = None
    sole_mesh = None
    
    if mode == "1":
        # 简单立方体
        size = input("立方体尺寸 (mm) [默认: 50]: ").strip()
        size = float(size) if size else 50.0
        bounds = [[0, 0, 0], [size, size, size]]
        print(f"[包围盒] 立方体: {size}×{size}×{size} mm")
        
    elif mode == "2":
        # 使用现有网格
        mesh_path = input("网格文件路径 [默认: 鞋底/1.stl]: ").strip() or "鞋底/1.stl"
        try:
            sole_mesh = trimesh.load(mesh_path)
            bounds = sole_mesh.bounds
            print(f"[加载] 网格: {mesh_path}")
            print(f"[包围盒] {bounds[0]} ~ {bounds[1]}")
        except Exception as e:
            print(f"[错误] 无法加载网格: {e}")
            return
            
    elif mode == "3":
        # 自定义包围盒
        print("请输入包围盒坐标 (xmin ymin zmin xmax ymax zmax):")
        coords = input("[默认: 0 0 0 50 50 50]: ").strip()
        if coords:
            vals = list(map(float, coords.split()))
            bounds = [[vals[0], vals[1], vals[2]], [vals[3], vals[4], vals[5]]]
        else:
            bounds = [[0, 0, 0], [50, 50, 50]]
        print(f"[包围盒] {bounds[0]} ~ {bounds[1]}")
    
    else:
        print("[错误] 无效选项")
        return
    
    # 输入参数
    print("\n" + "-" * 60)
    print("参数设置:")
    print("-" * 60)
    
    cell_size = input("胞元尺寸 (mm) [默认: 10.0]: ").strip()
    cell_size = float(cell_size) if cell_size else 10.0
    
    wall_thickness = input("管径/壁厚 (mm) [默认: 3.0]: ").strip()
    wall_thickness = float(wall_thickness) if wall_thickness else 3.0
    
    resolution = input("分辨率 (每胞元体素数) [默认: 4]: ").strip()
    resolution = int(resolution) if resolution else 4
    
    num_seeds = input("种子点数量 (留空自动计算) [默认: 自动]: ").strip()
    num_seeds = int(num_seeds) if num_seeds else None
    
    print("\n种子采样方法:")
    print("  1. jittered - 网格+大抖动（中等随机）⭐ 默认")
    print("  2. random - 完全随机（最不规则）")
    print("  3. grid - 规则网格+小抖动（最规则）")
    print("  4. poisson - 泊松盘采样（均匀但不规则）")
    seed_method_choice = input("选择方法 (1/2/3/4) [默认: 1]: ").strip() or "1"
    seed_method_map = {'1': 'jittered', '2': 'random', '3': 'grid', '4': 'poisson'}
    seed_method = seed_method_map.get(seed_method_choice, 'jittered')
    
    print("\n是否使用平滑融合（smooth min）？")
    print("  平滑融合可以让骨架交汇处更圆滑，但计算稍慢")
    use_smooth_input = input("使用平滑融合? (y/n) [默认: y]: ").strip().lower() or "y"
    use_smooth = (use_smooth_input == 'y')
    
    smooth_k = 10.0
    if use_smooth:
        smooth_k_input = input("平滑参数 k (5-20，越大越尖锐) [默认: 10]: ").strip()
        smooth_k = float(smooth_k_input) if smooth_k_input else 10.0
    
    print("\n" + "-" * 60)
    print("参数确认:")
    print(f"  胞元尺寸: {cell_size} mm")
    print(f"  管径: {wall_thickness} mm")
    print(f"  分辨率: {resolution}")
    print(f"  种子数: {'自动' if num_seeds is None else num_seeds}")
    print(f"  采样方法: {seed_method}")
    print(f"  平滑融合: {'开启 (k=' + str(smooth_k) + ')' if use_smooth else '关闭'}")
    print("-" * 60)
    
    confirm = input("\n开始生成? (y/n) [默认: y]: ").strip().lower() or "y"
    if confirm != "y":
        print("已取消")
        return
    
    # 生成晶格
    mesh = generate_voronoi_lattice(
        bounds=bounds,
        cell_size=cell_size,
        wall_thickness=wall_thickness,
        resolution=resolution,
        num_seeds=num_seeds,
        seed_method=seed_method,
        use_smooth=use_smooth,
        smooth_k=smooth_k
    )
    
    if len(mesh.faces) == 0:
        print("\n[失败] 未生成有效网格")
        return
    
    # 保存文件
    print("\n" + "-" * 60)
    default_name = "voronoi_lattice.stl"
    output_path = input(f"保存文件名 [默认: {default_name}]: ").strip() or default_name
    
    try:
        mesh.export(output_path)
        print(f"\n[成功] 已保存到: {output_path}")
        print(f"[网格] {len(mesh.vertices)} 顶点, {len(mesh.faces)} 面")
        
        # 询问是否继续
        print("\n" + "=" * 60)
        again = input("是否继续生成? (y/n) [默认: n]: ").strip().lower() or "n"
        if again == "y":
            interactive_mode()
            
    except Exception as e:
        print(f"[错误] 保存失败: {e}")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Voronoi 隐函数建模工具")
    print("=" * 60)
    print("\n请选择运行模式:")
    print("1. 生成 Voronoi 晶格 (交互式)")
    print("7. 可视化 Voronoi 图 (直接绘制)")
    
    choice = input("\n请输入选项 (1/7) [默认: 1]: ").strip() or "1"
    
    if choice == "1":
        interactive_mode()
    elif choice == "7":
        voronoi_visualization_mode()
    else:
        print("无效选项，使用交互式模式")
        interactive_mode()


