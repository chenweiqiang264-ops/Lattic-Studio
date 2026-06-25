"""
八叉树 SDF 存储
用于高效存储和查询稀疏 SDF 数据，大幅减少内存占用
"""

import numpy as np
from typing import Tuple, Optional, Callable
import time

# 尝试导入 C++ 加速模块
try:
    from core.cpp.cpp_octree_sdf import build_octree_from_periodic_sdf as cpp_build_octree
    CPP_OCTREE_AVAILABLE = True
    print("[加速] C++ 八叉树构建模块已加载")
except ImportError:
    CPP_OCTREE_AVAILABLE = False
    print("[提示] C++ 八叉树模块未找到，将使用 Python 版本（较慢）")
    print("       运行 'python core/cpp/build_octree_sdf.py build_ext --inplace' 来编译")


class OctreeNode:
    """八叉树节点"""
    
    def __init__(self, center: np.ndarray, size: float, depth: int = 0):
        self.center = np.array(center, dtype=np.float32)
        self.size = size
        self.depth = depth
        
        # 子节点（8个）或 None（叶节点）
        self.children = None
        
        # 叶节点数据
        self.is_leaf = True
        self.sdf_value = None  # 如果是均匀区域，存储单个值
        self.sdf_grid = None   # 如果是复杂区域，存储小网格
        
        # 统计信息
        self.min_sdf = float('inf')
        self.max_sdf = float('-inf')
    
    def is_uniform(self, threshold=0.01):
        """检查区域是否均匀（SDF变化小）"""
        if self.sdf_grid is not None:
            return (self.max_sdf - self.min_sdf) < threshold
        return True
    
    def subdivide(self):
        """细分为8个子节点"""
        if not self.is_leaf:
            return
        
        self.is_leaf = False
        self.children = []
        
        half_size = self.size / 2
        quarter_size = self.size / 4
        
        # 8个子节点的偏移
        offsets = [
            [-1, -1, -1], [1, -1, -1], [-1, 1, -1], [1, 1, -1],
            [-1, -1, 1], [1, -1, 1], [-1, 1, 1], [1, 1, 1]
        ]
        
        for offset in offsets:
            child_center = self.center + np.array(offset, dtype=np.float32) * quarter_size
            child = OctreeNode(child_center, half_size, self.depth + 1)
            self.children.append(child)
    
    def get_memory_usage(self):
        """估算内存占用（字节）"""
        base_size = 64  # 基础结构
        
        if self.is_leaf:
            if self.sdf_value is not None:
                return base_size + 4  # 单个float32
            elif self.sdf_grid is not None:
                return base_size + self.sdf_grid.nbytes
            else:
                return base_size
        else:
            total = base_size
            for child in self.children:
                total += child.get_memory_usage()
            return total


class OctreeSDF:
    """八叉树 SDF 存储"""
    
    def __init__(self, bbox_min: np.ndarray, bbox_max: np.ndarray, max_depth: int = 8):
        """
        参数:
            bbox_min: 包围盒最小点
            bbox_max: 包围盒最大点
            max_depth: 最大深度（深度越大，精度越高，但内存越多）
        """
        self.bbox_min = np.array(bbox_min, dtype=np.float32)
        self.bbox_max = np.array(bbox_max, dtype=np.float32)
        self.max_depth = max_depth
        
        # 创建根节点
        center = (self.bbox_min + self.bbox_max) / 2
        size = np.max(self.bbox_max - self.bbox_min)
        self.root = OctreeNode(center, size, depth=0)
        
        # 统计信息
        self.num_nodes = 0
        self.num_leaf_nodes = 0
        self.memory_saved_ratio = 0.0
    
    def build_from_function(
        self, 
        sdf_func: Callable[[float, float, float], float],
        leaf_resolution: int = 4,
        uniform_threshold: float = 0.01,
        verbose: bool = True
    ):
        """
        从 SDF 函数构建八叉树
        
        参数:
            sdf_func: SDF 查询函数 f(x, y, z) -> float
            leaf_resolution: 叶节点内部网格分辨率（如4表示4³=64个采样点）
            uniform_threshold: 均匀性阈值（SDF变化小于此值视为均匀）
            verbose: 是否显示进度
        """
        if verbose:
            print(f"  构建八叉树 SDF（最大深度: {self.max_depth}, 叶节点分辨率: {leaf_resolution}³）...")
        
        t_start = time.time()
        
        self._build_node(self.root, sdf_func, leaf_resolution, uniform_threshold)
        
        # 统计
        self._count_nodes(self.root)
        
        t_end = time.time()
        
        if verbose:
            print(f"  ✓ 八叉树构建完成 (耗时: {t_end - t_start:.1f}s)")
            print(f"  节点总数: {self.num_nodes:,}")
            print(f"  叶节点数: {self.num_leaf_nodes:,}")
            
            # 估算内存占用
            octree_memory = self.root.get_memory_usage()
            
            # 对比完整网格的内存
            full_resolution = 2 ** self.max_depth * leaf_resolution
            full_memory = full_resolution ** 3 * 4  # float32
            
            self.memory_saved_ratio = 1.0 - (octree_memory / full_memory)
            
            print(f"  八叉树内存: {octree_memory / (1024**2):.1f} MB")
            print(f"  完整网格内存: {full_memory / (1024**2):.1f} MB")
            print(f"  节省内存: {self.memory_saved_ratio * 100:.1f}%")
    
    def _build_node(
        self, 
        node: OctreeNode, 
        sdf_func: Callable,
        leaf_resolution: int,
        uniform_threshold: float
    ):
        """递归构建节点"""
        # 在节点范围内采样
        half = node.size / 2
        sample_points = [
            node.center + np.array([dx, dy, dz], dtype=np.float32) * half
            for dx in [-1, 0, 1]
            for dy in [-1, 0, 1]
            for dz in [-1, 0, 1]
        ]
        
        # 查询 SDF 值
        sdf_values = [sdf_func(*p) for p in sample_points]
        node.min_sdf = min(sdf_values)
        node.max_sdf = max(sdf_values)
        
        # 判断是否需要细分
        sdf_range = node.max_sdf - node.min_sdf
        
        # 如果区域均匀或达到最大深度，创建叶节点
        if sdf_range < uniform_threshold or node.depth >= self.max_depth:
            # 叶节点
            if sdf_range < uniform_threshold:
                # 均匀区域：只存储一个值
                node.sdf_value = np.mean(sdf_values)
            else:
                # 复杂区域：存储小网格
                node.sdf_grid = self._sample_grid(node, sdf_func, leaf_resolution)
            return
        
        # 细分
        node.subdivide()
        for child in node.children:
            self._build_node(child, sdf_func, leaf_resolution, uniform_threshold)
    
    def _sample_grid(
        self, 
        node: OctreeNode, 
        sdf_func: Callable,
        resolution: int
    ) -> np.ndarray:
        """在节点内采样网格"""
        half = node.size / 2
        
        # 创建采样点
        x = np.linspace(-half, half, resolution, dtype=np.float32)
        y = np.linspace(-half, half, resolution, dtype=np.float32)
        z = np.linspace(-half, half, resolution, dtype=np.float32)
        
        grid = np.zeros((resolution, resolution, resolution), dtype=np.float32)
        
        for i, dx in enumerate(x):
            for j, dy in enumerate(y):
                for k, dz in enumerate(z):
                    pos = node.center + np.array([dx, dy, dz], dtype=np.float32)
                    grid[i, j, k] = sdf_func(*pos)
        
        return grid
    
    def _count_nodes(self, node: OctreeNode):
        """统计节点数"""
        self.num_nodes += 1
        
        if node.is_leaf:
            self.num_leaf_nodes += 1
        else:
            for child in node.children:
                self._count_nodes(child)
    
    def query(self, x: float, y: float, z: float) -> float:
        """查询指定位置的 SDF 值"""
        pos = np.array([x, y, z], dtype=np.float32)
        return self._query_node(self.root, pos)
    
    def _query_node(self, node: OctreeNode, pos: np.ndarray) -> float:
        """在节点中查询"""
        if node.is_leaf:
            if node.sdf_value is not None:
                # 均匀区域：返回单个值
                return node.sdf_value
            elif node.sdf_grid is not None:
                # 复杂区域：三线性插值
                return self._interpolate_grid(node, pos)
            else:
                # 空节点
                return 0.0
        
        # 找到包含该点的子节点
        child_idx = self._get_child_index(node, pos)
        return self._query_node(node.children[child_idx], pos)
    
    def _get_child_index(self, node: OctreeNode, pos: np.ndarray) -> int:
        """获取包含指定位置的子节点索引"""
        idx = 0
        if pos[0] > node.center[0]:
            idx |= 1
        if pos[1] > node.center[1]:
            idx |= 2
        if pos[2] > node.center[2]:
            idx |= 4
        return idx
    
    def _interpolate_grid(self, node: OctreeNode, pos: np.ndarray) -> float:
        """在节点网格中三线性插值"""
        grid = node.sdf_grid
        resolution = grid.shape[0]
        
        # 转换为网格坐标
        half = node.size / 2
        local_pos = pos - node.center + half  # [0, size]
        grid_pos = local_pos / node.size * (resolution - 1)  # [0, resolution-1]
        
        # 限制在有效范围内
        grid_pos = np.clip(grid_pos, 0, resolution - 1)
        
        # 三线性插值
        i0, j0, k0 = np.floor(grid_pos).astype(int)
        i1, j1, k1 = np.ceil(grid_pos).astype(int)
        
        # 限制索引
        i0, i1 = np.clip([i0, i1], 0, resolution - 1)
        j0, j1 = np.clip([j0, j1], 0, resolution - 1)
        k0, k1 = np.clip([k0, k1], 0, resolution - 1)
        
        # 插值权重
        fx = grid_pos[0] - i0
        fy = grid_pos[1] - j0
        fz = grid_pos[2] - k0
        
        # 8个角点的值
        c000 = grid[i0, j0, k0]
        c001 = grid[i0, j0, k1]
        c010 = grid[i0, j1, k0]
        c011 = grid[i0, j1, k1]
        c100 = grid[i1, j0, k0]
        c101 = grid[i1, j0, k1]
        c110 = grid[i1, j1, k0]
        c111 = grid[i1, j1, k1]
        
        # 三线性插值
        c00 = c000 * (1 - fx) + c100 * fx
        c01 = c001 * (1 - fx) + c101 * fx
        c10 = c010 * (1 - fx) + c110 * fx
        c11 = c011 * (1 - fx) + c111 * fx
        
        c0 = c00 * (1 - fy) + c10 * fy
        c1 = c01 * (1 - fy) + c11 * fy
        
        return c0 * (1 - fz) + c1 * fz
    
    def to_dense_grid(self, resolution: Tuple[int, int, int]) -> np.ndarray:
        """
        转换为密集网格（用于 Marching Cubes）
        
        参数:
            resolution: 目标分辨率 (nx, ny, nz)
        
        返回:
            dense_grid: 密集 SDF 网格
        """
        print(f"  将八叉树转换为密集网格 ({resolution[0]}×{resolution[1]}×{resolution[2]})...")
        
        t_start = time.time()
        
        nx, ny, nz = resolution
        grid = np.zeros(resolution, dtype=np.float32)
        
        # 计算采样点
        x = np.linspace(self.bbox_min[0], self.bbox_max[0], nx)
        y = np.linspace(self.bbox_min[1], self.bbox_max[1], ny)
        z = np.linspace(self.bbox_min[2], self.bbox_max[2], nz)
        
        # 查询每个点
        total_points = nx * ny * nz
        progress_step = max(1, total_points // 20)
        
        idx = 0
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    grid[i, j, k] = self.query(x[i], y[j], z[k])
                    
                    idx += 1
                    if idx % progress_step == 0:
                        progress = idx / total_points * 100
                        print(f"    进度: {progress:.0f}%", end='\r')
        
        t_end = time.time()
        print(f"    进度: 100% (耗时: {t_end - t_start:.1f}s)          ")
        
        return grid
    
    def save(self, filename: str):
        """保存八叉树到文件"""
        import pickle
        with open(filename, 'wb') as f:
            pickle.dump(self, f)
        print(f"  ✓ 八叉树已保存: {filename}")
    
    @staticmethod
    def load(filename: str):
        """从文件加载八叉树"""
        import pickle
        with open(filename, 'rb') as f:
            octree = pickle.load(f)
        print(f"  ✓ 八叉树已加载: {filename}")
        return octree


def build_octree_from_grid(
    sdf_grid: np.ndarray,
    bbox_min: np.ndarray,
    bbox_max: np.ndarray,
    max_depth: int = 8,
    leaf_resolution: int = 4,
    uniform_threshold: float = 0.01
) -> OctreeSDF:
    """
    从现有的密集网格构建八叉树
    
    参数:
        sdf_grid: 密集 SDF 网格
        bbox_min: 包围盒最小点
        bbox_max: 包围盒最大点
        max_depth: 最大深度
        leaf_resolution: 叶节点分辨率
        uniform_threshold: 均匀性阈值
    
    返回:
        octree: 八叉树 SDF
    """
    # 创建查询函数
    resolution = np.array(sdf_grid.shape)
    spacing = (bbox_max - bbox_min) / (resolution - 1)
    
    def sdf_func(x, y, z):
        # 转换为网格索引
        pos = np.array([x, y, z])
        indices = ((pos - bbox_min) / spacing).astype(int)
        indices = np.clip(indices, 0, resolution - 1)
        return sdf_grid[indices[0], indices[1], indices[2]]
    
    # 构建八叉树
    octree = OctreeSDF(bbox_min, bbox_max, max_depth)
    octree.build_from_function(sdf_func, leaf_resolution, uniform_threshold)
    
    return octree



def deserialize_node_from_cpp(node_dict: dict) -> OctreeNode:
    """
    从 C++ 返回的字典反序列化节点
    
    参数:
        node_dict: C++ 返回的节点字典
    
    返回:
        node: OctreeNode 对象
    """
    center = node_dict['center']
    size = node_dict['size']
    depth = node_dict['depth']
    
    node = OctreeNode(center, size, depth)
    node.is_leaf = node_dict['is_leaf']
    node.min_sdf = node_dict['min_sdf']
    node.max_sdf = node_dict['max_sdf']
    
    if node.is_leaf:
        node_type = node_dict['type']
        if node_type == 'uniform':
            # 均匀区域
            node.sdf_value = node_dict['sdf_value']
        elif node_type == 'grid':
            # 复杂区域
            node.sdf_grid = node_dict['sdf_grid']
    else:
        # 非叶节点：递归反序列化子节点
        node.children = []
        for child_dict in node_dict['children']:
            node.children.append(deserialize_node_from_cpp(child_dict))
    
    return node


def build_octree_from_periodic_sdf_cpp(
    unit_sdf: np.ndarray,
    target_bbox_min: np.ndarray,
    target_bbox_max: np.ndarray,
    cell_size: float,
    unit_bbox_min: np.ndarray,
    unit_resolution: int,
    max_depth: int = 8,
    leaf_resolution: int = 4,
    uniform_threshold: float = 0.01,
    verbose: bool = True
) -> OctreeSDF:
    """
    使用 C++ 加速从周期性 SDF 构建八叉树
    
    参数:
        unit_sdf: 单个晶胞的 SDF
        target_bbox_min: 目标包围盒最小点
        target_bbox_max: 目标包围盒最大点
        cell_size: 晶胞尺寸
        unit_bbox_min: 晶胞包围盒最小点
        unit_resolution: 晶胞 SDF 分辨率
        max_depth: 最大深度
        leaf_resolution: 叶节点分辨率
        uniform_threshold: 均匀性阈值
        verbose: 是否显示进度
    
    返回:
        octree: 八叉树 SDF
    """
    if not CPP_OCTREE_AVAILABLE:
        raise ImportError("C++ 八叉树模块未加载，请先编译")
    
    # 调用 C++ 函数
    result_dict = cpp_build_octree(
        unit_sdf.astype(np.float32),
        target_bbox_min.astype(np.float32),
        target_bbox_max.astype(np.float32),
        float(cell_size),
        unit_bbox_min.astype(np.float32),
        int(unit_resolution),
        int(max_depth),
        int(leaf_resolution),
        float(uniform_threshold),
        bool(verbose)
    )
    
    # 创建 OctreeSDF 对象
    octree = OctreeSDF(target_bbox_min, target_bbox_max, max_depth)
    
    # 反序列化根节点
    octree.root = deserialize_node_from_cpp(result_dict['root'])
    
    # 设置统计信息
    octree.num_nodes = result_dict['total_nodes']
    octree.num_leaf_nodes = result_dict['leaf_nodes']
    
    # 计算内存节省比例
    octree_memory = result_dict['memory_bytes']
    
    # 估算完整网格的内存
    size = target_bbox_max - target_bbox_min
    full_resolution = 2 ** max_depth * leaf_resolution
    full_memory = full_resolution ** 3 * 4  # float32
    
    octree.memory_saved_ratio = 1.0 - (octree_memory / full_memory)
    
    if verbose:
        print(f"  八叉树内存: {octree_memory / (1024**2):.1f} MB")
        print(f"  完整网格内存: {full_memory / (1024**2):.1f} MB")
        print(f"  节省内存: {octree.memory_saved_ratio * 100:.1f}%")
    
    return octree
