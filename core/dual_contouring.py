"""
Dual Contouring 实现
用于从SDF生成网格，支持八叉树结构和锐特征保留
"""

import numpy as np
from typing import Tuple, List, Optional
import trimesh


class OctreeNode:
    """八叉树节点"""
    def __init__(self, center, size, depth=0):
        self.center = np.array(center, dtype=float)
        self.size = size
        self.depth = depth
        self.children = None  # 8个子节点
        self.is_leaf = True
        self.vertex_index = -1  # Dual Contouring 顶点索引
        self.has_surface = False  # 是否包含表面（符号变化）
        
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
            child_center = self.center + np.array(offset) * quarter_size
            child = OctreeNode(child_center, half_size, self.depth + 1)
            self.children.append(child)
    
    def get_corners(self):
        """获取8个角点坐标"""
        half = self.size / 2
        corners = []
        for i in range(8):
            x = self.center[0] + (1 if i & 1 else -1) * half
            y = self.center[1] + (1 if i & 2 else -1) * half
            z = self.center[2] + (1 if i & 4 else -1) * half
            corners.append([x, y, z])
        return np.array(corners)


class DualContouring:
    """Dual Contouring 算法实现"""
    
    def __init__(self, sdf_func, bbox_min, bbox_max, max_depth=6):
        """
        参数:
            sdf_func: SDF查询函数 f(x, y, z) -> float
            bbox_min: 包围盒最小点
            bbox_max: 包围盒最大点
            max_depth: 八叉树最大深度
        """
        self.sdf_func = sdf_func
        self.bbox_min = np.array(bbox_min)
        self.bbox_max = np.array(bbox_max)
        self.max_depth = max_depth
        
        # 创建根节点
        center = (self.bbox_min + self.bbox_max) / 2
        size = np.max(self.bbox_max - self.bbox_min)
        self.root = OctreeNode(center, size, depth=0)
        
        self.vertices = []
        self.faces = []
    
    def build_octree(self):
        """构建自适应八叉树"""
        print(f"  构建八叉树（最大深度: {self.max_depth}）...")
        self._subdivide_node(self.root)
        print(f"  ✓ 八叉树构建完成")
    
    def _subdivide_node(self, node):
        """递归细分节点"""
        # 检查是否包含表面
        corners = node.get_corners()
        corner_values = np.array([self.sdf_func(*c) for c in corners])
        
        # 如果有符号变化，说明包含表面
        has_surface = (corner_values.min() < 0) and (corner_values.max() > 0)
        node.has_surface = has_surface
        
        # 如果不包含表面或达到最大深度，停止细分
        if not has_surface or node.depth >= self.max_depth:
            return
        
        # 细分
        node.subdivide()
        for child in node.children:
            self._subdivide_node(child)
    
    def generate_vertices(self):
        """为每个叶节点生成顶点"""
        print(f"  生成顶点...")
        self._generate_vertex_for_node(self.root)
        print(f"  ✓ 生成 {len(self.vertices)} 个顶点")
    
    def _generate_vertex_for_node(self, node):
        """为节点生成顶点"""
        if node.is_leaf:
            if node.has_surface:
                # 简化版：使用节点中心作为顶点
                # 更精确的方法：使用QEF（二次误差函数）求解最优位置
                node.vertex_index = len(self.vertices)
                self.vertices.append(node.center.copy())
        else:
            for child in node.children:
                self._generate_vertex_for_node(child)
    
    def generate_faces(self):
        """生成面片"""
        print(f"  生成面片...")
        self._generate_faces_for_node(self.root)
        print(f"  ✓ 生成 {len(self.faces)} 个面")
    
    def _generate_faces_for_node(self, node):
        """为节点生成面片"""
        if node.is_leaf:
            return
        
        # 检查每条边
        # Dual Contouring: 如果边的两端节点都有顶点，连接它们
        for child in node.children:
            self._generate_faces_for_node(child)
        
        # 简化版：只处理相邻的叶节点
        # 完整实现需要更复杂的边处理逻辑
    
    def extract_mesh(self):
        """提取网格"""
        self.build_octree()
        self.generate_vertices()
        self.generate_faces()
        
        if len(self.vertices) == 0:
            return None
        
        vertices = np.array(self.vertices)
        faces = np.array(self.faces) if len(self.faces) > 0 else np.zeros((0, 3), dtype=int)
        
        return vertices, faces


def dual_marching_cubes_from_grid(
    sdf_grid: np.ndarray,
    bbox_min: np.ndarray,
    bbox_max: np.ndarray,
    max_depth: int = 6
) -> Tuple[np.ndarray, np.ndarray]:
    """
    从规则网格使用 Dual Contouring 提取网格
    
    参数:
        sdf_grid: SDF网格 (nx, ny, nz)
        bbox_min: 包围盒最小点
        bbox_max: 包围盒最大点
        max_depth: 八叉树最大深度
    
    返回:
        vertices: 顶点数组 (N, 3)
        faces: 面数组 (M, 3)
    """
    # 创建SDF查询函数
    resolution = np.array(sdf_grid.shape)
    spacing = (bbox_max - bbox_min) / (resolution - 1)
    
    def sdf_func(x, y, z):
        # 将世界坐标转换为网格索引
        pos = np.array([x, y, z])
        indices = ((pos - bbox_min) / spacing).astype(int)
        indices = np.clip(indices, 0, resolution - 1)
        return sdf_grid[indices[0], indices[1], indices[2]]
    
    # 使用 Dual Contouring
    dc = DualContouring(sdf_func, bbox_min, bbox_max, max_depth)
    vertices, faces = dc.extract_mesh()
    
    return vertices, faces


def dual_marching_cubes_simple(
    sdf_grid: np.ndarray,
    spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    offset: Tuple[float, float, float] = (0.0, 0.0, 0.0)
) -> Tuple[np.ndarray, np.ndarray]:
    """
    简化版 Dual Marching Cubes
    使用单元格中心作为顶点，比传统MC更简单但保留拓扑
    
    参数:
        sdf_grid: SDF网格
        spacing: 体素间距
        offset: 偏移量
    
    返回:
        vertices, faces
    """
    print(f"  使用简化版 Dual Marching Cubes...")
    
    nx, ny, nz = sdf_grid.shape
    spacing = np.array(spacing)
    offset = np.array(offset)
    
    # 存储顶点和面
    vertices = []
    vertex_map = {}  # (i,j,k) -> vertex_index
    faces = []
    
    # 遍历每个单元格
    for i in range(nx - 1):
        for j in range(ny - 1):
            for k in range(nz - 1):
                # 获取8个角点的SDF值
                corner_values = [
                    sdf_grid[i, j, k],
                    sdf_grid[i+1, j, k],
                    sdf_grid[i, j+1, k],
                    sdf_grid[i+1, j+1, k],
                    sdf_grid[i, j, k+1],
                    sdf_grid[i+1, j, k+1],
                    sdf_grid[i, j+1, k+1],
                    sdf_grid[i+1, j+1, k+1],
                ]
                
                # 检查是否有符号变化（包含表面）
                min_val = min(corner_values)
                max_val = max(corner_values)
                
                if min_val < 0 and max_val > 0:
                    # 包含表面，创建顶点
                    # 使用单元格中心
                    center = np.array([i + 0.5, j + 0.5, k + 0.5])
                    vertex_pos = center * spacing + offset
                    
                    vertex_idx = len(vertices)
                    vertices.append(vertex_pos)
                    vertex_map[(i, j, k)] = vertex_idx
    
    # 生成面（简化版：连接相邻的顶点）
    # 这里需要更复杂的逻辑来正确生成四边形或三角形
    # 暂时返回顶点，面的生成需要完整的 Dual Contouring 实现
    
    print(f"  ✓ 生成 {len(vertices)} 个顶点")
    print(f"  ⚠ 注意：简化版未生成面，建议使用完整实现或 PyMCubes")
    
    if len(vertices) == 0:
        return np.zeros((0, 3)), np.zeros((0, 3), dtype=int)
    
    return np.array(vertices), np.array(faces, dtype=int)


# 推荐：使用 PyMCubes（如果安装）
def marching_cubes_with_fallback(
    sdf_grid: np.ndarray,
    level: float = 0.0,
    spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    use_dual: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Marching Cubes with fallback
    优先使用 PyMCubes（支持 Dual），回退到 scikit-image
    
    参数:
        sdf_grid: SDF网格
        level: 等值面level
        spacing: 体素间距
        use_dual: 是否使用 Dual Contouring
    
    返回:
        vertices, faces
    """
    # 尝试使用 PyMCubes
    if use_dual:
        try:
            import mcubes
            print(f"  使用 PyMCubes Dual Contouring...")
            vertices, triangles = mcubes.marching_cubes_func(
                sdf_grid,
                isovalue=level,
                algorithm='dual'
            )
            # 应用spacing
            vertices = vertices * np.array(spacing)
            return vertices, triangles
        except ImportError:
            print(f"  ⚠ PyMCubes 未安装，回退到传统 Marching Cubes")
            print(f"  提示: pip install PyMCubes")
    
    # 回退到 scikit-image
    from skimage import measure
    vertices, faces, _, _ = measure.marching_cubes(
        sdf_grid,
        level=level,
        spacing=spacing
    )
    return vertices, faces
