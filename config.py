from __future__ import annotations

from pathlib import Path

# =============================================================================
# 路径配置
# =============================================================================

BASE_DIR = Path(__file__).resolve().parent

# 默认路径 - 设置为空，让用户手动选择
# 建议：
#   - 鞋底模型放在 resources/ 文件夹
#   - 晶格单元放在 resources/ 文件夹
#   - 输出结果保存到 results/ 文件夹
SOLE_STL_PATH = ""  # 鞋底模型路径（留空，需要用户选择）
LATTICE_STL_PATH = ""  # 晶格单元路径（留空，需要用户选择）
OUTPUT_STL_PATH = ""  # 输出路径（留空，需要用户选择）

# 推荐的目录结构
RESOURCES_DIR = BASE_DIR / "resources"
RESULTS_DIR = BASE_DIR / "results"

# =============================================================================
# 生成参数
# =============================================================================

# 晶格生成方法：
# - "tile_unit": 使用 lattice.stl（见 LATTICE_LAYOUT）
# - "voronoi_2_5d": 程序化 Voronoi 薄壁（几何构造）
# - "voronoi_implicit": 隐函数 Voronoi（平滑壁面，类似 Gyroid）
# - "gyroid": TPMS Gyroid 曲面（最接近参考文件，3D 体积填充）
LATTICE_METHOD: str = "gyroid"

# 仅 LATTICE_METHOD="tile_unit" 时有效：
# - "grid"：均匀缩小单元后在底面 XY 单层顺序密铺
# - "single_scaled"：单份等比缩放（见 LATTICE_SINGLE_FIT）
LATTICE_LAYOUT: str = "grid"

# 仅 LATTICE_LAYOUT="grid" 且为 True：先整体均匀缩小（不单独拉 Z），再在底面 XY 单层铺格
LATTICE_TILE_MATCH_SOLE_HEIGHT: bool = True

# 密铺前对 lattice.stl 单元整体均匀缩小（<1 越小格越密）
LATTICE_TILE_SHRINK: float = 0.28

# 仅 LATTICE_LAYOUT="single_scaled" 时有效：单份晶格与鞋底 AABB 的关系
# - "inside"：等比 min 缩放，完全落在盒内（细长鞋底往往只在中间一坨有晶格）
# - "cover"：等比 max 缩放，先大于盒再裁剪，更易沿鞋长/鞋宽铺满（推荐）
# - "stretch"：三轴独立缩放铺满盒（会改变杆夹角，仅当可接受变形时用）
LATTICE_SINGLE_FIT: str = "cover"

# 晶格单元整体缩放（1.0 = 与 STL 一致；<1 更密）
LATTICE_SCALE: float = 1.0

# 鞋底 Z 向缩放（1.0 = 不拉伸）
SOLE_HEIGHT_SCALE: float = 1.0

# =============================================================================
# Gyroid（TPMS 旋转极小曲面）参数
# =============================================================================

# Gyroid 单胞周期长度（mm）：越小结构越密，越大越疏
# 参考文件约 5~8mm 为一个周期
GYROID_CELL_SIZE: float = 4.0

# Gyroid 壁厚控制：等值面偏移量（0 = 无限薄，0.3~0.8 = 有厚度的壁）
GYROID_ISOVALUE: float = 0.3

# Gyroid 网格分辨率：每个周期内的体素数（越大越精细，越慢）
# 建议 20~40，参考文件约等效 30
GYROID_RESOLUTION: int = 30

# Gyroid 体素 mask 射线批大小（越大通常越快，但占内存更多）
GYROID_MASK_BATCH: int = 16384

# Gyroid 体素 mask 并行线程数（1=不并行，建议 2~8）
GYROID_MASK_WORKERS: int = 4

# 是否优先使用 C++ 扩展计算 mask（需先编译 cpp_mask 扩展）
GYROID_USE_CPP_MASK: bool = True

# 启用 C++ mask 的最小点数阈值（点数过小时直接走 Python）
GYROID_CPP_MASK_MIN_POINTS: int = 200000

# C++ mask 的 XY 网格分桶尺寸（越大候选三角越少，但建表稍慢）
GYROID_CPP_MASK_GRID: int = 96

# Gyroid 结果缓存（参数不变时直接复用最终晶格，二次运行速度提升明显）
GYROID_RESULT_USE_CACHE: bool = True
GYROID_RESULT_CACHE_DIR: str = str(BASE_DIR / ".cache")

# tile_unit 最终结果缓存（同鞋底/晶格/参数时直接复用最终晶格）
TILE_RESULT_USE_CACHE: bool = True
TILE_RESULT_CACHE_DIR: str = str(BASE_DIR / ".cache")

# tile_unit：是否优先使用 C++ 扩展做批量点内判定
TILE_USE_CPP_MASK: bool = True

# tile_unit：启用 C++ 判定的最小点数阈值（中心点/角点总数过小时直接走 Python）
TILE_CPP_MASK_MIN_POINTS: int = 4096

# tile_unit：C++ 判定使用的 XY 分桶尺寸
TILE_CPP_MASK_GRID: int = 96

# Gyroid 体素 mask 缓存（不影响质量，二次运行显著提速）
GYROID_MASK_USE_CACHE: bool = True
GYROID_MASK_CACHE_DIR: str = str(BASE_DIR / ".cache")

# 是否启用体素 mask（保证晶格贴合鞋底轮廓；关闭会更快但可能越界）
GYROID_USE_VOXEL_MASK: bool = True

# Voronoi：是否优先使用 C++ 扩展做批量点内判定
VORONOI_USE_CPP_MASK: bool = True

# Voronoi：启用 C++ 判定的最小点数阈值
VORONOI_CPP_MASK_MIN_POINTS: int = 2048

# Voronoi：C++ 判定使用的 XY 分桶尺寸
VORONOI_CPP_MASK_GRID: int = 96

# Voronoi 单参数（核心）：目标胞元尺度（mm）
# 值越小，分割越密、结构越细。
VORONOI_CELL_SIZE: float = 4.2

# Voronoi 杆/壁厚度（mm）
VORONOI_STRUT_THICKNESS: float = 0.55

# =============================================================================
# Voronoi Implicit（隐函数 Voronoi）参数
# =============================================================================

# Voronoi Implicit 胞元尺寸（mm）：控制种子点密度
# 值越小，结构越密；值越大，结构越疏
VORONOI_IMPLICIT_CELL_SIZE: float = 4.2

# Voronoi Implicit 壁厚参数（mm）：控制骨架杆件的粗细
# 建议设置为 cell_size 的 0.3-0.5 倍（例如 cell_size=4.2 时，wall_thickness=1.5-2.0）
# 值越大，杆越粗，孔隙率越低
VORONOI_IMPLICIT_WALL_THICKNESS: float = 1.8

# Voronoi Implicit 分辨率：每个胞元内的体素数
# 建议 15~30，值越大越精细但越慢
VORONOI_IMPLICIT_RESOLUTION: int = 20

# Voronoi Implicit 种子点密度系数：控制种子点数量
# 实际种子数 = 体积 / (cell_size^3) * density_factor
# 1.0 = 标准密度，>1.0 = 更密，<1.0 = 更疏
VORONOI_IMPLICIT_SEED_DENSITY_FACTOR: float = 1.0

# Voronoi Implicit 是否使用 C++ 加速（需编译 cpp_voronoi_implicit）
VORONOI_IMPLICIT_USE_CPP: bool = True

# Voronoi Implicit 结果缓存
VORONOI_IMPLICIT_USE_CACHE: bool = True
VORONOI_IMPLICIT_CACHE_DIR: str = str(BASE_DIR / ".cache")

# 晶格单元减面（0 = 不减面，完全保留 lattice.stl 形状；>0 则减到约该面数）
LATTICE_UNIT_TARGET_FACES: int = 0  # 默认禁用减面，保持最高精度

# 限制最多平铺的晶格单元 / Voronoi 边段数，防止意外生成超大网格
MAX_LATTICE_UNITS: int = 50000

# 平铺步长系数（<1 有重叠，更易铺满）
TILE_SPACING_FACTOR: float = 0.96

# tile_unit 平铺时预留的边界安全距离（mm），越大越不易越界
TILE_BOUNDARY_MARGIN: float = 1.0

# 是否启用 tile_unit 的补充候选格点，用于减少前掌/后跟/边缘漏铺
TILE_EDGE_FILL: bool = False

# True：三轴步长相同 = 单元包围盒平均边长 × TILE_SPACING_FACTOR（排列更均匀）
# False：按单元 X/Y/Z 外接盒分别乘系数（旧行为）
TILE_ISOTROPIC_STEP: bool = True

# True：格点只要「单元中心在鞋内」或「单元 AABB 任一角点在鞋内」就放置（贴边、补角）
# False：仅中心在鞋内（旧逻辑，离侧壁远、漏铺）
TILE_PLACE_IF_CORNER_INSIDE: bool = True

# True：若上述 contains 全失败，仍保留「晶格单元 AABB 与鞋底 AABB 相交」的格点（薄底/非水密 STL 必需）
TILE_PLACE_IF_AABB_OVERLAP: bool = True

# 密铺候选格点超过该数量时自动放大步长（避免内存爆炸）
MAX_TILE_CANDIDATES: int = 350000

# Voronoi：沿鞋底厚度方向叠加的层数（>=2 时不再是单个平面）
VORONOI_Z_LAYERS: int = 7

# Voronoi：边段上采样点数，全部需在鞋底体内（XY 平面 z=z_mid）才保留该边
VORONOI_SEGMENT_SAMPLES: int = 9

# 生成后裁剪到鞋底实体内部：
# - True 且 LATTICE_TRIM_MODE="boolean"：与鞋底做布尔交（沿鞋曲面截断，边缘杆件可不全；需 pip install manifold3d）
# - True 且 "contains"：仅保留面心在鞋内的三角（快，边界偏阶梯）
# - False：不裁剪
CLIP_LATTICE_TO_SOLE: bool = True
LATTICE_TRIM_MODE: str = "contains"  # "boolean" | "contains"

# 最终裁剪阶段：是否优先使用 C++ 扩展做批量点内判定
CLIP_USE_CPP_MASK: bool = True

# 最终裁剪阶段：启用 C++ 判定的最小点数阈值
CLIP_CPP_MASK_MIN_POINTS: int = 4096

# 最终裁剪阶段：C++ 判定使用的 XY 分桶尺寸
CLIP_CPP_MASK_GRID: int = 96

# =============================================================================
# 平面裁剪参数
# =============================================================================

# 平面裁剪模式（用于选项卡4的区域分割）：
# - False（默认）：平衡模式 - 面中心 OR 至少2个顶点在保留侧（推荐）
# - True：严格模式 - 所有顶点都在保留侧（无拖尾，但可能有空隙）
PLANE_CLIP_STRICT_MODE: bool = False

# =============================================================================
# 调试
# =============================================================================

VERBOSE: bool = True

