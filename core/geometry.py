from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
import time
from multiprocessing.pool import ThreadPool

import numpy as np
import trimesh
from scipy.spatial import Voronoi

from config import (
    SOLE_STL_PATH,
    LATTICE_STL_PATH,
    LATTICE_METHOD,
    LATTICE_LAYOUT,
    LATTICE_SINGLE_FIT,
    LATTICE_TILE_MATCH_SOLE_HEIGHT,
    LATTICE_TILE_SHRINK,
    LATTICE_SCALE,
    SOLE_HEIGHT_SCALE,
    GYROID_CELL_SIZE,
    GYROID_ISOVALUE,
    GYROID_RESOLUTION,
    GYROID_MASK_BATCH,
    GYROID_MASK_WORKERS,
    GYROID_USE_CPP_MASK,
    GYROID_CPP_MASK_MIN_POINTS,
    GYROID_CPP_MASK_GRID,
    GYROID_MASK_USE_CACHE,
    GYROID_MASK_CACHE_DIR,
    GYROID_RESULT_USE_CACHE,
    GYROID_RESULT_CACHE_DIR,
    GYROID_USE_VOXEL_MASK,
    TILE_RESULT_USE_CACHE,
    TILE_RESULT_CACHE_DIR,
    TILE_USE_CPP_MASK,
    TILE_CPP_MASK_MIN_POINTS,
    TILE_CPP_MASK_GRID,
    VORONOI_USE_CPP_MASK,
    VORONOI_CPP_MASK_MIN_POINTS,
    VORONOI_CPP_MASK_GRID,
    VORONOI_CELL_SIZE,
    VORONOI_STRUT_THICKNESS,
    VORONOI_Z_LAYERS,
    VORONOI_SEGMENT_SAMPLES,
    VORONOI_IMPLICIT_CELL_SIZE,
    VORONOI_IMPLICIT_WALL_THICKNESS,
    VORONOI_IMPLICIT_RESOLUTION,
    VORONOI_IMPLICIT_SEED_DENSITY_FACTOR,
    VORONOI_IMPLICIT_USE_CPP,
    VORONOI_IMPLICIT_USE_CACHE,
    VORONOI_IMPLICIT_CACHE_DIR,
    LATTICE_UNIT_TARGET_FACES,
    MAX_LATTICE_UNITS,
    TILE_SPACING_FACTOR,
    TILE_BOUNDARY_MARGIN,
    TILE_EDGE_FILL,
    TILE_PLACE_IF_CORNER_INSIDE,
    TILE_PLACE_IF_AABB_OVERLAP,
    MAX_TILE_CANDIDATES,
    TILE_ISOTROPIC_STEP,
    CLIP_LATTICE_TO_SOLE,
    LATTICE_TRIM_MODE,
    CLIP_USE_CPP_MASK,
    CLIP_CPP_MASK_MIN_POINTS,
    CLIP_CPP_MASK_GRID,
    VERBOSE,
)


def _log(message: str) -> None:
    if VERBOSE:
        print(message)


@dataclass(frozen=True)
class MeshStats:
    vertices: int
    faces: int


def _stats(mesh: trimesh.Trimesh) -> MeshStats:
    return MeshStats(vertices=int(len(mesh.vertices)), faces=int(len(mesh.faces)))


def _gyroid_runtime_hint(n_voxels: int, mask_cache_hit: bool) -> str:
    """根据体素规模和缓存命中情况给出粗略耗时提示。"""
    if n_voxels < 2_000_000:
        scale = "小"
    elif n_voxels < 8_000_000:
        scale = "中"
    else:
        scale = "大"

    if mask_cache_hit:
        return f"体素规模={scale}（{n_voxels:,}），命中mask缓存，预计主要耗时在 Marching Cubes"
    return f"体素规模={scale}（{n_voxels:,}），未命中mask缓存，预计本次为慢速冷启动"


_FILE_HASH_CACHE: dict[str, str] = {}


def _file_content_sha1(path: str) -> str:
    abs_path = os.path.abspath(path)
    cached = _FILE_HASH_CACHE.get(abs_path)
    if cached is not None:
        return cached
    try:
        h = hashlib.sha1()
        with open(abs_path, "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        digest = h.hexdigest()[:16]
    except OSError:
        digest = "missing"
    _FILE_HASH_CACHE[abs_path] = digest
    return digest


def _runtime_code_signature() -> str:
    return _file_content_sha1(__file__)


def _gyroid_mask_cache_path(
    sole_mesh: trimesh.Trimesh,
    nx: int,
    ny: int,
    nz: int,
    cell_size: float,
    isovalue: float,
    voxel_size: float,
) -> str:
    sb = np.asarray(sole_mesh.bounds, dtype=np.float64)
    key = (
        f"b={sb.round(6).tolist()}|n={nx},{ny},{nz}|"
        f"cell={cell_size:.6f}|iso={isovalue:.6f}|vox={voxel_size:.8f}"
    )
    h = hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]
    os.makedirs(GYROID_MASK_CACHE_DIR, exist_ok=True)
    return os.path.join(GYROID_MASK_CACHE_DIR, f"gyroid_mask_{h}.npy")


def _tile_result_cache_path(
    sole_path: str,
    lattice_path: str,
    sole_mesh: trimesh.Trimesh,
) -> str:
    sole_abs = os.path.abspath(sole_path)
    lattice_abs = os.path.abspath(lattice_path)
    try:
        sole_mtime = os.path.getmtime(sole_abs)
    except OSError:
        sole_mtime = -1.0
    try:
        lattice_mtime = os.path.getmtime(lattice_abs)
    except OSError:
        lattice_mtime = -1.0
    sb = np.asarray(sole_mesh.bounds, dtype=np.float64)
    sole_sig = _file_content_sha1(sole_abs)
    lattice_sig = _file_content_sha1(lattice_abs)
    code_sig = _runtime_code_signature()
    key = (
        f"sole={sole_abs}|sole_mtime={sole_mtime:.6f}|sole_sig={sole_sig}|"
        f"lattice={lattice_abs}|lattice_mtime={lattice_mtime:.6f}|lattice_sig={lattice_sig}|"
        f"code_sig={code_sig}|b={sb.round(6).tolist()}|layout={LATTICE_LAYOUT}|single_fit={LATTICE_SINGLE_FIT}|"
        f"match_h={int(bool(LATTICE_TILE_MATCH_SOLE_HEIGHT))}|shrink={LATTICE_TILE_SHRINK:.6f}|"
        f"scale={LATTICE_SCALE:.6f}|unit_faces={int(LATTICE_UNIT_TARGET_FACES)}|"
        f"spacing={TILE_SPACING_FACTOR:.6f}|margin={TILE_BOUNDARY_MARGIN:.6f}|"
        f"edge_fill={int(bool(TILE_EDGE_FILL))}|corner_inside={int(bool(TILE_PLACE_IF_CORNER_INSIDE))}|"
        f"aabb_overlap={int(bool(TILE_PLACE_IF_AABB_OVERLAP))}|isotropic={int(bool(TILE_ISOTROPIC_STEP))}|"
        f"clip={int(bool(CLIP_LATTICE_TO_SOLE))}|trim={LATTICE_TRIM_MODE}"
    )
    h = hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]
    os.makedirs(TILE_RESULT_CACHE_DIR, exist_ok=True)
    return os.path.join(TILE_RESULT_CACHE_DIR, f"tile_result_{h}.npz")


def _load_cached_mesh_npz(path: str) -> trimesh.Trimesh | None:
    try:
        z = np.load(path)
        v = np.asarray(z["vertices"], dtype=np.float64)
        f = np.asarray(z["faces"], dtype=np.int64)
        if v.ndim != 2 or f.ndim != 2 or v.shape[1] != 3 or f.shape[1] != 3:
            return None
        return trimesh.Trimesh(vertices=v, faces=f, process=False)
    except Exception:
        return None


def _save_cached_mesh_npz(path: str, mesh: trimesh.Trimesh, metadata: dict | None = None) -> None:
    """保存缓存网格，同时保存参数元数据到 .json 文件"""
    np.savez_compressed(
        path,
        vertices=np.asarray(mesh.vertices, dtype=np.float32),
        faces=np.asarray(mesh.faces, dtype=np.int32),
    )
    
    # 保存参数元数据
    if metadata is not None:
        import json
        from datetime import datetime
        
        json_path = path.replace('.npz', '.json')
        metadata_with_stats = {
            'cached_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'vertices': int(len(mesh.vertices)),
            'faces': int(len(mesh.faces)),
            'file_size_mb': round(os.path.getsize(path) / (1024 * 1024), 2) if os.path.exists(path) else 0,
            'parameters': metadata,
        }
        try:
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(metadata_with_stats, f, indent=2, ensure_ascii=False)
            _log(f"[cache] 已保存参数记录: {json_path}")
        except Exception as e:
            _log(f"[cache] 保存参数记录失败: {e}")


def _tile_region_cache_path(
    sole_mesh: trimesh.Trimesh,
    lattice_path: str,
    shrink: float,
    spacing: float,
    margin: float,
    bounds: np.ndarray = None,
    unit_target_faces: int = 0,
) -> str:
    """为选项卡3/4的区域晶格生成缓存路径（基于网格哈希和参数）"""
    lattice_abs = os.path.abspath(lattice_path)
    try:
        lattice_mtime = os.path.getmtime(lattice_abs)
    except OSError:
        lattice_mtime = -1.0
    
    # 使用鞋底网格的哈希值（顶点+面）
    sole_verts = np.asarray(sole_mesh.vertices, dtype=np.float32)
    sole_faces = np.asarray(sole_mesh.faces, dtype=np.int32)
    sole_hash = hashlib.sha1(sole_verts.tobytes() + sole_faces.tobytes()).hexdigest()[:12]
    
    sb = np.asarray(sole_mesh.bounds, dtype=np.float64)
    lattice_sig = _file_content_sha1(lattice_abs)
    code_sig = _runtime_code_signature()
    
    # 包含 bounds 参数到缓存键中
    # 如果 bounds 为 None，使用鞋底的 bounds
    # 如果 bounds 不为 None，使用自定义 bounds（区域包围盒）
    if bounds is None:
        bounds_key = f"bounds=None|sb={sb.round(6).tolist()}"
    else:
        bounds_arr = np.asarray(bounds, dtype=np.float64)
        bounds_key = f"bounds={bounds_arr.round(6).tolist()}"
    
    # 使用传入的 unit_target_faces，如果为0则使用全局配置
    target_faces = unit_target_faces if unit_target_faces > 0 else LATTICE_UNIT_TARGET_FACES
    
    key = (
        f"sole_hash={sole_hash}|{bounds_key}|"
        f"lattice={lattice_abs}|lattice_mtime={lattice_mtime:.6f}|lattice_sig={lattice_sig}|"
        f"code_sig={code_sig}|shrink={shrink:.6f}|"
        f"spacing={spacing:.6f}|margin={margin:.6f}|"
        f"unit_faces={int(target_faces)}|"
        f"edge_fill={int(bool(TILE_EDGE_FILL))}|corner_inside={int(bool(TILE_PLACE_IF_CORNER_INSIDE))}|"
        f"aabb_overlap={int(bool(TILE_PLACE_IF_AABB_OVERLAP))}|isotropic={int(bool(TILE_ISOTROPIC_STEP))}"
    )
    h = hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]
    os.makedirs(TILE_RESULT_CACHE_DIR, exist_ok=True)
    return os.path.join(TILE_RESULT_CACHE_DIR, f"tile_region_{h}.npz")


def generate_tile_lattice_with_cache(
    sole_mesh: trimesh.Trimesh,
    lattice_path: str,
    shrink: float,
    spacing: float,
    margin: float,
    use_cache: bool = True,
    bounds: np.ndarray = None,
    unit_target_faces: int = 0,
) -> trimesh.Trimesh:
    """
    为区域生成 tile 晶格（带缓存支持）
    
    参数:
        sole_mesh: 鞋底/区域网格（用于体内判断）
        lattice_path: 晶格单元文件路径
        shrink: 单元缩放系数
        spacing: 平铺间距系数
        margin: 边界安全距离
        use_cache: 是否使用缓存
        bounds: 可选的自定义包围盒 shape=(2,3)，用于区域优化。
                如果为None则使用 sole_mesh.bounds
        unit_target_faces: 晶格单元目标面数（0=不减面，>0=减面到指定面数）
    
    返回:
        生成的晶格网格
    """
    cache_path = ""
    if use_cache and TILE_RESULT_USE_CACHE:
        cache_path = _tile_region_cache_path(sole_mesh, lattice_path, shrink, spacing, margin, bounds, unit_target_faces)
        if os.path.exists(cache_path):
            mesh_cached = _load_cached_mesh_npz(cache_path)
            if mesh_cached is not None:
                if bounds is None:
                    _log(f"[tile_region] 命中缓存（完整鞋底）: {cache_path}")
                else:
                    _log(f"[tile_region] 命中缓存（区域包围盒）: {cache_path}")
                return mesh_cached
            else:
                _log("[tile_region] 缓存文件损坏，重新生成")
        else:
            if bounds is None:
                _log("[tile_region] 缓存未命中（完整鞋底），开始生成")
            else:
                _log("[tile_region] 缓存未命中（区域包围盒），开始生成")
    
    # 加载并准备晶格单元
    lattice_raw = load_mesh(lattice_path)
    
    # 优先使用传入的 unit_target_faces，否则使用全局配置
    target_faces = unit_target_faces if unit_target_faces > 0 else LATTICE_UNIT_TARGET_FACES
    if target_faces > 0:
        faces_before = len(lattice_raw.faces)
        lattice_raw = simplify_mesh(lattice_raw, target_faces)
        faces_after = len(lattice_raw.faces)
        if faces_before != faces_after:
            _log(f"[tile_unit] 晶格单元减面: {faces_before} -> {faces_after} 面")
    
    lattice_unit = normalize_lattice_unit(lattice_raw)
    
    # 准备单元（应用 shrink 参数）
    unit_prepared = prepare_lattice_unit_for_planar_tiling(lattice_unit, sole_mesh, shrink=shrink)
    
    # 生成晶格
    lattice = generate_lattice_in_sole(
        sole_mesh=sole_mesh,
        lattice_unit=unit_prepared,
        planar_tiling=True,
        spacing_factor=spacing,
        boundary_margin=margin,
        bounds=bounds,
    )
    
    # 写入缓存
    if use_cache and TILE_RESULT_USE_CACHE and cache_path and len(lattice.faces) > 0:
        try:
            metadata = {
                'method': 'tile_region',
                'shrink': float(shrink),
                'spacing': float(spacing),
                'margin': float(margin),
                'has_bounds': bounds is not None,
                'unit_target_faces': int(unit_target_faces),
            }
            if bounds is not None:
                metadata['bounds'] = bounds.tolist()
            
            _save_cached_mesh_npz(cache_path, lattice, metadata)
            
            if bounds is None:
                _log(f"[tile_region] 已写入缓存（完整鞋底）: {cache_path}")
            else:
                _log(f"[tile_region] 已写入缓存（区域包围盒）: {cache_path}")
        except Exception as e:
            _log(f"[tile_region] 写缓存失败: {e}")
    
    return lattice


def _inside_by_ray_batch(sole_mesh: trimesh.Trimesh, pts: np.ndarray, batch: int) -> np.ndarray:
    """批量射线法判断点是否在鞋底体内（奇偶规则）。"""
    inside = np.zeros(len(pts), dtype=bool)
    intersector = trimesh.ray.ray_triangle.RayMeshIntersector(sole_mesh)
    ray_dirs = np.tile([0.0, 0.0, 1.0], (batch, 1))
    for i in range(0, len(pts), batch):
        sl = slice(i, min(i + batch, len(pts)))
        p = pts[sl]
        nb = len(p)
        _, ray_idx, _ = intersector.intersects_location(
            ray_origins=p,
            ray_directions=ray_dirs[:nb],
            multiple_hits=True,
        )
        if len(ray_idx) > 0:
            counts = np.bincount(ray_idx, minlength=nb)
            inside[i:i + nb] = (counts % 2) == 1
    return inside


def _inside_by_cpp_mask(
    sole_mesh: trimesh.Trimesh,
    pts: np.ndarray,
    nx: int,
    ny: int,
    grid_size: int,
) -> np.ndarray | None:
    """调用 C++ 扩展计算 inside mask；失败时返回 None 以回退 Python。"""
    try:
        from core.cpp import cpp_mask  # type: ignore

        verts = np.asarray(sole_mesh.vertices, dtype=np.float64)
        faces = np.asarray(sole_mesh.faces, dtype=np.int32)
        pts64 = np.asarray(pts, dtype=np.float64)
        inside = cpp_mask.compute_inside_mask(
            verts,
            faces,
            pts64,
            int(nx),
            int(ny),
            int(grid_size),
        )
        return np.asarray(inside, dtype=bool)
    except Exception as e:
        _log(f"[gyroid] C++ mask 不可用，回退 Python ({type(e).__name__}: {e})")
        return None


def _inside_points_tile(
    sole_mesh: trimesh.Trimesh,
    pts: np.ndarray,
) -> np.ndarray:
    """tile_unit 批量点内判定：优先走 C++，失败则回退 Python contains。"""
    pts = np.asarray(pts, dtype=np.float64)
    if pts.size == 0:
        return np.zeros(0, dtype=bool)
    if TILE_USE_CPP_MASK and len(pts) >= int(TILE_CPP_MASK_MIN_POINTS):
        bounds = np.asarray(sole_mesh.bounds, dtype=np.float64)
        ext = np.maximum(bounds[1] - bounds[0], 1e-9)
        span_xy = np.max(ext[:2])
        nx = max(8, int(np.ceil(span_xy / max(ext[0] / max(int(TILE_CPP_MASK_GRID), 8), 1e-9))))
        ny = max(8, int(np.ceil(span_xy / max(ext[1] / max(int(TILE_CPP_MASK_GRID), 8), 1e-9))))
        inside = _inside_by_cpp_mask(
            sole_mesh=sole_mesh,
            pts=pts,
            nx=nx,
            ny=ny,
            grid_size=int(TILE_CPP_MASK_GRID),
        )
        if inside is not None:
            _log(f"[tile_unit] 使用 C++ 点内判定: 点数={len(pts)}")
            return inside
        _log("[tile_unit] C++ 点内判定失败，回退 Python contains")
    return _safe_contains(sole_mesh, pts)


def _inside_points_voronoi(
    sole_mesh: trimesh.Trimesh,
    pts: np.ndarray,
) -> np.ndarray:
    """voronoi 批量点内判定：优先走 C++，失败则回退 Python contains。"""
    pts = np.asarray(pts, dtype=np.float64)
    if pts.size == 0:
        return np.zeros(0, dtype=bool)
    if VORONOI_USE_CPP_MASK and len(pts) >= int(VORONOI_CPP_MASK_MIN_POINTS):
        bounds = np.asarray(sole_mesh.bounds, dtype=np.float64)
        ext = np.maximum(bounds[1] - bounds[0], 1e-9)
        span_xy = np.max(ext[:2])
        nx = max(8, int(np.ceil(span_xy / max(ext[0] / max(int(VORONOI_CPP_MASK_GRID), 8), 1e-9))))
        ny = max(8, int(np.ceil(span_xy / max(ext[1] / max(int(VORONOI_CPP_MASK_GRID), 8), 1e-9))))
        inside = _inside_by_cpp_mask(
            sole_mesh=sole_mesh,
            pts=pts,
            nx=nx,
            ny=ny,
            grid_size=int(VORONOI_CPP_MASK_GRID),
        )
        if inside is not None:
            _log(f"[voronoi] 使用 C++ 点内判定: 点数={len(pts)}")
            return inside
        _log("[voronoi] C++ 点内判定失败，回退 Python contains")
    return _safe_contains(sole_mesh, pts)


def _inside_points_clip(
    sole_mesh: trimesh.Trimesh,
    pts: np.ndarray,
    clip_stats: dict[str, int] | None = None,
) -> np.ndarray:
    """最终裁剪阶段的批量点内判定：优先走 C++，失败则回退 Python contains。"""
    pts = np.asarray(pts, dtype=np.float64)
    if pts.size == 0:
        return np.zeros(0, dtype=bool)
    if CLIP_USE_CPP_MASK and len(pts) >= int(CLIP_CPP_MASK_MIN_POINTS):
        bounds = np.asarray(sole_mesh.bounds, dtype=np.float64)
        ext = np.maximum(bounds[1] - bounds[0], 1e-9)
        span_xy = np.max(ext[:2])
        nx = max(8, int(np.ceil(span_xy / max(ext[0] / max(int(CLIP_CPP_MASK_GRID), 8), 1e-9))))
        ny = max(8, int(np.ceil(span_xy / max(ext[1] / max(int(CLIP_CPP_MASK_GRID), 8), 1e-9))))
        inside = _inside_by_cpp_mask(
            sole_mesh=sole_mesh,
            pts=pts,
            nx=nx,
            ny=ny,
            grid_size=int(CLIP_CPP_MASK_GRID),
        )
        if inside is not None:
            if clip_stats is not None:
                clip_stats["cpp_batches"] = clip_stats.get("cpp_batches", 0) + 1
                clip_stats["cpp_points"] = clip_stats.get("cpp_points", 0) + int(len(pts))
            return inside
        _log("[clip] C++ 点内判定失败，回退 Python contains")
    return _safe_contains(sole_mesh, pts)


def _z_scale_mesh(mesh: trimesh.Trimesh, scale: float) -> trimesh.Trimesh:
    if scale == 1.0:
        return mesh
    out = mesh.copy()
    verts = out.vertices.copy()
    verts[:, 2] *= scale
    out.vertices = verts
    return out


def _safe_contains(mesh: trimesh.Trimesh, points: np.ndarray) -> np.ndarray:
    """contains 在非封闭网格上可能报错，统一在这里兜底。"""
    try:
        return mesh.contains(points)
    except Exception as e:
        _log(f"[contains] 回退到全False ({type(e).__name__}: {e})")
        return np.zeros(len(points), dtype=bool)


def clip_lattice_to_sole(sole_mesh: trimesh.Trimesh, lattice: trimesh.Trimesh) -> trimesh.Trimesh:
    """
    去掉穿出鞋底体积的三角面：采用适中的多点体内检测，兼顾保留率与收边效果。

    若鞋底网格非闭合或 contains 不可靠导致「一面不剩」，则依次回退：
    1) 用鞋底 AABB 裁剪（面心在包围盒内）
    2) 仍为空则保留未裁剪晶格，避免查看器只剩鞋底
    """
    if lattice.faces.size == 0:
        return lattice

    orig = lattice.copy()
    lattice = orig.copy()
    try:
        tris = np.asarray(lattice.triangles, dtype=np.float64)
        cents = np.asarray(lattice.triangles_center, dtype=np.float64)
    except Exception as e:
        _log(f"[严格裁剪] triangles 获取失败 ({e})，跳过裁剪")
        return orig

    n = len(cents)
    if n == 0:
        return orig

    v0 = tris[:, 0, :]
    v1 = tris[:, 1, :]
    v2 = tris[:, 2, :]
    samples = np.stack([
        cents,
        0.5 * (v0 + v1),
        0.5 * (v1 + v2),
        0.5 * (v2 + v0),
    ], axis=1)

    keep = np.zeros(n, dtype=bool)
    batch = 4096
    clip_stats: dict[str, int] = {"cpp_batches": 0, "cpp_points": 0}
    for i in range(0, n, batch):
        sl = slice(i, min(i + batch, n))
        pts = samples[sl].reshape(-1, 3)
        inside = _inside_points_clip(sole_mesh, pts, clip_stats=clip_stats).reshape(-1, 4)
        score = inside.mean(axis=1)
        keep[sl] = score >= 0.58

    if clip_stats["cpp_batches"] > 0:
        _log(
            f"[clip] 使用 C++ 点内判定: 批次={clip_stats['cpp_batches']}, 总点数={clip_stats['cpp_points']}"
        )

    kept = int(np.count_nonzero(keep))
    if kept > 0:
        lattice.update_faces(keep)
        lattice.remove_unreferenced_vertices()
        _log(f"[平衡裁剪] {n} -> {kept} 面（多点体内检测）")
        return lattice

    sb = sole_mesh.bounds
    in_box = (cents >= sb[0]).all(axis=1) & (cents <= sb[1]).all(axis=1)
    kb = int(np.count_nonzero(in_box))
    if kb > 0:
        _log(
            f"[clip_lattice] 当 contains 未保留任何一面（鞋底可能非水密），"
            f"回退到鞋底 AABB 裁剪：{n} -> {kb}"
        )
        lattice = orig.copy()
        lattice.update_faces(in_box)
        lattice.remove_unreferenced_vertices()
        return lattice

    _log(
        "[clip_lattice] contains 与 AABB 均无保留面，返回未裁剪结果；"
        "请检查 lattice 是否与鞋底同坐标系、鞋底 STL 是否过小/未闭合。"
    )
    return orig


def trim_lattice_to_sole_volume(
    sole_mesh: trimesh.Trimesh,
    lattice: trimesh.Trimesh,
    prefer_boolean: bool = False,
) -> trimesh.Trimesh:
    """
    将晶格限制在鞋底实体内部。prefer_boolean=True 时优先尝试布尔裁剪，
    失败再回退到 contains 裁剪。
    """
    if lattice.faces.size == 0:
        return lattice

    mode = (LATTICE_TRIM_MODE or "contains").strip().lower()
    if mode not in ("boolean", "contains"):
        mode = "contains"

    try_boolean = prefer_boolean or (mode == "boolean")
    lattice_before = lattice.copy()

    if try_boolean:
        try:
            import trimesh.boolean

            inter = trimesh.boolean.intersection(
                [lattice, sole_mesh],
                engine="manifold",
                check_volume=False,
            )
            if inter is not None and len(inter.faces) > 0:
                _log(f"[布尔裁剪] 成功: {len(inter.faces)} 面（按鞋底体积截断）")
                return inter
            _log("[布尔裁剪] 结果为空，回退到 contains 裁剪")
        except Exception as e:
            _log(f"[布尔裁剪] 失败 ({type(e).__name__}: {e})，回退到 contains 裁剪")

    return clip_lattice_to_sole(sole_mesh, lattice_before)


def _batch_segments_inside_sole(
    sole_mesh: trimesh.Trimesh,
    segments: np.ndarray,
    z_mid: float,
    n_samples: int,
) -> np.ndarray:
    """
    对每条 2D 线段在 z=z_mid 上采样 n_samples 个点，全部在鞋体内部为 True。
    segments: (S, 2, 2) 为 XY 上的两个端点。
    """
    if segments.size == 0:
        return np.array([], dtype=bool)
    p0 = segments[:, 0, :]
    p1 = segments[:, 1, :]
    s = len(segments)
    t = np.linspace(0.0, 1.0, max(3, n_samples))
    keep = np.ones(s, dtype=bool)
    for ti in t:
        xy = (1.0 - ti) * p0 + ti * p1
        pts = np.column_stack([xy, np.full(s, z_mid)])
        keep &= _inside_points_voronoi(sole_mesh, pts)
    return keep


def _batch_voronoi_wall_inside_sole(
    sole_mesh: trimesh.Trimesh,
    segments: np.ndarray,
    z_mid: float,
    n_along: int,
    thickness: float,
) -> np.ndarray:
    """对 Voronoi 薄壁做更严格的体内筛选：同时检查中心线与法向厚度偏移。"""
    if segments.size == 0:
        return np.array([], dtype=bool)

    p0 = segments[:, 0, :]
    p1 = segments[:, 1, :]
    v = p1 - p0
    lens = np.linalg.norm(v, axis=1)
    keep = lens > 1e-6
    if not np.any(keep):
        return keep

    dirs = np.zeros_like(v)
    dirs[keep] = v[keep] / lens[keep, None]
    normals = np.column_stack([-dirs[:, 1], dirs[:, 0]])
    offsets = np.array([0.0, -0.5 * thickness, 0.5 * thickness], dtype=np.float64)
    ts = np.linspace(0.0, 1.0, max(5, int(n_along)))

    for off in offsets:
        shifted_p0 = p0 + normals * off
        shifted_p1 = p1 + normals * off
        for ti in ts:
            xy = (1.0 - ti) * shifted_p0 + ti * shifted_p1
            pts = np.column_stack([xy, np.full(len(segments), z_mid)])
            keep &= _inside_points_voronoi(sole_mesh, pts)
            if not np.any(keep):
                return keep
    return keep


def _trim_faces_by_contains(
    sole_mesh: trimesh.Trimesh,
    mesh: trimesh.Trimesh,
) -> trimesh.Trimesh:
    """按三角面心是否在鞋体内进行快速裁剪。"""
    if len(mesh.faces) == 0:
        return mesh
    centers = np.asarray(mesh.triangles_center, dtype=np.float64)
    if len(centers) == 0:
        return mesh
    keep = _safe_contains(sole_mesh, centers)
    trimmed = mesh.copy()
    trimmed.update_faces(keep)
    trimmed.remove_unreferenced_vertices()
    return trimmed


def simplify_mesh(mesh: trimesh.Trimesh, target_faces: int) -> trimesh.Trimesh:
    """
    尝试把网格减面到 target_faces 左右，避免后续平铺导致面数爆炸。

    优先使用 open3d（Windows 下通常更稳定）。如果 open3d 不可用，
    再尝试 trimesh 的 quadratic decimation。全部不可用则原样返回。
    """
    if target_faces <= 0:
        return mesh
    if len(mesh.faces) <= target_faces:
        return mesh

    # 优先使用 open3d（Windows 下更稳定）
    try:
        import open3d as o3d  # type: ignore

        o3m = o3d.geometry.TriangleMesh(
            vertices=o3d.utility.Vector3dVector(mesh.vertices),
            triangles=o3d.utility.Vector3iVector(mesh.faces),
        )
        o3m.remove_duplicated_vertices()
        o3m.remove_duplicated_triangles()
        o3m.remove_degenerate_triangles()
        o3m.remove_unreferenced_vertices()

        o3m_s = o3m.simplify_quadric_decimation(int(target_faces))
        vertices = np.asarray(o3m_s.vertices, dtype=np.float64)
        faces = np.asarray(o3m_s.triangles, dtype=np.int64)
        simplified = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)

        _log(
            f"[减面] {len(mesh.faces)} -> {len(simplified.faces)} 面"
            f"（目标={target_faces}, 后端=open3d）"
        )
        return simplified
    except Exception as e:
        _log(f"[减面] open3d 不可用 ({type(e).__name__}: {e})")

    # 退回 trimesh 自带减面（如果环境支持）
    try:
        simplified = mesh.simplify_quadratic_decimation(target_faces)
        _log(
            f"[减面] {len(mesh.faces)} -> {len(simplified.faces)} 面"
            f"（目标={target_faces}, 后端=trimesh）"
        )
        return simplified
    except Exception as e:
        _log(
            f"[减面] 跳过减面 ({type(e).__name__}: {e})，提示: pip install open3d"
        )
        return mesh


def load_mesh(path: str) -> trimesh.Trimesh:
    """加载 STL 网格，统一为 Trimesh。"""
    mesh = trimesh.load(path, force="mesh")
    if not isinstance(mesh, trimesh.Trimesh):
        mesh = trimesh.util.concatenate(mesh.dump())

    mesh.remove_unreferenced_vertices()
    # 新版 trimesh 中去重面片在 repair 模块中
    try:
        trimesh.repair.remove_duplicate_faces(mesh)
    except Exception:
        pass
    try:
        mesh.remove_degenerate_faces()
    except Exception:
        pass
    mesh.rezero()
    st = _stats(mesh)
    _log(f"[load_mesh] {path}: {st.vertices} vertices, {st.faces} faces")
    return mesh


def _batch_unit_aabb_overlaps_sole(
    sole_bounds: np.ndarray,
    unit_bounds: np.ndarray,
    centers: np.ndarray,
) -> np.ndarray:
    """
    单元平移到 centers 后，其轴对齐包围盒是否与鞋底包围盒相交。
    用于薄底、非水密 mesh（contains 全假）时仍能铺格。
    """
    u_lo = centers + unit_bounds[0]
    u_hi = centers + unit_bounds[1]
    sb0, sb1 = sole_bounds[0], sole_bounds[1]
    return np.all(u_hi >= sb0, axis=1) & np.all(u_lo <= sb1, axis=1)


def _unit_aabb_corners_local(bounds: np.ndarray) -> np.ndarray:
    """轴对齐包围盒的 8 个角点 (8, 3)。"""
    lo, hi = bounds[0], bounds[1]
    return np.array(
        [
            [lo[0], lo[1], lo[2]],
            [hi[0], lo[1], lo[2]],
            [lo[0], hi[1], lo[2]],
            [hi[0], hi[1], lo[2]],
            [lo[0], lo[1], hi[2]],
            [hi[0], lo[1], hi[2]],
            [lo[0], hi[1], hi[2]],
            [hi[0], hi[1], hi[2]],
        ],
        dtype=np.float64,
    )


def normalize_lattice_unit(lattice_mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """
    将晶格单元平移到以其包围盒中心为原点的坐标系内，
    并按 LATTICE_SCALE 进行整体缩放。
    """
    bounds = lattice_mesh.bounds  # shape (2, 3)
    center = bounds.mean(axis=0)
    extents = bounds[1] - bounds[0]

    lattice = lattice_mesh.copy()
    lattice.apply_translation(-center)

    if LATTICE_SCALE != 1.0:
        lattice.apply_scale(LATTICE_SCALE)
        extents *= LATTICE_SCALE

    if VERBOSE:
        _log(f"[normalize_lattice_unit] center={center}, extents={extents}")

    return lattice


def prepare_lattice_unit_for_planar_tiling(
    lattice_unit: trimesh.Trimesh,
    sole_mesh: trimesh.Trimesh,
    shrink: float = None,
) -> trimesh.Trimesh:
    """平铺前缩放单元：XY 用 shrink 控制密度，Z 尽量匹配鞋底厚度以覆盖完整高度。"""
    unit = lattice_unit.copy()
    if shrink is None:
        sh = float(LATTICE_TILE_SHRINK)
    else:
        sh = float(shrink)
    if sh <= 0.0:
        sh = 1.0

    T = np.eye(4, dtype=np.float64)
    T[0, 0] = sh
    T[1, 1] = sh
    T[2, 2] = sh
    unit.apply_transform(T)

    ue = unit.bounds[1] - unit.bounds[0]
    sz_sole = float(sole_mesh.bounds[1][2] - sole_mesh.bounds[0][2])
    if ue[2] > 1e-6 and sz_sole > 1e-6:
        z_scale = max(0.1, 0.96 * sz_sole / ue[2])
        Tz = np.eye(4, dtype=np.float64)
        Tz[2, 2] = z_scale
        unit.apply_transform(Tz)
        ue = unit.bounds[1] - unit.bounds[0]

    _log(
        f"[prepare_planar_tile] XY缩放={sh:.4f}；适配后单元尺寸={ue.round(4).tolist()}，鞋底厚={sz_sole:.4f}"
    )
    return unit


def _principal_xy_rotation(vertices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """返回 (3x3 旋转矩阵 R, 鞋底中心 mid)。p_rot = R @ (p - mid)，使 XY 上第一主轴对齐到 +X。"""
    mid = vertices.mean(axis=0)
    xy = vertices[:, :2] - mid[:2]
    if len(xy) < 3:
        return np.eye(3, dtype=np.float64), mid
    cov = np.cov(xy.T)
    evals, evecs = np.linalg.eigh(cov)
    v_main = evecs[:, int(np.argmax(evals))]
    theta = float(np.arctan2(v_main[1], v_main[0]))
    c, s = np.cos(-theta), np.sin(-theta)
    R = np.eye(3, dtype=np.float64)
    R[0, 0], R[0, 1] = c, -s
    R[1, 0], R[1, 1] = s, c
    return R, mid


def generate_lattice_single_in_sole_bbox(
    sole_mesh: trimesh.Trimesh,
    lattice_unit: trimesh.Trimesh,
) -> trimesh.Trimesh:
    """
    单份晶格：normalize 后的单元缩放到鞋底 AABB，再与盒中心对齐；不做网格平铺。

    LATTICE_SINGLE_FIT:
    - inside: 等比 min 缩放，整体落在盒内（短轴决定比例，长轴方向易留白 → 只见中间一坨）
    - cover: 等比 max 缩放，外接盒包住鞋底盒，再靠后续裁剪进鞋体（易铺满长条鞋底）
    - stretch: 三轴独立缩放使外接盒与鞋底盒对齐（会拉歪杆件角度）
    """
    sole_bounds = sole_mesh.bounds
    unit_bounds = lattice_unit.bounds
    unit_ext = unit_bounds[1] - unit_bounds[0]
    sole_ext = sole_bounds[1] - sole_bounds[0]
    ratios = sole_ext / np.maximum(unit_ext, 1e-9)

    fit = (LATTICE_SINGLE_FIT or "cover").strip().lower()
    lattice = lattice_unit.copy()

    if fit == "inside":
        s = float(np.min(ratios))
        lattice.apply_scale(s)
        _log(f"[single_in_bbox] 模式=inside 等比系数=min(ratios)={s:.6f}")
    elif fit == "cover":
        s = float(np.max(ratios))
        lattice.apply_scale(s)
        _log(f"[single_in_bbox] 模式=cover 等比系数=max(ratios)={s:.6f}（再裁剪进鞋体）")
    elif fit in ("stretch", "fill", "bbox"):
        sx, sy, sz = float(ratios[0]), float(ratios[1]), float(ratios[2])
        T = np.eye(4, dtype=np.float64)
        T[0, 0], T[1, 1], T[2, 2] = sx, sy, sz
        lattice.apply_transform(T)
        _log(f"[single_in_bbox] 模式=stretch 非等比 sx,sy,sz={sx:.6f},{sy:.6f},{sz:.6f}")
    else:
        _log(f"[single_in_bbox] 未知 LATTICE_SINGLE_FIT={LATTICE_SINGLE_FIT!r}，改用 cover")
        s = float(np.max(ratios))
        lattice.apply_scale(s)

    lc = lattice.bounds.mean(axis=0)
    sc = sole_bounds.mean(axis=0)
    lattice.apply_translation(sc - lc)
    _log(f"[single_in_bbox] 对齐鞋底 AABB 中心，三角面 {len(lattice.faces)}")
    return lattice


def generate_lattice_in_sole(
    sole_mesh: trimesh.Trimesh,
    lattice_unit: trimesh.Trimesh,
    margin: float = 0.0,
    planar_tiling: bool = False,
    spacing_factor: float = None,
    boundary_margin: float = None,
    bounds: np.ndarray = None,
) -> trimesh.Trimesh:
    """
    顺序密铺；planar_tiling=True 时按鞋底主方向在 XY 铺满，并继续走统一裁剪。
    
    参数:
        sole_mesh: 鞋底网格（用于体内判断）
        lattice_unit: 晶格单元
        margin: 边距
        planar_tiling: 是否平面铺设
        spacing_factor: 间距系数
        boundary_margin: 边界边距
        bounds: 可选的自定义包围盒 shape=(2,3)，用于区域优化。
                如果为None则使用 sole_mesh.bounds
    """
    t_total_start = time.perf_counter()
    t_candidate_start = t_total_start
    
    # 使用自定义包围盒或默认从mesh获取
    if bounds is None:
        sole_bounds = np.asarray(sole_mesh.bounds, dtype=np.float64)
        work_bounds = sole_bounds
    else:
        sole_bounds = np.asarray(sole_mesh.bounds, dtype=np.float64)
        work_bounds = np.asarray(bounds, dtype=np.float64)
    
    unit_bounds = np.asarray(lattice_unit.bounds, dtype=np.float64)
    raw_ext = unit_bounds[1] - unit_bounds[0]
    sole_mid_z = float(sole_bounds.mean(axis=0)[2])
    fac = float(spacing_factor if spacing_factor is not None else TILE_SPACING_FACTOR)
    boundary_margin_val = max(0.0, float(boundary_margin if boundary_margin is not None else TILE_BOUNDARY_MARGIN)) if planar_tiling else 0.0

    use_rot = bool(planar_tiling) and len(sole_mesh.vertices) >= 4
    if use_rot:
        R, mid = _principal_xy_rotation(np.asarray(sole_mesh.vertices, dtype=np.float64))
        vc = np.asarray(sole_mesh.vertices, dtype=np.float64) - mid
        v_rot = (R @ vc.T).T
        # 如果使用自定义包围盒，需要转换到旋转坐标系
        if bounds is not None:
            corners = np.array([
                [work_bounds[0, 0], work_bounds[0, 1], work_bounds[0, 2]],
                [work_bounds[0, 0], work_bounds[0, 1], work_bounds[1, 2]],
                [work_bounds[0, 0], work_bounds[1, 1], work_bounds[0, 2]],
                [work_bounds[0, 0], work_bounds[1, 1], work_bounds[1, 2]],
                [work_bounds[1, 0], work_bounds[0, 1], work_bounds[0, 2]],
                [work_bounds[1, 0], work_bounds[0, 1], work_bounds[1, 2]],
                [work_bounds[1, 0], work_bounds[1, 1], work_bounds[0, 2]],
                [work_bounds[1, 0], work_bounds[1, 1], work_bounds[1, 2]],
            ])
            corners_rot = (R @ (corners - mid).T).T
            work_bounds = np.stack([corners_rot.min(axis=0), corners_rot.max(axis=0)])
        else:
            work_bounds = np.stack([v_rot.min(axis=0), v_rot.max(axis=0)])
        _log("[generate_lattice_in_sole] planar_tiling 按鞋底主方向对齐")
    else:
        R = np.eye(3, dtype=np.float64)
        mid = sole_mesh.bounds.mean(axis=0)

    if planar_tiling:
        pad = np.array([raw_ext[0] * 0.5, raw_ext[1] * 0.5, 0.0], dtype=np.float64)
        min_corner = work_bounds[0] + margin + np.array([boundary_margin_val, boundary_margin_val, 0.0]) - pad
        max_corner = work_bounds[1] - margin - np.array([boundary_margin_val, boundary_margin_val, 0.0]) + pad
        if TILE_ISOTROPIC_STEP:
            mean_xy = float(np.mean(raw_ext[:2]))
            step_xy = max(mean_xy * fac, 1e-9)
            step = np.array([step_xy, step_xy, 1.0], dtype=np.float64)
        else:
            step = np.array(
                [max(raw_ext[0] * fac, 1e-9), max(raw_ext[1] * fac, 1e-9), 1.0],
                dtype=np.float64,
            )
        span_xy = np.array([max_corner[0] - min_corner[0], max_corner[1] - min_corner[1]], dtype=np.float64)
        n_xy = np.ceil(span_xy / step[:2]).astype(np.int64)
        n_xy = np.maximum(n_xy, 1)
        cells = int(n_xy[0] * n_xy[1])
        if cells > MAX_TILE_CANDIDATES:
            scale = (cells / float(MAX_TILE_CANDIDATES)) ** 0.5
            step[0] *= scale
            step[1] *= scale
            n_xy = np.ceil(span_xy / step[:2]).astype(np.int64)
            n_xy = np.maximum(n_xy, 1)
        xs = min_corner[0] + (np.arange(n_xy[0], dtype=np.float64) + 0.5) * step[0]
        ys = min_corner[1] + (np.arange(n_xy[1], dtype=np.float64) + 0.5) * step[1]
        z0 = 0.0 if use_rot else sole_mid_z
        candidate_rot = np.array(np.meshgrid(xs, ys, np.array([z0]), indexing="xy"), dtype=np.float64).reshape(3, -1).T

        if TILE_EDGE_FILL and len(xs) > 1 and len(ys) > 1:
            xs_shift = np.clip(xs + 0.5 * step[0], min_corner[0] + 0.5 * step[0], max_corner[0] - 0.5 * step[0])
            ys_shift = np.clip(ys + 0.5 * step[1], min_corner[1] + 0.5 * step[1], max_corner[1] - 0.5 * step[1])
            stagger = np.array(np.meshgrid(xs_shift, ys_shift, np.array([z0]), indexing="xy"), dtype=np.float64).reshape(3, -1).T
            edge_x = np.array([min_corner[0] + 0.35 * step[0], max_corner[0] - 0.35 * step[0]], dtype=np.float64)
            edge_y = np.array([min_corner[1] + 0.35 * step[1], max_corner[1] - 0.35 * step[1]], dtype=np.float64)
            strips1 = np.array(np.meshgrid(edge_x, ys, np.array([z0]), indexing="xy"), dtype=np.float64).reshape(3, -1).T
            strips2 = np.array(np.meshgrid(xs, edge_y, np.array([z0]), indexing="xy"), dtype=np.float64).reshape(3, -1).T
            candidate_rot = np.concatenate([candidate_rot, stagger, strips1, strips2], axis=0)

        candidate_rot = np.unique(np.round(candidate_rot, 6), axis=0)
        end_band_mask = np.zeros(len(candidate_rot), dtype=bool)
    else:
        pad = raw_ext * 0.5
        min_corner = sole_bounds[0] + margin - pad
        max_corner = sole_bounds[1] - margin + pad
        if TILE_ISOTROPIC_STEP:
            mean_ext = float(np.mean(raw_ext))
            step = np.full(3, max(mean_ext * fac, 1e-9))
        else:
            step = np.maximum(raw_ext * fac, 1e-9)
        span = max_corner - min_corner
        n = np.ceil(span / step).astype(np.int64)
        n = np.maximum(n, 1)
        cells = int(n[0] * n[1] * n[2])
        if cells > MAX_TILE_CANDIDATES:
            scale = (cells / float(MAX_TILE_CANDIDATES)) ** (1.0 / 3.0)
            step = step * scale
            n = np.ceil(span / step).astype(np.int64)
            n = np.maximum(n, 1)
        xs = min_corner[0] + (np.arange(n[0], dtype=np.float64) + 0.5) * step[0]
        ys = min_corner[1] + (np.arange(n[1], dtype=np.float64) + 0.5) * step[1]
        zs = min_corner[2] + (np.arange(n[2], dtype=np.float64) + 0.5) * step[2]
        candidate_rot = np.array(np.meshgrid(xs, ys, zs, indexing="xy"), dtype=np.float64).reshape(3, -1).T
        end_band_mask = np.zeros(len(candidate_rot), dtype=bool)

    if use_rot:
        candidate_centers = (R.T @ candidate_rot.T).T + mid
        corner_local = _unit_aabb_corners_local(unit_bounds)
        corner_offsets = (R.T @ corner_local.T).T
    else:
        candidate_centers = candidate_rot
        corner_offsets = _unit_aabb_corners_local(unit_bounds)

    t_candidate_end = time.perf_counter()
    _log(f"[tile_unit][timing] 候选生成: {t_candidate_end - t_candidate_start:.3f}s, 候选数={len(candidate_centers)}")
    t_filter_start = t_candidate_end

    placed_units: list[trimesh.Trimesh] = []
    batch_size = 512
    n_cand = len(candidate_centers)
    inside_mask = np.zeros(n_cand, dtype=bool)

    for i in range(0, n_cand, batch_size):
        batch = candidate_centers[i : i + batch_size]
        b = len(batch)
        centers_in = _inside_points_tile(sole_mesh, batch)
        cw = batch[:, None, :] + corner_offsets[None, :, :]
        flat = cw.reshape(-1, 3)
        corner_hits = _inside_points_tile(sole_mesh, flat).reshape(b, 8)
        if TILE_PLACE_IF_CORNER_INSIDE:
            vol_ok = centers_in | corner_hits.any(axis=1)
        else:
            vol_ok = centers_in
        if TILE_PLACE_IF_AABB_OVERLAP:
            aabb_ok = _batch_unit_aabb_overlaps_sole(sole_bounds, unit_bounds, batch)
            inside_mask[i : i + b] = vol_ok | aabb_ok
        else:
            inside_mask[i : i + b] = vol_ok

    valid_centers = candidate_centers[inside_mask]
    t_filter_end = time.perf_counter()
    _log(f"[tile_unit][timing] 体内筛选: {t_filter_end - t_filter_start:.3f}s, 保留={len(valid_centers)}/{n_cand}")
    if len(valid_centers) == 0:
        _log("[generate_lattice_in_sole] no lattice units placed.")
        _log(f"[tile_unit][timing] 总耗时: {time.perf_counter() - t_total_start:.3f}s")
        return trimesh.Trimesh()

    order = np.lexsort((valid_centers[:, 0], valid_centers[:, 1], valid_centers[:, 2]))
    valid_centers = valid_centers[order]
    _log(f"[平铺晶格] 候选中心点: {n_cand}, 过滤后放置: {len(valid_centers)}")

    if len(valid_centers) > MAX_LATTICE_UNITS:
        valid_centers = valid_centers[:MAX_LATTICE_UNITS]
        _log(f"[generate_lattice_in_sole] truncating units -> {MAX_LATTICE_UNITS}")

    rot_block = R.T if use_rot else np.eye(3, dtype=np.float64)
    t_assemble_start = time.perf_counter()
    for center in valid_centers:
        unit_copy = lattice_unit.copy()
        if use_rot:
            T = np.eye(4, dtype=np.float64)
            T[:3, :3] = rot_block
            T[:3, 3] = center
            unit_copy.apply_transform(T)
        else:
            unit_copy.apply_translation(center)
        placed_units.append(unit_copy)

    lattice_all = trimesh.util.concatenate(placed_units)
    t_assemble_end = time.perf_counter()
    _log(f"[tile_unit][timing] 单元拼接: {t_assemble_end - t_assemble_start:.3f}s, 单元数={len(valid_centers)}")
    st = _stats(lattice_all)
    _log(f"[generate_lattice_in_sole] lattice after tiling: {st.vertices} vertices, {st.faces} faces")
    _log(f"[tile_unit][timing] 总耗时: {t_assemble_end - t_total_start:.3f}s")
    return lattice_all


def _compute_voronoi_seed_count(bounds_xy: np.ndarray, cell_size: float) -> int:
    extents = bounds_xy[1] - bounds_xy[0]
    area = max(1e-6, float(extents[0] * extents[1]))
    approx_cells = max(24.0, 1.8 * area / max(1e-6, cell_size * cell_size))
    return int(min(MAX_LATTICE_UNITS, max(24, round(approx_cells))))


def _sample_inside_points_2d(
    sole_mesh: trimesh.Trimesh,
    count: int,
    z_level: float,
    rng_seed: int,
    bounds_xy: np.ndarray = None,
) -> np.ndarray:
    """
    在指定 z 高度进行分层采样：优先抖动网格，再补随机点，减少空区。
    
    参数:
        sole_mesh: 鞋底网格（用于体内判断）
        count: 目标种子数量
        z_level: 采样的 Z 高度
        rng_seed: 随机种子
        bounds_xy: 可选的 XY 平面包围盒 shape=(2,2)，用于区域优化。
                   如果为None则使用 sole_mesh.bounds 的 XY 部分
    """
    # 使用自定义包围盒或默认从mesh获取
    if bounds_xy is None:
        bounds = sole_mesh.bounds
        min_xy = bounds[0, :2]
        max_xy = bounds[1, :2]
    else:
        min_xy = np.asarray(bounds_xy[0], dtype=np.float64)
        max_xy = np.asarray(bounds_xy[1], dtype=np.float64)
    
    span_xy = np.maximum(max_xy - min_xy, 1e-6)
    rng = np.random.default_rng(rng_seed)

    accepted: list[np.ndarray] = []

    grid_n = max(5, int(np.ceil(np.sqrt(max(count * 2, 25)))))
    xs = np.linspace(min_xy[0], max_xy[0], grid_n, endpoint=False) + 0.5 * span_xy[0] / grid_n
    ys = np.linspace(min_xy[1], max_xy[1], grid_n, endpoint=False) + 0.5 * span_xy[1] / grid_n
    grid_xy = np.array(np.meshgrid(xs, ys, indexing="xy"), dtype=np.float64).reshape(2, -1).T
    jitter = (rng.random(grid_xy.shape) - 0.5) * (span_xy / grid_n) * 0.7
    grid_xy = np.clip(grid_xy + jitter, min_xy, max_xy)
    grid_pts = np.column_stack([grid_xy, np.full(len(grid_xy), z_level)])
    mask = _inside_points_voronoi(sole_mesh, grid_pts)
    if np.any(mask):
        accepted.append(grid_pts[mask][:, :2])

    current = sum(len(a) for a in accepted)
    need = max(0, count - current)
    max_rounds = 24
    for _ in range(max_rounds):
        if need <= 0:
            break
        batch_n = max(need * 5, 96)
        xs = rng.uniform(min_xy[0], max_xy[0], size=batch_n)
        ys = rng.uniform(min_xy[1], max_xy[1], size=batch_n)
        pts = np.column_stack([xs, ys, np.full(batch_n, z_level)])
        mask = _inside_points_voronoi(sole_mesh, pts)
        if np.any(mask):
            accepted.append(pts[mask][:, :2])
            current = sum(len(a) for a in accepted)
            need = count - current

    if not accepted:
        return np.zeros((0, 2), dtype=float)
    all_pts = np.vstack(accepted)
    if len(all_pts) > count:
        idx = rng.choice(len(all_pts), size=count, replace=False)
        all_pts = all_pts[idx]
    return all_pts


def _voronoi_finite_segments(vor: Voronoi) -> np.ndarray:
    """提取 Voronoi 有限边段，返回 shape=(N,2,2)。"""
    segs: list[np.ndarray] = []
    for rv in vor.ridge_vertices:
        if len(rv) != 2 or rv[0] < 0 or rv[1] < 0:
            continue
        p0 = vor.vertices[rv[0]]
        p1 = vor.vertices[rv[1]]
        segs.append(np.array([p0, p1], dtype=float))
    if not segs:
        return np.zeros((0, 2, 2), dtype=float)
    return np.stack(segs, axis=0)


def _segment_wall_mesh(start: np.ndarray, end: np.ndarray, thickness: float) -> trimesh.Trimesh:
    """把 3D 线段生成为圆柱杆件。"""
    start = np.asarray(start, dtype=np.float64)
    end = np.asarray(end, dtype=np.float64)
    
    # 检查长度
    direction = end - start
    length = float(np.linalg.norm(direction))
    if length < 1e-6:
        return trimesh.Trimesh()
    
    # 计算半径
    radius = max(0.08, 0.5 * float(thickness))
    
    try:
        # 使用 segment 参数生成圆柱体
        cylinder = trimesh.creation.cylinder(
            radius=radius,
            sections=16,
            segment=[start, end],
        )
        
        # 验证生成的网格
        if cylinder.is_empty or len(cylinder.faces) == 0:
            return trimesh.Trimesh()
        
        # 检查是否有退化的顶点或面
        if not np.all(np.isfinite(cylinder.vertices)):
            _log(f"[voronoi] 警告：圆柱体包含无效顶点，跳过")
            return trimesh.Trimesh()
        
        return cylinder
        
    except Exception as e:
        _log(f"[voronoi] 圆柱体生成失败: {e}")
        return trimesh.Trimesh()


def _voronoi_layer_z_values(z_min: float, z_max: float, n_layers: int) -> np.ndarray:
    """生成更贴近鞋底厚度方向的 Voronoi 层高。"""
    n = max(3, int(n_layers))
    thickness = max(1e-6, z_max - z_min)
    margin = min(thickness * 0.08, 0.6)
    if z_max - z_min <= 2.0 * margin:
        return np.linspace(z_min, z_max, n)
    return np.linspace(z_min + margin, z_max - margin, n)



def generate_gyroid_lattice(
    sole_mesh: trimesh.Trimesh,
    cell_size: float = GYROID_CELL_SIZE,
    isovalue: float = GYROID_ISOVALUE,
    resolution: int = GYROID_RESOLUTION,
    bounds: np.ndarray = None,
) -> trimesh.Trimesh:
    """
    在鞋底体积内生成 Gyroid TPMS 晶格结构。

    算法：
    1. 在鞋底包围盒内建立体素网格
    2. 计算每个体素的 Gyroid 隐式函数值
    3. 用 Marching Cubes 提取等值面
    4. 裁剪到鞋底体积内
    
    参数:
        sole_mesh: 鞋底网格（用于体内判断，必须水密）
        cell_size: Gyroid单胞尺寸
        isovalue: 等值面参数
        resolution: 分辨率
        bounds: 可选的自定义包围盒 shape=(2,3)，用于区域优化。
                如果为None则使用 sole_mesh.bounds
    """
    try:
        from skimage import measure as skm  # type: ignore
    except ImportError:
        _log("[gyroid] 需要 scikit-image: pip install scikit-image")
        return trimesh.Trimesh()

    # 使用自定义包围盒或默认从mesh获取
    if bounds is None:
        bounds = sole_mesh.bounds
    else:
        bounds = np.asarray(bounds, dtype=np.float64)
    lo, hi = bounds[0], bounds[1]
    extents = hi - lo

    # 结果缓存键（同鞋底范围+参数+分辨率）
    result_cache_path = ""
    if GYROID_RESULT_USE_CACHE:
        code_sig = _runtime_code_signature()
        key = (
            f"res|code_sig={code_sig}|b={np.asarray(bounds).round(6).tolist()}|"
            f"cell={cell_size:.6f}|iso={isovalue:.6f}|res={int(resolution)}"
        )
        h = hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]
        os.makedirs(GYROID_RESULT_CACHE_DIR, exist_ok=True)
        result_cache_path = os.path.join(GYROID_RESULT_CACHE_DIR, f"gyroid_result_{h}.npz")
        _log(f"[gyroid] 最终结果缓存键: {h}")
        if os.path.exists(result_cache_path):
            try:
                z = np.load(result_cache_path)
                v = np.asarray(z["vertices"], dtype=np.float64)
                f = np.asarray(z["faces"], dtype=np.int64)
                mesh_cached = trimesh.Trimesh(vertices=v, faces=f, process=False)
                _log(f"[gyroid] 命中最终结果缓存: {result_cache_path}")
                _log("[gyroid] 缓存命中：跳过 mask 和 Marching Cubes，预计秒级返回")
                return mesh_cached
            except Exception:
                pass
        else:
            _log("[gyroid] 最终结果缓存未命中：将执行完整计算（预计较慢）")

    # 每个周期内有 resolution 个体素
    # 体素大小 = cell_size / resolution
    voxel_size = cell_size / resolution
    nx = max(4, int(np.ceil(extents[0] / voxel_size)) + 2)
    ny = max(4, int(np.ceil(extents[1] / voxel_size)) + 2)
    nz = max(4, int(np.ceil(extents[2] / voxel_size)) + 2)

    _log(f"[gyroid] 体素网格: {nx}x{ny}x{nz} = {nx*ny*nz:,} 体素, 体素尺寸={voxel_size:.3f}mm")

    # 构建坐标网格（世界坐标）
    xs = np.linspace(lo[0], hi[0], nx)
    ys = np.linspace(lo[1], hi[1], ny)
    zs = np.linspace(lo[2], hi[2], nz)
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing='ij')

    # Gyroid 隐式函数：sin(x)cos(y) + sin(y)cos(z) + sin(z)cos(x) = isovalue
    # 坐标归一化到周期 2π/cell_size
    freq = 2.0 * np.pi / cell_size
    Xf = X * freq
    Yf = Y * freq
    Zf = Z * freq
    F = (np.sin(Xf) * np.cos(Yf)
         + np.sin(Yf) * np.cos(Zf)
         + np.sin(Zf) * np.cos(Xf))

    _log(f"[gyroid] 函数值范围: [{F.min():.3f}, {F.max():.3f}], isovalue={isovalue}")

    # 建立鞋底体积的 voxel mask（精确，无损）：
    # 支持缓存与并行，不降低精度
    if GYROID_USE_VOXEL_MASK:
        batch = max(1024, int(GYROID_MASK_BATCH))
        workers = max(1, int(GYROID_MASK_WORKERS))
        if os.name == "nt" and workers > 1:
            # Windows + rtree 在多线程下可能触发访问冲突，强制回退单线程保证稳定性。
            _log("[gyroid] Windows 下 rtree 射线求交并行不稳定，自动改为单线程")
            workers = 1
        cache_path = _gyroid_mask_cache_path(
            sole_mesh, nx, ny, nz, cell_size, isovalue, voxel_size
        )

        inside_3d: np.ndarray | None = None
        if GYROID_MASK_USE_CACHE and os.path.exists(cache_path):
            try:
                inside_3d = np.load(cache_path)
                if inside_3d.shape != (nx, ny, nz):
                    inside_3d = None
                else:
                    _log(f"[gyroid] 命中mask缓存: {cache_path}（预计可节省较多时间）")
            except Exception:
                inside_3d = None

        if inside_3d is None:
            _log(f"[gyroid] 建立体素 mask（精确模式，线程数={workers}，预计耗时较长）...")
            pts = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)
            n_pts = len(pts)

            if workers == 1 or n_pts < batch * 2:
                inside = None
                if GYROID_USE_CPP_MASK and n_pts >= int(GYROID_CPP_MASK_MIN_POINTS):
                    _log("[gyroid] 尝试使用 C++ 扩展计算 mask...")
                    inside = _inside_by_cpp_mask(
                        sole_mesh=sole_mesh,
                        pts=pts,
                        nx=nx,
                        ny=ny,
                        grid_size=int(GYROID_CPP_MASK_GRID),
                    )
                if inside is None:
                    inside = _inside_by_ray_batch(sole_mesh, pts, batch)
            else:
                try:
                    chunk = int(np.ceil(n_pts / workers))
                    chunks = [pts[i:i + chunk] for i in range(0, n_pts, chunk)]

                    def _run(ch: np.ndarray) -> np.ndarray:
                        return _inside_by_ray_batch(sole_mesh, ch, batch)

                    with ThreadPool(processes=workers) as pool:
                        parts = pool.map(_run, chunks)
                    inside = np.concatenate(parts, axis=0)
                except Exception as e:
                    _log(f"[gyroid] 并行mask失败，回退单线程 ({type(e).__name__}: {e})")
                    inside = None

                if inside is None and GYROID_USE_CPP_MASK and n_pts >= int(GYROID_CPP_MASK_MIN_POINTS):
                    _log("[gyroid] 尝试使用 C++ 扩展计算 mask（并行失败后）...")
                    inside = _inside_by_cpp_mask(
                        sole_mesh=sole_mesh,
                        pts=pts,
                        nx=nx,
                        ny=ny,
                        grid_size=int(GYROID_CPP_MASK_GRID),
                    )

                if inside is None:
                    inside = _inside_by_ray_batch(sole_mesh, pts, batch)

            inside_3d = inside.reshape(nx, ny, nz)
            if GYROID_MASK_USE_CACHE:
                try:
                    np.save(cache_path, inside_3d)
                    _log(f"[gyroid] 已写入mask缓存: {cache_path}")
                except Exception:
                    pass

        F_masked = F.copy()
        F_masked[~inside_3d] = 3.0  # Gyroid 最大值是 sqrt(3)≈1.73，3.0 远离 isovalue
        _log(f"[gyroid] mask 内体素: {int(inside_3d.sum()):,} / {inside_3d.size:,}")
        F = F_masked
    else:
        _log("[gyroid] 已关闭 voxel mask（更快，但边界可能越界）")

    # Marching Cubes 提取等值面
    try:
        verts_idx, faces_idx, _, _ = skm.marching_cubes(
            F, level=isovalue, spacing=(voxel_size, voxel_size, voxel_size)
        )
    except Exception as e:
        _log(f"[gyroid] marching_cubes 失败: {e}")
        return trimesh.Trimesh()

    # 体素坐标 → 世界坐标
    verts_world = verts_idx + lo
    gyroid_mesh = trimesh.Trimesh(
        vertices=verts_world,
        faces=faces_idx,
        process=False,
    )
    gyroid_mesh.fix_normals()
    # 尝试修复为水密网格，提高布尔交成功率
    try:
        trimesh.repair.fill_holes(gyroid_mesh)
        trimesh.repair.fix_winding(gyroid_mesh)
    except Exception:
        pass

    st = _stats(gyroid_mesh)
    _log(f"[gyroid] Marching Cubes 结果: {st.vertices} 顶点, {st.faces} 面")

    # 裁剪：voxel mask 已保证大部分面在鞋底内，只做 AABB 兜底去除边缘浮点误差
    sb = sole_mesh.bounds
    cents = gyroid_mesh.triangles_center
    in_box = (
        (cents[:, 0] >= sb[0][0]) & (cents[:, 0] <= sb[1][0]) &
        (cents[:, 1] >= sb[0][1]) & (cents[:, 1] <= sb[1][1]) &
        (cents[:, 2] >= sb[0][2]) & (cents[:, 2] <= sb[1][2])
    )
    n_before = len(gyroid_mesh.faces)
    gyroid_mesh.update_faces(in_box)
    gyroid_mesh.remove_unreferenced_vertices()
    _log(f"[gyroid] AABB 兜底裁剪: {n_before} -> {len(gyroid_mesh.faces)} 面")

    st = _stats(gyroid_mesh)
    _log(f"[gyroid] 最终: {st.vertices} 顶点, {st.faces} 面")

    if GYROID_RESULT_USE_CACHE and result_cache_path:
        try:
            # 准备元数据
            metadata = {
                'method': 'gyroid',
                'cell_size': float(cell_size),
                'isovalue': float(isovalue),
                'resolution': int(resolution),
                'voxel_grid': f'{nx}x{ny}x{nz}',
                'total_voxels': int(nx * ny * nz),
                'voxel_size': float(voxel_size),
                'bounds': bounds.tolist(),
                'use_voxel_mask': bool(GYROID_USE_VOXEL_MASK),
            }
            
            np.savez_compressed(
                result_cache_path,
                vertices=np.asarray(gyroid_mesh.vertices, dtype=np.float32),
                faces=np.asarray(gyroid_mesh.faces, dtype=np.int32),
            )
            _log(f"[gyroid] 已写入最终结果缓存: {result_cache_path}")
            
            # 保存参数元数据
            import json
            from datetime import datetime
            json_path = result_cache_path.replace('.npz', '.json')
            metadata_with_stats = {
                'cached_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'vertices': int(len(gyroid_mesh.vertices)),
                'faces': int(len(gyroid_mesh.faces)),
                'file_size_mb': round(os.path.getsize(result_cache_path) / (1024 * 1024), 2),
                'parameters': metadata,
            }
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(metadata_with_stats, f, indent=2, ensure_ascii=False)
            _log(f"[gyroid] 已保存参数记录: {json_path}")
        except Exception as e:
            _log(f"[gyroid] 写缓存失败 ({type(e).__name__}: {e})")

    return gyroid_mesh


def generate_gyroid_unit_cell(
    cell_size: float = GYROID_CELL_SIZE,
    isovalue: float = GYROID_ISOVALUE,
    resolution: int = GYROID_RESOLUTION,
) -> trimesh.Trimesh:
    """生成单个 Gyroid 单胞，用于观察当前公式对应的原始晶格形态。"""
    try:
        from skimage import measure as skm  # type: ignore
    except ImportError:
        _log("[gyroid_unit] 需要 scikit-image: pip install scikit-image")
        return trimesh.Trimesh()

    samples = max(16, int(resolution) + 1)
    xs = np.linspace(0.0, cell_size, samples)
    ys = np.linspace(0.0, cell_size, samples)
    zs = np.linspace(0.0, cell_size, samples)
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")

    freq = 2.0 * np.pi / cell_size
    Xf = X * freq
    Yf = Y * freq
    Zf = Z * freq
    F = (
        np.sin(Xf) * np.cos(Yf)
        + np.sin(Yf) * np.cos(Zf)
        + np.sin(Zf) * np.cos(Xf)
    )

    step = float(cell_size) / float(samples - 1)
    try:
        verts, faces, _, _ = skm.marching_cubes(
            F,
            level=float(isovalue),
            spacing=(step, step, step),
        )
    except Exception as e:
        _log(f"[gyroid_unit] marching_cubes 失败: {e}")
        return trimesh.Trimesh()

    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    try:
        mesh.fix_normals()
    except Exception:
        pass
    st = _stats(mesh)
    _log(f"[gyroid_unit] 单胞生成完成: {st.vertices} 顶点, {st.faces} 面")
    return mesh


def generate_voronoi_lattice_in_sole(
    sole_mesh: trimesh.Trimesh,
    cell_size: float = VORONOI_CELL_SIZE,
    strut_thickness: float = VORONOI_STRUT_THICKNESS,
    bounds: np.ndarray = None,
) -> trimesh.Trimesh:
    """
    多层 Voronoi 杆系：按各层 z 真正采样，并增强层间与平面覆盖。
    
    参数:
        sole_mesh: 鞋底网格（用于体内判断，必须水密）
        cell_size: Voronoi单胞尺寸
        strut_thickness: 杆件厚度
        bounds: 可选的自定义包围盒 shape=(2,3)，用于区域优化。
                如果为None则使用 sole_mesh.bounds
    """
    t_total_start = time.perf_counter()
    t_sample_total = 0.0
    t_segment_filter_total = 0.0
    
    # 使用自定义包围盒或默认从mesh获取
    if bounds is None:
        bounds = sole_mesh.bounds
    else:
        bounds = np.asarray(bounds, dtype=np.float64)
    
    bounds_xy = bounds[:, :2]
    z_min, z_max = float(bounds[0, 2]), float(bounds[1, 2])
    z_layers = _voronoi_layer_z_values(z_min, z_max, VORONOI_Z_LAYERS)

    layer_segments: list[tuple[np.ndarray, float]] = []
    connector_candidates: list[np.ndarray] = []
    seed_count = _compute_voronoi_seed_count(bounds_xy, cell_size)

    for layer_idx, z_level in enumerate(z_layers):
        t_layer_sample_start = time.perf_counter()
        
        # 在完整鞋底内采样（不使用区域包围盒限制）
        # 这样可以确保有足够的种子点
        seeds = _sample_inside_points_2d(
            sole_mesh=sole_mesh,
            count=seed_count,
            z_level=z_level,
            rng_seed=42 + layer_idx * 97,
            bounds_xy=None,  # 不限制采样范围，使用完整鞋底
        )
        
        # 如果使用了自定义包围盒，在采样后过滤种子点
        if bounds is not None and bounds_xy is not None:
            in_region = (
                (seeds[:, 0] >= bounds_xy[0, 0]) & (seeds[:, 0] <= bounds_xy[1, 0]) &
                (seeds[:, 1] >= bounds_xy[0, 1]) & (seeds[:, 1] <= bounds_xy[1, 1])
            )
            seeds_in_region = seeds[in_region]
            
            # 如果区域内种子太少，补充一些
            if len(seeds_in_region) < 4:
                _log(f"[voronoi] layer={layer_idx} 区域内种子不足({len(seeds_in_region)})，使用完整采样")
                # 使用完整采样结果
            else:
                seeds = seeds_in_region
                _log(f"[voronoi] layer={layer_idx} 区域过滤: {len(seeds)}/{len(seeds)} 种子在区域内")
        
        t_layer_sample_end = time.perf_counter()
        t_sample_total += (t_layer_sample_end - t_layer_sample_start)
        _log(f"[voronoi] layer={layer_idx} z={z_level:.3f} seed_count target={seed_count}, accepted={len(seeds)}")
        if len(seeds) < 4:
            continue

        connector_candidates.append(seeds)
        vor = Voronoi(seeds)
        segments = _voronoi_finite_segments(vor)
        if len(segments) == 0:
            continue

        t_layer_filter_start = time.perf_counter()
        inside_seg = _batch_voronoi_wall_inside_sole(
            sole_mesh=sole_mesh,
            segments=segments,
            z_mid=z_level,
            n_along=VORONOI_SEGMENT_SAMPLES,
            thickness=strut_thickness,
        )
        segments = segments[inside_seg]
        t_layer_filter_end = time.perf_counter()
        t_segment_filter_total += (t_layer_filter_end - t_layer_filter_start)
        _log(f"[voronoi] layer={layer_idx} inside wall segments={len(segments)}")
        if len(segments) == 0:
            continue

        if len(segments) > MAX_LATTICE_UNITS:
            segments = segments[:MAX_LATTICE_UNITS]
            _log(f"[voronoi] layer={layer_idx} truncated to MAX_LATTICE_UNITS={MAX_LATTICE_UNITS}")

        layer_segments.append((segments, z_level))

    _log(f"[voronoi][timing] 分层采样: {t_sample_total:.3f}s")
    _log(f"[voronoi][timing] 边段筛选: {t_segment_filter_total:.3f}s")

    rods: list[trimesh.Trimesh] = []
    t_rods_start = time.perf_counter()
    for segments, z_level in layer_segments:
        for seg in segments:
            start3 = np.array([seg[0, 0], seg[0, 1], z_level], dtype=np.float64)
            end3 = np.array([seg[1, 0], seg[1, 1], z_level], dtype=np.float64)
            rod = _segment_wall_mesh(start3, end3, strut_thickness)
            # 只添加有效的杆件
            if not rod.is_empty and len(rod.faces) > 0:
                rods.append(rod)

    if connector_candidates and len(z_layers) >= 2:
        all_xy = np.vstack(connector_candidates)
        keep_n = min(len(all_xy), max(36, seed_count // 2))
        step = max(1, len(all_xy) // keep_n)
        sampled_xy = all_xy[::step][:keep_n]
        for xy in sampled_xy:
            for z0, z1 in zip(z_layers[:-1], z_layers[1:]):
                start3 = np.array([xy[0], xy[1], z0], dtype=np.float64)
                end3 = np.array([xy[0], xy[1], z1], dtype=np.float64)
                check_pts = np.array([
                    start3,
                    end3,
                    0.5 * (start3 + end3),
                ])
                if np.all(_safe_contains(sole_mesh, check_pts)):
                    rod = _segment_wall_mesh(start3, end3, strut_thickness * 0.9)
                    # 只添加有效的杆件
                    if not rod.is_empty and len(rod.faces) > 0:
                        rods.append(rod)

    t_rods_end = time.perf_counter()
    _log(f"[voronoi][timing] 杆件生成: {t_rods_end - t_rods_start:.3f}s, 杆件数={len(rods)}")

    if not rods:
        _log(f"[voronoi][timing] 总耗时: {time.perf_counter() - t_total_start:.3f}s")
        _log("[voronoi] 警告：没有生成任何有效杆件")
        return trimesh.Trimesh()

    # 拼接前再次验证
    valid_rods = [r for r in rods if not r.is_empty and len(r.faces) > 0]
    if not valid_rods:
        _log("[voronoi] 警告：所有杆件都无效")
        return trimesh.Trimesh()
    
    _log(f"[voronoi] 有效杆件数: {len(valid_rods)}/{len(rods)}")
    
    try:
        lattice = trimesh.util.concatenate(valid_rods)
    except Exception as e:
        _log(f"[voronoi] 杆件拼接失败: {e}")
        return trimesh.Trimesh()
    
    # 验证拼接结果
    if lattice.is_empty or len(lattice.faces) == 0:
        _log("[voronoi] 警告：拼接后网格为空")
        return trimesh.Trimesh()
    
    # 检查并修复法向量
    try:
        lattice.fix_normals()
    except Exception as e:
        _log(f"[voronoi] 法向量修复失败: {e}")
    
    n_before = len(lattice.faces)
    t_trim_start = time.perf_counter()
    lattice = trim_lattice_to_sole_volume(sole_mesh, lattice)
    t_trim_end = time.perf_counter()
    _log(f"[voronoi][timing] 最终裁剪: {t_trim_end - t_trim_start:.3f}s")
    _log(f"[voronoi] 强约束裁剪: {n_before} -> {len(lattice.faces)} 面")
    st = _stats(lattice)
    _log(f"[voronoi] lattice mesh: {st.vertices} vertices, {st.faces} faces")
    _log(f"[voronoi][timing] 总耗时: {t_trim_end - t_total_start:.3f}s")
    return lattice




def build_sole_with_lattice(
    sole_path: str = SOLE_STL_PATH,
    lattice_path: str = LATTICE_STL_PATH,
    enable_final_trim: bool = True,
) -> tuple[trimesh.Trimesh, trimesh.Trimesh, trimesh.Trimesh]:
    """
    从路径加载鞋底和晶格单元，生成鞋底内部的晶格整体，
    并返回：(鞋底网格, 晶格整体网格, 合并视图用的简单并集网格)。
    """
    sole_mesh = _z_scale_mesh(load_mesh(sole_path), SOLE_HEIGHT_SCALE)
    try:
        sole_mesh.fix_normals()
    except Exception:
        pass
    if not sole_mesh.is_watertight:
        _log(
            "[build_sole_with_lattice] 鞋底网格非闭合 (is_watertight=False)，"
            "contains 裁剪可能不准；建议在 CAD 中修复 STL。"
        )

    # 缓存命中标志（用于避免重复写入缓存）
    cache_hit = False

    if LATTICE_METHOD == "gyroid":
        lattice_in_sole = generate_gyroid_lattice(
            sole_mesh,
            cell_size=GYROID_CELL_SIZE,
            isovalue=GYROID_ISOVALUE,
            resolution=GYROID_RESOLUTION,
        )
    elif LATTICE_METHOD == "voronoi_2_5d":
        lattice_in_sole = generate_voronoi_lattice_in_sole(
            sole_mesh=sole_mesh,
            cell_size=VORONOI_CELL_SIZE,
            strut_thickness=VORONOI_STRUT_THICKNESS,
        )
    elif LATTICE_METHOD == "voronoi_implicit":
        lattice_in_sole = generate_voronoi_lattice_implicit(
            sole_mesh=sole_mesh,
            cell_size=VORONOI_IMPLICIT_CELL_SIZE,
            wall_thickness=VORONOI_IMPLICIT_WALL_THICKNESS,
            resolution=VORONOI_IMPLICIT_RESOLUTION,
            seed_density_factor=VORONOI_IMPLICIT_SEED_DENSITY_FACTOR,
        )
    elif LATTICE_METHOD == "tile_unit":
        tile_cache_path = ""
        if TILE_RESULT_USE_CACHE:
            tile_cache_path = _tile_result_cache_path(sole_path, lattice_path, sole_mesh)
            if os.path.exists(tile_cache_path):
                mesh_cached = _load_cached_mesh_npz(tile_cache_path)
                if mesh_cached is not None:
                    _log(f"[tile_unit] 命中最终结果缓存: {tile_cache_path}")
                    _log("[tile_unit] 缓存命中：跳过 tile 平铺与裁剪重算，预计可快速返回")
                    lattice_in_sole = mesh_cached
                    # 命中缓存，跳过后续生成和写入
                    cache_hit = True
                else:
                    _log("[tile_unit] 缓存文件存在但读取失败，回退到冷启动重算")
                    cache_hit = False
            else:
                _log("[tile_unit] 最终结果缓存未命中：本次将执行完整 tile 平铺与裁剪")
                cache_hit = False
        else:
            _log("[tile_unit] 最终结果缓存已关闭：本次将执行完整 tile 平铺与裁剪")
            cache_hit = False

        if not cache_hit:
            lattice_raw = load_mesh(lattice_path)
            lattice_raw = simplify_mesh(lattice_raw, LATTICE_UNIT_TARGET_FACES)
            lattice_unit = normalize_lattice_unit(lattice_raw)
            layout = (LATTICE_LAYOUT or "single_scaled").strip().lower()
            if layout == "grid":
                if LATTICE_TILE_MATCH_SOLE_HEIGHT:
                    unit_t = prepare_lattice_unit_for_planar_tiling(lattice_unit, sole_mesh)
                    lattice_in_sole = generate_lattice_in_sole(
                        sole_mesh, unit_t, planar_tiling=True
                    )
                else:
                    lattice_in_sole = generate_lattice_in_sole(sole_mesh, lattice_unit)
            elif layout in ("single_scaled", "single", "one"):
                lattice_in_sole = generate_lattice_single_in_sole_bbox(sole_mesh, lattice_unit)
            else:
                _log(f"[build_sole_with_lattice] 未知 LATTICE_LAYOUT={LATTICE_LAYOUT!r}，改用 single_scaled")
                lattice_in_sole = generate_lattice_single_in_sole_bbox(sole_mesh, lattice_unit)
    else:
        raise ValueError(f"不支持的晶格生成方法: {LATTICE_METHOD!r}")

    if len(lattice_in_sole.faces) == 0:
        _log(
            "[build_sole_with_lattice] 晶格三角面数为 0："
            "请确认 lattice.stl 存在且与鞋底在同一尺度；"
            "或尝试略增大 SOLE_HEIGHT_SCALE、调整 TILE_SPACING_FACTOR。"
        )

    if enable_final_trim and CLIP_LATTICE_TO_SOLE and len(lattice_in_sole.faces) > 0 and LATTICE_METHOD not in ("voronoi_2_5d", "voronoi_implicit"):
        lattice_in_sole = trim_lattice_to_sole_volume(
            sole_mesh,
            lattice_in_sole,
            prefer_boolean=(LATTICE_METHOD == "tile_unit"),
        )

    # 只在未命中缓存时写入缓存
    if LATTICE_METHOD == "tile_unit" and TILE_RESULT_USE_CACHE and tile_cache_path and not cache_hit and len(lattice_in_sole.faces) > 0:
        try:
            # 准备元数据
            metadata = {
                'method': 'tile_unit',
                'sole_path': sole_path,
                'lattice_path': lattice_path,
                'layout': LATTICE_LAYOUT,
                'single_fit': LATTICE_SINGLE_FIT,
                'tile_match_sole_height': bool(LATTICE_TILE_MATCH_SOLE_HEIGHT),
                'tile_shrink': float(LATTICE_TILE_SHRINK),
                'lattice_scale': float(LATTICE_SCALE),
                'unit_target_faces': int(LATTICE_UNIT_TARGET_FACES),
                'spacing_factor': float(TILE_SPACING_FACTOR),
                'boundary_margin': float(TILE_BOUNDARY_MARGIN),
                'edge_fill': bool(TILE_EDGE_FILL),
                'corner_inside': bool(TILE_PLACE_IF_CORNER_INSIDE),
                'aabb_overlap': bool(TILE_PLACE_IF_AABB_OVERLAP),
                'isotropic_step': bool(TILE_ISOTROPIC_STEP),
                'clip_to_sole': bool(CLIP_LATTICE_TO_SOLE),
                'trim_mode': LATTICE_TRIM_MODE,
            }
            
            _save_cached_mesh_npz(tile_cache_path, lattice_in_sole, metadata)
            _log(f"[tile_unit] 已写入最终结果缓存: {tile_cache_path}")
        except Exception as e:
            _log(f"[tile_unit] 写缓存失败 ({type(e).__name__}: {e})")

    if len(lattice_in_sole.faces) == 0:
        combined = sole_mesh.copy()
    else:
        combined = trimesh.util.concatenate([sole_mesh, lattice_in_sole])

    st = _stats(combined)
    _log(f"[生成完成] 合并网格: {st.vertices} 顶点, {st.faces} 面")

    return sole_mesh, lattice_in_sole, combined



# =============================================================================
# Voronoi Implicit（隐函数 Voronoi）
# =============================================================================

def _sample_seeds_3d_uniform(
    sole_mesh: trimesh.Trimesh,
    cell_size: float,
    density_factor: float = 1.0,
    bounds: np.ndarray = None,
) -> np.ndarray:
    """
    在 3D 空间均匀采样种子点（标准 Voronoi 方法）
    
    参数:
        sole_mesh: 鞋底网格（用于体内判断）
        cell_size: 胞元尺寸（mm）
        density_factor: 密度系数（1.0 = 标准密度）
        bounds: 可选的自定义包围盒 shape=(2,3)
    
    返回:
        seeds: 种子点坐标 shape=(n_seeds, 3)
    """
    # 获取包围盒
    if bounds is None:
        bounds = sole_mesh.bounds
    else:
        bounds = np.asarray(bounds, dtype=np.float64)
    
    lo, hi = bounds[0], bounds[1]
    extents = hi - lo
    
    # 计算目标种子点数量（基于体积）
    volume = np.prod(extents)
    target_seeds = int(volume / (cell_size ** 3) * density_factor)
    target_seeds = max(10, target_seeds)  # 至少 10 个种子点
    
    _log(f"[voronoi_implicit] 目标种子点数: {target_seeds}")
    _log(f"[voronoi_implicit] 包围盒体积: {volume:.2f} mm³")
    
    # 采样策略：先网格采样，再随机补充
    seeds_list = []
    
    # 1. 网格采样（确保均匀分布）
    n_per_dim = max(2, int(np.ceil(target_seeds ** (1/3))))
    xs = np.linspace(lo[0], hi[0], n_per_dim)
    ys = np.linspace(lo[1], hi[1], n_per_dim)
    zs = np.linspace(lo[2], hi[2], n_per_dim)
    
    grid_points = np.array(np.meshgrid(xs, ys, zs, indexing='ij')).reshape(3, -1).T
    
    # 添加随机抖动（避免过于规则）
    jitter = (np.random.random(grid_points.shape) - 0.5) * (extents / n_per_dim) * 0.5
    grid_points = np.clip(grid_points + jitter, lo, hi)
    
    # 筛选在鞋底内的点
    inside_mask = _safe_contains(sole_mesh, grid_points)
    if np.any(inside_mask):
        seeds_list.append(grid_points[inside_mask])
    
    current_count = sum(len(s) for s in seeds_list)
    _log(f"[voronoi_implicit] 网格采样: {current_count} 个种子点")
    
    # 2. 随机补充（如果网格采样不足）
    if current_count < target_seeds:
        need = target_seeds - current_count
        max_attempts = need * 20  # 最多尝试 20 倍
        
        _log(f"[voronoi_implicit] 需要补充 {need} 个种子点，开始随机采样...")
        
        batch_size = min(10000, max_attempts)
        attempts = 0
        
        while current_count < target_seeds and attempts < max_attempts:
            # 批量生成随机点
            batch = np.random.uniform(
                lo, hi, 
                size=(batch_size, 3)
            )
            
            # 筛选在鞋底内的点
            inside_mask = _safe_contains(sole_mesh, batch)
            if np.any(inside_mask):
                seeds_list.append(batch[inside_mask])
                current_count = sum(len(s) for s in seeds_list)
            
            attempts += batch_size
        
        _log(f"[voronoi_implicit] 随机采样: 总共 {current_count} 个种子点")
    
    # 合并所有种子点
    if not seeds_list:
        _log("[voronoi_implicit] 警告：未能采样到任何种子点")
        return np.zeros((0, 3), dtype=np.float64)
    
    seeds = np.vstack(seeds_list)
    
    # 如果超过目标数量，随机选择
    if len(seeds) > target_seeds:
        indices = np.random.choice(len(seeds), target_seeds, replace=False)
        seeds = seeds[indices]
    
    _log(f"[voronoi_implicit] 最终种子点数: {len(seeds)}")
    
    return seeds.astype(np.float64)


def _extract_voronoi_edges(seeds: np.ndarray, bounds: np.ndarray) -> np.ndarray:
    """
    计算 3D Voronoi 图并提取所有边（骨架线段）
    
    在 3D Voronoi 图中，我们使用 ridge_points 来获取连接相邻种子点的边。
    每条 ridge 对应两个种子点，我们直接连接这两个种子点作为骨架边。
    
    参数:
        seeds: 种子点坐标 shape=(n_seeds, 3)
        bounds: 包围盒 shape=(2, 3)
    
    返回:
        edges: 边的端点坐标 shape=(n_edges, 2, 3)
               每条边由两个 3D 点定义
    """
    from scipy.spatial import Voronoi
    
    _log("[voronoi_implicit] 计算 3D Voronoi 图...")
    t_start = time.perf_counter()
    
    try:
        # 计算 Voronoi 图
        vor = Voronoi(seeds)
        
        # 提取 Voronoi 边：连接相邻种子点
        edges_list = []
        
        # ridge_points 包含每条 ridge 对应的两个种子点索引
        for point_indices in vor.ridge_points:
            p0_idx, p1_idx = point_indices
            p0 = seeds[p0_idx]
            p1 = seeds[p1_idx]
            
            # 检查边是否在包围盒内（或附近）
            lo, hi = bounds[0], bounds[1]
            margin = np.max(hi - lo) * 0.1  # 10% 边距
            
            # 检查边的中点是否在包围盒附近
            mid = (p0 + p1) / 2.0
            if np.all(mid >= lo - margin) and np.all(mid <= hi + margin):
                edges_list.append([p0, p1])
        
        if not edges_list:
            _log("[voronoi_implicit] 警告：未提取到任何有效的 Voronoi 边")
            return np.zeros((0, 2, 3), dtype=np.float64)
        
        edges = np.array(edges_list, dtype=np.float64)
        
        t_end = time.perf_counter()
        _log(f"[voronoi_implicit] Voronoi 图计算耗时: {t_end - t_start:.2f}s")
        _log(f"[voronoi_implicit] 提取到 {len(edges)} 条 Voronoi 边")
        
        return edges
        
    except Exception as e:
        _log(f"[voronoi_implicit] Voronoi 图计算失败: {e}")
        import traceback
        _log(f"[voronoi_implicit] 详细错误: {traceback.format_exc()}")
        return np.zeros((0, 2, 3), dtype=np.float64)


def _point_to_segment_distance(points: np.ndarray, seg_start: np.ndarray, seg_end: np.ndarray) -> np.ndarray:
    """
    计算点到线段的最短距离（向量化）
    
    参数:
        points: 查询点 shape=(n, 3)
        seg_start: 线段起点 shape=(3,)
        seg_end: 线段终点 shape=(3,)
    
    返回:
        distances: 距离 shape=(n,)
    """
    # 线段方向向量
    seg_vec = seg_end - seg_start
    seg_len_sq = np.dot(seg_vec, seg_vec)
    
    if seg_len_sq < 1e-12:
        # 退化为点
        return np.linalg.norm(points - seg_start, axis=1)
    
    # 计算投影参数 t（点在线段上的投影位置）
    # t = 0 表示在 seg_start，t = 1 表示在 seg_end
    t = np.dot(points - seg_start, seg_vec) / seg_len_sq
    t = np.clip(t, 0.0, 1.0)  # 限制在线段范围内
    
    # 计算最近点
    closest = seg_start + t[:, np.newaxis] * seg_vec
    
    # 计算距离
    distances = np.linalg.norm(points - closest, axis=1)
    
    return distances


def _compute_voronoi_distance_field_python(
    X: np.ndarray,
    Y: np.ndarray,
    Z: np.ndarray,
    seeds: np.ndarray,
    wall_thickness: float,
    bounds: np.ndarray
) -> np.ndarray:
    """
    Python 实现的 Voronoi 骨架距离场计算（正确流程）
    
    流程：
    1. 计算 3D Voronoi 图
    2. 提取 Voronoi 边（骨架线段）
    3. 计算每个体素到最近骨架线的距离
    4. 构建管状隐函数：F = distance_to_skeleton - wall_thickness/2
    
    参数:
        X, Y, Z: 体素坐标网格 shape=(nx, ny, nz)
        seeds: 种子点坐标 shape=(n_seeds, 3)
        wall_thickness: 骨架管径（建议为 cell_size 的 0.3-0.5 倍）
        bounds: 包围盒 shape=(2, 3)
    
    返回:
        F: 隐函数值 shape=(nx, ny, nz)
           F < 0: 在骨架管内（保留）
           F > 0: 在骨架管外（移除）
    """
    _log("[voronoi_implicit] 使用正确流程：Voronoi 图 → 提取边 → 距离场")
    _log(f"[voronoi_implicit] 参数: wall_thickness={wall_thickness:.3f}")
    t_start = time.perf_counter()
    
    # 1. 提取 Voronoi 边
    edges = _extract_voronoi_edges(seeds, bounds)
    
    if len(edges) == 0:
        _log("[voronoi_implicit] 错误：未能提取到 Voronoi 边")
        return np.ones(X.shape, dtype=np.float32) * 100.0
    
    # 2. 计算每个体素到最近骨架线的距离
    _log("[voronoi_implicit] 计算到骨架线的距离场...")
    t_dist_start = time.perf_counter()
    
    # 展平坐标网格
    points = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
    n_points = len(points)
    
    # 初始化距离为无穷大
    min_distances = np.full(n_points, np.inf, dtype=np.float32)
    
    # 对每条边计算距离，保留最小值
    batch_size = 10000  # 批处理以节省内存
    
    for i, edge in enumerate(edges):
        if (i + 1) % 100 == 0:
            _log(f"[voronoi_implicit] 处理边 {i+1}/{len(edges)}...")
        
        seg_start, seg_end = edge[0], edge[1]
        
        # 批处理计算距离
        for j in range(0, n_points, batch_size):
            batch_points = points[j:j+batch_size]
            batch_distances = _point_to_segment_distance(batch_points, seg_start, seg_end)
            min_distances[j:j+batch_size] = np.minimum(
                min_distances[j:j+batch_size],
                batch_distances
            )
    
    t_dist_end = time.perf_counter()
    _log(f"[voronoi_implicit] 距离场计算耗时: {t_dist_end - t_dist_start:.2f}s")
    
    # 3. 构建管状隐函数
    # F(x) = distance_to_skeleton - radius
    # 其中 radius = wall_thickness / 2
    radius = wall_thickness / 2.0
    F = min_distances - radius
    
    # 统计信息
    n_inside = np.sum(F < 0)
    ratio = n_inside / len(F) * 100
    _log(f"[voronoi_implicit] 骨架体积占比: {ratio:.1f}%")
    _log(f"[voronoi_implicit] 距离场范围: [{min_distances.min():.3f}, {min_distances.max():.3f}]")
    
    if ratio < 0.5:
        _log(f"[voronoi_implicit] 警告：骨架体积占比过低（{ratio:.1f}%），建议增大 wall_thickness")
        _log(f"[voronoi_implicit] 提示：wall_thickness 应为 cell_size 的 0.3-0.5 倍")
    elif ratio > 40.0:
        _log(f"[voronoi_implicit] 警告：骨架体积占比过高（{ratio:.1f}%），建议减小 wall_thickness")
    
    t_end = time.perf_counter()
    _log(f"[voronoi_implicit] Python 距离场总耗时: {t_end - t_start:.2f}s")
    
    return F.reshape(X.shape).astype(np.float32)


def _compute_voronoi_distance_field_cpp(
    X: np.ndarray,
    Y: np.ndarray,
    Z: np.ndarray,
    seeds: np.ndarray,
    wall_thickness: float,
    bounds: np.ndarray
) -> np.ndarray:
    """
    C++ 加速的 Voronoi 隐函数距离场计算
    
    参数:
        X, Y, Z: 体素坐标网格 shape=(nx, ny, nz)
        seeds: 种子点坐标 shape=(n_seeds, 3)
        wall_thickness: 壁厚参数
        bounds: 包围盒 shape=(2, 3)
    
    返回:
        F: 隐函数值 shape=(nx, ny, nz)
    """
    try:
        from core.cpp import cpp_voronoi_implicit  # type: ignore
        
        _log("[voronoi_implicit] 使用 C++ 加速计算距离场...")
        t_start = time.perf_counter()
        
        # 展平坐标网格
        voxel_coords = np.column_stack([
            X.ravel(), Y.ravel(), Z.ravel()
        ]).astype(np.float64)
        
        # 调用 C++ 函数
        F_flat = cpp_voronoi_implicit.compute_voronoi_distance_field(
            voxel_coords,
            seeds.astype(np.float64),
            float(wall_thickness)
        )
        
        t_end = time.perf_counter()
        _log(f"[voronoi_implicit] C++ 距离场计算耗时: {t_end - t_start:.2f}s")
        
        return F_flat.reshape(X.shape)
        
    except ImportError:
        _log("[voronoi_implicit] C++ 扩展未安装，回退到 Python 实现")
        return _compute_voronoi_distance_field_python(X, Y, Z, seeds, wall_thickness, bounds)
    except Exception as e:
        _log(f"[voronoi_implicit] C++ 计算失败: {e}，回退到 Python 实现")
        return _compute_voronoi_distance_field_python(X, Y, Z, seeds, wall_thickness, bounds)


def generate_voronoi_lattice_implicit(
    sole_mesh: trimesh.Trimesh,
    cell_size: float = VORONOI_IMPLICIT_CELL_SIZE,
    wall_thickness: float = VORONOI_IMPLICIT_WALL_THICKNESS,
    resolution: int = VORONOI_IMPLICIT_RESOLUTION,
    seed_density_factor: float = VORONOI_IMPLICIT_SEED_DENSITY_FACTOR,
    bounds: np.ndarray = None,
) -> trimesh.Trimesh:
    """
    用隐函数方法生成 Voronoi 晶格（标准 3D 方法）
    
    算法流程:
    1. 在鞋底包围盒内 3D 均匀采样种子点
    2. 建立体素网格
    3. 计算每个体素的 Voronoi 隐函数值（到最近和第二近种子点的距离差）
    4. 用 Marching Cubes 提取等值面
    5. 裁剪到鞋底体积内
    
    参数:
        sole_mesh: 鞋底网格（用于体内判断，必须水密）
        cell_size: Voronoi 胞元尺寸（mm），控制种子点密度
        wall_thickness: 壁厚参数（mm），类似 Gyroid 的 isovalue
        resolution: 分辨率，每个胞元内的体素数
        seed_density_factor: 种子点密度系数（1.0 = 标准密度）
        bounds: 可选的自定义包围盒 shape=(2,3)，用于区域优化
    
    返回:
        Voronoi 晶格网格
    """
    try:
        from skimage import measure as skm  # type: ignore
    except ImportError:
        _log("[voronoi_implicit] 需要 scikit-image: pip install scikit-image")
        return trimesh.Trimesh()
    
    t_total_start = time.perf_counter()
    
    # 1. 获取包围盒
    if bounds is None:
        bounds = sole_mesh.bounds
    else:
        bounds = np.asarray(bounds, dtype=np.float64)
    
    lo, hi = bounds[0], bounds[1]
    extents = hi - lo
    
    _log(f"[voronoi_implicit] 包围盒: {lo} ~ {hi}")
    _log(f"[voronoi_implicit] 尺寸: {extents}")
    
    # 2. 缓存检查
    result_cache_path = ""
    if VORONOI_IMPLICIT_USE_CACHE:
        code_sig = _runtime_code_signature()
        key = (
            f"voronoi_implicit_3d|code_sig={code_sig}|"
            f"bounds={np.asarray(bounds).round(6).tolist()}|"
            f"cell={cell_size:.6f}|wall={wall_thickness:.6f}|"
            f"res={int(resolution)}|density={seed_density_factor:.3f}"
        )
        h = hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]
        os.makedirs(VORONOI_IMPLICIT_CACHE_DIR, exist_ok=True)
        result_cache_path = os.path.join(
            VORONOI_IMPLICIT_CACHE_DIR,
            f"voronoi_implicit_3d_{h}.npz"
        )
        
        _log(f"[voronoi_implicit] 缓存键: {h}")
        
        if os.path.exists(result_cache_path):
            try:
                z = np.load(result_cache_path)
                v = np.asarray(z["vertices"], dtype=np.float64)
                f = np.asarray(z["faces"], dtype=np.int64)
                mesh_cached = trimesh.Trimesh(vertices=v, faces=f, process=False)
                _log(f"[voronoi_implicit] 命中缓存: {result_cache_path}")
                _log(f"[voronoi_implicit] 缓存命中，跳过计算（秒级返回）")
                return mesh_cached
            except Exception as e:
                _log(f"[voronoi_implicit] 缓存加载失败: {e}")
        else:
            _log("[voronoi_implicit] 缓存未命中，将执行完整计算")
    
    # 3. 采样种子点（标准 3D 均匀采样）
    _log("[voronoi_implicit] 开始 3D 均匀采样种子点...")
    t_sample_start = time.perf_counter()
    
    seeds = _sample_seeds_3d_uniform(
        sole_mesh=sole_mesh,
        cell_size=cell_size,
        density_factor=seed_density_factor,
        bounds=bounds,
    )
    
    t_sample_end = time.perf_counter()
    _log(f"[voronoi_implicit] 种子采样耗时: {t_sample_end - t_sample_start:.2f}s")
    
    if len(seeds) == 0:
        _log("[voronoi_implicit] 错误：未能采样到任何种子点")
        return trimesh.Trimesh()
    
    _log(f"[voronoi_implicit] 总种子点数: {len(seeds)}")
    
    # 4. 建立体素网格
    voxel_size = cell_size / resolution
    nx = max(4, int(np.ceil(extents[0] / voxel_size)) + 2)
    ny = max(4, int(np.ceil(extents[1] / voxel_size)) + 2)
    nz = max(4, int(np.ceil(extents[2] / voxel_size)) + 2)
    
    _log(f"[voronoi_implicit] 体素网格: {nx}×{ny}×{nz} = {nx*ny*nz:,} 体素")
    _log(f"[voronoi_implicit] 体素尺寸: {voxel_size:.3f} mm")
    
    # 构建坐标网格
    xs = np.linspace(lo[0], hi[0], nx)
    ys = np.linspace(lo[1], hi[1], ny)
    zs = np.linspace(lo[2], hi[2], nz)
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing='ij')
    
    # 5. 计算距离场（C++ 或 Python）
    if VORONOI_IMPLICIT_USE_CPP:
        F = _compute_voronoi_distance_field_cpp(X, Y, Z, seeds, wall_thickness, bounds)
    else:
        F = _compute_voronoi_distance_field_python(X, Y, Z, seeds, wall_thickness, bounds)
    
    _log(f"[voronoi_implicit] 隐函数值范围: [{F.min():.3f}, {F.max():.3f}]")
    
    # 6. Marching Cubes 提取等值面
    _log("[voronoi_implicit] 提取等值面（Marching Cubes）...")
    t_mc_start = time.perf_counter()
    
    try:
        verts, faces, _, _ = skm.marching_cubes(
            F,
            level=0.0,
            spacing=(voxel_size, voxel_size, voxel_size)
        )
        
        # 调整坐标到世界空间
        verts += lo
        
        mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
        
        t_mc_end = time.perf_counter()
        _log(f"[voronoi_implicit] Marching Cubes 耗时: {t_mc_end - t_mc_start:.2f}s")
        _log(f"[voronoi_implicit] 提取网格: {len(verts)} 顶点, {len(faces)} 面")
        
    except Exception as e:
        _log(f"[voronoi_implicit] Marching Cubes 失败: {e}")
        return trimesh.Trimesh()
    
    # 7. 裁剪到鞋底体积
    _log("[voronoi_implicit] 裁剪到鞋底体积...")
    t_trim_start = time.perf_counter()
    
    n_before = len(mesh.faces)
    mesh = trim_lattice_to_sole_volume(sole_mesh, mesh)
    
    t_trim_end = time.perf_counter()
    _log(f"[voronoi_implicit] 裁剪耗时: {t_trim_end - t_trim_start:.2f}s")
    _log(f"[voronoi_implicit] 裁剪: {n_before} -> {len(mesh.faces)} 面")
    
    # 8. 保存缓存
    if VORONOI_IMPLICIT_USE_CACHE and result_cache_path and len(mesh.faces) > 0:
        try:
            np.savez_compressed(
                result_cache_path,
                vertices=mesh.vertices,
                faces=mesh.faces
            )
            _log(f"[voronoi_implicit] 已保存缓存: {result_cache_path}")
        except Exception as e:
            _log(f"[voronoi_implicit] 缓存保存失败: {e}")
    
    # 9. 统计信息
    st = _stats(mesh)
    _log(f"[voronoi_implicit] 最终网格: {st.vertices} 顶点, {st.faces} 面")
    
    t_total_end = time.perf_counter()
    _log(f"[voronoi_implicit] 总耗时: {t_total_end - t_total_start:.2f}s")
    
    return mesh
