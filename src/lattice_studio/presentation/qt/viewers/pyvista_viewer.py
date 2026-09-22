"""
基于 PyVista (VTK) 的高质量渲染器
支持 PBR + SSAO + 阴影 + 抗锯齿

"""

from __future__ import annotations

import numpy as np
import trimesh
import warnings
import logging
import sys
import os

# 在导入 VTK/PyVista 之前设置环境变量
os.environ['VTK_SILENCE_GET_VOID_POINTER_WARNINGS'] = '1'
os.environ['VTK_DEBUG_LEAKS'] = '0'
# Keep the normal desktop default while allowing automated/offscreen callers to
# select a render backend before this module is imported.
os.environ.setdefault('PYVISTA_OFF_SCREEN', '0')

# 现在导入 pyvista
import pyvista as pv
from pyvistaqt import QtInteractor
from PyQt5 import QtWidgets, QtCore
from dataclasses import dataclass
from typing import Callable, Optional, Literal

from lattice_studio.engine.implicit.field import SampledImplicitField
from lattice_studio.engine.implicit.field_viewer import FieldPlaneIsoline, FieldPlaneSample

# 抑制 VTK 的 OpenGL 上下文警告（Windows 特有问题）
warnings.filterwarnings('ignore', category=UserWarning, module='vtkmodules')
logging.getLogger('vtkmodules').setLevel(logging.ERROR)

# 更激进的方法：重定向 VTK 的 C++ 错误输出
# 这些错误来自 VTK 的 C++ 层，无法通过 Python logging 抑制
try:
    import vtkmodules.vtkCommonCore as vtkcc
    # 创建一个输出窗口来捕获 VTK 错误
    vtk_output_window = vtkcc.vtkOutputWindow()
    vtk_output_window.SetInstance(vtk_output_window)
    # 禁用所有 VTK 输出
    vtk_output_window.GlobalWarningDisplayOff()
except:
    pass

# 禁用 PyVista 的详细输出
try:
    pv.set_error_output_file(os.devnull if os.name != 'nt' else 'nul')
except:
    pass


RenderMode = Literal["solid", "wireframe"]
MouseMode = Literal["rotate", "pan"]
RenderQuality = Literal["low", "medium", "high", "ultra"]
DEFAULT_RENDER_QUALITY: RenderQuality = "high"


@dataclass(frozen=True)
class RenderQualitySettings:
    """Display-only quality settings for GPU volume and surface rendering."""

    sample_distance_factor: float
    interpolation: Literal["nearest", "linear"]
    anti_aliasing: Literal["none", "fxaa", "ssaa"]
    use_jittering: bool
    global_illumination_reach: float
    volumetric_scattering_blending: float
    use_ssao: bool
    use_shadows: bool


RENDER_QUALITY_SETTINGS: dict[RenderQuality, RenderQualitySettings] = {
    "low": RenderQualitySettings(
        1.25, "nearest", "none", False, 0.0, 0.0, False, False
    ),
    "medium": RenderQualitySettings(
        0.65, "linear", "fxaa", False, 0.06, 0.04, False, False
    ),
    "high": RenderQualitySettings(
        0.42, "linear", "fxaa", False, 0.16, 0.08, False, False
    ),
    # VTK's SSAA render pass omits embedded vtkVolume props.
    "ultra": RenderQualitySettings(
        0.24, "linear", "fxaa", False, 0.28, 0.14, True, True
    ),
}

MESH_DISPLAY_FACE_LIMITS: dict[RenderQuality, int] = {
    "low": 100_000,
    "medium": 250_000,
    "high": 750_000,
    "ultra": 1_500_000,
}

MESH_DISPLAY_MINIMUM_RATIOS: dict[RenderQuality, float] = {
    "low": 0.40,
    "medium": 0.70,
    "high": 0.85,
    "ultra": 1.00,
}

MAX_RESIDENT_MESH_LAYERS = 8
MAX_RESIDENT_IMPLICIT_LAYERS = 8
DEFAULT_IMPLICIT_CACHE_BUDGET_BYTES = 512 * 1024 * 1024


def mesh_display_face_limit(source_faces: int, quality: RenderQuality) -> int:
    """Select a face budget without erasing high-frequency lattice detail."""

    count = int(source_faces)
    if count < 0:
        raise ValueError("source_faces must be non-negative")
    if quality not in MESH_DISPLAY_FACE_LIMITS:
        raise ValueError(f"unknown render quality: {quality}")
    proportional = int(np.ceil(count * MESH_DISPLAY_MINIMUM_RATIOS[quality]))
    return min(count, max(MESH_DISPLAY_FACE_LIMITS[quality], proportional))


@dataclass(frozen=True)
class RenderBackendStatus:
    """User-facing status for the active implicit rendering path."""

    active_backend: str
    using_gpu: bool
    device_name: str
    fallback_reason: str | None = None

    @classmethod
    def from_capabilities(
        cls,
        capability_report: str,
        *,
        supports_opengl: bool,
        mapper_supported: bool,
    ) -> "RenderBackendStatus":
        report = str(capability_report or "")
        renderer_name = "未知图形设备"
        for line in report.splitlines():
            key, separator, value = line.partition(":")
            if separator and key.strip().lower() == "opengl renderer string":
                renderer_name = value.strip() or renderer_name
                break

        software_tokens = (
            "llvmpipe",
            "softpipe",
            "software rasterizer",
            "gdi generic",
            "mesa offscreen",
            "swiftshader",
        )
        is_software = any(
            token in renderer_name.lower() for token in software_tokens
        )
        if not supports_opengl:
            return cls(
                "OpenGL 不可用",
                False,
                renderer_name,
                "当前图形环境不支持所需的 OpenGL 功能",
            )
        if not mapper_supported:
            return cls(
                "OpenGL 兼容渲染",
                False,
                renderer_name,
                "GPU 隐式体光线投射不受当前设备支持",
            )
        if is_software:
            return cls(
                "软件 OpenGL 光线投射",
                False,
                renderer_name,
                "检测到软件光栅器，未使用独立 GPU",
            )
        return cls("OpenGL GPU 光线投射", True, renderer_name)


@dataclass(frozen=True)
class ImplicitRenderCacheStatistics:
    """Observable state of the resident implicit-render resource cache."""

    cache_hits: int
    cache_misses: int
    resident_layers: int
    resident_bytes: int
    budget_bytes: int


def build_studio_lights(center: np.ndarray, scene_size: float) -> list[pv.Light]:
    """Build a restrained CAD studio rig around the model bounds.

    One dominant key light establishes the surface shape.  The remaining
    lights are intentionally weak: strong opposing fills create visible
    Lambertian terminator bands on GPU-rendered implicit surfaces.
    """

    target = np.asarray(center, dtype=np.float64).reshape(3)
    size = max(float(scene_size), 1e-3)
    light_specs = (
        ((1.35, 1.15, 2.10), (1.00, 0.985, 0.970), 1.00),
        ((-1.45, 0.55, 1.05), (0.96, 0.98, 1.00), 0.08),
        ((0.30, -1.80, 1.45), (0.97, 0.985, 1.00), 0.12),
        ((-0.35, -0.20, -0.85), (0.97, 0.985, 1.00), 0.02),
    )
    lights: list[pv.Light] = []
    for offset, color, intensity in light_specs:
        position = target + size * np.asarray(offset, dtype=np.float64)
        lights.append(
            pv.Light(
                position=position,
                focal_point=target,
                color=color,
                intensity=intensity,
                positional=False,
            )
        )
    return lights


def build_studio_environment_texture(
    width: int = 256,
    height: int = 128,
) -> pv.Texture:
    """Create a compact equirectangular studio map for mesh PBR shading.

    The texture is used only for image-based lighting and is not shown as the
    viewport background.  A dark neutral room plus two broad softboxes keeps
    matte CAD materials readable without producing several point highlights.
    """

    if width < 16 or height < 8:
        raise ValueError("studio environment dimensions are too small")
    yy, xx = np.mgrid[0:height, 0:width]
    pixels = np.full((height, width, 3), 68.0, dtype=np.float64)
    key = np.exp(
        -(
            ((xx - 0.25 * width) / (0.125 * width)) ** 2
            + ((yy - 0.25 * height) / (0.14 * height)) ** 2
        )
    )
    fill = np.exp(
        -(
            ((xx - 0.78 * width) / (0.19 * width)) ** 2
            + ((yy - 0.48 * height) / (0.24 * height)) ** 2
        )
    )
    pixels += 170.0 * key[..., None]
    pixels += 40.0 * fill[..., None]
    texture = pv.Texture(np.clip(pixels, 0.0, 255.0).astype(np.uint8))
    texture.interpolate = True
    texture.mipmap = True
    texture.repeat = True
    return texture


def _pbr_style_volume_coefficients(
    metallic: float,
    roughness: float,
) -> tuple[float, float, float, float]:
    """Map metallic/roughness controls to VTK's lit isosurface model.

    VTK's GPU volume isosurface mapper does not expose the polygon actor's
    PBR interpolation mode.  These coefficients preserve the same material
    controls while the surface remains an implicit GPU ray-cast volume.
    """

    metal = float(np.clip(metallic, 0.0, 1.0))
    rough = float(np.clip(roughness, 0.05, 1.0))
    ambient = 0.12 + 0.04 * rough
    diffuse = (1.0 - 0.72 * metal) * (0.84 - 0.18 * rough)
    specular = 0.12 + 0.34 * metal + 0.08 * (1.0 - rough)
    specular_power = float(np.clip(8.0 / (rough**2), 4.0, 64.0))
    return ambient, diffuse, specular, specular_power


@dataclass
class MeshData:
    """网格数据类（保持与原 viewer.py 兼容）"""
    vertices: np.ndarray
    faces: np.ndarray
    color: tuple[float, float, float, float] = (0.7, 0.7, 0.7, 1.0)
    render_mode: RenderMode = "solid"
    unlit: bool = False
    line_width: float = 1.5
    metallic: float = 0.3  # 金属度 (0-1)
    roughness: float = 0.5  # 粗糙度 (0-1)
    show_edges: bool = False
    edge_color: tuple[float, float, float] | str = "#4B5563"
    cache_key: object | None = None
    
    def __post_init__(self):
        self.vertices = np.asarray(self.vertices)
        self.faces = np.asarray(self.faces)
        if self.vertices.ndim != 2 or self.vertices.shape[1] != 3:
            raise ValueError("vertices must have shape (N, 3)")
        if self.faces.ndim != 2 or self.faces.shape[1] != 3:
            raise ValueError("faces must have shape (M, 3)")


@dataclass
class ContourData:
    """Display-only 3-D polylines extracted from one implicit layer."""

    polylines: list[np.ndarray]
    color: tuple[float, float, float] = (0.10, 0.35, 0.85)
    line_width: float = 2.0
    cache_key: object | None = None


@dataclass(frozen=True)
class FieldPlaneRenderData:
    """Display settings for one persistent field-viewer plane overlay."""

    sample: FieldPlaneSample
    colormap: Literal["implicit", "turbo", "distance"]
    value_range: tuple[float, float]
    opacity: float
    isolines: tuple[FieldPlaneIsoline, ...] = ()
    show_probe: bool = True
    clip_to_source_bounds: bool = True
    cache_key: object | None = None

    def __post_init__(self) -> None:
        lower, upper = (float(value) for value in self.value_range)
        opacity = float(self.opacity)
        if self.colormap not in {"implicit", "turbo", "distance"}:
            raise ValueError("unknown field-plane colormap")
        if not np.isfinite((lower, upper)).all() or not lower < upper:
            raise ValueError("field-plane value range must be finite and ordered")
        if not np.isfinite(opacity) or not 0.0 <= opacity <= 1.0:
            raise ValueError("field-plane opacity must be between 0 and 1")
        object.__setattr__(self, "value_range", (lower, upper))
        object.__setattr__(self, "opacity", opacity)


def build_contour_polydata(contour: ContourData) -> pv.PolyData:
    """Convert display polylines to VTK line cells without surface meshing."""

    polylines: list[np.ndarray] = []
    for raw in contour.polylines:
        points = np.asarray(raw, dtype=np.float32)
        if points.ndim == 2 and points.shape[1] == 3 and len(points) >= 2:
            polylines.append(points)
    if not polylines:
        return pv.PolyData()
    points = np.concatenate(polylines, axis=0)
    cells = []
    offset = 0
    for polyline in polylines:
        count = len(polyline)
        cells.append(
            np.concatenate(
                (
                    np.array([count], dtype=np.int32),
                    np.arange(offset, offset + count, dtype=np.int32),
                )
            )
        )
        offset += count
    return pv.PolyData(points, lines=np.concatenate(cells))


def build_field_plane_polydata(data: FieldPlaneRenderData) -> pv.DataSet:
    """Build the visible finite portion of a sampled Field Viewer plane.

    An ``ImplicitBody`` is defined only inside its finite bounds.  Clipping
    the inspection plane there prevents synthetic outside-positive values from
    being presented as engineering field values.
    """

    sample = data.sample
    shape = sample.values.shape
    grid = pv.StructuredGrid()
    grid.dimensions = (shape[0], shape[1], 1)
    grid.points = np.ascontiguousarray(
        sample.points.reshape(-1, 3, order="F"), dtype=np.float64
    )
    grid["field_value"] = np.ascontiguousarray(
        sample.values.ravel(order="F"), dtype=np.float32
    )
    bounds = sample.source_bounds_mm
    clip_bounds = (
        float(bounds[0, 0]),
        float(bounds[1, 0]),
        float(bounds[0, 1]),
        float(bounds[1, 1]),
        float(bounds[0, 2]),
        float(bounds[1, 2]),
    )
    if data.clip_to_source_bounds:
        return grid.clip_box(clip_bounds, invert=False, crinkle=False)

    # Preserve the user-selected plane size while clearly marking locations
    # outside the source field's finite definition as unavailable.
    valid = np.asarray(sample.valid_mask, dtype=bool).ravel(order="F")
    full_values = np.asarray(grid["field_value"], dtype=np.float32).copy()
    full_values[~valid] = np.nan
    grid["field_value"] = full_values
    return grid


@dataclass
class _MeshLayerRecord:
    """Persistent VTK resources for one display mesh layer."""

    key: object
    face_limit: int
    mesh_data: MeshData
    polydata: pv.PolyData
    actor: object


@dataclass
class _ImplicitLayerRecord:
    """Persistent VTK resources for one sampled implicit field."""

    key: object
    field: ImplicitData
    image: pv.ImageData
    actor: object
    attached: bool = False


def build_mesh_display_polydata(
    vertices: np.ndarray,
    faces: np.ndarray,
    face_limit: int,
) -> pv.PolyData:
    """Build a disposable VTK display proxy without changing source geometry."""

    points = np.ascontiguousarray(vertices, dtype=np.float32)
    triangles = np.asarray(faces)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("vertices must have shape (N, 3)")
    if triangles.ndim != 2 or triangles.shape[1] != 3:
        raise ValueError("faces must have shape (M, 3)")
    if face_limit < 4:
        raise ValueError("face_limit must be at least 4")
    vtk_faces = np.empty((len(triangles), 4), dtype=np.int64)
    vtk_faces[:, 0] = 3
    vtk_faces[:, 1:] = triangles
    source = pv.PolyData(points, vtk_faces.ravel())
    if source.n_cells <= face_limit:
        return source

    from vtkmodules.vtkFiltersCore import vtkDecimatePro

    decimator = vtkDecimatePro()
    decimator.SetInputData(source)
    decimator.SetTargetReduction(1.0 - face_limit / source.n_cells)
    decimator.PreserveTopologyOn()
    decimator.SplittingOff()
    decimator.BoundaryVertexDeletionOff()
    decimator.SetFeatureAngle(30.0)
    decimator.Update()
    proxy = pv.wrap(decimator.GetOutput()).copy(deep=True)
    if proxy.n_cells == 0:
        return source
    return proxy


ImplicitData = SampledImplicitField


def build_implicit_volume_actor(
    field: ImplicitData,
    quality: RenderQuality = "medium",
):
    """Build a VTK GPU volume actor configured for the zero isosurface."""

    from vtkmodules.vtkRenderingCore import vtkVolume, vtkVolumeProperty
    from vtkmodules.vtkRenderingVolumeOpenGL2 import vtkOpenGLGPUVolumeRayCastMapper

    values = np.ascontiguousarray(field.values, dtype=np.float32)
    image = pv.ImageData(
        dimensions=values.shape,
        spacing=tuple(float(value) for value in field.spacing),
        origin=tuple(float(value) for value in field.origin),
    )
    image.point_data["implicit"] = values.ravel(order="F")

    mapper = vtkOpenGLGPUVolumeRayCastMapper()
    mapper.SetInputData(image)
    mapper.SetBlendModeToIsoSurface()
    prop = vtkVolumeProperty()

    volume = vtkVolume()
    volume.SetMapper(mapper)
    volume.SetProperty(prop)
    _configure_implicit_volume_actor(field, volume, quality)
    return image, volume


def _configure_implicit_volume_actor(
    field: ImplicitData,
    volume,
    quality: RenderQuality,
) -> None:
    """Update display-only quality and material without replacing GPU data."""

    if quality not in RENDER_QUALITY_SETTINGS:
        raise ValueError(f"unknown render quality: {quality}")
    settings = RENDER_QUALITY_SETTINGS[quality]
    from vtkmodules.vtkCommonDataModel import vtkPiecewiseFunction
    from vtkmodules.vtkRenderingCore import vtkColorTransferFunction

    mapper = volume.GetMapper()
    mapper.SetAutoAdjustSampleDistances(quality != "ultra")
    mapper.SetLockSampleDistanceToInputSpacing(False)
    mapper.SetSampleDistance(
        float(np.min(field.spacing)) * settings.sample_distance_factor
    )
    mapper.SetUseJittering(settings.use_jittering)
    mapper.SetGlobalIlluminationReach(settings.global_illumination_reach)
    mapper.SetVolumetricScatteringBlending(
        settings.volumetric_scattering_blending
    )
    mapper.SetComputeNormalFromOpacity(False)

    prop = volume.GetProperty()
    if settings.interpolation == "nearest":
        prop.SetInterpolationTypeToNearest()
    else:
        prop.SetInterpolationTypeToLinear()
    ambient, diffuse, specular, specular_power = _pbr_style_volume_coefficients(
        field.metallic,
        field.roughness,
    )
    prop.SetAmbient(ambient)
    prop.SetShade(True)
    prop.SetDiffuse(diffuse)
    prop.SetSpecular(specular)
    prop.SetSpecularPower(specular_power)
    prop.SetScatteringAnisotropy(0.0)
    prop.GetIsoSurfaceValues().SetValue(0, 0.0)

    values = field.values
    color = tuple(float(np.clip(value, 0.0, 1.0)) for value in field.color[:3])
    opacity = float(np.clip(field.color[3], 0.0, 1.0))
    scalar_range = max(float(np.max(np.abs(values))), 1e-6)
    scalar_opacity = vtkPiecewiseFunction()
    scalar_opacity.AddPoint(-scalar_range, 0.0)
    scalar_opacity.AddPoint(0.0, opacity)
    scalar_opacity.AddPoint(scalar_range, 0.0)
    prop.SetScalarOpacity(scalar_opacity)
    color_transfer = vtkColorTransferFunction()
    color_transfer.AddRGBPoint(-scalar_range, *color)
    color_transfer.AddRGBPoint(scalar_range, *color)
    prop.SetColor(color_transfer)


class PyVistaRenderer(QtInteractor):
    """
    基于 PyVista 的高质量渲染器
    
    特性：
    - PBR 材质
    - SSAO (环境光遮蔽)
    - 实时阴影
    - FXAA 抗锯齿
    - 多光源
    
    交互：
    - 左键：选择对象或操纵器
    - 右键拖动：旋转视角
    - 中键拖动：平移视角
    - 滚轮：缩放
    """
    
    # 定义信号（保持与原 GLMeshViewer 兼容）
    lasso_completed = QtCore.pyqtSignal(list)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # 首次初始化时显示提示（仅一次）
        if not hasattr(PyVistaRenderer, '_init_message_shown'):
            PyVistaRenderer._init_message_shown = True
            print("=" * 60)
            print("PyVista 渲染器已启动")
            print("- 如果看到 VTK OpenGL 错误（wglMakeCurrent），可以忽略")
            print("- 这是 Windows 上 VTK 的已知问题，不影响功能")
            print("=" * 60)
        
        # 状态变量
        self.meshes: list[MeshData] = []
        # Display-only geometry such as analytic-field object previews.  It
        # is independent from generated meshes and never enters export paths.
        self.scene_overlays: list[MeshData] = []
        self.implicit_fields: list[ImplicitData] = []
        self.contours: list[ContourData] = []
        self.render_quality: RenderQuality = DEFAULT_RENDER_QUALITY
        self.mouse_mode: MouseMode = "rotate"
        self._bbox_center = np.zeros(3, dtype=np.float32)
        self._bbox_size = 1.0
        self._bbox_dimensions = np.ones(3, dtype=np.float32)  # 添加三维尺寸存储
        
        # 渲染选项
        self.show_axes = True
        self.show_ground_grid = False
        self.show_ground_plane = False
        self.use_smooth_shading = True
        self.enable_pbr = True  # 默认启用PBR
        self.enable_ssao_flag = True
        self.enable_shadows_flag = True
        self.enable_antialiasing = True
        self._studio_environment_texture = build_studio_environment_texture()
        
        # 平台相关
        self._ground_plane_actor = None
        self._ground_grid_actor = None
        
        # 平面可视化
        self.show_plane = False
        self.plane_point = None
        self.plane_normal = None
        self.plane_size = 100.0
        self.plane_actor = None

        # The Field Viewer is an independent 2-D overlay.  It never changes
        # the authoritative field or replaces the model scene.
        self._field_plane_data: FieldPlaneRenderData | None = None
        self._field_plane_actor = None
        self._field_plane_outline_actor = None
        self._field_plane_isoline_actor = None
        self._field_plane_probe_actor = None
        self._field_plane_probe_text = None
        self._field_plane_probe_value: float | None = None
        self._field_plane_highlight_value: float | None = None
        self._field_plane_probe_callback: Callable[[float | None], None] | None = None
        self._field_plane_probe_observer_id: int | None = None
        self._field_plane_probe_pending: (
            tuple[np.ndarray, float, tuple[int, int]] | None
        ) = None
        self._field_plane_probe_refresh_timer = QtCore.QTimer(self)
        self._field_plane_probe_refresh_timer.setSingleShot(True)
        self._field_plane_probe_refresh_timer.setInterval(33)
        self._field_plane_probe_refresh_timer.timeout.connect(
            self._flush_field_plane_probe_update
        )

        # Optional section-box clipping used to inspect the interior of a
        # displayed mesh.  The source trimesh data is never modified.
        self.section_enabled = False
        self.section_bounds = None
        self.section_planes = None
        self.section_plane = None
        
        # PyVista 内部对象
        self._pv_meshes = []  # 存储 PyVista mesh 对象
        self._pv_actors = []  # 存储 actor 对象
        self._implicit_actors = []
        self._implicit_images = []
        self._contour_actors = []
        self._implicit_layer_cache: dict[object, _ImplicitLayerRecord] = {}
        self._implicit_cache_budget_bytes = DEFAULT_IMPLICIT_CACHE_BUDGET_BYTES
        self._implicit_cache_hits = 0
        self._implicit_cache_misses = 0
        self._mesh_display_cache: dict[tuple[object, int], pv.PolyData] = {}
        self._mesh_layer_records: dict[object, _MeshLayerRecord] = {}
        self._scene_mode: Literal[
            "empty", "mesh", "implicit", "contours", "cell_map", "field_plane"
        ] = "empty"
        self._cell_map_preview = None
        self._cell_map_actor = None
        self._cell_map_frame_actors = []
        self._scene_bounds = np.array(
            [[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]], dtype=np.float64
        )
        self._left_navigation_style = None
        self._left_navigation_suspended = False
        
        # 初始化渲染器
        self._setup_renderer()
        self._configure_navigation_controls()
        self._configure_field_plane_probe()
        
        # 调整鼠标滚轮缩放灵敏度（降低缩放速度）
        self._adjust_zoom_sensitivity()
        
        # 添加相机交互回调，在相机移动后修复裁剪面
        self._setup_camera_callback()
    
    def _setup_renderer(self):
        """设置渲染器参数"""
        # Keep a bright workspace while retaining enough tonal separation for
        # light CAD materials.  The environment map lights PBR meshes only.
        self.set_background("#F1F3F6", top="#FFFFFF")
        try:
            self.set_environment_texture(
                self._studio_environment_texture,
                is_srgb=True,
                show_background=False,
            )
        except (AttributeError, RuntimeError, TypeError) as exc:
            print(f"[警告] 工作室环境光初始化失败: {exc}")
        
        self._configure_anti_aliasing()
        
        # 启用深度剥离（提高透明度渲染质量）
        try:
            self.enable_depth_peeling()
        except:
            pass

    def _configure_navigation_controls(self) -> None:
        """Configure camera navigation without assigning any action to left drag.

        ``enable_custom_trackball_style`` currently does not accept ``None``
        for an action in all supported PyVista versions, despite documenting
        it as a valid value.  Configure the style with a compatibility
        placeholder, then remove its left-button callbacks.  VTK also has a
        built-in ``ProcessEvents`` path that bypasses those callbacks, so the
        high-priority interactor observers detach the style for the complete
        left-button press interval.
        """

        self.enable_custom_trackball_style(
            left="rotate",
            shift_left="rotate",
            control_left="rotate",
            middle="pan",
            shift_middle="pan",
            control_middle="pan",
            right="rotate",
            shift_right="rotate",
            control_right="rotate",
        )
        style = self.interactor.GetInteractorStyle()
        if style is not None:
            style.RemoveObservers("LeftButtonPressEvent")
            style.RemoveObservers("LeftButtonReleaseEvent")
        self.interactor.AddObserver(
            "LeftButtonPressEvent",
            self._suspend_left_camera_navigation,
            2.0,
        )
        self.interactor.AddObserver(
            "LeftButtonReleaseEvent",
            self._restore_left_camera_navigation,
            2.0,
        )
        # Replace VTK's distance-based dolly with a parallel-camera zoom.  The
        # latter remains usable when the cursor is very close to the model.
        if style is not None:
            style.RemoveObservers("MouseWheelForwardEvent")
            style.RemoveObservers("MouseWheelBackwardEvent")
        self.interactor.AddObserver(
            "MouseWheelForwardEvent",
            lambda caller, event: PyVistaRenderer._zoom_in(self, caller, event),
            2.0,
        )
        self.interactor.AddObserver(
            "MouseWheelBackwardEvent",
            lambda caller, event: PyVistaRenderer._zoom_out(self, caller, event),
            2.0,
        )
    def _suspend_left_camera_navigation(self, _caller, _event) -> None:
        """Detach VTK's camera style for the duration of a left-button drag."""

        if self._left_navigation_suspended:
            return
        style = self.interactor.GetInteractorStyle()
        if style is None:
            return
        self._left_navigation_style = style
        self._left_navigation_suspended = True
        self.interactor.SetInteractorStyle(None)

    def _restore_left_camera_navigation(self, _caller, _event) -> None:
        """Restore camera navigation after left-button selection is released."""

        style = self._left_navigation_style
        self._left_navigation_style = None
        self._left_navigation_suspended = False
        if style is not None:
            self.interactor.SetInteractorStyle(style)
    
    def _adjust_zoom_sensitivity(self):
        """调整鼠标滚轮缩放灵敏度"""
        try:
            # 获取 VTK 交互器样式
            interactor_style = self.interactor.GetInteractorStyle()
            
            # 设置更小的缩放因子（默认是 1.1，改为 1.05 让缩放更平滑）
            if hasattr(interactor_style, 'SetMouseWheelMotionFactor'):
                interactor_style.SetMouseWheelMotionFactor(0.5)  # 降低滚轮灵敏度
            
            # 另一种方法：直接设置缩放因子
            if hasattr(self.interactor, 'SetDollyFactor'):
                self.interactor.SetDollyFactor(1.05)  # 从默认的 1.1 降到 1.05
                
        except Exception as e:
            print(f"[提示] 无法调整缩放灵敏度: {e}")

    def _zoom_in(self, _caller, _event) -> None:
        self._apply_wheel_zoom(1.0)

    def _zoom_out(self, _caller, _event) -> None:
        self._apply_wheel_zoom(-1.0)

    def _apply_wheel_zoom(self, direction: float) -> None:
        """Zoom with a CAD-style parallel scale instead of dollying to zero."""

        camera = self.camera
        camera.parallel_projection = True
        factor = 1.18 if direction > 0 else 1.0 / 1.18
        minimum = max(self._bbox_size * 1.0e-7, 1.0e-9)
        maximum = max(self._bbox_size * 100.0, minimum * 10.0)
        camera.parallel_scale = float(
            np.clip(camera.parallel_scale / factor, minimum, maximum)
        )
        self._fix_camera_clipping()
        self.render()
    
    def _setup_camera_callback(self):
        """设置相机交互回调，在相机移动后自动修复裁剪面"""
        try:
            # 添加相机修改事件的观察者
            def on_camera_modified(obj, event):
                """相机修改时的回调"""
                try:
                    self._fix_camera_clipping()
                except:
                    pass
            
            # 监听相机的修改事件
            self.camera.AddObserver('ModifiedEvent', on_camera_modified)
            
        except Exception as e:
            print(f"[提示] 无法设置相机回调: {e}")
    
    def set_mouse_mode(self, mode: MouseMode):
        """设置鼠标模式"""
        self.mouse_mode = "pan" if mode == "pan" else "rotate"
        # PyVista 的交互模式由内部处理，这里只记录状态
    
    def get_view_state(self) -> dict[str, float]:
        """获取当前视图状态"""
        camera = self.camera
        return {
            "position": camera.position,
            "focal_point": camera.focal_point,
            "view_up": camera.up,
            "distance": camera.distance,
        }
    
    def apply_view_state(self, state: dict):
        """应用视图状态"""
        try:
            camera = self.camera
            if "position" in state:
                camera.position = state["position"]
            if "focal_point" in state:
                camera.focal_point = state["focal_point"]
            if "view_up" in state:
                camera.up = state["view_up"]
            self.render()
        except:
            pass
    
    def reset_view(self):
        """Fit mesh and implicit bounds in a neutral CAD-style camera."""

        if not self._has_scene_content():
            self.reset_camera()
            self._fix_camera_clipping()
            return
        self._set_camera_orientation((1.0, 1.0, 0.8), (0.0, 0.0, 1.0))

    def _has_scene_content(self) -> bool:
        return bool(
            self.meshes
            or getattr(self, "scene_overlays", ())
            or self.implicit_fields
            or self.contours
            or self._cell_map_preview is not None
            or self._field_plane_data is not None
        )

    def _set_camera_orientation(
        self,
        direction: tuple[float, float, float],
        view_up: tuple[float, float, float],
    ) -> None:
        if not self._has_scene_content():
            return
        bounds = np.asarray(self._scene_bounds, dtype=float)
        center = bounds.mean(axis=0)
        extent = bounds[1] - bounds[0]
        radius = max(float(np.linalg.norm(extent)) * 0.5, 1.0e-6)
        direction_array = np.asarray(direction, dtype=float)
        direction_array /= max(float(np.linalg.norm(direction_array)), 1.0e-12)
        distance = max(radius * 4.0, self._bbox_size * 2.0, 1.0)
        camera = self.camera
        camera.focal_point = center
        camera.position = center + direction_array * distance
        camera.up = view_up
        camera.parallel_projection = True
        camera.parallel_scale = max(radius * 2.2, 1.0e-6)
        self._fix_camera_clipping()
        self.render()
    
    def set_view_front(self):
        """Look from -Y toward +Y; X is right and Z is up."""
        self._set_camera_orientation((0.0, -1.0, 0.0), (0.0, 0.0, 1.0))
    
    def set_view_back(self):
        """Look from +Y toward -Y; X is left and Z is up."""
        self._set_camera_orientation((0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
    
    def set_view_left(self):
        """Look from -X toward +X; Y is right and Z is up."""
        self._set_camera_orientation((-1.0, 0.0, 0.0), (0.0, 0.0, 1.0))
    
    def set_view_right(self):
        """Look from +X toward -X; Y is left and Z is up."""
        self._set_camera_orientation((1.0, 0.0, 0.0), (0.0, 0.0, 1.0))
    
    def set_view_top(self):
        """Look from +Z toward -Z; Y is up and X is right."""
        self._set_camera_orientation((0.0, 0.0, 1.0), (0.0, 1.0, 0.0))
    
    def set_view_bottom(self):
        """Look from -Z toward +Z; Y is up and X is left."""
        self._set_camera_orientation((0.0, 0.0, -1.0), (0.0, 1.0, 0.0))
    
    def set_view_isometric(self):
        """Look from +X,+Y,+Z with world Z as the vertical axis."""
        self._set_camera_orientation((1.0, 1.0, 0.8), (0.0, 0.0, 1.0))
    
    def _fix_camera_clipping(self):
        """修复相机裁剪面，防止在特定距离时模型显示异常"""
        try:
            camera = self.camera
            
            distance = max(float(camera.distance), 0.0)
            scene_size = max(float(self._bbox_size), 1.0e-6)
            near_clip = max(min(distance * 0.1, scene_size * 1.0e-4), 1.0e-9)
            far_clip = max(distance + scene_size * 4.0, near_clip * 10.0)
            
            # 设置裁剪范围
            camera.clipping_range = (near_clip, far_clip)
            
        except Exception as e:
            print(f"[警告] 修复相机裁剪面失败: {e}")
    
    def set_meshes(self, meshes: list[MeshData], reset_view: bool = True):
        """Set visible mesh layers while retaining unchanged GPU resources."""

        self.meshes = list(meshes)
        self.implicit_fields = []
        self.contours = []
        self._cell_map_preview = None
        if (
            self._scene_mode == "mesh"
            and not self.scene_overlays
            and not self.show_ground_plane
            and not self.show_ground_grid
        ):
            self._sync_mesh_layers()
        else:
            self._update_scene()
        
        if reset_view:
            self.reset_view()
        else:
            self._fix_camera_clipping()
            self.render()

    def set_implicit_fields(
        self,
        fields: list[ImplicitData],
        reset_view: bool = True,
    ) -> None:
        """Render sampled implicit fields while retaining hidden GPU layers."""

        self.implicit_fields = list(fields)
        self.meshes = []
        self.contours = []
        self._cell_map_preview = None
        if (
            self._scene_mode == "implicit"
            and not self.scene_overlays
            and not self.show_ground_plane
            and not self.show_ground_grid
        ):
            self._sync_implicit_layers()
        else:
            self._update_scene()
        if reset_view:
            self.reset_view()
        else:
            self._fix_camera_clipping()
            self.render()

    def set_scene_geometry(
        self,
        meshes: list[MeshData],
        fields: list[ImplicitData],
        reset_view: bool = True,
    ) -> None:
        """Render mesh and implicit layers together without changing either source."""

        self.meshes = list(meshes)
        self.implicit_fields = list(fields)
        self.contours = []
        self._cell_map_preview = None
        self._update_scene()
        if reset_view:
            self.reset_view()
        else:
            self._fix_camera_clipping()
            self.render()

    def set_scene_overlay_meshes(
        self,
        meshes: list[MeshData],
        *,
        rebuild_scene: bool = True,
    ) -> None:
        """Set display-only meshes, optionally batching their scene rebuild."""

        self.scene_overlays = list(meshes)
        if not rebuild_scene:
            return
        # Cached sampled implicit layers are reattached rather than resampled.
        self._update_scene()
        self._fix_camera_clipping()
        self.render()

    def set_implicit_cache_budget_mb(self, budget_mb: float) -> None:
        """Set the resident implicit-render cache budget and evict old layers."""

        value = float(budget_mb)
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError("implicit cache budget must be finite and positive")
        self._implicit_cache_budget_bytes = int(value * 1024 * 1024)
        active_keys = {
            self._implicit_layer_key(field) for field in self.implicit_fields
        }
        self._prune_inactive_implicit_layers(active_keys)

    def get_implicit_cache_statistics(self) -> ImplicitRenderCacheStatistics:
        """Return counters and resident memory estimates for diagnostics."""

        return ImplicitRenderCacheStatistics(
            cache_hits=self._implicit_cache_hits,
            cache_misses=self._implicit_cache_misses,
            resident_layers=len(self._implicit_layer_cache),
            resident_bytes=sum(
                record.field.estimated_bytes
                for record in self._implicit_layer_cache.values()
            ),
            budget_bytes=self._implicit_cache_budget_bytes,
        )

    def discard_inactive_implicit_layers(self) -> None:
        """Release obsolete sampled fields after an authoritative geometry edit."""

        active_keys = {
            self._implicit_layer_key(field) for field in self.implicit_fields
        }
        for key in list(self._implicit_layer_cache):
            if key in active_keys:
                continue
            record = self._implicit_layer_cache.pop(key)
            self._release_implicit_layer(record)

    def _release_implicit_layer(self, record: _ImplicitLayerRecord) -> None:
        """Detach one volume and release its OpenGL allocations deterministically."""

        if record.attached:
            self.remove_actor(
                record.actor,
                reset_camera=False,
                render=False,
            )
            record.attached = False
        render_window = getattr(self, "ren_win", None)
        if render_window is not None:
            try:
                record.actor.ReleaseGraphicsResources(render_window)
            except (AttributeError, RuntimeError):
                pass
            mapper = record.actor.GetMapper()
            if mapper is not None:
                try:
                    mapper.ReleaseGraphicsResources(render_window)
                except (AttributeError, RuntimeError):
                    pass

    def set_render_quality(self, quality: RenderQuality) -> None:
        """Change display quality without resampling implicit fields."""

        if quality not in RENDER_QUALITY_SETTINGS:
            raise ValueError(f"unknown render quality: {quality}")
        if quality == self.render_quality:
            return
        self.render_quality = quality
        self._mesh_display_cache.clear()
        self._configure_anti_aliasing()
        self._update_scene()
        self._fix_camera_clipping()
        self.render()

    def set_cell_map_preview(
        self,
        cell_map,
        domain_mesh: MeshData | None = None,
        reset_view: bool = True,
    ) -> None:
        """Show a standalone Cell Map grid with optional design-domain context."""

        if not hasattr(cell_map, "bounds") or not hasattr(cell_map, "axes"):
            raise ValueError("cell_map must expose bounds and axes()")
        self._cell_map_preview = cell_map
        self.meshes = [domain_mesh] if domain_mesh is not None else []
        self.implicit_fields = []
        self.contours = []
        self._update_scene()
        if reset_view:
            self.reset_view()
        else:
            self._fix_camera_clipping()
            self.render()

    def clear_cell_map_preview(self) -> None:
        """Remove the standalone Cell Map layer."""

        if self._cell_map_preview is None:
            return
        self._cell_map_preview = None
        self._update_scene()
        self.render()
    
    def _update_scene(self):
        """更新场景（重新构建所有网格）"""
        # PyVista removes props from the renderer; cached VTK resources remain
        # owned by this viewer and can be attached again without re-uploading.
        for record in self._implicit_layer_cache.values():
            record.attached = False
        self.clear()
        self._pv_meshes.clear()
        self._pv_actors.clear()
        self._implicit_actors.clear()
        self._implicit_images.clear()
        self._contour_actors.clear()
        self._mesh_layer_records.clear()
        self._cell_map_actor = None
        self._cell_map_frame_actors.clear()
        self._field_plane_actor = None
        self._field_plane_outline_actor = None
        self._field_plane_isoline_actor = None
        self._field_plane_probe_actor = None
        self._field_plane_probe_text = None
        self._field_plane_probe_value = None
        self._field_plane_highlight_value = None
        
        if (
            not self.meshes
            and not self.scene_overlays
            and not self.implicit_fields
            and not self.contours
            and self._cell_map_preview is None
            and self._field_plane_data is None
        ):
            self._scene_mode = "empty"
            return

        if self._cell_map_preview is not None:
            self._scene_mode = "cell_map"
        elif self.implicit_fields:
            self._scene_mode = "implicit"
        elif self.contours:
            self._scene_mode = "contours"
        elif self._field_plane_data is not None:
            self._scene_mode = "field_plane"
        else:
            self._scene_mode = "mesh"
        
        # 计算包围盒
        self._update_bbox()
        
        # 添加网格平台（在模型之前，这样模型在上面）
        # A layer-contour inspection is a single diagnostic plane.  Ground
        # helpers belong to the normal modeling scene and can otherwise be
        # mistaken for geometry extending away from that plane.
        if self._scene_mode != "contours" and (
            self.show_ground_plane or self.show_ground_grid
        ):
            self._add_ground_platform()
        
        # 添加所有网格
        for mesh_data in self.meshes:
            self._add_mesh_to_scene(mesh_data)

        for field in self.implicit_fields:
            self._add_implicit_field_to_scene(field)

        # Analytic preview meshes are attached after implicit volumes so their
        # opaque geometry participates in the final depth test.  This keeps
        # the part outside a design domain visible while the embedded part is
        # correctly occluded by the domain surface.
        for mesh_data in self.scene_overlays:
            self._add_mesh_to_scene(mesh_data)

        for contour in self.contours:
            self._add_contour_to_scene(contour)

        if self.implicit_fields:
            self._prune_inactive_implicit_layers(
                {self._implicit_layer_key(field) for field in self.implicit_fields}
            )

        if self._cell_map_preview is not None:
            self._add_cell_map_preview_to_scene(self._cell_map_preview)

        if self._field_plane_data is not None:
            self._add_field_plane_to_scene(self._field_plane_data)
        
        # 添加坐标轴
        if self.show_axes:
            self.add_axes(
                line_width=2,
                color="#263238",
                x_color="#D33F49",
                y_color="#1B9E62",
                z_color="#2563EB",
                xlabel="X",
                ylabel="Y",
                zlabel="Z",
            )
        
        # 设置光照
        self._setup_lighting()
        
        # 启用高级特性
        self._enable_advanced_features()
        
        # 添加模型信息显示（左上角）
        self._add_model_info_overlay()
        
        # 重新渲染平面（如果需要）
        if self.show_plane and self.plane_point is not None:
            self._update_plane()
    
    @staticmethod
    def _mesh_layer_key(mesh_data: MeshData) -> object:
        """Return the stable identity used for resident geometry resources."""

        key = mesh_data.cache_key
        if key is None:
            key = (
                id(mesh_data.vertices),
                id(mesh_data.faces),
                mesh_data.vertices.shape,
                mesh_data.faces.shape,
            )
        try:
            hash(key)
        except TypeError as exc:
            raise ValueError("MeshData.cache_key must be hashable") from exc
        return key

    @staticmethod
    def _implicit_layer_key(field: ImplicitData) -> object:
        """Return the immutable sampled-field identity used by the cache."""

        key = getattr(field, "cache_key", None)
        if key is None:
            key = (
                id(field.values),
                field.values.shape,
                tuple(float(value) for value in field.origin),
                tuple(float(value) for value in field.spacing),
            )
        try:
            hash(key)
        except TypeError as exc:
            raise ValueError("SampledImplicitField.cache_key must be hashable") from exc
        return key

    def _acquire_implicit_layer(
        self,
        field: ImplicitData,
    ) -> _ImplicitLayerRecord:
        """Return a recent resident layer or create its VTK resources once."""

        key = self._implicit_layer_key(field)
        record = self._implicit_layer_cache.pop(key, None)
        if record is None:
            image, actor = build_implicit_volume_actor(field, self.render_quality)
            record = _ImplicitLayerRecord(key, field, image, actor)
            self._implicit_cache_misses += 1
        else:
            record.field = field
            _configure_implicit_volume_actor(
                field,
                record.actor,
                self.render_quality,
            )
            self._implicit_cache_hits += 1
        self._implicit_layer_cache[key] = record
        return record

    def _show_implicit_layer(self, record: _ImplicitLayerRecord) -> None:
        """Attach and show one resident layer without replacing its mapper."""

        self._apply_implicit_clipping(record.actor.GetMapper())
        if not record.attached:
            self.add_actor(
                record.actor,
                reset_camera=False,
                pickable=False,
                render=False,
                name=f"implicit:{record.field.name}:{id(record.actor)}",
            )
            record.attached = True
        record.actor.SetVisibility(True)
        self._implicit_images.append(record.image)
        self._implicit_actors.append(record.actor)

    def _sync_implicit_layers(self) -> None:
        """Synchronize visibility while keeping sampled GPU resources resident."""

        desired_keys: set[object] = set()
        self._implicit_images.clear()
        self._implicit_actors.clear()
        self._scene_mode = "implicit"
        for field in self.implicit_fields:
            key = self._implicit_layer_key(field)
            if key in desired_keys:
                raise ValueError(f"duplicate implicit layer key: {key!r}")
            desired_keys.add(key)
            record = self._acquire_implicit_layer(field)
            self._show_implicit_layer(record)

        for key, record in self._implicit_layer_cache.items():
            if key not in desired_keys and record.attached:
                record.actor.SetVisibility(False)

        self._prune_inactive_implicit_layers(desired_keys)
        self._update_bbox()
        self._setup_lighting()
        self._enable_advanced_features()
        self._add_model_info_overlay()

    def _prune_inactive_implicit_layers(self, active_keys: set[object]) -> None:
        """Evict least-recent inactive fields until count and byte limits fit."""

        resident_bytes = sum(
            record.field.estimated_bytes
            for record in self._implicit_layer_cache.values()
        )
        inactive_keys = [
            key for key in self._implicit_layer_cache if key not in active_keys
        ]
        while inactive_keys and (
            len(self._implicit_layer_cache) > MAX_RESIDENT_IMPLICIT_LAYERS
            or resident_bytes > self._implicit_cache_budget_bytes
        ):
            key = inactive_keys.pop(0)
            record = self._implicit_layer_cache.pop(key)
            resident_bytes -= record.field.estimated_bytes
            self._release_implicit_layer(record)

    def _sync_mesh_layers(self) -> None:
        """Synchronize visible layers without clearing resident VTK actors."""

        desired_keys: set[object] = set()
        self._pv_meshes.clear()
        self._pv_actors.clear()
        for mesh_data in self.meshes:
            key = self._mesh_layer_key(mesh_data)
            if key in desired_keys:
                raise ValueError(f"duplicate mesh layer key: {key!r}")
            desired_keys.add(key)
            face_limit = mesh_display_face_limit(
                len(mesh_data.faces),
                self.render_quality,
            )
            record = self._mesh_layer_records.get(key)
            if record is None or record.face_limit != face_limit:
                if record is not None:
                    self.remove_actor(record.actor, reset_camera=False, render=False)
                    del self._mesh_layer_records[key]
                self._add_mesh_to_scene(mesh_data)
                continue

            record.mesh_data = mesh_data
            self._apply_mesh_actor_style(record.actor, mesh_data)
            record.actor.SetVisibility(True)
            self._apply_mesh_clipping(record.actor.GetMapper())
            self._pv_meshes.append(record.polydata)
            self._pv_actors.append(record.actor)

        for key, record in self._mesh_layer_records.items():
            if key not in desired_keys:
                record.actor.SetVisibility(False)

        self._prune_inactive_mesh_layers(desired_keys)

        self._update_bbox()
        self._setup_lighting()
        self._enable_advanced_features()
        self._add_model_info_overlay()

    def _prune_inactive_mesh_layers(self, active_keys: set[object]) -> None:
        """Bound resident CPU/GPU resources while retaining recent layers."""

        inactive_keys = [
            key for key in self._mesh_layer_records if key not in active_keys
        ]
        while (
            len(self._mesh_layer_records) > MAX_RESIDENT_MESH_LAYERS
            and inactive_keys
        ):
            key = inactive_keys.pop(0)
            record = self._mesh_layer_records.pop(key)
            self.remove_actor(record.actor, reset_camera=False, render=False)

    def _apply_mesh_actor_style(self, actor, mesh_data: MeshData) -> None:
        """Update material properties without replacing actor or mapper."""

        color = tuple(float(value) for value in mesh_data.color)
        rgb = color[:3]
        if any(value > 1.0 for value in rgb):
            rgb = tuple(value / 255.0 for value in rgb)
        prop = actor.GetProperty()
        prop.SetColor(*tuple(float(np.clip(value, 0.0, 1.0)) for value in rgb))
        prop.SetOpacity(float(np.clip(color[3], 0.0, 1.0)))
        prop.SetLineWidth(float(mesh_data.line_width))
        if mesh_data.render_mode == "wireframe":
            prop.SetRepresentationToWireframe()
            prop.SetLighting(False)
            return

        prop.SetRepresentationToSurface()
        prop.SetLighting(not mesh_data.unlit)
        if self.enable_pbr and not mesh_data.unlit:
            prop.SetInterpolationToPBR()
            prop.SetMetallic(float(np.clip(mesh_data.metallic, 0.0, 1.0)))
            prop.SetRoughness(float(np.clip(mesh_data.roughness, 0.05, 1.0)))
        else:
            prop.SetInterpolationToPhong()

    def _add_mesh_to_scene(self, mesh_data: MeshData):
        """添加单个网格到场景"""
        if len(mesh_data.vertices) == 0 or len(mesh_data.faces) == 0:
            return
        
        # 转换为 PyVista 格式
        try:
            key = self._mesh_layer_key(mesh_data)
            if key in self._mesh_layer_records:
                raise ValueError(f"duplicate mesh layer key: {key!r}")
            face_limit = mesh_display_face_limit(
                len(mesh_data.faces),
                self.render_quality,
            )
            pv_mesh = self._display_polydata(mesh_data, face_limit)
            
            # 提取颜色（转换为 0-1 范围）
            color = mesh_data.color
            if all(c <= 1.0 for c in color[:3]):
                color_rgb = color[:3]
            else:
                color_rgb = tuple(c / 255.0 for c in color[:3])
            
            # 转换为十六进制颜色
            color_hex = '#{:02x}{:02x}{:02x}'.format(
                int(color_rgb[0] * 255),
                int(color_rgb[1] * 255),
                int(color_rgb[2] * 255)
            )
            
            # 添加到场景
            if mesh_data.render_mode == "wireframe":
                actor = self.add_mesh(
                    pv_mesh,
                    color=color_hex,
                    style='wireframe',
                    line_width=mesh_data.line_width,
                    lighting=False
                )
            else:
                # 实体渲染 - 使用合理的PBR参数
                if self.enable_pbr and not mesh_data.unlit:
                    # 使用适中的金属度和粗糙度，启用双面渲染
                    actor = self.add_mesh(
                        pv_mesh,
                        color=color_hex,
                        pbr=True,
                        metallic=float(np.clip(mesh_data.metallic, 0.0, 1.0)),
                        roughness=float(np.clip(mesh_data.roughness, 0.05, 1.0)),
                        smooth_shading=self.use_smooth_shading,
                        show_edges=mesh_data.show_edges,
                        edge_color=mesh_data.edge_color,
                        lighting=True,
                        backface_culling=False  # 启用双面渲染
                    )
                else:
                    # 非PBR渲染，也启用双面渲染
                    actor = self.add_mesh(
                        pv_mesh,
                        color=color_hex,
                        smooth_shading=self.use_smooth_shading,
                        show_edges=mesh_data.show_edges,
                        edge_color=mesh_data.edge_color,
                        lighting=not mesh_data.unlit,
                        backface_culling=False  # 启用双面渲染
                    )
            
            self._apply_mesh_actor_style(actor, mesh_data)
            self._apply_mesh_clipping(actor.GetMapper())
            self._pv_meshes.append(pv_mesh)
            self._pv_actors.append(actor)
            record = _MeshLayerRecord(
                key,
                face_limit,
                mesh_data,
                pv_mesh,
                actor,
            )
            self._mesh_layer_records[key] = record
            
        except Exception as e:
            print(f"[错误] 添加网格失败: {e}")
            import traceback
            traceback.print_exc()

    def _display_polydata(
        self,
        mesh_data: MeshData,
        face_limit: int,
    ) -> pv.PolyData:
        """Return one cached quality-bounded proxy for a source mesh."""

        source_key = PyVistaRenderer._mesh_layer_key(mesh_data)
        key = (source_key, int(face_limit))
        cached = self._mesh_display_cache.get(key)
        if cached is not None:
            return cached
        proxy = build_mesh_display_polydata(
            mesh_data.vertices,
            mesh_data.faces,
            face_limit,
        )
        if len(self._mesh_display_cache) >= 8:
            self._mesh_display_cache.pop(next(iter(self._mesh_display_cache)))
        self._mesh_display_cache[key] = proxy
        return proxy

    def _add_implicit_field_to_scene(self, field: ImplicitData) -> None:
        """Add one sampled implicit field through VTK's GPU iso-surface mapper."""

        try:
            record = self._acquire_implicit_layer(field)
            self._show_implicit_layer(record)
        except Exception as exc:
            print(f"[警告] GPU 隐式体渲染失败，跳过 {field.name}: {exc}")

    def _add_contour_to_scene(self, contour: ContourData) -> None:
        """Add contour polylines without triangulating or decimating them."""

        polydata = build_contour_polydata(contour)
        if polydata.n_cells == 0:
            return
        actor = self.add_mesh(
            polydata,
            color=tuple(float(np.clip(value, 0.0, 1.0)) for value in contour.color),
            line_width=float(max(contour.line_width, 1.0)),
            lighting=False,
            render_lines_as_tubes=True,
            pickable=False,
            name=f"contours:{contour.cache_key!r}",
        )
        self._contour_actors.append(actor)

    def set_contours(
        self,
        contours: list[ContourData],
        reset_view: bool = True,
    ) -> None:
        """Show layer contours as a lightweight, isolated inspection scene.

        Layer contours are a 2-D diagnostic view represented by 3-D
        polylines.  They must not share visible geometry with the normal
        model scene: a stale primitive preview, field plane, or Cell Map can
        otherwise expand the scene bounds and look like the contour is being
        extruded along the camera direction.  The application restores those
        sources when contour inspection is disabled.
        """

        self.contours = list(contours)
        self.meshes = []
        self.implicit_fields = []
        self.scene_overlays = []
        self._cell_map_preview = None
        self._field_plane_data = None
        self._update_scene()
        if reset_view:
            self.reset_view()
        else:
            self._fix_camera_clipping()
            self.render()

    def _configure_field_plane_probe(self) -> None:
        """Install one lightweight hover observer for Field Viewer probes."""

        try:
            self._field_plane_probe_observer_id = self.interactor.AddObserver(
                "MouseMoveEvent",
                self._on_field_plane_mouse_move,
                0.25,
            )
        except (AttributeError, RuntimeError):
            self._field_plane_probe_observer_id = None

    @staticmethod
    def _same_vtk_actor(first, second) -> bool:
        """Compare VTK actors across possible distinct Python wrappers."""

        if first is second:
            return True
        if first is None or second is None:
            return False
        try:
            return first.GetAddressAsString("") == second.GetAddressAsString("")
        except AttributeError:
            return False

    @staticmethod
    def _field_plane_colormap(name: str) -> str:
        """Map documented Field Viewer schemes to stable Matplotlib palettes."""

        palettes = {
            "implicit": "coolwarm",
            "turbo": "turbo",
            "distance": "viridis",
        }
        try:
            return palettes[name]
        except KeyError as exc:
            raise ValueError(f"unknown field-plane colormap: {name}") from exc

    def set_field_plane(self, data: FieldPlaneRenderData) -> None:
        """Show a scalar field as a persistent colored plane overlay."""

        if not isinstance(data, FieldPlaneRenderData):
            raise TypeError("data must be a FieldPlaneRenderData instance")
        self._field_plane_data = data
        self._remove_field_plane_actors()
        self._add_field_plane_to_scene(data)
        # The plane can be larger than the source field's finite bounds.  Keep
        # camera clipping derived from its actual user-selected extent so that
        # expanding the plane never leaves the new area behind the old range.
        self._update_bbox()
        self._fix_camera_clipping()
        self.render()

    def set_field_plane_probe_callback(
        self,
        callback: Callable[[float | None], None] | None,
    ) -> None:
        """Notify the surrounding UI whenever Field Viewer probe value changes."""

        if callback is not None and not callable(callback):
            raise TypeError("field-plane probe callback must be callable")
        self._field_plane_probe_callback = callback

    def clear_field_plane(self) -> None:
        """Remove Field Viewer overlays while preserving the model scene."""

        self._field_plane_data = None
        self._remove_field_plane_actors()
        self.render()

    def _remove_field_plane_actors(self) -> None:
        """Detach all overlay props owned by the Field Viewer."""

        self._field_plane_probe_refresh_timer.stop()
        self._field_plane_probe_pending = None
        for attribute in (
            "_field_plane_actor",
            "_field_plane_outline_actor",
            "_field_plane_isoline_actor",
            "_field_plane_probe_actor",
            "_field_plane_probe_text",
        ):
            actor = getattr(self, attribute, None)
            if actor is None:
                continue
            try:
                self.remove_actor(actor, reset_camera=False, render=False)
            except (AttributeError, RuntimeError):
                try:
                    self.renderer.RemoveActor(actor)
                    self.renderer.RemoveActor2D(actor)
                except (AttributeError, RuntimeError):
                    pass
            setattr(self, attribute, None)
        self._field_plane_probe_value = None
        self._field_plane_highlight_value = None

    def _add_field_plane_to_scene(self, data: FieldPlaneRenderData) -> None:
        """Create the plane surface and line props without rebuilding geometry."""

        sample = data.sample
        grid = build_field_plane_polydata(data)
        if grid.n_cells == 0:
            raise ValueError("field plane does not intersect the selected field bounds")
        self._field_plane_actor = self.add_mesh(
            grid,
            scalars="field_value",
            cmap=self._field_plane_colormap(data.colormap),
            clim=data.value_range,
            opacity=data.opacity,
            lighting=False,
            show_scalar_bar=False,
            nan_color="#FFFFFF",
            nan_opacity=0.0,
            pickable=True,
            name="field_viewer_plane",
        )
        outline = grid.outline()
        self._field_plane_outline_actor = self.add_mesh(
            outline,
            color="#1E293B",
            line_width=1.4,
            lighting=False,
            pickable=False,
            name="field_viewer_plane_outline",
        )
        all_polylines = [
            polyline
            for isoline in data.isolines
            for polyline in isoline.polylines
        ]
        if all_polylines:
            self._field_plane_isoline_actor = self.add_mesh(
                build_contour_polydata(
                    ContourData(
                        polylines=all_polylines,
                        color=(0.08, 0.12, 0.20),
                        line_width=1.15,
                    )
                ),
                color="#162033",
                line_width=1.15,
                lighting=False,
                render_lines_as_tubes=False,
                pickable=False,
                name="field_viewer_isolines",
            )

    def _on_field_plane_mouse_move(self, _caller, _event) -> None:
        """Display a local field value and the corresponding isoline on hover."""

        data = self._field_plane_data
        if (
            data is None
            or not data.show_probe
            or self._field_plane_actor is None
        ):
            self._clear_field_plane_probe()
            return
        try:
            position = self.interactor.GetEventPosition()
            # The sampled field plane has an explicit position and basis, so
            # ray-plane intersection is exact here and avoids an O(cells)
            # VTK cell-pick for every mouse-move event.
            point = self._field_plane_point_from_display(position, data)
            value = self._field_plane_value_at(data, point)
            if value is None:
                self._clear_field_plane_probe()
                return
            self._queue_field_plane_probe_update(point, value, position)
        except (AttributeError, RuntimeError, ValueError):
            self._clear_field_plane_probe()

    def _queue_field_plane_probe_update(
        self,
        point: np.ndarray,
        value: float,
        display_position: tuple[int, int] | np.ndarray,
    ) -> None:
        """Coalesce high-frequency hover events before rendering feedback."""

        position = tuple(
            int(component)
            for component in np.asarray(display_position).reshape(2)
        )
        self._field_plane_probe_pending = (
            np.asarray(point, dtype=np.float64).reshape(3).copy(),
            float(value),
            position,
        )
        if not self._field_plane_probe_refresh_timer.isActive():
            self._field_plane_probe_refresh_timer.start()

    def _flush_field_plane_probe_update(self) -> None:
        """Render only the latest queued probe value at a bounded rate."""

        pending = self._field_plane_probe_pending
        self._field_plane_probe_pending = None
        if pending is None:
            return
        data = self._field_plane_data
        if data is None or not data.show_probe:
            return
        point, value, display_position = pending
        if self._field_plane_value_at(data, point) is None:
            return
        self._set_field_plane_probe(point, value, display_position)

    def _field_plane_point_from_display(
        self,
        display_position: tuple[int, int] | np.ndarray,
        data: FieldPlaneRenderData,
    ) -> np.ndarray | None:
        """Intersect the display ray with the finite plane as a picker fallback."""

        position = np.asarray(display_position, dtype=np.float64).reshape(2)
        endpoints = []
        for depth in (0.0, 1.0):
            self.renderer.SetDisplayPoint(float(position[0]), float(position[1]), depth)
            self.renderer.DisplayToWorld()
            world = np.asarray(self.renderer.GetWorldPoint(), dtype=np.float64)
            if abs(float(world[3])) <= np.finfo(np.float64).eps:
                return None
            endpoints.append(world[:3] / world[3])
        direction = endpoints[1] - endpoints[0]
        length = float(np.linalg.norm(direction))
        if length <= np.finfo(np.float64).eps:
            return None
        direction /= length
        state = data.sample.state
        denominator = float(np.dot(direction, state.normal))
        if abs(denominator) <= 1.0e-10:
            return None
        distance = float(np.dot(state.center_mm - endpoints[0], state.normal) / denominator)
        if distance < 0.0:
            return None
        point = endpoints[0] + distance * direction
        return point if self._field_plane_value_at(data, point) is not None else None

    @staticmethod
    def _field_plane_value_at(
        data: FieldPlaneRenderData,
        point: np.ndarray | None,
    ) -> float | None:
        """Return a value only where the selected finite field is defined."""

        if point is None:
            return None
        position = np.asarray(point, dtype=np.float64).reshape(3)
        bounds = data.sample.source_bounds_mm
        if not np.logical_and(
            position >= bounds[0] - 1.0e-9,
            position <= bounds[1] + 1.0e-9,
        ).all():
            return None
        return data.sample.value_at(position)

    def _set_field_plane_probe(
        self,
        point: np.ndarray,
        value: float,
        display_position: tuple[int, int] | np.ndarray,
    ) -> None:
        """Update the textual probe and emphasize the nearest rendered isoline."""

        data = self._field_plane_data
        if data is None:
            return
        nearest = None
        if data.isolines:
            nearest = min(data.isolines, key=lambda item: abs(item.value - value))
        nearest_value = nearest.value if nearest is not None else None
        unchanged = (
            self._field_plane_probe_value is not None
            and abs(self._field_plane_probe_value - value) <= 1.0e-7
            and self._field_plane_highlight_value == nearest_value
        )
        if unchanged:
            return
        self._field_plane_probe_value = float(value)
        if self._field_plane_probe_callback is not None:
            self._field_plane_probe_callback(float(value))
        if self._field_plane_probe_text is None:
            from vtkmodules.vtkRenderingCore import vtkTextActor

            text_actor = vtkTextActor()
            text_property = text_actor.GetTextProperty()
            text_property.SetFontSize(16)
            text_property.SetColor(0.05, 0.10, 0.20)
            text_property.SetBackgroundColor(1.0, 1.0, 1.0)
            text_property.SetBackgroundOpacity(0.88)
            text_property.SetFrame(True)
            text_property.SetFrameColor(0.20, 0.32, 0.48)
            self.renderer.AddActor2D(text_actor)
            self._field_plane_probe_text = text_actor
        try:
            self._field_plane_probe_text.SetInput(f"场值：{value:.5g} mm")
            position = np.asarray(display_position, dtype=float).reshape(2)
            self._field_plane_probe_text.SetPosition(
                float(position[0] + 14.0),
                float(position[1] + 18.0),
            )
            self._field_plane_probe_text.SetVisibility(True)
        except AttributeError:
            pass
        if nearest_value != self._field_plane_highlight_value:
            if self._field_plane_probe_actor is not None:
                try:
                    self.remove_actor(
                        self._field_plane_probe_actor,
                        reset_camera=False,
                        render=False,
                    )
                except (AttributeError, RuntimeError):
                    pass
                self._field_plane_probe_actor = None
            self._field_plane_highlight_value = nearest_value
            if nearest is not None:
                self._field_plane_probe_actor = self.add_mesh(
                    build_contour_polydata(
                        ContourData(
                            polylines=list(nearest.polylines),
                            color=(1.0, 0.79, 0.12),
                            line_width=3.0,
                        )
                    ),
                    color="#F59E0B",
                    line_width=3.0,
                    lighting=False,
                    render_lines_as_tubes=True,
                    pickable=False,
                    name="field_viewer_probe_isoline",
                )
        self.render()

    def _clear_field_plane_probe(self) -> None:
        """Hide the hover-specific props without discarding the sampled plane."""

        self._field_plane_probe_pending = None
        self._field_plane_probe_refresh_timer.stop()
        if self._field_plane_probe_value is None and self._field_plane_probe_actor is None:
            return
        self._field_plane_probe_value = None
        if self._field_plane_probe_callback is not None:
            self._field_plane_probe_callback(None)
        self._field_plane_highlight_value = None
        if self._field_plane_probe_text is not None:
            try:
                self._field_plane_probe_text.SetVisibility(False)
            except AttributeError:
                pass
        if self._field_plane_probe_actor is not None:
            try:
                self.remove_actor(
                    self._field_plane_probe_actor,
                    reset_camera=False,
                    render=False,
                )
            except (AttributeError, RuntimeError):
                pass
            self._field_plane_probe_actor = None
        self.render()

    def _apply_implicit_clipping(self, mapper) -> None:
        """Apply the current display-only clipping state to a volume mapper."""

        if not hasattr(mapper, "RemoveAllClippingPlanes"):
            return
        mapper.RemoveAllClippingPlanes()
        if self.section_planes is not None:
            mapper.SetClippingPlanes(self.section_planes)
        elif self.section_plane is not None:
            mapper.AddClippingPlane(self.section_plane)
        elif self.section_bounds is not None:
            from vtkmodules.vtkCommonDataModel import vtkPlane

            bounds = self.section_bounds
            lower_origin = np.array((bounds[0], bounds[2], bounds[4]), dtype=float)
            upper_origin = np.array((bounds[1], bounds[3], bounds[5]), dtype=float)
            for axis in range(3):
                lower = vtkPlane()
                lower.SetOrigin(*lower_origin)
                lower.SetNormal(*np.eye(3, dtype=float)[axis])
                upper = vtkPlane()
                upper.SetOrigin(*upper_origin)
                upper.SetNormal(*(-np.eye(3, dtype=float)[axis]))
                mapper.AddClippingPlane(lower)
                mapper.AddClippingPlane(upper)

    def _apply_mesh_clipping(self, mapper) -> None:
        """Update polygon clipping on the mapper without rebuilding geometry."""

        if not hasattr(mapper, "RemoveAllClippingPlanes"):
            return
        mapper.RemoveAllClippingPlanes()
        if not self.section_enabled:
            return
        planes = self._mesh_clipping_plane_collection()
        if planes is not None:
            mapper.SetClippingPlanes(planes)

    def _mesh_clipping_plane_collection(self):
        """Invert implicit-volume planes for VTK polygon mapper semantics."""

        from vtkmodules.vtkCommonDataModel import vtkPlane, vtkPlaneCollection

        if self.section_planes is not None:
            source_planes = [
                self.section_planes.GetPlane(index)
                for index in range(self.section_planes.GetNumberOfPlanes())
            ]
        elif self.section_plane is not None:
            source_planes = [self.section_plane]
        else:
            return None
        collection = vtkPlaneCollection()
        for source in source_planes:
            plane = vtkPlane()
            plane.SetOrigin(source.GetOrigin())
            plane.SetNormal(*(-np.asarray(source.GetNormal(), dtype=float)))
            collection.AddItem(plane)
        return collection

    def _update_mesh_clipping(self) -> None:
        for actor in self._pv_actors:
            mapper = actor.GetMapper() if hasattr(actor, "GetMapper") else None
            if mapper is not None:
                self._apply_mesh_clipping(mapper)

    def _update_implicit_clipping(self) -> None:
        """Update clipping planes without rebuilding sampled volume actors."""

        for actor in self._implicit_actors:
            self._apply_implicit_clipping(actor.GetMapper())
    
    def _update_bbox(self):
        """更新包围盒"""
        bounds = []
        valid = [m for m in self.meshes if len(m.vertices) > 0]
        valid.extend(
            m for m in getattr(self, "scene_overlays", ()) if len(m.vertices) > 0
        )
        bounds.extend(
            (
                np.asarray(mesh.vertices, dtype=np.float64).min(axis=0),
                np.asarray(mesh.vertices, dtype=np.float64).max(axis=0),
            )
            for mesh in valid
        )
        bounds.extend((field.bounds[0], field.bounds[1]) for field in self.implicit_fields)
        for contour in self.contours:
            polylines = [
                np.asarray(polyline, dtype=np.float64)
                for polyline in contour.polylines
                if np.asarray(polyline).ndim == 2 and len(polyline) > 0
            ]
            if polylines:
                points = np.concatenate(polylines, axis=0)
                bounds.append((points.min(axis=0), points.max(axis=0)))
        if self._cell_map_preview is not None:
            preview_bounds = np.asarray(self._cell_map_preview.bounds, dtype=np.float64)
            bounds.append((preview_bounds[0], preview_bounds[1]))
        if self._field_plane_data is not None:
            plane_points = np.asarray(
                self._field_plane_data.sample.points,
                dtype=np.float64,
            ).reshape((-1, 3))
            bounds.append((plane_points.min(axis=0), plane_points.max(axis=0)))
        if not bounds:
            self._bbox_center = np.zeros(3, dtype=np.float32)
            self._bbox_size = 1.0
            self._bbox_dimensions = np.ones(3, dtype=np.float32)  # 添加三维尺寸
            self._scene_bounds = np.array(
                [[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]], dtype=np.float64
            )
            return

        vmin = np.min(np.stack([item[0] for item in bounds]), axis=0)
        vmax = np.max(np.stack([item[1] for item in bounds]), axis=0)
        self._scene_bounds = np.vstack((vmin, vmax))
        self._bbox_center = (vmin + vmax) * 0.5
        self._bbox_dimensions = vmax - vmin  # 保存三维尺寸
        self._bbox_size = max(float(np.max(vmax - vmin)), 1.0)

    @staticmethod
    def _cell_map_wireframe(cell_map):
        """Create a 3-D line grid from cell-map axes without triangulating it."""

        segments = np.asarray(cell_map.wireframe_segments(), dtype=np.float64)
        if segments.ndim != 3 or segments.shape[1:] != (2, 3):
            raise ValueError("Cell Map wireframe segments must have shape (N, 2, 3)")
        points = segments.reshape((-1, 3))
        line_cells = np.column_stack(
            (
                np.full(len(segments), 2, dtype=np.int32),
                np.arange(len(points), dtype=np.int32).reshape((-1, 2)),
            )
        ).ravel()
        return pv.PolyData(
            points,
            lines=line_cells,
        )

    def _add_cell_map_preview_to_scene(self, cell_map) -> None:
        grid = self._cell_map_wireframe(cell_map)
        self._cell_map_actor = self.add_mesh(
            grid,
            color="#AAB2BD",
            line_width=1.0,
            lighting=False,
            show_edges=False,
        )
        frame = cell_map.frame
        arrow_length = max(
            float(np.min(cell_map.spacing_mm)) * 1.6,
            float(np.max(cell_map.extent_mm)) * 0.12,
        )
        endpoint_points = []
        endpoint_labels = []
        for label, axis, color in (
            ("U", frame.u_axis, "#E85B65"),
            ("V", frame.v_axis, "#52C788"),
            ("W", frame.w_axis, "#5BA8E8"),
        ):
            arrow = pv.Arrow(
                start=frame.origin,
                direction=axis,
                scale=arrow_length,
                tip_length=0.22,
                tip_radius=0.08,
                shaft_radius=0.025,
            )
            self._cell_map_frame_actors.append(
                self.add_mesh(arrow, color=color, lighting=False, pickable=False)
            )
            endpoint_points.append(frame.origin + axis * arrow_length * 1.12)
            endpoint_labels.append(label)
        label_actor = self.add_point_labels(
            np.asarray(endpoint_points),
            endpoint_labels,
            font_size=13,
            text_color="#263238",
            point_size=0,
            shape=None,
            always_visible=True,
            show_points=False,
        )
        self._cell_map_frame_actors.append(label_actor)
    
    def _add_ground_platform(self):
        """添加网格平台（类似 3D 打印软件的打印平台）"""
        try:
            # 计算平台尺寸（更宽）
            platform_size = self._bbox_size * 3.0  # 从 2.5 增加到 3.0，更宽
            
            # 找到模型的最低点
            if hasattr(self, "_scene_bounds"):
                z_min = float(self._scene_bounds[0, 2])
                z_max = float(self._scene_bounds[1, 2])
            else:
                vertices = np.concatenate(
                    [mesh.vertices for mesh in self.meshes if len(mesh.vertices)],
                    axis=0,
                )
                z_min = float(vertices[:, 2].min())
                z_max = float(vertices[:, 2].max())
            
            # 平台跟随当前可见模型的最低点，避免非零坐标模型悬空。
            platform_z = z_min
            platform_center = [self._bbox_center[0], self._bbox_center[1], platform_z]
            
            # 高度只取决于模型尺寸，不受模型绝对坐标偏移影响。
            cube_height = max(
                self._bbox_size * 2.0,
                (z_max - z_min) + self._bbox_size * 1.5,
            )
            
            # 创建地面平台（半透明平面）
            if self.show_ground_plane:
                plane = pv.Plane(
                    center=platform_center,
                    direction=[0, 0, 1],
                    i_size=platform_size,
                    j_size=platform_size
                )
                
                self._ground_plane_actor = self.add_mesh(
                    plane,
                    color='#888888',
                    opacity=0.3,
                    show_edges=False,
                    lighting=False
                )
            
            # 创建网格线（虚线效果）
            if self.show_ground_grid:
                # 网格间距
                grid_spacing = platform_size / 30
                
                # 创建网格线
                lines = []
                
                # X 方向的线
                for i in range(31):
                    y = -platform_size/2 + i * grid_spacing
                    line = pv.Line(
                        [platform_center[0] - platform_size/2, platform_center[1] + y, platform_z],
                        [platform_center[0] + platform_size/2, platform_center[1] + y, platform_z]
                    )
                    lines.append(line)
                
                # Y 方向的线
                for i in range(31):
                    x = -platform_size/2 + i * grid_spacing
                    line = pv.Line(
                        [platform_center[0] + x, platform_center[1] - platform_size/2, platform_z],
                        [platform_center[0] + x, platform_center[1] + platform_size/2, platform_z]
                    )
                    lines.append(line)
                
                # 合并所有线
                grid = lines[0]
                for line in lines[1:]:
                    grid = grid + line
                
                # 网格线更虚（降低透明度）
                self._ground_grid_actor = self.add_mesh(
                    grid,
                    color='#999999',  # 中灰色
                    line_width=1.0,  # 细线
                    opacity=0.4,  # 更透明（虚）
                    lighting=False,
                    style='wireframe'  # 线框模式
                )
                
                # 创建立方体边框（12条边）
                half_size = platform_size / 2
                cube_top_z = platform_z + cube_height
                
                # 底部4条边
                bottom_edges = [
                    # 前边
                    pv.Line(
                        [platform_center[0] - half_size, platform_center[1] - half_size, platform_z],
                        [platform_center[0] + half_size, platform_center[1] - half_size, platform_z]
                    ),
                    # 后边
                    pv.Line(
                        [platform_center[0] - half_size, platform_center[1] + half_size, platform_z],
                        [platform_center[0] + half_size, platform_center[1] + half_size, platform_z]
                    ),
                    # 左边
                    pv.Line(
                        [platform_center[0] - half_size, platform_center[1] - half_size, platform_z],
                        [platform_center[0] - half_size, platform_center[1] + half_size, platform_z]
                    ),
                    # 右边
                    pv.Line(
                        [platform_center[0] + half_size, platform_center[1] - half_size, platform_z],
                        [platform_center[0] + half_size, platform_center[1] + half_size, platform_z]
                    ),
                ]
                
                # 顶部4条边
                top_edges = [
                    # 前边
                    pv.Line(
                        [platform_center[0] - half_size, platform_center[1] - half_size, cube_top_z],
                        [platform_center[0] + half_size, platform_center[1] - half_size, cube_top_z]
                    ),
                    # 后边
                    pv.Line(
                        [platform_center[0] - half_size, platform_center[1] + half_size, cube_top_z],
                        [platform_center[0] + half_size, platform_center[1] + half_size, cube_top_z]
                    ),
                    # 左边
                    pv.Line(
                        [platform_center[0] - half_size, platform_center[1] - half_size, cube_top_z],
                        [platform_center[0] - half_size, platform_center[1] + half_size, cube_top_z]
                    ),
                    # 右边
                    pv.Line(
                        [platform_center[0] + half_size, platform_center[1] - half_size, cube_top_z],
                        [platform_center[0] + half_size, platform_center[1] + half_size, cube_top_z]
                    ),
                ]
                
                # 垂直4条边（连接底部和顶部）
                vertical_edges = [
                    # 左前
                    pv.Line(
                        [platform_center[0] - half_size, platform_center[1] - half_size, platform_z],
                        [platform_center[0] - half_size, platform_center[1] - half_size, cube_top_z]
                    ),
                    # 右前
                    pv.Line(
                        [platform_center[0] + half_size, platform_center[1] - half_size, platform_z],
                        [platform_center[0] + half_size, platform_center[1] - half_size, cube_top_z]
                    ),
                    # 左后
                    pv.Line(
                        [platform_center[0] - half_size, platform_center[1] + half_size, platform_z],
                        [platform_center[0] - half_size, platform_center[1] + half_size, cube_top_z]
                    ),
                    # 右后
                    pv.Line(
                        [platform_center[0] + half_size, platform_center[1] + half_size, platform_z],
                        [platform_center[0] + half_size, platform_center[1] + half_size, cube_top_z]
                    ),
                ]
                
                # 合并所有边框
                all_edges = bottom_edges + top_edges + vertical_edges
                cube_border = all_edges[0]
                for edge in all_edges[1:]:
                    cube_border = cube_border + edge
                
                # 添加黑色立方体边框（细线）
                self.add_mesh(
                    cube_border,
                    color='#333333',  # 深灰色（不是纯黑）
                    line_width=2,  # 细一点（从3改为2）
                    opacity=1.0,  # 完全不透明
                    lighting=False
                )
            
        except Exception as e:
            print(f"[警告] 添加网格平台失败: {e}")
            import traceback
            traceback.print_exc()
    
    def _setup_lighting(self):
        """Apply a model-centered studio rig for solid form readability."""

        self.remove_all_lights()
        for light in build_studio_lights(self._bbox_center, self._bbox_size):
            self.add_light(light)

    def get_render_backend_status(self) -> RenderBackendStatus:
        """Inspect the embedded OpenGL context and active volume mappers."""

        try:
            supports_opengl = bool(self.ren_win.SupportsOpenGL())
            capability_report = self.ren_win.ReportCapabilities()
            mapper_supported = True
            for actor in self._implicit_actors:
                mapper_supported = mapper_supported and bool(
                    actor.GetMapper().IsRenderSupported(
                        self.ren_win,
                        actor.GetProperty(),
                    )
                )
            return RenderBackendStatus.from_capabilities(
                capability_report,
                supports_opengl=supports_opengl,
                mapper_supported=mapper_supported,
            )
        except Exception as exc:
            return RenderBackendStatus(
                "渲染后端探测失败",
                False,
                "未知图形设备",
                f"{type(exc).__name__}: {exc}",
            )

    def _configure_anti_aliasing(self) -> None:
        """Apply the anti-aliasing portion of the current quality preset."""

        try:
            self.disable_anti_aliasing()
            aa_type = RENDER_QUALITY_SETTINGS[self.render_quality].anti_aliasing
            if self.enable_antialiasing and aa_type != "none":
                self.enable_anti_aliasing(aa_type)
        except (AttributeError, RuntimeError) as exc:
            print(f"[警告] 抗锯齿配置失败: {exc}")
    
    def _enable_advanced_features(self):
        """启用高级渲染特性"""
        quality = RENDER_QUALITY_SETTINGS[self.render_quality]
        # 启用 SSAO（环境光遮蔽）- 使用更保守的参数
        # PyVista's SSAO render pass can omit VTK volume props in the embedded
        # Qt renderer. Implicit fields use GPU volume ray casting and must use
        # VTK's normal render pass.
        if self.implicit_fields:
            try:
                if hasattr(self, 'disable_ssao'):
                    self.disable_ssao()
                if hasattr(self, 'disable_shadows'):
                    self.disable_shadows()
            except (AttributeError, RuntimeError):
                pass
            return

        if self.enable_ssao_flag and quality.use_ssao:
            try:
                # 使用更小的半径和更大的偏移，减少过度阴影
                self.enable_ssao(
                    radius=self._bbox_size * 0.03,  # 减小半径（从0.06降到0.03）
                    bias=0.03,  # 增大偏移（从0.015增到0.03），减少假阴影
                    kernel_size=16,  # 减少采样数（从24降到16），提高性能
                    blur=True  # 启用模糊，使效果更柔和
                )
                print("[提示] SSAO 已启用（保守参数）")
            except Exception as e:
                print(f"[警告] SSAO 启用失败: {e}")
        else:
            # 尝试禁用 SSAO
            try:
                # PyVista 没有 disable_ssao() 方法，但我们可以尝试关闭
                if hasattr(self, 'disable_ssao'):
                    self.disable_ssao()
                # 或者尝试设置极小的参数来最小化效果
                elif hasattr(self, 'enable_ssao'):
                    # 不调用 enable_ssao，让它保持未启用状态
                    pass
            except:
                pass

        try:
            if self.enable_shadows_flag and quality.use_shadows:
                self.enable_shadows()
            elif hasattr(self, 'disable_shadows'):
                self.disable_shadows()
        except (AttributeError, RuntimeError) as exc:
            print(f"[警告] 阴影配置失败: {exc}")
    
    def _add_model_info_overlay(self):
        """在左上角添加模型信息显示"""
        if not self.meshes:
            self.remove_actor("model_info", reset_camera=False, render=False)
            return
        
        # 统计信息
        total_vertices = sum(len(m.vertices) for m in self.meshes)
        total_faces = sum(len(m.faces) for m in self.meshes)
        display_vertices = sum(mesh.n_points for mesh in self._pv_meshes)
        display_faces = sum(mesh.n_cells for mesh in self._pv_meshes)
        num_meshes = len(self.meshes)
        
        # 获取三维尺寸
        size_x, size_y, size_z = self._bbox_dimensions
        
        # 构建信息文本（只显示模型信息）
        info_lines = [
            "--- Model Info ---",
            f"Meshes  : {num_meshes}",
            f"Source vertices : {total_vertices:,}",
            f"Source faces    : {total_faces:,}",
            f"Display vertices: {display_vertices:,}",
            f"Display faces   : {display_faces:,}",
            f"Size X  : {size_x:.1f} mm",
            f"Size Y  : {size_y:.1f} mm",
            f"Size Z  : {size_z:.1f} mm",
        ]
        
        info_text = "\n".join(info_lines)
        background = np.asarray(self.renderer.GetBackground(), dtype=np.float64)
        luminance = float(np.dot(background, (0.2126, 0.7152, 0.0722)))
        text_color = "#263238" if luminance >= 0.55 else "#F3F4F6"
        
        # 添加文本到左上角
        try:
            self.add_text(
                info_text,
                position='upper_left',
                font_size=11,
                color=text_color,
                font='courier',
                shadow=False,
                name="model_info",
            )
        except Exception as e:
            # 如果失败，尝试最简单的方式
            try:
                self.add_text(
                    info_text,
                    position='upper_left',
                    font_size=12,
                    color=text_color,
                    name="model_info",
                )
            except:
                print(f"[警告] 添加信息显示失败: {e}")
    
    def set_plane(self, point: np.ndarray, normal: np.ndarray, size: float = None):
        """设置要显示的平面"""
        self.plane_point = np.asarray(point, dtype=np.float32)
        self.plane_normal = np.asarray(normal, dtype=np.float32)
        self.plane_normal = self.plane_normal / np.linalg.norm(self.plane_normal)
        
        if size is not None:
            self.plane_size = size
        else:
            self.plane_size = self._bbox_size * 1.5  # 增大平面尺寸
        
        self.show_plane = True
        self._update_plane()
    
    def hide_plane(self):
        """隐藏平面"""
        self.show_plane = False
        if self.plane_actor is not None:
            try:
                self.remove_actor(self.plane_actor)
                self.plane_actor = None
            except:
                pass
        self.render()

    def set_section_box(self, bounds: np.ndarray):
        """Apply a display-only axis-aligned section box to all mesh layers."""

        values = np.asarray(bounds, dtype=np.float64)
        if values.shape != (6,) or not np.isfinite(values).all():
            raise ValueError("section bounds must contain six finite values")
        if not (
            values[0] < values[1]
            and values[2] < values[3]
            and values[4] < values[5]
        ):
            raise ValueError("section bounds must have positive extents")
        from vtkmodules.vtkCommonDataModel import vtkPlanes

        planes = vtkPlanes()
        planes.SetBounds(values)
        self.section_bounds = values
        self.section_planes = planes
        self.section_plane = None
        self.section_enabled = True
        if self._implicit_actors and not self.meshes:
            self._update_implicit_clipping()
        elif self._pv_actors:
            self._update_mesh_clipping()
        else:
            self._update_scene()
        self.render()

    def set_section_planes(self, planes):
        """Apply a display-only oriented clipping box from a vtkPlanes object."""

        if planes is None or not hasattr(planes, "GetNumberOfPlanes"):
            raise ValueError("planes must be a vtkPlanes object")
        if planes.GetNumberOfPlanes() == 0:
            raise ValueError("planes must contain at least one plane")
        self.section_planes = planes
        self.section_bounds = None
        self.section_plane = None
        self.section_enabled = True
        if self._implicit_actors and not self.meshes:
            self._update_implicit_clipping()
        elif self._pv_actors:
            self._update_mesh_clipping()
        else:
            self._update_scene()
        self.render()

    def set_section_plane(self, plane):
        """Apply a display-only single-plane section to all mesh layers.

        The positive side of the plane normal is removed.  The source mesh is
        not changed; polygon actors update mapper clipping in place.
        """

        if plane is None or not hasattr(plane, "EvaluateFunction"):
            raise ValueError("plane must be a VTK implicit plane")
        self.section_plane = plane
        self.section_planes = None
        self.section_bounds = None
        self.section_enabled = True
        if self._implicit_actors and not self.meshes:
            self._update_implicit_clipping()
        elif self._pv_actors:
            self._update_mesh_clipping()
        else:
            self._update_scene()
        self.render()

    def clear_section_box(self):
        """Disable display-only section clipping and restore complete meshes."""

        self.section_enabled = False
        self.section_bounds = None
        self.section_planes = None
        self.section_plane = None
        if self._implicit_actors and not self.meshes:
            self._update_implicit_clipping()
        elif self._pv_actors:
            self._update_mesh_clipping()
        else:
            self._update_scene()
        self.render()
    
    def _update_plane(self):
        """更新平面显示"""
        # 移除旧的平面
        if self.plane_actor is not None:
            try:
                self.remove_actor(self.plane_actor)
            except:
                pass
        
        if not self.show_plane or self.plane_point is None:
            return
        
        try:
            # 创建平面
            plane = pv.Plane(
                center=self.plane_point,
                direction=self.plane_normal,
                i_size=self.plane_size,
                j_size=self.plane_size
            )
            
            # 添加到场景（半透明）
            self.plane_actor = self.add_mesh(
                plane,
                color='yellow',
                opacity=0.3,
                show_edges=True,
                edge_color='orange',
                lighting=False
            )
            
            self.render()
            
        except Exception as e:
            print(f"[错误] 平面显示失败: {e}")
    
    def enable_lasso_mode(self, mesh):
        """启用套索模式（暂不支持）"""
        print("[警告] PyVista 渲染器暂不支持套索模式")
    
    def disable_lasso_mode(self):
        """禁用套索模式"""
        pass

    def close(self) -> None:
        """Release custom VTK resources in the render-window lifetime."""

        if getattr(self, "_closed", False):
            return
        for timer_name in (
            "_field_plane_probe_refresh_timer",
            "render_timer",
        ):
            timer = getattr(self, timer_name, None)
            if timer is not None:
                timer.stop()
        self._field_plane_probe_callback = None
        # Detach custom resident actors while the render window and OpenGL
        # context are still valid.  Destroying their volume mappers after
        # BasePlotter.close() is unsafe on Windows after rapid scene updates.
        for record in self._implicit_layer_cache.values():
            self._release_implicit_layer(record)
        self.clear()
        self._implicit_layer_cache.clear()
        self._mesh_layer_records.clear()
        self._mesh_display_cache.clear()
        self._implicit_actors.clear()
        self._implicit_images.clear()
        self._pv_actors.clear()
        self._pv_meshes.clear()
        self._contour_actors.clear()
        self._cell_map_frame_actors.clear()
        self._studio_environment_texture = None
        self._field_plane_actor = None
        self._field_plane_outline_actor = None
        self._field_plane_isoline_actor = None
        self._field_plane_probe_actor = None
        self._field_plane_probe_text = None
        super().close()

    def resizeEvent(self, event):
        """处理窗口大小改变事件"""
        super().resizeEvent(event)
        if getattr(self, "_closed", False) or not self.updatesEnabled():
            return
        try:
            self.render()
        except (AttributeError, RuntimeError):
            pass


# 为了兼容性，创建一个别名
GLMeshViewer = PyVistaRenderer


if __name__ == "__main__":
    """测试渲染器"""
    import sys
    from pathlib import Path
    
    app = QtWidgets.QApplication(sys.argv)
    
    # 创建窗口
    window = QtWidgets.QMainWindow()
    window.setWindowTitle("PyVista 渲染器测试")
    window.resize(1200, 900)
    
    # 创建渲染器
    renderer = PyVistaRenderer(window)
    window.setCentralWidget(renderer)
    
    # 加载测试模型
    model_path = None
    for path in [
        "resources/examples/design_domains/sole-1.stl",
        "build/test-artifacts/voronoi_lattice.stl",
    ]:
        if Path(path).exists():
            model_path = path
            break
    
    if model_path:
        print(f"加载模型: {model_path}")
        mesh = trimesh.load(model_path)
        
        mesh_data = MeshData(
            vertices=mesh.vertices,
            faces=mesh.faces,
            color=(0.29, 0.56, 0.89, 1.0),
            metallic=0.3,
            roughness=0.4
        )
        
        renderer.set_meshes([mesh_data], reset_view=True)
        print("✓ 模型已加载")
        print("  - PBR 材质: 启用")
        print("  - SSAO: 启用")
        print("  - 阴影: 启用")
        print("  - 抗锯齿: 启用")
    else:
        print("未找到测试模型，创建测试几何体...")
        # 创建测试球体
        sphere = trimesh.creation.icosphere(subdivisions=4, radius=50.0)
        mesh_data = MeshData(
            vertices=sphere.vertices,
            faces=sphere.faces,
            color=(0.29, 0.56, 0.89, 1.0),
            metallic=0.3,
            roughness=0.4
        )
        renderer.set_meshes([mesh_data], reset_view=True)
    
    window.show()
    sys.exit(app.exec_())
