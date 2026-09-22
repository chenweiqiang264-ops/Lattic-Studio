"""
Implicit surface module - 隐式曲面模块
包含 Voronoi 等隐式曲面算法
"""

from importlib import import_module

from lattice_studio.domain.cell_map import CellMap, CellMapBoundaryMode, CellMapFrame
from lattice_studio.engine.implicit.custom_unit_cell import (
    CustomUnitCellError,
    CustomUnitCellImportReport,
    PeriodicSeamAxisReport,
    PeriodicSeamReport,
    PreparedCustomUnitCell,
    UnitCellDomain,
    make_periodic_lattice,
    prepare_stl_unit_cell,
)
from lattice_studio.engine.implicit.field import (
    ImplicitBody,
    SampledFieldCleanupReport,
    SampledImplicitField,
    remove_small_negative_islands,
)
from lattice_studio.engine.implicit.precise_render import (
    PreciseRenderCamera,
    PreciseRenderMaterial,
    PreciseRenderReport,
    PreciseRenderResult,
    PreciseRenderSettings,
    render_precise_implicit,
)
from lattice_studio.engine.implicit.layer_contours import (
    ContourFieldSource,
    LayerAxis,
    LayerContour,
    LayerContourCache,
    LayerContourResult,
    available_layer_positions,
    sample_implicit_body_layer,
    sample_sampled_field_layer,
)
from lattice_studio.engine.implicit.shell import (
    ShellParameters,
    fuse_shell_lattice_fields,
    make_coordinated_shell_lattice_union,
    make_interior_shell,
    make_shell_lattice_union,
    smooth_union_values,
)
from lattice_studio.engine.implicit.lattice_pipeline import (
    AutomaticLatticeFieldPipeline,
    CpuLatticeFieldPipeline,
    LatticePipelineStatus,
    NumbaCudaLatticeFieldPipeline,
    get_default_lattice_pipeline,
)
from lattice_studio.engine.implicit.gradient_lattice import (
    GRADIENT_MODES,
    GRADIENT_RESOLUTION_STRATEGIES,
    WALL_THICKNESS_METHODS,
    GradientAxis,
    GradientControls,
    GradientMode,
    GradientResolutionStrategy,
    WallThicknessMethod,
    estimate_layer_density,
    evaluate_gradient_tpms_field,
    gradient_profile,
    gradient_projection_range,
    matlab_gradient_voxel_size,
)
from lattice_studio.engine.implicit.transition import (
    TransitionBuildResult,
    TransitionDiagnostics,
    TransitionOperand,
    TransitionQualityReport,
    TransitionSpec,
    build_transition_body,
    recommended_transition_width,
    transition_weights,
    validate_transition,
)

__all__ = [
    'CellMap',
    'CellMapBoundaryMode',
    'CellMapFrame',
    'CustomUnitCellError',
    'CustomUnitCellImportReport',
    'PeriodicSeamAxisReport',
    'PeriodicSeamReport',
    'ImplicitBody',
    'PreparedCustomUnitCell',
    'SampledFieldCleanupReport',
    'SampledImplicitField',
    'UnitCellDomain',
    'TransitionBuildResult',
    'TransitionDiagnostics',
    'TransitionOperand',
    'TransitionQualityReport',
    'TransitionSpec',
    'build_transition_body',
    'make_periodic_lattice',
    'prepare_stl_unit_cell',
    'recommended_transition_width',
    'remove_small_negative_islands',
    'PreciseRenderCamera',
    'PreciseRenderMaterial',
    'PreciseRenderReport',
    'PreciseRenderResult',
    'PreciseRenderSettings',
    'render_precise_implicit',
    'ContourFieldSource',
    'LayerAxis',
    'LayerContour',
    'LayerContourCache',
    'LayerContourResult',
    'available_layer_positions',
    'sample_implicit_body_layer',
    'sample_sampled_field_layer',
    'ShellParameters',
    'fuse_shell_lattice_fields',
    'make_coordinated_shell_lattice_union',
    'make_interior_shell',
    'make_shell_lattice_union',
    'smooth_union_values',
    'AutomaticLatticeFieldPipeline',
    'CpuLatticeFieldPipeline',
    'LatticePipelineStatus',
    'NumbaCudaLatticeFieldPipeline',
    'get_default_lattice_pipeline',
    'GRADIENT_MODES',
    'GRADIENT_RESOLUTION_STRATEGIES',
    'WALL_THICKNESS_METHODS',
    'GradientAxis',
    'GradientControls',
    'GradientMode',
    'GradientResolutionStrategy',
    'WallThicknessMethod',
    'estimate_layer_density',
    'evaluate_gradient_tpms_field',
    'gradient_profile',
    'gradient_projection_range',
    'matlab_gradient_voxel_size',
    'transition_weights',
    'validate_transition',
    'voronoi',
]


def __getattr__(name: str):
    """Load optional implicit algorithms only when they are requested."""

    if name == "voronoi":
        module = import_module("lattice_studio.engine.implicit.voronoi")
        globals()[name] = module
        return module
    raise AttributeError(name)
