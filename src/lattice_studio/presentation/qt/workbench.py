"""TPMS and STL-backed custom-cell implicit filling workbench.

This module is intentionally self-contained for the first TPMS milestone:

* Five standard periodic TPMS functions (G, D, IWP, Primitive, Neovius) are
  available through one type selector;
  watertight custom STL cells use the same Cell Map placement pipeline.
* A wall thickness in millimetres is converted to an implicit shell using the
  local field gradient.  It is therefore a physical thickness parameter,
  rather than an arbitrary isovalue-only control.
* The STL is sampled on an automatically recommended voxel grid.  CUDA is
  preferred for mesh SDF, implicit intersection, and Marching Cubes, with
  C++/PyVista and CPU extraction as safe fallbacks.
* Three explicit processing modes are available: one-shot evaluation and
  extraction, batched field evaluation followed by one extraction, and
  batched Marching-Cubes extraction with stitched surface chunks.
* The PyQt5 UI runs generation in a worker thread and reuses the project's
  PyVista renderer, including its PBR material and multi-light setup.

The supported entry point is ``python -m lattice_studio``.  Add
``--generate-example`` to generate the two example STL files without opening
the GUI.
"""

from __future__ import annotations

import argparse
from io import BytesIO
import json
from shutil import rmtree
import sys
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from tempfile import mkdtemp
from typing import TYPE_CHECKING, Callable, Literal, TypedDict

if TYPE_CHECKING:
    from PyQt5.QtWidgets import QWidget

import numpy as np
import trimesh


BACKGROUND_THREAD_SPECS = (
    ("thread", "正在取消晶格生成，请稍候再关闭。"),
    ("stl_thread", "正在取消 STL 重建，请稍候再关闭。"),
    ("shell_thread", "正在取消抽壳或壳体融合，请稍候再关闭。"),
    ("precise_render_thread", "正在取消精确渲染，请稍候再关闭。"),
    (
        "display_refinement_thread",
        "正在取消高质量显示场细化，请稍候再关闭。",
    ),
)


def request_background_thread_shutdown(
    owner,
    *,
    timeout_ms: int = 10_000,
) -> tuple[bool, str | None]:
    """Cooperatively stop every worker thread owned by the workbench."""

    active_threads = []
    for attribute, message in BACKGROUND_THREAD_SPECS:
        thread = getattr(owner, attribute, None)
        if thread is not None and thread.isRunning():
            active_threads.append((thread, message))

    for thread, _message in active_threads:
        thread.requestInterruption()
        thread.quit()

    deadline = time.monotonic() + max(int(timeout_ms), 0) / 1000.0
    for thread, message in active_threads:
        remaining_ms = max(0, int((deadline - time.monotonic()) * 1000.0))
        if not thread.wait(remaining_ms):
            return False, message
    return True, None


def request_backend_process_shutdown(owner) -> tuple[bool, str | None]:
    """Stop the loopback backend started by this workbench, if any."""

    backend_process = getattr(owner, "backend_process", None)
    if backend_process is None:
        return True, None
    try:
        backend_process.stop()
    except Exception as exc:
        return False, f"local backend shutdown failed: {type(exc).__name__}: {exc}"
    owner.backend_process = None
    owner.backend_client = None
    return True, None


PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lattice_studio.domain.cell_map import CellMap, CellMapFrame
from lattice_studio.engine.implicit.custom_unit_cell import (
    CustomUnitCellImportReport,
    PreparedCustomUnitCell,
    make_periodic_lattice,
    prepare_stl_unit_cell,
)
from lattice_studio.engine.implicit.field import (
    ImplicitBody,
    SampledImplicitField,
)
from lattice_studio.engine.implicit.primitives import (
    ImplicitPrimitive,
    PrimitiveKind,
    euler_deg_from_rotation_matrix,
)
from lattice_studio.engine.implicit.precise_render import (
    PreciseRenderCamera,
    PreciseRenderMaterial,
    PreciseRenderResult,
    PreciseRenderSettings,
    render_precise_implicit,
)
from lattice_studio.application.workspace import (
    AnalyticDesignDomain,
    DesignDomainValue,
    ManagedDesignWorkspace as DesignWorkspace,
    MeshDesignDomain,
    PersistedDerivedMesh,
)
from lattice_studio.presentation.http.task_results import (
    BackendGenerationResult,
    batch_display_field_artifact_ids,
    display_field_artifact_id,
    read_generation_batch,
    read_generation_result,
    read_sampled_field,
)
from lattice_studio.presentation.http.workspace_results import (
    read_workspace_projection,
)
from lattice_studio.engine.implicit.field_viewer import (
    FieldPlaneCache,
    FieldPlaneState,
    FieldSourceData,
    expand_implicit_body_bounds,
    extract_field_plane_isolines,
    field_plane_cache_key,
    field_surface_hit_from_ray,
    orthonormal_plane_axes,
    resolve_field_value_range,
    sample_field_plane,
    source_sampling_spacing,
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
    evaluate_gradient_tpms_field,
    gradient_projection_range,
    make_tpms_gradient_spec,
    matlab_gradient_voxel_size,
)
from lattice_studio.engine.implicit.geometry_compute import get_default_geometry_backend
from lattice_studio.engine.implicit.lattice_pipeline import get_default_lattice_pipeline
from lattice_studio.engine.implicit.layer_contours import (
    ContourFieldSource,
    LayerContourCache,
    LayerContourResult,
    available_layer_positions,
    resolve_layer_spacing,
    sample_implicit_body_layer,
    sample_sampled_field_layer,
)
from lattice_studio.engine.implicit.shell import (
    fuse_shell_lattice_fields,
    make_coordinated_shell_lattice_union,
    make_interior_shell,
    make_shell_lattice_union,
)
from lattice_studio.engine.implicit.surface_extraction import (
    ExtractionStatistics,
    grid_definition as extraction_grid_definition,
)
from lattice_studio.engine.implicit.tpms_compute import (
    TPMSFieldSpec as ComputeTPMSFieldSpec,
    TransitionBlendSpec as ComputeTransitionBlendSpec,
    get_default_tpms_backend,
    sigmoid_transition_weights as compute_sigmoid_transition_weights,
)
from lattice_studio.engine.implicit.transition import (
    TransitionDiagnostics,
    TransitionOperand,
    TransitionQualityReport,
    TransitionSpec as ProviderTransitionSpec,
    TransitionWeightKind,
    build_transition_body as build_provider_transition_body,
    recommended_transition_width as recommend_provider_transition_width,
    validate_transition as validate_provider_transition,
)


from lattice_studio.engine.workflows import (
    TPMSKind,
    LatticeKind,
    TPMS_KINDS,
    CellSize,
    CellMapMode,
    PlaneAxis,
    TransitionDriverMode,
    ProcessingMode,
    ExportSpacingMode,
    ProgressCallback,
    EvaluationStageCallback,
    TPMSParameterState,
    FRAME_INPUT_DECIMALS,
    FRAME_ORIGIN_DEFAULT_ATOL_MM,
    PROGRESS_DOMAIN_SDF,
    PROGRESS_TPMS_FIELD,
    PROGRESS_INTERSECTION,
    PROGRESS_MARCHING_CUBES,
    _tpms_backend_label,
    _geometry_backend_label,
    _lattice_pipeline_backend_label,
    TPMSParameters,
    CustomUnitCellParameters,
    LatticeParameters,
    SamplingParameters,
    SamplingRecommendation,
    ImplicitGenerationResult,
    DisplayRefinementReport,
    DISPLAY_FIELD_REFINEMENT_FACTORS,
    DISPLAY_QUALITY_LABELS,
    StlReconstructionTarget,
    PreciseRenderTarget,
    TransitionGenerationMetadata,
    TransitionParameters,
    MeshQuality,
    SimplificationReport,
    MeshConditioningReport,
    NumericalFragmentCleanupReport,
    FieldFragmentCleanupReport,
    ExportMeshResult,
    ExportGridEstimate,
    CellDisplaySampleEstimate,
    _format_spacing_mm,
    describe_display_sampling,
    export_spacing_limit_labels,
    _close_small_field_gaps,
    _sample_triangle_quality,
    _condition_mesh_for_slicing,
    _reconstruct_mesh_by_processing_mode,
    reconstruct_implicit_mesh,
    _compiled_simplify_mesh,
    _mesh_deviation_mm,
    simplify_mesh_with_quality,
    cell_map_for_parameters,
    _parameters_for_cell_map,
    transition_cell_maps,
    transition_cell_map,
    lattice_transition_cell_maps,
    lattice_sampling_recommendation,
    custom_cell_sampling_recommendation,
    recommend_voxel_size,
    recommendation_for_voxel_size,
    _grid_definition,
    recommend_display_voxel_size,
    _display_recommendation,
    estimate_cell_display_samples,
    _sample_implicit_body_on_grid,
    _sample_implicit_body,
    refine_implicit_display_field,
    _display_field_matches_grid,
    _sampled_fields_share_grid,
    _display_field_meets_recommendation,
    _sample_intersection_with_domain_cache,
    _make_design_domain_body,
    vtk_bounds_from_implicit_body,
    _make_lattice_feature_body,
    _make_lattice_body,
    _make_custom_lattice_feature_body,
    _make_transition_operand,
    _make_transition_body,
    _batch_ranges,
    _grid_batch_points,
    _sample_grid_points,
    _display_batch_progress_text,
    _sample_display_grid_points,
    _sample_grid_slab,
    _stitch_mesh_chunks,
    marching_cubes_chunked,
    _marching_cubes_field,
    _generate_mesh_by_processing_mode,
    _rotation_matrix,
    transition_plane_geometry,
    sigmoid_transition_weights,
    _tpms_shell_field,
    _cell_map_tpms_shell_field,
    _tpms_field_spec,
    _gradient_controls,
    _tpms_transition_chunk_field,
    _tpms_transition_blended_field,
    _compute_design_domain_sdf,
    implicit_intersection,
    generate_tpms_lattice,
    generate_tpms_transition,
    generate_implicit_lattice,
    generate_implicit_custom_lattice,
    generate_implicit_lattice_transition,
    generate_implicit_transition,
    cell_map_for_design_domain,
    _display_recommendation_for_bounds,
    _sample_implicit_body_for_bounds,
    _make_lattice_feature_body_for_design_domain,
    _make_lattice_body_for_design_domain,
    _domain_sampling_recommendation,
    generate_implicit_domain_preview_for_design_domain,
    generate_implicit_lattice_for_design_domain,
    generate_implicit_custom_lattice_for_design_domain,
    _transition_plane_geometry_for_bounds,
    _make_transition_operand_for_design_domain,
    generate_implicit_lattice_transition_for_design_domain,
    generate_implicit_domain_preview,
    design_domain_extraction_map,
    shell_sampling_recommendation,
    generate_implicit_shell,
    generate_implicit_shell_lattice_union,
)
from lattice_studio.engine.mesh_quality import (
    _clean_display_field_components,
    inspect_mesh,
    recommended_floating_component_extent_mm,
    remove_numerical_fragments,
    remove_small_negative_field_components,
    repair_mesh,
    safe_mesh_volume,
)
from lattice_studio.engine.reconstruction_grid import (
    estimate_export_grid,
    recommend_export_spacing,
    resolve_export_spacing,
)





# Imported float32 STL bounds and persisted UI values may already be rounded to
# four decimals even though current controls expose six. Treat that display-
# precision round trip as the same automatic origin.




UI_THEME: dict[str, str] = {
    "background": "#17181D",
    "navigation": "#1D1F25",
    "surface": "#202229",
    "surface_raised": "#262831",
    "surface_active": "#2D303A",
    "border": "#353842",
    "border_strong": "#484C59",
    "primary": "#7C3AED",
    "primary_hover": "#8B5CF6",
    "primary_pressed": "#6D28D9",
    "focus": "#A78BFA",
    "accent": "#0891B2",
    "accent_hover": "#0E7490",
    "text_primary": "#F4F4F6",
    "text_secondary": "#C2C4CC",
    "text_muted": "#8F929E",
    "success": "#22C55E",
    "warning": "#F59E0B",
    "danger": "#EF4444",
    "viewport_border": "#C8CBD2",
}


def _application_stylesheet() -> str:
    """Return the token-driven QSS shared by the complete workbench UI."""

    t = UI_THEME
    return f"""
        QMainWindow, QDialog, QMessageBox, QWidget {{
            background-color: {t['background']};
            color: {t['text_secondary']};
            font-family: "Segoe UI", "Microsoft YaHei UI", sans-serif;
            font-size: 13px;
        }}
        QTabWidget::pane {{
            border: 1px solid {t['border']};
            background: {t['navigation']};
            top: -1px;
        }}
        QTabBar {{
            background: {t['background']};
        }}
        QTabBar::tab {{
            background: {t['navigation']};
            color: {t['text_muted']};
            border: 1px solid {t['border']};
            border-bottom: 2px solid transparent;
            padding: 8px 12px;
            min-width: 88px;
            min-height: 24px;
        }}
        QTabBar::tab:first {{ border-top-left-radius: 5px; }}
        QTabBar::tab:last {{ border-top-right-radius: 5px; }}
        QTabBar::tab:selected {{
            background: #292532;
            color: {t['text_primary']};
            border-bottom-color: {t['primary']};
        }}
        QTabBar::tab:hover:!selected {{
            background: {t['surface_raised']};
            color: {t['text_secondary']};
        }}
        QTabBar::tab:focus {{ border-color: {t['focus']}; }}
        QScrollArea, QScrollArea > QWidget > QWidget {{
            border: none;
            background: {t['navigation']};
        }}
        QGroupBox {{
            background: {t['surface']};
            border: 1px solid {t['border']};
            border-radius: 6px;
            margin-top: 18px;
            padding: 12px 10px 10px 10px;
            color: {t['text_primary']};
            font-weight: 600;
        }}
        QGroupBox::title {{
            subcontrol-origin: margin;
            subcontrol-position: top left;
            left: 10px;
            padding: 1px 6px;
            color: {t['text_primary']};
            background: {t['surface']};
        }}
        QGroupBox[collapsible="true"] {{
            background: {t['surface']};
        }}
        QGroupBox[collapsible="true"]::title {{ color: {t['text_secondary']}; }}
        QGroupBox[collapsed="true"] {{
            background: {t['navigation']};
            border-color: {t['border']};
        }}
        QGroupBox[collapsed="true"]::title {{
            color: {t['text_muted']};
            background: {t['navigation']};
        }}
        QLabel {{
            color: {t['text_secondary']};
            background: transparent;
            font-weight: 400;
        }}
        QLabel#helper {{
            color: {t['text_muted']};
            font-size: 12px;
        }}
        QLabel#sectionTitle {{
            color: {t['text_primary']};
            font-size: 13px;
            font-weight: 600;
            padding-top: 8px;
            padding-bottom: 2px;
        }}
        QLabel#viewportHint {{
            color: #555A66;
            background: #F1F3F6;
            font-size: 12px;
        }}
        QLabel[status="success"] {{ color: {t['success']}; }}
        QLabel[status="warning"] {{ color: {t['warning']}; }}
        QLabel[status="error"] {{ color: {t['danger']}; }}
        QLineEdit, QComboBox, QDoubleSpinBox, QSpinBox {{
            background: {t['surface_raised']};
            border: 1px solid {t['border_strong']};
            border-radius: 4px;
            padding: 4px 7px;
            color: {t['text_primary']};
            min-height: 24px;
            selection-background-color: {t['primary']};
            selection-color: #FFFFFF;
        }}
        QLineEdit:hover, QComboBox:hover, QDoubleSpinBox:hover, QSpinBox:hover {{
            border-color: #5A5E6C;
        }}
        QLineEdit:focus, QComboBox:focus, QDoubleSpinBox:focus, QSpinBox:focus {{
            border: 1px solid {t['focus']};
        }}
        QLineEdit:read-only {{
            background: {t['navigation']};
            color: {t['text_muted']};
            border-color: {t['border']};
        }}
        QLineEdit:disabled, QComboBox:disabled,
        QDoubleSpinBox:disabled, QSpinBox:disabled {{
            background: #1B1D23;
            color: #676A75;
            border-color: #2A2D35;
        }}
        QComboBox QAbstractItemView {{
            background: {t['surface_raised']};
            color: {t['text_primary']};
            selection-background-color: {t['primary']};
            selection-color: #FFFFFF;
            border: 1px solid {t['border_strong']};
            outline: 0;
        }}
        QComboBox::drop-down {{
            border: none;
            width: 24px;
        }}
        QPushButton {{
            background: {t['surface_raised']};
            border: 1px solid {t['border_strong']};
            border-radius: 5px;
            padding: 4px 12px;
            min-height: 26px;
            color: {t['text_primary']};
            font-weight: 500;
        }}
        QPushButton:hover {{
            border-color: #626674;
            background: {t['surface_active']};
        }}
        QPushButton:pressed {{
            background: #1C1E24;
            border-color: {t['focus']};
        }}
        QPushButton:focus {{ border: 1px solid {t['focus']}; }}
        QPushButton:checked {{
            color: #FFFFFF;
            background: {t['primary']};
            border-color: {t['primary_hover']};
        }}
        QPushButton:disabled {{
            color: #666975;
            background: #1B1D23;
            border-color: #2A2D35;
        }}
        QPushButton[role="primary"] {{
            color: #FFFFFF;
            background: {t['primary']};
            border-color: {t['primary']};
            font-weight: 600;
        }}
        QPushButton[role="primary"]:hover {{
            background: {t['primary_hover']};
            border-color: {t['primary_hover']};
        }}
        QPushButton[role="primary"]:pressed {{
            background: {t['primary_pressed']};
            border-color: {t['primary_pressed']};
        }}
        QPushButton[role="accent"] {{
            color: #E6FAFF;
            background: #123A45;
            border-color: {t['accent']};
        }}
        QPushButton[role="accent"]:hover {{
            background: #164955;
            border-color: #22A6C3;
        }}
        QPushButton[role="danger"] {{
            color: #FCA5A5;
            background: #351D22;
            border-color: #7F3038;
        }}
        QPushButton[role="danger"]:hover {{
            color: #FFFFFF;
            background: #7F1D1D;
            border-color: {t['danger']};
        }}
        QPushButton[role="toolbar"] {{
            min-width: 42px;
            min-height: 24px;
            padding: 3px 9px;
            background: {t['navigation']};
        }}
        QCheckBox, QRadioButton {{
            spacing: 8px;
            color: {t['text_secondary']};
            background: transparent;
            font-weight: 400;
            min-height: 24px;
        }}
        QCheckBox:disabled, QRadioButton:disabled {{ color: #676A75; }}
        QSlider::groove:horizontal {{
            height: 4px;
            background: {t['border']};
            border-radius: 2px;
        }}
        QSlider::sub-page:horizontal {{
            background: {t['primary']};
            border-radius: 2px;
        }}
        QSlider::handle:horizontal {{
            background: {t['text_primary']};
            border: 2px solid {t['primary']};
            width: 14px;
            margin: -6px 0;
            border-radius: 8px;
        }}
        QProgressBar {{
            border: 1px solid {t['border']};
            border-radius: 4px;
            background: {t['surface_raised']};
            text-align: center;
            color: {t['text_primary']};
            min-height: 22px;
        }}
        QProgressBar::chunk {{
            background: {t['primary']};
            border-radius: 3px;
        }}
        QMenu {{
            background: {t['surface_raised']};
            color: {t['text_primary']};
            border: 1px solid {t['border_strong']};
            padding: 4px;
        }}
        QMenu::item {{ padding: 6px 24px 6px 10px; border-radius: 3px; }}
        QMenu::item:selected {{ background: {t['primary']}; color: #FFFFFF; }}
        QMenu::separator {{ height: 1px; background: {t['border']}; margin: 4px 8px; }}
        QToolTip {{
            color: {t['text_primary']};
            background: #2B2E36;
            border: 1px solid {t['border_strong']};
            padding: 5px 7px;
        }}
        QScrollBar:vertical {{
            background: {t['navigation']};
            width: 10px;
            margin: 0;
        }}
        QScrollBar::handle:vertical {{
            background: #4B4E5A;
            min-height: 36px;
            border-radius: 4px;
        }}
        QScrollBar::handle:vertical:hover {{ background: #626674; }}
        QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
        QScrollBar:horizontal {{
            background: {t['navigation']};
            height: 10px;
            margin: 0;
        }}
        QScrollBar::handle:horizontal {{
            background: #4B4E5A;
            min-width: 36px;
            border-radius: 4px;
        }}
        QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{ width: 0; }}
        QFrame#viewportToolbar {{
            background: {t['navigation']};
            border: 1px solid {t['border']};
            border-radius: 5px;
        }}
        QWidget#viewerPanel {{
            background: #F1F3F6;
            border: 1px solid {t['viewport_border']};
            border-radius: 5px;
        }}
    """








MATERIAL_PRESETS: dict[str, tuple[float, float, float]] = {
    "CAD 银白": (0.68, 0.70, 0.74),
    "银灰": (0.48, 0.52, 0.58),
    "白色": (0.78, 0.79, 0.80),
    "黑色": (0.10, 0.11, 0.14),
    "橙色": (0.72, 0.34, 0.08),
    "蓝色": (0.14, 0.31, 0.62),
    "青色": (0.08, 0.50, 0.58),
    "绿色": (0.13, 0.47, 0.24),
}

BACKGROUND_PRESETS: dict[str, str | tuple[str, str]] = {
    "CAD 亮白": ("#F1F3F6", "#FFFFFF"),
    "纯白": "#FFFFFF",
    "深灰": "#292B2F",
    "中灰": "#5A5A5A",
    "浅灰": "#8A8A8A",
    "黑色": "#1A1A1A",
}

PBR_MATERIAL_PARAMETERS: dict[str, dict[str, float]] = {
    "domain": {"metallic": 0.0, "roughness": 0.60},
    "G": {"metallic": 0.0, "roughness": 0.56},
    "D": {"metallic": 0.0, "roughness": 0.58},
    "IWP": {"metallic": 0.0, "roughness": 0.57},
    "Primitive": {"metallic": 0.0, "roughness": 0.56},
    "Neovius": {"metallic": 0.0, "roughness": 0.59},
    "Custom": {"metallic": 0.0, "roughness": 0.55},
    "Transition": {"metallic": 0.0, "roughness": 0.54},
    "Shell": {"metallic": 0.0, "roughness": 0.48},
    "ShellUnion": {"metallic": 0.0, "roughness": 0.50},
}

SHELL_RESULT_KEY = "Shell"
SHELL_UNION_RESULT_KEY = "ShellUnion"


def _configure_console_encoding() -> None:
    """Avoid the existing package's Unicode status messages breaking imports on Windows."""

    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure") and getattr(stream, "encoding", "").lower() not in {"utf-8", "utf8"}:
            stream.reconfigure(encoding="utf-8", errors="replace")


_configure_console_encoding()


def _default_sole_path(project_root: Path) -> Path:
    """Return the repository's canonical example design domain."""

    preferred = (
        project_root
        / "resources"
        / "examples"
        / "design_domains"
        / "sole-1.stl"
    )
    if preferred.is_file():
        return preferred

    paths = sorted(
        (project_root / "resources").glob("**/*.stl"),
        key=lambda item: item.stat().st_size,
    )
    for path in paths:
        try:
            # Discovery only needs extents. Full merge/repair processing can
            # take minutes for an unrelated high-face-count 1.stl candidate.
            mesh = trimesh.load(path, force="mesh", process=False)
            extents = mesh.bounds[1] - mesh.bounds[0]
            if extents[2] < 30.0 and extents[1] > extents[0] * 1.5:
                return path
        except (OSError, ValueError):
            continue
    raise FileNotFoundError("could not find the example sole design domain")


class _GenerationWorker:  # replaced by Qt worker class when the UI is imported
    pass


def _backend_lattice_parameters_payload(
    parameters: LatticeParameters,
    resolve_path: Callable[[Path], Path],
) -> dict[str, object]:
    """Convert a UI parameter object into the backend's JSON-only DTO."""

    payload = asdict(parameters)
    if isinstance(parameters, CustomUnitCellParameters):
        payload["source_path"] = str(resolve_path(parameters.source_path))
    return payload


def _build_collapsible_group_box_class(QtCore, QtWidgets):
    """Build the progressive-disclosure group without importing Qt globally."""

    class CollapsibleGroupBox(QtWidgets.QGroupBox):
        """A QGroupBox whose title toggles content without changing its state."""

        _COLLAPSED_HEIGHT = 40

        def __init__(self, title: str, *, collapsed: bool = False):
            super().__init__(title)
            self._base_title = str(title)
            self._initial_collapsed = bool(collapsed)
            self._expanded = True
            self._saved_visibility: dict[QWidget, bool] = {}
            self._expanded_minimum_height = 0
            self._expanded_maximum_height = 16777215
            self.setProperty("collapsible", True)
            self.setFocusPolicy(QtCore.Qt.StrongFocus)
            self.setToolTip("单击标题可展开或收起")

        def finalize_initial_state(self) -> None:
            """Apply the requested startup state after child controls exist."""

            self._expanded_minimum_height = self.minimumHeight()
            self._expanded_maximum_height = self.maximumHeight()
            self.set_expanded(not self._initial_collapsed)

        def is_expanded(self) -> bool:
            return self._expanded

        def _content_widgets(self) -> list[QWidget]:
            return self.findChildren(
                QtWidgets.QWidget,
                options=QtCore.Qt.FindDirectChildrenOnly,
            )

        def _refresh_dynamic_style(self) -> None:
            self.style().unpolish(self)
            self.style().polish(self)
            self.updateGeometry()
            self.update()

        def set_expanded(self, expanded: bool) -> None:
            expanded = bool(expanded)
            if expanded == self._expanded and not self._initial_collapsed:
                self.setTitle(f"▾  {self._base_title}")
                return
            if expanded:
                self.setMinimumHeight(self._expanded_minimum_height)
                self.setMaximumHeight(self._expanded_maximum_height)
                for widget, was_visible in self._saved_visibility.items():
                    widget.setVisible(was_visible)
                self._saved_visibility.clear()
            else:
                self._saved_visibility = {
                    widget: not widget.isHidden()
                    for widget in self._content_widgets()
                }
                for widget in self._saved_visibility:
                    widget.hide()
                self.setMinimumHeight(self._COLLAPSED_HEIGHT)
                self.setMaximumHeight(self._COLLAPSED_HEIGHT)
            self._expanded = expanded
            self._initial_collapsed = False
            self.setProperty("collapsed", not expanded)
            self.setTitle(f"{'▾' if expanded else '▸'}  {self._base_title}")
            self._refresh_dynamic_style()

        def mouseReleaseEvent(self, event) -> None:
            title_height = max(28, self.fontMetrics().height() + 12)
            if (
                event.button() == QtCore.Qt.LeftButton
                and event.pos().y() <= title_height
            ):
                self.set_expanded(not self._expanded)
                event.accept()
                return
            super().mouseReleaseEvent(event)

        def keyPressEvent(self, event) -> None:
            if event.key() in (QtCore.Qt.Key_Space, QtCore.Qt.Key_Return):
                self.set_expanded(not self._expanded)
                event.accept()
                return
            super().keyPressEvent(event)

    return CollapsibleGroupBox


def _build_qt_app():
    """Create the GUI class lazily so the numerical API stays importable."""

    from PyQt5 import QtCore, QtGui, QtWidgets
    from lattice_studio.presentation.qt.viewers.pyvista_viewer import ContourData, FieldPlaneRenderData, MeshData
    from lattice_studio.presentation.qt.tools.interactive_section_viewer import (
        InteractiveSectionViewer,
        vtk_bounds_from_mesh,
    )

    _CollapsibleGroupBox = _build_collapsible_group_box_class(QtCore, QtWidgets)

    def raise_if_worker_interrupted() -> None:
        if QtCore.QThread.currentThread().isInterruptionRequested():
            raise RuntimeError("后台计算已取消")

    class _ParameterWheelGuard(QtCore.QObject):
        """Keep page scrolling from silently changing parameter controls."""

        _CONTROL_TYPES = (
            QtWidgets.QAbstractSpinBox,
            QtWidgets.QComboBox,
            QtWidgets.QSlider,
        )

        def install_on(self, root: QtWidgets.QWidget) -> None:
            controls = (
                root.findChildren(QtWidgets.QAbstractSpinBox)
                + root.findChildren(QtWidgets.QComboBox)
                + root.findChildren(QtWidgets.QSlider)
            )
            for control in controls:
                control.installEventFilter(self)
                editor = (
                    control.lineEdit()
                    if isinstance(
                        control,
                        (QtWidgets.QAbstractSpinBox, QtWidgets.QComboBox),
                    )
                    else None
                )
                if editor is not None:
                    editor.installEventFilter(self)

        @classmethod
        def _parameter_control_for(cls, widget: QtCore.QObject):
            current = widget
            while isinstance(current, QtWidgets.QWidget):
                if isinstance(current, cls._CONTROL_TYPES):
                    return current
                current = current.parentWidget()
            return None

        @staticmethod
        def _scroll_area_for(widget: QtWidgets.QWidget):
            current = widget.parentWidget()
            while current is not None:
                if isinstance(current, QtWidgets.QAbstractScrollArea):
                    return current
                current = current.parentWidget()
            return None

        def eventFilter(self, watched, event):
            if event.type() != QtCore.QEvent.Wheel:
                return super().eventFilter(watched, event)
            control = self._parameter_control_for(watched)
            if control is None:
                return super().eventFilter(watched, event)
            scroll_area = self._scroll_area_for(control)
            if scroll_area is not None:
                scroll_bar = scroll_area.verticalScrollBar()
                pixel_delta = event.pixelDelta().y()
                if pixel_delta:
                    distance = int(pixel_delta)
                else:
                    wheel_steps = event.angleDelta().y() / 120.0
                    distance = int(
                        round(
                            wheel_steps
                            * QtWidgets.QApplication.wheelScrollLines()
                            * max(scroll_bar.singleStep(), 1)
                        )
                    )
                scroll_bar.setValue(scroll_bar.value() - distance)
            event.accept()
            return True

    class GenerationWorker(QtCore.QObject):
        progress = QtCore.pyqtSignal(str, float)
        finished = QtCore.pyqtSignal(str, int, object)
        failed = QtCore.pyqtSignal(str, int, str)

        def __init__(
            self,
            mesh: trimesh.Trimesh,
            jobs: dict[str, LatticeParameters],
            sampling: SamplingParameters,
            transition_job: tuple[LatticeParameters, LatticeParameters, TransitionParameters] | tuple[LatticeParameters, LatticeParameters, TransitionParameters, ImplicitBody] | None = None,
            display_voxel_size_mm: float | None = None,
            display_memory_budget_mb: float = 512.0,
            prepared_custom_cell: PreparedCustomUnitCell | None = None,
            display_batch_count: int | None = None,
            design_domain: AnalyticDesignDomain | None = None,
            design_identifier: str = "",
            design_revision: int = 0,
        ):
            super().__init__()
            self.mesh = mesh
            self.jobs = jobs
            self.sampling = sampling
            self.transition_job = transition_job
            self.display_voxel_size_mm = display_voxel_size_mm
            self.display_memory_budget_mb = display_memory_budget_mb
            self.prepared_custom_cell = prepared_custom_cell
            self.display_batch_count = display_batch_count
            self.design_domain = design_domain
            self.design_identifier = design_identifier
            self.design_revision = int(design_revision)

        @QtCore.pyqtSlot()
        def run(self):
            try:
                result = {}
                items = list(self.jobs.items())
                total = len(items) + (1 if self.transition_job is not None else 0) + 1
                visible_field_count = max(total, 1)
                candidate_parameters = [parameters for _kind, parameters in items]
                if self.transition_job is not None:
                    candidate_parameters.extend(self.transition_job[:2])
                if self.design_domain is None:
                    recommendations = [
                        lattice_sampling_recommendation(
                            self.mesh,
                            parameters,
                            self.sampling,
                        )
                        for parameters in candidate_parameters
                    ]
                else:
                    recommendations = [
                        _domain_sampling_recommendation(
                            self.design_domain,
                            parameters,
                            self.sampling,
                        )
                        for parameters in candidate_parameters
                    ]
                base_recommendation = min(
                    recommendations,
                    key=lambda item: item.voxel_size_mm,
                )
                def domain_callback(message: str, value: float):
                    raise_if_worker_interrupted()
                    self.progress.emit(message, value / total)

                if self.design_domain is None:
                    domain_field = generate_implicit_domain_preview(
                        self.mesh,
                        self.sampling,
                        base_recommendation,
                        display_voxel_size_mm=self.display_voxel_size_mm,
                        display_memory_budget_mb=self.display_memory_budget_mb,
                        visible_field_count=visible_field_count,
                        display_batch_count=self.display_batch_count,
                        progress=domain_callback,
                    )
                else:
                    domain_field = generate_implicit_domain_preview_for_design_domain(
                        self.design_domain,
                        self.sampling,
                        base_recommendation,
                        display_voxel_size_mm=self.display_voxel_size_mm,
                        display_memory_budget_mb=self.display_memory_budget_mb,
                        visible_field_count=visible_field_count,
                        display_batch_count=self.display_batch_count,
                        progress=domain_callback,
                    )
                result["domain"] = (domain_field, base_recommendation)
                for index, (kind, params) in enumerate(items):
                    raise_if_worker_interrupted()

                    def callback(message: str, value: float, offset=index + 1, total=total):
                        raise_if_worker_interrupted()
                        self.progress.emit(message, (offset + value) / total)

                    common_arguments = {
                        "display_voxel_size_mm": self.display_voxel_size_mm,
                        "display_memory_budget_mb": self.display_memory_budget_mb,
                        "visible_field_count": visible_field_count,
                        "display_batch_count": self.display_batch_count,
                        "domain_display_field": domain_field,
                        "progress": callback,
                    }
                    if isinstance(params, CustomUnitCellParameters) and self.design_domain is None:
                        lattice = generate_implicit_custom_lattice(
                            self.mesh,
                            params,
                            self.sampling,
                            prepared_unit_cell=self.prepared_custom_cell,
                            **common_arguments,
                        )
                    elif isinstance(params, CustomUnitCellParameters):
                        lattice = generate_implicit_custom_lattice_for_design_domain(
                            self.design_domain,
                            params,
                            self.sampling,
                            prepared_unit_cell=self.prepared_custom_cell,
                            **common_arguments,
                        )
                    elif self.design_domain is None:
                        lattice = generate_implicit_lattice(
                            self.mesh,
                            params,
                            self.sampling,
                            **common_arguments,
                        )
                    else:
                        lattice = generate_implicit_lattice_for_design_domain(
                            self.design_domain,
                            params,
                            self.sampling,
                            **common_arguments,
                        )
                    result[kind] = (lattice, lattice.recommendation)
                if self.transition_job is not None:
                    first_parameters, second_parameters, transition = self.transition_job[:3]
                    driver = self.transition_job[3] if len(self.transition_job) == 4 else None

                    def transition_callback(message: str, value: float, offset=len(items) + 1, total=total):
                        raise_if_worker_interrupted()
                        self.progress.emit(message, (offset + value) / total)

                    transition_arguments = {
                        "display_voxel_size_mm": self.display_voxel_size_mm,
                        "display_memory_budget_mb": self.display_memory_budget_mb,
                        "visible_field_count": visible_field_count,
                        "display_batch_count": self.display_batch_count,
                        "domain_display_field": domain_field,
                        "progress": transition_callback,
                        "prepared_custom_cell": self.prepared_custom_cell,
                        "transition_driver": driver,
                    }
                    if self.design_domain is None:
                        lattice = generate_implicit_lattice_transition(
                            self.mesh,
                            first_parameters,
                            second_parameters,
                            transition,
                            self.sampling,
                            **transition_arguments,
                        )
                    else:
                        lattice = generate_implicit_lattice_transition_for_design_domain(
                            self.design_domain,
                            first_parameters,
                            second_parameters,
                            transition,
                            self.sampling,
                            **transition_arguments,
                        )
                    result["Transition"] = (lattice, lattice.recommendation)
                raise_if_worker_interrupted()
                self.finished.emit(
                    self.design_identifier,
                    self.design_revision,
                    result,
                )
            except Exception as exc:  # surface the original cause in the UI
                self.failed.emit(
                    self.design_identifier,
                    self.design_revision,
                    f"{type(exc).__name__}: {exc}",
                )

    class BackendTaskWorker(QtCore.QObject):
        """Poll one loopback task outside the Qt event thread.

        The worker owns no numerical objects.  A task result reader may only
        deserialize downloadable artifacts into frontend-safe display data.
        """

        progress = QtCore.pyqtSignal(str, float)
        finished = QtCore.pyqtSignal(object)
        failed = QtCore.pyqtSignal(str)

        def __init__(
            self,
            client,
            task_kind: str,
            payload: dict[str, object],
            result_reader: Callable[[object], object],
        ) -> None:
            super().__init__()
            self.client = client
            self.task_kind = str(task_kind)
            self.payload = dict(payload)
            self.result_reader = result_reader

        @QtCore.pyqtSlot()
        def run(self) -> None:
            try:
                task = self.client.submit_task(self.task_kind, self.payload)
                cancellation_requested = False
                while task.state not in {"succeeded", "failed", "cancelled"}:
                    if QtCore.QThread.currentThread().isInterruptionRequested():
                        if not cancellation_requested:
                            task = self.client.cancel_task(task.identifier)
                            cancellation_requested = True
                        time.sleep(0.05)
                        continue
                    self.progress.emit(task.message, task.progress)
                    time.sleep(0.08)
                    task = self.client.get_task(task.identifier)
                self.progress.emit(task.message, task.progress)
                if task.state == "succeeded":
                    self.finished.emit(self.result_reader(task))
                    return
                message = task.error or task.message or "backend task did not complete"
                self.failed.emit(message)
            except Exception as exc:
                self.failed.emit(f"{type(exc).__name__}: {exc}")

    @dataclass(frozen=True)
    class BackendStlReconstructionResult:
        """A mesh artifact decoded from a backend-owned reconstruction task."""

        kind: str
        mesh: trimesh.Trimesh
        triangle_count: int

    class DisplayRefinementWorker(QtCore.QObject):
        """Build quality-specific sampled fields without blocking the UI thread."""

        progress = QtCore.pyqtSignal(str, float)
        finished = QtCore.pyqtSignal(str, int, str, object)
        failed = QtCore.pyqtSignal(str, int, str, str)

        def __init__(
            self,
            targets: dict[
                object,
                tuple[ImplicitBody, SampledImplicitField, float],
            ],
            *,
            quality: str,
            display_batch_count: int | None,
            design_identifier: str,
            design_revision: int,
        ):
            super().__init__()
            self.targets = dict(targets)
            if quality not in DISPLAY_FIELD_REFINEMENT_FACTORS:
                raise ValueError("display refinement requires High or Ultra quality")
            self.quality = quality
            self.display_batch_count = display_batch_count
            self.design_identifier = design_identifier
            self.design_revision = int(design_revision)

        @QtCore.pyqtSlot()
        def run(self) -> None:
            try:
                results = {}
                total = max(len(self.targets), 1)
                for index, (cache_key, (body, source, field_budget_mb)) in enumerate(
                    self.targets.items()
                ):
                    def callback(
                        message: str,
                        value: float,
                        offset: int = index,
                    ) -> None:
                        if QtCore.QThread.currentThread().isInterruptionRequested():
                            raise RuntimeError("display refinement cancelled")
                        self.progress.emit(message, (offset + value) / total)

                    results[cache_key] = refine_implicit_display_field(
                        body,
                        source,
                        quality=self.quality,
                        display_memory_budget_mb=field_budget_mb,
                        display_batch_count=self.display_batch_count,
                        progress=callback,
                    )
                raise_if_worker_interrupted()
                self.finished.emit(
                    self.design_identifier,
                    self.design_revision,
                    self.quality,
                    results,
                )
            except Exception as exc:
                self.failed.emit(
                    self.design_identifier,
                    self.design_revision,
                    self.quality,
                    f"{type(exc).__name__}: {exc}",
                )

    class PreciseRenderWorker(QtCore.QObject):
        """Render one authoritative implicit object without touching viewport caches."""

        progress = QtCore.pyqtSignal(str, float)
        finished = QtCore.pyqtSignal(str, int, str, str, object, object)
        failed = QtCore.pyqtSignal(str, int, str)

        def __init__(
            self,
            target: PreciseRenderTarget,
            *,
            target_key: str,
            camera: PreciseRenderCamera,
            settings: PreciseRenderSettings,
            material: PreciseRenderMaterial,
            background_lower: tuple[float, float, float],
            background_upper: tuple[float, float, float],
            output_path: str,
            design_identifier: str,
            design_revision: int,
        ):
            super().__init__()
            self.target = target
            self.target_key = str(target_key)
            self.camera = camera
            self.settings = settings
            self.material = material
            self.background_lower = background_lower
            self.background_upper = background_upper
            self.output_path = str(output_path)
            self.design_identifier = design_identifier
            self.design_revision = int(design_revision)

        @QtCore.pyqtSlot()
        def run(self) -> None:
            try:
                def callback(message: str, value: float) -> None:
                    if QtCore.QThread.currentThread().isInterruptionRequested():
                        raise RuntimeError("精确渲染已取消")
                    self.progress.emit(message, value)

                result = render_precise_implicit(
                    self.target.body,
                    camera=self.camera,
                    settings=self.settings,
                    material=self.material,
                    background_lower=self.background_lower,
                    background_upper=self.background_upper,
                    progress=callback,
                )
                raise_if_worker_interrupted()
                self.finished.emit(
                    self.design_identifier,
                    self.design_revision,
                    self.target_key,
                    self.output_path,
                    result,
                    self._cache_key(),
                )
            except Exception as exc:
                self.failed.emit(
                    self.design_identifier,
                    self.design_revision,
                    f"{type(exc).__name__}: {exc}",
                )

        def _cache_key(self) -> tuple[object, ...]:
            return (
                self.target_key,
                self.design_revision,
                self.camera,
                self.settings,
                self.material,
                self.background_lower,
                self.background_upper,
            )

    class StlReconstructionWorker(QtCore.QObject):
        progress = QtCore.pyqtSignal(str, float)
        finished = QtCore.pyqtSignal(object)
        failed = QtCore.pyqtSignal(str)

        def __init__(
            self,
            reconstruction_targets: dict[str, StlReconstructionTarget],
            tolerance_mm: float,
            spacing_mode: ExportSpacingMode,
            repair_tolerance_mm: float,
            clean_numerical_fragments: bool,
            processing_mode: ProcessingMode = "single_pass",
            batch_count: int = 1,
            optimize_for_slicing: bool = True,
        ):
            super().__init__()
            self.reconstruction_targets = reconstruction_targets
            self.tolerance_mm = float(tolerance_mm)
            self.spacing_mode = spacing_mode
            self.repair_tolerance_mm = float(repair_tolerance_mm)
            self.clean_numerical_fragments = bool(clean_numerical_fragments)
            self.processing_mode = processing_mode
            self.batch_count = int(batch_count)
            self.optimize_for_slicing = bool(optimize_for_slicing)

        @QtCore.pyqtSlot()
        def run(self):
            try:
                outputs = {}
                items = list(self.reconstruction_targets.items())
                total = max(len(items), 1)
                for index, (kind, target) in enumerate(items):
                    raise_if_worker_interrupted()

                    def callback(message: str, value: float, offset=index):
                        raise_if_worker_interrupted()
                        self.progress.emit(
                            f"{kind}：{message}",
                            (offset + value) / total,
                        )

                    outputs[kind] = reconstruct_implicit_mesh(
                        target.body,
                        target.extraction_map,
                        self.tolerance_mm,
                        target.minimum_feature_mm,
                        repair_tolerance_mm=self.repair_tolerance_mm,
                        clean_numerical_fragments=self.clean_numerical_fragments,
                        progress=callback,
                        processing_mode=self.processing_mode,
                        batch_count=self.batch_count,
                        spacing_mode=self.spacing_mode,
                        optimize_for_slicing=self.optimize_for_slicing,
                        enforce_feature_limits=target.enforce_feature_limits,
                    )
                raise_if_worker_interrupted()
                self.finished.emit(outputs)
            except Exception as exc:
                self.failed.emit(f"{type(exc).__name__}: {exc}")

    class ShellGenerationWorker(QtCore.QObject):
        """Build a shell-derived implicit result without blocking the Qt thread."""

        progress = QtCore.pyqtSignal(str, float)
        finished = QtCore.pyqtSignal(str, object)
        failed = QtCore.pyqtSignal(str)

        def __init__(
            self,
            mesh: trimesh.Trimesh,
            sampling: SamplingParameters,
            thickness_mm: float,
            fusion_radius_mm: float,
            lattice_generation: ImplicitGenerationResult | None,
            domain_display_field: SampledImplicitField | None,
            display_voxel_size_mm: float | None,
            display_memory_budget_mb: float,
            display_batch_count: int | None,
            visible_field_count: int,
        ):
            super().__init__()
            self.mesh = mesh
            self.sampling = sampling
            self.thickness_mm = float(thickness_mm)
            self.fusion_radius_mm = float(fusion_radius_mm)
            self.lattice_generation = lattice_generation
            self.domain_display_field = domain_display_field
            self.display_voxel_size_mm = display_voxel_size_mm
            self.display_memory_budget_mb = float(display_memory_budget_mb)
            self.display_batch_count = display_batch_count
            self.visible_field_count = int(visible_field_count)

        @QtCore.pyqtSlot()
        def run(self) -> None:
            try:
                def callback(message: str, value: float) -> None:
                    raise_if_worker_interrupted()
                    self.progress.emit(message, value)

                if self.lattice_generation is None:
                    result = generate_implicit_shell(
                        self.mesh,
                        self.thickness_mm,
                        self.sampling,
                        display_voxel_size_mm=self.display_voxel_size_mm,
                        display_memory_budget_mb=self.display_memory_budget_mb,
                        visible_field_count=self.visible_field_count,
                        display_batch_count=self.display_batch_count,
                        progress=callback,
                    )
                    raise_if_worker_interrupted()
                    self.finished.emit(SHELL_RESULT_KEY, result)
                    return
                result = generate_implicit_shell_lattice_union(
                    self.mesh,
                    self.lattice_generation,
                    self.thickness_mm,
                    self.fusion_radius_mm,
                    self.sampling,
                    display_voxel_size_mm=self.display_voxel_size_mm,
                    display_memory_budget_mb=self.display_memory_budget_mb,
                    visible_field_count=self.visible_field_count,
                    display_batch_count=self.display_batch_count,
                    domain_display_field=self.domain_display_field,
                    progress=callback,
                )
                raise_if_worker_interrupted()
                self.finished.emit(SHELL_UNION_RESULT_KEY, result)
            except Exception as exc:
                self.failed.emit(f"{type(exc).__name__}: {exc}")

    class TPMSWindow(QtWidgets.QMainWindow):
        def __init__(self):
            super().__init__()
            self.setWindowTitle("晶格填充工作台 · TPMS / 自定义晶胞")
            self.resize(1480, 900)
            self.setMinimumSize(1050, 700)
            self.root = PROJECT_ROOT
            self.backend_process = None
            self.backend_client = None
            self.backend_workspace_id: str | None = None
            self._backend_generation_input_root: Path | None = None
            self._backend_shell_input_root: Path | None = None
            self.design_workspace = DesignWorkspace()
            self._workspace_manifest_path: Path | None = None
            self._workspace_loading = False
            self.sole_mesh = trimesh.Trimesh(process=False)
            self.results: dict[str, trimesh.Trimesh] = {}
            self.raw_results: dict[str, trimesh.Trimesh] = {}
            self.repaired_results: dict[str, trimesh.Trimesh] = {}
            self.implicit_results: dict[str, ImplicitBody] = {}
            self.implicit_generation_results: dict[str, ImplicitGenerationResult] = {}
            self.backend_generation_handles: dict[str, BackendGenerationResult] = {}
            self.implicit_fields: dict[str, SampledImplicitField] = {}
            self.domain_implicit_field: SampledImplicitField | None = None
            self._shell_fusion_domain_field: SampledImplicitField | None = None
            self.stl_reconstruction_results: dict[str, ExportMeshResult] = {}
            self._layer_contour_cache = LayerContourCache()
            self._field_plane_cache = FieldPlaneCache()
            self._field_plane_state: FieldPlaneState | None = None
            self._field_plane_refresh_timer = QtCore.QTimer(self)
            self._field_plane_refresh_timer.setSingleShot(True)
            self._field_plane_refresh_timer.setInterval(120)
            self._field_plane_refresh_timer.timeout.connect(
                self._refresh_field_viewer_plane
            )
            self._analytic_domain_preview_timer = QtCore.QTimer(self)
            self._analytic_domain_preview_timer.setSingleShot(True)
            self._analytic_domain_preview_timer.setInterval(160)
            self._analytic_domain_preview_timer.timeout.connect(
                self._flush_pending_analytic_domain_preview
            )
            self._pending_analytic_domain_preview: str | None = None
            self.domain_implicit_body: ImplicitBody | None = None
            self.field_primitives: dict[str, ImplicitPrimitive] = {}
            self.field_primitive_visibility: dict[str, bool] = {}
            self._next_primitive_index = 1
            self._selected_primitive_identifier: str | None = None
            self._primitive_preview_signature: tuple[object, ...] | None = None
            self._primitive_preview_mesh_cache: dict[object, tuple[np.ndarray, np.ndarray]] = {}
            self._primitive_gizmo_sync_in_progress = False
            self._cell_map_gizmo_sync_in_progress = False
            self.material_colors = {
                "Transition": MATERIAL_PRESETS["CAD 银白"],
                "domain": MATERIAL_PRESETS["CAD 银白"],
                "G": MATERIAL_PRESETS["CAD 银白"],
                "D": MATERIAL_PRESETS["CAD 银白"],
                "IWP": MATERIAL_PRESETS["CAD 银白"],
                "Primitive": MATERIAL_PRESETS["CAD 银白"],
                "Neovius": MATERIAL_PRESETS["CAD 银白"],
                "Custom": MATERIAL_PRESETS["CAD 银白"],
                "Shell": MATERIAL_PRESETS["蓝色"],
                "ShellUnion": MATERIAL_PRESETS["青色"],
            }
            self.prepared_custom_cell: PreparedCustomUnitCell | None = None
            self.thread = None
            self.worker = None
            self.shell_thread = None
            self.shell_worker = None
            self.stl_thread = None
            self.stl_worker = None
            self.display_refinement_thread = None
            self.display_refinement_worker = None
            self._display_refinement_refresh_pending = False
            self.precise_render_thread = None
            self.precise_render_worker = None
            self._is_closing = False
            self._close_pending = False
            self._close_retry_timer = QtCore.QTimer(self)
            self._close_retry_timer.setInterval(100)
            self._close_retry_timer.timeout.connect(self._retry_pending_close)
            self._build_ui()
            self._parameter_wheel_guard = _ParameterWheelGuard(self)
            self._parameter_wheel_guard.install_on(self)
            # Capture the fully constructed UI once. New Design documents
            # must start from this baseline rather than inheriting values from
            # whichever document happened to be active at creation time.
            self._default_workspace_control_state = self._workspace_control_snapshot()
            self._default_tpms_parameter_states = {
                kind: dict(state)
                for kind, state in self._tpms_parameter_states.items()
            }
            self._initialize_empty_workspace()
            self._start_local_backend()

        def _start_local_backend(self) -> None:
            """Own a private backend process without coupling Qt to its transport."""

            from lattice_studio.presentation.http.backend_process import (
                BackendProcessError,
                LocalBackendProcess,
            )

            try:
                backend_process = LocalBackendProcess()
                self.backend_client = backend_process.start()
                self.backend_process = backend_process
                self.backend_workspace_id = self.backend_client.create_workspace().identifier
            except BackendProcessError as exc:
                self.backend_process = None
                self.backend_client = None
                self.backend_workspace_id = None
                self.status.setText(f"local backend unavailable: {exc}")

        def _backend_apply_workspace_commands(
            self,
            commands: list[dict[str, object]],
        ) -> bool:
            """Commit a UI document mutation before changing its local projection."""

            client = self.backend_client
            workspace_id = self.backend_workspace_id
            if client is None or workspace_id is None:
                return True
            try:
                client.apply_workspace_commands(workspace_id, commands)
            except Exception as exc:
                if (
                    "unknown active design" in str(exc)
                    or "unknown design" in str(exc)
                ) and self._bootstrap_backend_workspace_projection():
                    try:
                        client.apply_workspace_commands(workspace_id, commands)
                    except Exception as retry_exc:
                        self.status.setText(
                            f"local backend workspace update failed: {retry_exc}"
                        )
                        return False
                    return True
                self.status.setText(f"local backend workspace update failed: {exc}")
                return False
            return True

        @staticmethod
        def _backend_field_scene_payload(
            primitives: dict[str, ImplicitPrimitive],
            visibility: dict[str, bool],
        ) -> list[dict[str, object]]:
            return [
                {"primitive": asdict(primitive), "visible": bool(visibility[identifier])}
                for identifier, primitive in primitives.items()
            ]

        def _sync_workspace_projection_to_backend(self) -> bool:
            """Publish editable definitions and client-created mesh records before save."""

            if self.backend_client is None or self.backend_workspace_id is None:
                return True
            commands: list[dict[str, object]] = []
            documents = tuple(self.design_workspace.documents.values()) + tuple(
                self.design_workspace.archived_documents.values()
            )
            for document in documents:
                if isinstance(document.domain, MeshDesignDomain):
                    source_path = document.domain.asset_path
                    if source_path is None:
                        raise ValueError("mesh design domain requires an STL source path")
                    commands.append(
                        {
                            "kind": "document.replace.mesh",
                            "document_id": document.identifier,
                            "domain_name": document.domain.name,
                            "source_path": str(
                                self._workspace_asset_source_path(source_path).resolve()
                            ),
                        }
                    )
                else:
                    commands.append(
                        {
                            "kind": "document.replace.analytic",
                            "document_id": document.identifier,
                            "domain_name": document.domain.name,
                            "primitive": asdict(document.domain.primitive),
                        }
                    )
                commands.extend(
                    (
                        {
                            "kind": "document.settings.replace",
                            "document_id": document.identifier,
                            "settings": document.settings,
                        },
                        {
                            "kind": "document.field_scene.replace",
                            "document_id": document.identifier,
                            "field_primitives": self._backend_field_scene_payload(
                                document.field_primitives,
                                document.field_primitive_visibility,
                            ),
                        },
                        {
                            "kind": "document.derived_meshes.replace",
                            "document_id": document.identifier,
                            "derived_meshes": [
                                {
                                    "identifier": record.identifier,
                                    "asset_path": record.asset_path,
                                    "provenance": record.provenance,
                                }
                                for record in document.derived_meshes.values()
                            ],
                        },
                    )
                )
            return self._backend_apply_workspace_commands(commands)

        def _bootstrap_backend_workspace_projection(self) -> bool:
            """Register a legacy local projection in an empty backend session once."""

            client = self.backend_client
            workspace_id = self.backend_workspace_id
            if client is None or workspace_id is None:
                return False
            try:
                snapshot = client.get_workspace_snapshot(workspace_id)
                existing = {
                    str(item["document_id"])
                    for key in ("documents", "archived_documents")
                    for item in snapshot.get(key, [])
                    if isinstance(item, dict) and "document_id" in item
                }
                commands: list[dict[str, object]] = []
                active_documents = tuple(self.design_workspace.documents.values())
                archived_documents = tuple(self.design_workspace.archived_documents.values())
                for document in active_documents + archived_documents:
                    if document.identifier not in existing:
                        if isinstance(document.domain, MeshDesignDomain):
                            if document.domain.asset_path is None:
                                return False
                            command = {
                                "kind": "document.create.mesh",
                                "document_id": document.identifier,
                                "name": document.name,
                                "domain_name": document.domain.name,
                                "source_path": str(
                                    self._workspace_asset_source_path(
                                        document.domain.asset_path
                                    ).resolve()
                                ),
                                "activate": False,
                            }
                        else:
                            command = {
                                "kind": "document.create.analytic",
                                "document_id": document.identifier,
                                "name": document.name,
                                "domain_name": document.domain.name,
                                "primitive": asdict(document.domain.primitive),
                                "activate": False,
                            }
                        commands.append(command)
                    # The desktop may have created its projection before the
                    # local backend session. Keep existing documents current too.
                    commands.extend(
                        (
                            {
                                "kind": "document.settings.replace",
                                "document_id": document.identifier,
                                "settings": document.settings,
                            },
                            {
                                "kind": "document.field_scene.replace",
                                "document_id": document.identifier,
                                "field_primitives": self._backend_field_scene_payload(
                                    document.field_primitives,
                                    document.field_primitive_visibility,
                                ),
                            },
                        )
                    )
                commands.extend(
                    {
                        "kind": "document.archive",
                        "document_id": document.identifier,
                    }
                    for document in archived_documents
                    if document.identifier not in existing
                )
                active_identifier = self.design_workspace.active_design_id
                if active_identifier is not None:
                    commands.append(
                        {
                            "kind": "document.activate",
                            "document_id": active_identifier,
                        }
                    )
                if commands:
                    client.apply_workspace_commands(workspace_id, commands)
                return True
            except Exception:
                return False

        def _initialize_empty_workspace(self) -> None:
            """Present a clean document until the user creates or imports a design."""

            self.sole_mesh = trimesh.Trimesh(process=False)
            self._primitive_preview_mesh_cache.clear()
            self._primitive_preview_signature = None
            if hasattr(self, "_default_workspace_control_state"):
                previous_loading = self._workspace_loading
                self._workspace_loading = True
                try:
                    self._tpms_parameter_states = {
                        kind: dict(state)
                        for kind, state in self._default_tpms_parameter_states.items()
                    }
                    self._active_tpms_kind = "G"
                    self._restore_workspace_control_snapshot(
                        self._default_workspace_control_state
                    )
                    kind_index = self.tpms_widgets["kind"].findData("G")
                    self.tpms_widgets["kind"].setCurrentIndex(kind_index)
                    self._restore_tpms_parameter_state("G")
                finally:
                    self._workspace_loading = previous_loading
            self.results = {}
            self.raw_results = {}
            self.repaired_results = {}
            self.implicit_results = {}
            self.implicit_generation_results = {}
            self.backend_generation_handles = {}
            self.implicit_fields = {}
            self.domain_implicit_body = None
            self.domain_implicit_field = None
            self._shell_fusion_domain_field = None
            self.field_primitives = {}
            self.field_primitive_visibility = {}
            self._selected_primitive_identifier = None
            self.prepared_custom_cell = None
            self.path_edit.clear()
            self.domain_info.setText("空工作区：请选择 STL 设计域")
            self.status.setText("空工作区：导入设计域后即可生成晶格")
            self.show_domain.setChecked(False)
            self._set_result_controls_enabled(False)
            self._sync_contour_result_options()
            self._sync_field_viewer_options()
            self._sync_stl_reconstruction_target_options()
            self._sync_section_target_options()
            self._sync_shell_lattice_options()
            self._sync_design_workspace_controls()
            self._refresh_scene(reset_view=True)

        def _register_workspace_controls(self) -> None:
            """Register stable parameter controls that belong to each Design."""

            self._workspace_controls = []
            control_types = (
                QtWidgets.QLineEdit,
                QtWidgets.QCheckBox,
                QtWidgets.QComboBox,
                QtWidgets.QSpinBox,
                QtWidgets.QDoubleSpinBox,
                QtWidgets.QSlider,
            )
            excluded = {
                id(self.path_edit),
                id(self.design_selector),
                id(self.archived_design_selector),
                id(self.field_primitive_combo),
                id(self.transition_driver_combo),
                id(self.field_viewer_target_combo),
                id(self.contour_result_combo),
                id(self.section_target_combo),
                id(self.stl_reconstruction_target_combo),
                id(self.shell_lattice_combo),
            }
            for control_type in control_types:
                for index, control in enumerate(self.findChildren(control_type)):
                    if id(control) in excluded:
                        continue
                    self._workspace_controls.append(
                        (f"{control_type.__name__}:{index}", control)
                    )

        def _workspace_control_snapshot(self) -> dict[str, object]:
            """Capture static parameter widgets without renderer or result state."""

            snapshot: dict[str, object] = {}
            for key, control in self._workspace_controls:
                if isinstance(control, QtWidgets.QLineEdit):
                    snapshot[key] = control.text()
                elif isinstance(control, QtWidgets.QCheckBox):
                    snapshot[key] = control.isChecked()
                elif isinstance(control, QtWidgets.QComboBox):
                    snapshot[key] = control.currentIndex()
                elif isinstance(control, (QtWidgets.QSpinBox, QtWidgets.QDoubleSpinBox)):
                    snapshot[key] = control.value()
                elif isinstance(control, QtWidgets.QSlider):
                    snapshot[key] = control.value()
            return snapshot

        def _restore_workspace_control_snapshot(self, snapshot: object) -> None:
            """Restore the static controls while suppressing dependency signals."""

            if not isinstance(snapshot, dict):
                return
            for key, control in self._workspace_controls:
                if key not in snapshot:
                    continue
                value = snapshot[key]
                control.blockSignals(True)
                try:
                    if isinstance(control, QtWidgets.QLineEdit) and isinstance(value, str):
                        control.setText(value)
                    elif isinstance(control, QtWidgets.QCheckBox):
                        control.setChecked(bool(value))
                    elif isinstance(control, QtWidgets.QComboBox):
                        index = int(value)
                        if -1 <= index < control.count():
                            control.setCurrentIndex(index)
                    elif isinstance(control, QtWidgets.QDoubleSpinBox):
                        control.setValue(float(value))
                    elif isinstance(control, QtWidgets.QSpinBox):
                        control.setValue(int(value))
                    elif isinstance(control, QtWidgets.QSlider):
                        control.setValue(int(value))
                finally:
                    control.blockSignals(False)

        def _store_active_design_state(self) -> None:
            """Move the current window state into the Active Design document."""

            document = self.design_workspace.active_document
            if document is None:
                return
            if self._active_tpms_kind in TPMS_KINDS:
                self._tpms_parameter_states[
                    self._active_tpms_kind
                ] = self._tpms_state_from_widgets()
            control_state = self._workspace_control_snapshot()
            tpms_parameter_states = json.loads(
                json.dumps(
                    {
                        kind: dict(state)
                        for kind, state in self._tpms_parameter_states.items()
                    }
                )
            )
            persistent_state = {
                "ui_control_state": control_state,
                "active_tpms_kind": self._active_tpms_kind,
                "tpms_parameter_states": tpms_parameter_states,
            }
            settings_changed = any(
                document.settings.get(key) != value
                for key, value in persistent_state.items()
            )
            field_scene_changed = (
                document.field_primitives != self.field_primitives
                or document.field_primitive_visibility
                != {
                    key: bool(value)
                    for key, value in self.field_primitive_visibility.items()
                }
            )
            commands = []
            if settings_changed:
                updated_settings = dict(document.settings)
                updated_settings.update(persistent_state)
                commands.append(
                    {
                        "kind": "document.settings.replace",
                        "document_id": document.identifier,
                        "settings": updated_settings,
                    }
                )
            if field_scene_changed:
                commands.append(
                    {
                        "kind": "document.field_scene.replace",
                        "document_id": document.identifier,
                        "field_primitives": self._backend_field_scene_payload(
                            self.field_primitives,
                            self.field_primitive_visibility,
                        ),
                    }
                )
            if commands and not self._backend_apply_workspace_commands(commands):
                return
            if settings_changed:
                document.settings.update(persistent_state)
                document.touch()
                self.design_workspace.dirty = True
            if field_scene_changed:
                document.replace_field_object_scene(
                    self.field_primitives,
                    self.field_primitive_visibility,
                )
                self.design_workspace.dirty = True
            runtime = document.runtime
            runtime.domain_implicit_body = self.domain_implicit_body
            runtime.domain_implicit_field = self.domain_implicit_field
            runtime.shell_fusion_domain_field = self._shell_fusion_domain_field
            runtime.implicit_results = dict(self.implicit_results)
            runtime.implicit_generation_results = dict(
                self.implicit_generation_results
            )
            runtime.backend_generation_handles = dict(self.backend_generation_handles)
            runtime.implicit_fields = dict(self.implicit_fields)
            runtime.raw_results = dict(self.raw_results)
            runtime.repaired_results = dict(self.repaired_results)
            runtime.results = dict(self.results)
            runtime.stl_reconstruction_results = dict(self.stl_reconstruction_results)
            runtime.prepared_custom_cell = self.prepared_custom_cell
            runtime.ui_transient = {
                "active_tpms_kind": self._active_tpms_kind,
                "tpms_parameter_states": {
                    kind: dict(state)
                    for kind, state in self._tpms_parameter_states.items()
                },
                "next_primitive_index": self._next_primitive_index,
                "selected_primitive_identifier": self._selected_primitive_identifier,
            }

        def _new_design_document_settings(self) -> dict[str, object]:
            """Return the persistent baseline shared by every new Design."""

            return {
                "ui_control_state": dict(self._default_workspace_control_state),
                "active_tpms_kind": "G",
                "tpms_parameter_states": json.loads(
                    json.dumps(
                        {
                            kind: dict(state)
                            for kind, state in self._default_tpms_parameter_states.items()
                        }
                    )
                ),
            }

        def _initialize_new_design_document(self, document) -> None:
            """Give a newly created Design its own immutable UI baseline."""

            document.settings.update(self._new_design_document_settings())
            document.runtime.ui_transient = {
                "active_tpms_kind": "G",
                "tpms_parameter_states": {
                    kind: dict(state)
                    for kind, state in self._default_tpms_parameter_states.items()
                },
                "next_primitive_index": 1,
                "selected_primitive_identifier": None,
            }

        def _document_switch_is_locked(self) -> bool:
            """Report whether a legacy document-bound task still owns the UI."""

            operation = None
            if self.precise_render_thread is not None:
                operation = "精确渲染"
            elif self.stl_thread is not None:
                operation = "STL 重建"
            elif self.shell_thread is not None:
                operation = "抽壳或壳体融合"
            if operation is None:
                return False
            self.status.setText(f"正在执行{operation}，完成后才可切换设计。")
            return True

        def _restore_active_design_state(self) -> None:
            """Load the Active Design document into the existing UI shell."""

            document = self.design_workspace.active_document
            if document is None:
                self._initialize_empty_workspace()
                return
            self._primitive_preview_mesh_cache.clear()
            self._primitive_preview_signature = None
            self._workspace_loading = True
            try:
                if isinstance(document.domain, MeshDesignDomain):
                    self.sole_mesh = document.domain.mesh.copy()
                    self.path_edit.setText(str(document.domain.asset_path or ""))
                else:
                    # Analytic domains retain their SDF authority.  The mesh is
                    # only a view proxy until the domain-aware generator is used.
                    self.sole_mesh = document.domain.preview_mesh()
                    self.path_edit.setText("解析设计域")
                runtime = document.runtime
                self.results = dict(runtime.results)
                self.raw_results = dict(runtime.raw_results)
                self.repaired_results = dict(runtime.repaired_results)
                self.implicit_results = dict(runtime.implicit_results)
                self.implicit_generation_results = dict(
                    runtime.implicit_generation_results
                )
                self.backend_generation_handles = dict(
                    runtime.backend_generation_handles
                )
                self.implicit_fields = dict(runtime.implicit_fields)
                self.stl_reconstruction_results = dict(runtime.stl_reconstruction_results)
                self.domain_implicit_body = runtime.domain_implicit_body
                self.domain_implicit_field = runtime.domain_implicit_field
                self._shell_fusion_domain_field = runtime.shell_fusion_domain_field
                self.prepared_custom_cell = runtime.prepared_custom_cell
                self.field_primitives = dict(document.field_primitives)
                self.field_primitive_visibility = dict(
                    document.field_primitive_visibility
                )
                transient = runtime.ui_transient
                saved_states = transient.get(
                    "tpms_parameter_states",
                    document.settings.get("tpms_parameter_states"),
                )
                if isinstance(saved_states, dict) and set(saved_states) == set(TPMS_KINDS):
                    self._tpms_parameter_states = {
                        kind: dict(saved_states[kind])
                        for kind in TPMS_KINDS
                    }
                else:
                    self._tpms_parameter_states = {
                        kind: dict(state)
                        for kind, state in self._default_tpms_parameter_states.items()
                    }
                active_kind = transient.get(
                    "active_tpms_kind",
                    document.settings.get("active_tpms_kind", "G"),
                )
                self._active_tpms_kind = (
                    active_kind if active_kind in TPMS_KINDS else "G"
                )
                self._next_primitive_index = int(
                    transient.get("next_primitive_index", 1)
                )
                self._selected_primitive_identifier = transient.get(
                    "selected_primitive_identifier"
                )
                self._restore_workspace_control_snapshot(
                    document.settings.get(
                        "ui_control_state",
                        self._default_workspace_control_state,
                    )
                )
                custom_source = document.settings.get("custom_unit_cell_source_path")
                if isinstance(custom_source, str) and custom_source:
                    source_path = Path(custom_source)
                    if not source_path.is_absolute() and self._workspace_manifest_path:
                        source_path = self._workspace_manifest_path.parent / source_path
                    self.custom_widgets["source_path"].setText(str(source_path))
                # Prepared custom cells keep native acceleration resources and
                # are intentionally rebuilt from the managed source on demand.
                self.prepared_custom_cell = None
                kind_index = self.tpms_widgets["kind"].findData(
                    self._active_tpms_kind
                )
                self.tpms_widgets["kind"].setCurrentIndex(kind_index)
                self._restore_tpms_parameter_state(self._active_tpms_kind)
                bounds = self.sole_mesh.bounds
                extent = bounds[1] - bounds[0]
                domain_kind = "解析设计域" if not isinstance(
                    document.domain, MeshDesignDomain
                ) else "设计域 STL"
                self.domain_info.setText(
                    f"{domain_kind} · {len(self.sole_mesh.faces):,} 个面 · "
                    f"{extent[0]:.1f} × {extent[1]:.1f} × {extent[2]:.1f} mm"
                )
            finally:
                self._workspace_loading = False
            self._layer_contour_cache.clear()
            self._field_plane_cache.clear()
            self._sync_field_primitive_options(self._selected_primitive_identifier)
            self._sync_transition_field_objects()
            self._sync_contour_result_options()
            self._sync_field_viewer_options()
            self._sync_stl_reconstruction_target_options()
            self._sync_section_target_options()
            self._sync_shell_lattice_options()
            self._set_result_controls_enabled(bool(self.results))
            self.show_stl_edges.setEnabled(bool(self.results))
            if not self.results:
                self.show_stl_edges.blockSignals(True)
                self.show_stl_edges.setChecked(False)
                self.show_stl_edges.blockSignals(False)
            self.generate_button.setEnabled(
                self.design_workspace.active_document is not None
                and self.thread is None
            )
            self._update_recommendation()
            self._sync_design_workspace_controls()
            self._refresh_scene(reset_view=True)

        def _sync_design_workspace_controls(self) -> None:
            if "design_selector" not in self.__dict__:
                return
            self._workspace_loading = True
            try:
                current = self.design_workspace.active_design_id
                self.design_selector.blockSignals(True)
                self.design_selector.clear()
                for identifier, document in self.design_workspace.documents.items():
                    self.design_selector.addItem(document.name, identifier)
                index = self.design_selector.findData(current)
                self.design_selector.setCurrentIndex(index if index >= 0 else -1)
                self.design_selector.blockSignals(False)
                archived_selection = self.archived_design_selector.currentData()
                self.archived_design_selector.clear()
                for identifier, document in self.design_workspace.archived_documents.items():
                    self.archived_design_selector.addItem(document.name, identifier)
                archived_index = self.archived_design_selector.findData(archived_selection)
                if archived_index >= 0:
                    self.archived_design_selector.setCurrentIndex(archived_index)
                has_active = self.design_workspace.active_document is not None
                self.duplicate_design_button.setEnabled(has_active)
                self.rename_design_button.setEnabled(has_active)
                self.archive_design_button.setEnabled(has_active)
                self.restore_design_button.setEnabled(
                    self.archived_design_selector.count() > 0
                )
                self.clear_archived_design_button.setEnabled(
                    self.archived_design_selector.count() > 0
                )
            finally:
                self._workspace_loading = False

        def _on_active_design_changed(self, _index: int = 0) -> None:
            if self._workspace_loading:
                return
            identifier = self.design_selector.currentData()
            if identifier is None or identifier == self.design_workspace.active_design_id:
                return
            if self._document_switch_is_locked():
                self._sync_design_workspace_controls()
                return
            self._store_active_design_state()
            if not self._backend_apply_workspace_commands(
                [{"kind": "document.activate", "document_id": identifier}]
            ):
                self._sync_design_workspace_controls()
                return
            self.design_workspace.activate(identifier)
            self._restore_active_design_state()

        def _duplicate_active_design(self) -> None:
            if self._document_switch_is_locked():
                return
            document = self.design_workspace.active_document
            if document is None:
                return
            self._store_active_design_state()
            duplicate_identifier = self.design_workspace.new_identifier()
            if not self._backend_apply_workspace_commands(
                [
                    {
                        "kind": "document.duplicate",
                        "document_id": document.identifier,
                        "new_document_id": duplicate_identifier,
                        "name": f"{document.name} 副本",
                    }
                ]
            ):
                return
            duplicate = self.design_workspace.duplicate_document(
                document.identifier,
                f"{document.name} 副本",
                new_identifier=duplicate_identifier,
            )
            self._restore_active_design_state()
            self.status.setText(f"已创建独立设计：{duplicate.name}")

        def _create_analytic_design(self, kind: PrimitiveKind) -> None:
            """Create a Design whose primitive SDF is the authoritative domain."""

            if self._document_switch_is_locked():
                return
            names = {"sphere": "球体设计", "cylinder": "圆柱设计", "box": "立方体设计"}
            identifier = f"domain-{kind}-{len(self.design_workspace.documents) + 1}"
            primitive = ImplicitPrimitive(
                identifier=identifier,
                name=names[kind],
                kind=kind,
                radius_mm=20.0,
                height_mm=40.0,
                size_mm=(40.0, 40.0, 40.0),
            )
            document_identifier = self.design_workspace.new_identifier()
            initial_settings = self._new_design_document_settings()
            initial_settings["analytic_domain_primitive_id"] = identifier
            if not self._backend_apply_workspace_commands(
                [
                    {
                        "kind": "document.create.analytic",
                        "document_id": document_identifier,
                        "name": names[kind],
                        "domain_name": names[kind],
                        "primitive": asdict(primitive),
                    },
                    {
                        "kind": "document.settings.replace",
                        "document_id": document_identifier,
                        "settings": initial_settings,
                    },
                    {
                        "kind": "document.field_scene.replace",
                        "document_id": document_identifier,
                        "field_primitives": self._backend_field_scene_payload(
                            {identifier: primitive},
                            {identifier: True},
                        ),
                    },
                ]
            ):
                return
            document = self.design_workspace.create_analytic_document(
                primitive,
                names[kind],
                identifier=document_identifier,
            )
            self._initialize_new_design_document(document)
            # One primitive may serve as both the authoritative Design domain
            # and a selectable transition driver. It remains one SDF definition.
            document.field_primitives[identifier] = primitive
            document.field_primitive_visibility[identifier] = True
            document.settings["analytic_domain_primitive_id"] = identifier
            document.touch()
            self._restore_active_design_state()
            self.show_domain.setChecked(True)
            self._refresh_domain_implicit_preview()
            # The preview sampler only updates the field cache.  Rebuild the
            # scene now so a newly created analytic Design domain is visible
            # immediately, rather than waiting for a later UI change.
            self._refresh_scene(reset_view=True)
            self._store_active_design_state()
            self.status.setText(f"已创建{names[kind]}；可直接作为设计域或过渡驱动场。")

        def _archive_active_design(self) -> None:
            if self._document_switch_is_locked():
                return
            document = self.design_workspace.active_document
            if document is None:
                return
            self._store_active_design_state()
            if not self._backend_apply_workspace_commands(
                [{"kind": "document.archive", "document_id": document.identifier}]
            ):
                return
            self.design_workspace.archive_document(document.identifier)
            self._restore_active_design_state()

        def _restore_archived_design(self) -> None:
            if self._document_switch_is_locked():
                return
            identifier = self.archived_design_selector.currentData()
            if identifier is None:
                return
            if not self._backend_apply_workspace_commands(
                [
                    {
                        "kind": "document.restore",
                        "document_id": identifier,
                        "activate": True,
                    }
                ]
            ):
                return
            self.design_workspace.restore_document(identifier, activate=True)
            self._restore_active_design_state()

        def _rename_active_design(self) -> None:
            document = self.design_workspace.active_document
            if document is None:
                return
            name, confirmed = QtWidgets.QInputDialog.getText(
                self, "重命名当前设计", "设计名称", text=document.name,
            )
            if not confirmed:
                return
            if not name.strip():
                QtWidgets.QMessageBox.warning(self, "名称无效", "设计名称不能为空。")
                return
            if not self._backend_apply_workspace_commands(
                [
                    {
                        "kind": "document.rename",
                        "document_id": document.identifier,
                        "name": name,
                    }
                ]
            ):
                return
            self.design_workspace.rename_document(document.identifier, name)
            self._sync_design_workspace_controls()
            self.status.setText(f"当前设计已命名为：{document.name}")

        def _clear_archived_design(self) -> None:
            identifier = self.archived_design_selector.currentData()
            document = self.design_workspace.archived_documents.get(identifier)
            if document is None:
                return
            answer = QtWidgets.QMessageBox.question(
                self, "永久清除设计",
                f"确定永久清除回收站中的设计“{document.name}”？\n"
                "此操作无法通过回收站恢复，保存工程后生效。\n"
                "不会删除外部 STL 文件或其他设计使用的资源。",
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
                QtWidgets.QMessageBox.No,
            )
            if answer != QtWidgets.QMessageBox.Yes:
                return
            if not self._backend_apply_workspace_commands(
                [{"kind": "document.remove_archived", "document_id": identifier}]
            ):
                return
            self.design_workspace.permanently_remove_archived_document(identifier)
            self._sync_design_workspace_controls()
            self.status.setText(f"已永久清除设计：{document.name}；请保存工程。")

        def _workspace_asset_source_path(self, asset_path: Path) -> Path:
            """Resolve a managed or external source path before packaging it."""

            path = Path(asset_path)
            if path.is_absolute():
                return path
            if self._workspace_manifest_path is not None:
                return self._workspace_manifest_path.parent / path
            return self.root / path

        def _custom_source_control_key(self) -> str | None:
            source_control = self.custom_widgets["source_path"]
            for key, control in self._workspace_controls:
                if control is source_control:
                    return key
            return None

        def _package_workspace_assets(self, workspace_root: Path) -> None:
            """Ensure persisted Design documents never retain external STL paths."""

            source_control_key = self._custom_source_control_key()
            all_documents = (
                list(self.design_workspace.documents.values())
                + list(self.design_workspace.archived_documents.values())
            )
            for document in all_documents:
                if isinstance(document.domain, MeshDesignDomain):
                    source_path = document.domain.asset_path
                    if source_path is None:
                        raise ValueError(
                            f"设计“{document.name}”缺少 STL 源文件，无法保存工程"
                        )
                    self.design_workspace.package_mesh_domain_asset(
                        document.identifier,
                        self._workspace_asset_source_path(source_path),
                        workspace_root,
                    )
                source_text = document.settings.get("custom_unit_cell_source_path")
                if not isinstance(source_text, str) or not source_text.strip():
                    continue
                target = DesignWorkspace.package_stl_asset(
                    self._workspace_asset_source_path(Path(source_text)),
                    workspace_root,
                    "custom_cells",
                    document.identifier,
                )
                relative = str(target.relative_to(workspace_root))
                document.settings["custom_unit_cell_source_path"] = relative
                controls = document.settings.get("ui_control_state")
                if isinstance(controls, dict) and source_control_key is not None:
                    controls[source_control_key] = relative

        def _package_derived_meshes(self, workspace_root: Path) -> None:
            """Persist generated meshes, while leaving sampled fields disposable."""

            active = self.design_workspace.active_document
            if active is not None:
                self._store_active_design_state()
            for document in self.design_workspace.documents.values():
                meshes = document.runtime.results or document.runtime.raw_results
                if not meshes:
                    continue
                for kind, mesh in meshes.items():
                    if not isinstance(mesh, trimesh.Trimesh) or not mesh.faces.size:
                        continue
                    asset = workspace_root / "assets" / "results" / (
                        f"{document.identifier}_{kind}.stl"
                    )
                    asset.parent.mkdir(parents=True, exist_ok=True)
                    mesh.export(asset)
                    document.register_derived_mesh(
                        PersistedDerivedMesh(
                            identifier=str(kind),
                            asset_path=asset.relative_to(workspace_root),
                            provenance={
                                "source": "authoritative_implicit_body",
                                "design_revision": document.revision,
                                "kind": str(kind),
                            },
                        )
                    )

        def _save_workspace(self) -> None:
            if self.design_workspace.active_document is None:
                QtWidgets.QMessageBox.information(
                    self,
                    "保存工程",
                    "空工作区没有可保存的设计。",
                )
                return
            default_path = self._workspace_manifest_path or (
                self.root / "build" / "exports" / "tpms_workspace.json"
            )
            path, _ = QtWidgets.QFileDialog.getSaveFileName(
                self,
                "保存设计工程",
                str(default_path),
                "TPMS 工作区 (*.json)",
            )
            if not path:
                return
            manifest_path = Path(path).with_suffix(".json")
            try:
                self._store_active_design_state()
                self._package_workspace_assets(manifest_path.parent)
                self._package_derived_meshes(manifest_path.parent)
                if self.backend_client is not None and self.backend_workspace_id is not None:
                    if not self._sync_workspace_projection_to_backend():
                        return
                    self.backend_client.save_workspace(
                        self.backend_workspace_id,
                        str(manifest_path),
                    )
                    self.design_workspace.dirty = False
                else:
                    self.design_workspace.save(manifest_path)
                self._workspace_manifest_path = manifest_path
                self.status.setText(f"工程已保存：{manifest_path}")
            except (OSError, ValueError, TypeError) as exc:
                QtWidgets.QMessageBox.critical(
                    self,
                    "保存工程失败",
                    f"{type(exc).__name__}: {exc}",
                )

        def _open_workspace(self) -> None:
            if any((self.thread, self.shell_thread, self.stl_thread)):
                QtWidgets.QMessageBox.warning(
                    self,
                    "无法打开工程",
                    "请等待当前后台任务完成后再打开其他工程。",
                )
                return
            if self.design_workspace.dirty:
                answer = QtWidgets.QMessageBox.question(
                    self,
                    "未保存的更改",
                    "当前工程存在未保存的更改。是否先保存？",
                    QtWidgets.QMessageBox.Save
                    | QtWidgets.QMessageBox.Discard
                    | QtWidgets.QMessageBox.Cancel,
                    QtWidgets.QMessageBox.Save,
                )
                if answer == QtWidgets.QMessageBox.Cancel:
                    return
                if answer == QtWidgets.QMessageBox.Save:
                    self._save_workspace()
                    if self.design_workspace.dirty:
                        return
            path, _ = QtWidgets.QFileDialog.getOpenFileName(
                self,
                "打开设计工程",
                str(self._workspace_manifest_path or self.root),
                "TPMS 工作区 (*.json)",
            )
            if not path:
                return
            manifest_path = Path(path)
            try:
                if self.backend_client is not None:
                    session = self.backend_client.load_workspace(str(manifest_path))
                    snapshot = self.backend_client.get_workspace_snapshot(session.identifier)
                    self.design_workspace = read_workspace_projection(
                        snapshot,
                        manifest_path.parent,
                    )
                    self.backend_workspace_id = session.identifier
                else:
                    self.design_workspace = DesignWorkspace.load(manifest_path)
                self._workspace_manifest_path = manifest_path
                self._restore_persisted_derived_meshes()
                self._restore_active_design_state()
                self.status.setText(f"工程已打开：{manifest_path}")
            except (OSError, ValueError, TypeError) as exc:
                QtWidgets.QMessageBox.critical(
                    self,
                    "打开工程失败",
                    f"{type(exc).__name__}: {exc}",
                )

        def _restore_persisted_derived_meshes(self) -> None:
            """Reload managed derived meshes without treating them as implicit caches."""

            if self._workspace_manifest_path is None:
                return
            root = self._workspace_manifest_path.parent
            for document in self.design_workspace.documents.values():
                runtime_meshes: dict[str, trimesh.Trimesh] = {}
                for identifier, record in document.derived_meshes.items():
                    path = root / record.asset_path
                    if not path.is_file():
                        continue
                    mesh = trimesh.load(path, force="mesh", process=False)
                    if isinstance(mesh, trimesh.Trimesh) and mesh.faces.size:
                        runtime_meshes[identifier] = mesh
                document.runtime.results = runtime_meshes
                document.runtime.raw_results = {
                    identifier: mesh.copy()
                    for identifier, mesh in runtime_meshes.items()
                }

        def _active_design_domain(self) -> DesignDomainValue | None:
            document = self.design_workspace.active_document
            return None if document is None else document.domain

        def _active_domain_bounds(self) -> np.ndarray:
            domain = self._active_design_domain()
            if domain is not None:
                return np.asarray(domain.bounds, dtype=np.float64)
            return np.asarray(self.sole_mesh.bounds, dtype=np.float64)

        def _active_domain_frame_points(self) -> np.ndarray:
            domain = self._active_design_domain()
            if domain is not None:
                return np.asarray(domain.frame_points, dtype=np.float64)
            return np.asarray(self.sole_mesh.vertices, dtype=np.float64)

        def _cell_map_for_active_design(
            self,
            parameters: LatticeParameters,
        ) -> CellMap:
            domain = self._active_design_domain()
            if domain is not None:
                return cell_map_for_design_domain(domain, parameters)
            return cell_map_for_parameters(self.sole_mesh, parameters)

        def _sampling_for_active_design(
            self,
            parameters: LatticeParameters,
            sampling: SamplingParameters,
            *,
            cell_map: CellMap | None = None,
            minimum_feature_mm: float | None = None,
        ) -> SamplingRecommendation:
            domain = self._active_design_domain()
            if domain is not None:
                return _domain_sampling_recommendation(
                    domain,
                    parameters,
                    sampling,
                    cell_map=cell_map,
                    minimum_feature_mm=minimum_feature_mm,
                )
            if isinstance(parameters, CustomUnitCellParameters):
                return custom_cell_sampling_recommendation(
                    self.sole_mesh,
                    parameters,
                    sampling,
                    cell_map=cell_map,
                    minimum_feature_mm=minimum_feature_mm,
                )
            return lattice_sampling_recommendation(
                self.sole_mesh,
                parameters,
                sampling,
            )

        def _mark_active_design_definition_changed(
            self,
            keys: set[str] | None = None,
        ) -> None:
            """Invalidate results and advance the task-ownership revision."""

            if self._workspace_loading:
                return
            document = self.design_workspace.active_document
            if document is None:
                return
            if keys is None:
                document.clear_generated()
                document.touch()
            else:
                document.invalidate_generated_results(keys)
            self.design_workspace.dirty = True

        def _build_ui(self):
            self.setStyleSheet(_application_stylesheet())
            root = QtWidgets.QWidget()
            self.setCentralWidget(root)
            layout = QtWidgets.QHBoxLayout(root)
            layout.setContentsMargins(12, 12, 12, 12)
            layout.setSpacing(12)

            tabs = QtWidgets.QTabWidget()
            tabs.setDocumentMode(True)
            tabs.setMinimumWidth(440)
            tabs.setMaximumWidth(520)
            tabs.setSizePolicy(
                QtWidgets.QSizePolicy.Preferred,
                QtWidgets.QSizePolicy.Expanding,
            )
            generation_page = QtWidgets.QWidget()
            generation_page.setMinimumWidth(420)
            left_layout = QtWidgets.QVBoxLayout(generation_page)
            left_layout.setContentsMargins(12, 12, 12, 12)
            left_layout.setSpacing(12)
            section_page = QtWidgets.QWidget()
            section_page.setMinimumWidth(420)
            section_layout = QtWidgets.QVBoxLayout(section_page)
            section_layout.setContentsMargins(12, 12, 12, 12)
            section_layout.setSpacing(12)
            shell_page = QtWidgets.QWidget()
            shell_page.setMinimumWidth(420)
            shell_layout = QtWidgets.QVBoxLayout(shell_page)
            shell_layout.setContentsMargins(12, 12, 12, 12)
            shell_layout.setSpacing(12)
            field_object_page = QtWidgets.QWidget()
            field_object_page.setMinimumWidth(420)
            field_object_layout = QtWidgets.QVBoxLayout(field_object_page)
            field_object_layout.setContentsMargins(12, 12, 12, 12)
            field_object_layout.setSpacing(12)
            generation_scroll = QtWidgets.QScrollArea()
            generation_scroll.setWidgetResizable(True)
            generation_scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
            generation_scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
            generation_scroll.setWidget(generation_page)
            section_scroll = QtWidgets.QScrollArea()
            section_scroll.setWidgetResizable(True)
            section_scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
            section_scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
            section_scroll.setWidget(section_page)
            shell_scroll = QtWidgets.QScrollArea()
            shell_scroll.setWidgetResizable(True)
            shell_scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
            shell_scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
            shell_scroll.setWidget(shell_page)
            field_object_scroll = QtWidgets.QScrollArea()
            field_object_scroll.setWidgetResizable(True)
            field_object_scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
            field_object_scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
            field_object_scroll.setWidget(field_object_page)
            tabs.addTab(generation_scroll, "晶格生成")
            tabs.addTab(section_scroll, "截面与等值线")
            tabs.addTab(field_object_scroll, "场对象")
            tabs.addTab(shell_scroll, "抽壳与融合")
            layout.addWidget(tabs)

            self.design_workspace_group = self._make_design_workspace_group()
            left_layout.addWidget(self.design_workspace_group)
            file_group = QtWidgets.QGroupBox("设计域")
            file_form = QtWidgets.QFormLayout(file_group)
            self.path_edit = QtWidgets.QLineEdit()
            self.path_edit.setReadOnly(True)
            self.path_edit.setToolTip("通过“设计工作区 > 新建 STL 设计”导入设计域")
            file_form.addRow("模型", self.path_edit)
            self.domain_info = QtWidgets.QLabel("等待模型")
            self.domain_info.setObjectName("helper")
            file_form.addRow("状态", self.domain_info)
            self.repair_domain_button = QtWidgets.QPushButton("修复设计域网格")
            self.repair_domain_button.setToolTip("去除重复/退化面片，填补简单孔洞并修复法向")
            self.repair_domain_button.clicked.connect(self._repair_design_domain)
            file_form.addRow(self.repair_domain_button)
            left_layout.addWidget(file_group)

            self.tpms_group, self.tpms_widgets = self._make_tpms_group(
                "TPMS 晶胞",
                "G",
                (0.96, 0.58, 0.18),
            )
            self._active_tpms_kind: TPMSKind = "G"
            self._tpms_parameter_states: dict[TPMSKind, TPMSParameterState] = {
                kind: self._default_tpms_parameter_state(kind)
                for kind in TPMS_KINDS
            }
            self._tpms_parameter_states["G"] = self._tpms_state_from_widgets()
            # Keep the old attribute names as compatibility aliases for the
            # transition and Cell Map code.  There is only one visible TPMS
            # configuration panel; its type selector determines the job key.
            self.g_group = self.tpms_group
            self.d_group = self.tpms_group
            self.g_widgets = self.tpms_widgets
            self.d_widgets = self.tpms_widgets
            self.custom_group, self.custom_widgets = self._make_custom_cell_group()
            left_layout.addWidget(self.tpms_group)
            left_layout.addWidget(self.custom_group)
            self.cell_map_group = self._make_cell_map_group()
            left_layout.addWidget(self.cell_map_group)
            self.transition_group = self._make_transition_group()
            left_layout.addWidget(self.transition_group)
            self.field_viewer_group = self._make_field_viewer_group()
            section_layout.addWidget(self.field_viewer_group)
            self.section_group = self._make_section_group()
            section_layout.addWidget(self.section_group)
            self.contour_group = self._make_contour_group()
            section_layout.addWidget(self.contour_group)
            section_layout.addStretch()
            self.field_objects_group = self._make_field_objects_group()
            field_object_layout.addWidget(self.field_objects_group)
            field_object_layout.addStretch()
            self.shell_group = self._make_shell_group()
            shell_layout.addWidget(self.shell_group)
            shell_layout.addStretch()

            sample_group = self.sampling_group = _CollapsibleGroupBox(
                "隐式体渲染",
                collapsed=True,
            )
            sample_form = QtWidgets.QFormLayout(sample_group)
            self.target_voxels = QtWidgets.QLineEdit("4000000")
            self.target_voxels.setPlaceholderText("例如：4000000")
            self.target_voxels.setToolTip("仅用于推荐体素大小，不设置最大体素上限；过大的值会增加内存和计算时间")
            self.target_voxels.editingFinished.connect(self._update_recommendation)
            self.voxel_info = QtWidgets.QLabel("自动推荐将在模型加载后显示")
            self.voxel_info.setObjectName("helper")
            sample_form.addRow("目标体素数", self.target_voxels)
            sample_form.addRow("自动采样建议", self.voxel_info)
            self.display_voxel_size = QtWidgets.QDoubleSpinBox()
            self.display_voxel_size.setRange(0.001, 1000.0)
            self.display_voxel_size.setDecimals(4)
            self.display_voxel_size.setButtonSymbols(QtWidgets.QAbstractSpinBox.NoButtons)
            self.display_voxel_size.setReadOnly(True)
            self.display_voxel_size.setToolTip("实时隐式预览的采样间距；只影响显示缓存，不影响权威计算或 STL 导出")
            self.display_voxel_auto = QtWidgets.QCheckBox("自动推荐显示体素")
            self.display_voxel_auto.setChecked(True)
            self.display_voxel_auto.toggled.connect(self._on_display_voxel_mode_changed)
            self.display_voxel_size.valueChanged.connect(self._update_recommendation)
            sample_form.addRow("显示体素大小", self.display_voxel_size)
            sample_form.addRow("显示分辨率", self.display_voxel_auto)
            self.display_memory_budget = QtWidgets.QSpinBox()
            self.display_memory_budget.setRange(64, 16384)
            self.display_memory_budget.setValue(512)
            self.display_memory_budget.setSuffix(" MB")
            self.display_memory_budget.setToolTip("实时 GPU 隐式显示缓存的独立预算；超出时自动增大显示体素")
            self.display_memory_budget.valueChanged.connect(self._update_recommendation)
            sample_form.addRow("显示显存预算", self.display_memory_budget)
            self.display_batched_sampling = QtWidgets.QCheckBox(
                "启用隐式体分批采样"
            )
            self.display_batched_sampling.setChecked(False)
            self.display_batched_sampling.setToolTip(
                "将显示标量场沿 X 方向分批计算；降低场求值的临时内存峰值，"
                "不改变显示体素、最终隐式场或 STL 重建设置"
            )
            self.display_batched_sampling.toggled.connect(
                self._on_display_batch_mode_changed
            )
            sample_form.addRow("显示采样方式", self.display_batched_sampling)
            self.display_batch_count = QtWidgets.QSpinBox()
            self.display_batch_count.setRange(2, 512)
            self.display_batch_count.setValue(4)
            self.display_batch_count.setSuffix(" 批")
            self.display_batch_count.setEnabled(False)
            self.display_batch_count.setToolTip(
                "只控制隐式显示场的外层采样批次；每批内部仍自动使用微分片"
            )
            self.display_batch_count.valueChanged.connect(
                self._update_recommendation
            )
            sample_form.addRow("显示采样批数", self.display_batch_count)
            self.cell_sample_info = QtWidgets.QLabel(
                "模型加载后显示单晶胞采样估算"
            )
            self.cell_sample_info.setObjectName("helper")
            self.cell_sample_info.setWordWrap(True)
            self.cell_sample_info.setTextInteractionFlags(
                QtCore.Qt.TextSelectableByMouse
            )
            self.cell_sample_info.setToolTip(
                "按 Cell Map 的 U/V/W 晶胞尺寸除以当前显示体素大小，"
                "估算每轴采样数和单晶胞总体素采样数"
            )
            sample_form.addRow("单晶胞采样估算", self.cell_sample_info)
            self.use_cpp_sdf = QtWidgets.QCheckBox("自动 GPU/C++ SDF 加速")
            self.use_cpp_sdf.setChecked(True)
            self.use_cpp_sdf.setToolTip(
                "优先使用 CUDA GPU BVH；GPU 不可用或失败时回退 C++ BVH，再回退 PyVista"
            )
            sample_form.addRow(self.use_cpp_sdf)
            self.compute_backend_info = QtWidgets.QLabel(
                "TPMS、几何计算与设备常驻裁剪流水线：自动探测中"
            )
            self.compute_backend_info.setObjectName("helper")
            self.compute_backend_info.setWordWrap(True)
            self.compute_backend_info.setToolTip(
                "分别显示 TPMS kernel、独立几何运算以及 SDF+TPMS+交集常驻流水线的真实后端状态"
            )
            sample_form.addRow("数值计算后端", self.compute_backend_info)
            self._implicit_sampling_summary = "隐式显示：尚未生成"
            self._stl_sampling_summary = "STL 重建：尚未生成"
            self.applied_sampling_info = QtWidgets.QLabel()
            self.applied_sampling_info.setObjectName("helper")
            self.applied_sampling_info.setWordWrap(True)
            self.applied_sampling_info.setTextInteractionFlags(
                QtCore.Qt.TextSelectableByMouse
            )
            sample_form.addRow("实际采样结果", self.applied_sampling_info)
            self._refresh_applied_sampling_info()
            left_layout.addWidget(sample_group)

            output_group = QtWidgets.QGroupBox("显示与生成")
            output_layout = QtWidgets.QVBoxLayout(output_group)
            self.show_domain = QtWidgets.QCheckBox("显示设计域")
            self.show_domain.setChecked(True)
            self.show_domain.setToolTip("隐藏灰色设计域后，可以直接检查内部 TPMS 晶格")
            self.show_domain.stateChanged.connect(self._refresh_scene)
            self.domain_display_source_combo = QtWidgets.QComboBox()
            self.domain_display_source_combo.addItem("隐式体", "implicit")
            self.domain_display_source_combo.addItem("STL 网格", "mesh")
            self.domain_display_source_combo.setToolTip(
                "独立选择设计域的显示表示；不改变晶格结果的 STL 重建或导出。"
            )
            self.domain_display_source_combo.currentIndexChanged.connect(
                self._refresh_scene
            )
            self.show_g = QtWidgets.QCheckBox("显示 G 结果")
            self.show_g.setChecked(True)
            self.show_d = QtWidgets.QCheckBox("显示 D 结果")
            self.show_d.setChecked(True)
            self.show_custom = QtWidgets.QCheckBox("显示自定义晶胞结果")
            self.show_custom.setChecked(True)
            output_layout.addWidget(self.show_domain)
            domain_source_form = QtWidgets.QFormLayout()
            domain_source_form.setContentsMargins(0, 0, 0, 0)
            domain_source_form.addRow("设计域显示源", self.domain_display_source_combo)
            output_layout.addLayout(domain_source_form)
            self.show_g.stateChanged.connect(self._refresh_scene)
            self.show_d.stateChanged.connect(self._refresh_scene)
            self.show_custom.stateChanged.connect(self._refresh_scene)
            output_layout.addWidget(self.show_g)
            output_layout.addWidget(self.show_d)
            output_layout.addWidget(self.show_custom)
            self.show_iwp = QtWidgets.QCheckBox("显示 IWP 结果")
            self.show_primitive = QtWidgets.QCheckBox("显示 Primitive 结果")
            self.show_neovius = QtWidgets.QCheckBox("显示 Neovius 结果")
            for checkbox in (self.show_iwp, self.show_primitive, self.show_neovius):
                checkbox.setChecked(True)
                checkbox.stateChanged.connect(self._refresh_scene)
                output_layout.addWidget(checkbox)
            self.show_transition = QtWidgets.QCheckBox("显示晶胞过渡结果")
            self.show_transition.setChecked(True)
            self.show_transition.stateChanged.connect(self._refresh_scene)
            output_layout.addWidget(self.show_transition)
            self.show_shell = QtWidgets.QCheckBox("显示抽壳结果")
            self.show_shell.setChecked(True)
            self.show_shell.stateChanged.connect(self._refresh_scene)
            output_layout.addWidget(self.show_shell)
            self.show_shell_union = QtWidgets.QCheckBox("显示壳体融合结果")
            self.show_shell_union.setChecked(True)
            self.show_shell_union.stateChanged.connect(self._refresh_scene)
            output_layout.addWidget(self.show_shell_union)
            self.view_stl_mesh = QtWidgets.QCheckBox("显示生成 STL 模型")
            self.view_stl_mesh.setEnabled(False)
            self.view_stl_mesh.setToolTip(
                "启用后显示按 STL 容差生成的网格；关闭时显示交互隐式体"
            )
            self.view_stl_mesh.toggled.connect(self._refresh_scene)
            output_layout.addWidget(self.view_stl_mesh)
            self.show_stl_edges = QtWidgets.QCheckBox("显示 STL 三角网格")
            self.show_stl_edges.setChecked(False)
            self.show_stl_edges.setEnabled(False)
            self.show_stl_edges.setToolTip(
                "仅在显示生成 STL 模型时显示三角面边线，不改变 STL 文件和隐式体渲染"
            )
            self.show_stl_edges.toggled.connect(self._refresh_scene)
            output_layout.addWidget(self.show_stl_edges)
            advanced_render_group = self.advanced_render_group = _CollapsibleGroupBox(
                "高级渲染与材质",
                collapsed=True,
            )
            advanced_render_layout = QtWidgets.QVBoxLayout(advanced_render_group)
            style_form = QtWidgets.QFormLayout()
            self.render_quality_combo = QtWidgets.QComboBox()
            for label, quality in (
                ("低质量", "low"),
                ("中等", "medium"),
                ("高级", "high"),
                ("极佳", "ultra"),
            ):
                self.render_quality_combo.addItem(label, quality)
            self.render_quality_combo.setCurrentIndex(
                self.render_quality_combo.findData("high")
            )
            self.render_quality_combo.setToolTip(
                "调整 GPU 光线采样、抗锯齿和大网格显示代理精度；"
                "高级和极佳模式会在显示预算内从权威隐式体重建不同密度的显示场，"
                "不改变导出 STL"
            )
            self.render_quality_combo.currentIndexChanged.connect(
                self._on_render_quality_changed
            )
            style_form.addRow("渲染质量", self.render_quality_combo)
            self.precise_render_target_combo = QtWidgets.QComboBox()
            self.precise_render_target_combo.setToolTip(
                "选择一个权威隐式体进行静态高精度渲染；不会读取 STL 网格或交互显示缓存"
            )
            style_form.addRow("精确渲染对象", self.precise_render_target_combo)
            self.precise_render_resolution_combo = QtWidgets.QComboBox()
            for label, size in (
                ("1920 × 1080", (1920, 1080)),
                ("2560 × 1440", (2560, 1440)),
                ("3840 × 2160", (3840, 2160)),
                ("自定义", None),
            ):
                self.precise_render_resolution_combo.addItem(label, size)
            self.precise_render_resolution_combo.currentIndexChanged.connect(
                self._sync_precise_render_resolution_controls
            )
            style_form.addRow("精确渲染分辨率", self.precise_render_resolution_combo)
            precise_resolution_row = QtWidgets.QWidget()
            precise_resolution_layout = QtWidgets.QHBoxLayout(precise_resolution_row)
            precise_resolution_layout.setContentsMargins(0, 0, 0, 0)
            self.precise_render_width = QtWidgets.QSpinBox()
            self.precise_render_width.setRange(320, 16384)
            self.precise_render_width.setValue(1920)
            self.precise_render_width.setSuffix(" px")
            self.precise_render_height = QtWidgets.QSpinBox()
            self.precise_render_height.setRange(240, 16384)
            self.precise_render_height.setValue(1080)
            self.precise_render_height.setSuffix(" px")
            precise_resolution_layout.addWidget(self.precise_render_width)
            precise_resolution_layout.addWidget(QtWidgets.QLabel("×"))
            precise_resolution_layout.addWidget(self.precise_render_height)
            style_form.addRow("自定义尺寸", precise_resolution_row)
            self.precise_render_button = QtWidgets.QPushButton("精确渲染 PNG")
            self.precise_render_button.setProperty("role", "accent")
            self.precise_render_button.setToolTip(
                "从当前相机对选定权威隐式体进行静态精确渲染，并保存为 PNG"
            )
            self.precise_render_button.clicked.connect(self._start_precise_render)
            self.precise_render_info = QtWidgets.QLabel(
                "精确渲染将直接求值权威隐式体，不改变交互显示场"
            )
            self.precise_render_info.setObjectName("helper")
            self.precise_render_info.setWordWrap(True)
            self.precise_render_info.setTextInteractionFlags(
                QtCore.Qt.TextSelectableByMouse
            )
            self.render_backend_info = QtWidgets.QLabel("正在探测 OpenGL 渲染设备")
            self.render_backend_info.setObjectName("helper")
            self.render_backend_info.setWordWrap(True)
            self.render_backend_info.setToolTip(
                "隐式体使用 GPU 零等值面光线投射；若检测到软件 OpenGL，将明确显示降级原因"
            )
            style_form.addRow("图形渲染后端", self.render_backend_info)
            self.background_combo = QtWidgets.QComboBox()
            for name, color in BACKGROUND_PRESETS.items():
                self.background_combo.addItem(name, color)
            self.background_combo.currentIndexChanged.connect(self._on_background_changed)
            style_form.addRow("背景颜色", self.background_combo)
            self.domain_material_combo = self._make_material_combo("domain")
            self.g_material_combo = self._make_material_combo("G")
            self.d_material_combo = self._make_material_combo("D")
            self.iwp_material_combo = self._make_material_combo("IWP")
            self.primitive_material_combo = self._make_material_combo("Primitive")
            self.neovius_material_combo = self._make_material_combo("Neovius")
            self.custom_material_combo = self._make_material_combo("Custom")
            self.transition_material_combo = self._make_material_combo("Transition")
            self.shell_material_combo = self._make_material_combo("Shell")
            self.shell_union_material_combo = self._make_material_combo("ShellUnion")
            style_form.addRow("设计域材质", self.domain_material_combo)
            style_form.addRow("G 材质", self.g_material_combo)
            style_form.addRow("D 材质", self.d_material_combo)
            style_form.addRow("IWP 材质", self.iwp_material_combo)
            style_form.addRow("Primitive 材质", self.primitive_material_combo)
            style_form.addRow("Neovius 材质", self.neovius_material_combo)
            style_form.addRow("自定义晶胞材质", self.custom_material_combo)
            style_form.addRow("过渡材质", self.transition_material_combo)
            style_form.addRow("抽壳材质", self.shell_material_combo)
            style_form.addRow("壳体融合材质", self.shell_union_material_combo)
            advanced_render_layout.addLayout(style_form)
            advanced_render_layout.addWidget(self.precise_render_button)
            advanced_render_layout.addWidget(self.precise_render_info)
            output_layout.addWidget(advanced_render_group)
            self.check_button = QtWidgets.QPushButton("检查模型")
            self.check_button.setToolTip("生成前检查设计域、参数、体素规模；生成后检查结果网格")
            self.check_button.clicked.connect(self._check_model)
            output_layout.addWidget(self.check_button)
            self.generate_button = QtWidgets.QPushButton("生成选中的晶格")
            self.generate_button.setProperty("role", "primary")
            self.generate_button.clicked.connect(self._generate)
            output_layout.addWidget(self.generate_button)
            self.progress = QtWidgets.QProgressBar()
            self.progress.setRange(0, 100)
            self.progress.setValue(0)
            output_layout.addWidget(self.progress)
            self.status = QtWidgets.QLabel("就绪")
            self.status.setObjectName("helper")
            self.status.setWordWrap(True)
            output_layout.addWidget(self.status)
            left_layout.addWidget(output_group)
            model_group = self.stl_processing_group = _CollapsibleGroupBox(
                "结果处理、STL 重建与导出",
                collapsed=True,
            )
            model_layout = QtWidgets.QVBoxLayout(model_group)
            simplify_form = QtWidgets.QFormLayout()
            self.export_tolerance = QtWidgets.QDoubleSpinBox()
            self.export_tolerance.setRange(0.02, 5.0)
            self.export_tolerance.setDecimals(3)
            self.export_tolerance.setSingleStep(0.05)
            self.export_tolerance.setValue(0.25)
            self.export_tolerance.setSuffix(" mm")
            self.export_tolerance.setToolTip(
                "STL 重建请求的最大采样间距；实际间距还受最小特征厚度 / 3 "
                "和晶胞尺寸 / 16 约束；不使用显示体素大小"
            )
            self.export_tolerance.valueChanged.connect(
                self._update_stl_grid_estimate
            )
            simplify_form.addRow("STL 输入间距", self.export_tolerance)
            self.export_spacing_auto = QtWidgets.QCheckBox(
                "自动应用特征精度约束"
            )
            self.export_spacing_auto.setChecked(True)
            self.export_spacing_auto.setToolTip(
                "开启：输入值是采样上限，系统还会应用特征厚度 / 3 和晶胞尺寸 / 16；"
                "关闭：三轴严格使用输入间距，不再自动改小"
            )
            self.export_spacing_auto.toggled.connect(
                self._on_export_spacing_mode_changed
            )
            simplify_form.addRow("STL 采样模式", self.export_spacing_auto)
            self.export_grid_info = QtWidgets.QLabel(
                "生成隐式体或创建解析设计域后显示 STL 网格尺寸和总体素数"
            )
            self.export_grid_info.setObjectName("helper")
            self.export_grid_info.setWordWrap(True)
            self.export_grid_info.setTextInteractionFlags(
                QtCore.Qt.TextSelectableByMouse
            )
            simplify_form.addRow("STL 体素预估", self.export_grid_info)
            self.processing_mode = QtWidgets.QComboBox()
            self.processing_mode.addItem("单次计算（不分块）", "single_pass")
            self.processing_mode.addItem(
                "分批采样 + 单次 Marching Cubes", "batched_field"
            )
            self.processing_mode.addItem(
                "分批采样 + 分块 Marching Cubes", "chunked_marching_cubes"
            )
            self.processing_mode.setToolTip(
                "单次模式一次性计算完整场并执行一次 Marching Cubes；"
                "分批模式分批采样后执行一次 Marching Cubes；"
                "分块模式逐块提取并拼接网格。"
            )
            self.processing_mode.currentIndexChanged.connect(
                self._on_processing_mode_changed
            )
            simplify_form.addRow("计算模式", self.processing_mode)
            self.batch_count = QtWidgets.QSpinBox()
            self.batch_count.setRange(2, 512)
            self.batch_count.setValue(4)
            self.batch_count.setSuffix(" 批")
            self.batch_count.setToolTip(
                "批次数越多，单批内存通常越低，但 SDF 和网格提取的调度及拼接开销越大"
            )
            self.batch_count.setEnabled(False)
            self.batch_count.valueChanged.connect(self._update_recommendation)
            simplify_form.addRow("分批数量", self.batch_count)
            self.repair_tolerance = QtWidgets.QDoubleSpinBox()
            self.repair_tolerance.setRange(0.0, 5.0)
            self.repair_tolerance.setDecimals(3)
            self.repair_tolerance.setSingleStep(0.05)
            self.repair_tolerance.setValue(0.20)
            self.repair_tolerance.setSuffix(" mm")
            self.repair_tolerance.setToolTip("仅闭合不大于该尺度的局部隐式场间隙")
            simplify_form.addRow("修复容差", self.repair_tolerance)
            self.clean_numerical_fragments = QtWidgets.QCheckBox("清理数值碎片")
            self.clean_numerical_fragments.setChecked(True)
            self.clean_numerical_fragments.setToolTip(
                "仅移除体积同时低于 STL 容差立方和主组件百万分之一的闭合伪影；不会按面片数删除组件"
            )
            simplify_form.addRow("拓扑清理", self.clean_numerical_fragments)
            self.optimize_stl_for_slicing = QtWidgets.QCheckBox(
                "启用拓扑保持的切片优化"
            )
            self.optimize_stl_for_slicing.setChecked(True)
            self.optimize_stl_for_slicing.setToolTip(
                "在最终网格合格后最多减少约 20% 面片；仅当水密、法向、单连通、"
                "三角形质量和几何偏差检查全部通过时采用，否则自动回滚"
            )
            simplify_form.addRow("切片网格优化", self.optimize_stl_for_slicing)
            self.simplify_percent = QtWidgets.QDoubleSpinBox()
            self.simplify_percent.setRange(0.0, 90.0)
            self.simplify_percent.setDecimals(1)
            self.simplify_percent.setSingleStep(5.0)
            self.simplify_percent.setValue(0.0)
            self.simplify_percent.setSuffix(" %")
            self.simplify_percent.setToolTip("按当前生成结果的面片数减少百分比；0% 表示保留原始网格")
            simplify_form.addRow("面片简化", self.simplify_percent)
            self.stl_reconstruction_target_combo = QtWidgets.QComboBox()
            self.stl_reconstruction_target_combo.setToolTip(
                "选择一个权威隐式体；仅对该对象执行 STL 重建、拓扑清理和可选修复"
            )
            self.stl_reconstruction_target_combo.currentIndexChanged.connect(
                self._on_stl_reconstruction_target_changed
            )
            simplify_form.addRow("STL 重建对象", self.stl_reconstruction_target_combo)
            model_layout.addLayout(simplify_form)
            self.build_stl_button = QtWidgets.QPushButton("生成 STL 网格")
            self.build_stl_button.setProperty("role", "accent")
            self.build_stl_button.setEnabled(False)
            self.build_stl_button.clicked.connect(
                lambda: self._start_stl_reconstruction(repair=False)
            )
            model_layout.addWidget(self.build_stl_button)
            self.use_repaired_result = QtWidgets.QCheckBox("查看并导出修复结果")
            self.use_repaired_result.setEnabled(False)
            self.use_repaired_result.toggled.connect(self._select_result_variant)
            model_layout.addWidget(self.use_repaired_result)
            self.simplify_button = QtWidgets.QPushButton("应用面片简化")
            self.simplify_button.setEnabled(False)
            self.simplify_button.clicked.connect(self._simplify_results)
            model_layout.addWidget(self.simplify_button)
            self.repair_result_button = QtWidgets.QPushButton("修复生成结果")
            self.repair_result_button.setEnabled(False)
            self.repair_result_button.setToolTip(
                "系统生成结果从权威隐式场重建；外部网格执行保守网格修复"
            )
            self.repair_result_button.clicked.connect(self._repair_results)
            model_layout.addWidget(self.repair_result_button)
            export_row = QtWidgets.QHBoxLayout()
            self.export_g_button = QtWidgets.QPushButton("导出 G STL")
            self.export_d_button = QtWidgets.QPushButton("导出 D STL")
            self.export_custom_button = QtWidgets.QPushButton("导出自定义 STL")
            self.export_all_button = QtWidgets.QPushButton("导出全部")
            self.export_transition_button = QtWidgets.QPushButton("导出过渡 STL")
            self.export_shell_union_button = QtWidgets.QPushButton("导出壳体融合 STL")
            for button in (
                self.export_g_button,
                self.export_d_button,
                self.export_custom_button,
                self.export_transition_button,
                self.export_shell_union_button,
                self.export_all_button,
            ):
                button.setEnabled(False)
                export_row.addWidget(button)
            self.export_g_button.clicked.connect(lambda: self._export_result("G"))
            self.export_d_button.clicked.connect(lambda: self._export_result("D"))
            self.export_custom_button.clicked.connect(
                lambda: self._export_result("Custom")
            )
            self.export_transition_button.clicked.connect(lambda: self._export_result("Transition"))
            self.export_shell_union_button.clicked.connect(
                lambda: self._export_result(SHELL_UNION_RESULT_KEY)
            )
            self.export_all_button.clicked.connect(self._export_all_results)
            model_layout.addLayout(export_row)
            self.export_selected_stl_button = QtWidgets.QPushButton("导出当前重建 STL")
            self.export_selected_stl_button.setEnabled(False)
            self.export_selected_stl_button.setToolTip(
                "导出“STL 重建对象”中当前选中并已完成重建的对象，包含解析基本设计域。"
            )
            self.export_selected_stl_button.clicked.connect(
                self._export_selected_stl_reconstruction
            )
            model_layout.addWidget(self.export_selected_stl_button)
            left_layout.addWidget(model_group)
            left_layout.addStretch()

            viewer_panel = QtWidgets.QWidget()
            viewer_panel.setObjectName("viewerPanel")
            viewer_layout = QtWidgets.QVBoxLayout(viewer_panel)
            viewer_layout.setContentsMargins(8, 8, 8, 8)
            viewer_layout.setSpacing(8)
            toolbar_frame = QtWidgets.QFrame()
            toolbar_frame.setObjectName("viewportToolbar")
            toolbar = QtWidgets.QHBoxLayout(toolbar_frame)
            toolbar.setContentsMargins(6, 5, 6, 5)
            toolbar.setSpacing(6)
            for label, method in (("前", "set_view_front"), ("左", "set_view_left"), ("顶", "set_view_top"), ("等轴", "set_view_isometric"), ("重置", "reset_view")):
                button = QtWidgets.QPushButton(label)
                button.setProperty("role", "toolbar")
                button.setToolTip(f"切换{label}视图")
                button.clicked.connect(getattr(self, method))
                toolbar.addWidget(button)
            self.field_viewer_toolbar_button = QtWidgets.QPushButton("场")
            self.field_viewer_toolbar_button.setProperty("role", "toolbar")
            self.field_viewer_toolbar_button.setCheckable(True)
            self.field_viewer_toolbar_button.setToolTip("切换场可视化（F）")
            self.field_viewer_toolbar_button.toggled.connect(
                self._on_field_viewer_toolbar_toggled
            )
            toolbar.addWidget(self.field_viewer_toolbar_button)
            self.field_viewer_shortcut = QtWidgets.QShortcut(
                QtGui.QKeySequence("F"),
                self,
            )
            self.field_viewer_shortcut.activated.connect(
                self._toggle_field_viewer_shortcut
            )
            toolbar.addStretch()
            viewer_layout.addWidget(toolbar_frame)
            self.viewer = InteractiveSectionViewer(self)
            self.viewer.show_ground_plane = False
            self.viewer.show_ground_grid = False
            self.viewer.enable_pbr = True
            self.viewer.enable_ssao_flag = True
            self._apply_viewport_background(BACKGROUND_PRESETS["CAD 亮白"])
            if hasattr(self.viewer, "set_field_surface_pick_callback"):
                self.viewer.set_field_surface_pick_callback(
                    self._place_field_plane_from_surface_pick
                )
            if hasattr(self.viewer, "set_scene_mesh_pick_callback"):
                self.viewer.set_scene_mesh_pick_callback(
                    self._on_scene_mesh_selected
                )
            if hasattr(self.viewer, "set_field_plane_probe_callback"):
                self.viewer.set_field_plane_probe_callback(
                    self._on_field_plane_probe_value_changed
                )
            viewer_layout.addWidget(self.viewer, 1)
            legend = QtWidgets.QLabel(
                "对象显示与材质可在左侧独立设置    |    剖切模式：左键操纵器，右键旋转视角，中键平移，滚轮缩放；"
                "普通模式：左键选择 · 右键旋转 · 中键平移 · 滚轮缩放"
            )
            legend.setObjectName("viewportHint")
            viewer_layout.addWidget(legend)
            layout.addWidget(viewer_panel, 1)
            for form_layout in self.findChildren(QtWidgets.QFormLayout):
                form_layout.setHorizontalSpacing(12)
                form_layout.setVerticalSpacing(8)
                form_layout.setLabelAlignment(
                    QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter
                )
            for group_box in self.findChildren(QtWidgets.QGroupBox):
                group_layout = group_box.layout()
                if group_layout is not None:
                    group_layout.setSpacing(8)
            for collapsible_group in self.findChildren(_CollapsibleGroupBox):
                collapsible_group.finalize_initial_state()
            self._register_workspace_controls()

        def _make_design_workspace_group(self):
            """Build controls for switching isolated Design documents."""

            group = QtWidgets.QGroupBox("设计工作区")
            form = QtWidgets.QFormLayout(group)
            self.design_selector = QtWidgets.QComboBox()
            self.design_selector.setToolTip(
                "只有当前设计会参与渲染、生成、检查和导出；其他设计保持隔离。"
            )
            self.design_selector.currentIndexChanged.connect(
                self._on_active_design_changed
            )
            self.rename_design_button = QtWidgets.QPushButton("重命名")
            self.rename_design_button.clicked.connect(self._rename_active_design)
            name_row = QtWidgets.QHBoxLayout()
            name_row.addWidget(self.design_selector, 1)
            name_row.addWidget(self.rename_design_button)
            form.addRow("当前设计", name_row)

            action_row = QtWidgets.QGridLayout()
            self.new_stl_design_button = QtWidgets.QPushButton("新建 STL 设计")
            self.new_stl_design_button.clicked.connect(
                lambda: self._browse_sole(create_design=True)
            )
            self.new_sphere_design_button = QtWidgets.QPushButton("球体")
            self.new_sphere_design_button.clicked.connect(
                lambda: self._create_analytic_design("sphere")
            )
            self.new_cylinder_design_button = QtWidgets.QPushButton("圆柱")
            self.new_cylinder_design_button.clicked.connect(
                lambda: self._create_analytic_design("cylinder")
            )
            self.new_box_design_button = QtWidgets.QPushButton("立方体")
            self.new_box_design_button.clicked.connect(
                lambda: self._create_analytic_design("box")
            )
            self.duplicate_design_button = QtWidgets.QPushButton("复制当前设计")
            self.duplicate_design_button.clicked.connect(self._duplicate_active_design)
            self.archive_design_button = QtWidgets.QPushButton("移至回收站")
            self.archive_design_button.setProperty("role", "danger")
            self.archive_design_button.clicked.connect(self._archive_active_design)
            action_row.addWidget(self.new_stl_design_button, 0, 0)
            action_row.addWidget(self.duplicate_design_button, 0, 1)
            action_row.addWidget(self.new_sphere_design_button, 1, 0)
            action_row.addWidget(self.new_cylinder_design_button, 1, 1)
            action_row.addWidget(self.new_box_design_button, 2, 0)
            action_row.addWidget(self.archive_design_button, 2, 1)
            form.addRow(action_row)

            self.archived_design_selector = QtWidgets.QComboBox()
            self.archived_design_selector.setToolTip(
                "已删除设计会保留在回收站中，直到明确永久删除。"
            )
            self.restore_design_button = QtWidgets.QPushButton("恢复设计")
            self.restore_design_button.clicked.connect(self._restore_archived_design)
            self.clear_archived_design_button = QtWidgets.QPushButton("清除设计")
            self.clear_archived_design_button.setProperty("role", "danger")
            self.clear_archived_design_button.setToolTip("永久清除回收站中选中的设计，需确认。")
            self.clear_archived_design_button.clicked.connect(self._clear_archived_design)
            restore_row = QtWidgets.QHBoxLayout()
            restore_row.addWidget(self.restore_design_button)
            restore_row.addWidget(self.clear_archived_design_button)
            form.addRow("回收站", self.archived_design_selector)
            form.addRow(restore_row)

            project_row = QtWidgets.QHBoxLayout()
            self.open_workspace_button = QtWidgets.QPushButton("打开工程")
            self.open_workspace_button.clicked.connect(self._open_workspace)
            self.save_workspace_button = QtWidgets.QPushButton("保存工程")
            self.save_workspace_button.clicked.connect(self._save_workspace)
            project_row.addWidget(self.open_workspace_button)
            project_row.addWidget(self.save_workspace_button)
            form.addRow(project_row)
            return group

        def _make_cell_map_group(self):
            group = _CollapsibleGroupBox("Cell Map 查看", collapsed=True)
            form = QtWidgets.QFormLayout(group)
            self.cell_map_kind = QtWidgets.QComboBox()
            for kind, label in (
                ("G", "G Cell Map"),
                ("D", "D Cell Map"),
                ("IWP", "IWP Cell Map"),
                ("Primitive", "Primitive Cell Map"),
                ("Neovius", "Neovius Cell Map"),
            ):
                self.cell_map_kind.addItem(label, kind)
            self.cell_map_kind.addItem("自定义晶胞 Cell Map", "Custom")
            self.cell_map_kind.currentIndexChanged.connect(
                self._on_cell_map_preview_kind_changed
            )
            form.addRow("查看对象", self.cell_map_kind)
            self.cell_map_info = QtWidgets.QLabel("选择对象后查看规则 Cell Map")
            self.cell_map_info.setObjectName("helper")
            self.cell_map_info.setWordWrap(True)
            form.addRow("状态", self.cell_map_info)
            self.cell_map_interactive = QtWidgets.QCheckBox("交互调整 UVW Frame")
            self.cell_map_interactive.setChecked(False)
            self.cell_map_interactive.setToolTip(
                "在视口中拖动三轴箭头平移 Cell Map 原点，拖动旋转环调整 U/V/W 方向；"
                "交互只更新参数和预览，不会自动重新生成晶格。"
            )
            self.cell_map_interactive.toggled.connect(
                self._on_cell_map_interactive_changed
            )
            form.addRow(self.cell_map_interactive)
            self.cell_map_show_button = QtWidgets.QPushButton("显示 Cell Map")
            self.cell_map_show_button.setToolTip(
                "只显示设计域和对应的规则 Cell Map 网格"
            )
            self.cell_map_show_button.clicked.connect(
                lambda: self._show_cell_map(reset_view=True)
            )
            self.cell_map_reset_button = QtWidgets.QPushButton("重置当前 Frame")
            self.cell_map_reset_button.setToolTip(
                "恢复设计域最小点原点和世界 X/Y/Z 方向"
            )
            self.cell_map_reset_button.clicked.connect(
                self._reset_current_cell_map_frame
            )
            action_row = QtWidgets.QHBoxLayout()
            action_row.addWidget(self.cell_map_show_button)
            action_row.addWidget(self.cell_map_reset_button)
            form.addRow(action_row)
            return group

        def _on_cell_map_preview_kind_changed(self, *_args) -> None:
            if self.cell_map_interactive.isChecked():
                self._activate_cell_map_editor_kind(self.cell_map_kind.currentData())
            if getattr(self.viewer, "_cell_map_preview", None) is not None:
                self._show_cell_map(reset_view=False)

        def _activate_cell_map_editor_kind(self, kind: LatticeKind) -> None:
            """Expose the selected Cell Map family's controls in the shared panel."""

            if kind not in TPMS_KINDS:
                return
            combo = self.tpms_widgets["kind"]
            index = combo.findData(kind)
            if index >= 0 and combo.currentIndex() != index:
                combo.setCurrentIndex(index)

        def _on_cell_map_interactive_changed(self, enabled: bool) -> None:
            disable_transform = getattr(
                self.viewer,
                "disable_cell_map_transform",
                None,
            )
            if not enabled:
                if disable_transform is not None:
                    disable_transform()
                return
            if self.field_viewer_enabled.isChecked():
                self.field_viewer_enabled.setChecked(False)
            if self.contour_enabled.isChecked():
                self.contour_enabled.setChecked(False)
            if self.section_enabled.isChecked():
                self.section_enabled.setChecked(False)
            kind = self.cell_map_kind.currentData()
            self._activate_cell_map_editor_kind(kind)
            self._show_cell_map(
                reset_view=getattr(self.viewer, "_cell_map_preview", None) is None
            )

        def _disable_cell_map_interaction(self) -> None:
            control = getattr(self, "cell_map_interactive", None)
            if control is not None and control.isChecked():
                control.blockSignals(True)
                control.setChecked(False)
                control.blockSignals(False)
            disable_transform = getattr(
                self.viewer,
                "disable_cell_map_transform",
                None,
            )
            if disable_transform is not None:
                disable_transform()

        @staticmethod
        def _cell_map_gizmo_bounds(cell_map: CellMap) -> np.ndarray:
            """Size the manipulator from one cell instead of the full map."""

            origin = np.asarray(cell_map.frame.origin, dtype=np.float64)
            handle_extent = max(
                2.0 * float(np.max(cell_map.spacing_mm)),
                1.0,
            )
            half = 0.5 * handle_extent
            return np.asarray(
                (
                    origin[0] - half,
                    origin[0] + half,
                    origin[1] - half,
                    origin[1] + half,
                    origin[2] - half,
                    origin[2] + half,
                ),
                dtype=np.float64,
            )

        def _sync_cell_map_gizmo(self, cell_map: CellMap) -> None:
            enable_transform = getattr(
                self.viewer,
                "enable_cell_map_transform",
                None,
            )
            disable_transform = getattr(
                self.viewer,
                "disable_cell_map_transform",
                None,
            )
            if not self.cell_map_interactive.isChecked() or enable_transform is None:
                if disable_transform is not None:
                    disable_transform()
                return
            basis = np.asarray(cell_map.frame.axes, dtype=np.float64).T
            gizmo = getattr(self.viewer, "_cell_map_gizmo", None)
            if gizmo is None:
                self._cell_map_gizmo_sync_in_progress = True
                try:
                    enable_transform(
                        self._cell_map_gizmo_bounds(cell_map),
                        np.asarray(cell_map.frame.origin, dtype=np.float64),
                        basis,
                        self._on_cell_map_gizmo_changed,
                    )
                finally:
                    self._cell_map_gizmo_sync_in_progress = False
                return
            gizmo.origin = np.asarray(cell_map.frame.origin, dtype=np.float64).copy()
            gizmo.basis = basis.copy()
            gizmo.update_geometry()
            gizmo.add_to_scene()

        def _on_cell_map_gizmo_changed(
            self,
            origin: np.ndarray,
            basis: np.ndarray,
        ) -> None:
            if self._cell_map_gizmo_sync_in_progress:
                return
            kind = self.cell_map_kind.currentData()
            self._activate_cell_map_editor_kind(kind)
            try:
                basis_array = np.asarray(basis, dtype=np.float64)
                frame = CellMapFrame.from_origin_axes(
                    origin,
                    basis_array[:, 0],
                    basis_array[:, 1],
                )
            except (ValueError, IndexError) as exc:
                self.cell_map_info.setText(f"交互变换无效：{exc}")
                return
            widgets = self._lattice_widgets(kind)
            follow = widgets["frame_origin_follow"]
            follow.blockSignals(True)
            follow.setChecked(False)
            follow.blockSignals(False)
            for control in widgets["frame_inputs"]["origin"]:
                control.setEnabled(True)
            for key, values in (
                ("origin", frame.origin),
                ("u", frame.u_axis),
                ("v", frame.v_axis),
            ):
                for control, value in zip(widgets["frame_inputs"][key], values):
                    control.blockSignals(True)
                    control.setValue(float(value))
                    control.blockSignals(False)
            self._on_cell_map_frame_changed(kind)

        def _reset_current_cell_map_frame(self, _checked: bool = False) -> None:
            kind = self.cell_map_kind.currentData()
            self._activate_cell_map_editor_kind(kind)
            self._reset_cell_map_frame(kind)

        def _show_cell_map(self, *, reset_view: bool = True):
            if not self.sole_mesh.faces.size:
                return
            kind = self.cell_map_kind.currentData()
            parameters = self._params(kind)
            try:
                cell_map = self._cell_map_for_active_design(parameters)
                domain_layer = MeshData(
                    self.sole_mesh.vertices,
                    self.sole_mesh.faces,
                    color=(0.63, 0.66, 0.70, 1.0),
                    metallic=0.0,
                    roughness=0.68,
                    cache_key=("domain", id(self.sole_mesh)),
                )
                clear_field_plane = getattr(self.viewer, "clear_field_plane", None)
                if clear_field_plane is not None:
                    clear_field_plane()
                set_overlays = getattr(self.viewer, "set_scene_overlay_meshes", None)
                if set_overlays is not None:
                    set_overlays([], rebuild_scene=False)
                self.viewer.set_cell_map_preview(
                    cell_map,
                    domain_layer,
                    reset_view=reset_view,
                )
                self._sync_cell_map_gizmo(cell_map)
                counts = " × ".join(str(value) for value in cell_map.cell_counts)
                extent = " × ".join(f"{value:.2f}" for value in cell_map.extent_mm)
                target = tuple(round(value, 2) for value in cell_map.requested_spacing_mm)
                actual = tuple(round(value, 2) for value in cell_map.spacing_mm)
                mode = "贴合设计域" if cell_map.boundary_mode == "fit_bounds" else "完整晶胞扩展"
                signed_range = " / ".join(
                    f"{start}:{start + count}"
                    for start, count in zip(cell_map.index_min, cell_map.cell_counts)
                )
                frame_origin = tuple(round(value, 3) for value in cell_map.frame.origin)
                self.cell_map_info.setText(
                    f"{kind}：{mode}；{counts} 个单元；范围 {extent} mm；"
                    f"目标 {target} mm；实际 {actual} mm；"
                    f"UVW 索引 {signed_range}；原点 {frame_origin} mm"
                )
                self.status.setText(f"正在查看 {kind} Cell Map")
            except ValueError as exc:
                QtWidgets.QMessageBox.warning(self, "Cell Map 参数错误", str(exc))

        def _make_material_combo(self, target: str):
            combo = QtWidgets.QComboBox()
            for name, color in MATERIAL_PRESETS.items():
                combo.addItem(name, color)
            default_color = self.material_colors[target]
            default_index = next(
                (index for index in range(combo.count()) if combo.itemData(index) == default_color),
                0,
            )
            combo.setCurrentIndex(default_index)
            combo.currentIndexChanged.connect(
                lambda index, key=target: self._on_material_changed(key, combo.itemData(index))
            )
            return combo

        def _on_background_changed(self, index: int):
            color = self.background_combo.itemData(index)
            if color:
                self._apply_viewport_background(color)
                self.viewer.render()

        def _apply_viewport_background(
            self,
            color: str | tuple[str, str] | list[str],
        ) -> None:
            if isinstance(color, (tuple, list)) and len(color) == 2:
                try:
                    self.viewer.set_background(color[0], top=color[1])
                except TypeError as exc:
                    if "unexpected keyword argument 'top'" not in str(exc):
                        raise
                    self.viewer.set_background(color[0])
                return
            self.viewer.set_background(str(color))

        def _on_render_quality_changed(self, index: int):
            quality = self.render_quality_combo.itemData(index)
            if not quality or not hasattr(self, "viewer"):
                return
            self.viewer.set_render_quality(quality)
            self.status.setText(f"渲染质量已切换为{self.render_quality_combo.currentText()}")
            self._refresh_scene(reset_view=False)

        def _on_material_changed(self, target: str, color):
            self.material_colors[target] = tuple(float(value) for value in color)
            self._refresh_scene(reset_view=False)

        def _make_cell_map_frame_box(self, kind: LatticeKind):
            frame_box = _CollapsibleGroupBox(
                "Cell Map Frame (UVW)",
                collapsed=True,
            )
            frame_box.setMinimumHeight(212)
            frame_box.setSizePolicy(
                QtWidgets.QSizePolicy.Preferred,
                QtWidgets.QSizePolicy.Fixed,
            )
            frame_layout = QtWidgets.QGridLayout(frame_box)
            frame_layout.setContentsMargins(8, 8, 8, 8)
            frame_layout.setHorizontalSpacing(6)
            frame_layout.setVerticalSpacing(4)
            for row in (1, 2, 3):
                frame_layout.setRowMinimumHeight(row, 28)
            frame_layout.setRowMinimumHeight(4, 24)
            frame_layout.setRowMinimumHeight(5, 34)
            frame_layout.setRowMinimumHeight(6, 32)
            for column, axis_name in enumerate(("X", "Y", "Z"), start=1):
                axis_label = QtWidgets.QLabel(axis_name)
                axis_label.setAlignment(QtCore.Qt.AlignCenter)
                frame_layout.addWidget(axis_label, 0, column)
            frame_values = {
                "origin": (0.0, 0.0, 0.0),
                "u": (1.0, 0.0, 0.0),
                "v": (0.0, 1.0, 0.0),
            }
            frame_labels = {"origin": "原点", "u": "U 轴", "v": "V 轴"}
            frame_inputs = {}
            for row, key in enumerate(("origin", "u", "v"), start=1):
                frame_layout.addWidget(QtWidgets.QLabel(frame_labels[key]), row, 0)
                controls = []
                for column, value in enumerate(frame_values[key], start=1):
                    control = QtWidgets.QDoubleSpinBox()
                    control.setRange(-1_000_000.0, 1_000_000.0)
                    control.setDecimals(FRAME_INPUT_DECIMALS)
                    control.setSingleStep(0.1 if key == "origin" else 0.05)
                    control.setValue(value)
                    control.setKeyboardTracking(False)
                    control.setMinimumWidth(72)
                    control.valueChanged.connect(
                        lambda _value, lattice_kind=kind: self._on_cell_map_frame_changed(
                            lattice_kind
                        )
                    )
                    frame_layout.addWidget(control, row, column)
                    controls.append(control)
                frame_inputs[key] = controls
            frame_origin_follow = QtWidgets.QCheckBox("原点随 U/V 自动定位")
            frame_origin_follow.setChecked(True)
            frame_origin_follow.setToolTip(
                "开启时，原点自动移动到设计域在当前 UVW 坐标系下的最小角点；"
                "关闭后可手动编辑原点以控制晶格相位"
            )
            for control in frame_inputs["origin"]:
                control.setEnabled(False)
            frame_origin_follow.toggled.connect(
                lambda enabled, lattice_kind=kind: self._on_frame_origin_mode_changed(
                    lattice_kind,
                    enabled,
                )
            )
            frame_layout.addWidget(frame_origin_follow, 4, 0, 1, 4)
            frame_status = QtWidgets.QLabel("W 轴由 U × V 自动计算")
            frame_status.setObjectName("helper")
            frame_status.setWordWrap(True)
            frame_layout.addWidget(frame_status, 5, 0, 1, 4)
            frame_reset = QtWidgets.QPushButton("恢复默认 Frame")
            frame_reset.setToolTip(
                "原点恢复为设计域包围盒最小点，U/V/W 恢复为世界 X/Y/Z"
            )
            frame_reset.clicked.connect(
                lambda _checked=False, lattice_kind=kind: self._reset_cell_map_frame(
                    lattice_kind
                )
            )
            frame_layout.addWidget(frame_reset, 6, 0, 1, 4)
            return (
                frame_box,
                frame_inputs,
                frame_status,
                frame_reset,
                frame_origin_follow,
            )

        def _make_custom_cell_group(self):
            group = QtWidgets.QGroupBox("自定义 STL 晶胞")
            form = QtWidgets.QFormLayout(group)
            enabled = QtWidgets.QCheckBox("参与生成")
            enabled.setChecked(False)
            enabled.setToolTip("使用输入 STL 的实体几何作为一个周期晶胞")
            source_path = QtWidgets.QLineEdit()
            source_path.setReadOnly(True)
            source_path.setPlaceholderText("请选择水密的晶胞 STL")
            browse = QtWidgets.QPushButton("选择 STL")
            browse.clicked.connect(self._browse_custom_cell)
            source_row = QtWidgets.QHBoxLayout()
            source_row.addWidget(source_path, 1)
            source_row.addWidget(browse)
            source_status = QtWidgets.QLabel("尚未选择晶胞 STL")
            source_status.setObjectName("helper")
            source_status.setWordWrap(True)
            boundary_mode = QtWidgets.QComboBox()
            boundary_mode.addItem("贴合设计域", "fit_bounds")
            boundary_mode.addItem("完整晶胞扩展", "complete_cells")
            boundary_mode.setToolTip(
                "默认世界 Frame 可贴合设计域；自定义 Frame 自动使用完整晶胞覆盖"
            )
            cells = {}
            defaults = {"x": 12.0, "y": 12.0, "z": 8.0}
            for axis in ("x", "y", "z"):
                cell = QtWidgets.QDoubleSpinBox()
                cell.setRange(0.5, 100.0)
                cell.setDecimals(2)
                cell.setSingleStep(0.5)
                cell.setValue(defaults[axis])
                cell.setToolTip(
                    "输入 STL 的对应源轴将缩放到该 Cell Map 周期，单位 mm"
                )
                cell.valueChanged.connect(self._update_recommendation)
                cells[axis] = cell
            target_feature = QtWidgets.QDoubleSpinBox()
            target_feature.setRange(0.0, 100.0)
            target_feature.setDecimals(3)
            target_feature.setSingleStep(0.1)
            target_feature.setValue(0.0)
            target_feature.setSpecialValueText("使用源模型厚度")
            target_feature.setToolTip(
                "0 表示保留源 STL 厚度；正值通过隐式等距偏移设置目标特征厚度，单位 mm"
            )
            target_feature.valueChanged.connect(self._update_recommendation)
            bridge_row = QtWidgets.QHBoxLayout()
            bridge_checks = {}
            for direction in ("U", "V", "W"):
                check = QtWidgets.QCheckBox(direction)
                check.setToolTip(
                    f"勾选后在周期接缝 {direction} 方向增加局部桥接材料；不勾选则只保留原始晶胞"
                )
                check.toggled.connect(self._update_recommendation)
                bridge_row.addWidget(check)
                bridge_checks[direction] = check
            bridge_row.addStretch(1)
            bridge_depth = QtWidgets.QDoubleSpinBox()
            bridge_depth.setRange(0.0, 100.0)
            bridge_depth.setDecimals(3)
            bridge_depth.setSingleStep(0.1)
            bridge_depth.setValue(0.0)
            bridge_depth.setSpecialValueText("自动")
            bridge_depth.setToolTip(
                "接缝桥接层向每侧延伸的深度；0 使用保守自动值，单位 mm"
            )
            bridge_depth.valueChanged.connect(self._update_recommendation)
            frame_box, frame_inputs, frame_status, frame_reset, frame_origin_follow = (
                self._make_cell_map_frame_box("Custom")
            )
            form.addRow(enabled)
            form.addRow("晶胞模型", source_row)
            form.addRow("导入状态", source_status)
            form.addRow("Cell Map 边界", boundary_mode)
            form.addRow("目标尺寸 U (mm)", cells["x"])
            form.addRow("目标尺寸 V (mm)", cells["y"])
            form.addRow("目标尺寸 W (mm)", cells["z"])
            form.addRow("目标特征厚度 (mm)", target_feature)
            form.addRow("周期接缝桥接方向", bridge_row)
            form.addRow("桥接深度 (mm)", bridge_depth)
            form.addRow(frame_box)
            enabled.toggled.connect(self._update_recommendation)
            boundary_mode.currentIndexChanged.connect(self._update_recommendation)
            return group, {
                "enabled": enabled,
                "source_path": source_path,
                "source_status": source_status,
                "browse": browse,
                "cell": cells["x"],
                "cell_x": cells["x"],
                "cell_y": cells["y"],
                "cell_z": cells["z"],
                "target_feature": target_feature,
                "bridge_checks": bridge_checks,
                "bridge_depth": bridge_depth,
                "cell_map_mode": boundary_mode,
                "frame_inputs": frame_inputs,
                "frame_status": frame_status,
                "frame_reset": frame_reset,
                "frame_origin_follow": frame_origin_follow,
            }

        def _make_tpms_group(self, title: str, kind: TPMSKind, color):
            group = QtWidgets.QGroupBox(title)
            form = QtWidgets.QFormLayout(group)
            enabled = QtWidgets.QCheckBox("参与生成")
            enabled.setChecked(True)
            kind_combo = QtWidgets.QComboBox()
            kind_labels = {
                "G": "G / Gyroid",
                "D": "D / Diamond",
                "IWP": "IWP",
                "Primitive": "Primitive",
                "Neovius": "Neovius",
            }
            for value in ("G", "D", "IWP", "Primitive", "Neovius"):
                kind_combo.addItem(kind_labels[value], value)
            kind_combo.setCurrentIndex(max(kind_combo.findData(kind), 0))
            kind_combo.setToolTip("选择解析 TPMS 晶胞类型")
            kind_combo.currentIndexChanged.connect(self._on_tpms_kind_changed)
            kind_combo.currentIndexChanged.connect(self._update_recommendation)
            kind_combo.currentIndexChanged.connect(self._update_transition_lattice_summary)
            cells = {}
            boundary_mode = QtWidgets.QComboBox()
            boundary_mode.addItem("贴合设计域", "fit_bounds")
            boundary_mode.addItem("完整晶胞扩展", "complete_cells")
            boundary_mode.setToolTip(
                "贴合设计域时，外边界与模型包围盒重合，并以最接近目标尺寸的整数晶胞数计算实际尺寸"
            )
            default_cell = 10.0 if kind == "G" else 12.0
            for axis in ("x", "y", "z"):
                cell = QtWidgets.QDoubleSpinBox()
                cell.setRange(0.5, 100.0)
                cell.setDecimals(2)
                cell.setSingleStep(0.5)
                cell.setValue(default_cell)
                cell.setToolTip("该方向一个周期的物理长度，单位 mm；越小越密")
                cells[axis] = cell
            wall = QtWidgets.QDoubleSpinBox()
            wall.setRange(0.05, 20.0)
            wall.setDecimals(2)
            wall.setSingleStep(0.1)
            wall.setValue(0.9 if kind == "G" else 1.0)
            wall.setToolTip("晶格实体壁的目标厚度，单位 mm")
            level = QtWidgets.QDoubleSpinBox()
            level.setRange(-2.0, 2.0)
            level.setDecimals(3)
            level.setSingleStep(0.05)
            level.setValue(0.0)
            level.setToolTip("TPMS 中心曲面的隐式场偏移，0 为标准对称曲面")
            frame_box, frame_inputs, frame_status, frame_reset, frame_origin_follow = (
                self._make_cell_map_frame_box(kind)
            )
            form.addRow(enabled)
            form.addRow("TPMS 类型", kind_combo)
            form.addRow("Cell Map 边界", boundary_mode)
            form.addRow("目标尺寸 U (mm)", cells["x"])
            form.addRow("目标尺寸 V (mm)", cells["y"])
            form.addRow("目标尺寸 W (mm)", cells["z"])
            form.addRow("壁厚 (mm)", wall)
            form.addRow("中心面 level", level)
            form.addRow(frame_box)
            gradient_enabled = QtWidgets.QCheckBox("启用梯度填充")
            gradient_enabled.setToolTip("沿 Cell Map 的 U/V/W 局部坐标改变壁厚和/或等值面偏移")
            gradient_axis = QtWidgets.QComboBox()
            for axis in ("U", "V", "W"):
                gradient_axis.addItem(axis, axis)
            gradient_axis.setCurrentIndex(2)
            gradient_mode = QtWidgets.QComboBox()
            for profile in GRADIENT_MODES:
                gradient_mode.addItem(
                    {
                        "linear": "线性",
                        "power": "幂函数",
                        "sigmoid": "Sigmoid",
                        "layered": "分层",
                    }[profile],
                    profile,
                )
            gradient_power = QtWidgets.QDoubleSpinBox()
            gradient_power.setRange(0.1, 20.0)
            gradient_power.setDecimals(2)
            gradient_power.setValue(5.0)
            gradient_layers = QtWidgets.QSpinBox()
            gradient_layers.setRange(2, 512)
            gradient_layers.setValue(5)
            gradient_sharpness = QtWidgets.QDoubleSpinBox()
            gradient_sharpness.setRange(0.1, 100.0)
            gradient_sharpness.setDecimals(2)
            gradient_sharpness.setValue(10.0)
            gradient_resolution_strategy = QtWidgets.QComboBox()
            gradient_resolution_strategy.addItem("自动推荐", "automatic")
            gradient_resolution_strategy.addItem("MATLAB Def 规则", "matlab_def")
            gradient_resolution_strategy.setToolTip(
                "MATLAB Def 规则按每个半晶胞的采样间隔数计算："
                "体素间距 = 最小晶胞尺寸 / (2 × Def)；"
                "仅在启用梯度填充且未手动指定 STL 间距时生效"
            )
            gradient_def = QtWidgets.QSpinBox()
            gradient_def.setRange(1, 4096)
            gradient_def.setValue(28)
            gradient_def.setSuffix(" samples/half-cell")
            gradient_def.setToolTip(
                "MATLAB 原型中的 Def：每个半晶胞长度上的采样间隔数；"
                "数值越大，体素越小、精度越高、内存和计算量越大"
            )
            gradient_resolution_strategy.setEnabled(False)
            gradient_def.setEnabled(False)
            thickness_gradient = QtWidgets.QCheckBox("启用壁厚梯度")
            thickness_gradient.setChecked(True)
            thickness_soft = QtWidgets.QDoubleSpinBox()
            thickness_soft.setRange(0.05, 20.0)
            thickness_soft.setDecimals(3)
            thickness_soft.setValue(0.6)
            thickness_stiff = QtWidgets.QDoubleSpinBox()
            thickness_stiff.setRange(0.05, 20.0)
            thickness_stiff.setDecimals(3)
            thickness_stiff.setValue(1.8)
            offset_gradient = QtWidgets.QCheckBox("启用等值面偏移梯度")
            offset_gradient.setChecked(True)
            offset_soft = QtWidgets.QDoubleSpinBox()
            offset_soft.setRange(-2.0, 2.0)
            offset_soft.setDecimals(3)
            offset_soft.setValue(-0.5)
            offset_stiff = QtWidgets.QDoubleSpinBox()
            offset_stiff.setRange(-2.0, 2.0)
            offset_stiff.setDecimals(3)
            offset_stiff.setValue(-0.15)
            wall_method = QtWidgets.QComboBox()
            wall_method.addItem("梯度归一化", "gradient_normalized")
            wall_method.addItem("场值阈值", "field_threshold")
            wall_method.setToolTip("梯度归一化保持物理壁厚；场值阈值与 MATLAB 原型的阈值定义一致")
            manual_counts_enabled = QtWidgets.QCheckBox("手动设置晶胞数量")
            manual_counts = []
            count_row = QtWidgets.QHBoxLayout()
            for axis in "UVW":
                count = QtWidgets.QSpinBox()
                count.setRange(1, 100000)
                count.setValue(1)
                count.setSuffix(f" {axis}")
                count.valueChanged.connect(self._update_recommendation)
                count_row.addWidget(count)
                manual_counts.append(count)
            for control in (
                gradient_enabled,
                gradient_axis,
                gradient_mode,
                gradient_power,
                gradient_layers,
                gradient_sharpness,
                gradient_resolution_strategy,
                gradient_def,
                thickness_gradient,
                thickness_soft,
                thickness_stiff,
                offset_gradient,
                offset_soft,
                offset_stiff,
                wall_method,
                manual_counts_enabled,
            ):
                signal = getattr(control, "valueChanged", None)
                if signal is None:
                    signal = getattr(control, "currentIndexChanged", None)
                if signal is None:
                    signal = getattr(control, "toggled", None)
                if signal is not None:
                    signal.connect(self._update_recommendation)

            def update_gradient_resolution_controls(*_args):
                enabled = gradient_enabled.isChecked()
                strategy_enabled = (
                    enabled
                    and gradient_resolution_strategy.currentData() == "matlab_def"
                )
                gradient_resolution_strategy.setEnabled(enabled)
                gradient_def.setEnabled(strategy_enabled)

            gradient_enabled.toggled.connect(update_gradient_resolution_controls)
            gradient_resolution_strategy.currentIndexChanged.connect(
                update_gradient_resolution_controls
            )
            gradient_title = QtWidgets.QLabel("梯度设计参数（可选）")
            gradient_title.setObjectName("sectionTitle")
            gradient_title.setToolTip(
                "仅在启用梯度填充后生效；基础晶胞尺寸和壁厚仍位于上方。"
            )
            form.addRow(gradient_title)
            form.addRow("梯度填充", gradient_enabled)
            form.addRow("梯度方向", gradient_axis)
            form.addRow("梯度曲线", gradient_mode)
            form.addRow("幂指数", gradient_power)
            form.addRow("分层数量", gradient_layers)
            form.addRow("梯度 Sigmoid 锐度", gradient_sharpness)
            form.addRow("梯度分辨率策略", gradient_resolution_strategy)
            form.addRow("MATLAB Def", gradient_def)
            form.addRow("壁厚梯度", thickness_gradient)
            form.addRow("软侧壁厚 (mm)", thickness_soft)
            form.addRow("硬侧壁厚 (mm)", thickness_stiff)
            form.addRow("偏移梯度", offset_gradient)
            form.addRow("软侧 offset", offset_soft)
            form.addRow("硬侧 offset", offset_stiff)
            form.addRow("壁厚求值方法", wall_method)
            form.addRow(manual_counts_enabled)
            form.addRow("晶胞数量 U/V/W", count_row)
            for cell in cells.values():
                cell.valueChanged.connect(self._update_recommendation)
                cell.valueChanged.connect(self._update_transition_lattice_summary)
            boundary_mode.currentIndexChanged.connect(self._update_recommendation)
            boundary_mode.currentIndexChanged.connect(self._update_transition_lattice_summary)
            wall.valueChanged.connect(self._update_recommendation)
            level.valueChanged.connect(self._update_recommendation)
            wall.valueChanged.connect(self._update_transition_lattice_summary)
            return group, {
                "enabled": enabled,
                "kind": kind_combo,
                "cell": cells["x"],
                "cell_x": cells["x"],
                "cell_y": cells["y"],
                "cell_z": cells["z"],
                "cell_map_mode": boundary_mode,
                "wall": wall,
                "level": level,
                "color": color,
                "frame_inputs": frame_inputs,
                "frame_status": frame_status,
                "frame_reset": frame_reset,
                "frame_origin_follow": frame_origin_follow,
                "gradient_enabled": gradient_enabled,
                "gradient_axis": gradient_axis,
                "gradient_mode": gradient_mode,
                "gradient_power": gradient_power,
                "gradient_layers": gradient_layers,
                "gradient_sharpness": gradient_sharpness,
                "gradient_resolution_strategy": gradient_resolution_strategy,
                "gradient_def": gradient_def,
                "thickness_gradient": thickness_gradient,
                "thickness_soft": thickness_soft,
                "thickness_stiff": thickness_stiff,
                "offset_gradient": offset_gradient,
                "offset_soft": offset_soft,
                "offset_stiff": offset_stiff,
                "wall_method": wall_method,
                "manual_counts_enabled": manual_counts_enabled,
                "manual_counts": tuple(manual_counts),
            }

        def _default_tpms_parameter_state(self, kind: TPMSKind) -> TPMSParameterState:
            """Return independent defaults for an analytic TPMS family."""

            cell_size_mm = 10.0 if kind == "G" else 12.0
            wall_thickness_mm = 0.9 if kind == "G" else 1.0
            return {
                "cell_size_mm": (cell_size_mm, cell_size_mm, cell_size_mm),
                "wall_thickness_mm": wall_thickness_mm,
                "level": 0.0,
                "cell_map_mode": "fit_bounds",
                "frame_origin_follow": True,
                "frame_origin_mm": (0.0, 0.0, 0.0),
                "frame_u_axis": (1.0, 0.0, 0.0),
                "frame_v_axis": (0.0, 1.0, 0.0),
                "gradient_enabled": False,
                "gradient_axis": "W",
                "gradient_mode": "linear",
                "gradient_power": 5.0,
                "gradient_layers": 5,
                "gradient_sigmoid_sharpness": 10.0,
                "gradient_resolution_strategy": "automatic",
                "gradient_def_per_half_cell": 28,
                "use_thickness_gradient": True,
                "thickness_soft_mm": 0.6,
                "thickness_stiff_mm": 1.8,
                "use_offset_gradient": True,
                "offset_soft": -0.5,
                "offset_stiff": -0.15,
                "wall_thickness_method": "gradient_normalized",
                "manual_cell_counts_enabled": False,
                "manual_cell_counts": (1, 1, 1),
            }

        def _tpms_state_from_widgets(self) -> TPMSParameterState:
            """Capture every TPMS-specific control from the shared panel."""

            widgets = self.tpms_widgets
            return {
                "cell_size_mm": tuple(
                    float(widgets[f"cell_{axis}"].value())
                    for axis in ("x", "y", "z")
                ),
                "wall_thickness_mm": float(widgets["wall"].value()),
                "level": float(widgets["level"].value()),
                "cell_map_mode": widgets["cell_map_mode"].currentData(),
                "frame_origin_follow": widgets["frame_origin_follow"].isChecked(),
                "frame_origin_mm": tuple(
                    float(control.value())
                    for control in widgets["frame_inputs"]["origin"]
                ),
                "frame_u_axis": tuple(
                    float(control.value())
                    for control in widgets["frame_inputs"]["u"]
                ),
                "frame_v_axis": tuple(
                    float(control.value())
                    for control in widgets["frame_inputs"]["v"]
                ),
                "gradient_enabled": widgets["gradient_enabled"].isChecked(),
                "gradient_axis": widgets["gradient_axis"].currentData(),
                "gradient_mode": widgets["gradient_mode"].currentData(),
                "gradient_power": float(widgets["gradient_power"].value()),
                "gradient_layers": int(widgets["gradient_layers"].value()),
                "gradient_sigmoid_sharpness": float(
                    widgets["gradient_sharpness"].value()
                ),
                "gradient_resolution_strategy": widgets[
                    "gradient_resolution_strategy"
                ].currentData(),
                "gradient_def_per_half_cell": int(widgets["gradient_def"].value()),
                "use_thickness_gradient": widgets["thickness_gradient"].isChecked(),
                "thickness_soft_mm": float(widgets["thickness_soft"].value()),
                "thickness_stiff_mm": float(widgets["thickness_stiff"].value()),
                "use_offset_gradient": widgets["offset_gradient"].isChecked(),
                "offset_soft": float(widgets["offset_soft"].value()),
                "offset_stiff": float(widgets["offset_stiff"].value()),
                "wall_thickness_method": widgets["wall_method"].currentData(),
                "manual_cell_counts_enabled": widgets[
                    "manual_counts_enabled"
                ].isChecked(),
                "manual_cell_counts": tuple(
                    int(control.value()) for control in widgets["manual_counts"]
                ),
            }

        @staticmethod
        def _set_combo_data(control, value: object) -> None:
            index = control.findData(value)
            if index < 0:
                raise ValueError(f"unsupported combo value: {value!r}")
            control.blockSignals(True)
            control.setCurrentIndex(index)
            control.blockSignals(False)

        @staticmethod
        def _set_control_value(control, value: object) -> None:
            control.blockSignals(True)
            control.setValue(value)
            control.blockSignals(False)

        def _sync_tpms_gradient_resolution_controls(self) -> None:
            widgets = self.tpms_widgets
            enabled = widgets["gradient_enabled"].isChecked()
            widgets["gradient_resolution_strategy"].setEnabled(enabled)
            widgets["gradient_def"].setEnabled(
                enabled
                and widgets["gradient_resolution_strategy"].currentData()
                == "matlab_def"
            )

        def _restore_tpms_parameter_state(self, kind: TPMSKind) -> None:
            """Restore one family state into the shared, visible TPMS panel."""

            state = self._tpms_parameter_states[kind]
            widgets = self.tpms_widgets
            for axis, value in zip(("x", "y", "z"), state["cell_size_mm"]):
                self._set_control_value(widgets[f"cell_{axis}"], value)
            self._set_control_value(widgets["wall"], state["wall_thickness_mm"])
            self._set_control_value(widgets["level"], state["level"])
            self._set_combo_data(widgets["cell_map_mode"], state["cell_map_mode"])
            for key, state_key in (
                ("origin", "frame_origin_mm"),
                ("u", "frame_u_axis"),
                ("v", "frame_v_axis"),
            ):
                for control, value in zip(widgets["frame_inputs"][key], state[state_key]):
                    self._set_control_value(control, value)
            widgets["frame_origin_follow"].blockSignals(True)
            widgets["frame_origin_follow"].setChecked(
                bool(state["frame_origin_follow"])
            )
            widgets["frame_origin_follow"].blockSignals(False)
            for control in widgets["frame_inputs"]["origin"]:
                control.setEnabled(not bool(state["frame_origin_follow"]))
            for key, state_key in (
                ("gradient_enabled", "gradient_enabled"),
                ("thickness_gradient", "use_thickness_gradient"),
                ("offset_gradient", "use_offset_gradient"),
                ("manual_counts_enabled", "manual_cell_counts_enabled"),
            ):
                widgets[key].blockSignals(True)
                widgets[key].setChecked(bool(state[state_key]))
                widgets[key].blockSignals(False)
            for key, state_key in (
                ("gradient_axis", "gradient_axis"),
                ("gradient_mode", "gradient_mode"),
                ("gradient_resolution_strategy", "gradient_resolution_strategy"),
                ("wall_method", "wall_thickness_method"),
            ):
                self._set_combo_data(widgets[key], state[state_key])
            for key, state_key in (
                ("gradient_power", "gradient_power"),
                ("gradient_layers", "gradient_layers"),
                ("gradient_sharpness", "gradient_sigmoid_sharpness"),
                ("gradient_def", "gradient_def_per_half_cell"),
                ("thickness_soft", "thickness_soft_mm"),
                ("thickness_stiff", "thickness_stiff_mm"),
                ("offset_soft", "offset_soft"),
                ("offset_stiff", "offset_stiff"),
            ):
                self._set_control_value(widgets[key], state[state_key])
            for control, value in zip(
                widgets["manual_counts"], state["manual_cell_counts"]
            ):
                self._set_control_value(control, value)
            self._sync_tpms_gradient_resolution_controls()
            if widgets["frame_origin_follow"].isChecked():
                self._update_followed_frame_origin(kind)
            self._sync_frame_boundary_mode(kind)
            self._tpms_parameter_states[kind] = self._tpms_state_from_widgets()

        def _on_tpms_kind_changed(self, _index: int = 0) -> None:
            """Save the old family panel state before restoring the new one."""

            if not hasattr(self, "_tpms_parameter_states"):
                return
            kind = self.tpms_widgets["kind"].currentData()
            if kind not in TPMS_KINDS:
                return
            previous = self._active_tpms_kind
            if kind == previous:
                return
            self._tpms_parameter_states[previous] = self._tpms_state_from_widgets()
            self._active_tpms_kind = kind
            self._restore_tpms_parameter_state(kind)
            if hasattr(self, "transition_first_controls"):
                self._sync_transition_operand_controls(self.transition_first_controls)
                self._sync_transition_operand_controls(self.transition_second_controls)
                self._update_transition_lattice_summary()

        def _make_transition_group(self):
            group = QtWidgets.QGroupBox("晶胞空间过渡（Ramp）")
            form = QtWidgets.QFormLayout(group)
            self._transition_form = form
            self.transition_enabled = QtWidgets.QCheckBox("参与过渡生成")
            self.transition_enabled.setChecked(False)
            self.transition_enabled.setToolTip("在任意两个已配置晶胞之间生成空间过渡")
            self.transition_enabled.toggled.connect(self._on_transition_ui_changed)
            form.addRow(self.transition_enabled)

            def make_operand_controls(title: str, default_kind: LatticeKind):
                panel = QtWidgets.QGroupBox(title)
                panel_form = QtWidgets.QFormLayout(panel)
                kind = QtWidgets.QComboBox()
                for text, value in (
                    ("G 晶胞", "G"),
                    ("D 晶胞", "D"),
                    ("IWP 晶胞", "IWP"),
                    ("Primitive 晶胞", "Primitive"),
                    ("Neovius 晶胞", "Neovius"),
                    ("自定义 STL 晶胞", "Custom"),
                ):
                    kind.addItem(text, value)
                kind.setCurrentIndex(max(kind.findData(default_kind), 0))
                inherit = QtWidgets.QCheckBox("使用上方该晶胞的参数")
                inherit.setChecked(True)
                sizes = []
                size_row = QtWidgets.QHBoxLayout()
                for axis in "UVW":
                    control = QtWidgets.QDoubleSpinBox()
                    control.setRange(0.5, 100.0)
                    control.setDecimals(2)
                    control.setSuffix(f" {axis}")
                    control.setValue(10.0)
                    control.setEnabled(False)
                    control.valueChanged.connect(self._on_transition_ui_changed)
                    size_row.addWidget(control)
                    sizes.append(control)
                feature = QtWidgets.QDoubleSpinBox()
                feature.setRange(0.05, 100.0)
                feature.setDecimals(3)
                feature.setSuffix(" mm")
                feature.setValue(1.0)
                feature.setEnabled(False)
                feature.valueChanged.connect(self._on_transition_ui_changed)
                panel_form.addRow("晶胞类型", kind)
                panel_form.addRow(inherit)
                panel_form.addRow("独立 U/V/W", size_row)
                panel_form.addRow("独立特征厚度", feature)
                controls = {
                    "panel": panel,
                    "kind": kind,
                    "inherit": inherit,
                    "sizes": tuple(sizes),
                    "feature": feature,
                }
                kind.currentIndexChanged.connect(
                    lambda _index, item=controls: self._sync_transition_operand_controls(item)
                )
                inherit.toggled.connect(
                    lambda _checked, item=controls: self._sync_transition_operand_controls(item)
                )
                return controls

            self.transition_first_controls = make_operand_controls("侧 A 晶胞", "G")
            self.transition_second_controls = make_operand_controls("侧 B 晶胞", "D")
            form.addRow(self.transition_first_controls["panel"])
            form.addRow(self.transition_second_controls["panel"])

            self.transition_driver_mode = QtWidgets.QComboBox()
            self.transition_driver_mode.addItem("平面", "plane")
            self.transition_driver_mode.addItem("场对象", "field")
            self.transition_driver_mode.currentIndexChanged.connect(
                self._on_transition_driver_changed
            )
            form.addRow("过渡区域定义", self.transition_driver_mode)

            self.transition_driver_combo = QtWidgets.QComboBox()
            self.transition_driver_combo.setToolTip(
                "选择一个独立解析场对象；其 SDF 值区间驱动两侧晶胞的 Ramp 融合。"
            )
            self.transition_driver_combo.currentIndexChanged.connect(
                self._on_transition_driver_changed
            )
            form.addRow("场对象", self.transition_driver_combo)

            self.transition_field_lower = QtWidgets.QDoubleSpinBox()
            self.transition_field_upper = QtWidgets.QDoubleSpinBox()
            for control, value in (
                (self.transition_field_lower, -2.0),
                (self.transition_field_upper, 2.0),
            ):
                control.setRange(-1_000_000.0, 1_000_000.0)
                control.setDecimals(3)
                control.setSingleStep(0.1)
                control.setSuffix(" mm")
                control.setValue(value)
                control.valueChanged.connect(self._on_transition_ui_changed)
            field_interval = QtWidgets.QHBoxLayout()
            field_interval.setContentsMargins(0, 0, 0, 0)
            field_interval.addWidget(self.transition_field_lower)
            field_interval.addWidget(QtWidgets.QLabel("至"))
            field_interval.addWidget(self.transition_field_upper)
            self.transition_field_interval_container = QtWidgets.QWidget()
            self.transition_field_interval_container.setLayout(field_interval)
            self.transition_field_interval_container.setToolTip(
                "SDF 小于等于下限时完全采用侧 A；大于等于上限时完全采用侧 B。"
            )
            form.addRow("场值区间", self.transition_field_interval_container)

            self.transition_axis = QtWidgets.QComboBox()
            self.transition_axis.addItems(["X 轴", "Y 轴", "Z 轴"])
            self.transition_axis.setCurrentIndex(1)
            self.transition_axis.currentIndexChanged.connect(self._on_transition_axis_changed)
            form.addRow("分界面轴", self.transition_axis)

            self.transition_position_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
            self.transition_position_slider.setRange(0, 1000)
            self.transition_position_slider.setValue(500)
            self.transition_position_slider.valueChanged.connect(self._on_transition_position_changed)
            self.transition_position_label = QtWidgets.QLabel("位置：0.00 mm")
            position_layout = QtWidgets.QVBoxLayout()
            position_layout.setContentsMargins(0, 0, 0, 0)
            position_layout.addWidget(self.transition_position_slider)
            position_layout.addWidget(self.transition_position_label)
            self.transition_position_container = QtWidgets.QWidget()
            self.transition_position_container.setLayout(position_layout)
            form.addRow("分界面位置", self.transition_position_container)

            self.transition_angle1 = QtWidgets.QDoubleSpinBox()
            self.transition_angle1.setRange(-180.0, 180.0)
            self.transition_angle1.setDecimals(1)
            self.transition_angle1.setSuffix("°")
            self.transition_angle1.valueChanged.connect(self._on_transition_ui_changed)
            self.transition_angle1_label = QtWidgets.QLabel("绕Y")
            form.addRow(self.transition_angle1_label, self.transition_angle1)

            self.transition_angle2 = QtWidgets.QDoubleSpinBox()
            self.transition_angle2.setRange(-180.0, 180.0)
            self.transition_angle2.setDecimals(1)
            self.transition_angle2.setSuffix("°")
            self.transition_angle2.valueChanged.connect(self._on_transition_ui_changed)
            self.transition_angle2_label = QtWidgets.QLabel("绕Z")
            form.addRow(self.transition_angle2_label, self.transition_angle2)

            self.transition_width = QtWidgets.QDoubleSpinBox()
            self.transition_width.setRange(0.05, 1000.0)
            self.transition_width.setDecimals(2)
            self.transition_width.setValue(10.0)
            self.transition_width.setSuffix(" mm")
            self.transition_width.valueChanged.connect(self._on_transition_ui_changed)
            self.transition_width.setToolTip("Ramp 从侧 A 完全过渡到侧 B 的物理距离")
            form.addRow("过渡带宽度", self.transition_width)

            self.transition_width_status = QtWidgets.QLabel()
            self.transition_width_status.setObjectName("helper")
            self.transition_width_status.setWordWrap(True)
            form.addRow("宽度建议", self.transition_width_status)

            self.transition_center_offset = QtWidgets.QDoubleSpinBox()
            self.transition_center_offset.setRange(-1000.0, 1000.0)
            self.transition_center_offset.setDecimals(2)
            self.transition_center_offset.setSuffix(" mm")
            self.transition_center_offset.valueChanged.connect(self._on_transition_ui_changed)
            self.transition_center_offset.setToolTip("Ramp 中心相对分界面的偏移")
            form.addRow("中心偏移", self.transition_center_offset)

            self.transition_weight = QtWidgets.QComboBox()
            for text, value in (
                ("自动（Smootherstep / C2）", "automatic"),
                ("线性 / C0", "linear"),
                ("Smoothstep / C1", "smoothstep"),
                ("Smootherstep / C2", "smootherstep"),
                ("Sigmoid / Tanh", "sigmoid"),
                ("余弦", "cosine"),
            ):
                self.transition_weight.addItem(text, value)
            self.transition_weight.setToolTip("控制平面距离场到混合权重的 Ramp 连续性")
            self.transition_weight.currentIndexChanged.connect(self._on_transition_weight_changed)
            form.addRow("Ramp 连续性", self.transition_weight)

            self.transition_sharpness = QtWidgets.QDoubleSpinBox()
            self.transition_sharpness.setRange(0.1, 10.0)
            self.transition_sharpness.setDecimals(2)
            self.transition_sharpness.setValue(1.0)
            self.transition_sharpness.setEnabled(False)
            self.transition_sharpness.valueChanged.connect(self._on_transition_ui_changed)
            self.transition_sharpness.setToolTip("仅用于 Sigmoid/Tanh，不改变过渡带物理宽度")
            form.addRow("Sigmoid 锐度", self.transition_sharpness)

            self.transition_minimum_feature = QtWidgets.QDoubleSpinBox()
            self.transition_minimum_feature.setRange(0.0, 100.0)
            self.transition_minimum_feature.setDecimals(3)
            self.transition_minimum_feature.setSuffix(" mm")
            self.transition_minimum_feature.setSpecialValueText("自动")
            self.transition_minimum_feature.setToolTip("0 自动取两侧特征厚度的较小值")
            self.transition_minimum_feature.valueChanged.connect(self._on_transition_ui_changed)
            form.addRow("过渡最小特征厚度", self.transition_minimum_feature)

            self.transition_registration = QtWidgets.QCheckBox("自动局部配准")
            self.transition_registration.setChecked(True)
            self.transition_registration.setToolTip("仅在过渡带内调整局部相位，带外 Cell Map 不变")
            self.transition_registration.toggled.connect(self._on_transition_ui_changed)
            form.addRow(self.transition_registration)

            self.transition_topology = QtWidgets.QCheckBox("拓扑约束修正")
            self.transition_topology.setChecked(True)
            self.transition_topology.setToolTip("在过渡带内进行最小幅度增减材以改善连通和特征厚度")
            self.transition_topology.toggled.connect(self._on_transition_ui_changed)
            form.addRow(self.transition_topology)

            self.transition_side = QtWidgets.QComboBox()
            self.transition_side.addItems(["负侧 A / 正侧 B", "负侧 B / 正侧 A"])
            self.transition_side.currentIndexChanged.connect(self._on_transition_ui_changed)
            form.addRow("晶格侧分配", self.transition_side)

            self.transition_side_info = QtWidgets.QLabel()
            self.transition_side_info.setObjectName("helper")
            form.addRow("当前分配", self.transition_side_info)

            self.transition_first_info = QtWidgets.QLabel()
            self.transition_first_info.setObjectName("helper")
            self.transition_first_info.setWordWrap(True)
            form.addRow("侧 A", self.transition_first_info)

            self.transition_second_info = QtWidgets.QLabel()
            self.transition_second_info.setObjectName("helper")
            self.transition_second_info.setWordWrap(True)
            form.addRow("侧 B", self.transition_second_info)

            self.show_transition_plane = QtWidgets.QCheckBox("显示过渡分界面")
            self.show_transition_plane.setChecked(True)
            self.show_transition_plane.stateChanged.connect(self._on_transition_plane_visibility_changed)
            form.addRow(self.show_transition_plane)
            self._transition_plane_row_fields = (
                self.transition_axis,
                self.transition_position_container,
                self.transition_angle1,
                self.transition_angle2,
                self.transition_width,
                self.transition_width_status,
                self.transition_center_offset,
            )
            self._sync_transition_operand_controls(self.transition_first_controls)
            self._sync_transition_operand_controls(self.transition_second_controls)
            self._sync_transition_field_objects()
            self._sync_transition_driver_ui()
            self._update_transition_lattice_summary()
            self._update_transition_angle_labels()
            return group

        def _make_field_objects_group(self):
            """Build the independent analytic field-object scene controls."""

            group = QtWidgets.QGroupBox("解析场对象")
            layout = QtWidgets.QVBoxLayout(group)
            helper = QtWidgets.QLabel(
                "球、有限圆柱和立方体均以解析 SDF 定义。预览网格仅用于摆放与选择；"
                "场对象可被场可视化和晶格过渡复用。"
            )
            helper.setObjectName("helper")
            helper.setWordWrap(True)
            layout.addWidget(helper)

            actions = QtWidgets.QHBoxLayout()
            for label, kind in (("添加球", "sphere"), ("添加圆柱", "cylinder"), ("添加立方体", "box")):
                button = QtWidgets.QPushButton(label)
                button.clicked.connect(
                    lambda _checked=False, value=kind: self._add_field_primitive(value)
                )
                actions.addWidget(button)
            layout.addLayout(actions)

            form = QtWidgets.QFormLayout()
            self.field_primitive_combo = QtWidgets.QComboBox()
            self.field_primitive_combo.currentIndexChanged.connect(
                self._on_field_primitive_selection_changed
            )
            form.addRow("对象", self.field_primitive_combo)

            self.field_primitive_visible = QtWidgets.QCheckBox("显示预览")
            self.field_primitive_visible.setChecked(True)
            self.field_primitive_visible.toggled.connect(self._on_field_primitive_visibility_changed)
            form.addRow(self.field_primitive_visible)

            self.field_primitive_name = QtWidgets.QLineEdit()
            self.field_primitive_name.editingFinished.connect(self._on_field_primitive_changed)
            form.addRow("名称", self.field_primitive_name)

            self.field_primitive_center = self._make_field_viewer_vector_inputs(
                (-1_000_000.0, 1_000_000.0),
                (0.0, 0.0, 0.0),
                0.1,
                self._on_field_primitive_changed,
            )
            form.addRow("中心 X / Y / Z", self._field_viewer_vector_row(self.field_primitive_center))
            self.field_primitive_rotation = self._make_field_viewer_vector_inputs(
                (-180.0, 180.0),
                (0.0, 0.0, 0.0),
                1.0,
                self._on_field_primitive_changed,
            )
            for control in self.field_primitive_rotation:
                control.setSuffix(" deg")
            form.addRow("旋转 X / Y / Z", self._field_viewer_vector_row(self.field_primitive_rotation))

            self.field_primitive_radius = QtWidgets.QDoubleSpinBox()
            self.field_primitive_height = QtWidgets.QDoubleSpinBox()
            for control, value in ((self.field_primitive_radius, 5.0), (self.field_primitive_height, 10.0)):
                control.setRange(0.001, 1_000_000.0)
                control.setDecimals(3)
                control.setSingleStep(0.25)
                control.setSuffix(" mm")
                control.setValue(value)
                control.valueChanged.connect(self._on_field_primitive_changed)
            form.addRow("半径", self.field_primitive_radius)
            form.addRow("圆柱高度", self.field_primitive_height)

            self.field_primitive_size = self._make_field_viewer_vector_inputs(
                (0.001, 1_000_000.0),
                (10.0, 10.0, 10.0),
                0.25,
                self._on_field_primitive_changed,
            )
            form.addRow("尺寸 X / Y / Z", self._field_viewer_vector_row(self.field_primitive_size))
            layout.addLayout(form)

            transform = QtWidgets.QHBoxLayout()
            self.field_primitive_manipulator = QtWidgets.QCheckBox("显示三轴操纵器")
            self.field_primitive_manipulator.setChecked(True)
            self.field_primitive_manipulator.toggled.connect(self._sync_primitive_gizmo)
            transform.addWidget(self.field_primitive_manipulator)
            self.delete_field_primitive_button = QtWidgets.QPushButton("删除对象")
            self.delete_field_primitive_button.setProperty("role", "danger")
            self.delete_field_primitive_button.clicked.connect(self._delete_selected_field_primitive)
            transform.addWidget(self.delete_field_primitive_button)
            layout.addLayout(transform)

            self.field_primitive_status = QtWidgets.QLabel("添加一个场对象后可在视图中选中并拖动操纵器。")
            self.field_primitive_status.setObjectName("helper")
            self.field_primitive_status.setWordWrap(True)
            layout.addWidget(self.field_primitive_status)
            self._set_field_primitive_controls_enabled(False)
            return group

        def _make_shell_group(self):
            """Build the design-domain shell and implicit-union controls."""

            group = QtWidgets.QGroupBox("设计域抽壳与晶格融合")
            layout = QtWidgets.QVBoxLayout(group)
            helper = QtWidgets.QLabel(
                "抽壳由设计域 SDF 的内侧偏移构造，外表面保持设计域边界。"
                "融合结果使用受设计域约束的隐式平滑并集。"
            )
            helper.setObjectName("helper")
            helper.setWordWrap(True)
            layout.addWidget(helper)

            form = QtWidgets.QFormLayout()
            self.shell_thickness = QtWidgets.QDoubleSpinBox()
            self.shell_thickness.setRange(0.02, 100.0)
            self.shell_thickness.setDecimals(3)
            self.shell_thickness.setSingleStep(0.10)
            self.shell_thickness.setValue(1.0)
            self.shell_thickness.setSuffix(" mm")
            self.shell_thickness.setToolTip(
                "从设计域外表面向内保留的实体壳厚。抽壳与融合结果均采用该物理厚度。"
            )
            self.shell_thickness.valueChanged.connect(self._on_shell_parameter_changed)
            form.addRow("壳厚", self.shell_thickness)

            self.shell_lattice_combo = QtWidgets.QComboBox()
            self.shell_lattice_combo.setToolTip(
                "选择一个已生成的权威晶格隐式体，作为壳体布尔并集的另一输入。"
            )
            self.shell_lattice_combo.currentIndexChanged.connect(
                self._on_shell_parameter_changed
            )
            form.addRow("融合晶格", self.shell_lattice_combo)

            self.shell_fusion_radius = QtWidgets.QDoubleSpinBox()
            self.shell_fusion_radius.setRange(0.0, 20.0)
            self.shell_fusion_radius.setDecimals(3)
            self.shell_fusion_radius.setSingleStep(0.10)
            self.shell_fusion_radius.setValue(0.0)
            self.shell_fusion_radius.setSuffix(" mm")
            self.shell_fusion_radius.setToolTip(
                "壳体与晶格的融合系数（物理融合半径）。0 mm 为严格布尔并集；"
                "值越大，交界越平滑，但会更明显地改变局部几何。"
            )
            self.shell_fusion_radius.valueChanged.connect(
                self._on_shell_parameter_changed
            )
            form.addRow("融合系数", self.shell_fusion_radius)
            layout.addLayout(form)

            self.shell_generate_button = QtWidgets.QPushButton("生成抽壳隐式体")
            self.shell_generate_button.setToolTip(
                "基于当前水密设计域生成内侧壳体，并加入隐式显示和 STL 重建结果。"
            )
            self.shell_generate_button.clicked.connect(
                lambda: self._start_shell_generation(combine=False)
            )
            layout.addWidget(self.shell_generate_button)

            self.shell_union_button = QtWidgets.QPushButton("生成壳体与晶格融合结果")
            self.shell_union_button.setProperty("role", "primary")
            self.shell_union_button.setToolTip(
                "将当前壳体参数与所选晶格进行受设计域限制的硬/平滑布尔并集。"
            )
            self.shell_union_button.clicked.connect(
                lambda: self._start_shell_generation(combine=True)
            )
            layout.addWidget(self.shell_union_button)

            self.shell_status = QtWidgets.QLabel("导入水密设计域后可生成抽壳结果")
            self.shell_status.setObjectName("helper")
            self.shell_status.setWordWrap(True)
            self.shell_status.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
            layout.addWidget(self.shell_status)
            self._sync_shell_lattice_options()
            return group

        def _sync_shell_lattice_options(self) -> None:
            """Synchronize selectable lattice inputs after result changes."""

            if "shell_lattice_combo" not in self.__dict__:
                return
            current = self.shell_lattice_combo.currentData()
            self.shell_lattice_combo.blockSignals(True)
            self.shell_lattice_combo.clear()
            for kind in self.implicit_generation_results:
                if kind not in (SHELL_RESULT_KEY, SHELL_UNION_RESULT_KEY):
                    self.shell_lattice_combo.addItem(kind, kind)
            for kind in self.backend_generation_handles:
                if kind not in (SHELL_RESULT_KEY, SHELL_UNION_RESULT_KEY):
                    self.shell_lattice_combo.addItem(f"{kind}（后端）", kind)
            if current is not None:
                index = self.shell_lattice_combo.findData(current)
                if index >= 0:
                    self.shell_lattice_combo.setCurrentIndex(index)
            self.shell_lattice_combo.blockSignals(False)
            has_domain = bool(self.sole_mesh.faces.size and self.sole_mesh.is_watertight)
            has_lattice = self.shell_lattice_combo.count() > 0
            self.shell_lattice_combo.setEnabled(has_lattice)
            self.shell_generate_button.setEnabled(has_domain)
            self.shell_union_button.setEnabled(has_domain and has_lattice)
            if not has_domain:
                self.shell_status.setText("抽壳需要先导入水密设计域 STL")
            elif not has_lattice:
                self.shell_status.setText("已可生成抽壳；生成晶格后可执行壳体融合")

        def _on_shell_parameter_changed(self, *_args) -> None:
            if "shell_status" not in self.__dict__:
                return
            if self.shell_lattice_combo.count() > 0:
                self.shell_status.setText("参数已更新；请重新生成壳体或融合结果以应用。")

        def _make_field_viewer_group(self):
            """Build the nTop-style 2-D Field Viewer controls."""

            group = QtWidgets.QGroupBox("场可视化")
            form = QtWidgets.QFormLayout(group)
            self.field_viewer_enabled = QtWidgets.QCheckBox("启用场可视化")
            self.field_viewer_enabled.setChecked(False)
            self.field_viewer_enabled.setToolTip(
                "在完整模型上叠加一个可移动的二维场平面；不修改隐式体或 STL。"
            )
            self.field_viewer_enabled.toggled.connect(self._on_field_viewer_mode_changed)
            form.addRow(self.field_viewer_enabled)

            self.field_viewer_show_object = QtWidgets.QCheckBox("显示三维对象")
            self.field_viewer_show_object.setChecked(True)
            self.field_viewer_show_object.setToolTip(
                "关闭后仅显示场平面、等值线和操纵器；场的采样与计算不受影响。"
            )
            self.field_viewer_show_object.toggled.connect(
                self._on_field_viewer_object_visibility_changed
            )
            form.addRow(self.field_viewer_show_object)

            self.field_viewer_target_combo = QtWidgets.QComboBox()
            self.field_viewer_target_combo.setToolTip(
                "显式选择需要检查的设计域、晶格、过渡或壳体结果。"
            )
            self.field_viewer_target_combo.currentIndexChanged.connect(
                self._on_field_viewer_target_changed
            )
            form.addRow("结果对象", self.field_viewer_target_combo)

            self.field_viewer_source_combo = QtWidgets.QComboBox()
            self.field_viewer_source_combo.addItem("隐式体渲染场", "render")
            self.field_viewer_source_combo.addItem("STL 重建场", "stl_reconstruction")
            self.field_viewer_source_combo.addItem("权威隐式体场", "authoritative")
            self.field_viewer_source_combo.setToolTip(
                "明确指定场数据来源，以便比较显示缓存、STL 重建采样和权威求值。"
            )
            self.field_viewer_source_combo.currentIndexChanged.connect(
                self._on_field_viewer_source_changed
            )
            form.addRow("场源", self.field_viewer_source_combo)

            self.field_viewer_domain_extension = QtWidgets.QDoubleSpinBox()
            self.field_viewer_domain_extension.setRange(0.0, 1_000_000.0)
            self.field_viewer_domain_extension.setDecimals(4)
            self.field_viewer_domain_extension.setSingleStep(1.0)
            self.field_viewer_domain_extension.setSuffix(" mm")
            self.field_viewer_domain_extension.setValue(0.0)
            self.field_viewer_domain_extension.setToolTip(
                "支持设计域及解析场对象的真实 SDF 外部查询。"
                "扩展查询包围盒，可配合平面尺寸查看更大范围，不修改模型或导出结果。"
            )
            self.field_viewer_domain_extension.valueChanged.connect(
                self._on_field_viewer_domain_extension_changed
            )
            form.addRow("场外扩展", self.field_viewer_domain_extension)

            self.field_viewer_colormap_combo = QtWidgets.QComboBox()
            self.field_viewer_colormap_combo.addItem("Implicit", "implicit")
            self.field_viewer_colormap_combo.addItem("Turbo", "turbo")
            self.field_viewer_colormap_combo.addItem("Distance Field", "distance")
            self.field_viewer_colormap_combo.currentIndexChanged.connect(
                self._on_field_viewer_style_changed
            )
            form.addRow("颜色映射", self.field_viewer_colormap_combo)

            self.field_viewer_range_mode = QtWidgets.QComboBox()
            self.field_viewer_range_mode.addItem("自动范围", "auto")
            self.field_viewer_range_mode.addItem("自定义范围", "custom")
            self.field_viewer_range_mode.currentIndexChanged.connect(
                self._on_field_viewer_range_mode_changed
            )
            form.addRow("数值范围", self.field_viewer_range_mode)

            self.field_viewer_range_min = QtWidgets.QDoubleSpinBox()
            self.field_viewer_range_max = QtWidgets.QDoubleSpinBox()
            for control, value in (
                (self.field_viewer_range_min, -1.0),
                (self.field_viewer_range_max, 1.0),
            ):
                control.setRange(-1_000_000.0, 1_000_000.0)
                control.setDecimals(5)
                control.setSingleStep(0.1)
                control.setValue(value)
                control.valueChanged.connect(self._on_field_viewer_style_changed)
            range_row = QtWidgets.QHBoxLayout()
            range_row.setContentsMargins(0, 0, 0, 0)
            range_row.addWidget(self.field_viewer_range_min)
            range_row.addWidget(QtWidgets.QLabel("至"))
            range_row.addWidget(self.field_viewer_range_max)
            form.addRow("范围下限 / 上限", range_row)

            self.field_viewer_opacity = QtWidgets.QDoubleSpinBox()
            self.field_viewer_opacity.setRange(0.05, 1.0)
            self.field_viewer_opacity.setDecimals(2)
            self.field_viewer_opacity.setSingleStep(0.05)
            self.field_viewer_opacity.setValue(0.72)
            self.field_viewer_opacity.valueChanged.connect(self._on_field_viewer_style_changed)
            form.addRow("平面不透明度", self.field_viewer_opacity)

            self.field_viewer_isolines = QtWidgets.QCheckBox("显示等值线")
            self.field_viewer_isolines.setChecked(True)
            self.field_viewer_isolines.toggled.connect(self._on_field_viewer_style_changed)
            form.addRow(self.field_viewer_isolines)

            self.field_viewer_isoline_interval = QtWidgets.QDoubleSpinBox()
            self.field_viewer_isoline_interval.setRange(0.0, 1_000_000.0)
            self.field_viewer_isoline_interval.setDecimals(5)
            self.field_viewer_isoline_interval.setSingleStep(0.05)
            self.field_viewer_isoline_interval.setValue(0.0)
            self.field_viewer_isoline_interval.setSpecialValueText("自适应")
            self.field_viewer_isoline_interval.setToolTip(
                "0 使用自适应间隔；正值使用固定的场值间隔。"
            )
            self.field_viewer_isoline_interval.valueChanged.connect(
                self._on_field_viewer_style_changed
            )
            form.addRow("等值线间隔", self.field_viewer_isoline_interval)

            self.field_viewer_probe = QtWidgets.QCheckBox("启用悬停探针")
            self.field_viewer_probe.setChecked(True)
            self.field_viewer_probe.toggled.connect(self._on_field_viewer_style_changed)
            form.addRow(self.field_viewer_probe)
            self.field_viewer_probe_value = QtWidgets.QLabel("—")
            self.field_viewer_probe_value.setObjectName("helper")
            self.field_viewer_probe_value.setToolTip(
                "鼠标悬停在有效场区域时显示该位置的隐式场值。"
            )
            form.addRow("探针值", self.field_viewer_probe_value)

            self.field_viewer_extent_mode = QtWidgets.QComboBox()
            self.field_viewer_extent_mode.addItem("完整用户平面", False)
            self.field_viewer_extent_mode.addItem("仅有效场范围", True)
            self.field_viewer_extent_mode.setToolTip(
                "完整用户平面保留所设平面尺寸；场有效边界外透明显示，"
                "因为该区域没有可信的有限场值。"
            )
            self.field_viewer_extent_mode.currentIndexChanged.connect(
                self._on_field_viewer_style_changed
            )
            form.addRow("平面显示范围", self.field_viewer_extent_mode)

            self.field_viewer_sampling_mode = QtWidgets.QComboBox()
            self.field_viewer_sampling_mode.addItem("自动推荐", "auto")
            self.field_viewer_sampling_mode.addItem("固定分辨率", "manual")
            self.field_viewer_sampling_mode.currentIndexChanged.connect(
                self._on_field_viewer_sampling_mode_changed
            )
            form.addRow("平面采样", self.field_viewer_sampling_mode)

            self.field_viewer_resolution = QtWidgets.QSpinBox()
            self.field_viewer_resolution.setRange(32, 2048)
            self.field_viewer_resolution.setSingleStep(32)
            self.field_viewer_resolution.setValue(256)
            self.field_viewer_resolution.setSuffix(" × 采样")
            self.field_viewer_resolution.setToolTip(
                "固定模式下平面 U、V 两个方向的采样点数。"
            )
            self.field_viewer_resolution.valueChanged.connect(
                self._on_field_viewer_geometry_changed
            )
            form.addRow("固定分辨率", self.field_viewer_resolution)

            self.field_viewer_center_inputs = self._make_field_viewer_vector_inputs(
                (-1_000_000.0, 1_000_000.0),
                (0.0, 0.0, 0.0),
                0.1,
                self._on_field_viewer_geometry_changed,
            )
            form.addRow("平面中心 X / Y / Z", self._field_viewer_vector_row(self.field_viewer_center_inputs))

            self.field_viewer_normal_inputs = self._make_field_viewer_vector_inputs(
                (-1.0, 1.0),
                (0.0, 0.0, 1.0),
                0.05,
                self._on_field_viewer_geometry_changed,
            )
            form.addRow("平面法向 X / Y / Z", self._field_viewer_vector_row(self.field_viewer_normal_inputs))

            self.field_viewer_size = QtWidgets.QDoubleSpinBox()
            self.field_viewer_size.setRange(0.001, 1_000_000.0)
            self.field_viewer_size.setDecimals(4)
            self.field_viewer_size.setSingleStep(1.0)
            self.field_viewer_size.setSuffix(" mm")
            self.field_viewer_size.valueChanged.connect(self._on_field_viewer_geometry_changed)
            form.addRow("平面尺寸", self.field_viewer_size)

            field_actions = QtWidgets.QHBoxLayout()
            reset_plane = QtWidgets.QPushButton("重置平面")
            reset_plane.clicked.connect(self._reset_field_viewer_plane)
            fit_view = QtWidgets.QPushButton("匹配当前视图")
            fit_view.clicked.connect(self._fit_field_plane_to_view)
            field_actions.addWidget(reset_plane)
            field_actions.addWidget(fit_view)
            form.addRow(field_actions)

            self.field_viewer_status = QtWidgets.QLabel("导入设计域或生成隐式体后可查看场")
            self.field_viewer_status.setObjectName("helper")
            self.field_viewer_status.setWordWrap(True)
            self.field_viewer_status.setTextInteractionFlags(
                QtCore.Qt.TextSelectableByMouse
            )
            form.addRow("状态", self.field_viewer_status)
            self._sync_field_viewer_range_controls()
            self._sync_field_viewer_sampling_controls()
            self._sync_field_viewer_domain_extension_control()
            self._sync_field_viewer_options()
            return group

        def _make_field_viewer_vector_inputs(
            self,
            limits: tuple[float, float],
            values: tuple[float, float, float],
            step: float,
            callback,
        ) -> list[QtWidgets.QDoubleSpinBox]:
            controls = []
            for value in values:
                control = QtWidgets.QDoubleSpinBox()
                control.setRange(*limits)
                control.setDecimals(5)
                control.setSingleStep(step)
                control.setValue(value)
                control.valueChanged.connect(callback)
                controls.append(control)
            return controls

        @staticmethod
        def _field_viewer_vector_row(controls):
            row = QtWidgets.QHBoxLayout()
            row.setContentsMargins(0, 0, 0, 0)
            for control in controls:
                row.addWidget(control)
            return row

        def _sync_field_viewer_options(self) -> None:
            """Keep explicit Field Viewer target choices aligned with results."""

            if "field_viewer_target_combo" not in self.__dict__:
                return
            combo = self.field_viewer_target_combo
            current = combo.currentData()
            document = self.design_workspace.active_document
            analytic_domain_primitive_id = (
                document.settings.get("analytic_domain_primitive_id")
                if document is not None
                and isinstance(document.domain, AnalyticDesignDomain)
                else None
            )
            combo.blockSignals(True)
            combo.clear()
            if self.domain_implicit_body is not None:
                domain_label = "设计域 STL"
                if document is not None and isinstance(
                    document.domain, AnalyticDesignDomain
                ):
                    domain_label = f"设计域：{document.domain.primitive.name}"
                combo.addItem(domain_label, "domain")
            for kind in self.implicit_generation_results:
                combo.addItem(kind, kind)
            for identifier, primitive in self.field_primitives.items():
                if identifier == analytic_domain_primitive_id:
                    continue
                combo.addItem(f"场对象：{primitive.name}", f"field:{identifier}")
            index = combo.findData(current)
            combo.setCurrentIndex(index if index >= 0 else (0 if combo.count() else -1))
            combo.setEnabled(combo.count() > 0)
            combo.blockSignals(False)
            # Repopulation is deliberately signal-blocked.  Apply target
            # restrictions once afterwards so a newly added primitive cannot
            # leave the Field Viewer set to its unavailable render cache.
            self._on_field_viewer_target_changed()
            available = combo.count() > 0
            self.field_viewer_enabled.setEnabled(available)
            if hasattr(self, "field_viewer_toolbar_button"):
                self.field_viewer_toolbar_button.setEnabled(available)
            if hasattr(self, "field_viewer_domain_extension"):
                self._sync_field_viewer_domain_extension_control()
            if not available and self.field_viewer_enabled.isChecked():
                self.field_viewer_enabled.setChecked(False)

        def _field_viewer_target_body(self) -> ImplicitBody | None:
            target = self.field_viewer_target_combo.currentData()
            if target == "domain":
                return self.domain_implicit_body
            if isinstance(target, str) and target.startswith("field:"):
                return self.field_primitives.get(target.removeprefix("field:")).as_implicit_body() if target.removeprefix("field:") in self.field_primitives else None
            result = self.implicit_generation_results.get(target)
            return result.body if result is not None else None

        def _field_viewer_target_field(self) -> SampledImplicitField | None:
            target = self.field_viewer_target_combo.currentData()
            if target == "domain":
                return self.domain_implicit_field
            if isinstance(target, str) and target.startswith("field:"):
                return None
            result = self.implicit_generation_results.get(target)
            if result is not None:
                return result.display_field
            backend_result = self.backend_generation_handles.get(target)
            return backend_result.display_field if backend_result is not None else None

        def _field_viewer_can_extend_domain(self) -> bool:
            """Whether the selected source can truthfully query exterior SDF."""

            return bool(
                self.field_viewer_source_combo.currentData() != "render"
                and self._field_viewer_target_body() is not None
            )

        def _field_viewer_domain_extension_mm(self) -> float:
            if not self._field_viewer_can_extend_domain():
                return 0.0
            return float(self.field_viewer_domain_extension.value())

        def _field_viewer_source_data(self) -> FieldSourceData:
            source = self.field_viewer_source_combo.currentData()
            if source == "render":
                field = self._field_viewer_target_field()
                if field is None:
                    raise RuntimeError("所选对象尚未准备隐式体渲染场")
                return field
            body = self._field_viewer_target_body()
            if body is None:
                raise RuntimeError("所选对象尚未准备权威隐式体场")
            target = self.field_viewer_target_combo.currentData()
            body = expand_implicit_body_bounds(
                body, self._field_viewer_domain_extension_mm()
            )
            if isinstance(target, str) and target.startswith("field:"):
                # Analytic primitives have a valid SDF everywhere.  Their
                # tight placement AABB must not clip a user-enlarged Field
                # Viewer plane as if it were an unknown sampled field.
                state = self._field_plane_state
                if state is not None:
                    corners = np.asarray(
                        [
                            state.center_mm
                            + sign_u * 0.5 * state.width_mm * state.u_axis
                            + sign_v * 0.5 * state.height_mm * state.v_axis
                            for sign_u in (-1.0, 1.0)
                            for sign_v in (-1.0, 1.0)
                        ],
                        dtype=np.float64,
                    )
                    bounds = np.vstack(
                        (
                            np.minimum(body.bounds[0], corners.min(axis=0)),
                            np.maximum(body.bounds[1], corners.max(axis=0)),
                        )
                    )
                    return ImplicitBody(
                        name=body.name,
                        bounds=bounds,
                        evaluate=body.evaluate,
                    )
            return body

        def _field_viewer_sampling_spacing(self) -> np.ndarray:
            """Resolve the source-specific physical detail used for auto sampling."""

            source = self.field_viewer_source_combo.currentData()
            target = self.field_viewer_target_combo.currentData()
            if source == "render":
                field = self._field_viewer_target_field()
                if field is None:
                    raise RuntimeError("所选对象尚未准备隐式体渲染场")
                return np.asarray(field.spacing, dtype=np.float64)
            if source == "authoritative":
                field = self._field_viewer_target_field()
                if field is not None:
                    return np.asarray(field.spacing, dtype=np.float64)
                return source_sampling_spacing(self._field_viewer_source_data())
            if source != "stl_reconstruction":
                raise ValueError(f"未知场源：{source}")
            if isinstance(target, str) and target.startswith("field:"):
                raise RuntimeError("解析场对象没有 STL 重建场；请选择权威隐式体场")
            if target == "domain":
                return np.full(3, float(self.export_tolerance.value()), dtype=np.float64)
            generation = self.implicit_generation_results.get(target)
            if generation is None:
                raise RuntimeError("当前没有可查看的隐式结果")
            export = self.stl_reconstruction_results.get(target)
            if export is not None:
                return np.asarray(export.spacing_mm, dtype=np.float64)
            if generation.cell_map is None or generation.minimum_feature_mm is None:
                raise RuntimeError("当前结果缺少 STL 重建采样信息")
            return resolve_export_spacing(
                generation.cell_map,
                float(self.export_tolerance.value()),
                float(generation.minimum_feature_mm),
                self._export_spacing_mode(),
            )

        def _initialize_field_viewer_plane(self) -> FieldPlaneState:
            source = self._field_viewer_source_data()
            bounds = np.asarray(source.bounds, dtype=np.float64)
            extent = bounds[1] - bounds[0]
            size = max(float(np.linalg.norm(extent)) * 1.05, 0.01)
            u_axis, v_axis = orthonormal_plane_axes(np.array((0.0, 0.0, 1.0)))
            state = FieldPlaneState(
                center_mm=bounds.mean(axis=0),
                u_axis=u_axis,
                v_axis=v_axis,
                width_mm=size,
                height_mm=size,
            )
            self._field_plane_state = state
            self._sync_field_viewer_plane_controls()
            return state

        def _sync_field_viewer_plane_controls(self) -> None:
            state = self._field_plane_state
            if state is None or "field_viewer_center_inputs" not in self.__dict__:
                return
            for control, value in zip(self.field_viewer_center_inputs, state.center_mm):
                control.blockSignals(True)
                control.setValue(float(value))
                control.blockSignals(False)
            for control, value in zip(self.field_viewer_normal_inputs, state.normal):
                control.blockSignals(True)
                control.setValue(float(value))
                control.blockSignals(False)
            self.field_viewer_size.blockSignals(True)
            self.field_viewer_size.setValue(float(state.width_mm))
            self.field_viewer_size.blockSignals(False)

        def _sync_field_viewer_range_controls(self) -> None:
            custom = self.field_viewer_range_mode.currentData() == "custom"
            self.field_viewer_range_min.setEnabled(custom)
            self.field_viewer_range_max.setEnabled(custom)

        def _sync_field_viewer_sampling_controls(self) -> None:
            manual = self.field_viewer_sampling_mode.currentData() == "manual"
            self.field_viewer_resolution.setEnabled(manual)

        def _field_viewer_resolution_shape(self) -> tuple[int, int]:
            state = self._field_plane_state
            if state is None:
                raise RuntimeError("场平面尚未初始化")
            if self.field_viewer_sampling_mode.currentData() == "manual":
                value = int(self.field_viewer_resolution.value())
                return value, value
            spacing = self._field_viewer_sampling_spacing()
            # The authoritative evaluator has no intrinsic voxel size.  Its
            # display or STL-reconstruction counterpart provides a bounded
            # sampling hint through ``_field_viewer_sampling_spacing``.
            desired_u = int(np.ceil(state.width_mm / float(np.min(spacing)))) + 1
            desired_v = int(np.ceil(state.height_mm / float(np.min(spacing)))) + 1
            return (
                int(np.clip(desired_u, 96, 512)),
                int(np.clip(desired_v, 96, 512)),
            )

        def _field_viewer_cache_key(
            self,
            source: FieldSourceData,
            resolution: tuple[int, int],
        ) -> tuple[object, ...]:
            state = self._field_plane_state
            if state is None:
                raise RuntimeError("场平面尚未初始化")
            return field_plane_cache_key(
                (
                    self.field_viewer_target_combo.currentData(),
                    self.field_viewer_source_combo.currentData(),
                    id(
                        self._field_viewer_target_field()
                        if self.field_viewer_source_combo.currentData() == "render"
                        else self._field_viewer_target_body()
                    ),
                    round(self._field_viewer_domain_extension_mm(), 8),
                ),
                state,
                resolution,
            )

        def _field_viewer_value_range(self, sample) -> tuple[float, float]:
            custom_range = None
            if self.field_viewer_range_mode.currentData() == "custom":
                custom_range = (
                    float(self.field_viewer_range_min.value()),
                    float(self.field_viewer_range_max.value()),
                )
            return resolve_field_value_range(
                sample,
                self.field_viewer_colormap_combo.currentData(),
                custom_range=custom_range,
            )

        def _refresh_field_viewer_plane(self) -> None:
            if not self.field_viewer_enabled.isChecked() or not hasattr(self, "viewer"):
                return
            try:
                source = self._field_viewer_source_data()
                state = self._field_plane_state or self._initialize_field_viewer_plane()
                resolution = self._field_viewer_resolution_shape()
                key = self._field_viewer_cache_key(source, resolution)
                sample = self._field_plane_cache.get(key)
                cache_message = "缓存命中"
                if sample is None:
                    sample = sample_field_plane(source, state, resolution)
                    self._field_plane_cache.put(key, sample)
                    cache_message = "已采样并缓存"
                value_range = self._field_viewer_value_range(sample)
                isolines = ()
                if self.field_viewer_isolines.isChecked():
                    isolines = extract_field_plane_isolines(
                        sample,
                        value_range=value_range,
                        interval=float(self.field_viewer_isoline_interval.value()),
                    )
                self.viewer.set_field_plane(
                    FieldPlaneRenderData(
                        sample=sample,
                        colormap=self.field_viewer_colormap_combo.currentData(),
                        value_range=value_range,
                        opacity=float(self.field_viewer_opacity.value()),
                        isolines=isolines,
                        show_probe=self.field_viewer_probe.isChecked(),
                        clip_to_source_bounds=bool(
                            self.field_viewer_extent_mode.currentData()
                        ),
                        cache_key=key,
                    )
                )
                self.viewer.set_field_surface_pick_callback(
                    self._place_field_plane_from_surface_pick
                )
                self._sync_field_viewer_gizmo()
                self.field_viewer_status.setText(
                    f"{cache_message}；{resolution[0]} × {resolution[1]} 采样；"
                    f"范围 {value_range[0]:.4g} 至 {value_range[1]:.4g}；"
                    f"等值线 {len(isolines)} 组"
                    + (
                        f"；场外扩展 {self._field_viewer_domain_extension_mm():.4g} mm"
                        if self._field_viewer_domain_extension_mm() > 0.0
                        else ""
                    )
                )
            except (RuntimeError, ValueError, TypeError) as exc:
                self.field_viewer_status.setText(f"场可视化失败：{exc}")

        def _on_field_plane_probe_value_changed(self, value: float | None) -> None:
            """Keep a stable textual probe readout visible beside the controls."""

            if value is None:
                self.field_viewer_probe_value.setText("—")
                return
            self.field_viewer_probe_value.setText(f"{float(value):.5g} mm")

        def _sync_field_viewer_gizmo(self) -> None:
            state = self._field_plane_state
            if state is None:
                return
            try:
                source = self._field_viewer_source_data()
            except RuntimeError:
                return
            source_bounds = np.asarray(source.bounds, dtype=np.float64)
            bounds = np.asarray(
                (
                    source_bounds[0, 0], source_bounds[1, 0],
                    source_bounds[0, 1], source_bounds[1, 1],
                    source_bounds[0, 2], source_bounds[1, 2],
                ),
                dtype=np.float64,
            )
            gizmo = getattr(self.viewer, "_field_plane_gizmo", None)
            if gizmo is None:
                self.viewer.enable_field_plane_gizmo(
                    bounds,
                    state.center_mm,
                    state.u_axis,
                    state.v_axis,
                    self._on_field_plane_gizmo_changed,
                )
                return
            gizmo.origin = state.center_mm.copy()
            gizmo.basis = np.column_stack(
                (state.u_axis, state.v_axis, state.normal)
            )
            gizmo.update_geometry()
            gizmo.add_to_scene()

        def _schedule_field_viewer_refresh(self) -> None:
            if self.field_viewer_enabled.isChecked():
                self._field_plane_refresh_timer.start()

        def _on_field_viewer_mode_changed(self, enabled: bool) -> None:
            if hasattr(self, "field_viewer_toolbar_button"):
                self.field_viewer_toolbar_button.blockSignals(True)
                self.field_viewer_toolbar_button.setChecked(enabled)
                self.field_viewer_toolbar_button.blockSignals(False)
            if not hasattr(self, "viewer"):
                return
            if not enabled:
                self._field_plane_refresh_timer.stop()
                self.viewer.disable_field_plane_gizmo()
                self.viewer.clear_field_plane()
                self._refresh_scene(reset_view=False)
                return
            if self.section_enabled.isChecked():
                self.section_enabled.setChecked(False)
            if self.contour_enabled.isChecked():
                self.contour_enabled.setChecked(False)
            if self._field_viewer_target_body() is None:
                self.field_viewer_enabled.blockSignals(True)
                self.field_viewer_enabled.setChecked(False)
                self.field_viewer_enabled.blockSignals(False)
                self.field_viewer_status.setText("请先选择可查看的隐式体对象")
                return
            self._field_plane_state = None
            self._initialize_field_viewer_plane()
            self._refresh_scene(reset_view=False)
            self._refresh_field_viewer_plane()

        def _on_field_viewer_toolbar_toggled(self, enabled: bool) -> None:
            if not self.field_viewer_enabled.isEnabled():
                self.field_viewer_toolbar_button.blockSignals(True)
                self.field_viewer_toolbar_button.setChecked(False)
                self.field_viewer_toolbar_button.blockSignals(False)
                return
            self.field_viewer_enabled.setChecked(enabled)

        def _toggle_field_viewer_shortcut(self) -> None:
            if self.field_viewer_enabled.isEnabled():
                self.field_viewer_enabled.setChecked(
                    not self.field_viewer_enabled.isChecked()
                )

        def _on_field_viewer_target_changed(self, _index: int = 0) -> None:
            target = self.field_viewer_target_combo.currentData()
            is_primitive = isinstance(target, str) and target.startswith("field:")
            is_backend_result = target in self.backend_generation_handles
            if is_primitive:
                authoritative_index = self.field_viewer_source_combo.findData("authoritative")
                self.field_viewer_source_combo.blockSignals(True)
                self.field_viewer_source_combo.setCurrentIndex(authoritative_index)
                self.field_viewer_source_combo.blockSignals(False)
            elif is_backend_result:
                render_index = self.field_viewer_source_combo.findData("render")
                self.field_viewer_source_combo.blockSignals(True)
                self.field_viewer_source_combo.setCurrentIndex(render_index)
                self.field_viewer_source_combo.blockSignals(False)
            self.field_viewer_source_combo.setEnabled(
                not is_primitive and not is_backend_result
            )
            self._field_plane_state = None
            self._sync_field_viewer_object_visibility_control()
            self._sync_field_viewer_domain_extension_control()
            if self.field_viewer_enabled.isChecked():
                self._initialize_field_viewer_plane()
                self._refresh_scene(reset_view=False)
                self._refresh_field_viewer_plane()

        def _on_field_viewer_source_changed(self, _index: int = 0) -> None:
            self._sync_field_viewer_domain_extension_control()
            if self.field_viewer_enabled.isChecked():
                self._schedule_field_viewer_refresh()

        def _sync_field_viewer_domain_extension_control(self) -> None:
            """Enable exterior queries for evaluators with a global SDF."""

            enabled = self._field_viewer_can_extend_domain()
            self.field_viewer_domain_extension.setEnabled(enabled)
            if enabled:
                self.field_viewer_domain_extension.setToolTip(
                    "在设计域、生成晶格或解析场对象包围盒外继续查询真实隐式场。"
                    "该参数只影响场查看，不修改模型或任何生成结果。"
                )
            else:
                self.field_viewer_domain_extension.setToolTip(
                    "权威隐式体场和 STL 重建场支持场外扩展；"
                    "隐式体渲染场是固定采样网格，不能可靠外推。"
                )

        def _on_field_viewer_domain_extension_changed(self, _value: float) -> None:
            if not self._field_viewer_can_extend_domain():
                return
            if self.field_viewer_enabled.isChecked():
                self._sync_field_viewer_gizmo()
                self._schedule_field_viewer_refresh()

        def _on_field_viewer_range_mode_changed(self, _index: int = 0) -> None:
            self._sync_field_viewer_range_controls()
            self._on_field_viewer_style_changed()

        def _on_field_viewer_sampling_mode_changed(self, _index: int = 0) -> None:
            self._sync_field_viewer_sampling_controls()
            self._on_field_viewer_geometry_changed()

        def _on_field_viewer_style_changed(self, *_args) -> None:
            self._schedule_field_viewer_refresh()

        def _on_field_viewer_object_visibility_changed(self, _visible: bool) -> None:
            """Show or hide scene geometry without changing Field Viewer data."""

            if not self.field_viewer_enabled.isChecked():
                return
            self._refresh_scene(reset_view=False)
            self._refresh_field_viewer_plane()

        def _on_field_viewer_geometry_changed(self, *_args) -> None:
            state = self._field_plane_state
            if state is not None:
                center = np.asarray(
                    [control.value() for control in self.field_viewer_center_inputs],
                    dtype=np.float64,
                )
                requested_normal = np.asarray(
                    [control.value() for control in self.field_viewer_normal_inputs],
                    dtype=np.float64,
                )
                try:
                    u_axis, v_axis = orthonormal_plane_axes(
                        requested_normal,
                        state.u_axis,
                    )
                    self._field_plane_state = FieldPlaneState(
                        center,
                        u_axis,
                        v_axis,
                        float(self.field_viewer_size.value()),
                        float(self.field_viewer_size.value()),
                    )
                    self._sync_field_viewer_gizmo()
                except ValueError:
                    return
            self._schedule_field_viewer_refresh()

        def _on_field_plane_gizmo_changed(
            self,
            origin: np.ndarray,
            basis: np.ndarray,
        ) -> None:
            state = self._field_plane_state
            if state is None:
                return
            self._field_plane_state = FieldPlaneState(
                origin,
                basis[:, 0],
                basis[:, 1],
                state.width_mm,
                state.height_mm,
            )
            self._sync_field_viewer_plane_controls()
            self._schedule_field_viewer_refresh()

        def _reset_field_viewer_plane(self) -> None:
            if not self.field_viewer_enabled.isChecked():
                return
            try:
                self._initialize_field_viewer_plane()
                self._refresh_field_viewer_plane()
            except (RuntimeError, ValueError) as exc:
                self.field_viewer_status.setText(f"无法重置场平面：{exc}")

        def _fit_field_plane_to_view(self) -> None:
            state = self._field_plane_state
            if state is None:
                return
            try:
                window_width, window_height = self.viewer.ren_win.GetSize()
                aspect = max(float(window_width) / max(float(window_height), 1.0), 1.0)
                size = float(self.viewer.camera.parallel_scale) * 2.0 * aspect
                self._field_plane_state = state.with_size(max(size, 0.001))
                self._sync_field_viewer_plane_controls()
                self._refresh_field_viewer_plane()
            except (AttributeError, RuntimeError, ValueError) as exc:
                self.field_viewer_status.setText(f"无法匹配当前视图：{exc}")

        def _place_field_plane_from_surface_pick(
            self,
            ray_origin: np.ndarray,
            ray_direction: np.ndarray,
        ) -> None:
            if self.section_enabled.isChecked() or self.contour_enabled.isChecked():
                return
            try:
                if not self.field_viewer_enabled.isChecked():
                    if self._field_viewer_target_body() is None:
                        self.field_viewer_status.setText("请先选择可查看的隐式体对象")
                        return
                    self.field_viewer_enabled.setChecked(True)
                source = self._field_viewer_source_data()
                spacing = float(np.min(self._field_viewer_sampling_spacing()))
                hit = field_surface_hit_from_ray(
                    source,
                    ray_origin,
                    ray_direction,
                    sample_spacing_mm=spacing,
                )
                if hit is None:
                    self.field_viewer_status.setText("右键未命中所选场的零等值面")
                    return
                state = self._field_plane_state or self._initialize_field_viewer_plane()
                u_axis, v_axis = orthonormal_plane_axes(hit.normal, state.u_axis)
                self._field_plane_state = FieldPlaneState(
                    hit.point_mm,
                    u_axis,
                    v_axis,
                    state.width_mm,
                    state.height_mm,
                )
                self._sync_field_viewer_plane_controls()
                self._refresh_field_viewer_plane()
                self.field_viewer_status.setText(
                    "已按右键命中的零等值面位置和法向更新场平面"
                )
            except (RuntimeError, ValueError, TypeError) as exc:
                self.field_viewer_status.setText(f"场表面定位失败：{exc}")

        def _make_contour_group(self):
            group = QtWidgets.QGroupBox("逐层等值线查看")
            form = QtWidgets.QFormLayout(group)

            self.contour_enabled = QtWidgets.QCheckBox("启用逐层等值线")
            self.contour_enabled.setChecked(False)
            self.contour_enabled.setToolTip(
                "仅显示所选隐式场在当前层面的等值线，不生成或修改 STL。"
            )
            self.contour_enabled.toggled.connect(self._on_contour_mode_changed)
            form.addRow(self.contour_enabled)

            self.contour_result_combo = QtWidgets.QComboBox()
            self.contour_result_combo.setToolTip("选择要查看的晶格结果对象。")
            self.contour_result_combo.currentIndexChanged.connect(
                self._on_contour_source_changed
            )
            form.addRow("结果对象", self.contour_result_combo)

            self.contour_source_combo = QtWidgets.QComboBox()
            self.contour_source_combo.addItem("隐式体渲染场", "render")
            self.contour_source_combo.addItem("STL 重建场", "stl_reconstruction")
            self.contour_source_combo.addItem("权威隐式体场", "authoritative")
            self.contour_source_combo.currentIndexChanged.connect(
                self._on_contour_source_changed
            )
            form.addRow("场源", self.contour_source_combo)

            self.contour_axis_combo = QtWidgets.QComboBox()
            self.contour_axis_combo.addItems(["X", "Y", "Z"])
            self.contour_axis_combo.setCurrentIndex(2)
            self.contour_axis_combo.currentIndexChanged.connect(
                self._on_contour_axis_changed
            )
            form.addRow("层方向", self.contour_axis_combo)

            self.contour_fixed_layer_enabled = QtWidgets.QCheckBox("使用固定层高")
            self.contour_fixed_layer_enabled.setChecked(False)
            self.contour_fixed_layer_enabled.setToolTip(
                "启用后，仅改变层与层之间的间距；每一层等值线的平面内采样精度保持不变。"
            )
            self.contour_fixed_layer_enabled.toggled.connect(
                self._on_contour_fixed_layer_changed
            )
            form.addRow(self.contour_fixed_layer_enabled)

            self.contour_fixed_layer_height = QtWidgets.QDoubleSpinBox()
            self.contour_fixed_layer_height.setRange(0.001, 1000.0)
            self.contour_fixed_layer_height.setDecimals(4)
            self.contour_fixed_layer_height.setSingleStep(0.1)
            self.contour_fixed_layer_height.setValue(0.5)
            self.contour_fixed_layer_height.setSuffix(" mm")
            self.contour_fixed_layer_height.setToolTip(
                "固定层高，单位为 mm。该参数只控制沿层方向的层位置，不改变层面内采样间距。"
            )
            self.contour_fixed_layer_height.setEnabled(False)
            self.contour_fixed_layer_height.valueChanged.connect(
                self._on_contour_fixed_layer_changed
            )
            form.addRow("固定层高", self.contour_fixed_layer_height)

            self.contour_layer_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
            self.contour_layer_slider.setRange(0, 1000)
            self.contour_layer_slider.setValue(0)
            self.contour_layer_slider.valueChanged.connect(
                self._refresh_layer_contours
            )
            self.contour_layer_label = QtWidgets.QLabel("层位置：—")
            layer_layout = QtWidgets.QVBoxLayout()
            layer_layout.setContentsMargins(0, 0, 0, 0)
            layer_layout.addWidget(self.contour_layer_slider)
            layer_layout.addWidget(self.contour_layer_label)
            form.addRow("层位置", layer_layout)

            self.contour_level = QtWidgets.QDoubleSpinBox()
            self.contour_level.setRange(-10.0, 10.0)
            self.contour_level.setDecimals(4)
            self.contour_level.setSingleStep(0.05)
            self.contour_level.setValue(0.0)
            self.contour_level.valueChanged.connect(self._refresh_layer_contours)
            form.addRow("等值面值", self.contour_level)

            self.contour_status = QtWidgets.QLabel("生成晶格后可查看逐层等值线")
            self.contour_status.setObjectName("helper")
            self.contour_status.setWordWrap(True)
            self.contour_status.setTextInteractionFlags(
                QtCore.Qt.TextSelectableByMouse
            )
            form.addRow("状态", self.contour_status)

            refresh_button = QtWidgets.QPushButton("刷新当前层")
            refresh_button.clicked.connect(self._refresh_layer_contours)
            form.addRow(refresh_button)
            self._sync_contour_result_options()
            return group

        def _sync_contour_result_options(self) -> None:
            if "contour_result_combo" not in self.__dict__:
                return
            current = self.contour_result_combo.currentData()
            self.contour_result_combo.blockSignals(True)
            self.contour_result_combo.clear()
            if self.domain_implicit_body is not None:
                self.contour_result_combo.addItem("设计域 STL", "domain")
            for kind in self.implicit_generation_results:
                self.contour_result_combo.addItem(kind, kind)
            for kind in self.backend_generation_handles:
                self.contour_result_combo.addItem(f"{kind}（后端显示场）", kind)
            if current is not None:
                index = self.contour_result_combo.findData(current)
                if index >= 0:
                    self.contour_result_combo.setCurrentIndex(index)
            self.contour_result_combo.blockSignals(False)
            enabled = self.contour_result_combo.count() > 0
            if not enabled and self.contour_enabled.isChecked():
                self.contour_enabled.blockSignals(True)
                self.contour_enabled.setChecked(False)
                self.contour_enabled.blockSignals(False)
            self.contour_enabled.setEnabled(enabled)
            if not enabled and self.contour_fixed_layer_enabled.isChecked():
                self.contour_fixed_layer_enabled.blockSignals(True)
                self.contour_fixed_layer_enabled.setChecked(False)
                self.contour_fixed_layer_enabled.blockSignals(False)
            self._sync_contour_fixed_layer_controls()

        def _sync_stl_reconstruction_target_options(self) -> None:
            """Populate the explicit STL reconstruction target selector."""

            if "stl_reconstruction_target_combo" not in self.__dict__:
                return
            combo = self.stl_reconstruction_target_combo
            current = combo.currentData()
            combo.blockSignals(True)
            combo.clear()
            combo.addItem("请选择重建对象", None)
            if self._analytic_design_reconstruction_target() is not None:
                combo.addItem("设计域（解析隐式体）", "domain")
            for kind in self.implicit_generation_results:
                combo.addItem(kind, kind)
            for kind in self.backend_generation_handles:
                combo.addItem(f"{kind}（后端显示场）", kind)
            index = combo.findData(current)
            combo.setCurrentIndex(index if index >= 0 else 0)
            combo.setEnabled(combo.count() > 1)
            combo.blockSignals(False)
            self._update_stl_grid_estimate()
            self._sync_precise_render_target_options()

        def _sync_precise_render_target_options(self) -> None:
            """Expose only current-document authoritative implicit bodies."""

            combo = getattr(self, "precise_render_target_combo", None)
            if combo is None:
                return
            current = combo.currentData()
            combo.blockSignals(True)
            combo.clear()
            combo.addItem("请选择隐式对象", None)
            if self.domain_implicit_body is not None:
                combo.addItem("设计域", "domain")
            for kind, generation in self.implicit_generation_results.items():
                if isinstance(generation.body, ImplicitBody):
                    combo.addItem(str(kind), str(kind))
            for kind in self.backend_generation_handles:
                combo.addItem(f"{kind}（后端）", kind)
            index = combo.findData(current)
            combo.setCurrentIndex(index if index >= 0 else 0)
            combo.blockSignals(False)
            self._sync_precise_render_resolution_controls()

        def _sync_precise_render_resolution_controls(self, *_args) -> None:
            combo = getattr(self, "precise_render_resolution_combo", None)
            if combo is None:
                return
            custom = combo.currentData() is None
            self.precise_render_width.setEnabled(custom)
            self.precise_render_height.setEnabled(custom)

        def _selected_precise_render_target(self) -> PreciseRenderTarget | None:
            combo = getattr(self, "precise_render_target_combo", None)
            if combo is None:
                return None
            key = combo.currentData()
            if key == "domain" and self.domain_implicit_body is not None:
                extent = np.diff(self.domain_implicit_body.bounds, axis=0).reshape(3)
                return PreciseRenderTarget(
                    self.domain_implicit_body,
                    # Imported domains do not carry a lattice wall-thickness
                    # scale.  Use a conservative surface-detail scale instead
                    # of treating the entire shoe thickness as one feature.
                    float(max(np.min(extent) / 128.0, 0.10)),
                    "domain",
                )
            generation = self.implicit_generation_results.get(key)
            if generation is None or not isinstance(generation.body, ImplicitBody):
                return None
            minimum_feature = generation.minimum_feature_mm
            if minimum_feature is None:
                minimum_feature = float(np.min(np.diff(generation.body.bounds, axis=0)))
            return PreciseRenderTarget(
                generation.body,
                float(max(minimum_feature, 1.0e-6)),
                str(key),
            )

        def _analytic_design_reconstruction_target(
            self,
        ) -> StlReconstructionTarget | None:
            """Return the direct SDF extraction target for an analytic Design domain."""

            document = self.design_workspace.active_document
            body = self.domain_implicit_body
            if (
                document is None
                or not isinstance(document.domain, AnalyticDesignDomain)
                or body is None
            ):
                return None
            extent = np.diff(body.bounds, axis=0).reshape(3)
            minimum_feature_mm = float(np.min(extent))
            if not np.isfinite(minimum_feature_mm) or minimum_feature_mm <= 0.0:
                return None
            # A design domain has no periodic cell or wall feature. The map
            # supplies only a stable extraction-grid origin; tolerance controls
            # its spacing directly through ``enforce_feature_limits=False``.
            extraction_map = CellMap.from_bounds(
                body.bounds,
                tuple(float(value) for value in extent),
                boundary_mode="fit_bounds",
            )
            return StlReconstructionTarget(
                body=body,
                extraction_map=extraction_map,
                minimum_feature_mm=minimum_feature_mm,
                enforce_feature_limits=False,
            )

        def _selected_stl_reconstruction_results(
            self,
        ) -> dict[str, StlReconstructionTarget]:
            """Return the one authoritative implicit body selected for STL extraction."""

            if "stl_reconstruction_target_combo" not in self.__dict__:
                return {}
            kind = self.stl_reconstruction_target_combo.currentData()
            if kind == "domain":
                target = self._analytic_design_reconstruction_target()
                return {"domain": target} if target is not None else {}
            generation = self.implicit_generation_results.get(kind)
            if generation is None:
                return {}
            if generation.cell_map is None or generation.minimum_feature_mm is None:
                return {}
            return {
                kind: StlReconstructionTarget(
                    body=generation.body,
                    extraction_map=generation.cell_map,
                    minimum_feature_mm=generation.minimum_feature_mm,
                )
            }

        def _on_stl_reconstruction_target_changed(self, _index: int = 0) -> None:
            self._update_stl_grid_estimate()
            if self.stl_thread is None:
                self._set_result_controls_enabled(bool(self.results))

        def _precise_render_camera(self) -> PreciseRenderCamera:
            camera = self.viewer.camera
            return PreciseRenderCamera(
                position=tuple(float(value) for value in camera.position),
                focal_point=tuple(float(value) for value in camera.focal_point),
                view_up=tuple(float(value) for value in camera.up),
                parallel_projection=bool(camera.parallel_projection),
                parallel_scale=float(camera.parallel_scale),
                view_angle_deg=float(camera.view_angle),
            )

        def _precise_render_settings(self, target: PreciseRenderTarget) -> PreciseRenderSettings:
            selection = self.precise_render_resolution_combo.currentData()
            if selection is None:
                width = int(self.precise_render_width.value())
                height = int(self.precise_render_height.value())
            else:
                width, height = (int(value) for value in selection)
            return PreciseRenderSettings(
                width_px=width,
                height_px=height,
                minimum_feature_mm=float(target.minimum_feature_mm),
                tile_size_px=128,
            )

        def _backend_precise_render_settings(
            self,
            generation: BackendGenerationResult,
        ) -> PreciseRenderSettings:
            selection = self.precise_render_resolution_combo.currentData()
            if selection is None:
                width = int(self.precise_render_width.value())
                height = int(self.precise_render_height.value())
            else:
                width, height = (int(value) for value in selection)
            return PreciseRenderSettings(
                width_px=width,
                height_px=height,
                minimum_feature_mm=float(
                    max(generation.minimum_feature_mm or 1.0, 1.0e-6)
                ),
                tile_size_px=128,
            )

        def _start_precise_render(self) -> None:
            if self.precise_render_thread is not None:
                return
            if self._document_switch_is_locked():
                return
            target = self._selected_precise_render_target()
            backend_generation = self.backend_generation_handles.get(
                self.precise_render_target_combo.currentData()
            )
            if target is None and backend_generation is None:
                QtWidgets.QMessageBox.warning(
                    self,
                    "无法精确渲染",
                    "请先选择一个当前设计中的权威隐式对象。",
                )
                return
            try:
                settings = (
                    self._precise_render_settings(target)
                    if target is not None
                    else self._backend_precise_render_settings(backend_generation)
                )
                camera = self._precise_render_camera()
            except (TypeError, ValueError, AttributeError) as exc:
                QtWidgets.QMessageBox.warning(self, "精确渲染参数错误", str(exc))
                return
            default_path = self.root / "build" / "exports" / (
                f"precise_render_{target.material_key if target is not None else backend_generation.kind}.png"
            )
            output_path, _ = QtWidgets.QFileDialog.getSaveFileName(
                self,
                "保存精确渲染图片",
                str(default_path),
                "PNG 图片 (*.png)",
            )
            if not output_path:
                return
            document = self.design_workspace.active_document
            if document is None:
                return
            color = self.material_colors.get(
                target.material_key if target is not None else backend_generation.kind,
                self.material_colors.get("domain", MATERIAL_PRESETS["CAD 银白"]),
            )
            background = self.background_combo.currentData()
            if isinstance(background, (tuple, list)) and len(background) == 2:
                lower_hex, upper_hex = background
            else:
                lower_hex = upper_hex = background

            def rgb(value: str) -> tuple[float, float, float]:
                text = str(value).lstrip("#")
                if len(text) != 6:
                    raise ValueError("背景颜色格式无效")
                return tuple(int(text[index:index + 2], 16) / 255.0 for index in (0, 2, 4))

            try:
                lower_rgb = rgb(lower_hex)
                upper_rgb = rgb(upper_hex)
            except ValueError as exc:
                QtWidgets.QMessageBox.warning(self, "背景颜色错误", str(exc))
                return
            if backend_generation is not None:
                self._start_backend_precise_render(
                    document,
                    backend_generation,
                    settings,
                    camera,
                    color,
                    lower_rgb,
                    upper_rgb,
                    output_path,
                )
                return
            self.precise_render_button.setEnabled(False)
            self.generate_button.setEnabled(False)
            self.progress.setValue(0)
            self.status.setText("精确渲染：准备权威隐式体")
            self.precise_render_thread = QtCore.QThread(self)
            self.precise_render_worker = PreciseRenderWorker(
                target,
                target_key=target.material_key,
                camera=camera,
                settings=settings,
                material=PreciseRenderMaterial(
                    color=tuple(float(value) for value in color),
                    **PBR_MATERIAL_PARAMETERS.get(target.material_key, {}),
                ),
                background_lower=lower_rgb,
                background_upper=upper_rgb,
                output_path=output_path,
                design_identifier=document.identifier,
                design_revision=document.revision,
            )
            self.precise_render_worker.moveToThread(self.precise_render_thread)
            self.precise_render_thread.started.connect(self.precise_render_worker.run)
            self.precise_render_worker.progress.connect(
                lambda message, value: (
                    self.status.setText(message),
                    self.progress.setValue(int(value * 100)),
                )
            )
            self.precise_render_worker.finished.connect(self._precise_render_finished)
            self.precise_render_worker.failed.connect(self._precise_render_failed)
            self.precise_render_worker.finished.connect(self.precise_render_thread.quit)
            self.precise_render_worker.failed.connect(self.precise_render_thread.quit)
            self.precise_render_thread.finished.connect(self._precise_render_cleanup)
            self.precise_render_thread.start()

        def _start_backend_precise_render(
            self,
            document,
            generation: BackendGenerationResult,
            settings: PreciseRenderSettings,
            camera: PreciseRenderCamera,
            color,
            lower_rgb,
            upper_rgb,
            output_path: str,
        ) -> None:
            client = self.backend_client
            if client is None:
                self._precise_render_failed(
                    document.identifier,
                    document.revision,
                    "local backend is unavailable",
                )
                return
            payload = {
                "generation_id": generation.generation_id,
                "camera": asdict(camera),
                "settings": asdict(settings),
                "material": {
                    "color": [float(value) for value in color],
                    **PBR_MATERIAL_PARAMETERS.get(generation.kind, {}),
                },
                "background_lower": list(lower_rgb),
                "background_upper": list(upper_rgb),
            }

            def result_reader(snapshot: object):
                if snapshot.result is None or len(snapshot.artifacts) != 1:
                    raise ValueError("precise render task returned an invalid artifact")
                return snapshot.result, client.download_artifact(
                    snapshot.artifacts[0].identifier
                )

            self.precise_render_button.setEnabled(False)
            self.generate_button.setEnabled(False)
            self.progress.setValue(0)
            self.precise_render_thread = QtCore.QThread(self)
            self.precise_render_worker = BackendTaskWorker(
                client,
                "render.precise",
                payload,
                result_reader,
            )
            self.precise_render_worker.moveToThread(self.precise_render_thread)
            self.precise_render_thread.started.connect(self.precise_render_worker.run)
            self.precise_render_worker.progress.connect(
                lambda message, value: (
                    self.status.setText(message),
                    self.progress.setValue(int(value * 100)),
                )
            )
            self.precise_render_worker.finished.connect(
                lambda outcome, identifier=document.identifier, revision=document.revision,
                kind=generation.kind, path=output_path: self._backend_precise_render_finished(
                    identifier, revision, kind, path, outcome
                )
            )
            self.precise_render_worker.failed.connect(
                lambda message, identifier=document.identifier, revision=document.revision: self._precise_render_failed(
                    identifier, revision, message
                )
            )
            self.precise_render_worker.finished.connect(self.precise_render_thread.quit)
            self.precise_render_worker.failed.connect(self.precise_render_thread.quit)
            self.precise_render_thread.finished.connect(self._precise_render_cleanup)
            self.precise_render_thread.start()

        def _backend_precise_render_finished(
            self,
            design_identifier: str,
            design_revision: int,
            kind: str,
            output_path: str,
            outcome,
        ) -> None:
            if getattr(self, "_is_closing", False):
                return
            document = self.design_workspace.documents.get(design_identifier)
            if document is None or document.revision != int(design_revision):
                self.status.setText("precise render result is stale")
                return
            result, image = outcome
            path = Path(output_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(image)
            report = result.get("report", {})
            self.precise_render_info.setText(
                f"{kind} precise render complete; backend PNG; "
                f"hits {int(report.get('hit_pixels', 0)):,}"
            )
            self.progress.setValue(100)
            self.status.setText(f"precise render saved: {path}")

        def _precise_render_finished(
            self,
            design_identifier: str,
            design_revision: int,
            target_key: str,
            output_path: str,
            result: PreciseRenderResult,
            cache_key: tuple[object, ...],
        ) -> None:
            if getattr(self, "_is_closing", False):
                return
            document = self.design_workspace.documents.get(design_identifier)
            if document is None or document.revision != int(design_revision):
                self.status.setText("精确渲染结果已过期，未写入当前设计。")
                return
            from PyQt5 import QtGui

            array = np.ascontiguousarray(result.image_rgb)
            height, width, _channels = array.shape
            qimage = QtGui.QImage(
                array.data,
                width,
                height,
                width * 3,
                QtGui.QImage.Format_RGB888,
            ).copy()
            if not qimage.save(output_path, "PNG"):
                self.status.setText("精确渲染完成，但 PNG 保存失败。")
                return
            cache = document.runtime.renderer_cache.setdefault(
                "precise_render_images", {}
            )
            if isinstance(cache, dict):
                cache[cache_key] = {
                    "path": output_path,
                    "report": result.report,
                    "image": array,
                }
                while len(cache) > 4:
                    cache.pop(next(iter(cache)))
            report = result.report
            self.precise_render_info.setText(
                f"{target_key} 精确渲染完成：{report.output_size_px[0]}×"
                f"{report.output_size_px[1]}；像素覆盖 {report.pixel_footprint_mm:.5f} mm；"
                f"求交容差 {report.surface_tolerance_mm:.5f} mm；"
                f"命中 {report.hit_pixels:,} 像素；耗时 {report.elapsed_seconds:.2f} s"
            )
            self.progress.setValue(100)
            self.status.setText(f"精确渲染已保存：{output_path}")

        def _precise_render_failed(
            self,
            _design_identifier: str,
            _design_revision: int,
            message: str,
        ) -> None:
            if getattr(self, "_is_closing", False):
                return
            self.status.setText("精确渲染失败：" + message)
            QtWidgets.QMessageBox.critical(self, "精确渲染失败", message)

        def _precise_render_cleanup(self) -> None:
            self.precise_render_worker = None
            self.precise_render_thread = None
            self.precise_render_button.setEnabled(True)
            self.generate_button.setEnabled(self.design_workspace.active_document is not None)

        def _sync_contour_fixed_layer_controls(self) -> None:
            """Keep the fixed-layer-height editor consistent with its switch."""

            if "contour_fixed_layer_enabled" not in self.__dict__:
                return
            self.contour_fixed_layer_height.setEnabled(
                self.contour_fixed_layer_enabled.isEnabled()
                and self.contour_fixed_layer_enabled.isChecked()
            )

        def _on_contour_fixed_layer_changed(self, *_args) -> None:
            self._sync_contour_fixed_layer_controls()
            self._update_contour_layer_label()
            if self.contour_enabled.isChecked():
                self._refresh_layer_contours()

        def _contour_layer_position(self) -> float:
            kind = self.contour_result_combo.currentData()
            if kind == "domain":
                body = self.domain_implicit_body
                if body is None:
                    raise RuntimeError("设计域隐式体尚未准备完成")
                axis = self.contour_axis_combo.currentText()
                positions = available_layer_positions(
                    body.bounds,
                    axis,
                    self._contour_layer_spacing(
                        None, self.contour_source_combo.currentData()
                    ),
                )
                position_index = int(
                    round(
                        self.contour_layer_slider.value()
                        / 1000.0
                        * max(len(positions) - 1, 0)
                    )
                )
                return float(positions[position_index])
            body, _field = self._contour_result_fields(kind)
            generation = self.implicit_generation_results.get(kind)
            backend_generation = self.backend_generation_handles.get(kind)
            if body is None and backend_generation is None:
                raise RuntimeError("当前没有可查看的隐式结果")
            axis = self.contour_axis_combo.currentText()
            index = "XYZ".index(axis)
            bounds = (
                body.bounds
                if body is not None
                else backend_generation.display_field.bounds
            )
            positions = available_layer_positions(
                bounds,
                axis,
                self._contour_layer_spacing(
                    generation or backend_generation,
                    self.contour_source_combo.currentData(),
                ),
            )
            position_index = int(
                round(
                    self.contour_layer_slider.value()
                    / 1000.0
                    * max(len(positions) - 1, 0)
                )
            )
            return float(positions[position_index])

        def _contour_result_fields(
            self,
            kind: str | None,
        ) -> tuple[ImplicitBody | None, SampledImplicitField | None]:
            if kind == "domain":
                return self.domain_implicit_body, self.domain_implicit_field
            generation = self.implicit_generation_results.get(kind)
            if generation is not None:
                return generation.body, generation.display_field
            backend_generation = self.backend_generation_handles.get(kind)
            if backend_generation is not None:
                return None, backend_generation.display_field
            return None, None

        def _update_contour_layer_label(self) -> None:
            try:
                self.contour_layer_label.setText(
                    f"层位置：{self._contour_layer_position():.4f} mm"
                )
            except (RuntimeError, ValueError):
                self.contour_layer_label.setText("层位置：—")

        def _on_contour_axis_changed(self, _index: int = 0) -> None:
            self._update_contour_layer_label()
            if self.contour_enabled.isChecked():
                self._refresh_layer_contours()

        def _on_contour_source_changed(self, _index: int = 0) -> None:
            kind = self.contour_result_combo.currentData()
            if (
                kind in self.backend_generation_handles
                and self.contour_source_combo.currentData() != "render"
            ):
                render_index = self.contour_source_combo.findData("render")
                self.contour_source_combo.blockSignals(True)
                self.contour_source_combo.setCurrentIndex(render_index)
                self.contour_source_combo.blockSignals(False)
            self._update_contour_layer_label()
            if self.contour_enabled.isChecked():
                self._refresh_layer_contours()

        def _on_contour_mode_changed(self, enabled: bool) -> None:
            if enabled:
                self._disable_cell_map_interaction()
                if self.field_viewer_enabled.isChecked():
                    self.field_viewer_enabled.setChecked(False)
                if self.section_enabled.isChecked():
                    self.section_enabled.blockSignals(True)
                    self.section_enabled.setChecked(False)
                    self.section_enabled.blockSignals(False)
                    self.viewer.disable_interactive_section()
                self._refresh_primitive_preview(sync_gizmo=True)
                self._refresh_layer_contours()
            else:
                self._refresh_scene(reset_view=False)

        def _contour_cache_key(self) -> tuple[object, ...]:
            kind = self.contour_result_combo.currentData()
            source = self.contour_source_combo.currentData()
            axis = self.contour_axis_combo.currentText()
            if kind == "domain":
                body = self.domain_implicit_body
                if body is None:
                    raise RuntimeError("设计域隐式体尚未准备完成")
                spacing = self._contour_sampling_spacing(None, source)
                return (
                    kind,
                    source,
                    id(body),
                    tuple(round(float(value), 9) for value in spacing),
                    axis,
                    bool(self.contour_fixed_layer_enabled.isChecked()),
                    round(float(self.contour_fixed_layer_height.value()), 9),
                    round(self._contour_layer_position(), 9),
                    round(float(self.contour_level.value()), 9),
                )
            body, _field = self._contour_result_fields(kind)
            generation = self.implicit_generation_results.get(kind)
            backend_generation = self.backend_generation_handles.get(kind)
            if generation is None and backend_generation is None:
                raise RuntimeError("当前没有可查看的隐式结果")
            active_generation = generation or backend_generation
            spacing = self._contour_sampling_spacing(active_generation, source)
            return (
                kind,
                source,
                id(body if body is not None else backend_generation.display_field.values),
                tuple(round(float(value), 9) for value in spacing),
                axis,
                bool(self.contour_fixed_layer_enabled.isChecked()),
                round(float(self.contour_fixed_layer_height.value()), 9),
                round(self._contour_layer_position(), 9),
                round(float(self.contour_level.value()), 9),
            )

        def _contour_sampling_spacing(
            self,
            generation: ImplicitGenerationResult | BackendGenerationResult | None,
            source: ContourFieldSource,
        ) -> np.ndarray:
            if generation is None:
                if source == "render" and self.domain_implicit_field is not None:
                    return np.asarray(self.domain_implicit_field.spacing, dtype=np.float64)
                if source == "authoritative" and self.domain_implicit_field is not None:
                    return np.asarray(self.domain_implicit_field.spacing, dtype=np.float64)
                if source == "stl_reconstruction":
                    return np.full(3, float(self.export_tolerance.value()), dtype=np.float64)
                raise RuntimeError("设计域隐式采样场尚未准备完成")
            if source in ("render", "authoritative"):
                if isinstance(generation, BackendGenerationResult) and source != "render":
                    raise RuntimeError("backend results expose display fields only")
                return np.asarray(generation.display_field.spacing, dtype=np.float64)
            if source != "stl_reconstruction":
                raise ValueError(f"未知等值线场源：{source}")
            export = self.stl_reconstruction_results.get(
                self.contour_result_combo.currentData()
            )
            if export is not None:
                return np.asarray(export.spacing_mm, dtype=np.float64)
            if generation.cell_map is None or generation.minimum_feature_mm is None:
                raise RuntimeError("当前结果缺少 STL 重建采样信息")
            return resolve_export_spacing(
                generation.cell_map,
                float(self.export_tolerance.value()),
                float(generation.minimum_feature_mm),
                self._export_spacing_mode(),
            )

        def _contour_layer_spacing(
            self,
            generation: ImplicitGenerationResult | BackendGenerationResult | None,
            source: ContourFieldSource,
        ) -> np.ndarray:
            """Return spacing used to enumerate layers.

            The optional fixed height replaces only the spacing along the
            selected layer axis.  The two in-plane spacings continue to come
            from the selected field, so enabling this option does not reduce
            contour resolution within a layer.
            """

            fixed_height = (
                float(self.contour_fixed_layer_height.value())
                if self.contour_fixed_layer_enabled.isChecked()
                else None
            )
            return resolve_layer_spacing(
                self._contour_sampling_spacing(generation, source),
                self.contour_axis_combo.currentText(),
                fixed_height,
            )

        def _contour_layer_spacing_description(self) -> str:
            if self.contour_fixed_layer_enabled.isChecked():
                return f"固定 {self.contour_fixed_layer_height.value():.4f} mm"
            try:
                kind = self.contour_result_combo.currentData()
                spacing = self._contour_sampling_spacing(
                    self.implicit_generation_results.get(kind)
                    or self.backend_generation_handles.get(kind),
                    self.contour_source_combo.currentData(),
                )
                axis = self.contour_axis_combo.currentText()
                return f"跟随场采样 {spacing['XYZ'.index(axis)]:.4f} mm"
            except (RuntimeError, ValueError, TypeError):
                return "—"

        def _build_layer_contour(self) -> LayerContourResult:
            kind = self.contour_result_combo.currentData()
            if kind == "domain":
                body = self.domain_implicit_body
                field = self.domain_implicit_field
                if body is None or field is None:
                    raise RuntimeError("设计域隐式体尚未准备完成")
                source = self.contour_source_combo.currentData()
                axis = self.contour_axis_combo.currentText()
                position = self._contour_layer_position()
                level = float(self.contour_level.value())
                if source == "render":
                    return sample_sampled_field_layer(
                        field, axis, position, level=level, source=source
                    )
                return sample_implicit_body_layer(
                    body,
                    axis,
                    position,
                    self._contour_sampling_spacing(None, source),
                    level=level,
                    source=source,
                )
            generation = self.implicit_generation_results.get(kind)
            backend_generation = self.backend_generation_handles.get(kind)
            if generation is None and backend_generation is None:
                raise RuntimeError("当前没有可查看的隐式结果")
            source = self.contour_source_combo.currentData()
            axis = self.contour_axis_combo.currentText()
            position = self._contour_layer_position()
            level = float(self.contour_level.value())
            if backend_generation is not None:
                if source != "render":
                    raise RuntimeError("backend results expose display fields only")
                return sample_sampled_field_layer(
                    backend_generation.display_field,
                    axis,
                    position,
                    level=level,
                    source=source,
                )
            if source == "render":
                result = sample_sampled_field_layer(
                    generation.display_field,
                    axis,
                    position,
                    level=level,
                    source=source,
                )
            elif source == "authoritative":
                result = sample_implicit_body_layer(
                    generation.body,
                    axis,
                    position,
                    generation.display_field.spacing,
                    level=level,
                    source=source,
                )
            elif source == "stl_reconstruction":
                result = sample_implicit_body_layer(
                    generation.body,
                    axis,
                    position,
                    self._contour_sampling_spacing(generation, source),
                    level=level,
                    source=source,
                )
            else:
                raise ValueError(f"未知等值线场源：{source}")
            return result

        def _refresh_layer_contours(self, *_args) -> None:
            self._update_contour_layer_label()
            if not self.contour_enabled.isChecked():
                return
            if not hasattr(self, "viewer") or self.contour_result_combo.count() == 0:
                return
            try:
                key = self._contour_cache_key()
                result = self._layer_contour_cache.get(key)
                cache_label = "缓存命中"
                if result is None:
                    result = self._build_layer_contour()
                    self._layer_contour_cache.put(key, result)
                    cache_label = "已计算并缓存"
                polylines = [contour.points for contour in result.contours]
                self.viewer.set_contours(
                    [
                        ContourData(
                            polylines=polylines,
                            color=(0.04, 0.24, 0.80),
                            line_width=2.5,
                            cache_key=key,
                        )
                    ],
                    reset_view=False,
                )
                self.contour_status.setText(
                    f"{cache_label}；{result.source}；"
                    f"网格 {result.grid_shape[0]} × {result.grid_shape[1]}；"
                    f"轮廓 {len(result.contours)} 条，点数 {result.point_count:,}；"
                    f"层间距：{self._contour_layer_spacing_description()}"
                )
            except (RuntimeError, ValueError, TypeError) as exc:
                self.contour_status.setText(f"等值线提取失败：{exc}")

        def _make_section_group(self):
            group = QtWidgets.QGroupBox("交互剖切（内部查看）")
            layout = QtWidgets.QVBoxLayout(group)
            self.section_target_combo = QtWidgets.QComboBox()
            self.section_target_combo.setToolTip(
                "选择要单独显示并进行交互剖切的设计域或隐式体对象"
            )
            self.section_target_combo.currentIndexChanged.connect(
                self._on_section_target_changed
            )
            target_form = QtWidgets.QFormLayout()
            target_form.addRow("剖切查看对象", self.section_target_combo)
            self.section_render_source_combo = QtWidgets.QComboBox()
            self.section_render_source_combo.addItem("隐式体", "implicit")
            self.section_render_source_combo.addItem("STL 网格", "mesh")
            self.section_render_source_combo.setToolTip(
                "选择交互剖切使用的几何表示；STL 网格仅在该对象已完成重建后可用"
            )
            self.section_render_source_combo.currentIndexChanged.connect(
                self._on_section_render_source_changed
            )
            target_form.addRow("剖切显示源", self.section_render_source_combo)
            layout.addLayout(target_form)
            self.section_enabled = QtWidgets.QCheckBox("启用交互剖切")
            self.section_enabled.setChecked(False)
            self.section_enabled.setToolTip(
                "通过模型上的剖切操纵器调整位置和角度，仅改变显示，不修改原始 STL 或生成结果"
            )
            self.section_enabled.toggled.connect(self._on_section_mode_changed)
            layout.addWidget(self.section_enabled)

            helper = QtWidgets.QLabel(
                "鼠标悬停时操纵器高亮；左键命中并按住红/绿/蓝箭头沿 X/Y/Z 轴移动，"
                "按住对应颜色弧线绕该轴旋转；右键旋转视角，中键平移，滚轮缩放。正法向一侧被隐藏。"
            )
            helper.setObjectName("helper")
            helper.setWordWrap(True)
            layout.addWidget(helper)

            reset_button = QtWidgets.QPushButton("恢复剖切面")
            reset_button.clicked.connect(self._reset_interactive_section)
            layout.addWidget(reset_button)

            full_model_button = QtWidgets.QPushButton("显示完整模型")
            full_model_button.clicked.connect(self._disable_interactive_section)
            layout.addWidget(full_model_button)
            self._sync_section_target_options()
            return group

        def _sync_section_target_options(self) -> None:
            """Synchronize design-domain and implicit-body choices for sectioning."""

            if "section_target_combo" not in self.__dict__:
                return
            combo = self.section_target_combo
            current = combo.currentData()
            combo.blockSignals(True)
            combo.clear()
            combo.addItem("请选择查看对象", None)
            if self.domain_implicit_body is not None or self.sole_mesh.faces.size:
                combo.addItem("设计域 STL", "domain")
            for kind in self.implicit_generation_results:
                combo.addItem(kind, kind)
            index = combo.findData(current)
            selection_was_invalidated = current is not None and index < 0
            combo.setCurrentIndex(index if index >= 0 else 0)
            combo.setEnabled(combo.count() > 1)
            combo.blockSignals(False)
            if (
                (combo.count() <= 1 or selection_was_invalidated)
                and self.section_enabled.isChecked()
            ):
                self.section_enabled.setChecked(False)

        def _section_target_body(self) -> ImplicitBody | None:
            """Resolve the body currently selected for interactive sectioning."""

            if "section_target_combo" not in self.__dict__:
                return None
            kind = self.section_target_combo.currentData()
            if kind == "domain":
                return self.domain_implicit_body
            generation = self.implicit_generation_results.get(kind)
            return generation.body if generation is not None else None

        def _section_target_has_stl_mesh(self) -> bool:
            """Return whether the selected section target has a mesh result."""

            kind = self.section_target_combo.currentData()
            if kind == "domain":
                return bool(self.sole_mesh.faces.size)
            return kind in self.results

        def _on_section_target_changed(self, _index: int = 0) -> None:
            if not self.section_enabled.isChecked():
                return
            self.viewer.disable_interactive_section()
            self._on_section_mode_changed(True)

        def _on_section_render_source_changed(self, _index: int = 0) -> None:
            if not self.section_enabled.isChecked():
                return
            self.viewer.disable_interactive_section()
            self._on_section_mode_changed(True)

        def _on_section_mode_changed(self, enabled: bool):
            if not hasattr(self, "viewer"):
                return
            if not enabled:
                self.viewer.disable_interactive_section()
                self._refresh_scene(reset_view=False)
                return
            if self.field_viewer_enabled.isChecked():
                self.field_viewer_enabled.setChecked(False)
            if self.contour_enabled.isChecked():
                self.contour_enabled.blockSignals(True)
                self.contour_enabled.setChecked(False)
                self.contour_enabled.blockSignals(False)
            body = self._section_target_body()
            if body is None:
                self.section_enabled.blockSignals(True)
                self.section_enabled.setChecked(False)
                self.section_enabled.blockSignals(False)
                self.status.setText("请先选择需要交互剖切的对象")
                self._refresh_scene(reset_view=False)
                return
            if (
                self.section_render_source_combo.currentData() == "mesh"
                and not self._section_target_has_stl_mesh()
            ):
                self.section_enabled.blockSignals(True)
                self.section_enabled.setChecked(False)
                self.section_enabled.blockSignals(False)
                self.status.setText("所选对象尚未完成 STL 重建，请选择隐式体或先生成 STL 网格")
                self._refresh_scene(reset_view=False)
                return
            try:
                self._refresh_scene(reset_view=False)
                self.viewer.enable_interactive_section(vtk_bounds_from_implicit_body(body))
            except (TypeError, ValueError, RuntimeError) as exc:
                self.section_enabled.blockSignals(True)
                self.section_enabled.setChecked(False)
                self.section_enabled.blockSignals(False)
                self.status.setText(f"剖切显示失败：{exc}")
                self._refresh_scene(reset_view=False)

        def _reset_interactive_section(self):
            if hasattr(self, "viewer") and self.section_enabled.isChecked():
                self.viewer.reset_interactive_section()

        def _disable_interactive_section(self):
            if self.section_enabled.isChecked():
                self.section_enabled.setChecked(False)
            elif hasattr(self, "viewer"):
                self.viewer.disable_interactive_section()
                self._refresh_scene(reset_view=False)

        def _load_sole(self, path: Path, *, create_design: bool = False):
            if self._document_switch_is_locked():
                return
            if hasattr(self, "viewer"):
                self._disable_interactive_section()
            loaded_mesh = trimesh.load(path, force="mesh")
            if not isinstance(loaded_mesh, trimesh.Trimesh):
                raise TypeError("loaded design-domain STL is not a triangular mesh")

            if self.design_workspace.active_document is not None:
                self._store_active_design_state()
            if create_design or self.design_workspace.active_document is None:
                document_identifier = self.design_workspace.new_identifier()
                document_name = path.stem or "STL design"
                initial_settings = self._new_design_document_settings()
                if not self._backend_apply_workspace_commands(
                    [
                        {
                            "kind": "document.create.mesh",
                            "document_id": document_identifier,
                            "name": document_name,
                            "domain_name": document_name,
                            "source_path": str(path.resolve()),
                        },
                        {
                            "kind": "document.settings.replace",
                            "document_id": document_identifier,
                            "settings": initial_settings,
                        },
                        {
                            "kind": "document.field_scene.replace",
                            "document_id": document_identifier,
                            "field_primitives": [],
                        },
                    ]
                ):
                    return
                document = self.design_workspace.create_mesh_document(
                    loaded_mesh,
                    document_name,
                    asset_path=path,
                    identifier=document_identifier,
                )
                self._initialize_new_design_document(document)
                self._restore_active_design_state()
                self.show_domain.setChecked(True)
                # New documents have no persisted sampled-field cache. Build
                # the design-domain implicit preview before the first scene
                # refresh instead of briefly falling back to the imported STL.
                self._refresh_domain_implicit_preview()
                self._refresh_scene(reset_view=True)
                self._store_active_design_state()
                self.status.setText(f"已创建 STL 设计：{document.name}")
                return
            else:
                document = self.design_workspace.active_document
                assert document is not None
                if not self._backend_apply_workspace_commands(
                    [
                        {
                            "kind": "document.replace.mesh",
                            "document_id": document.identifier,
                            "domain_name": document.name,
                            "source_path": str(path.resolve()),
                        }
                    ]
                ):
                    return
                document.replace_domain(
                    MeshDesignDomain(
                        mesh=loaded_mesh,
                        name=document.name,
                        asset_path=path,
                    )
                )
                self.design_workspace.dirty = True

            # Results are tied to the previous design domain.  Clear them
            # before rebuilding the scene so a newly imported STL cannot be
            # displayed together with the old domain's generated layers.
            self.raw_results = {}
            self.repaired_results = {}
            self.results = {}
            self.implicit_results = {}
            self.implicit_generation_results = {}
            self.backend_generation_handles = {}
            self.implicit_fields = {}
            self.stl_reconstruction_results = {}
            contour_cache = self.__dict__.get("_layer_contour_cache")
            if contour_cache is not None:
                contour_cache.clear()
            field_plane_cache = self.__dict__.get("_field_plane_cache")
            if field_plane_cache is not None:
                field_plane_cache.clear()
            field_viewer_enabled = self.__dict__.get("field_viewer_enabled")
            if field_viewer_enabled is not None:
                field_viewer_enabled.setChecked(False)
            self.domain_implicit_body = None
            self.domain_implicit_field = None
            self._shell_fusion_domain_field = None
            self._reset_applied_sampling_info()
            self._set_result_controls_enabled(False)
            self._sync_contour_result_options()
            self._sync_field_viewer_options()
            self.sole_mesh = document.domain.mesh.copy()
            self._sync_stl_reconstruction_target_options()
            self._sync_section_target_options()
            self._sync_shell_lattice_options()
            if all(
                name in self.__dict__
                for name in ("tpms_widgets", "custom_widgets")
            ):
                self._reset_cell_map_frame(self._active_tpms_kind, notify=False)
                self._tpms_parameter_states[
                    self._active_tpms_kind
                ] = self._tpms_state_from_widgets()
                self._reset_cell_map_frame("Custom", notify=False)
            self._update_transition_axis_range(reset=True)
            self.path_edit.setText(str(path))
            bounds = self.sole_mesh.bounds
            extent = bounds[1] - bounds[0]
            self.domain_info.setText(f"{len(self.sole_mesh.faces):,} 个面 · {extent[0]:.1f} × {extent[1]:.1f} × {extent[2]:.1f} mm")
            self._update_recommendation()
            self._refresh_domain_implicit_preview()
            self._store_active_design_state()
            self._sync_design_workspace_controls()
            self._refresh_scene(reset_view=True)

        def set_view_front(self):
            self.viewer.set_view_front()

        def set_view_left(self):
            self.viewer.set_view_left()

        def set_view_top(self):
            self.viewer.set_view_top()

        def set_view_isometric(self):
            self.viewer.set_view_isometric()

        def reset_view(self):
            self.viewer.reset_view()

        def _transition_parameters(self) -> TransitionParameters:
            axis = ("X", "Y", "Z")[self.transition_axis.currentIndex()]
            driver_mode = self._transition_driver_mode()
            return TransitionParameters(
                plane_axis=axis,
                plane_position_mm=self._transition_position_mm(),
                angle1_deg=float(self.transition_angle1.value()),
                angle2_deg=float(self.transition_angle2.value()),
                transition_width_mm=float(self.transition_width.value()),
                center_offset_mm=float(self.transition_center_offset.value()),
                weight_kind=self.transition_weight.currentData(),
                sigmoid_sharpness=float(self.transition_sharpness.value()),
                g_on_negative_side=self.transition_side.currentIndex() == 0,
                minimum_feature_mm=(
                    float(self.transition_minimum_feature.value())
                    if self.transition_minimum_feature.value() > 0.0
                    else None
                ),
                automatic_registration=self.transition_registration.isChecked(),
                topology_correction=self.transition_topology.isChecked(),
                driver_mode=driver_mode,
                driver_identifier=(
                    self.transition_driver_combo.currentData()
                    if driver_mode == "field"
                    else None
                ),
                field_interval_lower_mm=float(self.transition_field_lower.value()),
                field_interval_upper_mm=float(self.transition_field_upper.value()),
            )

        def _transition_operand_parameters(self, controls) -> LatticeParameters:
            kind = controls["kind"].currentData()
            parameters = self._params(kind)
            if controls["inherit"].isChecked():
                return parameters
            cell_size = tuple(float(control.value()) for control in controls["sizes"])
            feature = float(controls["feature"].value())
            if isinstance(parameters, CustomUnitCellParameters):
                return replace(
                    parameters,
                    cell_size_mm=cell_size,
                    target_feature_mm=feature,
                )
            return replace(
                parameters,
                cell_size_mm=cell_size,
                wall_thickness_mm=feature,
            )

        def _sync_transition_operand_controls(self, controls) -> None:
            inherited = controls["inherit"].isChecked()
            try:
                parameters = self._params(controls["kind"].currentData())
                for control, value in zip(controls["sizes"], parameters.cell_size_xyz_mm):
                    if inherited:
                        control.blockSignals(True)
                        control.setValue(float(value))
                        control.blockSignals(False)
                if inherited:
                    feature = (
                        parameters.wall_thickness_mm
                        if isinstance(parameters, TPMSParameters)
                        else parameters.target_feature_mm
                    )
                    if feature is not None:
                        controls["feature"].blockSignals(True)
                        controls["feature"].setValue(float(feature))
                        controls["feature"].blockSignals(False)
            except (OSError, ValueError):
                pass
            for control in (*controls["sizes"], controls["feature"]):
                control.setEnabled(not inherited)
            if hasattr(self, "voxel_info"):
                self._on_transition_ui_changed()

        def _on_transition_weight_changed(self, *_args) -> None:
            self.transition_sharpness.setEnabled(
                self.transition_weight.currentData() == "sigmoid"
            )
            self._on_transition_ui_changed()

        def _set_field_primitive_controls_enabled(self, enabled: bool) -> None:
            if "field_primitive_combo" not in self.__dict__:
                return
            controls = (
                self.field_primitive_visible,
                self.field_primitive_name,
                *self.field_primitive_center,
                *self.field_primitive_rotation,
                self.field_primitive_radius,
                self.field_primitive_height,
                *self.field_primitive_size,
                self.field_primitive_manipulator,
                self.delete_field_primitive_button,
            )
            for control in controls:
                control.setEnabled(enabled)

        def _selected_field_primitive(self) -> ImplicitPrimitive | None:
            identifier = getattr(self, "_selected_primitive_identifier", None)
            if identifier is None and "field_primitive_combo" in self.__dict__:
                identifier = self.field_primitive_combo.currentData()
            return self.field_primitives.get(identifier)

        def _selected_transition_primitive(self) -> ImplicitPrimitive | None:
            if "transition_driver_combo" not in self.__dict__:
                return None
            return self.field_primitives.get(self.transition_driver_combo.currentData())

        def _sync_active_field_object_scene(self) -> None:
            """Persist the active Design's field-object definitions immediately."""

            if self._workspace_loading:
                return
            document = self.design_workspace.active_document
            if document is None:
                return
            if (
                document.field_primitives != self.field_primitives
                or document.field_primitive_visibility
                != {
                    key: bool(value)
                    for key, value in self.field_primitive_visibility.items()
                }
            ):
                if not self._backend_apply_workspace_commands(
                    [
                        {
                            "kind": "document.field_scene.replace",
                            "document_id": document.identifier,
                            "field_primitives": self._backend_field_scene_payload(
                                self.field_primitives,
                                self.field_primitive_visibility,
                            ),
                        }
                    ]
                ):
                    return
                document.replace_field_object_scene(
                    self.field_primitives,
                    self.field_primitive_visibility,
                )
                self.design_workspace.dirty = True

        def _transition_driver_mode(self) -> TransitionDriverMode:
            if "transition_driver_mode" not in self.__dict__:
                return "plane"
            return self.transition_driver_mode.currentData()

        def _add_field_primitive(self, kind: PrimitiveKind) -> None:
            index = self._next_primitive_index
            self._next_primitive_index += 1
            identifier = f"primitive-{index}"
            names = {"sphere": "球", "cylinder": "圆柱", "box": "立方体"}
            center = (
                tuple(float(value) for value in self._active_domain_bounds().mean(axis=0))
                if self.sole_mesh.faces.size
                else (0.0, 0.0, 0.0)
            )
            scale = (
                max(float(np.min(np.diff(self._active_domain_bounds(), axis=0))) * 0.12, 1.0)
                if self.sole_mesh.faces.size
                else 5.0
            )
            self.field_primitives[identifier] = ImplicitPrimitive(
                identifier=identifier,
                name=f"{names[kind]} {index}",
                kind=kind,
                center_mm=center,
                radius_mm=scale,
                height_mm=2.0 * scale,
                size_mm=(2.0 * scale, 2.0 * scale, 2.0 * scale),
            )
            self.field_primitive_visibility[identifier] = True
            self._sync_active_field_object_scene()
            self._sync_field_primitive_options(identifier)
            self._sync_transition_field_objects()
            self._sync_field_viewer_options()
            self._refresh_primitive_preview(sync_gizmo=True)
            self.field_primitive_status.setText("已添加解析场对象；左键单击预览可选中，拖动三轴操纵器可变换位置和方向。")

        def _delete_selected_field_primitive(self) -> None:
            primitive = self._selected_field_primitive()
            if primitive is None:
                return
            identifier = primitive.identifier
            self._invalidate_primitive_preview_cache(identifier)
            referenced = (
                self._transition_driver_mode() == "field"
                and self.transition_driver_combo.currentData() == identifier
            )
            self.field_primitives.pop(identifier, None)
            self.field_primitive_visibility.pop(identifier, None)
            self._selected_primitive_identifier = None
            self._sync_active_field_object_scene()
            self._sync_field_primitive_options()
            self._sync_transition_field_objects()
            self._sync_field_viewer_options()
            self._refresh_primitive_preview(sync_gizmo=True)
            if referenced:
                self._invalidate_transition_result()
                self.transition_driver_mode.setCurrentIndex(0)
                self.field_primitive_status.setText("已删除被过渡引用的场对象；过渡区域已切回平面。")
            else:
                self.field_primitive_status.setText("已删除场对象。")

        def _sync_field_primitive_options(self, preferred: str | None = None) -> None:
            if "field_primitive_combo" not in self.__dict__:
                return
            combo = self.field_primitive_combo
            current = preferred or self._selected_primitive_identifier or combo.currentData()
            combo.blockSignals(True)
            combo.clear()
            for identifier, primitive in self.field_primitives.items():
                combo.addItem(primitive.name, identifier)
            index = combo.findData(current)
            combo.setCurrentIndex(index if index >= 0 else (0 if combo.count() else -1))
            combo.blockSignals(False)
            self._selected_primitive_identifier = combo.currentData()
            self._sync_field_primitive_editor()

        def _sync_field_primitive_editor(self) -> None:
            primitive = self._selected_field_primitive()
            self._set_field_primitive_controls_enabled(primitive is not None)
            if primitive is None:
                return
            controls = (
                self.field_primitive_visible,
                self.field_primitive_name,
                *self.field_primitive_center,
                *self.field_primitive_rotation,
                self.field_primitive_radius,
                self.field_primitive_height,
                *self.field_primitive_size,
            )
            for control in controls:
                control.blockSignals(True)
            self.field_primitive_visible.setChecked(self.field_primitive_visibility.get(primitive.identifier, True))
            self.field_primitive_name.setText(primitive.name)
            for control, value in zip(self.field_primitive_center, primitive.center_mm):
                control.setValue(value)
            for control, value in zip(self.field_primitive_rotation, primitive.rotation_euler_deg):
                control.setValue(value)
            self.field_primitive_radius.setValue(primitive.radius_mm)
            self.field_primitive_height.setValue(primitive.height_mm)
            for control, value in zip(self.field_primitive_size, primitive.size_mm):
                control.setValue(value)
            for control in controls:
                control.blockSignals(False)
            is_sphere = primitive.kind == "sphere"
            is_cylinder = primitive.kind == "cylinder"
            self.field_primitive_radius.setVisible(is_sphere or is_cylinder)
            self.field_primitive_height.setVisible(is_cylinder)
            for control in self.field_primitive_size:
                control.setVisible(primitive.kind == "box")

        def _on_field_primitive_selection_changed(self, _index: int = 0) -> None:
            self._selected_primitive_identifier = self.field_primitive_combo.currentData()
            self._sync_field_primitive_editor()
            self._refresh_primitive_preview(sync_gizmo=True)

        def _on_field_primitive_changed(self, *_args) -> None:
            primitive = self._selected_field_primitive()
            if primitive is None:
                return
            updated = replace(
                primitive,
                name=self.field_primitive_name.text().strip() or primitive.name,
                center_mm=tuple(float(control.value()) for control in self.field_primitive_center),
                rotation_euler_deg=tuple(float(control.value()) for control in self.field_primitive_rotation),
                radius_mm=float(self.field_primitive_radius.value()),
                height_mm=float(self.field_primitive_height.value()),
                size_mm=tuple(float(control.value()) for control in self.field_primitive_size),
            )
            if updated == primitive:
                return
            self._cancel_display_refinement_for_geometry_edit()
            self.field_primitives[primitive.identifier] = updated
            self._invalidate_primitive_preview_cache(primitive.identifier)
            self.field_primitive_visibility[primitive.identifier] = self.field_primitive_visible.isChecked()
            self._apply_analytic_domain_primitive_change(updated)
            self._sync_active_field_object_scene()
            self._sync_field_primitive_options(updated.identifier)
            self._sync_transition_field_objects()
            self._sync_field_viewer_options()
            self._invalidate_primitive_dependents(primitive.identifier)
            self._refresh_primitive_preview(sync_gizmo=True)

        def _on_field_primitive_visibility_changed(self, visible: bool) -> None:
            """Change preview visibility without invalidating any geometry."""

            primitive = self._selected_field_primitive()
            if primitive is None:
                return
            identifier = primitive.identifier
            self.field_primitive_visibility[identifier] = bool(visible)
            document = self.design_workspace.active_document
            if document is not None and not self._workspace_loading:
                if not self._backend_apply_workspace_commands(
                    [
                        {
                            "kind": "document.field.visibility.set",
                            "document_id": document.identifier,
                            "primitive_id": identifier,
                            "visible": bool(visible),
                        }
                    ]
                ):
                    self.field_primitive_visibility[identifier] = document.field_primitive_visibility.get(
                        identifier,
                        True,
                    )
                    self._sync_field_primitive_editor()
                    return
                document.set_field_primitive_visibility(identifier, visible)
                self.design_workspace.dirty = True
            self._refresh_primitive_preview(sync_gizmo=True)

        def _on_primitive_gizmo_changed(self, origin: np.ndarray, basis: np.ndarray) -> None:
            # A viewer may synchronously report its initial gizmo state while
            # the controller is merely synchronizing the scene.  Only a
            # completed user drag is allowed to mutate a Design domain.
            if self._primitive_gizmo_sync_in_progress:
                return
            primitive = self._selected_field_primitive()
            if primitive is None:
                return
            updated = replace(
                primitive,
                center_mm=tuple(float(value) for value in origin),
                rotation_euler_deg=euler_deg_from_rotation_matrix(basis),
            )
            self._cancel_display_refinement_for_geometry_edit()
            self.field_primitives[primitive.identifier] = updated
            self._invalidate_primitive_preview_cache(primitive.identifier)
            self._apply_analytic_domain_primitive_change(updated)
            self._sync_active_field_object_scene()
            self._sync_field_primitive_editor()
            self._sync_transition_field_objects()
            self._sync_field_viewer_options()
            self._invalidate_primitive_dependents(primitive.identifier)
            self._refresh_primitive_preview(sync_gizmo=False)

        def _apply_analytic_domain_primitive_change(
            self,
            primitive: ImplicitPrimitive,
        ) -> None:
            """Keep a dual-role analytic Design domain and its field object unified."""

            document = self.design_workspace.active_document
            if (
                document is None
                or not isinstance(document.domain, AnalyticDesignDomain)
                or document.settings.get("analytic_domain_primitive_id")
                != primitive.identifier
            ):
                return
            if not self._backend_apply_workspace_commands(
                [
                    {
                        "kind": "document.replace.analytic",
                        "document_id": document.identifier,
                        "domain_name": document.name,
                        "primitive": asdict(primitive),
                    }
                ]
            ):
                return
            document.replace_domain(
                AnalyticDesignDomain(primitive=primitive, name=document.name)
            )
            self.sole_mesh = document.domain.preview_mesh()
            self.domain_implicit_body = None
            self.domain_implicit_field = None
            self._shell_fusion_domain_field = None
            self.implicit_results = {}
            self.implicit_generation_results = {}
            self.backend_generation_handles = {}
            self.implicit_fields = {}
            self.raw_results = {}
            self.repaired_results = {}
            self.results = {}
            self.stl_reconstruction_results = {}
            self._layer_contour_cache.clear()
            self._field_plane_cache.clear()
            self._update_transition_axis_range(reset=True)
            self._sync_stl_reconstruction_target_options()
            self._sync_section_target_options()
            self._sync_shell_lattice_options()
            self._pending_analytic_domain_preview = document.identifier
            self._analytic_domain_preview_timer.start()
            self.design_workspace.dirty = True

        def _flush_pending_analytic_domain_preview(self) -> None:
            """Rebuild one stable preview after a burst of dimension edits."""

            pending = self._pending_analytic_domain_preview
            self._pending_analytic_domain_preview = None
            if pending is None or getattr(self, "_is_closing", False):
                return
            document = self.design_workspace.active_document
            if (
                document is None
                or document.identifier != pending
                or not isinstance(document.domain, AnalyticDesignDomain)
            ):
                return
            self._refresh_domain_implicit_preview()
            # The lightweight primitive mesh remains interactive during the
            # debounce interval; replace it with the authoritative sampled
            # surface only after the dimensions settle.
            self._refresh_scene(reset_view=False)
            discard_inactive = getattr(
                self.viewer,
                "discard_inactive_implicit_layers",
                None,
            )
            if discard_inactive is not None:
                discard_inactive()

        def _invalidate_primitive_dependents(self, identifier: str) -> None:
            self._field_plane_cache.clear()
            if (
                self._transition_driver_mode() == "field"
                and self.transition_driver_combo.currentData() == identifier
            ):
                self._invalidate_transition_result()
            if self.field_viewer_enabled.isChecked() and self.field_viewer_target_combo.currentData() == f"field:{identifier}":
                self._schedule_field_viewer_refresh()

        def _primitive_preview_layers(self):
            from lattice_studio.presentation.qt.viewers.pyvista_viewer import MeshData

            if not self._should_show_primitive_previews():
                return []
            layers = []
            target_identifier = self._field_viewer_primitive_target_identifier()
            for identifier, primitive in self.field_primitives.items():
                if target_identifier is not None and identifier != target_identifier:
                    continue
                if not self.field_primitive_visibility.get(identifier, True):
                    continue
                cache_key = (identifier, primitive, 64)
                cached = self._primitive_preview_mesh_cache.get(cache_key)
                if cached is None:
                    cached = primitive.preview_mesh(subdivisions=64)
                    cached = (
                        np.asarray(cached[0], dtype=np.float32),
                        np.asarray(cached[1], dtype=np.int32),
                    )
                    self._primitive_preview_mesh_cache[cache_key] = cached
                vertices, faces = cached
                vertices = vertices @ primitive.rotation_matrix.T + np.asarray(primitive.center_mm)
                selected = identifier == self._selected_primitive_identifier
                layers.append(
                    MeshData(
                        vertices=vertices,
                        faces=faces,
                        color=(0.50, 0.52, 0.56, 1.0),
                        metallic=0.08,
                        roughness=0.42,
                        cache_key=("field-primitive", identifier, primitive),
                    )
                )
            return layers

        def _invalidate_primitive_preview_cache(self, identifier: str) -> None:
            self._primitive_preview_mesh_cache = {
                key: value
                for key, value in self._primitive_preview_mesh_cache.items()
                if key[0] != identifier
            }
            self._primitive_preview_signature = None

        def _field_viewer_primitive_target_identifier(self) -> str | None:
            """Return the selected analytic field-object identifier, if any."""

            combo = getattr(self, "field_viewer_target_combo", None)
            if combo is None:
                return None
            target = combo.currentData()
            if not isinstance(target, str) or not target.startswith("field:"):
                return None
            identifier = target.removeprefix("field:")
            return identifier if identifier in self.field_primitives else None

        def _sync_field_viewer_object_visibility_control(self) -> None:
            """Name the visibility control after its current Field Viewer target."""

            if self._field_viewer_primitive_target_identifier() is not None:
                self.field_viewer_show_object.setText("显示基本几何体")
                self.field_viewer_show_object.setToolTip(
                    "显示或隐藏当前被查看的基本几何体；不影响其 SDF 采样。"
                )
                return
            self.field_viewer_show_object.setText("显示三维对象")
            self.field_viewer_show_object.setToolTip(
                "关闭后仅显示场平面、等值线和操纵器；场的采样与计算不受影响。"
            )

        def _should_show_primitive_previews(self) -> bool:
            """Keep placement proxies out of analytical inspection modes."""

            if any(
                control is not None and control.isChecked()
                for control in (
                    getattr(self, "contour_enabled", None),
                    getattr(self, "section_enabled", None),
                )
            ):
                return False
            field_viewer = getattr(self, "field_viewer_enabled", None)
            if field_viewer is None or not field_viewer.isChecked():
                return True
            visibility = getattr(self, "field_viewer_show_object", None)
            return bool(
                visibility is not None
                and visibility.isChecked()
                and self._field_viewer_primitive_target_identifier() is not None
            )

        def _refresh_primitive_preview(
            self,
            *,
            sync_gizmo: bool,
            defer_scene_update: bool = False,
        ) -> None:
            if not hasattr(self, "viewer"):
                return
            layers = self._primitive_preview_layers()
            signature = tuple(
                (
                    layer.cache_key,
                    layer.color,
                    layer.render_mode,
                    layer.unlit,
                    layer.line_width,
                    layer.metallic,
                    layer.roughness,
                )
                for layer in layers
            )
            set_overlays = getattr(self.viewer, "set_scene_overlay_meshes", None)
            if set_overlays is not None and signature != self._primitive_preview_signature:
                set_overlays(layers, rebuild_scene=not defer_scene_update)
                self._primitive_preview_signature = signature
            if sync_gizmo:
                self._sync_primitive_gizmo()

        def _sync_primitive_gizmo(self, *_args) -> None:
            if not hasattr(self, "viewer"):
                return
            primitive = self._selected_field_primitive()
            field_viewer_active = bool(
                getattr(self, "field_viewer_enabled", None) is not None
                and self.field_viewer_enabled.isChecked()
            )
            if (
                primitive is None
                or field_viewer_active
                or not self.field_primitive_manipulator.isChecked()
                or not self.field_primitive_visibility.get(primitive.identifier, True)
                or not self._should_show_primitive_previews()
            ):
                disable_transform = getattr(
                    self.viewer,
                    "disable_primitive_transform",
                    None,
                )
                if disable_transform is not None:
                    disable_transform()
                return
            lower = primitive.bounds[0]
            upper = primitive.bounds[1]
            if self.sole_mesh.faces.size:
                domain_bounds = self._active_domain_bounds()
                lower = np.minimum(lower, domain_bounds[0])
                upper = np.maximum(upper, domain_bounds[1])
            bounds = np.array((lower[0], upper[0], lower[1], upper[1], lower[2], upper[2]))
            enable_transform = getattr(
                self.viewer,
                "enable_primitive_transform",
                None,
            )
            if enable_transform is not None:
                self._primitive_gizmo_sync_in_progress = True
                try:
                    enable_transform(
                        bounds,
                        np.asarray(primitive.center_mm),
                        primitive.rotation_matrix,
                        self._on_primitive_gizmo_changed,
                    )
                finally:
                    self._primitive_gizmo_sync_in_progress = False

        def _on_scene_mesh_selected(self, key: object) -> None:
            if not (isinstance(key, tuple) and len(key) >= 2 and key[0] == "field-primitive"):
                return
            identifier = key[1]
            if identifier not in self.field_primitives:
                return
            self._sync_field_primitive_options(identifier)
            self._refresh_primitive_preview(sync_gizmo=True)

        def _sync_transition_field_objects(self) -> None:
            if "transition_driver_combo" not in self.__dict__:
                return
            combo = self.transition_driver_combo
            current = combo.currentData()
            combo.blockSignals(True)
            combo.clear()
            for identifier, primitive in self.field_primitives.items():
                combo.addItem(primitive.name, identifier)
            index = combo.findData(current)
            combo.setCurrentIndex(index if index >= 0 else (0 if combo.count() else -1))
            combo.blockSignals(False)
            self._sync_transition_driver_ui()

        def _on_transition_driver_changed(self, *_args) -> None:
            self._sync_transition_driver_ui()
            self._on_transition_ui_changed()

        def _sync_transition_driver_ui(self) -> None:
            if "transition_driver_mode" not in self.__dict__:
                return
            field_mode = self._transition_driver_mode() == "field"
            has_driver = self.transition_driver_combo.count() > 0
            self.transition_driver_combo.setEnabled(field_mode and has_driver)
            self._set_transition_form_row_visible(
                self.transition_driver_combo,
                field_mode,
            )
            self._set_transition_form_row_visible(
                self.transition_field_interval_container,
                field_mode,
            )
            for control in self._transition_plane_row_fields:
                self._set_transition_form_row_visible(control, not field_mode)
            self._set_transition_form_row_visible(
                self.transition_registration,
                not field_mode,
            )
            self._set_transition_form_row_visible(
                self.show_transition_plane,
                not field_mode,
            )
            if field_mode:
                self.transition_width_status.setText(
                    "场值区间直接定义 Ramp 两端；曲面场不采用平面相位配准和跨平面质量检查。"
                )

        def _set_transition_form_row_visible(self, field, visible: bool) -> None:
            """Toggle a QFormLayout row together with its label."""

            label = self._transition_form.labelForField(field)
            if label is not None:
                label.setVisible(visible)
            field.setVisible(visible)

        def _transition_position_mm(self) -> float:
            minimum = getattr(self, "transition_slider_min", 0.0)
            maximum = getattr(self, "transition_slider_max", 1.0)
            ratio = self.transition_position_slider.value() / 1000.0
            return minimum + ratio * (maximum - minimum)

        def _update_transition_axis_range(self, reset: bool = False):
            if not self.sole_mesh.faces.size:
                return
            axis_index = self.transition_axis.currentIndex()
            bounds = self._active_domain_bounds()
            self.transition_slider_min = float(bounds[0, axis_index])
            self.transition_slider_max = float(bounds[1, axis_index])
            if reset:
                self.transition_position_slider.blockSignals(True)
                self.transition_position_slider.setValue(500)
                self.transition_position_slider.blockSignals(False)
            self._on_transition_position_changed()

        def _on_transition_axis_changed(self, _index: int = 0):
            self._update_transition_angle_labels()
            self._update_transition_axis_range(reset=False)
            self._on_transition_ui_changed()

        def _update_transition_angle_labels(self):
            """Expose the rotation axes used by the selected transition plane."""

            if not hasattr(self, "transition_angle1_label"):
                return
            axes = (("Y", "Z"), ("X", "Z"), ("X", "Y"))[self.transition_axis.currentIndex()]
            self.transition_angle1_label.setText(f"绕{axes[0]}")
            self.transition_angle2_label.setText(f"绕{axes[1]}")
            self.transition_angle1.setToolTip(f"绕 {axes[0]} 轴旋转过渡分界面")
            self.transition_angle2.setToolTip(f"绕 {axes[1]} 轴旋转过渡分界面")

        def _on_transition_position_changed(self, _value: int = 0):
            if not hasattr(self, "transition_position_label"):
                return
            position = self._transition_position_mm()
            axis = ("X", "Y", "Z")[self.transition_axis.currentIndex()]
            self.transition_position_label.setText(f"位置：{position:.2f} mm（{axis} 轴）")
            self._update_transition_plane()

        def _on_transition_ui_changed(self, *_args):
            self._invalidate_transition_result()
            self._update_transition_lattice_summary()
            self._update_transition_plane()
            self._update_recommendation()

        def _invalidate_transition_result(self) -> None:
            """Discard generated data derived from changed transition controls."""

            self._mark_active_design_definition_changed({"Transition"})
            changed = False
            for name in (
                "implicit_generation_results",
                "implicit_results",
                "implicit_fields",
                "raw_results",
                "repaired_results",
                "results",
            ):
                mapping = getattr(self, name, None)
                if mapping is not None and "Transition" in mapping:
                    mapping.pop("Transition", None)
                    changed = True
            if changed and hasattr(self, "viewer"):
                self._sync_stl_reconstruction_target_options()
                self._sync_section_target_options()
                self._set_result_controls_enabled(bool(getattr(self, "results", {})))
                self._refresh_scene(reset_view=False)

        def _update_transition_lattice_summary(self):
            """Show the physical operand parameters actually used by the transition."""

            if not hasattr(self, "transition_first_info"):
                return
            try:
                first = self._transition_operand_parameters(self.transition_first_controls)
                second = self._transition_operand_parameters(self.transition_second_controls)
                first_actual = first.cell_size_xyz_mm
                second_actual = second.cell_size_xyz_mm
                recommendation = None
                if self.sole_mesh.faces.size:
                    first_map = self._cell_map_for_active_design(first)
                    second_map = self._cell_map_for_active_design(second)
                    first_actual = tuple(round(value, 2) for value in first_map.spacing_mm)
                    second_actual = tuple(round(value, 2) for value in second_map.spacing_mm)
                    domain = self._active_design_domain()
                    if domain is not None:
                        first_operand, prepared = (
                            _make_transition_operand_for_design_domain(
                                domain,
                                first,
                                first_map,
                                self.prepared_custom_cell,
                            )
                        )
                        second_operand, _ = _make_transition_operand_for_design_domain(
                            domain,
                            second,
                            second_map,
                            prepared,
                        )
                    else:
                        first_operand, prepared = _make_transition_operand(
                            self.sole_mesh,
                            first,
                            first_map,
                            self.prepared_custom_cell,
                        )
                        second_operand, _ = _make_transition_operand(
                            self.sole_mesh,
                            second,
                            second_map,
                            prepared,
                        )
                    recommendation = recommend_provider_transition_width(
                        first_operand,
                        second_operand,
                    )

                def describe(parameters, actual) -> str:
                    feature = (
                        f"壁厚 {parameters.wall_thickness_mm:.3f} mm"
                        if isinstance(parameters, TPMSParameters)
                        else (
                            f"目标特征厚度 {parameters.target_feature_mm:.3f} mm"
                            if parameters.target_feature_mm is not None
                            else "特征厚度使用源模型"
                        )
                    )
                    return (
                        f"{parameters.kind}；目标 {parameters.cell_size_xyz_mm} mm；"
                        f"实际 {actual} mm；{feature}"
                    )

                self.transition_first_info.setText(describe(first, first_actual))
                self.transition_second_info.setText(describe(second, second_actual))
                first_name = first.kind
                second_name = second.kind
                side_text = (
                    f"负侧 {first_name} / 正侧 {second_name}"
                    if self.transition_side.currentIndex() == 0
                    else f"负侧 {second_name} / 正侧 {first_name}"
                )
                self.transition_side_info.setText(f"当前分配：{side_text}")
                if self._transition_driver_mode() == "field":
                    primitive = self._selected_transition_primitive()
                    name = primitive.name if primitive is not None else "未选择"
                    lower = float(self.transition_field_lower.value())
                    upper = float(self.transition_field_upper.value())
                    self.transition_width_status.setText(
                        f"场对象 {name}；区间 [{lower:.2f}, {upper:.2f}] mm，"
                        f"对应 Ramp 宽度 {upper - lower:.2f} mm。"
                    )
                    return
                if recommendation is None:
                    self.transition_width_status.setText("载入设计域后计算建议宽度")
                else:
                    current = float(self.transition_width.value())
                    risk = "；当前宽度偏窄，但仍允许计算" if current < recommendation else ""
                    self.transition_width_status.setText(
                        f"建议不小于 {recommendation:.2f} mm{risk}"
                    )
            except (OSError, ValueError) as exc:
                self.transition_first_info.setText(f"参数尚不可用：{exc}")
                self.transition_second_info.clear()
                self.transition_width_status.setText("无法计算宽度建议")

        def _on_transition_plane_visibility_changed(self, *_args):
            self._update_transition_plane()

        def _update_transition_plane(self):
            if not hasattr(self, "viewer") or not self.sole_mesh.faces.size:
                return
            if (
                not self.transition_enabled.isChecked()
                or self._transition_driver_mode() != "plane"
                or not self.show_transition_plane.isChecked()
            ):
                self.viewer.hide_plane()
                return
            try:
                params = self._transition_parameters()
                point, normal = _transition_plane_geometry_for_bounds(
                    self._active_domain_bounds(), params
                )
                plane_size = float(np.max(np.diff(self._active_domain_bounds(), axis=0))) * 1.5
                self.viewer.set_plane(point, normal, plane_size)
            except ValueError:
                self.viewer.hide_plane()

        def _browse_sole(self, *, create_design: bool = False):
            path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "选择设计域 STL", str(self.root), "STL (*.stl)")
            if path:
                try:
                    self._load_sole(Path(path), create_design=create_design)
                except Exception as exc:
                    QtWidgets.QMessageBox.critical(self, "模型加载失败", f"{type(exc).__name__}: {exc}")

        def _browse_custom_cell(self):
            start = (
                self.custom_widgets["source_path"].text()
                or str(self.root)
            )
            path, _ = QtWidgets.QFileDialog.getOpenFileName(
                self,
                "选择自定义晶胞 STL",
                start,
                "STL (*.stl)",
            )
            if not path:
                return
            QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
            try:
                prepared = prepare_stl_unit_cell(Path(path))
            except Exception as exc:
                self.prepared_custom_cell = None
                self.custom_widgets["source_status"].setText(
                    f"导入失败：{type(exc).__name__}: {exc}"
                )
                QtWidgets.QMessageBox.critical(
                    self,
                    "自定义晶胞导入失败",
                    f"{type(exc).__name__}: {exc}",
                )
                return
            finally:
                QtWidgets.QApplication.restoreOverrideCursor()
            self.prepared_custom_cell = prepared
            self.custom_widgets["source_path"].setText(str(prepared.report.source_path))
            document = self.design_workspace.active_document
            if document is not None:
                document.settings["custom_unit_cell_source_path"] = str(
                    prepared.report.source_path
                )
                document.touch()
                self.design_workspace.dirty = True
            report = prepared.report
            size = " × ".join(f"{value:.3f}" for value in prepared.source_size_mm)
            self.custom_widgets["source_status"].setText(
                f"已就绪：{report.prepared_faces:,} 个面，{size} mm；"
                f"水密={report.watertight}，连通量={report.connected_components}；"
                f"源特征厚度={report.characteristic_thickness_mm:.3f} mm；"
                f"移除退化面={report.removed_degenerate_faces}"
            )
            self._invalidate_lattice_dependents("Custom")
            self._update_recommendation()
            self.status.setText("自定义晶胞 STL 已准备，可生成或查看 Cell Map")

        def _current_sampling(self):
            raw_target = self.target_voxels.text().replace(",", "").strip()
            try:
                target_voxels = int(raw_target)
            except ValueError as exc:
                raise ValueError("目标体素数必须是正整数") from exc
            processing_mode = self.processing_mode.currentData()
            return SamplingParameters(
                target_voxels=target_voxels,
                use_cpp_sdf=self.use_cpp_sdf.isChecked(),
                batch_count=(
                    1
                    if processing_mode == "single_pass"
                    else self.batch_count.value()
                ),
                processing_mode=processing_mode,
            )

        def _on_display_voxel_mode_changed(self, automatic: bool):
            self.display_voxel_size.setReadOnly(automatic)
            self._update_recommendation()

        def _on_display_batch_mode_changed(self, enabled: bool):
            self.display_batch_count.setEnabled(enabled)
            self._update_recommendation()

        def _display_sampling_batch_count(self) -> int | None:
            if not self.display_batched_sampling.isChecked():
                return None
            value = int(self.display_batch_count.value())
            if value < 2:
                raise ValueError("隐式体显示采样批数必须至少为 2")
            return value

        def _export_spacing_mode(self) -> ExportSpacingMode:
            return (
                "recommended"
                if self.export_spacing_auto.isChecked()
                else "exact"
            )

        def _on_export_spacing_mode_changed(self, automatic: bool):
            if automatic:
                self.export_tolerance.setToolTip(
                    "STL 重建请求的最大采样间距；实际间距还受最小特征厚度 / 3 "
                    "和晶胞尺寸 / 16 约束；不使用显示体素大小"
                )
            else:
                self.export_tolerance.setToolTip(
                    "STL 重建三轴严格使用该采样间距；系统不会按特征厚度或晶胞尺寸自动改小"
                )
            self._update_stl_grid_estimate()

        def _update_stl_grid_estimate(self, *_args) -> None:
            if "export_grid_info" not in self.__dict__:
                return
            selected_targets = self._selected_stl_reconstruction_results()
            has_analytic_domain_target = (
                self._analytic_design_reconstruction_target() is not None
            )
            selected_backend = self.backend_generation_handles.get(
                self.stl_reconstruction_target_combo.currentData()
            )
            if (
                not self.implicit_generation_results
                and not self.backend_generation_handles
                and not has_analytic_domain_target
            ):
                self.export_grid_info.setText(
                    "生成隐式体或创建解析设计域后显示 STL 网格尺寸和总体素数"
                )
                return
            if selected_backend is not None:
                self.export_grid_info.setText(
                    "当前对象由本地后端持有；重建时将在后端按所选容差计算网格"
                )
                return
            if not selected_targets:
                self.export_grid_info.setText("请选择一个隐式体对象后估算 STL 网格")
                return
            try:
                tolerance = float(self.export_tolerance.value())
                spacing_mode = self._export_spacing_mode()
                lines = []
                for kind, target in selected_targets.items():
                    estimate = estimate_export_grid(
                        target.body,
                        target.extraction_map,
                        tolerance,
                        target.minimum_feature_mm,
                        spacing_mode,
                        target.enforce_feature_limits,
                    )
                    shape = estimate.grid_shape
                    line = (
                        f"{kind}：{shape[0]}×{shape[1]}×{shape[2]} = "
                        f"{estimate.total_voxels:,} 体素；"
                        f"实际间距 {_format_spacing_mm(estimate.spacing_mm)}"
                    )
                    if spacing_mode == "exact" and target.enforce_feature_limits:
                        recommended = recommend_export_spacing(
                            target.extraction_map,
                            tolerance,
                            target.minimum_feature_mm,
                        )
                        if np.any(
                            np.asarray(estimate.spacing_mm)
                            > recommended + np.maximum(recommended * 1.0e-6, 1.0e-9)
                        ):
                            line += (
                                f"；安全建议不大于 {_format_spacing_mm(recommended)}，"
                                "当前设置可能丢失薄壁或细节"
                            )
                    lines.append(line)
                self.export_grid_info.setText("\n".join(lines))
            except (RuntimeError, ValueError, OverflowError) as exc:
                self.export_grid_info.setText(f"无法估算：{exc}")

        def _on_processing_mode_changed(self, _index: int = 0):
            enabled = self.processing_mode.currentData() != "single_pass"
            self.batch_count.setEnabled(enabled)
            self._update_recommendation()

        def _display_voxel_size(self) -> float | None:
            if self.display_voxel_auto.isChecked():
                return None
            value = float(self.display_voxel_size.value())
            if value <= 0:
                raise ValueError("自定义显示体素大小必须是正数")
            return value

        def _display_memory_budget_mb(self) -> float:
            value = float(self.display_memory_budget.value())
            if value <= 0:
                raise ValueError("显示显存预算必须是正数")
            return value

        def _refresh_applied_sampling_info(self) -> None:
            self.applied_sampling_info.setText(
                self._implicit_sampling_summary
                + "\n"
                + self._stl_sampling_summary
            )

        def _reset_applied_sampling_info(self) -> None:
            self._implicit_sampling_summary = "隐式显示：尚未生成"
            self._stl_sampling_summary = "STL 重建：尚未生成"
            if "applied_sampling_info" in self.__dict__:
                self._refresh_applied_sampling_info()
            if "export_grid_info" in self.__dict__:
                self._update_stl_grid_estimate()

        def _lattice_widgets(self, kind: LatticeKind):
            if kind in ("G", "D", "IWP", "Primitive", "Neovius"):
                # Keep the helper usable by lightweight, pre-Qt test doubles
                # and by callers created before the unified TPMS panel was
                # introduced.  The live window always owns ``tpms_widgets``;
                # the legacy ``g_widgets`` shape is only a compatibility
                # fallback and is deliberately not duplicated elsewhere.
                widgets = self.__dict__.get("tpms_widgets")
                if widgets is not None:
                    return widgets
                widgets = self.__dict__.get("g_widgets")
                if widgets is not None:
                    return widgets
                raise RuntimeError("TPMS widgets have not been initialized")
            if kind == "Custom":
                return self.custom_widgets
            raise ValueError(f"unknown lattice kind: {kind}")

        def _control_lattice_kind(self, kind: LatticeKind) -> LatticeKind:
            """Resolve callbacks from the shared TPMS panel to its active type."""

            if kind in TPMS_KINDS and "tpms_widgets" in self.__dict__:
                selected = self.tpms_widgets["kind"].currentData()
                if selected in TPMS_KINDS:
                    return selected
            return kind

        def _frame_input_values(self, kind: LatticeKind):
            kind = self._control_lattice_kind(kind)
            inputs = self._lattice_widgets(kind)["frame_inputs"]
            return {
                key: tuple(float(control.value()) for control in controls)
                for key, controls in inputs.items()
            }

        def _update_followed_frame_origin(self, kind: LatticeKind) -> CellMapFrame:
            """Move the Frame start to the design domain's UVW minimum."""

            kind = self._control_lattice_kind(kind)
            values = self._frame_input_values(kind)
            frame_points = self._active_domain_frame_points()
            if len(frame_points):
                frame = CellMapFrame.aligned_to_points_minimum(
                    frame_points,
                    values["u"],
                    values["v"],
                )
            else:
                frame = CellMapFrame.from_origin_axes(
                    (0.0, 0.0, 0.0),
                    values["u"],
                    values["v"],
                )
            origin_controls = self._lattice_widgets(kind)["frame_inputs"]["origin"]
            for control, value in zip(origin_controls, frame.origin):
                control.blockSignals(True)
                control.setValue(float(value))
                control.blockSignals(False)
            return frame

        def _on_frame_origin_mode_changed(
            self,
            kind: LatticeKind,
            automatic: bool,
        ) -> None:
            kind = self._control_lattice_kind(kind)
            origin_controls = self._lattice_widgets(kind)["frame_inputs"]["origin"]
            for control in origin_controls:
                control.setEnabled(not automatic)
            self._on_cell_map_frame_changed(kind)

        def _frame_is_default(self, kind: LatticeKind) -> bool:
            kind = self._control_lattice_kind(kind)
            values = self._frame_input_values(kind)
            default_origin = (
                self._active_domain_bounds()[0]
                if self.sole_mesh.faces.size
                else np.zeros(3, dtype=np.float64)
            )
            return bool(
                np.allclose(
                    values["origin"],
                    default_origin,
                    rtol=0.0,
                    atol=FRAME_ORIGIN_DEFAULT_ATOL_MM,
                )
                and np.allclose(values["u"], (1.0, 0.0, 0.0), rtol=0.0, atol=1.0e-9)
                and np.allclose(values["v"], (0.0, 1.0, 0.0), rtol=0.0, atol=1.0e-9)
            )

        def _sync_frame_boundary_mode(self, kind: LatticeKind) -> None:
            kind = self._control_lattice_kind(kind)
            combo = self._lattice_widgets(kind)["cell_map_mode"]
            is_default = self._frame_is_default(kind)
            if not is_default:
                complete_index = combo.findData("complete_cells")
                if complete_index >= 0 and combo.currentIndex() != complete_index:
                    combo.blockSignals(True)
                    combo.setCurrentIndex(complete_index)
                    combo.blockSignals(False)
            combo.setEnabled(is_default)
            combo.setToolTip(
                "自定义 Frame 保持目标晶胞尺寸并使用完整晶胞覆盖"
                if not is_default
                else "默认世界 Frame 可选择贴合设计域或完整晶胞扩展"
            )

        def _invalidate_lattice_dependents(self, kind: LatticeKind) -> None:
            self._mark_active_design_definition_changed({kind, "Transition"})
            changed_keys = (kind, "Transition")
            had_results = any(
                key in mapping
                for mapping in (
                    self.implicit_generation_results,
                    self.implicit_results,
                    self.implicit_fields,
                    self.raw_results,
                    self.repaired_results,
                    self.results,
                )
                for key in changed_keys
            )
            for mapping in (
                self.implicit_generation_results,
                self.implicit_results,
                self.implicit_fields,
                self.raw_results,
                self.repaired_results,
                self.results,
            ):
                for key in changed_keys:
                    mapping.pop(key, None)
            if had_results:
                self.use_repaired_result.setEnabled(bool(self.repaired_results))
                self.view_stl_mesh.setEnabled(bool(self.results))
                if not self.results:
                    self.view_stl_mesh.blockSignals(True)
                    self.view_stl_mesh.setChecked(False)
                    self.view_stl_mesh.blockSignals(False)
                    self.show_stl_edges.blockSignals(True)
                    self.show_stl_edges.setChecked(False)
                    self.show_stl_edges.setEnabled(False)
                    self.show_stl_edges.blockSignals(False)
                self._set_result_controls_enabled(bool(self.results))
            self._sync_stl_reconstruction_target_options()
            self._sync_section_target_options()
            self._update_stl_grid_estimate()

        def _on_cell_map_frame_changed(self, kind: LatticeKind) -> None:
            kind = self._control_lattice_kind(kind)
            widgets = self._lattice_widgets(kind)
            status = widgets["frame_status"]
            try:
                if widgets["frame_origin_follow"].isChecked():
                    frame = self._update_followed_frame_origin(kind)
                else:
                    values = self._frame_input_values(kind)
                    frame = CellMapFrame.from_origin_axes(
                        values["origin"], values["u"], values["v"]
                    )
            except ValueError as exc:
                status.setProperty("status", "error")
                status.style().unpolish(status)
                status.style().polish(status)
                status.setText(f"Frame 参数无效：{exc}")
                self.status.setText(f"{kind} Cell Map Frame 参数无效")
                return
            status.setProperty("status", "")
            status.style().unpolish(status)
            status.style().polish(status)
            status.setText(
                "原点 = "
                + ", ".join(f"{value:.3f}" for value in frame.origin)
                + " mm；W 轴 = "
                + ", ".join(f"{value:.4f}" for value in frame.w_axis)
            )
            self._sync_frame_boundary_mode(kind)
            self._invalidate_lattice_dependents(kind)
            self._update_transition_lattice_summary()
            self._update_recommendation()
            if (
                getattr(self.viewer, "_cell_map_preview", None) is not None
                and self.cell_map_kind.currentData() == kind
            ):
                self._show_cell_map(reset_view=False)
            else:
                self._refresh_scene(reset_view=False)
            self.status.setText(f"{kind} Cell Map Frame 已更新，请重新生成相关结果")

        def _reset_cell_map_frame(
            self,
            kind: LatticeKind,
            *,
            notify: bool = True,
        ) -> None:
            kind = self._control_lattice_kind(kind)
            widgets = self._lattice_widgets(kind)
            inputs = widgets["frame_inputs"]
            origin = (
                tuple(float(value) for value in self._active_domain_bounds()[0])
                if self.sole_mesh.faces.size
                else (0.0, 0.0, 0.0)
            )
            defaults = {
                "origin": origin,
                "u": (1.0, 0.0, 0.0),
                "v": (0.0, 1.0, 0.0),
            }
            for key, controls in inputs.items():
                for control, value in zip(controls, defaults[key]):
                    control.blockSignals(True)
                    control.setValue(value)
                    control.blockSignals(False)
            follow = widgets["frame_origin_follow"]
            follow.blockSignals(True)
            follow.setChecked(True)
            follow.blockSignals(False)
            for control in inputs["origin"]:
                control.setEnabled(False)
            frame_status = widgets["frame_status"]
            frame_status.setProperty("status", "")
            frame_status.style().unpolish(frame_status)
            frame_status.style().polish(frame_status)
            widgets["frame_status"].setText(
                "原点 = "
                + ", ".join(f"{value:.3f}" for value in origin)
                + " mm；W 轴 = 0.0000, 0.0000, 1.0000"
            )
            self._sync_frame_boundary_mode(kind)
            if notify:
                self._on_cell_map_frame_changed(kind)

        def _tpms_state_for_kind(self, kind: TPMSKind) -> TPMSParameterState:
            """Return live values for the active type and saved values otherwise."""

            if kind == self._active_tpms_kind:
                return self._tpms_state_from_widgets()
            return self._tpms_parameter_states[kind]

        def _tpms_parameters_from_state(
            self,
            kind: TPMSKind,
            state: TPMSParameterState,
        ) -> TPMSParameters:
            """Build parameters without temporarily mutating the visible panel."""

            frame_origin = tuple(float(value) for value in state["frame_origin_mm"])
            frame_u_axis = tuple(float(value) for value in state["frame_u_axis"])
            frame_v_axis = tuple(float(value) for value in state["frame_v_axis"])
            if bool(state["frame_origin_follow"]):
                frame_points = self._active_domain_frame_points()
                if len(frame_points):
                    resolved_frame = CellMapFrame.aligned_to_points_minimum(
                        frame_points,
                        frame_u_axis,
                        frame_v_axis,
                    )
                else:
                    resolved_frame = CellMapFrame.from_origin_axes(
                        (0.0, 0.0, 0.0),
                        frame_u_axis,
                        frame_v_axis,
                    )
                frame_origin = tuple(float(value) for value in resolved_frame.origin)
                frame_u_axis = tuple(float(value) for value in resolved_frame.u_axis)
                frame_v_axis = tuple(float(value) for value in resolved_frame.v_axis)
                default_origin = (
                    self._active_domain_bounds()[0]
                    if self.sole_mesh.faces.size
                    else np.zeros(3, dtype=np.float64)
                )
                uses_default_frame = (
                    resolved_frame.is_world_aligned
                    and np.allclose(
                        resolved_frame.origin,
                        default_origin,
                        rtol=0.0,
                        atol=FRAME_ORIGIN_DEFAULT_ATOL_MM,
                    )
                )
            else:
                frame = CellMapFrame.from_origin_axes(
                    frame_origin,
                    frame_u_axis,
                    frame_v_axis,
                )
                default_origin = (
                    self._active_domain_bounds()[0]
                    if self.sole_mesh.faces.size
                    else np.zeros(3, dtype=np.float64)
                )
                uses_default_frame = (
                    frame.is_world_aligned
                    and np.allclose(
                        frame.origin,
                        default_origin,
                        rtol=0.0,
                        atol=FRAME_ORIGIN_DEFAULT_ATOL_MM,
                    )
                )
            manual_counts = (
                tuple(int(value) for value in state["manual_cell_counts"])
                if bool(state["manual_cell_counts_enabled"])
                else None
            )
            return TPMSParameters(
                kind=kind,
                cell_size_mm=tuple(float(value) for value in state["cell_size_mm"]),
                wall_thickness_mm=float(state["wall_thickness_mm"]),
                level=float(state["level"]),
                cell_map_mode=state["cell_map_mode"],
                frame_origin_mm=None if uses_default_frame else frame_origin,
                frame_u_axis=frame_u_axis,
                frame_v_axis=frame_v_axis,
                gradient_enabled=bool(state["gradient_enabled"]),
                gradient_axis=state["gradient_axis"],
                gradient_mode=state["gradient_mode"],
                gradient_power=float(state["gradient_power"]),
                gradient_layers=int(state["gradient_layers"]),
                gradient_sigmoid_sharpness=float(
                    state["gradient_sigmoid_sharpness"]
                ),
                gradient_resolution_strategy=state[
                    "gradient_resolution_strategy"
                ],
                gradient_def_per_half_cell=int(
                    state["gradient_def_per_half_cell"]
                ),
                use_thickness_gradient=bool(state["use_thickness_gradient"]),
                thickness_soft_mm=float(state["thickness_soft_mm"]),
                thickness_stiff_mm=float(state["thickness_stiff_mm"]),
                use_offset_gradient=bool(state["use_offset_gradient"]),
                offset_soft=float(state["offset_soft"]),
                offset_stiff=float(state["offset_stiff"]),
                wall_thickness_method=state["wall_thickness_method"],
                manual_cell_counts=manual_counts,
            )

        def _params(self, kind: LatticeKind) -> LatticeParameters:
            if kind in TPMS_KINDS:
                return self._tpms_parameters_from_state(
                    kind,
                    self._tpms_state_for_kind(kind),
                )
            if kind != "Custom":
                raise ValueError(f"unknown lattice kind: {kind}")
            widgets = self._lattice_widgets(kind)
            frame_values = self._frame_input_values(kind)
            if widgets["frame_origin_follow"].isChecked():
                frame_points = self._active_domain_frame_points()
                if len(frame_points):
                    resolved_frame = CellMapFrame.aligned_to_points_minimum(
                        frame_points,
                        frame_values["u"],
                        frame_values["v"],
                    )
                else:
                    resolved_frame = CellMapFrame.from_origin_axes(
                        (0.0, 0.0, 0.0),
                        frame_values["u"],
                        frame_values["v"],
                    )
                frame_values = {
                    "origin": tuple(float(value) for value in resolved_frame.origin),
                    "u": tuple(float(value) for value in resolved_frame.u_axis),
                    "v": tuple(float(value) for value in resolved_frame.v_axis),
                }
                default_origin = (
                    self._active_domain_bounds()[0]
                    if self.sole_mesh.faces.size
                    else np.zeros(3, dtype=np.float64)
                )
                uses_default_frame = (
                    resolved_frame.is_world_aligned
                    and np.allclose(
                        resolved_frame.origin,
                        default_origin,
                        rtol=0.0,
                        atol=FRAME_ORIGIN_DEFAULT_ATOL_MM,
                    )
                )
            else:
                uses_default_frame = self._frame_is_default(kind)
            frame_origin = None if uses_default_frame else frame_values["origin"]
            return CustomUnitCellParameters(
                source_path=Path(widgets["source_path"].text().strip()),
                cell_size_mm=(
                    widgets["cell_x"].value(),
                    widgets["cell_y"].value(),
                    widgets["cell_z"].value(),
                ),
                cell_map_mode=widgets["cell_map_mode"].currentData(),
                frame_origin_mm=frame_origin,
                frame_u_axis=frame_values["u"],
                frame_v_axis=frame_values["v"],
                target_feature_mm=(
                    widgets["target_feature"].value()
                    if widgets["target_feature"].value() > 0.0
                    else None
                ),
                bridge_directions=tuple(
                    direction
                    for direction, check in widgets["bridge_checks"].items()
                    if check.isChecked()
                ),
                bridge_depth_mm=(
                    widgets["bridge_depth"].value()
                    if widgets["bridge_depth"].value() > 0.0
                    else None
                ),
            )

        def _selected_jobs(self):
            jobs = {}
            if self.tpms_widgets["enabled"].isChecked():
                kind = self.tpms_widgets["kind"].currentData()
                jobs[kind] = self._params(kind)
            if self.custom_widgets["enabled"].isChecked():
                jobs["Custom"] = self._params("Custom")
            return jobs

        def _selected_transition_job(self):
            if not self.transition_enabled.isChecked():
                return None
            driver = None
            if self._transition_driver_mode() == "field":
                primitive = self._selected_transition_primitive()
                if primitive is None:
                    raise ValueError("请选择一个场对象作为过渡区域")
                driver = primitive.as_implicit_body()
            job = (
                self._transition_operand_parameters(self.transition_first_controls),
                self._transition_operand_parameters(self.transition_second_controls),
                self._transition_parameters(),
            )
            return (*job, driver) if driver is not None else job

        def _preflight_report(
            self,
            jobs,
            sampling: SamplingParameters,
            transition_job: tuple[LatticeParameters, LatticeParameters, TransitionParameters] | tuple[LatticeParameters, LatticeParameters, TransitionParameters, ImplicitBody] | None = None,
        ) -> tuple[str, bool]:
            lines = ["模型生成前检查", "=" * 24]
            valid = True
            if not self.sole_mesh.faces.size:
                return "错误：设计域为空。", False
            domain = self._active_design_domain()
            bounds = self._active_domain_bounds()
            extent = bounds[1] - bounds[0]
            if isinstance(domain, AnalyticDesignDomain):
                lines.extend([
                    f"设计域：解析 {domain.primitive.kind} SDF（不经 STL 转换）",
                    f"尺寸：{extent[0]:.2f} × {extent[1]:.2f} × {extent[2]:.2f} mm",
                    "设计域 SDF 后端：解析求值",
                ])
            else:
                quality = inspect_mesh(self.sole_mesh)
                if not quality.finite:
                    lines.append("错误：设计域包含非有限坐标。")
                    valid = False
                if not quality.watertight:
                    valid = False
                    lines.append("警告：设计域 STL 不是封闭网格，SDF 符号可能不可靠。")
                lines.extend([
                    f"设计域：{len(self.sole_mesh.vertices):,} 个顶点 / {len(self.sole_mesh.faces):,} 个面",
                    f"尺寸：{extent[0]:.2f} × {extent[1]:.2f} × {extent[2]:.2f} mm",
                ])
                domain_volume = safe_mesh_volume(self.sole_mesh)
                lines.append(
                    f"体积：{domain_volume:.2f} mm³"
                    if domain_volume is not None
                    else "体积：不可可靠计算（设计域非水密）"
                )
            try:
                sampling.validate()
                if not isinstance(domain, AnalyticDesignDomain):
                    lines.append(
                        "设计域 SDF 后端："
                        + (_geometry_backend_label() if sampling.use_cpp_sdf else "PyVista")
                    )
                    if sampling.use_cpp_sdf and any(
                        isinstance(value, TPMSParameters) for value in jobs.values()
                    ):
                        lines.append(
                            "TPMS 裁剪流水线：" + _lattice_pipeline_backend_label()
                        )
                lines.append(f"目标体素数：{sampling.target_voxels:,}")
                display_batches = self._display_sampling_batch_count()
                lines.append(
                    f"隐式体显示采样：{display_batches} 批"
                    if display_batches is not None
                    else "隐式体显示采样：单批（内部自动微分片）"
                )
                mode_labels = {
                    "single_pass": "单次计算（不分块）",
                    "batched_field": "分批采样 + 单次 Marching Cubes",
                    "chunked_marching_cubes": "分批采样 + 分块 Marching Cubes",
                }
                lines.append(f"计算模式：{mode_labels[sampling.processing_mode]}")
                if sampling.processing_mode != "single_pass":
                    lines.append(f"分批数量：{sampling.batch_count} 批")
                for kind, params in jobs.items():
                    params.validate()
                    cell_map = self._cell_map_for_active_design(params)
                    if isinstance(params, CustomUnitCellParameters):
                        prepared = self.prepared_custom_cell
                        if (
                            prepared is None
                            or prepared.report.source_path != params.source_path.resolve()
                        ):
                            prepared = prepare_stl_unit_cell(params.source_path)
                            self.prepared_custom_cell = prepared
                        minimum_feature = params.effective_feature_mm(
                            prepared.characteristic_feature_mm(cell_map)
                        )
                        rec = self._sampling_for_active_design(
                            params,
                            sampling,
                            cell_map=cell_map,
                            minimum_feature_mm=minimum_feature,
                        )
                        report = prepared.report
                        lines.append(
                            f"{kind}：目标晶胞={params.cell_size_xyz_mm} mm, "
                            f"实际晶胞={tuple(round(value, 3) for value in cell_map.spacing_mm)} mm, "
                            f"特征厚度={minimum_feature:.3f} mm, "
                            f"voxel={rec.voxel_size_mm:.3f} mm, grid={rec.estimated_voxels:,}"
                        )
                        lines.append(
                            f"  源 STL：{report.prepared_faces:,} 个面，"
                            f"水密={report.watertight}，连通量={report.connected_components}，"
                            f"移除退化面={report.removed_degenerate_faces}"
                        )
                        seam_report = prepared.seam_report(
                            cell_map,
                            samples_per_axis=24,
                            target_feature_mm=params.target_feature_mm,
                            bridge_directions=params.bridge_directions,
                            bridge_depth_mm=params.bridge_depth_mm,
                        )
                        def seam_status(axis) -> str:
                            if axis.compatible:
                                return "通过"
                            if axis.bridge_enabled and axis.effective_compatible:
                                return "源不兼容→桥接后预计通过"
                            if axis.bridge_enabled:
                                return "桥接深度内无实体，仍不兼容"
                            return "不兼容"

                        seam_summary = ", ".join(
                            f"{axis.direction}={seam_status(axis)}"
                            f"({axis.union_solid_area_mm2:.2f} mm²，"
                            f"探测深度 {axis.probe_depth_mm:.3f} mm)"
                            for axis in seam_report.axes
                        )
                        lines.append(f"  周期接缝检查：{seam_summary}（仅警告）")
                        if params.bridge_directions:
                            bridge_text = ", ".join(params.bridge_directions)
                            bridge_depth_text = (
                                "自动"
                                if params.bridge_depth_mm is None
                                else f"{params.bridge_depth_mm:.3f} mm"
                            )
                            lines.append(
                                f"  已启用桥接：{bridge_text}，深度="
                                f"{bridge_depth_text}"
                            )
                    else:
                        rec = self._sampling_for_active_design(
                            params,
                            sampling,
                            cell_map=cell_map,
                        )
                        lines.append(
                            f"{kind}：目标晶胞={params.cell_size_xyz_mm} mm, "
                            f"实际晶胞={tuple(round(value, 3) for value in cell_map.spacing_mm)} mm, "
                            f"wall={params.minimum_wall_thickness_mm:.2f} mm, "
                            f"voxel={rec.voxel_size_mm:.3f} mm, grid={rec.estimated_voxels:,}"
                        )
                        if params.gradient_enabled:
                            resolution_text = (
                                f"MATLAB Def={params.gradient_def_per_half_cell}"
                                if params.gradient_resolution_strategy == "matlab_def"
                                else "自动推荐"
                            )
                            lines.append(
                                f"  梯度：方向={params.gradient_axis}，曲线={params.gradient_mode}，"
                                f"壁厚方法={params.wall_thickness_method}，"
                                f"分辨率={resolution_text}"
                            )
                    if rec.samples_per_wall < sampling.min_samples_per_wall:
                        lines.append(f"警告：{kind} 壁厚采样数 {rec.samples_per_wall:.1f} 低于建议值 {sampling.min_samples_per_wall:.1f}。")
            except ValueError as exc:
                lines.append(f"错误：{exc}")
                valid = False
            if transition_job is not None:
                try:
                    first_parameters, second_parameters, transition = transition_job[:3]
                    first_parameters.validate()
                    second_parameters.validate()
                    transition.validate()
                    transition_recommendation = min(
                        (
                            self._sampling_for_active_design(
                                first_parameters, sampling
                            ),
                            self._sampling_for_active_design(
                                second_parameters, sampling
                            ),
                        ),
                        key=lambda item: item.voxel_size_mm,
                    )
                    lines.append(
                        f"过渡：{first_parameters.kind} → {second_parameters.kind}，"
                        f"Ramp={transition.weight_kind}，分界面={transition.plane_axis}，"
                        f"宽度={transition.transition_width_mm:.2f} mm，"
                        f"体素={transition_recommendation.voxel_size_mm:.3f} mm，"
                        f"网格={transition_recommendation.estimated_voxels:,}"
                    )
                    if transition_recommendation.samples_per_wall < sampling.min_samples_per_wall:
                        lines.append("警告：过渡结果的壁厚采样数低于建议值。")
                    if transition.driver_mode == "field":
                        lines.append(
                            "场对象过渡："
                            f"对象={transition.driver_identifier}，"
                            f"区间=[{transition.field_interval_lower_mm:.3f}, "
                            f"{transition.field_interval_upper_mm:.3f}] mm；"
                            "曲面场不采用平面跨带质量检查。"
                        )
                except ValueError as exc:
                    lines.append(f"错误：过渡参数无效：{exc}")
                    valid = False
            return "\n".join(lines), valid

        def _show_preflight(
            self,
            jobs,
            sampling: SamplingParameters,
            transition_job: tuple[LatticeParameters, LatticeParameters, TransitionParameters] | tuple[LatticeParameters, LatticeParameters, TransitionParameters, ImplicitBody] | None = None,
        ) -> bool:
            report, valid = self._preflight_report(jobs, sampling, transition_job)
            box = QtWidgets.QMessageBox(self)
            box.setWindowTitle("生成前模型检查")
            box.setIcon(QtWidgets.QMessageBox.Information if valid else QtWidgets.QMessageBox.Warning)
            box.setText("检查通过，可以开始生成。" if valid else "检查发现问题，请先修正参数或模型。")
            box.setDetailedText(report)
            box.setStandardButtons(QtWidgets.QMessageBox.Ok | QtWidgets.QMessageBox.Cancel)
            box.setDefaultButton(QtWidgets.QMessageBox.Ok)
            if not valid:
                box.button(QtWidgets.QMessageBox.Ok).setEnabled(False)
            return box.exec_() == QtWidgets.QMessageBox.Ok

        def _result_report(self) -> str:
            if not self.results and self.implicit_results:
                lines = ["当前隐式结果检查", "=" * 24]
                for kind, body in self.implicit_results.items():
                    field = self.implicit_fields.get(kind)
                    cache_text = "无显示缓存"
                    if field is not None:
                        cache_text = (
                            f"显示网格={field.values.shape[0]}×{field.values.shape[1]}×{field.values.shape[2]}，"
                            f"缓存={field.estimated_bytes / (1024 * 1024):.1f} MB"
                        )
                    state = "仅预览" if body.is_preview_only else "可用于实体运算"
                    lines.append(f"{kind}：隐式体={state}，{cache_text}")
                return "\n".join(lines)
            if not self.results:
                return "当前没有已生成的晶格结果。"
            lines = ["当前结果检查", "=" * 24]
            domain_bounds = self.sole_mesh.bounds
            tolerance = 2.0 * float(self.export_tolerance.value())
            for kind, mesh in self.results.items():
                finite = bool(np.isfinite(mesh.vertices).all())
                outside = bool(((mesh.vertices < domain_bounds[0] - tolerance) | (mesh.vertices > domain_bounds[1] + tolerance)).any())
                quality = inspect_mesh(mesh)
                lines.append(
                    f"  质量：水密={quality.watertight}，"
                    f"方向一致={quality.winding_consistent}，连通分量={quality.connected_components}"
                )
                lines.append(f"{kind}：{len(mesh.vertices):,} 个顶点 / {len(mesh.faces):,} 个面")
                result_volume = safe_mesh_volume(mesh)
                volume_text = (
                    f"{result_volume:.2f} mm³"
                    if result_volume is not None
                    else "不可可靠计算（非水密）"
                )
                lines.append(
                    f"  有限数据={finite}，水密={mesh.is_watertight}，体积={volume_text}"
                )
                if outside:
                    lines.append("  警告：发现超出设计域包围盒的顶点。")
                if not mesh.is_watertight:
                    lines.append("  警告：结果网格不是封闭网格，导出前应确认是否符合制造要求。")
            return "\n".join(lines)

        def _repair_design_domain(self):
            document = self.design_workspace.active_document
            if (
                document is None
                or not isinstance(document.domain, MeshDesignDomain)
                or not self.sole_mesh.faces.size
            ):
                QtWidgets.QMessageBox.warning(self, "No design domain", "Load an STL first.")
                return
            source_domain = document.domain
            before = inspect_mesh(source_domain.mesh)
            try:
                repaired = repair_mesh(source_domain.mesh)
            except (TypeError, ValueError, RuntimeError) as exc:
                QtWidgets.QMessageBox.critical(self, "Design-domain repair failed", str(exc))
                return
            after = inspect_mesh(repaired)
            document.replace_domain(
                MeshDesignDomain(
                    mesh=repaired,
                    name=source_domain.name,
                    asset_path=source_domain.asset_path,
                )
            )
            self.design_workspace.dirty = True
            self.sole_mesh = repaired
            if all(
                name in self.__dict__
                for name in ("tpms_widgets", "custom_widgets")
            ):
                self._reset_cell_map_frame(self._active_tpms_kind, notify=False)
                self._tpms_parameter_states[
                    self._active_tpms_kind
                ] = self._tpms_state_from_widgets()
                self._reset_cell_map_frame("Custom", notify=False)
            self.raw_results = {}
            self.repaired_results = {}
            self.results = {}
            self.implicit_results = {}
            self.implicit_generation_results = {}
            self.backend_generation_handles = {}
            self.implicit_fields = {}
            self.domain_implicit_field = None
            self._shell_fusion_domain_field = None
            self._reset_applied_sampling_info()
            self._set_result_controls_enabled(False)
            self._load_sole_mesh_metadata()
            self._update_recommendation()
            self._refresh_domain_implicit_preview()
            self._refresh_scene(reset_view=True)
            detail = (
                f"Before: {before.connected_components} component(s), "
                f"watertight={before.watertight}\n"
                f"After: {after.connected_components} component(s), "
                f"watertight={after.watertight}, winding={after.winding_consistent}\n\n"
                "Simple defects were repaired. Regenerate TPMS results after changing the design domain."
            )
            box = QtWidgets.QMessageBox(self)
            box.setWindowTitle("Design-domain repair result")
            box.setIcon(QtWidgets.QMessageBox.Information if after.watertight else QtWidgets.QMessageBox.Warning)
            box.setText("Repair completed." if after.watertight else "Repair completed, but the mesh is still not watertight.")
            box.setDetailedText(detail)
            box.exec_()

        def _load_sole_mesh_metadata(self):
            bounds = self.sole_mesh.bounds
            extent = bounds[1] - bounds[0]
            self.domain_info.setText(
                f"{len(self.sole_mesh.faces):,} 个面 · "
                f"{extent[0]:.1f} × {extent[1]:.1f} × {extent[2]:.1f} mm"
            )

        def _refresh_domain_implicit_preview(self) -> None:
            """Build the design-domain display field without generating TPMS."""

            if not self.sole_mesh.faces.size or not hasattr(self.viewer, "set_implicit_fields"):
                return
            try:
                sampling = self._current_sampling()
                document = self.design_workspace.active_document
                if document is not None and isinstance(
                    document.domain, AnalyticDesignDomain
                ):
                    base = _domain_sampling_recommendation(
                        document.domain,
                        self._params("G"),
                        sampling,
                    )
                    self.domain_implicit_body = document.domain.as_implicit_body()
                    self.domain_implicit_field = (
                        generate_implicit_domain_preview_for_design_domain(
                            document.domain,
                            sampling,
                            base,
                            display_voxel_size_mm=self._display_voxel_size(),
                            display_memory_budget_mb=self._display_memory_budget_mb(),
                            visible_field_count=1,
                            display_batch_count=self._display_sampling_batch_count(),
                        )
                    )
                else:
                    base = lattice_sampling_recommendation(
                        self.sole_mesh, self._params("G"), sampling
                    )
                    self.domain_implicit_body = _make_design_domain_body(
                        self.sole_mesh,
                        sampling,
                        shell_thickness_mm=max(base.voxel_size_mm, 0.25),
                    )
                    self.domain_implicit_field = generate_implicit_domain_preview(
                        self.sole_mesh,
                        sampling,
                        base,
                        display_voxel_size_mm=self._display_voxel_size(),
                        display_memory_budget_mb=self._display_memory_budget_mb(),
                        visible_field_count=1,
                        display_batch_count=self._display_sampling_batch_count(),
                    )
                self._sync_contour_result_options()
                self._sync_field_viewer_options()
                self._sync_stl_reconstruction_target_options()
                self._sync_section_target_options()
                state = "预览薄壳" if self.domain_implicit_field.is_preview_only else "隐式体"
                self.domain_info.setText(self.domain_info.text() + f" · {state}显示已就绪")
            except (ValueError, RuntimeError) as exc:
                self.domain_implicit_body = None
                self.domain_implicit_field = None
                self.status.setText(f"设计域隐式预览失败：{exc}")

        def _repair_results(self):
            if self._selected_stl_reconstruction_results():
                self._start_stl_reconstruction(repair=True)
                return
            if not self.results:
                return
            repaired = {}
            try:
                for kind, mesh in self.results.items():
                    repaired[kind] = repair_mesh(mesh)
            except (TypeError, ValueError, RuntimeError) as exc:
                QtWidgets.QMessageBox.critical(self, "Result repair failed", str(exc))
                return
            self.repaired_results = repaired
            self.use_repaired_result.setEnabled(True)
            self.use_repaired_result.setChecked(True)
            self._select_result_variant(True)
            self._set_result_controls_enabled(True)
            self.status.setText("网格修复完成，请重新检查拓扑状态。")
            self._refresh_scene(reset_view=False)

        def _start_stl_reconstruction(self, repair: bool):
            backend_generation = self.backend_generation_handles.get(
                self.stl_reconstruction_target_combo.currentData()
            )
            if backend_generation is not None:
                if self.stl_thread is None:
                    self._start_backend_stl_reconstruction(backend_generation, repair)
                return
            selected_results = self._selected_stl_reconstruction_results()
            if not selected_results or self.stl_thread is not None:
                if (
                    self.implicit_generation_results
                    or self._analytic_design_reconstruction_target() is not None
                ) and not selected_results:
                    self.status.setText("请先选择需要重建的隐式体对象")
                return
            tolerance = float(self.export_tolerance.value())
            repair_tolerance = float(self.repair_tolerance.value()) if repair else 0.0
            self.build_stl_button.setEnabled(False)
            self.repair_result_button.setEnabled(False)
            self.stl_reconstruction_target_combo.setEnabled(False)
            self.progress.setValue(0)
            self.status.setText("正在从权威隐式场重建 STL……")
            self.stl_thread = QtCore.QThread(self)
            sampling = self._current_sampling()
            self.stl_worker = StlReconstructionWorker(
                selected_results,
                tolerance,
                self._export_spacing_mode(),
                repair_tolerance,
                self.clean_numerical_fragments.isChecked(),
                sampling.processing_mode,
                sampling.batch_count,
                self.optimize_stl_for_slicing.isChecked(),
            )
            self.stl_worker.moveToThread(self.stl_thread)
            self.stl_thread.started.connect(self.stl_worker.run)
            self.stl_worker.progress.connect(
                lambda message, value: (
                    self.status.setText(message),
                    self.progress.setValue(int(value * 100)),
                )
            )
            self.stl_worker.finished.connect(
                lambda outputs, mode=repair: self._stl_reconstruction_finished(
                    outputs, mode
                )
            )
            self.stl_worker.failed.connect(self._stl_reconstruction_failed)
            self.stl_worker.finished.connect(self.stl_thread.quit)
            self.stl_worker.failed.connect(self.stl_thread.quit)
            self.stl_thread.finished.connect(self._stl_reconstruction_cleanup)
            self.stl_thread.start()

        def _start_backend_stl_reconstruction(
            self,
            generation: BackendGenerationResult,
            repair: bool,
        ) -> None:
            """Reconstruct a backend handle without returning an implicit body."""

            client = self.backend_client
            if client is None:
                self._stl_reconstruction_failed("local backend is unavailable")
                return
            sampling = self._current_sampling()
            payload = {
                "generation_id": generation.generation_id,
                "tolerance_mm": float(self.export_tolerance.value()),
                "repair_tolerance_mm": (
                    float(self.repair_tolerance.value()) if repair else 0.0
                ),
                "spacing_mode": self._export_spacing_mode(),
                "clean_numerical_fragments": self.clean_numerical_fragments.isChecked(),
                "processing_mode": sampling.processing_mode,
                "batch_count": sampling.batch_count,
                "optimize_for_slicing": self.optimize_stl_for_slicing.isChecked(),
            }

            def result_reader(snapshot: object) -> BackendStlReconstructionResult:
                if snapshot.result is None:
                    raise ValueError("reconstruction task returned no result")
                artifacts = snapshot.artifacts
                if len(artifacts) != 1 or artifacts[0].name != "lattice.stl":
                    raise ValueError("reconstruction task must publish lattice.stl")
                mesh = trimesh.load_mesh(
                    BytesIO(client.download_artifact(artifacts[0].identifier)),
                    file_type="stl",
                    process=False,
                )
                if not isinstance(mesh, trimesh.Trimesh):
                    raise ValueError("reconstruction artifact is not a triangular mesh")
                return BackendStlReconstructionResult(
                    kind=generation.kind,
                    mesh=mesh,
                    triangle_count=int(snapshot.result["triangle_count"]),
                )

            self.build_stl_button.setEnabled(False)
            self.repair_result_button.setEnabled(False)
            self.stl_reconstruction_target_combo.setEnabled(False)
            self.progress.setValue(0)
            self.status.setText("正在由本地后端重建 STL……")
            self.stl_thread = QtCore.QThread(self)
            self.stl_worker = BackendTaskWorker(
                client,
                "stl.reconstruct",
                payload,
                result_reader,
            )
            self.stl_worker.moveToThread(self.stl_thread)
            self.stl_thread.started.connect(self.stl_worker.run)
            self.stl_worker.progress.connect(
                lambda message, value: (
                    self.status.setText(message),
                    self.progress.setValue(int(value * 100)),
                )
            )
            self.stl_worker.finished.connect(
                lambda result, mode=repair: self._backend_stl_reconstruction_finished(
                    result,
                    mode,
                )
            )
            self.stl_worker.failed.connect(self._stl_reconstruction_failed)
            self.stl_worker.finished.connect(self.stl_thread.quit)
            self.stl_worker.failed.connect(self.stl_thread.quit)
            self.stl_thread.finished.connect(self._stl_reconstruction_cleanup)
            self.stl_thread.start()

        def _backend_stl_reconstruction_finished(
            self,
            result: BackendStlReconstructionResult,
            repair: bool,
        ) -> None:
            if getattr(self, "_is_closing", False):
                return
            self._layer_contour_cache.clear()
            self._field_plane_cache.clear()
            mesh = result.mesh.copy()
            self.stl_reconstruction_results = {}
            if repair:
                self.repaired_results = {result.kind: mesh}
                self.use_repaired_result.setEnabled(True)
                self.use_repaired_result.setChecked(True)
            else:
                self.raw_results = {result.kind: mesh}
                self.repaired_results = {}
                self.use_repaired_result.blockSignals(True)
                self.use_repaired_result.setChecked(False)
                self.use_repaired_result.blockSignals(False)
                self.use_repaired_result.setEnabled(False)
            self._select_result_variant(self.use_repaired_result.isChecked())
            self.view_stl_mesh.setEnabled(True)
            self.view_stl_mesh.setChecked(True)
            self.show_stl_edges.setEnabled(True)
            self.progress.setValue(100)
            self._stl_sampling_summary = (
                f"STL 重建：{result.kind}；后端产物；"
                f"三角面 {result.triangle_count:,}"
            )
            self._refresh_applied_sampling_info()
            self.status.setText(
                f"后端 STL 重建完成：{result.kind}；三角面 {result.triangle_count:,}"
            )
            self._set_result_controls_enabled(True)
            self._refresh_scene(reset_view=True)
            self._store_active_design_state()

        def _stl_reconstruction_finished(self, outputs, repair: bool):
            if getattr(self, "_is_closing", False):
                return
            self._layer_contour_cache.clear()
            self._field_plane_cache.clear()
            self.stl_reconstruction_results = dict(outputs)
            if getattr(self, "field_viewer_enabled", None) is not None and self.field_viewer_enabled.isChecked():
                self._schedule_field_viewer_refresh()
            meshes = {kind: result.mesh.copy() for kind, result in outputs.items()}
            if repair:
                self.repaired_results = meshes
                self.use_repaired_result.setEnabled(True)
                self.use_repaired_result.setChecked(True)
            else:
                self.raw_results = meshes
                self.repaired_results = {}
                self.use_repaired_result.blockSignals(True)
                self.use_repaired_result.setChecked(False)
                self.use_repaired_result.blockSignals(False)
                self.use_repaired_result.setEnabled(False)
            self._select_result_variant(self.use_repaired_result.isChecked())
            self.view_stl_mesh.setEnabled(True)
            self.view_stl_mesh.setChecked(True)
            self.show_stl_edges.setEnabled(True)
            self.progress.setValue(100)
            status_parts = []
            sampling_reports = []
            for kind, result in outputs.items():
                cleanup = result.fragment_cleanup
                cleanup_text = ""
                field_cleanup = result.field_fragment_cleanup
                if field_cleanup is not None and field_cleanup.removed_components:
                    cleanup_text = (
                        f"，隐式场清理={field_cleanup.removed_components} 个悬空分量/"
                        f"{field_cleanup.removed_voxels:,} 个采样点"
                    )
                if cleanup is not None:
                    cleanup_text = (
                        cleanup_text
                        + f"，数值碎片={cleanup.removed_components} 个/"
                        f"{cleanup.removed_faces:,} 面/"
                        f"{cleanup.removed_volume_mm3:.6g} mm³"
                    )
                conditioning_text = ""
                if result.conditioning is not None:
                    report = result.conditioning
                    if report.applied:
                        conditioning_text = (
                            f"，切片优化减少 {report.removed_faces:,} 面，"
                            f"最大偏差 {report.max_deviation_mm:.4f} mm"
                        )
                    else:
                        conditioning_text = "，切片优化未采用（质量门控回滚）"
                extraction_text = "稠密网格"
                if result.extraction_backend == "chunked-marching-cubes":
                    extraction_text = "分块 Marching Cubes（已拼接）"
                elif result.extraction_backend == "batched-field-dense-grid":
                    extraction_text = "分批采样 + 单次 Marching Cubes"
                elif result.extraction_backend == "dense-grid-repair":
                    extraction_text = "完整场 + 修复 Marching Cubes"
                if result.extraction_fallback_reason:
                    extraction_text += f"（{result.extraction_fallback_reason}）"
                if len(set(result.spacing_limits)) == 1:
                    spacing_limit_text = result.spacing_limits[0]
                else:
                    spacing_limit_text = "，".join(
                        f"{axis}={limit}"
                        for axis, limit in zip(
                            "XYZ", result.spacing_limits, strict=True
                        )
                    )
                spacing_text = _format_spacing_mm(result.spacing_mm)
                input_label = (
                    "用户指定间距"
                    if result.spacing_mode == "exact"
                    else "输入采样上限"
                )
                voxel_text = ""
                if result.extraction_statistics is not None:
                    voxel_text = (
                        f"，规则网格参考体素 {result.extraction_statistics.dense_grid_points:,}，"
                        f"实际评估点 {result.extraction_statistics.evaluated_points:,}"
                    )
                sampling_reports.append(
                    f"{kind}：{input_label} {result.tolerance_mm:.4f} mm，"
                    f"实际间距 {spacing_text}，决定约束：{spacing_limit_text}"
                    f"{voxel_text}"
                )
                status_parts.append(
                    f"{kind}: 水密={result.quality.watertight}, "
                    f"连通量={result.quality.connected_components}, "
                    f"提取={extraction_text}, "
                    f"{input_label}={result.tolerance_mm:.4f} mm, "
                    f"实际采样={spacing_text}, "
                    f"限制={spacing_limit_text}, "
                    f"面片={len(result.mesh.faces):,}{cleanup_text}{conditioning_text}"
                )
            self._stl_sampling_summary = (
                "STL 重建：\n" + "\n".join(sampling_reports)
                if sampling_reports
                else "STL 重建：无结果"
            )
            self._refresh_applied_sampling_info()
            status = "；".join(status_parts)
            prefix = "修复重建完成" if repair else "STL 网格生成完成"
            self.status.setText(f"{prefix}；{status}")
            self._set_result_controls_enabled(True)
            self._refresh_scene(reset_view=True)

        def _stl_reconstruction_failed(self, message: str):
            if getattr(self, "_is_closing", False):
                return
            self.status.setText("STL 重建失败：" + message)
            QtWidgets.QMessageBox.critical(self, "STL 重建失败", message)

        def _stl_reconstruction_cleanup(self):
            self.stl_worker = None
            self.stl_thread = None
            self._sync_stl_reconstruction_target_options()
            self._set_result_controls_enabled(bool(self.results))

        def _select_result_variant(self, repaired: bool):
            source = self.repaired_results if repaired and self.repaired_results else self.raw_results
            self.results = {kind: mesh.copy() for kind, mesh in source.items()}
            self._set_result_controls_enabled(bool(self.results))
            if self.results:
                self._refresh_scene(reset_view=False)

        def _check_model(self):
            if self.results or self.implicit_results:
                report = self._result_report()
                title = "结果模型检查"
            else:
                jobs = self._selected_jobs()
                transition_job = self._selected_transition_job()
                if not jobs and transition_job is None:
                    QtWidgets.QMessageBox.warning(
                        self,
                        "无法检查",
                        "请至少勾选 G、D、自定义晶胞，或启用晶胞过渡。",
                    )
                    return
                report, _ = self._preflight_report(
                    jobs,
                    self._current_sampling(),
                    transition_job,
                )
                title = "生成前模型检查"
            QtWidgets.QMessageBox.information(self, title, report)

        def _set_result_controls_enabled(self, enabled: bool):
            self.simplify_button.setEnabled(enabled)
            has_stl_target = bool(self._selected_stl_reconstruction_results())
            self.repair_result_button.setEnabled(
                enabled or has_stl_target
            )
            self.build_stl_button.setEnabled(has_stl_target and self.stl_thread is None)
            self.export_g_button.setEnabled(enabled and "G" in self.results)
            self.export_d_button.setEnabled(enabled and "D" in self.results)
            self.export_custom_button.setEnabled(
                enabled and "Custom" in self.results
            )
            self.export_transition_button.setEnabled(enabled and "Transition" in self.results)
            self.export_shell_union_button.setEnabled(
                enabled and SHELL_UNION_RESULT_KEY in self.results
            )
            self.export_all_button.setEnabled(enabled and bool(self.results))
            selected_kind = self.stl_reconstruction_target_combo.currentData()
            self.export_selected_stl_button.setEnabled(
                isinstance(selected_kind, str) and selected_kind in self.results
            )

        def _simplify_results(self):
            source_results = (
                self.repaired_results
                if self.use_repaired_result.isChecked() and self.repaired_results
                else self.raw_results
            )
            if not source_results:
                return
            percentage = float(self.simplify_percent.value())
            if percentage <= 0.0:
                self.results = {kind: mesh.copy() for kind, mesh in source_results.items()}
                self.status.setText("已恢复原始面片")
                self._refresh_scene(reset_view=False)
                return
            try:
                simplified = {}
                reports = []
                tolerance = float(self.export_tolerance.value())
                for kind, source in source_results.items():
                    target = max(4, int(round(len(source.faces) * (1.0 - percentage / 100.0))))
                    candidate, report = simplify_mesh_with_quality(
                        source,
                        target,
                        tolerance,
                    )
                    simplified[kind] = candidate.copy()
                    reports.append(f"{kind}: {report.message}")
                self.results = simplified
                self.status.setText("；".join(reports))
                self._refresh_scene(reset_view=False)
            except Exception as exc:
                QtWidgets.QMessageBox.critical(self, "简化失败", f"{type(exc).__name__}: {exc}")

        def _export_result(self, kind: str):
            mesh = self.results.get(kind)
            if mesh is None or not mesh.faces.size:
                QtWidgets.QMessageBox.warning(self, "无法导出", f"没有可导出的 {kind} 结果。")
                return
            quality = inspect_mesh(mesh)
            if not (
                quality.watertight
                and quality.winding_consistent
                and quality.connected_components == 1
            ):
                answer = QtWidgets.QMessageBox.warning(
                    self,
                    "结果网格检查警告",
                    f"{kind} 拓扑不合格：水密={quality.watertight}，"
                    f"法向一致={quality.winding_consistent}，"
                    f"连通量={quality.connected_components}。\n仍然导出 STL？",
                    QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
                    QtWidgets.QMessageBox.No,
                )
                if answer != QtWidgets.QMessageBox.Yes:
                    return
            suffix = (
                "design_domain"
                if kind == "domain"
                else ("custom_lattice" if kind == "Custom" else "tpms")
            )
            default_path = self.root / "build" / "exports" / f"{kind}_{suffix}.stl"
            path, _ = QtWidgets.QFileDialog.getSaveFileName(self, f"导出 {kind} STL", str(default_path), "STL (*.stl)")
            if path:
                export_path = Path(path).with_suffix(".stl")
                mesh.export(export_path)
                reloaded = trimesh.load(export_path, force="mesh", process=True)
                exported_quality = inspect_mesh(reloaded)
                self.status.setText(
                    f"已导出：{export_path}；水密={exported_quality.watertight}，"
                    f"连通量={exported_quality.connected_components}"
                )

        def _export_selected_stl_reconstruction(self) -> None:
            """Export the active reconstruction target, including analytic domains."""

            kind = self.stl_reconstruction_target_combo.currentData()
            if not isinstance(kind, str) or kind not in self.results:
                QtWidgets.QMessageBox.warning(
                    self,
                    "无法导出",
                    "当前 STL 重建对象尚未生成网格。",
                )
                return
            self._export_result(kind)

        def _export_all_results(self):
            if not self.results:
                return
            invalid = []
            for kind, mesh in self.results.items():
                quality = inspect_mesh(mesh)
                if not (
                    quality.watertight
                    and quality.winding_consistent
                    and quality.connected_components == 1
                ):
                    invalid.append(
                        f"{kind}: 水密={quality.watertight}, "
                        f"法向一致={quality.winding_consistent}, "
                        f"连通量={quality.connected_components}"
                    )
            if invalid:
                answer = QtWidgets.QMessageBox.warning(
                    self,
                    "结果网格检查警告",
                    "以下结果拓扑不合格：\n"
                    + "\n".join(invalid)
                    + "\n仍然导出全部 STL？",
                    QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
                    QtWidgets.QMessageBox.No,
                )
                if answer != QtWidgets.QMessageBox.Yes:
                    return
            directory = QtWidgets.QFileDialog.getExistingDirectory(self, "选择导出目录", str(self.root / "build" / "exports"))
            if not directory:
                return
            output_dir = Path(directory)
            exported = []
            for kind, mesh in self.results.items():
                suffix = (
                    "design_domain"
                    if kind == "domain"
                    else ("custom_lattice" if kind == "Custom" else "tpms")
                )
                path = output_dir / f"{kind}_{suffix}.stl"
                mesh.export(path)
                reloaded = trimesh.load(path, force="mesh", process=True)
                quality = inspect_mesh(reloaded)
                exported.append(
                    f"{path.name}(水密={quality.watertight}, 连通量={quality.connected_components})"
                )
            self.status.setText("已导出：" + ", ".join(exported))

        def _update_recommendation(self):
            if not self.sole_mesh.faces.size:
                return
            try:
                sampling = self._current_sampling()
                selected = list(self._selected_jobs().values())
                if self.transition_enabled.isChecked():
                    selected.extend(
                        (
                            self._transition_operand_parameters(self.transition_first_controls),
                            self._transition_operand_parameters(self.transition_second_controls),
                        )
                    )
                if not selected:
                    selected = [self._params("G"), self._params("D")]
                recs = []
                cell_maps: dict[str, CellMap] = {}
                for parameters in selected:
                    cell_map = self._cell_map_for_active_design(parameters)
                    cell_maps[parameters.kind] = cell_map
                    minimum_feature = None
                    if (
                        isinstance(parameters, CustomUnitCellParameters)
                        and self.prepared_custom_cell is not None
                    ):
                        minimum_feature = parameters.effective_feature_mm(
                            self.prepared_custom_cell.characteristic_feature_mm(
                                cell_map
                            )
                        )
                    recs.append(
                        self._sampling_for_active_design(
                            parameters,
                            sampling,
                            cell_map=cell_map,
                            minimum_feature_mm=minimum_feature,
                        )
                    )

                # The unified TPMS selector chooses the active generation
                # operand, but the sampling panel historically showed the
                # comparable G/D estimates side by side.  Keep that useful
                # reference information for the two baseline families
                # without allowing an inactive reference to change the
                # display spacing chosen for the active job.
                if (
                    not self.transition_enabled.isChecked()
                    and not self.custom_widgets["enabled"].isChecked()
                    and len(selected) == 1
                    and selected[0].kind in ("G", "D")
                ):
                    reference_kind = "D" if selected[0].kind == "G" else "G"
                    reference_parameters = self._params(reference_kind)
                    reference_map = self._cell_map_for_active_design(
                        reference_parameters
                    )
                    cell_maps[reference_kind] = reference_map
                rec = min(recs, key=lambda item: item.voxel_size_mm)
                display_step = _display_recommendation_for_bounds(
                    self._active_domain_bounds(),
                    rec,
                    None,
                    self._display_memory_budget_mb(),
                    max(1, len(self._selected_jobs()) + int(self.transition_enabled.isChecked()) + 1),
                ).voxel_size_mm
                if not self.display_voxel_auto.isChecked():
                    # Manual preview spacing is authoritative; the memory budget
                    # remains an informational risk indicator in this mode.
                    display_step = float(self.display_voxel_size.value())
                else:
                    self.display_voxel_size.blockSignals(True)
                    self.display_voxel_size.setValue(display_step)
                    self.display_voxel_size.blockSignals(False)
                display_batches = self._display_sampling_batch_count()
                cell_sample_lines = []
                for kind, cell_map in cell_maps.items():
                    estimate = estimate_cell_display_samples(
                        cell_map,
                        display_step,
                    )
                    u_samples, v_samples, w_samples = estimate.samples_per_axis
                    cell_sample_lines.append(
                        f"{kind}：U/V/W = {u_samples:.1f}/{v_samples:.1f}/{w_samples:.1f} samples；"
                        f"单晶胞约 {estimate.voxel_samples_per_cell:,.0f} voxel/samples"
                    )
                self.cell_sample_info.setText("\n".join(cell_sample_lines))
                self.voxel_info.setText(
                    f"推荐间距：{rec.voxel_size_mm:.4f} mm；"
                    f"估算网格：{rec.estimated_voxels:,} voxels；"
                    f"{rec.samples_per_cell:.1f} samples/cell；"
                    f"{rec.samples_per_wall:.1f} samples/wall\n"
                    f"实时显示：{display_step:.4f} mm，约 {int(np.prod(np.ceil((self._active_domain_bounds()[1] - self._active_domain_bounds()[0]) / display_step).astype(int) + 3)):,} voxels/层；"
                    + (
                        f"分 {display_batches} 批采样"
                        if display_batches is not None
                        else "单批采样（内部自动微分片）"
                    )
                )
            except ValueError as exc:
                self.voxel_info.setText(str(exc))
                if "cell_sample_info" in self.__dict__:
                    self.cell_sample_info.setText(f"无法估算：{exc}")

        def _generate(self):
            document = self.design_workspace.active_document
            if document is None or not self.sole_mesh.faces.size:
                return
            jobs = self._selected_jobs()
            transition_job = self._selected_transition_job()
            if not jobs and transition_job is None:
                QtWidgets.QMessageBox.warning(
                    self,
                    "没有生成对象",
                    "请至少启用一个 TPMS/自定义晶胞，或启用晶胞过渡。",
                )
                return
            try:
                sampling = self._current_sampling()
                display_voxel_size = self._display_voxel_size()
                display_budget_mb = self._display_memory_budget_mb()
                display_batch_count = self._display_sampling_batch_count()
                if not self._show_preflight(jobs, sampling, transition_job):
                    return
            except ValueError as exc:
                QtWidgets.QMessageBox.warning(self, "参数错误", str(exc))
                return
            backend_payload = self._backend_generation_payload(
                document,
                jobs,
                sampling,
                transition_job,
                display_voxel_size,
                display_budget_mb,
                display_batch_count,
            )
            self.generate_button.setEnabled(False)
            self.progress.setValue(0)
            self.status.setText("正在计算，请稍候……")
            self.thread = QtCore.QThread(self)
            if backend_payload is not None:
                self._start_backend_generation(
                    document,
                    backend_payload,
                )
                return
            self.worker = GenerationWorker(
                mesh=self.sole_mesh.copy(),
                jobs=jobs,
                sampling=sampling,
                transition_job=transition_job,
                display_voxel_size_mm=display_voxel_size,
                display_memory_budget_mb=display_budget_mb,
                prepared_custom_cell=self.prepared_custom_cell,
                display_batch_count=display_batch_count,
                design_domain=(
                    document.domain
                    if isinstance(document.domain, AnalyticDesignDomain)
                    else None
                ),
                design_identifier=document.identifier,
                design_revision=document.revision,
            )
            self.worker.moveToThread(self.thread)
            self.thread.started.connect(self.worker.run)
            self.worker.progress.connect(lambda message, value: (self.status.setText(message), self.progress.setValue(int(value * 100))))
            self.worker.finished.connect(self._generation_finished)
            self.worker.failed.connect(self._generation_failed)
            self.worker.finished.connect(self.thread.quit)
            self.worker.failed.connect(self.thread.quit)
            self.thread.finished.connect(self._generation_cleanup)
            self.thread.start()

        def _backend_generation_payload(
            self,
            document,
            jobs: dict[str, LatticeParameters],
            sampling: SamplingParameters,
            transition_job,
            display_voxel_size_mm: float | None,
            display_memory_budget_mb: float,
            display_batch_count: int | None,
        ) -> dict[str, object] | None:
            """Build one or many serialized-design requests for the backend task API."""

            if self.backend_client is None or not isinstance(
                document.domain, (MeshDesignDomain, AnalyticDesignDomain)
            ):
                return None
            task_kind = None
            parameters_payload = None
            if len(jobs) == 1 and transition_job is None:
                parameters = next(iter(jobs.values()))
                parameters_payload = asdict(parameters)
                if isinstance(parameters, TPMSParameters):
                    task_kind = "tpms.generate"
                elif isinstance(parameters, CustomUnitCellParameters):
                    task_kind = "custom.generate"
                    parameters_payload["source_path"] = str(
                        self._workspace_asset_source_path(parameters.source_path)
                    )
            elif not jobs and transition_job is not None and len(transition_job) in (3, 4):
                first, second, transition = transition_job[:3]
                task_kind = "transition.generate"
                parameters_payload = {
                    "first_parameters": _backend_lattice_parameters_payload(
                        first,
                        self._workspace_asset_source_path,
                    ),
                    "second_parameters": _backend_lattice_parameters_payload(
                        second,
                        self._workspace_asset_source_path,
                    ),
                    "transition": asdict(transition),
                }
                if transition.driver_mode == "field":
                    primitive = self.field_primitives.get(transition.driver_identifier)
                    if primitive is None:
                        return None
                    parameters_payload["driver_primitive"] = asdict(primitive)

            input_root: Path | None = None
            domain_payload: dict[str, object] | None = None
            if isinstance(document.domain, MeshDesignDomain):
                input_root = Path(mkdtemp(prefix="lattice-studio-backend-"))
                mesh_path = input_root / "design-domain.stl"
                try:
                    # The child process reads a fresh snapshot. It must not depend
                    # on a possibly stale source asset after interactive repairs.
                    self.sole_mesh.export(mesh_path)
                except Exception:
                    rmtree(input_root, ignore_errors=True)
                    raise
                mesh_path_value: str | None = str(mesh_path)
                domain_payload = {
                    "kind": "mesh",
                    "name": document.domain.name,
                    "source_path": mesh_path_value,
                }
            else:
                mesh_path_value = None
                domain_payload = {
                    "kind": "analytic",
                    "name": document.domain.name,
                    "primitive": asdict(document.domain.primitive),
                }
            common_payload = {
                "sampling": asdict(sampling),
                "display_voxel_size_mm": display_voxel_size_mm,
                "display_memory_budget_mb": display_memory_budget_mb,
                "display_batch_count": display_batch_count,
            }
            if mesh_path_value is not None:
                common_payload["mesh_path"] = mesh_path_value
            if domain_payload is not None:
                common_payload["domain"] = domain_payload
            if task_kind is not None and parameters_payload is not None:
                self._backend_generation_input_root = input_root
                payload = dict(common_payload)
                if task_kind == "transition.generate":
                    payload.update(parameters_payload)
                else:
                    payload["parameters"] = parameters_payload
                payload["_task_kind"] = task_kind
                return payload

            requests = []
            for kind, parameters in jobs.items():
                if isinstance(parameters, TPMSParameters):
                    request_kind = "tpms.generate"
                elif isinstance(parameters, CustomUnitCellParameters):
                    request_kind = "custom.generate"
                else:
                    if input_root is not None:
                        rmtree(input_root, ignore_errors=True)
                    return None
                request_payload = dict(common_payload)
                request_payload["parameters"] = _backend_lattice_parameters_payload(
                    parameters,
                    self._workspace_asset_source_path,
                )
                requests.append(
                    {
                        "request_id": str(kind),
                        "kind": request_kind,
                        "payload": request_payload,
                    }
                )
            if transition_job is not None and len(transition_job) in (3, 4):
                first, second, transition = transition_job[:3]
                transition_payload = {
                    **common_payload,
                    "first_parameters": _backend_lattice_parameters_payload(
                        first, self._workspace_asset_source_path
                    ),
                    "second_parameters": _backend_lattice_parameters_payload(
                        second, self._workspace_asset_source_path
                    ),
                    "transition": asdict(transition),
                }
                if transition.driver_mode == "field":
                    primitive = self.field_primitives.get(transition.driver_identifier)
                    if primitive is None:
                        if input_root is not None:
                            rmtree(input_root, ignore_errors=True)
                        return None
                    transition_payload["driver_primitive"] = asdict(primitive)
                requests.append(
                    {
                        "request_id": "Transition",
                        "kind": "transition.generate",
                        "payload": transition_payload,
                    }
                )
            if len(requests) < 2:
                if input_root is not None:
                    rmtree(input_root, ignore_errors=True)
                return None
            self._backend_generation_input_root = input_root
            return {"_task_kind": "generation.batch", "requests": requests}

        def _start_backend_generation(
            self,
            document,
            payload: dict[str, object],
        ) -> None:
            """Run one supported generation request through the loopback backend."""

            client = self.backend_client
            if client is None:
                raise RuntimeError("local backend is unavailable")
            task_kind = str(payload.pop("_task_kind", "tpms.generate"))

            def result_reader(snapshot: object):
                if task_kind == "generation.batch":
                    artifact_ids = batch_display_field_artifact_ids(snapshot)
                    return read_generation_batch(
                        snapshot,
                        {
                            request_id: client.download_artifact(artifact_id)
                            for request_id, artifact_id in artifact_ids.items()
                        },
                    )
                artifact_id = display_field_artifact_id(snapshot)
                return read_generation_result(
                    snapshot,
                    client.download_artifact(artifact_id),
                )

            self.worker = BackendTaskWorker(
                client,
                task_kind,
                payload,
                result_reader,
            )
            self.worker.moveToThread(self.thread)
            self.thread.started.connect(self.worker.run)
            self.worker.progress.connect(
                lambda message, value: (
                    self.status.setText(message),
                    self.progress.setValue(int(value * 100)),
                )
            )
            if task_kind == "generation.batch":
                self.worker.finished.connect(
                    lambda result, identifier=document.identifier, revision=document.revision: self._backend_generation_batch_finished(
                        identifier, revision, result
                    )
                )
            else:
                self.worker.finished.connect(
                    lambda result, identifier=document.identifier, revision=document.revision: self._backend_generation_finished(
                        identifier,
                        revision,
                        result,
                    )
                )
            self.worker.failed.connect(
                lambda message, identifier=document.identifier, revision=document.revision: self._generation_failed(
                    identifier,
                    revision,
                    message,
                )
            )
            self.worker.finished.connect(self.thread.quit)
            self.worker.failed.connect(self.thread.quit)
            self.thread.finished.connect(self._generation_cleanup)
            self.thread.start()

        def _backend_generation_batch_finished(
            self,
            design_identifier: str,
            design_revision: int,
            generations: dict[str, BackendGenerationResult],
        ) -> None:
            """Store several opaque generation handles from one backend task."""

            if getattr(self, "_is_closing", False):
                return
            document = self.design_workspace.documents.get(design_identifier)
            if document is None or document.revision != design_revision:
                if self.design_workspace.active_design_id == design_identifier:
                    self.status.setText("设计已修改；已丢弃过期的生成结果。")
                return
            runtime = document.runtime
            runtime.implicit_generation_results = {}
            runtime.implicit_results = {}
            runtime.backend_generation_handles = dict(generations)
            runtime.implicit_fields = {
                kind: generation.display_field for kind, generation in generations.items()
            }
            runtime.raw_results = {}
            runtime.repaired_results = {}
            runtime.results = {}
            runtime.stl_reconstruction_results = {}
            runtime.renderer_cache.clear()
            if self.design_workspace.active_design_id != design_identifier:
                self.design_workspace.dirty = True
                return
            self._layer_contour_cache.clear()
            self._field_plane_cache.clear()
            self.implicit_generation_results = {}
            self.implicit_results = {}
            self.backend_generation_handles = dict(generations)
            self.implicit_fields = {
                kind: generation.display_field for kind, generation in generations.items()
            }
            self.stl_reconstruction_results = {}
            self.raw_results = {}
            self.repaired_results = {}
            self.results = {}
            self.show_domain.blockSignals(True)
            self.show_domain.setChecked(False)
            self.show_domain.blockSignals(False)
            self._sync_contour_result_options()
            self._sync_field_viewer_options()
            self._sync_stl_reconstruction_target_options()
            self._sync_section_target_options()
            self._sync_shell_lattice_options()
            self._set_result_controls_enabled(False)
            self.view_stl_mesh.setChecked(False)
            self.view_stl_mesh.setEnabled(False)
            self.show_stl_edges.setChecked(False)
            self.show_stl_edges.setEnabled(False)
            self.use_repaired_result.setChecked(False)
            self.use_repaired_result.setEnabled(False)
            self.progress.setValue(100)
            self.status.setText(f"完成后端隐式显示：{', '.join(generations)}")
            self._refresh_compute_backend_info()
            if self.field_viewer_enabled.isChecked():
                self.field_viewer_enabled.setChecked(False)
            self._refresh_scene(reset_view=True)
            self._store_active_design_state()

        def _backend_generation_finished(
            self,
            design_identifier: str,
            design_revision: int,
            generation: BackendGenerationResult,
        ) -> None:
            """Store only a display cache and opaque handle in the Qt process."""

            if getattr(self, "_is_closing", False):
                return
            document = self.design_workspace.documents.get(design_identifier)
            if document is None or document.revision != design_revision:
                if self.design_workspace.active_design_id == design_identifier:
                    self.status.setText("设计已修改；已丢弃过期的生成结果。")
                return
            runtime = document.runtime
            runtime.implicit_generation_results = {}
            runtime.implicit_results = {}
            runtime.backend_generation_handles = {generation.kind: generation}
            runtime.implicit_fields = {generation.kind: generation.display_field}
            runtime.raw_results = {}
            runtime.repaired_results = {}
            runtime.results = {}
            runtime.stl_reconstruction_results = {}
            runtime.renderer_cache.clear()
            if self.design_workspace.active_design_id != design_identifier:
                self.design_workspace.dirty = True
                return

            self._layer_contour_cache.clear()
            self._field_plane_cache.clear()
            self.implicit_generation_results = {}
            self.implicit_results = {}
            self.backend_generation_handles = {generation.kind: generation}
            self.implicit_fields = {generation.kind: generation.display_field}
            self.stl_reconstruction_results = {}
            self.raw_results = {}
            self.repaired_results = {}
            self.results = {}
            document.runtime.renderer_cache.clear()
            self.show_domain.blockSignals(True)
            self.show_domain.setChecked(False)
            self.show_domain.blockSignals(False)
            self._sync_contour_result_options()
            self._sync_field_viewer_options()
            self._sync_stl_reconstruction_target_options()
            self._sync_section_target_options()
            self._sync_shell_lattice_options()
            self._set_result_controls_enabled(False)
            self.view_stl_mesh.setChecked(False)
            self.view_stl_mesh.setEnabled(False)
            self.show_stl_edges.setChecked(False)
            self.show_stl_edges.setEnabled(False)
            self.use_repaired_result.setChecked(False)
            self.use_repaired_result.setEnabled(False)
            field = generation.display_field
            self._implicit_sampling_summary = describe_display_sampling(
                generation.kind,
                field,
                generation.recommendation,
                None,
            )
            self._stl_sampling_summary = "STL 重建：尚未生成"
            self._refresh_applied_sampling_info()
            self.progress.setValue(100)
            self.status.setText(
                f"完成后端隐式显示：{generation.kind} "
                f"{field.values.shape[0]}x{field.values.shape[1]}x{field.values.shape[2]}"
            )
            self._refresh_compute_backend_info()
            if self.field_viewer_enabled.isChecked():
                self.field_viewer_enabled.setChecked(False)
            self._refresh_scene(reset_view=True)
            self._store_active_design_state()

        def _store_background_generation_results(
            self,
            document,
            results,
        ) -> None:
            """Store a completed task without touching the Active Design UI."""

            runtime = document.runtime
            runtime.domain_implicit_field = results.get("domain", (None, None))[0]
            runtime.domain_implicit_body = document.domain.as_implicit_body()
            runtime.implicit_generation_results = {
                kind: item[0] for kind, item in results.items() if kind != "domain"
            }
            runtime.implicit_results = {
                kind: item[0].body for kind, item in results.items() if kind != "domain"
            }
            runtime.implicit_fields = {
                kind: item[0].display_field
                for kind, item in results.items()
                if kind != "domain"
            }
            runtime.raw_results = {}
            runtime.repaired_results = {}
            runtime.results = {}
            runtime.stl_reconstruction_results = {}
            runtime.renderer_cache.clear()

        def _generation_finished(
            self,
            design_identifier: str,
            design_revision: int,
            results,
        ):
            if getattr(self, "_is_closing", False):
                return
            document = self.design_workspace.documents.get(design_identifier)
            if document is None or document.revision != design_revision:
                if self.design_workspace.active_design_id == design_identifier:
                    self.status.setText("设计已修改；已丢弃过期的生成结果。")
                return
            if self.design_workspace.active_design_id != design_identifier:
                self._store_background_generation_results(document, results)
                self.design_workspace.dirty = True
                return
            self._layer_contour_cache.clear()
            self._field_plane_cache.clear()
            self.stl_reconstruction_results = {}
            self.domain_implicit_field = results.get("domain", (None, None))[0]
            document = self.design_workspace.active_document
            if document is not None and isinstance(
                document.domain, AnalyticDesignDomain
            ):
                self.domain_implicit_body = document.domain.as_implicit_body()
            self.implicit_generation_results = {
                kind: item[0]
                for kind, item in results.items()
                if kind != "domain"
            }
            self.implicit_results = {
                kind: item[0].body
                for kind, item in results.items()
                if kind != "domain"
            }
            self.implicit_fields = {
                kind: item[0].display_field
                for kind, item in results.items()
                if kind != "domain"
            }
            document.runtime.renderer_cache.clear()
            if self.implicit_fields:
                # The design-domain overlay is intentionally opaque. Keep it
                # available as context, but do not let it mask a newly
                # generated interior result on the first post-generation view.
                self.show_domain.blockSignals(True)
                self.show_domain.setChecked(False)
                self.show_domain.blockSignals(False)
            self._sync_contour_result_options()
            self._sync_field_viewer_options()
            self._sync_stl_reconstruction_target_options()
            self._sync_section_target_options()
            self._sync_shell_lattice_options()
            self.raw_results = {}
            self.repaired_results = {}
            self.results = {}
            self._set_result_controls_enabled(bool(self.results))
            self.view_stl_mesh.blockSignals(True)
            self.view_stl_mesh.setChecked(False)
            self.view_stl_mesh.setEnabled(False)
            self.view_stl_mesh.blockSignals(False)
            self.show_stl_edges.blockSignals(True)
            self.show_stl_edges.setChecked(False)
            self.show_stl_edges.setEnabled(False)
            self.show_stl_edges.blockSignals(False)
            self.use_repaired_result.blockSignals(True)
            self.use_repaired_result.setChecked(False)
            self.use_repaired_result.setEnabled(False)
            self.use_repaired_result.blockSignals(False)
            self.progress.setValue(100)
            custom_generation = self.implicit_generation_results.get("Custom")
            if (
                custom_generation is not None
                and isinstance(custom_generation.metadata, CustomUnitCellImportReport)
            ):
                report = custom_generation.metadata
                self.custom_widgets["source_status"].setText(
                    f"已生成：{report.prepared_faces:,} 个源面；"
                    f"水密={report.watertight}，连通量={report.connected_components}；"
                    f"特征厚度={custom_generation.minimum_feature_mm:.3f} mm"
                )
            cleanup_details = []
            for kind, generation in self.implicit_generation_results.items():
                cleanup = generation.display_fragment_cleanup
                if cleanup is not None and cleanup.removed_components:
                    cleanup_details.append(
                        f"{kind} 清理显示碎片 {cleanup.removed_components} 个"
                    )
            transition_details = ""
            transition_generation = self.implicit_generation_results.get("Transition")
            if (
                transition_generation is not None
                and isinstance(
                    transition_generation.metadata,
                    TransitionGenerationMetadata,
                )
            ):
                transition_metadata = transition_generation.metadata
                quality = transition_metadata.quality
                diagnostics = transition_metadata.diagnostics
                quality_label = (
                    "平面检查未适用"
                    if not quality.inspection_supported
                    else ("通过" if quality.passed else "需检查")
                )
                transition_details = (
                    f"；过渡质量={quality_label}"
                    f"（跨带连通={quality.cross_band_connected}，"
                    f"最小厚度={quality.minimum_feature_satisfied}，"
                    f"孤立块={quality.isolated_fragment_count}）；"
                    f"后端={diagnostics.execution_backend}；"
                    "配准位移="
                    + ", ".join(
                        f"{value:.3f}"
                        for value in diagnostics.registration_translation_mm
                    )
                    + " mm"
                )
                if not quality.inspection_supported and quality.status_message:
                    transition_details += f"；{quality.status_message}"
                if quality.issue_locations_mm:
                    first_location = quality.issue_locations_mm[0]
                    transition_details += (
                        "；首个风险位置="
                        + ", ".join(f"{value:.2f}" for value in first_location)
                        + " mm"
                    )
            details = "、".join(
                f"{kind} 隐式体 {field.values.shape[0]}×{field.values.shape[1]}×{field.values.shape[2]}，"
                f"实际采样 {_format_spacing_mm(field.spacing)}"
                for kind, field in self.implicit_fields.items()
            )
            if cleanup_details:
                details += "；" + "、".join(cleanup_details)
            details += transition_details
            requested_display_spacing = (
                self.worker.display_voxel_size_mm
                if self.worker is not None
                else None
            )
            display_batch_count = (
                self.worker.display_batch_count
                if self.worker is not None
                else None
            )
            display_reports = []
            domain_item = results.get("domain")
            if domain_item is not None:
                domain_field, domain_recommendation = domain_item
                display_reports.append(
                    describe_display_sampling(
                        "设计域",
                        domain_field,
                        domain_recommendation,
                        requested_display_spacing,
                    )
                )
            for kind, generation in self.implicit_generation_results.items():
                display_reports.append(
                    describe_display_sampling(
                        kind,
                        generation.display_field,
                        generation.recommendation,
                        requested_display_spacing,
                    )
                )
            self._implicit_sampling_summary = (
                "隐式显示"
                + (
                    f"（分批采样 {display_batch_count} 批）"
                    if display_batch_count is not None
                    else "（单批采样，内部自动微分片）"
                )
                + "：\n"
                + "\n".join(display_reports)
                if display_reports
                else "隐式显示：无结果"
            )
            self._stl_sampling_summary = "STL 重建：尚未生成"
            self._refresh_applied_sampling_info()
            self._update_stl_grid_estimate()
            self.status.setText("完成隐式显示：" + (details or "无结果"))
            self._refresh_compute_backend_info()
            # Field Viewer is an inspection mode with an intentionally empty
            # main scene. A completed generation must return to the result
            # scene instead of leaving a selected field object in front.
            if self.field_viewer_enabled.isChecked():
                self.field_viewer_enabled.setChecked(False)
            self._refresh_scene(reset_view=True)
            self._store_active_design_state()

        def _generation_failed(
            self,
            design_identifier: str,
            design_revision: int,
            message: str,
        ):
            if getattr(self, "_is_closing", False):
                return
            document = self.design_workspace.documents.get(design_identifier)
            if (
                document is None
                or document.revision != design_revision
                or self.design_workspace.active_design_id != design_identifier
            ):
                return
            self.status.setText("生成失败：" + message)
            self._refresh_compute_backend_info()
            QtWidgets.QMessageBox.critical(self, "生成失败", message)

        def _start_shell_generation(self, combine: bool) -> None:
            if self.shell_thread is not None:
                return
            if not self.sole_mesh.faces.size or not self.sole_mesh.is_watertight:
                QtWidgets.QMessageBox.warning(
                    self,
                    "无法抽壳",
                    "抽壳和壳体融合需要非空且水密的设计域 STL。请先修复设计域网格。",
                )
                return
            lattice_generation = None
            backend_lattice_generation = None
            if combine:
                lattice_kind = self.shell_lattice_combo.currentData()
                lattice_generation = self.implicit_generation_results.get(lattice_kind)
                backend_lattice_generation = self.backend_generation_handles.get(
                    lattice_kind
                )
                if (
                    lattice_generation is None
                    and backend_lattice_generation is None
                    or lattice_kind in (
                    SHELL_RESULT_KEY,
                    SHELL_UNION_RESULT_KEY,
                    )
                ):
                    QtWidgets.QMessageBox.warning(
                        self,
                        "无法融合",
                        "请先生成并选择一个晶格隐式体。",
                    )
                    return
            try:
                sampling = self._current_sampling()
                sampling.validate()
                display_batch_count = self._display_sampling_batch_count()
            except ValueError as exc:
                QtWidgets.QMessageBox.warning(self, "参数错误", str(exc))
                return
            if self.backend_client is not None and (
                not combine or backend_lattice_generation is not None
            ):
                self._start_backend_shell_generation(
                    combine,
                    sampling,
                    display_batch_count,
                    backend_lattice_generation,
                )
                return
            self.shell_generate_button.setEnabled(False)
            self.shell_union_button.setEnabled(False)
            self.progress.setValue(0)
            operation = "壳体与晶格融合" if combine else "设计域抽壳"
            self.status.setText(f"正在执行{operation}……")
            self.shell_status.setText(f"正在执行{operation}……")
            self.shell_thread = QtCore.QThread(self)
            self.shell_worker = ShellGenerationWorker(
                mesh=self.sole_mesh.copy(),
                sampling=sampling,
                thickness_mm=float(self.shell_thickness.value()),
                fusion_radius_mm=float(self.shell_fusion_radius.value()),
                lattice_generation=lattice_generation,
                domain_display_field=(
                    self._shell_fusion_domain_field
                    or self.domain_implicit_field
                ),
                display_voxel_size_mm=self._display_voxel_size(),
                display_memory_budget_mb=self._display_memory_budget_mb(),
                display_batch_count=display_batch_count,
                visible_field_count=max(len(self.implicit_generation_results) + 2, 1),
            )
            self.shell_worker.moveToThread(self.shell_thread)
            self.shell_thread.started.connect(self.shell_worker.run)
            self.shell_worker.progress.connect(
                lambda message, value: (
                    self.status.setText(message),
                    self.shell_status.setText(message),
                    self.progress.setValue(int(value * 100)),
                )
            )
            self.shell_worker.finished.connect(self._shell_generation_finished)
            self.shell_worker.failed.connect(self._shell_generation_failed)
            self.shell_worker.finished.connect(self.shell_thread.quit)
            self.shell_worker.failed.connect(self.shell_thread.quit)
            self.shell_thread.finished.connect(self._shell_generation_cleanup)
            self.shell_thread.start()

        def _start_backend_shell_generation(
            self,
            combine: bool,
            sampling: SamplingParameters,
            display_batch_count: int | None,
            lattice_generation: BackendGenerationResult | None,
        ) -> None:
            client = self.backend_client
            if client is None:
                return
            input_root = Path(mkdtemp(prefix="lattice-studio-backend-shell-"))
            mesh_path = input_root / "design-domain.stl"
            try:
                self.sole_mesh.export(mesh_path)
            except Exception:
                rmtree(input_root, ignore_errors=True)
                raise
            self._backend_shell_input_root = input_root
            kind = "shell.union" if combine else "shell.generate"
            payload: dict[str, object] = {
                "mesh_path": str(mesh_path),
                "thickness_mm": float(self.shell_thickness.value()),
                "sampling": asdict(sampling),
                "display_voxel_size_mm": self._display_voxel_size(),
                "display_memory_budget_mb": self._display_memory_budget_mb(),
                "display_batch_count": display_batch_count,
            }
            result_key = SHELL_UNION_RESULT_KEY if combine else SHELL_RESULT_KEY
            if combine:
                assert lattice_generation is not None
                payload.update(
                    {
                        "generation_id": lattice_generation.generation_id,
                        "fusion_radius_mm": float(self.shell_fusion_radius.value()),
                    }
                )

            def result_reader(snapshot: object) -> BackendGenerationResult:
                return read_generation_result(
                    snapshot,
                    client.download_artifact(display_field_artifact_id(snapshot)),
                )

            self.shell_generate_button.setEnabled(False)
            self.shell_union_button.setEnabled(False)
            self.progress.setValue(0)
            self.shell_thread = QtCore.QThread(self)
            self.shell_worker = BackendTaskWorker(client, kind, payload, result_reader)
            self.shell_worker.moveToThread(self.shell_thread)
            self.shell_thread.started.connect(self.shell_worker.run)
            self.shell_worker.progress.connect(
                lambda message, value: (
                    self.status.setText(message),
                    self.shell_status.setText(message),
                    self.progress.setValue(int(value * 100)),
                )
            )
            self.shell_worker.finished.connect(
                lambda result, target=result_key: self._backend_shell_generation_finished(
                    target, result
                )
            )
            self.shell_worker.failed.connect(self._shell_generation_failed)
            self.shell_worker.finished.connect(self.shell_thread.quit)
            self.shell_worker.failed.connect(self.shell_thread.quit)
            self.shell_thread.finished.connect(self._shell_generation_cleanup)
            self.shell_thread.start()

        def _backend_shell_generation_finished(
            self,
            kind: str,
            generation: BackendGenerationResult,
        ) -> None:
            if getattr(self, "_is_closing", False):
                return
            self._layer_contour_cache.clear()
            self._field_plane_cache.clear()
            targets = (
                (SHELL_RESULT_KEY, SHELL_UNION_RESULT_KEY)
                if kind == SHELL_RESULT_KEY
                else (SHELL_UNION_RESULT_KEY,)
            )
            for target in targets:
                self.implicit_generation_results.pop(target, None)
                self.implicit_results.pop(target, None)
                self.backend_generation_handles.pop(target, None)
                self.implicit_fields.pop(target, None)
            self.backend_generation_handles[kind] = generation
            self.implicit_fields[kind] = generation.display_field
            self.stl_reconstruction_results = {}
            self.raw_results = {}
            self.repaired_results = {}
            self.results = {}
            self._sync_contour_result_options()
            self._sync_field_viewer_options()
            self._sync_stl_reconstruction_target_options()
            self._sync_section_target_options()
            self._sync_shell_lattice_options()
            self._set_result_controls_enabled(False)
            self.progress.setValue(100)
            self.status.setText(f"backend {kind} generation complete")
            self.shell_status.setText(self.status.text())
            self._refresh_scene(reset_view=True)
            self._store_active_design_state()

        def _shell_generation_finished(
            self,
            kind: str,
            generation: ImplicitGenerationResult,
        ) -> None:
            if getattr(self, "_is_closing", False):
                return
            self._layer_contour_cache.clear()
            self._field_plane_cache.clear()
            if kind == SHELL_RESULT_KEY:
                # A new shell thickness invalidates any previously fused body.
                for target in (SHELL_RESULT_KEY, SHELL_UNION_RESULT_KEY):
                    self.implicit_generation_results.pop(target, None)
                    self.implicit_results.pop(target, None)
                    self.implicit_fields.pop(target, None)
            else:
                self.implicit_generation_results.pop(SHELL_UNION_RESULT_KEY, None)
                self.implicit_results.pop(SHELL_UNION_RESULT_KEY, None)
                self.implicit_fields.pop(SHELL_UNION_RESULT_KEY, None)
            self.implicit_generation_results[kind] = generation
            self.implicit_results[kind] = generation.body
            self.implicit_fields[kind] = generation.display_field
            if kind == SHELL_UNION_RESULT_KEY and isinstance(generation.metadata, dict):
                domain_field = generation.metadata.get("domain_display_field")
                self._shell_fusion_domain_field = (
                    domain_field
                    if isinstance(domain_field, SampledImplicitField)
                    else None
                )
            self.stl_reconstruction_results = {}
            self.raw_results = {}
            self.repaired_results = {}
            self.results = {}
            self.use_repaired_result.blockSignals(True)
            self.use_repaired_result.setChecked(False)
            self.use_repaired_result.setEnabled(False)
            self.use_repaired_result.blockSignals(False)
            self.view_stl_mesh.blockSignals(True)
            self.view_stl_mesh.setChecked(False)
            self.view_stl_mesh.setEnabled(False)
            self.view_stl_mesh.blockSignals(False)
            self.show_stl_edges.blockSignals(True)
            self.show_stl_edges.setChecked(False)
            self.show_stl_edges.setEnabled(False)
            self.show_stl_edges.blockSignals(False)
            self._set_result_controls_enabled(False)
            self._sync_contour_result_options()
            self._sync_field_viewer_options()
            self._sync_stl_reconstruction_target_options()
            self._sync_section_target_options()
            self._sync_shell_lattice_options()
            self._update_stl_grid_estimate()
            shell_display_report = describe_display_sampling(
                "抽壳" if kind == SHELL_RESULT_KEY else "壳体融合",
                generation.display_field,
                generation.recommendation,
                self._display_voxel_size(),
            )
            self._implicit_sampling_summary = (
                self._implicit_sampling_summary + "\n" + shell_display_report
            )
            self._stl_sampling_summary = "STL 重建：尚未生成"
            self._refresh_applied_sampling_info()
            self.progress.setValue(100)
            if kind == SHELL_RESULT_KEY:
                message = f"抽壳完成：壳厚 {self.shell_thickness.value():.3f} mm"
            else:
                source = self.shell_lattice_combo.currentText()
                message = (
                    f"壳体融合完成：{source}；壳厚 {self.shell_thickness.value():.3f} mm；"
                    f"融合系数 {self.shell_fusion_radius.value():.3f} mm"
                )
                backend = generation.metadata.get("display_fusion_backend")
                if backend == "cached-fields-cpu":
                    message += "；显示场：复用设计域与晶格缓存"
                elif backend == "cached-lattice-field-cpu":
                    message += "；显示场：复用晶格缓存"
            if (
                kind == SHELL_UNION_RESULT_KEY
                and isinstance(generation.metadata, dict)
                and generation.metadata.get("display_fusion_backend")
                == "coordinated-resample"
            ):
                message += "；显示场：每个微分片只计算一次设计域 SDF"
            self.status.setText(message)
            self.shell_status.setText(message)
            self._refresh_scene(reset_view=True)

        def _shell_generation_failed(self, message: str) -> None:
            if getattr(self, "_is_closing", False):
                return
            self.status.setText("抽壳/融合失败：" + message)
            self.shell_status.setText("抽壳/融合失败：" + message)
            QtWidgets.QMessageBox.critical(self, "抽壳或融合失败", message)

        def _shell_generation_cleanup(self) -> None:
            input_root = self._backend_shell_input_root
            self._backend_shell_input_root = None
            if input_root is not None:
                rmtree(input_root, ignore_errors=True)
            self.shell_worker = None
            self.shell_thread = None
            self._sync_shell_lattice_options()

        def _refresh_compute_backend_info(self):
            tpms_status = get_default_tpms_backend().status
            geometry_status = get_default_geometry_backend().status
            pipeline_status = get_default_lattice_pipeline().status

            def format_status(name, status):
                value = f"{name}：{status.active_backend}"
                if status.using_gpu:
                    value += f" · {status.device_name}"
                if status.fallback_reason:
                    value += f"（GPU 回退：{status.fallback_reason}）"
                return value

            self.compute_backend_info.setText(
                format_status("TPMS 场", tpms_status)
                + "；"
                + format_status("设计域 SDF/交集/Marching Cubes", geometry_status)
                + "；"
                + format_status("TPMS 裁剪常驻流水线", pipeline_status)
            )

        def _refresh_render_backend_info(self):
            if not hasattr(self.viewer, "get_render_backend_status"):
                self.render_backend_info.setText("当前查看器不支持 GPU 后端探测")
                return
            status = self.viewer.get_render_backend_status()
            text = f"{status.active_backend} · {status.device_name}"
            if status.fallback_reason:
                text += f"（{status.fallback_reason}）"
            self.render_backend_info.setText(text)

        def _generation_cleanup(self):
            input_root = self._backend_generation_input_root
            self._backend_generation_input_root = None
            if input_root is not None:
                rmtree(input_root, ignore_errors=True)
            self.worker = None
            self.thread = None
            self.generate_button.setEnabled(
                self.design_workspace.active_document is not None
            )

        def _display_refinement_cache(self) -> dict[object, object]:
            document = self.design_workspace.active_document
            if document is None:
                return {}
            cache = document.runtime.renderer_cache.setdefault(
                "quality_implicit_fields",
                {},
            )
            if not isinstance(cache, dict):
                cache = {}
                document.runtime.renderer_cache["quality_implicit_fields"] = cache
            return cache

        def _display_refinement_cache_key(
            self,
            kind: str,
            source: SampledImplicitField,
            visible_field_count: int,
            quality: str,
        ) -> tuple[object, ...]:
            field_budget = self._display_refinement_field_budget_mb(
                source,
                visible_field_count,
            )
            return (
                quality,
                kind,
                id(source.values),
                source.values.shape,
                tuple(float(value) for value in source.spacing),
                round(field_budget, 6),
            )

        def _display_refinement_field_budget_mb(
            self,
            source: SampledImplicitField,
            visible_field_count: int,
        ) -> float:
            """Reserve the display budget for resident CPU and GPU copies."""

            layer_working_set_mb = self._display_memory_budget_mb() / (
                2.0 * max(int(visible_field_count), 1)
            )
            source_mb = source.estimated_bytes / (1024.0 * 1024.0)
            return max(layer_working_set_mb - source_mb, 0.001)

        def _cancel_display_refinement_for_geometry_edit(self) -> None:
            """Stop quality sampling that belongs to the previous geometry revision."""

            thread = self.display_refinement_thread
            if thread is not None and thread.isRunning():
                thread.requestInterruption()
                self._display_refinement_refresh_pending = True

        def _display_refinement_progress(
            self,
            design_identifier: str,
            design_revision: int,
            message: str,
            value: float,
        ) -> None:
            document = self.design_workspace.documents.get(design_identifier)
            if (
                document is None
                or document.revision != int(design_revision)
                or self.design_workspace.active_design_id != design_identifier
            ):
                return
            self.status.setText(message)
            self.progress.setValue(int(value * 100))

        def _display_field_for_quality(
            self,
            kind: str,
            source: SampledImplicitField,
            visible_field_count: int,
        ) -> SampledImplicitField:
            quality = self.render_quality_combo.currentData()
            if quality not in DISPLAY_FIELD_REFINEMENT_FACTORS:
                return source
            cache_key = self._display_refinement_cache_key(
                kind,
                source,
                visible_field_count,
                quality,
            )
            cached = self._display_refinement_cache().get(cache_key)
            if not isinstance(cached, tuple) or len(cached) != 2:
                return source
            field, _report = cached
            if not isinstance(field, SampledImplicitField):
                return source
            return replace(
                field,
                color=source.color,
                metallic=source.metallic,
                roughness=source.roughness,
            )

        def _prune_display_refinement_cache(
            self,
            cache: dict[object, object],
            protected_keys: set[object],
        ) -> None:
            """Bound CPU quality caches while retaining the just-built fields."""

            budget_bytes = int(
                self._display_memory_budget_mb() * 1024 * 1024 * 0.5
            )

            def item_bytes(item: object) -> int:
                if not isinstance(item, tuple) or not item:
                    return 0
                field = item[0]
                return (
                    field.estimated_bytes
                    if isinstance(field, SampledImplicitField)
                    else 0
                )

            resident_bytes = sum(item_bytes(item) for item in cache.values())
            for cache_key in list(cache):
                if resident_bytes <= budget_bytes:
                    break
                if cache_key in protected_keys:
                    continue
                resident_bytes -= item_bytes(cache.pop(cache_key))

        def _start_display_refinement(
            self,
            sources: list[
                tuple[str, ImplicitBody | None, SampledImplicitField]
            ],
        ) -> None:
            quality = self.render_quality_combo.currentData()
            if (
                quality not in DISPLAY_FIELD_REFINEMENT_FACTORS
                or not sources
                or not hasattr(self.viewer, "get_render_backend_status")
            ):
                return
            if self.display_refinement_thread is not None:
                self._display_refinement_refresh_pending = True
                return
            document = self.design_workspace.active_document
            if document is None:
                return
            visible_count = len(sources)
            cache = self._display_refinement_cache()
            targets = {}
            backend_targets = {}
            for kind, body, source in sources:
                cache_key = self._display_refinement_cache_key(
                    kind,
                    source,
                    visible_count,
                    quality,
                )
                if cache_key in cache:
                    continue
                if body is not None:
                    targets[cache_key] = (
                        body,
                        source,
                        self._display_refinement_field_budget_mb(
                            source,
                            visible_count,
                        ),
                    )
                else:
                    generation = self.backend_generation_handles.get(kind)
                    if generation is not None:
                        backend_targets[cache_key] = (
                            generation,
                            self._display_refinement_field_budget_mb(
                                source,
                                visible_count,
                            ),
                        )
            if not targets:
                if backend_targets:
                    cache_key, (generation, field_budget_mb) = next(
                        iter(backend_targets.items())
                    )
                    self._start_backend_display_refinement(
                        document,
                        cache_key,
                        generation,
                        field_budget_mb,
                        quality,
                    )
                return

            self._display_refinement_refresh_pending = False
            self.display_refinement_thread = QtCore.QThread(self)
            self.display_refinement_worker = DisplayRefinementWorker(
                targets,
                quality=quality,
                display_batch_count=self._display_sampling_batch_count(),
                design_identifier=document.identifier,
                design_revision=document.revision,
            )
            self.display_refinement_worker.moveToThread(
                self.display_refinement_thread
            )
            self.display_refinement_thread.started.connect(
                self.display_refinement_worker.run
            )
            self.display_refinement_worker.progress.connect(
                lambda message, value, identifier=document.identifier,
                revision=document.revision: self._display_refinement_progress(
                    identifier,
                    revision,
                    message,
                    value,
                )
            )
            self.display_refinement_worker.finished.connect(
                self._display_refinement_finished
            )
            self.display_refinement_worker.failed.connect(
                self._display_refinement_failed
            )
            self.display_refinement_worker.finished.connect(
                self.display_refinement_thread.quit
            )
            self.display_refinement_worker.failed.connect(
                self.display_refinement_thread.quit
            )
            self.display_refinement_thread.finished.connect(
                self._display_refinement_cleanup
            )
            self.status.setText(
                f"正在从权威隐式体构建{DISPLAY_QUALITY_LABELS[quality]}显示场……"
            )
            self.progress.setValue(0)
            self.display_refinement_thread.start()

        def _start_backend_display_refinement(
            self,
            document,
            cache_key: object,
            generation: BackendGenerationResult,
            field_budget_mb: float,
            quality: str,
        ) -> None:
            client = self.backend_client
            if client is None:
                return
            payload = {
                "generation_id": generation.generation_id,
                "quality": quality,
                "display_memory_budget_mb": field_budget_mb,
                "display_batch_count": self._display_sampling_batch_count(),
            }

            def result_reader(snapshot: object):
                if snapshot.result is None:
                    raise ValueError("display refinement task returned no result")
                report = DisplayRefinementReport(**snapshot.result["report"])
                return (
                    read_sampled_field(
                        client.download_artifact(display_field_artifact_id(snapshot))
                    ),
                    report,
                )

            self._display_refinement_refresh_pending = False
            self.display_refinement_thread = QtCore.QThread(self)
            self.display_refinement_worker = BackendTaskWorker(
                client,
                "display.refine",
                payload,
                result_reader,
            )
            self.display_refinement_worker.moveToThread(
                self.display_refinement_thread
            )
            self.display_refinement_thread.started.connect(
                self.display_refinement_worker.run
            )
            self.display_refinement_worker.progress.connect(
                lambda message, value, identifier=document.identifier,
                revision=document.revision: self._display_refinement_progress(
                    identifier,
                    revision,
                    message,
                    value,
                )
            )
            self.display_refinement_worker.finished.connect(
                lambda outcome, identifier=document.identifier,
                revision=document.revision, requested=quality, key=cache_key: self._display_refinement_finished(
                    identifier,
                    revision,
                    requested,
                    {key: outcome},
                )
            )
            self.display_refinement_worker.failed.connect(
                lambda message, identifier=document.identifier,
                revision=document.revision, requested=quality: self._display_refinement_failed(
                    identifier,
                    revision,
                    requested,
                    message,
                )
            )
            self.display_refinement_worker.finished.connect(
                self.display_refinement_thread.quit
            )
            self.display_refinement_worker.failed.connect(
                self.display_refinement_thread.quit
            )
            self.display_refinement_thread.finished.connect(
                self._display_refinement_cleanup
            )
            self.progress.setValue(0)
            self.display_refinement_thread.start()

        def _display_refinement_finished(
            self,
            design_identifier: str,
            design_revision: int,
            quality: str,
            results: dict[object, object],
        ) -> None:
            if getattr(self, "_is_closing", False):
                return
            document = self.design_workspace.documents.get(design_identifier)
            if document is None:
                return
            if document.revision != int(design_revision):
                if self.design_workspace.active_design_id == design_identifier:
                    self._display_refinement_refresh_pending = True
                return
            cache = document.runtime.renderer_cache.setdefault(
                "quality_implicit_fields",
                {},
            )
            if isinstance(cache, dict):
                cache.update(results)
                self._prune_display_refinement_cache(
                    cache,
                    set(results),
                )
            if self.design_workspace.active_design_id != design_identifier:
                return
            if self.render_quality_combo.currentData() != quality:
                return
            reports = [item[1] for item in results.values()]
            refined = [report for report in reports if report.refined]
            limited = [report for report in reports if report.budget_limited]
            self.progress.setValue(100)
            quality_label = DISPLAY_QUALITY_LABELS[quality]
            if refined:
                spacing = min(report.applied_spacing_mm for report in refined)
                message = (
                    f"{quality_label}显示场已就绪："
                    f"最小实际间距 {spacing:.4f} mm"
                )
                if limited:
                    message += "；部分对象受显示内存预算限制"
            else:
                message = (
                    f"{quality_label}模式受显示内存预算限制，继续使用原显示场"
                )
            self.status.setText(message)
            self._refresh_scene(reset_view=False)

        def _display_refinement_failed(
            self,
            design_identifier: str,
            design_revision: int,
            quality: str,
            message: str,
        ) -> None:
            if getattr(self, "_is_closing", False):
                return
            document = self.design_workspace.documents.get(design_identifier)
            if (
                document is not None
                and document.revision != int(design_revision)
                and self.design_workspace.active_design_id == design_identifier
            ):
                self._display_refinement_refresh_pending = True
                return
            if (
                document is not None
                and document.revision == int(design_revision)
                and self.design_workspace.active_design_id == design_identifier
                and self.render_quality_combo.currentData() == quality
            ):
                self.status.setText(
                    f"{DISPLAY_QUALITY_LABELS[quality]}显示场细化失败，"
                    "已保留原显示场：" + message
                )

        def _display_refinement_cleanup(self) -> None:
            self.display_refinement_worker = None
            self.display_refinement_thread = None
            if getattr(self, "_is_closing", False):
                self._display_refinement_refresh_pending = False
                return
            if self._display_refinement_refresh_pending:
                self._display_refinement_refresh_pending = False
                QtCore.QTimer.singleShot(
                    0,
                    lambda: self._refresh_scene(reset_view=False),
                )

        def _make_mesh_layer(
            self,
            kind: str,
            mesh: trimesh.Trimesh,
        ) -> MeshData:
            return MeshData(
                mesh.vertices,
                mesh.faces,
                color=self.material_colors[kind] + (1.0,),
                cache_key=(kind, id(mesh)),
                show_edges=bool(self.show_stl_edges.isChecked()),
                **PBR_MATERIAL_PARAMETERS[kind],
            )

        def _scene_uses_stl_mesh(self, section_target: str | None) -> bool:
            """Resolve the mesh source separately for normal and section views."""

            if section_target is None:
                return self.view_stl_mesh.isChecked()
            return self.section_render_source_combo.currentData() == "mesh"

        def _refresh_scene(self, *_args, reset_view=False):
            if getattr(self.viewer, "_cell_map_preview", None) is not None:
                self._disable_cell_map_interaction()
            # Analytic primitives are placement helpers.  Their SDFs remain
            # available in every mode, while their preview meshes stay out of
            # contour, section, and field-viewer inspection views.
            refresh_primitive_preview = getattr(
                self,
                "_refresh_primitive_preview",
                None,
            )
            if refresh_primitive_preview is not None:
                refresh_primitive_preview(
                    sync_gizmo=True,
                    defer_scene_update=True,
                )
            if getattr(self, "contour_enabled", None) is not None and self.contour_enabled.isChecked():
                self._refresh_layer_contours()
                return

            # Field Viewer inspection may intentionally show only its sampled
            # plane.  Clearing visible scene layers must not clear the plane,
            # its probe state, or the authoritative field selected by the UI.
            field_viewer_enabled = getattr(self, "field_viewer_enabled", None)
            field_viewer_show_object = getattr(
                self,
                "field_viewer_show_object",
                None,
            )
            field_viewer_only = bool(
                field_viewer_enabled is not None
                and field_viewer_show_object is not None
                and field_viewer_enabled.isChecked()
                and (
                    not field_viewer_show_object.isChecked()
                    or self._field_viewer_primitive_target_identifier() is not None
                )
            )
            if field_viewer_only:
                if hasattr(self.viewer, "set_scene_geometry"):
                    self.viewer.set_scene_geometry([], [], reset_view=reset_view)
                elif hasattr(self.viewer, "set_implicit_fields"):
                    self.viewer.set_implicit_fields([], reset_view=reset_view)
                else:
                    self.viewer.set_meshes([], reset_view=reset_view)
                self._refresh_render_backend_info()
                self._update_transition_plane()
                return

            section_target = (
                self.section_target_combo.currentData()
                if self.section_enabled.isChecked()
                else None
            )
            use_stl_mesh = self._scene_uses_stl_mesh(section_target)
            show_domain = section_target == "domain" or (
                section_target is None and self.show_domain.isChecked()
            )
            domain_uses_stl_mesh = bool(
                section_target == "domain"
                and use_stl_mesh
                or section_target is None
                and self.domain_display_source_combo.currentData() == "mesh"
            )
            if hasattr(self.viewer, "set_implicit_cache_budget_mb"):
                self.viewer.set_implicit_cache_budget_mb(
                    self._display_memory_budget_mb()
                )
            mesh_layers = []
            implicit_sources: list[
                tuple[str, ImplicitBody | None, SampledImplicitField]
            ] = []
            if show_domain:
                if domain_uses_stl_mesh and self.sole_mesh.faces.size:
                    document = self.design_workspace.active_document
                    domain_mesh = (
                        self.results["domain"]
                        if document is not None
                        and isinstance(document.domain, AnalyticDesignDomain)
                        and "domain" in self.results
                        else self.sole_mesh
                    )
                    mesh_layers.append(self._make_mesh_layer("domain", domain_mesh))
                elif self.domain_implicit_field is not None:
                    implicit_sources.append(
                        (
                            "domain",
                            self.domain_implicit_body,
                        replace(
                            self.domain_implicit_field,
                            color=self.material_colors["domain"] + (1.0,),
                            **PBR_MATERIAL_PARAMETERS["domain"],
                            ),
                        )
                    )
            for kind, checkbox, opacity in (
                ("G", self.show_g, 1.0),
                ("D", self.show_d, 1.0),
                ("IWP", self.show_iwp, 1.0),
                ("Primitive", self.show_primitive, 1.0),
                ("Neovius", self.show_neovius, 1.0),
                ("Custom", self.show_custom, 1.0),
                ("Transition", self.show_transition, 1.0),
                (SHELL_RESULT_KEY, self.show_shell, 1.0),
                (SHELL_UNION_RESULT_KEY, self.show_shell_union, 1.0),
            ):
                show_kind = section_target == kind or (
                    section_target is None and checkbox.isChecked()
                )
                if not show_kind:
                    continue
                if use_stl_mesh and kind in self.results:
                    mesh_layers.append(self._make_mesh_layer(kind, self.results[kind]))
                elif not use_stl_mesh and kind in self.implicit_fields:
                    implicit_sources.append(
                        (
                            kind,
                            self.implicit_results.get(kind),
                        replace(
                            self.implicit_fields[kind],
                            color=self.material_colors[kind] + (opacity,),
                            **PBR_MATERIAL_PARAMETERS[kind],
                            ),
                        )
                    )
            visible_implicit_count = len(implicit_sources)
            implicit_layers = [
                self._display_field_for_quality(
                    kind,
                    source,
                    visible_implicit_count,
                )
                for kind, _body, source in implicit_sources
            ]
            if hasattr(self.viewer, "set_scene_geometry"):
                self.viewer.set_scene_geometry(
                    mesh_layers,
                    implicit_layers,
                    reset_view=reset_view,
                )
            elif implicit_layers and hasattr(self.viewer, "set_implicit_fields"):
                self.viewer.set_implicit_fields(implicit_layers, reset_view=reset_view)
            else:
                self.viewer.set_meshes(mesh_layers, reset_view=reset_view)
            self._refresh_render_backend_info()
            self._update_transition_plane()
            self._start_display_refinement(implicit_sources)

        def _update_recommendation_from_ui(self, *_args):
            self._update_recommendation()

        def _retry_pending_close(self) -> None:
            if not self._close_pending:
                self._close_retry_timer.stop()
                return
            if any(
                thread is not None and thread.isRunning()
                for attribute, _message in BACKGROUND_THREAD_SPECS
                for thread in (getattr(self, attribute, None),)
            ):
                return
            self._close_retry_timer.stop()
            self._close_pending = False
            self.close()

        def _close_viewport(self) -> None:
            """Release VTK/OpenGL resources before the Qt window is destroyed."""

            if getattr(self, "_viewport_closed", False):
                return
            viewer = getattr(self, "viewer", None)
            if viewer is not None:
                viewer.setUpdatesEnabled(False)
                for timer in viewer.findChildren(QtCore.QTimer):
                    timer.stop()
                viewer.hide()
                for child in viewer.findChildren(QtCore.QObject):
                    QtCore.QCoreApplication.removePostedEvents(child)
                QtCore.QCoreApplication.removePostedEvents(viewer)
                viewer.close()
            self._viewport_closed = True

        def closeEvent(self, event) -> None:
            self._is_closing = True
            self._display_refinement_refresh_pending = False
            self._analytic_domain_preview_timer.stop()
            self._pending_analytic_domain_preview = None
            self._field_plane_refresh_timer.stop()
            stopped, message = request_background_thread_shutdown(self)
            if not stopped:
                self._close_pending = True
                self.status.setText(message or "正在安全结束后台计算，请稍候。")
                self._close_retry_timer.start()
                event.ignore()
                return
            stopped, message = request_backend_process_shutdown(self)
            if not stopped:
                self._close_pending = True
                self.status.setText(message or "local backend is still stopping")
                self._close_retry_timer.start()
                event.ignore()
                return
            self._close_pending = False
            self._close_retry_timer.stop()
            self._close_viewport()
            super().closeEvent(event)

    return TPMSWindow


def run_gui() -> int:
    from PyQt5 import QtWidgets

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window_class = _build_qt_app()
    window = window_class()
    window.show()
    return app.exec_()


def generate_example(project_root: Path) -> dict[str, Path]:
    """Generate G, D and G/D-transition example results for the example sole."""

    sole_path = _default_sole_path(project_root)
    mesh = trimesh.load(sole_path, force="mesh")
    output_dir = project_root / "build" / "examples"
    output_dir.mkdir(parents=True, exist_ok=True)
    sampling = SamplingParameters(target_voxels=2_500_000)
    g_parameters = TPMSParameters("G", 10.0, 0.9)
    d_parameters = TPMSParameters("D", 12.0, 1.0)
    outputs = {}
    voxel_size_mm = 0.5
    for params in (g_parameters, d_parameters):
        lattice, rec = generate_tpms_lattice(
            mesh,
            params,
            sampling,
            voxel_size_mm=voxel_size_mm,
            progress=lambda msg, value: print(f"[{value:5.1%}] {msg}"),
        )
        if not lattice.faces.size:
            raise RuntimeError(f"{params.kind} result is empty")
        path = output_dir / f"1_{params.kind}_tpms.stl"
        lattice.export(path)
        outputs[params.kind] = path
        print(f"{params.kind}: voxel={rec.voxel_size_mm:.3f} mm, grid={rec.grid_shape}, faces={len(lattice.faces):,}, output={path}")
    transition = TransitionParameters(
        plane_axis="Y",
        plane_position_mm=float(mesh.bounds.mean(axis=0)[1]),
        transition_width_mm=10.0,
    )
    lattice, rec = generate_tpms_transition(
        mesh,
        g_parameters,
        d_parameters,
        transition,
        sampling,
        voxel_size_mm=voxel_size_mm,
        progress=lambda msg, value: print(f"[{value:5.1%}] {msg}"),
    )
    if not lattice.faces.size:
        raise RuntimeError("Transition result is empty")
    path = output_dir / "1_GD_transition_tpms.stl"
    lattice.export(path)
    outputs["Transition"] = path
    print(f"Transition: voxel={rec.voxel_size_mm:.3f} mm, grid={rec.grid_shape}, faces={len(lattice.faces):,}, output={path}")
    return outputs


def main() -> int:
    parser = argparse.ArgumentParser(description="G/D TPMS implicit filling workbench")
    parser.add_argument("--generate-example", action="store_true", help="generate the example STL files and exit")
    args = parser.parse_args()
    root = PROJECT_ROOT
    if args.generate_example:
        generate_example(root)
        return 0
    return run_gui()


if __name__ == "__main__":
    raise SystemExit(main())
