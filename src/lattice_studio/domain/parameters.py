"""User-facing lattice design parameters and validation rules."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypedDict

import numpy as np

from lattice_studio.domain.cell_map import CellMapFrame
from lattice_studio.domain.gradient import (
    GRADIENT_RESOLUTION_STRATEGIES,
    GradientAxis,
    GradientControls,
    GradientMode,
    GradientResolutionStrategy,
    WallThicknessMethod,
)
from lattice_studio.domain.transition import (
    TransitionSpec as ProviderTransitionSpec,
    TransitionWeightKind,
)

TPMSKind = Literal["G", "D", "IWP", "Primitive", "Neovius"]


LatticeKind = Literal["G", "D", "IWP", "Primitive", "Neovius", "Custom"]


TPMS_KINDS: tuple[TPMSKind, ...] = ("G", "D", "IWP", "Primitive", "Neovius")


CellSize = float | tuple[float, float, float]


CellMapMode = Literal["fit_bounds", "complete_cells"]


PlaneAxis = Literal["X", "Y", "Z"]


TransitionDriverMode = Literal["plane", "field"]


ProcessingMode = Literal[
    "single_pass",
    "batched_field",
    "chunked_marching_cubes",
]


ExportSpacingMode = Literal["recommended", "exact"]


class TPMSParameterState(TypedDict):
    """Persistent values for one TPMS family shown in the shared UI panel."""

    cell_size_mm: tuple[float, float, float]
    wall_thickness_mm: float
    level: float
    cell_map_mode: CellMapMode
    frame_origin_follow: bool
    frame_origin_mm: tuple[float, float, float]
    frame_u_axis: tuple[float, float, float]
    frame_v_axis: tuple[float, float, float]
    gradient_enabled: bool
    gradient_axis: GradientAxis
    gradient_mode: GradientMode
    gradient_power: float
    gradient_layers: int
    gradient_sigmoid_sharpness: float
    gradient_resolution_strategy: GradientResolutionStrategy
    gradient_def_per_half_cell: int
    use_thickness_gradient: bool
    thickness_soft_mm: float
    thickness_stiff_mm: float
    use_offset_gradient: bool
    offset_soft: float
    offset_stiff: float
    wall_thickness_method: WallThicknessMethod
    manual_cell_counts_enabled: bool
    manual_cell_counts: tuple[int, int, int]


FRAME_INPUT_DECIMALS = 6


FRAME_ORIGIN_DEFAULT_ATOL_MM = 0.5 * 10.0**-4


@dataclass(frozen=True)
class TPMSParameters:
    """User-facing parameters for one TPMS family."""

    kind: TPMSKind
    cell_size_mm: CellSize = 10.0
    wall_thickness_mm: float = 0.9
    level: float = 0.0
    cell_map_mode: CellMapMode = "fit_bounds"
    frame_origin_mm: tuple[float, float, float] | None = None
    frame_u_axis: tuple[float, float, float] = (1.0, 0.0, 0.0)
    frame_v_axis: tuple[float, float, float] = (0.0, 1.0, 0.0)
    gradient_enabled: bool = False
    gradient_axis: GradientAxis = "W"
    gradient_mode: GradientMode = "linear"
    gradient_power: float = 5.0
    gradient_layers: int = 5
    gradient_sigmoid_sharpness: float = 10.0
    gradient_resolution_strategy: GradientResolutionStrategy = "automatic"
    gradient_def_per_half_cell: int = 28
    use_thickness_gradient: bool = True
    thickness_soft_mm: float | None = None
    thickness_stiff_mm: float | None = None
    use_offset_gradient: bool = True
    offset_soft: float | None = None
    offset_stiff: float | None = None
    wall_thickness_method: WallThicknessMethod = "gradient_normalized"
    manual_cell_counts: tuple[int, int, int] | None = None

    @property
    def cell_size_xyz_mm(self) -> tuple[float, float, float]:
        values = np.asarray(self.cell_size_mm, dtype=np.float64)
        if values.ndim == 0:
            values = np.repeat(values, 3)
        if values.shape != (3,):
            raise ValueError("cell_size_mm must be a scalar or three values")
        return tuple(float(value) for value in values)

    @property
    def minimum_wall_thickness_mm(self) -> float:
        """Return the smallest wall thickness that can occur in this field."""

        values = [float(self.wall_thickness_mm)]
        if self.gradient_enabled and self.use_thickness_gradient:
            if self.thickness_soft_mm is not None:
                values.append(float(self.thickness_soft_mm))
            if self.thickness_stiff_mm is not None:
                values.append(float(self.thickness_stiff_mm))
        return min(values)

    def validate(self) -> None:
        if self.kind not in ("G", "D", "IWP", "Primitive", "Neovius"):
            raise ValueError("unsupported TPMS kind")
        cell_sizes = np.asarray(self.cell_size_xyz_mm, dtype=np.float64)
        if not np.isfinite(cell_sizes).all() or np.any((cell_sizes < 0.5) | (cell_sizes > 100.0)):
            raise ValueError("cell_size_mm values must be in [0.5, 100]")
        if not 0.05 <= self.wall_thickness_mm <= 20.0:
            raise ValueError("wall_thickness_mm must be in [0.05, 20]")
        if self.gradient_resolution_strategy not in GRADIENT_RESOLUTION_STRATEGIES:
            raise ValueError("unsupported gradient resolution strategy")
        if (
            int(self.gradient_def_per_half_cell) != self.gradient_def_per_half_cell
            or self.gradient_def_per_half_cell < 1
        ):
            raise ValueError("gradient_def_per_half_cell must be a positive integer")
        thickness_values = [self.wall_thickness_mm]
        if self.gradient_enabled and self.use_thickness_gradient:
            thickness_values.extend(
                value
                for value in (self.thickness_soft_mm, self.thickness_stiff_mm)
                if value is not None
            )
        if any(float(value) >= float(np.min(cell_sizes)) * 0.45 for value in thickness_values):
            raise ValueError("all wall-thickness values must be smaller than 45% of cell size")
        if not -2.0 <= self.level <= 2.0:
            raise ValueError("level must be in [-2, 2]")
        if self.cell_map_mode not in ("fit_bounds", "complete_cells"):
            raise ValueError("cell_map_mode must be 'fit_bounds' or 'complete_cells'")
        CellMapFrame.from_origin_axes(
            self.frame_origin_mm if self.frame_origin_mm is not None else (0.0, 0.0, 0.0),
            self.frame_u_axis,
            self.frame_v_axis,
        )
        controls = GradientControls(
            enabled=self.gradient_enabled,
            axis=self.gradient_axis,
            mode=self.gradient_mode,
            power=self.gradient_power,
            layers=self.gradient_layers,
            sigmoid_sharpness=self.gradient_sigmoid_sharpness,
            use_thickness_gradient=self.use_thickness_gradient,
            thickness_soft_mm=self.thickness_soft_mm,
            thickness_stiff_mm=self.thickness_stiff_mm,
            use_offset_gradient=self.use_offset_gradient,
            offset_soft=self.offset_soft,
            offset_stiff=self.offset_stiff,
            wall_thickness_method=self.wall_thickness_method,
        )
        controls.validate()
        if self.manual_cell_counts is not None:
            counts = np.asarray(self.manual_cell_counts)
            if counts.shape != (3,) or not np.isfinite(counts).all():
                raise ValueError("manual_cell_counts must contain three integers")
            if any(int(value) != value or int(value) < 1 for value in counts):
                raise ValueError("manual_cell_counts must contain positive integers")

    def cell_map_frame(self, default_origin) -> CellMapFrame:
        """Resolve this family's frame against the current design domain."""

        return CellMapFrame.from_origin_axes(
            self.frame_origin_mm if self.frame_origin_mm is not None else default_origin,
            self.frame_u_axis,
            self.frame_v_axis,
        )


@dataclass(frozen=True)
class CustomUnitCellParameters:
    """User-facing placement parameters for one STL-backed unit cell."""

    source_path: Path
    cell_size_mm: CellSize = (12.0, 12.0, 8.0)
    cell_map_mode: CellMapMode = "fit_bounds"
    frame_origin_mm: tuple[float, float, float] | None = None
    frame_u_axis: tuple[float, float, float] = (1.0, 0.0, 0.0)
    frame_v_axis: tuple[float, float, float] = (0.0, 1.0, 0.0)
    target_feature_mm: float | None = None
    bridge_directions: tuple[str, ...] = ()
    bridge_depth_mm: float | None = None
    kind: Literal["Custom"] = "Custom"

    @property
    def cell_size_xyz_mm(self) -> tuple[float, float, float]:
        values = np.asarray(self.cell_size_mm, dtype=np.float64)
        if values.ndim == 0:
            values = np.repeat(values, 3)
        if values.shape != (3,):
            raise ValueError("cell_size_mm must be a scalar or three values")
        return tuple(float(value) for value in values)

    def validate(self) -> None:
        source = Path(self.source_path)
        if not source.is_file() or source.suffix.lower() != ".stl":
            raise ValueError("custom unit-cell source must be an existing STL file")
        cell_sizes = np.asarray(self.cell_size_xyz_mm, dtype=np.float64)
        if not np.isfinite(cell_sizes).all() or np.any(
            (cell_sizes < 0.5) | (cell_sizes > 100.0)
        ):
            raise ValueError("cell_size_mm values must be in [0.5, 100]")
        if self.cell_map_mode not in ("fit_bounds", "complete_cells"):
            raise ValueError("cell_map_mode must be 'fit_bounds' or 'complete_cells'")
        if self.target_feature_mm is not None and (
            not np.isfinite(self.target_feature_mm)
            or self.target_feature_mm <= 0.0
        ):
            raise ValueError("target_feature_mm must be finite and positive or None")
        directions = tuple(str(direction).upper() for direction in self.bridge_directions)
        if any(direction not in {"U", "V", "W"} for direction in directions):
            raise ValueError("bridge_directions must contain only U, V, and W")
        if self.bridge_depth_mm is not None and (
            not np.isfinite(self.bridge_depth_mm) or self.bridge_depth_mm <= 0.0
        ):
            raise ValueError("bridge_depth_mm must be finite and positive or None")
        selected_indices = ["UVW".index(direction) for direction in directions]
        if self.bridge_depth_mm is not None and any(
            self.bridge_depth_mm >= 0.5 * cell_sizes[index]
            for index in selected_indices
        ):
            raise ValueError("bridge_depth_mm must be less than half the selected cell spacing")
        CellMapFrame.from_origin_axes(
            self.frame_origin_mm
            if self.frame_origin_mm is not None
            else (0.0, 0.0, 0.0),
            self.frame_u_axis,
            self.frame_v_axis,
        )

    def cell_map_frame(self, default_origin) -> CellMapFrame:
        return CellMapFrame.from_origin_axes(
            self.frame_origin_mm if self.frame_origin_mm is not None else default_origin,
            self.frame_u_axis,
            self.frame_v_axis,
        )

    def effective_feature_mm(self, native_feature_mm: float) -> float:
        """Resolve the optional user target against the scaled source body."""

        native = float(native_feature_mm)
        if not np.isfinite(native) or native <= 0.0:
            raise ValueError("native_feature_mm must be finite and positive")
        return (
            native
            if self.target_feature_mm is None
            else float(self.target_feature_mm)
        )


LatticeParameters = TPMSParameters | CustomUnitCellParameters


@dataclass(frozen=True)
class SamplingParameters:
    """Controls for voxel sampling and the selected memory strategy.

    ``single_pass`` evaluates the complete scalar field and extracts it with
    one Marching-Cubes call.  ``batched_field`` evaluates slabs separately but
    stores one complete scalar field before one Marching-Cubes call.
    ``chunked_marching_cubes`` keeps the existing low-peak-memory extraction
    path and stitches the independently extracted surface chunks.
    """

    target_voxels: int = 4_000_000
    min_samples_per_cell: float = 16.0
    min_samples_per_wall: float = 2.5
    use_cpp_sdf: bool = True
    batch_count: int = 1
    processing_mode: ProcessingMode = "single_pass"

    def validate(self) -> None:
        if self.target_voxels < 100_000:
            raise ValueError("target_voxels is too small")
        if self.min_samples_per_cell < 8.0:
            raise ValueError("min_samples_per_cell must be at least 8")
        if self.min_samples_per_wall < 2.0:
            raise ValueError("min_samples_per_wall must be at least 2")
        if self.processing_mode not in (
            "single_pass",
            "batched_field",
            "chunked_marching_cubes",
        ):
            raise ValueError("processing_mode is invalid")
        if self.batch_count < 1:
            raise ValueError("batch_count must be at least 1")
        if self.processing_mode == "single_pass" and self.batch_count != 1:
            raise ValueError("single_pass requires batch_count=1")
        if self.processing_mode != "single_pass" and self.batch_count < 2:
            raise ValueError("batched modes require batch_count at least 2")


@dataclass(frozen=True)
class TransitionParameters:
    """User-facing controls for a provider-agnostic implicit transition."""

    plane_axis: PlaneAxis = "Y"
    plane_position_mm: float = 0.0
    angle1_deg: float = 0.0
    angle2_deg: float = 0.0
    transition_width_mm: float = 10.0
    center_offset_mm: float = 0.0
    weight_kind: TransitionWeightKind = "automatic"
    sigmoid_sharpness: float = 1.0
    g_on_negative_side: bool = True
    minimum_feature_mm: float | None = None
    automatic_registration: bool = True
    topology_correction: bool = True
    driver_mode: TransitionDriverMode = "plane"
    driver_identifier: str | None = None
    field_interval_lower_mm: float = -2.0
    field_interval_upper_mm: float = 2.0

    def validate(self) -> None:
        if self.plane_axis not in ("X", "Y", "Z"):
            raise ValueError("plane_axis must be 'X', 'Y' or 'Z'")
        values = (
            self.plane_position_mm,
            self.angle1_deg,
            self.angle2_deg,
            self.transition_width_mm,
            self.center_offset_mm,
            self.sigmoid_sharpness,
            self.field_interval_lower_mm,
            self.field_interval_upper_mm,
        )
        if not all(np.isfinite(value) for value in values):
            raise ValueError("transition parameters must be finite")
        if self.sigmoid_sharpness <= 0:
            raise ValueError("sigmoid_sharpness must be positive")
        if self.driver_mode not in ("plane", "field"):
            raise ValueError("driver_mode must be 'plane' or 'field'")
        if self.driver_mode == "field":
            if not self.driver_identifier:
                raise ValueError("field-driven transition requires a field object")
            if self.field_interval_lower_mm >= self.field_interval_upper_mm:
                raise ValueError("field transition lower limit must be less than upper limit")
        elif self.transition_width_mm <= 0:
            raise ValueError("transition_width_mm must be positive")
        ProviderTransitionSpec(
            width_mm=(
                self.field_interval_upper_mm - self.field_interval_lower_mm
                if self.driver_mode == "field"
                else self.transition_width_mm
            ),
            center_offset_mm=(
                0.5 * (self.field_interval_lower_mm + self.field_interval_upper_mm)
                if self.driver_mode == "field"
                else self.center_offset_mm
            ),
            weight_kind=self.weight_kind,
            sigmoid_sharpness=self.sigmoid_sharpness,
            first_on_negative_side=self.g_on_negative_side,
            minimum_feature_mm=self.minimum_feature_mm,
            automatic_registration=self.automatic_registration,
            topology_correction=self.topology_correction,
        ).validate()

    def provider_spec(self) -> ProviderTransitionSpec:
        """Translate UI controls to the generic transition module interface."""

        self.validate()
        width_mm = (
            self.field_interval_upper_mm - self.field_interval_lower_mm
            if self.driver_mode == "field"
            else self.transition_width_mm
        )
        center_offset_mm = (
            0.5 * (self.field_interval_lower_mm + self.field_interval_upper_mm)
            if self.driver_mode == "field"
            else self.center_offset_mm
        )
        return ProviderTransitionSpec(
            width_mm=width_mm,
            center_offset_mm=center_offset_mm,
            weight_kind=self.weight_kind,
            sigmoid_sharpness=self.sigmoid_sharpness,
            first_on_negative_side=self.g_on_negative_side,
            minimum_feature_mm=self.minimum_feature_mm,
            automatic_registration=self.automatic_registration,
            topology_correction=self.topology_correction,
        )


__all__ = [
    "TPMSKind",
    "LatticeKind",
    "TPMS_KINDS",
    "CellSize",
    "CellMapMode",
    "PlaneAxis",
    "TransitionDriverMode",
    "ProcessingMode",
    "ExportSpacingMode",
    "TPMSParameterState",
    "FRAME_INPUT_DECIMALS",
    "FRAME_ORIGIN_DEFAULT_ATOL_MM",
    "TPMSParameters",
    "CustomUnitCellParameters",
    "LatticeParameters",
    "SamplingParameters",
    "TransitionParameters",
]
