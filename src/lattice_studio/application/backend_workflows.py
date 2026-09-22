"""Backend-owned implementations of serializable implicit-workflow tasks."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path
from struct import pack
from threading import RLock
from uuid import uuid4
from zlib import crc32, compress

import numpy as np
import trimesh

from lattice_studio.application.tasks import TaskContext, TaskOutcome
from lattice_studio.domain.parameters import (
    CustomUnitCellParameters,
    SamplingParameters,
    TPMSParameters,
    TransitionParameters,
)
from lattice_studio.engine.design_domains import AnalyticDesignDomain, MeshDesignDomain
from lattice_studio.engine.implicit.primitives import ImplicitPrimitive
from lattice_studio.engine.implicit.precise_render import (
    PreciseRenderCamera,
    PreciseRenderMaterial,
    PreciseRenderSettings,
    render_precise_implicit,
)
from lattice_studio.engine.contracts import ImplicitGenerationResult
from lattice_studio.engine.workflows import (
    generate_implicit_custom_lattice,
    generate_implicit_custom_lattice_for_design_domain,
    generate_implicit_lattice,
    generate_implicit_lattice_for_design_domain,
    generate_implicit_lattice_transition,
    generate_implicit_lattice_transition_for_design_domain,
    generate_implicit_shell,
    generate_implicit_shell_lattice_union,
    refine_implicit_display_field,
    reconstruct_implicit_mesh,
)


class ImplicitWorkflowTasks:
    """Run TPMS and STL tasks while retaining authority in the backend process."""

    def __init__(self) -> None:
        self._lock = RLock()
        self._generations: dict[str, ImplicitGenerationResult] = {}

    def handlers(self) -> dict[str, object]:
        return {
            "tpms.generate": self.generate_tpms,
            "custom.generate": self.generate_custom,
            "transition.generate": self.generate_transition,
            "generation.batch": self.generate_batch,
            "shell.generate": self.generate_shell,
            "shell.union": self.generate_shell_union,
            "display.refine": self.refine_display,
            "render.precise": self.render_precise,
            "stl.reconstruct": self.reconstruct_stl,
        }

    def generate_tpms(
        self,
        payload: Mapping[str, object],
        context: TaskContext,
        *,
        field_name: str = "display-field.npz",
    ) -> TaskOutcome:
        context.report("loading design domain", 0.02)
        parameters = TPMSParameters(**_object(payload["parameters"], "parameters"))
        sampling = SamplingParameters(**_object(payload.get("sampling", {}), "sampling"))
        display_voxel_size = _optional_positive_float(
            payload.get("display_voxel_size_mm"), "display_voxel_size_mm"
        )
        display_memory_budget = float(payload.get("display_memory_budget_mb", 512.0))
        display_batch_count = _optional_positive_int(
            payload.get("display_batch_count"), "display_batch_count"
        )
        if display_memory_budget <= 0.0:
            raise ValueError("display_memory_budget_mb must be positive")

        def progress(message: str, value: float) -> None:
            context.report(message, 0.02 + 0.96 * min(max(float(value), 0.0), 1.0))

        domain = _design_domain(payload)
        if domain is None:
            result = generate_implicit_lattice(
                _load_mesh(payload["mesh_path"]),
                parameters,
                sampling,
                display_voxel_size_mm=display_voxel_size,
                display_memory_budget_mb=display_memory_budget,
                display_batch_count=display_batch_count,
                progress=progress,
            )
        else:
            result = generate_implicit_lattice_for_design_domain(
                domain,
                parameters,
                sampling,
                display_voxel_size_mm=display_voxel_size,
                display_memory_budget_mb=display_memory_budget,
                display_batch_count=display_batch_count,
                progress=progress,
            )
        return self._publish_generation(result, parameters.kind, context, field_name=field_name)

    def generate_custom(
        self,
        payload: Mapping[str, object],
        context: TaskContext,
        *,
        field_name: str = "display-field.npz",
    ) -> TaskOutcome:
        context.report("loading design domain", 0.02)
        parameters = _custom_parameters(_object(payload["parameters"], "parameters"))
        sampling = SamplingParameters(**_object(payload.get("sampling", {}), "sampling"))

        def progress(message: str, value: float) -> None:
            context.report(message, 0.02 + 0.96 * _bounded_progress(value))

        domain = _design_domain(payload)
        arguments = {
            "display_voxel_size_mm": _optional_positive_float(
                payload.get("display_voxel_size_mm"), "display_voxel_size_mm"
            ),
            "display_memory_budget_mb": _display_memory_budget(payload),
            "display_batch_count": _optional_positive_int(
                payload.get("display_batch_count"), "display_batch_count"
            ),
            "progress": progress,
        }
        if domain is None:
            result = generate_implicit_custom_lattice(
                _load_mesh(payload["mesh_path"]),
                parameters,
                sampling,
                **arguments,
            )
        else:
            result = generate_implicit_custom_lattice_for_design_domain(
                domain,
                parameters,
                sampling,
                **arguments,
            )
        return self._publish_generation(result, "Custom", context, field_name=field_name)

    def generate_transition(
        self,
        payload: Mapping[str, object],
        context: TaskContext,
        *,
        field_name: str = "display-field.npz",
    ) -> TaskOutcome:
        context.report("loading design domain", 0.02)
        first = _lattice_parameters(_object(payload["first_parameters"], "first_parameters"))
        second = _lattice_parameters(_object(payload["second_parameters"], "second_parameters"))
        transition = TransitionParameters(
            **_object(payload["transition"], "transition")
        )
        sampling = SamplingParameters(**_object(payload.get("sampling", {}), "sampling"))

        def progress(message: str, value: float) -> None:
            context.report(message, 0.02 + 0.96 * _bounded_progress(value))

        domain = _design_domain(payload)
        arguments = {
            "display_voxel_size_mm": _optional_positive_float(
                payload.get("display_voxel_size_mm"), "display_voxel_size_mm"
            ),
            "display_memory_budget_mb": _display_memory_budget(payload),
            "display_batch_count": _optional_positive_int(
                payload.get("display_batch_count"), "display_batch_count"
            ),
            "progress": progress,
        }
        driver = _transition_driver(payload, transition)
        if domain is None:
            if driver is not None:
                raise ValueError("field-driven transition requires a serialized design domain")
            result = generate_implicit_lattice_transition(
                _load_mesh(payload["mesh_path"]),
                first,
                second,
                transition,
                sampling,
                **arguments,
            )
        else:
            result = generate_implicit_lattice_transition_for_design_domain(
                domain,
                first,
                second,
                transition,
                sampling,
                transition_driver=driver,
                **arguments,
            )
        return self._publish_generation(result, "Transition", context, field_name=field_name)

    def generate_batch(
        self,
        payload: Mapping[str, object],
        context: TaskContext,
    ) -> TaskOutcome:
        """Generate multiple fixed lattice requests within one cancellable task."""

        requests = _generation_requests(payload.get("requests"))
        created_generation_ids: list[str] = []
        artifacts: list[Path] = []
        generations: list[dict[str, object]] = []
        try:
            for index, request in enumerate(requests):
                request_id = _non_empty_string(request.get("request_id"), "request_id")
                task_kind = str(request.get("kind"))
                request_payload = _object(request.get("payload"), "generation payload")
                field_name = f"generation-{index + 1}-display-field.npz"
                child_context = _ScaledTaskContext(
                    context,
                    start=index / len(requests),
                    end=(index + 1) / len(requests),
                    label=request_id,
                )
                if task_kind == "tpms.generate":
                    outcome = self.generate_tpms(
                        request_payload,
                        child_context,
                        field_name=field_name,
                    )
                elif task_kind == "custom.generate":
                    outcome = self.generate_custom(
                        request_payload,
                        child_context,
                        field_name=field_name,
                    )
                elif task_kind == "transition.generate":
                    outcome = self.generate_transition(
                        request_payload,
                        child_context,
                        field_name=field_name,
                    )
                else:
                    raise ValueError(f"unsupported batch generation kind: {task_kind}")
                result = _object(outcome.result, "generation result")
                generation_id = _non_empty_string(
                    result.get("generation_id"), "generation_id"
                )
                created_generation_ids.append(generation_id)
                generations.append(
                    {
                        **result,
                        "request_id": request_id,
                        "display_field_name": field_name,
                    }
                )
                artifacts.extend(outcome.artifacts)
            context.report("batch display fields ready", 1.0)
            return TaskOutcome(result={"generations": generations}, artifacts=tuple(artifacts))
        except Exception:
            self._discard_generations(created_generation_ids)
            raise

    def generate_shell(
        self,
        payload: Mapping[str, object],
        context: TaskContext,
    ) -> TaskOutcome:
        context.report("loading design domain", 0.02)
        mesh = _load_mesh(payload["mesh_path"])
        sampling = SamplingParameters(**_object(payload.get("sampling", {}), "sampling"))

        def progress(message: str, value: float) -> None:
            context.report(message, 0.02 + 0.96 * _bounded_progress(value))

        result = generate_implicit_shell(
            mesh,
            _positive_float(payload["thickness_mm"], "thickness_mm"),
            sampling,
            display_voxel_size_mm=_optional_positive_float(
                payload.get("display_voxel_size_mm"), "display_voxel_size_mm"
            ),
            display_memory_budget_mb=_display_memory_budget(payload),
            display_batch_count=_optional_positive_int(
                payload.get("display_batch_count"), "display_batch_count"
            ),
            progress=progress,
        )
        return self._publish_generation(result, "Shell", context)

    def generate_shell_union(
        self,
        payload: Mapping[str, object],
        context: TaskContext,
    ) -> TaskOutcome:
        lattice = self._generation(payload["generation_id"])
        context.report("loading design domain", 0.02)
        mesh = _load_mesh(payload["mesh_path"])
        sampling = SamplingParameters(**_object(payload.get("sampling", {}), "sampling"))

        def progress(message: str, value: float) -> None:
            context.report(message, 0.02 + 0.96 * _bounded_progress(value))

        result = generate_implicit_shell_lattice_union(
            mesh,
            lattice,
            _positive_float(payload["thickness_mm"], "thickness_mm"),
            _non_negative_float(payload["fusion_radius_mm"], "fusion_radius_mm"),
            sampling,
            display_voxel_size_mm=_optional_positive_float(
                payload.get("display_voxel_size_mm"), "display_voxel_size_mm"
            ),
            display_memory_budget_mb=_display_memory_budget(payload),
            display_batch_count=_optional_positive_int(
                payload.get("display_batch_count"), "display_batch_count"
            ),
            progress=progress,
        )
        return self._publish_generation(result, "ShellUnion", context)

    def refine_display(
        self,
        payload: Mapping[str, object],
        context: TaskContext,
    ) -> TaskOutcome:
        generation_id = str(payload["generation_id"])
        generation = self._generation(generation_id)
        quality = str(payload["quality"])

        def progress(message: str, value: float) -> None:
            context.report(message, _bounded_progress(value))

        field, report = refine_implicit_display_field(
            generation.body,
            generation.display_field,
            quality=quality,
            display_memory_budget_mb=_positive_float(
                payload["display_memory_budget_mb"], "display_memory_budget_mb"
            ),
            display_batch_count=_optional_positive_int(
                payload.get("display_batch_count"), "display_batch_count"
            ),
            progress=progress,
        )
        context.raise_if_cancelled()
        output = context.work_dir / "display-field.npz"
        _write_field_value(output, field)
        context.report("display field ready", 1.0)
        return TaskOutcome(
            result={"generation_id": generation_id, "quality": quality, "report": asdict(report)},
            artifacts=(output,),
        )

    def render_precise(
        self,
        payload: Mapping[str, object],
        context: TaskContext,
    ) -> TaskOutcome:
        generation_id = str(payload["generation_id"])
        generation = self._generation(generation_id)
        camera = PreciseRenderCamera(**_object(payload["camera"], "camera"))
        settings = PreciseRenderSettings(**_object(payload["settings"], "settings"))
        material = PreciseRenderMaterial(
            **_object(payload.get("material", {}), "material")
        )
        background_lower = _color(payload.get("background_lower"), "background_lower")
        background_upper = _color(payload.get("background_upper"), "background_upper")

        def progress(message: str, value: float) -> None:
            context.report(message, _bounded_progress(value))

        rendered = render_precise_implicit(
            generation.body,
            camera=camera,
            settings=settings,
            material=material,
            background_lower=background_lower,
            background_upper=background_upper,
            progress=progress,
        )
        context.raise_if_cancelled()
        output = context.work_dir / "precise-render.png"
        _write_png(output, rendered.image_rgb)
        context.report("render ready", 1.0)
        return TaskOutcome(
            result={"generation_id": generation_id, "report": asdict(rendered.report)},
            artifacts=(output,),
        )

    def reconstruct_stl(
        self,
        payload: Mapping[str, object],
        context: TaskContext,
    ) -> TaskOutcome:
        generation_id = str(payload["generation_id"])
        result = self._generation(generation_id)
        if result.cell_map is None or result.minimum_feature_mm is None:
            raise ValueError("generation cannot be reconstructed as STL")
        tolerance_mm = _positive_float(payload.get("tolerance_mm", 0.2), "tolerance_mm")
        repair_tolerance_mm = float(payload.get("repair_tolerance_mm", 0.0))
        processing_mode = str(payload.get("processing_mode", "single_pass"))
        batch_count = int(payload.get("batch_count", 1))
        spacing_mode = str(payload.get("spacing_mode", "recommended"))
        clean_numerical_fragments = bool(payload.get("clean_numerical_fragments", True))
        optimize_for_slicing = bool(payload.get("optimize_for_slicing", True))

        def progress(message: str, value: float) -> None:
            context.report(message, min(max(float(value), 0.0), 1.0))

        mesh_result = reconstruct_implicit_mesh(
            result.body,
            result.cell_map,
            tolerance_mm,
            result.minimum_feature_mm,
            repair_tolerance_mm=repair_tolerance_mm,
            clean_numerical_fragments=clean_numerical_fragments,
            processing_mode=processing_mode,  # validated by the workflow
            batch_count=batch_count,
            spacing_mode=spacing_mode,  # validated by the workflow
            optimize_for_slicing=optimize_for_slicing,
            progress=progress,
        )
        context.raise_if_cancelled()
        output = context.work_dir / "lattice.stl"
        mesh_result.mesh.export(output)
        context.report("STL ready", 1.0)
        return TaskOutcome(
            result={
                "generation_id": generation_id,
                "triangle_count": len(mesh_result.mesh.faces),
                "spacing_mm": list(mesh_result.spacing_mm),
                "tolerance_mm": mesh_result.tolerance_mm,
                "extraction_backend": mesh_result.extraction_backend,
            },
            artifacts=(output,),
        )

    def _generation(self, value: object) -> ImplicitGenerationResult:
        generation_id = str(value)
        with self._lock:
            result = self._generations.get(generation_id)
        if result is None:
            raise KeyError(f"unknown backend generation: {generation_id}")
        return result

    def _publish_generation(
        self,
        result: ImplicitGenerationResult,
        kind: str,
        context: TaskContext,
        *,
        field_name: str = "display-field.npz",
    ) -> TaskOutcome:
        context.raise_if_cancelled()
        generation_id = uuid4().hex
        with self._lock:
            self._generations[generation_id] = result
        field_path = context.work_dir / field_name
        _write_field(field_path, result)
        context.report("display field ready", 1.0)
        return TaskOutcome(
            result={
                "generation_id": generation_id,
                "kind": str(kind),
                "recommendation": _recommendation_payload(result),
                "cell_map": _cell_map_payload(result),
                "minimum_feature_mm": result.minimum_feature_mm,
            },
            artifacts=(field_path,),
        )

    def _discard_generations(self, identifiers: list[str]) -> None:
        with self._lock:
            for identifier in identifiers:
                self._generations.pop(identifier, None)


def _load_mesh(value: object) -> trimesh.Trimesh:
    path = Path(str(value)).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"design-domain mesh does not exist: {path}")
    # STL has no shared vertex table.  Let trimesh merge its duplicated face
    # vertices at this file boundary so watertightness has its geometric meaning.
    mesh = trimesh.load_mesh(path, force="mesh", process=True)
    if not isinstance(mesh, trimesh.Trimesh) or len(mesh.faces) == 0:
        raise ValueError("design-domain mesh must contain triangular faces")
    return mesh


def _design_domain(payload: Mapping[str, object]):
    """Restore a backend-local domain definition when a DTO was supplied."""

    value = payload.get("domain")
    if value is None:
        return None
    domain = _object(value, "domain")
    kind = str(domain.get("kind"))
    name = _non_empty_string(domain.get("name"), "domain name")
    if kind == "mesh":
        source = _non_empty_string(domain.get("source_path"), "domain source_path")
        return MeshDesignDomain(_load_mesh(source), name, Path(source).expanduser().resolve())
    if kind == "analytic":
        return AnalyticDesignDomain(_primitive(domain.get("primitive")), name)
    raise ValueError("unsupported design-domain kind")


def _transition_driver(
    payload: Mapping[str, object],
    transition: TransitionParameters,
):
    if transition.driver_mode == "plane":
        if payload.get("driver_primitive") is not None:
            raise ValueError("plane-driven transition must not include a field object")
        return None
    primitive = _primitive(payload.get("driver_primitive"))
    if primitive.identifier != transition.driver_identifier:
        raise ValueError("transition field object does not match driver_identifier")
    return primitive.as_implicit_body()


def _primitive(value: object) -> ImplicitPrimitive:
    entry = _object(value, "primitive")
    try:
        return ImplicitPrimitive(
            identifier=_non_empty_string(entry.get("identifier"), "primitive identifier"),
            name=_non_empty_string(entry.get("name"), "primitive name"),
            kind=str(entry["kind"]),
            center_mm=_triple(entry.get("center_mm"), "center_mm"),
            rotation_euler_deg=_triple(
                entry.get("rotation_euler_deg"), "rotation_euler_deg"
            ),
            radius_mm=float(entry["radius_mm"]),
            height_mm=float(entry["height_mm"]),
            size_mm=_triple(entry.get("size_mm"), "size_mm"),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("invalid primitive") from exc


def _triple(value: object, name: str) -> tuple[float, float, float]:
    if not isinstance(value, list) or len(value) != 3:
        raise ValueError(f"{name} must contain exactly three values")
    return tuple(float(item) for item in value)  # type: ignore[return-value]


def _write_field(path: Path, result: ImplicitGenerationResult) -> None:
    _write_field_value(path, result.display_field)


def _write_field_value(path: Path, field) -> None:
    np.savez_compressed(
        path,
        name=np.asarray(field.name),
        values=field.values,
        origin=field.origin,
        spacing=field.spacing,
        color=np.asarray(field.color),
        is_preview_only=np.asarray(field.is_preview_only),
    )


def _recommendation_payload(result: ImplicitGenerationResult) -> dict[str, object]:
    recommendation = result.recommendation
    return {
        "voxel_size_mm": recommendation.voxel_size_mm,
        "grid_shape": list(recommendation.grid_shape),
        "estimated_voxels": recommendation.estimated_voxels,
        "samples_per_cell": recommendation.samples_per_cell,
        "samples_per_wall": recommendation.samples_per_wall,
        "limiting_constraint": recommendation.limiting_constraint,
    }


def _cell_map_payload(result: ImplicitGenerationResult) -> dict[str, object] | None:
    cell_map = result.cell_map
    if cell_map is None:
        return None
    return {
        "bounds": cell_map.bounds.tolist(),
        "spacing_mm": list(cell_map.spacing_mm),
        "cell_counts": list(cell_map.cell_counts),
        "boundary_mode": cell_map.boundary_mode,
        "index_min": list(cell_map.index_min),
        "frame_origin": cell_map.frame.origin.tolist(),
        "frame_axes": cell_map.frame.axes.tolist(),
    }


def _object(value: object, name: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a JSON object")
    return value


def _generation_requests(value: object) -> list[dict[str, object]]:
    if not isinstance(value, list) or len(value) < 2:
        raise ValueError("generation batch requires at least two requests")
    if len(value) > 16:
        raise ValueError("generation batch may contain at most 16 requests")
    requests = [_object(item, "generation request") for item in value]
    request_ids = [
        _non_empty_string(request.get("request_id"), "request_id")
        for request in requests
    ]
    if len(set(request_ids)) != len(request_ids):
        raise ValueError("batch request_id values must be unique")
    return requests


def _non_empty_string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


class _ScaledTaskContext:
    """Map one batch entry's local progress into its parent task range."""

    def __init__(
        self,
        parent: TaskContext,
        *,
        start: float,
        end: float,
        label: str,
    ) -> None:
        self._parent = parent
        self._start = float(start)
        self._end = float(end)
        self._label = label

    @property
    def work_dir(self) -> Path:
        return self._parent.work_dir

    @property
    def cancelled(self) -> bool:
        return self._parent.cancelled

    def raise_if_cancelled(self) -> None:
        self._parent.raise_if_cancelled()

    def report(self, message: str, progress: float) -> None:
        local = _bounded_progress(progress)
        self._parent.report(
            f"{self._label}: {message}",
            self._start + (self._end - self._start) * local,
        )


def _positive_float(value: object, name: str) -> float:
    numeric = float(value)
    if numeric <= 0.0:
        raise ValueError(f"{name} must be positive")
    return numeric


def _non_negative_float(value: object, name: str) -> float:
    numeric = float(value)
    if numeric < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return numeric


def _optional_positive_float(value: object, name: str) -> float | None:
    return None if value is None else _positive_float(value, name)


def _optional_positive_int(value: object, name: str) -> int | None:
    if value is None:
        return None
    numeric = int(value)
    if numeric < 1:
        raise ValueError(f"{name} must be at least one")
    return numeric


def _display_memory_budget(payload: Mapping[str, object]) -> float:
    return _positive_float(
        payload.get("display_memory_budget_mb", 512.0),
        "display_memory_budget_mb",
    )


def _lattice_parameters(value: Mapping[str, object]):
    if str(value.get("kind")) == "Custom":
        return _custom_parameters(value)
    return TPMSParameters(**value)


def _custom_parameters(value: Mapping[str, object]) -> CustomUnitCellParameters:
    payload = dict(value)
    try:
        payload["source_path"] = Path(str(payload["source_path"]))
    except KeyError as exc:
        raise ValueError("custom parameters require source_path") from exc
    bridge_directions = payload.get("bridge_directions")
    if isinstance(bridge_directions, list):
        payload["bridge_directions"] = tuple(str(item) for item in bridge_directions)
    return CustomUnitCellParameters(**payload)


def _bounded_progress(value: object) -> float:
    return min(max(float(value), 0.0), 1.0)


def _color(value: object, name: str) -> tuple[float, float, float]:
    if not isinstance(value, list) or len(value) != 3:
        raise ValueError(f"{name} must contain three values")
    values = tuple(float(item) for item in value)
    if not np.isfinite(values).all() or any(item < 0.0 or item > 1.0 for item in values):
        raise ValueError(f"{name} values must be between 0 and 1")
    return values


def _write_png(path: Path, image: np.ndarray) -> None:
    """Write a compact RGB PNG without adding a backend GUI/image dependency."""

    pixels = np.ascontiguousarray(image, dtype=np.uint8)
    if pixels.ndim != 3 or pixels.shape[2] != 3:
        raise ValueError("precise render image must have shape (height, width, 3)")
    height, width, _channels = pixels.shape
    rows = b"".join(b"\x00" + row.tobytes() for row in pixels)

    def chunk(kind: bytes, payload: bytes) -> bytes:
        return (
            pack(">I", len(payload))
            + kind
            + payload
            + pack(">I", crc32(kind + payload) & 0xFFFFFFFF)
        )

    path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", compress(rows, level=6))
        + chunk(b"IEND", b"")
    )


__all__ = ["ImplicitWorkflowTasks"]
