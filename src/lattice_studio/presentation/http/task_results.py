"""Client-side readers for serializable local-backend task outputs.

The backend remains the owner of every implicit evaluator.  The desktop process
may only materialize disposable display fields and opaque generation handles
from task responses.
"""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from typing import Mapping

import numpy as np

from lattice_studio.application.tasks import TaskSnapshot
from lattice_studio.domain.cell_map import CellMap, CellMapFrame
from lattice_studio.engine.contracts import SamplingRecommendation
from lattice_studio.engine.implicit.field import SampledImplicitField


@dataclass(frozen=True)
class BackendGenerationResult:
    """A frontend-safe reference to an implicit result owned by the backend."""

    generation_id: str
    kind: str
    display_field: SampledImplicitField
    recommendation: SamplingRecommendation
    cell_map: CellMap | None
    minimum_feature_mm: float | None


def read_generation_result(
    snapshot: TaskSnapshot,
    display_field_bytes: bytes,
) -> BackendGenerationResult:
    """Decode a succeeded backend generation task without reconstructing its body."""

    if snapshot.state != "succeeded" or snapshot.result is None:
        raise ValueError("generation task must have succeeded")
    result = _mapping(snapshot.result, "generation result")
    return _read_generation_payload(result, display_field_bytes)


def read_generation_batch(
    snapshot: TaskSnapshot,
    display_fields: Mapping[str, bytes],
) -> dict[str, BackendGenerationResult]:
    """Decode a batch task into frontend-safe handles keyed by request ID."""

    if snapshot.state != "succeeded" or snapshot.result is None:
        raise ValueError("generation batch task must have succeeded")
    result = _mapping(snapshot.result, "generation batch result")
    entries = _sequence(result.get("generations"), "generations")
    if len(entries) < 2:
        raise ValueError("generation batch must contain at least two results")
    decoded: dict[str, BackendGenerationResult] = {}
    for entry in entries:
        payload = _mapping(entry, "generation batch entry")
        request_id = _string(payload.get("request_id"), "request_id")
        _string(payload.get("display_field_name"), "display_field_name")
        if request_id in decoded:
            raise ValueError("generation batch request_id values must be unique")
        try:
            display_field = display_fields[request_id]
        except KeyError as exc:
            raise ValueError(f"missing display field artifact: {request_id}") from exc
        decoded[request_id] = _read_generation_payload(payload, display_field)
    return decoded


def batch_display_field_artifact_ids(snapshot: TaskSnapshot) -> dict[str, str]:
    """Map each batch request ID to its downloadable NPZ artifact."""

    if snapshot.state != "succeeded" or snapshot.result is None:
        raise ValueError("generation batch task must have succeeded")
    result = _mapping(snapshot.result, "generation batch result")
    artifacts = {artifact.name: artifact.identifier for artifact in snapshot.artifacts}
    identifiers: dict[str, str] = {}
    for entry in _sequence(result.get("generations"), "generations"):
        payload = _mapping(entry, "generation batch entry")
        request_id = _string(payload.get("request_id"), "request_id")
        field_name = _string(payload.get("display_field_name"), "display_field_name")
        if request_id in identifiers or field_name not in artifacts:
            raise ValueError("generation batch artifacts do not match its result metadata")
        identifiers[request_id] = artifacts[field_name]
    return identifiers


def _read_generation_payload(
    result: Mapping[str, object],
    display_field_bytes: bytes,
) -> BackendGenerationResult:
    return BackendGenerationResult(
        generation_id=_string(result.get("generation_id"), "generation_id"),
        kind=_string(result.get("kind"), "kind"),
        display_field=read_sampled_field(display_field_bytes),
        recommendation=_recommendation(_mapping(result.get("recommendation"), "recommendation")),
        cell_map=_cell_map(result.get("cell_map")),
        minimum_feature_mm=_optional_float(
            result.get("minimum_feature_mm"), "minimum_feature_mm"
        ),
    )


def display_field_artifact_id(snapshot: TaskSnapshot) -> str:
    """Return the sole display-field artifact ID from a generation task."""

    matches = [artifact for artifact in snapshot.artifacts if artifact.name == "display-field.npz"]
    if len(matches) != 1:
        raise ValueError("generation task must publish exactly one display-field.npz")
    return matches[0].identifier


def read_sampled_field(raw: bytes) -> SampledImplicitField:
    """Load the NPZ display cache published by the backend with pickle disabled."""

    try:
        with np.load(BytesIO(raw), allow_pickle=False) as archive:
            name = str(archive["name"].item())
            values = np.asarray(archive["values"], dtype=np.float32)
            origin = np.asarray(archive["origin"], dtype=np.float64)
            spacing = np.asarray(archive["spacing"], dtype=np.float64)
            color = tuple(float(value) for value in archive["color"])
            is_preview_only = bool(archive["is_preview_only"].item())
    except (KeyError, OSError, ValueError) as exc:
        raise ValueError("invalid display-field artifact") from exc
    return SampledImplicitField(
        name=name,
        values=values,
        origin=origin,
        spacing=spacing,
        color=color,
        is_preview_only=is_preview_only,
    )


def _recommendation(value: Mapping[str, object]) -> SamplingRecommendation:
    grid_shape = tuple(int(item) for item in _sequence(value.get("grid_shape"), "grid_shape"))
    if len(grid_shape) != 3:
        raise ValueError("grid_shape must contain three values")
    return SamplingRecommendation(
        voxel_size_mm=_float(value.get("voxel_size_mm"), "voxel_size_mm"),
        grid_shape=grid_shape,
        estimated_voxels=int(value["estimated_voxels"]),
        samples_per_cell=_float(value.get("samples_per_cell"), "samples_per_cell"),
        samples_per_wall=_float(value.get("samples_per_wall"), "samples_per_wall"),
        limiting_constraint=_string(value.get("limiting_constraint"), "limiting_constraint"),
    )


def _cell_map(value: object) -> CellMap | None:
    if value is None:
        return None
    payload = _mapping(value, "cell_map")
    frame = CellMapFrame(
        np.asarray(_sequence(payload.get("frame_origin"), "frame_origin"), dtype=np.float64),
        np.asarray(_sequence(payload.get("frame_axes"), "frame_axes"), dtype=np.float64),
    )
    return CellMap(
        bounds=np.asarray(_sequence(payload.get("bounds"), "bounds"), dtype=np.float64),
        spacing_mm=tuple(
            _float(item, "spacing_mm")
            for item in _sequence(payload.get("spacing_mm"), "spacing_mm")
        ),
        cell_counts=tuple(
            int(item) for item in _sequence(payload.get("cell_counts"), "cell_counts")
        ),
        boundary_mode=_string(payload.get("boundary_mode"), "boundary_mode"),
        frame=frame,
        index_min=tuple(
            int(item) for item in _sequence(payload.get("index_min"), "index_min")
        ),
    )


def _mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return value


def _sequence(value: object, name: str) -> list[object]:
    if not isinstance(value, list):
        raise ValueError(f"{name} must be an array")
    return value


def _string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _float(value: object, name: str) -> float:
    numeric = float(value)
    if not np.isfinite(numeric):
        raise ValueError(f"{name} must be finite")
    return numeric


def _optional_float(value: object, name: str) -> float | None:
    return None if value is None else _float(value, name)


__all__ = [
    "BackendGenerationResult",
    "batch_display_field_artifact_ids",
    "display_field_artifact_id",
    "read_generation_batch",
    "read_generation_result",
    "read_sampled_field",
]
