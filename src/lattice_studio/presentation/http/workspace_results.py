"""Materialize a Qt-side workspace projection from a backend JSON snapshot.

The backend snapshot contains definitions and asset references only.  This
module may load those references for the desktop display, but it never sends a
mesh, evaluator, Qt object, or runtime cache back through the transport.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

import trimesh

from lattice_studio.domain.workspace import (
    DesignDocument,
    DesignWorkspace,
    PersistedDerivedMesh,
)
from lattice_studio.engine.design_domains import AnalyticDesignDomain, MeshDesignDomain
from lattice_studio.engine.implicit.primitives import ImplicitPrimitive


def read_workspace_projection(
    snapshot: Mapping[str, object],
    asset_root: Path,
) -> DesignWorkspace:
    """Build the disposable desktop projection of a backend-owned workspace."""

    root = Path(asset_root).resolve()
    active = _documents(snapshot.get("documents"), root, archived=False)
    archived = _documents(snapshot.get("archived_documents"), root, archived=True)
    active_by_id = {document.identifier: document for document in active}
    archived_by_id = {document.identifier: document for document in archived}
    if len(active) != len(active_by_id) or len(archived) != len(archived_by_id):
        raise ValueError("workspace snapshot document identifiers must be unique")
    active_design_id = snapshot.get("active_design_id")
    return DesignWorkspace(
        active_by_id,
        archived_by_id,
        None if active_design_id is None else _string(active_design_id, "active_design_id"),
        bool(snapshot.get("dirty", False)),
    )


def _documents(value: object, root: Path, *, archived: bool) -> list[DesignDocument]:
    if not isinstance(value, list):
        raise ValueError("workspace snapshot documents must be an array")
    return [_document(entry, root, archived=archived) for entry in value]


def _document(value: object, root: Path, *, archived: bool) -> DesignDocument:
    entry = _mapping(value, "workspace document")
    domain = _domain(entry.get("domain"), root)
    primitives: dict[str, ImplicitPrimitive] = {}
    visibility: dict[str, bool] = {}
    field_primitives = entry.get("field_primitives", [])
    if not isinstance(field_primitives, list):
        raise ValueError("field_primitives must be an array")
    for item in field_primitives:
        field = _mapping(item, "field primitive")
        primitive = _primitive(field.get("primitive"))
        if primitive.identifier in primitives:
            raise ValueError("field primitive identifiers must be unique")
        primitives[primitive.identifier] = primitive
        visibility[primitive.identifier] = bool(field.get("visible", True))
    derived: dict[str, PersistedDerivedMesh] = {}
    records = entry.get("derived_meshes", [])
    if not isinstance(records, list):
        raise ValueError("derived_meshes must be an array")
    for item in records:
        record_value = _mapping(item, "derived mesh")
        record = PersistedDerivedMesh(
            _string(record_value.get("identifier"), "derived mesh identifier"),
            _relative_path(record_value.get("asset_path"), root),
            _json_object(record_value.get("provenance"), "derived mesh provenance"),
        )
        if record.identifier in derived:
            raise ValueError("derived mesh identifiers must be unique")
        derived[record.identifier] = record
    return DesignDocument(
        identifier=_string(entry.get("document_id"), "document_id"),
        name=_string(entry.get("name"), "document name"),
        domain=domain,
        settings=_json_object(entry.get("settings", {}), "settings"),
        field_primitives=primitives,
        field_primitive_visibility=visibility,
        derived_meshes=derived,
        revision=int(entry.get("revision", 0)),
        archived=archived,
    )


def _domain(value: object, root: Path):
    entry = _mapping(value, "domain")
    kind = _string(entry.get("kind"), "domain kind")
    name = _string(entry.get("name"), "domain name")
    if kind == "analytic":
        return AnalyticDesignDomain(_primitive(entry.get("primitive")), name)
    if kind == "mesh":
        relative = _relative_path(entry.get("asset_path"), root)
        mesh = trimesh.load(root / relative, force="mesh", process=False)
        if not isinstance(mesh, trimesh.Trimesh) or not mesh.faces.size:
            raise ValueError("workspace mesh asset must contain triangular faces")
        return MeshDesignDomain(mesh, name, Path(relative))
    raise ValueError("unsupported workspace domain kind")


def _primitive(value: object) -> ImplicitPrimitive:
    entry = _mapping(value, "primitive")
    try:
        return ImplicitPrimitive(
            identifier=_string(entry.get("identifier"), "primitive identifier"),
            name=_string(entry.get("name"), "primitive name"),
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


def _relative_path(value: object, root: Path) -> str:
    path = Path(_string(value, "asset_path"))
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("workspace asset paths must be relative to the manifest")
    resolved = (root / path).resolve()
    if root not in resolved.parents:
        raise ValueError("workspace asset path escapes the manifest root")
    return str(path)


def _mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return value


def _json_object(value: object, name: str) -> dict[str, object]:
    try:
        checked = json.loads(json.dumps(value, allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be JSON-safe") from exc
    if not isinstance(checked, dict):
        raise ValueError(f"{name} must be an object")
    return checked


def _string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _triple(value: object, name: str) -> tuple[float, float, float]:
    if not isinstance(value, list) or len(value) != 3:
        raise ValueError(f"{name} must contain exactly three values")
    return tuple(float(item) for item in value)  # type: ignore[return-value]


__all__ = ["read_workspace_projection"]
