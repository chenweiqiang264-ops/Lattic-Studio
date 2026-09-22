"""Versioned JSON persistence for the public workspace model.

The serializer is deliberately kept outside the domain aggregate.  It owns
file paths, mesh loading, managed assets, and schema migrations.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import trimesh

from lattice_studio.engine.implicit.primitives import ImplicitPrimitive
from lattice_studio.domain.workspace import (
    DesignDocument,
    DesignWorkspace,
    PersistedDerivedMesh,
)
from lattice_studio.engine.design_domains import AnalyticDesignDomain, MeshDesignDomain


CURRENT_VERSION = 2


def _primitive_to_json(primitive: ImplicitPrimitive) -> dict[str, Any]:
    return {
        "identifier": primitive.identifier,
        "name": primitive.name,
        "kind": primitive.kind,
        "center_mm": list(primitive.center_mm),
        "rotation_euler_deg": list(primitive.rotation_euler_deg),
        "radius_mm": primitive.radius_mm,
        "height_mm": primitive.height_mm,
        "size_mm": list(primitive.size_mm),
    }


def _primitive_from_json(value: object) -> ImplicitPrimitive:
    if not isinstance(value, dict):
        raise ValueError("primitive manifest entry must be an object")
    try:
        return ImplicitPrimitive(
            identifier=str(value["identifier"]),
            name=str(value["name"]),
            kind=str(value["kind"]),
            center_mm=tuple(float(item) for item in value["center_mm"]),
            rotation_euler_deg=tuple(float(item) for item in value["rotation_euler_deg"]),
            radius_mm=float(value["radius_mm"]),
            height_mm=float(value["height_mm"]),
            size_mm=tuple(float(item) for item in value["size_mm"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("invalid primitive manifest entry") from exc


def _relative_asset(path: Path | str, root: Path) -> str:
    candidate = Path(path)
    if candidate.is_absolute():
        candidate = candidate.resolve()
    else:
        candidate = (root / candidate).resolve()
    try:
        return str(candidate.relative_to(root.resolve()))
    except ValueError as exc:
        raise ValueError("workspace assets must be located below the workspace root") from exc


def _document_to_json(document: DesignDocument, root: Path) -> dict[str, Any]:
    domain: dict[str, Any]
    if isinstance(document.domain, MeshDesignDomain):
        if document.domain.asset_path is None:
            raise ValueError("mesh design domain must have a managed asset path")
        domain = {
            "kind": "mesh",
            "name": document.domain.name,
            "asset_path": _relative_asset(document.domain.asset_path, root),
        }
    elif isinstance(document.domain, AnalyticDesignDomain):
        domain = {
            "kind": "analytic",
            "name": document.domain.name,
            "primitive": _primitive_to_json(document.domain.primitive),
        }
    else:
        raise TypeError("unsupported design-domain implementation")

    return {
        "identifier": document.identifier,
        "name": document.name,
        "revision": document.revision,
        "domain": domain,
        "settings": document.settings,
        "field_primitives": [
            {
                "primitive": _primitive_to_json(primitive),
                "visible": document.field_primitive_visibility[identifier],
            }
            for identifier, primitive in document.field_primitives.items()
        ],
        "derived_meshes": [
            {
                "identifier": record.identifier,
                "asset_path": _relative_asset(record.asset_path, root),
                "provenance": record.provenance,
            }
            for record in document.derived_meshes.values()
        ],
    }


def _entries(payload: dict[str, Any], key: str) -> list[object]:
    entries = payload.get(key, [])
    if not isinstance(entries, list):
        raise ValueError(f"workspace manifest '{key}' must be a list")
    return entries


def _migrate(payload: dict[str, Any]) -> dict[str, Any]:
    version = payload.get("version", 1)
    if version == CURRENT_VERSION:
        return payload
    if version != 1:
        raise ValueError(f"unsupported workspace manifest version: {version}")
    migrated = dict(payload)
    migrated["version"] = CURRENT_VERSION
    migrated.setdefault("archived_documents", [])
    return migrated


def _document_from_json(
    value: object,
    root: Path,
    *,
    archived: bool,
) -> DesignDocument:
    if not isinstance(value, dict) or not isinstance(value.get("domain"), dict):
        raise ValueError("invalid design manifest entry")
    domain_value = value["domain"]
    kind = domain_value.get("kind")
    name = str(domain_value.get("name", ""))
    if kind == "mesh":
        relative = Path(str(domain_value["asset_path"]))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("mesh asset path must be workspace-relative")
        mesh = trimesh.load(root / relative, force="mesh")
        if not isinstance(mesh, trimesh.Trimesh):
            raise TypeError("managed design-domain asset is not a triangular mesh")
        domain = MeshDesignDomain(mesh, name, relative)
    elif kind == "analytic":
        domain = AnalyticDesignDomain(_primitive_from_json(domain_value["primitive"]), name)
    else:
        raise ValueError("unsupported design-domain kind")

    primitives: dict[str, ImplicitPrimitive] = {}
    visibility: dict[str, bool] = {}
    for entry in _entries(value, "field_primitives"):
        if not isinstance(entry, dict):
            raise ValueError("invalid field primitive entry")
        primitive = _primitive_from_json(entry["primitive"])
        if primitive.identifier in primitives:
            raise ValueError("field primitive identifiers must be unique")
        primitives[primitive.identifier] = primitive
        visibility[primitive.identifier] = bool(entry.get("visible", True))

    derived: dict[str, PersistedDerivedMesh] = {}
    for entry in _entries(value, "derived_meshes"):
        if not isinstance(entry, dict):
            raise ValueError("invalid derived mesh entry")
        record = PersistedDerivedMesh(
            str(entry["identifier"]),
            str(entry["asset_path"]),
            entry.get("provenance", {}),
        )
        if record.identifier in derived:
            raise ValueError("derived mesh identifiers must be unique")
        derived[record.identifier] = record

    return DesignDocument(
        identifier=str(value["identifier"]),
        name=str(value["name"]),
        domain=domain,
        settings=value.get("settings", {}),
        field_primitives=primitives,
        field_primitive_visibility=visibility,
        derived_meshes=derived,
        revision=int(value.get("revision", 0)),
        archived=archived,
    )


def save_workspace(workspace: DesignWorkspace, manifest_path: Path) -> None:
    path = Path(manifest_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": CURRENT_VERSION,
        "active_design_id": workspace.active_design_id,
        "documents": [_document_to_json(item, path.parent) for item in workspace.documents.values()],
        "archived_documents": [
            _document_to_json(item, path.parent)
            for item in workspace.archived_documents.values()
        ],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    workspace.dirty = False


def load_workspace(manifest_path: Path) -> DesignWorkspace:
    path = Path(manifest_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("workspace manifest must be an object")
    payload = _migrate(payload)
    documents = [_document_from_json(item, path.parent, archived=False) for item in _entries(payload, "documents")]
    archived = [_document_from_json(item, path.parent, archived=True) for item in _entries(payload, "archived_documents")]
    active = {item.identifier: item for item in documents}
    archived_by_id = {item.identifier: item for item in archived}
    if len(active) != len(documents) or len(archived_by_id) != len(archived):
        raise ValueError("design document identifiers must be unique")
    return DesignWorkspace(active, archived_by_id, payload.get("active_design_id"), False)


def package_stl_asset(source_path: Path, workspace_root: Path, category: str, identifier: str) -> Path:
    source = Path(source_path)
    if not source.is_file() or source.suffix.lower() != ".stl":
        raise ValueError("managed mesh source must be an existing STL file")
    if not category.strip() or not identifier.strip() or Path(identifier).name != identifier:
        raise ValueError("asset category and identifier must be safe names")
    target = Path(workspace_root) / "assets" / category / f"{identifier}.stl"
    target.parent.mkdir(parents=True, exist_ok=True)
    if source.resolve() != target.resolve():
        shutil.copy2(source, target)
    return target


__all__ = ["CURRENT_VERSION", "load_workspace", "package_stl_asset", "save_workspace"]
