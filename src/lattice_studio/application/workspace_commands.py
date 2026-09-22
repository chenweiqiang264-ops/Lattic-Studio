"""Serializable document commands for backend-owned workspaces.

The command boundary intentionally carries definitions and file references,
never Qt, trimesh, or implicit-evaluator objects.  It gives the loopback
backend an atomic mutation surface while leaving presentation code free to
materialize disposable local display state when necessary.
"""

from __future__ import annotations

import copy
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import trimesh

from lattice_studio.domain.workspace import (
    DesignDocument,
    DesignWorkspace,
    PersistedDerivedMesh,
)
from lattice_studio.engine.design_domains import AnalyticDesignDomain, MeshDesignDomain
from lattice_studio.engine.implicit.primitives import ImplicitPrimitive


@dataclass(frozen=True)
class WorkspaceMutation:
    """One command outcome returned with the resulting workspace snapshot."""

    kind: str
    document_id: str | None = None

    def to_dict(self) -> dict[str, object]:
        value: dict[str, object] = {"kind": self.kind}
        if self.document_id is not None:
            value["document_id"] = self.document_id
        return value


class WorkspaceCommandService:
    """Apply a fixed JSON command vocabulary transactionally."""

    def apply(
        self,
        workspace: DesignWorkspace,
        commands: Sequence[Mapping[str, object]],
    ) -> tuple[DesignWorkspace, tuple[WorkspaceMutation, ...]]:
        if not commands:
            raise ValueError("workspace command batch must not be empty")
        if len(commands) > 128:
            raise ValueError("workspace command batch may contain at most 128 commands")

        candidate = _copy_workspace(workspace)
        mutations = tuple(self._apply_one(candidate, command) for command in commands)
        return candidate, mutations

    def _apply_one(
        self,
        workspace: DesignWorkspace,
        command: Mapping[str, object],
    ) -> WorkspaceMutation:
        kind = _non_empty_string(command.get("kind"), "workspace command kind")
        if kind == "document.create.mesh":
            document = _create_document(
                workspace,
                _mesh_domain(command),
                _name(command),
                command,
            )
            return WorkspaceMutation(kind, document.identifier)
        if kind == "document.create.analytic":
            document = _create_document(
                workspace,
                AnalyticDesignDomain(_primitive(command.get("primitive")), _domain_name(command)),
                _name(command),
                command,
            )
            return WorkspaceMutation(kind, document.identifier)

        identifier = _non_empty_string(command.get("document_id"), "document_id")
        if kind == "document.activate":
            workspace.activate(identifier)
            return WorkspaceMutation(kind, identifier)
        if kind == "document.rename":
            workspace.rename_document(identifier, _name(command))
            return WorkspaceMutation(kind, identifier)
        if kind == "document.duplicate":
            source = _document(workspace, identifier)
            duplicate_id = command.get("new_document_id")
            document = source.duplicate(
                (
                    workspace.new_identifier()
                    if duplicate_id is None
                    else _non_empty_string(duplicate_id, "new_document_id")
                ),
                _name(command),
            )
            workspace.add_document(document, activate=bool(command.get("activate", True)))
            return WorkspaceMutation(kind, document.identifier)
        if kind == "document.archive":
            workspace.archive_document(identifier)
            return WorkspaceMutation(kind, identifier)
        if kind == "document.restore":
            workspace.restore_document(
                identifier,
                activate=bool(command.get("activate", True)),
            )
            return WorkspaceMutation(kind, identifier)
        if kind == "document.remove_archived":
            workspace.permanently_remove_archived_document(identifier)
            return WorkspaceMutation(kind, identifier)

        document = _document(workspace, identifier)
        if kind == "document.replace.mesh":
            document.replace_domain(_mesh_domain(command, default_name=document.name))
            workspace.dirty = True
        elif kind == "document.replace.analytic":
            document.replace_domain(
                AnalyticDesignDomain(
                    _primitive(command.get("primitive")),
                    _domain_name(command, default=document.name),
                )
            )
            workspace.dirty = True
        elif kind == "document.setting.set":
            document.set_setting(
                _non_empty_string(command.get("key"), "setting key"),
                _json_value(command.get("value"), "setting value"),
            )
            workspace.dirty = True
        elif kind == "document.settings.replace":
            settings = _json_object(command.get("settings"), "settings")
            if document.settings != settings:
                document.settings = settings
                document.clear_generated()
                document.touch()
                workspace.dirty = True
        elif kind == "document.field.upsert":
            document.upsert_field_primitive(
                _primitive(command.get("primitive")),
                visible=bool(command.get("visible", True)),
            )
            workspace.dirty = True
        elif kind == "document.field.remove":
            document.remove_field_primitive(
                _non_empty_string(command.get("primitive_id"), "primitive_id")
            )
            workspace.dirty = True
        elif kind == "document.field.visibility.set":
            document.set_field_primitive_visibility(
                _non_empty_string(command.get("primitive_id"), "primitive_id"),
                bool(command.get("visible", True)),
            )
            workspace.dirty = True
        elif kind == "document.field_scene.replace":
            primitives, visibility = _field_scene(command.get("field_primitives"))
            if document.replace_field_object_scene(primitives, visibility):
                document.clear_generated()
                workspace.dirty = True
        elif kind == "document.derived_meshes.replace":
            document.derived_meshes = _derived_meshes(command.get("derived_meshes"))
            workspace.dirty = True
        else:
            raise ValueError(f"unsupported workspace command: {kind}")
        return WorkspaceMutation(kind, identifier)


def workspace_snapshot(
    workspace_id: str,
    workspace: DesignWorkspace,
    *,
    mutations: Sequence[WorkspaceMutation] = (),
) -> dict[str, object]:
    """Return the JSON-safe editable workspace state for a transport client."""

    return {
        "workspace_id": workspace_id,
        "document_count": len(workspace.documents),
        "archived_document_count": len(workspace.archived_documents),
        "active_design_id": workspace.active_design_id,
        "dirty": workspace.dirty,
        "documents": [_document_snapshot(item) for item in workspace.documents.values()],
        "archived_documents": [
            _document_snapshot(item) for item in workspace.archived_documents.values()
        ],
        "mutations": [mutation.to_dict() for mutation in mutations],
    }


def _copy_workspace(workspace: DesignWorkspace) -> DesignWorkspace:
    documents = {
        identifier: _copy_document(document, archived=False)
        for identifier, document in workspace.documents.items()
    }
    archived = {
        identifier: _copy_document(document, archived=True)
        for identifier, document in workspace.archived_documents.items()
    }
    return DesignWorkspace(documents, archived, workspace.active_design_id, workspace.dirty)


def _copy_document(document: DesignDocument, *, archived: bool) -> DesignDocument:
    duplicate_domain = getattr(document.domain, "duplicate_definition", None)
    domain = duplicate_domain() if duplicate_domain else copy.deepcopy(document.domain)
    return DesignDocument(
        identifier=document.identifier,
        name=document.name,
        domain=domain,
        settings=copy.deepcopy(document.settings),
        field_primitives=copy.deepcopy(document.field_primitives),
        field_primitive_visibility=dict(document.field_primitive_visibility),
        derived_meshes=copy.deepcopy(document.derived_meshes),
        revision=document.revision,
        archived=archived,
    )


def _document_snapshot(document: DesignDocument) -> dict[str, object]:
    domain: dict[str, object]
    if isinstance(document.domain, MeshDesignDomain):
        domain = {
            "kind": "mesh",
            "name": document.domain.name,
            "asset_path": (
                None
                if document.domain.asset_path is None
                else str(document.domain.asset_path)
            ),
        }
    elif isinstance(document.domain, AnalyticDesignDomain):
        domain = {
            "kind": "analytic",
            "name": document.domain.name,
            "primitive": _primitive_snapshot(document.domain.primitive),
        }
    else:  # pragma: no cover - domain aggregate already rejects unsupported values
        raise TypeError("unsupported design-domain implementation")
    return {
        "document_id": document.identifier,
        "name": document.name,
        "archived": document.archived,
        "revision": document.revision,
        "domain": domain,
        "settings": _json_value(document.settings, "settings"),
        "field_primitives": [
            {
                "primitive": _primitive_snapshot(primitive),
                "visible": document.field_primitive_visibility[identifier],
            }
            for identifier, primitive in document.field_primitives.items()
        ],
        "derived_meshes": [
            {
                "identifier": record.identifier,
                "asset_path": record.asset_path,
                "provenance": _json_value(record.provenance, "derived mesh provenance"),
            }
            for record in document.derived_meshes.values()
        ],
    }


def _mesh_domain(command: Mapping[str, object], *, default_name: str | None = None) -> MeshDesignDomain:
    source = Path(_non_empty_string(command.get("source_path"), "source_path")).expanduser()
    path = source.resolve()
    if not path.is_file() or path.suffix.lower() != ".stl":
        raise ValueError("source_path must reference an existing STL file")
    mesh = trimesh.load_mesh(path, force="mesh", process=True)
    if not isinstance(mesh, trimesh.Trimesh) or len(mesh.faces) == 0:
        raise ValueError("source_path must contain a triangular mesh")
    return MeshDesignDomain(mesh, _domain_name(command, default=default_name), path)


def _field_scene(value: object) -> tuple[dict[str, ImplicitPrimitive], dict[str, bool]]:
    if not isinstance(value, list):
        raise ValueError("field_primitives must be a JSON array")
    primitives: dict[str, ImplicitPrimitive] = {}
    visibility: dict[str, bool] = {}
    for item in value:
        entry = _mapping(item, "field primitive")
        primitive = _primitive(entry.get("primitive"))
        if primitive.identifier in primitives:
            raise ValueError("field primitive identifiers must be unique")
        primitives[primitive.identifier] = primitive
        visibility[primitive.identifier] = bool(entry.get("visible", True))
    return primitives, visibility


def _primitive(value: object) -> ImplicitPrimitive:
    entry = _mapping(value, "primitive")
    try:
        return ImplicitPrimitive(
            identifier=_non_empty_string(entry.get("identifier"), "primitive identifier"),
            name=_non_empty_string(entry.get("name"), "primitive name"),
            kind=str(entry["kind"]),
            center_mm=tuple(float(item) for item in _triple(entry.get("center_mm"), "center_mm")),
            rotation_euler_deg=tuple(
                float(item)
                for item in _triple(entry.get("rotation_euler_deg"), "rotation_euler_deg")
            ),
            radius_mm=float(entry["radius_mm"]),
            height_mm=float(entry["height_mm"]),
            size_mm=tuple(float(item) for item in _triple(entry.get("size_mm"), "size_mm")),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("invalid primitive") from exc


def _primitive_snapshot(primitive: ImplicitPrimitive) -> dict[str, object]:
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


def _create_document(
    workspace: DesignWorkspace,
    domain: MeshDesignDomain | AnalyticDesignDomain,
    name: str,
    command: Mapping[str, object],
) -> DesignDocument:
    identifier = command.get("document_id")
    return workspace.create_document(
        domain,
        name,
        activate=bool(command.get("activate", True)),
        identifier=(
            None if identifier is None else _non_empty_string(identifier, "document_id")
        ),
    )


def _document(workspace: DesignWorkspace, identifier: str) -> DesignDocument:
    document = workspace.documents.get(identifier) or workspace.archived_documents.get(identifier)
    if document is None:
        raise KeyError(f"unknown design: {identifier}")
    return document


def _derived_meshes(value: object) -> dict[str, PersistedDerivedMesh]:
    if not isinstance(value, list):
        raise ValueError("derived_meshes must be a JSON array")
    records: dict[str, PersistedDerivedMesh] = {}
    for entry in value:
        payload = _mapping(entry, "derived mesh")
        record = PersistedDerivedMesh(
            _non_empty_string(payload.get("identifier"), "derived mesh identifier"),
            _non_empty_string(payload.get("asset_path"), "derived mesh asset_path"),
            _json_object(payload.get("provenance"), "derived mesh provenance"),
        )
        if record.identifier in records:
            raise ValueError("derived mesh identifiers must be unique")
        records[record.identifier] = record
    return records


def _name(command: Mapping[str, object]) -> str:
    return _non_empty_string(command.get("name"), "document name")


def _domain_name(command: Mapping[str, object], *, default: str | None = None) -> str:
    value = command.get("domain_name", default)
    return _non_empty_string(value, "domain_name")


def _non_empty_string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


def _mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a JSON object")
    return value


def _json_object(value: object, name: str) -> dict[str, Any]:
    checked = _json_value(value, name)
    if not isinstance(checked, dict):
        raise ValueError(f"{name} must be a JSON object")
    return checked


def _json_value(value: object, name: str) -> Any:
    try:
        return json.loads(json.dumps(value, allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be JSON-safe") from exc


def _triple(value: object, name: str) -> tuple[object, object, object]:
    if not isinstance(value, list) or len(value) != 3:
        raise ValueError(f"{name} must contain exactly three values")
    return value[0], value[1], value[2]


__all__ = ["WorkspaceCommandService", "WorkspaceMutation", "workspace_snapshot"]
