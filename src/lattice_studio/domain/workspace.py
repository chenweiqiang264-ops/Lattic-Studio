"""Persistence-agnostic workspace aggregate and design state."""

from __future__ import annotations

import copy
import uuid
from dataclasses import dataclass, field
from typing import Any, Protocol, TypeAlias


JsonValue: TypeAlias = (
    None | bool | int | float | str | list["JsonValue"] | dict[str, "JsonValue"]
)


class DesignDomain(Protocol):
    """Authoritative geometry definition owned by one design document."""

    kind: str
    name: str

    def duplicate_definition(self) -> "DesignDomain": ...


class FieldPrimitive(Protocol):
    identifier: str


@dataclass(frozen=True)
class PersistedDerivedMesh:
    """A managed derived mesh reference and its reproducibility metadata."""

    identifier: str
    asset_path: str
    provenance: dict[str, JsonValue]

    def __post_init__(self) -> None:
        if not self.identifier.strip():
            raise ValueError("derived mesh identifier must not be empty")
        if not str(self.asset_path).strip():
            raise ValueError("derived mesh asset path must not be empty")
        object.__setattr__(self, "asset_path", str(self.asset_path))
        object.__setattr__(self, "provenance", copy.deepcopy(self.provenance))


@dataclass
class DesignRuntimeState:
    """Disposable computation and rendering state; never persisted."""

    domain_implicit_body: object | None = None
    domain_implicit_field: object | None = None
    shell_fusion_domain_field: object | None = None
    implicit_results: dict[str, object] = field(default_factory=dict)
    implicit_generation_results: dict[str, object] = field(default_factory=dict)
    backend_generation_handles: dict[str, object] = field(default_factory=dict)
    implicit_fields: dict[str, object] = field(default_factory=dict)
    raw_results: dict[str, object] = field(default_factory=dict)
    repaired_results: dict[str, object] = field(default_factory=dict)
    results: dict[str, object] = field(default_factory=dict)
    stl_reconstruction_results: dict[str, object] = field(default_factory=dict)
    prepared_custom_cell: object | None = None
    ui_transient: dict[str, object] = field(default_factory=dict)
    renderer_cache: dict[str, object] = field(default_factory=dict)

    def clear_generated(self) -> None:
        self.invalidate_generated()

    def invalidate_generated(self, keys: set[str] | None = None) -> None:
        mappings = (
            self.implicit_results,
            self.implicit_generation_results,
            self.backend_generation_handles,
            self.implicit_fields,
            self.raw_results,
            self.repaired_results,
            self.results,
            self.stl_reconstruction_results,
        )
        if keys is None:
            self.domain_implicit_body = None
            self.domain_implicit_field = None
            self.shell_fusion_domain_field = None
            for mapping in mappings:
                mapping.clear()
        else:
            for mapping in mappings:
                for key in keys:
                    mapping.pop(key, None)
        self.renderer_cache.clear()


@dataclass
class DesignDocument:
    """One isolated design and its authoritative editable definitions."""

    identifier: str
    name: str
    domain: DesignDomain
    settings: dict[str, JsonValue] = field(default_factory=dict)
    field_primitives: dict[str, FieldPrimitive] = field(default_factory=dict)
    field_primitive_visibility: dict[str, bool] = field(default_factory=dict)
    derived_meshes: dict[str, PersistedDerivedMesh] = field(default_factory=dict)
    runtime: DesignRuntimeState = field(default_factory=DesignRuntimeState, repr=False)
    revision: int = 0
    archived: bool = False

    def __post_init__(self) -> None:
        if not self.identifier.strip() or not self.name.strip():
            raise ValueError("design identifier and name must not be empty")
        if not hasattr(self.domain, "kind") or not hasattr(self.domain, "name"):
            raise TypeError("design document requires a supported design domain")
        if int(self.revision) < 0:
            raise ValueError("design revision must not be negative")
        self.revision = int(self.revision)
        self.settings = copy.deepcopy(self.settings)
        self.field_primitives = dict(self.field_primitives)
        self.field_primitive_visibility = {
            key: bool(value) for key, value in self.field_primitive_visibility.items()
        }
        self.derived_meshes = dict(self.derived_meshes)
        if set(self.field_primitives) != set(self.field_primitive_visibility):
            raise ValueError("field primitive visibility must match field primitives")
        for identifier, primitive in self.field_primitives.items():
            if primitive.identifier != identifier:
                raise ValueError("field primitive dictionary key must match identifier")

    def touch(self) -> int:
        self.revision += 1
        return self.revision

    def clear_generated(self) -> None:
        self.derived_meshes.clear()
        self.runtime.clear_generated()

    def invalidate_generated_results(self, keys: set[str]) -> None:
        if not keys:
            raise ValueError("at least one generated-result key is required")
        self.runtime.invalidate_generated(keys)
        for key in keys:
            self.derived_meshes.pop(key, None)
        self.touch()

    def replace_domain(self, domain: DesignDomain) -> None:
        if not hasattr(domain, "kind") or not hasattr(domain, "name"):
            raise TypeError("design document requires a supported design domain")
        self.domain = domain
        self.clear_generated()
        self.touch()

    def upsert_field_primitive(
        self, primitive: FieldPrimitive, *, visible: bool = True
    ) -> None:
        if not getattr(primitive, "identifier", ""):
            raise TypeError("field object requires a non-empty identifier")
        self.field_primitives[primitive.identifier] = primitive
        self.field_primitive_visibility[primitive.identifier] = bool(visible)
        self.clear_generated()
        self.touch()

    def remove_field_primitive(self, identifier: str) -> None:
        if identifier not in self.field_primitives:
            raise KeyError(f"unknown field object: {identifier}")
        del self.field_primitives[identifier]
        del self.field_primitive_visibility[identifier]
        self.clear_generated()
        self.touch()

    def set_field_primitive_visibility(self, identifier: str, visible: bool) -> None:
        if identifier not in self.field_primitives:
            raise KeyError(f"unknown field object: {identifier}")
        self.field_primitive_visibility[identifier] = bool(visible)

    def replace_field_object_scene(
        self,
        primitives: dict[str, FieldPrimitive],
        visibility: dict[str, bool],
    ) -> bool:
        updated_visibility = {key: bool(value) for key, value in visibility.items()}
        if set(primitives) != set(updated_visibility):
            raise ValueError("field primitive visibility must match field primitives")
        for identifier, primitive in primitives.items():
            if primitive.identifier != identifier:
                raise ValueError("field primitive dictionary key must match identifier")
        if self.field_primitives == primitives and self.field_primitive_visibility == updated_visibility:
            return False
        self.field_primitives = dict(primitives)
        self.field_primitive_visibility = updated_visibility
        self.touch()
        return True

    def set_setting(self, key: str, value: JsonValue) -> None:
        if not key.strip():
            raise ValueError("setting key must not be empty")
        self.settings[key] = copy.deepcopy(value)
        self.clear_generated()
        self.touch()

    def register_derived_mesh(self, record: PersistedDerivedMesh) -> None:
        if not isinstance(record, PersistedDerivedMesh):
            raise TypeError("derived result must be a PersistedDerivedMesh")
        self.derived_meshes[record.identifier] = record

    def duplicate(self, identifier: str, name: str) -> "DesignDocument":
        duplicate_domain = getattr(self.domain, "duplicate_definition", None)
        domain = duplicate_domain() if duplicate_domain else copy.deepcopy(self.domain)
        return DesignDocument(
            identifier=identifier,
            name=name,
            domain=domain,
            settings=copy.deepcopy(self.settings),
            field_primitives=copy.deepcopy(self.field_primitives),
            field_primitive_visibility=dict(self.field_primitive_visibility),
        )


@dataclass
class DesignWorkspace:
    """Aggregate root for active and recoverably archived designs."""

    documents: dict[str, DesignDocument] = field(default_factory=dict)
    archived_documents: dict[str, DesignDocument] = field(default_factory=dict)
    active_design_id: str | None = None
    dirty: bool = False

    def __post_init__(self) -> None:
        self.documents = dict(self.documents)
        self.archived_documents = dict(self.archived_documents)
        if set(self.documents) & set(self.archived_documents):
            raise ValueError("active and archived documents must not overlap")
        if any(document.archived for document in self.documents.values()):
            raise ValueError("active design documents must not be archived")
        if any(not document.archived for document in self.archived_documents.values()):
            raise ValueError("archived design documents must be marked archived")
        if self.active_design_id is not None and self.active_design_id not in self.documents:
            raise ValueError("active design must belong to active documents")
        if self.active_design_id is None and self.documents:
            self.active_design_id = next(iter(self.documents))

    @property
    def active_document(self) -> DesignDocument | None:
        return self.documents.get(self.active_design_id) if self.active_design_id else None

    @property
    def is_empty(self) -> bool:
        return not self.documents

    def new_identifier(self) -> str:
        identifier = uuid.uuid4().hex
        while identifier in self.documents or identifier in self.archived_documents:
            identifier = uuid.uuid4().hex
        return identifier

    def add_document(self, document: DesignDocument, *, activate: bool = True) -> None:
        if document.identifier in self.documents or document.identifier in self.archived_documents:
            raise ValueError("design identifier already exists in this workspace")
        if document.archived:
            raise ValueError("cannot add an archived document as active")
        self.documents[document.identifier] = document
        if activate or self.active_design_id is None:
            self.active_design_id = document.identifier
        self.dirty = True

    def create_document(
        self,
        domain: DesignDomain,
        name: str,
        *,
        activate: bool = True,
        identifier: str | None = None,
    ) -> DesignDocument:
        document = DesignDocument(identifier or self.new_identifier(), name, domain)
        self.add_document(document, activate=activate)
        return document

    def activate(self, identifier: str) -> DesignDocument:
        if identifier not in self.documents:
            raise KeyError(f"unknown active design: {identifier}")
        if self.active_design_id != identifier:
            self.active_design_id = identifier
            self.dirty = True
        return self.documents[identifier]

    def duplicate_document(
        self,
        identifier: str,
        name: str,
        *,
        new_identifier: str | None = None,
    ) -> DesignDocument:
        source = self.documents.get(identifier)
        if source is None:
            raise KeyError(f"unknown active design: {identifier}")
        duplicate = source.duplicate(new_identifier or self.new_identifier(), name)
        self.add_document(duplicate)
        return duplicate

    def rename_document(self, identifier: str, name: str) -> None:
        document = self.documents.get(identifier)
        if document is None:
            raise KeyError(f"unknown active design: {identifier}")
        name = name.strip()
        if not name:
            raise ValueError("design name must not be empty")
        if document.name != name:
            document.name = name
            self.dirty = True

    def archive_document(self, identifier: str) -> DesignDocument:
        document = self.documents.pop(identifier, None)
        if document is None:
            raise KeyError(f"unknown active design: {identifier}")
        document.archived = True
        self.archived_documents[identifier] = document
        if self.active_design_id == identifier:
            self.active_design_id = next(iter(self.documents), None)
        self.dirty = True
        return document

    def restore_document(self, identifier: str, *, activate: bool = True) -> DesignDocument:
        document = self.archived_documents.pop(identifier, None)
        if document is None:
            raise KeyError(f"unknown archived design: {identifier}")
        document.archived = False
        self.documents[identifier] = document
        if activate or self.active_design_id is None:
            self.active_design_id = identifier
        self.dirty = True
        return document

    def permanently_remove_archived_document(self, identifier: str) -> None:
        if identifier not in self.archived_documents:
            raise KeyError(f"unknown archived design: {identifier}")
        del self.archived_documents[identifier]
        self.dirty = True

    def mark_active_changed(self) -> int:
        if self.active_document is None:
            raise RuntimeError("an empty workspace has no active design")
        self.dirty = True
        return self.active_document.touch()


__all__ = [
    "DesignDocument",
    "DesignDomain",
    "DesignRuntimeState",
    "DesignWorkspace",
    "FieldPrimitive",
    "JsonValue",
    "PersistedDerivedMesh",
]
