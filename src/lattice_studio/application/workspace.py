"""Application commands for persistent design workspaces."""

from __future__ import annotations

from pathlib import Path

import trimesh

from lattice_studio.engine.implicit.primitives import ImplicitPrimitive
from lattice_studio.domain.workspace import DesignDocument, PersistedDerivedMesh
from lattice_studio.domain.workspace import DesignWorkspace
from lattice_studio.engine.design_domains import (
    AnalyticDesignDomain,
    DesignDomainValue,
    MeshDesignDomain,
)
from lattice_studio.infrastructure.workspace_persistence import load_workspace, save_workspace
from lattice_studio.infrastructure.workspace_persistence import package_stl_asset


class ManagedDesignWorkspace(DesignWorkspace):
    """Workspace use-case facade with managed geometry and persistence actions."""

    def create_mesh_document(
        self,
        mesh: trimesh.Trimesh,
        name: str,
        *,
        asset_path: Path | None = None,
        activate: bool = True,
    ) -> DesignDocument:
        return self.create_document(
            MeshDesignDomain(mesh, name, asset_path), name, activate=activate
        )

    def create_analytic_document(
        self,
        primitive: ImplicitPrimitive,
        name: str,
        *,
        activate: bool = True,
    ) -> DesignDocument:
        return self.create_document(
            AnalyticDesignDomain(primitive, name), name, activate=activate
        )

    def package_mesh_domain_asset(
        self,
        identifier: str,
        source_path: Path,
        workspace_root: Path,
    ) -> Path:
        document = self.documents.get(identifier) or self.archived_documents.get(identifier)
        if document is None:
            raise KeyError(f"unknown design document: {identifier}")
        if not isinstance(document.domain, MeshDesignDomain):
            raise TypeError("only mesh design domains have STL source assets")
        target = package_stl_asset(
            source_path, workspace_root, "design_domains", identifier
        )
        document.domain.asset_path = target.relative_to(workspace_root)
        self.dirty = True
        return target

    package_stl_asset = staticmethod(package_stl_asset)

    def save(self, manifest_path: Path) -> None:
        save_workspace(self, manifest_path)

    @classmethod
    def load(cls, manifest_path: Path) -> "ManagedDesignWorkspace":
        restored = load_workspace(manifest_path)
        return cls(
            restored.documents,
            restored.archived_documents,
            restored.active_design_id,
            restored.dirty,
        )


class WorkspaceService:
    """Own workspace lifecycle without coupling it to Qt or file dialogs."""

    def create(self) -> ManagedDesignWorkspace:
        return ManagedDesignWorkspace()

    def save(self, workspace: DesignWorkspace, manifest_path: Path) -> None:
        save_workspace(workspace, manifest_path)

    def load(self, manifest_path: Path) -> DesignWorkspace:
        return load_workspace(manifest_path)

    def package_mesh_asset(
        self,
        source_path: Path,
        workspace_root: Path,
        *,
        category: str,
        identifier: str,
    ) -> Path:
        """Copy a source STL into the managed workspace asset area."""

        return package_stl_asset(source_path, workspace_root, category, identifier)


__all__ = [
    "AnalyticDesignDomain",
    "DesignDomainValue",
    "ManagedDesignWorkspace",
    "MeshDesignDomain",
    "PersistedDerivedMesh",
    "WorkspaceService",
]
