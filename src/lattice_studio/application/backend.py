"""Application-facing local backend for non-Qt clients.

The backend owns short-lived workspace sessions behind a small interface. HTTP,
Qt, and command-line code adapt to this module instead of retaining their own
workspace registries.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

from lattice_studio.application.workspace import WorkspaceService
from lattice_studio.application.backend_workflows import ImplicitWorkflowTasks
from lattice_studio.application.workspace_commands import (
    WorkspaceCommandService,
    workspace_snapshot,
)
from lattice_studio.application.tasks import (
    BackendTaskService,
    TaskOutcome,
    TaskSnapshot,
)
from lattice_studio.domain.workspace import DesignWorkspace


SUPPORTED_OPERATIONS = (
    "task.submit",
    "task.get",
    "task.cancel",
    "task.artifact.get",
    "workspace.create",
    "workspace.get",
    "workspace.snapshot",
    "workspace.commands",
    "workspace.load",
    "workspace.save",
)


@dataclass(frozen=True)
class BackendHealth:
    """Observable readiness data for a client connecting to the backend."""

    version: str
    operations: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "status": "ready",
            "version": self.version,
            "operations": list(self.operations),
        }

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> "BackendHealth":
        return cls(
            version=str(value["version"]),
            operations=tuple(str(item) for item in value["operations"]),
        )


@dataclass(frozen=True)
class WorkspaceSession:
    """Serializable summary of one backend-owned design workspace."""

    identifier: str
    document_count: int
    archived_document_count: int
    active_design_id: str | None
    dirty: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "workspace_id": self.identifier,
            "document_count": self.document_count,
            "archived_document_count": self.archived_document_count,
            "active_design_id": self.active_design_id,
            "dirty": self.dirty,
        }

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> "WorkspaceSession":
        active_design_id = value.get("active_design_id")
        return cls(
            identifier=str(value["workspace_id"]),
            document_count=int(value["document_count"]),
            archived_document_count=int(value["archived_document_count"]),
            active_design_id=None if active_design_id is None else str(active_design_id),
            dirty=bool(value["dirty"]),
        )


class UnknownWorkspace(KeyError):
    """Raised when a caller references a session absent from this backend."""


class LocalBackend:
    """Own backend sessions while delegating workspace behavior to application use cases.

    Sessions live only for the lifetime of this process. Clients persist work by
    explicitly calling :meth:`save_workspace`; no transport or UI state leaks
    into the saved workspace model.
    """

    def __init__(
        self,
        workspace_service: WorkspaceService | None = None,
        task_service: BackendTaskService | None = None,
    ) -> None:
        self._workspace_service = workspace_service or WorkspaceService()
        self._workspaces: dict[str, DesignWorkspace] = {}
        self._workspace_commands = WorkspaceCommandService()
        self._implicit_workflow_tasks = ImplicitWorkflowTasks()
        self._task_service = task_service or BackendTaskService(
            {
                "workspace.save": self._save_workspace_task,
                **self._implicit_workflow_tasks.handlers(),
            }
        )

    def health(self) -> BackendHealth:
        """Report the stable operations implemented by this backend version."""

        return BackendHealth(version="0.1.2", operations=SUPPORTED_OPERATIONS)

    def create_workspace(self) -> WorkspaceSession:
        """Create and retain an empty design workspace."""

        return self._register(self._workspace_service.create())

    def get_workspace(self, identifier: str) -> WorkspaceSession:
        """Return a summary for an existing backend-owned workspace session."""

        return self._summary(identifier, self._workspace(identifier))

    def workspace_snapshot(self, identifier: str) -> dict[str, object]:
        """Return a transport-safe view of editable definitions in a session."""

        return workspace_snapshot(identifier, self._workspace(identifier))

    def apply_workspace_commands(
        self,
        identifier: str,
        commands: list[dict[str, object]],
    ) -> dict[str, object]:
        """Commit an all-or-nothing batch of fixed workspace commands."""

        workspace = self._workspace(identifier)
        updated, mutations = self._workspace_commands.apply(workspace, commands)
        self._workspaces[identifier] = updated
        return workspace_snapshot(identifier, updated, mutations=mutations)

    def load_workspace(self, manifest_path: str | Path) -> WorkspaceSession:
        """Load a persisted workspace into a new backend session."""

        workspace = self._workspace_service.load(self._manifest_path(manifest_path))
        return self._register(workspace)

    def save_workspace(self, identifier: str, manifest_path: str | Path) -> WorkspaceSession:
        """Persist an existing session and return its post-save summary."""

        workspace = self._workspace(identifier)
        path = self._manifest_path(manifest_path)
        self._package_workspace_assets(workspace, path.parent)
        self._workspace_service.save(workspace, path)
        return self._summary(identifier, workspace)

    def submit_task(
        self,
        kind: str,
        payload: dict[str, object],
    ) -> TaskSnapshot:
        """Queue a backend-owned operation from the explicit task registry."""

        return self._task_service.submit(kind, payload)

    def get_task(self, identifier: str) -> TaskSnapshot:
        return self._task_service.get(identifier)

    def cancel_task(self, identifier: str) -> TaskSnapshot:
        return self._task_service.cancel(identifier)

    def task_artifact_path(self, identifier: str):
        """Return artifact metadata and a process-local file path for transport."""

        return self._task_service.artifact_path(identifier)

    def close(self) -> None:
        """Release task workers when the backend server shuts down."""

        self._task_service.close()

    def _save_workspace_task(
        self,
        payload: dict[str, object],
        context,
    ) -> TaskOutcome:
        context.report("saving workspace", 0.1)
        session = self.save_workspace(
            str(payload["workspace_id"]),
            str(payload["manifest_path"]),
        )
        context.report("workspace saved", 1.0)
        return TaskOutcome(result={"workspace": session.to_dict()})

    def _register(self, workspace: DesignWorkspace) -> WorkspaceSession:
        identifier = uuid4().hex
        self._workspaces[identifier] = workspace
        return self._summary(identifier, workspace)

    def _workspace(self, identifier: str) -> DesignWorkspace:
        try:
            return self._workspaces[identifier]
        except KeyError as exc:
            raise UnknownWorkspace(identifier) from exc

    def _package_workspace_assets(self, workspace: DesignWorkspace, root: Path) -> None:
        """Make imported document assets portable before writing a manifest."""

        from lattice_studio.engine.design_domains import MeshDesignDomain

        documents = tuple(workspace.documents.values()) + tuple(
            workspace.archived_documents.values()
        )
        for document in documents:
            domain = document.domain
            if not isinstance(domain, MeshDesignDomain):
                continue
            if domain.asset_path is None:
                raise ValueError("mesh design domain must have a source STL path")
            source = Path(domain.asset_path)
            if not source.is_absolute():
                source = root / source
            target = self._workspace_service.package_mesh_asset(
                source,
                root,
                category="design_domains",
                identifier=document.identifier,
            )
            domain.asset_path = target.relative_to(root)

            custom_source = document.settings.get("custom_unit_cell_source_path")
            if isinstance(custom_source, str) and custom_source.strip():
                custom_path = Path(custom_source)
                if not custom_path.is_absolute():
                    custom_path = root / custom_path
                custom_target = self._workspace_service.package_mesh_asset(
                    custom_path,
                    root,
                    category="custom_cells",
                    identifier=document.identifier,
                )
                document.settings["custom_unit_cell_source_path"] = str(
                    custom_target.relative_to(root)
                )

    @staticmethod
    def _manifest_path(value: str | Path) -> Path:
        path = Path(value).expanduser()
        if not str(path):
            raise ValueError("workspace manifest path must not be empty")
        return path.resolve()

    @staticmethod
    def _summary(identifier: str, workspace: DesignWorkspace) -> WorkspaceSession:
        return WorkspaceSession(
            identifier=identifier,
            document_count=len(workspace.documents),
            archived_document_count=len(workspace.archived_documents),
            active_design_id=workspace.active_design_id,
            dirty=workspace.dirty,
        )


__all__ = [
    "BackendHealth",
    "LocalBackend",
    "SUPPORTED_OPERATIONS",
    "UnknownWorkspace",
    "WorkspaceSession",
]
