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
from lattice_studio.domain.workspace import DesignWorkspace


SUPPORTED_OPERATIONS = (
    "workspace.create",
    "workspace.get",
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

    def __init__(self, workspace_service: WorkspaceService | None = None) -> None:
        self._workspace_service = workspace_service or WorkspaceService()
        self._workspaces: dict[str, DesignWorkspace] = {}

    def health(self) -> BackendHealth:
        """Report the stable operations implemented by this backend version."""

        return BackendHealth(version="0.1.1", operations=SUPPORTED_OPERATIONS)

    def create_workspace(self) -> WorkspaceSession:
        """Create and retain an empty design workspace."""

        return self._register(self._workspace_service.create())

    def get_workspace(self, identifier: str) -> WorkspaceSession:
        """Return a summary for an existing backend-owned workspace session."""

        return self._summary(identifier, self._workspace(identifier))

    def load_workspace(self, manifest_path: str | Path) -> WorkspaceSession:
        """Load a persisted workspace into a new backend session."""

        workspace = self._workspace_service.load(self._manifest_path(manifest_path))
        return self._register(workspace)

    def save_workspace(self, identifier: str, manifest_path: str | Path) -> WorkspaceSession:
        """Persist an existing session and return its post-save summary."""

        workspace = self._workspace(identifier)
        self._workspace_service.save(workspace, self._manifest_path(manifest_path))
        return self._summary(identifier, workspace)

    def _register(self, workspace: DesignWorkspace) -> WorkspaceSession:
        identifier = uuid4().hex
        self._workspaces[identifier] = workspace
        return self._summary(identifier, workspace)

    def _workspace(self, identifier: str) -> DesignWorkspace:
        try:
            return self._workspaces[identifier]
        except KeyError as exc:
            raise UnknownWorkspace(identifier) from exc

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
