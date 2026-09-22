"""Supported scripting interface for Lattice Studio.

The interface is intentionally small while application use cases are migrated
out of the historical workbench. New public operations belong here instead of
being exported from Qt modules.
"""

from __future__ import annotations

from lattice_studio.application.generation import LatticeGenerationService
from lattice_studio.application.workspace import WorkspaceService


class LatticeStudio:
    """Facade for headless lattice design workflows."""

    def __init__(self) -> None:
        self.workspace = WorkspaceService()
        self.generation = LatticeGenerationService()
        self._ready = True

    @property
    def ready(self) -> bool:
        """Return whether the facade is ready to accept workflows."""

        return self._ready
