"""Contract tests for the loopback frontend-to-backend transport."""

from __future__ import annotations

import pytest

from lattice_studio.application.backend import LocalBackend
from lattice_studio.presentation.http.local_backend import (
    BackendRequestError,
    LocalBackendClient,
    LocalBackendServer,
)


def test_local_backend_reports_supported_workspace_operations() -> None:
    health = LocalBackend().health()

    assert health.version == "0.1.1"
    assert health.operations == (
        "workspace.create",
        "workspace.get",
        "workspace.load",
        "workspace.save",
    )


def test_loopback_client_owns_workspace_lifecycle(tmp_path) -> None:
    manifest_path = tmp_path / "workspace.json"

    with LocalBackendServer() as server:
        client = LocalBackendClient(server.base_url)

        assert client.health().version == "0.1.1"
        created = client.create_workspace()
        assert created.document_count == 0
        assert created.dirty is False

        saved = client.save_workspace(created.identifier, str(manifest_path))
        assert manifest_path.is_file()
        assert saved.identifier == created.identifier
        assert saved.dirty is False

        loaded = client.load_workspace(str(manifest_path))
        assert loaded.identifier != created.identifier
        assert loaded.document_count == 0
        assert client.get_workspace(loaded.identifier) == loaded


def test_loopback_client_reports_unknown_workspace() -> None:
    with LocalBackendServer() as server:
        client = LocalBackendClient(server.base_url)

        with pytest.raises(BackendRequestError, match="unknown workspace") as error:
            client.get_workspace("missing")

    assert error.value.status_code == 404
