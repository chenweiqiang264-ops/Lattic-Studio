"""Contract tests for the loopback frontend-to-backend transport."""

from __future__ import annotations

import pytest

from lattice_studio.application.backend import LocalBackend
from lattice_studio.presentation.http.local_backend import (
    BackendRequestError,
    LocalBackendClient,
    LocalBackendServer,
)
from lattice_studio.presentation.http.workspace_results import (
    read_workspace_projection,
)


def test_local_backend_reports_supported_workspace_operations() -> None:
    health = LocalBackend().health()

    assert health.version == "0.1.2"
    assert health.operations == (
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


def test_loopback_client_owns_workspace_lifecycle(tmp_path) -> None:
    manifest_path = tmp_path / "workspace.json"

    with LocalBackendServer() as server:
        client = LocalBackendClient(server.base_url)

        assert client.health().version == "0.1.2"
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


def test_loopback_workspace_commands_are_atomic_and_persisted(tmp_path) -> None:
    design_domain = tmp_path / "domain.stl"
    manifest_path = tmp_path / "workspace" / "workspace.json"
    trimesh = pytest.importorskip("trimesh")
    trimesh.creation.box(extents=(4.0, 4.0, 4.0)).export(design_domain)

    with LocalBackendServer() as server:
        client = LocalBackendClient(server.base_url)
        session = client.create_workspace()
        snapshot = client.apply_workspace_commands(
            session.identifier,
            [
                {
                    "kind": "document.create.mesh",
                    "document_id": "design-1",
                    "name": "Fixture",
                    "domain_name": "Fixture domain",
                    "source_path": str(design_domain),
                },
            ],
        )
        assert snapshot["document_count"] == 1
        document = snapshot["documents"][0]
        document_id = document["document_id"]
        assert document_id == "design-1"
        assert document["domain"]["kind"] == "mesh"

        updated = client.apply_workspace_commands(
            session.identifier,
            [
                {
                    "kind": "document.setting.set",
                    "document_id": document_id,
                    "key": "ui_control_state",
                    "value": {"sample": 7},
                },
                {
                    "kind": "document.field.upsert",
                    "document_id": document_id,
                    "primitive": {
                        "identifier": "field-1",
                        "name": "Driver",
                        "kind": "sphere",
                        "center_mm": [0.0, 0.0, 0.0],
                        "rotation_euler_deg": [0.0, 0.0, 0.0],
                        "radius_mm": 1.0,
                        "height_mm": 2.0,
                        "size_mm": [2.0, 2.0, 2.0],
                    },
                },
            ],
        )
        assert updated["dirty"] is True
        assert updated["documents"][0]["settings"] == {"ui_control_state": {"sample": 7}}
        assert len(updated["documents"][0]["field_primitives"]) == 1

        with pytest.raises(BackendRequestError, match="unknown active design"):
            client.apply_workspace_commands(
                session.identifier,
                [
                    {"kind": "document.rename", "document_id": document_id, "name": "Changed"},
                    {"kind": "document.rename", "document_id": "missing", "name": "Broken"},
                ],
            )
        assert client.get_workspace_snapshot(session.identifier)["documents"][0]["name"] == "Fixture"

        client.save_workspace(session.identifier, str(manifest_path))
        assert (manifest_path.parent / "assets" / "design_domains" / f"{document_id}.stl").is_file()
        loaded = client.load_workspace(str(manifest_path))
        restored = client.get_workspace_snapshot(loaded.identifier)
        assert restored["documents"][0]["settings"] == {"ui_control_state": {"sample": 7}}
        assert restored["documents"][0]["field_primitives"][0]["primitive"]["identifier"] == "field-1"
        projection = read_workspace_projection(restored, manifest_path.parent)
        projected = projection.active_document
        assert projected is not None
        assert projected.identifier == document_id
        assert projected.domain.kind == "mesh"
        assert projected.field_primitives["field-1"].name == "Driver"
