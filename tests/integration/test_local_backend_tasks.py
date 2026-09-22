"""Task-state and artifact contracts for the local backend boundary."""

from __future__ import annotations

import json
from io import BytesIO
import threading
import time

import pytest
import numpy as np
import trimesh

from lattice_studio.application.backend import LocalBackend
from lattice_studio.application.tasks import BackendTaskService, TaskOutcome
from lattice_studio.presentation.http.local_backend import (
    BackendRequestError,
    LocalBackendClient,
    LocalBackendServer,
)
from lattice_studio.presentation.http.task_results import (
    batch_display_field_artifact_ids,
    display_field_artifact_id,
    read_generation_batch,
    read_generation_result,
)
from lattice_studio.presentation.http.backend_process import LocalBackendProcess


def _wait_for_terminal(
    client: LocalBackendClient,
    identifier: str,
    *,
    timeout_seconds: float = 5.0,
):
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        snapshot = client.get_task(identifier)
        if snapshot.state in {"succeeded", "failed", "cancelled"}:
            return snapshot
        time.sleep(0.01)
    raise AssertionError("task did not reach a terminal state")


def test_loopback_task_returns_serializable_result_and_artifact() -> None:
    def create_artifact(payload, context):
        context.report("writing artifact", 0.5)
        output = context.work_dir / "result.json"
        output.write_text(json.dumps({"value": payload["value"]}), encoding="utf-8")
        return TaskOutcome(result={"accepted": True}, artifacts=(output,))

    backend = LocalBackend(
        task_service=BackendTaskService({"test.artifact": create_artifact})
    )
    with LocalBackendServer(backend) as server:
        client = LocalBackendClient(server.base_url)
        queued = client.submit_task("test.artifact", {"value": 42})
        assert queued.kind == "test.artifact"
        assert queued.state in {"queued", "running"}

        completed = _wait_for_terminal(client, queued.identifier)
        assert completed.state == "succeeded"
        assert completed.progress == 1.0
        assert completed.result == {"accepted": True}
        assert len(completed.artifacts) == 1
        assert client.download_artifact(completed.artifacts[0].identifier) == b'{"value": 42}'


def test_tasks_capture_failure_and_reject_unregistered_kinds() -> None:
    def fail(_payload, _context):
        raise RuntimeError("expected failure")

    backend = LocalBackend(task_service=BackendTaskService({"test.fail": fail}))
    with LocalBackendServer(backend) as server:
        client = LocalBackendClient(server.base_url)
        failed = _wait_for_terminal(client, client.submit_task("test.fail", {}).identifier)
        assert failed.state == "failed"
        assert failed.error == "RuntimeError: expected failure"

        with pytest.raises(BackendRequestError, match="unsupported task kind") as error:
            client.submit_task("__import__", {})

    assert error.value.status_code == 400


def test_tasks_can_be_cancelled_while_running() -> None:
    started = threading.Event()

    def wait_for_cancellation(_payload, context):
        started.set()
        while not context.cancelled:
            time.sleep(0.005)
        context.raise_if_cancelled()
        raise AssertionError("unreachable")

    backend = LocalBackend(
        task_service=BackendTaskService({"test.cancel": wait_for_cancellation})
    )
    with LocalBackendServer(backend) as server:
        client = LocalBackendClient(server.base_url)
        task = client.submit_task("test.cancel", {})
        assert started.wait(timeout=2.0)
        requested = client.cancel_task(task.identifier)
        assert requested.state in {"running", "cancelled"}

        cancelled = _wait_for_terminal(client, task.identifier)
        assert cancelled.state == "cancelled"
        assert cancelled.error is None


def test_owned_backend_process_exposes_loopback_health_and_stops() -> None:
    process = LocalBackendProcess(startup_timeout_seconds=15.0)
    client = process.start()

    assert client.health().version == "0.1.2"
    assert process.is_running is True

    process.stop()

    assert process.is_running is False


def test_backend_runs_tpms_generation_then_reconstructs_stl(tmp_path) -> None:
    design_domain = tmp_path / "design-domain.stl"
    trimesh.creation.box(extents=(6.0, 6.0, 6.0)).export(design_domain)

    with LocalBackendServer() as server:
        client = LocalBackendClient(server.base_url, timeout_seconds=30.0)
        generation = client.submit_task(
            "tpms.generate",
            {
                "mesh_path": str(design_domain),
                "parameters": {
                    "kind": "G",
                    "cell_size_mm": [4.0, 4.0, 4.0],
                    "wall_thickness_mm": 0.5,
                },
                "sampling": {
                    "target_voxels": 100_000,
                    "min_samples_per_cell": 8.0,
                    "min_samples_per_wall": 2.0,
                },
                "display_voxel_size_mm": 0.75,
            },
        )
        generated = _wait_for_terminal(
            client,
            generation.identifier,
            timeout_seconds=60.0,
        )
        assert generated.state == "succeeded", generated.error
        assert generated.result is not None
        assert generated.result["kind"] == "G"
        assert len(generated.artifacts) == 1
        display_field = client.download_artifact(display_field_artifact_id(generated))
        frontend_result = read_generation_result(generated, display_field)
        assert frontend_result.generation_id == generated.result["generation_id"]
        assert frontend_result.kind == "G"
        assert frontend_result.cell_map is not None
        assert frontend_result.minimum_feature_mm is not None
        assert not hasattr(frontend_result, "body")
        field = np.load(BytesIO(display_field))
        assert field["values"].ndim == 3

        reconstruction = client.submit_task(
            "stl.reconstruct",
            {
                "generation_id": generated.result["generation_id"],
                "tolerance_mm": 0.9,
            },
        )
        reconstructed = _wait_for_terminal(
            client,
            reconstruction.identifier,
            timeout_seconds=60.0,
        )
        assert reconstructed.state == "succeeded", reconstructed.error
        assert reconstructed.result is not None
        assert reconstructed.result["triangle_count"] > 0
        assert len(client.download_artifact(reconstructed.artifacts[0].identifier)) > 84


def test_backend_runs_shell_and_refines_its_display_field(tmp_path) -> None:
    design_domain = tmp_path / "design-domain.stl"
    trimesh.creation.box(extents=(6.0, 6.0, 6.0)).export(design_domain)
    sampling = {
        "target_voxels": 100_000,
        "min_samples_per_cell": 8.0,
        "min_samples_per_wall": 2.0,
    }

    with LocalBackendServer() as server:
        client = LocalBackendClient(server.base_url, timeout_seconds=30.0)
        shell = _wait_for_terminal(
            client,
            client.submit_task(
                "shell.generate",
                {
                    "mesh_path": str(design_domain),
                    "thickness_mm": 0.5,
                    "sampling": sampling,
                    "display_voxel_size_mm": 0.75,
                },
            ).identifier,
            timeout_seconds=60.0,
        )
        assert shell.state == "succeeded", shell.error
        shell_result = read_generation_result(
            shell,
            client.download_artifact(display_field_artifact_id(shell)),
        )
        assert shell_result.kind == "Shell"

        refined = _wait_for_terminal(
            client,
            client.submit_task(
                "display.refine",
                {
                    "generation_id": shell_result.generation_id,
                    "quality": "High",
                    "display_memory_budget_mb": 16.0,
                },
            ).identifier,
            timeout_seconds=60.0,
        )
        assert refined.state == "succeeded", refined.error
        assert refined.result is not None
        assert refined.result["generation_id"] == shell_result.generation_id
        assert len(refined.artifacts) == 1
        assert (
            np.load(BytesIO(client.download_artifact(refined.artifacts[0].identifier)))[
                "values"
            ].ndim
            == 3
        )


def test_backend_precise_render_publishes_png_artifact(tmp_path) -> None:
    design_domain = tmp_path / "design-domain.stl"
    trimesh.creation.box(extents=(6.0, 6.0, 6.0)).export(design_domain)

    with LocalBackendServer() as server:
        client = LocalBackendClient(server.base_url, timeout_seconds=30.0)
        generated = _wait_for_terminal(
            client,
            client.submit_task(
                "tpms.generate",
                {
                    "mesh_path": str(design_domain),
                    "parameters": {
                        "kind": "G",
                        "cell_size_mm": [4.0, 4.0, 4.0],
                        "wall_thickness_mm": 0.5,
                    },
                    "sampling": {
                        "target_voxels": 100_000,
                        "min_samples_per_cell": 8.0,
                        "min_samples_per_wall": 2.0,
                    },
                    "display_voxel_size_mm": 0.75,
                },
            ).identifier,
            timeout_seconds=60.0,
        )
        assert generated.state == "succeeded", generated.error
        render = _wait_for_terminal(
            client,
            client.submit_task(
                "render.precise",
                {
                    "generation_id": generated.result["generation_id"],
                    "camera": {
                        "position": [12.0, 12.0, 12.0],
                        "focal_point": [0.0, 0.0, 0.0],
                        "view_up": [0.0, 0.0, 1.0],
                        "parallel_projection": False,
                        "parallel_scale": 8.0,
                        "view_angle_deg": 30.0,
                    },
                    "settings": {
                        "width_px": 48,
                        "height_px": 36,
                        "minimum_feature_mm": 0.5,
                        "tile_size_px": 16,
                        "maximum_iterations": 128,
                        "root_refinement_steps": 4,
                    },
                    "material": {"color": [0.7, 0.7, 0.7]},
                    "background_lower": [0.9, 0.9, 0.9],
                    "background_upper": [1.0, 1.0, 1.0],
                },
            ).identifier,
            timeout_seconds=60.0,
        )
        assert render.state == "succeeded", render.error
        assert render.artifacts[0].media_type == "image/png"
        assert client.download_artifact(render.artifacts[0].identifier).startswith(
            b"\x89PNG\r\n\x1a\n"
        )


def test_backend_runs_custom_transition_and_shell_union_tasks(tmp_path) -> None:
    design_domain = tmp_path / "design-domain.stl"
    unit_cell = tmp_path / "unit-cell.stl"
    trimesh.creation.box(extents=(6.0, 6.0, 6.0)).export(design_domain)
    trimesh.creation.box(extents=(2.0, 2.0, 2.0)).export(unit_cell)
    sampling = {
        "target_voxels": 100_000,
        "min_samples_per_cell": 8.0,
        "min_samples_per_wall": 2.0,
    }
    tpms_g = {
        "kind": "G",
        "cell_size_mm": [4.0, 4.0, 4.0],
        "wall_thickness_mm": 0.5,
    }
    tpms_d = {**tpms_g, "kind": "D"}

    with LocalBackendServer() as server:
        client = LocalBackendClient(server.base_url, timeout_seconds=30.0)
        custom = _wait_for_terminal(
            client,
            client.submit_task(
                "custom.generate",
                {
                    "mesh_path": str(design_domain),
                    "parameters": {
                        "kind": "Custom",
                        "source_path": str(unit_cell),
                        "cell_size_mm": [4.0, 4.0, 4.0],
                    },
                    "sampling": sampling,
                    "display_voxel_size_mm": 0.9,
                },
            ).identifier,
            timeout_seconds=60.0,
        )
        assert custom.state == "succeeded", custom.error
        assert custom.result["kind"] == "Custom"

        transition = _wait_for_terminal(
            client,
            client.submit_task(
                "transition.generate",
                {
                    "mesh_path": str(design_domain),
                    "first_parameters": tpms_g,
                    "second_parameters": tpms_d,
                    "transition": {
                        "plane_axis": "X",
                        "plane_position_mm": 0.0,
                        "transition_width_mm": 2.0,
                    },
                    "sampling": sampling,
                    "display_voxel_size_mm": 0.9,
                },
            ).identifier,
            timeout_seconds=60.0,
        )
        assert transition.state == "succeeded", transition.error
        assert transition.result["kind"] == "Transition"

        shell_union = _wait_for_terminal(
            client,
            client.submit_task(
                "shell.union",
                {
                    "mesh_path": str(design_domain),
                    "generation_id": custom.result["generation_id"],
                    "thickness_mm": 0.5,
                    "fusion_radius_mm": 0.0,
                    "sampling": sampling,
                    "display_voxel_size_mm": 0.9,
                },
            ).identifier,
            timeout_seconds=60.0,
        )
        assert shell_union.state == "succeeded", shell_union.error
        assert shell_union.result["kind"] == "ShellUnion"


def test_backend_runs_atomic_batch_generation_and_publishes_each_field(tmp_path) -> None:
    design_domain = tmp_path / "design-domain.stl"
    trimesh.creation.box(extents=(6.0, 6.0, 6.0)).export(design_domain)
    sampling = {
        "target_voxels": 100_000,
        "min_samples_per_cell": 8.0,
        "min_samples_per_wall": 2.0,
    }
    tpms = {
        "cell_size_mm": [4.0, 4.0, 4.0],
        "wall_thickness_mm": 0.5,
    }

    with LocalBackendServer() as server:
        client = LocalBackendClient(server.base_url, timeout_seconds=30.0)
        completed = _wait_for_terminal(
            client,
            client.submit_task(
                "generation.batch",
                {
                    "requests": [
                        {
                            "request_id": "G",
                            "kind": "tpms.generate",
                            "payload": {
                                "mesh_path": str(design_domain),
                                "parameters": {**tpms, "kind": "G"},
                                "sampling": sampling,
                                "display_voxel_size_mm": 0.9,
                            },
                        },
                        {
                            "request_id": "D",
                            "kind": "tpms.generate",
                            "payload": {
                                "mesh_path": str(design_domain),
                                "parameters": {**tpms, "kind": "D"},
                                "sampling": sampling,
                                "display_voxel_size_mm": 0.9,
                            },
                        },
                    ]
                },
            ).identifier,
            timeout_seconds=120.0,
        )
        assert completed.state == "succeeded", completed.error
        artifact_ids = batch_display_field_artifact_ids(completed)
        assert set(artifact_ids) == {"G", "D"}
        results = read_generation_batch(
            completed,
            {
                request_id: client.download_artifact(artifact_id)
                for request_id, artifact_id in artifact_ids.items()
            },
        )
        assert set(results) == {"G", "D"}
        assert results["G"].generation_id != results["D"].generation_id
        assert results["G"].display_field.values.ndim == 3


def test_backend_runs_analytic_domain_and_field_driven_transition() -> None:
    domain = {
        "kind": "analytic",
        "name": "Sphere domain",
        "primitive": {
            "identifier": "domain-sphere",
            "name": "Domain",
            "kind": "sphere",
            "center_mm": [0.0, 0.0, 0.0],
            "rotation_euler_deg": [0.0, 0.0, 0.0],
            "radius_mm": 3.0,
            "height_mm": 6.0,
            "size_mm": [6.0, 6.0, 6.0],
        },
    }
    driver = {
        "identifier": "driver-sphere",
        "name": "Driver",
        "kind": "sphere",
        "center_mm": [0.0, 0.0, 0.0],
        "rotation_euler_deg": [0.0, 0.0, 0.0],
        "radius_mm": 1.5,
        "height_mm": 3.0,
        "size_mm": [3.0, 3.0, 3.0],
    }
    sampling = {
        "target_voxels": 100_000,
        "min_samples_per_cell": 8.0,
        "min_samples_per_wall": 2.0,
    }
    parameters = {
        "cell_size_mm": [4.0, 4.0, 4.0],
        "wall_thickness_mm": 0.5,
    }

    with LocalBackendServer() as server:
        client = LocalBackendClient(server.base_url, timeout_seconds=30.0)
        analytic = _wait_for_terminal(
            client,
            client.submit_task(
                "tpms.generate",
                {
                    "domain": domain,
                    "parameters": {**parameters, "kind": "G"},
                    "sampling": sampling,
                    "display_voxel_size_mm": 0.9,
                },
            ).identifier,
            timeout_seconds=60.0,
        )
        assert analytic.state == "succeeded", analytic.error
        assert read_generation_result(
            analytic,
            client.download_artifact(display_field_artifact_id(analytic)),
        ).kind == "G"

        transition = _wait_for_terminal(
            client,
            client.submit_task(
                "transition.generate",
                {
                    "domain": domain,
                    "first_parameters": {**parameters, "kind": "G"},
                    "second_parameters": {**parameters, "kind": "D"},
                    "transition": {
                        "driver_mode": "field",
                        "driver_identifier": "driver-sphere",
                        "field_interval_lower_mm": -0.5,
                        "field_interval_upper_mm": 0.5,
                    },
                    "driver_primitive": driver,
                    "sampling": sampling,
                    "display_voxel_size_mm": 0.9,
                },
            ).identifier,
            timeout_seconds=60.0,
        )
        assert transition.state == "succeeded", transition.error
        result = read_generation_result(
            transition,
            client.download_artifact(display_field_artifact_id(transition)),
        )
        assert result.kind == "Transition"
        assert result.cell_map is not None
