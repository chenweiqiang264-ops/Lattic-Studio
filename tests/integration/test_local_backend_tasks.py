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

    assert client.health().version == "0.1.1"
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
        field = np.load(BytesIO(client.download_artifact(generated.artifacts[0].identifier)))
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
