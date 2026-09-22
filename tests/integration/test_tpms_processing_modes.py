"""Behavioral tests for the three scalar-field processing modes."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import trimesh

TESTS_DIR = Path(__file__).resolve().parent
if str(TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(TESTS_DIR))

from lattice_studio.presentation.qt import workbench as app


def _small_domain() -> trimesh.Trimesh:
    return trimesh.creation.box(extents=(4.0, 4.0, 4.0))


def _run_mode(
    monkeypatch,
    mode: str,
) -> tuple[trimesh.Trimesh, list[int], list[int], list[str]]:
    domain_calls: list[int] = []
    tpms_calls: list[int] = []
    progress_messages: list[str] = []

    def fake_domain(_mesh, points, _progress, _use_cpp):
        domain_calls.append(len(points))
        return np.full(len(points), -10.0, dtype=np.float32)

    def fake_tpms(_kind, points, _params, max_tile_points=None):
        tpms_calls.append(len(points))
        center = np.array((2.0, 2.0, 2.0), dtype=np.float32)
        return (np.sum((points.astype(np.float32) - center) ** 2, axis=1) - 1.5).astype(np.float32)

    monkeypatch.setattr(app, "_compute_design_domain_sdf", fake_domain)
    monkeypatch.setattr(app, "_tpms_shell_field", fake_tpms)
    sampling = app.SamplingParameters(
        target_voxels=100_000,
        use_cpp_sdf=False,
        processing_mode=mode,
        batch_count=1 if mode == "single_pass" else 2,
    )
    mesh, _ = app.generate_tpms_lattice(
        _small_domain(),
        app.TPMSParameters("G", cell_size_mm=2.0, wall_thickness_mm=0.2),
        sampling,
        voxel_size_mm=1.0,
        progress=lambda message, _value: progress_messages.append(message),
    )
    return mesh, domain_calls, tpms_calls, progress_messages


def test_single_pass_uses_one_full_point_evaluation(monkeypatch):
    mesh, domain_calls, tpms_calls, _messages = _run_mode(monkeypatch, "single_pass")

    assert len(mesh.faces) > 0
    assert len(domain_calls) == 1
    assert len(tpms_calls) == 1
    assert domain_calls[0] == 7 * 7 * 7


def test_batched_field_samples_slabs_but_extracts_one_surface(monkeypatch):
    mesh, domain_calls, tpms_calls, _messages = _run_mode(monkeypatch, "batched_field")

    assert len(mesh.faces) > 0
    assert len(domain_calls) == 2
    assert len(tpms_calls) == 2
    assert sum(domain_calls) == 7 * 7 * 7


def test_chunked_mode_samples_and_extracts_each_requested_chunk(monkeypatch):
    mesh, domain_calls, tpms_calls, _messages = _run_mode(monkeypatch, "chunked_marching_cubes")

    assert len(mesh.faces) > 0
    assert len(domain_calls) == 2
    assert len(tpms_calls) == 2
    assert sum(domain_calls) > 7 * 7 * 7


def test_all_modes_report_the_actual_generation_stages(monkeypatch):
    for mode in ("single_pass", "batched_field", "chunked_marching_cubes"):
        _mesh, _domain_calls, _tpms_calls, messages = _run_mode(monkeypatch, mode)

        assert any("计算设计域 SDF" in message for message in messages)
        assert any("计算 TPMS 场" in message for message in messages)
        assert any("计算设计域与 TPMS 场交集" in message for message in messages)
        assert any("执行 Marching Cubes" in message for message in messages)
        assert not any("计算体素场" in message for message in messages)
        assert not any("等值面提取" in message for message in messages)
        if mode == "chunked_marching_cubes":
            assert any("划分计算分块" in message for message in messages)
            assert any("微分片" in message for message in messages)
            assert any("拼接分块网格" in message for message in messages)


def test_micro_slices_limit_the_number_of_points_per_evaluation() -> None:
    shape = (3, 200, 200)
    axes = [np.arange(size, dtype=float) for size in shape]
    evaluated_sizes: list[int] = []

    def evaluate(points: np.ndarray, _tile_index: int, _tile_count: int) -> np.ndarray:
        evaluated_sizes.append(len(points))
        return np.zeros(len(points), dtype=np.float32)

    field = app._sample_grid_points(
        axes,
        shape,
        0,
        shape[0],
        evaluate,
        max_tile_points=50_000,
    )

    assert field.shape == shape
    assert len(evaluated_sizes) == 3
    assert max(evaluated_sizes) <= 50_000
