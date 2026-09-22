"""Tests for crash-isolated automatic CUDA discovery."""

from __future__ import annotations

from types import SimpleNamespace
from pathlib import Path
import json

import lattice_studio.engine.implicit.cuda_probe as cuda_probe


def test_frozen_probe_uses_worker_command_instead_of_python_c(monkeypatch):
    monkeypatch.delenv("TPMS_DISABLE_NUMBA_CUDA", raising=False)
    monkeypatch.delenv("TPMS_ENABLE_NUMBA_CUDA", raising=False)
    monkeypatch.setattr(cuda_probe.sys, "frozen", True, raising=False)

    def run_worker(command, **kwargs):
        assert command[1] == "--cuda-probe-worker"
        Path(command[2]).write_text(json.dumps({"available": True,
            "device_name": "Test GPU", "reason": ""}), encoding="utf-8")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(cuda_probe.subprocess, "run", run_worker)
    assert cuda_probe.probe_numba_cuda().device_name == "Test GPU"


def test_probe_automatically_accepts_a_working_numba_runtime(monkeypatch) -> None:
    monkeypatch.delenv("TPMS_DISABLE_NUMBA_CUDA", raising=False)
    monkeypatch.delenv("TPMS_ENABLE_NUMBA_CUDA", raising=False)

    calls = []

    def fake_run(*args, **kwargs):
        calls.append((args, kwargs))
        return SimpleNamespace(
            returncode=0,
            stdout="NVIDIA GeForce RTX 4060 Laptop GPU\n",
            stderr="",
        )

    monkeypatch.setattr(cuda_probe.subprocess, "run", fake_run)

    result = cuda_probe.probe_numba_cuda()

    assert result.available is True
    assert result.device_name == "NVIDIA GeForce RTX 4060 Laptop GPU"
    assert result.reason == ""
    assert len(calls) == 1


def test_probe_can_be_explicitly_disabled(monkeypatch) -> None:
    monkeypatch.setenv("TPMS_DISABLE_NUMBA_CUDA", "true")

    def unexpected_run(*_args, **_kwargs):
        raise AssertionError("disabled CUDA must not start a child probe")

    monkeypatch.setattr(cuda_probe.subprocess, "run", unexpected_run)

    result = cuda_probe.probe_numba_cuda()

    assert result.available is False
    assert result.reason == "Numba CUDA disabled by TPMS_DISABLE_NUMBA_CUDA=1"


def test_probe_preserves_legacy_explicit_disable(monkeypatch) -> None:
    monkeypatch.delenv("TPMS_DISABLE_NUMBA_CUDA", raising=False)
    monkeypatch.setenv("TPMS_ENABLE_NUMBA_CUDA", "0")

    def unexpected_run(*_args, **_kwargs):
        raise AssertionError("legacy disabled CUDA must not start a child probe")

    monkeypatch.setattr(cuda_probe.subprocess, "run", unexpected_run)

    result = cuda_probe.probe_numba_cuda()

    assert result.available is False
    assert result.reason == "Numba CUDA disabled by TPMS_ENABLE_NUMBA_CUDA=0"


def test_probe_reports_child_runtime_failure(monkeypatch) -> None:
    monkeypatch.delenv("TPMS_DISABLE_NUMBA_CUDA", raising=False)
    monkeypatch.delenv("TPMS_ENABLE_NUMBA_CUDA", raising=False)
    monkeypatch.setattr(
        cuda_probe.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(
            returncode=1,
            stdout="",
            stderr="CudaSupportError: initialization error",
        ),
    )

    result = cuda_probe.probe_numba_cuda()

    assert result.available is False
    assert result.reason == "CudaSupportError: initialization error"
