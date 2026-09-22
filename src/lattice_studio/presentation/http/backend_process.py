"""Lifecycle adapter for the private loopback backend child process."""

from __future__ import annotations

import socket
import subprocess
import sys
import time
from pathlib import Path

from lattice_studio.presentation.http.local_backend import (
    BackendRequestError,
    LocalBackendClient,
)


class BackendProcessError(RuntimeError):
    """The desktop client could not start or stop its owned backend process."""


class LocalBackendProcess:
    """Start and stop exactly one loopback backend process for a desktop client."""

    def __init__(
        self,
        *,
        executable: str | Path | None = None,
        startup_timeout_seconds: float = 10.0,
    ) -> None:
        if startup_timeout_seconds <= 0:
            raise ValueError("startup_timeout_seconds must be positive")
        self._executable = str(executable or sys.executable)
        self._startup_timeout_seconds = float(startup_timeout_seconds)
        self._process: subprocess.Popen[bytes] | None = None
        self._client: LocalBackendClient | None = None

    @property
    def client(self) -> LocalBackendClient:
        if self._client is None:
            raise BackendProcessError("local backend process has not been started")
        return self._client

    @property
    def is_running(self) -> bool:
        return self._process is not None and self._process.poll() is None

    def start(self) -> LocalBackendClient:
        """Launch the backend and wait until its loopback health endpoint is ready."""

        if self.is_running:
            return self.client
        self.stop()
        port = _reserve_loopback_port()
        base_url = f"http://127.0.0.1:{port}"
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        command = [self._executable]
        if not getattr(sys, "frozen", False):
            command.extend(("-m", "lattice_studio"))
        command.extend(("--serve-backend", "--port", str(port)))
        self._process = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=creationflags,
        )
        client = LocalBackendClient(base_url, timeout_seconds=0.25)
        deadline = time.monotonic() + self._startup_timeout_seconds
        while time.monotonic() < deadline:
            if self._process.poll() is not None:
                self._process = None
                raise BackendProcessError("local backend exited before becoming ready")
            try:
                client.health()
            except BackendRequestError:
                time.sleep(0.05)
            else:
                self._client = client
                return client
        self.stop()
        raise BackendProcessError("timed out waiting for the local backend")

    def stop(self, *, timeout_seconds: float = 5.0) -> None:
        """Terminate only the child process owned by this adapter."""

        process = self._process
        self._client = None
        self._process = None
        if process is None or process.poll() is not None:
            return
        process.terminate()
        try:
            process.wait(timeout=max(float(timeout_seconds), 0.1))
        except subprocess.TimeoutExpired:
            process.kill()
            try:
                process.wait(timeout=1.0)
            except subprocess.TimeoutExpired as exc:  # pragma: no cover - OS failure
                raise BackendProcessError("could not stop local backend process") from exc

    def __enter__(self) -> LocalBackendClient:
        return self.start()

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.stop()


def _reserve_loopback_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


__all__ = ["BackendProcessError", "LocalBackendProcess"]
