"""Loopback HTTP adapter for the local Lattice Studio backend.

The transport is deliberately narrow: it carries serializable workspace-session
data only and listens exclusively on ``127.0.0.1``. Rendering resources,
numerical caches, and Qt objects remain process-local implementation details.
"""

from __future__ import annotations

import argparse
import json
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread
from urllib.error import HTTPError, URLError
from urllib.parse import unquote, urlparse
from urllib.request import ProxyHandler, Request, build_opener

from lattice_studio.application.backend import (
    BackendHealth,
    LocalBackend,
    UnknownWorkspace,
    WorkspaceSession,
)
from lattice_studio.application.tasks import (
    TaskArtifact,
    TaskSnapshot,
    UnknownArtifact,
    UnknownTask,
)


class BackendRequestError(RuntimeError):
    """An HTTP response that a backend client cannot treat as a success."""

    def __init__(self, status_code: int, message: str) -> None:
        self.status_code = int(status_code)
        self.message = str(message)
        super().__init__(f"backend request failed ({self.status_code}): {self.message}")


class LocalBackendClient:
    """Client adapter for a locally running :class:`LocalBackendServer`."""

    def __init__(self, base_url: str, *, timeout_seconds: float = 10.0) -> None:
        normalized = str(base_url).rstrip("/")
        parsed = urlparse(normalized)
        if parsed.scheme != "http" or parsed.hostname != "127.0.0.1":
            raise ValueError("local backend clients must use an http://127.0.0.1 URL")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        self._base_url = normalized
        self._timeout_seconds = float(timeout_seconds)
        # A loopback backend must never escape through a corporate/system
        # proxy.  Besides being unnecessary, proxy routing can turn a healthy
        # private backend into a misleading 502 response.
        self._opener = build_opener(ProxyHandler({}))

    def health(self) -> BackendHealth:
        return BackendHealth.from_dict(self._request("GET", "/v1/health"))

    def create_workspace(self) -> WorkspaceSession:
        return WorkspaceSession.from_dict(self._request("POST", "/v1/workspaces"))

    def get_workspace(self, identifier: str) -> WorkspaceSession:
        return WorkspaceSession.from_dict(
            self._request("GET", f"/v1/workspaces/{identifier}")
        )

    def get_workspace_snapshot(self, identifier: str) -> dict[str, object]:
        """Read editable workspace definitions without materializing engine objects."""

        return self._request("GET", f"/v1/workspaces/{identifier}/snapshot")

    def apply_workspace_commands(
        self,
        identifier: str,
        commands: list[dict[str, object]],
    ) -> dict[str, object]:
        """Apply a fixed, atomic batch of JSON document commands."""

        return self._request(
            "POST",
            f"/v1/workspaces/{identifier}/commands",
            {"commands": commands},
        )

    def load_workspace(self, manifest_path: str) -> WorkspaceSession:
        return WorkspaceSession.from_dict(
            self._request("POST", "/v1/workspaces/load", {"manifest_path": manifest_path})
        )

    def save_workspace(self, identifier: str, manifest_path: str) -> WorkspaceSession:
        return WorkspaceSession.from_dict(
            self._request(
                "POST",
                f"/v1/workspaces/{identifier}/save",
                {"manifest_path": manifest_path},
            )
        )

    def submit_task(self, kind: str, payload: dict[str, object]) -> TaskSnapshot:
        return TaskSnapshot.from_dict(
            self._request("POST", "/v1/tasks", {"kind": kind, "payload": payload})
        )

    def get_task(self, identifier: str) -> TaskSnapshot:
        return TaskSnapshot.from_dict(self._request("GET", f"/v1/tasks/{identifier}"))

    def cancel_task(self, identifier: str) -> TaskSnapshot:
        return TaskSnapshot.from_dict(
            self._request("POST", f"/v1/tasks/{identifier}/cancel")
        )

    def download_artifact(self, identifier: str) -> bytes:
        request = Request(
            f"{self._base_url}/v1/artifacts/{identifier}", method="GET"
        )
        try:
            with self._opener.open(request, timeout=self._timeout_seconds) as response:
                return response.read()
        except HTTPError as exc:
            try:
                message = str(_decode_payload(exc.read()).get("error", exc.reason))
            except (TypeError, ValueError):
                message = str(exc.reason)
            raise BackendRequestError(exc.code, message) from exc
        except URLError as exc:
            raise BackendRequestError(0, str(exc.reason)) from exc

    def _request(
        self,
        method: str,
        path: str,
        payload: dict[str, object] | None = None,
    ) -> dict[str, object]:
        body = None if payload is None else json.dumps(payload).encode("utf-8")
        request = Request(
            f"{self._base_url}{path}",
            data=body,
            headers={"Content-Type": "application/json"} if body is not None else {},
            method=method,
        )
        try:
            with self._opener.open(request, timeout=self._timeout_seconds) as response:
                return _decode_payload(response.read())
        except HTTPError as exc:
            try:
                message = str(_decode_payload(exc.read()).get("error", exc.reason))
            except (TypeError, ValueError):
                message = str(exc.reason)
            raise BackendRequestError(exc.code, message) from exc
        except URLError as exc:
            raise BackendRequestError(0, str(exc.reason)) from exc


class LocalBackendServer:
    """A process-owned HTTP server that exposes a :class:`LocalBackend`."""

    def __init__(self, backend: LocalBackend | None = None, *, port: int = 0) -> None:
        if not 0 <= int(port) <= 65535:
            raise ValueError("port must be between 0 and 65535")
        self.backend = backend or LocalBackend()
        self._httpd = ThreadingHTTPServer(
            ("127.0.0.1", int(port)), _handler_for(self.backend)
        )
        self._thread: Thread | None = None

    @property
    def base_url(self) -> str:
        host, port = self._httpd.server_address[:2]
        return f"http://{host}:{port}"

    def start(self) -> "LocalBackendServer":
        """Serve requests on a background thread for an embedding client or test."""

        if self._thread is None:
            self._thread = Thread(target=self._httpd.serve_forever, daemon=True)
            self._thread.start()
        return self

    def serve_forever(self) -> None:
        """Serve requests in the calling process until interrupted or shut down."""

        self._httpd.serve_forever()

    def close(self) -> None:
        """Stop accepting requests and release the loopback socket."""

        if self._thread is not None:
            self._httpd.shutdown()
            self._thread.join(timeout=5)
            self._thread = None
        self._httpd.server_close()
        self.backend.close()

    def __enter__(self) -> "LocalBackendServer":
        return self.start()

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()


def _handler_for(backend: LocalBackend) -> type[BaseHTTPRequestHandler]:
    class BackendRequestHandler(BaseHTTPRequestHandler):
        server_version = "LatticeStudioBackend/0.1"
        protocol_version = "HTTP/1.1"

        def do_GET(self) -> None:  # noqa: N802 - stdlib callback name
            self._dispatch()

        def do_POST(self) -> None:  # noqa: N802 - stdlib callback name
            self._dispatch()

        def log_message(self, format: str, *args: object) -> None:
            """Keep the library adapter quiet; callers choose their own logging."""

        def _dispatch(self) -> None:
            try:
                payload = self._route()
            except _ResponseAlreadyWritten:
                return
            except UnknownWorkspace as exc:
                self._write_json(
                    HTTPStatus.NOT_FOUND,
                    {"error": f"unknown workspace: {exc.args[0]}"},
                )
            except UnknownTask as exc:
                self._write_json(
                    HTTPStatus.NOT_FOUND,
                    {"error": f"unknown task: {exc.args[0]}"},
                )
            except UnknownArtifact as exc:
                self._write_json(
                    HTTPStatus.NOT_FOUND,
                    {"error": f"unknown artifact: {exc.args[0]}"},
                )
            except (FileNotFoundError, KeyError, TypeError, ValueError) as exc:
                self._write_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
            except Exception:  # pragma: no cover - defensive transport boundary
                self._write_json(
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                    {"error": "internal backend error"},
                )
            else:
                self._write_json(HTTPStatus.OK, payload)

        def _route(self) -> dict[str, object]:
            path = urlparse(self.path).path
            if self.command == "GET" and path == "/v1/health":
                return backend.health().to_dict()
            if self.command == "POST" and path == "/v1/workspaces":
                return backend.create_workspace().to_dict()
            if self.command == "POST" and path == "/v1/workspaces/load":
                return backend.load_workspace(self._json_body()["manifest_path"]).to_dict()
            if self.command == "POST" and path == "/v1/tasks":
                body = self._json_body()
                return backend.submit_task(
                    str(body["kind"]),
                    _object_payload(body.get("payload", {}), "task payload"),
                ).to_dict()

            task_prefix = "/v1/tasks/"
            if path.startswith(task_prefix):
                remainder = unquote(path.removeprefix(task_prefix))
                if self.command == "POST" and remainder.endswith("/cancel"):
                    return backend.cancel_task(remainder.removesuffix("/cancel")).to_dict()
                if self.command == "GET" and "/" not in remainder:
                    return backend.get_task(remainder).to_dict()

            artifact_prefix = "/v1/artifacts/"
            if self.command == "GET" and path.startswith(artifact_prefix):
                identifier = unquote(path.removeprefix(artifact_prefix))
                if "/" not in identifier:
                    artifact, artifact_path = backend.task_artifact_path(identifier)
                    self._write_artifact(artifact, artifact_path)
                    raise _ResponseAlreadyWritten()

            prefix = "/v1/workspaces/"
            if path.startswith(prefix):
                remainder = unquote(path.removeprefix(prefix))
                if self.command == "GET" and remainder.endswith("/snapshot"):
                    identifier = remainder.removesuffix("/snapshot")
                    if "/" not in identifier:
                        return backend.workspace_snapshot(identifier)
                if self.command == "POST" and remainder.endswith("/commands"):
                    identifier = remainder.removesuffix("/commands")
                    if "/" not in identifier:
                        body = self._json_body()
                        commands = body.get("commands")
                        if not isinstance(commands, list):
                            raise ValueError("commands must be a JSON array")
                        if not all(isinstance(command, dict) for command in commands):
                            raise ValueError("each workspace command must be a JSON object")
                        return backend.apply_workspace_commands(identifier, commands)
                if self.command == "POST" and remainder.endswith("/save"):
                    identifier = remainder.removesuffix("/save")
                    return backend.save_workspace(
                        identifier, self._json_body()["manifest_path"]
                    ).to_dict()
                if self.command == "GET" and "/" not in remainder:
                    return backend.get_workspace(remainder).to_dict()

            self._write_json(HTTPStatus.NOT_FOUND, {"error": "unknown endpoint"})
            raise _ResponseAlreadyWritten()

        def _json_body(self) -> dict[str, object]:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > 1_048_576:
                raise ValueError("request body must contain at most 1 MiB of JSON")
            payload = _decode_payload(self.rfile.read(length))
            return payload

        def _write_json(self, status: HTTPStatus, payload: dict[str, object]) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _write_artifact(self, artifact: TaskArtifact, path) -> None:
            body = path.read_bytes()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", artifact.media_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header(
                "Content-Disposition",
                f'attachment; filename="{artifact.name}"',
            )
            self.end_headers()
            self.wfile.write(body)

    return BackendRequestHandler


class _ResponseAlreadyWritten(Exception):
    """Stop dispatch after an adapter has already sent a response."""


def _decode_payload(raw: bytes) -> dict[str, object]:
    value = json.loads(raw.decode("utf-8"))
    if not isinstance(value, dict):
        raise ValueError("response JSON must be an object")
    return value


def _object_payload(value: object, name: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a JSON object")
    return value


def main(argv: list[str] | None = None) -> int:
    """Run the local backend process used by a separate desktop client."""

    parser = argparse.ArgumentParser(description="Run the local Lattice Studio backend")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args(argv)
    server = LocalBackendServer(port=args.port)
    print(f"Lattice Studio backend listening on {server.base_url}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        return 0
    finally:
        server.close()


__all__ = ["BackendRequestError", "LocalBackendClient", "LocalBackendServer", "main"]
