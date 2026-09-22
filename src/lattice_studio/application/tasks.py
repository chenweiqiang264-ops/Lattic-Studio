"""Asynchronous task primitives owned by the local backend process.

The desktop client only sees serializable task snapshots and artifact metadata.
Handlers are registered in-process by the backend, which prevents the HTTP
boundary from becoming an arbitrary callable-execution endpoint.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from shutil import rmtree
from tempfile import mkdtemp
from threading import Event, RLock
from typing import Literal
from uuid import uuid4


TaskState = Literal["queued", "running", "succeeded", "failed", "cancelled"]
TERMINAL_TASK_STATES: frozenset[TaskState] = frozenset(
    {"succeeded", "failed", "cancelled"}
)


class UnknownTask(KeyError):
    """Raised when a task identifier is unknown to this backend process."""


class UnknownArtifact(KeyError):
    """Raised when an artifact identifier is unknown to this backend process."""


class TaskCancelled(RuntimeError):
    """Raised by a handler after the client has requested cancellation."""


@dataclass(frozen=True)
class TaskArtifact:
    """A backend-managed, downloadable task output."""

    identifier: str
    name: str
    media_type: str
    size_bytes: int

    def to_dict(self) -> dict[str, object]:
        return {
            "artifact_id": self.identifier,
            "name": self.name,
            "media_type": self.media_type,
            "size_bytes": self.size_bytes,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "TaskArtifact":
        return cls(
            identifier=str(value["artifact_id"]),
            name=str(value["name"]),
            media_type=str(value["media_type"]),
            size_bytes=int(value["size_bytes"]),
        )


@dataclass(frozen=True)
class TaskSnapshot:
    """JSON-safe status exposed to a client polling an asynchronous task."""

    identifier: str
    kind: str
    state: TaskState
    progress: float
    message: str
    result: dict[str, object] | None
    error: str | None
    artifacts: tuple[TaskArtifact, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "task_id": self.identifier,
            "kind": self.kind,
            "state": self.state,
            "progress": self.progress,
            "message": self.message,
            "result": self.result,
            "error": self.error,
            "artifacts": [artifact.to_dict() for artifact in self.artifacts],
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "TaskSnapshot":
        raw_result = value.get("result")
        result = None if raw_result is None else dict(_mapping(raw_result, "result"))
        raw_artifacts = value.get("artifacts", ())
        if not isinstance(raw_artifacts, list):
            raise ValueError("task artifacts must be a JSON array")
        return cls(
            identifier=str(value["task_id"]),
            kind=str(value["kind"]),
            state=_task_state(value["state"]),
            progress=float(value["progress"]),
            message=str(value["message"]),
            result=result,
            error=None if value.get("error") is None else str(value["error"]),
            artifacts=tuple(
                TaskArtifact.from_dict(_mapping(item, "artifact"))
                for item in raw_artifacts
            ),
        )


@dataclass(frozen=True)
class TaskOutcome:
    """Handler result plus its files that should be exposed to the client."""

    result: Mapping[str, object] | None = None
    artifacts: tuple[Path, ...] = ()


TaskHandler = Callable[[Mapping[str, object], "TaskContext"], TaskOutcome]


class TaskContext:
    """A handler's narrow capability to report state and publish artifacts."""

    def __init__(self, service: "BackendTaskService", identifier: str, cancelled: Event) -> None:
        self._service = service
        self._identifier = identifier
        self._cancelled = cancelled
        self._work_dir = service._task_work_dir(identifier)

    @property
    def work_dir(self) -> Path:
        self._work_dir.mkdir(parents=True, exist_ok=True)
        return self._work_dir

    @property
    def cancelled(self) -> bool:
        return self._cancelled.is_set()

    def raise_if_cancelled(self) -> None:
        if self.cancelled:
            raise TaskCancelled("task cancelled")

    def report(self, message: str, progress: float) -> None:
        self.raise_if_cancelled()
        self._service._report(self._identifier, message, progress)


@dataclass
class _TaskRecord:
    identifier: str
    kind: str
    state: TaskState = "queued"
    progress: float = 0.0
    message: str = "queued"
    result: dict[str, object] | None = None
    error: str | None = None
    artifacts: tuple[TaskArtifact, ...] = ()
    cancelled: Event | None = None
    future: Future[None] | None = None


class BackendTaskService:
    """Run a fixed task registry and retain task/artifact state for one process."""

    def __init__(
        self,
        handlers: Mapping[str, TaskHandler] | None = None,
        *,
        artifact_root: Path | None = None,
        max_workers: int = 1,
    ) -> None:
        if max_workers < 1:
            raise ValueError("max_workers must be at least one")
        self._handlers = dict(handlers or {})
        self._artifact_root = (
            Path(artifact_root).resolve()
            if artifact_root is not None
            else Path(mkdtemp(prefix="lattice-studio-backend-"))
        )
        self._owns_artifact_root = artifact_root is None
        self._artifact_root.mkdir(parents=True, exist_ok=True)
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="lattice-backend-task",
        )
        self._lock = RLock()
        self._records: dict[str, _TaskRecord] = {}
        self._artifact_paths: dict[str, Path] = {}

    @property
    def supported_kinds(self) -> tuple[str, ...]:
        return tuple(sorted(self._handlers))

    def submit(self, kind: str, payload: Mapping[str, object]) -> TaskSnapshot:
        handler = self._handlers.get(str(kind))
        if handler is None:
            raise ValueError(f"unsupported task kind: {kind}")
        identifier = uuid4().hex
        cancelled = Event()
        record = _TaskRecord(identifier=identifier, kind=str(kind), cancelled=cancelled)
        with self._lock:
            self._records[identifier] = record
            record.future = self._executor.submit(
                self._run,
                identifier,
                handler,
                dict(payload),
                cancelled,
            )
            return self._snapshot(record)

    def get(self, identifier: str) -> TaskSnapshot:
        with self._lock:
            return self._snapshot(self._record(identifier))

    def cancel(self, identifier: str) -> TaskSnapshot:
        with self._lock:
            record = self._record(identifier)
            if record.state in TERMINAL_TASK_STATES:
                return self._snapshot(record)
            assert record.cancelled is not None
            record.cancelled.set()
            if record.future is not None and record.future.cancel():
                record.state = "cancelled"
                record.progress = 1.0
                record.message = "cancelled"
            else:
                record.message = "cancellation requested"
            return self._snapshot(record)

    def artifact_path(self, identifier: str) -> tuple[TaskArtifact, Path]:
        with self._lock:
            path = self._artifact_paths.get(identifier)
            if path is None:
                raise UnknownArtifact(identifier)
            for record in self._records.values():
                for artifact in record.artifacts:
                    if artifact.identifier == identifier:
                        return artifact, path
        raise UnknownArtifact(identifier)

    def close(self, *, wait: bool = True) -> None:
        self._executor.shutdown(wait=wait, cancel_futures=True)
        if self._owns_artifact_root:
            rmtree(self._artifact_root, ignore_errors=True)

    def _run(
        self,
        identifier: str,
        handler: TaskHandler,
        payload: Mapping[str, object],
        cancelled: Event,
    ) -> None:
        with self._lock:
            record = self._record(identifier)
            if cancelled.is_set():
                record.state = "cancelled"
                record.progress = 1.0
                record.message = "cancelled"
                return
            record.state = "running"
            record.message = "running"
        context = TaskContext(self, identifier, cancelled)
        try:
            outcome = handler(payload, context)
            context.raise_if_cancelled()
            artifacts = self._register_artifacts(identifier, outcome.artifacts)
            result = None if outcome.result is None else dict(outcome.result)
            with self._lock:
                record = self._record(identifier)
                record.state = "succeeded"
                record.progress = 1.0
                record.message = "completed"
                record.result = result
                record.artifacts = artifacts
        except TaskCancelled:
            with self._lock:
                record = self._record(identifier)
                record.state = "cancelled"
                record.progress = 1.0
                record.message = "cancelled"
        except Exception as exc:
            with self._lock:
                record = self._record(identifier)
                record.state = "failed"
                record.progress = 1.0
                record.message = "failed"
                record.error = f"{type(exc).__name__}: {exc}"

    def _report(self, identifier: str, message: str, progress: float) -> None:
        bounded = min(max(float(progress), 0.0), 1.0)
        with self._lock:
            record = self._record(identifier)
            if record.state != "running":
                return
            record.message = str(message)
            record.progress = bounded

    def _register_artifacts(
        self,
        task_identifier: str,
        paths: tuple[Path, ...],
    ) -> tuple[TaskArtifact, ...]:
        work_dir = self._task_work_dir(task_identifier).resolve()
        artifacts = []
        for path in paths:
            resolved = Path(path).resolve()
            if not resolved.is_file() or work_dir not in resolved.parents:
                raise ValueError("task artifacts must be files inside the task work directory")
            identifier = uuid4().hex
            artifact = TaskArtifact(
                identifier=identifier,
                name=resolved.name,
                media_type=_media_type_for(resolved),
                size_bytes=resolved.stat().st_size,
            )
            with self._lock:
                self._artifact_paths[identifier] = resolved
            artifacts.append(artifact)
        return tuple(artifacts)

    def _task_work_dir(self, identifier: str) -> Path:
        return self._artifact_root / identifier

    def _record(self, identifier: str) -> _TaskRecord:
        try:
            return self._records[identifier]
        except KeyError as exc:
            raise UnknownTask(identifier) from exc

    @staticmethod
    def _snapshot(record: _TaskRecord) -> TaskSnapshot:
        return TaskSnapshot(
            identifier=record.identifier,
            kind=record.kind,
            state=record.state,
            progress=record.progress,
            message=record.message,
            result=None if record.result is None else dict(record.result),
            error=record.error,
            artifacts=record.artifacts,
        )


def _task_state(value: object) -> TaskState:
    state = str(value)
    if state not in {"queued", "running", "succeeded", "failed", "cancelled"}:
        raise ValueError(f"unsupported task state: {state}")
    return state  # type: ignore[return-value]


def _mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"task {name} must be a JSON object")
    return value


def _media_type_for(path: Path) -> str:
    return {
        ".json": "application/json",
        ".npz": "application/x-npz",
        ".png": "image/png",
        ".stl": "model/stl",
    }.get(path.suffix.lower(), "application/octet-stream")


__all__ = [
    "BackendTaskService",
    "TaskArtifact",
    "TaskCancelled",
    "TaskContext",
    "TaskOutcome",
    "TaskSnapshot",
    "TaskState",
    "UnknownArtifact",
    "UnknownTask",
]
