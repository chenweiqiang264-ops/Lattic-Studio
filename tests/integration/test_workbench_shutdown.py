"""Regression tests for safe workbench shutdown while tasks are active."""

from __future__ import annotations

import inspect
from types import SimpleNamespace

from lattice_studio.presentation.qt import workbench as app
from lattice_studio.presentation.qt.viewers.pyvista_viewer import PyVistaRenderer


class _FakeThread:
    def __init__(self, *, running: bool = True, stops: bool = True) -> None:
        self.running = running
        self.stops = stops
        self.interruption_requests = 0
        self.quit_requests = 0
        self.wait_timeouts: list[int] = []

    def isRunning(self) -> bool:
        return self.running

    def requestInterruption(self) -> None:
        self.interruption_requests += 1

    def quit(self) -> None:
        self.quit_requests += 1

    def wait(self, timeout_ms: int) -> bool:
        self.wait_timeouts.append(timeout_ms)
        if self.stops:
            self.running = False
        return self.stops


def _owner_with_threads(threads: dict[str, _FakeThread]) -> SimpleNamespace:
    values = {attribute: None for attribute, _message in app.BACKGROUND_THREAD_SPECS}
    values.update(threads)
    return SimpleNamespace(**values)


def test_shutdown_requests_every_background_task_before_waiting() -> None:
    threads = {
        attribute: _FakeThread()
        for attribute, _message in app.BACKGROUND_THREAD_SPECS
    }
    owner = _owner_with_threads(threads)

    stopped, message = app.request_background_thread_shutdown(
        owner,
        timeout_ms=1_000,
    )

    assert stopped is True
    assert message is None
    assert all(thread.interruption_requests == 1 for thread in threads.values())
    assert all(thread.quit_requests == 1 for thread in threads.values())
    assert all(len(thread.wait_timeouts) == 1 for thread in threads.values())
    assert all(not thread.running for thread in threads.values())


def test_shutdown_reports_unresponsive_task_without_destroying_it() -> None:
    thread = _FakeThread(stops=False)
    owner = _owner_with_threads({"stl_thread": thread})

    stopped, message = app.request_background_thread_shutdown(
        owner,
        timeout_ms=25,
    )

    assert stopped is False
    assert message == "正在取消 STL 重建，请稍候再关闭。"
    assert thread.interruption_requests == 1
    assert thread.quit_requests == 1
    assert thread.running is True


def test_shutdown_ignores_threads_that_are_already_finished() -> None:
    thread = _FakeThread(running=False)
    owner = _owner_with_threads({"shell_thread": thread})

    stopped, message = app.request_background_thread_shutdown(owner)

    assert stopped is True
    assert message is None
    assert thread.interruption_requests == 0
    assert thread.quit_requests == 0
    assert thread.wait_timeouts == []


def test_viewport_shutdown_is_idempotent() -> None:
    from PyQt5 import QtCore

    class _Viewer(QtCore.QObject):
        def __init__(self) -> None:
            super().__init__()
            self.close_count = 0
            self.updates_enabled = True
            self.hidden = False

        def setUpdatesEnabled(self, enabled: bool) -> None:
            self.updates_enabled = enabled

        def findChildren(self, _kind):
            return []

        def hide(self) -> None:
            self.hidden = True

        def close(self) -> None:
            self.close_count += 1

    viewer = _Viewer()
    owner = SimpleNamespace(viewer=viewer, _viewport_closed=False)
    window_class = app._build_qt_app()

    window_class._close_viewport(owner)
    window_class._close_viewport(owner)

    assert viewer.close_count == 1
    assert viewer.updates_enabled is False
    assert viewer.hidden is True
    assert owner._viewport_closed is True


def test_resize_handler_does_not_reenter_the_qt_event_loop() -> None:
    """A native WM_CLOSE may resize the viewport while shutdown is active."""

    source = inspect.getsource(PyVistaRenderer.resizeEvent)

    assert "processEvents" not in source
    assert "updatesEnabled" in source
    assert 'getattr(self, "_closed", False)' in source
