"""Single-thread dispatch for Python backend executors."""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any


class _ExecutorState:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.busy = False
        self.closed = False


class _GuardedResponse:
    """Complete the shared envelope and release admission exactly once."""

    def __init__(self, response: Any, state: _ExecutorState) -> None:
        self._response = response
        self._state = state
        self._released = False

    def set_result(self, result: Any) -> None:
        self._complete(self._response.set_result, result)

    def set_exception(self, error: BaseException) -> None:
        self._complete(self._response.set_exception, error)

    def _complete(self, complete: Any, value: Any) -> None:
        # Serialize publication with submit: a consumer can wake as soon as the
        # envelope is filled, before its setter returns to Python.
        with self._state.lock:
            if self._released:
                raise RuntimeError("response already completed")
            complete(value)
            self._release_locked()

    def _release(self) -> None:
        with self._state.lock:
            self._release_locked()

    def _release_locked(self) -> None:
        if not self._released:
            self._released = True
            self._state.busy = False


class ThreadedExecutor:
    """Own one persistent Python thread per backend replica.

    ``submit`` only schedules work and returns immediately. Plugin submissions
    are serialized on the pool thread, and plugin exceptions complete the Rust
    response envelope instead of escaping the pool future.
    """

    def __init__(self, executor: Any) -> None:
        self._executor = executor
        self._pool = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="infinity-python"
        )
        self._state = _ExecutorState()

    def submit(self, route: str, response: Any, payload: dict[str, Any]) -> None:
        guarded = _GuardedResponse(response, self._state)
        with self._state.lock:
            if self._state.closed:
                raise RuntimeError("executor is closed")
            if self._state.busy:
                raise RuntimeError("queue is full: Python executor is busy")
            self._state.busy = True
            try:
                self._pool.submit(self._run, route, guarded, payload)
            except BaseException:
                guarded._release_locked()
                raise

    def _run(self, route: str, response: _GuardedResponse, payload: dict[str, Any]) -> None:
        try:
            self._executor.submit(route, response, payload)
        except BaseException as error:  # noqa: BLE001 - delivered to the Rust side
            try:
                response.set_exception(error)
            except BaseException:  # noqa: BLE001 - completion already failed
                response._release()

    def healthy(self) -> bool:
        with self._state.lock:
            if self._state.closed:
                return False
            future = self._pool.submit(self._executor.healthy)
        return bool(future.result())

    def close(self) -> None:
        with self._state.lock:
            if self._state.closed:
                return
            self._state.closed = True
            close = self._pool.submit(self._executor.close)
        # Close is the final task, after accepted submissions and on their
        # execution thread. Python's waits release the GIL.
        try:
            close.result()
        finally:
            self._pool.shutdown(wait=True, cancel_futures=False)
