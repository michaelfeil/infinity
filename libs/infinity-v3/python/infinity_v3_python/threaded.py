"""Single-thread dispatch for Python backend executors."""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any


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
        self._lock = threading.Lock()
        self._closed = False

    def submit(self, route: str, response: Any, payload: dict[str, Any]) -> None:
        with self._lock:
            if self._closed:
                response.set_exception(RuntimeError("executor is closed"))
                return
            self._pool.submit(self._run, route, response, payload)

    def _run(self, route: str, response: Any, payload: dict[str, Any]) -> None:
        try:
            self._executor.submit(route, response, payload)
        except BaseException as error:  # noqa: BLE001 - delivered to the Rust side
            try:
                response.set_exception(error)
            except BaseException:  # noqa: BLE001 - completion already failed
                pass

    def healthy(self) -> bool:
        with self._lock:
            if self._closed:
                return False
            future = self._pool.submit(self._executor.healthy)
        return bool(future.result())

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            close = self._pool.submit(self._executor.close)
        # Close is the final task, after accepted submissions and on their
        # execution thread. Python's waits release the GIL.
        try:
            close.result()
        finally:
            self._pool.shutdown(wait=True, cancel_futures=False)
