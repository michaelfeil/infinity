import threading
from queue import Queue
import unittest
import weakref
from unittest.mock import patch

from infinity_v3_python.threaded import ThreadedExecutor


class Response:
    def __init__(self):
        self.done = threading.Event()
        self.result = None
        self.error = None

    def set_result(self, result):
        self.result = result
        self.done.set()

    def set_exception(self, error):
        self.error = error
        self.done.set()


class Executor:
    def __init__(self, submit):
        self._submit = submit

    def submit(self, route, response, payload):
        self._submit(route, response, payload)

    def healthy(self):
        return True

    def close(self):
        pass


def wait(response, timeout=2):
    assert response.done.wait(timeout), "response did not complete"
    return response


def start(executor):
    ready = Response()
    threaded = ThreadedExecutor(lambda **_: executor, {}, ready)
    assert wait(ready).error is None
    return threaded


class ThreadedExecutorTests(unittest.TestCase):
    def test_factory_signals_ready_and_uses_the_execution_thread(self):
        state = threading.local()
        entered = threading.Event()
        release = threading.Event()
        closed = []
        destroyed = []
        config = {"device_id": 7, "serving_limits": {"max_batch_tokens": 8192}}

        def factory(**kwargs):
            state.config = kwargs
            state.thread = threading.get_ident()
            entered.set()
            assert release.wait(2), "factory release timed out"
            inner = Executor(lambda _, response, __: response.set_result(
                (state.thread, state.config)
            ))
            inner.healthy = lambda: state.config == config
            inner.close = lambda: closed.append(state.thread)
            weakref.finalize(inner, lambda: destroyed.append(threading.get_ident()))
            return inner

        ready = Response()
        executor = ThreadedExecutor(factory, config, ready)
        try:
            self.assertTrue(entered.wait(1))
            self.assertFalse(ready.done.is_set(), "startup returned before initialization")
            release.set()
            self.assertIsNone(wait(ready).error)
            response = wait_after_submit(executor)
            worker, received = response.result
            self.assertNotEqual(worker, threading.get_ident())
            self.assertEqual(received, config)
            self.assertTrue(executor.healthy())
        finally:
            release.set()
            executor.close()
        self.assertEqual(closed, [worker])
        self.assertEqual(destroyed, [worker])

    def test_eight_factories_initialize_concurrently(self):
        barrier = threading.Barrier(8)

        def factory(**config):
            barrier.wait(timeout=2)
            return Executor(lambda _, response, __: response.set_result(config["device_id"]))

        ready = [Response() for _ in range(8)]
        replicas = []
        try:
            for device in range(8):
                replicas.append(ThreadedExecutor(factory, {"device_id": device}, ready[device]))
            self.assertTrue(all(wait(response).error is None for response in ready))
            self.assertEqual([wait_after_submit(replica).result for replica in replicas], list(range(8)))
        finally:
            for replica in replicas:
                replica.close()

    def test_factory_error_completes_startup_and_allows_cleanup(self):
        workers = []

        def factory(**config):
            workers.append(threading.current_thread())
            raise ValueError("factory exploded")

        ready = Response()
        executor = ThreadedExecutor(factory, {}, ready)
        try:
            self.assertIsInstance(wait(ready).error, ValueError)
            self.assertIn("factory exploded", str(ready.error))
        finally:
            executor.close()
        self.assertFalse(workers[0].is_alive(), "failed startup leaked its worker")
        self.assertFalse(executor.healthy())

    def test_submit_returns_while_plugin_submit_is_blocked(self):
        started = threading.Event()
        release = threading.Event()

        def submit(route, response, payload):
            started.set()
            assert release.wait(2), "release gate timed out"
            response.set_result("done")

        executor = start(Executor(submit))
        response = Response()
        executor.submit("embed", response, {})
        self.assertTrue(started.wait(1), "plugin submit did not start")
        self.assertFalse(response.done.is_set(), "envelope completed before plugin work")
        with self.assertRaisesRegex(RuntimeError, "queue is full"):
            executor.submit("embed", Response(), {})
        release.set()
        self.assertEqual(wait(response).result, "done")
        after = Response()
        executor.submit("embed", after, {})
        self.assertEqual(wait(after).result, "done")
        executor.close()

    def test_two_replicas_overlap(self):
        barrier = threading.Barrier(2)

        def submit(route, response, payload):
            barrier.wait(timeout=2)
            response.set_result(len(payload))

        first = start(Executor(submit))
        second = start(Executor(submit))
        responses = [Response(), Response()]
        first.submit("embed", responses[0], {"row": 1})
        second.submit("embed", responses[1], {"row": 2})
        self.assertEqual([wait(item).result for item in responses], [1, 1])
        first.close()
        second.close()

    def test_worker_thread_is_persistent(self):
        threads = set()

        def submit(route, response, payload):
            threads.add(threading.get_ident())
            response.set_result(threading.get_ident())

        executor = start(Executor(submit))
        ids = []
        for _ in range(3):
            response = Response()
            executor.submit("embed", response, {})
            ids.append(wait(response).result)
        self.assertEqual(len(threads), 1)
        self.assertEqual(set(ids), threads)
        executor.close()

    def test_failures_release_admission(self):
        def submit(route, response, payload):
            raise RuntimeError("submit exploded")

        executor = start(Executor(submit))
        with patch.object(executor._pool, "submit", side_effect=RuntimeError("schedule failed")):
            with self.assertRaisesRegex(RuntimeError, "schedule failed"):
                executor.submit("embed", Response(), {})
        response = wait_after_submit(executor)
        self.assertIsInstance(response.error, RuntimeError)
        self.assertIn("submit exploded", str(response.error))
        self.assertIsInstance(wait_after_submit(executor).error, RuntimeError)
        executor.close()

    def test_busy_until_terminal_completion(self):
        class InvalidResult(Response):
            def set_result(self, result):
                raise ValueError("invalid result")

        pending = Queue()
        executor = start(Executor(lambda _, response, __: pending.put(response)))
        first = InvalidResult()
        executor.submit("embed", first, {})
        active = pending.get(timeout=2)
        self.assertTrue(executor.healthy())  # submit returned; envelope is still pending.
        with self.assertRaisesRegex(RuntimeError, "queue is full"):
            executor.submit("embed", Response(), {})
        with self.assertRaisesRegex(ValueError, "invalid result"):
            active.set_result(None)
        with self.assertRaisesRegex(RuntimeError, "queue is full"):
            executor.submit("embed", Response(), {})
        active.set_exception(ValueError("invalid result"))
        self.assertIsInstance(wait(first).error, ValueError)

        second = Response()
        executor.submit("embed", second, {})
        next_response = pending.get(timeout=2)
        with self.assertRaisesRegex(RuntimeError, "already completed"):
            active.set_result(None)
        with self.assertRaisesRegex(RuntimeError, "queue is full"):
            executor.submit("embed", Response(), {})
        next_response.set_result("ok")
        self.assertEqual(wait(second).result, "ok")
        executor.close()

    def test_legacy_async_plugin_can_complete_later(self):
        def submit(route, response, payload):
            threading.Timer(0.01, response.set_result, args=(payload["row"],)).start()

        executor = start(Executor(submit))
        response = Response()
        executor.submit("embed", response, {"row": 7})
        self.assertEqual(wait(response).result, 7)
        executor.close()

    def test_close_uses_execution_thread_and_runs_once(self):
        state = threading.local()
        closed = []

        def submit(route, response, payload):
            state.thread = threading.get_ident()
            response.set_result(state.thread)

        inner = Executor(submit)
        inner.close = lambda: closed.append(state.thread)
        executor = start(inner)
        response = wait_after_submit(executor)
        executor.close()
        executor.close()
        self.assertEqual(closed, [response.result])

    def test_close_reports_failure_and_stops_accepting_work(self):
        def close():
            raise RuntimeError("cleanup failed")

        inner = Executor(lambda *args: None)
        inner.close = close
        executor = start(inner)
        with self.assertRaisesRegex(RuntimeError, "cleanup failed"):
            executor.close()
        self.assertFalse(executor.healthy())
        with self.assertRaisesRegex(RuntimeError, "closed"):
            executor.submit("embed", Response(), {})

    def test_close_drains_queue_and_rejects_late_submit(self):
        started = threading.Event()
        release = threading.Event()

        def submit(route, response, payload):
            started.set()
            assert release.wait(2), "release gate timed out"
            response.set_result(payload["row"])

        executor = start(Executor(submit))
        first = Response()
        executor.submit("embed", first, {"row": 1})
        self.assertTrue(started.wait(1))
        with self.assertRaisesRegex(RuntimeError, "queue is full"):
            executor.submit("embed", Response(), {"row": 2})

        close_started = threading.Event()

        def close():
            close_started.set()
            executor.close()

        closer = threading.Thread(target=close)
        closer.start()
        self.assertTrue(close_started.wait(1), "close did not start")
        self.assertTrue(closer.is_alive(), "close did not wait for in-flight work")
        release.set()
        closer.join(2)
        self.assertFalse(closer.is_alive())
        self.assertEqual(wait(first).result, 1)
        with self.assertRaisesRegex(RuntimeError, "closed"):
            executor.submit("embed", Response(), {"row": 3})
        executor.close()


def wait_after_submit(executor):
    response = Response()
    executor.submit("embed", response, {})
    return wait(response)


if __name__ == "__main__":
    unittest.main()
