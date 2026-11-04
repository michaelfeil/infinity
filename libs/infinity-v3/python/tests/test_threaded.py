import threading
from queue import Queue
import unittest

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


class ThreadedExecutorTests(unittest.TestCase):
    def test_submit_returns_while_plugin_submit_is_blocked(self):
        started = threading.Event()
        release = threading.Event()

        def submit(route, response, payload):
            started.set()
            assert release.wait(2), "release gate timed out"
            response.set_result("done")

        executor = ThreadedExecutor(Executor(submit))
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

        first = ThreadedExecutor(Executor(submit))
        second = ThreadedExecutor(Executor(submit))
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

        executor = ThreadedExecutor(Executor(submit))
        ids = []
        for _ in range(3):
            response = Response()
            executor.submit("embed", response, {})
            ids.append(wait(response).result)
        self.assertEqual(len(threads), 1)
        self.assertEqual(set(ids), threads)
        executor.close()

    def test_plugin_exception_completes_response(self):
        def submit(route, response, payload):
            raise RuntimeError("submit exploded")

        executor = ThreadedExecutor(Executor(submit))
        response = wait_after_submit(executor)
        self.assertIsInstance(response.error, RuntimeError)
        self.assertIn("submit exploded", str(response.error))
        self.assertIsInstance(wait_after_submit(executor).error, RuntimeError)
        executor.close()

    def test_legacy_async_pending_rejects_until_callback_delivers(self):
        release = threading.Event()

        def submit(route, response, payload):
            def complete():
                assert release.wait(2), "release gate timed out"
                response.set_result(payload["row"])

            threading.Timer(0, complete).start()

        executor = ThreadedExecutor(Executor(submit))
        first = Response()
        executor.submit("embed", first, {"row": 7})
        with self.assertRaisesRegex(RuntimeError, "queue is full"):
            executor.submit("embed", Response(), {"row": 8})
        release.set()
        self.assertEqual(wait(first).result, 7)
        after = Response()
        executor.submit("embed", after, {"row": 9})
        self.assertEqual(wait(after).result, 9)
        executor.close()

    def test_scheduling_failure_releases_single_flight_guard(self):
        executor = ThreadedExecutor(
            Executor(lambda route, response, payload: response.set_result("ok"))
        )
        original = executor._pool.submit
        executor._pool.submit = lambda *args, **kwargs: (_ for _ in ()).throw(
            RuntimeError("schedule failed")
        )
        with self.assertRaisesRegex(RuntimeError, "schedule failed"):
            executor.submit("embed", Response(), {})
        executor._pool.submit = original
        self.assertEqual(wait_after_submit(executor).result, "ok")
        executor.close()

    def test_repeated_completion_does_not_release_another_request(self):
        pending = Queue()
        executor = ThreadedExecutor(Executor(lambda _, response, __: pending.put(response)))
        first = Response()
        executor.submit("embed", first, {})
        completed = pending.get(timeout=2)
        completed.set_result(4)
        self.assertEqual(wait(first).result, 4)
        second = Response()
        executor.submit("embed", second, {})
        active = pending.get(timeout=2)
        with self.assertRaisesRegex(RuntimeError, "already completed"):
            completed.set_exception(RuntimeError("late"))
        with self.assertRaisesRegex(RuntimeError, "queue is full"):
            executor.submit("embed", Response(), {})
        active.set_result(5)
        self.assertEqual(wait(second).result, 5)
        executor.close()

    def test_invalid_result_keeps_admission_until_error_completion(self):
        class InvalidResult(Response):
            def set_result(self, result):
                raise ValueError("invalid result")

        pending = Queue()
        executor = ThreadedExecutor(Executor(lambda _, response, __: pending.put(response)))
        response = InvalidResult()
        executor.submit("embed", response, {})
        active = pending.get(timeout=2)
        with self.assertRaisesRegex(ValueError, "invalid result"):
            active.set_result(None)
        with self.assertRaisesRegex(RuntimeError, "queue is full"):
            executor.submit("embed", Response(), {})
        active.set_exception(ValueError("invalid result"))
        self.assertIsInstance(wait(response).error, ValueError)
        after = Response()
        executor.submit("embed", after, {})
        pending.get(timeout=2).set_result("ok")
        self.assertEqual(wait(after).result, "ok")
        executor.close()

    def test_legacy_async_plugin_can_complete_later(self):
        def submit(route, response, payload):
            threading.Timer(0.01, response.set_result, args=(payload["row"],)).start()

        executor = ThreadedExecutor(Executor(submit))
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
        executor = ThreadedExecutor(inner)
        response = wait_after_submit(executor)
        executor.close()
        executor.close()
        self.assertEqual(closed, [response.result])

    def test_close_reports_failure_and_stops_accepting_work(self):
        def close():
            raise RuntimeError("cleanup failed")

        inner = Executor(lambda *args: None)
        inner.close = close
        executor = ThreadedExecutor(inner)
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

        executor = ThreadedExecutor(Executor(submit))
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
