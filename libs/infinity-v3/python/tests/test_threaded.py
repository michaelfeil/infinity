import threading
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
        release.set()
        self.assertEqual(wait(response).result, "done")
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
        responses = [Response() for _ in range(3)]
        for response in responses:
            executor.submit("embed", response, {})
        ids = {wait(response).result for response in responses}
        self.assertEqual(len(threads), 1)
        self.assertEqual(ids, threads)
        executor.close()

    def test_plugin_exception_completes_response(self):
        def submit(route, response, payload):
            raise RuntimeError("submit exploded")

        executor = ThreadedExecutor(Executor(submit))
        response = wait_after_submit(executor)
        self.assertIsInstance(response.error, RuntimeError)
        self.assertIn("submit exploded", str(response.error))
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
        self.assertIn("closed", str(wait_after_submit(executor).error))

    def test_close_drains_queue_and_rejects_late_submit(self):
        started = threading.Event()
        release = threading.Event()

        def submit(route, response, payload):
            started.set()
            assert release.wait(2), "release gate timed out"
            response.set_result(payload["row"])

        executor = ThreadedExecutor(Executor(submit))
        first = Response()
        second = Response()
        executor.submit("embed", first, {"row": 1})
        executor.submit("embed", second, {"row": 2})
        self.assertTrue(started.wait(1))

        close_started = threading.Event()

        def close():
            close_started.set()
            executor.close()

        closer = threading.Thread(target=close)
        closer.start()
        self.assertTrue(close_started.wait(1), "close did not start")
        self.assertTrue(closer.is_alive(), "close did not wait for queued work")
        release.set()
        closer.join(2)
        self.assertFalse(closer.is_alive())
        self.assertEqual(wait(first).result, 1)
        self.assertEqual(wait(second).result, 2)

        late = Response()
        executor.submit("embed", late, {"row": 3})
        self.assertIn("closed", str(wait(late).error))
        executor.close()


def wait_after_submit(executor):
    response = Response()
    executor.submit("embed", response, {})
    return wait(response)


if __name__ == "__main__":
    unittest.main()
