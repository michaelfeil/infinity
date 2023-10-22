"""High-level support for working with threads in asyncio"""

import asyncio
import contextvars
import functools
from concurrent.futures import ThreadPoolExecutor

__all__ = ["to_thread"]


# class EventTS:
#     """Throw-away async event.
#     wait and set once, and forget.

#     Save only for one reading and one writing thread.
#     """

#     def __init__(self, tp=None):
#         self._waiter = None
#         self._value = False

#     def is_set(self):
#         """Return True if and only if the internal flag is true."""
#         return self._value

#     def set(self):
#         """Set the internal flag to true. All coroutines waiting for it to
#         become true are awakened. Coroutine that call wait() once the flag is
#         true will not block at all.
#         """
#         if not self._value:
#             self._value = True

#             if self._waiter:
#                 self._waiter.set_result(True)

#     async def wait(self):
#         """Block until the internal flag is true.

#         If the internal flag is true on entry, return True
#         immediately.  Otherwise, block until another coroutine calls
#         set() to set the flag to true, then return True.
#         """
#         if self._value:
#             return True

#         fut = asyncio.events._get_running_loop().create_future()
#         self._waiter = fut
#         try:
#             await fut
#             return True
#         finally:
#             self._waiter = None


async def to_thread(func, tp: ThreadPoolExecutor, /, *args, **kwargs):
    """Asynchronously run function *func* in a separate thread.

    Any *args and **kwargs supplied for this function are directly passed
    to *func*. Also, the current :class:`contextvars.Context` is propagated,
    allowing context variables from the main thread to be accessed in the
    separate thread.

    Return a coroutine that can be awaited to get the eventual result of *func*.
    """
    loop = asyncio.events.get_running_loop()
    ctx = contextvars.copy_context()
    func_call = functools.partial(ctx.run, func, *args, **kwargs)
    return await loop.run_in_executor(tp, func_call)
