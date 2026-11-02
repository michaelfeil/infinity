# SPDX-License-Identifier: MIT
# Copyright (c) 2023-now michaelfeil

"""High-level support for working with threads in asyncio"""

import asyncio
import contextvars
import functools
from concurrent.futures import ThreadPoolExecutor

__all__ = ["to_thread"]


async def to_thread(func, tp: ThreadPoolExecutor, /, *args, **kwargs):
    """Asynchronously run function *func* in a separate thread.

    Any *args and **kwargs supplied for this function are directly passed
    to *func*. Also, the current :class:`contextvars.Context` is propagated,
    allowing context variables from the main thread to be accessed in the
    separate thread.

    Return a coroutine that can be awaited to get the eventual result of *func*.
    """
    loop = asyncio.get_running_loop()
    ctx = contextvars.copy_context()
    func_call = functools.partial(ctx.run, func, *args, **kwargs)
    return await loop.run_in_executor(tp, func_call)
