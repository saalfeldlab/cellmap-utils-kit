"""cellmap_utils_kit.parallel_utils: Utility functions for parallelization."""

import asyncio
import functools
from typing import Callable, ParamSpec, TypeVar

P = ParamSpec("P")
R = TypeVar("R")


def background(f: Callable[P, R]) -> Callable[P, asyncio.Future[R]]:
    """A decorator that allows a function to be run in a separate thread,
    allowing it to be executed asynchronously.

    This decorator wraps a function and sumits it to an executor, which allows
    the function to run in a non-blocking way. It can accept any arguments and
    keyword arguments and returns an asynctio.Future that can be awaited.

    Args:
        f (Callable[P, R]): The function to be decorated. It can accept any
            parameters and return any type.

    Returns:
        Callable[P, asyncio.Future[R]]: A wrapped function that, when called,
            executes the original function in a separate thread and returns an
            asyncio.Future

    """

    def wrapped(*args: P.args, **kwargs: P.kwargs) -> asyncio.Future[R]:
        ff: Callable[..., R]  # generic callable
        if kwargs:
            ff = functools.partial(f, **kwargs)
        else:
            ff = f
        return asyncio.get_event_loop().run_in_executor(None, ff, *args)

    return wrapped
