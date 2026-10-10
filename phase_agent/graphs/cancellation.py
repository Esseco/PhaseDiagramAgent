"""Cooperative stop propagated to scientific workers; no thread is killed mid-write."""

import asyncio
from contextlib import contextmanager
from contextvars import ContextVar
from threading import Event

_signal = ContextVar("scientific_stop_signal", default=None)


class ScientificRunCancelled(BaseException):
    pass


@contextmanager
def cancellation_scope(signal):
    token = _signal.set(signal)
    try:
        yield
    finally:
        _signal.reset(token)


def check_cancelled():
    signal = _signal.get()
    if signal is not None and signal.is_set():
        raise ScientificRunCancelled("Studio run was paused or deleted")


async def cancellable_worker(callback, *args):
    signal = Event()

    def worker():
        with cancellation_scope(signal):
            check_cancelled()
            result = callback(*args)
            check_cancelled()
            return result

    task = asyncio.create_task(asyncio.to_thread(worker))
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        signal.set()
        # Do not release the run/lock while its old worker can still dispatch work.
        while not task.done():
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError:
                continue
            except BaseException:
                break
        if task.done() and not task.cancelled():
            try:
                task.result()
            except BaseException:
                pass
        raise
