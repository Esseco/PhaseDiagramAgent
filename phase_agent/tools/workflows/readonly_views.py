"""Request-local read-only presentation mode; never changes scientific authority."""

from contextvars import ContextVar
from functools import wraps

_read_only = ContextVar("phase_read_only_presentation", default=False)


def is_read_only():
    return _read_only.get()


def read_only_view(callback):
    @wraps(callback)
    def read(*args, **kwargs):
        token = _read_only.set(True)
        try:
            return callback(*args, **kwargs)
        finally:
            _read_only.reset(token)

    return read
