"""Studio threads share a local project; legacy HTTP sessions keep their guard."""

from contextlib import contextmanager
from contextvars import ContextVar

_project_session = ContextVar("studio_project_session", default=False)
_history = ContextVar("studio_thread_history", default=None)


def is_project_session():
    return _project_session.get()


def thread_history():
    return _history.get()


@contextmanager
def studio_project_session(messages=None):
    token = _project_session.set(True)
    history_token = _history.set(messages) if messages is not None else None
    try:
        yield
    finally:
        if history_token is not None:
            _history.reset(history_token)
        _project_session.reset(token)
