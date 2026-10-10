"""Invocation-local display preference, never a scientific configuration change."""

from contextlib import contextmanager
from contextvars import ContextVar

_detail = ContextVar("response_detail", default="brief")


def detailed_response():
    return _detail.get() == "detailed"


@contextmanager
def response_detail(value):
    if value not in {"brief", "detailed"}:
        raise ValueError("response_detail must be brief or detailed")
    token = _detail.set(value)
    try:
        yield
    finally:
        _detail.reset(token)
