"""Carry Studio message identity into the scientific invocation."""

from contextvars import ContextVar
from contextlib import contextmanager
import uuid

_message_identity = ContextVar("scientific_message_identity", default=None)


@contextmanager
def scientific_message(identity):
    token = _message_identity.set(identity)
    try:
        yield
    finally:
        _message_identity.reset(token)


def new_invocation(prefix="webui"):
    identity = _message_identity.get()
    return f"{prefix}-{identity}" if identity else f"{prefix}-{uuid.uuid4().hex}"


def lifecycle_request_id(action_id, human_feedback=None):
    """Message identity differs from the immutable action being approved."""
    identity = _message_identity.get()
    if identity:
        return "webui-" + identity
    if human_feedback is None:
        return action_id
    import hashlib
    import json

    digest = hashlib.sha256(
        json.dumps(human_feedback, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()[:20]
    return f"{action_id}:review:{digest}"


@contextmanager
def dialogue_message():
    """Give direct chat each turn an identity; preserve Studio receipt identity."""
    if _message_identity.get():
        yield
    else:
        with scientific_message(uuid.uuid4().hex):
            yield
