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
    # An approval/revision is bound to the durable action id.  A Studio
    # message id is intentionally used for ordinary turns, but using it here
    # would make the approval look like an unrelated lifecycle request and
    # could route it into the wrong checkpoint.
    if human_feedback is not None:
        return action_id
    identity = _message_identity.get()
    if identity:
        return "webui-" + identity
    return action_id


@contextmanager
def dialogue_message():
    """Give direct chat each turn an identity; preserve Studio receipt identity."""
    if _message_identity.get():
        yield
    else:
        with scientific_message(uuid.uuid4().hex):
            yield
