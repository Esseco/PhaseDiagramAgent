"""Small durable preparation markers; scientific state stays out of the journal."""

from contextlib import contextmanager
from contextvars import ContextVar
import json

from phase_agent.tools.state.execution_receipts import _connect, _event, _identity

_execution = ContextVar("input_preparation_execution", default=None)
_STAGES = {"started", "plan_verified", "effects_started", "complete"}


@contextmanager
def execution_scope(state_path, identity):
    token = _execution.set((state_path, identity) if state_path and identity else None)
    try:
        yield
    finally:
        _execution.reset(token)


def preparation_progress(stage, evidence):
    owner = _execution.get()
    if owner is None:
        return  # Standalone offline builders have no execution receipt.
    if stage not in _STAGES:
        raise ValueError("unsupported_preparation_stage")
    state_path, identity = owner
    key, serialized = _identity(state_path, identity)
    payload = {"identity": identity, "evidence": evidence}
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    if len(encoded.encode()) > 16384:
        raise ValueError("preparation_progress_must_be_small")
    connection = _connect(state_path)
    try:
        connection.execute("BEGIN IMMEDIATE")
        receipt = connection.execute(
            "SELECT identity,phase FROM receipts WHERE action_id=?", (key,)
        ).fetchone()
        if receipt != (serialized, "started"):
            raise ValueError("preparation_progress_identity_or_phase_changed")
        event = "input_preparation:" + stage
        previous = connection.execute(
            "SELECT payload FROM execution_events WHERE action_id=? AND event=?", (key, event)
        ).fetchone()
        if previous and json.loads(previous[0]) != payload:
            raise ValueError("preparation_progress_changed")
        _event(connection, key, event, payload)
        connection.commit()
    finally:
        connection.close()
