"""Bound wall-clock waiting; detached workers perform HTTP only, never actions."""

from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
from queue import Queue, Empty
from threading import Thread, BoundedSemaphore, RLock
from time import monotonic, perf_counter

_slots = BoundedSemaphore(4)
_budget_lock = RLock()


class APIDeadlineExceeded(TimeoutError):
    pass


class APITransportBusy(RuntimeError):
    pass


def read_response(request, socket_timeout, deadline):
    import urllib.request
    from phase_agent.graphs.cancellation import check_cancelled

    if monotonic() >= deadline:
        raise APIDeadlineExceeded()
    if not _slots.acquire(blocking=False):
        raise APITransportBusy()
    result = Queue(maxsize=1)

    def network_only():
        try:
            with urllib.request.urlopen(
                request, timeout=min(socket_timeout, max(0.001, deadline - monotonic()))
            ) as response:
                result.put((True, response.read()))
        except BaseException as error:
            result.put((False, error))
        finally:
            _slots.release()

    Thread(target=network_only, daemon=True, name="bounded-api-http").start()
    while True:
        check_cancelled()
        remaining = deadline - monotonic()
        if remaining <= 0:
            raise APIDeadlineExceeded()
        try:
            ok, value = result.get(timeout=min(0.2, remaining))
        except Empty:
            continue
        if monotonic() >= deadline:
            raise APIDeadlineExceeded()
        if ok:
            return value
        raise value


# Model-only limits. No scientific tool or side-effect retry belongs here.

_model_round = ContextVar("model_round", default=None)
_model_call = ContextVar("model_call", default=None)


class ModelRepairLimitExceeded(RuntimeError):
    pass


class ModelRoundDeadlineExceeded(TimeoutError):
    pass


def current_model_round():
    from phase_agent.runtime.turn_process import _current

    record = _current.get()
    return _model_round.get() or (
        record.get("_decision_model_round") if record is not None else None
    )


@contextmanager
def model_round(client=None, *, timeout=None, repairs=None):
    """Join one turn when available, otherwise bound this request chain."""
    from phase_agent.runtime.turn_process import _current
    import math

    budget = current_model_round()
    if budget is None:
        duration = (
            timeout if timeout is not None else getattr(client, "decision_timeout_seconds", 120.0)
        )
        limit = repairs if repairs is not None else getattr(client, "decision_repair_limit", 2)
        if type(duration) not in (int, float) or not math.isfinite(duration) or duration <= 0:
            raise ValueError("invalid_model_round_timeout")
        if type(limit) is not int or limit < 0:
            raise ValueError("invalid_model_round_repair_limit")
        budget = {
            "deadline": monotonic() + duration,
            "repair_limit": limit,
            "repairs": 0,
            "calls": [],
            "next_id": 0,
        }
        record = _current.get()
        if record is not None:
            record["_decision_model_round"] = budget
            record["model_call_chain"] = budget["calls"]
    token = _model_round.set(budget)
    try:
        check_model_deadline()
        yield budget
    finally:
        _model_round.reset(token)


def check_model_deadline():
    budget = current_model_round()
    if budget is not None and monotonic() >= budget["deadline"]:
        raise ModelRoundDeadlineExceeded("model_round_deadline_exceeded; no action is authorized")


def consume_model_repair(reason):
    check_model_deadline()
    budget = current_model_round()
    if budget is not None:
        with _budget_lock:
            if budget["repairs"] >= budget["repair_limit"]:
                raise ModelRepairLimitExceeded("model_round_repair_limit_exceeded: " + reason)
            budget["repairs"] += 1


def model_round_operation(callback):
    @wraps(callback)
    def wrapped(client, *args, **kwargs):
        if (
            getattr(client, "_codex_structured", False) is True
            or getattr(client, "_decision_source", None) == "codex"
        ):
            return callback(client, *args, **kwargs)
        with model_round(client):
            return callback(client, *args, **kwargs)

    return wrapped


def traced_model_call(client, request, *, repair=False):
    """Log metadata only; physical API rows carry already-encoded input bytes."""
    if (
        getattr(client, "_codex_structured", False) is True
        or getattr(client, "_decision_source", None) == "codex"
    ):
        return client(request)
    if _model_call.get() is not None and not repair:
        return client(request)
    with model_round(client) as budget:
        with _budget_lock:
            budget["next_id"] += 1
            parent = _model_call.get()
            if repair and parent is None and budget["calls"]:
                parent = budget["calls"][-1]["call_id"]
            row = {
                "call_id": budget["next_id"],
                "parent_call_id": parent,
                "reason": str(request.get("mode") or "model_request"),
                "repair": repair,
                "input_bytes": None,
                "usage": None,
                "invoked": False,
            }
            budget["calls"].append(row)
        token = _model_call.set(row["call_id"])
        started = perf_counter()
        try:
            if repair:
                consume_model_repair(str(request.get("mode") or "proposal_repair"))
            row["invoked"] = True
            result = client(request)
            row["usage"] = result.get("_llm_usage") if isinstance(result, dict) else None
            check_model_deadline()
            row["status"] = "returned"
            return result
        except Exception as error:
            row["status"] = type(error).__name__
            if getattr(error, "llm_usage", None):
                row["usage"] = error.llm_usage
            elif row["usage"]:
                error.llm_usage = row["usage"]
            raise
        finally:
            row["seconds"] = perf_counter() - started
            _model_call.reset(token)


def local_analysis(payload):
    """Saved facts only; never propose executable actions or infer authorization."""
    from phase_agent.runtime.turn_process import _current

    record = _current.get()
    if record and record.get("local_analysis"):
        return record["local_analysis"]
    state = payload.get("state") or {}
    tasks = state.get("tasks") if isinstance(state, dict) else None
    if isinstance(tasks, list) and all(isinstance(t, dict) for t in tasks):
        completed = sum(t.get("status") == "completed" for t in tasks)
        return f"本地记录：共 {len(tasks)} 个任务，已完成 {completed} 个；其余状态需核对回传登记。建议先核对结果完整性和收敛记录，待模型恢复后继续分析，不重复提交计算。"
    return "本地初判：模型未返回有效方案，现有任务状态不能据此判为失败。建议先核对已保存的回传与收敛记录，待服务恢复后继续分析，不重复提交计算。"
