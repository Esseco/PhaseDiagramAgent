"""Bound wall-clock waiting; detached workers perform HTTP only, never actions."""

from queue import Queue, Empty
from threading import Thread, BoundedSemaphore
from time import monotonic

_slots = BoundedSemaphore(4)


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
