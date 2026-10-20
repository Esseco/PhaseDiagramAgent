"""Small typed failure facts. No intent routing, retries, raw payloads or secrets."""

from datetime import datetime, timezone
import errno
import json
import logging
from pathlib import Path
import traceback
from uuid import uuid4

from filelock import Timeout as LockTimeout


def describe_failure(error, *, boundary, object_id=None):
    code, message, retry = (
        "internal_error",
        "程序内部异常，已保留故障位置；先检查执行收据，不能假定未执行",
        False,
    )
    if isinstance(error, json.JSONDecodeError):
        code, message = "invalid_json", "记录不是完整有效的 JSON；保留原文件并核对备份或重新回传"
    elif isinstance(error, FileNotFoundError):
        code, message = "missing_file", "所需文件缺失；核对登记路径或回传原结果，不要重投计算"
    elif isinstance(error, PermissionError):
        code, message = "permission_denied", "文件访问被拒绝；检查目录权限和文件占用后核对原事务"
    elif isinstance(error, OSError) and error.errno == errno.ENOSPC:
        code, message = "disk_full", "磁盘空间不足；释放空间后核对原事务，禁止重放计算"
    elif isinstance(error, LockTimeout):
        code, message = "project_busy", "项目锁被占用；先查看后台进度，本次不得并发推进"
    elif isinstance(error, (TimeoutError, ConnectionError)):
        code, message = (
            "dependency_unavailable",
            "依赖连接或等待失败；外部执行状态未知，先核对原作业与收据",
        )
    elif isinstance(error, ValueError):
        code, message = (
            "invalid_data",
            "数据或绑定校验未通过；核对诊断对象及原始凭据，不自动修复科学数据",
        )
    if boundary in {"presentation", "operation_record", "diagnostics"}:
        retry = True  # Only a read; never authority to replay the failed business action.
    return {
        "incident_id": uuid4().hex,
        "code": code,
        "boundary": boundary,
        "object_id": object_id,
        "message": message,
        "error_type": type(error).__name__,
        "retry_read_only": retry,
        "automatic_retry": False,
        "execution_effect": "unknown"
        if boundary in {"control_operation", "scientific_action"}
        else "not_inferred",
        "diagnostics_url": "/phase/diagnostics",
        "recovery_url": "/phase/approval",
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


def record_failure(state_path, error, *, boundary, object_id=None):
    fact = describe_failure(error, boundary=boundary, object_id=object_id)
    fact["locations"] = [
        {"file": Path(frame.filename).name, "function": frame.name, "line": frame.lineno}
        for frame in traceback.extract_tb(error.__traceback__)[-12:]
    ]
    if state_path is None:
        fact["evidence_saved"] = False
        return fact
    try:
        from phase_agent.tools.step_runner.file_protocol import _write_json

        path = Path(state_path).parent / "diagnostics/incidents" / (fact["incident_id"] + ".json")
        fact["evidence_file"] = str(path)
        _write_json(path, fact)
        fact["evidence_saved"] = True
    except Exception:
        # An unavailable disk/diagnostic must not replace the original failure.
        fact["evidence_saved"] = False
        logging.getLogger(__name__).warning(
            "Cannot persist incident %s (%s)", fact["incident_id"], fact["code"]
        )
    return fact
