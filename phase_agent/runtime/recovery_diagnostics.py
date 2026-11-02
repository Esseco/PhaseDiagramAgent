"""Current return failure, independent of scientific facts and conversational memory."""

from pathlib import Path
from phase_agent.tools.step_runner.file_protocol import read_json, write_json


def _path(state_path):
    path = Path(state_path)
    return path.with_name(path.name + ".recovery-error.json")


def _diagnostic(state_path):
    try:
        row = read_json(_path(state_path), {})
    except (OSError, ValueError, TypeError):
        return {"warning": "回收诊断记录无法读取"}
    if row == {} or row is None:
        return {}
    if not isinstance(row, dict) or not isinstance(row.get("error"), str):
        return {"warning": "回收诊断记录格式无效"}
    if row.get("state_path") != str(Path(state_path).resolve()):
        return {"warning": "回收诊断记录属于旧项目路径"}
    return row


def saved_recovery_error(state_path):
    return _diagnostic(state_path).get("error")


def recovery_diagnostic_notice(state_path):
    row = _diagnostic(state_path)
    if row.get("error"):
        from phase_agent.graphs.dialogue.decision import recovery_notice

        return recovery_notice(row["error"])
    if row.get("warning"):
        return row["warning"] + "；本次依据当前任务事实答复。合法回收重试可重新生成诊断。"
    return None


def record_recovery_error(state_path, error, fingerprint):
    write_json(
        _path(state_path),
        {
            "state_path": str(Path(state_path).resolve()),
            "error": str(error)[:500],
            "fingerprint": fingerprint,
        },
    )


def clear_recovery_error(state_path):
    _path(state_path).unlink(missing_ok=True)
