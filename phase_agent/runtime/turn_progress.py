"""Small in-memory progress snapshots, independent of the message worker lock."""

from pathlib import Path
import threading
from time import monotonic
import uuid

_lock = threading.RLock()
_active = {}

LABELS = {
    "read_project": "读取项目状态",
    "inspection": "检查当前项目",
    "config_import": "核对配置",
    "dft_recovery": "检查 DFT 回传",
    "export": "准备结果导出",
    "finetune": "检查模型更新",
    "mc": "检查 MC 状态",
    "rerun": "核对已确认的历史重做",
    "migration": "检查状态兼容",
    "restore_history": "读取对话记录",
    "route_request": "准备本轮请求",
    "decide": "准备 Agent 决策",
    "read_evidence": "读取科学证据",
    "review": "核对审批",
    "scientific_workflow": "执行已批准动作",
    "present": "整理本轮答复",
    "configure": "处理配置方案",
    "prepare_context": "准备上下文",
    "model": "核对当前模型",
    "scientific_feedback": "分析计算结果",
    "results_wait_gate": "核对回传完整性",
    "assess_and_export_round": "整理本轮结果",
    "bounded_action_graph": "执行已批准动作",
    "prepare_inputs_and_finalize": "准备输入并汇总",
}


def start_turn(root, turn_id):
    with _lock:
        _active[turn_id] = {
            "root": str(Path(root).resolve()),
            "started": monotonic(),
            "thread_id": threading.get_ident(),
            "updated": monotonic(),
            "steps": [],
        }


def set_local_analysis(turn_id, analysis):
    with _lock:
        if turn_id in _active:
            _active[turn_id]["local_analysis"] = analysis


def end_turn(turn_id):
    with _lock:
        _active.pop(turn_id, None)


def start_step(turn_id, name, category="node"):
    with _lock:
        item = _active.get(turn_id)
        if item is None:
            return None
        token = uuid.uuid4().hex
        label = (
            ("等待模型响应 · " + name)
            if category == "model"
            else "读取项目文件"
            if category == "io"
            else LABELS.get(name, name)
        )
        item["updated"] = monotonic()
        item["steps"].append(
            {
                "token": token,
                "label": label,
                "name": name,
                "category": category,
                "thread_id": threading.get_ident(),
                "started": monotonic(),
            }
        )
        return token


def end_step(turn_id, token):
    with _lock:
        item = _active.get(turn_id)
        if item is not None:
            item["steps"] = [s for s in item["steps"] if s["token"] != token]
            item["updated"] = monotonic()


def active_turns(runtime_config):
    root = str(Path(runtime_config).resolve().parent)
    now = monotonic()
    with _lock:
        result = []
        for item in _active.values():
            if item["root"] != root:
                continue
            step = item["steps"][-1] if item["steps"] else None
            result.append(
                {
                    "label": step["label"] if step else "处理本轮请求",
                    "local_analysis": item.get("local_analysis", ""),
                    "name": step["name"] if step else "turn",
                    "category": step["category"] if step else "node",
                    "thread_id": step["thread_id"] if step else item["thread_id"],
                    "step_seconds": round(
                        now - (step["started"] if step else item.get("updated", item["started"])), 1
                    ),
                    "total_seconds": round(now - item["started"], 1),
                    "last_progress_seconds": round(now - item.get("updated", item["started"]), 1),
                }
            )
        return result
