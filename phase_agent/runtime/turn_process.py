"""Read-only, bounded per-turn evidence for users; never model hidden reasoning."""

from contextlib import contextmanager

from contextvars import ContextVar

from datetime import datetime, timezone

from html import escape

from pathlib import Path

import json

import re

import uuid

from time import monotonic

from functools import wraps


_current = ContextVar("turn_process", default=None)


def clean(value, depth=0):

    if depth > 8:
        return "…"

    if isinstance(value, dict):
        return {
            str(k): clean(v, depth + 1)
            for k, v in list(value.items())[:40]
            if not any(
                word in str(k).lower()
                for word in (
                    "key",
                    "token",
                    "password",
                    "credential",
                    "secret",
                    "authorization",
                )
            )
        }

    if isinstance(value, (list, tuple)):
        return [clean(v, depth + 1) for v in value[:30]]

    if isinstance(value, str):
        value = re.sub(r"(?i)\bBearer\s+\S+|\bsk-[A-Za-z0-9_-]+", "[已隐藏凭据]", value)

        value = re.sub(
            r"(?i)((?:api[_ -]?key|password|secret|token|密码)\s*[:=：]\s*)\S+",
            r"\1[已隐藏凭据]",
            value,
        )

        return value[:4000]

    return value if value is None or isinstance(value, (bool, int, float)) else type(value).__name__


def reuse_initial_state(path):
    """Transfer ownership of the initial parsed state once, after a freshness check."""
    record = _current.get()
    cached = record.get("_initial_state") if record is not None else None
    if not cached or str(Path(path).resolve()) != cached["path"]:
        return False, None
    record.pop("_initial_state", None)
    try:
        stat = Path(path).stat()
        if (stat.st_size, stat.st_mtime_ns, stat.st_ino) == cached["signature"]:
            return True, cached["value"]
    except OSError:
        pass
    return False, None


def process_event(title, data):

    record = _current.get()

    if record is not None and len(record["events"]) < 30:
        record["events"].append({"title": title, "data": clean(data)})


def record_timing(name, seconds, *, category="node", status="completed", mode=None):

    record = _current.get()

    if record is not None:
        rows = record.setdefault("timings", [])
        important = category in {"processing", "training_inputs", "model"} or status == "failed"
        if not important and len(rows) >= 64:
            return
        if len(rows) >= 80:
            replace = next(
                (
                    i
                    for i, row in enumerate(rows)
                    if row.get("category") not in {"processing", "training_inputs", "model"}
                    and row.get("status") != "failed"
                ),
                None,
            )
            if replace is None:
                return
            rows.pop(replace)
        rows.append(
            {
                "name": name,
                "category": category,
                "seconds": round(seconds, 3),
                "status": status,
                "mode": mode,
            }
        )


def timed_call(name, callback, *args, category="node", mode=None, **kwargs):

    started = monotonic()
    from phase_agent.runtime.turn_progress import start_step, end_step

    record = _current.get()
    progress_token = start_step(record["id"], name, category) if record is not None else None

    status = "failed"

    try:
        result = callback(*args, **kwargs)

        status = "completed"

        return result

    finally:
        if record is not None:
            end_step(record["id"], progress_token)
        record_timing(name, monotonic() - started, category=category, status=status, mode=mode)


def timed_node(name, node):

    @wraps(node)
    def measured(state, runtime):

        return timed_call(name, node, state, runtime)

    return measured


def proposal_summary(action):

    if not isinstance(action, dict):
        return {"result_type": type(action).__name__}

    return {
        k: action[k]
        for k in (
            "kind",
            "answer",
            "tool",
            "target_ids",
            "parameters",
            "budget",
            "reason",
            "expected_purpose",
            "evidence_refs",
            "post_dft_review",
            "round_budget_review",
        )
        if k in action
    }


@contextmanager
def record_turn(runtime_config, message, thread_id):

    config_path = Path(runtime_config)

    root = config_path.resolve().parent

    record = {
        "id": uuid.uuid4().hex,
        "time": datetime.now(timezone.utc).isoformat(),
        "thread_id": thread_id,
        "message": clean(message),
        "events": [],
        "timings": [],
        "model_requests": [],
        "status": "进行中",
    }

    started = monotonic()

    token = _current.set(record)
    from phase_agent.runtime.turn_progress import start_turn, end_turn

    start_turn(root, record["id"])

    try:
        try:
            settings = json.loads(config_path.read_text(encoding="utf-8"))

            state_path = root / settings.get("state_path", "workflow_state/state.json")

            if state_path.is_file() and state_path.resolve().is_relative_to(root):
                before_stat = state_path.stat()
                state = timed_call(
                    "读取项目状态摘要",
                    lambda: json.loads(state_path.read_text(encoding="utf-8")),
                    category="io",
                )

                after_stat = state_path.stat()

                def signature(st):
                    return st.st_size, st.st_mtime_ns, st.st_ino

                if signature(before_stat) == signature(after_stat):
                    record["_initial_state"] = {
                        "path": str(state_path.resolve()),
                        "signature": signature(after_stat),
                        "value": state,
                    }

                tasks = state.get("tasks") or []
                completed = sum(t.get("status") == "completed" for t in tasks)
                failed = sum(t.get("status") == "failed" for t in tasks)
                record["local_analysis"] = (
                    f"本地记录：共 {len(tasks)} 个任务，已完成 {completed} 个，失败 {failed} 个。"
                    "未完成记录可能包含尚未登记的回传，不能据此重提计算。"
                    "建议先核对回传完整性、弛豫收敛和候选覆盖；模型恢复后再给出科学方案。"
                )
                from phase_agent.runtime.turn_progress import set_local_analysis

                set_local_analysis(record["id"], record["local_analysis"])
                process_event(
                    "读取的项目状态",
                    {
                        "配置版本": state.get("confirmed_config_version"),
                        "任务数量": len(state.get("tasks") or []),
                        "branch数量": len(state.get("branches") or []),
                        "待审批方案数量": len(state.get("pending_execution_policies") or {}),
                    },
                )

                del state, tasks

        except (OSError, ValueError, TypeError, AttributeError):
            process_event("读取的项目状态", {"提示": "未能读取状态摘要，以业务检查结果为准"})

        yield record

        record["status"] = "已返回答复（执行结果以业务记录为准）"

    except Exception as error:
        process_event("本轮异常", {"type": type(error).__name__, "message": str(error)})

        record["status"] = "异常结束"
        failed = next(
            (row for row in reversed(record["timings"]) if row.get("status") == "failed"),
            {},
        )
        record["failure_summary"] = {
            "type": type(error).__name__,
            "stage": failed.get("name", "本轮处理"),
        }
        code = getattr(error, "status_code", None)
        if type(code) is int:
            record["failure_summary"]["http_status"] = code

        raise

    finally:
        end_turn(record["id"])
        from phase_agent.runtime.token_usage import usage_summary

        record["token_usage"] = usage_summary(record)
        record.pop("_model_request_cache", None)
        record.pop("_initial_state", None)
        record["duration_seconds"] = round(monotonic() - started, 3)

        _current.reset(token)
        try:
            from phase_agent.runtime.runtime_diagnostics import observe_completed

            observe_completed(runtime_config, record)
        except Exception:
            pass  # Optional diagnostics cannot change a business outcome.

        try:
            directory = root / "workflow_state" / "dialogue_process"

            directory.mkdir(parents=True, exist_ok=True)

            target = directory / (record["id"] + ".json")

            target.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")

            # Diagnostics are retained; no automatic file deletion.

        except OSError:
            pass  # Diagnostics cannot turn a successful business action into a failure.


def render_process_panel(runtime_config):

    root = Path(runtime_config).resolve().parent
    from phase_agent.runtime.turn_progress import active_turns

    progress = active_turns(runtime_config)

    files = sorted(
        (root / "workflow_state" / "dialogue_process").glob("*.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )[:20]

    from phase_agent.runtime.runtime_diagnostics import (
        render_diagnostics,
        monitoring_active,
    )

    diagnostic_card = render_diagnostics(runtime_config)
    cards = [
        "<section><h2>本轮正在处理</h2><p>当前阶段："
        + escape(item["label"])
        + "；本阶段 "
        + str(item["step_seconds"])
        + " 秒；本轮累计 "
        + str(item["total_seconds"])
        + " 秒。</p><p>"
        + escape(item.get("local_analysis", ""))
        + "</p><p>本页每 3 秒刷新；不会批准或执行新动作。</p></section>"
        for item in progress
    ]

    for path in files:
        try:
            record = json.loads(path.read_text(encoding="utf-8"))

        except (OSError, ValueError):
            continue

        events = []

        usage = record.get("token_usage")
        if usage is not None:
            events.append(
                "<p>本轮 token：输入 "
                + str(usage["input_tokens"])
                + "，输出 "
                + str(usage["output_tokens"])
                + "；API 请求 "
                + str(usage["api_calls"])
                + " 次，重试 "
                + str(usage["retries"])
                + " 次。"
                + (
                    "另有 " + str(usage["unknown_usage_calls"]) + " 次用量未知，未按零计算。"
                    if usage["unknown_usage_calls"]
                    else ""
                )
                + "</p>"
            )
        timings = record.get("timings") or []

        if timings:
            models = sum(row.get("category") == "model" for row in timings)

            events.append(
                "<p>本轮总耗时："
                + escape(str(record.get("duration_seconds", "未知")))
                + " 秒；模型调用："
                + str(models)
                + " 次。</p><details><summary>节点与模型耗时</summary><pre>"
                + escape(
                    json.dumps(
                        sorted(timings, key=lambda row: row.get("seconds", 0), reverse=True),
                        ensure_ascii=False,
                        indent=2,
                    )
                )
                + "</pre><p>父节点包含子步骤耗时，不能相加作为总耗时。</p></details>"
            )

        for event in record.get("events", []):
            events.append(
                "<details><summary>"
                + escape(str(event["title"]))
                + "</summary><pre>"
                + escape(json.dumps(event["data"], ensure_ascii=False, indent=2))
                + "</pre></details>"
            )

        cards.append(
            "<section><h2>"
            + escape(str(record.get("message", "")))
            + "</h2><p>"
            + escape(str(record.get("time", "")) + " · " + str(record.get("status", "")))
            + "</p>"
            + "".join(events)
            + "</section>"
        )

    return (
        """<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">

<title>本轮过程</title><style>body{font:16px/1.6 system-ui;background:#f4f6f8;color:#203040;max-width:1000px;margin:32px auto;padding:0 20px}section,details{background:white;padding:16px;border:1px solid #dce3ea;border-radius:10px;margin:12px 0}pre{white-space:pre-wrap;overflow-wrap:anywhere}summary{cursor:pointer}h2{font-size:18px;overflow-wrap:anywhere}</style>

<h1>本轮过程</h1><p>查看实际读取的状态、模型提出的方案、校验错误及修正结果。这里只展示可核对的记录，不展示模型内部思维链。</p>

<p><a href="/phase/conversations">暂停／删除Studio对话</a> · <a href="/phase/process">刷新过程</a> · <a href="/phase/flow">科学流程与当前方案</a></p>

<p>依次展开：读取的项目状态 → 模型方案 → 方案校验 → 修正方案 → 本轮答复。出现校验错误表示该方案尚未通过；是否执行请核对最终答复和任务记录。本页不会重跑或批准任何操作。</p>"""
        + (
            '<meta http-equiv="refresh" content="3">'
            if progress
            else '<meta http-equiv="refresh" content="30">'
            if monitoring_active(runtime_config)
            else ""
        )
        + (
            diagnostic_card + "".join(cards)
            or "<section>暂无过程记录。更新并启动服务后，在 Studio 发送一条新消息，再刷新本页。历史对话不会被补写成调试记录。</section>"
        )
        + "</html>"
    )
