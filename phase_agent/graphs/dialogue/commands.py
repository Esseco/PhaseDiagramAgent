"""Explicit control commands and pure presentation helpers; no model routing."""

import json
from phase_agent.tools.step_runner.build_status_summary import build_status_summary
from phase_agent.graphs.dialogue.errors import OpenWebUIRequestError
from pathlib import Path
from copy import deepcopy
import os
import uuid
from phase_agent.graphs.invocation_context import new_invocation
from phase_agent.tools.step_runner.file_protocol import read_json, write_json
from phase_agent.runtime.workflow_reply_presentation import (
    format_workflow_reply,
    _is_model_failure_proposal,
)


def classify_user_decision(message: str) -> str:
    """Only a final, exact user line can approve or reject; all else is feedback."""
    lines = [line.strip() for line in str(message).splitlines() if line.strip()]
    final = lines[-1].lower() if lines else ""
    if final in {"approve", "同意"}:
        return "approve"
    if final in {"reject", "拒绝"}:
        return "reject"
    return "comment"


def _is_config_migration_approval(message: str) -> bool:
    """Recognize a direct migration approval without treating questions as consent."""
    lines = [line.strip().lower() for line in str(message or "").splitlines() if line.strip()]
    final = lines[-1] if lines else ""
    for mark in ("。", "！", "!", ".", "；", ";"):
        final = final.rstrip(mark).strip()
    if any(term in final for term in ("不批准", "不允许", "不要迁移", "先别迁移", "拒绝迁移")):
        return False
    return final in {"批准迁移", "同意迁移", "确认迁移", "允许迁移", "approve migration"} or any(
        final.endswith(term) for term in ("批准迁移", "同意迁移", "确认迁移", "允许迁移")
    )


def _mentions_explicit_branch_generation(message: str) -> bool:
    text = "".join(str(message or "").lower().split())
    return (
        any(token in text for token in ("branch", "分支"))
        and any(
            token in text
            for token in ("生成", "补充", "新增", "扩展", "generate", "regenerate", "create")
        )
        and not any(
            token in text
            for token in ("不要生成", "不生成", "暂不生成", "先不生成", "取消生成", "别生成")
        )
    )


def _drop_finished_pending(state):
    """Recover approvals left behind by a handler returning an older state copy."""
    pending = state.get("pending_execution_policies") or {}
    if not pending:
        return state, False
    finished = {
        "completed",
        "failed",
        "rejected",
        "rejected_by_user",
        "not_configured",
        "paused",
        "cancelled",
        "budget_exhausted",
    }
    records = {row.get("record_id"): row for row in state.get("action_records") or []}
    invocations = state.get("invocations") or {}
    obsolete = [
        key
        for key, item in pending.items()
        if (invocations.get(key) or records.get(item.get("record_id")) or {}).get("status")
        in finished
    ]
    if not obsolete:
        return state, False
    updated = deepcopy(state)
    for key in obsolete:
        updated["pending_execution_policies"].pop(key, None)
    return updated, True


def format_status_reply(state: dict) -> str:
    from phase_agent.runtime.status_presentation import format_progress

    return format_progress(state)


def format_history_prompt(state: dict, manager=None) -> str:
    summary = build_status_summary(state, config_version=state.get("confirmed_config_version"))
    ledger = getattr(manager, "data", {}) or {}
    actions = state.get("action_records") or state.get("decisions") or []
    latest = actions[-1] if actions else None
    compact = {
        "branches": len(ledger.get("branches") or {}),
        "structures": len(ledger.get("structures") or {}),
        "actions": len(actions),
        "latest_action": (
            {
                key: latest.get(key)
                for key in ("status", "record_id", "final_action")
                if latest.get(key) is not None
            }
            if isinstance(latest, dict)
            else None
        ),
        "tasks": len(state.get("tasks") or state.get("pending_tasks") or []),
        "pending_approvals": len(state.get("pending_execution_policies") or {}),
        "run_status": state.get("run_status") or state.get("status"),
        "config_version": summary.get("config_version"),
    }
    return (
        "检测到已配置的本地历史：\n```json\n"
        + json.dumps(compact, ensure_ascii=False, indent=2)
        + "\n```\n请明确回复 `继续` 或 `新建`。确认前不会提出或执行 action；`继续` 也不会批准待审批 action。"
    )


def _latest_user_message(messages):
    for message in reversed(messages):
        if not isinstance(message, dict) or message.get("role") != "user":
            continue
        content = message.get("content")
        if isinstance(content, str) and content.strip():
            return content.strip()
        if isinstance(content, list):
            text = "\n".join(
                item.get("text", "")
                for item in content
                if isinstance(item, dict) and item.get("type") in {"text", "input_text"}
            )
            if text.strip():
                return text.strip()
    raise OpenWebUIRequestError("messages must contain a non-empty user message")


def _open_webui_metadata_reply(message):
    """Answer Open WebUI helper requests without entering the search workflow."""
    text = str(message or "").lstrip()
    if not text.startswith("### Task:"):
        return None
    lowered = text.lower()
    if "suggest 3-5 relevant follow-up" in lowered or '"follow_ups"' in lowered:
        return '{"follow_ups": []}'
    if "broad tags categorizing" in lowered or '"tags"' in lowered:
        return '{"tags": ["材料相图搜索"]}'
    if "generate a concise" in lowered and "title" in lowered:
        return "相图搜索"
    return None


def _is_status_command(message):
    value = " ".join(str(message).strip().lower().split()).rstrip("？?。.!！")
    return value in {
        "/status",
        "/状态",
        "status",
        "progress",
        "状态",
        "进度",
        "当前状态",
        "当前进度",
        "查看状态",
        "查看进度",
        "查看预算",
        "查看相图",
        "现在什么进度",
        "现在是什么进度",
        "目前什么进度",
        "目前是什么进度",
        "现在进度如何",
        "当前进度如何",
        "现在到哪一步了",
        "做到哪一步了",
        "现在处于什么状态",
        "现在是什么状态",
        "目前处于什么状态",
        "现在什么状态",
        "当前是什么状态",
        "目前是什么状态",
        "现在到哪一步",
        "目前到哪一步了",
        "现在什么阶段",
        "现在是什么阶段",
        "目前什么阶段",
        "当前什么阶段",
        "目前处于什么阶段",
        "现在处于什么阶段",
        "当前阶段",
        "进行到什么阶段了",
        "现在什么阶段，下一步呢",
        "现在什么阶段 下一步呢",
    }


def _phase_csv_request(message):
    value = " ".join(str(message or "").lower().split())
    export_verb = any(word in value for word in ("导出", "输出", "给我"))
    named_artifact = "相图" in value or "csv" in value
    return (
        export_verb and named_artifact and ("当前" in value or "最新" in value or "相图" in value)
    )


def _classify_history_decision(message):
    value = " ".join(str(message).strip().lower().split())
    if value in {"继续", "continue"}:
        return "continue"
    if value in {"新建", "new", "new run"}:
        return "new"
    return None


def _deepseek_switch_reply(model):
    label = "DeepSeek V4.1 Flash" if model == "deepseek-flash" else "DeepSeek V4 Pro"
    return (
        f"已将本地 Agent 切换为 {label}（`{model}`），并保存到本地 Agent 运行时配置；"
        "从下一条消息起生效。搜索配置、API Key 和计算任务未修改；未调用计算后端。"
    )
