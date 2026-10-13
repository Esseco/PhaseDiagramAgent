"""Small conversation facts and explicitly requested, read-only evidence."""

from copy import deepcopy
from collections import Counter
import hashlib
import json
from phase_agent.analysis.state.workflow_blockers import refresh_result_facts, convergence_feedback


def progress_signature(facts):
    from phase_agent.analysis.state.project_authority import project_identity

    view = {
        "project": project_identity(facts),
        "tasks": [(row.get("task_id"), row.get("status")) for row in facts.get("tasks") or []],
        "diagrams": {
            k: v.get("version")
            for k, v in (facts.get("phase_diagrams") or {}).items()
            if isinstance(v, dict)
        },
        "refresh": {
            k: (facts.get("model_refresh") or {}).get(k) for k in ("status", "wave", "task_ids")
        },
        "pending": [
            (k, v.get("record_id"), v.get("revision"))
            for k, v in (facts.get("pending_execution_policies") or {}).items()
        ],
    }
    return hashlib.sha256(json.dumps(view, sort_keys=True, default=str).encode()).hexdigest()


def conversation_facts(facts, project):
    tasks = facts.get("tasks") or []
    refresh = refresh_result_facts(facts)
    return {
        "current_project": deepcopy(project),
        "current_status": facts.get("status"),
        "human_constraints": deepcopy(
            (facts.get("decision_memory") or {}).get("long_term_advice")
            or ((facts.get("decision_memory") or {}).get("long_term") or {}).get(
                "human_system_knowledge"
            )
            or []
        ),
        "task_counts": dict(Counter(t.get("status", "unknown") for t in tasks)),
        "task_stage_counts": [
            {"stage": stage, "status": status, "count": count}
            for (stage, status), count in Counter(
                (t.get("stage"), t.get("status")) for t in tasks
            ).items()
        ],
        "phase_diagrams": {
            method: {
                key: row.get(key) for key in ("status", "version", "model_version", "csv_path")
            }
            for method, row in (facts.get("phase_diagrams") or {}).items()
            if isinstance(row, dict)
        },
        "refresh_result_facts": refresh,
        "latest_convergence_check": convergence_feedback(facts),
        "gpu_usage_summary": deepcopy(facts.get("gpu_usage_summary") or {}),
        "pool_counts": {
            "branches": len(facts.get("branch_candidates") or []),
            "dft_candidates": len(facts.get("qbc_candidates") or []),
        },
        "recent_failures": [
            {
                "task_id": row.get("task_id"),
                "stage": row.get("stage"),
                "reason": str(row.get("error") or row.get("failure_reason") or "原因未记录")[:600],
            }
            for row in tasks
            if row.get("status") in {"failed", "timeout"}
        ][-3:],
    }


def read_topics(facts, topics):
    from phase_agent.analysis.state.build_state_snapshot import build_state_snapshot

    fields = {
        "mc": ("mc_entry_conditions", "available_branches", "current_phase_diagram"),
        "dft": (
            "qbc_candidates",
            "post_dft_assessment",
            "training_excluded_data_ids",
            "dft_magnetic_diagnostics",
        ),
        "tasks": (
            "task_stage_evidence",
            "task_round_evidence",
            "refresh_result_facts",
            "retryable_tasks",
        ),
        "convergence": ("latest_convergence_check", "current_phase_diagram", "post_dft_assessment"),
        "budget": ("round_budget_evidence", "calculation_cost_reference"),
        "history": (
            "recent_experience",
            "decision_feedback",
            "long_term_memory",
            "relevant_approved_knowledge",
        ),
        "model": (
            "model_refresh",
            "refresh_result_facts",
            "model_round_summary",
            "training_validation_planning",
            "post_dft_assessment",
        ),
        "configuration": (),
        "plans": (),
    }
    selected = {key for topic in topics for key in fields[topic]}
    context = (
        build_state_snapshot(facts, snapshot_index=int(facts.get("state_snapshot_index", 0)))[
            "decision_context"
        ]
        if selected
        else {}
    )
    result = {key: deepcopy(context.get(key)) for key in selected}
    if "configuration" in topics:
        result["confirmed_configuration"] = deepcopy(facts.get("confirmed_config") or {})
    if "model" in topics:
        result["active_model"] = deepcopy(facts.get("active_model") or {})
    if "convergence" in topics:
        result["convergence_rules"] = deepcopy(
            (facts.get("confirmed_config") or {}).get("convergence") or {}
        )
    if "plans" in topics:
        result["pending_plans"] = [
            {
                "plan_id": key,
                "revision": row.get("revision", 0),
                "action": {
                    name: deepcopy(value)
                    for name, value in (
                        (row.get("agent_proposal") or {}).get("raw_action") or {}
                    ).items()
                    if name not in {"decision_context", "_llm_usage"}
                },
                "authority": "只读方案事实，不是批准；审批仍绑定实际plan_id/版本/哈希。",
            }
            for key, row in (facts.get("pending_execution_policies") or {}).items()
        ]
    return result


def read_limit_reply(facts):
    refresh = refresh_result_facts(facts)
    groups = refresh["failure_groups"]
    if groups:
        causes = "；".join(f"{row['count']} 个任务：{row['reason']}" for row in groups[:3])
        return (
            "模型重复读取同一证据，已停止循环。当前已核对的失败："
            + causes
            + "。\n先恢复报错中缺失的模型文件或依赖，再核对并审核重试方案；重复读取或“同意”无法修复文件。本轮未执行任务。"
        )
    check = convergence_feedback(facts)
    if check.get("missing_evidence"):
        from phase_agent.runtime.workflow_reply_presentation import format_workflow_reply

        return format_workflow_reply({**check, "status": "insufficient_evidence"}, "state.json")
    return "本轮已读取所请求的证据，但模型重复请求同一内容，未形成新的有效答复。已停止重复调用，未执行任务；已有结果和待审批方案保留。"
