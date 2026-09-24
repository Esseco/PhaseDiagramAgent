"""连接状态摘要、动作选择、校验、回退、执行和留痕。"""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

from execution_layer.budget.check_budget import check_budget
from execution_layer.budget.record_budget_usage import record_budget_usage

from decision_layer.agent.choose_agent_action import choose_agent_action
from decision_layer.agent.choose_rule_action import choose_rule_action
from execution_layer.dispatch.execute_agent_action import execute_agent_action
from analysis_layer.state.summarize_agent_state import summarize_agent_state
from execution_layer.policy.validate_agent_action import validate_agent_action


def run_agent_scheduler(state: dict, *, agent_client=None, handlers=None, config=None, save_path=None) -> dict:
    current = deepcopy(state)
    summary = summarize_agent_state(current)
    settings = dict(config or {})
    limits = settings.get("budget_limits")
    llm_budget = check_budget(current, {"llm_usage": {"calls": 1}, "iteration": current.get("iteration", 0)}, limits) if limits and agent_client else {"allowed": True, "reasons": []}
    proposed = choose_agent_action(summary, agent_client=agent_client if llm_budget["allowed"] else None, config=config)
    if not llm_budget["allowed"]:
        proposed["fallback_reason"] = ",".join(llm_budget["reasons"])
    if proposed.get("_llm_usage"):
        current = record_budget_usage(current, {"llm_usage": proposed["_llm_usage"], "iteration": current.get("iteration", 0)})
    validation = validate_agent_action(proposed, summary, config=config)
    fallback_used = False
    if not validation["valid"]:
        proposed = choose_rule_action(summary, config=config)
        validation = validate_agent_action(proposed, summary, config=config)
        fallback_used = True
    execution = execute_agent_action(proposed, handlers=handlers) if validation["valid"] else {"status": "rejected", "error": validation["errors"]}
    record = {"decision_id": f"decision-{len(current.get('decision_history', [])) + 1:06d}", "summary": summary, "action": proposed, "validation": validation, "fallback_used": fallback_used, "execution": execution}
    current.setdefault("decision_history", []).append(record)
    if execution.get("status") == "completed":
        current["remaining_budget"] = max(0.0, float(current.get("remaining_budget", 0.0)) - float(proposed.get("budget", 0.0)))
        if limits:
            current = record_budget_usage(current, {"stage": proposed.get("stage"), "tasks": len(proposed.get("target_ids", [])) or 1, "relative_cost": proposed.get("budget", 0.0)})
    current["iteration"] = int(current.get("iteration", 0)) + 1
    if save_path:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(current, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return {"state": current, "record": record}


def load_scheduler_state(path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))
