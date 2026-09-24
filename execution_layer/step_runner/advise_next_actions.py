"""Online login-node step: turn one immutable summary into one proposed plan."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from execution_layer.step_runner.check_node_role import check_node_role
from execution_layer.step_runner.file_protocol import content_id, read_json, write_json


def advise_next_actions(summary_path, plan_directory, *, agent_client, config_version, node_role=None):
    check_node_role("login", role=node_role)
    summary = read_json(summary_path)
    if not summary:
        raise FileNotFoundError(f"status summary not found: {summary_path}")
    if summary.get("config_version") not in {None, config_version}:
        raise ValueError("summary config_version does not match the confirmed configuration")
    plan_key = content_id({"summary_id": summary["summary_id"], "config_version": config_version}, "plan")
    path = Path(plan_directory) / f"{plan_key}.json"
    existing = read_json(path)
    if existing:
        return {"status": "already_advised", "plan": existing, "plan_path": str(path)}
    response = agent_client({
        "instruction": (
            "Return a JSON action plan based only on the supplied factual summary. "
            "Do not invent or modify energy, Ehull, QBC, cost, convergence, or completed results. "
            "The plan must contain an actions list and reasons."
        ),
        "state_summary": summary,
        "config_version": config_version,
    })
    actions = response.get("actions")
    if actions is None and response.get("tool"):
        actions = [{key: value for key, value in response.items() if key != "_llm_usage"}]
    if not isinstance(actions, list):
        raise ValueError("Agent response must contain an actions list or one tool action")
    plan = {
        "plan_id": plan_key, "status": "proposed", "config_version": config_version,
        "summary_id": summary["summary_id"], "actions": actions,
        "reason": response.get("reason") or response.get("reasons"),
        "llm_usage": response.get("_llm_usage"),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    write_json(path, plan)
    return {"status": "proposed", "plan": plan, "plan_path": str(path)}
