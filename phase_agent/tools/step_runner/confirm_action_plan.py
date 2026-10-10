"""Record explicit user approval without changing the proposed actions."""

from copy import deepcopy
from datetime import datetime, timezone

from phase_agent.tools.step_runner.check_node_role import check_node_role
from phase_agent.tools.step_runner.file_protocol import read_json, write_json


def confirm_action_plan(plan_path, *, user_confirmed: bool, comment=None, node_role=None):
    check_node_role("login", role=node_role)
    plan = read_json(plan_path)
    if not plan:
        raise FileNotFoundError(plan_path)
    if plan.get("status") == "confirmed":
        return {"status": "already_confirmed", "plan": plan}
    updated = deepcopy(plan)
    updated["status"] = "confirmed" if user_confirmed else "rejected"
    updated["confirmation"] = {
        "explicit": bool(user_confirmed),
        "comment": comment,
        "at": datetime.now(timezone.utc).isoformat(),
    }
    write_json(plan_path, updated)
    return {"status": updated["status"], "plan": updated}
