"""在相同初态、预算和执行模拟器下比较两类调度。"""

from copy import deepcopy

from execution_layer.workflows.run_agent_scheduler import run_agent_scheduler


def compare_rule_and_agent(initial_state: dict, *, agent_client=None, handlers=None, config=None, steps=1) -> dict:
    results = {}
    for name, client in (("rule", None), ("agent", agent_client)):
        state = deepcopy(initial_state)
        records = []
        for _ in range(steps):
            output = run_agent_scheduler(state, agent_client=client, handlers=handlers, config=config)
            state, record = output["state"], output["record"]
            records.append(record)
        confirmed_rewards = [item["execution"].get("result", {}).get("confirmed_reward") for item in records if isinstance(item["execution"].get("result"), dict) and item["execution"]["result"].get("confirmed_reward") is not None]
        results[name] = {"final_state": state, "records": records, "spent_budget": float(initial_state.get("remaining_budget", 0.0)) - float(state.get("remaining_budget", 0.0)), "confirmed_reward": sum(confirmed_rewards) if confirmed_rewards else None}
    return {"status": "completed", "same_initial_budget": initial_state.get("remaining_budget"), "results": results, "note": "Reward remains unknown unless the shared simulator or real feedback reports it."}
