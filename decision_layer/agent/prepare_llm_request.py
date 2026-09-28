"""Compact transport-only context and choose reasoning for scientific decisions."""
from copy import deepcopy


def prepare_llm_request(payload):
    result = deepcopy(payload)
    state = result.get("state") or {}
    if not isinstance(state, dict):
        return result
    context = state.pop("decision_context", None)
    if context is not None:
        result.setdefault("decision_context", context)
    context = result.get("decision_context") or {}
    # These are aliases, not additional evidence. Persistent state is untouched.
    for alias, canonical in (("recent_action_history", "search_history"),
                             ("qbc_uncertainty", "uncertainty")):
        if state.get(alias) == state.get(canonical):
            state.pop(alias, None)
    if context.get("available_branches") == state.get("available_branches"):
        state.pop("available_branches", None)
    rows = context.get("available_branches")
    if isinstance(rows, list):
        excluded = {"structure", "initial_structure", "relaxed_structure", "sites",
                    "cart_coords", "frac_coords", "T", "tm_site_order", "provenance"}
        context["available_branches"] = [
            {k: v for k, v in row.items() if k not in excluded}
            if isinstance(row, dict) else row for row in rows]
        context["branch_detail_policy"] = "IDs and summary metrics only; full occupancy and structures remain in the local ledger."
    return result


def needs_deep_reasoning(payload):
    """No LLM call is needed to classify its own reasoning budget."""
    if payload.get("decision_kind") in {"strategy", "dft_selection", "model_update", "convergence"}:
        return True
    state = payload.get("state") or {}
    text = str(payload.get("human_comment") or payload.get("user_message") or state.get("user_message") or "").lower()
    if any(word in text for word in ("收敛", "策略", "dft选", "模型更新", "预算分配", "重新训练")):
        return True
    if any(word in text for word in ("生成branch", "重新生成", "准备", "输入文件", "读取配置", "查看状态")):
        return False
    # Only actual scientific evidence activates reasoning, not the instruction
    # listing every available tool, or the ordinary bootstrap/prepare flow.
    return bool(state.get("uncertainty") or state.get("qbc_uncertainty")
                or (state.get("mlip_status") or {}).get("stale_results")
                or any((hull or {}).get("stable_entries")
                       for hull in (state.get("current_convex_hull") or {}).values()))
