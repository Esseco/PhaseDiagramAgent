"""人工明确填写的长期建议；随 workflow state 保存，不自动吸收普通评论。"""

from copy import deepcopy

LONG_TERM_CATEGORIES = {
    "human_system_knowledge",
    "physical_priors",
    "frozen_parameter_advice",
    "search_rules",
}


def update_long_term_advice(state, advice, *, source):
    """None 保持原值，列表完整替换，空列表清空；保留旧版本。"""
    updated = deepcopy(state)
    if advice is None:
        return updated
    if not isinstance(advice, list) or any(not isinstance(x, str) or not x.strip() for x in advice):
        raise ValueError("long_term_advice 必须为非空文字条目的列表；[] 表示清空")
    advice = list(dict.fromkeys(x.strip() for x in advice))
    memory = updated.setdefault(
        "decision_memory", {"version": 0, "long_term_advice": [], "history": []}
    )
    if memory["long_term_advice"] == advice:
        return updated
    memory["version"] += 1
    memory["long_term_advice"] = advice
    long_term = memory.setdefault("long_term", {})
    long_term["human_system_knowledge"] = deepcopy(advice)
    long_term.setdefault("physical_priors", [])
    long_term.setdefault("frozen_parameter_advice", [])
    long_term.setdefault("search_rules", [])
    memory["source"] = source
    memory.setdefault("history", []).append(
        {"version": memory["version"], "advice": advice, "source": source}
    )
    return updated


def update_short_term_memory(state, snapshot):
    """Replace volatile memory from facts; never promote it into long-term knowledge."""
    updated = deepcopy(state)
    memory = updated.setdefault(
        "decision_memory", {"version": 0, "long_term_advice": [], "history": []}
    )
    memory["short_term"] = deepcopy(snapshot.get("short_term_memory") or {})
    memory["short_term"]["current_status"] = snapshot.get("current_status")
    memory["short_term"]["available_budget"] = deepcopy(snapshot.get("available_budget"))
    memory["short_term"]["uncertainty"] = deepcopy(snapshot.get("uncertainty") or [])
    memory["short_term"]["recent_actions"] = deepcopy(snapshot.get("recent_action_history") or [])
    memory["short_term"]["snapshot_id"] = snapshot.get("snapshot_id")
    return updated


def update_long_term_memory(state, updates, *, source):
    """Explicitly revise durable knowledge; volatile observations are rejected."""
    updated = deepcopy(state)
    if updates is None:
        return updated
    if not isinstance(updates, dict) or set(updates) - LONG_TERM_CATEGORIES:
        raise ValueError("long_term_memory contains unsupported categories")
    normalized = {}
    for category, items in updates.items():
        if not isinstance(items, list) or any(
            not isinstance(item, str) or not item.strip() for item in items
        ):
            raise ValueError(f"long_term_memory.{category} must be a list of non-empty strings")
        normalized[category] = list(dict.fromkeys(item.strip() for item in items))
    memory = updated.setdefault(
        "decision_memory", {"version": 0, "long_term_advice": [], "history": []}
    )
    long_term = memory.setdefault("long_term", {})
    changed = False
    for category in LONG_TERM_CATEGORIES:
        long_term.setdefault(category, [])
    for category, items in normalized.items():
        if long_term[category] != items:
            long_term[category] = items
            changed = True
    if not changed:
        return updated
    memory["version"] = int(memory.get("version", 0)) + 1
    memory["source"] = source
    memory["long_term_advice"] = deepcopy(long_term["human_system_knowledge"])
    memory.setdefault("history", []).append(
        {"version": memory["version"], "long_term": deepcopy(long_term), "source": source}
    )
    return updated
