"""Read an MC step target from feedback on a pending MC proposal."""

import re


def resolve_mc_budget_feedback(message, action):
    if (action or {}).get("tool") != "allocate_mc_bohb":
        return None
    text = re.sub(r"\s+", "", str(message or "").lower())
    if any(term in text for term in ("relax", "弛豫", "dft")):
        return None
    if not any(term in text for term in ("预算", "mc", "搜索步数", "本轮", "这轮")):
        return None
    matches = re.findall(r"(?<!\d)(\d+)步(?!\d)", text)
    if len(matches) != 1:
        return None
    value = int(matches[0])
    return value if value > 0 else None


def resolve_mc_full_plan_steps(message, action):
    """Resolve an explicit full/uncompressed choice from the frozen preview."""
    if (action or {}).get("tool") != "allocate_mc_bohb":
        return None
    text = re.sub(r"\s+", "", str(message or "").lower())
    if not any(term in text for term in (
        "完整运行", "全部运行", "全量运行", "不压缩", "完整方案", "全部branch",
    )):
        return None
    preview = ((action.get("parameters") or {}).get("budget_preview") or {})
    steps = preview.get("full_plan_steps")
    if isinstance(steps, bool) or not isinstance(steps, int) or steps <= 0:
        return None
    return steps
