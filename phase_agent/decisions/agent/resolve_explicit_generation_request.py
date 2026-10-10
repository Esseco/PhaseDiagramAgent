"""Route explicit user requests for branch generation without LLM ambiguity."""

from __future__ import annotations

import re


def _requested_max_det_H(message: str) -> int | None:
    text = re.sub(r"\s+", "", str(message or "").lower())
    match = re.search(
        r"(?:det\(?h\)?|h|超胞)(?:的)?(?:首轮|本轮|首次|当前轮)?"
        r"(?:上限|最大(?:行列式)?|不超过|<=|≤|限制(?:为|到)?)"
        r"(?:设为|改为|为|到|是|[:：=])?(\d+)",
        text,
    )
    return int(match.group(1)) if match else None


def _requested_branch_batch_size(message: str) -> int | None:
    """Read an explicit selected-branch limit, not the candidate generation quota."""
    text = re.sub(r"\s+", "", str(message or "").lower())
    match = re.search(
        r"(?:入选|选中|筛选|最终选择)(?:branch|分支)?(?:数量|数|上限|为|到|[:：=])*"
        r"(\d+)(?:个)?(?:branch|分支)?",
        text,
    )
    if match is None:
        match = re.search(r"(?:batch_size|入选上限)[:：=](\d+)", text)
    return int(match.group(1)) if match else None


def resolve_explicit_generation_request(
    message: str, *, allowed_tools: list[str], state=None
) -> dict | None:
    text = "".join(str(message or "").lower().split())
    if "generate_branches" not in allowed_tools:
        return None
    if any(
        token in text
        for token in ("不要生成", "不生成", "暂不生成", "先不生成", "取消生成", "别生成")
    ):
        return None
    mentions_branch = any(token in text for token in ("branch", "分支"))
    asks_generation = any(
        token in text
        for token in (
            "生成",
            "重新生成",
            "再生成",
            "补充",
            "新增",
            "扩展",
            "generate",
            "regenerate",
            "create",
        )
    )
    if not (mentions_branch and asks_generation):
        return None
    snapshot = (state or {}).get("snapshot_id") or "current"
    max_det_H = _requested_max_det_H(message)
    parameters = {"max_det_H": max_det_H} if max_det_H is not None else {}
    batch_size = _requested_branch_batch_size(message)
    if batch_size is not None:
        parameters["batch_size"] = batch_size
    return {
        "tool": "generate_branches",
        "action_type": "generate_branches",
        "task_key": f"generate_branches:user-request:{snapshot}",
        "target_ids": [],
        "parameters": parameters,
        "budget": 0.0,
        "reason": (
            f"按用户指令在 det(H) ≤ {max_det_H} 内生成 branch 候选。"
            if max_det_H is not None
            else "按用户明确指令生成新的 branch 候选。"
        ),
        "expected_purpose": "扩展合法 branch 候选池，供后续预筛与搜索使用。",
        "decision_source": "explicit_user_instruction",
    }
