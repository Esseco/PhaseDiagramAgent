"""Catch common scope/meaning mistakes before saving an Agent patch."""

import re


def validate_requested_config_patch(message, patch):
    text = re.sub(r"\s+", "", str(message).lower())
    explicit_candidates = any(word in text for word in ("候选", "生成量", "总配额", "total_quota"))
    branches = re.search(r"(\d+)(?:个)?branch", text)
    corrected = dict(patch)
    if branches and not explicit_candidates and "run.total_quota" in corrected:
        value = corrected.pop("run.total_quota")
        corrected.setdefault("run.batch_size", value)
    h_mentioned = re.search(r"(?:h|超胞)(?:的)?(?:首轮|本轮)?上限", text)
    global_scope = any(
        word in text
        for word in ("全局", "总边界", "所有轮", "存储上限", "budgets.structure_limits")
    )
    if (
        h_mentioned
        and not global_scope
        and any(
            path.startswith("budgets.structure_limits.")
            or path.startswith("qbc.budget_limits.structure_limits.")
            for path in corrected
        )
    ):
        raise ValueError(
            "H本轮/首轮上限不能写成全局结构预算边界；请仅修改本轮动作的max_det_H，或明确指定首轮配置字段"
        )
    return corrected
