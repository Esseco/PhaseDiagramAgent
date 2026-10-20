"""Evidence and explanation contract; the model chooses reuse or supplementation."""

from collections import Counter


INSTRUCTION = (
    "按需比较旧branch续搜、新增与DFT；每轮优先考虑新增branch，但不强制生成、不得凑数。"
    "结合旧branch的新模型基态/近简并/Ehull、已有模型下MC历史及收益、未搜产物的生成依据与有效记忆。"
    "已撤回的旧配额产物只能作历史分布证据，不能推荐搜索；用户已要求删除时必须遵从。"
    "顶层branch_pool_review须含choice=reuse_existing或supplement，sufficiency_for_next_search、rationale、"
    "alternative_reason为非空中文句子，missing_evidence、revisit_conditions、evidence_refs为字符串数组。"
    "若decision_context.branch_pool_facts.existing_branches存在，须含existing_branch_groups数组，"
    "每组decision=continue_search或defer，branch_indices列existing_branches中标明的branch_index整数（程序映射真实ID），reason给中文理由、evidence_refs给引用；"
    "可明确给default_existing_branch_decision=continue_search或defer、default_existing_branch_reason及default_existing_branch_evidence_refs，程序把未列成员按Agent此项明确决定展开；不设默认时须完整分组。最终各ID恰好一次，不只按相机械分组。"
    "基态及近简并优先考虑，不把刷新当MC完成；x=0没有Na/空位MC；同模型已完成任务不得重置。"
    "reuse_existing要求regions=[]及defer_reason；supplement要求具体regions，说明新branch的必要性与区域依据，"
    "不先定总数或配额，实际数生成后统计。可同时建议部分旧branch续搜与按需新增。"
    "本次动作仅生成提案；旧branch的分组仅为搜索建议，未授权准备输入或计算。"
    "所有evidence_refs必须逐字使用output_contracts.branch_review_evidence.references中的键；"
    "JSON字段路径、branch_id、branch_index和自拟名称均不是证据编号。引用对应来源的实际事实并说明关联，不凭编号存在推断科学结论。"
)


def pool_context(state, branches=None):
    """Summarize registered generation facts, without inferring search completion."""
    from phase_agent.tools.state.retired_generations import retired_task_keys, retirement_records
    from phase_agent.analysis.state.branch_search_evidence import existing_branch_evidence

    retired = retired_task_keys(state)
    histories = [
        row for row in state.get("generation_history") or [] if row.get("task_key") not in retired
    ]
    ids = set()
    structure_ids = set()
    for row in [next((row for row in reversed(histories) if row.get("new_branch_ids")), {})]:
        ids.update(row.get("new_branch_ids") or [])
        structure_ids.update(row.get("registered_ids") or [])
    tasks = state.get("tasks") or []
    related = [row for row in tasks if row.get("branch_id") in ids]
    result = {
        "registered_new_branch_count": len(ids),
        "registered_structure_count": len(structure_ids),
        "related_task_states": dict(Counter(str(row.get("status")) for row in related)),
        "active_model_version": state.get("active_model_version"),
        "instruction": "任务登记状态不能代替有效回收结果；无匹配任务也不能独立证明从未搜索。",
    }
    latest = next((row for row in reversed(histories) if row.get("new_branch_ids")), {})
    origin = {}
    for row in reversed(state.get("action_records") or []):
        raw = (row.get("agent_proposal") or {}).get("raw_action") or row.get("final_action") or {}
        if raw.get("task_key") == latest.get("task_key") and latest.get("task_key"):
            params = raw.get("parameters") or {}
            origin = {
                "sampling_mode": params.get("sampling_mode")
                or ("quota_plan" if params.get("total_quota") is not None else "unknown"),
                "planned_quota": params.get("total_quota"),
                "model_version": latest.get("model_version"),
                "task_key": latest.get("task_key"),
                "reason": raw.get("reason"),
            }
            break
    result["registered_batch_origin"] = origin
    result["existing_branches"] = existing_branch_evidence(state, branches)
    for index, row in enumerate(result["existing_branches"]):
        row["branch_index"] = index
    result["retired_batches"] = [
        {
            k: row.get(k)
            for k in ("task_key", "branch_count", "structure_count", "origin", "status", "reason")
        }
        for row in retirement_records(state)
    ]
    result["initialization_failures"] = [
        {
            "task_key": row.get("task_key"),
            "failed_branch_count": row["summary"]["initialization_failed_branches"],
            "reasons": row["summary"].get("initialization_failure_reasons") or {},
        }
        for row in state.get("generation_history") or []
        if (row.get("summary") or {}).get("initialization_failed_branches")
    ]
    result["prefer_new_branches"] = True
    if branches is not None:
        from phase_agent.science.structures.boundary_utils import det_H

        groups = Counter(
            (row["P"], det_H(row["H"]), str(row["x"]))
            for key, row in branches.items()
            if key in ids
        )
        result["registered_new_branch_distribution"] = [
            {"phase": phase, "det_H": size, "Na_per_O2": x, "branches": count}
            for (phase, size, x), count in sorted(groups.items())
        ]
        result["missing_ledger_branch_count"] = len(ids - set(branches))
    return result


def expand_review_indices(action, context):
    """Resolve only transport identifiers; group membership remains the model's choice."""
    from copy import deepcopy

    result = deepcopy(action)
    available = (context.get("branch_pool_facts") or {}).get("existing_branches") or []
    groups = (result.get("branch_pool_review") or {}).get("existing_branch_groups") or []
    for group in groups:
        if not isinstance(group, dict) or "branch_indices" not in group:
            continue
        indices = group["branch_indices"]
        if (
            "branch_ids" in group
            or not isinstance(indices, list)
            or any(type(i) is not int or not 0 <= i < len(available) for i in indices)
        ):
            raise ValueError("branch_pool_review.invalid_transport_indices")
        group["branch_ids"] = [available[i]["branch_id"] for i in indices]
        group.pop("branch_indices")
    review = result.get("branch_pool_review") or {}
    if "default_existing_branch_decision" in review:
        decision = review.pop("default_existing_branch_decision")
        reason = review.pop("default_existing_branch_reason", None)
        refs = review.pop("default_existing_branch_evidence_refs", None)
        if (
            decision not in {"continue_search", "defer"}
            or not isinstance(reason, str)
            or not reason.strip()
            or not isinstance(refs, list)
            or not refs
        ):
            raise ValueError(
                "branch_pool_review.explicit_default_requires_decision_reason_evidence"
            )
        assigned = {bid for group in groups for bid in group.get("branch_ids") or []}
        remaining = [row["branch_id"] for row in available if row["branch_id"] not in assigned]
        if remaining:
            review.setdefault("existing_branch_groups", []).append(
                {
                    "decision": decision,
                    "branch_ids": remaining,
                    "reason": reason,
                    "evidence_refs": refs,
                }
            )
    return result


def review_errors(action, context):
    if not context.get("branch_pool_review_required"):
        return []
    if (action.get("tool") or action.get("action_type")) != "generate_branches":
        return []
    try:
        action = expand_review_indices(action, context)
    except ValueError as error:
        return [str(error)]
    review = action.get("branch_pool_review")
    if not isinstance(review, dict):
        return ["branch_pool_review: required before supplementing a registered pool"]
    errors = []
    for key in ("sufficiency_for_next_search", "rationale", "alternative_reason"):
        if not isinstance(review.get(key), str) or not review[key].strip():
            errors.append("branch_pool_review." + key + ": nonempty explanation required")
    for key in ("missing_evidence", "revisit_conditions", "evidence_refs"):
        values = review.get(key)
        if not isinstance(values, list) or any(not isinstance(v, str) for v in values):
            errors.append("branch_pool_review." + key + ": string array required")
    refs = review.get("evidence_refs")
    if isinstance(refs, list) and not refs:
        errors.append("branch_pool_review.evidence_refs: at least one source required")
    catalog = (context.get("evidence_catalog") or {}).get("references") or {}
    if (
        isinstance(refs, list)
        and catalog
        and any(ref not in catalog for ref in refs if isinstance(ref, str))
    ):
        errors.append(
            "branch_pool_review.evidence_refs: unknown source "
            + repr([ref for ref in refs if isinstance(ref, str) and ref not in catalog])
        )
    expected = {
        row["branch_id"]
        for row in (context.get("branch_pool_facts") or {}).get("existing_branches") or []
    }
    if expected:
        groups = review.get("existing_branch_groups")
        seen = []
        if not isinstance(groups, list):
            errors.append("branch_pool_review.existing_branch_groups: required")
        else:
            for index, group in enumerate(groups):
                if not isinstance(group, dict):
                    errors.append("branch_pool_review.group: object required")
                    continue
                ids = group.get("branch_ids")
                if not isinstance(ids, list) or any(not isinstance(v, str) for v in ids):
                    errors.append("branch_pool_review.group.branch_ids: string array required")
                    continue
                seen.extend(ids)
                unavailable = {
                    row["branch_id"]
                    for row in (context.get("branch_pool_facts") or {}).get("existing_branches")
                    or []
                    if row.get("has_mobile_ions") is False
                }
                if group.get("decision") == "continue_search" and unavailable.intersection(ids):
                    errors.append(
                        "branch_pool_review.group: x=0 has no Na/vacancy MC; defer these branches or propose a separate explicit operation"
                    )
                if (
                    group.get("decision") not in {"continue_search", "defer"}
                    or not str(group.get("reason") or "").strip()
                ):
                    errors.append("branch_pool_review.group: decision and reason required")
                refs = group.get("evidence_refs")
                if (
                    not isinstance(refs, list)
                    or not refs
                    or any(
                        not isinstance(ref, str) or (catalog and ref not in catalog) for ref in refs
                    )
                ):
                    errors.append(
                        f"branch_pool_review.group[{index}].evidence_refs: valid evidence refs required; received={refs!r}"
                    )
            if len(seen) != len(set(seen)) or set(seen) != expected:
                errors.append(
                    "branch_pool_review.groups: complete unique partition required; missing="
                    + str(sorted(expected - set(seen)))
                    + "; unknown="
                    + str(sorted(set(seen) - expected))
                    + "; duplicated="
                    + str(sorted({bid for bid in seen if seen.count(bid) > 1}))
                )
    regions = (action.get("parameters") or {}).get("regions")
    choice = review.get("choice")
    if choice not in {"reuse_existing", "supplement"}:
        errors.append("branch_pool_review.choice: reuse_existing or supplement required")
    elif (choice == "supplement") != bool(regions):
        errors.append("branch_pool_review.choice: must match regions")
    return errors


def readable_review(review):
    if not review:
        return None
    return {
        "建议": {
            "reuse_existing": "先复用已有branch，暂不补充",
            "supplement": "补充指定区域",
        }.get(review.get("choice")),
        "是否够启动下一批搜索": review.get("sufficiency_for_next_search"),
        "判断理由": review.get("rationale"),
        "旧branch搜索建议": [
            {
                "建议": "继续搜索" if group["decision"] == "continue_search" else "暂缓",
                "数量": len(group["branch_ids"]),
                "branch编号": group["branch_ids"],
                "理由": group["reason"],
                "依据": group["evidence_refs"],
            }
            for group in review.get("existing_branch_groups") or []
        ],
        "旧branch数量汇总": {
            "建议继续搜索": sum(
                len(group["branch_ids"])
                for group in review.get("existing_branch_groups") or []
                if group["decision"] == "continue_search"
            ),
            "建议暂缓": sum(
                len(group["branch_ids"])
                for group in review.get("existing_branch_groups") or []
                if group["decision"] == "defer"
            ),
        },
        "另一条路径为何暂不优先": review.get("alternative_reason"),
        "尚缺的证据": review.get("missing_evidence"),
        "何时重新考虑补充": review.get("revisit_conditions"),
        "依据": review.get("evidence_refs"),
        "说明": "这是下一步路径建议，不是物理收敛结论，也不授予计算或提交权限。",
    }
