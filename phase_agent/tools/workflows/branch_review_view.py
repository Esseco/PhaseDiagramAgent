"""A Chinese reading view of recorded branch plans; no scientific decisions."""

from collections import Counter, defaultdict
import json
from fractions import Fraction

RAW_KEY = "程序记录（可折叠）"
STRATEGIES = {
    "coverage": "覆盖搜索空间",
    "composition": "补充组成分布",
    "competing_phase": "探索竞争相",
    "tm_ordering": "探索过渡金属排布",
    "periodic_extension": "周期扩展",
}


def raw_record(value):
    return value.get(RAW_KEY, value.get("99 原始记录（程序使用，可折叠）", value))


def display_cost(value):
    return (
        round(value, 2)
        if isinstance(value, (int, float)) and not isinstance(value, bool)
        else value
    )


def markdown_rationale(record):
    """Detailed report only; keep compact inventories and chat receipts unchanged."""
    from phase_agent.tools.workflows.compact_review import value

    review = record.get("branch_pool_review") or {}
    params = record.get("parameters") or {}
    plan = record.get("plan") or {}
    lines = [
        "### 本轮 branch 选点理由",
        "",
        "本轮目标与依据：" + value(record.get("reason") or record.get("quantity_reason")),
        "复用或补充判断：" + value(review.get("rationale")),
        "其他方案的取舍：" + value(review.get("alternative_reason")),
        "判断引用：" + value(review.get("evidence_refs")),
    ]
    notes = record.get("branch_region_review_notes")
    if notes:
        lines.append("本轮背景与复用说明：" + value(notes))
    if params.get("regions") == []:
        lines.append("暂不新增的理由：" + value(params.get("defer_reason")))
    regions = params.get("regions") or plan.get("allocations") or []
    if regions:
        lines.extend(
            [
                "",
                "| 区域或分配 | 相 | 选择策略 | 选点理由 | 证据引用 |",
                "| --- | --- | --- | --- | --- |",
            ]
        )
        for index, row in enumerate(regions, 1):
            cells = [
                row.get("region_id", index),
                row.get("phase"),
                STRATEGIES.get(row.get("strategy"), row.get("strategy")),
                row.get("reason"),
                row.get("evidence_refs"),
            ]
            lines.append("| " + " | ".join(value(item) for item in cells) + " |")
    groups = review.get("existing_branch_groups") or []
    if groups:
        lines.extend(
            [
                "",
                "已有 branch 分组判断：",
                "",
                "| branch 分组 | 建议 | 理由 | 证据引用 |",
                "| --- | --- | --- | --- |",
            ]
        )
        for row in groups:
            cells = [
                row.get("branch_ids") or row.get("branch_indices"),
                row.get("decision"),
                row.get("reason"),
                row.get("evidence_refs"),
            ]
            lines.append("| " + " | ".join(value(item) for item in cells) + " |")
    for key, label in (("missing_evidence", "尚缺证据"), ("revisit_conditions", "重新评估条件")):
        if review.get(key):
            lines.extend(["", label + "：" + value(review[key])])
    lines.extend(["", "以上仅展示本轮保存的判断；未记录的理由或引用不会自动补造。", ""])
    return "\n\n".join(lines[:6]) + "\n" + "\n".join(lines[6:])


def phase_rules(plan):
    rules = plan.get("phases")
    if isinstance(rules, list):
        return [
            {"相": phase, "钠含量_x": plan.get("Na_fractions") or "生成后确定"} for phase in rules
        ]
    if not isinstance(rules, dict):
        return [{"相": "未记录", "钠含量_x": "未记录"}]
    by_phase = defaultdict(list)
    for value, phases in (rules.get("at_x") or {}).items():
        for phase in phases:
            by_phase[phase].append(f"x = {value}")
    for phase in rules.get("intermediate") or []:
        by_phase[phase].append("中间钠含量（0 < x < 1），具体值生成后登记")
    return [
        {
            "相": phase,
            "允许钠含量_x": values,
            "计划数量": "未单独分配；生成后填写实际数量",
        }
        for phase, values in sorted(by_phase.items())
    ]


def supercell_options(plan):
    matrices = {}
    for row in plan.get("supercells") or []:
        key = json.dumps(row.get("H"), sort_keys=True)
        item = matrices.setdefault(
            key, {"detH": row.get("det_H"), "H矩阵": row.get("H"), "适用相": []}
        )
        if row.get("phase") not in item["适用相"]:
            item["适用相"].append(row.get("phase"))
    groups = defaultdict(list)
    for item in matrices.values():
        groups[item["detH"]].append({"H矩阵": item["H矩阵"], "适用相": sorted(item["适用相"])})
    return [
        {"超胞大小_detH": size, "不同H矩阵数量": len(items), "选项": items}
        for size, items in sorted(groups.items(), key=lambda pair: (pair[0] is None, pair[0] or 0))
    ]


def allocation_phases(plan, row):
    if row.get("phase") != "all":
        return [row.get("phase")]
    phases = {r.get("phase") for r in plan.get("supercells") or [] if r.get("phase")}
    rules = plan.get("phases")
    lo, hi = row.get("na_min"), row.get("na_max")
    if isinstance(rules, dict) and lo is not None and hi is not None:
        allowed = set()
        for x, items in (rules.get("at_x") or {}).items():
            if lo <= float(Fraction(x)) <= hi:
                allowed.update(items)
        if hi > 0 and lo < 1:
            allowed.update(rules.get("intermediate") or [])
        phases.intersection_update(allowed)
    return sorted(phases)


def readable_view(record):
    plan = record.get("plan") or {}
    actual = record.get("actual")
    completed = record.get("status") == "completed"
    cost = record.get("budget_preview") or {}
    allocations = plan.get("allocations") or []
    if not allocations:
        allocations = [
            {"strategy": name, "quota": number, "phase": "all"}
            for name, number in (plan.get("quotas") or {}).items()
        ]
    adaptive = bool(plan.get("branch_count_policy"))
    target = plan.get("candidate_quota")
    explicit = []
    for row in allocations:
        phases = allocation_phases(plan, row)
        sizes = sorted(
            {
                r.get("det_H")
                for r in plan.get("supercells") or []
                if r.get("phase") in phases
                and r.get("det_H") is not None
                and (row.get("min_det_H") is None or r["det_H"] >= row["min_det_H"])
                and (row.get("max_det_H") is None or r["det_H"] <= row["max_det_H"])
            }
        )
        explicit.append(
            {
                "生成策略": STRATEGIES.get(row.get("strategy"), row.get("strategy")),
                "相": phases,
                "计划branch数": row.get("quota"),
                "允许超胞倍数_detH": sizes,
                "钠含量_x范围": [row.get("na_min"), row.get("na_max")],
                "数量依据": row.get("reason") or "原方案未记录，需修订",
                "说明": "范围内仅取母胞允许的整数Na占位；本项数量在列出的相/超胞/钠含量之间共享，不是每个组合各生成此数量。",
            }
        )
    view = {
        "状态": "已生成" if completed else "待审批，尚未生成",
        "数量": {
            "候选采样配额": plan.get("candidate_quota"),
            "去重后选取branch上限": plan.get("selected_branch_upper"),
            "计划每个branch初始结构数": plan.get("initial_states_per_branch"),
            "数量说明": "配额是计划采样量，上限是最多选取量；两者都不是实际新增branch数。",
        },
        "相与钠含量": phase_rules(plan),
        "超胞": {
            "本次detH上限": plan.get("max_det_H"),
            "大小含义": "det(H)表示相对于该相母胞的超胞倍数，不是原子总数。",
            "允许选项（相同矩阵合并）": supercell_options(plan),
        },
        "候选配额分配": [
            {
                "生成策略": STRATEGIES.get(row.get("strategy"), row.get("strategy")),
                "相范围": "按相与钠含量规则" if row.get("phase") == "all" else row.get("phase"),
                "候选配额": row.get("quota"),
            }
            for row in allocations
        ],
        "后续计算预算": {
            "后续Relax_MC估计相对成本（约）": display_cost(cost.get("relax_mc_cost_upper")),
            "记录的剩余相对预算（约）": display_cost(cost.get("remaining_budget")),
            "预算提示": "估算超过剩余相对预算，请核对；不是已花费成本，也不是机时。"
            if cost.get("status") == "over_budget"
            else "以原方案预算记录为准",
        },
        "本次执行范围": "生成并登记branch，按原方案准备Relax+MC输入；不提交超算"
        if record.get("execution_scope") == "generate_and_prepare_relax_mc_inputs"
        else "只生成并登记branch；后续输入准备另行审批",
        "实际生成结果": {
            "状态": "尚未生成，审批执行后在同一文件补入实际清单",
            "新增branch数": None,
            "初始结构数": None,
            "按相_超胞_钠含量汇总": [],
            "结构文件清单": [],
        },
        RAW_KEY: record,
    }
    if adaptive:
        view["程序约束"] = {
            "数量范围": [100, 1000],
            "明确配置的分组限制": plan.get("selection_guards")
            or {"max_per_framework": None, "max_per_parent_branch": None},
            "分组说明": "null表示未明确配置额外限制；不套旧8/框架和2/父branch小默认。覆盖均衡仍参与排序。",
            "结构规则": "遵守确认的相/H/Na、固定变量及去重规则；非法候选不执行。",
            "执行规则": "当前模型和配置必须匹配；先人工审批，不自动提交。",
        }
        view["数量"] = {
            "本批目标branch数": target,
            "自适应选择范围": [100, 1000],
            "每个branch计划初始结构数": plan.get("initial_states_per_branch"),
            "初始结构数上界": target * plan.get("initial_states_per_branch", 0),
            "本次数量依据": record.get("quantity_reason")
            or "按下列分配理由审阅本批目标；不是固定默认值。",
            "不足处理": "去重或合法空间不足时报告实际数量，不用96截断，不自动补算；每轮重新评估目标。",
        }
        view["超胞"] = {
            "大小定义": "相对于各相母胞的倍数det(H)，不是原子数",
            "本次允许大小": sorted(
                {r.get("det_H") for r in plan.get("supercells") or [] if r.get("det_H") is not None}
            ),
            "分配规则": "每项允许大小见下方分配清单；H矩阵保留在末尾程序记录。",
        }
        view["候选配额分配"] = explicit
        per_phase = Counter()
        shared = []
        for row in allocations:
            if row.get("phase") == "all":
                shared.append(
                    {
                        "策略": STRATEGIES.get(row.get("strategy"), row.get("strategy")),
                        "适用相": allocation_phases(plan, row),
                        "合计数量": row.get("quota"),
                        "钠含量范围": [row.get("na_min"), row.get("na_max")],
                    }
                )
            else:
                per_phase[row.get("phase")] += row.get("quota") or 0
        view["相与钠含量"] = {
            "指定相的计划数量": dict(per_phase),
            "跨相共享计划": shared,
            "分配合计": sum(per_phase.values()) + sum(r["合计数量"] or 0 for r in shared),
            "说明": "跨相共享项未承诺逐相数量；具体超胞、Na区间和依据见逐项分配。所有数量是去重前目标，实际结果另列。",
        }
        view["后续计算预算"]["预算提示"] = "仅作粗估和参考，不据此缩减本批目标或截断branch。"
    if actual is not None:
        grouped = Counter()
        structures = Counter()
        branches = []
        for row in actual.get("branches") or []:
            key = (row.get("phase"), row.get("supercell_det"), row.get("Na_fraction"))
            grouped[key] += 1
            structures[key] += row.get("structure_count") or 0
            branches.append(
                {
                    "branch编号": row.get("branch_id"),
                    "是否本轮新增": row.get("is_new"),
                    "相": row.get("phase"),
                    "超胞大小_detH": row.get("supercell_det"),
                    "H矩阵": row.get("H"),
                    "钠含量_x": row.get("Na_fraction"),
                    "初态数": row.get("structure_count"),
                    "结构文件": [
                        {
                            "结构编号": s.get("structure_id"),
                            "原子数": s.get("atom_count"),
                            "路径": s.get("path"),
                        }
                        for s in row.get("structures") or []
                    ],
                }
            )
        view["实际生成结果"] = {
            "状态": "已生成",
            "新增branch数": actual.get("new_branch_count"),
            "登记branch数": actual.get("registered_branch_count"),
            "初始结构数": actual.get("registered_structure_count"),
            "按相_超胞_钠含量汇总": [
                {
                    "相": key[0],
                    "超胞大小_detH": key[1],
                    "钠含量_x": key[2],
                    "branch数量": count,
                    "初态数量": structures[key],
                }
                for key, count in grouped.items()
            ],
            "结构文件清单": branches,
        }
    # Keep machine data last even when optional human review sections are added.
    view.pop(RAW_KEY)
    view[RAW_KEY] = record
    return view
