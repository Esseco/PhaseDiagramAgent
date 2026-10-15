"""Chinese DFT selection details; values come only from recorded evidence."""

from phase_agent.tools.workflows.branch_review_view import RAW_KEY, display_cost

ACTION_NAMES = {
    "DFT_SINGLE_POINT": "DFT单点",
    "DFT_RELAX": "DFT结构优化",
    "DEFER": "暂缓",
    "REJECT": "不选",
}


def known(value):
    return "未记录" if value is None or value == "unknown" else value


def readable_view(record):
    preview = record.get("input_preview") or {}
    counts = (record.get("summary") or {}).get("counts_by_action") or {}
    actual = record.get("actual")
    statuses = {
        "awaiting_approval": "待审批，尚未准备输入",
        "prepared": "输入已准备，尚不代表计算完成",
        "tasks_created": "任务已登记，输入准备状态见下方",
        "input_preparation_failed": "输入准备未完成，请核对原因",
    }
    selected, deferred = [], []
    for row in record.get("decisions") or []:
        if row.get("action") not in {"DFT_SINGLE_POINT", "DFT_RELAX"}:
            deferred.append(
                {
                    "结构编号": row.get("candidate_id"),
                    "处理": ACTION_NAMES.get(row.get("action"), row.get("action")),
                    "理由": known(row.get("reason")),
                }
            )
            continue
        qbc = row.get("qbc") or {}
        selected.append(
            {
                "序号": len(selected) + 1,
                "结构编号": row.get("candidate_id"),
                "branch编号": known(row.get("branch_id")),
                "相": known(row.get("phase")),
                "超胞大小_detH": known(row.get("supercell_det")),
                "H矩阵": known(row.get("H")),
                "钠含量_x": known(row.get("Na_fraction")),
                "原子总数": known(row.get("atom_count")),
                "元素组成": known(row.get("composition")),
                "计算类型": ACTION_NAMES[row["action"]],
                "选点理由": known(row.get("reason")),
                "距凸包能量_Ehull": known(row.get("Ehull")),
                "Ehull单位": known(row.get("Ehull_unit")),
                "QBC不确定度": {
                    "每原子能量标准差": known(qbc.get("energy_per_atom_std")),
                    "最大力分歧": known(
                        qbc.get("force_max_atom_disagreement", qbc.get("f_std_max"))
                    ),
                    "说明": "保持记录的数值与单位，缺失项不补造",
                },
                "训练集关系": "未确认"
                if row.get("training_membership") is None
                or row.get("training_membership") == "unknown"
                else row.get("training_membership"),
                "已有DFT任务记录": row.get("known_DFT_tasks") or "当前记录未提供；不代表从未计算",
                "结构文件": known(row.get("structure_path")),
            }
        )
    estimate = (record.get("estimated_cost") or {}).get("estimated_total_cost")
    view = {
        "状态": statuses.get(record.get("status"), "以程序记录状态为准"),
        "本次目的": known(record.get("purpose")),
        "选点前的路径判断": {
            "本次建议": "对现有结构选点DFT",
            "已记录依据": known((record.get("route_review") or {}).get("rationale")),
            "已比较路线": (record.get("route_review") or {}).get("alternatives") or "未记录",
            "流程说明": "第一轮搜索完成后可直接选点DFT；后续轮次先比较再生成branch搜索与对现有结构选点DFT。此处只展示已保存判断，不重新决策。",
        },
        "数量": {
            "选中结构数": (record.get("summary") or {}).get("selected_count", len(selected)),
            "DFT单点数": counts.get("DFT_SINGLE_POINT", 0),
            "DFT结构优化数": counts.get("DFT_RELAX", 0),
            "暂缓数": counts.get("DEFER", 0),
            "不选数": counts.get("REJECT", 0),
        },
        "选中结构清单": selected,
        "暂缓或不选清单": deferred,
        "计算预算": {
            "估计相对成本": known(display_cost(preview.get("relative_cost", estimate))),
            "说明": "这是计划成本，不是机时，也不是已花费成本。",
        },
        "本次执行范围": "按批准的选点准备DFT输入；不提交超算。",
        "实际输入任务": "尚未准备；批准执行后在同一文件补入实际任务和输入目录。",
        RAW_KEY: record,
    }
    if actual is not None:
        view["实际输入任务"] = {
            "任务数": actual.get("task_count"),
            "输入准备状态": (actual.get("input_preparation") or {}).get("status", "未记录"),
            "输入目录与准备结果": actual.get("input_preparation") or "尚未准备输入",
            "任务清单": [
                {
                    "任务编号": t.get("task_id"),
                    "结构编号": t.get("structure_id"),
                    "计算类型": "DFT单点"
                    if t.get("stage") == "dft_single_point"
                    else "DFT结构优化"
                    if t.get("stage") == "dft_relax"
                    else known(t.get("stage")),
                    "计划相对成本": t.get("planned_relative_cost"),
                    "结构文件": known(t.get("structure_path")),
                }
                for t in actual.get("tasks") or []
            ],
            "未接受项": actual.get("rejected") or [],
        }
    return view
