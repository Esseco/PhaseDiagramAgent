"""One readable JSON for reviewing a region plan and its eventual actual counts."""

from collections import Counter
from copy import deepcopy
import hashlib
from pathlib import Path

from phase_agent.tools.relax_mc_batch.files import read, write
from phase_agent.tools.policy.region_sampling import CONTROLS


def path_for(state, action):
    identity = str(action.get("task_key")) + ":" + str(state.get("active_model_version"))
    digest = hashlib.sha256(identity.encode()).hexdigest()[:16]
    root = (state.get("workspace_layout") or {}).get("workspace_root")
    if not root:
        raise ValueError("region_review_requires_workspace_root")
    return Path(root) / "workflow_state/branch_plans" / f"branch_region_plan_{digest}.json"


def write_proposal(proposal, state):
    action = proposal["raw_action"]
    params = action["parameters"]
    compact_params = deepcopy(params)
    for item in (compact_params.get("region_preview") or {}).get("regions") or []:
        item.pop("cells", None)
    path = path_for(state, action)
    if path.exists() and (read(path).get("程序记录（可折叠）") or {}).get("status") == "completed":
        return {"json": str(path)}
    preview = params.get("region_preview") or {}
    legal = {row["region_id"]: row for row in preview.get("regions") or []}
    rows = []
    for row in sorted(params["regions"], key=lambda r: (r["priority"], r["region_id"])):
        detail = legal.get(row["region_id"]) or {}
        rows.append(
            {
                "区域": row["region_id"],
                "优先级": row["priority"],
                "相": row["phase"],
                "选择方法": row["strategy"],
                "超胞倍数det_H范围": [row["min_det_H"], row["max_det_H"]],
                "钠含量Na每O2范围": [row["na_min"], row["na_max"]],
                "选择理由": row["reason"],
                "证据": row["evidence_refs"],
                "合法相_超胞_钠含量组合数": detail.get("legal_cells"),
                "已有覆盖组合数": detail.get("covered_cells"),
                "组合数说明": "组合内可有多个TM排列；此数不是计划或实际branch数。",
            }
        )
    controls = params.get("sampling_controls") or CONTROLS
    payload = {
        "状态": "待审批，尚未执行新生成",
        "决策方式": "Agent选区域与优先级；程序受控生成、去重和停止；最后统计实际数量。",
        "方案来源": action.get("proposal_backend") or action.get("decision_source"),
        "本轮背景与复用说明": action.get("branch_region_review_notes"),
        "项目已有branch数": preview.get("existing_branch_count"),
        "本次选择理由": action.get("reason"),
        "本次选择的区域": rows,
        "本次新增branch数": None,
        "数量说明": "不预设总数或各区域配额；实际新增数与相、超胞、Na分布在生成登记后统计。",
        "停止规则": {
            "区域停止": f"连续{controls['stagnant_batches']}小批没有新覆盖或新的初始TM近邻相关性描述符。",
            "每小批最多尝试": controls["batch_attempts"],
            "暂停审阅保护阈值": {
                "尝试次数": controls["max_attempts"],
                "新增branch数": controls["review_after_new_branches"],
                "秒": controls["max_seconds"],
            },
            "说明": "保护阈值不是生成目标或最终截断数，达到即暂停等待审阅；随机停滞不证明穷尽，描述符不代表能量或物理收敛。",
        },
        "每个branch计划初始结构数": params.get("initial_states_per_branch", 3),
        "执行范围": "批准后仅生成并登记branch及初始结构，保存实际统计后停止。",
        "暂缓理由": params.get("defer_reason"),
        "程序记录（可折叠）": {
            "schema_version": 2,
            "status": "awaiting_approval",
            "task_key": action.get("task_key"),
            "config_version": state.get("confirmed_config_version"),
            "model_version": state.get("active_model_version"),
            "parameters": compact_params,
        },
    }
    write(path, payload, sort_keys=False)
    return {"json": str(path)}


def write_result(history, manager, registered, state, action):
    path = path_for(state, action)
    if not path.exists():
        write_proposal({"raw_action": action}, state)
    payload = read(path)
    new = set(history["new_branch_ids"])
    counts, inventory = Counter(), []
    for identity in sorted(new):
        row = manager.data["branches"][identity]
        matrix = row["H"]
        from phase_agent.science.structures.boundary_utils import det_H

        counts[(row["P"], det_H(matrix), str(row["x"]))] += 1
        inventory.append(
            {
                "branch_id": identity,
                "相": row["P"],
                "超胞H": matrix,
                "钠含量Na每O2": str(row["x"]),
                "TM排列": row["T"],
                "本次登记结构数": sum(r["branch_id"] == identity for r in registered),
            }
        )
    report = history["summary"].get("sampling_report") or {}
    payload.update(
        {
            "状态": "已生成，保护阈值触发后待审阅"
            if report.get("status") == "paused_for_review"
            else "已生成",
            "本次新增branch数": len(new),
            "本次登记结构数": len(registered),
            "实际分布": [
                {"相": p, "超胞倍数det_H": h, "钠含量Na每O2": x, "branch数": n}
                for (p, h, x), n in sorted(counts.items())
            ],
            "生成停止与失败统计": history["summary"],
            "新增branch清单": inventory,
        }
    )
    payload["程序记录（可折叠）"].update(status="completed", actual_new_branch_count=len(new))
    raw = payload.pop("程序记录（可折叠）")
    rows = []
    for identity in sorted({r["branch_id"] for r in registered}):
        branch = manager.data["branches"][identity]
        structures = [
            {
                "structure_id": r["structure_id"],
                "path": (manager.data.get("structures") or {})
                .get(r["structure_id"], {})
                .get("source_path"),
            }
            for r in registered
            if r["branch_id"] == identity
        ]
        rows.append(
            {
                "branch_id": identity,
                "is_new": identity in new,
                "phase": branch["P"],
                "H": branch["H"],
                "Na_fraction": str(branch["x"]),
                "structure_count": len(structures),
                "structures": structures,
            }
        )
    raw["actual"] = {
        "new_branch_count": len(new),
        "registered_branch_count": len(rows),
        "registered_structure_count": len(registered),
        "branches": rows,
    }
    payload["程序记录（可折叠）"] = raw
    write(path, payload, sort_keys=False)
    return {"json": str(path)}
