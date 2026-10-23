"""Region-first branch policy. Work limits are guards, never scientific targets."""

from copy import deepcopy
from fractions import Fraction
import hashlib
import json
import math

VERSION = "region-adaptive-branch-v2"
MODE = "region_adaptive"
CONTROLS = {
    "batch_attempts": 16,
    "stagnant_batches": 3,
    "max_attempts": 4096,
    "review_after_new_branches": 1000,
    "max_seconds": 300,
}
INSTRUCTION = (
    "新模型刷新完成后的首次新branch生成是必需步骤，须先审批生成并展示实际结果，再单独审批Relax/MC输入；本阶段不允许暂缓新增。"
    "完成首次生成后，Branch按需决策：每轮优先考虑新增，但不强制；先比较旧branch新模型结果与新增收益。需要新增时，再依据有效证据和记忆选择区域与优先级。"
    "parameters.sampling_mode=region_adaptive，regions为1至12个对象，每项含region_id、priority(1最高)、"
    "strategy、phase、min_det_H、max_det_H、na_min、na_max、reason和evidence_refs。"
    "strategy仅coverage/composition/competing_phase/periodic_extension/tm_ordering。"
    "每项phase必须是项目boundary.P允许的单个相字符串（如O1、O3、P2、P3、OP2，具体以项目边界为准），大小写严格；不能省略、传列表或填写all/多相，多个相须拆成不同region，且遵守项目边界。Na范围是Na/O2。禁止quota、quotas、total_quota、batch_size、generation_plan，"
    "不先定600或任何总数，不把100–1000当必须达到的数量。程序分小批生成，按新增覆盖和初始TM近邻"
    "相关性描述符的多样性控制继续/停止；连续无新颖性停止该区域。达到资源保护阈值暂停等待审阅，"
    "不声称物理收敛。生成后才统计实际branch及相/H/Na分布。"
    "每个branch初态1至3个，默认3，不沿用旧初态4。不能凭空声称缺口、能量优势或历史收益；无证据标注探索假设。允许regions=[]并给defer_reason，"
    "表示现有池应先消化，不执行新生成。只生成登记branch，不封装Relax/MC/DFT输入或提交。"
)


def is_region(action):
    return (action.get("tool") or action.get("action_type")) == "generate_branches" and (
        action.get("parameters") or {}
    ).get("sampling_mode") == MODE


def policy():
    return {
        "version": VERSION,
        "instruction": INSTRUCTION,
        "controls": deepcopy(CONTROLS),
        "count_is_result": True,
        "prefer_new_branches": True,
        "approval_required": True,
    }


def errors(params):
    output = []
    if params.get("sampling_mode") != MODE:
        return ["region_sampling_requires_proposal_revision"]
    for key in ("quota", "quotas", "total_quota", "batch_size", "generation_plan"):
        if key in params:
            output.append(f"region_sampling_forbids_target_count:{key}")
    rows = params.get("regions")
    if not isinstance(rows, list) or len(rows) > 12:
        return output + ["regions_requires_array_up_to_12"]
    if not rows and not str(params.get("defer_reason") or "").strip():
        output.append("empty_regions_requires_defer_reason")
    from phase_agent.decisions.agent.generation_plan import STRATEGIES

    ids = set()
    allowed = {
        "region_id",
        "priority",
        "strategy",
        "phase",
        "min_det_H",
        "max_det_H",
        "na_min",
        "na_max",
        "reason",
        "evidence_refs",
    }
    for index, row in enumerate(rows):
        prefix = f"region[{index}]"
        if not isinstance(row, dict):
            output.append(prefix + ":requires_object")
            continue
        if set(row) - allowed:
            output.append(prefix + ":unknown_fields")
        identity = row.get("region_id")
        if not isinstance(identity, str) or not identity.strip() or identity in ids:
            output.append(prefix + ":invalid_or_duplicate_id")
        ids.add(str(identity))
        if type(row.get("priority")) is not int or not 1 <= row["priority"] <= 12:
            output.append(prefix + ":invalid_priority")
        if row.get("strategy") not in STRATEGIES:
            output.append(prefix + ":invalid_strategy")
        if not isinstance(row.get("phase"), str) or row["phase"] not in {
            "O1",
            "O3",
            "P2",
            "P3",
            "OP2",
        }:
            output.append(
                prefix + f":explicit_phase_required; received={row.get('phase')!r}; "
                "expected one string from O1/O3/P2/P3/OP2; split multiple phases into separate regions"
            )
        low, high = row.get("min_det_H"), row.get("max_det_H")
        if type(low) is not int or type(high) is not int or not 1 <= low <= high:
            output.append(prefix + ":invalid_supercell_bounds")
        low, high = row.get("na_min"), row.get("na_max")
        if (
            any(
                isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
                for v in (low, high)
            )
            or not 0 <= low <= high <= 1
        ):
            output.append(prefix + ":invalid_na_bounds")
        if not isinstance(row.get("reason"), str) or not row["reason"].strip():
            output.append(prefix + ":reason_required")
        refs = row.get("evidence_refs")
        if not isinstance(refs, list) or any(not isinstance(v, str) for v in refs):
            output.append(prefix + ":evidence_refs_required")
    if "sampling_controls" in params and params["sampling_controls"] != CONTROLS:
        output.append("sampling_controls_changed_requires_revision")
    initial = params.get("initial_states_per_branch", 3)
    if type(initial) is not int or not 1 <= initial <= 3:
        output.append("initial_states_requires_1_to_3")
    if "seed" in params and type(params["seed"]) is not int:
        output.append("seed_requires_integer")
    if "max_det_H" in params and (type(params["max_det_H"]) is not int or params["max_det_H"] <= 0):
        output.append("max_det_H_requires_positive_integer")
    return output


def branch_key(row):
    return (
        row["P"],
        json.dumps(row["H"], separators=(",", ":")),
        str(Fraction(str(row["x"]))),
        tuple(row["T"]),
    )


def ledger_fingerprint(manager):
    values = sorted((str(key), branch_key(row)) for key, row in manager.data["branches"].items())
    content = {
        "branches": values,
        "structure_ids": sorted(manager.data.get("structures") or {}),
    }
    return hashlib.sha256(json.dumps(content, sort_keys=True).encode()).hexdigest()


def ledger_context(manager):
    from collections import Counter
    from phase_agent.science.structures.boundary_utils import det_H

    branches = manager.data["branches"]
    groups = Counter(
        (row["P"], det_H(row["H"]), str(Fraction(str(row["x"])))) for row in branches.values()
    )
    return {
        "ledger_fingerprint": ledger_fingerprint(manager),
        "branch_count": len(branches),
        "structure_count": len(manager.data.get("structures") or {}),
        "coverage": [
            {"phase": p, "det_H": h, "na_per_O2": x, "branches": n}
            for (p, h, x), n in sorted(groups.items())
        ],
        "instruction": "此为当前账本实际统计，优先于旧状态快照；数量覆盖不代表该区域已经完成能量搜索或物理收敛。结合有效模型结果和记忆选择区域，不能只凭数量多少判断科学价值。",
    }


def preview(params, manager, references, config, state):
    """Legality and coverage from parent metadata; never expands a structure."""
    problems = errors(params)
    if problems:
        raise ValueError(";".join(problems))
    from phase_agent.science.structures.boundary_utils import (
        allowed_phases,
        allowed_phases_at_x,
        allowed_supercells,
        det_H,
        load_structure,
    )
    from phase_agent.decisions.agent.generation_plan import (
        configured_generation_strategies,
    )

    system = (
        config.get("system_config")
        or config.get("system")
        or manager.data.get("system_config")
        or {}
    )
    enabled = configured_generation_strategies({**config, "system": system})
    boundary = manager.boundary
    space = system.get("configuration_space") or {}
    roles, fixed = space.get("roles") or {}, space.get("fixed_values") or {}
    limits = (config.get("budgets") or {}).get("structure_limits") or {}
    cached, rows = {}, []
    existing = {branch_key(row)[:3] for row in manager.data["branches"].values()}
    for region in params["regions"]:
        phase = region["phase"]
        if phase not in allowed_phases(boundary["P"]) or region["strategy"] not in enabled:
            raise ValueError(f"非法或禁用区域：{region['region_id']}")
        if phase not in cached:
            from pymatgen.core import Element

            reference = load_structure(references[phase])
            cached[phase] = (
                len(reference),
                sum(s.specie.symbol == "O" for s in reference),
                sum(s.specie.symbol == "Na" for s in reference),
                sum(Element(s.specie.symbol).is_transition_metal for s in reference),
            )
        atoms, oxygen, sodium, tm_sites = cached[phase]
        if not oxygen:
            raise ValueError("母结构缺少氧")
        cells = []
        for matrix in allowed_supercells(boundary["H"], phase):
            size = det_H(matrix)
            ratio = boundary.get("TM_ratio") or {}
            if ratio and any(
                abs(
                    tm_sites * size * value / sum(ratio.values())
                    - round(tm_sites * size * value / sum(ratio.values()))
                )
                > 1e-8
                for value in ratio.values()
            ):
                continue
            if not region["min_det_H"] <= size <= region["max_det_H"]:
                continue
            if params.get("max_det_H") is not None and size > params["max_det_H"]:
                continue
            if limits.get("max_det_H") is not None and size > limits["max_det_H"]:
                continue
            if limits.get("max_atoms") is not None and atoms * size > limits["max_atoms"]:
                continue
            if roles.get("P") == "fixed" and phase != str(fixed["P"]).upper():
                continue
            if roles.get("H") == "fixed" and matrix != fixed["H"]:
                continue
            for count in range(sodium * size + 1):
                x = Fraction(2 * count, oxygen * size)
                if roles.get("x") == "fixed" and x != Fraction(str(fixed["x"])):
                    continue
                if phase not in allowed_phases_at_x(boundary["P"], x):
                    continue
                if region["na_min"] <= float(x) <= region["na_max"]:
                    key = (phase, json.dumps(matrix, separators=(",", ":")), str(x))
                    cells.append(
                        {
                            "H": matrix,
                            "det_H": size,
                            "x": str(x),
                            "covered": key in existing,
                        }
                    )
        if not cells:
            raise ValueError(f"区域没有合法相/H/Na组合：{region['region_id']}")
        rows.append(
            {
                "region_id": region["region_id"],
                "legal_cells": len(cells),
                "covered_cells": sum(row["covered"] for row in cells),
                "cells": cells,
            }
        )
    return {
        "ledger_fingerprint": ledger_fingerprint(manager),
        "model_version": state.get("active_model_version"),
        "config_version": state.get("confirmed_config_version"),
        "regions": rows,
        "existing_branch_count": len(manager.data["branches"]),
        "expected_new_branch_count": None,
    }
