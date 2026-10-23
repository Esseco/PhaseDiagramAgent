"""Review independent D/H questions without prescribing a batch size or selecting points."""

from copy import deepcopy
from math import isfinite


def coverage_evidence(candidates):
    groups = {}
    for row in candidates:
        evidence = row.get("dft_sampling_evidence") or {}
        scope = evidence.get("sampling_evidence_scope")
        if evidence.get("status") != "available" or scope not in {
            "current_near_hull",
            "conditional_near_hull",
        }:
            continue
        d_group = evidence.get("D_candidate_id")
        h = evidence.get("H_single_max")
        # Same meV/O2 comparison tolerance as phase_quality/phase_low_energy_D.
        positive_h = type(h) in {int, float} and isfinite(h) and h > 1e-8
        if not d_group and not positive_h:
            continue
        key = (
            "D:" + str(d_group)
            if d_group
            else "H:" + str(evidence.get("geometry_key") or row["candidate_id"])
        )
        group = groups.setdefault(
            key,
            {
                "group_id": key,
                "candidate_ids": [],
                "phase": row.get("phase"),
                "scope": scope,
                "D_score": evidence.get("D_score"),
                "H_union_sequence": deepcopy(evidence.get("H_D_candidate_union_sequence")),
                "members": [],
            },
        )
        if row["candidate_id"] in group["candidate_ids"]:
            continue
        group["candidate_ids"].append(row["candidate_id"])
        group["members"].append(
            {
                "candidate_id": row["candidate_id"],
                "H_single_sequence": deepcopy(evidence.get("H_single_sequence")),
                "D_reference_structure_paths": deepcopy(
                    evidence.get("D_reference_structure_paths") or []
                ),
            }
        )
    return {
        "groups": list(groups.values()),
        "instruction": "逐组考虑当前/条件近凸包D簇及独立H疑点。每组至少一项已有合法候选须选择DFT，或用DEFER/REJECT及具体理由说明复用基准、重复环境、条件依赖或暂缓依据。可增加或替换代表，不默认每组都计算，不强制固定数量或选满100。条件H依赖前序假设，不能视为同时发生；缺失D/H未知而非零。D上方参考需核对可比DFT与精确几何，不凭旧结构ID声称已覆盖。同相已有代表不等于另一独立簇已验证；暂缓需权衡本组联合H及条件依赖，而非仅凭D较低或同相覆盖。",
    }


def coverage_errors(action, candidates):
    rows = (action.get("parameters") or {}).get("decisions") or []
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        return []  # Existing preview owns malformed decisions.
    decisions = {row.get("candidate_id"): row for row in rows}
    errors = []
    for group in coverage_evidence(candidates)["groups"]:
        considered = [decisions[cid] for cid in group["candidate_ids"] if cid in decisions]
        if any(row.get("action") in {"DFT_SINGLE_POINT", "DFT_RELAX"} for row in considered):
            continue
        if any(
            row.get("action") in {"DEFER", "REJECT"}
            and isinstance(row.get("reason"), str)
            and row["reason"].strip()
            for row in considered
        ):
            continue
        errors.append(
            "dft_DH_group_requires_selection_or_reasoned_deferral:"
            + group["group_id"]
            + ":"
            + ",".join(group["candidate_ids"])
        )
    return errors
