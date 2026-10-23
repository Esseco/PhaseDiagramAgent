"""Independent H/D evidence, strict existing geometry verdict, and formal/hypothetical tiers."""

from hashlib import sha256
import json


def annotate_dispositions(report, state, *, snapshot, points):
    from phase_agent.analysis.phase.phase_quality_restore import restored_records
    from phase_agent.analysis.phase.phase_quality_endpoints import endpoint_guards, retained_keys

    rows = report["structures"]
    by_key = {r["geometry_key"]: r for r in rows}
    cases = report["analysis_layers"][0]["joint_cases"]["cases"]
    islands = report.get("energy_islands", [])
    policy = report["screening_policy"]
    hcut, dcut, directcut = (
        policy[k]
        for k in ("hull_shift_mev_per_O2", "outlier_gap_mev_per_O2", "direct_removal_mev_per_O2")
    )
    protected = restored_records(state)
    decisions = {r["review_id"]: r for r in state.get("phase_quality_decisions", [])}
    retained_geometry = retained_keys(state, snapshot, policy, points)
    eligible = set()
    for target in cases + islands:
        keys = target["geometry_keys"]
        lift = target.get("hull_lift_mev_per_O2")
        if (
            lift is not None
            and lift > hcut
            and all(by_key[k]["distance_to_current_hull_mev_per_O2"] <= hcut for k in keys)
            and all(
                (by_key[k]["geometry"].get("framework") or {}).get("auto_confirmed") is True
                for k in keys
            )
        ):
            eligible.update(keys)
    for row in rows:
        key = row["geometry_key"]
        relevant = [c for c in islands if key in c["geometry_keys"]]
        joint = [c for c in cases if key in c["geometry_keys"]]
        row["energy_island_ids"] = [c["cluster_id"] for c in relevant]
        row["D_mev_per_O2"] = max(
            (c["gap_to_lowest_reference_mev_per_O2"] for c in relevant), default=None
        )
        row["H_mev_per_O2"] = row["hull_lift_mev_per_O2"]
        row["joint_hull_lift_mev_per_O2"] = max(
            (
                c["hull_lift_mev_per_O2"]
                for c in relevant + joint
                if c.get("hull_lift_mev_per_O2") is not None
            ),
            default=0.0,
        )
        row["near_current_hull"] = row["distance_to_current_hull_mev_per_O2"] <= hcut
        h, d = row["H_mev_per_O2"], row["D_mev_per_O2"]
        high_d = d is not None and d > dcut and row["near_current_hull"]
        high_h = row["near_current_hull"] and (
            (h is not None and h > hcut) or row["joint_hull_lift_mev_per_O2"] > hcut
        )
        conditional_near = [
            e
            for e in row.get("conditional_evidence", [])
            if e["distance_to_conditional_hull_mev_per_O2"] <= hcut
        ]
        conditional_high = any(
            (e.get("hull_lift_mev_per_O2") or 0) > hcut
            or (e.get("group_hull_lift_mev_per_O2") or 0) > hcut
            or (e.get("D_mev_per_O2") or 0) > dcut
            or (e.get("joint_hull_lift_mev_per_O2") or 0) > hcut
            for e in conditional_near
        )
        framework = row["geometry"].get("framework") or {}
        broken = framework.get("auto_confirmed") is True
        direct = (
            h is not None
            and h > directcut
            and report["model_version"] == state.get("active_model_version")
        )
        # A raw joint effect cannot authorize deleting one unconfirmed member.
        geometry_rule = (
            broken
            and row["near_current_hull"]
            and (high_d or (h is not None and h > hcut) or key in eligible)
        )
        reason = (
            "single_impact_above_100"
            if direct
            else "confirmed_framework_destruction"
            if geometry_rule
            else None
        )
        binding = {
            "rule": report["rule_version"],
            "model": report["model_version"],
            "hull": report["hull_version"],
            "policy": policy,
            "key": key,
            "records": row["record_ids"],
            "geometry": row["geometry"],
            "H": h,
            "D": d,
        }
        row["review_id"] = (
            "pq-" + sha256(json.dumps(binding, sort_keys=True).encode()).hexdigest()[:24]
        )
        restored = bool(row["record_ids"]) and all(
            (report["model_version"], rid) in protected for rid in row["record_ids"]
        )
        retained = (
            key in retained_geometry
            or decisions.get(row["review_id"], {}).get("decision") == "retain"
        )
        endpoint = row.get("endpoint_protected", False)
        row["framework_status"] = framework.get("status", "uncertain")
        row["automatic_exclusion_reason"] = None if endpoint else reason
        row["automatic_exclusion_eligible"] = (
            not endpoint and not restored and (direct or (geometry_rule and not retained))
        )
        review = high_h or high_d or conditional_high or h is None or endpoint
        if endpoint:
            disposition, label, tier = (
                "endpoint_protected",
                "端点基准保护／替代参考不足／待验证或补算",
                0,
            )
        elif restored:
            disposition, label, tier = "human_restored", "人工已恢复／继续DFT选点", 0
        elif direct:
            disposition, label, tier = "auto_exclude", f"第一档／当前H>{directcut:g}／有备份移除", 1
        elif retained:
            disposition, label, tier = "human_retained", "人工暂不处理／保留并继续DFT选点", 0
        elif row["automatic_exclusion_eligible"]:
            disposition, label, tier = "auto_exclude", "第一档／H或D超阈值且结构模块确认有问题", 1
        elif h is None and not (high_h or high_d or conditional_high):
            disposition, label, tier = "pending_human", "端点影响未知／保留端点范围并人工核查", 0
        elif review:
            disposition, label, tier = (
                "pending_human",
                "第二档／H或D证据较高／人工检查；右图假设排除",
                2,
            )
            if conditional_high and not (high_h or high_d):
                label += "／依赖前序假设"
        else:
            disposition, label, tier = "low_impact", "常规保留／不满足当前H/D处置条件", 0
        row.update(
            disposition=disposition,
            disposition_label=label,
            tier=tier,
            review_recommended=review,
            review_pending=disposition in {"auto_exclude", "pending_human", "endpoint_protected"},
            conditional_only=conditional_high and not (high_h or high_d),
            sampling_evidence_scope="current_near_hull"
            if row["near_current_hull"]
            else "conditional_near_hull"
            if conditional_near
            else "far_from_reviewed_hulls",
            quality_sampling_evidence_eligible=row["near_current_hull"] or bool(conditional_near),
            dft_evidence_priority_mev_per_O2=max(
                [(d or 0) if row["near_current_hull"] else 0]
                + [(e.get("D_mev_per_O2") or 0) for e in conditional_near]
            ),
        )
        if row.get("preview_excluded") is False and tier == 2:
            row["disposition_label"] = label.replace(
                "；右图假设排除", "；本次右图保留（端点或人工保留约束）"
            )
    auto_keys = {r["geometry_key"] for r in rows if r["automatic_exclusion_eligible"]}
    joint_guards = set(endpoint_guards(points, policy, auto_keys)) & auto_keys
    for row in rows:
        if row["geometry_key"] in joint_guards:
            row.update(
                endpoint_protected=True,
                endpoint_protection_reason="本次联合移除将耗尽独立同相替代参考",
                automatic_exclusion_eligible=False,
                automatic_exclusion_reason=None,
                disposition="endpoint_protected",
                disposition_label="端点基准保护／联合移除将耗尽同相替代参考／待验证或补算",
                tier=0,
                review_pending=True,
            )
    for target in cases + islands:
        members = [by_key[k] for k in target["geometry_keys"]]
        target["review_ids"] = [r["review_id"] for r in members]
        target["framework_statuses"] = {r["geometry_key"]: r["framework_status"] for r in members}
        target["framework_all_confirmed_destroyed"] = all(
            (r["geometry"].get("framework") or {}).get("auto_confirmed") is True for r in members
        )
        target["automatic_exclusion_eligible"] = all(
            r["automatic_exclusion_eligible"] for r in members
        )
        target["disposition"] = (
            "auto_exclude"
            if target["automatic_exclusion_eligible"]
            else "pending_human"
            if any(r["review_pending"] for r in members)
            else "retained"
        )
        target["decision_role"] = "supporting_evidence_alongside_ehull_and_qbc"
    report["pending_human_count"] = sum(
        r["disposition"] in {"pending_human", "endpoint_protected"} for r in rows
    )
    report["automatic_exclusion_count"] = sum(r["automatic_exclusion_eligible"] for r in rows)
    report["status"] = (
        "review_required" if any(r["review_pending"] for r in rows) else "screened_no_findings"
    )
