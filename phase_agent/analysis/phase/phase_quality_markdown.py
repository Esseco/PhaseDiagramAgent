"""Compact review items: one low-energy cluster per phase/composition, full paths in details."""


def review_groups(report):
    rows = report["structures"]
    by_key = {r["geometry_key"]: r for r in rows}
    result = []
    for item in report["review_items"]:
        members = [by_key[k] for k in item["geometry_keys"]]
        reference = max(members, key=lambda r: r.get("D_mev_per_O2") or 0)
        result.append(
            {
                **item,
                "members": members,
                "reference_count": reference["reference_count"],
                "reference_status": reference["reference_status"],
            }
        )
    result.sort(key=lambda g: (-(g["D"] or 0), -(g["H"] or 0), g["x"], g["phase"]))
    for number, g in enumerate(result, 1):
        g.update(
            number=number,
            review_ids=[r["review_id"] for r in g["members"]],
            disposition="端点保护／待验证"
            if all(r.get("endpoint_protected") for r in g["members"])
            else "自动移除"
            if all(r["tier"] == 1 for r in g["members"])
            else "人工审查",
        )
    return result


def review_markdown(report, *, bundle_id, plot_paths=()):
    def value(v):
        return "未知" if v is None else f"{v:.3f}"

    def link(path):
        target = str(path).removeprefix("\\\\?\\").replace("\\", "/")
        return f"[结构](<{target}>)（`{path}`）"

    policy = report["screening_policy"]
    groups = review_groups(report)
    lines = [
        f"# 相图审查：{bundle_id}",
        "",
        f"模型 `{report['model_version']}`；凸包 `{report['hull_version']}`。",
        "",
        f"仅两个可调阈值：H>{policy['hull_shift_mev_per_O2']:g}，D>{policy['outlier_gap_mev_per_O2']:g} meV/O₂。当前单结构H>100直接移除，但端点基准保护优先。",
        "H/D超阈值且现有结构模块确认有问题为第一档；其余高证据为第二档，交人工确认。右图仅对同一排除链实际选中的点打×，第二档正式保留。",
        "端点同相基准簇缺少独立替代参考时保留；近重复的基准簇成员和其他相不算替代。沿用内部参考要求：基准簇外至少5个精确几何，能量跨度≤50 meV/O₂。参考充足仅代表可评估，不证明正确；结构异常仍需验证或补算。",
        "",
        "每个Na含量＋相只列一条审查项，低能离群簇不重复展开；表中H为整项移除影响，不能替代单结构H>100条件。D为该项最大的同相同组分参考能差。",
        "本模块优先当前近凸包结构；沿用H审查窗口作为近凸包范围（默认50 meV/O₂），远点不因D大获得额外选点优先级，直到递归后进入近凸包。DFT正式Ehull来自已执行移除后的当前正式相图；待审右图、条件距离只作诊断补充，不替换正式Ehull，也不把D当作能量修正。",
        "",
        "|编号|Na含量 / 相|结构数|H（整项）|D|处理|",
        "|---:|---|---:|---:|---:|---|",
    ]
    for g in groups:
        lines.append(
            f"|{g['number']}|{g['x']:.6g} / {g['phase']}|{len(g['members'])}|{value(g['H'])}|{value(g['D'])}|{g['disposition']}|"
        )
    lines += [
        "",
        "未列为重点的结构保留。H、D不相加；第二档DFT证据主要看D，结合Ehull、QBC；QBC不是实际误差。",
        "",
        "左右图分别按各自保留集合的端点归一化，每张图所有相共用同一端点参考，×也使用右图参考。H采用固定原始参考，不能直接读作两图纵向差；D不变。内部最多5项联合检查，按H/D截止参数继续到无新增候选；后层依赖前层假设，联合抬升超过100或条件影响均不单独授权正式删除。",
        f"递归复核：{'已收敛' if report['recursive_converged'] else '未收敛，不能视为完成排查'}；共{len(report['analysis_layers'])}轮（含最后复核），终止原因 `{report['recursive_stop_reason']}`。",
        "",
    ]
    for g in groups:
        lines += [
            f"## 审查 {g['number']}：Na={g['x']:.6g} / {g['phase']}",
            f"H={value(g['H'])}；D={value(g['D'])} meV/O₂；{g['disposition']}。参考{g['reference_count']}个，{g['reference_status']}。",
            "结构地址及成员：",
        ]
        for r in g["members"]:
            framework = r["geometry"].get("framework") or {}
            lines.append(
                f"- {', '.join(r['structure_ids'])}；记录 {', '.join(r['record_ids'])}；单结构H={value(r['hull_lift_mev_per_O2'])}；{r.get('disposition_label', '待核查')}；结构结论：{framework.get('status', 'uncertain')}。"
            )
            lines.extend("  " + link(p) for p in r["structure_paths"])
            lines.append(f"  审核身份：`{r['review_id']}`。")
            conditional = r.get("conditional_evidence") or []
            if conditional:
                lines.append(
                    "  条件影响："
                    + "；".join(
                        f"层{e['analysis_layer']} H={value(e.get('hull_lift_mev_per_O2'))}、联合H={value(e.get('joint_hull_lift_mev_per_O2'))}、D={value(e.get('D_mev_per_O2'))}、距该层凸包={value(e.get('distance_to_conditional_hull_mev_per_O2'))}"
                        for e in conditional
                    )
                    + "；依赖前序假设，具体移除前提见JSON。"
                )
            lines.append(
                "  右图：" + ("假设排除并打×。" if r.get("preview_excluded") else "保留。")
            )
        lines.append("")
    lines += ["## 已执行排除与恢复", ""]
    for entry in report.get("applied_exclusions", []):
        lines.append(
            f"- 已执行排除：{entry.get('reason')}；备份/审计 `{entry.get('backup_path')}`。"
        )
        for e in entry.get("evidence") or []:
            lines.append(
                f"  记录 {', '.join(e.get('record_ids') or entry.get('record_ids') or [])}；操作时单结构H={value(e.get('hull_lift_mev_per_O2'))} meV/O₂。"
            )
            if e.get("automatic_exclusion_reason") == "single_impact_above_100":
                lines.append("  按H直接阈值移除，不代表已确认结构破坏。")
            lines.extend("  " + link(p) for p in e.get("structure_paths") or [])
    for event in report.get("applied_restorations", []):
        lines.append(
            f"- 已人工恢复记录：{', '.join(event['record_ids'])}；{event['reason']}；同模型同记录同几何不立即重复删除。"
        )
    for target in report.get("restoration_targets", []):
        lines.append(
            "- 可恢复编号："
            + ", ".join(f"`{v}`" for v in target.get("review_ids") or [])
            + f"；备份 `{target.get('backup_path')}`。"
        )
    lines += [
        "",
        "删除只移除选中活动结构和记录，保留归档、原始回传和branch其他成员；可人工恢复，以后相同结构可重新搜索和计算。",
        "",
        "## 文件",
        *[f"- `{p}`" for p in plot_paths],
        f"- 完整审查JSON：`{report.get('report_path')}`",
        "",
    ]
    history = report.get("policy_control", {}).get("history") or []
    if history:
        event = history[-1]
        lines.append(
            f"最近阈值调整：{event['source']}；{event['reason']}；前后值和版本见JSON审计。"
        )
    return "\n".join(lines)
