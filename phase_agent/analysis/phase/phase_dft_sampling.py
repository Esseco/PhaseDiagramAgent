"""Fixed D and conditional H are supporting DFT evidence, never a selection policy."""

from copy import deepcopy
import numpy as np

from phase_agent.analysis.phase.phase_low_energy_D import evaluate_low_energy_prefixes, RULE
from phase_agent.analysis.phase.phase_hull_sensitivity import HullSensitivity


def dft_sampling_reference():
    return {
        "role": "agent_reference_not_automatic_ranking",
        "primary_goal": "获得可靠相图优先，同时改善模型；以下是可调整的选点经验，不是自动评分或删除规则。",
        "D_definition": "fixed_snapshot_low_energy_prefix_gap_over_10meV_per_O2",
        "D_unit": "dimensionless",
        "H_unit": "meV/O2",
        "Ehull_unit": "use_recorded_unit_eV_per_atom",
        "reference_steps": [
            "遵守decision_context.independent_audit_candidate_ids中的已登记独立抽检要求；选实际DFT时至少包含一个可用抽检候选，不要求全部入选。若这些ID均不在当前有效池，报告绑定缺口并请求刷新，不能凭空编造或自行解除校验。",
            "先核对当前模型/相图版本、精确几何、已有DFT和结构审查；已有同一几何的有效DFT不重复提交，已排除结构须经恢复流程。",
            "以当前正式Ehull确定热力学相关性，并保留递归后暴露为近凸包的候选；当前Ehull偏高不能单独否决条件近凸包点。",
            "用dft_sampling_evidence.D_score识别固定快照内同组分同相的少数低能离群；用真实委员会QBC看模型分歧，两者任一突出可作为核查理由，不能要求同时高。",
            "结合单点H序列、D候选联合H序列及审查中的其他联合影响评估相图重要性；H最大值依赖对应前置假设，不能当成当前独立影响。",
            "按下列经验比较凸包验证、模型改进、覆盖和共同偏差抽检的收益；按预算选互补结构，无固定比例、权重或新增D硬阈值。",
            "兼顾Na区间、相、结构环境及已有DFT覆盖；覆盖只在各相获准的Na域内评估，端点限定相不要求补全区间，样本数比例不是热力学相含量。候选池本身缺相或缺组分时，应考虑补搜索，不能声称仅选DFT就补齐覆盖。",
        ],
        "evidence_patterns": [
            {
                "when": "当前近凸包且H较高",
                "consider": "优先验证决定凸包的结构，QBC低也有价值。",
                "purpose": "hull_verification",
            },
            {
                "when": "D与QBC都较高",
                "consider": "核查疑似低能失准，避免重复计权；先取同簇代表。",
                "purpose": "model_improvement",
            },
            {
                "when": "D较高但QBC低",
                "consider": "保留共同偏差抽检；委员会一致不能排除共享偏差。",
                "purpose": "bias_check",
            },
            {
                "when": "QBC较高但D不突出",
                "consider": "考虑模型改进，优先近凸包和已有DFT未覆盖的环境。",
                "purpose": "model_improvement",
            },
            {
                "when": "原始Ehull高、递归后暴露为近凸包",
                "consider": "保留条件凸包验证候选，说明前置移除假设。",
                "purpose": "hull_verification",
            },
            {
                "when": "D/H/QBC均不突出",
                "consider": "按已有DFT覆盖决定补充或延期；必要时保留少量普通点抽检。",
                "purpose": "coverage",
            },
        ],
        "representative_selection": (
            "同组分同相的同一疑似低能簇，默认先选一个代表，兼顾能量、局域环境和已有DFT，避免选一串近重复结构。"
            "缺少可靠DFT基准时，再考虑一个绑定D证据中的上方参考；基准须为同组分同相、计算设置和能量口径可比、已完成且质量合格的DFT。"
            "直接验证候选还需核对精确几何；有可用基准则复用，不重复计算。上方参考也须满足可用性和审批条件，缺失就报告，不能编造。"
            "这是一簇一个代表的默认起点，若簇内环境差异或DFT误差证据支持，可说明理由增加代表，并遵守已有预算与审批。"
        ),
        "adaptation": [
            "当近凸包或递归暴露范围内，同时高D、高H的独立疑点较多，应按验证需要增加本轮DFT数量并扩大相/Na/结构覆盖；不能把重复几何的数量当作独立疑点。100只是单点与优化合计的硬上限，不是目标数量，费用只作估算。",
            "若近凸包或递归暴露范围内多个独立低能簇的D突出，应增加本轮DFT代表数量并拓宽相/Na覆盖；按独立疑点而非高D结构总数扩点，同簇去重，结合H、QBC和已有DFT，并说明预算内增加或替换的取舍。缺少的诊断几何先补入合法候选池；不得伪造QBC或突破已确认单轮数量/成本上限。",
            "结合post_dft_assessment和decision_feedback中实际DFT误差、能量排序变化、覆盖缺口与重复收益，说明本轮为何调整优先级或代表数量；没有结果就保持为假设。",
            "若D高而QBC低的代表确有大DFT误差，可提高此类共同偏差核查优先级；若高D簇经DFT支持，可减少重复核查，不能据此宣称所有高D结构合理。",
            "误差大或排序不稳时，在有依据的Na/相/环境范围拓宽；同类点收益低时转向未覆盖环境。明确人工约束优先，调整经验不改D/H定义、不自动改cutoff或预算、不触发删除或执行。",
        ],
        "cautions": [
            "D与QBC可能重叠，不直接相加重复计权；高D低QBC仍可用于共同偏差核查，高QBC不能证明错误；低相关也不能证明结构合理。",
            "优先当前近凸包或递归暴露的候选；在所检查各层都远离凸包时不因D大自动提权，保留少量有依据的覆盖探索。",
            "缺失D/H/QBC表示未知，不填零、不伪造，也不归入三者均低；QBC缺失不自动否决近凸包或高H候选，必要时提议补委员会评估。",
            "正式Ehull使用已执行移除后的正式凸包；假设右图和条件距离仅作补充，不能替换predicted_Ehull。",
            "D_score无量纲，当前定义实质为能量间隙/10，不是异常概率；旧phase_quality_signal.D_mev_per_O2是原审查参数，禁止混用阈值。",
            "QBC、D、H都不等于实测误差；选点倾向须结合当前数据，不能从少量相关性结果推出普遍因果。",
        ],
        "review_requirements": (
            "沿用scientific_decision.selection_purposes，为每个已选候选标注hull_verification/model_improvement/coverage/bias_check之一，"
            "reason遵守现有长度限制；整批简述互补性、低能簇代表与上方参考的取舍、H前置假设、证据缺口及经验调整依据。不自动增加数量或执行。"
        ),
    }


def fixed_snapshot_sampling_evidence(snapshot, report):
    from phase_agent.analysis.phase.phase_quality import _points, snapshot_evidence_digest

    if (
        snapshot.get("model_version") != report.get("model_version")
        or snapshot.get("version") != report.get("hull_version")
        or snapshot_evidence_digest(snapshot) != report.get("snapshot_evidence_digest")
    ):
        raise ValueError("DFT选点D/H证据与当前相图快照不匹配")
    points = _points(snapshot, [])
    _, candidates, membership = evaluate_low_energy_prefixes(points)
    by_key = {p["key"]: p for p in points}
    by_id = {c["candidate_id"]: c for c in candidates}
    single = {p["key"]: [] for p in points}
    distances = {p["key"]: [] for p in points}
    joint = {c["candidate_id"]: [] for c in candidates}
    layers = report.get("analysis_layers") or []
    if not layers:
        raise ValueError("缺少已绑定的H递归路径")
    removed = set()
    for layer in layers:
        active = [p for p in points if p["key"] not in removed]
        evaluator = HullSensitivity(active)
        for p in points:
            present = p["key"] not in removed
            single[p["key"]].append(evaluator.lift([p["key"]]) if present else None)
            distances[p["key"]].append(
                max(
                    0.0,
                    p["energy_mev_per_O2"]
                    - float(np.interp(p["x"], evaluator.xs, evaluator.baseline)),
                )
                if present
                else None
            )
        for c in candidates:
            keys = set(c["geometry_keys"])
            joint[c["candidate_id"]].append(
                None if keys.intersection(removed) else evaluator.lift(keys)
            )
        if layer["applied_to_preview"]:
            removed.update(layer["joint_geometry_keys"])

    def paths(keys):
        return [
            r["structure_path"] for k in keys for r in by_key[k]["rows"] if r.get("structure_path")
        ]

    def maximum(values):
        return max((v for v in values if v is not None), default=None)

    result = {}
    for key, cluster in membership.items():
        witness = by_id.get(cluster["D_candidate_id"], {})
        union = joint.get(cluster["D_candidate_id"], [None] * len(layers))
        result[key] = {
            "status": "available",
            "decision_made": False,
            "geometry_key": key,
            "model_version": snapshot["model_version"],
            "hull_version": snapshot["version"],
            "snapshot_evidence_digest": report["snapshot_evidence_digest"],
            "D_rule": RULE,
            "composition_per_O2": dict(cluster["composition_per_O2"]),
            "phase": cluster["phase"],
            "D_score": cluster["D"],
            "D_status": cluster["D_status"],
            "D_unit": "dimensionless",
            "D_gap_meV_per_O2": witness.get("gap_meV_per_O2"),
            "D_candidate_id": cluster["D_candidate_id"],
            "D_candidate_size": witness.get("candidate_size"),
            "D_candidate_geometry_keys": deepcopy(witness.get("geometry_keys", [])),
            "D_candidate_structure_paths": paths(witness.get("geometry_keys", [])),
            "D_reference_structure_paths": paths(witness.get("reference_geometry_keys", [])),
            "H_unit": "meV/O2",
            "H_layers": [r["analysis_layer"] for r in layers],
            "H_single_sequence": single[key],
            "H_single_max": maximum(single[key]),
            "H_D_candidate_union_sequence": union,
            "H_D_candidate_union_max": maximum(union),
            "distance_to_layer_hull_meV_per_O2": distances[key],
            "H_path_reference": report.get("report_path")
            or "current_phase_diagram.mlip.quality_diagnostics.analysis_layers",
            "H_path_scope": "existing_review_path_not_D_v2_driven; maxima_are_conditional",
            "formal_Ehull_reference": "current_formal_hull_after_executed_exclusions",
        }
    return result
