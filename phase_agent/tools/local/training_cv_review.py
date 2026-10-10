"""Compare saved baseline predictions with matched out-of-fold results."""

import csv
import json
import math
from pathlib import Path

from phase_agent.analysis.feedback.dft_comparison_tables import (
    build_comparison_tables,
    comparison_metrics,
)


def prepare_cv_review(state, job, result):
    root = Path(job["directory"]) / "results"

    def rows(name):
        with (root / name).open(encoding="utf-8-sig", newline="") as handle:
            return list(csv.DictReader(handle))

    energies = rows("energy_comparison.csv")
    forces = rows("force_comparison.csv")
    ids = {row.get("task_id") for row in energies if row.get("task_id")}
    records = [
        r
        for r in state.get("dft_dataset_records", [])
        if r.get("task_id") in ids and r.get("model_version") == job["original_model_version"]
    ]
    baseline, baseline_forces = build_comparison_tables(
        records, state.get("dft_mlip_comparisons", []), round_name=Path(job["directory"]).name
    )
    old = {r["task_id"]: r for r in baseline if r["comparison_status"] == "completed"}
    matched = [
        r
        for r in energies
        if r.get("task_id") in old
        and abs(float(r["E_DFT_eV_per_atom"]) - old[r["task_id"]]["dft_energy_eV_per_atom"]) < 1e-8
    ]
    matched_ids = {r["task_id"] for r in matched}
    # Compare only identical labeled force components, with unchanged atom order.
    old_forces = {
        (r["task_id"], int(r["atom_index"]), "xyz".index(r["component"])): r
        for r in baseline_forces
        if r["comparison_status"] == "completed"
    }
    force_errors = []
    matched_old_forces = []
    for row in forces:
        if row.get("task_id") not in matched_ids:
            continue
        key = (
            row["task_id"],
            int(row["atom_index"]),
            int(row["component"]) if row["component"].isdigit() else "xyz".index(row["component"]),
        )
        previous = old_forces.get(key)
        if previous and abs(float(row["F_DFT_eV_per_A"]) - previous["dft_force_eV_per_A"]) < 1e-8:
            force_errors.append(float(row["F_MLIP_eV_per_A"]) - float(row["F_DFT_eV_per_A"]))
            matched_old_forces.append(previous)

    def stats(values):
        return (
            {
                "mae": sum(abs(v) for v in values) / len(values),
                "rmse": math.sqrt(sum(v * v for v in values) / len(values)),
            }
            if values
            else None
        )

    report = {
        "evaluation_type": "baseline_vs_grouped_out_of_fold",
        "activatable": False,
        "training_fingerprint": result["fingerprint"],
        "structures": len(matched),
        "force_components": len(force_errors),
        "returned_structures": len(energies),
        "old": comparison_metrics([old[k] for k in matched_ids], matched_old_forces, {}),
        "new": {
            "energy_per_atom": stats(
                [float(r["E_MLIP_eV_per_atom"]) - float(r["E_DFT_eV_per_atom"]) for r in matched]
            ),
            "forces": stats(force_errors),
        },
        "limitations": [
            "K折评估训练方法，不是最终全数据模型的独立测试。",
            "未匹配旧模型同帧预测的结构不用于改善幅度计算；未评估近凸包排序。",
        ],
        "next_proposal": "先审阅匹配结构的能量和力误差变化，再决定补DFT；需验证最终模型时，从未参与训练的同组成近凸包候选中选择结构，先核对已有DFT能否复用，再提出补算方案及预算。",
    }
    path = root / "cv_baseline_review.json"
    content = json.dumps(report, ensure_ascii=False, indent=2)
    if not path.exists() or path.read_text(encoding="utf-8") != content:
        path.write_text(content, encoding="utf-8")

    def metric(value, unit):
        return (
            f"{value['mae'] * 1000:.2f}/{value['rmse'] * 1000:.2f} {unit}"
            if value and value.get("mae") is not None
            else "缺少匹配数据"
        )

    message = (
        f"K折对比报告已生成：{path}。匹配结构 {len(matched)}/{len(energies)}，力分量 {len(force_errors)}。"
        + "\n旧模型→折外模型：能量MAE/RMSE "
        + metric(report["old"]["energy_per_atom"], "meV/atom")
        + " → "
        + metric(report["new"]["energy_per_atom"], "meV/atom")
        + "；力MAE/RMSE "
        + metric(report["old"]["forces"], "meV/Å")
        + " → "
        + metric(report["new"]["forces"], "meV/Å")
        + "。\n执行方案："
        + report["next_proposal"]
        + " 本报告不激活模型。"
    )
    return report, message


def register_cv_candidate(current, key, job, result, report, handoff):
    """Register a reviewed CV candidate; only a separate explicit command activates."""
    complete = (
        report["structures"] > 0
        and report["structures"] == report["returned_structures"]
        and report["force_components"] == int(result["kfold_metrics"]["force_components"])
    )
    improved = complete and all(
        report["new"][name][metric] <= report["old"][name][metric]
        for name in ("energy_per_atom", "forces")
        for metric in ("mae", "rmse")
    )
    if not improved:
        return False
    from copy import deepcopy
    from phase_agent.tools.local.training_handoff import digest_file

    model = deepcopy(next(row for row in result["models"] if row["is_main_model"]))
    version = "remote-" + key + "-" + result["fingerprint"][:12]
    model.update(
        version=version,
        model_path=model.get("remote_model_path") or model["model_path"],
        dataset_version=job["report"].get("training_round", Path(job["directory"]).name),
        members=deepcopy(result["models"]),
    )
    validation = {
        "status": "completed",
        "passed": True,
        "evaluation_type": "grouped_cross_validation_review",
        "independent_test_completed": False,
        "checks": {"fully_matched": complete, "energy_and_force_improved": improved},
        "old_metrics": report["old"],
        "new_metrics": report["new"],
        "validation_data_version": "cv-" + result["fingerprint"],
        "limitations": report["limitations"],
        "activate_new_model": False,
    }
    stored = current.setdefault("candidate_models", {}).get(version)
    if stored and stored.get("status") == "user_rejected":
        handoff.update(stage="candidate_rejected", reason="该候选已被拒绝，未重新申请激活。")
        return True
    current["candidate_models"][version] = {
        "model": model,
        "validation": validation,
        "old_model_version": job["original_model_version"],
        "dft_data_version": model["dataset_version"],
        "validation_data_version": validation["validation_data_version"],
        "status": "validated_candidate",
    }
    if stored and stored.get("validation") == validation and stored.get("model") == model:
        for field in ("agent_review", "agent_review_diagnostic"):
            if field in stored:
                current["candidate_models"][version][field] = deepcopy(stored[field])
    artifact = Path(job["directory"]) / "results" / "cv_baseline_review.json"
    handoff.update(
        stage="awaiting_activation_approval",
        candidate_model_version=version,
        evidence_type="grouped_cross_validation_review",
        review_sha256=digest_file(artifact),
        reason=f"本轮微调已完成，K折同帧能量和力误差均改善。建议激活候选 {version}，随后刷新已有结构与凸包，再判断收敛、针对关键结构补DFT或下一轮搜索。K折不等同最终模型独立测试。\n如同意，回复：激活候选 {version} 原因：K折误差改善，采用新模型刷新凸包后评估收敛与DFT缺口。",
    )
    return True
