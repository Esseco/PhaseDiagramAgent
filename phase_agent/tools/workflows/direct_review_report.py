"""Publish a readable review of a previously verified DIRECT receipt."""

from collections import Counter
from copy import deepcopy
import hashlib
from pathlib import Path

from phase_agent.tools.workflows.direct_extrapolation_sampling import valid_direct_receipt


def bind_direct_candidates(state, ledger, scope_id):
    """Enrich only verified selected frames with their registered parent metadata."""
    from ase.io import read

    receipt = (state.get("direct_sampling_results") or {}).get(scope_id) or {}
    if not valid_direct_receipt(receipt) or receipt.get("model_version") != state.get(
        "active_model_version"
    ):
        raise ValueError("direct_review_receipt_invalid")
    result = deepcopy(state)
    tasks = {row.get("task_id"): row for row in result.get("tasks") or []}
    candidates = {row.get("candidate_id"): row for row in result.get("qbc_candidates") or []}
    branches = ledger.get("branches") or {}
    for selected in receipt.get("selected") or []:
        task = tasks.get(selected.get("source_task_id")) or {}
        candidate = candidates.get(selected.get("candidate_id")) or {}
        branch_id = task.get("branch_id")
        branch = branches.get(branch_id) or {}
        try:
            path = Path(selected["structure_path"])
            actual = hashlib.sha256(path.read_bytes()).hexdigest()
        except (KeyError, OSError, TypeError) as exc:
            raise ValueError("direct_candidate_binding_invalid") from exc
        if (
            task.get("model_version") != receipt["model_version"]
            or not branch.get("P")
            or not branch.get("H")
            or candidate.get("structure_sha256") != actual
            or selected.get("structure_sha256") != actual
        ):
            raise ValueError("direct_candidate_binding_invalid")
        symbols = read(path, index=0).get_chemical_symbols()
        candidate["branch_id"] = branch_id
        candidate["phase"] = branch["P"]
        candidate["phase_basis"] = "parent_branch"
        candidate["H"] = deepcopy(branch["H"])
        candidate["composition"] = dict(Counter(symbols))
        candidate.setdefault("candidate_origin", "direct_recovery")
        candidate.setdefault("qbc", {"status": "not_available_exact_current_geometry"})
    return result, {"status": "bound", "selected_count": len(receipt.get("selected") or [])}


def publish_direct_review(state, ledger, scope_id):
    receipt = (state.get("direct_sampling_results") or {}).get(scope_id) or {}
    if not valid_direct_receipt(receipt) or receipt.get("model_version") != state.get(
        "active_model_version"
    ):
        raise ValueError("direct_review_receipt_invalid")
    tasks = {t.get("task_id"): t for t in state.get("tasks") or []}
    branches = ledger.get("branches") or {}
    rows = []
    for item in receipt.get("selected") or []:
        task = tasks.get(item.get("source_task_id")) or {}
        path = Path(item.get("structure_path") or "")
        if (
            task.get("model_version") != receipt["model_version"]
            or not path.is_file()
            or hashlib.sha256(path.read_bytes()).hexdigest() != item.get("structure_sha256")
        ):
            raise ValueError("direct_review_selected_frame_invalid")
        branch = branches.get(task.get("branch_id")) or {}
        phase = branch.get("P")
        matrix = branch.get("H")
        if not phase or not isinstance(matrix, list) or len(matrix) != 3:
            raise ValueError("direct_review_branch_identity_missing")
        from sympy import Matrix

        multiplier = abs(int(Matrix(matrix).det()))
        rows.append((item, task, phase, multiplier))
    if len(rows) != receipt.get("selected_count") or len(
        {x[0]["candidate_id"] for x in rows}
    ) != len(rows):
        raise ValueError("direct_review_selection_count_invalid")
    counts = Counter(row[2] for row in rows)
    failures = sum(item.get("extrapolation_failure") is True for item, _, _, _ in rows)
    cycle = state.get("extrapolation_recovery_cycle") or {}
    diagram_note = (
        f"本轮凸包 `{cycle['invalid_hull_version']}` 已作废；latest 已恢复上一版 `{cycle['baseline_hull_version']}`。"
        if cycle.get("direct_source_digest") == receipt.get("source_digest")
        else "本轮外推失败较多，当前轮相图不应作为可靠结论；相图正式作废和后续微调状态仍需通过 Agent 登记。"
    )
    lines = [
        "# DIRECT 全批采样审核方案",
        "",
        f"模型：`{receipt['model_version']}`；范围：`{scope_id}`。",
        f"输入任务 {receipt['batch_task_count']}；去重后结构 {receipt['unique_candidate_count']}；"
        f"历史 DFT 覆盖参考 {receipt['reference_count']}；目标 {receipt['target']}，实际选出 {len(rows)}。",
        f"选中外推失败 {failures} 项；母相分布："
        + "、".join(f"{p} {n}" for p, n in sorted(counts.items()))
        + "。",
        "相标签与扩胞倍数来自登记的母分支，不代表已重新鉴定弛豫后相。",
        "本文件供审核；DIRECT 本身不包含 DFT 能量和受力标签，尚未创建或提交 DFT。",
        diagram_note,
        "",
        "| 序号 | 候选 | 任务 | 结构 | 母相 | 扩胞倍数 | Na x | 外推失败 | 末态 SHA256 |",
        "|---:|---|---|---|---|---:|---:|---|---|",
    ]
    for i, (item, task, phase, multiplier) in enumerate(rows, 1):
        lines.append(
            f"| {i} | {item['candidate_id']} | {item['source_task_id']} | {item['structure_id']} | {phase} | {multiplier} | {item.get('na_fraction', '未知')} | {'是' if item.get('extrapolation_failure') else '否'} | {item['structure_sha256']} |"
        )
    path = Path(receipt["plan_path"]).parent / "review.md"
    from phase_agent.analysis.feedback.export_dft_products import _publish_bytes

    _publish_bytes(path, ("\n".join(lines) + "\n").encode("utf8"))
    return {
        "status": "completed",
        "path": str(path),
        "selected_count": len(rows),
        "extrapolation_failure_count": failures,
        "parent_phase_counts": dict(counts),
    }
