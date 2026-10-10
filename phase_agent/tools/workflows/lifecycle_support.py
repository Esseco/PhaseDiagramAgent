"""State I/O and factual lifecycle support; no orchestration decisions."""

from copy import deepcopy
import json
from pathlib import Path
from phase_agent.tools.step_runner.file_protocol import write_json


def _config_migration_block_message(migration):
    status = migration.get("status")
    source, target = migration.get("from") or "未知", migration.get("to") or "未知"
    if status == "approval_required":
        return (
            f"运行仍绑定配置 {source}，当前确认配置为 {target}；已有任务或结果。"
            "若要迁移预算上限、已确认开启第二段 MC，或调整 MC 步数成本估算系数，"
            "请核对配置后发送“批准迁移”；其他科学设置变化不会被迁移。"
        )
    if status == "rejected_non_budget_change":
        fields = "、".join(migration.get("changed_fields") or []) or "科学设置"
        evidence = migration.get("compatibility_rejection")
        evidence_text = f"兼容性核验：{evidence}。" if evidence else ""
        return (
            f"已收到迁移批准，但配置 {source} → {target} 改动了 {fields}。"
            f"{evidence_text}运行状态和历史结果未修改。"
        )
    if status == "rejected_budget_decrease":
        return f"配置 {source} → {target} 降低了已有运行的预算上限，不能迁移；运行状态和历史结果未修改。"
    return f"配置迁移被拒绝（{status}）；运行状态和历史结果未修改。"


def _count_newly_recovered(reconciled):
    return sum(row.get("status") in {"settled", "already_settled"} for row in (reconciled or []))


def _branch_candidates_for_agent(manager):
    """Expose bounded factual branch choices before the Agent proposes a batch."""
    rows = []
    if manager is None:
        return rows
    for branch_id in sorted(manager.data.get("branches", {})):
        branch = manager.data["branches"][branch_id]
        rows.append(
            {
                key: deepcopy(value)
                for key, value in {
                    "branch_id": branch_id,
                    "P": branch.get("P"),
                    "x": branch.get("x"),
                    "T": branch.get("T"),
                    "composition": branch.get("composition"),
                    "structure_count": len(branch.get("structure_ids") or []),
                }.items()
            }
        )
    return rows


def _read_state_for_runner(state, state_path):
    if isinstance(state, (str, Path)):
        return json.loads(Path(state).read_text(encoding="utf-8"))
    if state is not None:
        return deepcopy(state)
    if state_path and Path(state_path).is_file():
        return json.loads(Path(state_path).read_text(encoding="utf-8"))
    return {}


def _save_runner_state(state, state_path):
    if state_path is None:
        return
    write_json(state_path, state)
    from phase_agent.tools.state.execution_receipts import record_business_saved

    record_business_saved(state_path, state)
    from phase_agent.persistence.memory.publish_memory_views import publish_memory_views

    publish_memory_views(state, state_path)


def _dft_comparison_evaluator(adapters, config, state=None):
    if "final_frame_mlip_evaluator" in adapters:
        return adapters["final_frame_mlip_evaluator"]
    from phase_agent.tools.workflows.create_dft_comparison_evaluator import (
        create_dft_comparison_evaluator,
    )

    return create_dft_comparison_evaluator(config, state=state)
