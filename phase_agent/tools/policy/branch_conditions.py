"""Shared scientific lifecycle prerequisites, independent of LLM preferences."""

from phase_agent.tools.state.training_pending_validation import training_pending_validation
from phase_agent.tools.workflows.model_refresh_state import refresh_active


def branch_facts(state):
    refresh = state.get("model_refresh") or {}
    return {
        "config_version": state.get("confirmed_config_version"),
        "active_model_version": state.get("active_model_version"),
        "training_review_pending": training_pending_validation(state),
        "model_refresh_pending": refresh_active(state),
        "model_refresh_status": refresh.get("status"),
        "instruction": "这是进入条件事实，不是路线推荐；微调结果回收后先评估候选，模型切换后先完成结构与相图刷新。",
    }


def branch_entry_errors(action, state):
    tool = action.get("tool") or action.get("action_type")
    parameters = action.get("parameters") or {}
    errors = []
    if training_pending_validation(state) and tool == "update_mlip":
        errors.append("training_already_received_review_required")
    if refresh_active(state) and tool not in {"pause_search", "restart_failed_task"}:
        if tool != "prepare_local_batch_files" or parameters.get("mode") != "model_refresh_inputs":
            errors.append("new_model_structure_refresh_required")
    mc = mc_entry_facts(state)
    if (
        mc["required"]
        and not mc["ready"]
        and (
            tool == "allocate_mc_bohb"
            or (tool == "prepare_local_batch_files" and parameters.get("mode") == "mc_inputs")
        )
    ):
        errors.append("mc_requires_recovered_relax_and_current_hull")
    return errors


def mc_entry_facts(state):
    """Read only prerequisites; the model still chooses among feasible actions."""
    required = bool(state.get("generation_history"))
    version = state.get("active_model_version") or (
        (state.get("confirmed_config") or {}).get("calculation") or {}
    ).get("mlip_version")
    hull_version = state.get("current_branch_hull_version")
    pool = (state.get("branch_hull_batches") or {}).get(hull_version) or {}
    diagrams = state.get("phase_diagrams") or state.get("phase_diagram_state") or {}
    diagram = diagrams.get("diagrams", diagrams).get("mlip") or {}
    ready = bool(
        version
        and hull_version
        and pool.get("records")
        and pool.get("model_version") == version
        and diagram.get("status") == "completed"
        and diagram.get("version")
        and diagram.get("model_version") == version
    )
    return {
        "required": required,
        "ready": ready,
        "model_version": version,
        "hull_reference_version": hull_version,
        "phase_diagram_version": diagram.get("version"),
        "instruction": "Generated branches are initial structures, not Relax results. Before MC allocation or mc_inputs, prepare a separate prepare_local_batch_files action with mode=relax_inputs; the human uploads/submits, results are recovered and analyzed into a current-model Relax pool and completed MLIP hull. Only then choose MC steps and prepare its files. Never approve future MC allocation through the Relax preparation proposal. If inputs are already prepared or jobs pending, report upload/recovery status instead of repeating preparation.",
    }
