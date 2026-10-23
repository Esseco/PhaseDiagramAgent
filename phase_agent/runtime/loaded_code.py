"""Read-only comparison of loaded workflow code with its source on disk."""

from functools import lru_cache
import inspect
import os
from pathlib import Path
import sys
from types import CodeType, FunctionType

WATCHED_MODULES = (
    "phase_agent.runtime.interaction_reports",
    "phase_agent.tools.relax_mc_batch.failure_diagnostics",
    "phase_agent.runtime.agent_api",
    "phase_agent.runtime.control_request_contracts",
    "phase_agent.tools.workflows.lifecycle_initialization",
    "phase_agent.tools.workflows.readonly_views",
    "phase_agent.tools.workflows.compact_review",
    "phase_agent.tools.workflows.dft_selection_artifacts",
    "phase_agent.runtime.ui_operations",
    "phase_agent.runtime.boundary_diagnostics",
    "phase_agent.runtime.project_diagnostics",
    "phase_agent.runtime.operation_resolution",
    "phase_agent.persistence.ledger.result_validation",
    "phase_agent.persistence.ledger.collect_calculation_results",
    "phase_agent.tools.state.reconcile_task_results",
    "phase_agent.tools.workflows.apply_scientific_feedback",
    "phase_agent.analysis.state.summarize_agent_state",
    "phase_agent.runtime.result_recovery_status",
    "phase_agent.runtime.scientific_progress",
    "phase_agent.runtime.local_agent_control",
    "phase_agent.runtime.local_http_server",
    "phase_agent.runtime.flow_panel",
    "phase_agent.tools.relax_mc_batch.recovery",
    "phase_agent.tools.cost.recovery_job_accounting",
    "phase_agent.persistence.result_recovery_commit",
    "phase_agent.graphs.business_checkpoint",
    "phase_agent.tools.workflows.compact_action_history",
    "phase_agent.decisions.agent.generation_contracts",
    "phase_agent.decisions.agent.post_dft_review",
    "phase_agent.runtime.waiting_files_presentation",
    "phase_agent.tools.remote.transfer_bundle",
    "phase_agent.tools.remote.integrity",
    "phase_agent.tools.relax_mc_batch.run",
    "phase_agent.tools.relax_mc_batch.dependencies",
    "phase_agent.runtime.studio_payloads",
    "phase_agent.analysis.state.decision_evidence_catalog",
    "phase_agent.decisions.agent.branch_pool_review",
    "phase_agent.decisions.agent.branch_review_repair",
    "phase_agent.analysis.cost.estimate_proposal_cost",
    "phase_agent.analysis.cost.estimate_proposal_runtime",
    "phase_agent.analysis.convergence.mlip_relaxation",
    "phase_agent.science.mlip.slurm_executor",
    "phase_agent.science.mlip.mace_worker",
    "phase_agent.analysis.phase.ensure_phase_identification",
    "phase_agent.analysis.phase.refresh_identified_phases",
    "phase_agent.decisions.agent.create_deepseek_client",
    "phase_agent.analysis.phase.identify_result_phase",
    "phase_agent.tools.workflows.compact_review",
    "phase_agent.tools.workflows.generation_artifacts",
    "phase_agent.tools.workflows.region_review",
    "phase_agent.tools.workflows.dft_selection_artifacts",
    "phase_agent.tools.workflows.model_refresh_gate",
    "phase_agent.tools.policy.model_generation",
    "phase_agent.tools.policy.region_sampling",
    "phase_agent.decisions.agent.propose_tool_action",
    "phase_agent.tools.policy.branch_conditions",
    "phase_agent.tools.workflows.initial_tool_proposal",
    "phase_agent.tools.workflows.propose_followup",
    "phase_agent.analysis.state.training_context",
    "phase_agent.runtime.plan_queries",
    "phase_agent.runtime.read_only_queries",
    "phase_agent.graphs.dialogue.saved_control",
    "phase_agent.tools.workflows.create_active_learning_handlers",
    "phase_agent.tools.relax_mc_batch.analysis",
    "phase_agent.tools.budget.reserve_budget",
    "phase_agent.tools.local.prepare_model_refresh",
    "phase_agent.tools.local.prepare_relax_upload_batches",
    "phase_agent.tools.workflows.prepare_branch_relaxation",
    "phase_agent.tools.relax_mc_batch.prepare",
    "phase_agent.tools.relax_mc_batch.results",
    "phase_agent.graphs.dialogue.evidence",
    "phase_agent.tools.relax_mc_batch.plan",
    "phase_agent.tools.workflows.relax_proposal_feedback",
    "phase_agent.decisions.agent.summary_request",
    "phase_agent.graphs.batch_recovery_graph",
    "phase_agent.graphs.approval_graph",
    "phase_agent.graphs.direction_review_graph",
    "phase_agent.persistence.compact_checkpoints",
    "phase_agent.persistence.checkpoint_blobs",
    "phase_agent.tools.state.execution_receipts",
    "phase_agent.tools.state.execution_progress",
    "phase_agent.tools.state.execution_reconciliation",
    "phase_agent.tools.state.execution_recovery_evidence",
    "phase_agent.tools.state.recovered_input_plan",
    "phase_agent.tools.state.workflow_control",
    "phase_agent.tools.dispatch.execute_tool_action",
    "phase_agent.tools.state.execution_consistency",
    "phase_agent.tools.workflows.prepare_dft_proposal",
    "phase_agent.tools.workflows.reuse_dft_selection",
    "phase_agent.graphs.dialogue.nodes",
    "phase_agent.interface_compatibility",
    "phase_agent.runtime.loaded_code",
    "phase_agent.configuration.schema.state_schema_migrations",
    "phase_agent.configuration.runtime.authorize_budget_extension",
    "phase_agent.persistence.memory.decision_memory",
    "phase_agent.persistence.shared_dft",
    "phase_agent.analysis.convergence.single_point_validation",
    "phase_agent.analysis.phase.update_phase_diagram",
    "phase_agent.graphs.business_checkpoint",
    "phase_agent.tools.budget.reserve_dft_actions",
    "phase_agent.tools.state.state_manager",
    "phase_agent.tools.state.reconcile_task_results",
    "phase_agent.tools.policy.workflow_policy",
    "phase_agent.tools.remote.batch_runner",
    "phase_agent.tools.remote.manual_upload_runner",
    "phase_agent.tools.local.prepare_dft_upload_batches",
    "phase_agent.tools.workflows.lifecycle_initialization",
    "phase_agent.tools.workflows.run_event_loop",
    "phase_agent.tools.workflows.run_tool_step",
    "phase_agent.tools.workflows.tool_outcomes",
    "phase_agent.tools.workflows.create_dft_selection_handler",
    "phase_agent.tools.workflows.react_proposal",
    "phase_agent.runtime.studio_runtime",
    "phase_agent.runtime.local_http_server",
    "phase_agent.runtime.workflow_reply_presentation",
    "phase_agent.runtime.post_dft_presentation",
    "phase_agent.runtime.status_presentation",
    "phase_agent.runtime.concise_dialogue",
    "phase_agent.runtime.proposal_presentation",
    "phase_agent.runtime.chat_state_presentation",
    "phase_agent.runtime.studio_reply_presentation",
    "phase_agent.runtime.waiting_files_presentation",
    "phase_agent.runtime.training_reply",
    "phase_agent.runtime.handoff_presentation",
    "phase_agent.runtime.review_presentation",
    "phase_agent.runtime.review_file_presentation",
    "phase_agent.decisions.agent.post_dft_review",
)

_INTERFACES = {
    "prepare",
    "migrate_persisted_state",
    "authorize_budget_extension",
    "update_short_term_memory",
    "single_point_validation",
    "update_phase_diagram",
    "checkpoint_frame",
    "reserve_dft_actions",
    "update_state_snapshot",
    "agent_state_summary",
    "reconcile_task_results",
    "normalize_workflow_state",
    "prepare_dft_upload_batches",
    "_apply_execution_result",
}


@lru_cache(maxsize=128)
def _code_path(filename):
    return Path(filename).resolve()


@lru_cache(maxsize=64)
def _disk_codes(path, modified_ns, size):
    # Compile for comparison only: never execute or reload a module.
    code = compile(
        Path(path).read_bytes(), path, "exec", dont_inherit=True, optimize=sys.flags.optimize
    )
    codes = {}

    def collect(current):
        codes[current.co_qualname] = current
        for value in current.co_consts:
            if isinstance(value, CodeType):
                collect(value)

    collect(code)
    return codes


def _functions(module):
    for value in tuple(vars(module).values()):
        if isinstance(value, FunctionType):
            yield inspect.unwrap(value)
        elif isinstance(value, type) and value.__module__ == module.__name__:
            for item in tuple(vars(value).values()):
                if isinstance(item, (staticmethod, classmethod)):
                    item = item.__func__
                if isinstance(item, FunctionType):
                    yield inspect.unwrap(item)


def loaded_code_report(*, module_names=None, source_root=None):
    module_names = tuple(WATCHED_MODULES if module_names is None else module_names)
    root = Path(source_root or Path(__file__).resolve().parents[2]).resolve()
    modules = []
    for name in module_names:
        module = sys.modules.get(name)
        if module is None:
            modules.append({"module": name, "status": "not_loaded"})
            continue
        path = Path(getattr(module, "__file__", "")).resolve()
        expected = root.joinpath(*name.split(".")).with_suffix(".py")
        row = {"module": name, "loaded_path": str(path)}
        if path != expected:
            row["status"] = "unexpected_source_path"
        else:
            try:
                info = path.stat()
                codes = _disk_codes(str(path), info.st_mtime_ns, info.st_size)
                functions = {
                    function.__code__.co_qualname: function
                    for function in _functions(module)
                    if function.__module__ == name
                    and _code_path(function.__code__.co_filename) == path
                }
                changed = sorted(
                    key for key, value in functions.items() if codes.get(key) != value.__code__
                )
                row.update(
                    status="source_changed" if changed else "matches",
                    checked_functions=len(functions),
                    changed_functions=changed,
                    loaded_signatures={
                        key: str(inspect.signature(function))
                        for key, function in functions.items()
                        if function.__name__ in _INTERFACES
                    },
                )
            except (OSError, SyntaxError, ValueError) as error:
                row.update(status="comparison_unavailable", error_type=type(error).__name__)
        modules.append(row)
    stale = any(row["status"] not in {"not_loaded", "matches"} for row in modules)
    return {
        "status": "manual_reload_required" if stale else "checked_code_matches",
        "source_root": str(root),
        "pid": os.getpid(),
        "python": sys.executable,
        "modules": modules,
        "watched_module_count": len(module_names),
        "scope": "watched workflow, batch budget, proposal preflight, reply and checkpoint source-defined function code; globals, generated code and unloaded modules are not verified",
    }


class LoadedCodeChangedError(RuntimeError):
    """Known no-execution block with inspectable module evidence."""

    def __init__(self, message, report):
        super().__init__(message)
        self.report = report


def assert_loaded_code_current():
    report = loaded_code_report()
    if report["status"] == "manual_reload_required":
        changed = [
            row["module"]
            for row in report["modules"]
            if row["status"] not in {"not_loaded", "matches"}
        ]
        raise LoadedCodeChangedError(
            "运行中的代码与磁盘源码不一致；本次未推进科研任务。"
            "请先结束当前请求，再由用户手动重新启动项目服务加载一致版本。涉及模块："
            + ", ".join(changed),
            report,
        )
