"""Read-only comparison of loaded workflow code with its source on disk."""

from functools import lru_cache
import inspect
import os
from pathlib import Path
import sys
from types import CodeType, FunctionType

WATCHED_MODULES = (
    "phase_agent.tools.state.execution_receipts",
    "phase_agent.tools.state.execution_consistency",
    "phase_agent.tools.workflows.prepare_dft_proposal",
    "phase_agent.tools.workflows.initial_tool_proposal",
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
    module_names = WATCHED_MODULES if module_names is None else module_names
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
        "scope": "watched source-defined function code; globals, generated code and unloaded modules are not verified",
    }


def assert_loaded_code_current():
    report = loaded_code_report()
    if report["status"] == "manual_reload_required":
        changed = [
            row["module"]
            for row in report["modules"]
            if row["status"] not in {"not_loaded", "matches"}
        ]
        raise RuntimeError(
            "运行中的代码与磁盘源码不一致；本次未推进科研任务。"
            "请先结束当前请求，再由用户手动重新启动项目服务加载一致版本。涉及模块："
            + ", ".join(changed)
        )
