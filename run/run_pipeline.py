"""调用一次完整搜索迭代的统一入口。"""

from __future__ import annotations

from typing import Any, Callable

from run.search_iteration import run_search_iteration


def run_pipeline(
    manager: Any,
    phase_references: dict[str, Any],
    config: dict[str, Any],
    *,
    state: dict[str, Any] | str | None = None,
    dispatcher: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    recovered_results: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """按 config 运行一轮；未提供 dispatcher 时只生成 pending 任务。"""
    required = {
        "structure_directory",
        "total_quota",
        "batch_size",
        "initial_states_per_branch",
        "seed",
    }
    missing = required - config.keys()
    if missing:
        raise ValueError(f"运行配置缺少字段：{sorted(missing)}")
    return run_search_iteration(
        manager,
        phase_references,
        state,
        structure_directory=config["structure_directory"],
        total_quota=config["total_quota"],
        batch_size=config["batch_size"],
        initial_states_per_branch=config["initial_states_per_branch"],
        seed=config["seed"],
        dispatcher=dispatcher,
        recovered_results=recovered_results,
        generation_metrics=config.get("generation_metrics"),
        generation_options=config.get("generation_options"),
        state_path=config.get("state_path"),
        ledger_path=config.get("ledger_path"),
        phase_diagram_directory=config.get("phase_diagram_directory"),
        task_versions=config.get("task_versions"),
        budget_limits=config.get("budgets"),
        system_config=config.get("system_config"),
    )
