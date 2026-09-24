"""从已保存台账和状态继续下一轮搜索。"""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

from data_layer.ledger.phase_data_manager import PhaseDataManager
from run.run_pipeline import run_pipeline


def resume_pipeline(config: dict, phase_references: dict, *, dispatcher=None, recovered_results=None, additional_budget=0.0, stage_budget_increases=None, resume_existing_first=True) -> dict:
    ledger_path = Path(config["ledger_path"])
    state_path = Path(config["state_path"])
    if not ledger_path.exists():
        raise FileNotFoundError(f"台账不存在：{ledger_path}")
    if not state_path.exists():
        raise FileNotFoundError(f"搜索状态不存在：{state_path}")
    manager = PhaseDataManager.load(ledger_path)
    state = json.loads(state_path.read_text(encoding="utf-8"))
    resumed_config = deepcopy(config)
    resumed_config["budgets"] = deepcopy(state.get("budget_limits", config["budgets"]))
    if additional_budget < 0:
        raise ValueError("additional_budget 不能为负")
    resumed_config["budgets"]["total_relative_cost"] += float(additional_budget)
    for stage, increase in (stage_budget_increases or {}).items():
        if stage not in resumed_config["budgets"]["stage_limits"] or increase < 0:
            raise ValueError(f"非法阶段预算增量：{stage}={increase}")
        limit = resumed_config["budgets"]["stage_limits"][stage]
        if limit.get("max_cost") is not None:
            limit["max_cost"] += float(increase)
    if additional_budget or stage_budget_increases:
        state["run_status"] = "active"
        state.pop("pause_reasons", None)
    run_config = deepcopy(resumed_config)
    if resume_existing_first:
        run_config["total_quota"] = 0
    result = run_pipeline(manager, phase_references, run_config, state=state, dispatcher=dispatcher, recovered_results=recovered_results)
    result["manager"] = manager
    result["resumed"] = True
    result["effective_config"] = resumed_config
    return result
