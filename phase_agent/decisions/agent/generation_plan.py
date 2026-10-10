"""Validate explicit strategy allocations; no scientific values are inferred."""

from collections import Counter
from typing import get_args
from pydantic import ValidationError
from phase_agent.decisions.agent.generation_contracts import (
    GenerationAllocation,
    GenerationStrategy,
)

STRATEGIES = set(get_args(GenerationStrategy))


def derive_generation_totals(action):
    """Compute redundant counts from a validated model-selected allocation list.

    Explicit conflicting counts are retained for validation, never corrected.
    """
    from copy import deepcopy

    if (
        not isinstance(action, dict)
        or (action.get("tool") or action.get("action_type")) != "generate_branches"
    ):
        return action
    params = action.get("parameters") or {}
    plan = params.get("generation_plan")
    if not isinstance(plan, list) or not 1 <= len(plan) <= 12:
        return action
    try:
        rows = [GenerationAllocation.model_validate(row) for row in plan]
    except ValidationError:
        return action
    totals = Counter()
    for row in rows:
        totals[row.strategy] += row.quota
    result = deepcopy(action)
    params = result["parameters"]
    params.setdefault("quotas", dict(totals))
    params.setdefault("total_quota", sum(totals.values()))
    return result


def validate_generation_plan(params):
    plan = params.get("generation_plan")
    if not isinstance(plan, list) or not 1 <= len(plan) <= 12:
        raise ValueError("生成方案需包含1–12项实际策略分配")
    quotas = params.get("quotas")
    if not isinstance(quotas, dict) or any(
        k not in STRATEGIES or type(v) is not int or v < 0 for k, v in quotas.items()
    ):
        raise ValueError("quotas必须是合法策略到非负JSON整数的映射")
    if type(params.get("total_quota")) is not int or params["total_quota"] <= 0:
        raise ValueError("total_quota必须是正JSON整数")
    totals = Counter()
    for index, row in enumerate(plan):
        try:
            GenerationAllocation.model_validate(row)
        except ValidationError as error:
            messages = [
                f"generation_plan[{index}]." + ".".join(map(str, e["loc"])) + ": " + e["msg"]
                for e in error.errors(include_input=False, include_url=False)
            ]
            raise ValueError("; ".join(messages)) from None
        totals[row["strategy"]] += row["quota"]
    if dict(totals) != {k: v for k, v in quotas.items() if v}:
        raise ValueError(
            f"策略清单与quotas不一致：清单汇总={dict(totals)}，quotas={quotas}；请保持原科学分配意图并统一两处数量"
        )
    if sum(totals.values()) != params.get("total_quota"):
        raise ValueError("策略分配总数与total_quota不一致")
    return plan


def configured_generation_strategies(config):
    """Intersect explicit configuration lists; an empty list disables all strategies."""
    system = config.get("system") or config.get("system_config") or {}
    sections = [
        (system.get("generation") or {}, "enabled_strategies"),
        (config.get("generation_actions") or {}, "enabled"),
    ]
    enabled = set(STRATEGIES)
    for section, key in sections:
        if key in section:
            enabled.intersection_update(section[key] or [])
    # Boundary facts remain authoritative even for older unsynchronized snapshots.
    boundary = system.get("boundary") or {}
    if "P" in boundary:
        from phase_agent.science.structures.boundary_utils import allowed_phases

        if len(allowed_phases(boundary["P"])) < 2:
            enabled.discard("competing_phase")
    ratio = boundary.get("TM_ratio")
    if isinstance(ratio, dict) and len(ratio) == 1:
        enabled.discard("tm_ordering")
    roles = (system.get("configuration_space") or {}).get("roles") or {}
    for variable, strategy in {
        "T": "tm_ordering",
        "P": "competing_phase",
        "x": "composition",
        "H": "periodic_extension",
    }.items():
        if roles.get(variable) == "fixed":
            enabled.discard(strategy)
    return sorted(enabled)


def disabled_generation_allocations(parameters, enabled):
    used = {
        row.get("strategy")
        for row in parameters.get("generation_plan") or []
        if isinstance(row, dict) and row.get("quota")
    }
    used.update(name for name, quota in (parameters.get("quotas") or {}).items() if quota)
    return sorted(used - set(enabled))
